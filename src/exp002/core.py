"""Numerical core for EXP-002.

The functions in this module deliberately do not know about VeloEdit.  They
operate on already aligned velocity predictions and serializable records so
that the scientific arithmetic can be tested independently of a GPU runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TARGET_SIGMAS = (1.00, 0.90, 0.75, 0.50, 0.25, 0.10)


@dataclass(frozen=True)
class SigmaProbe:
    target_sigma: float
    actual_sigma: float
    schedule_index: int
    model_timestep: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def map_target_sigmas(
    schedule: Sequence[float],
    targets: Sequence[float] = DEFAULT_TARGET_SIGMAS,
) -> tuple[list[SigmaProbe], list[dict[str, Any]]]:
    """Map target sigmas to nearest non-terminal schedule entries.

    The terminal sigma is excluded because no model velocity is evaluated
    after the final Euler update. Duplicate nearest indices are retained only
    once and explicitly reported.
    """

    values = [float(value) for value in schedule]
    if len(values) < 2:
        raise ValueError("schedule must contain at least one model step and a terminal sigma")
    candidates = values[:-1]
    selected: list[SigmaProbe] = []
    duplicate_mappings: list[dict[str, Any]] = []
    seen: dict[int, float] = {}
    for raw_target in targets:
        target = float(raw_target)
        index = min(range(len(candidates)), key=lambda i: (abs(candidates[i] - target), i))
        if index in seen:
            duplicate_mappings.append(
                {
                    "target_sigma": target,
                    "deduplicated_to_target_sigma": seen[index],
                    "schedule_index": index,
                    "actual_sigma": candidates[index],
                }
            )
            continue
        seen[index] = target
        # FLUX.1-Kontext's analyzer receives 1000 * sigma and divides it by
        # 1000 immediately before the transformer call. Record the latter.
        selected.append(SigmaProbe(target, candidates[index], index, candidates[index]))
    selected.sort(key=lambda probe: probe.schedule_index)
    return selected, duplicate_mappings


def _l2(tensor: Any) -> float:
    return float(tensor.norm().item())


def _relative_l2(delta: Any, reference: Any, eps: float) -> float:
    return _l2(delta) / (_l2(reference) + eps)


def _cosine(a: Any, b: Any, eps: float) -> float:
    denominator = _l2(a) * _l2(b)
    if denominator <= eps:
        return math.nan
    return float((a.flatten() @ b.flatten()).item()) / denominator


def centered_velocity_metrics(
    velocities: Mapping[str, Any],
    *,
    torch_module: Any,
    eps: float = 1e-12,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute EXP-002 quantities using explicitly centered FP32 arithmetic.

    ``velocities`` must contain effective model velocities for 0/A/B/AB at
    exactly the same latent state and timestep. Returned tensors are FP32 and
    intended for optional diagnostic persistence, not the Euler update.
    """

    missing = [label for label in ("0", "A", "B", "AB") if label not in velocities]
    if missing:
        raise KeyError(f"missing conditions: {missing}")
    shapes = {label: tuple(velocities[label].shape) for label in ("0", "A", "B", "AB")}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"velocity shape mismatch: {shapes}")

    fp32 = {label: velocities[label].detach().to(torch_module.float32) for label in velocities}
    if not all(bool(torch_module.isfinite(value).all().item()) for value in fp32.values()):
        raise ValueError("non-finite velocity prediction")

    d_a = fp32["A"] - fp32["0"]
    d_b = fp32["B"] - fp32["0"]
    d_ab = fp32["AB"] - fp32["0"]
    additive = d_a + d_b
    r_ref = d_ab - additive

    # Reproduce the legacy EXP-001 expression in the prediction dtype, then
    # cast only the final residual to FP32 for a like-for-like comparison.
    legacy = velocities["AB"] - velocities["A"] - velocities["B"] + velocities["0"]
    legacy_fp32 = legacy.detach().to(torch_module.float32)
    discrepancy = legacy_fp32 - r_ref

    norms = {
        "dA": _l2(d_a),
        "dB": _l2(d_b),
        "dAB": _l2(d_ab),
        "dA_plus_dB": _l2(additive),
        "r_ref": _l2(r_ref),
    }
    metrics = {
        **{f"norm_{key}": value for key, value in norms.items()},
        "Q1": norms["r_ref"] / (norms["dAB"] + eps),
        "Q2": norms["r_ref"] / (norms["dA"] + norms["dB"] + eps),
        "cos_dA_dB": _cosine(d_a, d_b, eps),
        "cos_dA_dAB": _cosine(d_a, d_ab, eps),
        "cos_dB_dAB": _cosine(d_b, d_ab, eps),
        "cos_dAB_dA_plus_dB": _cosine(d_ab, additive, eps),
        "ratio_dA_to_dAB": norms["dA"] / (norms["dAB"] + eps),
        "ratio_dB_to_dAB": norms["dB"] / (norms["dAB"] + eps),
        "ratio_dA_plus_dB_to_dAB": norms["dA_plus_dB"] / (norms["dAB"] + eps),
        "legacy_bf16_vs_fp32_residual_l2": _l2(discrepancy),
        "legacy_bf16_vs_fp32_residual_relative_l2": _relative_l2(discrepancy, r_ref, eps),
        "legacy_bf16_vs_fp32_residual_max_abs": float(discrepancy.abs().max().item()),
    }
    tensors = {
        "dA_fp32": d_a,
        "dB_fp32": d_b,
        "dAB_fp32": d_ab,
        "dA_plus_dB_fp32": additive,
        "r_ref_fp32": r_ref,
        "r_ref_legacy_prediction_dtype_then_fp32": legacy_fp32,
    }
    return metrics, tensors


def classify_trend(values: Sequence[float]) -> str:
    """Classify a temporal sequence without imposing an effect threshold."""

    if len(values) < 2:
        return "insufficient"
    decreasing = all(right <= left for left, right in zip(values, values[1:]))
    increasing = all(right >= left for left, right in zip(values, values[1:]))
    if decreasing and increasing:
        return "constant"
    if decreasing:
        return "decreasing"
    if increasing:
        return "increasing"
    return "non_monotonic"


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def aggregate_seed_records(seed_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate valid per-seed probe records by actual schedule index."""

    by_probe: dict[int, list[Mapping[str, Any]]] = {}
    per_seed_q1: dict[str, list[float]] = {}
    excluded: list[dict[str, Any]] = []
    for seed_record in seed_records:
        seed = str(seed_record["seed"])
        per_seed_q1[seed] = []
        for probe in seed_record.get("probes", []):
            if not probe.get("valid", False):
                excluded.append({"seed": int(seed_record["seed"]), "schedule_index": probe.get("schedule_index"), "reason": probe.get("invalid_reason")})
                continue
            by_probe.setdefault(int(probe["schedule_index"]), []).append(probe)
            per_seed_q1[seed].append(float(probe["metrics"]["Q1"]))

    aggregate: list[dict[str, Any]] = []
    for index in sorted(by_probe):
        rows = by_probe[index]
        aggregate.append(
            {
                "schedule_index": index,
                "target_sigma": float(rows[0]["target_sigma"]),
                "actual_sigma": float(rows[0]["actual_sigma"]),
                "individual_seed_values": {
                    str(row["seed"]): {
                        "Q1": float(row["metrics"]["Q1"]),
                        "Q2": float(row["metrics"]["Q2"]),
                        "cos_dAB_dA_plus_dB": float(row["metrics"]["cos_dAB_dA_plus_dB"]),
                        "ratio_dA_plus_dB_to_dAB": float(row["metrics"]["ratio_dA_plus_dB_to_dAB"]),
                    }
                    for row in rows
                },
                "Q1": _summary(row["metrics"]["Q1"] for row in rows),
                "Q2": _summary(row["metrics"]["Q2"] for row in rows),
                "cos_dAB_dA_plus_dB": _summary(row["metrics"]["cos_dAB_dA_plus_dB"] for row in rows),
                "ratio_dA_plus_dB_to_dAB": _summary(row["metrics"]["ratio_dA_plus_dB_to_dAB"] for row in rows),
            }
        )

    trends = {seed: classify_trend(values) for seed, values in per_seed_q1.items()}
    counts = {name: sum(value == name for value in trends.values()) for name in ("decreasing", "increasing", "non_monotonic", "constant", "insufficient")}
    invalid_seeds = [
        {
            "seed": int(record["seed"]),
            "status": record.get("status"),
            "reasons": record.get("invalid_reasons"),
        }
        for record in seed_records
        if record.get("status") != "complete"
    ]
    return {
        "by_probe": aggregate,
        "Q1_temporal_trend_by_seed": trends,
        "Q1_temporal_trend_counts": counts,
        "excluded_invalid_probes": excluded,
        "invalid_or_incomplete_seeds": invalid_seeds,
    }
