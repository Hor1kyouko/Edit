"""Model-independent numerical helpers for EXP-003."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


PAIRED_METRICS = (
    "Q1",
    "Q2",
    "cos_dAB_dA_plus_dB",
    "ratio_dA_plus_dB_to_dAB",
    "cos_dA_dB",
    "ratio_dA_to_dAB",
    "ratio_dB_to_dAB",
    "norm_dA",
    "norm_dB",
    "norm_dAB",
    "norm_dA_plus_dB",
    "norm_r_ref",
)


def validate_atomic_conditions(
    conditions: Mapping[str, str], clause_a: str, clause_b: str
) -> None:
    """Require exact atomic reuse rather than semantic paraphrases."""

    if set(conditions) != {"0", "A", "B", "AB"}:
        raise ValueError("conditions must be exactly 0/A/B/AB")
    if conditions["A"] != clause_a or conditions["B"] != clause_b:
        raise ValueError("A and B must equal their registered atomic clauses")
    expected_ab = f"{clause_a} and {clause_b}"
    if conditions["AB"] != expected_ab:
        raise ValueError(f"AB must be the exact composition {expected_ab!r}")
    if conditions["0"] != "Keep the image unchanged.":
        raise ValueError("EXP-003 keeps the EXP-002 preservation reference unchanged")


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def paired_prompt_comparison(
    legacy_records: Sequence[Mapping[str, Any]],
    atomic_records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = PAIRED_METRICS,
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compare atomic and legacy measurements at matched seed/schedule index.

    No hypothesis-test threshold or automatic scientific conclusion is applied.
    All per-seed observations remain visible because n=5 is descriptive.
    """

    def index(
        records: Sequence[Mapping[str, Any]]
    ) -> dict[tuple[int, int], Mapping[str, Any]]:
        result: dict[tuple[int, int], Mapping[str, Any]] = {}
        for seed_record in records:
            seed = int(seed_record["seed"])
            for probe in seed_record.get("probes", []):
                if probe.get("valid", False):
                    result[(seed, int(probe["schedule_index"]))] = probe
        return result

    legacy = index(legacy_records)
    atomic = index(atomic_records)
    matched = sorted(set(legacy) & set(atomic))
    missing = sorted(set(atomic) - set(legacy))
    by_probe: list[dict[str, Any]] = []
    for schedule_index in sorted({key[1] for key in matched}):
        keys = [key for key in matched if key[1] == schedule_index]
        first = atomic[keys[0]]
        metric_rows: dict[str, Any] = {}
        for metric in metrics:
            individual: dict[str, Any] = {}
            deltas: list[float] = []
            percent_deltas: list[float] = []
            for key in keys:
                old = float(legacy[key]["metrics"][metric])
                new = float(atomic[key]["metrics"][metric])
                delta = new - old
                percent = math.nan if abs(old) <= epsilon else 100.0 * delta / abs(old)
                individual[str(key[0])] = {
                    "legacy": old,
                    "atomic": new,
                    "atomic_minus_legacy": delta,
                    "percent_change_relative_to_abs_legacy": percent,
                }
                deltas.append(delta)
                percent_deltas.append(percent)
            metric_rows[metric] = {
                "individual_seed_values": individual,
                "delta_summary": _summary(deltas),
                "percent_delta_summary": _summary(percent_deltas),
                "sign_consistency": {
                    "atomic_lower": sum(value < 0 for value in deltas),
                    "equal": sum(value == 0 for value in deltas),
                    "atomic_higher": sum(value > 0 for value in deltas),
                },
            }
        by_probe.append(
            {
                "schedule_index": schedule_index,
                "target_sigma": float(first["target_sigma"]),
                "actual_sigma": float(first["actual_sigma"]),
                "metrics": metric_rows,
            }
        )
    return {
        "comparison": "T_atomic - T_legacy, paired by seed and schedule index",
        "automatic_scientific_classification": None,
        "matched_pairs": [
            {"seed": seed, "schedule_index": index_} for seed, index_ in matched
        ],
        "atomic_pairs_missing_from_legacy": [
            {"seed": seed, "schedule_index": index_} for seed, index_ in missing
        ],
        "by_probe": by_probe,
    }
