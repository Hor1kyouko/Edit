"""EXP-003 runner for prompt control and minimal cross-image replication.

The VeloEdit checkout is imported read-only. Scientific arithmetic reuses the
validated EXP-002 implementation; this runner changes prompts/images and uses
targeted order-invariance spot checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from src.exp001.core import (
    CONDITION_ORDER_FORWARD,
    VelocityPair,
    evaluate_shared_state,
    validate_conditions,
)
from src.exp001.veloedit_flux import (
    REPO_ROOT,
    VELOEDIT_INSPECTED_COMMIT,
    _file_sha256,
    _load_runtime,
    _tensor_sha256,
)
from src.exp002.core import (
    aggregate_seed_records,
    centered_velocity_metrics,
    map_target_sigmas,
)
from src.exp002.veloedit_flux import (
    EXPECTED_MODEL_REVISION,
    _audit_prepared_non_text_state,
    _load_exp002_analyzer_model,
    _prediction_pair,
    _run_independent_trajectory,
)
from .core import paired_prompt_comparison, validate_atomic_conditions


REFERENCE = "Keep the image unchanged."
CASES = {
    "v03_atomic": {
        "image": str(REPO_ROOT / "assets" / "exp001" / "v03_input.png"),
        "clause_a": "change the black mug to a white mug",
        "clause_b": "change the beige vase with flowers to a blue vase",
        "seeds": (42, 43, 44, 45, 46),
        "role": "EXP-003A paired prompt-template control",
    },
    "pinkballoon": {
        "image": str(
            REPO_ROOT / "external" / "SplitFlow" / "example_images" / "pinkballoon.jpg"
        ),
        "clause_a": "change the red balloon to a purple balloon",
        "clause_b": "change the blue suitcase to a yellow suitcase",
        "seeds": (42, 43, 44),
        "role": "EXP-003B different-category replication",
    },
    "elk": {
        "image": str(
            REPO_ROOT / "external" / "SplitFlow" / "example_images" / "elk.jpg"
        ),
        "clause_a": "change the left elk's fur to white",
        "clause_b": "change the right elk's fur to dark brown",
        "seeds": (42, 43, 44),
        "role": "EXP-003B same-category instance-binding replication",
    },
}
DEFAULT_TARGET_SIGMAS = (1.00, 0.75, 0.50, 0.10)
EXPECTED_PROBE_INDICES = (0, 9, 17, 23)
DEFAULT_ORDER_CHECK_INDICES = (0, 17)


def conditions_for(case: str) -> dict[str, str]:
    spec = CASES[case]
    a = str(spec["clause_a"])
    b = str(spec["clause_b"])
    conditions = {"0": REFERENCE, "A": a, "B": b, "AB": f"{a} and {b}"}
    validate_conditions(conditions)
    validate_atomic_conditions(conditions, a, b)
    return conditions


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    temporary.replace(path)


def _prepare_conditions(
    torch: Any,
    analyzer: Any,
    image: Any,
    seed: int,
    steps: int,
    conditions: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for label in CONDITION_ORDER_FORWARD:
            prepared[label] = analyzer._prepare_inputs(
                image=image,
                prompt=conditions[label],
                num_inference_steps=steps,
                seed=seed,
            )
    return prepared, _audit_prepared_non_text_state(torch, prepared)


def _single_order_predictions(
    torch: Any, prepared: Mapping[str, Mapping[str, Any]], shared_z_t: Any, sigma: Any
) -> tuple[dict[str, VelocityPair], list[dict[str, Any]]]:
    snapshot = shared_z_t.detach().clone()
    predictions: dict[str, VelocityPair] = {}
    calls: list[dict[str, Any]] = []
    for label in CONDITION_ORDER_FORWARD:
        model_input = snapshot.detach().clone()
        before = model_input.detach().clone()
        pair = _prediction_pair(
            prepared[label]["v_pred_fn"](model_input, sigma.detach().clone())
        )
        if not torch.equal(model_input, before) or not torch.equal(
            shared_z_t, snapshot
        ):
            raise RuntimeError(f"shared latent mutated while evaluating {label}")
        for field in ("raw_conditional_model_velocity", "effective_velocity"):
            tensor = getattr(pair, field)
            if tuple(tensor.shape) != tuple(snapshot.shape):
                raise ValueError(
                    f"{label}/{field} shape {tuple(tensor.shape)} != latent shape {tuple(snapshot.shape)}"
                )
        delta = model_input.detach().float() - snapshot.detach().float()
        calls.append(
            {
                "condition": label,
                "max_absolute_difference_from_shared_z_t": float(
                    delta.abs().max().item()
                ),
                "shape": list(model_input.shape),
                "dtype": str(model_input.dtype),
            }
        )
        predictions[label] = pair
    return predictions, calls


def _evaluate_probe(
    torch: Any,
    prepared: Mapping[str, Mapping[str, Any]],
    shared_z_t: Any,
    sigma: Any,
    *,
    seed: int,
    target_sigma: float,
    schedule_index: int,
    order_check: bool,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> tuple[dict[str, Any], Any | None]:
    input_calls: list[dict[str, Any]] = []
    order_metrics: dict[str, Any] = {}
    blocking_reasons: list[str] = []

    if order_check:

        def predict(label: str, model_z: Any, model_sigma: Any) -> VelocityPair:
            delta = model_z.detach().float() - shared_z_t.detach().float()
            input_calls.append(
                {
                    "condition": label,
                    "max_absolute_difference_from_shared_z_t": float(
                        delta.abs().max().item()
                    ),
                    "shape": list(model_z.shape),
                    "dtype": str(model_z.dtype),
                }
            )
            return _prediction_pair(prepared[label]["v_pred_fn"](model_z, model_sigma))

        evaluation = evaluate_shared_state(
            predict,
            shared_z_t,
            sigma,
            max_abs_tolerance=max_abs_tolerance,
            relative_l2_tolerance=relative_l2_tolerance,
        )
        predictions = evaluation.predictions_by_order["order_1"]
        order_metrics = evaluation.order_metrics
        blocking_reasons.extend(evaluation.blocking_reasons)
        blocked = evaluation.blocked
    else:
        predictions, input_calls = _single_order_predictions(
            torch, prepared, shared_z_t, sigma
        )
        blocked = False

    effective = {
        label: predictions[label].effective_velocity
        for label in CONDITION_ORDER_FORWARD
    }
    computation_error = None
    try:
        metrics, _ = centered_velocity_metrics(effective, torch_module=torch)
    except (ValueError, RuntimeError) as exc:
        metrics = {}
        computation_error = str(exc)

    all_predictions_finite = all(
        bool(torch.isfinite(getattr(pair, field)).all().item())
        for pair in predictions.values()
        for field in ("raw_conditional_model_velocity", "effective_velocity")
    )
    scalar_metrics_finite = bool(metrics) and all(
        math.isfinite(float(value)) for value in metrics.values()
    )
    input_max_abs = max(
        item["max_absolute_difference_from_shared_z_t"] for item in input_calls
    )
    if input_max_abs != 0.0:
        blocking_reasons.append(
            f"model input differed from shared z_t (max_abs={input_max_abs})"
        )
    if not all_predictions_finite:
        blocking_reasons.append("non-finite prediction tensor")
    if not scalar_metrics_finite:
        blocking_reasons.append("non-finite scalar metric")
    if computation_error:
        blocking_reasons.append(
            f"FP32 diagnostic computation failed: {computation_error}"
        )
    valid = (
        not blocked
        and input_max_abs == 0.0
        and all_predictions_finite
        and scalar_metrics_finite
        and computation_error is None
    )

    record = {
        "seed": seed,
        "carrier_type": "AB",
        "prompt_identifiers": list(CONDITION_ORDER_FORWARD),
        "target_sigma": target_sigma,
        "actual_sigma": float(sigma.item()),
        "schedule_index": schedule_index,
        "model_timestep": float(sigma.item()),
        "shared_z_t": {
            "sha256": _tensor_sha256(shared_z_t),
            "shape": list(shared_z_t.shape),
            "dtype": str(shared_z_t.dtype),
            "maximum_input_difference": input_max_abs,
            "per_call": input_calls,
        },
        "order_check": {
            "performed": order_check,
            "order_1": list(CONDITION_ORDER_FORWARD),
            "order_2": ["AB", "B", "A", "0"] if order_check else None,
            "metrics": order_metrics if order_check else None,
            "blocked": blocked,
        },
        "prediction_contract": {
            label: {
                "raw_shape": list(
                    predictions[label].raw_conditional_model_velocity.shape
                ),
                "raw_dtype": str(
                    predictions[label].raw_conditional_model_velocity.dtype
                ),
                "effective_shape": list(predictions[label].effective_velocity.shape),
                "effective_dtype": str(predictions[label].effective_velocity.dtype),
                "raw_equals_effective": bool(
                    torch.equal(
                        predictions[label].raw_conditional_model_velocity,
                        predictions[label].effective_velocity,
                    )
                ),
            }
            for label in CONDITION_ORDER_FORWARD
        },
        "all_prediction_tensors_finite": all_predictions_finite,
        "valid": valid,
        "invalid_reason": blocking_reasons or None,
        "metrics": metrics,
        "arithmetic": {
            "primary": "r_ref = dAB - dA - dB",
            "dA": "fp32(v_A) - fp32(v_0)",
            "dB": "fp32(v_B) - fp32(v_0)",
            "dAB": "fp32(v_AB) - fp32(v_0)",
        },
    }
    return record, predictions["AB"].effective_velocity if valid else None


def _save_contact_sheet(Image: Any, outputs: Mapping[str, str], target: Path) -> None:
    images = [
        Image.open(outputs[label]).convert("RGB") for label in CONDITION_ORDER_FORWARD
    ]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 2) * width, (index // 2) * height))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=95)


def _manipulation_outputs(
    torch: Any,
    Image: Any,
    analyzer: Any,
    euler_step: Any,
    prepared: Mapping[str, Mapping[str, Any]],
    seed_dir: Path,
    carrier_final: Any,
) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    for label in ("0", "A", "B"):
        path = seed_dir / "manipulation" / f"condition_{label}.png"
        _run_independent_trajectory(torch, analyzer, euler_step, prepared[label], path)
        outputs[label] = str(path)
    ab_path = seed_dir / "manipulation" / "condition_AB.png"
    ab_path.parent.mkdir(parents=True, exist_ok=True)
    carrier_final.save(ab_path)
    outputs["AB"] = str(ab_path)
    _save_contact_sheet(Image, outputs, seed_dir / "contact_sheet.jpg")
    report = {
        "status": "PENDING_HUMAN_REVIEW",
        "allowed_labels": ["PASS", "PARTIAL", "FAIL"],
        "outputs": outputs,
        "AB_output_source": "decoded final state of the unmodified AB carrier; probes are read-only",
        "note": "No VLM scoring. Numerical records are not filtered by pending qualitative labels.",
    }
    _json_write(seed_dir / "manipulation_check.json", report)
    return report


def run_seed(
    torch: Any,
    Image: Any,
    analyzer: Any,
    euler_step: Any,
    image: Any,
    args: argparse.Namespace,
    conditions: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "complete"
            and prior.get("case") == args.case
            and prior.get("conditions") == conditions
        ):
            print(f"[EXP-003] seed {seed}: reusing complete checkpoint")
            return prior

    prepared, invariant_report = _prepare_conditions(
        torch, analyzer, image, seed, args.num_inference_steps, conditions
    )
    carrier = prepared["AB"]
    schedule = carrier["sigma_schedule"]
    schedule_values = [float(value.item()) for value in schedule]
    probes, duplicates = map_target_sigmas(schedule_values, args.target_sigmas)
    mapped_indices = tuple(probe.schedule_index for probe in probes)
    if mapped_indices != EXPECTED_PROBE_INDICES:
        raise RuntimeError(
            f"reduced probe mapping changed: expected {EXPECTED_PROBE_INDICES}, got {mapped_indices}"
        )
    probes_by_index = {probe.schedule_index: probe for probe in probes}
    z = carrier["latents"].detach().clone()
    record: dict[str, Any] = {
        "experiment": "EXP-003",
        "case": args.case,
        "conditions": dict(conditions),
        "seed": seed,
        "status": "running",
        "carrier_trajectory": "AB",
        "initial_carrier_sha256": _tensor_sha256(z),
        "sigma_schedule": schedule_values,
        "mapped_probes": [probe.to_dict() for probe in probes],
        "duplicate_probe_mappings": duplicates,
        "non_text_invariants": invariant_report,
        "probes": [],
    }
    _json_write(summary_path, record)
    if not invariant_report["valid"]:
        record["status"] = "invalid_non_text_invariants"
        record["invalid_reasons"] = invariant_report["invalid_reasons"]
        _json_write(summary_path, record)
        return record

    with torch.inference_mode():
        for index in range(len(schedule) - 1):
            sigma, sigma_next = schedule[index], schedule[index + 1]
            if index in probes_by_index:
                probe = probes_by_index[index]
                do_order_check = args.full_order_validation or (
                    seed == args.order_check_seed and index in args.order_check_indices
                )
                probe_record, carrier_velocity = _evaluate_probe(
                    torch,
                    prepared,
                    z.detach().clone(),
                    sigma,
                    seed=seed,
                    target_sigma=probe.target_sigma,
                    schedule_index=index,
                    order_check=do_order_check,
                    max_abs_tolerance=args.max_abs_tolerance,
                    relative_l2_tolerance=args.relative_l2_tolerance,
                )
                record["probes"].append(probe_record)
                _json_write(
                    seed_dir / "probes" / f"step_{index:04d}.json", probe_record
                )
                if carrier_velocity is None:
                    record["status"] = "blocked_measurement"
                    record["blocking_step"] = index
                    _json_write(summary_path, record)
                    return record
            else:
                carrier_velocity = carrier["v_pred_fn"](z, sigma)
            z, _ = euler_step(
                z, carrier_velocity, float(sigma.item()), float(sigma_next.item())
            )
            z = z.to(dtype=carrier["latents"].dtype)

        record["status"] = "complete"
        record["final_carrier_sha256"] = _tensor_sha256(z)
        carrier_image = analyzer._decode_latents(z, carrier["height"], carrier["width"])
    record["manipulation_check"] = _manipulation_outputs(
        torch, Image, analyzer, euler_step, prepared, seed_dir, carrier_image
    )
    _json_write(summary_path, record)
    return record


def _load_legacy_records(directory: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        path = directory / f"seed_{seed}" / "seed_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing paired EXP-002 record: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def run_exp003(args: argparse.Namespace) -> dict[str, Any]:
    torch, Image, Analyzer, flux_config, euler_step = _load_runtime()
    spec = CASES[args.case]
    conditions = conditions_for(args.case)
    expected_seeds = tuple(spec["seeds"])
    if any(seed not in expected_seeds for seed in args.seeds):
        raise ValueError(
            f"case {args.case} preregistered seeds are {expected_seeds}, got {args.seeds}"
        )
    image_path = Path(args.image or str(spec["image"])).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    config = flux_config()
    config.model.path = args.model_path
    config.model.dtype = args.dtype
    config.sampling.num_inference_steps = args.num_inference_steps
    config.sampling.guidance_scale = args.guidance_scale
    config.sampling.first_step_align_steps = args.first_step_align_steps
    analyzer = Analyzer(config, device=args.device, save_tensors=False, lora_path=None)
    _load_exp002_analyzer_model(torch, analyzer, args)
    analyzer.model_loaded = True
    image = Image.open(image_path).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "EXP-003",
        "case": args.case,
        "study_role": spec["role"],
        "conditions": conditions,
        "condition_0_construction": "The exact EXP-002 preservation instruction; not an unconditional branch and not a zero-velocity claim.",
        "source_image": str(image_path),
        "source_image_sha256": _file_sha256(image_path),
        "seeds": list(args.seeds),
        "canonical_case_seeds": list(expected_seeds),
        "target_sigmas": list(args.target_sigmas),
        "expected_schedule_indices": list(EXPECTED_PROBE_INDICES),
        "order_validation": {
            "policy": "full" if args.full_order_validation else "targeted spot checks",
            "seed": args.order_check_seed,
            "schedule_indices": list(args.order_check_indices),
            "escalation": "Any failed spot check blocks interpretation and requires full validation.",
        },
        "model_family": "FLUX.1-Kontext",
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "veloedit_inspected_commit": VELOEDIT_INSPECTED_COMMIT,
        "dtype": args.dtype,
        "guidance_scale": args.guidance_scale,
        "num_requested_inference_steps": args.num_inference_steps,
        "first_step_align_steps": args.first_step_align_steps,
        "cpu_offload_enabled": args.cpu_offload,
        "intervention_enabled": False,
        "attention_analysis_enabled": False,
        "prediction_levels": {
            "raw_conditional_model_velocity": "generated-token tensor directly returned by the transformer closure",
            "effective_velocity": "tensor passed to Euler after post-transformer operations",
            "post_transformer_guidance_applied": False,
            "raw_equals_effective_by_construction": True,
        },
    }
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError("model revision differs from EXP-001/002")
    _json_write(output_dir / "protocol.json", protocol)

    records = []
    for seed in args.seeds:
        print(f"[EXP-003] {args.case} seed {seed}: preparing and measuring")
        record = run_seed(
            torch, Image, analyzer, euler_step, image, args, conditions, seed
        )
        records.append(record)
        if record["status"] != "complete":
            print(
                f"[EXP-003][BLOCKED] seed {seed}: {record['status']}", file=sys.stderr
            )
            break
    aggregate = aggregate_seed_records(records)
    summary: dict[str, Any] = {
        **protocol,
        "status": (
            "complete"
            if len(records) == len(args.seeds)
            and all(record["status"] == "complete" for record in records)
            else "blocked_or_incomplete"
        ),
        "seed_status": {str(record["seed"]): record["status"] for record in records},
        "aggregate": aggregate,
    }
    _json_write(output_dir / "aggregate.json", summary)
    if args.case == "v03_atomic":
        legacy_records = _load_legacy_records(Path(args.legacy_results_dir), args.seeds)
        comparison = paired_prompt_comparison(legacy_records, records)
        _json_write(output_dir / "paired_legacy_vs_atomic.json", comparison)
        summary["paired_comparison_path"] = str(
            output_dir / "paired_legacy_vs_atomic.json"
        )
    return summary


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXP-003 prompt control and minimal cross-image replication"
    )
    parser.add_argument("--case", choices=tuple(CASES), default="v03_atomic")
    parser.add_argument(
        "--image", default=None, help="Override only for an exact approved case image."
    )
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--first-step-align-steps", type=int, default=4)
    parser.add_argument("--seeds", type=_comma_ints, default=(42,))
    parser.add_argument(
        "--target-sigmas", type=_comma_floats, default=DEFAULT_TARGET_SIGMAS
    )
    parser.add_argument("--order-check-seed", type=int, default=42)
    parser.add_argument(
        "--order-check-indices", type=_comma_ints, default=DEFAULT_ORDER_CHECK_INDICES
    )
    parser.add_argument("--full-order-validation", action="store_true")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--legacy-results-dir", default="results/EXP-002/v03_seed_robustness"
    )
    parser.add_argument("--output-dir", default="results/EXP-003/v03_prompt_control")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        "[EXP-003] condition 0 is the explicit preservation instruction; it is not unconditional."
    )
    summary = run_exp003(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
