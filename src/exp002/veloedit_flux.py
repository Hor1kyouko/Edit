"""EXP-002 seed-robustness runner for VeloEdit FLUX.1-Kontext.

This project-owned adapter imports ``external/VeloEdit`` as a read-only
dependency.  It changes only the AB carrier seed across repeats.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.exp001.core import (
    CONDITION_ORDER_FORWARD,
    OrderDependenceError,
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
from .core import (
    DEFAULT_TARGET_SIGMAS,
    aggregate_seed_records,
    centered_velocity_metrics,
    map_target_sigmas,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_MODEL_REVISION = "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"
CONDITIONS = {
    "0": "Keep the image unchanged.",
    "A": "change the black mug to a white mug while keeping the beige vase with flowers unchanged",
    "B": "keep the black mug unchanged and change the beige vase with flowers to a blue vase",
    "AB": "change the black mug to a white mug and change the beige vase with flowers to a blue vase",
}
TEXT_INDEPENDENT_CLOSURE_FIELDS = ("image_latents", "latent_ids", "guidance", "text_ids")


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _closure_values(function: Any) -> dict[str, Any]:
    closure = inspect.getclosurevars(function).nonlocals
    return dict(closure)


def _tensor_equality_report(torch: Any, first: Any, second: Any) -> dict[str, Any]:
    if first is None or second is None:
        equal = first is None and second is None
        return {"equal": equal, "shape": None, "dtype": None, "max_absolute_difference": 0.0 if equal else math.inf}
    delta = first.detach().float() - second.detach().float()
    return {
        "equal": bool(torch.equal(first, second)),
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "max_absolute_difference": float(delta.abs().max().item()),
    }


def _audit_prepared_non_text_state(torch: Any, prepared: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit both returned state and text-independent closure captures."""

    reference_prepared = prepared["AB"]
    reasons: list[str] = []
    returned: dict[str, Any] = {}
    for field in ("latents", "reference_latent", "sigma_schedule"):
        returned[field] = {}
        for label in CONDITION_ORDER_FORWARD:
            comparison = _tensor_equality_report(
                torch, prepared[label][field], reference_prepared[field]
            )
            returned[field][label] = comparison
            if not comparison["equal"]:
                reasons.append(
                    f"returned prepared tensor {field} differs for {label} and AB"
                )
    for field in ("height", "width", "original_height", "original_width"):
        returned[field] = {}
        for label in CONDITION_ORDER_FORWARD:
            equal = prepared[label][field] == reference_prepared[field]
            returned[field][label] = {
                "equal": equal,
                "value": prepared[label][field],
            }
            if not equal:
                reasons.append(
                    f"returned prepared scalar {field} differs for {label} and AB"
                )
    captures = {label: _closure_values(prepared[label]["v_pred_fn"]) for label in CONDITION_ORDER_FORWARD}
    report: dict[str, Any] = {"returned_fields": returned, "closure_fields": {}}
    for field in TEXT_INDEPENDENT_CLOSURE_FIELDS:
        reference = captures["AB"].get(field)
        report["closure_fields"][field] = {}
        for label in CONDITION_ORDER_FORWARD:
            current = captures[label].get(field)
            comparison = _tensor_equality_report(torch, current, reference)
            report["closure_fields"][field][label] = comparison
            if not comparison["equal"]:
                reasons.append(
                    f"text-independent closure field {field} differs for {label} and AB"
                )
    report["valid"] = not reasons
    report["invalid_reasons"] = reasons or None
    return report


def _prepare_conditions(torch: Any, analyzer: Any, image: Any, seed: int, steps: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for label in CONDITION_ORDER_FORWARD:
            prepared[label] = analyzer._prepare_inputs(
                image=image,
                prompt=CONDITIONS[label],
                num_inference_steps=steps,
                seed=seed,
            )
    return prepared, _audit_prepared_non_text_state(torch, prepared)


def _prediction_pair(tensor: Any) -> VelocityPair:
    # FLUX.1-Kontext uses embedded guidance. No post-transformer CFG or other
    # arithmetic is applied between this generated-token slice and Euler.
    return VelocityPair(
        raw_conditional_model_velocity=tensor,
        effective_velocity=tensor,
    )


def _load_exp002_analyzer_model(torch: Any, analyzer: Any, args: argparse.Namespace) -> None:
    """Load once, pinning the hub revision when the path is a model ID."""

    if not args.cpu_offload:
        # The upstream analyzer loader cannot accept a revision. Require a
        # local artifact rather than silently resolving a moving hub branch.
        if not Path(args.model_path).is_dir():
            raise RuntimeError(
                "non-offload EXP-002 requires a local --model-path so the "
                "fixed revision is not silently ignored"
            )
        analyzer.load_model()
        analyzer._exp002_revision_enforcement = (
            "local artifact provenance declared from EXP-001"
        )
        return

    from diffusers import FluxKontextPipeline

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype_map[args.dtype]}
    model_path = Path(args.model_path)
    revision_enforcement = "local artifact provenance declared from EXP-001"
    if not model_path.is_dir():
        load_kwargs["revision"] = args.model_revision
        revision_enforcement = "hub revision passed to from_pretrained"
    print(f"[EXP-002] loading {args.model_path} ({revision_enforcement})")
    analyzer.pipeline = FluxKontextPipeline.from_pretrained(
        args.model_path, **load_kwargs
    )
    if analyzer.pipeline.text_encoder is not None:
        analyzer.pipeline.text_encoder.to(dtype=dtype_map[args.dtype])
    if analyzer.pipeline.text_encoder_2 is not None:
        analyzer.pipeline.text_encoder_2.to(dtype=dtype_map[args.dtype])
    analyzer.pipeline.enable_sequential_cpu_offload(device=args.device)
    analyzer._exp002_revision_enforcement = revision_enforcement


def _save_optional_probe_tensors(torch: Any, directory: Path, evaluation: Any, centered: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for order, predictions in evaluation.predictions_by_order.items():
        for label, pair in predictions.items():
            target = directory / order / label
            target.mkdir(parents=True, exist_ok=True)
            torch.save(pair.raw_conditional_model_velocity.detach().cpu(), target / "raw_conditional_model_velocity.pt")
            torch.save(pair.effective_velocity.detach().cpu(), target / "effective_velocity.pt")
    centered_dir = directory / "centered_fp32"
    centered_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in centered.items():
        torch.save(tensor.detach().cpu(), centered_dir / f"{name}.pt")


def _evaluate_probe(
    torch: Any,
    prepared: Mapping[str, Mapping[str, Any]],
    shared_z_t: Any,
    sigma: Any,
    *,
    seed: int,
    target_sigma: float,
    schedule_index: int,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
    save_tensors: bool,
    tensor_dir: Path,
) -> tuple[dict[str, Any], Any | None]:
    input_differences: list[dict[str, Any]] = []

    def predict(label: str, model_z: Any, model_sigma: Any) -> VelocityPair:
        delta = model_z.detach().float() - shared_z_t.detach().float()
        input_differences.append(
            {
                "condition": label,
                "max_absolute_difference_from_shared_z_t": float(delta.abs().max().item()),
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
    prediction_tensors = {
        label: evaluation.predictions_by_order["order_1"][label].effective_velocity
        for label in CONDITION_ORDER_FORWARD
    }
    centered: dict[str, Any] = {}
    computation_error = None
    try:
        metrics, centered = centered_velocity_metrics(
            prediction_tensors, torch_module=torch
        )
    except (ValueError, RuntimeError) as exc:
        metrics = {}
        computation_error = str(exc)
    actual_sigma = float(sigma.item())
    input_max_abs = max(item["max_absolute_difference_from_shared_z_t"] for item in input_differences)
    all_predictions_finite = all(
        bool(torch.isfinite(getattr(pair, field)).all().item())
        for predictions in evaluation.predictions_by_order.values()
        for pair in predictions.values()
        for field in (
            "raw_conditional_model_velocity",
            "effective_velocity",
        )
    )
    finite = (
        bool(metrics)
        and all(math.isfinite(float(value)) for value in metrics.values())
        and all_predictions_finite
    )
    valid = (
        not evaluation.blocked
        and input_max_abs == 0.0
        and finite
        and computation_error is None
    )
    invalid_reasons = list(evaluation.blocking_reasons)
    if input_max_abs != 0.0:
        invalid_reasons.append(f"model input differed from shared z_t (max_abs={input_max_abs})")
    if not finite:
        invalid_reasons.append("non-finite scalar metric")
    if computation_error is not None:
        invalid_reasons.append(
            f"FP32 diagnostic computation failed: {computation_error}"
        )

    record = {
        "seed": seed,
        "carrier_type": "AB",
        "prompt_identifiers": ["0", "A", "B", "AB"],
        "target_sigma": target_sigma,
        "actual_sigma": actual_sigma,
        "schedule_index": schedule_index,
        "model_timestep": actual_sigma,
        "model_timestep_definition": "normalized timestep passed to the transformer after the analyzer's multiply-and-divide-by-1000 conversion",
        "shared_z_t": {
            "sha256": _tensor_sha256(shared_z_t),
            "shape": list(shared_z_t.shape),
            "dtype": str(shared_z_t.dtype),
            "maximum_input_difference": input_max_abs,
            "per_call": input_differences,
        },
        "orders": {"order_1": ["0", "A", "B", "AB"], "order_2": ["AB", "B", "A", "0"]},
        "order_metrics": evaluation.order_metrics,
        "blocked_measurement": evaluation.blocked,
        "valid": valid,
        "invalid_reason": invalid_reasons or None,
        "metrics": metrics,
        "all_prediction_tensors_finite": all_predictions_finite,
        "arithmetic": {
            "primary": "r_ref = dAB - dA - dB",
            "dA": "fp32(v_A) - fp32(v_0)",
            "dB": "fp32(v_B) - fp32(v_0)",
            "dAB": "fp32(v_AB) - fp32(v_0)",
            "legacy_bf16": "v_AB - v_A - v_B + v_0 in prediction dtype, then cast to FP32",
        },
    }
    if save_tensors and centered:
        _save_optional_probe_tensors(torch, tensor_dir, evaluation, centered)
    carrier_velocity = evaluation.carrier_effective_velocity if valid else None
    return record, carrier_velocity


def _run_independent_trajectory(torch: Any, analyzer: Any, euler_step: Any, prepared: Mapping[str, Any], output_path: Path) -> None:
    z = prepared["latents"].detach().clone()
    schedule = prepared["sigma_schedule"]
    with torch.inference_mode():
        for index in range(len(schedule) - 1):
            sigma = schedule[index]
            sigma_next = schedule[index + 1]
            velocity = prepared["v_pred_fn"](z, sigma)
            z, _ = euler_step(z, velocity, float(sigma.item()), float(sigma_next.item()))
            z = z.to(dtype=prepared["latents"].dtype)
        image = analyzer._decode_latents(z, prepared["height"], prepared["width"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _run_manipulation_outputs(torch: Any, analyzer: Any, euler_step: Any, prepared: Mapping[str, Mapping[str, Any]], seed_dir: Path) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    for label in CONDITION_ORDER_FORWARD:
        output_path = seed_dir / "manipulation" / f"condition_{label}.png"
        _run_independent_trajectory(torch, analyzer, euler_step, prepared[label], output_path)
        outputs[label] = str(output_path)
    report = {
        "status": "PENDING_HUMAN_REVIEW",
        "allowed_labels": ["PASS", "PARTIAL", "FAIL"],
        "review_unit": "each seed separately",
        "criteria": {
            "0": "source content remains materially unchanged",
            "A": "mug becomes white while the beige vase with flowers remains unchanged",
            "B": "mug remains black while the vase becomes blue",
            "AB": "mug becomes white and vase becomes blue",
        },
        "outputs": outputs,
        "note": "No VLM or automatic semantic judge is used. Numerical records are not filtered by this pending qualitative review.",
    }
    _json_write(seed_dir / "manipulation_check.json", report)
    return report


def run_seed(torch: Any, analyzer: Any, euler_step: Any, image: Any, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    prepared, invariant_report = _prepare_conditions(torch, analyzer, image, seed, args.num_inference_steps)
    carrier = prepared["AB"]
    schedule = carrier["sigma_schedule"]
    schedule_values = [float(value.item()) for value in schedule]
    probes, duplicate_mappings = map_target_sigmas(schedule_values, args.target_sigmas)
    probes_by_index = {probe.schedule_index: probe for probe in probes}
    z = carrier["latents"].detach().clone()
    record: dict[str, Any] = {
        "experiment": "EXP-002",
        "seed": seed,
        "status": "running",
        "carrier_trajectory": "AB",
        "initial_carrier_sha256": _tensor_sha256(z),
        "sigma_schedule": schedule_values,
        "mapped_probes": [probe.to_dict() for probe in probes],
        "duplicate_probe_mappings": duplicate_mappings,
        "non_text_invariants": invariant_report,
        "probes": [],
    }
    _json_write(seed_dir / "seed_summary.json", record)

    if not invariant_report["valid"]:
        record["status"] = "invalid_non_text_invariants"
        record["invalid_reasons"] = invariant_report["invalid_reasons"]
        record["manipulation_check"] = _run_manipulation_outputs(
            torch, analyzer, euler_step, prepared, seed_dir
        )
        _json_write(seed_dir / "seed_summary.json", record)
        return record

    with torch.inference_mode():
        for index in range(len(schedule) - 1):
            sigma = schedule[index]
            sigma_next = schedule[index + 1]
            if index in probes_by_index:
                probe = probes_by_index[index]
                probe_record, carrier_velocity = _evaluate_probe(
                    torch,
                    prepared,
                    z.detach().clone(),
                    sigma,
                    seed=seed,
                    target_sigma=probe.target_sigma,
                    schedule_index=index,
                    max_abs_tolerance=args.max_abs_tolerance,
                    relative_l2_tolerance=args.relative_l2_tolerance,
                    save_tensors=args.save_velocity_tensors,
                    tensor_dir=seed_dir / "probe_tensors" / f"step_{index:04d}",
                )
                record["probes"].append(probe_record)
                _json_write(seed_dir / "probes" / f"step_{index:04d}.json", probe_record)
                if carrier_velocity is None:
                    record["status"] = "blocked_measurement"
                    record["blocking_step"] = index
                    break
            else:
                carrier_velocity = carrier["v_pred_fn"](z, sigma)
            z, _ = euler_step(z, carrier_velocity, float(sigma.item()), float(sigma_next.item()))
            z = z.to(dtype=carrier["latents"].dtype)
        else:
            record["status"] = "complete"
            record["final_carrier_sha256"] = _tensor_sha256(z)

    # Manipulation checks are independent native trajectories, not the measured
    # AB carrier, and are generated even if order dependence blocks measurement.
    record["manipulation_check"] = _run_manipulation_outputs(torch, analyzer, euler_step, prepared, seed_dir)
    _json_write(seed_dir / "seed_summary.json", record)
    return record


def run_exp002(args: argparse.Namespace) -> dict[str, Any]:
    torch, Image, Analyzer, flux_config, euler_step = _load_runtime()
    validate_conditions(CONDITIONS)
    config = flux_config()
    config.model.path = args.model_path
    config.model.dtype = args.dtype
    config.sampling.num_inference_steps = args.num_inference_steps
    config.sampling.guidance_scale = args.guidance_scale
    config.sampling.first_step_align_steps = args.first_step_align_steps
    analyzer = Analyzer(config, device=args.device, save_tensors=False, lora_path=None)
    _load_exp002_analyzer_model(torch, analyzer, args)
    analyzer.model_loaded = True
    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": "EXP-002",
        "scientific_question": "Is centered reference-corrected non-additivity robust to AB carrier seed?",
        "only_manipulated_variable": "AB carrier random seed",
        "seeds": list(args.seeds),
        "conditions": CONDITIONS,
        "condition_0_construction": "The exact explicit preservation instruction 'Keep the image unchanged.'; it is neither an empty prompt nor an unconditional branch and is not claimed to produce zero velocity.",
        "linguistic_asymmetry_confound": "A and B contain explicit keep-unchanged clauses while AB does not. Prompts are intentionally not rewritten, so this remains a known confound.",
        "source_image": str(image_path),
        "source_image_sha256": _file_sha256(image_path),
        "model_family": "FLUX.1-Kontext",
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "expected_model_revision_from_EXP001": EXPECTED_MODEL_REVISION,
        "veloedit_inspected_commit": VELOEDIT_INSPECTED_COMMIT,
        "dtype": args.dtype,
        "guidance_scale": args.guidance_scale,
        "num_requested_inference_steps": args.num_inference_steps,
        "first_step_align_steps": args.first_step_align_steps,
        "target_sigmas": list(args.target_sigmas),
        "cpu_offload_enabled": args.cpu_offload,
        "model_revision_enforcement": getattr(
            analyzer,
            "_exp002_revision_enforcement",
            "local analyzer load; local model directory required",
        ),
        "intervention_enabled": False,
        "attention_analysis_enabled": False,
        "weights_or_adapters": None,
        "full_velocity_tensors_saved": args.save_velocity_tensors,
        "prediction_levels": {
            "raw_conditional_model_velocity": "generated-token tensor directly returned by the transformer closure",
            "effective_velocity": "tensor passed to Euler after post-transformer operations",
            "post_transformer_guidance_applied": False,
            "raw_equals_effective_by_construction": True,
        },
    }
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError("model revision differs from the fixed EXP-001 revision")
    _json_write(output_dir / "protocol.json", protocol)

    seed_records = []
    for seed in args.seeds:
        print(f"[EXP-002] seed {seed}: preparing and measuring")
        seed_records.append(run_seed(torch, analyzer, euler_step, image, args, seed))
    aggregate = aggregate_seed_records(seed_records)
    summary = {**protocol, "status": "complete" if all(record["status"] == "complete" for record in seed_records) else "blocked_or_incomplete", "seed_status": {str(record["seed"]): record["status"] for record in seed_records}, "aggregate": aggregate}
    _json_write(output_dir / "aggregate.json", summary)
    return summary


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-002 seed robustness of centered non-additivity")
    parser.add_argument("--image", default=str(REPO_ROOT / "assets" / "exp001" / "v03_input.png"))
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--first-step-align-steps", type=int, default=4)
    parser.add_argument("--seeds", type=_comma_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--target-sigmas", type=_comma_floats, default=DEFAULT_TARGET_SIGMAS)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="outputs/EXP-002/v03_seed_robustness")
    parser.add_argument("--save-velocity-tensors", action="store_true", help="Off by default to avoid large outputs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if tuple(args.seeds) != DEFAULT_SEEDS:
        print("[EXP-002][WARNING] non-preregistered seed list; do not mix with the canonical aggregate", file=sys.stderr)
    print("[EXP-002] condition 0 is the explicit preservation instruction 'Keep the image unchanged.'; it is not unconditional and is not a zero-velocity claim.")
    try:
        summary = run_exp002(args)
    except OrderDependenceError as exc:
        print(f"[EXP-002][BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
