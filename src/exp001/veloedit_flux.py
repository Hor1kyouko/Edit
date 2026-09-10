"""Project-owned EXP-001 runner for VeloEdit FLUX.1-Kontext.

This module imports VeloEdit as a read-only reference dependency. It does not
patch or write anything below external/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .core import (
    CONDITION_ORDER_FORWARD,
    CONDITION_ORDER_REVERSE,
    CONDITION_ZERO_DEFAULT,
    OrderDependenceError,
    SharedStateEvaluation,
    VelocityPair,
    evaluate_shared_state,
    resolve_probe_steps,
    validate_conditions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external"
VELOEDIT_ROOT = EXTERNAL_ROOT / "VeloEdit"
VELOEDIT_INSPECTED_COMMIT = "ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a"


def _load_runtime():
    if not VELOEDIT_ROOT.is_dir():
        raise RuntimeError(f"VeloEdit reference repository not found: {VELOEDIT_ROOT}")
    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    try:
        import torch
        from PIL import Image
        from VeloEdit.analyzers.flux import FLUXVelocityAnalyzer
        from VeloEdit.config.config import flux_config
        from VeloEdit.core.sampler import euler_step
    except ImportError as exc:
        raise RuntimeError(
            "Missing VeloEdit runtime dependency. Install external/VeloEdit/"
            "requirements.txt in the intended experiment environment."
        ) from exc
    return torch, Image, FLUXVelocityAnalyzer, flux_config, euler_step


def _load_analyzer_model(torch: Any, analyzer: Any, args: argparse.Namespace) -> None:
    """Load the unchanged pipeline with optional execution-only CPU offload."""

    if not args.cpu_offload:
        analyzer.load_model()
        return

    from diffusers import FluxKontextPipeline

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    print(f"[FLUX] Using dtype: {dtype}")
    print(f"[FLUX] Loading model from {args.model_path} with CPU offload...")
    analyzer.pipeline = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    if analyzer.pipeline.text_encoder is not None:
        analyzer.pipeline.text_encoder.to(dtype=dtype)
    if analyzer.pipeline.text_encoder_2 is not None:
        analyzer.pipeline.text_encoder_2.to(dtype=dtype)
    analyzer.pipeline.enable_sequential_cpu_offload(device=args.device)
    print("[FLUX] Model loaded successfully with sequential CPU offload.")


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    as_bytes = contiguous.view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(as_bytes).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equal_prepared_state(torch: Any, prepared: Mapping[str, Dict[str, Any]]) -> None:
    """Verify every prompt preparation reconstructed the same non-text state."""

    reference = prepared["AB"]
    tensor_fields = ("latents", "reference_latent", "sigma_schedule")
    scalar_fields = ("height", "width", "original_height", "original_width")
    for condition in CONDITION_ORDER_FORWARD:
        current = prepared[condition]
        for field in tensor_fields:
            if not torch.equal(current[field], reference[field]):
                raise RuntimeError(
                    f"non-text invariant failed: {field} differs for {condition} and AB"
                )
        for field in scalar_fields:
            if current[field] != reference[field]:
                raise RuntimeError(
                    f"non-text invariant failed: {field} differs for {condition} and AB"
                )


def _scalar_tensor_summary(torch: Any, tensor: Any) -> Dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "max_absolute_value": float(value.abs().max().item()),
    }


def _cpu_for_save(tensor: Any, save_dtype: str) -> Any:
    torch = __import__("torch")
    value = tensor.detach().cpu()
    if save_dtype == "float16":
        return value.to(torch.float16)
    if save_dtype == "bfloat16":
        return value.to(torch.bfloat16)
    if save_dtype == "float32":
        return value.to(torch.float32)
    return value


def _save_probe(
    torch: Any,
    output_dir: Path,
    step_index: int,
    sigma: float,
    evaluation: SharedStateEvaluation,
    *,
    save_tensors: bool,
    save_dtype: str,
    metadata: Dict[str, Any],
) -> Path:
    step_dir = output_dir / f"step_{step_index:04d}_sigma_{sigma:.6f}"
    step_dir.mkdir(parents=True, exist_ok=True)

    tensor_summaries: Dict[str, Any] = {}
    for order_name, predictions in evaluation.predictions_by_order.items():
        tensor_summaries[order_name] = {}
        for condition, pair in predictions.items():
            tensor_summaries[order_name][condition] = {}
            for field_name in (
                "raw_conditional_model_velocity",
                "effective_velocity",
            ):
                tensor = getattr(pair, field_name)
                tensor_summaries[order_name][condition][field_name] = (
                    _scalar_tensor_summary(torch, tensor)
                )
                if save_tensors:
                    target = step_dir / order_name / condition / f"{field_name}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(_cpu_for_save(tensor, save_dtype), target)

    residual_summaries: Dict[str, Any] = {}
    for level, residuals in evaluation.residuals.items():
        residual_summaries[level] = {}
        for residual_name, tensor in residuals.items():
            residual_summaries[level][residual_name] = _scalar_tensor_summary(
                torch, tensor
            )
            if save_tensors:
                target = step_dir / "residuals" / level / f"{residual_name}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_cpu_for_save(tensor, save_dtype), target)

    report = {
        **metadata,
        "step_index": step_index,
        "sigma": sigma,
        "blocked_measurement": evaluation.blocked,
        "blocking_reasons": evaluation.blocking_reasons,
        "valid_for_scientific_analysis": not evaluation.blocked,
        "order_metrics": evaluation.order_metrics,
        "tensor_summaries": tensor_summaries,
        "residual_summaries": residual_summaries,
        "primary_diagnostic": {
            "prediction_level": "effective",
            "quantity": "r_ref",
            "formula": "v_AB - v_A - v_B + v_0",
            "semantics": "algebraic reference-corrected non-additivity residual; "
            "no interaction interpretation is assigned",
        },
    }
    report_path = step_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def run_exp001(args: argparse.Namespace) -> Dict[str, Any]:
    torch, Image, Analyzer, flux_config, euler_step = _load_runtime()
    conditions = {
        "0": args.prompt_0,
        "A": args.prompt_a,
        "B": args.prompt_b,
        "AB": args.prompt_ab,
    }
    validate_conditions(conditions)

    config = flux_config()
    config.model.path = args.model_path
    config.model.dtype = args.dtype
    config.sampling.num_inference_steps = args.num_inference_steps
    config.sampling.guidance_scale = args.guidance_scale
    config.sampling.first_step_align_steps = args.first_step_align_steps

    analyzer = Analyzer(config, device=args.device, save_tensors=False, lora_path=None)
    _load_analyzer_model(torch, analyzer, args)
    analyzer.model_loaded = True
    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")

    # VeloEdit's closure captures text and fixed condition-image state. Repeating
    # preparation with the same seed creates one closure per text condition; the
    # exact equality checks below make this acceptable rather than assumed.
    prepared: Dict[str, Dict[str, Any]] = {}
    with torch.inference_mode():
        for label in CONDITION_ORDER_FORWARD:
            prepared[label] = analyzer._prepare_inputs(
                image=image,
                prompt=conditions[label],
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
            )
    _assert_equal_prepared_state(torch, prepared)

    carrier = prepared["AB"]
    z = carrier["latents"].detach().clone()
    sigma_schedule = carrier["sigma_schedule"]
    num_transitions = len(sigma_schedule) - 1
    requested_steps: Sequence[str] = [
        item.strip() for item in args.probe_steps.split(",") if item.strip()
    ]
    probe_steps = resolve_probe_steps(num_transitions, requested_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_metadata: Dict[str, Any] = {
        "experiment": "EXP-001",
        "model_family": "FLUX.1-Kontext",
        "implementation_basis": "external/VeloEdit read-only adapter",
        "veloedit_inspected_commit": VELOEDIT_INSPECTED_COMMIT,
        "model_path": args.model_path,
        "device": args.device,
        "model_dtype": args.dtype,
        "cpu_offload_enabled": args.cpu_offload,
        "source_image": str(image_path),
        "source_image_sha256": _file_sha256(image_path),
        "carrier_trajectory": "AB",
        "conditions": conditions,
        "condition_0_construction": {
            "text": conditions["0"],
            "role": "explicit preservation-instruction reference",
            "is_empty_prompt": False,
            "is_unconditional_branch": False,
            "claimed_zero_velocity": False,
            "note": "FLUX.1-Kontext has no canonical null-edit condition in the "
            "inspected interface. This operational reference must be reported "
            "verbatim and may be overridden with --prompt-0.",
        },
        "prediction_levels": {
            "raw_conditional_model_velocity": "generated-token slice directly "
            "returned by the transformer for the condition; embedded guidance "
            "is already an input to the transformer",
            "effective_velocity": "tensor passed to VeloEdit's Euler update",
            "post_transformer_guidance_applied": False,
            "raw_equals_effective_by_construction": True,
        },
        "orders": {
            "order_1": list(CONDITION_ORDER_FORWARD),
            "order_2": list(CONDITION_ORDER_REVERSE),
        },
        "order_invariance_tolerances": {
            "max_absolute_difference": args.max_abs_tolerance,
            "relative_l2_difference": args.relative_l2_tolerance,
            "blocking_rule": "blocked if either metric exceeds its tolerance "
            "for either prediction level in any condition",
        },
        "intervention_enabled": False,
        "attention_analysis_enabled": False,
        "prepared_non_text_state_exact_equality_verified": True,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_requested_inference_steps": args.num_inference_steps,
        "num_effective_transitions": num_transitions,
        "probe_steps": probe_steps,
        "probe_alias_definition": {
            "early": 0,
            "middle": num_transitions // 2,
            "late": num_transitions - 1,
        },
        "initial_carrier_sha256": _tensor_sha256(z),
        "reference_latent_sha256": _tensor_sha256(carrier["reference_latent"]),
        "sigma_schedule": [float(value.item()) for value in sigma_schedule],
    }

    run_summary: Dict[str, Any] = {
        **base_metadata,
        "status": "running",
        "probe_reports": [],
    }
    summary_path = output_dir / "run_summary.json"

    with torch.inference_mode():
        for step_index in range(num_transitions):
            sigma = sigma_schedule[step_index]
            sigma_next = sigma_schedule[step_index + 1]
            sigma_value = float(sigma.item())
            sigma_next_value = float(sigma_next.item())

            if step_index in probe_steps:
                shared_z_t = z.detach().clone()
                shared_sha256 = _tensor_sha256(shared_z_t)

                def predict(condition: str, model_z: Any, model_sigma: Any) -> VelocityPair:
                    # FLUX Kontext uses embedded/distilled guidance. The generated
                    # token slice returned by this closure is directly integrated;
                    # there is no separate post-transformer CFG operation.
                    transformer_generated_tokens = prepared[condition]["v_pred_fn"](
                        model_z, model_sigma
                    )
                    return VelocityPair(
                        raw_conditional_model_velocity=transformer_generated_tokens,
                        effective_velocity=transformer_generated_tokens,
                    )

                evaluation = evaluate_shared_state(
                    predict,
                    shared_z_t,
                    sigma,
                    max_abs_tolerance=args.max_abs_tolerance,
                    relative_l2_tolerance=args.relative_l2_tolerance,
                )
                report_path = _save_probe(
                    torch,
                    output_dir,
                    step_index,
                    sigma_value,
                    evaluation,
                    save_tensors=not args.no_save_tensors,
                    save_dtype=args.save_dtype,
                    metadata={
                        **base_metadata,
                        "shared_z_t_sha256": shared_sha256,
                    },
                )
                run_summary["probe_reports"].append(str(report_path))
                if evaluation.blocked:
                    run_summary["status"] = "blocked_order_dependence"
                    run_summary["blocking_step"] = step_index
                    run_summary["blocking_reasons"] = evaluation.blocking_reasons
                    summary_path.write_text(
                        json.dumps(run_summary, indent=2), encoding="utf-8"
                    )
                    raise OrderDependenceError(
                        "EXP-001 blocked by condition-order dependence at "
                        f"step {step_index}; see {report_path}"
                    )
                carrier_velocity = evaluation.carrier_effective_velocity
            else:
                carrier_velocity = carrier["v_pred_fn"](z, sigma)

            z, _ = euler_step(
                z,
                carrier_velocity,
                sigma_value,
                sigma_next_value,
            )
            z = z.to(dtype=carrier["latents"].dtype)

    run_summary["status"] = "complete"
    run_summary["final_carrier_sha256"] = _tensor_sha256(z)
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXP-001 same-latent 0/A/B/AB probe for VeloEdit FLUX.1-Kontext"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-a", required=True)
    parser.add_argument("--prompt-b", required=True)
    parser.add_argument("--prompt-ab", required=True)
    parser.add_argument(
        "--prompt-0",
        default=CONDITION_ZERO_DEFAULT,
        help="Operational preservation reference; saved verbatim. This is not "
        "an unconditional or guaranteed zero-velocity condition.",
    )
    parser.add_argument(
        "--model-path", default="black-forest-labs/FLUX.1-Kontext-dev"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload pipeline components between CPU and GPU without changing "
        "model weights, dtype, sampling, or measured tensors.",
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--first-step-align-steps", type=int, default=4)
    parser.add_argument("--probe-steps", default="early,middle,late")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="outputs/EXP-001")
    parser.add_argument(
        "--save-dtype",
        choices=("original", "float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--no-save-tensors",
        action="store_true",
        help="Save scalar reports only. Raw/effective names remain in metrics JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        "[EXP-001] condition 0 is the explicit preservation instruction: "
        f"{args.prompt_0!r}. It is not an unconditional or zero-velocity branch."
    )
    try:
        summary = run_exp001(args)
    except OrderDependenceError as exc:
        print(f"[EXP-001][BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
