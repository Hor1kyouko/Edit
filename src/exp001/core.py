"""Model-independent contracts for EXP-001.

Tensor operations import PyTorch lazily so that the experiment specification and
CLI help remain inspectable on machines without the VeloEdit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


CONDITION_ZERO_DEFAULT = "Keep the image unchanged."
CONDITION_ORDER_FORWARD = ("0", "A", "B", "AB")
CONDITION_ORDER_REVERSE = ("AB", "B", "A", "0")
REQUIRED_CONDITIONS = frozenset(CONDITION_ORDER_FORWARD)


class OrderDependenceError(RuntimeError):
    """Raised after results are saved when condition order invalidates a probe."""


@dataclass(frozen=True)
class VelocityPair:
    """Two explicitly named prediction levels.

    raw_conditional_model_velocity is the generated-token slice directly returned
    by the conditional transformer call. effective_velocity is the tensor that
    would be passed to the Euler update after any post-transformer guidance.
    These are equal for VeloEdit's guidance-distilled FLUX.1-Kontext path because
    guidance is an input to the transformer and no external CFG arithmetic follows.
    """

    raw_conditional_model_velocity: Any
    effective_velocity: Any


@dataclass
class SharedStateEvaluation:
    """Complete two-order evaluation at one immutable carrier state."""

    predictions_by_order: Dict[str, Dict[str, VelocityPair]]
    order_metrics: Dict[str, Dict[str, Dict[str, Any]]]
    residuals: Dict[str, Dict[str, Any]]
    blocked: bool
    blocking_reasons: list[str]

    @property
    def primary_reference_corrected_residual(self) -> Any:
        """Primary EXP-001 diagnostic: effective r_ref from the forward order."""

        return self.residuals["effective"]["r_ref"]

    @property
    def carrier_effective_velocity(self) -> Any:
        """AB effective velocity from order 1, valid only when not blocked."""

        if self.blocked:
            raise OrderDependenceError(
                "Order dependence blocked this measurement; carrier velocity is invalid."
            )
        return self.predictions_by_order["order_1"]["AB"].effective_velocity


def validate_conditions(conditions: Mapping[str, str]) -> None:
    """Require exactly the four preregistered EXP-001 condition labels."""

    labels = set(conditions)
    if labels != REQUIRED_CONDITIONS:
        missing = sorted(REQUIRED_CONDITIONS - labels)
        extra = sorted(labels - REQUIRED_CONDITIONS)
        raise ValueError(
            f"conditions must be exactly {sorted(REQUIRED_CONDITIONS)}; "
            f"missing={missing}, extra={extra}"
        )
    for label, prompt in conditions.items():
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"condition {label!r} must have a non-empty text prompt")


def resolve_probe_steps(
    num_transitions: int,
    requested: Sequence[str | int],
) -> list[int]:
    """Resolve early/middle/late aliases to pre-update transition indices."""

    if num_transitions < 3:
        raise ValueError("EXP-001 requires at least three transitions")
    aliases = {
        "early": 0,
        "middle": num_transitions // 2,
        "late": num_transitions - 1,
    }
    resolved: list[int] = []
    for item in requested:
        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized in aliases:
                index = aliases[normalized]
            else:
                try:
                    index = int(normalized)
                except ValueError as exc:
                    raise ValueError(f"unknown probe step {item!r}") from exc
        else:
            index = int(item)
        if index < 0 or index >= num_transitions:
            raise ValueError(
                f"probe step {index} outside [0, {num_transitions - 1}]"
            )
        if index not in resolved:
            resolved.append(index)
    if len(resolved) < 3:
        raise ValueError("select at least three distinct early/middle/late steps")
    return sorted(resolved)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "EXP-001 tensor evaluation requires PyTorch. Install the VeloEdit "
            "runtime before running a real or tensor-level smoke experiment."
        ) from exc
    return torch


def _clone_timestep(timestep: Any) -> Any:
    if hasattr(timestep, "detach") and hasattr(timestep, "clone"):
        return timestep.detach().clone()
    return timestep


def _assert_same_tensor(actual: Any, expected: Any, name: str) -> None:
    torch = _torch()
    if not torch.equal(actual, expected):
        raise RuntimeError(f"{name} changed during a supposedly read-only probe")


def _validate_pair(pair: VelocityPair, shared_state: Any, condition: str) -> None:
    for field_name in (
        "raw_conditional_model_velocity",
        "effective_velocity",
    ):
        tensor = getattr(pair, field_name)
        if tuple(tensor.shape) != tuple(shared_state.shape):
            raise ValueError(
                f"{condition} {field_name} shape {tuple(tensor.shape)} does not "
                f"match generated-token latent shape {tuple(shared_state.shape)}"
            )


def _difference_metrics(first: Any, second: Any, epsilon: float) -> Dict[str, Any]:
    torch = _torch()
    delta = first.detach().float() - second.detach().float()
    max_abs = float(delta.abs().max().item())
    delta_norm = torch.linalg.vector_norm(delta)
    first_norm = torch.linalg.vector_norm(first.detach().float())
    second_norm = torch.linalg.vector_norm(second.detach().float())
    denominator = torch.maximum(first_norm, second_norm).clamp_min(epsilon)
    relative_l2 = float((delta_norm / denominator).item())
    return {
        "max_absolute_difference": max_abs,
        "relative_l2_difference": relative_l2,
        "relative_l2_denominator": "max(l2(order_1), l2(order_2), epsilon)",
    }


def _residuals(predictions: Mapping[str, VelocityPair], field: str) -> Dict[str, Any]:
    v_0 = getattr(predictions["0"], field)
    v_a = getattr(predictions["A"], field)
    v_b = getattr(predictions["B"], field)
    v_ab = getattr(predictions["AB"], field)
    return {
        "r_raw": v_ab - v_a - v_b,
        "r_ref": v_ab - v_a - v_b + v_0,
    }


def evaluate_shared_state(
    predict: Callable[[str, Any, Any], VelocityPair],
    shared_z_t: Any,
    timestep: Any,
    *,
    max_abs_tolerance: float = 1e-5,
    relative_l2_tolerance: float = 1e-5,
    epsilon: float = 1e-12,
) -> SharedStateEvaluation:
    """Evaluate 0/A/B/AB twice without changing the shared state.

    A condition is blocking when either its raw or effective prediction exceeds
    either registered tolerance between the forward and reverse orders. No reset,
    averaging, or compensation is applied.
    """

    torch = _torch()
    if max_abs_tolerance < 0 or relative_l2_tolerance < 0:
        raise ValueError("order-invariance tolerances must be non-negative")

    shared_snapshot = shared_z_t.detach().clone()
    predictions_by_order: Dict[str, Dict[str, VelocityPair]] = {}
    orders: Iterable[tuple[str, tuple[str, ...]]] = (
        ("order_1", CONDITION_ORDER_FORWARD),
        ("order_2", CONDITION_ORDER_REVERSE),
    )

    for order_name, order in orders:
        predictions_by_order[order_name] = {}
        for condition in order:
            model_input = shared_snapshot.detach().clone()
            input_before = model_input.detach().clone()
            pair = predict(condition, model_input, _clone_timestep(timestep))
            if not isinstance(pair, VelocityPair):
                raise TypeError("predict must return VelocityPair")
            _assert_same_tensor(model_input, input_before, f"z_t input for {condition}")
            _assert_same_tensor(shared_z_t, shared_snapshot, "shared carrier z_t")
            _validate_pair(pair, shared_snapshot, condition)
            predictions_by_order[order_name][condition] = pair

    order_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    blocking_reasons: list[str] = []
    for condition in CONDITION_ORDER_FORWARD:
        order_metrics[condition] = {}
        first = predictions_by_order["order_1"][condition]
        second = predictions_by_order["order_2"][condition]
        for short_name, field_name in (
            ("raw_conditional_model_velocity", "raw_conditional_model_velocity"),
            ("effective_velocity", "effective_velocity"),
        ):
            metrics = _difference_metrics(
                getattr(first, field_name),
                getattr(second, field_name),
                epsilon,
            )
            metrics["exceeds_tolerance"] = bool(
                metrics["max_absolute_difference"] > max_abs_tolerance
                or metrics["relative_l2_difference"] > relative_l2_tolerance
            )
            order_metrics[condition][short_name] = metrics
            if metrics["exceeds_tolerance"]:
                blocking_reasons.append(
                    f"{condition}/{short_name}: max_abs="
                    f"{metrics['max_absolute_difference']:.6g}, relative_l2="
                    f"{metrics['relative_l2_difference']:.6g}"
                )

    first_order = predictions_by_order["order_1"]
    residuals = {
        "raw_conditional_model": _residuals(
            first_order, "raw_conditional_model_velocity"
        ),
        "effective": _residuals(first_order, "effective_velocity"),
    }
    return SharedStateEvaluation(
        predictions_by_order=predictions_by_order,
        order_metrics=order_metrics,
        residuals=residuals,
        blocked=bool(blocking_reasons),
        blocking_reasons=blocking_reasons,
    )
