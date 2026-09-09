"""EXP-001: exact same-latent conditional velocity diagnostics."""

from .core import (
    CONDITION_ORDER_FORWARD,
    CONDITION_ORDER_REVERSE,
    CONDITION_ZERO_DEFAULT,
    OrderDependenceError,
    VelocityPair,
    evaluate_shared_state,
    resolve_probe_steps,
)

__all__ = [
    "CONDITION_ORDER_FORWARD",
    "CONDITION_ORDER_REVERSE",
    "CONDITION_ZERO_DEFAULT",
    "OrderDependenceError",
    "VelocityPair",
    "evaluate_shared_state",
    "resolve_probe_steps",
]
