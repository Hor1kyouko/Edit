import unittest

from src.exp001.core import (
    CONDITION_ORDER_FORWARD,
    CONDITION_ORDER_REVERSE,
    CONDITION_ZERO_DEFAULT,
    VelocityPair,
    evaluate_shared_state,
    resolve_probe_steps,
    validate_conditions,
)


class ExperimentSpecificationTests(unittest.TestCase):
    def test_exact_four_conditions_are_required(self):
        conditions = {
            "0": CONDITION_ZERO_DEFAULT,
            "A": "Edit object A.",
            "B": "Edit object B.",
            "AB": "Edit object A and object B.",
        }
        validate_conditions(conditions)
        with self.assertRaises(ValueError):
            validate_conditions({key: conditions[key] for key in ("A", "B", "AB")})

    def test_reference_condition_is_explicit_text(self):
        self.assertEqual(CONDITION_ZERO_DEFAULT, "Keep the image unchanged.")
        self.assertTrue(CONDITION_ZERO_DEFAULT.strip())

    def test_orders_are_exact_reverses(self):
        self.assertEqual(CONDITION_ORDER_FORWARD, ("0", "A", "B", "AB"))
        self.assertEqual(CONDITION_ORDER_REVERSE, tuple(reversed(CONDITION_ORDER_FORWARD)))

    def test_early_middle_late_resolution(self):
        self.assertEqual(
            resolve_probe_steps(30, ["early", "middle", "late"]),
            [0, 15, 29],
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this shell")
class TensorEvaluationTests(unittest.TestCase):
    def test_residuals_and_order_invariance(self):
        values = {
            "0": torch.tensor([1.0, 1.0]),
            "A": torch.tensor([2.0, 3.0]),
            "B": torch.tensor([4.0, 5.0]),
            "AB": torch.tensor([6.0, 9.0]),
        }

        def predict(condition, z_t, timestep):
            del z_t, timestep
            value = values[condition].clone()
            return VelocityPair(value, value)

        result = evaluate_shared_state(
            predict, torch.zeros(2), torch.tensor(0.5)
        )
        self.assertFalse(result.blocked)
        torch.testing.assert_close(
            result.residuals["effective"]["r_raw"],
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            result.primary_reference_corrected_residual,
            torch.tensor([1.0, 2.0]),
        )

    def test_order_dependence_is_blocking(self):
        call_index = 0

        def predict(condition, z_t, timestep):
            nonlocal call_index
            del condition, z_t, timestep
            call_index += 1
            value = torch.tensor([float(call_index)])
            return VelocityPair(value, value)

        result = evaluate_shared_state(
            predict,
            torch.zeros(1),
            torch.tensor(0.5),
            max_abs_tolerance=0.0,
            relative_l2_tolerance=0.0,
        )
        self.assertTrue(result.blocked)
        self.assertTrue(result.blocking_reasons)


if __name__ == "__main__":
    unittest.main()
