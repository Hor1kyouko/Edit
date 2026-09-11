import unittest

from src.exp003.core import paired_prompt_comparison, validate_atomic_conditions


class PromptValidationTests(unittest.TestCase):
    def test_exact_atomic_composition_is_required(self):
        a = "change object A to white"
        b = "change object B to blue"
        conditions = {
            "0": "Keep the image unchanged.",
            "A": a,
            "B": b,
            "AB": f"{a} and {b}",
        }
        validate_atomic_conditions(conditions, a, b)
        conditions["AB"] = "make both requested edits"
        with self.assertRaises(ValueError):
            validate_atomic_conditions(conditions, a, b)


class PairedComparisonTests(unittest.TestCase):
    @staticmethod
    def _record(seed, q1):
        metrics = {
            "Q1": q1,
            "Q2": q1 / 2,
            "cos_dAB_dA_plus_dB": 0.9,
            "ratio_dA_plus_dB_to_dAB": 1.2,
            "cos_dA_dB": 0.8,
            "ratio_dA_to_dAB": 0.7,
            "ratio_dB_to_dAB": 0.8,
            "norm_dA": 1.0,
            "norm_dB": 2.0,
            "norm_dAB": 3.0,
            "norm_dA_plus_dB": 4.0,
            "norm_r_ref": 5.0,
        }
        return {
            "seed": seed,
            "probes": [
                {
                    "valid": True,
                    "schedule_index": 17,
                    "target_sigma": 0.5,
                    "actual_sigma": 0.49,
                    "metrics": metrics,
                }
            ],
        }

    def test_delta_and_sign_counts_are_paired(self):
        result = paired_prompt_comparison(
            [self._record(42, 2.0), self._record(43, 4.0)],
            [self._record(42, 1.0), self._record(43, 5.0)],
        )
        q1 = result["by_probe"][0]["metrics"]["Q1"]
        self.assertEqual(q1["delta_summary"]["mean"], 0.0)
        self.assertEqual(q1["sign_consistency"]["atomic_lower"], 1)
        self.assertEqual(q1["sign_consistency"]["atomic_higher"], 1)


if __name__ == "__main__":
    unittest.main()
