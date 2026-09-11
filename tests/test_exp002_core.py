import math
import unittest

from src.exp002.core import aggregate_seed_records, classify_trend, map_target_sigmas


class SigmaMappingTests(unittest.TestCase):
    def test_nearest_nonterminal_schedule_points_are_selected(self):
        probes, duplicates = map_target_sigmas([1.0, 0.91, 0.76, 0.49, 0.24, 0.11, 0.0])
        self.assertEqual([probe.schedule_index for probe in probes], [0, 1, 2, 3, 4, 5])
        self.assertEqual(duplicates, [])
        self.assertEqual(probes[1].actual_sigma, 0.91)

    def test_duplicate_schedule_points_are_reported(self):
        probes, duplicates = map_target_sigmas([1.0, 0.5, 0.0], [0.9, 0.8, 0.5])
        self.assertEqual([probe.schedule_index for probe in probes], [0, 1])
        self.assertEqual(len(duplicates), 1)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def _probe(seed, index, q1, q2):
        return {
            "seed": seed,
            "valid": True,
            "schedule_index": index,
            "target_sigma": 1.0 - index / 10,
            "actual_sigma": 1.0 - index / 10,
            "metrics": {
                "Q1": q1,
                "Q2": q2,
                "cos_dAB_dA_plus_dB": 0.9,
                "ratio_dA_plus_dB_to_dAB": 0.8,
            },
        }

    def test_aggregate_reports_seed_values_and_population_std(self):
        records = [
            {"seed": 42, "probes": [self._probe(42, 0, 1.0, 0.5), self._probe(42, 1, 0.5, 0.25)]},
            {"seed": 43, "probes": [self._probe(43, 0, 3.0, 1.5), self._probe(43, 1, 4.0, 2.0)]},
        ]
        result = aggregate_seed_records(records)
        self.assertEqual(result["by_probe"][0]["Q1"]["mean"], 2.0)
        self.assertEqual(result["by_probe"][0]["Q1"]["std"], 1.0)
        self.assertEqual(result["Q1_temporal_trend_counts"]["decreasing"], 1)
        self.assertEqual(result["Q1_temporal_trend_counts"]["increasing"], 1)
        self.assertTrue(math.isfinite(result["by_probe"][0]["Q2"]["median"]))

    def test_trend_is_threshold_free(self):
        self.assertEqual(classify_trend([1.0, 0.99, 0.991]), "non_monotonic")


if __name__ == "__main__":
    unittest.main()
