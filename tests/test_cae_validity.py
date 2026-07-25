import unittest

from agent_retrieval_bench.cae_validity import (
    bucket_by_metric,
    correlation_summary,
    ranks,
    summarize_pes_calibration,
)


class CaeValidityTest(unittest.TestCase):
    def test_spearman_summary_handles_monotonic_pairs(self):
        summary = correlation_summary([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)], "positive")
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["spearman"], 1.0)
        self.assertAlmostEqual(summary["pearson"], 1.0)

    def test_ranks_average_ties(self):
        self.assertEqual(ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_bucket_by_metric_uses_equal_count_splits(self):
        rows = [{"sample_id": str(index), "bcy": float(index)} for index in range(9)]
        buckets = bucket_by_metric(rows, "bcy")
        self.assertEqual([name for name, _ in buckets], ["low", "mid", "high"])
        self.assertEqual([len(bucket) for _, bucket in buckets], [3, 3, 3])
        self.assertEqual([bucket[0]["sample_id"] for _, bucket in buckets], ["0", "3", "6"])

    def test_pes_calibration_summary_recomputes_correlations(self):
        report = {
            "counts": {"paired_samples": 3},
            "overall": {
                "predicted_pes_mean": 2.0,
                "first_hit_delta_mean_both_hit": 2.0,
                "event_saving_mean": 1.0,
                "final_file_f1_delta_mean": 0.1,
                "rescue_rate": 0.2,
                "lost_rate": 0.0,
            },
            "samples": [
                {
                    "has_control": True,
                    "has_seeded": True,
                    "predicted_pes@20": 1.0,
                    "first_hit_delta": 1.0,
                    "event_saving": 0.0,
                    "final_file_f1_delta": 0.0,
                },
                {
                    "has_control": True,
                    "has_seeded": True,
                    "predicted_pes@20": 2.0,
                    "first_hit_delta": 2.0,
                    "event_saving": 1.0,
                    "final_file_f1_delta": 0.1,
                },
                {
                    "has_control": True,
                    "has_seeded": True,
                    "predicted_pes@20": 3.0,
                    "first_hit_delta": 3.0,
                    "event_saving": 2.0,
                    "final_file_f1_delta": 0.2,
                },
            ],
        }
        summary = summarize_pes_calibration(report)
        self.assertEqual(summary["paired_samples"], 3)
        self.assertAlmostEqual(summary["pes_vs_first_hit_delta"]["spearman"], 1.0)
        self.assertAlmostEqual(summary["pes_vs_event_saving"]["spearman"], 1.0)
        self.assertAlmostEqual(summary["pes_vs_final_file_f1_delta"]["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
