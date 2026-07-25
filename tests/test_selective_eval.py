import unittest

from agent_retrieval_bench.selective_eval import selective_detail, threshold_metrics, threshold_sweep


class SelectiveEvalTests(unittest.TestCase):
    def test_threshold_metrics_abstains_low_confidence_no_gold(self):
        details = [
            {"label": "positive", "confidence": 3.0, "metrics": {"Recall@20": 1.0}},
            {"label": "positive", "confidence": 0.5, "metrics": {"Recall@20": 1.0}},
            {"label": "no_gold", "confidence": 0.2},
            {"label": "no_gold", "confidence": 2.0},
        ]

        row = threshold_metrics(details, 1.0)

        self.assertEqual(row["confusion"]["positive_returned"], 1)
        self.assertEqual(row["confusion"]["positive_abstained"], 1)
        self.assertEqual(row["confusion"]["no_gold_abstained"], 1)
        self.assertEqual(row["confusion"]["no_gold_returned"], 1)
        self.assertEqual(row["positive_pass_rate"], 0.5)
        self.assertEqual(row["no_gold_abstain_rate"], 0.5)
        self.assertEqual(row["selective_accuracy"], 0.5)
        self.assertEqual(row["selective_success@20"], 0.5)

    def test_threshold_sweep_includes_never_and_always_abstain_points(self):
        details = [
            {"label": "positive", "confidence": 3.0, "metrics": {"Recall@20": 1.0}},
            {"label": "no_gold", "confidence": 0.2},
        ]

        rows = threshold_sweep(details)

        self.assertEqual(rows[0]["threshold_display"], "-inf")
        self.assertEqual(rows[0]["abstained"], 0)
        self.assertEqual(rows[-1]["abstained"], 2)

    def test_selective_detail_supports_bm25_ranker(self):
        sample = {
            "id": "s1",
            "task_type": "edit2ripple",
            "repo": "o/r",
            "base_commit": "base",
            "gold": {"files": ["src/many.py"]},
        }
        chunks = [
            {"chunk_id": "a", "path": "src/one.py", "kind": "file", "symbol": "", "text": "alpha beta beta beta"},
            {"chunk_id": "b", "path": "src/many.py", "kind": "file", "symbol": "", "text": "alpha alpha alpha beta"},
        ]

        detail = selective_detail(0, sample, ["src/many.py"], "alpha", chunks, "all_files", False, ranker="bm25")

        self.assertEqual(detail["ranker"], "bm25")
        self.assertEqual(detail["top_files"][0], "src/many.py")
        self.assertEqual(detail["gold_ranks"], {"src/many.py": 1})


if __name__ == "__main__":
    unittest.main()
