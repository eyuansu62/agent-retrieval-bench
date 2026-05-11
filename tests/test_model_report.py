import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.model_report import (
    infer_candidate_filter,
    model_label,
    report_model_leaderboard,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ModelReportTests(unittest.TestCase):
    def test_report_model_leaderboard_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            out = root / "reports" / "leaderboard.md"
            write_json(
                eval_dir / "lexical_summary.json",
                {
                    "mode": "corpus",
                    "evaluated": 2,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 2, "Recall@5": 0.5, "Recall@10": 0.5, "Recall@20": 1.0, "MRR": 0.75, "gold_coverage@8k": 0.5},
                        "code2test": {"samples": 1, "Recall@5": 0.0, "Recall@10": 1.0, "Recall@20": 1.0, "MRR": 0.1, "gold_coverage@8k": 0.0},
                    },
                },
            )
            write_json(
                eval_dir / "home-qinbowen-models-jina-code-embeddings-0.5b_tests_only_summary.json",
                {
                    "mode": "embedding",
                    "model": "/home/qinbowen/models/jina-code-embeddings-0.5b",
                    "candidate_filter": "tests_only",
                    "evaluated": 2,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 2, "Recall@5": 0.7, "Recall@10": 0.8, "Recall@20": 0.9, "MRR": 0.6, "gold_coverage@8k": 0.4},
                        "code2test": {"samples": 1, "Recall@5": 1.0, "Recall@10": 1.0, "Recall@20": 1.0, "MRR": 1.0, "gold_coverage@8k": 1.0},
                    },
                },
            )

            result = report_model_leaderboard(eval_dir, out)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            markdown = out.read_text(encoding="utf-8")

            self.assertEqual(result["summaries"], 2)
            self.assertEqual(data["row_count"], 4)
            self.assertIn("jina-code-embeddings-0.5b", markdown)
            self.assertIn("`tests_only`", markdown)
            self.assertIn("## overall", markdown)

    def test_report_model_leaderboard_sorts_by_mrr_before_recall20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            out = root / "reports" / "leaderboard.md"
            write_json(
                eval_dir / "high_recall_summary.json",
                {
                    "mode": "repomap",
                    "model": "high-recall",
                    "evaluated": 1,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 1, "Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 1.0, "MRR": 0.1, "gold_coverage@8k": 0.0}
                    },
                },
            )
            write_json(
                eval_dir / "high_mrr_summary.json",
                {
                    "mode": "embedding",
                    "model": "high-mrr",
                    "evaluated": 1,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 1, "Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.5, "MRR": 0.9, "gold_coverage@8k": 0.0}
                    },
                },
            )

            report_model_leaderboard(eval_dir, out)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            overall_rows = [row for row in data["rows"] if row["task"] == "overall"]

            self.assertEqual([row["model_label"] for row in overall_rows], ["high-mrr", "high-recall"])

    def test_report_helpers_infer_filter_and_labels(self):
        self.assertEqual(infer_candidate_filter(Path("foo_tests_only_summary.json")), "tests_only")
        self.assertEqual(infer_candidate_filter(Path("foo_summary.json")), "all_files")
        self.assertEqual(model_label("/home/qinbowen/models/jina-code-embeddings-0.5b", "embedding"), "jina-code-embeddings-0.5b")
        self.assertEqual(model_label("jinaai/jina-code-embeddings-0.5b", "embedding"), "jinaai/jina-code-embeddings-0.5b")
        self.assertEqual(model_label("lexical", "corpus"), "lexical")


if __name__ == "__main__":
    unittest.main()
