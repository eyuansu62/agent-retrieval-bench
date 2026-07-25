import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.rank_analysis import report_rank_analysis


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RankAnalysisTests(unittest.TestCase):
    def test_report_rank_analysis_summarizes_depths_and_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            samples = root / "benchmark" / "samples.jsonl"
            out = root / "reports" / "rank_analysis.md"
            sample_rows = [
                {
                    "id": "s1",
                    "task_type": "trace2code",
                    "repo": "owner/repo",
                    "query": {"failure_excerpt": "panic in parser"},
                    "gold": {"root_cause_files": ["src/parser.py"]},
                    "metadata": {"pr_url": "https://example.test/pr/1"},
                },
                {
                    "id": "s2",
                    "task_type": "comment2context",
                    "repo": "owner/repo",
                    "query": {"review_comment": "needs same behavior", "given_file": "tests/test_api.py"},
                    "gold": {"given_files": ["tests/test_api.py"], "must_context_files": [{"path": "src/api.py"}]},
                    "metadata": {},
                },
            ]
            write_jsonl(samples, sample_rows)
            write_json(
                eval_dir / "embedding_summary.json",
                {
                    "mode": "embedding",
                    "model": "/models/embedding",
                    "candidate_filter": "all_files",
                    "evaluated": 2,
                    "metrics": {"overall": {"samples": 2, "MRR": 0.25}},
                },
            )
            write_json(
                eval_dir / "repomap_summary.json",
                {
                    "mode": "repomap",
                    "model": "aider-style-repomap",
                    "candidate_filter": "all_files",
                    "evaluated": 2,
                    "metrics": {"overall": {"samples": 2, "MRR": 0.5}},
                },
            )
            write_jsonl(
                eval_dir / "embedding_details.jsonl",
                [
                    {
                        "sample_id": "s1",
                        "task_type": "trace2code",
                        "repo": "owner/repo",
                        "candidate_filter": "all_files",
                        "gold_files": ["src/parser.py"],
                        "gold_ranks": {"src/parser.py": 50},
                        "metrics": {"Recall@20": 0.0, "MRR": 0.02, "gold_coverage@8k": 0.0},
                        "top_files": ["tests/test_parser.py"],
                    },
                    {
                        "sample_id": "s2",
                        "task_type": "comment2context",
                        "repo": "owner/repo",
                        "candidate_filter": "all_files",
                        "gold_files": ["src/api.py"],
                        "gold_ranks": {"src/api.py": None},
                        "metrics": {"Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0},
                        "top_files": ["tests/test_api.py"],
                    },
                ],
            )
            write_jsonl(
                eval_dir / "repomap_details.jsonl",
                [
                    {
                        "sample_id": "s1",
                        "task_type": "trace2code",
                        "repo": "owner/repo",
                        "candidate_filter": "all_files",
                        "gold_files": ["src/parser.py"],
                        "gold_ranks": {"src/parser.py": 3},
                        "metrics": {"Recall@20": 1.0, "MRR": 1 / 3, "gold_coverage@8k": 0.0},
                        "top_files": ["src/router.py", "src/token.py", "src/parser.py"],
                    },
                    {
                        "sample_id": "s2",
                        "task_type": "comment2context",
                        "repo": "owner/repo",
                        "candidate_filter": "all_files",
                        "gold_files": ["src/api.py"],
                        "gold_ranks": {"src/api.py": None},
                        "metrics": {"Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0},
                        "top_files": ["tests/test_api.py"],
                    },
                ],
            )

            result = report_rank_analysis(eval_dir, samples, out, top_examples=3)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            markdown = out.read_text(encoding="utf-8")

            self.assertEqual(result["models"], 2)
            self.assertIn("Overall First-Gold CDF", markdown)
            overall_repomap = next(
                row for row in data["depth_rows"] if row["task"] == "overall" and row["model"] == "aider-style-repomap"
            )
            self.assertEqual(overall_repomap["any_gold@20"], 0.5)
            self.assertEqual(data["cross_model_coverage"]["overall"]["all_models_miss@20"], 1)
            self.assertEqual(len(data["examples"]["trace2code_repomap_hits_embeddings_miss@20"]), 1)
            self.assertEqual(len(data["examples"]["comment2context_given_file_top1_gold_miss@20"]), 1)


if __name__ == "__main__":
    unittest.main()
