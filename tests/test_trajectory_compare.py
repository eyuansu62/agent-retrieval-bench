import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.trajectory_compare import evaluate_ranked_context_as_trajectory


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class TrajectoryCompareTests(unittest.TestCase):
    def test_evaluate_ranked_context_as_same_budget_final_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details = root / "repomap_details.jsonl"
            write_jsonl(
                details,
                [
                    {
                        "sample_id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["tests/a.py"],
                        "top_files": ["tests/a.py", "src/a.py"],
                    },
                    {
                        "sample_id": "s2",
                        "task_type": "trace2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["src/x.py", "src/y.py"],
                        "top_files": ["src/z.py", "src/x.py", "src/y.py"],
                    },
                ],
            )

            result = evaluate_ranked_context_as_trajectory(
                baseline_details=details,
                top_k=2,
                out_path=root / "summary.json",
                details_path=root / "ranked_details.jsonl",
                model_label="repomap@2",
            )

            self.assertEqual(result["mode"], "ranked_context_as_trajectory")
            self.assertEqual(result["model"], "repomap@2")
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(result["context"]["average_final_files"], 2.0)
            self.assertAlmostEqual(result["metrics"]["overall"]["final_file_recall"], 0.75)
            self.assertAlmostEqual(result["metrics"]["overall"]["final_file_precision"], 0.5)
            self.assertAlmostEqual(result["metrics"]["overall"]["final_file_f1"], (2 / 3 + 0.5) / 2)
            self.assertTrue((root / "summary.json").exists())
            self.assertEqual(len((root / "ranked_details.jsonl").read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
