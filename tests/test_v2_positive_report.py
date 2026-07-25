import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.v2_positive_report import TASK_SAMPLES, report_v2_positive_leaderboard


class V2PositiveReportTests(unittest.TestCase):
    def test_report_combines_weighted_and_macro_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_dir = root / "core"
            extension_dir = root / "extension"
            core_dir.mkdir()
            extension_dir.mkdir()
            spec = {
                "label": "method-a",
                "family": "test",
                "core_summary": "core.json",
                "extension_summary": "extension.json",
                "core_details": "core_details.jsonl",
                "extension_details": "extension_details.jsonl",
                "core_bcy": "core-a",
                "extension_bcy": "extension-a",
            }
            core_metrics = {
                task: self._metrics(samples, value)
                for task, samples, value in (
                    ("code2test", 106, 0.1),
                    ("comment2context", 80, 0.2),
                    ("trace2code", 101, 0.3),
                )
            }
            self._write(core_dir / "core.json", {"metrics": core_metrics})
            self._write(
                extension_dir / "extension.json",
                {"metrics": {"edit2ripple": self._metrics(58, 0.4)}},
            )
            self._write_details(core_dir / "core_details.jsonl", 287, ("repo-large", "repo-small"), 0.2)
            self._write_details(extension_dir / "extension_details.jsonl", 58, ("repo-large",), 0.4, offset=287)
            core_bcy = self._bcy_report("core-a", {"code2test": 0.2, "comment2context": 0.3, "trace2code": 0.4})
            extension_bcy = self._bcy_report("extension-a", {"edit2ripple": 0.5})
            self._write(root / "core_bcy.json", core_bcy)
            self._write(root / "extension_bcy.json", extension_bcy)

            report_v2_positive_leaderboard(
                core_eval_dir=core_dir,
                extension_eval_dir=extension_dir,
                core_bcy_path=root / "core_bcy.json",
                extension_bcy_path=root / "extension_bcy.json",
                out_path=root / "leaderboard.md",
                method_specs=(spec,),
            )

            report = json.loads((root / "leaderboard.json").read_text())
            row = report["rows"][0]
            expected_weighted = sum(
                value * TASK_SAMPLES[task]
                for task, value in zip(TASK_SAMPLES, (0.1, 0.2, 0.3, 0.4))
            ) / 345
            self.assertAlmostEqual(row["weighted"]["MRR"], expected_weighted)
            self.assertAlmostEqual(row["macro"]["MRR"], 0.25)
            self.assertAlmostEqual(row["macro"]["BCY@8k"], 0.35)
            expected_repo_macro = (((144 * 0.2) + (58 * 0.4)) / 202 + 0.2) / 2
            self.assertAlmostEqual(row["repo_macro"]["MRR"], expected_repo_macro)
            self.assertEqual(report["repository_distribution"]["repository_count"], 2)
            self.assertIn("V2 Positive Leaderboard", (root / "leaderboard.md").read_text())

    def test_report_rejects_incomplete_task_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_dir = root / "core"
            extension_dir = root / "extension"
            core_dir.mkdir()
            extension_dir.mkdir()
            spec = {
                "label": "method-a",
                "family": "test",
                "core_summary": "core.json",
                "extension_summary": "extension.json",
                "core_details": "core_details.jsonl",
                "extension_details": "extension_details.jsonl",
                "core_bcy": "core-a",
                "extension_bcy": "extension-a",
            }
            self._write(
                core_dir / "core.json",
                {
                    "metrics": {
                        "code2test": self._metrics(105, 0.1),
                        "comment2context": self._metrics(80, 0.2),
                        "trace2code": self._metrics(101, 0.3),
                    }
                },
            )
            self._write(extension_dir / "extension.json", {"metrics": {"edit2ripple": self._metrics(58, 0.4)}})
            self._write_details(core_dir / "core_details.jsonl", 287, ("repo-a",), 0.2)
            self._write_details(extension_dir / "extension_details.jsonl", 58, ("repo-a",), 0.4, offset=287)
            self._write(root / "core_bcy.json", self._bcy_report("core-a", {"code2test": 0.2, "comment2context": 0.3, "trace2code": 0.4}))
            self._write(root / "extension_bcy.json", self._bcy_report("extension-a", {"edit2ripple": 0.5}))

            with self.assertRaisesRegex(ValueError, "expected 106"):
                report_v2_positive_leaderboard(
                    core_eval_dir=core_dir,
                    extension_eval_dir=extension_dir,
                    core_bcy_path=root / "core_bcy.json",
                    extension_bcy_path=root / "extension_bcy.json",
                    out_path=root / "leaderboard.md",
                    method_specs=(spec,),
                )

    @staticmethod
    def _metrics(samples, value):
        return {"samples": samples, "Recall@5": value, "Recall@10": value, "Recall@20": value, "MRR": value}

    @staticmethod
    def _bcy_report(label, values):
        return {
            "runs": [
                {
                    "label": label,
                    "by_task": {task: {"BCY@8000": value} for task, value in values.items()},
                }
            ]
        }

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _write_details(path, count, repos, value, offset=0):
        rows = []
        for index in range(count):
            repo = repos[index % len(repos)]
            metrics = {metric: value for metric in ("Recall@5", "Recall@10", "Recall@20", "MRR")}
            rows.append(json.dumps({"sample_id": f"sample-{offset + index}", "repo": repo, "metrics": metrics}))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
