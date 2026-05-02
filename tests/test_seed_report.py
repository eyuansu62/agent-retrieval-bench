import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.seed_report import report_v1_seed


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class SeedReportTests(unittest.TestCase):
    def test_report_v1_seed_compares_counts_metrics_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "base_samples.jsonl"
            seed_samples = root / "seed_samples.jsonl"
            base_eval = root / "base_eval.json"
            seed_eval = root / "seed_eval.json"
            audit = root / "audit.json"
            out = root / "report.md"
            json_out = root / "report.json"
            write_jsonl(
                base_samples,
                [
                    {"id": "base-1", "task_type": "code2test"},
                    {"id": "base-2", "task_type": "comment2context"},
                ],
            )
            write_jsonl(seed_samples, [{"id": "seed-1", "task_type": "code2test"}])
            base_eval.write_text(
                json.dumps({"evaluated": 2, "skipped": {}, "metrics": {"overall": {"samples": 2, "Recall@20": 0.7, "MRR": 0.3}}}),
                encoding="utf-8",
            )
            seed_eval.write_text(
                json.dumps({"evaluated": 1, "skipped": {}, "metrics": {"overall": {"samples": 1, "Recall@20": 0.4, "MRR": 0.1}}}),
                encoding="utf-8",
            )
            audit.write_text(json.dumps({"total": 60, "kept": 55, "dropped": 5, "pending": 0, "verdicts": {"valid": 55}}), encoding="utf-8")

            result = report_v1_seed(base_samples, base_eval, seed_samples, seed_eval, audit, out, json_out)
            report = json.loads(json_out.read_text())

            self.assertEqual(result["status"], "ready")
            self.assertEqual(report["base"]["samples"]["by_task"], {"code2test": 1, "comment2context": 1})
            self.assertEqual(report["seed"]["samples"]["by_task"], {"code2test": 1})
            self.assertEqual(report["audit"]["kept"], 55)
            self.assertIn("V1 Seed Comparison", out.read_text())

    def test_report_marks_audit_shortfall_without_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            audit.write_text(json.dumps({"total": 77, "kept": 42, "dropped": 35, "pending": 0}), encoding="utf-8")

            result = report_v1_seed(
                root / "base_missing.jsonl",
                root / "base_eval_missing.json",
                root / "seed_missing.jsonl",
                root / "seed_eval_missing.json",
                audit,
                root / "report.md",
                root / "report.json",
            )

            self.assertEqual(result["status"], "audit_shortfall")


if __name__ == "__main__":
    unittest.main()
