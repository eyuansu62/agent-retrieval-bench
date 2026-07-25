import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.dataset_validity import (
    build_gold_audit_summary,
    build_task_diversity_report,
    detect_query_leakage,
    ndcg_from_ranks,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def sample(sample_id="s1", task_type="code2test", query=None, gold=None, spans=True):
    row = {
        "id": sample_id,
        "task_type": task_type,
        "repo": "o/r",
        "base_commit": "base",
        "query": query or {"raw_signal": "auth fails"},
        "gold": gold
        or {
            "related_tests": ["tests/test_auth.py"],
            "root_cause_files": ["src/auth.py"],
            "root_cause_symbols": ["AuthManager"],
            "fix_commit": "abc123def4567890",
        },
        "query_provenance": "pr_summary" if task_type == "code2test" else "failure_trace",
    }
    if spans:
        row["gold_spans"] = [{"path": "tests/test_auth.py", "start_line": 10, "end_line": 20, "reason": "manual review"}]
        row["gold_blocks"] = [{"path": "tests/test_auth.py", "start_line": 10, "end_line": 20, "symbol": "test_auth_login", "reason": "manual review"}]
    return row


class DatasetValidityTests(unittest.TestCase):
    def test_leakage_detector_catches_shortcut_types(self):
        row = sample(
            query={"raw_signal": "FAILED tests/test_auth.py::test_auth_login AuthManager abc123def456 diff --git a/x b/x"},
        )

        result = detect_query_leakage(row)
        flag_types = set(result["flag_types"])

        self.assertIn("exact_gold_path", flag_types)
        self.assertIn("test_symbol", flag_types)
        self.assertIn("gold_symbol", flag_types)
        self.assertIn("raw_patch_marker", flag_types)
        self.assertIn("fix_commit_hash", flag_types)
        self.assertTrue(result["fatal"])

    def test_trace_stack_hint_is_diagnostic_not_fatal(self):
        row = sample(
            sample_id="t1",
            task_type="trace2code",
            query={"raw_signal": "Traceback\n  File \"/repo/src/auth.py\", line 42, in login"},
            gold={"root_cause_files": ["src/auth.py"], "related_tests": [], "fix_commit": "ffffeeee11112222"},
        )

        result = detect_query_leakage(row)

        self.assertIn("direct_stack_hint", result["flag_types"])
        self.assertNotIn("exact_gold_path", result["flag_types"])
        self.assertFalse(result["fatal"])

    def test_gold_audit_aggregates_review_evidence_and_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = sample("reviewed")
            pending = sample("pending", spans=False)
            validation = {
                "rows": [
                    {"sample_id": "reviewed", "errors": []},
                    {"sample_id": "pending", "errors": []},
                ]
            }
            validation_path = root / "validation.json"
            audit_path = root / "audit.jsonl"
            write_json(validation_path, validation)
            write_jsonl(audit_path, [{"sample_id": "reviewed", "task_type": "code2test", "repo": "o/r", "review_status": "accepted"}])

            result = build_gold_audit_summary(
                samples=[reviewed, pending],
                validation_path=validation_path,
                audit_paths=[audit_path],
                out_dir=root / "out",
                audit_packet_size=10,
                valid_threshold=0.9,
            )

            summary = result["summary"]
            self.assertEqual(summary["overall"]["reviewed"], 1)
            self.assertEqual(summary["overall"]["pending"], 1)
            self.assertEqual(summary["overall"]["valid"], 1)
            self.assertEqual(summary["audit_packet"]["rows"], 1)
            self.assertTrue((root / "out" / "gold_audit_packet.jsonl").exists())

    def test_task_diversity_reports_macro_spread_and_ndcg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [sample("a", "code2test"), sample("b", "comment2context"), sample("c", "trace2code")]
            eval_dir = root / "eval"
            metrics = {
                "overall": {"samples": 3, "Recall@20": 0.6, "MRR": 0.5, "gold_coverage@8k": 0.4},
                "code2test": {"samples": 1, "Recall@20": 1.0, "MRR": 1.0, "gold_coverage@8k": 1.0},
                "comment2context": {"samples": 1, "Recall@20": 0.5, "MRR": 0.5, "gold_coverage@8k": 0.0},
                "trace2code": {"samples": 1, "Recall@20": 0.3, "MRR": 0.2, "gold_coverage@8k": 0.2},
            }
            write_json(eval_dir / "model_a_summary.json", {"mode": "embedding", "model": "model-a", "evaluated": 3, "metrics": metrics})
            write_json(eval_dir / "model_b_summary.json", {"mode": "embedding", "model": "model-b", "evaluated": 3, "metrics": {task: {**row, "Recall@20": 0.1, "gold_coverage@8k": 0.1} for task, row in metrics.items()}})
            write_jsonl(
                eval_dir / "model_a_details.jsonl",
                [
                    {"sample_id": "a", "task_type": "code2test", "gold_ranks": {"x": 1}},
                    {"sample_id": "b", "task_type": "comment2context", "gold_ranks": {"x": 20}},
                    {"sample_id": "c", "task_type": "trace2code", "gold_ranks": {"x": None}},
                ],
            )

            result = build_task_diversity_report(
                samples=samples,
                samples_path=root / "samples.jsonl",
                corpus_manifest_path=root / "corpus_manifest.jsonl",
                eval_dirs=[eval_dir],
                out_dir=root / "out",
                min_task_spread=0.05,
            )

            self.assertEqual(result["summary"]["sample_distribution"]["by_task"], {"code2test": 1, "comment2context": 1, "trace2code": 1})
            self.assertGreaterEqual(result["summary"]["macro_average_rows"], 2)
            self.assertIn("Recall@20", json.loads((root / "out" / "task_diversity_report.json").read_text())["spread"])
            self.assertAlmostEqual(ndcg_from_ranks({"x": 1}), 1.0)


if __name__ == "__main__":
    unittest.main()
