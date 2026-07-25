import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.agentic_eval import evaluate_agentic_search
from agent_retrieval_bench.baseline import sample_metrics, span_metrics_at_budget
from agent_retrieval_bench.v1_2 import (
    merge_manual_annotations,
    report_context_pollution,
    report_runtime_cache,
    report_span_subset,
    validate_v1_2_benchmark,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class V12Tests(unittest.TestCase):
    def test_sample_metrics_include_precision_and_context_pollution(self):
        ranked = [
            {"path": "tests/test_auth.py", "text": "x" * 10},
            {"path": "src/auth.py", "text": "x" * 20},
            {"path": "docs/auth.md", "text": "x" * 30},
        ]

        metrics = sample_metrics(["src/auth.py"], ranked, hard_negative_files=["tests/test_auth.py"])

        self.assertEqual(metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(metrics["Precision@5"], 1 / 3)
        self.assertAlmostEqual(metrics["F0.5@5"], 5 / 13)
        self.assertEqual(metrics["hard_negative_hits@5"], 1.0)
        self.assertEqual(metrics["irrelevant_files@5"], 2.0)
        self.assertEqual(metrics["context_pollution_tokens@8k"], 40.0)
        self.assertAlmostEqual(metrics["gold_token_ratio@8k"], 20 / 60)

    def test_span_metrics_measure_line_overlap(self):
        spans = [{"path": "src/auth.py", "start_line": 10, "end_line": 20}]
        ranked = [
            {"path": "src/auth.py", "start_line": 15, "end_line": 25, "text": "x"},
            {"path": "src/other.py", "start_line": 1, "end_line": 10, "text": "x"},
        ]

        metrics = span_metrics_at_budget(spans, ranked)

        self.assertEqual(metrics["gold_lines"], 11.0)
        self.assertEqual(metrics["overlap_lines"], 6.0)
        self.assertGreater(metrics["span_f0.5@8k"], 0.0)

    def test_validate_v1_2_accepts_spans_and_rejects_role_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark"
            corpus = root / "corpus"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            valid_sample = self._sample("valid", hard_negative_files=["tests/test_auth.py"])
            invalid_sample = self._sample("invalid", hard_negative_files=["src/auth.py"])
            write_jsonl(derived / "samples.jsonl", [valid_sample, invalid_sample])
            write_jsonl(
                chunks_path,
                [
                    {"path": "src/auth.py", "kind": "file", "start_line": 1, "end_line": 50, "text": "auth"},
                    {"path": "tests/test_auth.py", "kind": "file", "start_line": 1, "end_line": 30, "text": "test"},
                ],
            )
            manifest = corpus / "corpus_manifest.jsonl"
            write_jsonl(manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            report = validate_v1_2_benchmark(derived, corpus_manifest_path=manifest)

            self.assertFalse(report["ready"])
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["span_samples"], 2)
            invalid = next(row for row in report["rows"] if row["sample_id"] == "invalid")
            self.assertTrue(any("overlap gold_files" in error for error in invalid["errors"]))

    def test_merge_manual_annotations_preserves_samples_and_rejects_invalid_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            annotations = root / "annotations.jsonl"
            out = root / "out"
            s2 = self._sample("s2", [])
            s2.pop("gold_spans")
            s2.pop("hard_negative_files")
            s2.pop("query_provenance")
            write_jsonl(base / "samples.jsonl", [self._sample("s1", []), s2])
            write_jsonl(
                annotations,
                [
                    {
                        "sample_id": "s1",
                        "gold_spans": [{"path": "src/auth.py", "start_line": 12, "end_line": 18}],
                        "hard_negative_files": ["tests/test_auth.py"],
                        "query_provenance": "failure_trace",
                    }
                ],
            )

            report = merge_manual_annotations(base, annotations, out)
            merged = [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertTrue(report["sample_ids_preserved"])
            self.assertEqual([row["id"] for row in merged], ["s1", "s2"])
            self.assertEqual(report["annotation_count"], 1)
            self.assertEqual(merged[0]["hard_negative_files"], ["tests/test_auth.py"])
            self.assertNotIn("gold_spans", merged[1])

            write_jsonl(
                annotations,
                [
                    {
                        "sample_id": "s1",
                        "gold_spans": [{"path": "tests/test_auth.py", "start_line": 1, "end_line": 2}],
                    }
                ],
            )
            with self.assertRaises(ValueError):
                merge_manual_annotations(base, annotations, out)

            write_jsonl(annotations, [{"sample_id": "s1", "hard_negative_files": ["src/auth.py"]}])
            with self.assertRaises(ValueError):
                merge_manual_annotations(base, annotations, out)

    def test_v1_2_reports_read_extended_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            write_json(
                eval_dir / "scripted-search-read_summary.json",
                {
                    "mode": "agentic_search",
                    "model": "scripted-search-read",
                    "candidate_filter": "all_files",
                    "evaluated": 1,
                    "runtime": {"wall_time_seconds": 3.5, "batch_size": None},
                    "metrics": {"overall": {"samples": 1, "MRR": 1.0}},
                },
            )
            write_jsonl(
                eval_dir / "scripted-search-read_details.jsonl",
                [
                    {
                        "sample_id": "s1",
                        "task_type": "trace2code",
                        "candidate_filter": "all_files",
                        "metrics": {
                            "Precision@20": 0.25,
                            "F0.5@20": 0.5,
                            "irrelevant_files@20": 3,
                            "hard_negative_hits@20": 1,
                            "context_pollution_tokens@8k": 700,
                            "gold_token_ratio@8k": 0.3,
                        },
                        "span_metrics": {
                            "span_recall@8k": 0.5,
                            "span_precision@8k": 0.25,
                            "span_f0.5@8k": 0.3,
                            "line_overlap_f0.5": 0.3,
                            "gold_lines": 10,
                            "predicted_lines": 20,
                            "overlap_lines": 5,
                        },
                    }
                ],
            )

            context = report_context_pollution(eval_dir, root / "reports" / "context.md")
            spans = report_span_subset(eval_dir, root / "reports" / "spans.md")
            runtime = report_runtime_cache(eval_dir, root / "reports" / "runtime.md")

            self.assertEqual(context["rows"][0]["Precision@20"], 0.25)
            self.assertEqual(spans["rows"][0]["span_recall@8k"], 0.5)
            self.assertEqual(runtime["rows"][0]["wall_time_seconds"], 3.5)

    def test_eval_agentic_records_budget_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(samples, [self._sample("s1", hard_negative_files=["tests/test_auth.py"])])
            write_jsonl(
                chunks_path,
                [
                    {"chunk_id": "c1", "path": "tests/test_auth.py", "kind": "file", "symbol": "", "start_line": 1, "end_line": 10, "text": "auth failure"},
                    {"chunk_id": "c2", "path": "src/auth.py", "kind": "file", "symbol": "", "start_line": 1, "end_line": 40, "text": "auth implementation failure"},
                ],
            )
            write_jsonl(corpus_dir / "corpus_manifest.jsonl", [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])
            details = root / "details.jsonl"

            result = evaluate_agentic_search([samples], corpus_dir, details_path=details, max_tool_calls_per_turn=2)
            row = json.loads(details.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(result["mode"], "agentic_search")
            self.assertLessEqual(row["agentic"]["tool_calls"], 2)
            self.assertIn("Precision@20", row["metrics"])
            self.assertEqual(row["hard_negative_files"], ["tests/test_auth.py"])

    def _sample(self, sample_id, hard_negative_files):
        return {
            "id": sample_id,
            "version": 2,
            "task_type": "trace2code",
            "repo": "o/r",
            "base_commit": "base",
            "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": "base"},
            "query": {"failure_excerpt": "auth failure"},
            "query_provenance": "failure_trace",
            "gold": {"root_cause_files": ["src/auth.py"], "related_tests": []},
            "gold_spans": [{"path": "src/auth.py", "start_line": 10, "end_line": 20, "reason": "root cause implementation"}],
            "hard_negative_files": hard_negative_files,
        }


if __name__ == "__main__":
    unittest.main()
