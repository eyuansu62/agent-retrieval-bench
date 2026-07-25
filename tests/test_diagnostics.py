import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.diagnostics import bucket_sample, diagnose_benchmark, file_ranks, query_gold_hints


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class DiagnosticsTests(unittest.TestCase):
    def test_query_gold_hints_detects_path_and_basename(self):
        path_hints = query_gold_hints('{"raw_signal": "see tests/test_auth.py"}', ["tests/test_auth.py"])
        basename_hints = query_gold_hints('{"raw_signal": "test_auth.py failed"}', ["tests/test_auth.py"])

        self.assertTrue(path_hints["has_gold_path_hint"])
        self.assertEqual(path_hints["gold_path_hits"], ["tests/test_auth.py"])
        self.assertTrue(basename_hints["has_gold_basename_hint"])
        self.assertEqual(basename_hints["gold_basename_hits"], ["tests/test_auth.py"])

    def test_file_ranks_reports_given_and_context_positions(self):
        self.assertEqual(
            file_ranks(["src/auth.py", "tests/test_auth.py"], ["tests/test_auth.py", "src/auth.py"]),
            {"src/auth.py": 2, "tests/test_auth.py": 1},
        )

    def test_bucket_sample_classifies_quality_slices(self):
        hint = {"has_gold_path_hint": True, "has_gold_basename_hint": False}
        no_hint = {"has_gold_path_hint": False, "has_gold_basename_hint": False}

        self.assertEqual(bucket_sample({"Recall@20": 1.0}, no_hint, ["missing.py"]), "invalid_missing_gold")
        self.assertEqual(bucket_sample({"Recall@20": 1.0}, hint, []), "too_easy_direct_hint")
        self.assertEqual(bucket_sample({"Recall@5": 1.0, "Recall@20": 1.0}, no_hint, []), "easy_lexical")
        self.assertEqual(bucket_sample({"Recall@5": 0.0, "Recall@20": 0.0}, no_hint, []), "hard_lexical_miss")
        self.assertEqual(bucket_sample({"Recall@5": 0.0, "Recall@20": 0.5}, no_hint, []), "partial_lexical")
        self.assertEqual(bucket_sample({"Recall@5": 0.0, "Recall@20": 1.0}, no_hint, []), "medium_lexical")

    def test_diagnose_benchmark_writes_report_and_filters_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "samples.jsonl"
            details_path = root / "eval" / "lexical_details.jsonl"
            corpus_manifest_path = root / "corpus" / "corpus_manifest.jsonl"
            chunks_a = root / "corpus" / "o__r" / "base.chunks.jsonl"
            chunks_b = root / "corpus" / "o__r" / "base2.chunks.jsonl"
            out_dir = root / "reports"

            write_jsonl(
                samples_path,
                [
                    {
                        "id": "hinted",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "run tests/test_auth.py"},
                        "gold": {"related_tests": ["tests/test_auth.py"]},
                    },
                    {
                        "id": "missing",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/missing.py"},
                        "gold": {"related_tests": ["tests/test_missing.py"]},
                    },
                    {
                        "id": "easy",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "base_commit": "base2",
                        "version": 2,
                        "query": {"comment": "auth regression", "path": "src/auth.py"},
                        "gold": {
                            "given_files": ["src/auth.py"],
                            "must_context_files": [{"path": "tests/test_auth.py", "evidence": ["human_verified_required"]}],
                            "root_cause_files": ["tests/test_auth.py"],
                        },
                    },
                    {
                        "id": "excluded-log",
                        "task_type": "testlog2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "FAILED tests/test_auth.py"},
                        "gold": {"root_cause_files": ["src/auth.py"]},
                    },
                ],
            )
            write_jsonl(
                details_path,
                [
                    {
                        "sample_id": "hinted",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["tests/test_auth.py"],
                        "top_files": ["tests/test_auth.py"],
                        "metrics": {"Recall@5": 1, "Recall@10": 1, "Recall@20": 1, "MRR": 1, "gold_coverage@8k": 1},
                    },
                    {
                        "sample_id": "missing",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["tests/test_missing.py"],
                        "top_files": ["src/missing.py"],
                        "metrics": {"Recall@5": 0, "Recall@10": 0, "Recall@20": 0, "MRR": 0, "gold_coverage@8k": 0},
                    },
                    {
                        "sample_id": "easy",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "base_commit": "base2",
                        "gold_files": ["tests/test_auth.py"],
                        "gold_ranks": {"tests/test_auth.py": 1},
                        "top_files": ["src/auth.py", "tests/test_auth.py"],
                        "metrics": {"Recall@5": 1, "Recall@10": 1, "Recall@20": 1, "MRR": 1, "gold_coverage@8k": 1},
                    },
                    {
                        "sample_id": "excluded-log",
                        "task_type": "testlog2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["src/auth.py"],
                        "top_files": ["src/auth.py"],
                        "metrics": {"Recall@5": 1, "Recall@10": 1, "Recall@20": 1, "MRR": 1, "gold_coverage@8k": 1},
                    },
                ],
            )
            write_jsonl(
                chunks_a,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "tests/test_auth.py", "kind": "file"},
                    {"repo": "o/r", "base_commit": "base", "path": "src/missing.py", "kind": "file"},
                ],
            )
            write_jsonl(
                chunks_b,
                [
                    {"repo": "o/r", "base_commit": "base2", "path": "src/auth.py", "kind": "file"},
                    {"repo": "o/r", "base_commit": "base2", "path": "tests/test_auth.py", "kind": "file"},
                ],
            )
            write_jsonl(
                corpus_manifest_path,
                [
                    {"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_a)},
                    {"repo": "o/r", "base_commit": "base2", "status": "ok", "chunks_path": str(chunks_b)},
                ],
            )

            result = diagnose_benchmark(
                samples_path=samples_path,
                corpus_manifest_path=corpus_manifest_path,
                details_path=details_path,
                out_dir=out_dir,
                tasks=["code2test", "comment2context"],
            )
            summary = json.loads((out_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
            diagnostics = [json.loads(line) for line in (out_dir / "sample_diagnostics.jsonl").read_text().splitlines()]
            report = (out_dir / "report.md").read_text(encoding="utf-8")

            self.assertEqual(result["samples"], 3)
            self.assertEqual(summary["distribution"]["by_task"], {"code2test": 2, "comment2context": 1})
            self.assertNotIn("testlog2code", summary["distribution"]["by_task"])
            self.assertEqual(summary["gold_corpus"]["samples_with_missing_gold"], 1)
            self.assertEqual(summary["query_hints"]["samples_with_gold_path_hint"], 1)
            self.assertEqual(summary["buckets"]["too_easy_direct_hint"], 1)
            self.assertEqual(summary["buckets"]["invalid_missing_gold"], 1)
            self.assertEqual({row["sample_id"] for row in diagnostics}, {"hinted", "missing", "easy"})
            easy = next(row for row in diagnostics if row["sample_id"] == "easy")
            self.assertEqual(easy["given_file_ranks"], {"src/auth.py": 1})
            self.assertEqual(easy["context_gold_ranks"], {"tests/test_auth.py": 1})
            self.assertIn("V0.2 Decisions", report)
            self.assertTrue((out_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
