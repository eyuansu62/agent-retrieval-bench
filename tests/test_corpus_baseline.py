import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.baseline import evaluate_lexical_baseline
from agent_retrieval_bench.cli import default_baseline_details_path
from agent_retrieval_bench.corpus import build_commit_chunks, chunks_for_file, sample_paths_from_derived
from agent_retrieval_bench.curate import export_curated_samples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class CorpusBaselineTests(unittest.TestCase):
    def test_chunks_for_file_extracts_file_and_symbols(self):
        chunks = chunks_for_file(
            "o/r",
            "base",
            "src/auth.py",
            "class Auth:\n    pass\n\ndef refresh_token(value):\n    return value\n",
        )

        self.assertEqual(chunks[0]["kind"], "file")
        self.assertIn("Auth", {chunk["symbol"] for chunk in chunks})
        self.assertIn("refresh_token", {chunk["symbol"] for chunk in chunks})

    @unittest.skipIf(shutil.which("git") is None, "git is required")
    def test_build_commit_chunks_reads_bare_repo_at_base_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            bare = Path(tmp) / "o__r.git"
            out = Path(tmp) / "corpus"
            work.mkdir()
            self._git(["init"], work)
            self._git(["config", "user.email", "test@example.com"], work)
            self._git(["config", "user.name", "Test User"], work)
            (work / "src").mkdir()
            (work / "src" / "auth.py").write_text("def refresh_token(value):\n    return value\n", encoding="utf-8")
            self._git(["add", "."], work)
            self._git(["commit", "-m", "seed"], work)
            base_commit = self._git(["rev-parse", "HEAD"], work).stdout.strip()
            subprocess.run(["git", "clone", "--bare", str(work), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result = build_commit_chunks("o/r", base_commit, bare, out)
            rows = [json.loads(line) for line in Path(result["chunks_path"]).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["status"], "ok")
            self.assertIn("src/auth.py", {row["path"] for row in rows})
            self.assertIn("refresh_token", {row["symbol"] for row in rows})

    def test_build_commit_chunks_reuses_existing_chunk_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus"
            chunks_path = out / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "src/auth.py", "kind": "file", "symbol": ""},
                    {"repo": "o/r", "base_commit": "base", "path": "src/auth.py", "kind": "symbol", "symbol": "refresh_token"},
                ],
            )

            result = build_commit_chunks("o/r", "base", Path(tmp) / "missing.git", out)

            self.assertTrue(result["reused"])
            self.assertEqual(result["chunk_count"], 2)
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["symbol_count"], 1)

    def test_lexical_baseline_reports_recall_and_skips_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "s1",
                        "task_type": "testlog2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "refresh token assertion failed in auth"},
                        "gold": {"root_cause_files": ["src/auth.py"], "related_tests": [], "fix_commit": "fix"},
                    },
                    {
                        "id": "s2",
                        "task_type": "testlog2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "diff --git a/x b/x"},
                        "gold": {"root_cause_files": ["src/x.py"], "related_tests": [], "fix_commit": "fix"},
                    },
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {"chunk_id": "c1", "path": "src/auth.py", "symbol": "refresh_token", "kind": "symbol", "text": "refresh token auth"},
                    {"chunk_id": "c2", "path": "src/other.py", "symbol": "", "kind": "file", "text": "unrelated"},
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = evaluate_lexical_baseline([samples], corpus_dir)

            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["skipped"]["query_leakage"], 1)
            self.assertEqual(result["metrics"]["testlog2code"]["Recall@5"], 1.0)
            self.assertEqual(result["metrics"]["testlog2code"]["MRR"], 1.0)

    def test_curated_export_and_baseline_use_keep_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "derived"
            keep_list = root / "keep.jsonl"
            benchmark = root / "benchmark"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            valid_sample = {
                "id": "keep-me",
                "task_type": "code2test",
                "repo": "o/r",
                "base_commit": "base",
                "query": {"changed_file": "src/auth.py"},
                "gold": {"root_cause_files": ["src/auth.py"], "related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
            }
            noisy_sample = {
                "id": "drop-me",
                "task_type": "code2test",
                "repo": "o/r",
                "base_commit": "base",
                "query": {"changed_file": "src/noisy.py"},
                "gold": {"root_cause_files": ["src/noisy.py"], "related_tests": ["tests/test_noisy.py"], "fix_commit": "fix"},
            }
            write_jsonl(derived / "code2test.jsonl", [valid_sample, noisy_sample])
            write_jsonl(keep_list, [{"sample_id": "keep-me", "task_type": "code2test", "repo": "o/r", "verdict": "valid"}])
            write_jsonl(
                chunks_path,
                [
                    {"chunk_id": "c1", "path": "tests/test_auth.py", "symbol": "", "kind": "file", "text": "auth test"},
                    {"chunk_id": "c2", "path": "tests/test_noisy.py", "symbol": "", "kind": "file", "text": "noisy test"},
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            export_result = export_curated_samples(derived, keep_list, benchmark)
            baseline_result = evaluate_lexical_baseline([derived / "code2test.jsonl"], corpus_dir, keep_list=keep_list)

            self.assertEqual(export_result["total"], 1)
            self.assertEqual(sample_paths_from_derived(benchmark), [benchmark / "samples.jsonl"])
            with (benchmark / "samples.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 1)
            self.assertEqual(baseline_result["evaluated"], 1)

    def test_curated_export_ignores_keep_ids_for_excluded_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "derived"
            keep_list = root / "keep.jsonl"
            benchmark = root / "benchmark"
            write_jsonl(
                derived / "code2test.jsonl",
                [
                    {
                        "id": "keep-code",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth.py"},
                        "gold": {"related_tests": ["tests/test_auth.py"]},
                    }
                ],
            )
            write_jsonl(
                derived / "testlog2code.jsonl",
                [
                    {
                        "id": "keep-log",
                        "task_type": "testlog2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "FAILED tests/test_auth.py"},
                        "gold": {"root_cause_files": ["src/auth.py"]},
                    }
                ],
            )
            write_jsonl(
                keep_list,
                [
                    {"sample_id": "keep-code", "task_type": "code2test", "repo": "o/r", "verdict": "valid"},
                    {"sample_id": "keep-log", "task_type": "testlog2code", "repo": "o/r", "verdict": "valid"},
                ],
            )

            result = export_curated_samples(derived, keep_list, benchmark, tasks=["code2test"])

            self.assertEqual(result["total"], 1)
            self.assertEqual(result["missing_keep_ids"], [])

    def test_baseline_details_default_follows_summary_path(self):
        self.assertEqual(
            default_baseline_details_path(Path("data/eval/v0_1/lexical_summary.json")),
            Path("data/eval/v0_1/lexical_details.jsonl"),
        )

    def _git(self, args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
