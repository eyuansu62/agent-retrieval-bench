import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.openai_context_agent import split_strict_final_files
from agent_retrieval_bench.trajectory_release import audit_strict_context_run, forbidden_release_member, package_trajectory_release


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class TrajectoryReleaseTests(unittest.TestCase):
    def test_split_strict_final_files_moves_unread_paths_to_suggestions(self):
        result = split_strict_final_files(
            {
                "context_files": ["src/read.py", "src/unread.py", "src/read.py"],
                "suggested_unread_files": ["docs/notes.md"],
            },
            ["src/read.py", "tests/read_test.py"],
        )

        self.assertEqual(result["raw_final_files"], ["src/read.py", "src/unread.py"])
        self.assertEqual(result["final_files"], ["src/read.py"])
        self.assertEqual(result["suggested_unread_files"], ["docs/notes.md", "src/unread.py"])

    def test_audit_strict_context_run_accepts_clean_release_with_raw_unread_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            corpus = root / "corpus" / "v1_2"
            base = root / "trajectory_runs" / "v1_4" / "run"
            run_name = "openai_gpt54mini_v2_strict_context"
            chunks = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                derived / "samples.jsonl",
                [{"id": "s1", "task_type": "code2test", "repo": "o/r", "base_commit": "base"}],
            )
            write_jsonl(
                chunks,
                [
                    {"path": "src/a.py", "kind": "file", "start_line": 1, "end_line": 10, "text": "a"},
                    {"path": "src/b.py", "kind": "file", "start_line": 1, "end_line": 10, "text": "b"},
                    {"path": "src/c.py", "kind": "file", "start_line": 1, "end_line": 10, "text": "c"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks)}],
            )
            write_json(
                base / f"{run_name}_manifest.json",
                {"model": "gpt-5.4-mini", "completed": 1, "skipped_existing": 0, "failures": []},
            )
            write_jsonl(
                base / f"logs_{run_name}" / "s1.jsonl",
                [
                    {"sample_id": "s1", "path": "src/a.py"},
                    {"sample_id": "s1", "path": "src/b.py"},
                    {"sample_id": "s1", "path": "src/c.py"},
                ],
            )
            (base / f"answers_{run_name}").mkdir(parents=True)
            (base / f"answers_{run_name}" / "s1.md").write_text("# answer\n", encoding="utf-8")
            write_json(
                base / f"traces_{run_name}" / "s1.json",
                {
                    "sample_id": "s1",
                    "repo": "o/r",
                    "base_commit": "base",
                    "read_paths": ["src/a.py", "src/b.py", "src/c.py"],
                    "raw_final_files": ["src/a.py", "src/unread.py"],
                    "final_files": ["src/a.py"],
                    "suggested_unread_files": ["src/unread.py"],
                    "actions": [],
                },
            )

            result = audit_strict_context_run(
                base=base,
                run_name=run_name,
                derived=derived,
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                out_path=root / "audit.json",
                markdown_out=root / "audit.md",
            )

            self.assertEqual(result["verdict"], "pass")
            self.assertEqual(result["counts"]["logs"], 1)
            self.assertEqual(result["counts"]["read_steps_total"], 3)
            self.assertEqual(result["issues"]["final_unread_samples"], [])
            self.assertEqual(len(result["issues"]["raw_unread_samples"]), 1)
            self.assertEqual(result["issues"]["missing_read_path_samples"], [])
            self.assertEqual(result["issues"]["missing_final_path_samples"], [])

    def test_forbidden_release_member_rejects_benchmark_corpus_and_smoke(self):
        self.assertTrue(forbidden_release_member("benchmark/v1_3/samples.jsonl"))
        self.assertTrue(forbidden_release_member("corpus/v1_2/corpus_manifest.jsonl"))
        self.assertTrue(forbidden_release_member("reports/v1_4/release_smoke.json"))
        self.assertFalse(forbidden_release_member("reports/v1_4/model_report.json"))

    def test_package_release_requires_checksum_path_to_match_release_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            include = data / "reports" / "v1_4" / "model_report.json"
            write_json(include, {"ok": True})

            with self.assertRaises(ValueError):
                package_trajectory_release(
                    data_root=data,
                    release_dir=root / "out",
                    archive_name="smoke.tar.zst",
                    include_paths=[include],
                    checksum_path_in_release="releases/smoke",
                )


if __name__ == "__main__":
    unittest.main()
