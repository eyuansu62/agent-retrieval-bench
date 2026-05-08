import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.cli import default_baseline_details_path, default_repomap_summary_path
from agent_retrieval_bench.repomap_eval import build_repomap_index, evaluate_repomap_baseline


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class RepoMapEvalTests(unittest.TestCase):
    def test_build_repomap_index_links_symbols_imports_and_tests(self):
        chunks = [
            {
                "path": "src/auth.py",
                "kind": "file",
                "symbol": "",
                "text": "def refresh_token(user):\n    return user.token\n",
            },
            {"path": "src/auth.py", "kind": "symbol", "symbol": "refresh_token", "text": "def refresh_token(user):"},
            {
                "path": "tests/test_auth.py",
                "kind": "file",
                "symbol": "",
                "text": "from src.auth import refresh_token\n\ndef test_refresh_token():\n    refresh_token(user)\n",
            },
            {"path": "tests/test_auth.py", "kind": "symbol", "symbol": "test_refresh_token", "text": "def test_refresh_token():"},
            {"path": "src/run.py", "kind": "symbol", "symbol": "run", "text": "def run():"},
        ]

        index = build_repomap_index(chunks)

        self.assertEqual(index.nodes["src/auth.py"].kind, "source")
        self.assertEqual(index.nodes["tests/test_auth.py"].kind, "test")
        self.assertIn("refresh_token", index.nodes["src/auth.py"].symbols)
        self.assertNotIn("run", index.nodes["src/run.py"].symbols)
        self.assertIn("tests/test_auth.py", index.graph["src/auth.py"])
        self.assertGreater(index.stats["edge_count"], 0)

    def test_evaluate_repomap_outputs_compatible_summary_and_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "code2test.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "eval" / "repomap_details.jsonl"
            summary_path = root / "eval" / "repomap_summary.json"
            write_jsonl(
                samples_path,
                [
                    {
                        "id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth.py", "pr_title": "Refresh token behavior"},
                        "gold": {
                            "root_cause_files": ["src/auth.py"],
                            "related_tests": ["tests/test_auth.py"],
                            "fix_commit": "fix123",
                        },
                    }
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "f1",
                        "path": "src/auth.py",
                        "kind": "file",
                        "symbol": "",
                        "text": "def refresh_token(user):\n    return user.token\n",
                    },
                    {
                        "chunk_id": "s1",
                        "path": "src/auth.py",
                        "kind": "symbol",
                        "symbol": "refresh_token",
                        "text": "def refresh_token(user):",
                    },
                    {
                        "chunk_id": "f2",
                        "path": "tests/test_auth.py",
                        "kind": "file",
                        "symbol": "",
                        "text": "from src.auth import refresh_token\n\ndef test_refresh_token():\n    refresh_token(user)\n",
                    },
                    {
                        "chunk_id": "s2",
                        "path": "tests/test_auth.py",
                        "kind": "symbol",
                        "symbol": "test_refresh_token",
                        "text": "def test_refresh_token():",
                    },
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = evaluate_repomap_baseline(
                [samples_path],
                corpus_dir,
                out_path=summary_path,
                details_path=details_path,
                candidate_filter="tests_only",
            )
            detail = json.loads(details_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(result["mode"], "repomap")
            self.assertEqual(result["model"], "aider-style-repomap")
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["skipped"], {})
            self.assertEqual(result["metrics"]["code2test"]["Recall@5"], 1.0)
            self.assertEqual(detail["top_files"][0], "tests/test_auth.py")
            self.assertIn("top_file_scores", detail)
            self.assertIn("repo_map_stats", detail)

    def test_repomap_summary_default_paths_match_eval_conventions(self):
        self.assertEqual(default_repomap_summary_path("all_files"), Path("data/eval/v0/repomap_summary.json"))
        self.assertEqual(default_repomap_summary_path("code_only"), Path("data/eval/v0/repomap_code_only_summary.json"))
        self.assertEqual(
            default_baseline_details_path(Path("data/eval/v1/repomap_summary.json")),
            Path("data/eval/v1/repomap_details.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
