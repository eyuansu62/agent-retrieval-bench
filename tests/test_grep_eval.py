import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.cli import default_baseline_details_path, default_grep_summary_path
from agent_retrieval_bench.grep_eval import evaluate_grep_baseline, extract_search_patterns, rank_chunks_by_grep


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class GrepEvalTests(unittest.TestCase):
    def test_extract_search_patterns_keeps_high_signal_terms(self):
        query = json.dumps(
            {
                "failure_excerpt": "listeners_test.go fails because sharedQUICState lacks `getEncryptedClientHelloKeys`",
                "pr_title": "Support --overwrite for create-vite",
            }
        )

        patterns = extract_search_patterns(query)
        by_term = {pattern.term: pattern.kind for pattern in patterns}

        self.assertEqual(by_term["getEncryptedClientHelloKeys"], "code")
        self.assertEqual(by_term["listeners_test.go"], "file_path")
        self.assertEqual(by_term["sharedQUICState"], "identifier")
        self.assertEqual(by_term["--overwrite"], "flag")
        self.assertNotIn("failure_excerpt", by_term)

    def test_rank_chunks_by_grep_aggregates_hits_to_file_ranking(self):
        chunks = [
            {
                "chunk_id": "c1",
                "path": "src/auth.py",
                "kind": "symbol",
                "symbol": "refresh_token",
                "text": "def refresh_token(user):\n    return user.token\n",
            },
            {
                "chunk_id": "c2",
                "path": "src/other.py",
                "kind": "file",
                "symbol": "",
                "text": "def render_page():\n    return page\n",
            },
        ]

        ranked, patterns, scores = rank_chunks_by_grep('{"failure_excerpt": "NameError in refresh_token"}', chunks)

        self.assertEqual(ranked[0]["path"], "src/auth.py")
        self.assertIn("refresh_token", {pattern.term for pattern in patterns})
        self.assertEqual(scores[0]["path"], "src/auth.py")

    def test_rank_chunks_by_grep_returns_empty_ranking_without_patterns(self):
        ranked, patterns, scores = rank_chunks_by_grep('{"pr_title": "fix the issue"}', [{"path": "src/auth.py", "text": "auth"}])

        self.assertEqual(ranked, [])
        self.assertEqual(patterns, [])
        self.assertEqual(scores, [])

    def test_evaluate_grep_outputs_compatible_summary_and_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "trace2code.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "eval" / "grep_strict_details.jsonl"
            summary_path = root / "eval" / "grep_strict_summary.json"
            write_jsonl(
                samples_path,
                [
                    {
                        "id": "s1",
                        "task_type": "trace2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"failure_excerpt": "NameError: refresh_token is missing"},
                        "gold": {"root_cause_files": ["src/auth.py"], "related_tests": []},
                    }
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "c1",
                        "path": "src/auth.py",
                        "kind": "symbol",
                        "symbol": "refresh_token",
                        "text": "def refresh_token(user):\n    return user.token\n",
                    },
                    {"chunk_id": "c2", "path": "src/other.py", "kind": "file", "symbol": "", "text": "unrelated"},
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = evaluate_grep_baseline([samples_path], corpus_dir, out_path=summary_path, details_path=details_path)
            detail = json.loads(details_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(result["mode"], "grep")
            self.assertEqual(result["model"], "grep-strict")
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["metrics"]["trace2code"]["Recall@5"], 1.0)
            self.assertEqual(detail["top_files"][0], "src/auth.py")
            self.assertEqual(detail["gold_ranks"], {"src/auth.py": 1})
            self.assertIn("patterns", detail)
            self.assertTrue(summary_path.exists())

    def test_grep_summary_default_paths_match_eval_conventions(self):
        self.assertEqual(default_grep_summary_path("strict", "all_files"), Path("data/eval/v0/grep_strict_summary.json"))
        self.assertEqual(default_grep_summary_path("expanded", "tests_only"), Path("data/eval/v0/grep_expanded_tests_only_summary.json"))
        self.assertEqual(
            default_baseline_details_path(Path("data/eval/v1/grep_strict_summary.json")),
            Path("data/eval/v1/grep_strict_details.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
