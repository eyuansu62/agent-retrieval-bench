from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.bcy_curve import count_tokens, report_bcy_budget_curve


class BcyCurveTests(unittest.TestCase):
    def test_tokenizer_excludes_whitespace(self) -> None:
        self.assertEqual(count_tokens("alpha beta\ngamma"), 3)
        self.assertEqual(count_tokens("### gold.py\n"), 6)

    def test_budget_curve_uses_content_tokens_for_gold_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks = root / "chunks.jsonl"
            write_jsonl(
                chunks,
                [
                    file_chunk("owner/repo", "abc", "gold.py", "alpha beta gamma"),
                    file_chunk("owner/repo", "abc", "noise.py", "one two three four five six"),
                ],
            )
            manifest = root / "corpus_manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "repo": "owner/repo",
                        "base_commit": "abc",
                        "status": "ok",
                        "chunks_path": str(chunks),
                    }
                ],
            )
            details = root / "details.jsonl"
            write_jsonl(
                details,
                [
                    {
                        "sample_id": "s1",
                        "task_type": "code2test",
                        "repo": "owner/repo",
                        "base_commit": "abc",
                        "gold_files": ["gold.py"],
                        "top_files": ["gold.py", "noise.py"],
                    }
                ],
            )

            report = report_bcy_budget_curve(
                corpus_manifest_path=manifest,
                out_path=root / "bcy.json",
                markdown_out_path=root / "bcy.md",
                budgets=(2, 10, 20),
                runs=(("toy", "test", details),),
                coverage_thresholds=(1, 2, 4),
            )

            overall = report["runs"][0]["overall"]
            self.assertEqual(overall["BCY@2"], 0.0)
            self.assertEqual(overall["BCY@10"], 1.0)
            self.assertEqual(overall["BCY@20"], 1.0)
            self.assertEqual(overall["coverage_sensitivity"]["10"]["1"], 1.0)
            self.assertEqual(overall["coverage_sensitivity"]["10"]["2"], 1.0)
            self.assertEqual(overall["coverage_sensitivity"]["10"]["4"], 1.0)
            self.assertIn("BCY@10", (root / "bcy.md").read_text())

    def test_partial_gold_requires_threshold_but_short_complete_file_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks = root / "chunks.jsonl"
            write_jsonl(chunks, [file_chunk("owner/repo", "abc", "gold.py", "alpha beta gamma")])
            manifest = root / "corpus_manifest.jsonl"
            write_jsonl(
                manifest,
                [{"repo": "owner/repo", "base_commit": "abc", "status": "ok", "chunks_path": str(chunks)}],
            )
            details = root / "details.jsonl"
            write_jsonl(
                details,
                [{
                    "sample_id": "s1",
                    "task_type": "code2test",
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "gold_files": ["gold.py"],
                    "top_files": ["gold.py"],
                }],
            )

            report = report_bcy_budget_curve(
                corpus_manifest_path=manifest,
                out_path=root / "bcy.json",
                markdown_out_path=root / "bcy.md",
                budgets=(8, 10),
                runs=(("toy", "test", details),),
                coverage_thresholds=(1, 2, 4),
            )

            overall = report["runs"][0]["overall"]
            self.assertEqual(overall["coverage_sensitivity"]["8"]["1"], 1.0)
            self.assertEqual(overall["coverage_sensitivity"]["8"]["2"], 1.0)
            self.assertEqual(overall["coverage_sensitivity"]["8"]["4"], 0.0)
            self.assertEqual(overall["coverage_sensitivity"]["10"]["4"], 1.0)


def file_chunk(repo: str, base_commit: str, path: str, text: str) -> dict[str, object]:
    return {
        "repo": repo,
        "base_commit": base_commit,
        "chunk_id": path,
        "kind": "file",
        "path": path,
        "symbol": "",
        "start_line": 1,
        "end_line": 1,
        "text": text,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()
