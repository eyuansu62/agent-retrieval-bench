from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.rank_fusion import report_rank_fusion


class RankFusionTests(unittest.TestCase):
    def test_rrf_promotes_shared_gold_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            write_jsonl(
                eval_dir / "a_details.jsonl",
                [
                    detail(
                        "s1",
                        "code2test",
                        ["src/noise.py", "tests/test_target.py"],
                        ["tests/test_target.py"],
                    ),
                    detail("s2", "trace2code", ["src/root.py", "src/other.py"], ["src/root.py"]),
                ],
            )
            write_jsonl(
                eval_dir / "b_details.jsonl",
                [
                    detail(
                        "s1",
                        "code2test",
                        ["tests/test_target.py", "src/other.py"],
                        ["tests/test_target.py"],
                    ),
                    detail("s2", "trace2code", ["src/other.py", "src/root.py"], ["src/root.py"]),
                ],
            )

            result = report_rank_fusion(
                eval_dir=eval_dir,
                out_dir=root / "fusion",
                report_out=root / "rank_fusion.md",
                json_out=root / "rank_fusion.json",
                components={
                    "a": {"label": "A", "details": "a_details.jsonl"},
                    "b": {"label": "B", "details": "b_details.jsonl"},
                },
                fusions=[{"name": "rrf_a_b", "label": "RRF(A+B)", "components": ["a", "b"]}],
            )

            self.assertEqual(result["fusions"], 1)
            summary = json.loads((root / "fusion" / "rrf_a_b_summary.json").read_text())
            self.assertEqual(summary["metrics"]["overall"]["samples"], 2)
            self.assertEqual(summary["metrics"]["overall"]["Recall@20"], 1.0)
            self.assertEqual(summary["metrics"]["overall"]["Any@20"], 1.0)
            rows = [json.loads(line) for line in (root / "fusion" / "rrf_a_b_details.jsonl").read_text().splitlines()]
            first = {row["sample_id"]: row for row in rows}
            self.assertEqual(first["s1"]["top_files"][0], "tests/test_target.py")
            self.assertIn("| RRF(A+B) | A + B |", (root / "rank_fusion.md").read_text())


def detail(sample_id: str, task_type: str, top_files: list[str], gold_files: list[str]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "repo": "owner/repo",
        "base_commit": "abc123",
        "candidate_filter": "all_files",
        "gold_files": gold_files,
        "top_files": top_files,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()
