import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.trace_preflight import trace_preflight


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TracePreflightTests(unittest.TestCase):
    def test_counts_only_repo_owned_root_cause_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 1,
                "url": "https://github.com/o/r/pull/1",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 1,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "review_comments.jsonl",
                [
                    {
                        "pr_number": 1,
                        "data": [
                            {
                                "id": 10,
                                "body": 'Traceback (most recent call last):\n  File "src/auth.py", line 7, in refresh\nRuntimeError: boom',
                            },
                            {
                                "id": 11,
                                "body": 'Traceback (most recent call last):\n  File "tests/test_auth.py", line 3, in test_refresh\nAssertionError',
                            },
                        ],
                    }
                ],
            )

            result = trace_preflight(root / "raw", root / "trace")
            rows = [json.loads(line) for line in (root / "trace" / "candidates.jsonl").read_text().splitlines()]

            self.assertEqual(result["candidates"], 1)
            self.assertEqual(rows[0]["root_cause_files"], ["src/auth.py"])
            self.assertEqual(result["quality_gate"]["ready_for_trace_audit"], False)


if __name__ == "__main__":
    unittest.main()
