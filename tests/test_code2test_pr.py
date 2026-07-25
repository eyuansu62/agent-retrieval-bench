import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.code2test_pr import build_pr_code2test_sample, mine_code2test_prs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def pr(number=1, body=None):
    return {
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "title": "Fix auth cache regression",
        "body": body
        if body is not None
        else "Fixes a cache invalidation bug and adds regression coverage for token refresh behavior.",
        "baseRefOid": "base",
        "mergeCommit": {"oid": "fix"},
        "createdAt": "2026-01-01T00:00:00Z",
        "mergedAt": "2026-01-02T00:00:00Z",
    }


def files(paths):
    return {"data": [{"filename": path, "status": "modified"} for path in paths], "pr_number": 1}


class Code2TestPrMiningTests(unittest.TestCase):
    def test_builds_pr_level_sample_without_test_path_in_query(self):
        sample, reason = build_pr_code2test_sample(
            repo="o/r",
            pr_number=1,
            pr=pr(),
            files_record=files(["src/auth/cache.py", "tests/auth/test_refresh.py"]),
            details_record={
                "data": [
                    {
                        "sha": "abc",
                        "files": [
                            {"filename": "src/auth/cache.py"},
                            {"filename": "tests/auth/test_refresh.py"},
                        ],
                    }
                ]
            },
        )

        self.assertIsNone(reason)
        self.assertEqual(sample["task_type"], "code2test")
        self.assertEqual(sample["gold"]["related_tests"], ["tests/auth/test_refresh.py"])
        query_text = json.dumps(sample["query"])
        self.assertNotIn("tests/auth/test_refresh.py", query_text)
        self.assertNotIn("test_refresh.py", query_text)
        self.assertIn("commit_detail_confirms_source_and_test_changes", sample["metadata"]["evidence"]["signals"])

    def test_drops_query_that_mentions_test_path_or_basename(self):
        sample, reason = build_pr_code2test_sample(
            repo="o/r",
            pr_number=1,
            pr=pr(body="Fix auth behavior; update tests/auth/test_refresh.py for this regression."),
            files_record=files(["src/auth/cache.py", "tests/auth/test_refresh.py"]),
        )

        self.assertIsNone(sample)
        self.assertEqual(reason, "test_path_leak")

    def test_mine_excludes_audited_clusters_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr(), "repo": "o/r", "type": "pull_request"}])
            write_jsonl(raw_repo / "pull_files.jsonl", [files(["src/auth/cache.py", "tests/auth/test_refresh.py"])])
            write_jsonl(raw_repo / "commit_details.jsonl", [])
            audit = root / "audited.csv"
            with audit.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "task_type", "repo", "verdict", "keep"])
                writer.writeheader()
                writer.writerow({"sample_id": "old", "task_type": "code2test", "repo": "o/r", "verdict": "valid", "keep": "true"})
            pool = root / "pool.jsonl"
            write_jsonl(
                pool,
                [
                    {
                        "sample_id": "old",
                        "repo": "o/r",
                        "pr_url": "https://github.com/o/r/pull/1",
                        "gold_files": ["tests/auth/test_refresh.py"],
                    }
                ],
            )

            result = mine_code2test_prs(
                raw_dir=root / "raw",
                out_dir=root / "benchmark",
                report_dir=root / "reports",
                audit_path=audit,
                audited_pool_path=pool,
            )

            self.assertEqual(result["total"], 0)
            self.assertEqual(result["dropped"]["already_audited"], 1)
            self.assertTrue((root / "benchmark" / "samples.jsonl").exists())
            self.assertTrue((root / "reports" / "audit_samples.csv").exists())


if __name__ == "__main__":
    unittest.main()
