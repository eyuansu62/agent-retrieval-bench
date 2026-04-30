import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.crawler import crawl_commit_details_for_raw, fetch_commit_details


class FakeAPI:
    authenticated = True

    def __init__(self):
        self.paths = []

    def get(self, path, params=None, accept=None):
        self.paths.append(path)
        return type(
            "Response",
            (),
            {
                "body": {
                    "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                    "files": [
                        {"filename": "src/auth.py", "status": "modified", "additions": 2, "deletions": 1, "changes": 3},
                        {"status": "modified"},
                    ],
                }
            },
        )()


class CrawlerTests(unittest.TestCase):
    def test_fetch_commit_details_records_changed_files(self):
        api = FakeAPI()

        details = fetch_commit_details(api, "o/r", [{"sha": "abc123"}])

        self.assertEqual(api.paths, ["/repos/o/r/commits/abc123"])
        self.assertEqual(details[0]["sha"], "abc123")
        self.assertEqual(details[0]["files"], [{"filename": "src/auth.py", "status": "modified", "additions": 2, "deletions": 1, "changes": 3}])

    def test_crawl_commit_details_backfills_raw_pull_commits(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            repo_dir = raw / "o__r"
            repo_dir.mkdir()
            with (repo_dir / "pull_commits.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"repo": "o/r", "pr_number": 1, "type": "pull_commits", "data": [{"sha": "abc123"}]}) + "\n")

            result = crawl_commit_details_for_raw(api, raw, "o/r")
            rows = [json.loads(line) for line in (repo_dir / "commit_details.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["fetched_prs"], 1)
            self.assertEqual(rows[0]["pr_number"], 1)
            self.assertEqual(rows[0]["data"][0]["files"][0]["filename"], "src/auth.py")


if __name__ == "__main__":
    unittest.main()
