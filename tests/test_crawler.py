import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.crawler import crawl_commit_details_for_raw, crawl_pr_checks, fetch_commit_details


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


class FakeCrawlerAPI:
    authenticated = True

    def __init__(self, prs):
        self.prs = prs
        self.graphql_variables = []
        self.paginate_calls = []
        self.get_calls = []

    def graphql(self, query, variables=None):
        self.graphql_variables.append(variables or {})
        return type(
            "Response",
            (),
            {
                "body": {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "nodes": self.prs,
                                "pageInfo": {"hasNextPage": False, "endCursor": "NEXT"},
                            }
                        },
                        "rateLimit": {"remaining": 4999, "resetAt": "2026-01-01T00:00:00Z"},
                    }
                }
            },
        )()

    def paginate(self, path, params=None, accept=None):
        self.paginate_calls.append(path)
        if path.endswith("/files"):
            return [{"filename": "src/auth.py"}, {"filename": "tests/test_auth.py"}]
        if path.endswith("/commits"):
            return [{"sha": "commit-sha", "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}}}]
        return []

    def get(self, path, params=None, accept=None):
        self.get_calls.append(path)
        if path.endswith("/check-runs"):
            sha = path.split("/commits/", 1)[1].split("/", 1)[0]
            return type(
                "Response",
                (),
                {
                    "body": {
                        "check_runs": [
                            {
                                "id": int("".join(str(ord(char)) for char in sha)[:8]),
                                "name": "unit tests",
                                "conclusion": "failure",
                                "app": {"slug": "github-actions"},
                                "output": {"summary": "FAILED tests/test_auth.py::test_refresh"},
                            }
                        ]
                    }
                },
            )()
        return type(
            "Response",
            (),
            {
                "body": {
                    "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                    "files": [{"filename": "src/auth.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1}],
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

    def test_crawl_pr_checks_skips_complete_existing_prs_without_consuming_accept_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            repo_dir = raw / "o__r"
            repo_dir.mkdir()
            (repo_dir / "crawl_state.json").write_text(json.dumps({"repo": "o/r", "graphql_cursor": "CUR", "last_summary": {"accepted_prs": 5}}))
            write_rows(repo_dir / "pull_files.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "pull_files", "data": []}])
            write_rows(repo_dir / "pull_commits.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "pull_commits", "data": [{"sha": "old"}]}])
            write_rows(repo_dir / "commit_details.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "commit_details", "data": []}])
            write_rows(repo_dir / "check_runs.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "check_runs", "ref_type": "head", "sha": "old", "data": []}])
            api = FakeCrawlerAPI([pr_node(2), pr_node(1)])

            result = crawl_pr_checks(api, "o/r", raw, limit_prs=1)
            rows = [json.loads(line) for line in (repo_dir / "check_runs.jsonl").read_text().splitlines()]

            self.assertEqual(api.graphql_variables[0]["cursor"], "CUR")
            self.assertEqual(result["accepted_prs"], 1)
            self.assertEqual(result["existing_skipped"], 1)
            self.assertEqual(result["new_check_runs"], 2)
            self.assertEqual(result["failed_github_actions_jobs"], 2)
            self.assertFalse(any("/pulls/2/files" in path for path in api.paginate_calls))
            self.assertEqual(sorted(row["pr_number"] for row in rows), [1, 1, 2])

    def test_crawl_pr_checks_refresh_replaces_existing_check_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            repo_dir = raw / "o__r"
            repo_dir.mkdir()
            write_rows(repo_dir / "pull_files.jsonl", [{"repo": "o/r", "pr_number": 1, "type": "pull_files", "data": []}])
            write_rows(repo_dir / "pull_commits.jsonl", [{"repo": "o/r", "pr_number": 1, "type": "pull_commits", "data": [{"sha": "commit-sha"}]}])
            write_rows(repo_dir / "commit_details.jsonl", [{"repo": "o/r", "pr_number": 1, "type": "commit_details", "data": []}])
            write_rows(
                repo_dir / "check_runs.jsonl",
                [{"repo": "o/r", "pr_number": 1, "type": "check_runs", "ref_type": "head", "sha": "head-1", "data": [{"id": 1, "name": "old"}]}],
            )
            api = FakeCrawlerAPI([pr_node(1)])

            result = crawl_pr_checks(api, "o/r", raw, limit_prs=1, refresh_existing_checks=True)
            rows = [json.loads(line) for line in (repo_dir / "check_runs.jsonl").read_text().splitlines()]

            self.assertEqual(result["accepted_prs"], 1)
            self.assertEqual(len(rows), 2)
            self.assertNotIn("old", json.dumps(rows))

    def test_crawl_pr_checks_repair_empty_state_does_not_override_valid_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            repo_dir = raw / "o__r"
            repo_dir.mkdir()
            (repo_dir / "crawl_state.json").write_text(json.dumps({"repo": "o/r", "graphql_cursor": "GOOD", "last_summary": {"accepted_prs": 0}}))
            write_rows(repo_dir / "pull_requests.jsonl", [{"type": "pull_request", "repo": "o/r", "data": {"number": 99}}])
            api = FakeCrawlerAPI([pr_node(1)])

            result = crawl_pr_checks(api, "o/r", raw, limit_prs=1, repair_empty_state=True)

            self.assertEqual(api.graphql_variables[0]["cursor"], "GOOD")
            self.assertFalse(result["state_repaired"])

    def test_crawl_pr_checks_max_pages_bounds_history_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakeCrawlerAPI([pr_node(1)])

            result = crawl_pr_checks(api, "o/r", raw, limit_prs=10, max_pages=1)

            self.assertEqual(result["pages"], 1)
            self.assertEqual(len(api.graphql_variables), 1)


def pr_node(number: int):
    return {
        "number": number,
        "title": f"PR {number}",
        "body": "",
        "url": f"https://github.com/o/r/pull/{number}",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "mergedAt": "2026-01-01T00:00:00Z",
        "baseRefOid": f"base-{number}",
        "headRefOid": f"head-{number}",
        "mergeCommit": {"oid": f"merge-{number}"},
        "changedFiles": 2,
    }


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    unittest.main()
