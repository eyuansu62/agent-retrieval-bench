import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.crawler import crawl_commit_details_for_raw, crawl_pr_checks, crawl_repo, crawl_review_comment_prs, fetch_commit_details


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


class FakeReviewCommentAPI:
    authenticated = False

    def __init__(self):
        self.get_calls = []
        self.paginate_calls = []

    def get(self, path, params=None, accept=None):
        self.get_calls.append((path, params or {}))
        if path.endswith("/pulls/comments"):
            return response(
                [
                    {
                        "id": 1,
                        "body": "Could this also cover the auth retry behavior with a test?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:10:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/7",
                        "user": {"login": "reviewer", "type": "User"},
                    },
                    {
                        "id": 2,
                        "body": "Bot suggestion that should not drive candidate selection.",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:11:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/8",
                        "user": {"login": "coderabbitai[bot]", "type": "Bot"},
                    },
                    {
                        "id": 3,
                        "body": "Please add a coverage test for this retry branch.",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:12:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/8",
                        "in_reply_to_id": 1,
                        "user": {"login": "reviewer", "type": "User"},
                    },
                    {
                        "id": 4,
                        "body": "Could we add a test for this behavior?\n\nLearnt from: hidden bot analysis",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:13:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/9",
                        "user": {"login": "reviewer", "type": "User"},
                    },
                ]
            )
        if path.endswith("/pulls/7"):
            return response(
                {
                    "number": 7,
                    "title": "Fix auth retry",
                    "body": "",
                    "html_url": "https://github.com/o/r/pull/7",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T01:00:00Z",
                    "merged_at": "2026-01-01T02:00:00Z",
                    "base": {"ref": "main", "sha": "base-sha"},
                    "head": {"ref": "branch", "sha": "head-sha"},
                    "merge_commit_sha": "merge-sha",
                    "changed_files": 2,
                    "labels": [],
                }
            )
        if path.endswith("/commits/fix-sha"):
            return response(
                {
                    "commit": {"committer": {"date": "2026-01-01T00:30:00Z"}},
                    "files": [{"filename": "tests/test_auth.py", "status": "modified", "additions": 3, "deletions": 0, "changes": 3}],
                }
            )
        raise AssertionError(f"unexpected get path {path}")

    def paginate(self, path, params=None, accept=None):
        self.paginate_calls.append(path)
        if path.endswith("/pulls/7/files"):
            return [{"filename": "src/auth.py", "status": "modified"}, {"filename": "tests/test_auth.py", "status": "modified"}]
        if path.endswith("/pulls/7/commits"):
            return [{"sha": "fix-sha", "commit": {"committer": {"date": "2026-01-01T00:30:00Z"}}}]
        return []


class FakePagedReviewCommentAPI:
    authenticated = False

    def __init__(self):
        self.get_calls = []

    def get(self, path, params=None, accept=None):
        self.get_calls.append((path, params or {}))
        if not path.endswith("/pulls/comments"):
            raise AssertionError(f"unexpected get path {path}")
        page = int((params or {}).get("page") or 1)
        if page == 1:
            comments = [review_comment(index) for index in range(100)]
            return response_with_headers(comments, {"link": '<https://api.github.test?page=2>; rel="next"'})
        if page == 2:
            return response_with_headers([review_comment(100), review_comment(101)], {})
        return response_with_headers([], {})


class FakeSkippedReviewCommentAPI:
    authenticated = False

    def __init__(self):
        self.get_calls = []
        self.paginate_calls = []

    def get(self, path, params=None, accept=None):
        self.get_calls.append((path, params or {}))
        if path.endswith("/pulls/comments"):
            return response(
                [
                    {
                        "id": 1,
                        "body": "Could we add a coverage test for this behavior?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:10:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/9",
                        "user": {"login": "reviewer", "type": "User"},
                    },
                    {
                        "id": 2,
                        "body": "Should this behavior also have a regression test?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:11:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/9",
                        "user": {"login": "reviewer", "type": "User"},
                    },
                    {
                        "id": 3,
                        "body": "Could this also cover the auth retry behavior with a test?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:12:00Z",
                        "pull_request_url": "https://api.github.com/repos/o/r/pulls/8",
                        "user": {"login": "reviewer", "type": "User"},
                    },
                ]
            )
        if path.endswith("/pulls/9"):
            return response(
                {
                    "number": 9,
                    "title": "Draft auth retry",
                    "body": "",
                    "html_url": "https://github.com/o/r/pull/9",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T01:00:00Z",
                    "merged_at": None,
                    "base": {"ref": "main", "sha": "base-sha"},
                    "head": {"ref": "branch", "sha": "head-sha"},
                    "merge_commit_sha": None,
                    "changed_files": 2,
                    "labels": [],
                }
            )
        if path.endswith("/pulls/8"):
            return response(
                {
                    "number": 8,
                    "title": "Fix auth retry",
                    "body": "",
                    "html_url": "https://github.com/o/r/pull/8",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T01:00:00Z",
                    "merged_at": "2026-01-01T02:00:00Z",
                    "base": {"ref": "main", "sha": "base-sha"},
                    "head": {"ref": "branch", "sha": "head-sha"},
                    "merge_commit_sha": "merge-sha",
                    "changed_files": 2,
                    "labels": [],
                }
            )
        raise AssertionError(f"unexpected get path {path}")

    def paginate(self, path, params=None, accept=None):
        self.paginate_calls.append(path)
        return []


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

    def test_crawl_repo_skips_complete_existing_prs_without_refetching_raw_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            repo_dir = raw / "o__r"
            repo_dir.mkdir()
            write_rows(repo_dir / "pull_files.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "pull_files", "data": []}])
            write_rows(repo_dir / "pull_file_summary.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "pull_file_summary", "implementation": [], "tests": [], "ignored": []}])
            write_rows(repo_dir / "pull_commits.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "pull_commits", "data": [{"sha": "old"}]}])
            write_rows(repo_dir / "commit_details.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "commit_details", "data": []}])
            write_rows(repo_dir / "review_comments.jsonl", [{"repo": "o/r", "pr_number": 2, "type": "review_comments", "data": []}])
            api = FakeCrawlerAPI([pr_node(2), pr_node(1)])

            result = crawl_repo(api, "o/r", raw, limit_prs=1, include_checks=False)
            pull_rows = [json.loads(line) for line in (repo_dir / "pull_files.jsonl").read_text().splitlines()]

            self.assertEqual(result["accepted_prs"], 1)
            self.assertEqual(result["existing_skipped"], 1)
            self.assertEqual(result["new_pull_files"], 1)
            self.assertFalse(any("/pulls/2/files" in path for path in api.paginate_calls))
            self.assertEqual(sorted(row["pr_number"] for row in pull_rows), [1, 2])

    def test_crawl_repo_persists_state_after_each_successful_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FailingSecondPRAPI([pr_node(1), pr_node(2)])

            with self.assertRaises(RuntimeError):
                crawl_repo(api, "o/r", raw, limit_prs=2, include_checks=False)

            state = json.loads((raw / "o__r" / "crawl_state.json").read_text())
            self.assertEqual(state["last_summary"]["accepted_prs"], 1)
            self.assertEqual(state["last_summary"]["new_pull_files"], 1)

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

    def test_crawl_review_comment_prs_starts_from_repo_comments_and_excludes_bots_replies_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakeReviewCommentAPI()

            result = crawl_review_comment_prs(api, ["o/r"], raw, limit_prs=3)
            repo_dir = raw / "o__r"
            comments = [json.loads(line) for line in (repo_dir / "review_comments.jsonl").read_text().splitlines()]
            details = [json.loads(line) for line in (repo_dir / "commit_details.jsonl").read_text().splitlines()]

            self.assertEqual(result["new_review_comments"], 1)
            self.assertEqual(result["fetched_review_comments"]["o/r"], 3)
            self.assertEqual(result["rankable_review_comments"]["o/r"], 1)
            self.assertEqual(result["filtered_review_comments"]["o/r"], 2)
            self.assertEqual(result["selected_prs"][0]["pr_number"], 7)
            self.assertEqual(comments[0]["pr_number"], 7)
            self.assertEqual(len(comments[0]["data"]), 1)
            self.assertEqual(comments[0]["data"][0]["user"]["login"], "reviewer")
            self.assertEqual(details[0]["data"][0]["files"][0]["filename"], "tests/test_auth.py")
            self.assertFalse(any("/pulls/8" in path for path, _params in api.get_calls))
            self.assertFalse(any("/pulls/9" in path for path, _params in api.get_calls))

    def test_crawl_review_comment_prs_paginates_comments_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakePagedReviewCommentAPI()

            result = crawl_review_comment_prs(
                api,
                ["o/r"],
                raw,
                comments_per_repo=101,
                limit_prs=101,
                max_prs_per_repo=101,
                dry_run=True,
            )

            comment_calls = [params for path, params in api.get_calls if path.endswith("/pulls/comments")]
            self.assertEqual([params["page"] for params in comment_calls], [1, 2])
            self.assertEqual([params["per_page"] for params in comment_calls], [100, 1])
            self.assertEqual(result["fetched_review_comments"]["o/r"], 101)
            self.assertEqual(result["rankable_review_comments"]["o/r"], 101)
            self.assertEqual(len(result["selected_prs"]), 101)

    def test_crawl_review_comment_prs_dry_run_respects_per_repo_limit_without_backup_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakePagedReviewCommentAPI()

            result = crawl_review_comment_prs(
                api,
                ["o/r"],
                raw,
                comments_per_repo=5,
                limit_prs=3,
                max_prs_per_repo=1,
                dry_run=True,
            )

            self.assertEqual(len(result["selected_prs"]), 1)
            self.assertEqual(result["selected_prs"][0]["pr_number"], 1004)
            self.assertEqual(result["pull_detail_attempt_limit"], 3)

    def test_crawl_review_comment_prs_metadata_only_defers_file_and_commit_fetches(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakeReviewCommentAPI()

            result = crawl_review_comment_prs(api, ["o/r"], raw, limit_prs=3, metadata_only=True)
            repo_dir = raw / "o__r"
            pull_rows = [json.loads(line) for line in (repo_dir / "pull_requests.jsonl").read_text().splitlines()]
            comment_rows = [json.loads(line) for line in (repo_dir / "review_comments.jsonl").read_text().splitlines()]

            self.assertTrue(result["metadata_only"])
            self.assertEqual(result["new_review_comments"], 1)
            self.assertEqual(result["new_pull_files"], 0)
            self.assertEqual(result["new_pull_commits"], 0)
            self.assertEqual(result["new_commit_details"], 0)
            self.assertTrue(result["selected_prs"][0]["details_deferred"])
            self.assertEqual(pull_rows[0]["data"]["baseRefOid"], "base-sha")
            self.assertEqual(comment_rows[0]["data"][0]["id"], 1)
            self.assertEqual(api.paginate_calls, [])
            self.assertFalse((repo_dir / "pull_files.jsonl").exists())

    def test_crawl_review_comment_prs_replaces_skipped_top_ranked_prs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            api = FakeSkippedReviewCommentAPI()

            result = crawl_review_comment_prs(api, ["o/r"], raw, limit_prs=1, max_prs_per_repo=1, metadata_only=True)

            self.assertEqual(result["selected_prs"][0]["pr_number"], 8)
            self.assertEqual(result["skipped_prs"][0]["pr_number"], 9)
            self.assertEqual(result["skipped_prs"][0]["reason"], "not_merged")
            self.assertEqual(result["pull_detail_attempts"], 2)
            self.assertTrue(any("/pulls/9" in path for path, _params in api.get_calls))
            self.assertTrue(any("/pulls/8" in path for path, _params in api.get_calls))


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


class FailingSecondPRAPI(FakeCrawlerAPI):
    def paginate(self, path, params=None, accept=None):
        if "/pulls/2/files" in path:
            raise RuntimeError("rate limit")
        return super().paginate(path, params=params, accept=accept)


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def review_comment(index: int):
    return {
        "id": index,
        "body": f"Could we add a coverage test for behavior {index}?",
        "path": "src/auth.py",
        "created_at": "2026-01-01T00:10:00Z",
        "pull_request_url": f"https://api.github.com/repos/o/r/pulls/{1000 + index}",
        "user": {"login": "reviewer", "type": "User"},
    }


def response(body):
    return type("Response", (), {"body": body, "headers": {}})()


def response_with_headers(body, headers):
    return type("Response", (), {"body": body, "headers": headers})()

if __name__ == "__main__":
    unittest.main()
