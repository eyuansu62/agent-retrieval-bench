import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.logs import crawl_job_logs


class FakeBytesResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self.body = body
        self.headers = headers or {}
        self.status = 200


class FakeGitHubAPI:
    def __init__(self, failures: set[int] | None = None):
        self.calls: list[str] = []
        self.failures = failures or set()

    def get_bytes(self, path, params=None, accept=None, max_bytes=None):
        self.calls.append(path)
        job_id = path.rsplit("/", 2)[-2]
        if int(job_id) in self.failures:
            raise OSError("missing log blob")
        return FakeBytesResponse(
            (
                f"FAILED tests/test_auth.py::test_refresh\n"
                f"Traceback (most recent call last):\n"
                f"  File \"src/auth.py\", line {job_id}, in refresh\n"
                f"RuntimeError: boom\n"
            ).encode(),
            {
                "x-ratelimit-limit": "5000",
                "x-ratelimit-remaining": "4999",
                "x-ratelimit-reset": "123",
                "x-ratelimit-used": "1",
                "x-ratelimit-resource": "core",
            },
        )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class CrawlJobLogsTests(unittest.TestCase):
    def test_existing_logs_do_not_consume_new_job_budget_and_metadata_is_deduped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "raw" / "o__r"
            (repo_dir / "job_logs").mkdir(parents=True)
            (repo_dir / "job_logs" / "3.txt").write_text("FAILED existing\n  File \"src/auth.py\", line 3\n", encoding="utf-8")
            write_jsonl(
                repo_dir / "job_logs.jsonl",
                [
                    {"type": "job_log", "repo": "o/r", "pr_number": 1, "job_id": 3, "log_path": "job_logs/3.txt"},
                    {"type": "job_log", "repo": "o/r", "pr_number": 1, "job_id": 3, "log_path": "job_logs/3.txt"},
                ],
            )
            write_jsonl(
                repo_dir / "check_runs.jsonl",
                [
                    {
                        "repo": "o/r",
                        "pr_number": 1,
                        "sha": "sha",
                        "ref_type": "merge",
                        "data": [
                            check_run(1, "unit tests", "failure"),
                            check_run(2, "integration tests", "failure"),
                            check_run(3, "more tests", "failure"),
                        ],
                    }
                ],
            )

            api = FakeGitHubAPI()
            summary = crawl_job_logs(api, root / "raw", "o/r", max_new_jobs=2)
            rows = [json.loads(line) for line in (repo_dir / "job_logs.jsonl").read_text().splitlines()]

            self.assertEqual(summary["candidate_jobs"], 3)
            self.assertEqual(summary["existing_skipped"], 1)
            self.assertEqual(summary["new_downloaded"], 2)
            self.assertEqual(summary["downloaded_or_existing"], 3)
            self.assertEqual(summary["errors"], 0)
            self.assertEqual(summary["rate_limit"]["remaining"], "4999")
            self.assertEqual(len(api.calls), 2)
            self.assertEqual(sorted(row["job_id"] for row in rows), [1, 2, 3])

    def test_candidate_jobs_skip_ignored_checks_and_prioritize_failure_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "raw" / "o__r"
            write_jsonl(
                repo_dir / "check_runs.jsonl",
                [
                    {
                        "repo": "o/r",
                        "pr_number": 1,
                        "sha": "sha",
                        "ref_type": "merge",
                        "data": [
                            check_run(10, "Dependabot", "failure"),
                            check_run(11, "build", "action_required"),
                            check_run(12, "unit tests", "failure"),
                        ],
                    }
                ],
            )

            api = FakeGitHubAPI()
            summary = crawl_job_logs(api, root / "raw", "o/r", max_new_jobs=2)
            rows = [json.loads(line) for line in (repo_dir / "job_logs.jsonl").read_text().splitlines()]

            self.assertEqual(summary["candidate_jobs"], 2)
            self.assertEqual([row["job_id"] for row in rows], [12, 11])
            self.assertNotIn("/10/", "\n".join(api.calls))

    def test_download_errors_are_recorded_without_aborting_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "raw" / "o__r"
            write_jsonl(
                repo_dir / "check_runs.jsonl",
                [
                    {
                        "repo": "o/r",
                        "pr_number": 1,
                        "sha": "sha",
                        "ref_type": "merge",
                        "data": [
                            check_run(21, "unit tests", "failure"),
                            check_run(22, "integration tests", "failure"),
                        ],
                    }
                ],
            )

            api = FakeGitHubAPI(failures={22})
            summary = crawl_job_logs(api, root / "raw", "o/r", max_new_jobs=2)
            rows = [json.loads(line) for line in (repo_dir / "job_logs.jsonl").read_text().splitlines()]

            self.assertEqual(summary["new_downloaded"], 2)
            self.assertEqual(summary["errors"], 1)
            self.assertEqual([row["type"] for row in rows], ["job_log_error", "job_log"])


def check_run(job_id: int, name: str, conclusion: str):
    return {
        "id": job_id,
        "name": name,
        "conclusion": conclusion,
        "html_url": f"https://github.com/o/r/actions/runs/1/job/{job_id}",
        "details_url": f"https://github.com/o/r/actions/runs/1/job/{job_id}",
        "app": {"slug": "github-actions"},
    }


if __name__ == "__main__":
    unittest.main()
