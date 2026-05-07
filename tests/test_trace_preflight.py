import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.trace_preflight import mine_trace2code, trace_debug_drops, trace_debug_summary, trace_preflight, trace_source_scan


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

    def test_drops_coderabbit_and_pure_path_lookup_noise(self):
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
            write_jsonl(raw_repo / "pull_files.jsonl", [{"pr_number": 1, "data": [{"filename": "src/auth.py", "status": "modified"}]}])
            write_jsonl(
                raw_repo / "review_comments.jsonl",
                [
                    {
                        "pr_number": 1,
                        "data": [
                            {
                                "id": 10,
                                "body": "<!-- This is an auto-generated reply by CodeRabbit -->\nTraceback\nFile \"src/auth.py\", line 7\nRuntimeError",
                            },
                            {"id": 11, "body": "src/auth.py:7"},
                        ],
                    }
                ],
            )

            result = trace_preflight(root / "raw", root / "trace")

            self.assertEqual(result["candidates"], 0)
            self.assertEqual(result["dropped"]["weak_review_comment_trace"], 2)

    def test_mine_trace2code_writes_benchmark_and_audit_from_job_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 2,
                "url": "https://github.com/o/r/pull/2",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 2,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 2,
                        "job_id": 99,
                        "check_name": "tests",
                        "html_url": "https://github.com/o/r/actions/runs/1/job/99",
                        "excerpt": (
                            "FAILED tests/test_auth.py::test_refresh\n"
                            "Traceback (most recent call last):\n"
                            "  File \"/home/runner/work/r/r/src/auth.py\", line 7, in refresh\n"
                            "RuntimeError: token expired"
                        ),
                    }
                ],
            )

            result = mine_trace2code(root / "raw", root / "benchmark", root / "report", audit_limit=10)
            samples = [json.loads(line) for line in (root / "benchmark" / "samples.jsonl").read_text().splitlines()]
            audit_csv = (root / "report" / "audit_samples.csv").read_text()

            self.assertEqual(result["samples"], 1)
            self.assertEqual(samples[0]["task_type"], "trace2code")
            self.assertEqual(samples[0]["gold"]["root_cause_files"], ["src/auth.py"])
            self.assertEqual(samples[0]["gold"]["related_tests"], ["tests/test_auth.py"])
            self.assertIn("not_root_cause", audit_csv)

    def test_job_log_test_frame_can_use_small_fix_diff_source_as_root_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 3,
                "url": "https://github.com/o/r/pull/3",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 3,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 3,
                        "job_id": 100,
                        "check_name": "tests",
                        "excerpt": (
                            "FAILED tests/test_auth.py::test_refresh\n"
                            "Traceback (most recent call last):\n"
                            "  File \"/home/runner/work/r/r/tests/test_auth.py\", line 12, in test_refresh\n"
                            "AssertionError: expected refresh to fail"
                        ),
                    }
                ],
            )

            result = mine_trace2code(root / "raw", root / "benchmark", root / "report")
            samples = [json.loads(line) for line in (root / "benchmark" / "samples.jsonl").read_text().splitlines()]

            self.assertEqual(result["samples"], 1)
            self.assertEqual(samples[0]["gold"]["root_cause_files"], ["src/auth.py"])
            self.assertIn("small_fix_diff_root_source", samples[0]["metadata"]["evidence"]["signals"])

    def test_trace_debug_drops_samples_weak_ci_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 4,
                "url": "https://github.com/o/r/pull/4",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 4,
                        "data": [
                            {"filename": f"src/module_{index}.py", "status": "modified"}
                            for index in range(6)
                        ]
                        + [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 4,
                        "job_id": 101,
                        "check_name": "tests",
                        "excerpt": (
                            "FAILED tests/test_auth.py::test_refresh\n"
                            "Traceback (most recent call last):\n"
                            "  File \"/home/runner/work/r/r/tests/test_auth.py\", line 12, in test_refresh\n"
                            "AssertionError: expected refresh to fail"
                        ),
                    }
                ],
            )

            result = trace_debug_drops(root / "raw", root / "debug", audit_limit=10)
            rows = [json.loads(line) for line in (root / "debug" / "weak_signals.jsonl").read_text().splitlines()]
            csv_text = (root / "debug" / "audit_samples.csv").read_text()

            self.assertEqual(result["weak_signals"], 1)
            self.assertEqual(rows[0]["drop_reason"], "no_root_candidate")
            self.assertEqual(rows[0]["source"], "job_log")
            self.assertIn("valid_root_cause", csv_text)

    def test_trace_debug_summary_writes_only_kept_recoverable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.csv"
            audit.write_text(
                "sample_id,repo,pr_number,source,check_name,drop_reason,failure_excerpt,trace_paths,implementation_files,test_files,candidate_root_files,verdict,reason,keep,notes\n"
                "a,o/r,1,job_log,tests,no_root_candidate,boom,tests/test_auth.py,src/auth.py,,src/auth.py,valid_root_cause,source trace maps to impl,true,\n"
                "b,o/r,2,job_log,tests,no_root_candidate,setup,,,,,infra_noise,cache setup,false,\n",
                encoding="utf-8",
            )

            result = trace_debug_summary(audit, root / "summary.json", root / "recoverable.jsonl")
            recoverable = [json.loads(line) for line in (root / "recoverable.jsonl").read_text().splitlines()]

            self.assertEqual(result["pending"], 0)
            self.assertEqual(result["invalid_verdicts"], {})
            self.assertEqual(result["recoverable"], 1)
            self.assertEqual(recoverable[0]["sample_id"], "a")
            self.assertEqual(recoverable[0]["implementation_files"], ["src/auth.py"])

    def test_setup_cache_failure_words_are_not_debug_trace_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 7,
                "url": "https://github.com/o/r/pull/7",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(raw_repo / "pull_files.jsonl", [{"pr_number": 7, "data": [{"filename": "src/auth.py", "status": "modified"}]}])
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 7,
                        "job_id": 107,
                        "check_name": "tests",
                        "excerpt": (
                            "Run actions/cache@v5\n"
                            "fail-on-cache-miss: false\n"
                            "Downloaded ruff\n"
                            "Prepared 14 packages in 403ms"
                        ),
                    }
                ],
            )

            result = trace_debug_drops(root / "raw", root / "debug", audit_limit=10)
            self.assertEqual(result["weak_signals"], 0)

    def test_trace_source_scan_ranks_real_repo_trace_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            pr = {
                "number": 8,
                "url": "https://github.com/o/r/pull/8",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 8,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 8,
                        "job_id": 108,
                        "check_name": "tests",
                        "excerpt": (
                            "FAILED tests/test_auth.py::test_refresh\n"
                            "Traceback (most recent call last):\n"
                            "  File \"/home/runner/work/r/r/src/auth.py\", line 7, in refresh\n"
                            "RuntimeError: token expired"
                        ),
                    }
                ],
            )

            result = trace_source_scan(root / "raw", root / "source", audit_limit=10)
            rows = [json.loads(line) for line in (root / "source" / "source_candidates.jsonl").read_text().splitlines()]
            csv_text = (root / "source" / "audit_samples.csv").read_text()

            self.assertEqual(result["candidates"], 1)
            self.assertEqual(result["quality_gate"]["ready_for_source_audit"], False)
            self.assertIn("source_trace_frame", rows[0]["tags"])
            self.assertIn("usable_trace_source", csv_text)

    def test_trace_source_scan_rejects_downstream_and_setup_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "pydantic__pydantic"
            pr = {
                "number": 9,
                "url": "https://github.com/pydantic/pydantic/pull/9",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "pydantic/pydantic", "type": "pull_request"}])
            write_jsonl(raw_repo / "pull_files.jsonl", [{"pr_number": 9, "data": [{"filename": "pydantic/main.py", "status": "modified"}]}])
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 9,
                        "job_id": 109,
                        "check_name": "Test Dify (main branch) on Python 3.12",
                        "excerpt": "Traceback\n  File \"api/tests/unit/test_config.py\", line 1\nAssertionError",
                    },
                    {
                        "type": "job_log",
                        "pr_number": 9,
                        "job_id": 110,
                        "check_name": "tests",
                        "excerpt": "Run actions/cache@v5\nDownloaded ruff\nPrepared 14 packages in 403ms",
                    },
                ],
            )

            result = trace_source_scan(root / "raw", root / "source", audit_limit=10)
            self.assertEqual(result["candidates"], 0)
            self.assertEqual(result["rejected"]["downstream_check"], 1)
            self.assertEqual(result["rejected"]["signal_noise"], 1)

    def test_timestamp_ansi_vitest_test_trace_maps_to_related_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "vitejs__vite"
            pr = {
                "number": 5,
                "url": "https://github.com/vitejs/vite/pull/5",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "vitejs/vite", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 5,
                        "data": [
                            {"filename": "packages/vite/src/node/server/pluginContainer.ts", "status": "modified"},
                            {"filename": "packages/vite/src/node/__tests__/plugins/hooks.spec.ts", "status": "modified"},
                        ],
                    }
                ],
            )
            write_jsonl(
                raw_repo / "job_logs.jsonl",
                [
                    {
                        "type": "job_log",
                        "pr_number": 5,
                        "job_id": 102,
                        "check_name": "vitest",
                        "excerpt": (
                            "2026-04-21T02:44:25.8098020Z \x1b[31mFAIL\x1b[0m packages/vite/src/node/__tests__/plugins/hooks.spec.ts > plugin hooks\n"
                            "2026-04-21T02:44:25.8098020Z Error: expected hook order\n"
                            "2026-04-21T02:44:25.8098020Z   at packages/vite/src/node/__tests__/plugins/hooks.spec.ts:42:7"
                        ),
                    }
                ],
            )

            result = mine_trace2code(root / "raw", root / "benchmark", root / "report", include_review_comments=False)
            samples = [json.loads(line) for line in (root / "benchmark" / "samples.jsonl").read_text().splitlines()]

            self.assertEqual(result["samples"], 1)
            self.assertEqual(samples[0]["gold"]["root_cause_files"], ["packages/vite/src/node/server/pluginContainer.ts"])
            self.assertIn("test_trace_to_impl_mapping", samples[0]["metadata"]["evidence"]["signals"])

    def test_java_dotted_source_trace_maps_to_repo_source_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "spring-projects__spring-boot"
            pr = {
                "number": 6,
                "url": "https://github.com/spring-projects/spring-boot/pull/6",
                "baseRefOid": "base",
                "mergeCommit": {"oid": "fix-sha"},
            }
            source = "spring-boot-project/spring-boot/src/main/java/org/springframework/boot/foo/Bar.java"
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr, "repo": "spring-projects/spring-boot", "type": "pull_request"}])
            write_jsonl(raw_repo / "pull_files.jsonl", [{"pr_number": 6, "data": [{"filename": source, "status": "modified"}]}])
            write_jsonl(
                raw_repo / "check_runs.jsonl",
                [
                    {
                        "pr_number": 6,
                        "data": [
                            {
                                "id": 103,
                                "name": "test",
                                "conclusion": "failure",
                                "html_url": "https://github.com/spring/actions/runs/1/job/103",
                                "output": {
                                    "summary": (
                                        "java.lang.AssertionError: boom\n"
                                        "at org.springframework.boot.foo.Bar.java:44\n"
                                    )
                                },
                            }
                        ],
                    }
                ],
            )

            result = mine_trace2code(root / "raw", root / "benchmark", root / "report", include_review_comments=False)
            samples = [json.loads(line) for line in (root / "benchmark" / "samples.jsonl").read_text().splitlines()]

            self.assertEqual(result["samples"], 1)
            self.assertEqual(samples[0]["gold"]["root_cause_files"], [source])
            self.assertIn("source_trace_frame", samples[0]["metadata"]["evidence"]["signals"])


if __name__ == "__main__":
    unittest.main()
