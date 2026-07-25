import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.trace_repro import (
    FAILURE_TRACE_RE,
    build_trace_repro_candidate,
    classify_failure_type,
    failure_excerpt_for_run,
    mine_trace_repro_runs,
    process_record,
    run_trace_repro,
    run_shell,
    run_variants_for_candidate,
    safe_log_label,
    trace_sample_from_repro_run,
    trace_repro_env_noise_reason,
    trace_repro_source,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def pr(number=1):
    return {
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "title": "Fix auth refresh regression",
        "body": "Fixes a token refresh runtime error and adds a regression test.",
        "baseRefOid": "base-sha",
        "mergeCommit": {"oid": "fix-sha"},
        "createdAt": "2026-01-01T00:00:00Z",
        "mergedAt": "2026-01-02T00:00:00Z",
    }


def files(rows):
    return {"pr_number": 1, "data": rows}


def run(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)


class TraceReproSourceTests(unittest.TestCase):
    def test_failure_trace_regex_accepts_go_compile_errors(self):
        output = "# github.com/gin-gonic/gin [github.com/gin-gonic/gin.test]\n./context_test.go:524:27: c.GetError undefined"

        self.assertRegex(output, FAILURE_TRACE_RE)

    def test_failure_trace_regex_accepts_go_assertions_and_package_compile_errors(self):
        assertion = "--- FAIL: TestThing (0.00s)\n    thing_test.go:32: got false, want true"
        compile_error = "modules/caddyhttp/metrics_test.go:529:16: unknown field OTLP in struct literal of type Metrics"

        self.assertRegex(assertion, FAILURE_TRACE_RE)
        self.assertRegex(compile_error, FAILURE_TRACE_RE)

    def test_classifies_pytest_failed_output_as_test_failure(self):
        output = "FAILED testing/python/show_fixtures_per_test.py::test_case\nE       Failed: nomatch: '*fixtures used by*'"

        self.assertEqual(classify_failure_type(output), "test_failure")

    def test_classifies_rust_assertion_failure(self):
        output = (
            "test try_recv_after_receiver_close_with_permit ... FAILED\n\n"
            "thread 'try_recv_after_receiver_close_with_permit' panicked at tokio/tests/sync_mpsc.rs:1010:5:\n"
            "assertion `left == right` failed\n"
            "  left: Err(Empty)\n"
            " right: Err(Disconnected)\n"
        )

        self.assertEqual(classify_failure_type(output), "assertion_failure")

    def test_failure_excerpt_prefers_failure_section_from_combined_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            combined = Path(tmp) / "combined.log"
            combined.write_text(
                "$ cargo test --test sync_mpsc\n"
                + "\n".join(f"test passing_{index} ... ok" for index in range(250))
                + "\nfailures:\n\n---- try_recv_after_receiver_close_with_permit stdout ----\n"
                "thread 'try_recv_after_receiver_close_with_permit' panicked at tokio/tests/sync_mpsc.rs:1010:5:\n"
                "assertion `left == right` failed\n"
                "  left: Err(Empty)\n"
                " right: Err(Disconnected)\n",
                encoding="utf-8",
            )

            excerpt = failure_excerpt_for_run({"combined_log": str(combined), "failure_excerpt": "test passing_0 ... ok"})

            self.assertTrue(excerpt.startswith("$ cargo test --test sync_mpsc"))
            self.assertIn("assertion `left == right` failed", excerpt)
            self.assertNotIn("test passing_0 ... ok", excerpt)
            sample, reason = trace_sample_from_repro_run(
                {
                    "id": "rust",
                    "repo": "tokio-rs/tokio",
                    "base_commit": "base-sha",
                    "status": "failed_expected",
                    "failure_trace_found": True,
                    "failure_excerpt": "test passing_0 ... ok",
                    "combined_log": str(combined),
                    "implementation_files": ["tokio/src/sync/mpsc/chan.rs"],
                    "test_files": ["tokio/tests/sync_mpsc.rs"],
                    "command_results": [{"command": "cargo test --test sync_mpsc", "returncode": 1}],
                },
                {"fix_commit": "fix-sha", "run": {"strategy": "cargo_test_focused"}},
            )

            self.assertIsNone(reason)
            self.assertIn("assertion `left == right` failed", sample["query"]["failure_excerpt"])
            self.assertIn("assertion_failure", sample["metadata"]["evidence"]["signals"])

    def test_detects_trace_repro_environment_noise(self):
        self.assertEqual(
            trace_repro_env_noise_reason("ModuleNotFoundError: No module named '_pytest._version'"),
            "env_missing_pytest_generated_version",
        )
        self.assertEqual(
            trace_repro_env_noise_reason("ERROR: pyproject.toml: 'minversion' requires pytest-2.0, actual pytest-0.1.dev1"),
            "env_pytest_version_metadata",
        )
        self.assertEqual(
            trace_repro_env_noise_reason("INTERNALERROR> pytest.PytestConfigWarning: Unknown config option: timeout"),
            "env_missing_pytest_plugin",
        )
        self.assertEqual(
            trace_repro_env_noise_reason("INTERNALERROR> File \"/x/site-packages/_pytest/logging.py\", line 1, in pytest_collection"),
            "env_pytest_internal_collection_error",
        )
        self.assertEqual(trace_repro_env_noise_reason("/bin/sh: 1: cargo: not found"), "env_missing_cargo")
        self.assertEqual(trace_repro_env_noise_reason("/bin/sh: 1: go: not found"), "env_missing_go")
        self.assertEqual(trace_repro_env_noise_reason("/bin/sh: 1: pnpm: not found"), "env_missing_node")
        self.assertEqual(trace_repro_env_noise_reason("error[E0432]: unresolved import `tokio::test`\nno `test` in the root"), "env_rust_missing_test_macro_feature")
        self.assertEqual(trace_repro_env_noise_reason("the package `x` does not contain these features: full"), "env_rust_unknown_feature")
        self.assertEqual(trace_repro_env_noise_reason("error: the package 'tokio-stream' does not contain this feature: test-util"), "env_rust_unknown_feature")
        self.assertEqual(trace_repro_env_noise_reason("error: no test target named `signal` in `tokio` package"), "env_rust_no_test_target")
        self.assertEqual(trace_repro_env_noise_reason("error: specification `tokio-stream` is ambiguous"), "env_rust_ambiguous_package_spec")

    def test_failed_environment_run_keeps_noise_reason(self):
        sample, reason = trace_sample_from_repro_run(
            {
                "id": "env",
                "repo": "o/r",
                "base_commit": "base-sha",
                "status": "failed_environment",
                "environment_noise_reason": "env_missing_pytest",
                "failure_excerpt": "No module named pytest",
                "implementation_files": ["src/auth.py"],
            },
            {"fix_commit": "fix-sha"},
        )

        self.assertIsNone(sample)
        self.assertEqual(reason, "env_missing_pytest")

    def test_run_variants_accept_string_fallback_and_safe_log_label(self):
        variants = run_variants_for_candidate(
            {
                "run": {
                    "commands": ["pytest tests/test_auth.py"],
                    "fallback_commands": ["python tests/test_auth.py"],
                }
            }
        )

        self.assertEqual(variants[1]["commands"], ["python tests/test_auth.py"])
        self.assertEqual(safe_log_label("fallback / one"), "fallback_one")

    def test_builds_base_plus_test_patch_repro_source(self):
        candidate, reason = build_trace_repro_candidate(
            repo="o/r",
            pr_number=1,
            pr=pr(),
            files_record=files(
                [
                    {"filename": "src/auth/cache.py", "status": "modified", "patch": "@@ -1 +1 @@\n-false\n+true"},
                    {
                        "filename": "tests/auth/test_refresh.py",
                        "status": "added",
                        "patch": "@@ -0,0 +1,3 @@\n+def test_refresh():\n+    assert refresh() == 'ok'\n",
                    },
                ]
            ),
            details_record={
                "data": [
                    {
                        "files": [
                            {"filename": "src/auth/cache.py"},
                            {"filename": "tests/auth/test_refresh.py"},
                        ]
                    }
                ]
            },
        )

        self.assertIsNone(reason)
        self.assertEqual(candidate["source_type"], "local_test_reproduction")
        self.assertEqual(candidate["implementation_files"], ["src/auth/cache.py"])
        self.assertEqual(candidate["test_files"], ["tests/auth/test_refresh.py"])
        self.assertEqual(candidate["run"]["strategy"], "pytest")
        self.assertIn("fallback_commands", candidate["run"])
        self.assertIn("assertion_or_failure_test_patch", candidate["evidence"])
        self.assertEqual(candidate["repro_plan"]["apply_patches"], "test_files_only")

    def test_rust_repro_source_uses_focused_cargo_test_commands(self):
        candidate, reason = build_trace_repro_candidate(
            repo="tokio-rs/tokio",
            pr_number=8075,
            pr=pr(8075),
            files_record=files(
                [
                    {
                        "filename": "tokio/src/sync/mpsc/bounded.rs",
                        "status": "modified",
                        "patch": "@@ -1 +1 @@\n-false\n+true",
                    },
                    {
                        "filename": "tokio/tests/sync_mpsc.rs",
                        "status": "modified",
                        "patch": "@@ -1 +1 @@\n-assert!(false)\n+assert!(true)",
                    },
                ]
            ),
            details_record={
                "data": [
                    {
                        "files": [
                            {"filename": "tokio/src/sync/mpsc/bounded.rs"},
                            {"filename": "tokio/tests/sync_mpsc.rs"},
                        ]
                    }
                ]
            },
        )

        self.assertIsNone(reason)
        self.assertEqual(candidate["run"]["strategy"], "cargo_test_focused")
        self.assertEqual(candidate["run"]["commands"], ["cargo test -p tokio --test sync_mpsc"])
        self.assertIn("fallback_commands", candidate["run"])
        self.assertEqual(
            candidate["run"]["fallback_commands"][0],
            {"name": "cargo_test_common_features", "commands": ["cargo test --features full,test-util -p tokio --test sync_mpsc"]},
        )

    def test_rust_repro_source_uses_module_selector_for_src_tests(self):
        candidate, reason = build_trace_repro_candidate(
            repo="tokio-rs/tokio",
            pr_number=8062,
            pr=pr(8062),
            files_record=files(
                [
                    {
                        "filename": "tokio/src/sync/mpsc/list.rs",
                        "status": "modified",
                        "patch": "@@ -1 +1 @@\n-false\n+true",
                    },
                    {
                        "filename": "tokio/src/sync/tests/loom_mpsc.rs",
                        "status": "modified",
                        "patch": "@@ -1 +1 @@\n-assert!(false)\n+assert!(true)",
                    },
                ]
            ),
            details_record={
                "data": [
                    {
                        "files": [
                            {"filename": "tokio/src/sync/mpsc/list.rs"},
                            {"filename": "tokio/src/sync/tests/loom_mpsc.rs"},
                        ]
                    }
                ]
            },
        )

        self.assertIsNone(reason)
        self.assertEqual(candidate["run"]["strategy"], "cargo_test_focused")
        self.assertEqual(candidate["run"]["commands"], ["cargo test -p tokio sync::tests::loom_mpsc"])

    def test_rejects_broad_pr_before_creating_repro_source(self):
        candidate, reason = build_trace_repro_candidate(
            repo="o/r",
            pr_number=1,
            pr=pr(),
            files_record=files(
                [{"filename": f"src/module_{index}.py", "status": "modified", "patch": "@@"} for index in range(6)]
                + [
                    {
                        "filename": "tests/test_regression.py",
                        "status": "added",
                        "patch": "@@ -0,0 +1 @@\n+def test_regression(): assert True",
                    }
                ]
            ),
            max_source_files=5,
        )

        self.assertIsNone(candidate)
        self.assertEqual(reason, "too_many_source_files")

    def test_mines_repro_source_outputs_and_audit_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr(), "repo": "o/r", "type": "pull_request"}])
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    files(
                        [
                            {"filename": "src/auth/cache.py", "status": "modified", "patch": "@@ -1 +1 @@\n-false\n+true"},
                            {
                                "filename": "tests/auth/test_refresh.py",
                                "status": "added",
                                "patch": "@@ -0,0 +1,3 @@\n+def test_refresh():\n+    assert refresh() == 'ok'\n",
                            },
                        ]
                    )
                ],
            )

            result = trace_repro_source(root / "raw", root / "reports", audit_limit=10)
            candidates = [json.loads(line) for line in (root / "reports" / "repro_candidates.jsonl").read_text().splitlines()]
            audit_csv = (root / "reports" / "audit_samples.csv").read_text()

            self.assertEqual(result["candidates"], 1)
            self.assertEqual(result["audit_rows"], 1)
            self.assertEqual(candidates[0]["repo"], "o/r")
            self.assertIn("runnable_repro_source", audit_csv)
            self.assertTrue((root / "reports" / "summary.json").exists())

    def test_run_trace_repro_applies_test_patch_and_captures_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            remote.mkdir()
            run(["git", "init"], cwd=remote)
            run(["git", "config", "user.email", "test@example.com"], cwd=remote)
            run(["git", "config", "user.name", "Test"], cwd=remote)
            (remote / "src").mkdir()
            (remote / "src" / "auth.py").write_text("def refresh():\n    return 'bug'\n", encoding="utf-8")
            run(["git", "add", "."], cwd=remote)
            run(["git", "commit", "-m", "base"], cwd=remote)
            base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=remote, check=True, text=True, capture_output=True).stdout.strip()
            raw_repo = root / "raw" / "o__r"
            test_patch = "@@ -0,0 +1,2 @@\n+from src.auth import refresh\n+assert refresh() == 'ok'\n"
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 7,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified", "patch": "@@ -1 +1 @@\n-def refresh():\n+def refresh():"},
                            {"filename": "tests/test_auth.py", "status": "added", "patch": test_patch},
                        ],
                    }
                ],
            )
            candidate = {
                "id": "cand",
                "repo": "o/r",
                "repo_url": str(remote),
                "pr_number": 7,
                "base_commit": base_commit,
                "implementation_files": ["src/auth.py"],
                "test_files": ["tests/test_auth.py"],
                "run": {"commands": ["PYTHONPATH=. python3 tests/test_auth.py"]},
            }
            write_jsonl(root / "candidates.jsonl", [candidate])

            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                result = run_trace_repro(
                    candidate_path=Path("candidates.jsonl"),
                    raw_dir=Path("raw"),
                    repos_dir=Path("worktrees"),
                    out_dir=Path("runs"),
                    candidate_ids=["cand"],
                    timeout_seconds=30,
                )
            finally:
                os.chdir(old_cwd)
            run_record = json.loads((root / "runs" / "runs" / "cand" / "run.json").read_text())

            self.assertEqual(result["status_counts"], {"failed_expected": 1})
            self.assertEqual(run_record["status"], "failed_expected")
            self.assertTrue(run_record["failure_trace_found"])
            self.assertIn("AssertionError", (root / "runs" / "runs" / "cand" / "combined.log").read_text())
            self.assertIn("tests/test_auth.py", (root / "runs" / "runs" / "cand" / "test_patch.diff").read_text())

    def test_run_trace_repro_retries_fallback_after_environment_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            remote.mkdir()
            run(["git", "init"], cwd=remote)
            run(["git", "config", "user.email", "test@example.com"], cwd=remote)
            run(["git", "config", "user.name", "Test"], cwd=remote)
            (remote / "src").mkdir()
            (remote / "src" / "auth.py").write_text("def refresh():\n    return 'bug'\n", encoding="utf-8")
            run(["git", "add", "."], cwd=remote)
            run(["git", "commit", "-m", "base"], cwd=remote)
            base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=remote, check=True, text=True, capture_output=True).stdout.strip()
            raw_repo = root / "raw" / "o__r"
            test_patch = "@@ -0,0 +1,2 @@\n+from src.auth import refresh\n+assert refresh() == 'ok'\n"
            write_jsonl(
                raw_repo / "pull_files.jsonl",
                [
                    {
                        "pr_number": 7,
                        "data": [
                            {"filename": "src/auth.py", "status": "modified", "patch": "@@ -1 +1 @@\n-def refresh():\n+def refresh():"},
                            {"filename": "tests/test_auth.py", "status": "added", "patch": test_patch},
                        ],
                    }
                ],
            )
            candidate = {
                "id": "cand",
                "repo": "o/r",
                "repo_url": str(remote),
                "pr_number": 7,
                "base_commit": base_commit,
                "implementation_files": ["src/auth.py"],
                "test_files": ["tests/test_auth.py"],
                "run": {
                    "commands": ["python3 -c \"import sys; sys.stderr.write('/python: No module named pytest'); sys.exit(1)\""],
                    "fallback_commands": [{"name": "local_script", "commands": ["PYTHONPATH=. python3 tests/test_auth.py"]}],
                },
            }
            write_jsonl(root / "candidates.jsonl", [candidate])

            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                result = run_trace_repro(
                    candidate_path=Path("candidates.jsonl"),
                    raw_dir=Path("raw"),
                    repos_dir=Path("worktrees"),
                    out_dir=Path("runs"),
                    candidate_ids=["cand"],
                    timeout_seconds=30,
                )
            finally:
                os.chdir(old_cwd)
            run_record = json.loads((root / "runs" / "runs" / "cand" / "run.json").read_text())

            self.assertEqual(result["status_counts"], {"failed_expected": 1})
            self.assertEqual([attempt["status"] for attempt in run_record["attempts"]], ["failed_environment", "failed_expected"])
            self.assertEqual(run_record["commands"], ["PYTHONPATH=. python3 tests/test_auth.py"])
            self.assertIn("AssertionError", (root / "runs" / "runs" / "cand" / "combined.log").read_text())

    def test_timeout_process_records_are_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            process = run_shell("python3 -c 'import time; print(\"start\"); time.sleep(2)'", cwd=root, timeout_seconds=1)
            record = process_record(process)

            self.assertEqual(record["returncode"], 124)
            self.assertIsInstance(record["stdout"], str)
            self.assertIsInstance(record["stderr"], str)
            json.dumps(record)

    def test_run_shell_replaces_invalid_utf8_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            process = run_shell(
                "python3 -c 'import sys; sys.stdout.buffer.write(bytes([255]) + b\"ok\")'",
                cwd=root,
                timeout_seconds=5,
            )

            self.assertEqual(process.returncode, 0)
            self.assertIn("\ufffdok", process.stdout)

    def test_mine_trace_repro_runs_converts_failed_runs_and_drops_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [
                {
                    "id": "ok",
                    "repo": "o/r",
                    "pr_url": "https://github.com/o/r/pull/1",
                    "fix_commit": "fix-sha",
                    "run": {"strategy": "pytest"},
                },
                {
                    "id": "broad",
                    "repo": "o/r",
                    "fix_commit": "fix-sha",
                    "run": {"strategy": "pytest"},
                },
                {
                    "id": "ok2",
                    "repo": "o/r",
                    "fix_commit": "fix-sha",
                    "run": {"strategy": "pytest"},
                },
                {
                    "id": "env-noise",
                    "repo": "o/r",
                    "fix_commit": "fix-sha",
                    "run": {"strategy": "pytest"},
                },
            ]
            runs = [
                {
                    "id": "ok2",
                    "repo": "o/r",
                    "pr_number": 1,
                    "base_commit": "base-sha",
                    "status": "failed_expected",
                    "failure_trace_found": True,
                    "failure_excerpt": "Traceback\nAssertionError\nfix-sha",
                    "implementation_files": ["src/auth.py"],
                    "test_files": ["tests/test_auth.py"],
                    "combined_log": "runs/ok/combined.log",
                    "command_results": [{"command": "python -m pytest tests/test_auth.py", "returncode": 1}],
                },
                {
                    "id": "skip",
                    "repo": "o/r",
                    "status": "patch_failed",
                    "failure_trace_found": False,
                    "implementation_files": ["src/skip.py"],
                },
                {
                    "id": "broad",
                    "repo": "o/r",
                    "base_commit": "base-sha",
                    "status": "failed_expected",
                    "failure_trace_found": True,
                    "failure_excerpt": "AssertionError",
                    "implementation_files": ["a.py", "b.py", "c.py", "d.py"],
                    "test_files": ["tests/test_many.py"],
                },
                {
                    "id": "ok",
                    "repo": "o/r",
                    "pr_number": 1,
                    "base_commit": "base-sha",
                    "status": "failed_without_trace",
                    "failure_trace_found": False,
                    "failure_excerpt": "--- FAIL: TestRefresh (0.00s)\n    test_auth.py:1: got bug",
                    "implementation_files": ["src/auth.py"],
                    "test_files": ["tests/test_auth.py"],
                    "command_results": [{"command": "python -m pytest tests/test_auth.py", "returncode": 1}],
                },
                {
                    "id": "env-noise",
                    "repo": "o/r",
                    "base_commit": "base-sha",
                    "status": "failed_expected",
                    "failure_trace_found": True,
                    "failure_excerpt": "INTERNALERROR> File \"/x/site-packages/_pytest/logging.py\", line 1, in pytest_collection",
                    "implementation_files": ["src/auth.py"],
                    "test_files": ["tests/test_auth.py"],
                    "command_results": [{"command": "python -m pytest tests/test_auth.py", "returncode": 3}],
                },
            ]
            write_jsonl(root / "candidates.jsonl", candidates)
            write_jsonl(root / "runs.jsonl", runs)

            result = mine_trace_repro_runs(
                candidates_path=root / "candidates.jsonl",
                runs_path=root / "runs.jsonl",
                out_dir=root / "benchmark",
                report_dir=root / "report",
                max_root_files=3,
            )
            samples = [json.loads(line) for line in (root / "benchmark" / "trace2code.jsonl").read_text().splitlines()]
            audit_csv = (root / "report" / "audit_samples.csv").read_text()

            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["dropped"]["run_status_patch_failed"], 1)
            self.assertEqual(result["dropped"]["too_broad_root_files"], 1)
            self.assertEqual(result["dropped"]["env_pytest_internal_collection_error"], 1)
            self.assertEqual(samples[0]["task_type"], "trace2code")
            self.assertEqual(samples[0]["gold"]["root_cause_files"], ["src/auth.py"])
            self.assertEqual(samples[0]["gold"]["related_tests"], ["tests/test_auth.py"])
            self.assertIn("assertion_failure", samples[0]["metadata"]["evidence"]["signals"])
            self.assertTrue(any(sample["metadata"]["evidence"].get("failure_trace_reclassified") for sample in samples))
            query_text = json.dumps(samples[0]["query"])
            self.assertNotIn("fix-sha", query_text)
            self.assertNotIn("src/auth.py", query_text)
            self.assertIn("not_root_cause", audit_csv)


if __name__ == "__main__":
    unittest.main()
