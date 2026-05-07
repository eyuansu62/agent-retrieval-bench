import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.trace_repro import (
    FAILURE_TRACE_RE,
    build_trace_repro_candidate,
    mine_trace_repro_runs,
    process_record,
    run_trace_repro,
    run_shell,
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
        self.assertIn("assertion_or_failure_test_patch", candidate["evidence"])
        self.assertEqual(candidate["repro_plan"]["apply_patches"], "test_files_only")

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

    def test_timeout_process_records_are_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            process = run_shell("python3 -c 'import time; print(\"start\"); time.sleep(2)'", cwd=root, timeout_seconds=1)
            record = process_record(process)

            self.assertEqual(record["returncode"], 124)
            self.assertIsInstance(record["stdout"], str)
            self.assertIsInstance(record["stderr"], str)
            json.dumps(record)

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
