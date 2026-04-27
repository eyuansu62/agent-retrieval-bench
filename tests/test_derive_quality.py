import unittest

from agent_retrieval_bench.derive import _comment2context, _failure_samples, _review_trace2code
from agent_retrieval_bench.quality import validate_sample


class DeriveQualityTests(unittest.TestCase):
    def test_comment_sample_uses_base_commit_and_sanitized_hunk(self):
        pr_by_number = {
            1: {
                "data": {
                    "number": 1,
                    "title": "Fix auth",
                    "body": "Body",
                    "url": "https://github.com/o/r/pull/1",
                    "baseRefOid": "base",
                    "mergeCommit": {"oid": "fix"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "mergedAt": "2026-01-01T01:00:00Z",
                }
            }
        }
        files_by_pr = {
            1: {
                "data": [
                    {"filename": "src/auth.py"},
                    {"filename": "tests/test_auth.py"},
                ]
            }
        }
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "This misses refresh.",
                        "path": "src/auth.py",
                        "line": 12,
                        "diff_hunk": "@@ -1 +1 @@\n-old\n+new\n context",
                        "commit_id": "head",
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["candidate_corpus"]["base_commit"], "base")
        self.assertNotIn("+new", samples[0]["query"]["diff_hunk_context"])
        self.assertEqual(validate_sample(samples[0]), [])

    def test_comment_sample_drops_suggestion_blocks(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Try this\n```suggestion\nreturn fixed\n```",
                        "path": "src/auth.py",
                        "line": 12,
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20))

        self.assertEqual(samples, [])

    def test_review_trace_requires_real_failure_signal(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/app.py", "tests/test_app.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Learnt from: bot\nFile src/app.py line 3",
                        "path": "tests/test_app.py",
                        "line": 12,
                    },
                    {
                        "id": 11,
                        "body": "Traceback (most recent call last):\n  File src/app.py:3\nRuntimeError: boom",
                        "path": "tests/test_app.py",
                        "line": 13,
                    },
                ]
            }
        }

        samples = list(_review_trace2code("o/r", pr_by_number, files_by_pr, comments_by_pr, 20))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["gold"]["root_cause_files"], ["src/app.py"])

    def test_failure_samples_drop_ci_noise_and_keep_real_failures(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/app.py", "tests/test_app.py"])
        checks_by_pr = {
            1: [
                {
                    "sha": "head",
                    "data": [
                        {
                            "id": 1,
                            "name": "DCO",
                            "conclusion": "failure",
                            "output": {"summary": "There is one commit incorrectly signed off."},
                        },
                        {
                            "id": 2,
                            "name": "Build&Test: node-24",
                            "conclusion": "failure",
                            "output": {"summary": ""},
                        },
                        {
                            "id": 3,
                            "name": "tests",
                            "conclusion": "failure",
                            "output": {
                                "summary": "FAILED tests/test_app.py::test_auth - AssertionError: boom\nTraceback\n  File src/app.py:3"
                            },
                        },
                    ],
                }
            ]
        }

        samples = list(_failure_samples("o/r", pr_by_number, files_by_pr, checks_by_pr, 20))

        self.assertEqual(len(samples), 1)
        self.assertIn(samples[0]["task_type"], {"testlog2code", "trace2code"})

    def _pr_and_files(self, paths):
        pr_by_number = {
            1: {
                "data": {
                    "number": 1,
                    "title": "Fix auth",
                    "body": "Body",
                    "url": "https://github.com/o/r/pull/1",
                    "baseRefOid": "base",
                    "mergeCommit": {"oid": "fix"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "mergedAt": "2026-01-01T01:00:00Z",
                }
            }
        }
        files_by_pr = {1: {"data": [{"filename": path} for path in paths]}}
        return pr_by_number, files_by_pr


if __name__ == "__main__":
    unittest.main()
