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
                        "body": "Could you please add a regression test for refresh?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                        "line": 12,
                        "diff_hunk": "@@ -1 +1 @@\n-old\n+new\n context",
                        "commit_id": "head",
                    }
                ]
            }
        }
        commits_by_pr = {
            1: {
                "data": [
                    {"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}},
                ]
            }
        }
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["version"], 2)
        self.assertEqual(samples[0]["candidate_corpus"]["base_commit"], "base")
        self.assertNotIn("+new", samples[0]["query"]["diff_hunk_context"])
        self.assertEqual(samples[0]["gold"]["given_files"], ["src/auth.py"])
        self.assertEqual(samples[0]["gold"]["must_context_files"][0]["path"], "tests/test_auth.py")
        self.assertIn("modified_after_review_comment", samples[0]["gold"]["must_context_files"][0]["evidence"])
        self.assertIn("behavior_test_for_reviewed_change", samples[0]["gold"]["must_context_files"][0]["evidence"])
        self.assertEqual(validate_sample(samples[0]), [])

    def test_comment_sample_drops_suggestion_blocks(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Try this\n```suggestion\nreturn fixed\n```",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                        "line": 12,
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, {}, {}))

        self.assertEqual(samples, [])

    def test_comment_sample_requires_post_comment_context(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add a regression test for refresh?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                        "line": 12,
                    }
                ]
            }
        }
        commits_by_pr = {
            1: {
                "data": [
                    {"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:10:00Z"}}},
                ]
            }
        }
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:10:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(samples, [])

    def test_comment_sample_uses_only_post_comment_commit_detail_gold(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py", "tests/test_other.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add a regression test for refresh?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                        "line": 12,
                    }
                ]
            }
        }
        commits_by_pr = {1: {"data": [{"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}}]}}
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual([item["path"] for item in samples[0]["gold"]["must_context_files"]], ["tests/test_auth.py"])

    def test_comment_sample_drops_post_hoc_and_large_solution_blocks(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {"id": 10, "body": "Fixed.", "path": "src/auth.py", "created_at": "2026-01-01T00:30:00Z"},
                    {
                        "id": 11,
                        "body": "Does this work?\n```py\n" + "\n".join(f"line_{i}" for i in range(12)) + "\n```",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    },
                ]
            }
        }
        commits_by_pr = {1: {"data": [{"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}}]}}
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(samples, [])

    def test_comment_sample_drops_replies_and_small_solution_blocks(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add coverage?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                        "in_reply_to_id": 9,
                    },
                    {
                        "id": 11,
                        "body": "Could this be simpler?\n```py\nreturn refresh()\n```",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    },
                ]
            }
        }
        commits_by_pr = {1: {"data": [{"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}}]}}
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(samples, [])

    def test_comment_sample_drops_direct_context_path_hints(self):
        pr_by_number, files_by_pr = self._pr_and_files(["src/auth.py", "tests/test_auth.py"])
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add coverage in tests/test_auth.py?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    }
                ]
            }
        }
        commits_by_pr = {1: {"data": [{"sha": "head", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}}]}}
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": "tests/test_auth.py", "status": "modified"}],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(samples, [])

    def test_comment_sample_uses_first_valid_commit_within_72h_and_caps_gold(self):
        pr_by_number, files_by_pr = self._pr_and_files(
            [
                "src/auth.py",
                "tests/test_auth.py",
                "tests/test_refresh.py",
                "tests/test_late.py",
                ".github/workflows/ci.yml",
            ]
        )
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add regression tests for refresh behavior?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    }
                ]
            }
        }
        commits_by_pr = {
            1: {
                "data": [
                    {"sha": "workflow", "commit": {"committer": {"date": "2026-01-01T00:35:00Z"}}},
                    {"sha": "first-valid", "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}}},
                    {"sha": "late", "commit": {"committer": {"date": "2026-01-05T00:45:00Z"}}},
                ]
            }
        }
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "workflow",
                        "commit": {"committer": {"date": "2026-01-01T00:35:00Z"}},
                        "files": [{"filename": ".github/workflows/ci.yml", "status": "modified"}],
                    },
                    {
                        "sha": "first-valid",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [
                            {"filename": "tests/test_auth.py", "status": "modified"},
                            {"filename": "tests/test_refresh.py", "status": "modified"},
                            {"filename": "tests/test_late.py", "status": "modified"},
                        ],
                    },
                    {
                        "sha": "late",
                        "commit": {"committer": {"date": "2026-01-05T00:45:00Z"}},
                        "files": [{"filename": "tests/test_late.py", "status": "modified"}],
                    },
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, commits_by_pr, details_by_pr))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["metadata"]["evidence"]["response_commit"], "first-valid")
        self.assertEqual(
            [item["path"] for item in samples[0]["gold"]["must_context_files"]],
            ["tests/test_auth.py", "tests/test_refresh.py"],
        )

    def test_comment_sample_drops_oversized_response_commits_and_low_value_gold(self):
        pr_by_number, files_by_pr = self._pr_and_files(
            [
                "src/auth.py",
                "tests/test_auth.py",
                "docs/auth.md",
                "pyproject.toml",
                ".github/workflows/ci.yml",
            ]
        )
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add coverage for refresh behavior?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    }
                ]
            }
        }
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "too-large",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [{"filename": f"tests/test_{index}.py", "status": "modified"} for index in range(9)],
                    },
                    {
                        "sha": "low-value",
                        "commit": {"committer": {"date": "2026-01-01T00:50:00Z"}},
                        "files": [
                            {"filename": "docs/auth.md", "status": "modified"},
                            {"filename": ".github/workflows/ci.yml", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    },
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, {}, details_by_pr))

        self.assertEqual(samples, [])

    def test_comment_sample_drops_added_given_or_gold_files(self):
        pr_by_number, files_by_pr = self._pr_and_files(
            ["src/auth.py", "tests/test_auth.py", "tests/test_new_auth.py"],
            statuses={"src/auth.py": "modified", "tests/test_auth.py": "modified", "tests/test_new_auth.py": "added"},
        )
        comments_by_pr = {
            1: {
                "data": [
                    {
                        "id": 10,
                        "body": "Could you please add a regression test for refresh behavior?",
                        "path": "src/auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    },
                    {
                        "id": 11,
                        "body": "Could this use the existing auth behavior?",
                        "path": "tests/test_new_auth.py",
                        "created_at": "2026-01-01T00:30:00Z",
                    },
                ]
            }
        }
        details_by_pr = {
            1: {
                "data": [
                    {
                        "sha": "head",
                        "commit": {"committer": {"date": "2026-01-01T00:45:00Z"}},
                        "files": [
                            {"filename": "tests/test_new_auth.py", "status": "modified"},
                            {"filename": "tests/test_auth.py", "status": "modified"},
                        ],
                    }
                ]
            }
        }

        samples = list(_comment2context("o/r", pr_by_number, files_by_pr, comments_by_pr, 20, {}, details_by_pr))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["query"]["path"], "src/auth.py")
        self.assertEqual(
            [item["path"] for item in samples[0]["gold"]["must_context_files"]],
            ["tests/test_auth.py"],
        )

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

    def _pr_and_files(self, paths, statuses=None):
        statuses = statuses or {}
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
        files_by_pr = {1: {"data": [{"filename": path, "status": statuses.get(path)} for path in paths]}}
        return pr_by_number, files_by_pr


if __name__ == "__main__":
    unittest.main()
