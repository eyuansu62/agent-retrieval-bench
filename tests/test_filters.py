import unittest

from agent_retrieval_bench.filters import (
    extract_repo_trace_paths,
    is_generated_or_lockfile,
    is_test_file,
    sanitize_diff_hunk,
    sanitize_review_body,
    split_changed_files,
)


class FilterTests(unittest.TestCase):
    def test_split_changed_files_classifies_sources_tests_and_ignored(self):
        implementation, tests, ignored = split_changed_files(
            ["src/app.py", "tests/test_app.py", "package-lock.json", "dist/bundle.js"]
        )

        self.assertEqual(implementation, ["src/app.py"])
        self.assertEqual(tests, ["tests/test_app.py"])
        self.assertEqual(ignored, ["package-lock.json", "dist/bundle.js"])

    def test_generated_and_test_file_detection(self):
        self.assertTrue(is_generated_or_lockfile("pnpm-lock.yaml"))
        self.assertTrue(is_generated_or_lockfile("vendor/generated.pb.go"))
        self.assertTrue(is_test_file("src/foo_test.go"))
        self.assertTrue(is_test_file("tests/FooTest.java"))

    def test_sanitize_diff_hunk_removes_patch_lines(self):
        hunk = "@@ -1,3 +1,3 @@\n context\n-old\n+new\n unchanged"

        self.assertEqual(sanitize_diff_hunk(hunk), "@@ -1,3 +1,3 @@\n context\n unchanged")

    def test_sanitize_review_body_removes_diff_blocks(self):
        body = "please add this\n```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n+secret fix\n```\nthanks"

        self.assertEqual(sanitize_review_body(body), "please add this\nthanks")

    def test_extract_repo_trace_paths_filters_third_party(self):
        text = "File \"/repo/pkg/app.py\", line 10\nFile \"/venv/site-packages/lib.py\", line 2\nsrc/main.ts:22"

        self.assertEqual(extract_repo_trace_paths(text), ["/repo/pkg/app.py", "src/main.ts"])


if __name__ == "__main__":
    unittest.main()
