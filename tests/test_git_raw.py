import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.git_raw import backfill_git_raw


class GitRawTests(unittest.TestCase):
    def test_backfill_git_raw_writes_files_commits_and_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            git(source, "init")
            git(source, "config", "user.email", "test@example.com")
            git(source, "config", "user.name", "Test User")
            (source / "src").mkdir()
            (source / "tests").mkdir()
            (source / "src" / "lib.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            (source / "tests" / "test_lib.py").write_text("from src.lib import answer\n\n\ndef test_answer():\n    assert answer() == 1\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "base")
            base = git_stdout(source, "rev-parse", "HEAD")

            (source / "src" / "lib.py").write_text("def answer():\n    return 2\n", encoding="utf-8")
            (source / "tests" / "test_lib.py").write_text("from src.lib import answer\n\n\ndef test_answer():\n    assert answer() == 2\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "fix answer")
            head = git_stdout(source, "rev-parse", "HEAD")

            raw = root / "raw"
            repo_raw = raw / "o__r"
            repo_raw.mkdir(parents=True)
            write_rows(
                repo_raw / "pull_requests.jsonl",
                [
                    {
                        "type": "pull_request",
                        "repo": "o/r",
                        "data": {
                            "number": 7,
                            "title": "Fix answer",
                            "body": "Regression fix with test coverage.",
                            "url": "https://example.test/o/r/pull/7",
                            "createdAt": "2026-01-01T00:00:00Z",
                            "mergedAt": "2026-01-01T01:00:00Z",
                            "baseRefOid": base,
                            "headRefOid": head,
                            "mergeCommit": {"oid": head},
                        },
                    }
                ],
            )

            result = backfill_git_raw(raw, "o/r", root / "clones", repo_url_template=str(source), limit_prs=1)
            files = [json.loads(line) for line in (repo_raw / "pull_files.jsonl").read_text().splitlines()][0]["data"]
            details = [json.loads(line) for line in (repo_raw / "commit_details.jsonl").read_text().splitlines()][0]["data"]

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["processed"], 1)
            self.assertEqual({file["filename"] for file in files}, {"src/lib.py", "tests/test_lib.py"})
            self.assertIn("@@", next(file for file in files if file["filename"] == "tests/test_lib.py")["patch"])
            self.assertEqual(details[0]["sha"], head)
            self.assertIn("src/lib.py", {file["filename"] for file in details[0]["files"]})

    def test_backfill_git_raw_can_infer_missing_base_from_head_and_base_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            git(source, "init")
            git(source, "config", "user.email", "test@example.com")
            git(source, "config", "user.name", "Test User")
            git(source, "branch", "-M", "main")
            (source / "src").mkdir()
            (source / "src" / "lib.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "base")
            base = git_stdout(source, "rev-parse", "HEAD")
            git(source, "checkout", "-b", "feature")
            (source / "src" / "lib.py").write_text("def answer():\n    return 2\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "feature")
            head = git_stdout(source, "rev-parse", "HEAD")

            raw = root / "raw"
            repo_raw = raw / "o__r"
            repo_raw.mkdir(parents=True)
            write_rows(
                repo_raw / "pull_requests.jsonl",
                [
                    {
                        "type": "pull_request",
                        "repo": "o/r",
                        "data": {
                            "number": 8,
                            "title": "Infer base",
                            "body": "",
                            "url": "https://example.test/o/r/pull/8",
                            "createdAt": "2026-01-01T00:00:00Z",
                            "mergedAt": "2026-01-01T01:00:00Z",
                            "baseRefName": "main",
                            "headRefOid": head,
                            "mergeCommit": {"oid": head},
                        },
                    }
                ],
            )

            result = backfill_git_raw(raw, "o/r", root / "clones", repo_url_template=str(source), limit_prs=1, infer_missing_base=True)
            pull = [json.loads(line) for line in (repo_raw / "pull_requests.jsonl").read_text().splitlines()][0]["data"]
            files = [json.loads(line) for line in (repo_raw / "pull_files.jsonl").read_text().splitlines()][0]["data"]

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["processed"], 1)
            self.assertEqual(pull["baseRefOid"], base)
            self.assertEqual(files[0]["filename"], "src/lib.py")

    def test_backfill_git_raw_skips_large_inferred_base_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            git(source, "init")
            git(source, "config", "user.email", "test@example.com")
            git(source, "config", "user.name", "Test User")
            git(source, "branch", "-M", "main")
            (source / "src").mkdir()
            (source / "src" / "lib.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "base")
            git(source, "checkout", "-b", "feature")
            for value in range(2, 5):
                (source / "src" / "lib.py").write_text(f"def answer():\n    return {value}\n", encoding="utf-8")
                git(source, "add", ".")
                git(source, "commit", "-m", f"feature {value}")
            head = git_stdout(source, "rev-parse", "HEAD")

            raw = root / "raw"
            repo_raw = raw / "o__r"
            repo_raw.mkdir(parents=True)
            write_rows(
                repo_raw / "pull_requests.jsonl",
                [
                    {
                        "type": "pull_request",
                        "repo": "o/r",
                        "data": {
                            "number": 9,
                            "title": "Large inferred range",
                            "body": "",
                            "url": "https://example.test/o/r/pull/9",
                            "baseRefName": "main",
                            "headRefOid": head,
                            "mergeCommit": {"oid": head},
                        },
                    }
                ],
            )

            result = backfill_git_raw(
                raw,
                "o/r",
                root / "clones",
                repo_url_template=str(source),
                limit_prs=1,
                infer_missing_base=True,
                max_inferred_commits=2,
            )

            self.assertEqual(result["processed"], 0)
            self.assertEqual(result["skipped"]["inferred_base_too_many_commits"], 1)
            self.assertFalse((repo_raw / "commit_details.jsonl").exists())


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=True)


def git_stdout(path: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=True).stdout.strip()


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
