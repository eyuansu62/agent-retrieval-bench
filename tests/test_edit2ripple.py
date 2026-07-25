import json
import tempfile
import unittest
import csv
import shutil
import subprocess
from pathlib import Path

from agent_retrieval_bench.baseline import given_files, query_has_leakage, target_gold_files
from agent_retrieval_bench.edit2ripple import (
    build_pr_edit2ripple_samples,
    diagnose_edit2ripple_leakage,
    is_format_only_patch,
    mine_edit2ripple,
    mine_edit2ripple_from_sample_commits,
    mine_edit2ripple_from_samples,
    parse_github_pr_url,
    report_edit2ripple_pilot,
)
from agent_retrieval_bench.quality import validate_sample


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def pr(body=None):
    return {
        "number": 7,
        "url": "https://github.com/o/r/pull/7",
        "title": "Support nested field option",
        "body": body if body is not None else "Add an option so nested fields can be included when walking payload values.",
        "baseRefOid": "base",
        "mergeCommit": {"oid": "fixsha"},
        "createdAt": "2026-01-01T00:00:00Z",
        "mergedAt": "2026-01-02T00:00:00Z",
    }


def file_record(filename, patch, additions=2, deletions=1, status="modified"):
    return {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions,
        "patch": patch,
    }


def files_record(body_mentions_gold=False):
    gold_patch = """@@ def collect_values(options):
-    return collect(payload)
+    return collect(payload, include_nested=options.include_nested)
"""
    return {
        "pr_number": 7,
        "data": [
            file_record(
                "src/runtime/options.py",
                """@@ class RuntimeOptions:
-    include_empty: bool = False
+    include_empty: bool = False
+    include_nested: bool = False
""",
                additions=1,
                deletions=0,
            ),
            file_record("src/runtime/traversal.py", gold_patch, additions=1, deletions=1),
        ],
    }


class FakePRAPI:
    def __init__(self):
        self.get_paths = []
        self.paginate_paths = []

    def get(self, path, params=None, accept=None):
        self.get_paths.append(path)
        return type(
            "Response",
            (),
            {
                "body": {
                    "number": 7,
                    "title": "Support nested field option",
                    "body": "Add an option so nested fields can be included when walking payload values.",
                    "html_url": "https://github.com/o/r/pull/7",
                    "created_at": "2026-01-01T00:00:00Z",
                    "merged_at": "2026-01-02T00:00:00Z",
                    "base": {"ref": "main", "sha": "current-base"},
                    "head": {"ref": "feature", "sha": "head"},
                    "merge_commit_sha": "fixsha",
                    "changed_files": 2,
                    "labels": [],
                }
            },
        )()

    def paginate(self, path, params=None, accept=None):
        self.paginate_paths.append(path)
        if path.endswith("/files"):
            return files_record()["data"]
        if path.endswith("/commits"):
            return [{"sha": "abc", "commit": {"message": "support nested option"}}]
        return []


class Edit2RippleMiningTests(unittest.TestCase):
    def test_parse_github_pr_url(self):
        self.assertEqual(parse_github_pr_url("https://github.com/o/r/pull/7"), ("o/r", 7))
        self.assertIsNone(parse_github_pr_url("https://example.com/o/r/pull/7"))

    def test_builds_sample_with_anchor_diff_only_and_gold_files(self):
        samples, reason = build_pr_edit2ripple_samples(
            repo="o/r",
            pr_number=7,
            pr=pr(),
            files_record=files_record(),
        )

        self.assertIsNone(reason)
        self.assertEqual(len(samples), 2)
        sample = next(item for item in samples if item["query"]["anchor_file"] == "src/runtime/options.py")
        self.assertEqual(sample["task_type"], "edit2ripple")
        self.assertEqual(sample["query"]["anchor_file"], "src/runtime/options.py")
        self.assertEqual(sample["gold"]["files"], ["src/runtime/traversal.py"])
        self.assertEqual(sample["gold"]["given_files"], ["src/runtime/options.py"])
        query_text = json.dumps(sample["query"])
        self.assertNotIn("src/runtime/traversal.py", query_text)
        self.assertNotIn("traversal.py", query_text)
        self.assertNotIn("diff --git", query_text)
        self.assertEqual(diagnose_edit2ripple_leakage(sample)["fatal"], [])

    def test_retains_gold_stem_as_nonfatal_hint(self):
        samples, reason = build_pr_edit2ripple_samples(
            repo="o/r",
            pr_number=7,
            pr=pr(body="Add nested fields and update traversal for the new option."),
            files_record=files_record(),
        )

        self.assertIsNone(reason)
        self.assertEqual(len(samples), 2)
        hinted = next(item for item in samples if item["query"]["anchor_file"] == "src/runtime/options.py")
        leakage = diagnose_edit2ripple_leakage(hinted)
        self.assertEqual(leakage["fatal"], [])
        self.assertIn("gold_stem_hint", leakage["nonfatal"])

    def test_detects_format_only_patch(self):
        self.assertTrue(
            is_format_only_patch(
                """@@ def run():
-    return call(value)
+  return call( value )
"""
            )
        )
        self.assertFalse(
            is_format_only_patch(
                """@@ def run():
-    return call(value)
+    return call(value, include_nested=True)
"""
            )
        )

    def test_mine_writes_candidates_and_filters_gold_missing_from_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_repo = root / "raw" / "o__r"
            write_jsonl(raw_repo / "pull_requests.jsonl", [{"data": pr(), "repo": "o/r", "type": "pull_request"}])
            write_jsonl(raw_repo / "pull_files.jsonl", [files_record()])
            write_jsonl(raw_repo / "commit_details.jsonl", [])
            corpus = root / "corpus"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/options.py"},
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/traversal.py"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = mine_edit2ripple(
                raw_dir=root / "raw",
                out_dir=root / "benchmark",
                report_dir=root / "reports",
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                require_gold_in_corpus=True,
            )

            self.assertEqual(result["total"], 2)
            self.assertTrue((root / "benchmark" / "edit2ripple.jsonl").exists())
            self.assertTrue((root / "reports" / "audit_samples.csv").exists())

            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            write_jsonl(chunks_path, [{"repo": "o/r", "base_commit": "base", "path": "src/runtime/options.py"}])
            result = mine_edit2ripple(
                raw_dir=root / "raw",
                out_dir=root / "benchmark_missing",
                report_dir=root / "reports_missing",
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                require_gold_in_corpus=True,
            )

            self.assertEqual(result["total"], 1)
            self.assertEqual(result["dropped"]["gold_missing_from_corpus"], 1)
            retained = json.loads(
                (root / "benchmark_missing" / "edit2ripple.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(retained["gold"]["files"], ["src/runtime/options.py"])

    def test_mine_from_samples_refetches_pr_url_and_uses_sample_base_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "seed",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {},
                        "gold": {"related_tests": ["tests/test_options.py"]},
                        "metadata": {"pr_url": "https://github.com/o/r/pull/7"},
                    }
                ],
            )
            corpus = root / "corpus"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/options.py"},
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/traversal.py"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            api = FakePRAPI()
            result = mine_edit2ripple_from_samples(
                api=api,
                sample_paths=[samples],
                out_dir=root / "benchmark",
                report_dir=root / "reports",
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                require_corpus=True,
                require_gold_in_corpus=True,
            )
            candidates = [
                json.loads(line)
                for line in (root / "benchmark" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result["input_prs"], 1)
            self.assertEqual(result["fetched_prs"], 1)
            self.assertEqual(result["total"], 2)
            self.assertEqual({candidate["base_commit"] for candidate in candidates}, {"base"})
            self.assertEqual({tuple(candidate["metadata"]["source_samples"]) for candidate in candidates}, {("seed",)})
            self.assertEqual(api.get_paths, ["/repos/o/r/pulls/7"])
            self.assertEqual(api.paginate_paths, ["/repos/o/r/pulls/7/files", "/repos/o/r/pulls/7/commits"])

    @unittest.skipIf(shutil.which("git") is None, "git is required")
    def test_mine_from_sample_commits_uses_git_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            self._git(["init"], work)
            self._git(["config", "user.email", "test@example.com"], work)
            self._git(["config", "user.name", "Test User"], work)
            (work / "src/runtime").mkdir(parents=True)
            (work / "src/runtime/options.py").write_text("class RuntimeOptions:\n    include_empty = False\n", encoding="utf-8")
            (work / "src/runtime/traversal.py").write_text("def collect_values(options):\n    return collect(payload)\n", encoding="utf-8")
            self._git(["add", "."], work)
            self._git(["commit", "-m", "seed"], work)
            base = self._git(["rev-parse", "HEAD"], work).stdout.strip()
            (work / "src/runtime/options.py").write_text(
                "class RuntimeOptions:\n    include_empty = False\n    include_nested = False\n",
                encoding="utf-8",
            )
            (work / "src/runtime/traversal.py").write_text(
                "def collect_values(options):\n    return collect(payload, include_nested=options.include_nested)\n",
                encoding="utf-8",
            )
            self._git(["add", "."], work)
            self._git(["commit", "-m", "Support nested field option"], work)
            fix = self._git(["rev-parse", "HEAD"], work).stdout.strip()
            repos_dir = root / "repos"
            repos_dir.mkdir()
            self._git(["clone", "--bare", str(work), str(repos_dir / "o__r.git")], root)
            samples = root / "samples.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "seed",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": base,
                        "query": {"pr_title": "Support nested field option", "pr_body": "Add an option for nested fields."},
                        "gold": {"fix_commit": fix, "related_tests": []},
                        "metadata": {"pr": 7, "pr_url": "https://github.com/o/r/pull/7"},
                    }
                ],
            )
            corpus = root / "corpus"
            chunks_path = corpus / "o__r" / f"{base}.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": base, "path": "src/runtime/options.py"},
                    {"repo": "o/r", "base_commit": base, "path": "src/runtime/traversal.py"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": base, "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = mine_edit2ripple_from_sample_commits(
                sample_paths=[samples],
                out_dir=root / "benchmark",
                report_dir=root / "reports",
                repos_dir=repos_dir,
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                require_corpus=True,
                require_gold_in_corpus=True,
            )
            rows = [json.loads(line) for line in (root / "benchmark" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["processed_commits"], 1)
            self.assertEqual(result["total"], 2)
            candidate = next(item for item in rows if item["query"]["anchor_file"] == "src/runtime/options.py")
            self.assertEqual(candidate["gold"]["files"], ["src/runtime/traversal.py"])
            self.assertEqual({item["metadata"]["source"] for item in rows}, {"git_diff"})

    def test_report_edit2ripple_pilot_checks_gates_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples, _reason = build_pr_edit2ripple_samples(
                repo="o/r",
                pr_number=7,
                pr=pr(),
                files_record=files_record(),
            )
            sample_path = root / "samples.jsonl"
            write_jsonl(sample_path, samples)
            corpus = root / "corpus"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/options.py"},
                    {"repo": "o/r", "base_commit": "base", "path": "src/runtime/traversal.py"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            audit = root / "audit.csv"
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "verdict"])
                writer.writeheader()
                for sample in samples:
                    writer.writerow({"sample_id": sample["id"], "verdict": "valid"})

            report = report_edit2ripple_pilot(
                sample_paths=[sample_path],
                corpus_manifest=corpus / "corpus_manifest.jsonl",
                audit_path=audit,
                out_path=root / "pilot.md",
                json_out_path=root / "pilot.json",
                min_samples=1,
            )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["valid_rate"], 1.0)
            self.assertTrue(report["gates"]["fatal_leakage_zero"])
            self.assertTrue(report["gates"]["anchor_not_test"])
            self.assertTrue(report["gates"]["every_gold_in_corpus"])
            self.assertTrue((root / "pilot.md").exists())
            self.assertTrue((root / "pilot.json").exists())

    def test_report_edit2ripple_pilot_requires_every_sample_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _reason = build_pr_edit2ripple_samples(
                repo="o/r",
                pr_number=7,
                pr=pr(),
                files_record=files_record(),
            )
            second, _reason = build_pr_edit2ripple_samples(
                repo="o/r",
                pr_number=8,
                pr={**pr(), "number": 8, "url": "https://github.com/o/r/pull/8"},
                files_record=files_record(),
            )
            sample_path = root / "samples.jsonl"
            write_jsonl(sample_path, first + second)
            audit = root / "audit.csv"
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "verdict"])
                writer.writeheader()
                writer.writerow({"sample_id": first[0]["id"], "verdict": "valid"})

            report = report_edit2ripple_pilot(
                sample_paths=[sample_path],
                audit_path=audit,
                min_samples=1,
            )

            self.assertEqual(report["status"], "needs_audit")
            self.assertFalse(report["gates"]["audit_complete"])

    def test_baseline_helpers_understand_edit2ripple_schema_and_leakage(self):
        sample = {
            "task_type": "edit2ripple",
            "query": {"intent": "Support nested fields", "anchor_file": "src/runtime/options.py", "anchor_diff": "@@ class Options:"},
            "gold": {"files": ["src/runtime/traversal.py"], "given_files": ["src/runtime/options.py"]},
            "metadata": {"fix_commit": "fixsha"},
        }

        self.assertEqual(target_gold_files(sample), ["src/runtime/traversal.py"])
        self.assertEqual(given_files(sample), ["src/runtime/options.py"])
        self.assertFalse(query_has_leakage(sample, json.dumps(sample["query"])))

        hinted = {**sample, "query": {**sample["query"], "intent": "Update traversal for nested fields"}}
        self.assertFalse(query_has_leakage(hinted, json.dumps(hinted["query"])))
        self.assertIn("gold_stem_hint", diagnose_edit2ripple_leakage(hinted)["nonfatal"])

        leaked = {**sample, "query": {**sample["query"], "intent": "Update traversal.py for nested fields"}}
        self.assertTrue(query_has_leakage(leaked, json.dumps(leaked["query"])))

    def test_quality_validator_understands_edit2ripple_schema(self):
        sample = {
            "id": "sample",
            "task_type": "edit2ripple",
            "repo": "o/r",
            "base_commit": "base",
            "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": "base"},
            "query": {
                "intent": "Support nested fields",
                "anchor_file": "src/runtime/options.py",
                "anchor_diff": "@@ class Options:\n+ include_nested = False",
            },
            "gold": {"files": ["src/runtime/traversal.py"], "given_files": ["src/runtime/options.py"]},
            "metadata": {"fix_commit": "fixsha"},
        }

        self.assertEqual(validate_sample(sample), [])

        leaked = {**sample, "query": {**sample["query"], "intent": "Update traversal.py for nested fields"}}
        self.assertIn("edit2ripple query leaks gold", validate_sample(leaked))

    def _git(self, args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
