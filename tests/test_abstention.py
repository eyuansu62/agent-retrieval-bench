import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.abstention import (
    apply_abstention_audit_verdicts,
    backfill_abstention_issue_base_commits,
    export_abstention_clean,
    finalize_abstention_pilot,
    merge_abstention_audit_packets,
    mine_abstention_counterfactuals,
    mine_abstention_organic_candidates,
    parse_github_issue_html,
    prepare_abstention_audit_worklist,
    report_abstention_audit,
    report_abstention_completion_audit,
    report_abstention_crawling_status,
    report_abstention_pilot,
    report_abstention_shard_progress,
    render_abstention_review_packets,
    shard_abstention_audit_worklist,
    write_abstention_audit_handoff_manifest,
)
from agent_retrieval_bench.quality import validate_sample


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class AbstentionTests(unittest.TestCase):
    def test_mine_counterfactual_wrong_repo_writes_candidates_and_audit_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_manifest = root / "corpus_manifest.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "source-1",
                        "task_type": "trace2code",
                        "repo": "fastapi/fastapi",
                        "base_commit": "source-base",
                        "query": {
                            "pr_title": "Fix request validation error handling",
                            "raw_signal": "FAILED test_validation_error - AssertionError: expected status code 422 but got 500",
                            "trace_paths": ["fastapi/routing.py"],
                        },
                        "gold": {"root_cause_files": ["fastapi/routing.py"], "fix_commit": "abc123"},
                        "metadata": {"pr_url": "https://github.com/fastapi/fastapi/pull/1"},
                    }
                ],
            )
            write_jsonl(
                corpus_manifest,
                [
                    {"repo": "fastapi/fastapi", "base_commit": "source-base", "chunks_path": "source.chunks.jsonl", "status": "ok"},
                    {"repo": "encode/uvicorn", "base_commit": "wrong-base", "chunks_path": "wrong.chunks.jsonl", "status": "ok"},
                    {"repo": "scrapy/scrapy", "base_commit": "wrong-base-2", "chunks_path": "wrong2.chunks.jsonl", "status": "ok"},
                ],
            )

            result = mine_abstention_counterfactuals(
                sample_paths=[samples],
                corpus_manifest_path=corpus_manifest,
                out_dir=root / "candidates",
                audit_dir=root / "audit",
                limit=10,
                pairs_per_sample=2,
            )

            self.assertEqual(result["selected"], 2)
            candidates = [
                json.loads(line)
                for line in (root / "candidates" / "counterfactual_wrong_repo_candidates.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({candidate["id"] for candidate in candidates}), 2)
            self.assertEqual({candidate["repo"] for candidate in candidates}, {"encode/uvicorn", "scrapy/scrapy"})
            for candidate in candidates:
                self.assertEqual(candidate["task_type"], "abstention")
                self.assertEqual(candidate["gold"], {"files": [], "no_gold": True, "reason": "counterfactual_wrong_repo"})
                self.assertNotIn("fastapi/routing.py", candidate["query"]["text"])
                self.assertEqual(validate_sample(candidate), [])
            packet = json.loads((root / "audit" / "abstention_audit_packet.jsonl").read_text().splitlines()[0])
            self.assertEqual(packet["proposed_no_gold_reason"], "counterfactual_wrong_repo")

    def test_mine_organic_ci_requires_same_sha_failure_then_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "raw" / "pytest-dev__pytest"
            write_jsonl(
                repo_dir / "pull_requests.jsonl",
                [
                    {
                        "repo": "pytest-dev/pytest",
                        "type": "pull_request",
                        "data": {
                            "number": 42,
                            "title": "Fix docs build timeout",
                            "url": "https://github.com/pytest-dev/pytest/pull/42",
                        },
                    }
                ],
            )
            write_jsonl(
                repo_dir / "check_runs.jsonl",
                [
                    {
                        "repo": "pytest-dev/pytest",
                        "pr_number": 42,
                        "sha": "head-sha",
                        "data": [
                            {
                                "id": 1,
                                "name": "docs / link check",
                                "head_sha": "head-sha",
                                "conclusion": "timed_out",
                                "completed_at": "2026-01-01T00:00:00Z",
                                "html_url": "https://example.test/fail",
                                "output": {"summary": "GitHub Actions timeout while checking links"},
                            },
                            {
                                "id": 2,
                                "name": "docs / link check",
                                "head_sha": "head-sha",
                                "conclusion": "success",
                                "completed_at": "2026-01-01T00:05:00Z",
                                "html_url": "https://example.test/pass",
                            },
                        ],
                    }
                ],
            )

            result = mine_abstention_organic_candidates([root / "raw"], root / "candidates", root / "audit")

            self.assertEqual(result["raw_candidates"], 1)
            self.assertEqual(result["prefiltered"], 1)
            packet = json.loads((root / "audit" / "organic_abstention_audit_packet.jsonl").read_text().splitlines()[0])
            self.assertEqual(packet["proposed_no_gold_reason"], "flaky_ci")
            self.assertEqual(packet["query"]["source"], "ci_log")
            self.assertIn("same_head_sha_later_passed", packet["evidence"]["rerun_status"])

    def test_mine_organic_issue_uses_maintainer_resolution_as_evidence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "raw" / "pytest-dev__pytest"
            write_jsonl(
                repo_dir / "issues.jsonl",
                [
                    {
                        "repo": "pytest-dev/pytest",
                        "issue_number": 7,
                        "type": "issue",
                        "candidate_base_commit": "base-before-resolution",
                        "candidate_base_commit_source": "default_branch_before_maintainer_evidence",
                        "data": {
                            "number": 7,
                            "title": "Crash when plugin imports dependency on Python 3.13",
                            "body": "Running the minimal plugin reproduction raises ImportError during collection.",
                            "html_url": "https://github.com/pytest-dev/pytest/issues/7",
                        },
                    }
                ],
            )
            write_jsonl(
                repo_dir / "issue_comments.jsonl",
                [
                    {
                        "repo": "pytest-dev/pytest",
                        "issue_number": 7,
                        "type": "issue_comments",
                        "data": [
                            {
                                "author_association": "MEMBER",
                                "created_at": "2026-01-01T00:00:00Z",
                                "body": "This is an upstream dependency bug; please follow the linked library issue.",
                                "user": {"login": "maintainer"},
                            }
                        ],
                    }
                ],
            )
            write_jsonl(
                repo_dir / "issue_events.jsonl",
                [{"repo": "pytest-dev/pytest", "issue_number": 7, "type": "issue_events", "data": [{"event": "closed"}]}],
            )

            result = mine_abstention_organic_candidates([root / "raw"], root / "candidates", root / "audit")

            self.assertEqual(result["raw_candidates"], 1)
            self.assertEqual(result["prefiltered"], 1)
            packet = json.loads((root / "audit" / "organic_abstention_audit_packet.jsonl").read_text().splitlines()[0])
            self.assertEqual(packet["query"]["source"], "issue")
            self.assertEqual(packet["proposed_no_gold_reason"], "upstream_dependency")
            self.assertNotIn("upstream dependency bug", packet["query"]["text"])
            self.assertIn("upstream dependency bug", packet["evidence"]["resolution_comments"][0])

    def test_parse_github_issue_html_preload_extracts_comments_and_linked_prs(self):
        payload = {
            "payload": {
                "preloadedQueries": [
                    {
                        "queryName": "IssueViewerViewQuery",
                        "result": {
                            "data": {
                                "repository": {
                                    "issue": {
                                        "number": 7,
                                        "title": "Crash in dependency resolver",
                                        "body": "Resolver raises a ValueError for this input.",
                                        "url": "https://github.com/pypa/pip/issues/7",
                                        "createdAt": "2026-01-01T00:00:00Z",
                                        "state": "CLOSED",
                                        "author": {"login": "reporter"},
                                        "linkedPullRequestsIncludingClosed": {
                                            "nodes": [{"url": "https://github.com/pypa/pip/pull/8"}]
                                        },
                                        "frontTimelineItems": {
                                            "edges": [
                                                {
                                                    "node": {
                                                        "__typename": "IssueComment",
                                                        "authorAssociation": "MEMBER",
                                                        "createdAt": "2026-01-02T00:00:00Z",
                                                        "body": "This is an upstream bug in the resolver library.",
                                                        "url": "https://github.com/pypa/pip/issues/7#issuecomment-1",
                                                        "author": {"login": "maintainer"},
                                                    }
                                                }
                                            ]
                                        },
                                        "backTimelineItems": {"edges": []},
                                    }
                                }
                            }
                        },
                    }
                ]
            }
        }
        html = f'<html><script type="application/json">{json.dumps(payload)}</script></html>'

        parsed = parse_github_issue_html(html, "pypa/pip", 7)

        self.assertEqual(parsed["issue"]["title"], "Crash in dependency resolver")
        self.assertEqual(parsed["issue"]["linked_pull_requests"], ["https://github.com/pypa/pip/pull/8"])
        self.assertEqual(parsed["comments"][0]["author_association"], "MEMBER")
        self.assertIn("upstream bug", parsed["comments"][0]["body"])
        self.assertEqual(parsed["events"][0]["event"], "closed")

    def test_backfill_issue_base_commits_uses_local_git_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "git" / "owner__repo"
            repo_dir.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.test"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test User"], check=True)
            (repo_dir / "file.txt").write_text("before\n", encoding="utf-8")
            env = {
                **os.environ,
                "GIT_AUTHOR_DATE": "2024-01-01T00:00:00Z",
                "GIT_COMMITTER_DATE": "2024-01-01T00:00:00Z",
            }
            subprocess.run(["git", "-C", str(repo_dir), "add", "file.txt"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-q", "-m", "before"], check=True, env=env)
            before_sha = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
            (repo_dir / "file.txt").write_text("after\n", encoding="utf-8")
            env = {
                **os.environ,
                "GIT_AUTHOR_DATE": "2024-03-01T00:00:00Z",
                "GIT_COMMITTER_DATE": "2024-03-01T00:00:00Z",
            }
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-q", "-am", "after"], check=True, env=env)

            raw_dir = root / "raw"
            issue_dir = raw_dir / "owner__repo"
            issue_dir.mkdir(parents=True)
            write_jsonl(
                issue_dir / "issues.jsonl",
                [
                    {
                        "type": "issue",
                        "repo": "owner/repo",
                        "issue_number": 7,
                        "candidate_base_commit": "",
                        "candidate_base_commit_source": "",
                        "data": {"number": 7, "title": "Dependency crash"},
                    }
                ],
            )
            write_jsonl(
                issue_dir / "issue_comments.jsonl",
                [
                    {
                        "type": "issue_comments",
                        "repo": "owner/repo",
                        "issue_number": 7,
                        "data": [
                            {
                                "author_association": "MEMBER",
                                "created_at": "2024-02-01T00:00:00Z",
                                "body": "This is an upstream bug.",
                                "user": {"login": "maintainer"},
                            }
                        ],
                    }
                ],
            )

            report = backfill_abstention_issue_base_commits(
                raw_dirs=[raw_dir],
                git_repos_dir=root / "git",
                report_out=root / "report.json",
            )

            issues = [json.loads(line) for line in (issue_dir / "issues.jsonl").read_text().splitlines()]
            self.assertEqual(report["backfilled"], 1)
            self.assertEqual(issues[0]["candidate_base_commit"], before_sha)
            self.assertEqual(issues[0]["candidate_base_commit_source"], "local_default_branch_before_maintainer_evidence")
            self.assertTrue((root / "report.md").exists())

    def test_export_clean_requires_valid_no_gold_audit_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            audit = root / "audit.csv"
            write_jsonl(
                packet,
                [
                    {
                        "id": "abstention__counterfactual__1",
                        "repo": "encode/uvicorn",
                        "base_commit": "wrong-base",
                        "query": {"source": "counterfactual_wrong_repo", "text": "Failure excerpt: AssertionError in request validation test"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": "https://example.test/pr/1",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_sample_id": "source-1",
                                "source_repo": "fastapi/fastapi",
                                "source_task_type": "trace2code",
                                "pairing_profile": "python:web",
                            },
                        },
                        "audit_fields": {"verdict": "", "notes": "", "has_local_gold_files": []},
                    },
                    {
                        "id": "abstention__counterfactual__2",
                        "repo": "pytest-dev/pytest",
                        "base_commit": "wrong-base-2",
                        "query": {"source": "counterfactual_wrong_repo", "text": "Failure excerpt: another test failure"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {"source_url": "", "why_no_local_fix": "Counterfactual wrong-repo sample."},
                    },
                ],
            )
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                writer.writerow({"id": "abstention__counterfactual__1", "verdict": "valid_no_gold", "notes": "plausible wrong repo"})
                writer.writerow({"id": "abstention__counterfactual__2", "verdict": "ambiguous", "notes": "too weak"})

            result = export_abstention_clean(packet, audit, root / "clean", report_out=root / "report.json")

            rows = [json.loads(line) for line in (root / "clean" / "abstention.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["abstention__counterfactual__1"])
            self.assertEqual(rows[0]["audit"]["verdict"], "valid_no_gold")
            self.assertFalse(result["gates"]["counterfactual_no_more_than_half"])
            self.assertTrue(result["gates"]["schema_valid"])

    def test_export_clean_can_balance_counterfactual_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            audit = root / "audit.csv"
            packets = []
            for index in range(60):
                packets.append(
                    {
                        "id": f"organic-{index}",
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "issue", "text": f"Issue title: dependency failure {index}"},
                        "proposed_no_gold_reason": "upstream_dependency",
                        "evidence": {
                            "source_url": f"https://example.test/issues/{index}",
                            "why_no_local_fix": "Maintainer confirmed upstream issue.",
                        },
                    }
                )
            for index in range(90):
                packets.append(
                    {
                        "id": f"counter-{index}",
                        "repo": "encode/uvicorn",
                        "base_commit": "base",
                        "query": {"source": "counterfactual_wrong_repo", "text": f"Failure excerpt: request validation error {index}"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": f"https://example.test/pr/{index}",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                        },
                    }
                )
            write_jsonl(packet, packets)
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                for row in packets:
                    writer.writerow({"id": row["id"], "verdict": "valid_no_gold", "notes": "reviewed"})

            result = export_abstention_clean(
                packet,
                audit,
                root / "clean",
                report_out=root / "report.json",
                max_samples=100,
                max_counterfactual_share=0.5,
            )

            rows = [json.loads(line) for line in (root / "clean" / "abstention.jsonl").read_text().splitlines()]
            counterfactual = [row for row in rows if row["gold"]["reason"] == "counterfactual_wrong_repo"]
            self.assertEqual(len(rows), 100)
            self.assertLessEqual(len(counterfactual), 50)
            self.assertEqual(result["selected_before_balancing"], 150)
            self.assertTrue(result["gates"]["counterfactual_no_more_than_half"])

    def test_report_abstention_audit_summarizes_verdicts_and_balance_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            audit = root / "audit.csv"
            packets = [
                {
                    "id": "organic-valid",
                    "repo": "pytest-dev/pytest",
                    "base_commit": "base",
                    "query": {"source": "issue", "text": "Issue title: dependency failure"},
                    "proposed_no_gold_reason": "upstream_dependency",
                    "evidence": {"source_url": "https://example.test/issues/1", "why_no_local_fix": "Maintainer confirmed external cause."},
                },
                {
                    "id": "counter-valid",
                    "repo": "encode/uvicorn",
                    "base_commit": "base",
                    "query": {"source": "counterfactual_wrong_repo", "text": "Failure excerpt: request validation error"},
                    "proposed_no_gold_reason": "counterfactual_wrong_repo",
                    "evidence": {"source_url": "https://example.test/pr/1", "why_no_local_fix": "Counterfactual wrong-repo sample."},
                },
                {
                    "id": "organic-local",
                    "repo": "pytest-dev/pytest",
                    "base_commit": "base",
                    "query": {"source": "issue", "text": "Issue title: local config failure"},
                    "proposed_no_gold_reason": "external_service",
                    "evidence": {"source_url": "https://example.test/issues/2", "why_no_local_fix": "Needs review."},
                },
            ]
            write_jsonl(packet, packets)
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                writer.writerow({"id": "organic-valid", "verdict": "valid_no_gold", "notes": "valid"})
                writer.writerow({"id": "counter-valid", "verdict": "valid_no_gold", "notes": "valid"})
                writer.writerow({"id": "organic-local", "verdict": "has_local_gold", "notes": "needs config"})

            result = report_abstention_audit(
                packet,
                audit,
                out_path=root / "audit_report.json",
                max_samples=2,
                max_counterfactual_share=0.5,
            )

            self.assertEqual(result["reviewed"], 3)
            self.assertEqual(result["verdicts"]["valid_no_gold"], 2)
            self.assertEqual(result["verdicts"]["has_local_gold"], 1)
            self.assertEqual(result["valid_rate"], 2 / 3)
            self.assertEqual(result["balanced_clean_preview"]["total"], 2)
            self.assertEqual(result["balanced_clean_preview"]["counterfactual"], 1)
            self.assertFalse(result["gates"]["valid_rate_ge_90"])
            self.assertTrue((root / "audit_report.md").exists())

    def test_apply_audit_verdicts_updates_only_valid_filled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.csv"
            source = root / "source.csv"
            out = root / "merged.csv"
            fields = ["id", "repo", "verdict", "notes", "has_local_gold_files"]
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"id": "row-1", "repo": "o/r", "verdict": "", "notes": "", "has_local_gold_files": ""})
                writer.writerow({"id": "row-2", "repo": "o/r", "verdict": "", "notes": "", "has_local_gold_files": ""})
                writer.writerow({"id": "row-3", "repo": "o/r", "verdict": "", "notes": "", "has_local_gold_files": ""})
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes", "has_local_gold_files"])
                writer.writeheader()
                writer.writerow({"id": "row-1", "verdict": "valid_no_gold", "notes": "reviewed", "has_local_gold_files": ""})
                writer.writerow({"id": "row-2", "verdict": "", "notes": "blank ignored", "has_local_gold_files": ""})
                writer.writerow({"id": "row-3", "verdict": "not_a_real_verdict", "notes": "bad", "has_local_gold_files": ""})
                writer.writerow({"id": "unknown", "verdict": "valid_no_gold", "notes": "unknown", "has_local_gold_files": ""})

            result = apply_abstention_audit_verdicts(
                source,
                target,
                out,
                report_out=root / "apply_report.json",
            )

            self.assertEqual(result["applied"], 1)
            self.assertEqual(result["ignored_blank"], 1)
            self.assertEqual(result["remaining_missing"], 2)
            self.assertEqual(result["invalid_verdicts"], [{"id": "row-3", "verdict": "not_a_real_verdict"}])
            self.assertEqual(result["unknown_ids"], ["unknown"])
            self.assertFalse(result["gates"]["invalid_verdicts_zero"])
            with out.open(newline="", encoding="utf-8") as handle:
                rows = {row["id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["row-1"]["verdict"], "valid_no_gold")
            self.assertEqual(rows["row-1"]["notes"], "reviewed")
            self.assertEqual(rows["row-2"]["verdict"], "")
            self.assertEqual(rows["row-3"]["verdict"], "")
            self.assertTrue((root / "apply_report.md").exists())

    def test_apply_audit_verdicts_accepts_multiple_source_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.csv"
            source_one = root / "source_one.csv"
            source_two = root / "source_two.csv"
            out = root / "merged.csv"
            fields = ["id", "repo", "verdict", "notes", "has_local_gold_files"]
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for sample_id in ["row-1", "row-2", "row-3"]:
                    writer.writerow({"id": sample_id, "repo": "o/r", "verdict": "", "notes": "", "has_local_gold_files": ""})
            for path, rows in [
                (source_one, [{"id": "row-1", "verdict": "valid_no_gold", "notes": "first shard", "has_local_gold_files": ""}]),
                (source_two, [{"id": "row-2", "verdict": "has_local_gold", "notes": "second shard", "has_local_gold_files": "src/local.py"}]),
            ]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes", "has_local_gold_files"])
                    writer.writeheader()
                    writer.writerows(rows)

            result = apply_abstention_audit_verdicts(
                [source_one, source_two],
                target,
                out,
                report_out=root / "apply_report.json",
            )

            self.assertEqual(result["inputs"]["source_audits"], [str(source_one), str(source_two)])
            self.assertEqual(result["applied"], 2)
            self.assertEqual(result["remaining_missing"], 1)
            self.assertEqual(result["verdicts_after"], {"has_local_gold": 1, "missing": 1, "valid_no_gold": 1})
            with out.open(newline="", encoding="utf-8") as handle:
                rows = {row["id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["row-1"]["notes"], "first shard")
            self.assertEqual(rows["row-2"]["has_local_gold_files"], "src/local.py")
            self.assertEqual(rows["row-3"]["verdict"], "")

    def test_finalize_pilot_waits_for_audit_and_exports_when_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            organic_prefiltered = root / "organic_prefiltered.jsonl"
            counterfactual_candidates = root / "counterfactual_candidates.jsonl"
            audit = root / "audit.csv"
            packets = []
            organic_candidates = []
            counterfactual_rows = []
            for index in range(75):
                sample_id = f"organic-{index:03d}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "issue", "text": f"Issue title: release command fails with status code {index}"},
                        "proposed_no_gold_reason": "external_service",
                        "evidence": {
                            "source_url": f"https://example.test/issues/{index}",
                            "why_no_local_fix": "Maintainer confirmed provider-side outage.",
                        },
                    }
                )
                organic_candidates.append({"id": sample_id, "candidate_reason": "external_service"})
            for index in range(75):
                sample_id = f"counter-{index:03d}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "encode/uvicorn",
                        "base_commit": "base",
                        "query": {"source": "counterfactual_wrong_repo", "text": f"Failure excerpt: request validation assertion {index}"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": f"https://example.test/pr/{index}",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_repo": "fastapi/fastapi",
                                "source_sample_id": f"source-{index}",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                )
                counterfactual_rows.append(
                    {
                        "id": sample_id,
                        "task_type": "abstention",
                        "gold": {"files": [], "no_gold": True, "reason": "counterfactual_wrong_repo"},
                    }
                )
            write_jsonl(packet, packets)
            write_jsonl(organic_prefiltered, organic_candidates)
            write_jsonl(counterfactual_candidates, counterfactual_rows)
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                for row in packets:
                    writer.writerow({"id": row["id"], "verdict": "", "notes": ""})

            not_ready = finalize_abstention_pilot(
                [counterfactual_candidates, organic_prefiltered],
                packet,
                audit,
                out_dir=root / "clean_not_ready",
                report_out=root / "finalization_not_ready.json",
                audit_report_out=root / "audit_report_not_ready.json",
                pilot_report_out=root / "pilot_not_ready.json",
                completion_report_out=root / "completion_not_ready.json",
                max_samples=100,
                max_counterfactual_share=0.5,
            )

            self.assertEqual(not_ready["status"], "not_ready")
            self.assertFalse(not_ready["exported"])
            self.assertFalse((root / "clean_not_ready" / "abstention.jsonl").exists())

            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                for row in packets:
                    writer.writerow({"id": row["id"], "verdict": "valid_no_gold", "notes": "reviewed"})

            ready = finalize_abstention_pilot(
                [counterfactual_candidates, organic_prefiltered],
                packet,
                audit,
                out_dir=root / "clean_ready",
                report_out=root / "finalization_ready.json",
                audit_report_out=root / "audit_report_ready.json",
                pilot_report_out=root / "pilot_ready.json",
                completion_report_out=root / "completion_ready.json",
                max_samples=100,
                max_counterfactual_share=0.5,
            )

            self.assertEqual(ready["status"], "complete")
            self.assertTrue(ready["exported"])
            rows = [json.loads(line) for line in (root / "clean_ready" / "abstention.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertEqual(sum(1 for row in rows if row["gold"]["reason"] == "counterfactual_wrong_repo"), 50)
            self.assertTrue((root / "finalization_ready.md").exists())

    def test_finalize_can_use_separate_crawling_stage_report_for_clean_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "clean_packet.jsonl"
            prefiltered = root / "clean_prefiltered.jsonl"
            audit = root / "clean_audit.csv"
            crawling_report = root / "crawling_status.json"
            packets = []
            prefiltered_rows = []
            for index in range(50):
                sample_id = f"organic-{index}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "issue", "text": f"Issue title: platform failure {index}"},
                        "proposed_no_gold_reason": "upstream_dependency",
                        "evidence": {
                            "source_url": f"https://example.test/issues/{index}",
                            "why_no_local_fix": "Maintainer confirmed the failure is outside the repository.",
                        },
                    }
                )
                prefiltered_rows.append({"id": sample_id, "candidate_reason": "upstream_dependency"})
            for index in range(30):
                sample_id = f"counter-{index}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "encode/uvicorn",
                        "base_commit": "base",
                        "query": {"source": "counterfactual_wrong_repo", "text": f"Failure excerpt: request validation error {index}"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": f"https://example.test/pr/{index}",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_repo": "fastapi/fastapi",
                                "source_sample_id": f"source-{index}",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                )
                prefiltered_rows.append(
                    {
                        "id": sample_id,
                        "task_type": "abstention",
                        "gold": {"files": [], "no_gold": True, "reason": "counterfactual_wrong_repo"},
                    }
                )
            write_jsonl(packet, packets)
            write_jsonl(prefiltered, prefiltered_rows)
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                for row in packets:
                    writer.writerow({"id": row["id"], "verdict": "valid_no_gold", "notes": "reviewed"})
            crawling_report.write_text(
                json.dumps(
                    {
                        "status": "ready_for_manual_audit",
                        "prefiltered_candidates": 150,
                        "prefiltered_by_pool": {"organic_no_gold": 75, "counterfactual_wrong_repo": 75},
                        "gates": {"prefiltered_candidates_ge_150": True},
                    }
                ),
                encoding="utf-8",
            )

            result = finalize_abstention_pilot(
                [prefiltered],
                packet,
                audit,
                out_dir=root / "clean",
                report_out=root / "finalization.json",
                audit_report_out=root / "audit_report.json",
                pilot_report_out=root / "pilot.json",
                completion_report_out=root / "completion.json",
                crawling_report_path=crawling_report,
                max_samples=100,
                max_counterfactual_share=0.5,
            )

            self.assertEqual(result["status"], "complete")
            completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
            requirements = {item["name"]: item["status"] for item in completion["requirements"]}
            self.assertEqual(requirements["prefiltered_candidates_ge_150"], "passed")
            self.assertEqual(completion["final_clean_status"]["total"], 80)

    def test_prepare_audit_worklist_balances_core_without_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            packets = []
            for index in range(3):
                packets.append(
                    {
                        "id": f"organic-{index}",
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "issue", "text": f"Issue title: upstream failure {index}"},
                        "proposed_no_gold_reason": "upstream_dependency",
                        "evidence": {
                            "source_url": f"https://example.test/issues/{index}",
                            "why_no_local_fix": "Maintainer confirmed upstream issue.",
                        },
                    }
                )
            for index in range(6):
                packets.append(
                    {
                        "id": f"counter-{index}",
                        "repo": "encode/uvicorn",
                        "base_commit": "base",
                        "query": {"source": "counterfactual_wrong_repo", "text": f"Failure excerpt: request validation error {index}"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": f"https://example.test/pr/{index}",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_sample_id": f"source-{index}",
                                "source_repo": "fastapi/fastapi",
                                "source_task_type": "trace2code",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                )
            write_jsonl(packet, packets)

            result = prepare_abstention_audit_worklist(
                packet,
                out_csv=root / "worklist.csv",
                out_jsonl=root / "worklist.jsonl",
                report_out=root / "worklist_report.json",
                target_size=6,
                max_counterfactual_share=0.5,
            )

            self.assertEqual(result["core"]["total"], 6)
            self.assertEqual(result["core"]["organic"], 3)
            self.assertEqual(result["core"]["counterfactual"], 3)
            self.assertEqual(result["reserve"]["total"], 3)
            self.assertTrue(result["gates"]["core_counterfactual_no_more_than_half"])
            with (root / "worklist.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[0]["audit_priority"], "core")
            self.assertEqual(rows[0]["verdict"], "")
            self.assertIn("valid_no_gold", rows[0]["allowed_verdicts"])
            self.assertTrue((root / "worklist_report.md").exists())

    def test_shard_audit_worklist_splits_core_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worklist = root / "worklist.csv"
            fields = ["audit_order", "audit_priority", "pool", "proposed_no_gold_reason", "id", "verdict"]
            with worklist.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(1, 8):
                    writer.writerow(
                        {
                            "audit_order": str(index),
                            "audit_priority": "core",
                            "pool": "organic_no_gold",
                            "proposed_no_gold_reason": "external_service",
                            "id": f"core-{index}",
                            "verdict": "",
                        }
                    )
                for index in range(8, 11):
                    writer.writerow(
                        {
                            "audit_order": str(index),
                            "audit_priority": "reserve",
                            "pool": "counterfactual_wrong_repo",
                            "proposed_no_gold_reason": "counterfactual_wrong_repo",
                            "id": f"reserve-{index}",
                            "verdict": "",
                        }
                    )

            report = shard_abstention_audit_worklist(
                worklist,
                root / "shards",
                report_out=root / "shards_report.json",
                shard_size=3,
                priority="core",
            )

            self.assertEqual(report["selected_rows"], 7)
            self.assertEqual([shard["rows"] for shard in report["shards"]], [3, 3, 1])
            self.assertEqual(report["selected_by_pool"], {"organic_no_gold": 7})
            first_shard = root / "shards" / "abstention_audit_core_shard_01.csv"
            with first_shard.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in rows], ["core-1", "core-2", "core-3"])
            self.assertTrue((root / "shards" / "abstention_audit_core_shard_01.jsonl").exists())
            self.assertTrue((root / "shards_report.md").exists())

    def test_report_shard_progress_counts_pending_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_one = root / "shard_one.csv"
            shard_two = root / "shard_two.csv"
            fields = ["id", "verdict", "notes"]
            with shard_one.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"id": "row-1", "verdict": "valid_no_gold", "notes": "reviewed"})
                writer.writerow({"id": "row-2", "verdict": "", "notes": ""})
            with shard_two.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"id": "row-3", "verdict": "not_real", "notes": "bad"})

            report = report_abstention_shard_progress(
                [shard_one, shard_two],
                out_path=root / "progress.json",
            )

            self.assertEqual(report["status"], "not_ready")
            self.assertEqual(report["total_rows"], 3)
            self.assertEqual(report["reviewed"], 1)
            self.assertEqual(report["pending"], 1)
            self.assertEqual(report["invalid_verdicts"], [{"id": "row-3", "verdict": "not_real", "path": str(shard_two)}])
            self.assertFalse(report["gates"]["invalid_verdicts_zero"])
            self.assertTrue((root / "progress.md").exists())

    def test_render_review_packets_expands_query_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard.csv"
            fields = [
                "audit_order",
                "id",
                "repo",
                "base_commit",
                "pool",
                "proposed_no_gold_reason",
                "query_source",
                "query_text",
                "source_url",
                "review_focus",
                "why_no_local_fix",
                "resolution_comments",
                "evidence_snippets",
                "verdict",
                "notes",
                "has_local_gold_files",
            ]
            with shard.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "audit_order": "1",
                        "id": "row-1",
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "pool": "organic_no_gold",
                        "proposed_no_gold_reason": "upstream_dependency",
                        "query_source": "issue",
                        "query_text": "Issue title: dependency failure",
                        "source_url": "https://example.test/issues/1",
                        "review_focus": "Verify upstream attribution.",
                        "why_no_local_fix": "Maintainer confirmed upstream.",
                        "resolution_comments": json.dumps(["This belongs upstream."]),
                        "evidence_snippets": json.dumps(["No linked local fix."]),
                        "verdict": "",
                        "notes": "",
                        "has_local_gold_files": "",
                    }
                )

            report = render_abstention_review_packets(
                [shard],
                root / "review_packets",
                report_out=root / "review_packets_report.json",
            )

            self.assertEqual(report["total_rows"], 1)
            self.assertEqual(report["total_pending"], 1)
            markdown = (root / "review_packets" / "shard_review.md").read_text()
            self.assertIn("Issue title: dependency failure", markdown)
            self.assertIn("This belongs upstream.", markdown)
            self.assertIn("Fill verdicts in the source CSV", markdown)
            self.assertTrue((root / "review_packets_report.md").exists())

    def test_write_handoff_manifest_fingerprints_review_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            shard_dir = audit_dir / "shards"
            review_dir = audit_dir / "review_packets"
            shard_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (audit_dir / "abstention_manual_audit_handoff.md").write_text("handoff\n", encoding="utf-8")
            write_jsonl(audit_dir / "abstention_audit_packet.jsonl", [{"id": "row-1"}])
            write_jsonl(audit_dir / "abstention_manual_audit_worklist.jsonl", [{"id": "row-1"}])
            for path in [
                audit_dir / "abstention_audit_packet.csv",
                audit_dir / "abstention_manual_audit_worklist.csv",
                shard_dir / "shard_01.csv",
            ]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["id", "verdict"])
                    writer.writeheader()
                    writer.writerow({"id": "row-1", "verdict": ""})
            write_jsonl(shard_dir / "shard_01.jsonl", [{"id": "row-1"}])
            (review_dir / "shard_01_review.md").write_text("# Review\n", encoding="utf-8")
            (audit_dir / "abstention_manual_audit_worklist_report.json").write_text(
                json.dumps({"status": "ready_for_manual_audit"}), encoding="utf-8"
            )
            (audit_dir / "abstention_shard_progress_report.json").write_text(
                json.dumps(
                    {
                        "status": "in_progress",
                        "total_rows": 1,
                        "reviewed": 0,
                        "pending": 1,
                        "gates": {
                            "invalid_verdicts_zero": True,
                            "duplicate_ids_zero": True,
                            "conflicting_verdicts_zero": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "abstention_review_packets_report.json").write_text(
                json.dumps({"status": "ready_for_manual_audit", "total_rows": 1}), encoding="utf-8"
            )
            (audit_dir / "abstention_crawling_status.json").write_text(
                json.dumps({"status": "ready_for_manual_audit"}), encoding="utf-8"
            )
            (audit_dir / "abstention_completion_audit.json").write_text(
                json.dumps({"status": "ready_for_manual_audit", "status_counts": {"passed": 1}}), encoding="utf-8"
            )

            report = write_abstention_audit_handoff_manifest(
                audit_dir=audit_dir,
                out_path=audit_dir / "abstention_audit_handoff_manifest.json",
                finalization_report_path=root / "missing_finalization.json",
            )

            self.assertEqual(report["status"], "ready_for_manual_audit")
            self.assertEqual(report["reviewer_shards"], {"csv": 1, "jsonl": 1})
            self.assertEqual(report["review_packets"], 1)
            csv_entries = [entry for entry in report["files"] if entry["path"].endswith("shard_01.csv")]
            self.assertEqual(csv_entries[0]["rows"], 1)
            self.assertIn("sha256", csv_entries[0])
            apply_command = report["next_commands"][1]
            self.assertIn("--source-audit", apply_command)
            self.assertIn("shard_01.csv", apply_command)
            self.assertNotIn("...", apply_command)
            self.assertIn("--require-complete", apply_command)
            markdown = (audit_dir / "abstention_audit_handoff_manifest.md").read_text(encoding="utf-8")
            self.assertIn("```bash", markdown)
            self.assertTrue((audit_dir / "abstention_audit_handoff_manifest.md").exists())

    def test_completion_audit_reports_manual_audit_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            organic_prefiltered = root / "organic_prefiltered.jsonl"
            counterfactual_candidates = root / "counterfactual_candidates.jsonl"
            audit = root / "audit.csv"
            packets = []
            organic_candidates = []
            counterfactual_rows = []
            for index in range(75):
                sample_id = f"organic-{index}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "issue", "text": f"Issue title: dependency failure {index}"},
                        "proposed_no_gold_reason": "upstream_dependency",
                        "evidence": {
                            "source_url": f"https://example.test/issues/{index}",
                            "why_no_local_fix": "Maintainer confirmed dependency-owned failure.",
                        },
                    }
                )
                organic_candidates.append({"id": sample_id, "candidate_reason": "upstream_dependency"})
            for index in range(75):
                sample_id = f"counter-{index}"
                packets.append(
                    {
                        "id": sample_id,
                        "repo": "encode/uvicorn",
                        "base_commit": "base",
                        "query": {"source": "counterfactual_wrong_repo", "text": f"Failure excerpt: request validation error {index}"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": f"https://example.test/pr/{index}",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_repo": "fastapi/fastapi",
                                "source_sample_id": f"source-{index}",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                )
                counterfactual_rows.append(
                    {
                        "id": sample_id,
                        "task_type": "abstention",
                        "gold": {"files": [], "no_gold": True, "reason": "counterfactual_wrong_repo"},
                    }
                )
            write_jsonl(packet, packets)
            write_jsonl(organic_prefiltered, organic_candidates)
            write_jsonl(counterfactual_candidates, counterfactual_rows)
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "notes"])
                writer.writeheader()
                for row in packets:
                    writer.writerow({"id": row["id"], "verdict": "", "notes": ""})

            result = report_abstention_completion_audit(
                [counterfactual_candidates, organic_prefiltered],
                packet,
                audit,
                clean_sample_paths=[],
                out_path=root / "completion.json",
            )

            self.assertEqual(result["status"], "ready_for_manual_audit")
            requirements = {item["name"]: item["status"] for item in result["requirements"]}
            self.assertEqual(requirements["prefiltered_candidates_ge_150"], "passed")
            self.assertEqual(requirements["manual_audit_all_packets_reviewed"], "pending")
            self.assertEqual(requirements["final_clean_no_gold_samples_ge_50"], "pending")
            self.assertTrue((root / "completion.md").exists())

    def test_merge_packets_and_report_crawling_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counterfactual_packet = root / "counterfactual.jsonl"
            organic_packet = root / "organic.jsonl"
            counterfactual_candidates = root / "counterfactual_candidates.jsonl"
            organic_prefiltered = root / "organic_prefiltered.jsonl"
            write_jsonl(
                counterfactual_packet,
                [
                    {
                        "id": "abstention__counterfactual__1",
                        "repo": "encode/uvicorn",
                        "base_commit": "wrong-base",
                        "query": {"source": "counterfactual_wrong_repo", "text": "Failure excerpt: request validation test returns 500"},
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": "https://example.test/pr/1",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_repo": "fastapi/fastapi",
                                "source_sample_id": "source-1",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                ],
            )
            write_jsonl(
                organic_packet,
                [
                    {
                        "id": "abstention_candidate__organic__1",
                        "repo": "pytest-dev/pytest",
                        "base_commit": "base",
                        "query": {"source": "ci_log", "text": "Check name: docs\nConclusion: timed_out\nFailure excerpt: link check timeout"},
                        "proposed_no_gold_reason": "flaky_ci",
                        "evidence": {
                            "source_url": "https://example.test/check",
                            "rerun_status": "same_head_sha_later_passed",
                            "why_no_local_fix": "Same SHA later passed.",
                        },
                    }
                ],
            )
            write_jsonl(counterfactual_candidates, [{"id": "abstention__counterfactual__1", "task_type": "abstention", "gold": {"reason": "counterfactual_wrong_repo"}}])
            write_jsonl(organic_prefiltered, [{"id": "abstention_candidate__organic__1", "candidate_reason": "flaky_ci"}])

            merge = merge_abstention_audit_packets(
                [counterfactual_packet, organic_packet],
                root / "merged.jsonl",
                root / "merged.csv",
                report_out=root / "merge_report.json",
            )
            self.assertEqual(merge["total"], 2)
            self.assertFalse(merge["ready_for_manual_audit"])
            self.assertEqual(merge["invalid_packets"], [])
            self.assertTrue((root / "merged.csv").exists())

            status = report_abstention_crawling_status(
                [counterfactual_candidates, organic_prefiltered],
                root / "merged.jsonl",
                out_path=root / "status.json",
            )
            self.assertEqual(status["prefiltered_candidates"], 2)
            self.assertEqual(status["audit_packet_candidates"], 2)
            self.assertFalse(status["gates"]["prefiltered_candidates_ge_150"])
            self.assertTrue(status["gates"]["prefiltered_ids_have_audit_packets"])
            self.assertTrue(status["gates"]["audit_packets_schema_valid"])
            self.assertTrue(status["gates"]["counterfactual_queries_avoid_source_identity"])
            self.assertTrue(status["gates"]["counterfactual_pairing_metadata_present"])
            self.assertEqual(status["counterfactual_diagnostics"]["by_pairing_profile"], {"python:web": 1})

    def test_crawling_status_flags_counterfactual_source_identity_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_packet = root / "packet.jsonl"
            candidates = root / "candidates.jsonl"
            write_jsonl(
                audit_packet,
                [
                    {
                        "id": "abstention__counterfactual__1",
                        "repo": "encode/uvicorn",
                        "base_commit": "wrong-base",
                        "query": {
                            "source": "counterfactual_wrong_repo",
                            "text": "Failure excerpt: FastAPI request validation returns 500",
                        },
                        "proposed_no_gold_reason": "counterfactual_wrong_repo",
                        "evidence": {
                            "source_url": "https://example.test/pr/1",
                            "why_no_local_fix": "Counterfactual wrong-repo sample.",
                            "counterfactual": {
                                "source_repo": "fastapi/fastapi",
                                "source_sample_id": "source-1",
                                "pairing_profile": "python:web",
                            },
                        },
                    }
                ],
            )
            write_jsonl(candidates, [{"id": "abstention__counterfactual__1", "task_type": "abstention"}])

            status = report_abstention_crawling_status([candidates], audit_packet)

            self.assertFalse(status["gates"]["no_packet_query_contains_resolution_answer"])
            self.assertFalse(status["gates"]["counterfactual_queries_avoid_source_identity"])
            self.assertEqual(
                status["counterfactual_diagnostics"]["source_identity_leaks"],
                [{"id": "abstention__counterfactual__1", "source_repo": "fastapi/fastapi", "wrong_repo": "encode/uvicorn"}],
            )

    def test_report_and_quality_reject_malformed_abstention_rows(self):
        valid = {
            "id": "abstention__1",
            "task_type": "abstention",
            "repo": "o/r",
            "base_commit": "base",
            "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": "base"},
            "query": {"source": "issue", "text": "Issue title: token permission error during release"},
            "gold": {"files": [], "no_gold": True, "reason": "external_service"},
        }
        invalid = {
            **valid,
            "id": "abstention__2",
            "gold": {"files": ["src/local.py"], "no_gold": True, "reason": "external_service"},
        }

        self.assertEqual(validate_sample(valid), [])
        self.assertIn("abstention gold.files must be empty", validate_sample(invalid))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "abstention.jsonl"
            write_jsonl(samples, [valid, invalid])
            report = report_abstention_pilot([samples])
            self.assertFalse(report["gates"]["schema_valid"])
            self.assertEqual(report["invalid_rows"][0]["id"], "abstention__2")


if __name__ == "__main__":
    unittest.main()
