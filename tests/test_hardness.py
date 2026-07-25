import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.hardness import (
    diagnose_hardness,
    filter_hard_pool,
    hard_query_hints,
    lexical_rank_bucket,
    merge_seed_audits,
    same_directory_gold,
    summarize_seed_audit,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class HardnessTests(unittest.TestCase):
    def test_hard_query_hints_ignore_given_file_tokens(self):
        hints = hard_query_hints(
            '{"comment": "Can we cover the src/auth/session.py behavior?"}',
            ["tests/auth/session_test.py"],
            ["src/auth/session.py"],
        )

        self.assertFalse(hints["direct_path_hint"])
        self.assertFalse(hints["basename_hint"])
        self.assertFalse(hints["module_overlap"])
        self.assertIn("session", hints["ignored_reference_tokens"])

    def test_hard_query_hints_detect_basename_and_module_overlap(self):
        basename = hard_query_hints(
            '{"comment": "Please add tests in session_test.py"}',
            ["tests/auth/session_test.py"],
            ["src/auth/session.py"],
        )
        module = hard_query_hints(
            '{"comment": "Does scheduler preserve fairness under load?"}',
            ["runtime/scheduler/fairness.rs"],
            ["runtime/task/mod.rs"],
        )

        self.assertTrue(basename["basename_hint"])
        self.assertTrue(module["module_overlap"])
        self.assertEqual(module["module_token_hits"], {"runtime/scheduler/fairness.rs": ["fairness", "scheduler"]})

    def test_same_directory_gold_uses_given_or_query_files(self):
        self.assertEqual(
            same_directory_gold(["src/auth/session_test.py", "tests/auth/test_api.py"], ["src/auth/session.py"]),
            ["src/auth/session_test.py"],
        )

    def test_lexical_rank_bucket_marks_deep_hits_as_hard_signal(self):
        self.assertEqual(lexical_rank_bucket({"Recall@20": 0.0, "MRR": 0.0}), "miss@20")
        self.assertEqual(lexical_rank_bucket({"Recall@20": 0.5, "MRR": 0.1}), "partial@20")
        self.assertEqual(lexical_rank_bucket({"Recall@5": 0.0, "Recall@20": 1.0, "MRR": 0.1}), "deep@20")
        self.assertEqual(lexical_rank_bucket({"Recall@5": 1.0, "Recall@20": 1.0, "MRR": 1.0}), "top5")

    def test_diagnose_hardness_writes_report_and_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "samples.jsonl"
            details_path = root / "eval" / "lexical_details.jsonl"
            corpus_manifest_path = root / "corpus" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "o__r" / "base.chunks.jsonl"
            keep_list = root / "audit" / "keep_samples.jsonl"
            out_dir = root / "reports"

            write_jsonl(
                samples_path,
                [
                    {
                        "id": "hard",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth/session.py", "pr_title": "tighten auth session expiry"},
                        "gold": {"related_tests": ["tests/auth/test_session_expiry.py"]},
                    },
                    {
                        "id": "too-easy",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"path": "src/auth/session.py", "comment": "Please update test_session_expiry.py too"},
                        "gold": {
                            "given_files": ["src/auth/session.py"],
                            "must_context_files": [{"path": "tests/auth/test_session_expiry.py"}],
                        },
                    },
                    {
                        "id": "missing",
                        "task_type": "trace2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"trace": "panic in missing.rs"},
                        "gold": {"root_cause_files": ["src/missing.rs"]},
                    },
                ],
            )
            write_jsonl(
                details_path,
                [
                    {
                        "sample_id": "hard",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["tests/auth/test_session_expiry.py"],
                        "top_files": ["src/auth/session.py"],
                        "metrics": {"Recall@5": 0, "Recall@10": 0, "Recall@20": 0, "MRR": 0, "gold_coverage@8k": 0},
                    },
                    {
                        "sample_id": "too-easy",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["tests/auth/test_session_expiry.py"],
                        "top_files": ["tests/auth/test_session_expiry.py", "src/auth/session.py"],
                        "metrics": {"Recall@5": 1, "Recall@10": 1, "Recall@20": 1, "MRR": 1, "gold_coverage@8k": 1},
                    },
                    {
                        "sample_id": "missing",
                        "task_type": "trace2code",
                        "repo": "o/r",
                        "base_commit": "base",
                        "gold_files": ["src/missing.rs"],
                        "top_files": ["src/auth/session.py"],
                        "metrics": {"Recall@5": 0, "Recall@10": 0, "Recall@20": 0, "MRR": 0, "gold_coverage@8k": 0},
                    },
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": "src/auth/session.py", "kind": "file"},
                    {"repo": "o/r", "base_commit": "base", "path": "tests/auth/test_session_expiry.py", "kind": "file"},
                ],
            )
            write_jsonl(
                corpus_manifest_path,
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            write_jsonl(
                keep_list,
                [
                    {"sample_id": "hard", "task_type": "code2test", "verdict": "valid"},
                    {"sample_id": "too-easy", "task_type": "comment2context", "verdict": "valid"},
                    {"sample_id": "missing", "task_type": "trace2code", "verdict": "valid"},
                ],
            )

            result = diagnose_hardness(
                sample_paths=[samples_path],
                corpus_manifest_path=corpus_manifest_path,
                details_path=details_path,
                out_dir=out_dir,
                keep_list=keep_list,
                tasks=["code2test", "comment2context", "trace2code"],
            )
            summary = json.loads((out_dir / "hardness_report.json").read_text(encoding="utf-8"))
            pool = [json.loads(line) for line in (out_dir / "candidate_keep_pool.jsonl").read_text().splitlines()]
            report = (out_dir / "hardness_report.md").read_text(encoding="utf-8")

            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["hard_curated"], 1)
            self.assertEqual(summary["distribution"]["by_label"]["hard"], 1)
            self.assertEqual(summary["distribution"]["by_label"]["too_easy_hint"], 1)
            self.assertEqual(summary["distribution"]["by_label"]["invalid_missing_gold"], 1)
            self.assertEqual(pool[0]["sample_id"], "hard")
            self.assertEqual(pool[0]["recommended_split"], "hard_curated")
            self.assertEqual(pool[-1]["recommended_split"], "drop")
            self.assertIn("V1 Target Gaps", report)

    def test_filter_hard_pool_applies_audit_and_noise_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path = root / "candidate_keep_pool.jsonl"
            audit_path = root / "audit.jsonl"
            out_path = root / "v1_seed_candidates.jsonl"
            summary_path = root / "v1_seed_summary.json"
            audit_out = root / "v1_seed_audit_samples.jsonl"
            audit_csv = root / "v1_seed_audit_samples.csv"

            def row(sample_id, **overrides):
                base = {
                    "sample_id": sample_id,
                    "task_type": "code2test",
                    "repo": "o/r",
                    "pr_url": f"https://example.test/{sample_id}",
                    "gold_files": [f"tests/{sample_id}.py"],
                    "gold_count": 1,
                    "gold_in_corpus": True,
                    "recommended_split": "hard_curated",
                    "hardness_score": 10.0,
                    "lexical_rank_bucket": "miss@20",
                    "direct_path_hint": False,
                    "basename_hint": False,
                    "module_overlap": False,
                    "same_directory_gold": False,
                    "query_excerpt": "behavior changes when invalid input returns an error",
                    "metrics": {"Recall@5": 0, "Recall@10": 0, "Recall@20": 0, "MRR": 0, "gold_coverage@8k": 0},
                }
                base.update(overrides)
                return base

            write_jsonl(
                pool_path,
                [
                    row("manual-valid", pr_url="https://example.test/cluster", gold_files=["tests/shared.py"]),
                    row("cluster-duplicate", pr_url="https://example.test/cluster", gold_files=["tests/shared.py"]),
                    row("manual-noisy"),
                    row("generated", query_excerpt="<!-- auto-generated comment: release notes by coderabbit.ai -->"),
                    row("template", query_excerpt="What does this PR do? Please replace this with a description."),
                    row("broad", gold_files=["a.py", "b.py", "c.py", "d.py"], gold_count=4),
                    row("same-dir", same_directory_gold=True, query_excerpt="add tests"),
                    row("manual-broad-valid", gold_files=["a.py", "b.py", "c.py", "d.py"], gold_count=4),
                    row("auto-keep"),
                ],
            )
            write_jsonl(
                audit_path,
                [
                    {"sample_id": "manual-valid", "verdict": "valid", "keep_for_v1_hard": True},
                    {"sample_id": "manual-noisy", "verdict": "noisy", "keep_for_v1_hard": False},
                    {"sample_id": "manual-broad-valid", "verdict": "valid", "keep_for_v1_hard": True},
                ],
            )

            result = filter_hard_pool(
                pool_path=pool_path,
                out_path=out_path,
                summary_path=summary_path,
                audit_path=audit_path,
                audit_out_path=audit_out,
                audit_csv_path=audit_csv,
            )
            selected = [json.loads(line) for line in out_path.read_text().splitlines()]
            summary = json.loads(summary_path.read_text())

            selected_ids = {row["sample_id"] for row in selected}
            self.assertEqual(result["kept"], 3)
            self.assertEqual(selected_ids, {"manual-valid", "manual-broad-valid", "auto-keep"})
            self.assertNotIn("cluster-duplicate", selected_ids)
            self.assertEqual(summary["drop_reasons"]["duplicate_cluster"], 1)
            self.assertEqual(summary["drop_reasons"]["audit_noisy"], 1)
            self.assertEqual(summary["drop_reasons"]["generated_summary"], 1)
            self.assertEqual(summary["drop_reasons"]["pr_template"], 1)
            self.assertEqual(summary["drop_reasons"]["broad_gold"], 1)
            self.assertEqual(summary["drop_reasons"]["weak_same_directory"], 1)
            self.assertTrue(summary["quality_gates"]["kept_ge_15"] is False)
            self.assertTrue(audit_out.exists())
            self.assertTrue(audit_csv.exists())

    def test_filter_hard_pool_can_exclude_audited_rows_and_prioritize_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path = root / "pool.jsonl"
            audit_path = root / "audit.jsonl"
            out_path = root / "seed.jsonl"
            summary_path = root / "summary.json"

            def row(sample_id, task_type):
                return {
                    "sample_id": sample_id,
                    "task_type": task_type,
                    "repo": "o/r",
                    "pr_url": f"https://example.test/{sample_id}",
                    "gold_files": [f"tests/{sample_id}.py"],
                    "gold_count": 1,
                    "gold_in_corpus": True,
                    "recommended_split": "hard_curated",
                    "hardness_score": 1.0,
                    "lexical_rank_bucket": "miss@20",
                    "direct_path_hint": False,
                    "basename_hint": False,
                    "module_overlap": False,
                    "same_directory_gold": False,
                    "query_excerpt": "behavior changes when invalid input returns an error",
                    "metrics": {"Recall@5": 0, "Recall@10": 0, "Recall@20": 0, "MRR": 0, "gold_coverage@8k": 0},
                }

            write_jsonl(pool_path, [row("old", "code2test"), row("new-comment", "comment2context"), row("new-code", "code2test")])
            write_jsonl(audit_path, [{"sample_id": "old", "verdict": "valid", "keep": True}])

            result = filter_hard_pool(
                pool_path=pool_path,
                out_path=out_path,
                summary_path=summary_path,
                audit_path=audit_path,
                exclude_audited=True,
                task_priority=["code2test", "comment2context"],
            )
            selected = [json.loads(line) for line in out_path.read_text().splitlines()]
            summary = json.loads(summary_path.read_text())

            self.assertEqual(result["kept"], 2)
            self.assertEqual([row["sample_id"] for row in selected], ["new-code", "new-comment"])
            self.assertEqual(summary["drop_reasons"]["already_audited"], 1)
            self.assertTrue(summary["inputs"]["exclude_audited"])

    def test_summarize_seed_audit_counts_extended_verdicts_and_writes_keep_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / "seed_audit.csv"
            out_path = root / "summary.json"
            keep_list = root / "keep.jsonl"
            audit_path.write_text(
                "\n".join(
                    [
                        "sample_id,task_type,repo,query_excerpt,gold_files,verdict,reason,keep,notes",
                        "valid-code,code2test,o/r,q,g,valid,,true,",
                        "valid-without-keep,code2test,o/r,q,g,valid,,,",
                        "too-easy,comment2context,o/r,q,g,too_easy,,false,",
                        "duplicate,comment2context,o/r,q,g,duplicate,,false,",
                        "not-root,trace2code,o/r,q,g,not_root_cause,,false,",
                        "pending,trace2code,o/r,q,g,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_seed_audit(audit_path, out_path, keep_list)
            kept = [json.loads(line) for line in keep_list.read_text().splitlines()]

            self.assertEqual(summary["total"], 6)
            self.assertEqual(summary["kept"], 1)
            self.assertEqual(summary["dropped"], 3)
            self.assertEqual(summary["pending"], 2)
            self.assertEqual(summary["verdicts"]["too_easy"], 1)
            self.assertEqual(summary["verdicts"]["duplicate"], 1)
            self.assertEqual(summary["verdicts"]["not_root_cause"], 1)
            self.assertEqual(summary["kept_by_task"], {"code2test": 1})
            self.assertEqual(kept[0]["sample_id"], "valid-code")
            self.assertTrue(kept[0]["keep"])

    def test_merge_seed_audits_dedupes_mixed_inputs_and_requires_keep_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_audit = root / "round1.csv"
            jsonl_audit = root / "round2.jsonl"
            out_path = root / "merged_summary.json"
            keep_list = root / "merged_keep.jsonl"
            csv_audit.write_text(
                "\n".join(
                    [
                        "sample_id,task_type,repo,query_excerpt,gold_files,verdict,reason,keep,notes",
                        "valid-code,code2test,o/r,q,g,valid,,true,",
                        "valid-without-keep,code2test,o/r,q,g,valid,,,",
                        "same-duplicate,comment2context,o/r,q,g,valid,,true,",
                        "conflict,comment2context,o/r,q,g,valid,,true,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_jsonl(
                jsonl_audit,
                [
                    {
                        "sample_id": "same-duplicate",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "query_excerpt": "q",
                        "gold_files": "g",
                        "verdict": "valid",
                        "keep": True,
                    },
                    {
                        "sample_id": "json-valid",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "query_excerpt": "q",
                        "gold_files": "g",
                        "verdict": "valid",
                        "keep": True,
                    },
                    {
                        "sample_id": "conflict",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "query_excerpt": "q",
                        "gold_files": "g",
                        "verdict": "noisy",
                        "keep": False,
                    },
                ],
            )

            summary = merge_seed_audits([csv_audit, jsonl_audit], out_path, keep_list)
            kept = [json.loads(line) for line in keep_list.read_text().splitlines()]

            self.assertEqual(summary["total_rows"], 7)
            self.assertEqual(summary["unique_rows"], 5)
            self.assertEqual(summary["duplicate_rows"], 2)
            self.assertEqual(summary["conflict_count"], 1)
            self.assertEqual(summary["kept"], 3)
            self.assertEqual(summary["pending"], 1)
            self.assertEqual([row["sample_id"] for row in kept], ["valid-code", "same-duplicate", "json-valid"])
            self.assertNotIn("conflict", {row["sample_id"] for row in kept})


if __name__ == "__main__":
    unittest.main()
