import json
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from agent_retrieval_bench.v1_1 import (
    assemble_v1_1_benchmark,
    auto_merge_v1_1_baseline_shards,
    baseline_label_matches,
    check_required_baseline_summaries,
    check_v1_1_baseline_status,
    check_v1_1_readiness,
    completion_doc_content_status,
    create_v1_1_baseline_transfer_bundle,
    frozen_v1_fingerprint_status,
    handoff_workflow_evidence_paths,
    report_v1_1_completion_audit,
    report_v1_1_release,
    render_v1_1_baseline_finalization_markdown,
    verify_v1_1_baseline_handoff,
    verify_v1_1_baseline_return_bundle,
    verify_v1_1_baseline_transfer_bundle,
    verify_v1_1_baseline_transfer_manifest,
    write_v1_1_baseline_apply_return_bundle_script,
    write_v1_1_baseline_finalization,
    write_v1_1_baseline_return_acceptance,
    write_v1_1_baseline_return_bundle_script,
    write_v1_1_baseline_return_manifest,
    write_v1_1_baseline_run_script,
    write_v1_1_baseline_shard_commands,
    write_v1_1_baseline_handoff,
    write_v1_1_baseline_transfer_unpack_script,
    write_v1_1_baseline_transfer_manifest,
    write_v1_1_external_runner_copy_packet,
    write_v1_1_external_runner_failfast_smoke_report,
    write_v1_1_external_runner_preflight_report,
    write_v1_1_merged_details,
    write_v1_1_sample_id_shards,
    write_v1_1_summary_from_details,
    write_v1_1_audit_packet,
)


V1_1_DOC_TEXT = """\
V1.1 is a focused benchmark improvement and targeted expansion over V1.
Keep `benchmark/v1` unchanged as frozen V1.
Expand `comment2context` to 80-100 samples and `trace2code` to 100+ samples.
Required open-source baselines include lexical, RepoMap, Jina, and Qwen.
"""


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def sample(sample_id, task_type, query, gold, repo="o/r", base_commit="base"):
    return {
        "id": sample_id,
        "version": 2,
        "task_type": task_type,
        "repo": repo,
        "base_commit": base_commit,
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": base_commit},
        "query": query,
        "gold": {"fix_commit": "fix", **gold},
        "metadata": {"pr_url": f"https://github.com/{repo}/pull/1", "evidence": {"source": "manual_audit"}},
    }


class V11ReadinessTests(unittest.TestCase):
    def test_required_embedding_baseline_labels_match_public_model_ids(self):
        self.assertTrue(baseline_label_matches("jinaai/jina-code-embeddings-0.5b", "jina-code-embeddings-0.5b"))
        self.assertTrue(baseline_label_matches("Qwen/Qwen3-Embedding-4B", "qwen3-embedding-4b"))
        self.assertTrue(baseline_label_matches("voyage-code-3", "voyage-code-3"))
        self.assertFalse(baseline_label_matches("jinaai/jina-code-embeddings-0.5b", "qwen3-embedding-4b"))

    def test_required_baseline_summary_preflight_checks_external_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_samples = 3
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}

            def summary(path, model=None, mode="embedding", evaluated=expected_samples, skipped=None, candidate_filter="all_files"):
                payload = {
                    "mode": mode,
                    "candidate_filter": candidate_filter,
                    "evaluated": evaluated,
                    "skipped": skipped or {},
                    "metrics": {
                        "overall": {"samples": evaluated, **zero_metrics},
                        "code2test": {"samples": evaluated, **zero_metrics},
                    },
                }
                if model:
                    payload["model"] = model
                write_json(path, payload)
                details_stem = path.stem.removesuffix("_summary")
                write_jsonl(
                    path.with_name(f"{details_stem}_details.jsonl"),
                    [
                        {
                            "sample_id": f"sample-{index}",
                            "task_type": "code2test",
                            "candidate_filter": candidate_filter,
                            "metrics": zero_metrics,
                        }
                        for index in range(evaluated)
                    ],
                )

            summary(root / "lexical_summary.json", mode="corpus")
            summary(root / "repomap_summary.json", model="aider-style-repomap", mode="repomap")
            summary(root / "jina-code-embeddings-0.5b_summary.json", model="jinaai/jina-code-embeddings-0.5b")
            summary(root / "qwen3-embedding-4b_summary.json", model="Qwen/Qwen3-Embedding-4B", evaluated=2, skipped={"oom": 1})

            result = check_required_baseline_summaries(sorted(root.glob("*_summary.json")), expected_samples=expected_samples)

            self.assertFalse(result["complete"])
            self.assertEqual(result["blocking_baselines"], ["qwen3-embedding-4b"])
            jina = [item for item in result["required_baselines"] if item["baseline"] == "jina-code-embeddings-0.5b"][0]
            self.assertTrue(jina["complete"])

            summary(root / "qwen3-embedding-4b_summary.json", model="Qwen/Qwen3-Embedding-4B", candidate_filter="changed_files")
            changed_files = check_required_baseline_summaries(sorted(root.glob("*_summary.json")), expected_samples=expected_samples)
            self.assertFalse(changed_files["complete"])
            self.assertEqual(changed_files["blocking_baselines"], ["qwen3-embedding-4b"])

            summary(root / "qwen3-embedding-4b_summary.json", model="Qwen/Qwen3-Embedding-4B")
            complete = check_required_baseline_summaries(sorted(root.glob("*_summary.json")), expected_samples=expected_samples)
            self.assertTrue(complete["complete"])

            stale_summary = json.loads((root / "lexical_summary.json").read_text(encoding="utf-8"))
            stale_summary["metrics"]["overall"]["MRR"] = 0.5
            write_json(root / "lexical_summary.json", stale_summary)
            mismatched_metrics = check_required_baseline_summaries(sorted(root.glob("*_summary.json")), expected_samples=expected_samples)
            self.assertFalse(mismatched_metrics["complete"])
            lexical = [item for item in mismatched_metrics["required_baselines"] if item["baseline"] == "lexical"][0]
            self.assertFalse(lexical["details"]["metrics_match"])
            self.assertEqual(lexical["details"]["metrics_mismatches"][0]["metric"], "MRR")
            summary(root / "lexical_summary.json", mode="corpus")

            mismatched_ids = check_required_baseline_summaries(
                sorted(root.glob("*_summary.json")),
                expected_samples=expected_samples,
                expected_sample_ids={"sample-0", "sample-1", "sample-x"},
            )
            self.assertFalse(mismatched_ids["complete"])
            lexical = [item for item in mismatched_ids["required_baselines"] if item["baseline"] == "lexical"][0]
            self.assertEqual(lexical["details"]["missing_sample_ids"], ["sample-x"])
            self.assertEqual(lexical["details"]["unexpected_sample_ids"], ["sample-2"])

            (root / "qwen3-embedding-4b_details.jsonl").unlink()
            missing_details = check_required_baseline_summaries(sorted(root.glob("*_summary.json")), expected_samples=expected_samples)
            self.assertFalse(missing_details["complete"])
            qwen = [item for item in missing_details["required_baselines"] if item["baseline"] == "qwen3-embedding-4b"][0]
            self.assertFalse(qwen["details"]["complete"])

    def test_v1_1_baseline_status_reports_runtime_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}

            def summary(path, model=None, mode="embedding"):
                payload = {
                    "mode": mode,
                    "candidate_filter": "all_files",
                    "evaluated": 1,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 1, **zero_metrics},
                        "code2test": {"samples": 1, **zero_metrics},
                    },
                }
                if model:
                    payload["model"] = model
                write_json(path, payload)
                write_jsonl(
                    path.with_name(f"{path.stem.removesuffix('_summary')}_details.jsonl"),
                    [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics}],
                )

            summary(root / "lexical_summary.json", mode="corpus")
            summary(root / "repomap_summary.json", model="aider-style-repomap", mode="repomap")
            write_jsonl(
                root / "jina-code-embeddings-0.5b_details.jsonl",
                [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics}],
            )
            runtime = {
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True, "version": "test", "cuda_available": False, "cuda_device_count": 0},
                "nvidia_smi": {"available": False, "path": None, "gpus": [], "error": None},
                "cuda_available": False,
                "voyage_api_key_set": False,
            }

            result = check_v1_1_baseline_status(
                sorted(root.glob("*_summary.json")),
                expected_samples=1,
                expected_sample_ids={"sample-1"},
                runtime_status=runtime,
                eval_dirs=[root],
            )

            self.assertFalse(result["complete"])
            by_baseline = {item["baseline"]: item for item in result["baseline_blockers"]}
            self.assertTrue(by_baseline["lexical"]["complete"])
            self.assertEqual(by_baseline["jina-code-embeddings-0.5b"]["partial_details"]["rows"], 1)
            self.assertTrue(by_baseline["jina-code-embeddings-0.5b"]["partial_details"]["complete_by_rows"])
            self.assertEqual(by_baseline["jina-code-embeddings-0.5b"]["reason"], "missing_summary_from_complete_details")
            self.assertEqual(by_baseline["qwen3-embedding-4b"]["reason"], "blocked_no_cuda_or_precomputed_cache")
            self.assertNotIn("voyage-code-3", by_baseline)

            missing_numpy_runtime = {
                **runtime,
                "numpy": {"installed": False},
                "voyage_api_key_set": True,
            }
            missing_numpy = check_v1_1_baseline_status(
                [],
                expected_samples=1,
                expected_baselines=["voyage-code-3"],
                expected_sample_ids={"sample-1"},
                runtime_status=missing_numpy_runtime,
                eval_dirs=[root],
            )
            self.assertEqual(missing_numpy["baseline_blockers"][0]["reason"], "blocked_missing_numpy")

    def test_baseline_status_uses_preflight_details_path_for_model_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            summary_path = root / "Qwen3-Embedding-4B_summary.json"
            details_path = root / "Qwen3-Embedding-4B_details.jsonl"
            write_json(
                summary_path,
                {
                    "mode": "embedding",
                    "model": "/models/Qwen3-Embedding-4B",
                    "candidate_filter": "all_files",
                    "evaluated": 1,
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": 1, **zero_metrics},
                        "code2test": {"samples": 1, **zero_metrics},
                    },
                },
            )
            write_jsonl(
                details_path,
                [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics}],
            )

            result = check_v1_1_baseline_status(
                [summary_path],
                expected_samples=1,
                expected_baselines=["qwen3-embedding-4b"],
                expected_sample_ids={"sample-1"},
                runtime_status={
                    "numpy": {"installed": True},
                    "sentence_transformers": {"installed": True},
                    "torch": {"installed": True, "version": "test", "cuda_available": False, "cuda_device_count": 0},
                    "nvidia_smi": {"available": False, "path": None, "gpus": [], "error": None},
                    "cuda_available": False,
                    "voyage_api_key_set": False,
                },
                eval_dirs=[root],
            )

            partial = result["partial_details"]["qwen3-embedding-4b"]
            self.assertTrue(result["complete"])
            self.assertEqual(partial["path"], str(details_path))
            self.assertTrue(partial["exists"])
            self.assertTrue(partial["complete_by_rows"])

    def test_v1_1_external_runner_preflight_reports_local_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            completion_json = root / "completion.json"
            transfer_verify = root / "transfer_verify.json"
            handoff_verify = root / "handoff_verify.json"
            bundle_verify = root / "bundle_verify.json"
            full_runner = root / "run_all.sh"
            gpu_runner = root / "run_gpu.sh"
            voyage_runner = root / "run_voyage.sh"
            package_script = root / "package_return.sh"
            gpu_package_script = root / "package_gpu_return.sh"
            voyage_package_script = root / "package_voyage_return.sh"
            for script in [full_runner, gpu_runner, voyage_runner, package_script, gpu_package_script, voyage_package_script]:
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            runtime = {
                "cuda_available": False,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": False,
                "nvidia_smi": {"available": False},
            }
            write_json(baseline_status, {"complete": False, "runtime": runtime, "blocking_baselines": ["voyage-code-3"]})
            write_json(
                return_acceptance,
                {
                    "complete": False,
                    "return_manifest": str(return_manifest),
                    "completion_json": str(completion_json),
                    "return_manifest_generated_at": "2026-05-15T00:00:00+00:00",
                    "completion_audit_generated_at": "2026-05-15T00:00:01+00:00",
                    "current_status": {"completion_audit_overall_status": "not_complete"},
                },
            )
            required_return_files = [
                root / "eval" / "jina-code-embeddings-0.5b_details.jsonl",
                root / "eval" / "jina-code-embeddings-0.5b_summary.json",
                root / "eval" / "qwen3-embedding-4b_details.jsonl",
                root / "eval" / "qwen3-embedding-4b_summary.json",
                root / "eval" / "voyage-code-3_details.jsonl",
                root / "eval" / "voyage-code-3_summary.json",
            ]
            for path in required_return_files[:4]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            write_json(
                return_manifest,
                {
                    "generated_at": "2026-05-15T00:00:00+00:00",
                    "artifacts_complete": False,
                    "required_files": [str(path) for path in required_return_files],
                    "missing_required_files": [str(path) for path in required_return_files[4:]],
                },
            )
            write_json(completion_json, {"generated_at": "2026-05-15T00:00:01+00:00", "overall_status": "not_complete"})
            for path in [transfer_verify, handoff_verify, bundle_verify]:
                write_json(path, {"complete": True, "failed_checks": []})

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
                gpu_return_bundle_script_path=gpu_package_script,
                voyage_return_bundle_script_path=voyage_package_script,
                out_path=root / "preflight.json",
                markdown_out_path=root / "preflight.md",
            )

            self.assertFalse(result["complete"])
            self.assertTrue(result["handoff_ready"])
            self.assertTrue(result["return_acceptance_ready"])
            self.assertFalse(result["local_runner_ready"])
            self.assertFalse(result["return_packaging_ready"])
            self.assertTrue(result["artifact_checks"]["return_acceptance"]["freshness"]["complete"])
            self.assertEqual(result["runner_scripts"]["full"]["expected_blockers"], ["missing_voyage_api_key", "missing_cuda"])
            self.assertEqual(result["runner_scripts"]["gpu"]["expected_blockers"], ["missing_cuda"])
            self.assertEqual(result["runner_scripts"]["voyage"]["expected_blockers"], ["missing_voyage_api_key"])
            self.assertEqual(result["return_packaging"]["expected_blocker"], "missing_required_return_files")
            self.assertTrue(result["return_packaging_scripts"]["gpu"]["ready"])
            self.assertEqual(result["return_packaging_scripts"]["voyage"]["missing_required_file_count"], 2)
            markdown = (root / "preflight.md").read_text(encoding="utf-8")
            self.assertIn("Return acceptance ready: `True`", markdown)
            self.assertIn("Return acceptance complete: `False`", markdown)
            self.assertIn("reported_complete=`False`", markdown)
            self.assertIn("artifact_ready=`True`", markdown)
            self.assertIn("missing_required_return_files", markdown)

    def test_v1_1_external_runner_preflight_flags_stale_return_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            completion_json = root / "completion.json"
            transfer_verify = root / "transfer_verify.json"
            handoff_verify = root / "handoff_verify.json"
            bundle_verify = root / "bundle_verify.json"
            runner = root / "run.sh"
            package_script = root / "package_return.sh"
            for script in [runner, package_script]:
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            runtime = {
                "cuda_available": True,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": True,
                "nvidia_smi": {"available": True},
            }
            write_json(baseline_status, {"complete": True, "runtime": runtime, "blocking_baselines": []})
            write_json(
                return_manifest,
                {
                    "generated_at": "2026-05-15T00:00:00+00:00",
                    "artifacts_complete": True,
                    "required_files": [],
                    "missing_required_files": [],
                },
            )
            write_json(completion_json, {"generated_at": "2026-05-15T00:00:01+00:00", "overall_status": "complete"})
            write_json(
                return_acceptance,
                {
                    "complete": True,
                    "return_manifest": str(return_manifest),
                    "completion_json": str(completion_json),
                    "return_manifest_generated_at": "2026-05-14T00:00:00+00:00",
                    "completion_audit_generated_at": "2026-05-14T00:00:01+00:00",
                },
            )
            for path in [transfer_verify, handoff_verify, bundle_verify]:
                write_json(path, {"complete": True, "failed_checks": []})

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                full_runner_path=runner,
                gpu_runner_path=runner,
                voyage_runner_path=runner,
                return_bundle_script_path=package_script,
                out_path=root / "preflight.json",
                markdown_out_path=root / "preflight.md",
            )

            acceptance_check = result["artifact_checks"]["return_acceptance"]
            self.assertFalse(result["complete"])
            self.assertFalse(result["return_acceptance_ready"])
            self.assertFalse(acceptance_check["complete"])
            self.assertFalse(acceptance_check["freshness"]["complete"])
            self.assertEqual(
                acceptance_check["freshness_mismatches"],
                ["return_manifest_generated_at_mismatch", "completion_audit_generated_at_mismatch"],
            )
            markdown = (root / "preflight.md").read_text(encoding="utf-8")
            self.assertIn("Return acceptance ready: `False`", markdown)
            self.assertIn("freshness_mismatches", markdown)

    def test_v1_1_external_runner_preflight_flags_stale_transfer_manifest_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            completion_json = root / "completion.json"
            transfer_manifest = root / "transfer_manifest.json"
            transfer_verify = root / "transfer_verify.json"
            handoff_verify = root / "handoff_verify.json"
            bundle_verify = root / "bundle_verify.json"
            runner = root / "run.sh"
            package_script = root / "package_return.sh"
            for script in [runner, package_script]:
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            runtime = {
                "cuda_available": True,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": True,
                "nvidia_smi": {"available": True},
            }
            write_json(baseline_status, {"complete": True, "runtime": runtime, "blocking_baselines": []})
            write_json(return_manifest, {"generated_at": "2026-05-15T00:00:00+00:00", "artifacts_complete": True})
            write_json(completion_json, {"generated_at": "2026-05-15T00:00:01+00:00", "overall_status": "complete"})
            write_json(
                return_acceptance,
                {
                    "complete": True,
                    "return_manifest": str(return_manifest),
                    "completion_json": str(completion_json),
                    "return_manifest_generated_at": "2026-05-15T00:00:00+00:00",
                    "completion_audit_generated_at": "2026-05-15T00:00:01+00:00",
                },
            )
            write_json(transfer_manifest, {"generated_at": "2026-05-15T00:01:00+00:00"})
            write_json(
                transfer_verify,
                {
                    "generated_at": "2026-05-15T00:00:30+00:00",
                    "manifest": str(transfer_manifest),
                    "complete": True,
                    "failed_checks": [],
                },
            )
            for path in [handoff_verify, bundle_verify]:
                write_json(path, {"complete": True, "failed_checks": []})

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                full_runner_path=runner,
                gpu_runner_path=runner,
                voyage_runner_path=runner,
                return_bundle_script_path=package_script,
                out_path=root / "preflight.json",
                markdown_out_path=root / "preflight.md",
            )

            transfer_check = result["artifact_checks"]["transfer_manifest_verify"]
            self.assertFalse(result["complete"])
            self.assertFalse(result["handoff_ready"])
            self.assertFalse(transfer_check["complete"])
            self.assertFalse(transfer_check["freshness"]["complete"])
            self.assertEqual(transfer_check["freshness_mismatches"], ["transfer_manifest_verify_stale"])
            self.assertIn("transfer_manifest_verify_freshness", transfer_check["failed_checks"])
            markdown = (root / "preflight.md").read_text(encoding="utf-8")
            self.assertIn("freshness_mismatches", markdown)
            self.assertIn("transfer_manifest_verify_stale", markdown)

    def test_v1_1_external_runner_preflight_flags_stale_handoff_and_bundle_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            completion_json = root / "completion.json"
            handoff = root / "handoff.json"
            transfer_manifest = root / "transfer_manifest.json"
            bundle_file = root / "transfer_bundle.tar.zst"
            transfer_verify = root / "transfer_verify.json"
            handoff_verify = root / "handoff_verify.json"
            bundle_verify = root / "bundle_verify.json"
            runner = root / "run.sh"
            package_script = root / "package_return.sh"
            for script in [runner, package_script]:
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            runtime = {
                "cuda_available": True,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": True,
                "nvidia_smi": {"available": True},
            }
            write_json(baseline_status, {"complete": True, "runtime": runtime, "blocking_baselines": []})
            write_json(return_manifest, {"generated_at": "2026-05-15T00:00:00+00:00", "artifacts_complete": True})
            write_json(completion_json, {"generated_at": "2026-05-15T00:00:01+00:00", "overall_status": "complete"})
            write_json(
                return_acceptance,
                {
                    "complete": True,
                    "return_manifest": str(return_manifest),
                    "completion_json": str(completion_json),
                    "return_manifest_generated_at": "2026-05-15T00:00:00+00:00",
                    "completion_audit_generated_at": "2026-05-15T00:00:01+00:00",
                },
            )
            write_json(handoff, {"generated_at": "2026-05-15T00:02:00+00:00"})
            write_json(transfer_manifest, {"generated_at": "2026-05-15T00:01:00+00:00"})
            bundle_file.write_bytes(b"new bundle bytes")
            write_json(
                transfer_verify,
                {
                    "generated_at": "2026-05-15T00:02:00+00:00",
                    "manifest": str(transfer_manifest),
                    "complete": True,
                    "failed_checks": [],
                },
            )
            write_json(
                handoff_verify,
                {
                    "generated_at": "2026-05-15T00:01:00+00:00",
                    "handoff": str(handoff),
                    "complete": True,
                    "failed_checks": [],
                },
            )
            write_json(
                bundle_verify,
                {
                    "generated_at": "2026-05-15T00:00:30+00:00",
                    "manifest": str(transfer_manifest),
                    "bundle": str(bundle_file),
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                    "complete": True,
                    "failed_checks": [],
                },
            )

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                full_runner_path=runner,
                gpu_runner_path=runner,
                voyage_runner_path=runner,
                return_bundle_script_path=package_script,
                out_path=root / "preflight.json",
                markdown_out_path=root / "preflight.md",
            )

            handoff_check = result["artifact_checks"]["handoff_verify"]
            bundle_check = result["artifact_checks"]["transfer_bundle_verify"]
            self.assertFalse(result["complete"])
            self.assertFalse(result["handoff_ready"])
            self.assertEqual(handoff_check["freshness_mismatches"], ["handoff_verify_stale"])
            self.assertEqual(
                bundle_check["freshness_mismatches"],
                ["transfer_bundle_verify_stale", "transfer_bundle_size_mismatch", "transfer_bundle_sha256_mismatch"],
            )
            self.assertIn("handoff_verify_freshness", handoff_check["failed_checks"])
            self.assertIn("transfer_bundle_verify_freshness", bundle_check["failed_checks"])
            markdown = (root / "preflight.md").read_text(encoding="utf-8")
            self.assertIn("handoff_verify_stale", markdown)
            self.assertIn("transfer_bundle_size_mismatch", markdown)

    def test_v1_1_external_runner_preflight_reports_dependency_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            runner = root / "run.sh"
            runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            runner.chmod(0o755)
            write_json(
                baseline_status,
                {
                    "complete": False,
                    "runtime": {
                        "cuda_available": False,
                        "numpy": {"installed": False},
                        "sentence_transformers": {"installed": False},
                        "torch": {"installed": True},
                        "voyage_api_key_set": False,
                    },
                    "blocking_baselines": ["jina-code-embeddings-0.5b", "voyage-code-3"],
                },
            )
            write_json(return_acceptance, {"complete": False})
            write_json(return_manifest, {"artifacts_complete": False, "missing_required_files": []})

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                full_runner_path=runner,
                gpu_runner_path=runner,
                voyage_runner_path=runner,
            )

            self.assertEqual(
                result["runner_scripts"]["full"]["expected_blockers"],
                ["missing_voyage_api_key", "missing_dependency", "missing_cuda"],
            )
            self.assertEqual(result["runner_scripts"]["gpu"]["expected_blockers"], ["missing_dependency", "missing_cuda"])
            self.assertEqual(result["runner_scripts"]["voyage"]["expected_blockers"], ["missing_voyage_api_key", "missing_dependency"])
            self.assertEqual(result["runtime"]["numpy_installed"], False)
            self.assertEqual(result["runtime"]["sentence_transformers_installed"], False)
            self.assertEqual(result["runtime"]["torch_installed"], True)
            self.assertEqual(result["runner_scripts"]["full"]["missing_dependencies"], ["numpy", "sentence_transformers"])
            self.assertEqual(result["runner_scripts"]["gpu"]["missing_dependencies"], ["numpy", "sentence_transformers"])
            self.assertEqual(result["runner_scripts"]["voyage"]["missing_dependencies"], ["numpy"])

    def test_v1_1_external_runner_preflight_accepts_unpacked_smoke_verifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "data" / "reports" / "v1_1"
            eval_dir = root / "data" / "eval" / "v1_1"
            baseline_status = report_dir / "baseline_status_v19.json"
            return_acceptance = report_dir / "baseline_return_acceptance_v19.json"
            return_manifest = report_dir / "baseline_return_manifest_v19.json"
            completion_json = report_dir / "completion_audit_v19.json"
            transfer_manifest = report_dir / "baseline_transfer_manifest_v19.json"
            transfer_verify = report_dir / "baseline_transfer_manifest_verify_v19.json"
            transfer_unpack_smoke = report_dir / "baseline_transfer_unpack_smoke_v19.json"
            handoff = report_dir / "baseline_handoff_v19.json"
            handoff_verify = report_dir / "baseline_handoff_verify_v19.json"
            handoff_unpack_smoke = report_dir / "baseline_handoff_unpack_smoke_v19.json"
            bundle_verify = report_dir / "baseline_transfer_bundle_verify_v19.json"
            full_runner = report_dir / "run_v19_baseline_shards.sh"
            gpu_runner = report_dir / "run_v19_gpu_baseline_shards.sh"
            voyage_runner = report_dir / "run_v19_voyage_baseline_shards.sh"
            package_script = report_dir / "package_v19_return_artifacts.sh"
            gpu_package_script = report_dir / "package_v19_gpu_return_artifacts.sh"
            voyage_package_script = report_dir / "package_v19_voyage_return_artifacts.sh"
            for script in [full_runner, gpu_runner, voyage_runner, package_script, gpu_package_script, voyage_package_script]:
                script.parent.mkdir(parents=True, exist_ok=True)
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            required_return_files = [
                eval_dir / "jina-code-embeddings-0.5b_details.jsonl",
                eval_dir / "jina-code-embeddings-0.5b_summary.json",
                eval_dir / "qwen3-embedding-4b_details.jsonl",
                eval_dir / "qwen3-embedding-4b_summary.json",
                eval_dir / "voyage-code-3_details.jsonl",
                eval_dir / "voyage-code-3_summary.json",
            ]
            write_json(
                return_manifest,
                {
                    "generated_at": "2026-05-15T00:00:00+00:00",
                    "artifacts_complete": False,
                    "required_files": [str(path) for path in required_return_files],
                    "missing_required_files": [str(path) for path in required_return_files],
                },
            )
            write_json(
                return_acceptance,
                {
                    "complete": False,
                    "return_manifest": str(return_manifest),
                    "completion_json": str(completion_json),
                    "return_manifest_generated_at": "2026-05-15T00:00:00+00:00",
                    "completion_audit_generated_at": "2026-05-15T00:00:02+00:00",
                    "current_status": {"completion_audit_overall_status": "not_complete"},
                },
            )
            write_json(completion_json, {"generated_at": "2026-05-15T00:00:02+00:00", "overall_status": "not_complete"})
            write_json(transfer_manifest, {"generated_at": "2026-05-15T00:00:00+00:00"})
            write_json(handoff, {"generated_at": "2026-05-15T00:00:00+00:00"})
            write_json(
                transfer_unpack_smoke,
                {
                    "generated_at": "2026-05-15T00:00:01+00:00",
                    "manifest": str(transfer_manifest),
                    "complete": True,
                    "failed_checks": [],
                },
            )
            write_json(
                handoff_unpack_smoke,
                {
                    "generated_at": "2026-05-15T00:00:01+00:00",
                    "handoff": str(handoff),
                    "complete": True,
                    "failed_checks": [],
                },
            )
            runtime = {
                "cuda_available": False,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": False,
                "nvidia_smi": {"available": False},
            }

            with mock.patch("agent_retrieval_bench.v1_1.embedding_runtime_status", return_value=runtime):
                result = write_v1_1_external_runner_preflight_report(
                    baseline_status_path=baseline_status,
                    return_acceptance_path=return_acceptance,
                    return_manifest_path=return_manifest,
                    transfer_manifest_verify_path=transfer_verify,
                    handoff_verify_path=handoff_verify,
                    transfer_bundle_verify_path=bundle_verify,
                    full_runner_path=full_runner,
                    gpu_runner_path=gpu_runner,
                    voyage_runner_path=voyage_runner,
                    return_bundle_script_path=package_script,
                    gpu_return_bundle_script_path=gpu_package_script,
                    voyage_return_bundle_script_path=voyage_package_script,
                    out_path=report_dir / "external_runner_preflight_v19.json",
                    markdown_out_path=report_dir / "external_runner_preflight_v19.md",
                )

            self.assertFalse(result["complete"])
            self.assertTrue(result["handoff_ready"])
            self.assertTrue(result["return_acceptance_ready"])
            self.assertFalse(result["return_acceptance_complete"])
            self.assertEqual(result["baseline_status_source"], "live_runtime_fallback")
            self.assertEqual(
                result["blocking_baselines"],
                ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"],
            )
            transfer_check = result["artifact_checks"]["transfer_manifest_verify"]
            handoff_check = result["artifact_checks"]["handoff_verify"]
            bundle_check = result["artifact_checks"]["transfer_bundle_verify"]
            return_acceptance_check = result["artifact_checks"]["return_acceptance"]
            self.assertTrue(transfer_check["complete"])
            self.assertTrue(handoff_check["complete"])
            self.assertTrue(return_acceptance_check["freshness"]["complete"])
            self.assertEqual(return_acceptance_check["freshness_mismatches"], [])
            self.assertTrue(transfer_check["used_fallback"])
            self.assertTrue(handoff_check["used_fallback"])
            self.assertEqual(transfer_check["primary_path"], str(transfer_verify))
            self.assertEqual(handoff_check["primary_path"], str(handoff_verify))
            self.assertFalse(bundle_check["exists"])
            self.assertFalse(bundle_check["required"])
            self.assertTrue(bundle_check["complete"])
            self.assertEqual(result["runner_scripts"]["full"]["expected_blockers"], ["missing_voyage_api_key", "missing_cuda"])
            markdown = (report_dir / "external_runner_preflight_v19.md").read_text(encoding="utf-8")
            self.assertIn("Baseline status source: `live_runtime_fallback`", markdown)

    def test_v1_1_external_runner_preflight_checks_copy_packet_bundle_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_status = root / "baseline_status.json"
            return_acceptance = root / "acceptance.json"
            return_manifest = root / "return_manifest.json"
            transfer_verify = root / "transfer_verify.json"
            handoff_verify = root / "handoff_verify.json"
            bundle_verify = root / "bundle_verify.json"
            bundle_report = root / "transfer_bundle.json"
            copy_packet = root / "copy_packet.json"
            bundle_file = root / "transfer_bundle.tar.zst"
            checksum_file = root / "transfer_bundle.tar.zst.sha256"
            unpack_script = root / "unpack_v19_transfer_bundle.sh"
            full_runner = root / "run_all.sh"
            gpu_runner = root / "run_gpu.sh"
            voyage_runner = root / "run_voyage.sh"
            bundle_file.write_bytes(b"transfer bundle")
            bundle_sha256 = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
            checksum_file.write_text(f"{bundle_sha256}  {bundle_file}\n", encoding="utf-8")
            unpack_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            package_script = root / "package_return.sh"
            for script in [full_runner, gpu_runner, voyage_runner, package_script]:
                script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                script.chmod(0o755)
            runtime = {
                "cuda_available": True,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": True,
                "nvidia_smi": {"available": True},
            }
            write_json(baseline_status, {"complete": True, "runtime": runtime, "blocking_baselines": []})
            write_json(return_acceptance, {"complete": True})
            write_json(return_manifest, {"artifacts_complete": True, "missing_required_files": []})
            for path in [transfer_verify, handoff_verify]:
                write_json(path, {"complete": True, "failed_checks": []})
            write_json(
                bundle_verify,
                {
                    "complete": True,
                    "sha256": bundle_sha256,
                    "size_bytes": bundle_file.stat().st_size,
                    "bundle_fingerprint": {"sha256": bundle_sha256, "size_bytes": bundle_file.stat().st_size},
                    "failed_checks": [],
                    "checks": [
                        {"name": "bundle_excludes_external_runner_copy_packet", "status": "pass"},
                    ],
                },
            )
            write_json(
                bundle_report,
                {
                    "generated_at": "2026-05-15T00:00:00+00:00",
                    "complete": True,
                    "sha256": bundle_sha256,
                    "size_bytes": bundle_file.stat().st_size,
                    "file_count": 7,
                    "failed_checks": [],
                    "checks": [
                        {"name": "manifest_excludes_external_runner_copy_packet", "status": "pass"},
                    ],
                    "verification": {
                        "checks": [
                            {"name": "bundle_excludes_external_runner_copy_packet", "status": "pass"},
                        ],
                    },
                    "external_acceptance": {
                        "external_baselines": ["voyage-code-3"],
                        "required_return_file_count": 2,
                        "required_return_files": [
                            "eval/details.jsonl",
                            "eval/summary.json",
                        ],
                        "required_return_files_by_baseline": {
                            "voyage-code-3": ["eval/details.jsonl", "eval/summary.json"],
                        },
                    },
                },
            )
            copy_packet_payload = {
                "generated_at": "2026-05-15T00:00:01+00:00",
                "complete": True,
                "transfer_bundle_report": str(bundle_report),
                "bundle_generated_at": "2026-05-15T00:00:00+00:00",
                "bundle_path": str(bundle_file),
                "checksum_path": str(checksum_file),
                "unpack_script_path": str(unpack_script),
                "bundle_sha256": bundle_sha256,
                "bundle_size_bytes": bundle_file.stat().st_size,
                "transfer_file_count": 7,
                "copy_to_external_runner": [str(unpack_script), str(bundle_file), str(checksum_file)],
                "required_return_file_count": 2,
                "required_return_files": [
                    "eval/details.jsonl",
                    "eval/summary.json",
                ],
                "required_return_files_by_baseline": {
                    "voyage-code-3": [
                        "eval/details.jsonl",
                        "eval/summary.json",
                    ]
                },
                "failed_checks": [],
            }
            write_json(copy_packet, copy_packet_payload)

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )

            self.assertTrue(result["complete"])
            self.assertTrue(result["handoff_ready"])
            self.assertTrue(result["copy_packet_ready"])
            self.assertTrue(result["artifact_checks"]["copy_packet"]["matches_transfer_bundle"])
            self.assertEqual(
                result["artifact_checks"]["copy_packet"]["transfer_bundle_report_manifest_copy_packet_exclusion"]["status"],
                "pass",
            )
            self.assertEqual(
                result["artifact_checks"]["copy_packet"]["comparison"]["transfer_bundle_report"][
                    "expected_required_return_file_count"
                ],
                2,
            )
            self.assertEqual(
                result["artifact_checks"]["copy_packet"]["comparison"]["transfer_bundle_report"][
                    "actual_required_return_files"
                ],
                ["eval/details.jsonl", "eval/summary.json"],
            )
            self.assertEqual(
                result["artifact_checks"]["copy_packet"]["comparison"]["copy_packet_bundle_file"]["actual_bundle_sha256"],
                bundle_sha256,
            )
            self.assertEqual(result["artifact_checks"]["copy_packet"]["checksum_file_sha256_value"], bundle_sha256)

            bundle_verify_payload = json.loads(bundle_verify.read_text(encoding="utf-8"))
            bundle_verify.unlink()
            missing_verify_result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )
            self.assertFalse(missing_verify_result["handoff_ready"])
            self.assertFalse(missing_verify_result["copy_packet_ready"])
            self.assertTrue(missing_verify_result["artifact_checks"]["transfer_bundle_verify"]["required"])
            self.assertFalse(missing_verify_result["artifact_checks"]["transfer_bundle_verify"]["complete"])
            self.assertIn(
                "missing_transfer_bundle_verify",
                missing_verify_result["artifact_checks"]["copy_packet"]["mismatches"],
            )
            write_json(bundle_verify, bundle_verify_payload)

            write_json(copy_packet, {**copy_packet_payload, "bundle_sha256": "b" * 64})
            stale_result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )

            copy_packet_status = stale_result["artifact_checks"]["copy_packet"]
            self.assertFalse(stale_result["complete"])
            self.assertFalse(stale_result["handoff_ready"])
            self.assertFalse(stale_result["copy_packet_ready"])
            self.assertFalse(copy_packet_status["matches_transfer_bundle"])
            self.assertIn("bundle_sha256_mismatch", copy_packet_status["mismatches"])
            self.assertIn("bundle_report_sha256_mismatch", copy_packet_status["mismatches"])
            self.assertIn("copy_packet_bundle_file_sha256_mismatch", copy_packet_status["mismatches"])
            self.assertIn("copy_packet_checksum_sha256_mismatch", copy_packet_status["mismatches"])

            write_json(
                copy_packet,
                {
                    **copy_packet_payload,
                    "required_return_files": ["eval/voyage-code-3_summary.json"],
                    "required_return_files_by_baseline": {"voyage-code-3": ["eval/voyage-code-3_summary.json"]},
                },
            )
            stale_returns_result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )
            stale_returns_status = stale_returns_result["artifact_checks"]["copy_packet"]
            self.assertFalse(stale_returns_result["complete"])
            self.assertFalse(stale_returns_result["copy_packet_ready"])
            self.assertIn("required_return_files_mismatch", stale_returns_status["mismatches"])
            self.assertIn("required_return_files_by_baseline_mismatch", stale_returns_status["mismatches"])

            write_json(
                copy_packet,
                {
                    **copy_packet_payload,
                    "checksum_path": str(root / "missing.sha256"),
                    "copy_to_external_runner": [str(unpack_script), str(bundle_file), str(root / "missing.sha256")],
                },
            )
            missing_copy_file_result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )
            missing_copy_file_status = missing_copy_file_result["artifact_checks"]["copy_packet"]
            self.assertFalse(missing_copy_file_result["complete"])
            self.assertFalse(missing_copy_file_result["copy_packet_ready"])
            self.assertIn("copy_packet_copy_file_missing", missing_copy_file_status["mismatches"])
            self.assertIn("copy_packet_checksum_file_missing", missing_copy_file_status["mismatches"])

            stale_report = json.loads(bundle_report.read_text(encoding="utf-8"))
            stale_report["checks"] = []
            write_json(bundle_report, stale_report)
            write_json(copy_packet, copy_packet_payload)
            missing_check_result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                transfer_bundle_verify_path=bundle_verify,
                copy_packet_path=copy_packet,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
            )
            missing_check_status = missing_check_result["artifact_checks"]["copy_packet"]
            self.assertFalse(missing_check_result["complete"])
            self.assertIn("transfer_bundle_report_missing_copy_packet_exclusion_check", missing_check_status["mismatches"])

    def test_v1_1_external_runner_failfast_smoke_runs_only_expected_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_runner = root / "run_all.sh"
            gpu_runner = root / "run_gpu.sh"
            voyage_runner = root / "run_voyage.sh"
            package_script = root / "package_return.sh"
            gpu_package_script = root / "package_gpu_return.sh"
            voyage_package_script = root / "package_voyage_return.sh"
            full_runner.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${VOYAGE_API_KEY:-}\" = 'ARB_FAILFAST_PLACEHOLDER' ]; then\n"
                "  echo 'CUDA is not available to torch; run Jina/Qwen shards on a CUDA-capable machine.' >&2\n"
                "  exit 1\n"
                "fi\n"
                "echo 'VOYAGE_API_KEY: Set VOYAGE_API_KEY' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            gpu_runner.write_text(
                "#!/usr/bin/env bash\necho 'CUDA is not available to torch; run Jina/Qwen shards on a CUDA-capable machine.' >&2\nexit 1\n",
                encoding="utf-8",
            )
            voyage_runner.write_text("#!/usr/bin/env bash\necho 'VOYAGE_API_KEY: Set VOYAGE_API_KEY' >&2\nexit 1\n", encoding="utf-8")
            package_script.write_text("#!/usr/bin/env bash\necho '\"missing_required_files\": 6'\nexit 1\n", encoding="utf-8")
            gpu_package_script.write_text("#!/usr/bin/env bash\necho 'Missing required return artifact: gpu'\nexit 1\n", encoding="utf-8")
            voyage_package_script.write_text("#!/usr/bin/env bash\necho 'Missing required return artifact: voyage'\nexit 1\n", encoding="utf-8")
            for script in [full_runner, gpu_runner, voyage_runner, package_script, gpu_package_script, voyage_package_script]:
                script.chmod(0o755)
            preflight = root / "preflight.json"
            write_json(
                preflight,
                {
                    "runner_scripts": {
                        "full": {"expected_blockers": ["missing_voyage_api_key", "missing_cuda"]},
                        "gpu": {"expected_blockers": ["missing_cuda"]},
                        "voyage": {"expected_blockers": ["missing_voyage_api_key"]},
                    },
                    "return_packaging": {"expected_blocker": "missing_required_return_files"},
                    "runtime": {"cuda_available": False},
                },
            )

            result = write_v1_1_external_runner_failfast_smoke_report(
                preflight_path=preflight,
                full_runner_path=full_runner,
                gpu_runner_path=gpu_runner,
                voyage_runner_path=voyage_runner,
                return_bundle_script_path=package_script,
                gpu_return_bundle_script_path=gpu_package_script,
                voyage_return_bundle_script_path=voyage_package_script,
                out_path=root / "smoke.json",
                markdown_out_path=root / "smoke.md",
                cwd=root,
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["steps"]["full"]["status"], "blocked_as_expected")
            self.assertEqual(result["steps"]["full"]["observed_blockers"], ["missing_voyage_api_key"])
            self.assertEqual(result["steps"]["full_cuda_probe"]["status"], "blocked_as_expected")
            self.assertEqual(result["steps"]["full_cuda_probe"]["observed_blockers"], ["missing_cuda"])
            self.assertEqual(result["steps"]["full_cuda_probe"]["env_overrides"], ["VOYAGE_API_KEY"])
            self.assertFalse(result["steps"]["gpu"]["long_run_started"])
            self.assertEqual(result["steps"]["return_packaging"]["observed_blockers"], ["missing_required_return_files"])
            self.assertEqual(result["steps"]["gpu_return_packaging"]["observed_blockers"], ["missing_required_return_files"])
            self.assertEqual(result["steps"]["voyage_return_packaging"]["observed_blockers"], ["missing_required_return_files"])
            self.assertIn("blocked_as_expected", (root / "smoke.md").read_text(encoding="utf-8"))

    def test_v1_1_baseline_status_reports_shard_details_ready_to_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            shard_zero = root / "eval" / "jina-code-embeddings-0.5b_shard00_details.jsonl"
            shard_one = root / "eval" / "jina-code-embeddings-0.5b_shard01_details.jsonl"
            write_jsonl(
                shard_zero,
                [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics}],
            )
            write_jsonl(
                shard_one,
                [{"sample_id": "sample-2", "task_type": "trace2code", "candidate_filter": "all_files", "metrics": zero_metrics}],
            )
            shard_commands = root / "commands.json"
            write_json(
                shard_commands,
                {
                    "baselines": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "shard_commands": [
                                {"index": 0, "artifacts": {"details": str(shard_zero)}},
                                {"index": 1, "artifacts": {"details": str(shard_one)}},
                            ],
                        }
                    ]
                },
            )
            runtime = {
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True, "version": "test", "cuda_available": False, "cuda_device_count": 0},
                "nvidia_smi": {"available": False, "path": None, "gpus": [], "error": None},
                "cuda_available": False,
                "voyage_api_key_set": False,
            }

            result = check_v1_1_baseline_status(
                [],
                expected_samples=2,
                expected_baselines=["jina-code-embeddings-0.5b"],
                expected_sample_ids={"sample-1", "sample-2"},
                runtime_status=runtime,
                eval_dirs=[root / "eval"],
                shard_commands_path=shard_commands,
            )

            blocker = result["baseline_blockers"][0]
            self.assertFalse(result["complete"])
            self.assertEqual(blocker["reason"], "ready_to_merge_shards")
            self.assertEqual(blocker["shard_details"]["rows"], 2)
            self.assertTrue(blocker["shard_details"]["complete_by_rows"])

    def test_v1_1_summary_from_details_recovers_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details_path = root / "jina-code-embeddings-0.5b_details.jsonl"
            out_path = root / "jina-code-embeddings-0.5b_summary.json"
            metrics_one = {"Recall@5": 1.0, "Recall@10": 1.0, "Recall@20": 1.0, "MRR": 1.0, "gold_coverage@8k": 1.0}
            metrics_zero = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            write_jsonl(
                details_path,
                [
                    {"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": metrics_one},
                    {"sample_id": "sample-2", "task_type": "trace2code", "candidate_filter": "all_files", "metrics": metrics_zero},
                ],
            )

            result = write_v1_1_summary_from_details(
                details_path=details_path,
                out_path=out_path,
                model="jinaai/jina-code-embeddings-0.5b",
                expected_samples=2,
                expected_sample_ids={"sample-1", "sample-2"},
            )

            summary = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(summary["candidate_filter"], "all_files")
            self.assertEqual(summary["model"], "jinaai/jina-code-embeddings-0.5b")
            self.assertEqual(summary["skipped"], {})
            self.assertEqual(summary["metrics"]["overall"]["samples"], 2)
            self.assertEqual(summary["metrics"]["overall"]["MRR"], 0.5)

    def test_v1_1_merge_details_validates_and_writes_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_one = {"Recall@5": 1.0, "Recall@10": 1.0, "Recall@20": 1.0, "MRR": 1.0, "gold_coverage@8k": 1.0}
            metrics_zero = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            shard_one = root / "shard-1.jsonl"
            shard_zero = root / "shard-0.jsonl"
            out_path = root / "merged.jsonl"
            write_jsonl(
                shard_one,
                [{"sample_id": "sample-2", "task_type": "trace2code", "candidate_filter": "all_files", "metrics": metrics_zero}],
            )
            write_jsonl(
                shard_zero,
                [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": metrics_one}],
            )

            result = write_v1_1_merged_details(
                details_paths=[shard_one, shard_zero],
                out_path=out_path,
                expected_samples=2,
                expected_sample_ids={"sample-1", "sample-2"},
                candidate_filter="all_files",
            )

            merged_rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(result["complete"])
            self.assertTrue(result["wrote_output"])
            self.assertEqual([row["sample_id"] for row in merged_rows], ["sample-1", "sample-2"])

    def test_v1_1_merge_details_blocks_duplicate_sample_ids_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            shard_one = root / "shard-1.jsonl"
            shard_two = root / "shard-2.jsonl"
            out_path = root / "merged.jsonl"
            duplicate_row = {"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": metrics}
            write_jsonl(shard_one, [duplicate_row])
            write_jsonl(shard_two, [duplicate_row])

            result = write_v1_1_merged_details(
                details_paths=[shard_one, shard_two],
                out_path=out_path,
                expected_samples=2,
                expected_sample_ids={"sample-1", "sample-2"},
                candidate_filter="all_files",
            )

            self.assertFalse(result["complete"])
            self.assertFalse(result["wrote_output"])
            self.assertFalse(out_path.exists())
            self.assertEqual(result["duplicate_sample_ids"], ["sample-1"])
            self.assertEqual(result["missing_sample_ids"], ["sample-2"])

    def test_v1_1_auto_merge_baseline_shards_writes_final_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_one = {"Recall@5": 1.0, "Recall@10": 1.0, "Recall@20": 1.0, "MRR": 1.0, "gold_coverage@8k": 1.0}
            metrics_zero = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            shard_zero = root / "eval" / "jina_shard00_details.jsonl"
            shard_one = root / "eval" / "jina_shard01_details.jsonl"
            final_details = root / "eval" / "jina-code-embeddings-0.5b_details.jsonl"
            final_summary = root / "eval" / "jina-code-embeddings-0.5b_summary.json"
            write_jsonl(
                shard_zero,
                [{"sample_id": "sample-1", "task_type": "code2test", "candidate_filter": "all_files", "metrics": metrics_one}],
            )
            write_jsonl(
                shard_one,
                [{"sample_id": "sample-2", "task_type": "trace2code", "candidate_filter": "all_files", "metrics": metrics_zero}],
            )
            shard_commands = root / "commands.json"
            write_json(
                shard_commands,
                {
                    "baselines": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "model": "jinaai/jina-code-embeddings-0.5b",
                            "shard_commands": [
                                {"index": 0, "artifacts": {"details": str(shard_zero)}},
                                {"index": 1, "artifacts": {"details": str(shard_one)}},
                            ],
                            "final_artifacts": {
                                "summary": str(final_summary),
                                "details": str(final_details),
                                "merge_report": str(root / "reports" / "jina_merge.json"),
                                "merge_markdown": str(root / "reports" / "jina_merge.md"),
                            },
                        }
                    ]
                },
            )

            result = auto_merge_v1_1_baseline_shards(
                shard_commands_path=shard_commands,
                expected_samples=2,
                expected_sample_ids={"sample-1", "sample-2"},
            )

            summary = json.loads(final_summary.read_text(encoding="utf-8"))
            merged_rows = [json.loads(line) for line in final_details.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(result["complete"])
            self.assertEqual(result["merged"], ["jina-code-embeddings-0.5b"])
            self.assertEqual(summary["evaluated"], 2)
            self.assertEqual(summary["model"], "jinaai/jina-code-embeddings-0.5b")
            self.assertEqual(summary["metrics"]["overall"]["MRR"], 0.5)
            self.assertEqual([row["sample_id"] for row in merged_rows], ["sample-1", "sample-2"])
            self.assertTrue((root / "reports" / "jina_merge.json").exists())

    def test_v1_1_auto_merge_baseline_shards_reports_missing_shard_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_zero = root / "eval" / "jina_shard00_details.jsonl"
            shard_one = root / "eval" / "jina_shard01_details.jsonl"
            final_details = root / "eval" / "jina-code-embeddings-0.5b_details.jsonl"
            final_summary = root / "eval" / "jina-code-embeddings-0.5b_summary.json"
            merge_report = root / "reports" / "jina_merge.json"
            shard_commands = root / "commands.json"
            write_json(
                shard_commands,
                {
                    "baselines": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "model": "jinaai/jina-code-embeddings-0.5b",
                            "shard_commands": [
                                {"index": 0, "artifacts": {"details": str(shard_zero)}},
                                {"index": 1, "artifacts": {"details": str(shard_one)}},
                            ],
                            "final_artifacts": {
                                "summary": str(final_summary),
                                "details": str(final_details),
                                "merge_report": str(merge_report),
                                "merge_markdown": str(root / "reports" / "jina_merge.md"),
                            },
                        }
                    ]
                },
            )

            result = auto_merge_v1_1_baseline_shards(
                shard_commands_path=shard_commands,
                expected_samples=2,
                expected_sample_ids={"sample-1", "sample-2"},
            )

            self.assertFalse(result["complete"])
            self.assertEqual(result["blocked"][0]["reason"], "missing_shard_details")
            self.assertEqual(result["blocked"][0]["missing_shard_detail_count"], 2)
            self.assertFalse(final_details.exists())
            self.assertFalse(final_summary.exists())
            self.assertEqual(json.loads(merge_report.read_text(encoding="utf-8"))["missing_shard_details"], [str(shard_zero), str(shard_one)])

    def test_v1_1_finalization_markdown_summarizes_auto_merge_blockers(self):
        markdown = render_v1_1_baseline_finalization_markdown(
            {
                "generated_at": "now",
                "complete": False,
                "next_required_action": "Run missing external baseline artifacts.",
                "handoff": "handoff.json",
                "paths": {},
                "steps": {
                    "auto_merge_shards": {
                        "complete": False,
                        "attempted": ["jina-code-embeddings-0.5b"],
                        "merged": [],
                        "blocked": [
                            {
                                "baseline": "jina-code-embeddings-0.5b",
                                "reason": "missing_shard_details",
                                "missing_shard_detail_count": 4,
                                "merge": {"missing_shard_details": ["a", "b", "c", "d"]},
                            }
                        ],
                    },
                    "return_manifest": {
                        "complete": True,
                        "artifacts_complete": False,
                        "missing_required_files": ["details.jsonl", "summary.json"],
                    },
                },
            }
        )

        self.assertIn("blocked=1; jina-code-embeddings-0.5b:missing_shard_details(4)", markdown)
        self.assertIn("missing_required_files=2", markdown)
        self.assertIn("## Next Action", markdown)
        self.assertIn("Run missing external baseline artifacts.", markdown)
        self.assertNotIn('"missing_shard_details": ["a", "b", "c", "d"]', markdown)

    def test_v1_1_sample_id_shards_match_embedding_modulo_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            write_jsonl(
                derived / "samples.jsonl",
                [{"id": f"sample-{index}", "task_type": "code2test"} for index in range(1, 6)],
            )

            result = write_v1_1_sample_id_shards(
                derived=derived,
                out_dir=root / "shards",
                shard_count=2,
                manifest_out_path=root / "shards" / "manifest.json",
                markdown_out_path=root / "shards" / "manifest.md",
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["sample_count"], 5)
            shard_zero = (root / "shards" / "sample_ids_shard00.txt").read_text(encoding="utf-8").splitlines()
            shard_one = (root / "shards" / "sample_ids_shard01.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(shard_zero, ["sample-1", "sample-3", "sample-5"])
            self.assertEqual(shard_one, ["sample-2", "sample-4"])
            self.assertTrue((root / "shards" / "manifest.json").exists())
            self.assertTrue((root / "shards" / "manifest.md").exists())

    def test_v1_1_sample_id_shards_can_keep_corpora_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            corpus = root / "corpus" / "v1_1"
            write_jsonl(
                derived / "samples.jsonl",
                [
                    {"id": "sample-a", "task_type": "code2test", "repo": "org/big", "base_commit": "aaa"},
                    {"id": "sample-b", "task_type": "code2test", "repo": "org/small", "base_commit": "bbb"},
                    {"id": "sample-c", "task_type": "trace2code", "repo": "org/big", "base_commit": "aaa"},
                    {"id": "sample-d", "task_type": "comment2context", "repo": "org/other", "base_commit": "ccc"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [
                    {"repo": "org/big", "base_commit": "aaa", "chunk_count": 1000},
                    {"repo": "org/small", "base_commit": "bbb", "chunk_count": 10},
                    {"repo": "org/other", "base_commit": "ccc", "chunk_count": 20},
                ],
            )

            result = write_v1_1_sample_id_shards(
                derived=derived,
                out_dir=root / "shards",
                shard_count=2,
                manifest_out_path=root / "shards" / "manifest.json",
                markdown_out_path=root / "shards" / "manifest.md",
                assignment_strategy="corpus_balanced",
                corpus_manifest_path=corpus / "corpus_manifest.jsonl",
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["selection_strategy"], "corpus_balanced")
            self.assertEqual(result["corpus_group_count"], 3)
            self.assertEqual(result["shard_estimated_weights"], [1000, 30])
            shard_zero = (root / "shards" / "sample_ids_shard00.txt").read_text(encoding="utf-8").splitlines()
            shard_one = (root / "shards" / "sample_ids_shard01.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(shard_zero, ["sample-a", "sample-c"])
            self.assertEqual(shard_one, ["sample-d", "sample-b"])
            self.assertEqual(result["files"][0]["corpus_group_count"], 1)
            self.assertEqual(result["files"][1]["corpus_group_count"], 2)
            markdown = (root / "shards" / "manifest.md").read_text(encoding="utf-8")
            self.assertIn("corpus_balanced", markdown)

    def test_v1_1_sample_id_shards_block_duplicates_without_writing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            write_jsonl(
                derived / "samples.jsonl",
                [{"id": "sample-1", "task_type": "code2test"}, {"id": "sample-1", "task_type": "trace2code"}],
            )

            result = write_v1_1_sample_id_shards(
                derived=derived,
                out_dir=root / "shards",
                shard_count=2,
                manifest_out_path=root / "shards" / "manifest.json",
            )

            self.assertFalse(result["complete"])
            self.assertFalse(result["wrote_files"])
            self.assertEqual(result["duplicate_sample_ids"], ["sample-1"])
            self.assertTrue((root / "shards" / "manifest.json").exists())
            self.assertFalse((root / "shards" / "sample_ids_shard00.txt").exists())

    def test_v1_1_baseline_shard_commands_rewrite_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            write_jsonl(
                derived / "samples.jsonl",
                [{"id": f"sample-{index}", "task_type": "code2test"} for index in range(1, 5)],
            )
            sample_shards = write_v1_1_sample_id_shards(
                derived=derived,
                out_dir=root / "shards",
                shard_count=2,
                manifest_out_path=root / "shards" / "manifest.json",
            )
            self.assertTrue(sample_shards["complete"])
            handoff_path = root / "handoff.json"
            eval_dir = root / "eval"
            cache_dir = root / "embeddings"
            summary = eval_dir / "jina-code-embeddings-0.5b_summary.json"
            details = eval_dir / "jina-code-embeddings-0.5b_details.jsonl"
            cache = cache_dir / "jina-code-embeddings-0.5b"
            shared = cache_dir / "jina-code-embeddings-0.5b_texts.sqlite"
            command = (
                "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-embedding "
                f"--model jinaai/jina-code-embeddings-0.5b --derived {derived} --corpus {root / 'corpus'} "
                f"--out {summary} --details {details} --cache {cache} --shared-text-cache {shared} "
                "--candidate-filter all_files --resume-details --no-keep-list"
            )
            write_json(
                handoff_path,
                {
                    "inputs": {"derived": str(derived), "report_dir": str(root / "reports")},
                    "jobs": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "model": "jinaai/jina-code-embeddings-0.5b",
                            "command": command,
                            "artifacts": {
                                "summary": str(summary),
                                "details": str(details),
                                "cache": str(cache),
                                "shared_text_cache": str(shared),
                            },
                        }
                    ],
                },
            )

            result = write_v1_1_baseline_shard_commands(
                handoff_path=handoff_path,
                sample_shards_path=root / "shards" / "manifest.json",
                out_path=root / "commands.json",
                markdown_out_path=root / "commands.md",
            )

            self.assertTrue(result["complete"])
            baseline = result["baselines"][0]
            self.assertEqual(baseline["shard_count"], 2)
            first_command = baseline["shard_commands"][0]["command"]
            self.assertIn("--sample-id-file", first_command)
            self.assertIn("sample_ids_shard00.txt", first_command)
            self.assertIn("jina-code-embeddings-0.5b_shard00_details.jsonl", first_command)
            self.assertIn("jina-code-embeddings-0.5b_shard00_summary.json", first_command)
            self.assertIn("jina-code-embeddings-0.5b_shard00_texts.sqlite", first_command)
            self.assertIn("v1-1-merge-details", baseline["merge_command"])
            self.assertIn("v1-1-summary-from-details", baseline["summary_from_details_command"])
            self.assertTrue((root / "commands.json").exists())
            self.assertTrue((root / "commands.md").exists())

            shared_result = write_v1_1_baseline_shard_commands(
                handoff_path=handoff_path,
                sample_shards_path=root / "shards" / "manifest.json",
                out_path=root / "commands_shared.json",
                use_shard_caches=False,
            )
            shared_command = shared_result["baselines"][0]["shard_commands"][0]["command"]
            self.assertIn(f"--cache {cache}", shared_command)
            self.assertIn(f"--shared-text-cache {shared}", shared_command)
            self.assertNotIn("jina-code-embeddings-0.5b_shard00_texts.sqlite", shared_command)

    def test_v1_1_baseline_run_script_writes_executable_shard_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_commands_path = root / "commands.json"
            script_path = root / "run.sh"
            markdown_path = root / "run.md"
            write_json(
                root / "handoff.json",
                {
                    "transfer_manifest_verification": {
                        "report": str(root / "transfer_verify.json"),
                        "markdown": str(root / "transfer_verify.md"),
                    }
                },
            )
            write_json(
                shard_commands_path,
                {
                    "complete": True,
                    "handoff": str(root / "handoff.json"),
                    "baselines": [
                        {
                            "baseline": "voyage-code-3",
                            "shard_commands": [
                                {
                                    "index": 0,
                                    "command": (
                                        "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-voyage "
                                        "--model voyage-code-3 --details shard00.jsonl"
                                    ),
                                    "artifacts": {
                                        "summary": "eval/voyage-code-3_shard00_summary.json",
                                        "details": "eval/voyage-code-3_shard00_details.jsonl",
                                        "cache": "cache/voyage-code-3_shard00",
                                        "shared_text_cache": "cache/voyage-code-3_shard00_texts.sqlite",
                                    },
                                },
                                {
                                    "index": 1,
                                    "command": (
                                        "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-voyage "
                                        "--model voyage-code-3 --details shard01.jsonl"
                                    ),
                                    "artifacts": {
                                        "summary": "eval/voyage-code-3_shard01_summary.json",
                                        "details": "eval/voyage-code-3_shard01_details.jsonl",
                                        "cache": "cache/voyage-code-3_shard01",
                                        "shared_text_cache": "cache/voyage-code-3_shard01_texts.sqlite",
                                    },
                                },
                            ],
                            "merge_command": "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-merge-details --out merged.jsonl",
                            "summary_from_details_command": (
                                "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-summary-from-details --out summary.json"
                            ),
                            "final_artifacts": {
                                "summary": "eval/voyage-code-3_summary.json",
                                "details": "eval/voyage-code-3_details.jsonl",
                                "merge_report": "reports/voyage-code-3_merge_report.json",
                                "merge_markdown": "reports/voyage-code-3_merge_report.md",
                            },
                        }
                    ],
                },
            )

            result = write_v1_1_baseline_run_script(
                shard_commands_path=shard_commands_path,
                out_path=script_path,
                markdown_out_path=markdown_path,
                transfer_manifest_path=root / "transfer.json",
                return_manifest_path=root / "return.json",
                return_manifest_markdown_path=root / "return.md",
                return_files_path=root / "return.files",
                return_bundle_script_path=root / "package_return.sh",
                include_return_shard_artifacts=True,
                finalization_path=root / "finalization.json",
                finalization_markdown_path=root / "finalization.md",
                return_acceptance_path=root / "acceptance.json",
                return_acceptance_markdown_path=root / "acceptance.md",
                completion_json_path=root / "completion.json",
                workflow_evidence_paths=[root / "return_bundle.verify.json"],
            )

            script = script_path.read_text(encoding="utf-8")
            self.assertTrue(result["complete"])
            self.assertEqual(result["command_count"], 10)
            self.assertTrue(result["include_transfer_manifest_check"])
            self.assertTrue(result["include_return_manifest_check"])
            self.assertTrue(result["include_return_bundle_script"])
            self.assertTrue(result["include_finalization_check"])
            self.assertTrue(result["include_return_acceptance_refresh"])
            self.assertEqual(result["return_bundle_script"], str(root / "package_return.sh"))
            self.assertEqual(result["return_acceptance"], str(root / "acceptance.json"))
            self.assertEqual(result["completion_json"], str(root / "completion.json"))
            self.assertEqual(result["transfer_manifest_verify"], str(root / "transfer_verify.json"))
            self.assertEqual(result["transfer_manifest_verify_markdown"], str(root / "transfer_verify.md"))
            self.assertEqual(result["required_env"], ["VOYAGE_API_KEY"])
            self.assertTrue(result["requires_numpy"])
            self.assertEqual(result["preflight_dir_count"], 5)
            self.assertIn("set -euo pipefail", script)
            self.assertIn("Missing repo source files", script)
            self.assertIn("src/agent_retrieval_bench/cli.py", script)
            self.assertIn("VOYAGE_API_KEY", script)
            self.assertIn("Missing numpy", script)
            self.assertIn("mkdir -p cache cache/voyage-code-3_shard00 cache/voyage-code-3_shard01 eval reports", script)
            self.assertIn("df -h cache cache/voyage-code-3_shard00 cache/voyage-code-3_shard01 eval reports", script)
            self.assertIn("v1-1-verify-transfer-manifest", script)
            self.assertIn(f"--out {root / 'transfer_verify.json'}", script)
            self.assertIn(f"--markdown-out {root / 'transfer_verify.md'}", script)
            self.assertIn("v1-1-verify-handoff", script)
            self.assertIn("eval-voyage", script)
            self.assertIn("v1-1-merge-details", script)
            self.assertIn("v1-1-summary-from-details", script)
            self.assertIn("v1-1-baseline-return-manifest", script)
            self.assertIn("package verified return bundle", script)
            self.assertIn(f"bash {root / 'package_return.sh'}", script)
            self.assertIn("v1-1-finalize-baselines", script)
            self.assertLess(script.index("package verified return bundle"), script.index("v1-1-finalize-baselines"))
            self.assertIn("refresh return acceptance report", script)
            self.assertIn("v1-1-baseline-return-acceptance", script)
            self.assertIn("--completion-json", script)
            self.assertIn(str(root / "completion.json"), script)
            self.assertIn("--require-complete", script)
            self.assertLess(script.index("v1-1-finalize-baselines"), script.index("v1-1-baseline-return-acceptance"))
            self.assertIn("--auto-merge-shards", script)
            self.assertIn("--doc README.md", script)
            self.assertIn("--doc PLAN.md", script)
            self.assertIn("--doc docs/v1_1_completion_audit.md", script)
            self.assertIn("--workflow-evidence", script)
            self.assertIn("return_bundle.verify.json", script)
            self.assertIn("--require-existing", script)
            run_markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn(f"Transfer manifest verify: `{root / 'transfer_verify.json'}`", run_markdown)
            self.assertIn("writes the configured verifier report", run_markdown)
            self.assertIn("--include-shard-artifacts", script)
            self.assertIn(str(root / "return_bundle.verify.json"), result["workflow_evidence"])
            self.assertIn("return_bundle.verify.json", markdown_path.read_text(encoding="utf-8"))
            self.assertIn("Return bundle packaging: `True`", markdown_path.read_text(encoding="utf-8"))
            self.assertIn("Return acceptance refresh: `True`", markdown_path.read_text(encoding="utf-8"))
            self.assertTrue(script_path.stat().st_mode & 0o111)
            self.assertTrue(markdown_path.exists())

    def test_v1_1_baseline_run_script_can_filter_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_commands_path = root / "commands.json"
            handoff_path = root / "handoff.json"
            write_json(
                handoff_path,
                {
                    "handoff_verification": {
                        "report": str(root / "baseline_handoff_verify.json"),
                        "markdown": str(root / "baseline_handoff_verify.md"),
                    }
                },
            )
            write_json(
                shard_commands_path,
                {
                    "complete": True,
                    "handoff": str(handoff_path),
                    "baselines": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "shard_commands": [
                                {
                                    "index": 0,
                                    "command": (
                                        "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-embedding "
                                        "--model jina --device cuda"
                                    ),
                                    "artifacts": {
                                        "summary": "eval/jina_shard00_summary.json",
                                        "details": "eval/jina_shard00_details.jsonl",
                                        "cache": "cache/jina_shard00",
                                        "shared_text_cache": "cache/jina_shard00_texts.sqlite",
                                    },
                                }
                            ],
                            "merge_command": "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-merge-details --out jina.jsonl",
                            "summary_from_details_command": (
                                "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-summary-from-details --out jina.json"
                            ),
                            "final_artifacts": {
                                "summary": "eval/jina_summary.json",
                                "details": "eval/jina_details.jsonl",
                            },
                        },
                        {
                            "baseline": "voyage-code-3",
                            "shard_commands": [
                                {
                                    "index": 0,
                                    "command": (
                                        "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-voyage "
                                        "--model voyage-code-3"
                                    ),
                                    "artifacts": {
                                        "summary": "eval/voyage_shard00_summary.json",
                                        "details": "eval/voyage_shard00_details.jsonl",
                                        "cache": "cache/voyage_shard00",
                                        "shared_text_cache": "cache/voyage_shard00_texts.sqlite",
                                    },
                                }
                            ],
                            "merge_command": "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-merge-details --out voyage.jsonl",
                            "summary_from_details_command": (
                                "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-summary-from-details --out voyage.json"
                            ),
                            "final_artifacts": {
                                "summary": "eval/voyage_summary.json",
                                "details": "eval/voyage_details.jsonl",
                            },
                        },
                    ],
                },
            )

            result = write_v1_1_baseline_run_script(
                shard_commands_path=shard_commands_path,
                out_path=root / "voyage.sh",
                baseline_filters=["voyage-code-3"],
            )
            script = (root / "voyage.sh").read_text(encoding="utf-8")

            self.assertTrue(result["complete"])
            self.assertEqual(result["selected_baselines"], ["voyage-code-3"])
            self.assertEqual(result["required_env"], ["VOYAGE_API_KEY"])
            self.assertTrue(result["requires_numpy"])
            self.assertFalse(result["requires_cuda"])
            self.assertEqual(result["preflight_dirs"], ["cache", "cache/voyage_shard00", "eval"])
            self.assertIn("eval-voyage", script)
            self.assertIn("v1-1-verify-handoff", script)
            self.assertIn("--out", script)
            self.assertIn("baseline_handoff_verify.json", script)
            self.assertIn("--markdown-out", script)
            self.assertIn("baseline_handoff_verify.md", script)
            self.assertIn("Missing numpy", script)
            self.assertIn("df -h cache cache/voyage_shard00 eval", script)
            self.assertNotIn("eval-embedding", script)
            self.assertNotIn("sentence_transformers", script)
            self.assertNotIn("cache/jina_shard00", script)
            self.assertIn(str(handoff_path), result["workflow_evidence"])
            self.assertIn(str(root / "baseline_handoff_verify.json"), result["workflow_evidence"])

            jina = write_v1_1_baseline_run_script(
                shard_commands_path=shard_commands_path,
                out_path=root / "jina.sh",
                baseline_filters=["jina-code-embeddings-0.5b"],
            )
            jina_script = (root / "jina.sh").read_text(encoding="utf-8")

            self.assertTrue(jina["complete"])
            self.assertEqual(jina["required_env"], [])
            self.assertTrue(jina["requires_numpy"])
            self.assertTrue(jina["requires_cuda"])
            self.assertEqual(jina["preflight_dirs"], ["cache", "cache/jina_shard00", "eval"])
            self.assertIn("sentence_transformers", jina_script)
            self.assertIn("baseline_handoff_verify.json", jina_script)
            self.assertIn("Missing optional embedding dependencies", jina_script)
            self.assertIn("CUDA is not available", jina_script)
            self.assertIn("df -h cache cache/jina_shard00 eval", jina_script)
            self.assertNotIn("cache/voyage_shard00", jina_script)

            missing = write_v1_1_baseline_run_script(
                shard_commands_path=shard_commands_path,
                out_path=root / "missing.sh",
                baseline_filters=["not-a-baseline"],
            )
            self.assertFalse(missing["complete"])
            self.assertEqual(missing["missing_requested_baselines"], ["not-a-baseline"])

    def test_generated_return_acceptance_refresh_infers_completion_json_from_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = root / "handoff.json"
            completion_json = root / "reports" / "completion_from_handoff.json"
            write_json(
                handoff_path,
                {
                    "verification_commands": [
                        (
                            "PYTHONPATH=src python -m agent_retrieval_bench.cli report-v1-1-completion-audit "
                            f"--json-out {completion_json}"
                        )
                    ]
                },
            )
            shard_commands_path = root / "commands.json"
            write_json(
                shard_commands_path,
                {
                    "complete": True,
                    "handoff": str(handoff_path),
                    "baselines": [
                        {
                            "baseline": "voyage-code-3",
                            "shard_commands": [
                                {
                                    "index": 0,
                                    "command": "PYTHONPATH=src python -m agent_retrieval_bench.cli eval-voyage",
                                }
                            ],
                            "merge_command": "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-merge-details",
                            "summary_from_details_command": "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-summary-from-details",
                            "final_artifacts": {
                                "summary": "eval/voyage-code-3_summary.json",
                                "details": "eval/voyage-code-3_details.jsonl",
                            },
                        }
                    ],
                },
            )

            run_script = root / "run.sh"
            run_result = write_v1_1_baseline_run_script(
                shard_commands_path=shard_commands_path,
                out_path=run_script,
                return_manifest_path=root / "return.json",
                finalization_path=root / "finalization.json",
                return_acceptance_path=root / "acceptance.json",
            )
            run_text = run_script.read_text(encoding="utf-8")

            self.assertEqual(run_result["completion_json"], str(completion_json))
            self.assertIn("--completion-json", run_text)
            self.assertIn(str(completion_json), run_text)
            self.assertIn("--require-complete", run_text)

            apply_script = root / "apply.sh"
            apply_result = write_v1_1_baseline_apply_return_bundle_script(
                handoff_path=handoff_path,
                out_path=apply_script,
                bundle_path=root / "return_bundle.tar.zst",
                finalization_path=root / "finalization.json",
                return_manifest_path=root / "return.json",
                return_acceptance_path=root / "acceptance.json",
            )
            apply_text = apply_script.read_text(encoding="utf-8")

            self.assertEqual(apply_result["completion_json"], str(completion_json))
            self.assertIn("--completion-json", apply_text)
            self.assertIn(str(completion_json), apply_text)
            self.assertIn("--require-complete", apply_text)

    def test_v1_1_baseline_return_manifest_lists_required_final_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            report_dir = root / "reports"
            handoff_path = root / "handoff.json"
            summary = eval_dir / "jina-code-embeddings-0.5b_summary.json"
            details = eval_dir / "jina-code-embeddings-0.5b_details.jsonl"
            write_json(
                handoff_path,
                {
                    "jobs": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "model": "jinaai/jina-code-embeddings-0.5b",
                            "artifacts": {"summary": str(summary), "details": str(details)},
                        }
                    ],
                },
            )
            shard_commands_path = root / "commands.json"
            write_json(
                shard_commands_path,
                {
                    "baselines": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "final_artifacts": {
                                "summary": str(summary),
                                "details": str(details),
                                "merge_report": str(report_dir / "jina_merge.json"),
                                "merge_markdown": str(report_dir / "jina_merge.md"),
                            },
                            "shard_commands": [
                                {
                                    "index": 0,
                                    "artifacts": {
                                        "summary": str(eval_dir / "jina-code-embeddings-0.5b_shard00_summary.json"),
                                        "details": str(eval_dir / "jina-code-embeddings-0.5b_shard00_details.jsonl"),
                                    },
                                }
                            ],
                        }
                    ]
                },
            )
            write_json(summary, {"model": "jinaai/jina-code-embeddings-0.5b"})

            missing = write_v1_1_baseline_return_manifest(
                handoff_path=handoff_path,
                shard_commands_path=shard_commands_path,
                out_path=root / "return.json",
                markdown_out_path=root / "return.md",
                files_out_path=root / "return.files",
                include_shard_artifacts=True,
            )

            self.assertTrue(missing["complete"])
            self.assertFalse(missing["artifacts_complete"])
            self.assertEqual(missing["required_files"], [str(details), str(summary)])
            self.assertIn(str(report_dir / "jina_merge.json"), missing["optional_files"])
            self.assertIn(str(summary), missing["existing_files"])
            self.assertIn(str(details), missing["missing_files"])
            self.assertIn(str(details), missing["missing_required_files"])
            self.assertEqual(
                missing["required_files_by_baseline"],
                {"jina-code-embeddings-0.5b": [str(details), str(summary)]},
            )
            self.assertEqual(
                missing["missing_required_files_by_baseline"],
                {"jina-code-embeddings-0.5b": [str(details)]},
            )
            self.assertEqual(
                missing["existing_files_by_baseline"],
                {"jina-code-embeddings-0.5b": [str(summary)]},
            )
            self.assertIn(str(summary), missing["files"])
            self.assertIn(str(eval_dir / "jina-code-embeddings-0.5b_shard00_details.jsonl"), missing["files"])
            self.assertTrue((root / "return.md").exists())
            self.assertTrue((root / "return.files").exists())
            self.assertIn("Missing Required Files By Baseline", (root / "return.md").read_text(encoding="utf-8"))

            stable_payload = json.loads((root / "return.json").read_text(encoding="utf-8"))
            stable_payload["generated_at"] = "2026-01-01T00:00:00+00:00"
            (root / "return.json").write_text(json.dumps(stable_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stable = write_v1_1_baseline_return_manifest(
                handoff_path=handoff_path,
                shard_commands_path=shard_commands_path,
                out_path=root / "return.json",
                markdown_out_path=root / "return.md",
                files_out_path=root / "return.files",
                include_shard_artifacts=True,
            )
            self.assertEqual(stable["generated_at"], "2026-01-01T00:00:00+00:00")
            self.assertIn("2026-01-01T00:00:00+00:00", (root / "return.md").read_text(encoding="utf-8"))

            write_jsonl(details, [{"sample_id": "sample-1"}])
            complete = write_v1_1_baseline_return_manifest(
                handoff_path=handoff_path,
                out_path=root / "return_complete.json",
            )
            self.assertTrue(complete["artifacts_complete"])

    def test_v1_1_baseline_return_manifest_can_filter_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            handoff_path = root / "handoff.json"
            write_json(
                handoff_path,
                {
                    "jobs": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "artifacts": {
                                "summary": str(eval_dir / "jina_summary.json"),
                                "details": str(eval_dir / "jina_details.jsonl"),
                            },
                        },
                        {
                            "baseline": "voyage-code-3",
                            "artifacts": {
                                "summary": str(eval_dir / "voyage_summary.json"),
                                "details": str(eval_dir / "voyage_details.jsonl"),
                            },
                        },
                    ],
                },
            )
            write_json(eval_dir / "voyage_summary.json", {"model": "voyage-code-3"})
            write_jsonl(eval_dir / "voyage_details.jsonl", [{"sample_id": "sample-1"}])

            filtered = write_v1_1_baseline_return_manifest(
                handoff_path=handoff_path,
                out_path=root / "return_voyage.json",
                markdown_out_path=root / "return_voyage.md",
                files_out_path=root / "return_voyage.files",
                baseline_filters=["voyage-code-3"],
            )

            self.assertTrue(filtered["complete"])
            self.assertTrue(filtered["artifacts_complete"])
            self.assertEqual(filtered["requested_baselines"], ["voyage-code-3"])
            self.assertEqual(filtered["selected_baselines"], ["voyage-code-3"])
            self.assertEqual(filtered["missing_requested_baselines"], [])
            self.assertEqual(filtered["required_files"], [str(eval_dir / "voyage_details.jsonl"), str(eval_dir / "voyage_summary.json")])
            self.assertEqual(
                filtered["required_files_by_baseline"],
                {"voyage-code-3": [str(eval_dir / "voyage_details.jsonl"), str(eval_dir / "voyage_summary.json")]},
            )
            self.assertEqual(filtered["missing_required_files_by_baseline"], {})
            self.assertNotIn(str(eval_dir / "jina_summary.json"), filtered["files"])
            self.assertIn("voyage-code-3", (root / "return_voyage.md").read_text(encoding="utf-8"))

            missing = write_v1_1_baseline_return_manifest(
                handoff_path=handoff_path,
                out_path=root / "return_missing.json",
                baseline_filters=["not-a-baseline"],
            )
            self.assertFalse(missing["complete"])
            self.assertEqual(missing["missing_requested_baselines"], ["not-a-baseline"])

    def test_v1_1_baseline_return_acceptance_summarizes_current_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            report_dir = root / "reports"
            required_details = eval_dir / "voyage-code-3_details.jsonl"
            required_summary = eval_dir / "voyage-code-3_summary.json"
            handoff_path = root / "handoff.json"
            return_manifest_path = report_dir / "return.json"
            completion_json_path = report_dir / "completion.json"
            write_json(
                handoff_path,
                {
                    "external_acceptance": {
                        "external_baselines": ["voyage-code-3"],
                        "required_return_files": [str(required_details), str(required_summary)],
                        "required_return_file_count": 2,
                        "run_scripts": [str(report_dir / "run_v19_voyage_baseline_shards.sh")],
                        "return_packaging_scripts": [str(report_dir / "package_v19_voyage_return_artifacts.sh")],
                        "return_apply_scripts": [str(report_dir / "apply_v19_voyage_return_artifacts.sh")],
                        "return_manifest": str(return_manifest_path),
                    }
                },
            )
            write_json(
                return_manifest_path,
                {
                    "generated_at": "2026-01-02T03:04:05+00:00",
                    "artifacts_complete": False,
                    "required_files": [str(required_details), str(required_summary)],
                    "missing_required_files": [str(required_details)],
                },
            )
            write_json(completion_json_path, {"generated_at": "2026-01-02T04:05:06+00:00", "overall_status": "not_complete"})

            result = write_v1_1_baseline_return_acceptance(
                handoff_path=handoff_path,
                return_manifest_path=return_manifest_path,
                completion_json_path=completion_json_path,
                out_path=report_dir / "acceptance.json",
                markdown_out_path=report_dir / "acceptance.md",
            )

            self.assertFalse(result["complete"])
            self.assertEqual(result["external_baselines"], ["voyage-code-3"])
            self.assertEqual(result["required_return_file_count"], 2)
            self.assertEqual(
                result["required_return_files_by_baseline"],
                {"voyage-code-3": [str(required_details), str(required_summary)]},
            )
            self.assertFalse(result["artifacts_complete"])
            self.assertTrue(result["return_manifest_exists"])
            self.assertEqual(result["return_manifest_generated_at"], "2026-01-02T03:04:05+00:00")
            self.assertTrue(result["completion_audit_exists"])
            self.assertEqual(result["completion_audit_generated_at"], "2026-01-02T04:05:06+00:00")
            self.assertEqual(result["missing_required_file_count"], 1)
            self.assertEqual(result["missing_required_files"], [str(required_details)])
            self.assertEqual(result["missing_required_files_by_baseline"], {"voyage-code-3": [str(required_details)]})
            self.assertEqual(result["completion_audit_overall_status"], "not_complete")
            self.assertEqual(result["current_status"]["return_manifest_generated_at"], "2026-01-02T03:04:05+00:00")
            self.assertEqual(result["current_status"]["completion_audit_generated_at"], "2026-01-02T04:05:06+00:00")
            self.assertEqual(result["current_status"]["missing_required_file_count"], 1)
            self.assertEqual(result["current_status"]["missing_required_files_by_baseline"], {"voyage-code-3": [str(required_details)]})
            self.assertEqual(result["current_status"]["completion_audit_overall_status"], "not_complete")
            self.assertIn(str(completion_json_path), result["completion_gate"])
            acceptance_markdown = (report_dir / "acceptance.md").read_text(encoding="utf-8")
            self.assertIn("Completion Gate", acceptance_markdown)
            self.assertIn("--require-complete", acceptance_markdown)
            self.assertIn("Required Return Files By Baseline", acceptance_markdown)
            self.assertIn("Missing required files by baseline", acceptance_markdown)
            self.assertIn("Return manifest generated at", acceptance_markdown)
            self.assertIn("2026-01-02T04:05:06+00:00", acceptance_markdown)
            self.assertIn("voyage-code-3_summary.json", acceptance_markdown)
            self.assertIn("not_complete", acceptance_markdown)

            write_jsonl(required_details, [{"sample_id": "sample-1"}])
            write_json(required_summary, {"model": "voyage-code-3"})
            write_json(
                return_manifest_path,
                {
                    "generated_at": "2026-01-02T05:06:07+00:00",
                    "artifacts_complete": True,
                    "required_files": [str(required_details), str(required_summary)],
                    "missing_required_files": [],
                },
            )
            write_json(completion_json_path, {"generated_at": "2026-01-02T06:07:08+00:00", "overall_status": "complete"})

            complete = write_v1_1_baseline_return_acceptance(
                handoff_path=handoff_path,
                return_manifest_path=return_manifest_path,
                completion_json_path=completion_json_path,
                out_path=report_dir / "acceptance_complete.json",
            )
            self.assertTrue(complete["complete"])
            self.assertTrue(complete["artifacts_complete"])
            self.assertEqual(complete["missing_required_file_count"], 0)
            self.assertEqual(complete["return_manifest_generated_at"], "2026-01-02T05:06:07+00:00")
            self.assertEqual(complete["completion_audit_generated_at"], "2026-01-02T06:07:08+00:00")
            self.assertEqual(complete["completion_audit_overall_status"], "complete")

    def test_v1_1_baseline_return_bundle_script_refreshes_manifest_and_packages_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = root / "handoff.json"
            shard_commands_path = root / "commands.json"
            write_json(
                handoff_path,
                {
                    "jobs": [
                        {
                            "baseline": "voyage-code-3",
                            "artifacts": {
                                "summary": "eval/voyage-code-3_summary.json",
                                "details": "eval/voyage-code-3_details.jsonl",
                            },
                        }
                    ]
                },
            )
            write_json(shard_commands_path, {"baselines": []})
            script_path = root / "package_return.sh"
            markdown_path = root / "package_return.md"
            bundle_path = root / "baseline_return_bundle.tar.zst"

            result = write_v1_1_baseline_return_bundle_script(
                handoff_path=handoff_path,
                shard_commands_path=shard_commands_path,
                return_manifest_path=root / "return.json",
                return_manifest_markdown_path=root / "return.md",
                return_files_path=root / "return.files",
                bundle_path=bundle_path,
                bundle_files_path=root / "return_existing.files",
                out_path=script_path,
                markdown_out_path=markdown_path,
                include_shard_artifacts=True,
                baseline_filters=["voyage-code-3"],
            )

            script = script_path.read_text(encoding="utf-8")
            self.assertTrue(result["complete"])
            self.assertEqual(result["bundle"], str(bundle_path))
            self.assertTrue(script_path.stat().st_mode & 0o111)
            self.assertTrue(markdown_path.exists())
            self.assertIn("v1-1-baseline-return-manifest", script)
            self.assertIn("--baseline voyage-code-3", script)
            self.assertIn("--require-existing", script)
            self.assertIn("--include-shard-artifacts", script)
            self.assertIn("check-baseline-summaries", script)
            self.assertIn("--required-baseline voyage-code-3", script)
            self.assertIn("baseline_return_bundle.tar.zst.preflight.json", script)
            self.assertIn("existing_files", script)
            self.assertIn("PurePosixPath", script)
            self.assertIn("Unsafe return artifact paths", script)
            self.assertIn("Path(path).is_symlink()", script)
            self.assertIn("Symlink return artifact paths are not allowed", script)
            self.assertIn("tar --files-from=", script)
            self.assertIn("sha256sum", script)
            self.assertIn("zstd -t", script)
            self.assertIn("tar --use-compress-program=zstd -tf", script)
            self.assertIn("Verified {len(actual)} return artifact bundle members.", script)
            self.assertIn("Bundle contains unexpected files", script)
            self.assertIn("return_existing.files.archive", script)
            self.assertIn("v1-1-verify-return-bundle", script)
            self.assertIn("baseline_return_bundle.tar.zst.verify.json", script)
            self.assertIn("--bundle-files", script)
            self.assertEqual(result["requested_baselines"], ["voyage-code-3"])
            self.assertEqual(result["bundle_preflight"], str(bundle_path) + ".preflight.json")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("voyage-code-3", markdown)
            self.assertIn("Bundle preflight", markdown)
            self.assertIn("absolute or parent-traversal", markdown)

    def test_v1_1_baseline_apply_return_bundle_script_unpacks_and_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = root / "handoff.json"
            shard_commands_path = root / "commands.json"
            write_json(handoff_path, {"jobs": []})
            write_json(shard_commands_path, {"baselines": []})
            script_path = root / "apply_return.sh"
            markdown_path = root / "apply_return.md"

            result = write_v1_1_baseline_apply_return_bundle_script(
                handoff_path=handoff_path,
                shard_commands_path=shard_commands_path,
                return_manifest_path=root / "return.json",
                return_manifest_markdown_path=root / "return.md",
                return_files_path=root / "return.files",
                bundle_path=root / "baseline_return_bundle.tar.zst",
                finalization_path=root / "finalization.json",
                finalization_markdown_path=root / "finalization.md",
                return_acceptance_path=root / "acceptance.json",
                return_acceptance_markdown_path=root / "acceptance.md",
                completion_json_path=root / "completion.json",
                out_path=script_path,
                markdown_out_path=markdown_path,
                include_shard_artifacts=True,
                workflow_evidence_paths=[root / "transfer_unpack_smoke.json"],
            )

            script = script_path.read_text(encoding="utf-8")
            self.assertTrue(result["complete"])
            self.assertTrue(script_path.stat().st_mode & 0o111)
            self.assertTrue(markdown_path.exists())
            self.assertIn("ARB_RETURN_BUNDLE", script)
            self.assertIn("ARB_RETURN_CHECKSUM", script)
            self.assertIn("checksum_path=", script)
            self.assertIn('sha256sum "$bundle_path"', script)
            self.assertIn("Return bundle checksum mismatch", script)
            self.assertIn('zstd -t "$bundle_path"', script)
            self.assertIn('tar --use-compress-program=zstd -tf "$bundle_path"', script)
            self.assertIn("Unsafe return bundle member paths", script)
            self.assertIn("Validated {len(members)} safe return bundle member paths.", script)
            self.assertIn('tar --use-compress-program=zstd -tvf "$bundle_path"', script)
            self.assertIn("Return bundle contains non-regular file members", script)
            self.assertIn("Validated {len(members)} regular file return bundle members.", script)
            self.assertIn("verify return bundle members against manifest", script)
            self.assertIn("Return bundle contains files not listed in return manifest", script)
            self.assertIn("Return bundle is missing required return files", script)
            self.assertIn("Validated {len(members)} return bundle members against {manifest_path}.", script)
            self.assertIn("v1-1-verify-return-bundle", script)
            self.assertIn('--bundle "$bundle_path"', script)
            self.assertIn('--checksum "$checksum_path"', script)
            self.assertIn("baseline_return_bundle.tar.zst.verify.json", script)
            self.assertIn('tar --use-compress-program=zstd -xf "$bundle_path"', script)
            self.assertIn("baseline_return_bundle.tar.zst.members", script)
            self.assertEqual(result["bundle_member_types"], str(root / "baseline_return_bundle.tar.zst.members.types"))
            self.assertIn("v1-1-baseline-return-manifest", script)
            self.assertIn("--require-existing", script)
            self.assertIn("v1-1-finalize-baselines", script)
            self.assertIn("refresh return acceptance report", script)
            self.assertIn("v1-1-baseline-return-acceptance", script)
            self.assertIn("--completion-json", script)
            self.assertIn(str(root / "completion.json"), script)
            self.assertIn("--require-complete", script)
            self.assertLess(script.index("v1-1-finalize-baselines"), script.index("v1-1-baseline-return-acceptance"))
            self.assertTrue(result["include_return_acceptance_refresh"])
            self.assertEqual(result["return_acceptance"], str(root / "acceptance.json"))
            self.assertEqual(result["completion_json"], str(root / "completion.json"))
            self.assertIn("--auto-merge-shards", script)
            self.assertIn("--include-shard-artifacts", script)
            self.assertIn("--doc README.md", script)
            self.assertIn("--doc PLAN.md", script)
            self.assertIn("--doc docs/v1_1_completion_audit.md", script)
            self.assertIn("--workflow-evidence", script)
            self.assertIn("baseline_return_bundle.tar.zst.verify.json", script)
            self.assertIn("transfer_unpack_smoke.json", script)
            self.assertIn(str(root / "transfer_unpack_smoke.json"), result["workflow_evidence"])
            self.assertIn("transfer_unpack_smoke.json", markdown_path.read_text(encoding="utf-8"))
            self.assertIn("Return acceptance refresh: `True`", markdown_path.read_text(encoding="utf-8"))

    def test_v1_1_baseline_apply_return_bundle_script_can_skip_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = root / "handoff.json"
            shard_commands_path = root / "commands.json"
            write_json(handoff_path, {"jobs": []})
            write_json(shard_commands_path, {"baselines": []})
            script_path = root / "apply_partial_return.sh"
            markdown_path = root / "apply_partial_return.md"

            result = write_v1_1_baseline_apply_return_bundle_script(
                handoff_path=handoff_path,
                shard_commands_path=shard_commands_path,
                return_manifest_path=root / "return_gpu.json",
                return_manifest_markdown_path=root / "return_gpu.md",
                return_files_path=root / "return_gpu.files",
                bundle_path=root / "baseline_return_bundle_gpu.tar.zst",
                finalization_path=root / "finalization.json",
                finalization_markdown_path=root / "finalization.md",
                out_path=script_path,
                markdown_out_path=markdown_path,
                include_shard_artifacts=True,
                run_finalization=False,
                baseline_filters=["jina-code-embeddings-0.5b", "qwen3-embedding-4b"],
                workflow_evidence_paths=[root / "transfer_unpack_smoke.json"],
            )

            script = script_path.read_text(encoding="utf-8")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertTrue(result["complete"])
            self.assertFalse(result["run_finalization"])
            self.assertIsNone(result["finalization"])
            self.assertEqual(result["requested_baselines"], ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"])
            self.assertEqual(result["workflow_evidence"], [])
            self.assertIn("ARB_RETURN_BUNDLE", script)
            self.assertIn("ARB_RETURN_CHECKSUM", script)
            self.assertIn("checksum_path=", script)
            self.assertIn('sha256sum "$bundle_path"', script)
            self.assertIn("Return bundle checksum mismatch", script)
            self.assertIn('zstd -t "$bundle_path"', script)
            self.assertIn('tar --use-compress-program=zstd -tf "$bundle_path"', script)
            self.assertIn("v1-1-verify-return-bundle", script)
            self.assertIn("v1-1-baseline-return-manifest", script)
            self.assertIn("--baseline jina-code-embeddings-0.5b", script)
            self.assertIn("--baseline qwen3-embedding-4b", script)
            self.assertEqual(script.count("--baseline jina-code-embeddings-0.5b"), 2)
            self.assertEqual(script.count("--baseline qwen3-embedding-4b"), 2)
            self.assertIn("write return manifest for bundle verification", script)
            preverify_section = script.split("echo '-- verify return bundle members against manifest --'", maxsplit=1)[0]
            self.assertIn("v1-1-baseline-return-manifest", preverify_section)
            self.assertIn("--baseline jina-code-embeddings-0.5b", preverify_section)
            self.assertIn("--baseline qwen3-embedding-4b", preverify_section)
            self.assertNotIn("--require-existing", preverify_section)
            post_unpack_section = script.split("echo '-- verify returned required artifacts --'", maxsplit=1)[1]
            self.assertIn("v1-1-baseline-return-manifest", post_unpack_section)
            self.assertIn("--require-existing", post_unpack_section)
            self.assertIn("--require-existing", script)
            self.assertIn('tar --use-compress-program=zstd -xf "$bundle_path"', script)
            self.assertIn("finalization disabled", script)
            self.assertNotIn("v1-1-finalize-baselines", script)
            self.assertNotIn("--workflow-evidence", script)
            self.assertIn('Requested baselines: `["jina-code-embeddings-0.5b", "qwen3-embedding-4b"]`', markdown)
            self.assertIn("Run finalization: `False`", markdown)
            self.assertIn("Finalization is disabled", markdown)
            self.assertIn("writes a filtered return manifest before member verification", markdown)

    def test_v1_1_baseline_apply_return_bundle_script_accepts_checksum_with_foreign_path(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed return bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
            (root / "src" / "agent_retrieval_bench").mkdir(parents=True)
            (root / "src" / "agent_retrieval_bench" / "cli.py").write_text("", encoding="utf-8")
            write_json(root / "handoff.json", {"jobs": []})
            payload_root = root / "payload"
            returned_file = payload_root / "returned" / "marker.txt"
            returned_file.parent.mkdir(parents=True)
            returned_file.write_text("ok\n", encoding="utf-8")
            bundle_path = root / "incoming_return_bundle.tar.zst"
            subprocess.run(
                ["tar", "--use-compress-program=zstd", "-cf", str(bundle_path), "-C", str(payload_root), "returned/marker.txt"],
                check=True,
            )
            digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            checksum_path = root / "incoming_return_bundle.tar.zst.sha256"
            checksum_path.write_text(f"{digest}  /external-runner/baseline_return_bundle.tar.zst\n", encoding="utf-8")
            script_path = root / "apply_return.sh"

            write_v1_1_baseline_apply_return_bundle_script(
                handoff_path=root / "handoff.json",
                bundle_path=root / "generated_default_missing.tar.zst",
                checksum_path=root / "generated_default_missing.tar.zst.sha256",
                finalization_path=root / "finalization.json",
                out_path=script_path,
                run_finalization=False,
            )

            env = {"ARB_RETURN_BUNDLE": str(bundle_path), "ARB_RETURN_CHECKSUM": str(checksum_path)}
            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, **env},
            )
            self.assertIn("safe return bundle member paths.", result.stdout)
            self.assertIn("regular file return bundle members.", result.stdout)
            self.assertIn("finalization was not run", result.stdout)
            self.assertEqual((root / "returned" / "marker.txt").read_text(encoding="utf-8"), "ok\n")

    def test_v1_1_return_bundle_verifier_checks_manifest_members(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed return bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            summary = eval_dir / "model_summary.json"
            details = eval_dir / "model_details.jsonl"
            optional = eval_dir / "model_shard00_details.jsonl"
            summary.write_text("{}\n", encoding="utf-8")
            details.write_text("{}\n", encoding="utf-8")
            optional.write_text("{}\n", encoding="utf-8")
            bundle = root / "return.tar.zst"
            checksum = root / "return.tar.zst.sha256"
            members = root / "return.members"
            bundle_files = root / "return.files"
            bundle_files.write_text("eval/model_details.jsonl\neval/model_summary.json\n", encoding="utf-8")
            manifest = root / "return_manifest.json"
            write_json(
                manifest,
                {
                    "files": ["eval/model_details.jsonl", "eval/model_summary.json", "eval/model_shard00_details.jsonl"],
                    "required_files": ["eval/model_details.jsonl", "eval/model_summary.json"],
                },
            )
            subprocess.run(
                [
                    "tar",
                    "--files-from",
                    str(bundle_files),
                    "--use-compress-program=zstd",
                    "-cf",
                    str(bundle),
                ],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            checksum.write_text(f"{digest}  {bundle}\n", encoding="utf-8")

            result = verify_v1_1_baseline_return_bundle(
                bundle_path=bundle,
                return_manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=members,
                bundle_files_path=bundle_files,
                out_path=root / "return_verify.json",
                markdown_out_path=root / "return_verify.md",
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_checks"], [])
            self.assertEqual(result["sha256"], digest)
            self.assertGreater(result["size_bytes"], 0)
            self.assertEqual(members.read_text(encoding="utf-8").splitlines(), ["eval/model_details.jsonl", "eval/model_summary.json"])
            saved_report = json.loads((root / "return_verify.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["sha256"], digest)
            self.assertEqual(saved_report["size_bytes"], result["size_bytes"])
            self.assertIn("SHA256", (root / "return_verify.md").read_text(encoding="utf-8"))
            compression_check = [item for item in result["checks"] if item["name"] == "return_bundle_compression_valid"][0]
            self.assertIsNone(compression_check["error"])
            self.assertIn("stderr", compression_check)
            regular_check = [item for item in result["checks"] if item["name"] == "return_bundle_members_are_regular_files"][0]
            self.assertEqual(regular_check["non_regular_member_count"], 0)
            self.assertTrue((root / "return_verify.md").exists())

            write_json(manifest, {"files": ["eval/model_details.jsonl"], "required_files": ["eval/model_details.jsonl", "eval/model_summary.json"]})
            failed = verify_v1_1_baseline_return_bundle(
                bundle_path=bundle,
                return_manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=root / "failed.members",
            )
            self.assertFalse(failed["complete"])
            self.assertIn("return_bundle_members_match_return_manifest", failed["failed_checks"])

            missing_bundle = verify_v1_1_baseline_return_bundle(
                bundle_path=root / "missing.tar.zst",
                return_manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=root / "missing.members",
            )
            self.assertFalse(missing_bundle["complete"])
            self.assertEqual(
                {check["name"]: check["status"] for check in missing_bundle["checks"]},
                {
                    "return_bundle_checksum_matches": "fail",
                    "return_bundle_compression_valid": "fail",
                    "return_bundle_members_are_safe_paths": "fail",
                    "return_bundle_members_are_regular_files": "fail",
                    "return_bundle_members_match_return_manifest": "fail",
                },
            )

    def test_v1_1_return_bundle_verifier_rejects_unsafe_members(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed return bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.txt"
            payload.write_text("unsafe\n", encoding="utf-8")
            raw_tar = root / "return.tar"
            bundle = root / "return.tar.zst"
            checksum = root / "return.tar.zst.sha256"
            manifest = root / "return_manifest.json"
            members = root / "return.members"
            with tarfile.open(raw_tar, "w") as archive:
                archive.add(payload, arcname="../payload.txt")
            subprocess.run(["zstd", "-q", "-f", str(raw_tar), "-o", str(bundle)], check=True)
            checksum.write_text(f"{hashlib.sha256(bundle.read_bytes()).hexdigest()}  {bundle}\n", encoding="utf-8")
            write_json(manifest, {"files": ["../payload.txt"], "required_files": ["../payload.txt"]})

            result = verify_v1_1_baseline_return_bundle(
                bundle_path=bundle,
                return_manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=members,
            )

            self.assertFalse(result["complete"])
            self.assertIn("return_bundle_members_are_safe_paths", result["failed_checks"])
            safe_check = [item for item in result["checks"] if item["name"] == "return_bundle_members_are_safe_paths"][0]
            self.assertEqual(safe_check["unsafe_member_count"], 1)
            self.assertEqual(safe_check["unsafe_members"], ["../payload.txt"])

    def test_v1_1_return_bundle_verifier_rejects_non_regular_members(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed return bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            link = root / "linked_summary.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            raw_tar = root / "return.tar"
            bundle = root / "return.tar.zst"
            checksum = root / "return.tar.zst.sha256"
            manifest = root / "return_manifest.json"
            with tarfile.open(raw_tar, "w") as archive:
                archive.add(link, arcname="eval/model_summary.json")
            subprocess.run(["zstd", "-q", "-f", str(raw_tar), "-o", str(bundle)], check=True)
            checksum.write_text(f"{hashlib.sha256(bundle.read_bytes()).hexdigest()}  {bundle}\n", encoding="utf-8")
            write_json(manifest, {"files": ["eval/model_summary.json"], "required_files": ["eval/model_summary.json"]})

            result = verify_v1_1_baseline_return_bundle(
                bundle_path=bundle,
                return_manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=root / "return.members",
            )

            self.assertFalse(result["complete"])
            self.assertIn("return_bundle_members_are_regular_files", result["failed_checks"])
            regular_check = [item for item in result["checks"] if item["name"] == "return_bundle_members_are_regular_files"][0]
            self.assertEqual(regular_check["non_regular_member_count"], 1)
            self.assertEqual(regular_check["non_regular_members"][0]["type"], "l")

    def test_v1_1_baseline_finalization_regenerates_release_gates_from_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_derived = root / "benchmark" / "v1"
            derived = root / "benchmark" / "v1_1"
            corpus = root / "corpus" / "v1_1"
            eval_dir = root / "eval" / "v1_1"
            report_dir = root / "reports" / "v1_1"
            cache_root = root / "embeddings" / "v1_1"
            handoff_path = report_dir / "handoff.json"
            doc_path = root / "README.md"
            doc_path.write_text(V1_1_DOC_TEXT, encoding="utf-8")

            base_code = sample("base-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            good_comment = sample(
                "good-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["cross_module_contract"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            good_trace = sample(
                "good-trace",
                "trace2code",
                {"trace": "Traceback: AssertionError in runtime fairness check"},
                {"root_cause_files": ["src/runtime/worker_impl.py"]},
            )
            write_jsonl(base_derived / "samples.jsonl", [base_code])
            write_jsonl(derived / "samples.jsonl", [base_code, good_comment, good_trace])
            write_json(
                derived / "manifest.json",
                {"require_audit_keep": True, "audit_paths": [str(report_dir / "audit.csv")], "accepted_expansion": 2},
            )
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                chunks_path,
                [
                    {"path": "tests/web/test_sanitizer.py", "kind": "file"},
                    {"path": "src/runtime/worker_impl.py", "kind": "file"},
                ],
            )
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path), "chunk_count": 2, "file_count": 2}],
            )
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            detail_rows = [
                {"sample_id": "base-code", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics},
                {"sample_id": "good-comment", "task_type": "comment2context", "candidate_filter": "all_files", "metrics": zero_metrics},
                {"sample_id": "good-trace", "task_type": "trace2code", "candidate_filter": "all_files", "metrics": zero_metrics},
            ]
            summaries = [
                ("lexical", "lexical", "corpus"),
                ("repomap", "aider-style-repomap", "repomap"),
                ("jina-code-embeddings-0.5b", "jinaai/jina-code-embeddings-0.5b", "embedding"),
                ("qwen3-embedding-4b", "Qwen/Qwen3-Embedding-4B", "embedding"),
                ("voyage-code-3", "voyage-code-3", "embedding"),
            ]
            for stem, model, mode in summaries:
                details = eval_dir / f"{stem}_details.jsonl"
                write_jsonl(details, detail_rows)
                write_v1_1_summary_from_details(
                    details_path=details,
                    out_path=eval_dir / f"{stem}_summary.json",
                    model=model,
                    mode=mode,
                    expected_samples=3,
                    expected_sample_ids={"base-code", "good-comment", "good-trace"},
                )

            handoff = write_v1_1_baseline_handoff(
                derived=derived,
                corpus=corpus,
                eval_dir=eval_dir,
                cache_root=cache_root,
                report_dir=report_dir,
                out_path=handoff_path,
                markdown_out_path=report_dir / "handoff.md",
                base_derived=base_derived,
            )
            for index, command in enumerate(handoff["verification_commands"]):
                if " v1-1-readiness " in command:
                    handoff["verification_commands"][index] = (
                        command
                        + " --min-comment2context 1 --max-comment2context 1 --min-comment-cross-module 1"
                        + " --min-trace2code 1 --min-trace-non-go-repos 1 --min-trace-languages 1 --min-trace-failure-types 1"
                    )
            write_json(handoff_path, handoff)

            result = write_v1_1_baseline_finalization(
                handoff_path=handoff_path,
                shard_commands_path=None,
                return_manifest_path=report_dir / "return.json",
                out_path=report_dir / "finalization.json",
                markdown_out_path=report_dir / "finalization.md",
                docs=[doc_path],
            )

            self.assertTrue(result["complete"])
            self.assertIsNone(result["next_required_action"])
            self.assertTrue(result["steps"]["return_manifest"]["artifacts_complete"])
            self.assertTrue(result["steps"]["baseline_status"]["complete"])
            self.assertTrue(result["steps"]["baseline_preflight"]["complete"])
            self.assertTrue(result["steps"]["leaderboard"]["contains_required_baselines"])
            self.assertTrue(result["steps"]["readiness"]["ready"])
            self.assertEqual(result["steps"]["release"]["status"], "ready")
            self.assertEqual(result["steps"]["completion_audit"]["overall_status"], "complete")
            self.assertTrue((report_dir / "finalization.md").exists())

            (eval_dir / "qwen3-embedding-4b_summary.json").unlink()
            (eval_dir / "qwen3-embedding-4b_details.jsonl").unlink()
            incomplete = write_v1_1_baseline_finalization(
                handoff_path=handoff_path,
                shard_commands_path=None,
                return_manifest_path=report_dir / "return_incomplete.json",
                out_path=report_dir / "finalization_incomplete.json",
                markdown_out_path=report_dir / "finalization_incomplete.md",
                docs=[doc_path],
            )

            self.assertFalse(incomplete["complete"])
            self.assertEqual(
                incomplete["next_required_action"],
                incomplete["steps"]["completion_audit"]["next_required_action"],
            )
            self.assertIn("qwen3-embedding-4b", incomplete["next_required_action"])
            self.assertIn("## Next Action", (report_dir / "finalization_incomplete.md").read_text(encoding="utf-8"))

    def test_baseline_handoff_contains_external_jobs_and_verifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            base_derived = root / "benchmark" / "v1"
            corpus = root / "corpus" / "v1_1"
            derived.mkdir(parents=True)
            base_derived.mkdir(parents=True)
            corpus.mkdir(parents=True)
            write_jsonl(
                derived / "samples.jsonl",
                [
                    {"id": "sample-1", "task_type": "comment2context"},
                    {"id": "sample-2", "task_type": "trace2code"},
                ],
            )
            write_jsonl(derived / "comment2context.jsonl", [{"id": "sample-1", "task_type": "comment2context"}])
            write_jsonl(derived / "trace2code.jsonl", [{"id": "sample-2", "task_type": "trace2code"}])
            write_jsonl(base_derived / "samples.jsonl", [{"id": "base-1", "task_type": "code2test"}])
            write_jsonl(base_derived / "code2test.jsonl", [{"id": "base-1", "task_type": "code2test"}])
            write_json(derived / "manifest.json", {"accepted_expansion": 2})
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(chunks_path, [{"repo": "o/r", "base_commit": "base", "path": "src/a.py", "kind": "file"}])
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path), "chunk_count": 1, "file_count": 1}],
            )
            result = write_v1_1_baseline_handoff(
                derived=derived,
                corpus=corpus,
                eval_dir=root / "eval" / "v1_1",
                cache_root=root / "embeddings" / "v1_1",
                report_dir=root / "reports" / "v1_1",
                out_path=root / "reports" / "v1_1" / "handoff.json",
                markdown_out_path=root / "reports" / "v1_1" / "handoff.md",
                base_derived=base_derived,
                baseline_status_path=root / "reports" / "v1_1" / "baseline_status_v19.json",
                baseline_status_markdown_path=root / "reports" / "v1_1" / "baseline_status_v19.md",
                baseline_preflight_path=root / "reports" / "v1_1" / "baseline_summary_preflight_v19.json",
                finalization_path=root / "reports" / "v1_1" / "baseline_finalization_v19.json",
                finalization_markdown_path=root / "reports" / "v1_1" / "baseline_finalization_v19.md",
                shard_commands_path=root / "reports" / "v1_1" / "baseline_shard_commands_v19.json",
                return_manifest_path=root / "reports" / "v1_1" / "baseline_return_manifest_v19.json",
                return_manifest_markdown_path=root / "reports" / "v1_1" / "baseline_return_manifest_v19.md",
                return_files_path=root / "reports" / "v1_1" / "baseline_return_manifest_v19.files",
                include_shard_artifacts=True,
                auto_merge_shards=True,
                workflow_evidence_paths=[root / "reports" / "v1_1" / "baseline_return_bundle_v19.tar.zst.verify.json"],
                transfer_manifest_path=root / "reports" / "v1_1" / "baseline_transfer_manifest_v19.json",
                transfer_manifest_markdown_path=root / "reports" / "v1_1" / "baseline_transfer_manifest_v19.md",
                transfer_manifest_verify_path=root / "reports" / "v1_1" / "baseline_transfer_manifest_verify_v19.json",
                transfer_manifest_verify_markdown_path=root / "reports" / "v1_1" / "baseline_transfer_manifest_verify_v19.md",
                transfer_files_path=root / "reports" / "v1_1" / "baseline_transfer_manifest_v19.files",
                transfer_unpack_script_path=root / "reports" / "v1_1" / "unpack_transfer.sh",
                transfer_unpack_script_markdown_path=root / "reports" / "v1_1" / "unpack_transfer.md",
                transfer_unpack_destination="external-checkout",
                transfer_unpack_transfer_verify_path=root / "reports" / "v1_1" / "transfer_unpack_smoke.json",
                transfer_unpack_handoff_verify_path=root / "reports" / "v1_1" / "handoff_unpack_smoke.json",
            )

            self.assertEqual([job["baseline"] for job in result["jobs"]], ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"])
            self.assertEqual(result["external_acceptance"]["external_baselines"], ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"])
            self.assertEqual(result["external_acceptance"]["required_return_file_count"], 4)
            self.assertEqual(
                sorted(result["external_acceptance"]["required_return_files_by_baseline"]),
                ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"],
            )
            self.assertIn("--require-complete", result["external_acceptance"]["completion_gate"])
            self.assertIn("--resume-details", result["jobs"][0]["command"])
            self.assertEqual(result["jobs"][0]["required_env"], [])
            self.assertEqual(result["jobs"][1]["required_env"], [])
            self.assertTrue(any("report-v1-1-completion-audit" in command for command in result["verification_commands"]))
            joined_verifiers = "\n".join(result["verification_commands"])
            completion_verifier = next(
                command for command in result["verification_commands"] if "report-v1-1-completion-audit" in command
            )
            self.assertIn("baseline_status_v19.json", joined_verifiers)
            self.assertIn("baseline_status_v19.md", joined_verifiers)
            self.assertIn("baseline_summary_preflight_v19.json", joined_verifiers)
            self.assertIn("v1-1-finalize-baselines", joined_verifiers)
            self.assertIn("baseline_finalization_v19.json", joined_verifiers)
            self.assertIn("baseline_finalization_v19.md", joined_verifiers)
            self.assertIn("baseline_handoff_verify.json", joined_verifiers)
            self.assertIn("--doc README.md", joined_verifiers)
            self.assertIn("--doc PLAN.md", joined_verifiers)
            self.assertIn("--doc docs/v1_1_completion_audit.md", joined_verifiers)
            self.assertIn("--shard-commands", joined_verifiers)
            self.assertIn("baseline_shard_commands_v19.json", joined_verifiers)
            self.assertIn("--return-manifest", joined_verifiers)
            self.assertIn("baseline_return_manifest_v19.json", joined_verifiers)
            self.assertIn("--include-shard-artifacts", joined_verifiers)
            self.assertIn("--auto-merge-shards", joined_verifiers)
            self.assertIn("--workflow-evidence", joined_verifiers)
            self.assertIn("baseline_return_bundle_v19.tar.zst.verify.json", joined_verifiers)
            self.assertIn("--workflow-evidence", completion_verifier)
            self.assertIn("baseline_handoff_verify.json", completion_verifier)
            self.assertIn("baseline_return_bundle_v19.tar.zst.verify.json", completion_verifier)
            self.assertIn("baseline_return_manifest_v19.json", completion_verifier)
            self.assertEqual(
                result["finalization"]["workflow_evidence"],
                [
                    str(root / "reports" / "v1_1" / "handoff.json"),
                    str(root / "reports" / "v1_1" / "baseline_handoff_verify.json"),
                    str(root / "reports" / "v1_1" / "baseline_transfer_bundle_verify.json"),
                    str(root / "reports" / "v1_1" / "transfer_unpack_smoke.json"),
                    str(root / "reports" / "v1_1" / "handoff_unpack_smoke.json"),
                    str(root / "reports" / "v1_1" / "baseline_return_bundle_v19.tar.zst.verify.json"),
                    str(root / "reports" / "v1_1" / "baseline_return_manifest_v19.json"),
                ],
            )
            joined_setup = "\n".join(result["setup_commands"])
            self.assertIn("v1-1-baseline-transfer-unpack-script", joined_setup)
            self.assertIn("unpack_transfer.sh", joined_setup)
            self.assertIn("external-checkout", joined_setup)
            self.assertEqual(result["transfer_unpack_script"]["script"], str(root / "reports" / "v1_1" / "unpack_transfer.sh"))
            self.assertEqual(result["transfer_unpack_script"]["destination"], "external-checkout")
            self.assertIn("v1-1-baseline-transfer-manifest", joined_setup)
            self.assertIn("baseline_transfer_manifest_v19.files", joined_setup)
            self.assertIn("README.md", joined_setup)
            self.assertIn("PLAN.md", joined_setup)
            self.assertIn("docs/v1_1_completion_audit.md", joined_setup)
            self.assertIn("pyproject.toml", joined_setup)
            self.assertIn("src/agent_retrieval_bench", joined_setup)
            self.assertEqual(
                result["transfer_includes"],
                [
                    "README.md",
                    "PLAN.md",
                    "docs/v1_1_completion_audit.md",
                    "pyproject.toml",
                    "src/agent_retrieval_bench",
                ],
            )
            self.assertIn("v1-1-baseline-transfer-bundle", joined_setup)
            self.assertIn("baseline_transfer_bundle.tar.zst", joined_setup)
            self.assertIn("baseline_transfer_bundle.tar.zst.sha256", joined_setup)
            self.assertIn("baseline_transfer_bundle_verify.json", joined_setup)
            self.assertIn("baseline_handoff_verify.json", joined_setup)
            self.assertIn("baseline_handoff_verify.md", joined_setup)
            self.assertIn("v1-1-verify-transfer-manifest", joined_setup)
            self.assertIn("baseline_transfer_manifest_verify_v19.json", joined_setup)
            self.assertIn("baseline_transfer_manifest_verify_v19.md", joined_setup)
            self.assertIn("v1-1-verify-handoff", joined_setup)
            self.assertEqual(
                result["transfer_manifest_verification"],
                {
                    "report": str(root / "reports" / "v1_1" / "baseline_transfer_manifest_verify_v19.json"),
                    "markdown": str(root / "reports" / "v1_1" / "baseline_transfer_manifest_verify_v19.md"),
                },
            )
            self.assertEqual(
                result["transfer_bundle"]["bundle"],
                str(root / "reports" / "v1_1" / "baseline_transfer_bundle.tar.zst"),
            )
            handoff_markdown = (root / "reports" / "v1_1" / "handoff.md").read_text(encoding="utf-8")
            self.assertIn("Minimal External Runner Path", handoff_markdown)
            self.assertIn("transfer bundle checksum", handoff_markdown)
            self.assertIn("Install the local package plus embedding dependencies", handoff_markdown)
            self.assertIn("including `numpy`", handoff_markdown)
            self.assertIn("sentence-transformers", handoff_markdown)
            self.assertIn("overall_status=complete", handoff_markdown)
            self.assertIn("full return-bundle verifier report is required workflow evidence", handoff_markdown)
            self.assertIn("Current prepared transfer bootstrap", handoff_markdown)
            self.assertIn("bash unpack_transfer.sh baseline_transfer_bundle.tar.zst", handoff_markdown)
            self.assertIn("Copy these files beside each other", handoff_markdown)
            self.assertIn("supplied bundle path", handoff_markdown)
            self.assertIn("repo-relative path", handoff_markdown)
            self.assertIn("Transfer Bundle", handoff_markdown)
            self.assertIn("Transfer Unpack Script", handoff_markdown)
            self.assertIn("Acceptance Checklist", handoff_markdown)
            self.assertIn("Required final return files: `4`", handoff_markdown)
            self.assertIn("--require-complete", handoff_markdown)
            self.assertNotIn("voyage-code-3_summary.json", handoff_markdown)
            joined_notes = "\n".join(result["notes"])
            self.assertIn("transfer bundle", joined_notes)
            self.assertIn("return-bundle script", joined_notes)
            self.assertIn("Minimal external-runner path", joined_notes)
            self.assertIn("transfer bundle checksum", joined_notes)
            self.assertIn("verify transfer and handoff fingerprints", joined_notes)
            self.assertIn("including numpy", joined_notes)
            self.assertIn("full return-bundle verifier report", joined_notes)
            self.assertIn("required workflow evidence", joined_notes)
            self.assertIn("local reporting checkout", joined_notes)
            fingerprints = result["input_fingerprints"]
            self.assertEqual(fingerprints["derived_samples"]["rows"], 2)
            self.assertEqual(fingerprints["derived_samples"]["unique_ids"], 2)
            self.assertEqual(fingerprints["base_samples"]["rows"], 1)
            self.assertRegex(fingerprints["assembly_manifest"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(fingerprints["corpus_manifest"]["sha256"], r"^[0-9a-f]{64}$")
            verify_result = verify_v1_1_baseline_handoff(
                root / "reports" / "v1_1" / "handoff.json",
                out_path=root / "reports" / "v1_1" / "handoff_verify.json",
                markdown_out_path=root / "reports" / "v1_1" / "handoff_verify.md",
            )
            self.assertTrue(verify_result["complete"])
            self.assertTrue((root / "reports" / "v1_1" / "handoff_verify.md").exists())
            transfer = write_v1_1_baseline_transfer_manifest(
                root / "reports" / "v1_1" / "handoff.json",
                out_path=root / "reports" / "v1_1" / "transfer.json",
                markdown_out_path=root / "reports" / "v1_1" / "transfer.md",
                files_out_path=root / "reports" / "v1_1" / "transfer.files",
            )
            self.assertTrue(transfer["complete"])
            self.assertIn(str(chunks_path), transfer["files"])
            self.assertEqual(
                transfer["benchmark_split_files"],
                [
                    str(base_derived / "code2test.jsonl"),
                    str(derived / "comment2context.jsonl"),
                    str(derived / "trace2code.jsonl"),
                ],
            )
            self.assertEqual(len(transfer["benchmark_split_file_fingerprints"]), 3)
            self.assertIn(str(derived / "comment2context.jsonl"), transfer["files"])
            self.assertIn(str(derived / "trace2code.jsonl"), transfer["files"])
            self.assertIn(str(base_derived / "code2test.jsonl"), transfer["files"])
            self.assertIn(str(root / "reports" / "v1_1" / "transfer.json"), transfer["files"])
            self.assertIn(str(root / "reports" / "v1_1" / "transfer.md"), transfer["files"])
            self.assertIn(str(root / "reports" / "v1_1" / "transfer.files"), transfer["files"])
            self.assertEqual(
                transfer["generated_output_files"],
                [
                    str(root / "reports" / "v1_1" / "transfer.files"),
                    str(root / "reports" / "v1_1" / "transfer.json"),
                    str(root / "reports" / "v1_1" / "transfer.md"),
                ],
            )
            self.assertEqual(transfer["chunk_count"], 1)
            self.assertTrue((root / "reports" / "v1_1" / "transfer.files").exists())
            stable_transfer_payload = json.loads((root / "reports" / "v1_1" / "transfer.json").read_text(encoding="utf-8"))
            stable_transfer_payload["generated_at"] = "2026-01-01T00:00:00+00:00"
            (root / "reports" / "v1_1" / "transfer.json").write_text(
                json.dumps(stable_transfer_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stable_transfer = write_v1_1_baseline_transfer_manifest(
                root / "reports" / "v1_1" / "handoff.json",
                out_path=root / "reports" / "v1_1" / "transfer.json",
                markdown_out_path=root / "reports" / "v1_1" / "transfer.md",
                files_out_path=root / "reports" / "v1_1" / "transfer.files",
            )
            self.assertEqual(stable_transfer["generated_at"], "2026-01-01T00:00:00+00:00")
            self.assertIn("2026-01-01T00:00:00+00:00", (root / "reports" / "v1_1" / "transfer.md").read_text(encoding="utf-8"))
            helper_file = root / "reports" / "v1_1" / "sample_shards" / "sample_ids_shard00.txt"
            helper_file.parent.mkdir(parents=True, exist_ok=True)
            helper_file.write_text("sample-1\n", encoding="utf-8")
            pycache_file = helper_file.parent / "__pycache__" / "ignored.pyc"
            pycache_file.parent.mkdir(parents=True, exist_ok=True)
            pycache_file.write_bytes(b"bytecode")
            copy_packet_json = helper_file.parent / "external_runner_copy_packet_v19.json"
            copy_packet_markdown = helper_file.parent / "external_runner_copy_packet_v19.md"
            copy_packet_json.write_text("{}\n", encoding="utf-8")
            copy_packet_markdown.write_text("# copy packet\n", encoding="utf-8")
            helper_report = root / "reports" / "v1_1" / "baseline_shard_commands.md"
            helper_report.write_text("commands\n", encoding="utf-8")
            transfer_with_helpers = write_v1_1_baseline_transfer_manifest(
                root / "reports" / "v1_1" / "handoff.json",
                out_path=root / "reports" / "v1_1" / "transfer_helpers.json",
                include_paths=[helper_file.parent, helper_report, copy_packet_json],
            )
            self.assertTrue(transfer_with_helpers["complete"])
            self.assertIn(str(helper_file), transfer_with_helpers["files"])
            self.assertNotIn(str(pycache_file), transfer_with_helpers["files"])
            self.assertNotIn(str(copy_packet_json), transfer_with_helpers["files"])
            self.assertNotIn(str(copy_packet_markdown), transfer_with_helpers["files"])
            self.assertIn(str(helper_report), transfer_with_helpers["files"])
            self.assertEqual(transfer_with_helpers["included_file_count"], 2)
            helper_fingerprints = {item["path"]: item for item in transfer_with_helpers["included_file_fingerprints"]}
            self.assertRegex(helper_fingerprints[str(helper_file)]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(helper_fingerprints[str(helper_report)]["sha256"], r"^[0-9a-f]{64}$")
            ignored_entries = [entry for entry in transfer_with_helpers["include_entries"] if entry.get("type") == "ignored"]
            self.assertEqual([entry["path"] for entry in ignored_entries], [str(copy_packet_json)])
            transfer_verify = verify_v1_1_baseline_transfer_manifest(
                root / "reports" / "v1_1" / "transfer_helpers.json",
                out_path=root / "reports" / "v1_1" / "transfer_verify.json",
                markdown_out_path=root / "reports" / "v1_1" / "transfer_verify.md",
            )
            self.assertTrue(transfer_verify["complete"])
            split_check = next(check for check in transfer_verify["checks"] if check["name"] == "benchmark_split_file_fingerprints_match")
            self.assertEqual(split_check["benchmark_split_file_count"], 3)
            self.assertTrue((root / "reports" / "v1_1" / "transfer_verify.md").exists())
            original_trace_split = (derived / "trace2code.jsonl").read_text(encoding="utf-8")
            write_jsonl(derived / "trace2code.jsonl", [{"id": "changed-sample", "task_type": "trace2code"}])
            failed_split_verify = verify_v1_1_baseline_transfer_manifest(root / "reports" / "v1_1" / "transfer_helpers.json")
            self.assertFalse(failed_split_verify["complete"])
            self.assertIn("benchmark_split_file_fingerprints_match", failed_split_verify["failed_checks"])
            (derived / "trace2code.jsonl").write_text(original_trace_split, encoding="utf-8")
            helper_report.write_text("changed commands\n", encoding="utf-8")
            failed_transfer_verify = verify_v1_1_baseline_transfer_manifest(root / "reports" / "v1_1" / "transfer_helpers.json")
            self.assertFalse(failed_transfer_verify["complete"])
            self.assertIn("included_file_fingerprints_match", failed_transfer_verify["failed_checks"])

            write_jsonl(
                derived / "samples.jsonl",
                [
                    {"id": "sample-1", "task_type": "comment2context"},
                    {"id": "sample-2", "task_type": "trace2code"},
                    {"id": "sample-3", "task_type": "trace2code"},
                ],
            )
            failed_verify = verify_v1_1_baseline_handoff(root / "reports" / "v1_1" / "handoff.json")
            self.assertFalse(failed_verify["complete"])
            self.assertIn("derived_samples", failed_verify["failed_checks"])
            self.assertTrue((root / "reports" / "v1_1" / "handoff.md").exists())

    def test_baseline_handoff_can_emit_return_acceptance_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_1"
            base_derived = root / "benchmark" / "v1"
            corpus = root / "corpus" / "v1_1"
            derived.mkdir(parents=True)
            base_derived.mkdir(parents=True)
            corpus.mkdir(parents=True)
            write_jsonl(derived / "samples.jsonl", [{"id": "sample-1", "task_type": "trace2code"}])
            write_jsonl(base_derived / "samples.jsonl", [{"id": "base-1", "task_type": "code2test"}])
            write_json(derived / "manifest.json", {"accepted_expansion": 1})
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(chunks_path, [{"repo": "o/r", "base_commit": "base", "path": "src/a.py", "kind": "file"}])
            write_jsonl(
                corpus / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path), "chunk_count": 1, "file_count": 1}],
            )
            report_dir = root / "reports" / "v1_1"
            result = write_v1_1_baseline_handoff(
                derived=derived,
                corpus=corpus,
                eval_dir=root / "eval" / "v1_1",
                cache_root=root / "embeddings" / "v1_1",
                report_dir=report_dir,
                out_path=report_dir / "handoff.json",
                markdown_out_path=report_dir / "handoff.md",
                base_derived=base_derived,
                return_manifest_path=report_dir / "baseline_return_manifest_v19.json",
                return_acceptance_path=report_dir / "baseline_return_acceptance_v19.json",
                return_acceptance_markdown_path=report_dir / "baseline_return_acceptance_v19.md",
            )

            joined_setup = "\n".join(result["setup_commands"])
            self.assertIn("v1-1-baseline-return-acceptance", joined_setup)
            self.assertIn("baseline_return_acceptance_v19.json", joined_setup)
            self.assertIn("--completion-json", joined_setup)
            self.assertNotIn(f"--include {report_dir / 'completion_audit.md'}", joined_setup)
            self.assertNotIn(f"--include {report_dir / 'completion_audit.json'}", joined_setup)
            self.assertNotIn(f"--include {report_dir / 'baseline_return_acceptance_v19.json'}", joined_setup)
            self.assertNotIn(f"--include {report_dir / 'baseline_return_acceptance_v19.md'}", joined_setup)
            self.assertEqual(result["return_acceptance"]["report"], str(report_dir / "baseline_return_acceptance_v19.json"))
            self.assertNotIn(str(report_dir / "completion_audit.md"), result["transfer_includes"])
            self.assertNotIn(str(report_dir / "completion_audit.json"), result["transfer_includes"])
            self.assertNotIn(str(report_dir / "baseline_return_acceptance_v19.json"), result["transfer_includes"])
            self.assertNotIn(str(report_dir / "baseline_return_acceptance_v19.md"), result["transfer_includes"])

    def test_external_runner_preflight_allows_missing_sender_side_acceptance_without_copy_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "data" / "reports" / "v1_1"
            report_dir.mkdir(parents=True)
            baseline_status = report_dir / "baseline_status_v19.json"
            return_acceptance = report_dir / "baseline_return_acceptance_v19.json"
            return_manifest = report_dir / "baseline_return_manifest_v19.json"
            transfer_verify = report_dir / "baseline_transfer_manifest_verify_v19.json"
            handoff_verify = report_dir / "baseline_handoff_verify_v19.json"
            runner = report_dir / "run.sh"
            runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            runner.chmod(0o755)
            runtime = {
                "cuda_available": False,
                "numpy": {"installed": True},
                "sentence_transformers": {"installed": True},
                "torch": {"installed": True},
                "voyage_api_key_set": False,
                "nvidia_smi": {"available": False},
            }
            write_json(baseline_status, {"complete": False, "runtime": runtime, "blocking_baselines": ["voyage-code-3"]})
            write_json(return_manifest, {"generated_at": "2026-05-15T00:00:00+00:00", "artifacts_complete": False})
            write_json(transfer_verify, {"complete": True, "failed_checks": []})
            write_json(handoff_verify, {"complete": True, "failed_checks": []})

            result = write_v1_1_external_runner_preflight_report(
                baseline_status_path=baseline_status,
                return_acceptance_path=return_acceptance,
                return_manifest_path=return_manifest,
                transfer_manifest_verify_path=transfer_verify,
                handoff_verify_path=handoff_verify,
                full_runner_path=runner,
                gpu_runner_path=runner,
                voyage_runner_path=runner,
            )

            acceptance = result["artifact_checks"]["return_acceptance"]
            self.assertTrue(result["return_acceptance_ready"])
            self.assertTrue(acceptance["complete"])
            self.assertTrue(acceptance["skipped"])
            self.assertEqual(acceptance["reason"], "sender_side_gate_snapshot_not_transferred")

    def test_handoff_workflow_evidence_excludes_return_acceptance_cycle(self):
        evidence = handoff_workflow_evidence_paths(
            {
                "handoff_verification": {"report": "reports/baseline_handoff_verify.json"},
                "transfer_unpack_script": {
                    "transfer_verify": "reports/baseline_transfer_unpack_smoke.json",
                    "handoff_verify": "reports/baseline_handoff_unpack_smoke.json",
                },
                "finalization": {
                    "workflow_evidence": [
                        "reports/baseline_handoff.json",
                        "reports/baseline_transfer_unpack_smoke.json",
                        "reports/baseline_return_manifest.json",
                    ]
                },
                "return_acceptance": {
                    "report": "reports/baseline_return_acceptance.json",
                    "markdown": "reports/baseline_return_acceptance.md",
                },
            }
        )

        self.assertEqual(
            evidence,
            [
                Path("reports/baseline_handoff_verify.json"),
                Path("reports/baseline_transfer_unpack_smoke.json"),
                Path("reports/baseline_handoff_unpack_smoke.json"),
                Path("reports/baseline_handoff.json"),
                Path("reports/baseline_return_manifest.json"),
            ],
        )
        self.assertNotIn(Path("reports/baseline_return_acceptance.json"), evidence)

    def test_frozen_v1_fingerprint_status_matches_current_release(self):
        if not Path("data/benchmark/v1/samples.jsonl").exists():
            self.skipTest("frozen V1 fixture is not available")

        status = frozen_v1_fingerprint_status({"inputs": {"base_samples": ["data/benchmark/v1/samples.jsonl"]}})

        self.assertTrue(status["applicable"])
        self.assertTrue(status["complete"])
        self.assertFalse([check for check in status["checks"] if check["mismatches"]])

        absolute_status = frozen_v1_fingerprint_status(
            {"inputs": {"base_samples": [str(Path("data/benchmark/v1/samples.jsonl").resolve())]}}
        )
        self.assertTrue(absolute_status["applicable"])
        self.assertTrue(absolute_status["complete"])

    def test_frozen_v1_fingerprint_status_ignores_noncanonical_fixtures(self):
        status = frozen_v1_fingerprint_status({"inputs": {"base_samples": ["/tmp/project/benchmark/v1/samples.jsonl"]}})

        self.assertFalse(status["applicable"])
        self.assertTrue(status["complete"])

    def test_v1_1_transfer_bundle_creator_packages_manifest_files_and_verifies(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed transfer bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            (root / "unpack_v19_transfer_bundle.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (root / "run_v19_baseline_shards.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (root / "package_v19_return_artifacts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (root / "apply_v19_return_artifacts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            write_json(
                root / "return_manifest.json",
                {
                    "required_files": [
                        "eval/jina-code-embeddings-0.5b_details.jsonl",
                        "eval/jina-code-embeddings-0.5b_summary.json",
                    ]
                },
            )
            write_json(
                root / "handoff.json",
                {
                    "finalization": {"return_manifest": "return_manifest.json"},
                    "jobs": [
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "artifacts": {
                                "details": "eval/jina-code-embeddings-0.5b_details.jsonl",
                                "summary": "eval/jina-code-embeddings-0.5b_summary.json",
                            },
                        }
                    ],
                },
            )
            bundle_members = [
                "a.txt",
                "b.txt",
                "unpack_v19_transfer_bundle.sh",
                "run_v19_baseline_shards.sh",
                "package_v19_return_artifacts.sh",
                "apply_v19_return_artifacts.sh",
                "handoff.json",
                "return_manifest.json",
            ]
            write_json(root / "transfer.json", {"handoff": "handoff.json", "files": bundle_members})

            result = create_v1_1_baseline_transfer_bundle(
                manifest_path=root / "transfer.json",
                bundle_path=Path("transfer.tar.zst"),
                checksum_path=Path("transfer.tar.zst.sha256"),
                archive_members_path=Path("transfer.members"),
                bundle_files_path=Path("transfer.files"),
                out_path=Path("transfer_bundle.json"),
                markdown_out_path=Path("transfer_bundle.md"),
                verify_out_path=Path("transfer_bundle_verify.json"),
                verify_markdown_out_path=Path("transfer_bundle_verify.md"),
                work_dir=root,
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_checks"], [])
            self.assertRegex(result["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(result["size_bytes"], 0)
            self.assertEqual(
                (root / "transfer.files").read_text(encoding="utf-8").splitlines(),
                sorted(bundle_members),
            )
            self.assertTrue((root / "transfer.tar.zst").exists())
            self.assertRegex((root / "transfer.tar.zst.sha256").read_text(encoding="utf-8"), r"^[0-9a-f]{64}  transfer\.tar\.zst")
            saved_report = json.loads((root / "transfer_bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["sha256"], result["sha256"])
            self.assertEqual(saved_report["size_bytes"], result["size_bytes"])
            self.assertEqual(
                (root / "transfer.members").read_text(encoding="utf-8").splitlines(),
                sorted(bundle_members),
            )
            self.assertEqual(result["transfer_unpack_scripts"], ["unpack_v19_transfer_bundle.sh"])
            self.assertEqual(result["external_acceptance"]["external_baselines"], ["jina-code-embeddings-0.5b"])
            self.assertEqual(result["external_acceptance"]["required_return_file_count"], 2)
            self.assertEqual(
                list(result["external_acceptance"]["required_return_files_by_baseline"]),
                ["jina-code-embeddings-0.5b"],
            )
            self.assertIn("--require-complete", result["external_acceptance"]["completion_gate"])
            self.assertIn("run_v19_baseline_shards.sh", result["external_acceptance"]["run_scripts"])
            self.assertTrue(result["verification"]["complete"])
            self.assertTrue((root / "transfer_bundle.json").exists())
            self.assertTrue((root / "transfer_bundle.md").exists())
            self.assertTrue((root / "transfer_bundle_verify.json").exists())
            self.assertTrue((root / "transfer_bundle_verify.md").exists())
            transfer_markdown = (root / "transfer_bundle.md").read_text(encoding="utf-8")
            self.assertIn("Acceptance Checklist", transfer_markdown)
            self.assertIn("eval/jina-code-embeddings-0.5b_details.jsonl", transfer_markdown)
            self.assertIn("Required files by baseline", transfer_markdown)
            self.assertIn("- `jina-code-embeddings-0.5b`: `2`", transfer_markdown)
            self.assertIn("run_v19_baseline_shards.sh", transfer_markdown)
            self.assertIn("overall_status=complete", transfer_markdown)
            self.assertIn("--require-complete", transfer_markdown)
            self.assertIn("Preferred bootstrap", transfer_markdown)
            self.assertIn("unpack_v19_transfer_bundle.sh", transfer_markdown)
            self.assertIn("pass local basenames instead", transfer_markdown)
            self.assertIn(
                "bash unpack_v19_transfer_bundle.sh transfer.tar.zst transfer.tar.zst.sha256",
                transfer_markdown,
            )
            self.assertIn("preserving the generated bundle path", transfer_markdown)
            self.assertIn("do not run `sha256sum -c`", transfer_markdown)
            self.assertIn("v1-1-verify-transfer-manifest", transfer_markdown)

    def test_v1_1_external_runner_copy_packet_tracks_transfer_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "baseline_transfer_bundle.tar.zst"
            checksum = root / "baseline_transfer_bundle.tar.zst.sha256"
            unpack = root / "unpack_v19_transfer_bundle.sh"
            bundle.write_bytes(b"bundle")
            checksum.write_text("0" * 64 + "  baseline_transfer_bundle.tar.zst\n", encoding="utf-8")
            unpack.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            transfer_report = root / "transfer_bundle.json"
            write_json(
                transfer_report,
                {
                    "generated_at": "2026-05-15T00:00:00+00:00",
                    "complete": True,
                    "bundle": str(bundle),
                    "checksum": str(checksum),
                    "markdown": str(root / "transfer_bundle.md"),
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                    "file_count": 7,
                    "external_acceptance": {
                        "external_baselines": ["voyage-code-3"],
                        "required_return_file_count": 2,
                        "required_return_files": [
                            "eval/details.jsonl",
                            "eval/summary.json",
                        ],
                        "required_return_files_by_baseline": {
                            "voyage-code-3": ["eval/details.jsonl", "eval/summary.json"],
                        },
                    },
                },
            )

            result = write_v1_1_external_runner_copy_packet(
                transfer_bundle_report_path=transfer_report,
                unpack_script_path=unpack,
                out_path=root / "copy_packet.json",
                markdown_out_path=root / "copy_packet.md",
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["bundle_sha256"], "a" * 64)
            self.assertEqual(result["bundle_size_bytes"], 123)
            self.assertEqual(result["transfer_file_count"], 7)
            self.assertEqual(result["required_external_baselines"], ["voyage-code-3"])
            self.assertEqual(
                result["required_return_files"],
                ["eval/details.jsonl", "eval/summary.json"],
            )
            self.assertEqual(
                result["required_return_files_by_baseline"],
                {
                    "voyage-code-3": [
                        "eval/details.jsonl",
                        "eval/summary.json",
                    ]
                },
            )
            self.assertEqual(result["bundle_path"], str(bundle))
            self.assertEqual(result["checksum_path"], str(checksum))
            self.assertEqual(
                result["copy_to_external_runner"],
                [str(unpack), str(bundle), str(checksum)],
            )
            self.assertIn("--copy-packet", result["sender_preflight_command"])
            self.assertIn("v1-1-external-runner-preflight", result["sender_preflight_command"])
            self.assertIn("baseline_transfer_bundle.tar.zst", result["external_runner_first_command"])
            self.assertNotIn(str(root), result["external_runner_first_command"])
            self.assertIn("pip install -e '.[embedding]'", result["external_runner_after_unpack"])
            self.assertIn("v1-1-external-runner-preflight", " ".join(result["external_runner_after_unpack"]))
            self.assertNotIn("cuda_jina_qwen", result["split_runner_scripts"])
            self.assertEqual(
                result["split_runner_scripts"]["api_voyage"],
                "data/reports/v1_1/run_v19_voyage_baseline_shards.sh",
            )
            self.assertIn("overall_status=complete", result["completion_gate"])
            self.assertEqual(result["completion_json"], "data/reports/v1_1/completion_audit_v19.json")
            self.assertEqual(result["completion_required_status"], "overall_status=complete")
            self.assertIn(str(unpack), result["copy_to_external_runner"])
            markdown = (root / "copy_packet.md").read_text(encoding="utf-8")
            self.assertIn("External Runner Copy Packet", markdown)
            self.assertIn("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", markdown)
            self.assertIn("Sender-Side Check", markdown)
            self.assertIn("First External Command", markdown)
            self.assertIn("Required Return Files", markdown)
            self.assertIn("Required Return Files By Baseline", markdown)
            self.assertIn("`voyage-code-3`: `2`", markdown)
            self.assertIn("eval/summary.json", markdown)
            self.assertNotIn("run_v19_gpu_baseline_shards.sh", markdown)
            self.assertIn("run_v19_voyage_baseline_shards.sh", markdown)
            self.assertIn("`data/reports/v1_1/completion_audit_v19.json` reporting `overall_status=complete`", markdown)
            self.assertNotIn("completion_audit_v19.json must report overall_status=complete", markdown)

    def test_v1_1_transfer_unpack_script_verifies_checksum_and_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_path = root / "unpack_transfer.sh"
            markdown_path = root / "unpack_transfer.md"

            result = write_v1_1_baseline_transfer_unpack_script(
                bundle_path=Path("baseline_transfer_bundle.tar.zst"),
                checksum_path=Path("baseline_transfer_bundle.tar.zst.sha256"),
                manifest_path=Path("data/reports/v1_1/baseline_transfer_manifest.json"),
                handoff_path=Path("data/reports/v1_1/baseline_handoff.json"),
                destination="external-checkout",
                transfer_verify_path=Path("data/reports/v1_1/transfer_unpack_smoke.json"),
                handoff_verify_path=Path("data/reports/v1_1/handoff_unpack_smoke.json"),
                out_path=script_path,
                markdown_out_path=markdown_path,
            )

            script = script_path.read_text(encoding="utf-8")
            self.assertTrue(result["complete"])
            self.assertTrue(script_path.stat().st_mode & 0o111)
            self.assertTrue(markdown_path.exists())
            self.assertEqual(result["destination"], "external-checkout")
            self.assertIn("Transfer bundle checksum mismatch", script)
            self.assertIn('sha256sum "$bundle"', script)
            self.assertIn("Transfer bundle contains unsafe tar member paths", script)
            self.assertIn('tar --use-compress-program=zstd -tvf "$bundle"', script)
            self.assertIn("Transfer bundle contains non-regular file members", script)
            self.assertIn("regular file transfer bundle members", script)
            self.assertIn("v1-1-verify-transfer-manifest", script)
            self.assertIn("v1-1-verify-handoff", script)
            self.assertIn("transfer_unpack_smoke.json", script)
            self.assertIn("handoff_unpack_smoke.json", script)
            self.assertIn("Transfer bundle verified and unpacked", script)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Baseline Transfer Unpack Script", markdown)
            self.assertIn("external-checkout", markdown)
            self.assertIn("supplied bundle path", markdown)
            self.assertIn("repo-relative bundle path", markdown)

    def test_v1_1_transfer_bundle_creator_rejects_unsafe_manifest_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "transfer.json", {"files": ["../outside.txt"]})

            result = create_v1_1_baseline_transfer_bundle(
                manifest_path=root / "transfer.json",
                bundle_path=Path("transfer.tar.zst"),
                out_path=Path("unsafe_bundle.json"),
                work_dir=root,
            )

            self.assertFalse(result["complete"])
            self.assertIn("manifest_members_are_safe_paths", result["failed_checks"])
            self.assertIn("bundle_created", result["failed_checks"])
            self.assertFalse((root / "transfer.tar.zst").exists())
            self.assertTrue((root / "unsafe_bundle.json").exists())

    def test_v1_1_transfer_bundle_creator_rejects_copy_packet_manifest_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copy_packet = root / "external_runner_copy_packet_v19.json"
            copy_packet.write_text("{}\n", encoding="utf-8")
            write_json(root / "transfer.json", {"files": ["external_runner_copy_packet_v19.json"]})

            result = create_v1_1_baseline_transfer_bundle(
                manifest_path=root / "transfer.json",
                bundle_path=Path("transfer.tar.zst"),
                out_path=Path("copy_packet_bundle.json"),
                work_dir=root,
            )

            self.assertFalse(result["complete"])
            self.assertIn("manifest_excludes_external_runner_copy_packet", result["failed_checks"])
            self.assertIn("bundle_created", result["failed_checks"])
            self.assertEqual(result["copy_packet_members"], ["external_runner_copy_packet_v19.json"])
            self.assertFalse((root / "transfer.tar.zst").exists())

    def test_v1_1_transfer_bundle_creator_rejects_symlink_manifest_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            write_json(root / "transfer.json", {"files": ["link.txt"]})

            result = create_v1_1_baseline_transfer_bundle(
                manifest_path=root / "transfer.json",
                bundle_path=Path("transfer.tar.zst"),
                out_path=Path("symlink_bundle.json"),
                work_dir=root,
            )

            self.assertFalse(result["complete"])
            self.assertIn("manifest_members_are_regular_files", result["failed_checks"])
            self.assertIn("bundle_created", result["failed_checks"])
            self.assertFalse((root / "transfer.tar.zst").exists())

    def test_v1_1_transfer_bundle_verifier_checks_checksum_and_members(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed transfer bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            manifest = root / "transfer.json"
            bundle = root / "transfer.tar.zst"
            checksum = root / "transfer.tar.zst.sha256"
            members = root / "transfer.members"
            write_json(manifest, {"files": ["a.txt", "b.txt"]})
            subprocess.run(
                ["tar", "--use-compress-program=zstd", "-cf", str(bundle), "a.txt", "b.txt"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            checksum.write_text(f"{digest}  {bundle}\n", encoding="utf-8")

            result = verify_v1_1_baseline_transfer_bundle(
                bundle_path=bundle,
                manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=members,
                out_path=root / "bundle_verify.json",
                markdown_out_path=root / "bundle_verify.md",
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_checks"], [])
            self.assertEqual(result["sha256"], digest)
            self.assertGreater(result["size_bytes"], 0)
            self.assertEqual(members.read_text(encoding="utf-8").splitlines(), ["a.txt", "b.txt"])
            saved_report = json.loads((root / "bundle_verify.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["sha256"], digest)
            self.assertEqual(saved_report["size_bytes"], result["size_bytes"])
            self.assertIn("SHA256", (root / "bundle_verify.md").read_text(encoding="utf-8"))
            member_check = [item for item in result["checks"] if item["name"] == "bundle_members_match_manifest"][0]
            self.assertEqual(member_check["actual_member_count"], 2)
            safe_check = [item for item in result["checks"] if item["name"] == "bundle_members_are_safe_paths"][0]
            self.assertEqual(safe_check["unsafe_member_count"], 0)
            regular_check = [item for item in result["checks"] if item["name"] == "bundle_members_are_regular_files"][0]
            self.assertEqual(regular_check["non_regular_member_count"], 0)
            compression_check = [item for item in result["checks"] if item["name"] == "bundle_compression_valid"][0]
            self.assertIsNone(compression_check["error"])
            self.assertIn("stderr", compression_check)
            self.assertTrue((root / "bundle_verify.md").exists())

            copy_packet = root / "external_runner_copy_packet_v19.json"
            copy_packet.write_text("{}\n", encoding="utf-8")
            copy_packet_manifest = root / "copy_packet_transfer.json"
            copy_packet_bundle = root / "copy_packet_transfer.tar.zst"
            copy_packet_checksum = root / "copy_packet_transfer.tar.zst.sha256"
            write_json(copy_packet_manifest, {"files": ["external_runner_copy_packet_v19.json"]})
            subprocess.run(
                ["tar", "--use-compress-program=zstd", "-cf", str(copy_packet_bundle), "external_runner_copy_packet_v19.json"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            copy_packet_checksum.write_text(
                f"{hashlib.sha256(copy_packet_bundle.read_bytes()).hexdigest()}  {copy_packet_bundle}\n",
                encoding="utf-8",
            )
            copy_packet_result = verify_v1_1_baseline_transfer_bundle(
                bundle_path=copy_packet_bundle,
                manifest_path=copy_packet_manifest,
                checksum_path=copy_packet_checksum,
                archive_members_path=root / "copy_packet.members",
            )
            self.assertFalse(copy_packet_result["complete"])
            self.assertIn("bundle_excludes_external_runner_copy_packet", copy_packet_result["failed_checks"])
            copy_packet_check = [
                item for item in copy_packet_result["checks"] if item["name"] == "bundle_excludes_external_runner_copy_packet"
            ][0]
            self.assertEqual(copy_packet_check["copy_packet_members"], ["external_runner_copy_packet_v19.json"])

            stale_manifest = root / "stale_transfer.json"
            write_json(stale_manifest, {"files": ["a.txt"]})
            failed = verify_v1_1_baseline_transfer_bundle(
                bundle_path=bundle,
                manifest_path=stale_manifest,
                checksum_path=checksum,
                archive_members_path=root / "stale.members",
            )
            self.assertFalse(failed["complete"])
            self.assertIn("bundle_members_match_manifest", failed["failed_checks"])

            missing_bundle = verify_v1_1_baseline_transfer_bundle(
                bundle_path=root / "missing.tar.zst",
                manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=root / "missing.members",
            )
            self.assertFalse(missing_bundle["complete"])
            self.assertEqual(
                {check["name"]: check["status"] for check in missing_bundle["checks"]},
                {
                    "bundle_checksum_matches": "fail",
                    "bundle_compression_valid": "fail",
                    "bundle_members_are_safe_paths": "fail",
                    "bundle_members_are_regular_files": "fail",
                    "bundle_excludes_external_runner_copy_packet": "pass",
                    "bundle_members_match_manifest": "fail",
                },
            )

    def test_v1_1_transfer_bundle_verifier_rejects_non_regular_members(self):
        if not shutil.which("zstd"):
            self.skipTest("zstd is required to create a compressed transfer bundle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            link = root / "linked.txt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            manifest = root / "transfer.json"
            bundle = root / "transfer.tar.zst"
            checksum = root / "transfer.tar.zst.sha256"
            members = root / "transfer.members"
            write_json(manifest, {"files": ["linked.txt"]})
            subprocess.run(
                ["tar", "--use-compress-program=zstd", "-cf", str(bundle), "linked.txt"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            checksum.write_text(f"{hashlib.sha256(bundle.read_bytes()).hexdigest()}  {bundle}\n", encoding="utf-8")

            result = verify_v1_1_baseline_transfer_bundle(
                bundle_path=bundle,
                manifest_path=manifest,
                checksum_path=checksum,
                archive_members_path=members,
            )

            self.assertFalse(result["complete"])
            self.assertIn("bundle_members_are_regular_files", result["failed_checks"])
            regular_check = [item for item in result["checks"] if item["name"] == "bundle_members_are_regular_files"][0]
            self.assertEqual(regular_check["non_regular_member_count"], 1)
            self.assertEqual(regular_check["non_regular_members"][0]["type"], "l")

    def test_completion_doc_content_status_requires_release_positioning_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_path = root / "README.md"
            doc_path.write_text("V1.1 docs\n", encoding="utf-8")

            incomplete = completion_doc_content_status([doc_path])

            self.assertFalse(incomplete["complete"])
            self.assertIn("focused_improvement_over_v1", incomplete["missing_markers"])
            self.assertIn("trace2code_expansion", incomplete["missing_markers"])

            doc_path.write_text(V1_1_DOC_TEXT, encoding="utf-8")
            complete = completion_doc_content_status([doc_path])

            self.assertTrue(complete["complete"])
            self.assertEqual(complete["missing_markers"], [])

    def test_completion_audit_maps_blockers_to_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness_path = root / "readiness.json"
            release_path = root / "release.json"
            baseline_status_path = root / "baseline_status.json"
            leaderboard_path = root / "leaderboard.json"
            out_path = root / "completion.md"
            json_out = root / "completion.json"
            doc_path = root / "README.md"
            doc_path.write_text(V1_1_DOC_TEXT, encoding="utf-8")
            corpus_manifest = root / "corpus_manifest.jsonl"
            corpus_manifest.write_text("{}\n", encoding="utf-8")
            leaderboard_markdown = root / "leaderboard.md"
            leaderboard_markdown.write_text("# Leaderboard\n", encoding="utf-8")
            lexical_summary = root / "lexical_summary.json"
            lexical_summary.write_text("{}\n", encoding="utf-8")
            repomap_summary = root / "repomap_summary.json"
            repomap_summary.write_text("{}\n", encoding="utf-8")
            jina_details = root / "jina-code-embeddings-0.5b_details.jsonl"
            jina_summary = root / "jina-code-embeddings-0.5b_summary.json"
            jina_shard_details = root / "jina-code-embeddings-0.5b_shard00_details.jsonl"
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            complete_details = {
                "exists": True,
                "rows": 287,
                "unique_sample_ids": 287,
                "candidate_filters": ["all_files"],
                "metrics_match": True,
                "complete": True,
            }
            missing_details = {"exists": False, "rows": 0, "unique_sample_ids": 0, "candidate_filters": [], "metrics_match": False, "complete": False}
            required_baselines = [
                {
                    "baseline": "lexical",
                    "found": True,
                    "source": str(lexical_summary),
                    "evaluated": 287,
                    "overall_samples": 287,
                    "skipped": {},
                    "complete": True,
                    "details": complete_details,
                },
                {
                    "baseline": "aider-style-repomap",
                    "found": True,
                    "source": str(repomap_summary),
                    "evaluated": 287,
                    "overall_samples": 287,
                    "skipped": {},
                    "complete": True,
                    "details": complete_details,
                },
                {
                    "baseline": "jina-code-embeddings-0.5b",
                    "found": False,
                    "source": None,
                    "evaluated": 0,
                    "overall_samples": 0,
                    "skipped": {},
                    "complete": False,
                    "details": missing_details,
                },
                {
                    "baseline": "qwen3-embedding-4b",
                    "found": False,
                    "source": None,
                    "evaluated": 0,
                    "overall_samples": 0,
                    "skipped": {},
                    "complete": False,
                    "details": missing_details,
                },
                {
                    "baseline": "voyage-code-3",
                    "found": False,
                    "source": None,
                    "evaluated": 0,
                    "overall_samples": 0,
                    "skipped": {},
                    "complete": False,
                    "details": missing_details,
                },
            ]
            gates = {
                "sample_paths_do_not_point_at_benchmark_v1": True,
                "contains_all_base_v1_ids": True,
                "code2test_count_preserved": True,
                "comment2context_count_ge_target": True,
                "comment2context_count_le_target_band": True,
                "trace2code_count_ge_target": True,
                "new_samples_schema_valid": True,
                "new_samples_have_clear_task_semantics": True,
                "new_samples_no_path_role_overlap": True,
                "new_samples_no_query_leakage": True,
                "new_samples_no_direct_gold_hints": True,
                "new_comment2context_have_given_files": True,
                "new_comment2context_no_same_directory_gold": True,
                "new_comment2context_cross_module_count_ge_min": True,
                "new_trace2code_gold_not_tests": True,
                "new_trace2code_has_non_go_gold": True,
                "new_trace2code_non_go_repo_count_ge_min": True,
                "new_trace2code_language_count_ge_min": True,
                "new_trace2code_failure_type_count_ge_min": True,
                "corpus_manifest_exists": True,
                "new_samples_gold_in_corpus": True,
                "new_samples_have_audit_evidence": True,
                "assembly_manifest_exists": True,
                "assembly_manifest_requires_audit_keep": True,
                "assembly_manifest_has_audit_paths": True,
                "leaderboard_reports_contain_required_baselines": False,
            }
            write_json(
                readiness_path,
                {
                    "ready": False,
                    "blocking_gates": ["required_baseline_summaries_complete", "leaderboard_reports_contain_required_baselines"],
                    "samples": 287,
                    "new_samples": 62,
                    "counts_by_task": {"code2test": 106, "comment2context": 80, "trace2code": 101},
                    "base_counts_by_task": {"code2test": 106, "comment2context": 51, "trace2code": 68},
                    "new_counts_by_task": {"comment2context": 29, "trace2code": 33},
                    "target_gaps": {"comment2context_samples": 0, "trace2code_samples": 0, "comment2context_cross_module_samples": 0},
                    "comment2context_new_cross_module_samples": 14,
                    "trace2code_new_non_go_repos": ["pytest-dev/pytest"],
                    "trace2code_new_gold_extensions": [".py", ".rs"],
                    "trace2code_new_failure_types": ["assertion", "panic"],
                    "trace2code_new_unknown_failure_samples": 0,
                    "missing_base_ids": [],
                    "assembly_manifest": {"accepted_expansion": 62},
                    "inputs": {"corpus_manifest": str(corpus_manifest)},
                    "required_baselines": required_baselines,
                    "leaderboard": {
                        "markdown": str(leaderboard_markdown),
                        "missing_baselines": ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"],
                    },
                    "gates": gates,
                },
            )
            write_json(release_path, {"status": "not_ready", "blocking_gates": ["required_baseline_summaries_complete"]})
            write_json(
                baseline_status_path,
                {
                    "complete": False,
                    "baseline_blockers": [
                        {"baseline": "lexical", "complete": True},
                        {"baseline": "aider-style-repomap", "complete": True},
                        {
                            "baseline": "jina-code-embeddings-0.5b",
                            "complete": False,
                            "reason": "blocked_no_cuda_or_precomputed_cache",
                            "next_action": "run jina",
                            "partial_details": {"path": str(jina_details), "exists": False},
                            "shard_details": {"paths": [str(jina_shard_details)]},
                        },
                        {"baseline": "qwen3-embedding-4b", "complete": False, "reason": "blocked_no_cuda_or_precomputed_cache", "next_action": "run qwen"},
                    ],
                },
            )
            write_json(leaderboard_path, {"row_count": 8, "missing_required_baselines": ["jina-code-embeddings-0.5b", "qwen3-embedding-4b"]})

            report = report_v1_1_completion_audit(
                readiness_path=readiness_path,
                release_json_path=release_path,
                baseline_status_path=baseline_status_path,
                leaderboard_json_path=leaderboard_path,
                out_path=out_path,
                json_out_path=json_out,
                docs=[doc_path],
            )

            self.assertEqual(report["overall_status"], "not_complete")
            self.assertTrue(out_path.exists())
            self.assertEqual(report["requirement_count"], len(report["checklist"]))
            self.assertEqual(report["blocked_or_partial_requirement_count"], len(report["blocked_or_partial_requirements"]))
            self.assertTrue(all(item["name"] == item["requirement"] for item in report["checklist"]))
            self.assertTrue(all(item["name"] == item["requirement"] for item in report["blocked_or_partial_requirements"]))
            aggregate_next_action = (
                "Run missing embedding baselines "
                "(jina-code-embeddings-0.5b, qwen3-embedding-4b) on CUDA/API-capable machines, "
                "package and apply a verified return bundle, then rerun v1-1-finalize-baselines."
            )
            self.assertEqual(report["next_required_action"], aggregate_next_action)
            self.assertNotIn("voyage-code-3", "\n".join(report["success_criteria"]))
            blocked = {item["requirement"]: item for item in report["blocked_or_partial_requirements"]}
            passed = {item["requirement"]: item for item in report["checklist"] if item["status"] == "pass"}
            frozen_v1 = passed["Keep frozen V1 unchanged and preserve all V1 IDs"]
            self.assertTrue(frozen_v1["evidence_all_present"])
            self.assertTrue(all(item["exists"] for item in frozen_v1["evidence_status"]))
            self.assertEqual(frozen_v1["evidence_status"][0]["kind"], "file")
            self.assertIn("Run jina-code-embeddings-0.5b all-files baseline with no skipped samples and metric-consistent details", blocked)
            jina_blocked = blocked["Run jina-code-embeddings-0.5b all-files baseline with no skipped samples and metric-consistent details"]
            self.assertFalse(jina_blocked["evidence_all_present"])
            self.assertIn(str(jina_details), jina_blocked["evidence"])
            self.assertIn(str(jina_summary), jina_blocked["evidence"])
            self.assertIn(str(jina_shard_details), jina_blocked["evidence"])
            missing_evidence = {item["path"] for item in jina_blocked["evidence_status"] if not item["exists"]}
            self.assertIn(str(jina_summary), missing_evidence)
            self.assertEqual(jina_blocked["next_action"], "run jina")
            self.assertEqual(report["summary"]["next_required_action"], aggregate_next_action)
            self.assertIn(aggregate_next_action, out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                blocked["Regenerate leaderboard and include every required baseline"]["next_action"],
                "Finish missing required baseline summaries/details, then rerun report-models with all required baselines.",
            )
            docs_item = blocked["Update docs to present V1.1 as focused benchmark improvement over V1"]
            self.assertEqual(docs_item["status"], "partial")
            self.assertTrue(docs_item["details"]["content_markers_complete"])
            self.assertEqual(docs_item["details"]["missing_content_markers"], [])
            self.assertEqual(
                docs_item["next_action"],
                "Finalize public V1.1 docs after the release report is ready.",
            )

    def test_completion_audit_checks_workflow_evidence_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness_path = root / "readiness.json"
            release_path = root / "release.json"
            baseline_status_path = root / "baseline_status.json"
            leaderboard_path = root / "leaderboard.json"
            out_path = root / "completion.md"
            json_out = root / "completion.json"
            doc_path = root / "README.md"
            doc_path.write_text(V1_1_DOC_TEXT, encoding="utf-8")
            workflow_report = root / "return_bundle_verify.json"
            write_json(
                workflow_report,
                {
                    "generated_at": "2026-05-16T00:00:00+00:00",
                    "bundle": "return_bundle.tar.zst",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                    "complete": False,
                    "failed_checks": [
                        "return_bundle_members_are_regular_files",
                        "return_bundle_members_match_return_manifest",
                    ],
                },
            )
            write_json(
                readiness_path,
                {
                    "ready": True,
                    "samples": 287,
                    "new_samples": 62,
                    "counts_by_task": {"code2test": 106, "comment2context": 80, "trace2code": 101},
                    "base_counts_by_task": {"code2test": 106, "comment2context": 51, "trace2code": 68},
                    "new_counts_by_task": {"comment2context": 29, "trace2code": 33},
                    "target_gaps": {},
                    "comment2context_new_cross_module_samples": 1,
                    "trace2code_new_non_go_repos": ["pytest-dev/pytest"],
                    "trace2code_new_gold_extensions": [".py", ".rs"],
                    "trace2code_new_failure_types": ["assertion", "panic"],
                    "trace2code_new_unknown_failure_samples": 0,
                    "missing_base_ids": [],
                    "assembly_manifest": {"accepted_expansion": 62},
                    "inputs": {"corpus_manifest": str(root / "corpus_manifest.jsonl")},
                    "required_baselines": [
                        {
                            "baseline": baseline,
                            "found": True,
                            "source": str(root / f"{baseline}_summary.json"),
                            "evaluated": 287,
                            "overall_samples": 287,
                            "skipped": {},
                            "complete": True,
                            "details": {"complete": True},
                        }
                        for baseline in ["lexical", "aider-style-repomap", "jina-code-embeddings-0.5b", "qwen3-embedding-4b", "voyage-code-3"]
                    ],
                    "leaderboard": {"markdown": str(root / "leaderboard.md"), "missing_baselines": []},
                    "gates": {
                        "sample_paths_do_not_point_at_benchmark_v1": True,
                        "contains_all_base_v1_ids": True,
                        "code2test_count_preserved": True,
                        "comment2context_count_ge_target": True,
                        "comment2context_count_le_target_band": True,
                        "trace2code_count_ge_target": True,
                        "new_samples_schema_valid": True,
                        "new_samples_have_clear_task_semantics": True,
                        "new_samples_no_path_role_overlap": True,
                        "new_samples_no_query_leakage": True,
                        "new_samples_no_direct_gold_hints": True,
                        "new_comment2context_have_given_files": True,
                        "new_comment2context_no_same_directory_gold": True,
                        "new_comment2context_cross_module_count_ge_min": True,
                        "new_trace2code_gold_not_tests": True,
                        "new_trace2code_has_non_go_gold": True,
                        "new_trace2code_non_go_repo_count_ge_min": True,
                        "new_trace2code_language_count_ge_min": True,
                        "new_trace2code_failure_type_count_ge_min": True,
                        "corpus_manifest_exists": True,
                        "new_samples_gold_in_corpus": True,
                        "new_samples_have_audit_evidence": True,
                        "assembly_manifest_exists": True,
                        "assembly_manifest_requires_audit_keep": True,
                        "assembly_manifest_has_audit_paths": True,
                        "leaderboard_reports_contain_required_baselines": True,
                    },
                },
            )
            (root / "corpus_manifest.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "leaderboard.md").write_text("# Leaderboard\n", encoding="utf-8")
            for baseline in ["lexical", "aider-style-repomap", "jina-code-embeddings-0.5b", "qwen3-embedding-4b", "voyage-code-3"]:
                (root / f"{baseline}_summary.json").write_text("{}\n", encoding="utf-8")
            write_json(release_path, {"status": "ready", "blocking_gates": []})
            write_json(baseline_status_path, {"complete": True, "baseline_blockers": []})
            write_json(leaderboard_path, {"row_count": 5, "missing_required_baselines": []})

            report = report_v1_1_completion_audit(
                readiness_path=readiness_path,
                release_json_path=release_path,
                baseline_status_path=baseline_status_path,
                leaderboard_json_path=leaderboard_path,
                out_path=out_path,
                json_out_path=json_out,
                docs=[doc_path],
                workflow_evidence_paths=[workflow_report],
            )

            blocked = {item["requirement"]: item for item in report["blocked_or_partial_requirements"]}
            self.assertIn("External baseline handoff, transfer, and return-bundle workflow evidence", blocked)
            workflow_check = blocked["External baseline handoff, transfer, and return-bundle workflow evidence"]["details"][
                "workflow_report_checks"
            ][0]
            self.assertFalse(workflow_check["complete"])
            self.assertEqual(workflow_check["generated_at"], "2026-05-16T00:00:00+00:00")
            self.assertEqual(workflow_check["bundle"], "return_bundle.tar.zst")
            self.assertEqual(workflow_check["sha256"], "a" * 64)
            self.assertEqual(workflow_check["size_bytes"], 123)
            self.assertEqual(
                workflow_check["failed_checks"],
                ["return_bundle_members_are_regular_files", "return_bundle_members_match_return_manifest"],
            )

            missing = report_v1_1_completion_audit(
                readiness_path=readiness_path,
                release_json_path=release_path,
                baseline_status_path=baseline_status_path,
                leaderboard_json_path=leaderboard_path,
                out_path=out_path,
                json_out_path=json_out,
                docs=[doc_path],
                workflow_evidence_paths=[root / "missing_return_bundle_verify.json"],
            )
            missing_blocked = {item["requirement"]: item for item in missing["blocked_or_partial_requirements"]}
            missing_check = missing_blocked["External baseline handoff, transfer, and return-bundle workflow evidence"]["details"][
                "workflow_report_checks"
            ][0]
            self.assertFalse(missing_check["exists"])
            self.assertEqual(missing_check["reason"], "missing")

            write_json(workflow_report, {"complete": True, "failed_checks": []})
            fixed = report_v1_1_completion_audit(
                readiness_path=readiness_path,
                release_json_path=release_path,
                baseline_status_path=baseline_status_path,
                leaderboard_json_path=leaderboard_path,
                out_path=out_path,
                json_out_path=json_out,
                docs=[doc_path],
                workflow_evidence_paths=[workflow_report],
            )
            self.assertNotIn(
                "External baseline handoff, transfer, and return-bundle workflow evidence",
                [item["requirement"] for item in fixed["blocked_or_partial_requirements"]],
            )

    def test_v1_1_audit_packet_writes_preflight_diagnostics_and_audit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            candidates = root / "candidates" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            out = root / "reports" / "audit_packet"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            good_trace = sample(
                "good-trace",
                "trace2code",
                {"trace": "Traceback: AssertionError in scheduler fairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            duplicate = sample(
                "old-code",
                "trace2code",
                {"trace": "Traceback: AssertionError in duplicate"},
                {"root_cause_files": ["src/runtime/duplicate.py"]},
            )
            leaked = sample(
                "leaked",
                "comment2context",
                {"path": "src/auth/session.py", "review_comment": "Please inspect tests/auth/test_session.py"},
                {
                    "given_files": ["src/auth/session.py"],
                    "must_context_files": [{"path": "tests/auth/test_session.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/auth/test_session.py"],
                },
            )
            pr_only = sample(
                "pr-only",
                "trace2code",
                {"trace": "Traceback: RuntimeError in worker pool"},
                {"root_cause_files": ["src/runtime/worker.py"]},
            )
            pr_only["metadata"] = {"pr_url": "https://github.com/o/r/pull/99"}
            no_signal = sample(
                "no-signal",
                "trace2code",
                {"command": "pytest tests/test_worker.py"},
                {"root_cause_files": ["src/runtime/no_signal.py"]},
            )
            write_jsonl(base_samples, [old_code])
            write_jsonl(candidates, [good_trace, duplicate, leaked, pr_only, no_signal])
            write_jsonl(chunks_path, [{"repo": "o/r", "base_commit": "base", "path": "src/runtime/scheduler.py", "kind": "file"}])
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            result = write_v1_1_audit_packet(
                candidate_sources=[candidates],
                out_dir=out,
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                require_corpus=True,
            )
            diagnostics = [json.loads(line) for line in (out / "candidate_diagnostics.jsonl").read_text().splitlines()]
            audit_rows = [json.loads(line) for line in (out / "audit_samples.jsonl").read_text().splitlines()]
            csv_text = (out / "audit_samples.csv").read_text(encoding="utf-8")

            self.assertEqual(result["diagnostics"], 5)
            self.assertEqual(result["audit_rows"], 1)
            self.assertEqual(result["preflight_counts"]["candidate"], 1)
            self.assertEqual(result["preflight_counts"]["duplicate_base_id"], 1)
            self.assertEqual(result["preflight_counts"]["direct_gold_hint"], 1)
            self.assertEqual(result["preflight_counts"]["missing_audit_evidence"], 1)
            self.assertEqual(result["preflight_counts"]["unclear_task_semantics"], 1)
            self.assertEqual([row["sample_id"] for row in audit_rows], ["good-trace"])
            self.assertEqual(audit_rows[0]["trace_failure_types"], "assertion; traceback")
            self.assertIn("preflight_reason", csv_text)
            self.assertEqual([row["preflight_status"] for row in diagnostics], ["candidate", "drop", "drop", "drop", "drop"])

    def test_assemble_v1_1_preserves_base_and_filters_expansion_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            expansion = root / "candidates" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            out_dir = root / "benchmark" / "v1_1"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            good_comment = sample(
                "good-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["cross_module_contract"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            good_trace = sample(
                "good-trace",
                "trace2code",
                {"trace": "Traceback: AssertionError in TestSchedulerFairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            duplicate_base = sample(
                "old-code",
                "comment2context",
                {"path": "src/other.py", "review_comment": "Needs context"},
                {
                    "given_files": ["src/other.py"],
                    "must_context_files": [{"path": "tests/other/test_context.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/other/test_context.py"],
                },
            )
            excluded_code = sample("new-code", "code2test", {"changed_file": "src/new.py"}, {"related_tests": ["tests/test_new.py"]})
            direct_hint = sample(
                "direct-hint",
                "comment2context",
                {"path": "src/auth/session.py", "review_comment": "Please update session_helper.py too"},
                {
                    "given_files": ["src/auth/session.py"],
                    "must_context_files": [{"path": "src/auth/session_helper.py", "evidence": ["manual"]}],
                    "root_cause_files": ["src/auth/session_helper.py"],
                },
            )
            missing_evidence = sample(
                "missing-evidence",
                "trace2code",
                {"trace": "Traceback: RuntimeError in worker"},
                {"root_cause_files": ["src/worker.py"]},
            )
            missing_evidence["metadata"] = {}

            write_jsonl(base_samples, [old_code])
            write_jsonl(expansion, [good_comment, good_trace, duplicate_base, excluded_code, direct_hint, missing_evidence])
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in ["tests/test_auth.py", "tests/web/test_sanitizer.py", "src/runtime/scheduler.py"]
                ],
            )
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            result = assemble_v1_1_benchmark(
                base_sample_paths=[base_samples],
                expansion_sources=[expansion],
                out_dir=out_dir,
                corpus_manifest_path=corpus_manifest,
                require_corpus=True,
            )
            rows = [json.loads(line) for line in (out_dir / "samples.jsonl").read_text().splitlines()]
            manifest = json.loads((out_dir / "manifest.json").read_text())

            self.assertEqual(result["base_total"], 1)
            self.assertEqual(result["accepted_expansion"], 2)
            self.assertEqual([row["id"] for row in rows], ["old-code", "good-comment", "good-trace"])
            self.assertEqual(manifest["counts_by_task"], {"code2test": 1, "comment2context": 1, "trace2code": 1})
            self.assertEqual(manifest["new_counts_by_task"], {"comment2context": 1, "trace2code": 1})
            self.assertEqual(manifest["dropped"]["duplicate_base_id"], 1)
            self.assertEqual(manifest["dropped"]["excluded_task"], 1)
            self.assertEqual(manifest["dropped"]["direct_gold_hint"], 1)
            self.assertEqual(manifest["dropped"]["missing_audit_evidence"], 1)
            self.assertEqual(len((out_dir / "comment2context.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((out_dir / "trace2code.jsonl").read_text().splitlines()), 1)

    def test_assemble_v1_1_can_require_manual_audit_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            expansion = root / "candidates" / "samples.jsonl"
            audit = root / "reports" / "audit_samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            out_dir = root / "benchmark" / "v1_1"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            unaudited_comment = sample(
                "unaudited-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            kept_trace = sample(
                "kept-trace",
                "trace2code",
                {"trace": "Traceback: AssertionError in TestSchedulerFairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            write_jsonl(base_samples, [old_code])
            write_jsonl(expansion, [unaudited_comment, kept_trace])
            write_jsonl(
                audit,
                [
                    {"sample_id": "unaudited-comment", "task_type": "comment2context", "repo": "o/r", "verdict": "valid", "keep": "no"},
                    {"sample_id": "kept-trace", "task_type": "trace2code", "repo": "o/r", "verdict": "valid", "keep": "yes"},
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in ["tests/test_auth.py", "tests/web/test_sanitizer.py", "src/runtime/scheduler.py"]
                ],
            )
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            result = assemble_v1_1_benchmark(
                base_sample_paths=[base_samples],
                expansion_sources=[expansion],
                out_dir=out_dir,
                corpus_manifest_path=corpus_manifest,
                require_corpus=True,
                audit_paths=[audit],
                require_audit_keep=True,
            )
            rows = [json.loads(line) for line in (out_dir / "samples.jsonl").read_text().splitlines()]

            self.assertEqual(result["accepted_expansion"], 1)
            self.assertEqual(result["audit_keep_ids"], 1)
            self.assertEqual(result["dropped"]["audit_not_kept"], 1)
            self.assertEqual([row["id"] for row in rows], ["old-code", "kept-trace"])

            readiness = check_v1_1_readiness(
                sample_paths=[out_dir / "samples.jsonl"],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                assembly_manifest_path=out_dir / "manifest.json",
                min_comment2context=0,
                max_comment2context=1,
                min_comment_cross_module=0,
                min_trace2code=1,
                min_trace_non_go_repos=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            self.assertTrue(readiness["summary"]["gates"]["assembly_manifest_requires_audit_keep"])
            self.assertTrue(readiness["summary"]["gates"]["assembly_manifest_has_audit_paths"])

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["require_audit_keep"] = False
            write_json(out_dir / "manifest.json", manifest)
            unsafe_readiness = check_v1_1_readiness(
                sample_paths=[out_dir / "samples.jsonl"],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                assembly_manifest_path=out_dir / "manifest.json",
                min_comment2context=0,
                max_comment2context=1,
                min_comment_cross_module=0,
                min_trace2code=1,
                min_trace_non_go_repos=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            self.assertIn("assembly_manifest_requires_audit_keep", unsafe_readiness["blocking_gates"])

    def test_assemble_v1_1_refuses_to_write_into_frozen_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(ValueError):
                assemble_v1_1_benchmark(
                    base_sample_paths=[],
                    expansion_sources=[],
                    out_dir=root / "benchmark" / "v1",
                )

    def test_v1_1_readiness_passes_targeted_new_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            v1_1_samples = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"

            old_code = sample(
                "old-code",
                "code2test",
                {"changed_file": "src/auth.py", "intent": "auth behavior"},
                {"related_tests": ["tests/test_auth.py"]},
            )
            old_comment = sample(
                "old-comment",
                "comment2context",
                {"path": "src/payments/handler.py", "review_comment": "Does this match the web response path?"},
                {
                    "given_files": ["src/payments/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_payments.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/web/test_payments.py"],
                },
            )
            old_trace = sample(
                "old-trace",
                "trace2code",
                {"trace": "panic in auth refresh"},
                {"root_cause_files": ["src/auth.py"]},
            )
            new_comment = sample(
                "new-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently with web responses?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["cross_module_contract"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            new_trace = sample(
                "new-trace",
                "trace2code",
                {"trace": "AssertionError in TestSchedulerFairness: expected ready task"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            all_samples = [old_code, old_comment, old_trace, new_comment, new_trace]
            write_jsonl(base_samples, [old_code, old_comment, old_trace])
            write_jsonl(v1_1_samples, all_samples)
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in [
                        "src/auth.py",
                        "tests/test_auth.py",
                        "src/payments/handler.py",
                        "tests/web/test_payments.py",
                        "src/api/handler.py",
                        "tests/web/test_sanitizer.py",
                        "src/runtime/scheduler.py",
                    ]
                ],
            )
            write_jsonl(
                corpus_manifest,
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=2,
                max_comment2context=3,
                min_trace2code=2,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )

            self.assertTrue(result["ready"])
            self.assertEqual(result["summary"]["new_counts_by_task"], {"comment2context": 1, "trace2code": 1})
            self.assertEqual(result["summary"]["target_gaps"]["comment2context_samples"], 0)
            self.assertEqual(result["summary"]["target_gaps"]["trace2code_samples"], 0)
            self.assertEqual(result["blocking_gates"], [])

    def test_v1_1_readiness_rejects_code2test_expansion_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            v1_1_samples = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            old_comment = sample(
                "old-comment",
                "comment2context",
                {"path": "src/payments/handler.py", "review_comment": "Does this match the web response path?"},
                {
                    "given_files": ["src/payments/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_payments.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/web/test_payments.py"],
                },
            )
            old_trace = sample("old-trace", "trace2code", {"trace": "panic in auth refresh"}, {"root_cause_files": ["src/auth.py"]})
            new_comment = sample(
                "new-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            new_trace = sample(
                "new-trace",
                "trace2code",
                {"trace": "AssertionError in TestSchedulerFairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            new_code = sample(
                "new-code",
                "code2test",
                {"changed_file": "src/feature.py", "intent": "feature behavior"},
                {"related_tests": ["tests/test_feature_behavior.py"]},
            )

            write_jsonl(base_samples, [old_code, old_comment, old_trace])
            write_jsonl(v1_1_samples, [old_code, old_comment, old_trace, new_comment, new_trace, new_code, old_code])
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in [
                        "src/auth.py",
                        "tests/test_auth.py",
                        "src/payments/handler.py",
                        "tests/web/test_payments.py",
                        "src/api/handler.py",
                        "tests/web/test_sanitizer.py",
                        "src/runtime/scheduler.py",
                        "tests/test_feature_behavior.py",
                    ]
                ],
            )
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            result = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=2,
                max_comment2context=3,
                min_trace2code=2,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )

            self.assertFalse(result["ready"])
            self.assertIn("sample_ids_unique", result["blocking_gates"])
            self.assertIn("code2test_count_preserved", result["blocking_gates"])
            self.assertIn("new_samples_are_target_tasks", result["blocking_gates"])
            self.assertEqual(result["summary"]["duplicate_sample_ids"], ["old-code"])
            self.assertEqual(result["summary"]["counts_by_task"]["code2test"], 3)
            self.assertEqual(result["summary"]["base_counts_by_task"]["code2test"], 1)

    def test_v1_1_readiness_checks_comment_cross_module_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            v1_1_samples = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            cross_module_comment = sample(
                "cross-module-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently with web responses?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["manual"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            same_module_comment = sample(
                "same-module-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Does this match the API helper behavior?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "src/api/tests/test_handler.py", "evidence": ["manual"]}],
                    "root_cause_files": ["src/api/tests/test_handler.py"],
                },
            )
            new_trace = sample(
                "new-trace",
                "trace2code",
                {"trace": "AssertionError in TestSchedulerFairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            write_jsonl(base_samples, [old_code])
            write_jsonl(v1_1_samples, [old_code, cross_module_comment, same_module_comment, new_trace])
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in [
                        "tests/test_auth.py",
                        "src/api/handler.py",
                        "tests/web/test_sanitizer.py",
                        "src/api/tests/test_handler.py",
                        "src/runtime/scheduler.py",
                    ]
                ],
            )
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            result = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=2,
                max_comment2context=3,
                min_comment_cross_module=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )

            self.assertFalse(result["ready"])
            self.assertEqual(result["summary"]["comment2context_new_cross_module_samples"], 1)
            self.assertEqual(result["summary"]["target_gaps"]["comment2context_cross_module_samples"], 1)
            self.assertIn("new_comment2context_cross_module_count_ge_min", result["blocking_gates"])

    def test_v1_1_readiness_reports_blocking_new_sample_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            markdown_out = root / "reports" / "readiness.md"

            bad_comment = sample(
                "bad-comment",
                "comment2context",
                {"path": "src/auth/session.py", "review_comment": "Please update session_helper.py too"},
                {
                    "given_files": ["src/auth/session.py"],
                    "must_context_files": [{"path": "src/auth/session_helper.py"}],
                    "root_cause_files": ["src/auth/session_helper.py"],
                },
            )
            bad_comment["metadata"] = {}
            bad_trace = sample(
                "bad-trace",
                "trace2code",
                {"trace": "FAIL auth_test.go::TestRefresh"},
                {"root_cause_files": ["tests/auth/auth_test.go"]},
            )
            unclear_trace = sample(
                "unclear-trace",
                "trace2code",
                {"command": "pytest tests/test_worker.py"},
                {"root_cause_files": ["src/auth/session.py"]},
            )
            write_jsonl(samples_path, [bad_comment, bad_trace, unclear_trace])
            write_jsonl(
                chunks_path,
                [{"repo": "o/r", "base_commit": "base", "path": "src/auth/session.py", "kind": "file"}],
            )
            write_jsonl(
                corpus_manifest,
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = check_v1_1_readiness(
                sample_paths=[samples_path],
                corpus_manifest_path=corpus_manifest,
                markdown_out_path=markdown_out,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            rows = {row["sample_id"]: row for row in result["sample_diagnostics"]}
            report = markdown_out.read_text(encoding="utf-8")

            self.assertFalse(result["ready"])
            self.assertIn("new_samples_gold_in_corpus", result["blocking_gates"])
            self.assertIn("new_samples_have_clear_task_semantics", result["blocking_gates"])
            self.assertIn("new_samples_no_direct_gold_hints", result["blocking_gates"])
            self.assertIn("new_comment2context_no_same_directory_gold", result["blocking_gates"])
            self.assertIn("new_trace2code_gold_not_tests", result["blocking_gates"])
            self.assertIn("missing_gold", rows["bad-comment"]["issues"])
            self.assertIn("gold_basename_hint", rows["bad-comment"]["issues"])
            self.assertIn("same_directory_gold", rows["bad-comment"]["issues"])
            self.assertIn("missing_audit_evidence", rows["bad-comment"]["issues"])
            self.assertIn("trace_gold_is_test", rows["bad-trace"]["issues"])
            self.assertIn("unclear_task_semantics", rows["unclear-trace"]["issues"])
            self.assertIn("New Sample Failures", report)

    def test_v1_1_path_role_overlap_is_blocking_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            audit_out = root / "reports" / "audit_packet"

            overlap_trace = sample(
                "overlap-trace",
                "trace2code",
                {"trace": "Traceback: AssertionError in scheduler fairness"},
                {
                    "root_cause_files": ["src/runtime/scheduler.py"],
                    "supporting_files": ["src/runtime/scheduler.py"],
                    "negative_distractors": ["src/runtime/scheduler.py"],
                },
            )
            write_jsonl(samples_path, [overlap_trace])
            write_jsonl(
                chunks_path,
                [{"repo": "o/r", "base_commit": "base", "path": "src/runtime/scheduler.py", "kind": "file"}],
            )
            write_jsonl(
                corpus_manifest,
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            audit = write_v1_1_audit_packet(candidate_sources=[samples_path], out_dir=audit_out)
            readiness = check_v1_1_readiness(
                sample_paths=[samples_path],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=0,
                max_comment2context=1,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            rows = {row["sample_id"]: row for row in readiness["sample_diagnostics"]}

            self.assertEqual(audit["audit_rows"], 0)
            self.assertEqual(audit["preflight_counts"]["gold_negative_distractor_overlap"], 1)
            self.assertIn("new_samples_no_path_role_overlap", readiness["blocking_gates"])
            self.assertIn("gold_supporting_overlap", rows["overlap-trace"]["issues"])
            self.assertIn("gold_negative_distractor_overlap", rows["overlap-trace"]["issues"])
            self.assertIn("support_negative_distractor_overlap", rows["overlap-trace"]["issues"])

    def test_v1_1_readiness_checks_trace_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            v1_1_samples = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_a = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            chunks_b = root / "corpus" / "v1_1" / "rust__r" / "base.chunks.jsonl"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            new_comment = sample(
                "new-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["cross_module_contract"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            py_trace = sample(
                "py-trace",
                "trace2code",
                {"trace": "AssertionError in TestSchedulerFairness: expected ready task"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            rust_trace = sample(
                "rust-trace",
                "trace2code",
                {"trace": "thread 'main' panicked while polling the scheduler fairness loop"},
                {"root_cause_files": ["crates/runtime/src/scheduler.rs"]},
                repo="rust/r",
            )
            unknown_trace = sample(
                "unknown-trace",
                "trace2code",
                {"failure_excerpt": "worker returned status 17 after running the background task"},
                {"root_cause_files": ["src/runtime/worker.py"]},
            )
            write_jsonl(base_samples, [old_code])
            write_jsonl(v1_1_samples, [old_code, new_comment, py_trace, rust_trace, unknown_trace])
            write_jsonl(
                chunks_a,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in [
                        "tests/test_auth.py",
                        "src/api/handler.py",
                        "tests/web/test_sanitizer.py",
                        "src/runtime/scheduler.py",
                        "src/runtime/worker.py",
                    ]
                ],
            )
            write_jsonl(
                chunks_b,
                [{"repo": "rust/r", "base_commit": "base", "path": "crates/runtime/src/scheduler.rs", "kind": "file"}],
            )
            write_jsonl(
                corpus_manifest,
                [
                    {"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_a)},
                    {"repo": "rust/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_b)},
                ],
            )

            result = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=3,
                min_trace_non_go_repos=2,
                min_trace_languages=2,
                min_trace_failure_types=2,
            )

            self.assertTrue(result["ready"])
            self.assertEqual(result["summary"]["trace2code_new_gold_extensions"], [".py", ".rs"])
            self.assertEqual(result["summary"]["trace2code_new_failure_types"], ["assertion", "panic"])
            self.assertEqual(result["summary"]["trace2code_new_unknown_failure_samples"], 1)
            self.assertEqual(
                result["summary"]["target_gaps"],
                {
                    "comment2context_cross_module_samples": 0,
                    "comment2context_samples": 0,
                    "trace2code_failure_types": 0,
                    "trace2code_languages": 0,
                    "trace2code_non_go_repos": 0,
                    "trace2code_samples": 0,
                },
            )

            too_strict = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=3,
                min_trace_non_go_repos=3,
                min_trace_languages=3,
                min_trace_failure_types=3,
            )

            self.assertFalse(too_strict["ready"])
            self.assertIn("new_trace2code_non_go_repo_count_ge_min", too_strict["blocking_gates"])
            self.assertIn("new_trace2code_language_count_ge_min", too_strict["blocking_gates"])
            self.assertIn("new_trace2code_failure_type_count_ge_min", too_strict["blocking_gates"])
            self.assertEqual(too_strict["summary"]["target_gaps"]["trace2code_non_go_repos"], 1)
            self.assertEqual(too_strict["summary"]["target_gaps"]["trace2code_languages"], 1)
            self.assertEqual(too_strict["summary"]["target_gaps"]["trace2code_failure_types"], 1)

    def test_v1_1_readiness_checks_required_baseline_and_report_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_samples = root / "benchmark" / "v1" / "samples.jsonl"
            v1_1_samples = root / "benchmark" / "v1_1" / "samples.jsonl"
            corpus_manifest = root / "corpus" / "v1_1" / "corpus_manifest.jsonl"
            chunks_path = root / "corpus" / "v1_1" / "o__r" / "base.chunks.jsonl"
            eval_dir = root / "eval" / "v1_1"
            leaderboard_md = root / "reports" / "model_leaderboard.md"
            leaderboard_json = root / "reports" / "model_leaderboard.json"

            old_code = sample("old-code", "code2test", {"changed_file": "src/auth.py"}, {"related_tests": ["tests/test_auth.py"]})
            new_comment = sample(
                "new-comment",
                "comment2context",
                {"path": "src/api/handler.py", "review_comment": "Should this sanitize consistently?"},
                {
                    "given_files": ["src/api/handler.py"],
                    "must_context_files": [{"path": "tests/web/test_sanitizer.py", "evidence": ["cross_module_contract"]}],
                    "root_cause_files": ["tests/web/test_sanitizer.py"],
                },
            )
            new_trace = sample(
                "new-trace",
                "trace2code",
                {"trace": "AssertionError in TestSchedulerFairness"},
                {"root_cause_files": ["src/runtime/scheduler.py"]},
            )
            all_samples = [old_code, new_comment, new_trace]
            zero_metrics = {"Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
            write_jsonl(base_samples, [old_code])
            write_jsonl(v1_1_samples, all_samples)
            write_jsonl(
                chunks_path,
                [
                    {"repo": "o/r", "base_commit": "base", "path": path, "kind": "file"}
                    for path in ["tests/test_auth.py", "src/api/handler.py", "tests/web/test_sanitizer.py", "src/runtime/scheduler.py"]
                ],
            )
            write_jsonl(corpus_manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])
            for filename, model, mode in [
                ("lexical_summary.json", None, "corpus"),
                ("repomap_summary.json", "aider-style-repomap", "repomap"),
                ("jina-code-embeddings-0.5b_summary.json", "jina-code-embeddings-0.5b", "embedding"),
                ("qwen3-embedding-4b_summary.json", "qwen3-embedding-4b", "embedding"),
            ]:
                payload = {
                    "mode": mode,
                    "evaluated": len(all_samples),
                    "skipped": {},
                    "metrics": {
                        "overall": {"samples": len(all_samples), **zero_metrics},
                        "code2test": {"samples": 1, **zero_metrics},
                        "comment2context": {"samples": 1, **zero_metrics},
                        "trace2code": {"samples": 1, **zero_metrics},
                    },
                }
                if model:
                    payload["model"] = model
                summary_path = eval_dir / filename
                write_json(summary_path, payload)
                write_jsonl(
                    summary_path.with_name(f"{summary_path.stem.removesuffix('_summary')}_details.jsonl"),
                    [
                        {
                            "sample_id": row["id"],
                            "task_type": row["task_type"],
                            "candidate_filter": "all_files",
                            "metrics": zero_metrics,
                        }
                        for row in all_samples
                    ],
                )
            labels = ["lexical", "aider-style-repomap", "jina-code-embeddings-0.5b", "qwen3-embedding-4b"]
            write_json(leaderboard_json, {"rows": [{"task": "overall", "model_label": label} for label in labels]})
            leaderboard_md.write_text("# Model Leaderboard\n", encoding="utf-8")

            result = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                eval_dir=eval_dir,
                leaderboard_path=leaderboard_md,
                leaderboard_json_path=leaderboard_json,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )

            self.assertTrue(result["ready"])
            self.assertTrue(result["summary"]["gates"]["required_baseline_summaries_complete"])
            self.assertTrue(result["summary"]["gates"]["leaderboard_reports_contain_required_baselines"])

            qwen_details = eval_dir / "qwen3-embedding-4b_details.jsonl"
            write_jsonl(
                qwen_details,
                [
                    {"sample_id": "not-current", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics},
                    {"sample_id": "old-code", "task_type": "code2test", "candidate_filter": "all_files", "metrics": zero_metrics},
                    {"sample_id": "new-comment", "task_type": "comment2context", "candidate_filter": "all_files", "metrics": zero_metrics},
                ],
            )
            mismatched_details = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                eval_dir=eval_dir,
                leaderboard_path=leaderboard_md,
                leaderboard_json_path=leaderboard_json,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            self.assertFalse(mismatched_details["ready"])
            qwen = [item for item in mismatched_details["summary"]["required_baselines"] if item["baseline"] == "qwen3-embedding-4b"][0]
            self.assertEqual(qwen["details"]["missing_sample_ids"], ["new-trace"])
            self.assertEqual(qwen["details"]["unexpected_sample_ids"], ["not-current"])

            qwen_details.unlink()
            missing_details = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                eval_dir=eval_dir,
                leaderboard_path=leaderboard_md,
                leaderboard_json_path=leaderboard_json,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )
            self.assertFalse(missing_details["ready"])
            qwen = [item for item in missing_details["summary"]["required_baselines"] if item["baseline"] == "qwen3-embedding-4b"][0]
            self.assertFalse(qwen["details"]["complete"])
            write_jsonl(
                qwen_details,
                [{"sample_id": row["id"], "task_type": row["task_type"], "candidate_filter": "all_files", "metrics": zero_metrics} for row in all_samples],
            )

            write_json(
                eval_dir / "qwen3-embedding-4b_summary.json",
                {
                    "mode": "embedding",
                    "model": "qwen3-embedding-4b",
                    "evaluated": len(all_samples) - 1,
                    "skipped": {"oom": 1},
                    "metrics": {"overall": {"samples": len(all_samples) - 1}},
                },
            )
            incomplete = check_v1_1_readiness(
                sample_paths=[v1_1_samples],
                base_sample_paths=[base_samples],
                corpus_manifest_path=corpus_manifest,
                eval_dir=eval_dir,
                leaderboard_path=leaderboard_md,
                leaderboard_json_path=leaderboard_json,
                min_comment2context=1,
                max_comment2context=2,
                min_trace2code=1,
                min_trace_languages=1,
                min_trace_failure_types=1,
            )

            self.assertFalse(incomplete["ready"])
            self.assertIn("required_baseline_summaries_complete", incomplete["blocking_gates"])
            qwen = [item for item in incomplete["summary"]["required_baselines"] if item["baseline"] == "qwen3-embedding-4b"][0]
            self.assertFalse(qwen["complete"])

    def test_report_v1_1_release_writes_public_summary_from_readiness_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = root / "readiness.json"
            leaderboard = root / "leaderboard.json"
            out = root / "analysis.md"
            json_out = root / "analysis.json"
            write_json(
                readiness,
                {
                    "summary": {
                        "ready": True,
                        "counts_by_task": {"code2test": 106, "comment2context": 82, "trace2code": 101},
                        "base_counts_by_task": {"code2test": 106, "comment2context": 51, "trace2code": 68},
                        "new_counts_by_task": {"comment2context": 31, "trace2code": 33},
                        "trace2code_new_gold_extensions": [".py", ".rs"],
                        "trace2code_new_non_go_repos": ["pytest-dev/pytest", "rust-lang/rust"],
                        "trace2code_new_failure_types": ["assertion", "panic"],
                        "trace2code_new_unknown_failure_samples": 1,
                        "comment2context_new_cross_module_samples": 25,
                        "target_gaps": {
                            "comment2context_cross_module_samples": 0,
                            "comment2context_samples": 0,
                            "trace2code_samples": 0,
                            "trace2code_non_go_repos": 0,
                            "trace2code_languages": 0,
                            "trace2code_failure_types": 0,
                        },
                        "gates": {"required_baseline_summaries_complete": True},
                        "required_baselines": [
                            {"baseline": "lexical", "complete": True, "evaluated": 289, "skipped": {}},
                            {"baseline": "voyage-code-3", "complete": True, "evaluated": 289, "skipped": {}},
                        ],
                    }
                },
            )
            write_json(
                leaderboard,
                {
                    "rows": [
                        {
                            "task": "overall",
                            "candidate_filter": "all_files",
                            "model_label": "voyage-code-3",
                            "samples": 289,
                            "MRR": 0.31,
                            "Recall@20": 0.66,
                        },
                        {
                            "task": "trace2code",
                            "candidate_filter": "all_files",
                            "model_label": "aider-style-repomap",
                            "samples": 101,
                            "MRR": 0.28,
                            "Recall@20": 0.81,
                        },
                    ]
                },
            )

            result = report_v1_1_release(readiness, leaderboard, out, json_out)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = out.read_text(encoding="utf-8")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(report["new_counts_by_task"], {"comment2context": 31, "trace2code": 33})
            self.assertEqual(report["comment_context"]["new_cross_module_samples"], 25)
            self.assertEqual(report["trace_diversity"]["new_unknown_failure_samples"], 1)
            self.assertEqual(report["target_gaps"]["trace2code_samples"], 0)
            self.assertEqual(report["leaderboard_best"]["overall"]["best_mrr_model"], "voyage-code-3")
            self.assertIn("Agent Retrieval Bench V1.1 Report", markdown)
            self.assertIn("Review Context", markdown)
            self.assertIn("Target Gaps", markdown)
            self.assertIn("pytest-dev/pytest", markdown)
            self.assertIn("voyage-code-3", markdown)

    def test_report_v1_1_release_marks_blocking_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = root / "readiness.json"
            leaderboard = root / "leaderboard.json"
            write_json(
                readiness,
                {"summary": {"ready": False, "gates": {"new_samples_gold_in_corpus": False, "trace2code_count_ge_target": True}}},
            )
            write_json(leaderboard, {"rows": []})

            result = report_v1_1_release(readiness, leaderboard, root / "analysis.md", root / "analysis.json")

            self.assertEqual(result["status"], "not_ready")
            self.assertEqual(result["blocking_gates"], ["new_samples_gold_in_corpus"])


if __name__ == "__main__":
    unittest.main()
