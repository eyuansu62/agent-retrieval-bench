import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.baseline import block_metrics_at_budget
from agent_retrieval_bench.trajectory import evaluate_trajectories
from agent_retrieval_bench.trajectory_collect import prepare_trajectory_runs, record_trajectory_step
from agent_retrieval_bench.v1_3 import derive_v1_3_blocks, validate_v1_3_benchmark, write_v1_3_span_worklist


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class V13Tests(unittest.TestCase):
    def test_block_metrics_measure_block_overlap(self):
        blocks = [{"path": "src/auth.py", "start_line": 10, "end_line": 20, "kind": "symbol", "symbol": "refresh"}]
        ranked = [
            {"path": "src/auth.py", "start_line": 10, "end_line": 20, "kind": "symbol", "symbol": "refresh", "text": "x"},
            {"path": "src/noise.py", "start_line": 1, "end_line": 5, "kind": "symbol", "symbol": "noise", "text": "x"},
        ]

        metrics = block_metrics_at_budget(blocks, ranked)

        self.assertEqual(metrics["gold_blocks"], 1.0)
        self.assertEqual(metrics["predicted_blocks"], 2.0)
        self.assertEqual(metrics["matched_blocks"], 1.0)
        self.assertEqual(metrics["block_recall@8k"], 1.0)
        self.assertEqual(metrics["block_precision@8k"], 0.5)

    def test_derive_and_validate_v1_3_blocks_from_corpus_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_2"
            corpus = root / "corpus" / "v1_2"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            write_jsonl(derived / "samples.jsonl", [self._sample("s1")])
            write_jsonl(
                chunks_path,
                [
                    {"chunk_id": "f1", "path": "src/auth.py", "kind": "file", "symbol": "", "start_line": 1, "end_line": 50, "text": "auth"},
                    {"chunk_id": "b1", "path": "src/auth.py", "kind": "symbol", "symbol": "refresh", "start_line": 8, "end_line": 25, "text": "def refresh"},
                ],
            )
            manifest = corpus / "corpus_manifest.jsonl"
            write_jsonl(manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            report = derive_v1_3_blocks(derived, manifest, root / "benchmark" / "v1_3")
            validate = validate_v1_3_benchmark(root / "benchmark" / "v1_3", corpus_manifest_path=manifest)
            merged = [json.loads(line) for line in (root / "benchmark" / "v1_3" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(report["block_samples"], 1)
            self.assertEqual(merged[0]["gold_blocks"][0]["chunk_id"], "b1")
            self.assertTrue(validate["ready"])
            self.assertEqual(validate["block_samples"], 1)

    def test_trajectory_evaluator_reports_file_line_and_block_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            trajectories = root / "trajectories.jsonl"
            sample = self._sample("s1")
            sample["gold_blocks"] = [
                {"path": "src/auth.py", "start_line": 8, "end_line": 25, "kind": "symbol", "symbol": "refresh", "chunk_id": "b1"}
            ]
            write_jsonl(derived / "samples.jsonl", [sample])
            write_jsonl(
                trajectories,
                [
                    {
                        "sample_id": "s1",
                        "trajectory": [
                            {"step": 1, "tool": "search", "path": "src/noise.py"},
                            {
                                "step": 2,
                                "tool": "read",
                                "path": "src/auth.py",
                                "start_line": 8,
                                "end_line": 25,
                                "kind": "symbol",
                                "symbol": "refresh",
                                "content_hash": "b1",
                                "is_final_context": True,
                                "is_utilized_context": True,
                            },
                        ],
                    }
                ],
            )

            result = evaluate_trajectories(derived=derived, trajectory_paths=[trajectories])
            metrics = result["metrics"]["overall"]

            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(metrics["retrieved_file_recall"], 1.0)
            self.assertGreater(metrics["line_recall@trajectory"], 0.0)
            self.assertEqual(metrics["block_recall@trajectory"], 1.0)

    def test_trajectory_evaluator_reports_supporting_context_metrics_from_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            trajectories = root / "trajectories.jsonl"
            sample = self._sample("s1")
            sample["gold"]["supporting_context_files"] = ["src/helper.py"]
            write_jsonl(derived / "samples.jsonl", [sample])
            write_jsonl(
                trajectories,
                [
                    {
                        "sample_id": "s1",
                        "trajectory": [
                            {
                                "step": 1,
                                "tool": "read",
                                "path": "src/helper.py",
                                "is_final_context": True,
                                "is_utilized_context": True,
                            }
                        ],
                    }
                ],
            )

            result = evaluate_trajectories(derived=derived, trajectory_paths=[trajectories])
            metrics = result["metrics"]["overall"]

            self.assertEqual(metrics["final_file_recall"], 0.0)
            self.assertEqual(metrics["final_supporting_file_recall"], 1.0)
            self.assertEqual(metrics["final_gold_or_supporting_file_recall"], 0.5)

    def test_trajectory_evaluator_reports_supporting_context_metrics_from_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            trajectories = root / "trajectories.jsonl"
            annotations = root / "supporting.jsonl"
            write_jsonl(derived / "samples.jsonl", [self._sample("s1")])
            write_jsonl(annotations, [{"sample_id": "s1", "supporting_context_files": [{"path": "src/helper.py"}]}])
            write_jsonl(
                trajectories,
                [
                    {
                        "sample_id": "s1",
                        "trajectory": [
                            {
                                "step": 1,
                                "tool": "read",
                                "path": "src/helper.py",
                                "is_final_context": True,
                                "is_utilized_context": True,
                            }
                        ],
                    }
                ],
            )

            result = evaluate_trajectories(derived=derived, trajectory_paths=[trajectories], supporting_context_annotations=annotations)
            metrics = result["metrics"]["overall"]

            self.assertEqual(metrics["final_file_recall"], 0.0)
            self.assertEqual(metrics["final_supporting_file_recall"], 1.0)
            self.assertEqual(metrics["final_gold_or_supporting_file_recall"], 0.5)

    def test_prepare_trajectory_runs_writes_gold_free_prompt_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            sample = self._sample("s1")
            sample["query"] = {"failure_excerpt": "auth failure", "visible_path": "src/query.py"}
            sample["gold"] = {"root_cause_files": ["src/secret_gold.py"], "related_tests": []}
            write_jsonl(derived / "samples.jsonl", [sample])

            result = prepare_trajectory_runs(derived=derived, out_dir=root / "runs", limit_samples=1, model_label="codex")
            prompt = (root / "runs" / "prompts" / "s1.md").read_text(encoding="utf-8")
            runs = [json.loads(line) for line in (root / "runs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["samples"], 1)
            self.assertEqual(runs[0]["sample_id"], "s1")
            self.assertIn("auth failure", prompt)
            self.assertIn("src/query.py", prompt)
            self.assertNotIn("src/secret_gold.py", prompt)
            self.assertNotIn("Review-Comment Search Guidance", prompt)

    def test_prepare_trajectory_runs_adds_review_comment_guidance_without_gold_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            sample = self._sample("s1")
            sample["task_type"] = "comment2context"
            sample["query_provenance"] = "review_comment"
            sample["query"] = {
                "review_comment": "Should this also update the regression test and config policy?",
                "path": "src/query.py",
                "given_file": "src/query.py",
            }
            sample["gold"] = {"must_context_files": [{"path": "src/secret_gold.py"}], "related_tests": ["tests/hidden_policy.py"]}
            write_jsonl(derived / "samples.jsonl", [sample])

            prepare_trajectory_runs(derived=derived, out_dir=root / "runs", limit_samples=1, model_label="codex")
            prompt = (root / "runs" / "prompts" / "s1.md").read_text(encoding="utf-8")

            self.assertIn("## Review-Comment Search Guidance", prompt)
            self.assertIn("semver, error wording, regression coverage, process protocols, or conversion registries", prompt)
            self.assertIn("policy, config, test, and implementation follow-up files", prompt)
            self.assertIn("src/query.py", prompt)
            self.assertNotIn("src/secret_gold.py", prompt)
            self.assertNotIn("tests/hidden_policy.py", prompt)

    def test_record_trajectory_step_outputs_evaluable_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            repo = root / "repo"
            repo.mkdir()
            (repo / "src").mkdir()
            (repo / "src" / "auth.py").write_text("one\nroot cause\nthird\n", encoding="utf-8")
            sample = self._sample("s1")
            sample["gold_spans"] = [{"path": "src/auth.py", "start_line": 2, "end_line": 2, "reason": "root cause"}]
            sample["gold_blocks"] = [{"path": "src/auth.py", "start_line": 2, "end_line": 2, "kind": "block", "chunk_id": ""}]
            write_jsonl(derived / "samples.jsonl", [sample])
            log_path = root / "trajectory.jsonl"

            first = record_trajectory_step(
                log_path=log_path,
                sample_id="s1",
                path="src/noise.py",
                tool="rg",
            )
            second = record_trajectory_step(
                log_path=log_path,
                sample_id="s1",
                path="src/auth.py",
                start_line=2,
                end_line=2,
                repo_root=repo,
                is_final_context=True,
                is_utilized_context=True,
            )
            result = evaluate_trajectories(derived=derived, trajectory_paths=[log_path])
            metrics = result["metrics"]["overall"]

            self.assertEqual(first["step"], 1)
            self.assertEqual(second["step"], 2)
            self.assertTrue(second["content_hash"])
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(metrics["retrieved_file_recall"], 1.0)
            self.assertEqual(metrics["line_recall@trajectory"], 1.0)

    def test_span_worklist_suggests_target_file_chunks_without_gold_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = root / "benchmark" / "v1_3"
            corpus = root / "corpus" / "v1_2"
            chunks_path = corpus / "o__r" / "base.chunks.jsonl"
            sample = self._sample("s1")
            sample["task_type"] = "comment2context"
            sample["query"] = {
                "review_comment": "Please move the needle helper into the cached top-level function.",
                "given_file": "src/query.py",
                "path": "src/query.py",
                "line": 12,
            }
            sample["gold"] = {"must_context_files": [{"path": "src/target.py"}], "given_files": ["src/query.py"]}
            sample.pop("gold_spans")
            write_jsonl(derived / "samples.jsonl", [sample])
            write_jsonl(
                chunks_path,
                [
                    {"chunk_id": "file", "path": "src/target.py", "kind": "file", "symbol": "", "start_line": 1, "end_line": 80, "text": "target"},
                    {"chunk_id": "noise", "path": "src/target.py", "kind": "symbol", "symbol": "unrelated", "start_line": 1, "end_line": 10, "text": "other code"},
                    {"chunk_id": "needle", "path": "src/target.py", "kind": "symbol", "symbol": "needle_helper", "start_line": 20, "end_line": 30, "text": "def needle_helper(): pass"},
                ],
            )
            manifest = corpus / "corpus_manifest.jsonl"
            write_jsonl(manifest, [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}])

            report = write_v1_3_span_worklist(
                derived=derived,
                corpus_manifest_path=manifest,
                out_path=root / "report.json",
                jsonl_out_path=root / "suggestions.jsonl",
            )
            suggestion = json.loads((root / "suggestions.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(report["missing_span_samples"], 1)
            self.assertEqual(report["with_candidates"], 1)
            self.assertEqual(report["rows"][0]["candidate_spans"][0]["chunk_id"], "needle")
            self.assertNotIn("gold_spans", suggestion)

    def _sample(self, sample_id):
        return {
            "id": sample_id,
            "version": 3,
            "task_type": "trace2code",
            "repo": "o/r",
            "base_commit": "base",
            "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": "base"},
            "query": {"failure_excerpt": "auth failure"},
            "query_provenance": "failure_trace",
            "gold": {"root_cause_files": ["src/auth.py"], "related_tests": []},
            "gold_spans": [{"path": "src/auth.py", "start_line": 10, "end_line": 20, "reason": "root cause implementation"}],
        }


if __name__ == "__main__":
    unittest.main()
