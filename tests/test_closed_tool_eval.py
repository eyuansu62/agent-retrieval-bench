import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_retrieval_bench.closed_tool_eval import (
    append_detail_jsonl,
    choose_grep_patterns,
    detect_codex_protocol_violations,
    grep_files,
    load_existing_details,
    load_seed_files,
    parse_tool_action,
    prepare_details_resume,
    report_closed_tool_budget_curve,
    report_closed_tool_seed_intervention,
    run_closed_tool_codex_sample,
    run_closed_tool_llm_sample,
    run_closed_tool_sample,
)


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def create(self, **kwargs):
        self.prompts.append(kwargs["input"])
        if not self.outputs:
            raise AssertionError("fake client exhausted")
        return SimpleNamespace(output_text=self.outputs.pop(0))


class FakeClient:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


class ClosedToolEvalTest(unittest.TestCase):
    def test_choose_grep_patterns_prefers_title_and_code_terms(self):
        query = {
            "pr_title": "Fix timeout panic in async join",
            "changed_file": "src/time/timeout.rs",
            "pr_body": "The timeout future should not panic.",
        }
        patterns = choose_grep_patterns(query, max_patterns=5)
        self.assertIn("timeout", patterns)
        self.assertIn("panic", patterns)

    def test_grep_files_counts_path_and_text_matches(self):
        files = {
            "src/time/timeout.rs": "pub fn timeout() {}",
            "tests/time_panic.rs": "timeout panic timeout",
        }
        hits = grep_files(files, "timeout", top_k=2)
        self.assertEqual({hit["path"] for hit in hits}, {"src/time/timeout.rs", "tests/time_panic.rs"})
        self.assertGreaterEqual(next(hit for hit in hits if hit["path"] == "tests/time_panic.rs")["matches"], 2)

    def test_run_closed_tool_sample_finds_gold_file(self):
        sample = {
            "id": "s1",
            "task_type": "code2test",
            "repo": "example/repo",
            "base_commit": "abc",
            "query": {
                "pr_title": "Fix timeout panic in async join",
                "changed_file": "src/time/timeout.rs",
                "pr_body": "The timeout future should not panic.",
            },
            "gold": {"related_tests": ["tests/time_panic.rs"]},
            "gold_spans": [{"path": "tests/time_panic.rs", "start_line": 1, "end_line": 3}],
        }
        files = {
            "src/time/timeout.rs": "pub fn timeout() { /* implementation */ }",
            "tests/time_panic.rs": "test timeout panic\nassert timeout does not panic\n",
            "tests/other.rs": "unrelated parser test",
        }
        detail = run_closed_tool_sample(
            sample=sample,
            gold_files=["tests/time_panic.rs"],
            query_text='{"pr_title": "Fix timeout panic in async join", "changed_file": "src/time/timeout.rs"}',
            files=files,
            max_tool_calls=8,
            max_read_tokens=200,
            max_read_tokens_per_file=80,
            max_grep_calls=4,
            final_k=2,
            grep_top_k=3,
        )
        self.assertIn("tests/time_panic.rs", detail["closed_tool"]["final_files"])
        self.assertGreater(detail["metrics"]["final_file_f1"], 0.0)

    def test_parse_tool_action_accepts_fenced_json(self):
        action = parse_tool_action('Use this:\n```json\n{"action":"grep","pattern":"timeout","top_k":3}\n```')
        self.assertEqual(action["action"], "grep")
        self.assertEqual(action["pattern"], "timeout")

    def test_run_closed_tool_llm_sample_uses_closed_tools(self):
        sample = {
            "id": "s1",
            "task_type": "code2test",
            "repo": "example/repo",
            "base_commit": "abc",
            "query": {"pr_title": "Fix timeout panic"},
            "gold": {"related_tests": ["tests/time_panic.rs"]},
            "gold_spans": [{"path": "tests/time_panic.rs", "start_line": 1, "end_line": 3}],
        }
        files = {
            "src/time/timeout.rs": "pub fn timeout() { /* implementation */ }",
            "tests/time_panic.rs": "test timeout panic\nassert timeout does not panic\n",
            "tests/other.rs": "unrelated parser test",
        }
        client = FakeClient(
            [
                '{"action":"grep","pattern":"timeout panic","top_k":3}',
                '{"action":"read_file","path":"tests/time_panic.rs"}',
                '{"action":"submit","files":["tests/time_panic.rs"]}',
            ]
        )
        detail = run_closed_tool_llm_sample(
            sample=sample,
            gold_files=["tests/time_panic.rs"],
            query_text='{"pr_title": "Fix timeout panic"}',
            files=files,
            model="fake-model",
            client=client,
            max_tool_calls=8,
            max_read_tokens=200,
            max_read_tokens_per_file=80,
            final_k=2,
            grep_top_k=3,
            max_model_turns=5,
        )
        self.assertEqual(detail["closed_tool"]["policy"], "openai_closed_tool")
        self.assertIn("tests/time_panic.rs", detail["closed_tool"]["final_files"])
        self.assertGreater(detail["metrics"]["final_file_f1"], 0.0)
        self.assertEqual(detail["closed_tool"]["tool_calls"], 3)

    def test_run_closed_tool_llm_sample_can_submit_seed_context(self):
        sample = {
            "id": "s1",
            "task_type": "code2test",
            "repo": "example/repo",
            "base_commit": "abc",
            "query": {"pr_title": "Fix timeout panic"},
            "gold": {"related_tests": ["tests/time_panic.rs"]},
            "gold_spans": [{"path": "tests/time_panic.rs", "start_line": 1, "end_line": 3}],
        }
        files = {
            "src/time/timeout.rs": "pub fn timeout() { /* implementation */ }",
            "tests/time_panic.rs": "test timeout panic\nassert timeout does not panic\n",
        }
        client = FakeClient(['{"action":"submit","files":["tests/time_panic.rs"]}'])
        detail = run_closed_tool_llm_sample(
            sample=sample,
            gold_files=["tests/time_panic.rs"],
            query_text='{"pr_title": "Fix timeout panic"}',
            files=files,
            model="fake-model",
            client=client,
            max_tool_calls=8,
            max_read_tokens=200,
            max_read_tokens_per_file=80,
            final_k=2,
            grep_top_k=3,
            max_model_turns=5,
            seed_files=["tests/time_panic.rs"],
            seed_label="oracle_seed",
            max_seed_tokens=40,
            max_seed_tokens_per_file=40,
        )
        self.assertEqual(detail["closed_tool"]["first_gold_step"], 0)
        self.assertEqual(detail["closed_tool"]["tool_calls"], 1)
        self.assertEqual(detail["closed_tool"]["seed_files"], ["tests/time_panic.rs"])
        self.assertTrue(detail["closed_tool"]["seed_any_gold"])
        self.assertIn("tests/time_panic.rs", detail["closed_tool"]["final_files"])
        self.assertGreater(detail["metrics"]["final_file_f1"], 0.0)

    def test_load_seed_files_reads_top_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "details.jsonl"
            path.write_text(
                json.dumps({"sample_id": "s1", "top_files": ["a.py", "b.py", "c.py"]}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_seed_files(path, top_k=2), {"s1": ["a.py", "b.py"]})

    def test_detect_codex_protocol_violations_flags_shell_tool(self):
        violations = detect_codex_protocol_violations(
            [
                {"type": "assistant_message", "text": "hello"},
                {"type": "tool_call", "name": "exec_command", "arguments": {"cmd": "ls"}},
            ]
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("exec_command", violations[0]["markers"])

    def test_prepare_details_resume_appends_and_force_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            details_path = Path(tmp) / "details.jsonl"
            details_path.write_text(json.dumps({"sample_id": "old"}) + "\n", encoding="utf-8")

            existing, handle = prepare_details_resume(details_path, force=False)
            with handle:
                append_detail_jsonl(handle, {"sample_id": "new"})
            self.assertEqual(set(existing), {"old"})
            self.assertEqual(set(load_existing_details(details_path)), {"old", "new"})

            existing, handle = prepare_details_resume(details_path, force=True)
            with handle:
                append_detail_jsonl(handle, {"sample_id": "fresh"})
            self.assertEqual(existing, {})
            self.assertEqual(set(load_existing_details(details_path)), {"fresh"})

    def test_run_closed_tool_codex_sample_uses_codex_action_adapter(self):
        sample = {
            "id": "s1",
            "task_type": "code2test",
            "repo": "example/repo",
            "base_commit": "abc",
            "query": {"pr_title": "Fix timeout panic"},
            "gold": {"related_tests": ["tests/time_panic.rs"]},
            "gold_spans": [{"path": "tests/time_panic.rs", "start_line": 1, "end_line": 3}],
        }
        files = {
            "src/time/timeout.rs": "pub fn timeout() { /* implementation */ }",
            "tests/time_panic.rs": "test timeout panic\nassert timeout does not panic\n",
            "tests/other.rs": "unrelated parser test",
        }
        outputs = [
            '{"action":"grep","pattern":"timeout panic","top_k":3}',
            '{"action":"read_file","path":"tests/time_panic.rs"}',
            '{"action":"submit","files":["tests/time_panic.rs"]}',
        ]

        def fake_codex_action(**kwargs):
            index = kwargs["turn"] - 1
            return {
                "turn": kwargs["turn"],
                "returncode": 0,
                "output_text": outputs[index],
                "json_events": 1,
                "protocol_violations": [],
                "stderr_excerpt": "",
            }

        with patch("agent_retrieval_bench.closed_tool_eval.run_codex_exec_action", side_effect=fake_codex_action):
            detail = run_closed_tool_codex_sample(
                sample=sample,
                gold_files=["tests/time_panic.rs"],
                query_text='{"pr_title": "Fix timeout panic"}',
                files=files,
                model="fake-codex",
                codex_bin=Path("/tmp/codex"),
                work_root=Path("/tmp"),
                timeout_seconds=5,
                max_tool_calls=8,
                max_read_tokens=200,
                max_read_tokens_per_file=80,
                final_k=2,
                grep_top_k=3,
                max_model_turns=5,
            )
        self.assertEqual(detail["closed_tool"]["policy"], "codex_closed_tool")
        self.assertIn("tests/time_panic.rs", detail["closed_tool"]["final_files"])
        self.assertEqual(detail["closed_tool"]["codex_exec_calls"], 3)
        self.assertEqual(detail["closed_tool"]["protocol_violations"], [])

    def test_report_closed_tool_budget_curve_scores_prefix_context(self):
        detail = {
            "sample_id": "s1",
            "task_type": "code2test",
            "repo": "example/repo",
            "base_commit": "abc",
            "gold_files": ["src/gold.py"],
            "supporting_context_files": [],
            "gold_spans": [],
            "gold_blocks": [],
            "steps": [
                {
                    "step": 2,
                    "tool": "read_file",
                    "path": "src/noise.py",
                    "start_line": None,
                    "end_line": None,
                    "kind": "file",
                    "symbol": "",
                    "content_hash": "",
                    "is_final_context": False,
                    "is_utilized_context": False,
                },
                {
                    "step": 4,
                    "tool": "read_file",
                    "path": "src/gold.py",
                    "start_line": None,
                    "end_line": None,
                    "kind": "file",
                    "symbol": "",
                    "content_hash": "",
                    "is_final_context": False,
                    "is_utilized_context": False,
                },
            ],
            "metrics": {},
            "closed_tool": {
                "policy": "codex_closed_tool",
                "model": "fake-codex",
                "max_tool_calls": 8,
                "trace": [
                    {"tool": "list_dir", "path": ""},
                    {"tool": "read_file", "path": "src/noise.py", "tokens": 10},
                    {"tool": "grep", "pattern": "gold"},
                    {"tool": "read_file", "path": "src/gold.py", "tokens": 20},
                    {"tool": "submit", "files": ["src/gold.py"]},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            details_path = tmp_path / "details.jsonl"
            details_path.write_text(json.dumps(detail) + "\n", encoding="utf-8")
            report = report_closed_tool_budget_curve(
                details_path=details_path,
                out_path=tmp_path / "curve.json",
                markdown_out_path=tmp_path / "curve.md",
                budgets=[2, 4],
            )

        rows = {row["budget"]: row for row in report["rows"]}
        self.assertEqual(rows[2]["metrics"]["overall"]["final_file_f1"], 0.0)
        self.assertGreater(rows[4]["metrics"]["overall"]["final_file_f1"], 0.0)
        self.assertEqual(rows[2]["closed_tool_metrics"]["overall"]["any_gold_rate"], 0.0)
        self.assertEqual(rows[4]["closed_tool_metrics"]["overall"]["any_gold_rate"], 1.0)
        self.assertEqual(rows[4]["closed_tool_metrics"]["overall"]["mean_read_tokens"], 30.0)

    def test_report_closed_tool_seed_intervention_adds_paired_analysis(self):
        control_rows = [
            {
                "sample_id": "s1",
                "task_type": "code2test",
                "repo": "example/repo",
                "base_commit": "abc",
                "gold_files": ["src/gold.py"],
                "steps": [],
                "metrics": {"final_file_f1": 0.0, "final_file_recall": 0.0, "final_file_precision": 0.0},
                "closed_tool": {"first_gold_step": None, "final_files": [], "read_tokens": 100, "seed_tokens": 0},
            },
            {
                "sample_id": "s2",
                "task_type": "code2test",
                "repo": "example/repo",
                "base_commit": "abc",
                "gold_files": ["src/other.py"],
                "steps": [],
                "metrics": {"final_file_f1": 1.0, "final_file_recall": 1.0, "final_file_precision": 1.0},
                "closed_tool": {"first_gold_step": 2, "final_files": ["src/other.py"], "read_tokens": 120, "seed_tokens": 0},
            },
        ]
        arm_rows = [
            {
                "sample_id": "s1",
                "task_type": "code2test",
                "repo": "example/repo",
                "base_commit": "abc",
                "gold_files": ["src/gold.py"],
                "steps": [],
                "metrics": {"final_file_f1": 1.0, "final_file_recall": 1.0, "final_file_precision": 1.0},
                "closed_tool": {
                    "first_gold_step": 0,
                    "final_files": ["src/gold.py"],
                    "read_tokens": 80,
                    "seed_tokens": 30,
                    "seed_any_gold": True,
                    "seed_files": ["src/gold.py"],
                },
            },
            {
                "sample_id": "s2",
                "task_type": "code2test",
                "repo": "example/repo",
                "base_commit": "abc",
                "gold_files": ["src/other.py"],
                "steps": [],
                "metrics": {"final_file_f1": 0.0, "final_file_recall": 0.0, "final_file_precision": 0.0},
                "closed_tool": {
                    "first_gold_step": None,
                    "final_files": [],
                    "read_tokens": 90,
                    "seed_tokens": 40,
                    "seed_any_gold": False,
                    "seed_files": ["src/noise.py"],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control = tmp_path / "control.jsonl"
            arm = tmp_path / "arm.jsonl"
            control.write_text("".join(json.dumps(row) + "\n" for row in control_rows), encoding="utf-8")
            arm.write_text("".join(json.dumps(row) + "\n" for row in arm_rows), encoding="utf-8")
            report = report_closed_tool_seed_intervention(
                control_details=control,
                arms={"seeded": arm},
                out_path=tmp_path / "report.json",
                markdown_out_path=tmp_path / "report.md",
            )

        seeded = next(row for row in report["rows"] if row["label"] == "seeded")
        self.assertEqual(seeded["seed_metrics"]["mean_post_seed_read_tokens"], 50.0)
        self.assertEqual(seeded["paired_vs_no_seed"]["improved_f1_count"], 1)
        self.assertEqual(seeded["paired_vs_no_seed"]["worsened_f1_count"], 1)
        self.assertEqual(seeded["paired_vs_no_seed"]["rescued_any_gold_count"], 1)
        self.assertEqual(seeded["paired_vs_no_seed"]["lost_any_gold_count"], 1)
        self.assertEqual(seeded["paired_vs_no_seed"]["rescue_rate_given_no_seed_miss"], 1.0)


if __name__ == "__main__":
    unittest.main()
