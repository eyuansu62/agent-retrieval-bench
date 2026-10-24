import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_retrieval_bench.bcy_curve import count_tokens
from agent_retrieval_bench.closed_tool_eval import (
    append_detail_jsonl,
    build_seed_context,
    choose_grep_patterns,
    detect_codex_protocol_violations,
    evaluate_closed_tool_codex,
    evaluate_closed_tool_grep,
    evaluate_closed_tool_openai,
    format_read_file_observation,
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
    token_prefix,
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
    def _bounded_read_policy(self, suffix, *, per_file=4, total=8):
        prefix = "anchorcommon\n" + "padding " * 200
        files = {"a.py": prefix + "nothing\n", "b.py": prefix + suffix + "\n"}
        sample = dict(
            id="bounded-read-control", repo="example/repo", base_commit="base",
            task_type="trace2code", gold=dict(root_cause_files=["a.py"]),
        )
        detail = run_closed_tool_sample(
            sample=sample, gold_files=["a.py"],
            query_text=json.dumps(dict(pr_title="anchorcommon anchorcommon", pr_body="rarebonus")),
            files=files, max_tool_calls=5, max_read_tokens=total,
            max_read_tokens_per_file=per_file, max_grep_calls=1,
            final_k=1, grep_top_k=2,
        )
        policy = detail["closed_tool"]
        observations = {
            event["path"]: format_read_file_observation(event["path"], files[event["path"]], event["tokens"])
            for event in policy["trace"] if event["tool"] == "read_file"
        }
        return policy, observations

    def test_scripted_policy_cannot_score_unobserved_suffix(self):
        for total, read_sizes in ((8, [4, 4]), (6, [4, 2])):
            with self.subTest(total=total):
                control, before = self._bounded_read_policy("nothing", total=total)
                changed, after = self._bounded_read_policy("rarebonus", total=total)
                self.assertEqual(before, after)
                self.assertNotIn("rarebonus", after["b.py"])
                self.assertEqual(control["trace"][:-1], changed["trace"][:-1])
                self.assertEqual(control["read_tokens"], total)
                self.assertEqual(changed["read_tokens"], total)
                self.assertEqual(
                    [event["tokens"] for event in changed["trace"] if event["tool"] == "read_file"],
                    read_sizes,
                )
                self.assertEqual(control["final_files"], changed["final_files"])

    def test_scripted_policy_can_score_visible_suffix(self):
        control, before = self._bounded_read_policy("nothing", per_file=256, total=512)
        changed, after = self._bounded_read_policy("rarebonus", per_file=256, total=512)
        self.assertNotIn("rarebonus", before["b.py"])
        self.assertIn("rarebonus", after["b.py"])
        self.assertEqual(control["final_files"], ["a.py"])
        self.assertEqual(changed["final_files"], ["b.py"])

    def test_token_prefix_preserves_text_at_and_below_limit(self):
        for text, cap, expected in (
            ("", 1, ""),
            (" \n", 1, " \n"),
            ("你好🙂\n", 3, "你好🙂\n"),
            (" alpha beta \n", 2, " alpha beta \n"),
            (" alpha beta \n", 3, " alpha beta \n"),
            (" alpha beta \n", 10**100, " alpha beta \n"),
            (" alpha beta gamma\n", 2, " alpha beta\n[TRUNCATED]"),
            ("one\r\ntwo three", 2, "one\r\ntwo\n[TRUNCATED]"),
        ):
            with self.subTest(text=text, cap=cap):
                self.assertEqual(token_prefix(text, cap), expected)

    def test_token_prefix_uses_the_charged_code_tokenizer(self):
        for text, cap, expected in (
            ("call(x, y)", 2, "call(\n[TRUNCATED]"),
            (";;;", 1, ";\n[TRUNCATED]"),
            ("你好🙂\n", 1, "你\n[TRUNCATED]"),
        ):
            with self.subTest(text=text):
                visible = token_prefix(text, cap)
                self.assertEqual(visible, expected)
                content = visible.removesuffix("\n[TRUNCATED]")
                self.assertEqual(count_tokens(content), cap)

    def test_seed_context_respects_the_charged_code_tokenizer(self):
        context = build_seed_context(
            files={"sample.py": "call(x, y)"}, seed_files=["sample.py"],
            seed_label="test", max_seed_tokens=2, max_seed_tokens_per_file=2,
            max_read_tokens=2,
        )
        self.assertEqual(context["tokens"], 2)
        self.assertIn("tokens=2\ncall(\n[TRUNCATED]", context["observation"])
        self.assertNotIn("call(x", context["observation"])

    def test_model_read_respects_the_charged_code_tokenizer(self):
        client = FakeClient([
            '{"action":"read_file","path":"sample.py"}',
            '{"action":"submit","files":["sample.py"]}',
        ])
        detail = run_closed_tool_llm_sample(
            sample=dict(id="tokenizer-control", repo="example/repo", base_commit="base",
                        task_type="trace2code", gold=dict(root_cause_files=["sample.py"])),
            gold_files=["sample.py"], query_text='{"pr_title":"call"}',
            files={"sample.py": "call(x, y)"}, model="fake", client=client,
            max_tool_calls=2, max_read_tokens=2, max_read_tokens_per_file=2,
            final_k=1, grep_top_k=1, max_model_turns=2,
        )
        self.assertEqual(detail["closed_tool"]["read_tokens"], 2)
        self.assertEqual(len(client.responses.prompts), 2)
        self.assertIn("tokens=2\ncall(\n[TRUNCATED]", client.responses.prompts[1])
        self.assertNotIn("call(x", client.responses.prompts[1])

    def test_scripted_policy_cannot_score_past_punctuation_budget(self):
        prefix = "anchorcommon\n" + "; " * 200

        def evaluate(suffix, per_file, total):
            detail = run_closed_tool_sample(
                sample=dict(id="punctuation-control", repo="example/repo", base_commit="base",
                            task_type="trace2code", gold=dict(root_cause_files=["a.py"])),
                gold_files=["a.py"],
                query_text=json.dumps(dict(pr_title="anchorcommon anchorcommon", pr_body="rarebonus")),
                files={"a.py": prefix + "nothing\n", "b.py": prefix + suffix + "\n"},
                max_tool_calls=5, max_read_tokens=total, max_read_tokens_per_file=per_file,
                max_grep_calls=1, final_k=1, grep_top_k=2,
            )
            return detail["closed_tool"]

        control, changed = evaluate("nothing", 2, 4), evaluate("rarebonus", 2, 4)
        self.assertEqual(control["read_tokens"], 4)
        self.assertEqual(control["trace"][:-1], changed["trace"][:-1])
        self.assertEqual(control["final_files"], changed["final_files"])
        self.assertEqual(evaluate("nothing", 256, 512)["final_files"], ["a.py"])
        self.assertEqual(evaluate("rarebonus", 256, 512)["final_files"], ["b.py"])

    def test_token_prefix_stops_after_one_lookahead_match(self):
        from agent_retrieval_bench import closed_tool_eval

        pattern = closed_tool_eval.TOKEN_RE
        for cap in (1, 3, 20):
            consumed = []

            def bounded_matches(text):
                for match in pattern.finditer(text):
                    consumed.append(match)
                    self.assertLessEqual(len(consumed), cap + 1)
                    yield match

            with self.subTest(cap=cap), patch.object(
                closed_tool_eval, "TOKEN_RE", SimpleNamespace(finditer=bounded_matches)
            ):
                self.assertEqual(
                    token_prefix("word " * 10_000, cap),
                    " ".join(["word"] * cap) + "\n[TRUNCATED]",
                )
                self.assertEqual(len(consumed), cap + 1)

    def test_token_prefix_nonpositive_limit_does_not_scan(self):
        with patch("agent_retrieval_bench.closed_tool_eval.TOKEN_RE") as pattern:
            for cap in (0, -1):
                self.assertEqual(token_prefix("unobserved text", cap), "")
            pattern.finditer.assert_not_called()

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

    def test_grep_files_snippets_use_original_unicode_offsets(self):
        for prefix, word, query in (
            ("İ" * 200, "needle", "NEEDLE"),
            ("é" * 200, "needle", "needle"),
            ("z" * 200, "İ", "i\u0307"),
        ):
            with self.subTest(word=word, prefix=prefix[:1]):
                text = f"{prefix} {word}\n"
                hit, = grep_files({"subject.py": text}, query, top_k=1)
                self.assertEqual(hit["matches"], 1)
                self.assertFalse(hit["path_match"])
                self.assertEqual(hit["snippet"], text[len(prefix) + 1 - 80:].replace("\n", "\\n"))

    def test_grep_files_path_only_match_has_no_body_snippet(self):
        hit, = grep_files({"src/timeout.py": "def clean(): return 1\n"}, "TIMEOUT", top_k=1)
        self.assertEqual(hit["matches"], 1)
        self.assertTrue(hit["path_match"])
        self.assertEqual(hit["snippet"], "")

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


class ClosedToolSampleIdentityTest(unittest.TestCase):
    def _inputs(self, root):
        sample = dict(
            id="sample-one", repo="example/repo", base_commit="base",
            task_type="trace2code", query=dict(pr_title="anchorapple"),
            gold=dict(root_cause_files=["subject.py"]),
        )
        chunks = root / "chunks.jsonl"
        self._write(chunks, [dict(path="subject.py", kind="file", text="def anchorapple(): return 1\n")])
        self._write(root / "corpus_manifest.jsonl", [dict(
            repo="example/repo", base_commit="base", status="ok", chunks_path=str(chunks),
        )])
        paths = [root / name for name in ("first.jsonl", "identical.jsonl", "conflicting.jsonl", "unique.jsonl")]
        for path, row in zip(paths, [sample, sample, dict(sample, query=dict(pr_title="differentintent")), dict(sample, id="sample-two")]):
            self._write(path, [row])
        return paths

    def _write(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_duplicate_ids_fail_before_dispatch_or_output_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, identical, conflicting, _unique = self._inputs(root)
            out, details, report = (root / name for name in ("summary.json", "details.jsonl", "report.md"))
            for path in (out, details, report):
                path.write_text("preserve existing output\n", encoding="utf-8")
            for evaluator in (evaluate_closed_tool_grep, evaluate_closed_tool_openai, evaluate_closed_tool_codex):
                for second in (first, identical, conflicting):
                    with self.subTest(evaluator=evaluator.__name__, second=second.name):
                        kwargs = dict(sample_paths=[first, second], corpus_dir=root,
                                      out_path=out, details_path=details, markdown_out_path=report, force=True)
                        if evaluator is evaluate_closed_tool_codex:
                            kwargs["codex_bin"] = Path(sys.executable)
                        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch(
                            "agent_retrieval_bench.closed_tool_eval.run_closed_tool_codex_sample",
                            side_effect=AssertionError("must not dispatch invalid inputs"),
                        ) as dispatch:
                            with self.assertRaisesRegex(ValueError, "duplicate sample id.*sample-one"):
                                evaluator(**kwargs)
                            dispatch.assert_not_called()
                        for path in (out, details, report):
                            self.assertEqual(path.read_text(encoding="utf-8"), "preserve existing output\n")

    def test_unique_ids_keep_both_rows_and_resume_them_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _identical, _conflicting, unique = self._inputs(root)
            for force in (True, False):
                summary = evaluate_closed_tool_grep(
                    sample_paths=[first, unique], corpus_dir=root, out_path=root / "summary.json",
                    details_path=root / "details.jsonl", markdown_out_path=root / "report.md", force=force,
                )
                rows = [json.loads(line) for line in (root / "details.jsonl").read_text().splitlines()]
                self.assertEqual(summary["evaluated"], 2)
                self.assertEqual([row["sample_id"] for row in rows], ["sample-one", "sample-two"])
            self.assertEqual(summary["skipped"], {"existing": 2})

    def test_limit_validates_only_the_selected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, *_rest = self._inputs(root)
            summary = evaluate_closed_tool_grep(
                sample_paths=[first, first], corpus_dir=root, out_path=root / "summary.json",
                details_path=root / "details.jsonl", markdown_out_path=root / "report.md",
                limit_samples=1, force=True,
            )
            self.assertEqual(summary["evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
