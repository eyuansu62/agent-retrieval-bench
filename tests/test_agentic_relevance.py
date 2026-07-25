import unittest

from agent_retrieval_bench.agentic_relevance import annotate_sample, semantic_overlap_score, tokenize


class AgenticRelevanceTest(unittest.TestCase):
    def test_trace2code_is_causal_indirect(self):
        row = {
            "id": "trace",
            "task_type": "trace2code",
            "repo": "example/repo",
            "query": {"failure_excerpt": "foo_test.py fails with expected 3 got 4"},
            "gold": {"root_cause_files": ["src/foo.py"], "related_tests": ["tests/foo_test.py"]},
            "gold_spans": [{"path": "src/foo.py"}],
        }
        annotation = annotate_sample(row)
        self.assertEqual(annotation["primary_relevance_type"], "causal_indirect")

    def test_code2test_defaults_to_workflow_conventional(self):
        row = {
            "id": "code",
            "task_type": "code2test",
            "repo": "example/repo",
            "query": {"changed_file": "src/cache.go", "pr_title": "add progress notification"},
            "gold": {"related_tests": ["tests/watch_api_test.go"]},
            "gold_spans": [{"path": "tests/watch_api_test.go"}],
            "metadata": {"evidence": {"signals": ["same_pr_changed_tests"]}},
        }
        annotation = annotate_sample(row)
        self.assertEqual(annotation["primary_relevance_type"], "workflow_conventional")
        self.assertIn("workflow_conventional", annotation["secondary_relevance_types"] + [annotation["primary_relevance_type"]])

    def test_comment_same_module_defaults_to_structural(self):
        row = {
            "id": "comment",
            "task_type": "comment2context",
            "repo": "example/repo",
            "query": {"given_file": "src/types/function.rs", "review_comment": "move this helper to a cached top-level function"},
            "gold": {
                "must_context_files": [
                    {"path": "src/types/infer/builder.rs", "evidence": ["same_module_context", "explicit_behavior_or_api_dependency"]}
                ]
            },
            "gold_spans": [{"path": "src/types/infer/builder.rs"}],
        }
        annotation = annotate_sample(row)
        self.assertEqual(annotation["primary_relevance_type"], "structural_indirect")

    def test_strong_query_gold_overlap_can_be_semantic_direct(self):
        row = {
            "id": "semantic",
            "task_type": "comment2context",
            "repo": "example/repo",
            "query": {"review_comment": "The enum __new__ inheritance behavior should be covered."},
            "gold": {
                "must_context_files": [
                    {"path": "src/types/enums.rs", "evidence": ["symbol_or_path_overlap"]}
                ]
            },
            "gold_blocks": [{"path": "src/types/enums.rs", "symbol": "enum_new_inheritance"}],
            "gold_spans": [{"path": "src/types/enums.rs"}],
        }
        annotation = annotate_sample(row)
        self.assertEqual(annotation["primary_relevance_type"], "semantic_direct")

    def test_overlap_score_uses_min_denominator(self):
        self.assertGreater(semantic_overlap_score(tokenize("enum new inheritance"), tokenize("enum new inheritance behavior")), 0.9)


if __name__ == "__main__":
    unittest.main()
