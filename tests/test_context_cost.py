import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.context_cost import report_context_acquisition_cost


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ContextCostTests(unittest.TestCase):
    def test_report_context_cost_joins_agent_misses_to_retriever_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agent.jsonl"
            retriever_path = root / "retriever.jsonl"
            write_jsonl(
                agent_path,
                [
                    {
                        "sample_id": "s1",
                        "task_type": "code2test",
                        "gold_files": ["tests/test_a.py"],
                        "steps": [{"path": "tests/test_a.py", "step": 1, "tool": "read", "is_final_context": True}],
                    },
                    {
                        "sample_id": "s2",
                        "task_type": "trace2code",
                        "gold_files": ["src/fix.py"],
                        "steps": [{"path": "tests/test_fix.py", "step": 1, "tool": "read", "is_final_context": True}],
                    },
                    {
                        "sample_id": "s3",
                        "task_type": "trace2code",
                        "gold_files": ["src/late.py"],
                        "steps": [{"path": "src/late.py", "step": 4, "tool": "read", "is_final_context": True}],
                    },
                ],
            )
            write_jsonl(
                retriever_path,
                [
                    {"sample_id": "s1", "task_type": "code2test", "gold_files": ["tests/test_a.py"], "gold_ranks": {"tests/test_a.py": 5}},
                    {"sample_id": "s2", "task_type": "trace2code", "gold_files": ["src/fix.py"], "gold_ranks": {"src/fix.py": 10}},
                    {"sample_id": "s3", "task_type": "trace2code", "gold_files": ["src/late.py"], "gold_ranks": {"src/late.py": 3}},
                ],
            )

            result = report_context_acquisition_cost(
                out_path=root / "report.json",
                markdown_out_path=root / "report.md",
                agent_details=[("agent", agent_path)],
                retriever_details=[("retriever", retriever_path)],
                late_hit_threshold=3,
            )

            agent = result["agents"][0]
            self.assertEqual(agent["samples"], 3)
            self.assertAlmostEqual(agent["any_gold_rate"], 2 / 3)
            self.assertAlmostEqual(agent["miss_rate"], 1 / 3)
            self.assertEqual(agent["late_hit_count"], 1)

            join = result["joins"][0]["subsets"]
            self.assertEqual(join["agent_miss"]["samples"], 1)
            self.assertAlmostEqual(join["agent_miss"]["retriever_any@10"], 1.0)
            self.assertEqual(join["agent_late_hit"]["samples"], 1)
            self.assertAlmostEqual(join["agent_late_hit"]["retriever_any@3"], 1.0)
            comp10 = result["joins"][0]["complementarity"]["@10"]
            self.assertAlmostEqual(comp10["both_hit_rate"], 2 / 3)
            self.assertAlmostEqual(comp10["retriever_only_rate"], 1 / 3)
            self.assertAlmostEqual(comp10["agent_only_rate"], 0.0)
            self.assertAlmostEqual(comp10["both_miss_rate"], 0.0)
            self.assertAlmostEqual(comp10["retriever_only_given_agent_miss"], 1.0)
            self.assertAlmostEqual(comp10["potential_exploration_savings"], 1.0)
            self.assertTrue((root / "report.md").exists())
            self.assertIn("Trajectory-Conditioned Complementarity", (root / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
