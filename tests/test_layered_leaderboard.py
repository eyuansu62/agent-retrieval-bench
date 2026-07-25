from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.layered_leaderboard import report_layered_leaderboard


class LayeredLeaderboardTests(unittest.TestCase):
    def test_report_uses_complementarity_matrix_without_rescue_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "toy_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "evaluated": 2,
                        "metrics": {
                            "overall": {
                                "MRR": 0.5,
                                "Recall@20": 0.75,
                                "samples": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            details = root / "toy_details.jsonl"
            details.write_text("", encoding="utf-8")
            bcy_report = root / "bcy.json"
            bcy_report.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "label": "Toy",
                                "overall": {
                                    "samples": 2,
                                    "BCY@4000": 0.2,
                                    "BCY@8000": 0.4,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            context_selection = root / "context_selection.json"
            context_selection.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "method": "Toy",
                                "type": "retriever",
                                "context": "top-3",
                                "avg_final_files": 3,
                                "file_recall": 0.2,
                                "file_precision": 0.1,
                                "file_f1": 0.15,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            context_cost = root / "context_cost.json"
            context_cost.write_text(
                json.dumps(
                    {
                        "agents": [
                            {
                                "label": "Agent",
                                "events_per_sample_mean": 4,
                                "any_gold_rate": 0.5,
                                "miss_rate": 0.5,
                                "late_hit_rate": 0.25,
                                "median_first_hit_step": 2,
                            }
                        ],
                        "joins": [
                            {
                                "agent": "Agent",
                                "retriever": "Toy",
                                "samples": 2,
                                "complementarity": {
                                    "@20": {
                                        "samples": 2,
                                        "both_hit_rate": 0.25,
                                        "retriever_only_rate": 0.25,
                                        "agent_only_rate": 0.25,
                                        "both_miss_rate": 0.25,
                                        "potential_exploration_savings": 0.5,
                                        "retriever_only_given_agent_miss": 0.5,
                                    }
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = report_layered_leaderboard(
                bcy_report_path=bcy_report,
                context_selection_path=context_selection,
                context_cost_path=context_cost,
                out_path=root / "layered.json",
                markdown_out_path=root / "layered.md",
                runs=(("Toy", "retriever", details),),
            )

            self.assertEqual(report["rank_leaderboard"][0]["method"], "Toy")
            self.assertEqual(report["rank_leaderboard"][0]["BCY@8000"], 0.4)
            self.assertIn("trajectory_complementarity", report)
            self.assertNotIn("trajectory_join", report)
            self.assertNotIn("Rescue@20", report["metric_notes"])
            row = report["trajectory_complementarity"][0]
            self.assertEqual(row["retriever_only_rate"], 0.25)
            markdown = (root / "layered.md").read_text(encoding="utf-8")
            self.assertIn("Trajectory-Conditioned Complementarity @20", markdown)
            self.assertIn("Retriever only", markdown)
            self.assertNotIn("Rescue@20", markdown)


if __name__ == "__main__":
    unittest.main()
