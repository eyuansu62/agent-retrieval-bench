import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.pes_calibration import report_pes_calibration


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def detail(sample_id, task_type, gold_files, paths, final_f1=0.0):
    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "repo": "owner/repo",
        "gold_files": gold_files,
        "steps": [{"step": index, "path": path} for index, path in enumerate(paths, start=1)],
        "metrics": {"final_file_f1": final_f1},
    }


class PesCalibrationTests(unittest.TestCase):
    def test_report_pes_calibration_compares_seeded_and_control_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet.jsonl"
            control = root / "control.jsonl"
            seeded = root / "seeded.jsonl"
            write_jsonl(
                packet,
                [
                    {
                        "sample_id": "early",
                        "group": "high_pes",
                        "task_type": "code2test",
                        "repo": "owner/repo",
                        "retriever_hit@20": True,
                        "retriever_first_gold_rank": 2,
                        "PES@20": 5,
                    },
                    {
                        "sample_id": "rescue",
                        "group": "rescue_candidate",
                        "task_type": "trace2code",
                        "repo": "owner/repo",
                        "retriever_hit@20": True,
                        "retriever_first_gold_rank": 1,
                        "PES@20": 0,
                    },
                    {
                        "sample_id": "lost",
                        "group": "negative_control",
                        "task_type": "comment2context",
                        "repo": "owner/repo",
                        "retriever_hit@20": False,
                        "retriever_first_gold_rank": None,
                        "PES@20": 0,
                    },
                ],
            )
            write_jsonl(
                control,
                [
                    detail("early", "code2test", ["gold.py"], ["a.py", "b.py", "gold.py"], final_f1=0.2),
                    detail("rescue", "trace2code", ["fix.py"], ["a.py", "b.py"], final_f1=0.0),
                    detail("lost", "comment2context", ["ctx.py"], ["ctx.py"], final_f1=1.0),
                ],
            )
            write_jsonl(
                seeded,
                [
                    detail("early", "code2test", ["gold.py"], ["gold.py", "a.py"], final_f1=0.6),
                    detail("rescue", "trace2code", ["fix.py"], ["hint.py", "fix.py"], final_f1=0.5),
                    detail("lost", "comment2context", ["ctx.py"], ["wrong.py"], final_f1=0.0),
                ],
            )

            result = report_pes_calibration(
                packet_path=packet,
                control_details_path=control,
                seeded_details_paths=[seeded],
                out_path=root / "report.json",
                markdown_out_path=root / "report.md",
            )

            self.assertEqual(result["counts"]["paired_samples"], 3)
            overall = result["overall"]
            self.assertEqual(overall["first_hit_improved_count"], 1)
            self.assertEqual(overall["rescue_count"], 1)
            self.assertEqual(overall["lost_count"], 1)
            self.assertAlmostEqual(overall["first_hit_delta_mean_both_hit"], 2.0)
            self.assertAlmostEqual(overall["event_saving_mean"], 1 / 3)
            self.assertAlmostEqual(overall["final_file_f1_delta_mean"], -0.1 / 3)
            by_group = result["by_group"]
            self.assertEqual(by_group["rescue_candidate"]["rescue_count"], 1)
            self.assertTrue((root / "report.md").exists())
            self.assertIn("PES Calibration Report", (root / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
