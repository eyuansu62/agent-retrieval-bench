from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.selective_cv import (
    assign_repo_folds,
    decision_metrics,
    evaluate_group_cv,
    report_selective_group_cv,
    select_balanced_threshold,
)


def make_row(
    sample_id: str,
    repo: str,
    label: str,
    confidence: float,
    recall_at20: float = 0.0,
    no_gold_reason: str | None = None,
) -> dict:
    row = {
        "sample_id": sample_id,
        "repo": repo,
        "label": label,
        "confidence": confidence,
    }
    if label == "positive":
        row["metrics"] = {"Recall@20": recall_at20}
    elif no_gold_reason is not None:
        row["no_gold_reason"] = no_gold_reason
    return row


class SelectiveCvTests(unittest.TestCase):
    def test_repo_grouped_folds_keep_repositories_intact_and_both_classes_present(self) -> None:
        rows = []
        for index in range(10):
            repo = f"org/repo-{index}"
            rows.append(make_row(f"p-{index}", repo, "positive", 0.8, 1.0))
            rows.append(make_row(f"n-{index}", repo, "no_gold", 0.2))

        assignments = assign_repo_folds(rows, fold_count=5)

        self.assertEqual(len(assignments), 10)
        for fold_index in range(5):
            held_out = [row for row in rows if assignments[row["repo"]] == fold_index]
            self.assertEqual({row["label"] for row in held_out}, {"positive", "no_gold"})

    def test_balanced_threshold_and_out_of_fold_metrics(self) -> None:
        rows = []
        for index in range(10):
            repo = f"org/repo-{index}"
            rows.append(make_row(f"p-{index}", repo, "positive", 0.9, 1.0))
            rows.append(make_row(f"n-{index}", repo, "no_gold", 0.1))
        assignments = assign_repo_folds(rows, fold_count=5)

        selected = select_balanced_threshold(rows)
        result = evaluate_group_cv(rows, assignments, fold_count=5)

        self.assertGreater(selected["threshold"], 0.1)
        self.assertLess(selected["threshold"], 0.9)
        self.assertAlmostEqual(result["out_of_fold"]["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(result["out_of_fold"]["selective_success@20"], 1.0)
        self.assertEqual(sum(fold["test_samples"] for fold in result["folds"]), len(rows))
        self.assertEqual(len(result["predictions"]), len(rows))
        self.assertEqual(len({row["sample_id"] for row in result["predictions"]}), len(rows))

    def test_decision_metrics_counts_positive_utility(self) -> None:
        rows = [
            make_row("p-hit", "org/a", "positive", 0.9, 1.0),
            make_row("p-miss", "org/b", "positive", 0.8, 0.0),
            make_row("n", "org/c", "no_gold", 0.1),
        ]
        metrics = decision_metrics(
            rows,
            {"p-hit": False, "p-miss": False, "n": True},
        )

        self.assertAlmostEqual(metrics["positive_pass_rate"], 1.0)
        self.assertAlmostEqual(metrics["no_gold_abstain_rate"], 1.0)
        self.assertAlmostEqual(metrics["selective_success@20"], 2 / 3)
        self.assertAlmostEqual(metrics["accepted_positive_hit@20"], 0.5)

    def test_report_rejects_misaligned_method_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            left = [
                make_row("p", "org/a", "positive", 0.9, 1.0),
                make_row("n", "org/b", "no_gold", 0.1),
            ]
            right = [
                make_row("different", "org/a", "positive", 0.9, 1.0),
                make_row("n", "org/b", "no_gold", 0.1),
            ]
            left_path = tmp_path / "left.jsonl"
            right_path = tmp_path / "right.jsonl"
            left_path.write_text(
                "".join(json.dumps(row) + "\n" for row in left),
                encoding="utf-8",
            )
            right_path.write_text(
                "".join(json.dumps(row) + "\n" for row in right),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sample alignment differs"):
                report_selective_group_cv(
                    {"left": left_path, "right": right_path},
                    tmp_path / "report.md",
                    fold_count=2,
                )

    def test_report_separates_counterfactual_no_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rows = []
            for index in range(10):
                repo = f"org/repo-{index}"
                rows.append(make_row(f"p-{index}", repo, "positive", 0.9, 1.0))
                rows.append(
                    make_row(
                        f"n-{index}",
                        repo,
                        "no_gold",
                        0.1,
                        no_gold_reason="upstream_dependency",
                    )
                )
                if index < 4:
                    rows.append(
                        make_row(
                            f"c-{index}",
                            repo,
                            "no_gold",
                            0.05,
                            no_gold_reason="counterfactual_wrong_repo",
                        )
                    )

            detail_path = tmp_path / "details.jsonl"
            detail_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            json_path = tmp_path / "report.json"
            report_selective_group_cv(
                {"ranker": detail_path},
                tmp_path / "report.md",
                json_out_path=json_path,
                fold_count=5,
            )

            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["no_gold_reason_counts"],
                {
                    "counterfactual_wrong_repo": 4,
                    "upstream_dependency": 10,
                },
            )
            self.assertEqual(report["natural_only"]["samples"], 20)
            self.assertEqual(report["natural_only"]["no_gold_samples"], 10)
            self.assertAlmostEqual(
                report["natural_only"]["methods"]["ranker"]["out_of_fold"]["balanced_accuracy"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
