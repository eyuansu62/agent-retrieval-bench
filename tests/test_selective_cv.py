from __future__ import annotations

import json

import pytest

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


def test_repo_grouped_folds_keep_repositories_intact_and_both_classes_present():
    rows = []
    for index in range(10):
        repo = f"org/repo-{index}"
        rows.append(make_row(f"p-{index}", repo, "positive", 0.8, 1.0))
        rows.append(make_row(f"n-{index}", repo, "no_gold", 0.2))

    assignments = assign_repo_folds(rows, fold_count=5)

    assert len(assignments) == 10
    for fold_index in range(5):
        held_out = [
            row for row in rows
            if assignments[row["repo"]] == fold_index
        ]
        assert {row["label"] for row in held_out} == {"positive", "no_gold"}


def test_balanced_threshold_and_out_of_fold_metrics():
    rows = []
    for index in range(10):
        repo = f"org/repo-{index}"
        rows.append(make_row(f"p-{index}", repo, "positive", 0.9, 1.0))
        rows.append(make_row(f"n-{index}", repo, "no_gold", 0.1))
    assignments = assign_repo_folds(rows, fold_count=5)

    selected = select_balanced_threshold(rows)
    result = evaluate_group_cv(rows, assignments, fold_count=5)

    assert 0.1 < selected["threshold"] < 0.9
    assert result["out_of_fold"]["balanced_accuracy"] == pytest.approx(1.0)
    assert result["out_of_fold"]["selective_success@20"] == pytest.approx(1.0)
    assert sum(fold["test_samples"] for fold in result["folds"]) == len(rows)
    assert len(result["predictions"]) == len(rows)
    assert len({row["sample_id"] for row in result["predictions"]}) == len(rows)


def test_decision_metrics_counts_positive_utility():
    rows = [
        make_row("p-hit", "org/a", "positive", 0.9, 1.0),
        make_row("p-miss", "org/b", "positive", 0.8, 0.0),
        make_row("n", "org/c", "no_gold", 0.1),
    ]
    metrics = decision_metrics(
        rows,
        {"p-hit": False, "p-miss": False, "n": True},
    )

    assert metrics["positive_pass_rate"] == pytest.approx(1.0)
    assert metrics["no_gold_abstain_rate"] == pytest.approx(1.0)
    assert metrics["selective_success@20"] == pytest.approx(2 / 3)
    assert metrics["accepted_positive_hit@20"] == pytest.approx(0.5)


def test_report_rejects_misaligned_method_samples(tmp_path):
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

    with pytest.raises(ValueError, match="sample alignment differs"):
        report_selective_group_cv(
            {"left": left_path, "right": right_path},
            tmp_path / "report.md",
            fold_count=2,
        )


def test_report_separates_counterfactual_no_gold(tmp_path):
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
    assert report["no_gold_reason_counts"] == {
        "counterfactual_wrong_repo": 4,
        "upstream_dependency": 10,
    }
    assert report["natural_only"]["samples"] == 20
    assert report["natural_only"]["no_gold_samples"] == 10
    assert report["natural_only"]["methods"]["ranker"]["out_of_fold"][
        "balanced_accuracy"
    ] == pytest.approx(1.0)
