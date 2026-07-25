from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from .io import ensure_parent, read_jsonl, write_json


DEFAULT_SELECTIVE_DETAILS = {
    "lexical": Path(
        "data/eval/v2_selective/"
        "v2_selective_retrieval_natural_all_files_selective_lexical_details.jsonl"
    ),
    "BM25": Path(
        "data/eval/v2_selective/"
        "v2_selective_retrieval_natural_all_files_selective_bm25_details.jsonl"
    ),
    "Jina-0.5B": Path(
        "data/eval/v2_selective_retrieval_natural/"
        "jina-code-embeddings-0.5b_selective_details.jsonl"
    ),
}

CONFUSION_KEYS = (
    "no_gold_abstained",
    "no_gold_returned",
    "positive_returned",
    "positive_abstained",
)

COUNTERFACTUAL_NO_GOLD_REASON = "counterfactual_wrong_repo"


def report_selective_group_cv(
    detail_paths: Mapping[str, Path],
    out_path: Path,
    json_out_path: Path | None = None,
    fold_count: int = 5,
) -> dict[str, Any]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not detail_paths:
        raise ValueError("at least one details file is required")

    details_by_method = {
        method: _load_and_validate_details(path, method)
        for method, path in detail_paths.items()
    }
    reference_method = next(iter(details_by_method))
    reference_rows = details_by_method[reference_method]
    _validate_aligned_samples(details_by_method, reference_method)

    repo_folds = assign_repo_folds(reference_rows, fold_count)
    fold_summary = summarize_folds(reference_rows, repo_folds, fold_count)
    method_results = {
        method: evaluate_group_cv(rows, repo_folds, fold_count)
        for method, rows in details_by_method.items()
    }

    natural_details_by_method = {
        method: [row for row in rows if _is_natural_selective_row(row)]
        for method, rows in details_by_method.items()
    }
    natural_reference_rows = natural_details_by_method[reference_method]
    natural_repo_folds = assign_repo_folds(natural_reference_rows, fold_count)
    natural_fold_summary = summarize_folds(
        natural_reference_rows,
        natural_repo_folds,
        fold_count,
    )
    natural_method_results = {
        method: evaluate_group_cv(rows, natural_repo_folds, fold_count)
        for method, rows in natural_details_by_method.items()
    }

    labels = Counter(str(row["label"]) for row in reference_rows)
    result = {
        "protocol": {
            "name": "repo_grouped_out_of_fold_threshold_calibration",
            "folds": fold_count,
            "group_key": "repo",
            "selection_objective": "balanced_accuracy",
            "selection_tiebreakers": [
                "selective_accuracy",
                "selective_success@20",
            ],
            "confidence": "top retrieval score",
            "abstain_rule": "confidence < fold-specific threshold",
        },
        "samples": len(reference_rows),
        "repos": len(repo_folds),
        "positive_samples": labels["positive"],
        "no_gold_samples": labels["no_gold"],
        "no_gold_reason_counts": dict(
            sorted(
                Counter(
                    str(row.get("no_gold_reason") or "unspecified")
                    for row in reference_rows
                    if row["label"] == "no_gold"
                ).items()
            )
        ),
        "folds": fold_summary,
        "repo_folds": dict(sorted(repo_folds.items())),
        "methods": method_results,
        "natural_only": {
            "definition": (
                "All positive samples plus no-gold samples whose reason is not "
                f"{COUNTERFACTUAL_NO_GOLD_REASON}."
            ),
            "samples": len(natural_reference_rows),
            "repos": len(natural_repo_folds),
            "positive_samples": sum(
                row["label"] == "positive" for row in natural_reference_rows
            ),
            "no_gold_samples": sum(
                row["label"] == "no_gold" for row in natural_reference_rows
            ),
            "folds": natural_fold_summary,
            "repo_folds": dict(sorted(natural_repo_folds.items())),
            "methods": natural_method_results,
        },
        "inputs": {method: str(path) for method, path in detail_paths.items()},
    }

    ensure_parent(out_path)
    out_path.write_text(render_group_cv_report(result), encoding="utf-8")
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, result)
    return {
        "samples": result["samples"],
        "repos": result["repos"],
        "folds": fold_count,
        "methods": len(method_results),
        "natural_samples": result["natural_only"]["samples"],
        "markdown": str(out_path),
        "json": str(json_path),
    }


def assign_repo_folds(
    rows: list[dict[str, Any]],
    fold_count: int,
) -> dict[str, int]:
    by_repo: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_repo[str(row["repo"])][str(row["label"])] += 1
    if len(by_repo) < fold_count:
        raise ValueError(
            f"cannot create {fold_count} repo-grouped folds from {len(by_repo)} repos"
        )

    total_positive = sum(counts["positive"] for counts in by_repo.values())
    total_no_gold = sum(counts["no_gold"] for counts in by_repo.values())
    total_samples = total_positive + total_no_gold
    target_positive = total_positive / fold_count
    target_no_gold = total_no_gold / fold_count
    target_total = total_samples / fold_count
    target_groups = len(by_repo) / fold_count

    def group_priority(item: tuple[str, Counter[str]]) -> tuple[float, int, str]:
        repo, counts = item
        positive_ratio = counts["positive"] / max(target_positive, 1.0)
        no_gold_ratio = counts["no_gold"] / max(target_no_gold, 1.0)
        size = counts["positive"] + counts["no_gold"]
        return (-max(positive_ratio, no_gold_ratio), -size, repo)

    fold_counts = [
        {"positive": 0, "no_gold": 0, "groups": 0}
        for _ in range(fold_count)
    ]
    assignments: dict[str, int] = {}
    for repo, counts in sorted(by_repo.items(), key=group_priority):
        candidates: list[tuple[float, int, int]] = []
        for fold_index in range(fold_count):
            projected = [dict(values) for values in fold_counts]
            projected[fold_index]["positive"] += counts["positive"]
            projected[fold_index]["no_gold"] += counts["no_gold"]
            projected[fold_index]["groups"] += 1
            cost = 0.0
            for values in projected:
                fold_total = values["positive"] + values["no_gold"]
                cost += (
                    ((values["positive"] - target_positive) / max(target_positive, 1.0)) ** 2
                    + ((values["no_gold"] - target_no_gold) / max(target_no_gold, 1.0)) ** 2
                    + 0.25 * ((fold_total - target_total) / max(target_total, 1.0)) ** 2
                    + 0.01 * ((values["groups"] - target_groups) / max(target_groups, 1.0)) ** 2
                )
            current_size = (
                fold_counts[fold_index]["positive"]
                + fold_counts[fold_index]["no_gold"]
            )
            candidates.append((cost, current_size, fold_index))
        _, _, selected_fold = min(candidates)
        assignments[repo] = selected_fold
        fold_counts[selected_fold]["positive"] += counts["positive"]
        fold_counts[selected_fold]["no_gold"] += counts["no_gold"]
        fold_counts[selected_fold]["groups"] += 1

    for fold_index, counts in enumerate(fold_counts):
        if not counts["positive"] or not counts["no_gold"]:
            raise ValueError(
                f"fold {fold_index} lacks a class after repo grouping: {counts}"
            )
    return assignments


def summarize_folds(
    rows: list[dict[str, Any]],
    repo_folds: Mapping[str, int],
    fold_count: int,
) -> list[dict[str, Any]]:
    fold_rows: list[list[dict[str, Any]]] = [[] for _ in range(fold_count)]
    fold_repos: list[set[str]] = [set() for _ in range(fold_count)]
    for row in rows:
        repo = str(row["repo"])
        fold_index = repo_folds[repo]
        fold_rows[fold_index].append(row)
        fold_repos[fold_index].add(repo)

    output = []
    for fold_index, held_out in enumerate(fold_rows):
        labels = Counter(str(row["label"]) for row in held_out)
        output.append(
            {
                "fold": fold_index,
                "samples": len(held_out),
                "positive": labels["positive"],
                "no_gold": labels["no_gold"],
                "repos": sorted(fold_repos[fold_index]),
            }
        )
    return output


def evaluate_group_cv(
    rows: list[dict[str, Any]],
    repo_folds: Mapping[str, int],
    fold_count: int,
) -> dict[str, Any]:
    fold_results = []
    decisions: dict[str, bool] = {}
    predictions = []
    thresholds = []
    for fold_index in range(fold_count):
        train = [
            row for row in rows
            if repo_folds[str(row["repo"])] != fold_index
        ]
        test = [
            row for row in rows
            if repo_folds[str(row["repo"])] == fold_index
        ]
        selected = select_balanced_threshold(train)
        threshold = float(selected["threshold"])
        thresholds.append(threshold)
        test_decisions = {
            str(row["sample_id"]): _predict_abstain(row, threshold)
            for row in test
        }
        decisions.update(test_decisions)
        predictions.extend(
            {
                "sample_id": str(row["sample_id"]),
                "repo": str(row["repo"]),
                "label": str(row["label"]),
                "fold": fold_index,
                "confidence": float(row.get("confidence") or 0.0),
                "threshold": threshold,
                "abstained": test_decisions[str(row["sample_id"])],
            }
            for row in test
        )
        train_labels = Counter(str(row["label"]) for row in train)
        test_labels = Counter(str(row["label"]) for row in test)
        fold_results.append(
            {
                "fold": fold_index,
                "threshold": threshold,
                "train_samples": len(train),
                "train_positive": train_labels["positive"],
                "train_no_gold": train_labels["no_gold"],
                "test_samples": len(test),
                "test_positive": test_labels["positive"],
                "test_no_gold": test_labels["no_gold"],
                "calibration_metrics": selected,
                "test_metrics": decision_metrics(test, test_decisions),
            }
        )

    aggregate = decision_metrics(rows, decisions)
    no_abstain = decision_metrics(
        rows,
        {str(row["sample_id"]): False for row in rows},
    )
    finite_thresholds = [value for value in thresholds if math.isfinite(value)]
    return {
        "out_of_fold": aggregate,
        "no_abstain": no_abstain,
        "threshold_summary": {
            "mean": mean(finite_thresholds) if finite_thresholds else None,
            "std": pstdev(finite_thresholds) if len(finite_thresholds) > 1 else 0.0,
            "min": min(finite_thresholds) if finite_thresholds else None,
            "max": max(finite_thresholds) if finite_thresholds else None,
            "values": thresholds,
        },
        "folds": fold_results,
        "predictions": sorted(predictions, key=lambda row: row["sample_id"]),
    }


def select_balanced_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = threshold_sweep(rows)
    if not candidates:
        raise ValueError("cannot calibrate a threshold from no rows")
    return max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["selective_accuracy"],
            row["selective_success@20"],
        ),
    )


def threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    scores = sorted({float(row.get("confidence") or 0.0) for row in rows})
    thresholds = [-math.inf]
    thresholds.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    thresholds.append(scores[-1] + max(1e-9, abs(scores[-1]) * 1e-9))
    return [
        decision_metrics(
            rows,
            {
                str(row["sample_id"]): _predict_abstain(row, threshold)
                for row in rows
            },
            threshold=threshold,
        )
        for threshold in thresholds
    ]


def decision_metrics(
    rows: list[dict[str, Any]],
    decisions: Mapping[str, bool],
    threshold: float | None = None,
) -> dict[str, Any]:
    confusion = Counter({key: 0 for key in CONFUSION_KEYS})
    accepted_positive_recalls = []
    accepted_positive_hits = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in decisions:
            raise ValueError(f"missing abstention decision for sample {sample_id}")
        predict_abstain = decisions[sample_id]
        is_no_gold = row["label"] == "no_gold"
        if is_no_gold and predict_abstain:
            confusion["no_gold_abstained"] += 1
        elif is_no_gold:
            confusion["no_gold_returned"] += 1
        elif predict_abstain:
            confusion["positive_abstained"] += 1
        else:
            confusion["positive_returned"] += 1
            recall_at20 = float((row.get("metrics") or {}).get("Recall@20") or 0.0)
            accepted_positive_recalls.append(recall_at20)
            accepted_positive_hits += int(recall_at20 > 0.0)

    positive_total = confusion["positive_returned"] + confusion["positive_abstained"]
    no_gold_total = confusion["no_gold_abstained"] + confusion["no_gold_returned"]
    total = positive_total + no_gold_total
    accepted = confusion["positive_returned"] + confusion["no_gold_returned"]
    positive_pass_rate = confusion["positive_returned"] / positive_total if positive_total else 0.0
    no_gold_abstain_rate = confusion["no_gold_abstained"] / no_gold_total if no_gold_total else 0.0
    selective_accuracy = (
        confusion["positive_returned"] + confusion["no_gold_abstained"]
    ) / total if total else 0.0
    balanced_accuracy = (
        positive_pass_rate + no_gold_abstain_rate
    ) / 2.0 if positive_total and no_gold_total else selective_accuracy
    selective_success = (
        confusion["no_gold_abstained"] + accepted_positive_hits
    ) / total if total else 0.0
    output = {
        "total": total,
        "positive_total": positive_total,
        "no_gold_total": no_gold_total,
        "coverage": accepted / total if total else 0.0,
        "positive_pass_rate": positive_pass_rate,
        "no_gold_abstain_rate": no_gold_abstain_rate,
        "selective_accuracy": selective_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "selective_success@20": selective_success,
        "accepted_positive_recall@20": (
            mean(accepted_positive_recalls) if accepted_positive_recalls else 0.0
        ),
        "accepted_positive_hit@20": (
            accepted_positive_hits / confusion["positive_returned"]
            if confusion["positive_returned"]
            else 0.0
        ),
        "confusion": dict(confusion),
    }
    if threshold is not None:
        output["threshold"] = threshold
    return output


def render_group_cv_report(result: dict[str, Any]) -> str:
    lines = [
        f"# Selective Retrieval: Repo-Grouped {result['protocol']['folds']}-Fold Cross-Validation",
        "",
        f"- Samples: {result['samples']} "
        f"({result['positive_samples']} positive, {result['no_gold_samples']} no-gold)",
        "- No-gold reasons: "
        + ", ".join(
            f"{reason}={count}"
            for reason, count in result["no_gold_reason_counts"].items()
        ),
        f"- Repository groups: {result['repos']}",
        f"- Folds: {result['protocol']['folds']}",
        "- Calibration: maximize balanced accuracy on four folds; evaluate once on held-out repositories.",
        "- Aggregate metrics use one out-of-fold decision per sample.",
        "",
        "## Mixed Out-of-Fold Results",
        "",
        "| Ranker | Pos. pass | No-gold abstain | Balanced acc. | Selective acc. | Success@20 | Coverage | No-abstain Success@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_result in result["methods"].items():
        metrics = method_result["out_of_fold"]
        baseline = method_result["no_abstain"]
        lines.append(
            f"| {method} | {metrics['positive_pass_rate']:.4f} | "
            f"{metrics['no_gold_abstain_rate']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"{metrics['selective_accuracy']:.4f} | "
            f"{metrics['selective_success@20']:.4f} | "
            f"{metrics['coverage']:.4f} | "
            f"{baseline['selective_success@20']:.4f} |"
        )

    natural = result["natural_only"]
    lines.extend(
        [
            "",
            "## Natural Evidence-Only Out-of-Fold Results",
            "",
            f"This view contains {natural['positive_samples']} positives and "
            f"{natural['no_gold_samples']} natural no-gold samples. Thresholds are "
            "recalibrated within the natural-only repo-grouped folds; counterfactual "
            "wrong-repository negatives are excluded from both calibration and evaluation.",
            "",
            "| Ranker | Pos. pass | No-gold abstain | Balanced acc. | Selective acc. | Success@20 | Coverage | No-abstain Success@20 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, method_result in natural["methods"].items():
        metrics = method_result["out_of_fold"]
        baseline = method_result["no_abstain"]
        lines.append(
            f"| {method} | {metrics['positive_pass_rate']:.4f} | "
            f"{metrics['no_gold_abstain_rate']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"{metrics['selective_accuracy']:.4f} | "
            f"{metrics['selective_success@20']:.4f} | "
            f"{metrics['coverage']:.4f} | "
            f"{baseline['selective_success@20']:.4f} |"
        )

    lines.extend(["", "## Fold Composition", ""])
    lines.extend(
        [
            "| Fold | Samples | Positive | No-gold | Repositories |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for fold in result["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['samples']} | {fold['positive']} | "
            f"{fold['no_gold']} | {', '.join(fold['repos'])} |"
        )

    for method, method_result in result["methods"].items():
        lines.extend(
            [
                "",
                f"## {method} Fold Results",
                "",
                "| Fold | Threshold | Test n | Pos. pass | No-gold abstain | Balanced acc. | Success@20 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in method_result["folds"]:
            metrics = fold["test_metrics"]
            lines.append(
                f"| {fold['fold']} | {fold['threshold']:.6g} | "
                f"{fold['test_samples']} | {metrics['positive_pass_rate']:.4f} | "
                f"{metrics['no_gold_abstain_rate']:.4f} | "
                f"{metrics['balanced_accuracy']:.4f} | "
                f"{metrics['selective_success@20']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _load_and_validate_details(path: Path, method: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no selective details found for {method}: {path}")
    required = {"sample_id", "repo", "label", "confidence"}
    seen = set()
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"{method} row {index} is missing fields: {sorted(missing)}"
            )
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"{method} has duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if row["label"] not in {"positive", "no_gold"}:
            raise ValueError(f"{method} has unknown label: {row['label']!r}")
    return rows


def _is_natural_selective_row(row: Mapping[str, Any]) -> bool:
    return not (
        row.get("label") == "no_gold"
        and row.get("no_gold_reason") == COUNTERFACTUAL_NO_GOLD_REASON
    )


def _validate_aligned_samples(
    details_by_method: Mapping[str, list[dict[str, Any]]],
    reference_method: str,
) -> None:
    reference = {
        str(row["sample_id"]): (
            str(row["repo"]),
            str(row["label"]),
            str(row.get("no_gold_reason") or ""),
        )
        for row in details_by_method[reference_method]
    }
    for method, rows in details_by_method.items():
        observed = {
            str(row["sample_id"]): (
                str(row["repo"]),
                str(row["label"]),
                str(row.get("no_gold_reason") or ""),
            )
            for row in rows
        }
        if observed != reference:
            missing = sorted(set(reference) - set(observed))
            extra = sorted(set(observed) - set(reference))
            changed = sorted(
                sample_id
                for sample_id in set(reference) & set(observed)
                if reference[sample_id] != observed[sample_id]
            )
            raise ValueError(
                f"{method} sample alignment differs from {reference_method}: "
                f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
            )


def _predict_abstain(row: dict[str, Any], threshold: float) -> bool:
    return float(row.get("confidence") or 0.0) < threshold
