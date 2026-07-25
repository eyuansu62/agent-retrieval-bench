from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .baseline import (
    block_metrics_at_budget,
    f_score,
    gold_blocks,
    gold_spans,
    line_metrics_at_budget,
    path_values,
    supporting_context_files,
    target_gold_files,
)
from .corpus import sample_paths_from_derived
from .io import ensure_parent, read_jsonl, utc_now

TRAJECTORY_METRIC_KEYS = (
    "retrieved_file_recall",
    "retrieved_file_precision",
    "retrieved_file_f1",
    "final_file_recall",
    "final_file_precision",
    "final_file_f1",
    "utilized_file_recall",
    "utilized_file_precision",
    "utilized_file_f1",
    "retrieved_supporting_file_recall",
    "retrieved_supporting_file_precision",
    "retrieved_supporting_file_f1",
    "final_supporting_file_recall",
    "final_supporting_file_precision",
    "final_supporting_file_f1",
    "utilized_supporting_file_recall",
    "utilized_supporting_file_precision",
    "utilized_supporting_file_f1",
    "retrieved_gold_or_supporting_file_recall",
    "retrieved_gold_or_supporting_file_precision",
    "retrieved_gold_or_supporting_file_f1",
    "final_gold_or_supporting_file_recall",
    "final_gold_or_supporting_file_precision",
    "final_gold_or_supporting_file_f1",
    "utilized_gold_or_supporting_file_recall",
    "utilized_gold_or_supporting_file_precision",
    "utilized_gold_or_supporting_file_f1",
    "final_usage_drop",
    "utilization_drop",
    "trajectory_redundancy",
    "line_recall@trajectory",
    "line_precision@trajectory",
    "line_f1@trajectory",
    "block_recall@trajectory",
    "block_precision@trajectory",
    "block_f1@trajectory",
)


def evaluate_trajectories(
    *,
    derived: Path,
    trajectory_paths: Iterable[Path],
    out_path: Path | None = None,
    details_path: Path | None = None,
    model_label: str = "trajectory-log",
    supporting_context_annotations: Path | None = None,
) -> dict[str, Any]:
    samples = {str(row.get("id") or ""): row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)}
    groups = load_trajectory_groups(trajectory_paths)
    support_annotations = load_supporting_context_annotations(supporting_context_annotations)
    details: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    for sample_id, steps in sorted(groups.items()):
        sample = samples.get(sample_id)
        if not sample:
            skipped["missing_sample"] += 1
            continue
        gold_files = target_gold_files(sample)
        if not gold_files:
            skipped["no_gold"] += 1
            continue
        normalized_steps = [step for step in (normalize_trajectory_step(row, index + 1) for index, row in enumerate(steps)) if step]
        if not normalized_steps:
            skipped["empty_trajectory"] += 1
            continue
        details.append(
            trajectory_detail(
                sample,
                gold_files,
                normalized_steps,
                supporting_files=support_annotations.get(sample_id),
            )
        )
    summary = {
        "mode": "trajectory",
        "model": model_label,
        "derived": str(derived),
        "trajectory_files": [str(path) for path in trajectory_paths],
        "evaluated": len(details),
        "skipped": dict(skipped),
        "metrics": summarize_trajectory_details(details),
        "generated_at": utc_now(),
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        write_jsonl(details_path, details)
    return summary


def load_supporting_context_annotations(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    annotations: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if not sample_id:
            continue
        values = row.get("supporting_context_files")
        if values is None:
            values = row.get("supporting_files")
        if values is None:
            values = row.get("paths")
        annotations[sample_id] = dedupe_paths(path_values(values or []))
    return annotations


def load_trajectory_groups(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            trajectory = row.get("trajectory")
            if isinstance(trajectory, list):
                if sample_id:
                    groups[sample_id].extend(item for item in trajectory if isinstance(item, dict))
                continue
            if sample_id:
                groups[sample_id].append(row)
    return groups


def normalize_trajectory_step(row: dict[str, Any], default_step: int) -> dict[str, Any] | None:
    path = str(row.get("path") or "")
    if not path:
        return None
    start_line = optional_int(row.get("start_line"))
    end_line = optional_int(row.get("end_line"))
    if start_line is not None and end_line is not None and (start_line <= 0 or end_line < start_line):
        start_line = None
        end_line = None
    return {
        "step": optional_int(row.get("step")) or optional_int(row.get("turn")) or default_step,
        "tool": str(row.get("tool") or row.get("action") or "read"),
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "kind": str(row.get("kind") or "block"),
        "symbol": str(row.get("symbol") or ""),
        "content_hash": str(row.get("content_hash") or ""),
        "is_final_context": bool(row.get("is_final_context") or row.get("final") or row.get("selected")),
        "is_utilized_context": bool(row.get("is_utilized_context") or row.get("utilized") or row.get("used")),
    }


def trajectory_detail(
    sample: dict[str, Any],
    gold_files: list[str],
    steps: list[dict[str, Any]],
    *,
    supporting_files: list[str] | None = None,
) -> dict[str, Any]:
    retrieved = [step for step in steps if step.get("path")]
    final = [step for step in retrieved if step.get("is_final_context")] or retrieved
    utilized = [step for step in retrieved if step.get("is_utilized_context")] or final
    spans = gold_spans(sample)
    blocks = gold_blocks(sample)
    supporting = dedupe_paths(supporting_context_files(sample) if supporting_files is None else supporting_files)
    supporting = [path for path in supporting if path not in set(gold_files)]
    gold_or_supporting = dedupe_paths([*gold_files, *supporting])
    retrieved_file = file_set_metrics(gold_files, [str(step["path"]) for step in retrieved])
    final_file = file_set_metrics(gold_files, [str(step["path"]) for step in final])
    utilized_file = file_set_metrics(gold_files, [str(step["path"]) for step in utilized])
    metrics = {
        "retrieved_file_recall": retrieved_file["recall"],
        "retrieved_file_precision": retrieved_file["precision"],
        "retrieved_file_f1": retrieved_file["f1"],
        "final_file_recall": final_file["recall"],
        "final_file_precision": final_file["precision"],
        "final_file_f1": final_file["f1"],
        "utilized_file_recall": utilized_file["recall"],
        "utilized_file_precision": utilized_file["precision"],
        "utilized_file_f1": utilized_file["f1"],
        "final_usage_drop": max(0.0, retrieved_file["recall"] - final_file["recall"]),
        "utilization_drop": max(0.0, retrieved_file["recall"] - utilized_file["recall"]),
        "trajectory_redundancy": trajectory_redundancy(retrieved),
    }
    if supporting:
        add_named_file_metrics(metrics, "retrieved_supporting_file", supporting, retrieved)
        add_named_file_metrics(metrics, "final_supporting_file", supporting, final)
        add_named_file_metrics(metrics, "utilized_supporting_file", supporting, utilized)
        add_named_file_metrics(metrics, "retrieved_gold_or_supporting_file", gold_or_supporting, retrieved)
        add_named_file_metrics(metrics, "final_gold_or_supporting_file", gold_or_supporting, final)
        add_named_file_metrics(metrics, "utilized_gold_or_supporting_file", gold_or_supporting, utilized)
    detail = {
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "gold_files": gold_files,
        "supporting_context_files": supporting,
        "gold_or_supporting_context_files": gold_or_supporting,
        "gold_spans": spans,
        "gold_blocks": blocks,
        "steps": steps,
        "metrics": metrics,
    }
    if spans:
        line_metrics = line_metrics_at_budget(spans, steps_to_chunks(retrieved), context_budget=10**12)
        detail["line_metrics"] = line_metrics
        metrics.update(
            {
                "line_recall@trajectory": line_metrics["line_recall@8k"],
                "line_precision@trajectory": line_metrics["line_precision@8k"],
                "line_f1@trajectory": line_metrics["line_f1@8k"],
            }
        )
    if blocks:
        block_metrics = block_metrics_at_budget(blocks, steps_to_chunks(retrieved), context_budget=10**12)
        detail["block_metrics"] = block_metrics
        metrics.update(
            {
                "block_recall@trajectory": block_metrics["block_recall@8k"],
                "block_precision@trajectory": block_metrics["block_precision@8k"],
                "block_f1@trajectory": block_metrics["block_f1@8k"],
            }
        )
    return detail


def file_set_metrics(gold_files: list[str], predicted_files: list[str]) -> dict[str, float]:
    gold = set(gold_files)
    predicted = {path for path in predicted_files if path}
    overlap = len(gold & predicted)
    recall = overlap / len(gold) if gold else 0.0
    precision = overlap / len(predicted) if predicted else 0.0
    return {"recall": recall, "precision": precision, "f1": f_score(precision, recall)}


def add_named_file_metrics(metrics: dict[str, float], prefix: str, target_files: list[str], steps: list[dict[str, Any]]) -> None:
    values = file_set_metrics(target_files, [str(step["path"]) for step in steps])
    metrics[f"{prefix}_recall"] = values["recall"]
    metrics[f"{prefix}_precision"] = values["precision"]
    metrics[f"{prefix}_f1"] = values["f1"]


def dedupe_paths(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def trajectory_redundancy(steps: list[dict[str, Any]]) -> float:
    keys = [(step.get("path"), step.get("start_line"), step.get("end_line"), step.get("content_hash")) for step in steps]
    if not keys:
        return 0.0
    return max(0, len(keys) - len(set(keys))) / len(keys)


def steps_to_chunks(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for step in steps:
        chunks.append(
            {
                "path": step.get("path"),
                "start_line": step.get("start_line") or 0,
                "end_line": step.get("end_line") or 0,
                "kind": step.get("kind") or "block",
                "symbol": step.get("symbol") or "",
                "chunk_id": step.get("content_hash") or "",
                "text": "x",
            }
        )
    return chunks


def summarize_trajectory_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for detail in details:
        task = str(detail.get("task_type") or "unknown")
        grouped["overall"].append(detail["metrics"])
        grouped[task].append(detail["metrics"])
    return {task: average_trajectory_metrics(rows) for task, rows in sorted(grouped.items())}


def average_trajectory_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        **{
            key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows)
            for key in TRAJECTORY_METRIC_KEYS
            if any(key in row for row in rows)
        },
    }


def optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
