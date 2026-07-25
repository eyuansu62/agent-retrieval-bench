from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import ensure_parent, read_jsonl, utc_now, write_jsonl
from .trajectory import summarize_trajectory_details, trajectory_detail


def evaluate_ranked_context_as_trajectory(
    *,
    baseline_details: Path,
    top_k: int = 3,
    out_path: Path | None = None,
    details_path: Path | None = None,
    model_label: str | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    source_rows = read_jsonl(baseline_details)
    details: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    final_file_counts: Counter[int] = Counter()
    for row in source_rows:
        gold_files = [str(path) for path in row.get("gold_files") or [] if path]
        if not gold_files:
            skipped["no_gold"] += 1
            continue
        top_files = [str(path) for path in row.get("top_files") or [] if path][:top_k]
        if not top_files:
            skipped["no_top_files"] += 1
            continue
        final_file_counts[len(top_files)] += 1
        sample = {
            "id": row.get("sample_id"),
            "task_type": row.get("task_type"),
            "repo": row.get("repo"),
            "base_commit": row.get("base_commit"),
            "gold_spans": row.get("gold_spans") or [],
            "gold_blocks": row.get("gold_blocks") or [],
            "supporting_context_files": row.get("supporting_context_files") or [],
        }
        steps = [
            {
                "step": index,
                "tool": "ranked-context",
                "path": path,
                "start_line": None,
                "end_line": None,
                "kind": "file",
                "symbol": "",
                "content_hash": "",
                "is_final_context": True,
                "is_utilized_context": True,
            }
            for index, path in enumerate(top_files, start=1)
        ]
        detail = trajectory_detail(sample, gold_files, steps)
        detail["source_baseline_detail"] = str(baseline_details)
        detail["top_k"] = top_k
        details.append(detail)

    label = model_label or f"{baseline_details.stem}@{top_k}-context"
    summary = {
        "mode": "ranked_context_as_trajectory",
        "model": label,
        "baseline_details": str(baseline_details),
        "top_k": top_k,
        "evaluated": len(details),
        "skipped": dict(skipped),
        "context": {
            "average_final_files": average_final_files(final_file_counts),
            "final_file_count_distribution": {str(key): value for key, value in sorted(final_file_counts.items())},
        },
        "metrics": summarize_trajectory_details(details),
        "generated_at": utc_now(),
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        write_jsonl(details_path, details)
    return summary


def average_final_files(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(size * count for size, count in counts.items()) / total
