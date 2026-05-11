from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .baseline import CANDIDATE_FILTERS
from .io import ensure_parent, read_json, utc_now, write_json

METRIC_KEYS = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")
TASK_ORDER = ("overall", "code2test", "comment2context", "trace2code", "testlog2code")
CANDIDATE_FILTER_ORDER = {name: index for index, name in enumerate(CANDIDATE_FILTERS)}


def report_model_leaderboard(
    eval_dir: Path,
    out_path: Path,
    json_out_path: Path | None = None,
) -> dict[str, Any]:
    summaries = load_eval_summaries(eval_dir)
    rows = leaderboard_rows(summaries)
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "summary_count": len(summaries),
        "row_count": len(rows),
        "rows": rows,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_model_leaderboard_markdown(report), encoding="utf-8")
    return {
        "eval_dir": str(eval_dir),
        "summaries": len(summaries),
        "rows": len(rows),
        "markdown": str(out_path),
        "json": str(json_path),
    }


def load_eval_summaries(eval_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        summary = read_json(path, {})
        if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
            continue
        summaries.append(normalize_summary(path, summary))
    return summaries


def normalize_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    model = str(summary.get("model") or "lexical")
    mode = str(summary.get("mode") or ("embedding" if summary.get("model") else "lexical"))
    candidate_filter = str(summary.get("candidate_filter") or infer_candidate_filter(path))
    return {
        "path": str(path),
        "filename": path.name,
        "mode": mode,
        "model": model,
        "model_label": model_label(model, mode),
        "candidate_filter": candidate_filter,
        "evaluated": int(summary.get("evaluated") or 0),
        "skipped": summary.get("skipped") or {},
        "metrics": summary.get("metrics") or {},
    }


def infer_candidate_filter(path: Path) -> str:
    stem = path.stem
    for candidate_filter in CANDIDATE_FILTERS:
        if candidate_filter != "all_files" and f"_{candidate_filter}_summary" in stem:
            return candidate_filter
    return "all_files"


def model_label(model: str, mode: str) -> str:
    if model == "lexical" or mode in {"corpus", "dry_run", "lexical"}:
        return "lexical"
    path = Path(model)
    if path.is_absolute() or model.startswith(("./", "../", "~/", "models/")) or model.count("/") > 1:
        name = path.name
        return name or model
    return model


def leaderboard_rows(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        skipped = summary.get("skipped") or {}
        skipped_total = sum(int(value) for value in skipped.values())
        for task, metrics in sorted((summary.get("metrics") or {}).items(), key=lambda item: task_sort_key(item[0])):
            rows.append(
                {
                    "model": summary["model"],
                    "model_label": summary["model_label"],
                    "mode": summary["mode"],
                    "candidate_filter": summary["candidate_filter"],
                    "task": task,
                    "samples": int(metrics.get("samples") or 0),
                    "evaluated": summary["evaluated"],
                    "skipped": skipped,
                    "skipped_total": skipped_total,
                    "source": summary["path"],
                    **{key: float(metrics.get(key) or 0.0) for key in METRIC_KEYS},
                }
            )
    rows.sort(key=row_sort_key)
    return rows


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task_sort_key(str(row["task"])),
        CANDIDATE_FILTER_ORDER.get(str(row["candidate_filter"]), 99),
        -float(row["MRR"]),
        -float(row["Recall@20"]),
        str(row["model_label"]),
    )


def task_sort_key(task: str) -> tuple[int, str]:
    if task in TASK_ORDER:
        return (TASK_ORDER.index(task), task)
    return (len(TASK_ORDER), task)


def render_model_leaderboard_markdown(report: dict[str, Any]) -> str:
    rows = list(report["rows"])
    lines = [
        "# Model Leaderboard",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Summary files: `{report['summary_count']}`",
        f"- Rows: `{report['row_count']}`",
        "",
    ]
    for task in sorted({row["task"] for row in rows}, key=task_sort_key):
        task_rows = [row for row in rows if row["task"] == task]
        lines.extend(render_task_table(task, task_rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_task_table(task: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {task}",
        "",
        "| Model | Candidate | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | `{candidate}` | {samples} | {r5} | {r10} | {r20} | {mrr} | {gold} | `{source}` |".format(
                model=escape_markdown_cell(str(row["model_label"])),
                candidate=row["candidate_filter"],
                samples=row["samples"],
                r5=format_metric(row["Recall@5"]),
                r10=format_metric(row["Recall@10"]),
                r20=format_metric(row["Recall@20"]),
                mrr=format_metric(row["MRR"]),
                gold=format_metric(row["gold_coverage@8k"]),
                source=Path(str(row["source"])).name,
            )
        )
    return lines


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")
