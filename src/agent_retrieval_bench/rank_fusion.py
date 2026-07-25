from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json
from .model_report import format_metric, task_sort_key

DEFAULT_COMPONENTS: dict[str, dict[str, str]] = {
    "qwen3_8b": {
        "label": "Qwen3-Embedding-8B",
        "details": "Qwen3-Embedding-8B_details.jsonl",
        "summary": "Qwen3-Embedding-8B_summary.json",
    },
    "qwen3_4b": {
        "label": "Qwen3-Embedding-4B",
        "details": "Qwen3-Embedding-4B_details.jsonl",
        "summary": "Qwen3-Embedding-4B_summary.json",
    },
    "repomap": {
        "label": "RepoMap",
        "details": "repomap_all_files_details.jsonl",
        "summary": "repomap_all_files_summary.json",
    },
}

DEFAULT_FUSIONS: list[dict[str, Any]] = [
    {
        "name": "rrf_qwen8b_repomap",
        "label": "RRF(Qwen3-8B + RepoMap)",
        "components": ["qwen3_8b", "repomap"],
    },
    {
        "name": "rrf_qwen4b_repomap",
        "label": "RRF(Qwen3-4B + RepoMap)",
        "components": ["qwen3_4b", "repomap"],
    },
    {
        "name": "rrf_qwen8b_qwen4b_repomap",
        "label": "RRF(Qwen3-8B + Qwen3-4B + RepoMap)",
        "components": ["qwen3_8b", "qwen3_4b", "repomap"],
    },
]

METRIC_KEYS = (
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "Precision@5",
    "Precision@10",
    "Precision@20",
    "F0.5@5",
    "F0.5@10",
    "F0.5@20",
    "MRR",
    "Any@5",
    "Any@10",
    "Any@20",
    "nDCG@20",
    "coverage_auc@20",
)


def report_rank_fusion(
    eval_dir: Path = Path("data/eval/v1_3_reviewed"),
    out_dir: Path = Path("data/eval/v1_4/rank_fusion"),
    report_out: Path = Path("data/reports/v1_4/rank_fusion_report.md"),
    json_out: Path | None = None,
    rrf_k: int = 60,
    components: dict[str, dict[str, str]] | None = None,
    fusions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    component_specs = components or DEFAULT_COMPONENTS
    fusion_specs = fusions or DEFAULT_FUSIONS
    loaded_components = load_components(eval_dir, component_specs)
    summaries = load_component_summaries(eval_dir, component_specs)

    fusion_results = []
    for spec in fusion_specs:
        result = evaluate_fusion(spec, loaded_components, rrf_k=rrf_k)
        write_fusion_outputs(out_dir, result)
        fusion_results.append(result)

    rows = report_rows(fusion_results)
    reference_rows = single_run_reference_rows(summaries)
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "rrf_k": rrf_k,
        "components": component_specs,
        "fusions": [
            {
                "name": result["name"],
                "label": result["label"],
                "components": result["component_labels"],
                "evaluated": result["evaluated"],
                "summary_path": result["summary_path"],
                "details_path": result["details_path"],
            }
            for result in fusion_results
        ],
        "rows": rows,
        "single_run_reference_rows": reference_rows,
    }
    json_path = json_out or report_out.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(report_out)
    report_out.write_text(render_rank_fusion_markdown(report), encoding="utf-8")
    return {
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "fusions": len(fusion_results),
        "rows": len(rows),
        "markdown": str(report_out),
        "json": str(json_path),
    }


def load_components(eval_dir: Path, component_specs: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for key, spec in component_specs.items():
        details_path = eval_dir / spec["details"]
        if not details_path.exists():
            raise FileNotFoundError(f"Missing component details for {key}: {details_path}")
        rows = [row for row in read_jsonl(details_path) if row.get("sample_id")]
        by_sample = {str(row["sample_id"]): row for row in rows}
        loaded[key] = {
            "key": key,
            "label": spec.get("label") or key,
            "details_path": details_path,
            "details": by_sample,
        }
    return loaded


def load_component_summaries(eval_dir: Path, component_specs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    summaries = []
    for key, spec in component_specs.items():
        summary_name = spec.get("summary")
        if not summary_name:
            continue
        summary_path = eval_dir / summary_name
        summary = read_json(summary_path, {})
        if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
            continue
        summaries.append(
            {
                "key": key,
                "label": spec.get("label") or key,
                "path": str(summary_path),
                "metrics": summary["metrics"],
            }
        )
    return summaries


def evaluate_fusion(
    spec: dict[str, Any],
    components: dict[str, dict[str, Any]],
    rrf_k: int,
) -> dict[str, Any]:
    component_keys = list(spec["components"])
    missing = [key for key in component_keys if key not in components]
    if missing:
        raise KeyError(f"Unknown fusion components: {', '.join(missing)}")
    sample_ids = common_sample_ids(components[key]["details"] for key in component_keys)
    if not sample_ids:
        raise ValueError(f"No shared samples for fusion {spec['name']}")

    details = []
    for sample_id in sample_ids:
        component_rows = [components[key]["details"][sample_id] for key in component_keys]
        reference = component_rows[0]
        ranked_files = reciprocal_rank_fusion(
            [list(row.get("top_files") or []) for row in component_rows],
            rrf_k=rrf_k,
            weights=spec.get("weights"),
            component_keys=component_keys,
        )[:20]
        gold_files = list(reference.get("gold_files") or [])
        metrics = file_rank_metrics(gold_files, ranked_files)
        details.append(
            {
                "sample_id": sample_id,
                "task_type": reference.get("task_type"),
                "repo": reference.get("repo"),
                "base_commit": reference.get("base_commit"),
                "candidate_filter": reference.get("candidate_filter") or "all_files",
                "mode": "rank_fusion",
                "model": spec["label"],
                "fusion_name": spec["name"],
                "components": component_keys,
                "component_labels": [components[key]["label"] for key in component_keys],
                "rrf_k": rrf_k,
                "gold_files": gold_files,
                "gold_ranks": gold_ranks(gold_files, ranked_files),
                "top_files": ranked_files,
                "metrics": metrics,
            }
        )

    summary = {
        "mode": "rank_fusion",
        "model": spec["label"],
        "candidate_filter": "all_files",
        "evaluated": len(details),
        "skipped": {},
        "fusion_name": spec["name"],
        "components": component_keys,
        "component_labels": [components[key]["label"] for key in component_keys],
        "rrf_k": rrf_k,
        "metrics": summarize_fusion_details(details),
    }
    return {
        "name": spec["name"],
        "label": spec["label"],
        "component_keys": component_keys,
        "component_labels": [components[key]["label"] for key in component_keys],
        "evaluated": len(details),
        "details": details,
        "summary": summary,
    }


def common_sample_ids(detail_maps: Iterable[dict[str, dict[str, Any]]]) -> list[str]:
    iterator = iter(detail_maps)
    try:
        ids = set(next(iterator))
    except StopIteration:
        return []
    for detail_map in iterator:
        ids &= set(detail_map)
    return sorted(ids)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    component_keys: list[str] | None = None,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    component_keys = component_keys or [str(index) for index in range(len(ranked_lists))]
    for component_key, ranked in zip(component_keys, ranked_lists):
        seen = set()
        weight = float((weights or {}).get(component_key, 1.0))
        for rank, path in enumerate(ranked, start=1):
            if not path or path in seen:
                continue
            scores[path] += weight / float(rrf_k + rank)
            seen.add(path)
    return sorted(scores, key=lambda path: (-scores[path], path))


def file_rank_metrics(gold_files: list[str], ranked_files: list[str]) -> dict[str, float]:
    gold = {path for path in gold_files if path}
    metrics = {}
    for depth in (5, 10, 20):
        recall = recall_at(gold, ranked_files, depth)
        precision = precision_at(gold, ranked_files, depth)
        metrics[f"Recall@{depth}"] = recall
        metrics[f"Precision@{depth}"] = precision
        metrics[f"F0.5@{depth}"] = f_score(precision, recall, beta=0.5)
        metrics[f"Any@{depth}"] = any_gold_at(gold, ranked_files, depth)
    metrics["MRR"] = reciprocal_rank(gold, ranked_files)
    metrics["nDCG@20"] = ndcg_at(gold, ranked_files, 20)
    metrics["coverage_auc@20"] = coverage_auc_at(gold, ranked_files, 20)
    return metrics


def recall_at(gold: set[str], ranked_files: list[str], depth: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(ranked_files[:depth])) / len(gold)


def precision_at(gold: set[str], ranked_files: list[str], depth: int) -> float:
    if depth <= 0:
        return 0.0
    return len(gold.intersection(ranked_files[:depth])) / depth


def f_score(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def any_gold_at(gold: set[str], ranked_files: list[str], depth: int) -> float:
    return 1.0 if gold.intersection(ranked_files[:depth]) else 0.0


def reciprocal_rank(gold: set[str], ranked_files: list[str]) -> float:
    for rank, path in enumerate(ranked_files, start=1):
        if path in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at(gold: set[str], ranked_files: list[str], depth: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, path in enumerate(ranked_files[:depth], start=1) if path in gold)
    ideal_hits = min(len(gold), depth)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def coverage_auc_at(gold: set[str], ranked_files: list[str], depth: int) -> float:
    if depth <= 0:
        return 0.0
    return sum(recall_at(gold, ranked_files, rank) for rank in range(1, depth + 1)) / depth


def gold_ranks(gold_files: list[str], ranked_files: list[str]) -> dict[str, int | None]:
    ranks = {path: rank for rank, path in enumerate(ranked_files, start=1)}
    return {path: ranks.get(path) for path in gold_files}


def summarize_fusion_details(details: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for detail in details:
        metrics = detail.get("metrics") or {}
        grouped["overall"].append(metrics)
        grouped[str(detail.get("task_type") or "unknown")].append(metrics)
    summary = {}
    for task, rows in sorted(grouped.items(), key=lambda item: task_sort_key(item[0])):
        summary[task] = {"samples": len(rows)}
        for key in METRIC_KEYS:
            summary[task][key] = mean(float(row.get(key) or 0.0) for row in rows)
    return summary


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def write_fusion_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    ensure_parent(out_dir / "placeholder")
    summary_path = out_dir / f"{result['name']}_summary.json"
    details_path = out_dir / f"{result['name']}_details.jsonl"
    result["summary_path"] = str(summary_path)
    result["details_path"] = str(details_path)
    write_json(summary_path, result["summary"])
    write_jsonl(details_path, result["details"])


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def report_rows(fusion_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in fusion_results:
        for task, metrics in result["summary"]["metrics"].items():
            rows.append(
                {
                    "fusion": result["label"],
                    "fusion_name": result["name"],
                    "components": " + ".join(result["component_labels"]),
                    "task": task,
                    "samples": int(metrics.get("samples") or 0),
                    **{key: float(metrics.get(key) or 0.0) for key in METRIC_KEYS},
                }
            )
    rows.sort(key=lambda row: (task_sort_key(str(row["task"])), -float(row["MRR"]), str(row["fusion"])))
    return rows


def single_run_reference_rows(component_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in component_summaries:
        for task, metrics in (summary.get("metrics") or {}).items():
            rows.append(
                {
                    "model": summary["label"],
                    "task": task,
                    "samples": int(metrics.get("samples") or 0),
                    "Recall@5": float(metrics.get("Recall@5") or 0.0),
                    "Recall@10": float(metrics.get("Recall@10") or 0.0),
                    "Recall@20": float(metrics.get("Recall@20") or 0.0),
                    "MRR": float(metrics.get("MRR") or 0.0),
                    "nDCG@20": float(metrics.get("nDCG@20") or metrics.get("coverage_auc@20") or 0.0),
                    "source": summary["path"],
                }
            )
    rows.sort(key=lambda row: (task_sort_key(str(row["task"])), -float(row["MRR"]), str(row["model"])))
    return rows


def render_rank_fusion_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# Rank Fusion Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Output dir: `{report['out_dir']}`",
        f"- RRF k: `{report['rrf_k']}`",
        "",
        "## Overall",
        "",
        "| Fusion | Components | Samples | R@5 | R@10 | R@20 | MRR | Any@20 | nDCG@20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [row for row in rows if row["task"] == "overall"]:
        lines.append(render_fusion_row(row))

    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | Fusion | Samples | R@20 | MRR | Any@20 | nDCG@20 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [row for row in rows if row["task"] != "overall"]:
        lines.append(
            "| {task} | {fusion} | {samples} | {r20} | {mrr} | {any20} | {ndcg} |".format(
                task=row["task"],
                fusion=row["fusion"],
                samples=row["samples"],
                r20=format_metric(row["Recall@20"]),
                mrr=format_metric(row["MRR"]),
                any20=format_metric(row["Any@20"]),
                ndcg=format_metric(row["nDCG@20"]),
            )
        )

    reference = report.get("single_run_reference_rows") or []
    if reference:
        lines.extend(
            [
                "",
                "## Component Reference",
                "",
                "| Task | Model | Samples | R@20 | MRR | Source |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in reference:
            lines.append(
                "| {task} | {model} | {samples} | {r20} | {mrr} | `{source}` |".format(
                    task=row["task"],
                    model=row["model"],
                    samples=row["samples"],
                    r20=format_metric(row["Recall@20"]),
                    mrr=format_metric(row["MRR"]),
                    source=Path(str(row["source"])).name,
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def render_fusion_row(row: dict[str, Any]) -> str:
    return "| {fusion} | {components} | {samples} | {r5} | {r10} | {r20} | {mrr} | {any20} | {ndcg} |".format(
        fusion=row["fusion"],
        components=row["components"],
        samples=row["samples"],
        r5=format_metric(row["Recall@5"]),
        r10=format_metric(row["Recall@10"]),
        r20=format_metric(row["Recall@20"]),
        mrr=format_metric(row["MRR"]),
        any20=format_metric(row["Any@20"]),
        ndcg=format_metric(row["nDCG@20"]),
    )
