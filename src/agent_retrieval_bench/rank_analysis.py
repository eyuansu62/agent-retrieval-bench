from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .baseline import query_text_for_eval
from .io import ensure_parent, read_json, read_jsonl, truncate_text, utc_now, write_json
from .model_report import model_label, normalize_summary, task_sort_key

DEPTHS = (1, 2, 3, 5, 10, 20, 50, 100, 200)


def report_rank_analysis(
    eval_dir: Path,
    samples_path: Path,
    out_path: Path,
    json_out_path: Path | None = None,
    candidate_filter: str = "all_files",
    top_examples: int = 5,
) -> dict[str, Any]:
    samples = {str(row.get("id")): row for row in read_jsonl(samples_path)}
    runs = load_detail_runs(eval_dir, candidate_filter=candidate_filter)
    if not runs:
        raise FileNotFoundError(f"No *_details.jsonl files found in {eval_dir} for candidate_filter={candidate_filter}")

    depth_rows = []
    for run in runs:
        depth_rows.extend(summarize_run_depths(run))

    samples_by_id = sample_level_rows(runs, samples)
    examples = select_error_examples(samples_by_id, runs, top_examples=top_examples)
    coverage = summarize_cross_model_coverage(samples_by_id)
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "samples": str(samples_path),
        "candidate_filter": candidate_filter,
        "model_count": len(runs),
        "models": [
            {"label": run["label"], "mode": run["mode"], "details": str(run["details_path"])}
            for run in runs
        ],
        "depths": list(DEPTHS),
        "depth_rows": depth_rows,
        "cross_model_coverage": coverage,
        "examples": examples,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_rank_analysis_markdown(report), encoding="utf-8")
    return {
        "eval_dir": str(eval_dir),
        "samples": str(samples_path),
        "candidate_filter": candidate_filter,
        "models": len(runs),
        "depth_rows": len(depth_rows),
        "markdown": str(out_path),
        "json": str(json_path),
    }


def load_detail_runs(eval_dir: Path, candidate_filter: str = "all_files") -> list[dict[str, Any]]:
    runs = []
    for details_path in sorted(eval_dir.glob("*_details.jsonl")):
        summary_path = details_path.with_name(details_path.name.replace("_details.jsonl", "_summary.json"))
        summary = read_json(summary_path, {})
        if isinstance(summary, dict) and summary.get("metrics"):
            normalized = normalize_summary(summary_path, summary)
            label = str(normalized["model_label"])
            mode = str(normalized["mode"])
            model = str(normalized["model"])
        else:
            stem = details_path.name.replace("_details.jsonl", "")
            label = stem
            mode = "unknown"
            model = stem
        details = [
            row
            for row in read_jsonl(details_path)
            if str(row.get("candidate_filter") or "all_files") == candidate_filter
        ]
        if not details:
            continue
        runs.append(
            {
                "details_path": details_path,
                "summary_path": summary_path if summary_path.exists() else None,
                "model": model,
                "label": model_label(label, mode) if mode == "embedding" else label,
                "mode": mode,
                "details": details,
            }
        )
    runs.sort(key=lambda run: (mode_sort_key(str(run["mode"])), str(run["label"])))
    return runs


def mode_sort_key(mode: str) -> int:
    return {"embedding": 0, "repomap": 1, "corpus": 2, "lexical": 2}.get(mode, 9)


def summarize_run_depths(run: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in run["details"]:
        grouped["overall"].append(detail)
        grouped[str(detail.get("task_type") or "unknown")].append(detail)

    rows = []
    for task, details in sorted(grouped.items(), key=lambda item: task_sort_key(item[0])):
        first_ranks = [first_gold_rank(detail) for detail in details]
        found_ranks = [rank for rank in first_ranks if rank is not None]
        row: dict[str, Any] = {
            "model": run["label"],
            "mode": run["mode"],
            "task": task,
            "samples": len(details),
            "missing_first_gold_rank": sum(1 for rank in first_ranks if rank is None),
            "median_first_gold_rank_hit_only": percentile_nearest(found_ranks, 0.5),
            "p90_first_gold_rank_hit_only": percentile_nearest(found_ranks, 0.9),
            "mean_mrr": mean(float((detail.get("metrics") or {}).get("MRR") or 0.0) for detail in details),
            "mean_gold_coverage@8k": mean(
                float((detail.get("metrics") or {}).get("gold_coverage@8k") or 0.0) for detail in details
            ),
        }
        for depth in DEPTHS:
            row[f"any_gold@{depth}"] = mean(
                1.0 if rank is not None and rank <= depth else 0.0 for rank in first_ranks
            )
            row[f"gold_recall@{depth}"] = mean(gold_recall_at(detail, depth) for detail in details)
        rows.append(row)
    return rows


def sample_level_rows(
    runs: list[dict[str, Any]],
    samples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for run in runs:
        for detail in run["details"]:
            sample_id = str(detail.get("sample_id") or "")
            if not sample_id:
                continue
            sample = samples.get(sample_id, {})
            entry = by_sample.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "task": str(detail.get("task_type") or sample.get("task_type") or ""),
                    "repo": str(detail.get("repo") or sample.get("repo") or ""),
                    "gold_files": list(detail.get("gold_files") or []),
                    "given_files": list(((sample.get("gold") or {}).get("given_files") or [])),
                    "pr_url": str(((sample.get("metadata") or {}).get("pr_url") or "")),
                    "query_excerpt": one_line(truncate_text(query_text_for_eval(sample), 280)),
                    "models": {},
                },
            )
            entry["models"][run["label"]] = {
                "mode": run["mode"],
                "first_gold_rank": first_gold_rank(detail),
                "Recall@20": float((detail.get("metrics") or {}).get("Recall@20") or 0.0),
                "MRR": float((detail.get("metrics") or {}).get("MRR") or 0.0),
                "top_files": list(detail.get("top_files") or [])[:5],
                "gold_ranks": detail.get("gold_ranks") or {},
            }
    return sorted(by_sample.values(), key=lambda row: (task_sort_key(row["task"]), row["repo"], row["sample_id"]))


def summarize_cross_model_coverage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped["overall"].append(sample)
        grouped[str(sample["task"])].append(sample)
    result = {}
    for task, rows in sorted(grouped.items(), key=lambda item: task_sort_key(item[0])):
        counts = Counter(models_hit_at_20(row) for row in rows)
        result[task] = {
            "samples": len(rows),
            "hit_model_count_distribution": {str(key): counts.get(key, 0) for key in range(max(counts or {0}) + 1)},
            "all_models_miss@20": counts.get(0, 0),
            "any_model_hit@20": len(rows) - counts.get(0, 0),
        }
    return result


def select_error_examples(
    samples: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    top_examples: int,
) -> dict[str, Any]:
    embedding_labels = [run["label"] for run in runs if run["mode"] == "embedding"]
    repomap_labels = [run["label"] for run in runs if run["mode"] == "repomap"]
    repomap_label = repomap_labels[0] if repomap_labels else ""
    all_misses_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in sorted({str(row["task"]) for row in samples}, key=task_sort_key):
        rows = [row for row in samples if row["task"] == task and models_hit_at_20(row) == 0]
        all_misses_by_task[task] = [format_example(row) for row in rows[:top_examples]]

    structure_wins = []
    embedding_wins = []
    for row in samples:
        if row["task"] != "trace2code" or not repomap_label:
            continue
        repomap = row["models"].get(repomap_label, {})
        embedding_hit = any(row["models"].get(label, {}).get("Recall@20", 0.0) > 0.0 for label in embedding_labels)
        if repomap.get("Recall@20", 0.0) > 0.0 and not embedding_hit:
            structure_wins.append(format_example(row, focus_models=[repomap_label, *embedding_labels[:2]]))
        if repomap.get("Recall@20", 0.0) <= 0.0 and embedding_hit:
            best_embedding = best_model(row, include=embedding_labels)
            embedding_wins.append(format_example(row, focus_models=[best_embedding, repomap_label]))

    given_file_traps = []
    for row in samples:
        if row["task"] != "comment2context" or not row["given_files"]:
            continue
        given = set(row["given_files"])
        for label, result in sorted(row["models"].items()):
            top_files = list(result.get("top_files") or [])
            if top_files and top_files[0] in given and float(result.get("Recall@20") or 0.0) <= 0.0:
                example = format_example(row, focus_models=[label])
                example["trap_model"] = label
                given_file_traps.append(example)
                break

    return {
        "all_model_misses@20_by_task": all_misses_by_task,
        "trace2code_repomap_hits_embeddings_miss@20": structure_wins[:top_examples],
        "trace2code_embedding_hits_repomap_miss@20": embedding_wins[:top_examples],
        "comment2context_given_file_top1_gold_miss@20": given_file_traps[:top_examples],
    }


def format_example(row: dict[str, Any], focus_models: list[str] | None = None) -> dict[str, Any]:
    labels = focus_models or [best_model(row)]
    model_results = {}
    for label in labels:
        if not label or label not in row["models"]:
            continue
        result = row["models"][label]
        model_results[label] = {
            "first_gold_rank": result.get("first_gold_rank"),
            "Recall@20": result.get("Recall@20"),
            "MRR": result.get("MRR"),
            "top_files": result.get("top_files"),
        }
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "repo": row["repo"],
        "gold_files": row["gold_files"],
        "given_files": row["given_files"],
        "query_excerpt": row["query_excerpt"],
        "pr_url": row["pr_url"],
        "model_results": model_results,
    }


def best_model(row: dict[str, Any], include: list[str] | None = None) -> str:
    labels = include or list(row["models"])
    labels = [label for label in labels if label in row["models"]]
    if not labels:
        return ""
    return max(
        labels,
        key=lambda label: (
            float(row["models"][label].get("MRR") or 0.0),
            float(row["models"][label].get("Recall@20") or 0.0),
            -(row["models"][label].get("first_gold_rank") or 10**12),
            label,
        ),
    )


def models_hit_at_20(row: dict[str, Any]) -> int:
    return sum(1 for result in row["models"].values() if float(result.get("Recall@20") or 0.0) > 0.0)


def first_gold_rank(detail: dict[str, Any]) -> int | None:
    ranks = []
    for value in (detail.get("gold_ranks") or {}).values():
        if value is None:
            continue
        rank = int(value)
        if rank > 0:
            ranks.append(rank)
    return min(ranks) if ranks else None


def gold_recall_at(detail: dict[str, Any], depth: int) -> float:
    gold_files = list(detail.get("gold_files") or [])
    if not gold_files:
        return 0.0
    ranks = detail.get("gold_ranks") or {}
    hits = 0
    for path in gold_files:
        rank = ranks.get(path)
        if rank is not None and int(rank) <= depth:
            hits += 1
    return hits / len(gold_files)


def mean(values: Any) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(float(value) for value in rows) / len(rows)


def percentile_nearest(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def render_rank_analysis_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rank and Error Analysis",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Samples: `{report['samples']}`",
        f"- Candidate filter: `{report['candidate_filter']}`",
        f"- Models: `{report['model_count']}`",
        "",
        "## Overall First-Gold CDF",
        "",
        "| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = [row for row in report["depth_rows"] if row["task"] == "overall"]
    for row in sorted(overall, key=lambda item: (-float(item["mean_mrr"]), str(item["model"]))):
        lines.append(render_depth_row(row))
    lines.extend(["", "## Task First-Gold CDF", ""])
    for task in sorted({row["task"] for row in report["depth_rows"] if row["task"] != "overall"}, key=task_sort_key):
        lines.extend([f"### {task}", ""])
        lines.extend(
            [
                "| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        task_rows = [row for row in report["depth_rows"] if row["task"] == task]
        for row in sorted(task_rows, key=lambda item: (-float(item["mean_mrr"]), str(item["model"]))):
            lines.append(render_depth_row(row))
        lines.append("")

    lines.extend(render_coverage(report["cross_model_coverage"]))
    lines.extend(render_examples(report["examples"]))
    return "\n".join(lines).rstrip() + "\n"


def render_depth_row(row: dict[str, Any]) -> str:
    return (
        "| {model} | `{mode}` | {a1} | {a5} | {a10} | {a20} | {a50} | {median} | {misses} | {mrr} | {gold} |"
    ).format(
        model=escape_cell(str(row["model"])),
        mode=row["mode"],
        a1=format_metric(row["any_gold@1"]),
        a5=format_metric(row["any_gold@5"]),
        a10=format_metric(row["any_gold@10"]),
        a20=format_metric(row["any_gold@20"]),
        a50=format_metric(row["any_gold@50"]),
        median=row["median_first_gold_rank_hit_only"] if row["median_first_gold_rank_hit_only"] is not None else "",
        misses=row["missing_first_gold_rank"],
        mrr=format_metric(row["mean_mrr"]),
        gold=format_metric(row["mean_gold_coverage@8k"]),
    )


def render_coverage(coverage: dict[str, Any]) -> list[str]:
    lines = ["", "## Cross-Model Coverage", ""]
    lines.extend(
        [
            "| Task | Samples | Any Model Hit@20 | All Models Miss@20 | Hit Model Count Distribution |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for task, row in sorted(coverage.items(), key=lambda item: task_sort_key(item[0])):
        distribution = ", ".join(f"{count} models: {value}" for count, value in row["hit_model_count_distribution"].items())
        lines.append(
            f"| {task} | {row['samples']} | {row['any_model_hit@20']} | {row['all_models_miss@20']} | {escape_cell(distribution)} |"
        )
    return lines


def render_examples(examples: dict[str, Any]) -> list[str]:
    lines = ["", "## Representative Error Slices", ""]
    for title, rows in examples.get("all_model_misses@20_by_task", {}).items():
        lines.extend([f"### All Models Miss@20: {title}", ""])
        lines.extend(render_example_table(rows))
        lines.append("")
    named_sections = [
        ("Trace2Code RepoMap Hits, Embeddings Miss@20", examples.get("trace2code_repomap_hits_embeddings_miss@20", [])),
        ("Trace2Code Embedding Hits, RepoMap Miss@20", examples.get("trace2code_embedding_hits_repomap_miss@20", [])),
        ("Comment2Context Given-File Top1 Traps", examples.get("comment2context_given_file_top1_gold_miss@20", [])),
    ]
    for title, rows in named_sections:
        lines.extend([f"### {title}", ""])
        lines.extend(render_example_table(rows))
        lines.append("")
    return lines


def render_example_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No examples found."]
    lines = [
        "| Sample | Repo | Gold | Focus Model Results | Query Excerpt |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        model_text = "<br>".join(
            f"{escape_cell(label)}: rank={result.get('first_gold_rank')}, R@20={format_metric(result.get('Recall@20') or 0.0)}, top={escape_cell(', '.join(result.get('top_files') or []))}"
            for label, result in row.get("model_results", {}).items()
        )
        lines.append(
            "| {sample} | {repo} | {gold} | {models} | {query} |".format(
                sample=row["sample_id"],
                repo=escape_cell(row["repo"]),
                gold=escape_cell("<br>".join(row.get("gold_files") or [])),
                models=model_text,
                query=escape_cell(row.get("query_excerpt") or ""),
            )
        )
    return lines


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def one_line(value: str) -> str:
    return " ".join(str(value).split())


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")
