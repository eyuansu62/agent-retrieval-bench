from __future__ import annotations

import json
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .baseline import average_metrics, given_files, query_text_for_eval, target_gold_files
from .curate import filter_samples, load_keep_ids
from .diagnostics import file_ranks, paths_for_pair, query_gold_hints
from .io import ensure_parent, read_jsonl, truncate_text, utc_now, write_json

METRIC_KEYS = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")
V1_TARGET_COUNTS = {"code2test": 150, "comment2context": 150, "trace2code": 50}
SEED_AUDIT_FIELDS = ("sample_id", "task_type", "repo", "query_excerpt", "gold_files", "verdict", "reason", "keep", "notes")
DROP_AUDIT_VERDICTS = {"ambiguous", "duplicate", "leaked", "noisy", "too_easy"}
KEEP_AUDIT_VERDICTS = {"valid"}
GENERIC_MODULE_TOKENS = {
    "api",
    "app",
    "apps",
    "base",
    "bench",
    "benchmark",
    "benchmarks",
    "build",
    "common",
    "config",
    "core",
    "doc",
    "docs",
    "example",
    "examples",
    "file",
    "files",
    "helper",
    "helpers",
    "impl",
    "index",
    "init",
    "integration",
    "internal",
    "java",
    "js",
    "json",
    "lib",
    "main",
    "mod",
    "module",
    "package",
    "packages",
    "py",
    "repo",
    "rs",
    "src",
    "spec",
    "suite",
    "test",
    "tests",
    "testing",
    "ts",
    "tsx",
    "unit",
    "util",
    "utils",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def diagnose_hardness(
    sample_paths: Iterable[Path],
    corpus_manifest_path: Path,
    details_path: Path,
    out_dir: Path,
    pool_out_path: Path | None = None,
    keep_list: Path | None = None,
    tasks: Iterable[str] | None = None,
    hard_recall20_threshold: float = 1.0,
    hard_mrr_threshold: float = 0.25,
) -> dict[str, Any]:
    sample_paths = list(sample_paths)
    allowed_tasks = set(tasks or [])
    keep_ids = load_keep_ids(keep_list, tasks=allowed_tasks)
    samples = list(
        dedupe_samples(
            filter_samples(
                (
                    {**sample, "_source_path": str(path)}
                    for path in sample_paths
                    for sample in read_jsonl(path)
                    if not allowed_tasks or str(sample.get("task_type", "")) in allowed_tasks
                ),
                keep_ids,
            )
        )
    )
    details_by_id = {str(record.get("sample_id")): record for record in read_jsonl(details_path)}
    corpus_by_pair = {
        (record.get("repo"), record.get("base_commit")): Path(str(record.get("chunks_path", "")))
        for record in read_jsonl(corpus_manifest_path)
        if record.get("status") == "ok"
    }
    task_counts = Counter(str(sample.get("task_type", "")) for sample in samples)
    path_cache: dict[tuple[str, str], set[str]] = {}

    rows: list[dict[str, Any]] = []
    for sample in samples:
        detail = details_by_id.get(str(sample.get("id")), {})
        row = hardness_row(
            sample=sample,
            detail=detail,
            corpus_by_pair=corpus_by_pair,
            path_cache=path_cache,
            task_counts=task_counts,
            hard_recall20_threshold=hard_recall20_threshold,
            hard_mrr_threshold=hard_mrr_threshold,
        )
        rows.append(row)

    summary = summarize_hardness(
        rows=rows,
        sample_paths=list(sample_paths),
        corpus_manifest_path=corpus_manifest_path,
        details_path=details_path,
        keep_list=keep_list,
        hard_recall20_threshold=hard_recall20_threshold,
        hard_mrr_threshold=hard_mrr_threshold,
    )
    pool = sorted(rows, key=candidate_sort_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "hardness_report.json"
    report_path = out_dir / "hardness_report.md"
    samples_out_path = out_dir / "hardness_samples.jsonl"
    pool_path = pool_out_path or (out_dir / "candidate_keep_pool.jsonl")

    write_json(summary_path, summary)
    write_jsonl(samples_out_path, rows)
    write_jsonl(pool_path, pool)
    report_path.write_text(render_hardness_markdown(summary, pool), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "report": str(report_path),
        "sample_diagnostics": str(samples_out_path),
        "candidate_keep_pool": str(pool_path),
        "samples": summary["samples"],
        "hard_curated": summary["splits"].get("hard_curated", {"samples": 0})["samples"],
        "all_curated": summary["splits"].get("all_curated", {"samples": 0})["samples"],
        "tasks": summary["distribution"]["by_task"],
    }


def dedupe_samples(samples: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        yield sample


def hardness_row(
    sample: dict[str, Any],
    detail: dict[str, Any],
    corpus_by_pair: dict[tuple[str, str], Path],
    path_cache: dict[tuple[str, str], set[str]],
    task_counts: Counter[str],
    hard_recall20_threshold: float = 1.0,
    hard_mrr_threshold: float = 0.25,
) -> dict[str, Any]:
    sample_id = str(sample.get("id", ""))
    repo = str(sample.get("repo", ""))
    base_commit = str(sample.get("base_commit", ""))
    task_type = str(sample.get("task_type", ""))
    gold_files = target_gold_files(sample)
    known_given_files = given_files(sample)
    query_files = explicit_query_files(sample)
    reference_files = dedupe(known_given_files + query_files)
    query_text = query_text_for_eval(sample)
    corpus_paths = paths_for_pair(repo, base_commit, corpus_by_pair, path_cache)
    missing_gold = [path for path in gold_files if path not in corpus_paths]
    metrics = metrics_for_detail(detail)
    hint_details = hard_query_hints(query_text, gold_files, reference_files)
    top_files = list(detail.get("top_files") or [])[:20]
    lexical_bucket = lexical_rank_bucket(metrics, bool(detail), hard_mrr_threshold)
    label = hardness_label(
        metrics=metrics,
        has_detail=bool(detail),
        missing_gold=missing_gold,
        direct_path_hint=hint_details["direct_path_hint"],
        basename_hint=hint_details["basename_hint"],
        hard_recall20_threshold=hard_recall20_threshold,
        hard_mrr_threshold=hard_mrr_threshold,
    )
    split = recommended_split(label, missing_gold)
    score = hardness_score(
        metrics=metrics,
        label=label,
        hint_details=hint_details,
        same_directory_gold=bool(same_directory_gold(gold_files, reference_files)),
        gold_count=len(gold_files),
        task_balance_weight=task_balance_weight(task_type, task_counts),
    )
    same_directory = same_directory_gold(gold_files, reference_files)
    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "repo": repo,
        "base_commit": base_commit,
        "source_path": sample.get("_source_path", ""),
        "pr_url": ((sample.get("metadata") or {}).get("pr_url") or ""),
        "query_excerpt": truncate_text(query_text, 2000),
        "given_files": known_given_files,
        "query_files": query_files,
        "gold_files": gold_files,
        "gold_count": len(gold_files),
        "gold_in_corpus": not missing_gold,
        "missing_gold_files": missing_gold,
        "direct_path_hint": hint_details["direct_path_hint"],
        "basename_hint": hint_details["basename_hint"],
        "module_overlap": hint_details["module_overlap"],
        "same_directory_gold": bool(same_directory),
        "same_directory_gold_files": same_directory,
        "lexical_rank_bucket": lexical_bucket,
        "task_balance_weight": task_balance_weight(task_type, task_counts),
        "hardness_label": label,
        "hardness_score": round(score, 6),
        "recommended_split": split,
        "query_hint_details": hint_details,
        "given_file_ranks": file_ranks(known_given_files, top_files),
        "context_gold_ranks": detail.get("gold_ranks") or file_ranks(gold_files, top_files),
        "metrics": metrics,
        "top_files": top_files,
    }


def hard_query_hints(query_text: str, gold_files: list[str], reference_files: list[str] | None = None) -> dict[str, Any]:
    direct_hints = query_gold_hints(query_text, gold_files)
    query_tokens = set(tokens_for_text(query_text))
    reference_tokens = tokens_for_paths(reference_files or [])
    module_hits: dict[str, list[str]] = {}
    for path in gold_files:
        tokens = sorted(tokens_for_paths([path]) - reference_tokens)
        hits = [token for token in tokens if token in query_tokens]
        if hits:
            module_hits[path] = hits
    return {
        "direct_path_hint": bool(direct_hints["has_gold_path_hint"]),
        "basename_hint": bool(direct_hints["has_gold_basename_hint"]),
        "module_overlap": bool(module_hits),
        "gold_path_hits": direct_hints["gold_path_hits"],
        "gold_basename_hits": direct_hints["gold_basename_hits"],
        "module_token_hits": module_hits,
        "ignored_reference_tokens": sorted(reference_tokens),
    }


def lexical_rank_bucket(metrics: dict[str, float], has_detail: bool = True, hard_mrr_threshold: float = 0.25) -> str:
    if not has_detail:
        return "missing_detail"
    recall5 = metrics.get("Recall@5", 0.0)
    recall20 = metrics.get("Recall@20", 0.0)
    mrr = metrics.get("MRR", 0.0)
    if recall20 <= 0.0:
        return "miss@20"
    if recall20 < 1.0:
        return "partial@20"
    if mrr < hard_mrr_threshold:
        return "deep@20"
    if recall5 >= 1.0:
        return "top5"
    return "top20"


def hardness_label(
    metrics: dict[str, float],
    has_detail: bool,
    missing_gold: list[str],
    direct_path_hint: bool,
    basename_hint: bool,
    hard_recall20_threshold: float,
    hard_mrr_threshold: float,
) -> str:
    if missing_gold:
        return "invalid_missing_gold"
    if not has_detail:
        return "unknown_lexical"
    if direct_path_hint or basename_hint:
        return "too_easy_hint"
    if metrics.get("Recall@20", 0.0) < hard_recall20_threshold or metrics.get("MRR", 0.0) < hard_mrr_threshold:
        return "hard"
    if metrics.get("Recall@5", 0.0) >= 1.0:
        return "too_easy_lexical"
    return "medium"


def recommended_split(label: str, missing_gold: list[str]) -> str:
    if missing_gold or label == "invalid_missing_gold":
        return "drop"
    if label == "hard":
        return "hard_curated"
    return "all_curated"


def hardness_score(
    metrics: dict[str, float],
    label: str,
    hint_details: dict[str, Any],
    same_directory_gold: bool,
    gold_count: int,
    task_balance_weight: float,
) -> float:
    score = (1.0 - metrics.get("Recall@20", 0.0)) + (0.25 - min(metrics.get("MRR", 0.0), 0.25))
    if label == "hard":
        score += 1.0
    if hint_details["direct_path_hint"] or hint_details["basename_hint"]:
        score -= 1.0
    if hint_details["module_overlap"]:
        score -= 0.25
    if same_directory_gold:
        score -= 0.15
    if gold_count > 1:
        score += min(0.2, 0.05 * (gold_count - 1))
    return score * max(task_balance_weight, 0.1)


def task_balance_weight(task_type: str, task_counts: Counter[str]) -> float:
    target = V1_TARGET_COUNTS.get(task_type)
    if not target:
        return 1.0
    current = max(1, task_counts.get(task_type, 0))
    return round(target / current, 6)


def same_directory_gold(gold_files: list[str], reference_files: list[str]) -> list[str]:
    if not reference_files:
        return []
    reference_dirs = {str(PurePosixPath(path).parent) for path in reference_files if path}
    return [path for path in gold_files if str(PurePosixPath(path).parent) in reference_dirs]


def explicit_query_files(sample: dict[str, Any]) -> list[str]:
    query = sample.get("query") or {}
    values: list[str] = []
    for key in ("changed_file", "given_file", "path", "file"):
        value = query.get(key)
        if isinstance(value, str) and "/" in value:
            values.append(value)
    return dedupe(values)


def tokens_for_paths(paths: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for path in paths:
        normalized = path.replace("\\", "/")
        pure_path = PurePosixPath(normalized)
        parts = list(pure_path.parts)
        if pure_path.suffix:
            parts.append(pure_path.stem)
        for part in parts:
            tokens.update(tokens_for_text(part))
    return {token for token in tokens if token not in GENERIC_MODULE_TOKENS and len(token) >= 3}


def tokens_for_text(text: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in TOKEN_RE.findall(spaced)]


def summarize_hardness(
    rows: list[dict[str, Any]],
    sample_paths: list[Path],
    corpus_manifest_path: Path,
    details_path: Path,
    keep_list: Path | None,
    hard_recall20_threshold: float,
    hard_mrr_threshold: float,
) -> dict[str, Any]:
    by_task = Counter(row["task_type"] for row in rows)
    by_repo = Counter(row["repo"] for row in rows)
    by_label = Counter(row["hardness_label"] for row in rows)
    by_split = Counter(row["recommended_split"] for row in rows)
    hint_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    metrics_by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    metrics_by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    hard_by_task = Counter()
    for row in rows:
        task_type = row["task_type"]
        metrics_by_task[task_type].append(row["metrics"])
        metrics_by_task["overall"].append(row["metrics"])
        if row["recommended_split"] == "drop":
            metrics_by_split["drop"].append(row["metrics"])
        else:
            metrics_by_split["all_curated"].append(row["metrics"])
        if row["recommended_split"] == "hard_curated":
            metrics_by_split["hard_curated"].append(row["metrics"])
            hard_by_task[task_type] += 1
        if row["direct_path_hint"]:
            hint_by_task[task_type]["direct_path_hint"] += 1
        if row["basename_hint"]:
            hint_by_task[task_type]["basename_hint"] += 1
        if row["module_overlap"]:
            hint_by_task[task_type]["module_overlap"] += 1
        if row["same_directory_gold"]:
            hint_by_task[task_type]["same_directory_gold"] += 1

    return {
        "generated_at": utc_now(),
        "inputs": {
            "samples": [str(path) for path in sample_paths],
            "corpus_manifest": str(corpus_manifest_path),
            "baseline_details": str(details_path),
            "keep_list": str(keep_list) if keep_list else None,
        },
        "thresholds": {
            "hard_recall20_threshold": hard_recall20_threshold,
            "hard_mrr_threshold": hard_mrr_threshold,
        },
        "samples": len(rows),
        "distribution": {
            "by_task": dict(sorted(by_task.items())),
            "by_repo": dict(sorted(by_repo.items())),
            "by_label": dict(sorted(by_label.items())),
            "by_split": dict(sorted(by_split.items())),
        },
        "splits": split_metrics(metrics_by_split),
        "metrics_by_task": {task: average_metrics(metrics) for task, metrics in sorted(metrics_by_task.items())},
        "hint_counts": {
            "direct_path_hint": sum(1 for row in rows if row["direct_path_hint"]),
            "basename_hint": sum(1 for row in rows if row["basename_hint"]),
            "module_overlap": sum(1 for row in rows if row["module_overlap"]),
            "same_directory_gold": sum(1 for row in rows if row["same_directory_gold"]),
            "by_task": {task: dict(counts) for task, counts in sorted(hint_by_task.items())},
        },
        "gold_corpus": {
            "samples_with_all_gold": sum(1 for row in rows if row["gold_in_corpus"]),
            "samples_with_missing_gold": sum(1 for row in rows if not row["gold_in_corpus"]),
        },
        "v1_targets": {
            task: {
                "target": target,
                "all_curated_current": by_task.get(task, 0),
                "hard_current": hard_by_task.get(task, 0),
                "hard_gap": max(0, target - hard_by_task.get(task, 0)),
            }
            for task, target in V1_TARGET_COUNTS.items()
        },
        "candidate_pool": {
            "hard_candidates": sum(1 for row in rows if row["recommended_split"] == "hard_curated"),
            "all_candidates": sum(1 for row in rows if row["recommended_split"] in {"all_curated", "hard_curated"}),
            "drops": sum(1 for row in rows if row["recommended_split"] == "drop"),
        },
        "acceptance_gates": acceptance_gates(rows, metrics_by_split),
    }


def acceptance_gates(rows: list[dict[str, Any]], metrics_by_split: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    hard_metrics = average_metrics(metrics_by_split.get("hard_curated", []))
    all_metrics = average_metrics(metrics_by_split.get("all_curated", []))
    hard_valid_count = sum(1 for row in rows if row["recommended_split"] == "hard_curated")
    return {
        "hard_valid_count_ge_50": hard_valid_count >= 50,
        "lexical_overall_recall20_le_0_65": all_metrics["Recall@20"] <= 0.65,
        "lexical_overall_mrr_le_0_25": all_metrics["MRR"] <= 0.25,
        "hard_curated_metrics": hard_metrics,
        "all_curated_metrics": all_metrics,
    }


def split_metrics(metrics_by_split: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, Any]]:
    splits = {split: list(metrics) for split, metrics in metrics_by_split.items()}
    for split in ("all_curated", "hard_curated", "drop"):
        splits.setdefault(split, [])
    return {
        split: {"samples": len(metrics), "metrics": average_metrics(metrics)}
        for split, metrics in sorted(splits.items())
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    split_rank = {"hard_curated": 0, "all_curated": 1, "drop": 2}.get(row["recommended_split"], 3)
    hint_rank = int(row["direct_path_hint"]) + int(row["basename_hint"]) + int(row["module_overlap"])
    return (
        split_rank,
        hint_rank,
        int(row["same_directory_gold"]),
        -float(row["hardness_score"]),
        row["task_type"],
        row["repo"],
        row["sample_id"],
    )


def render_hardness_markdown(summary: dict[str, Any], pool: list[dict[str, Any]]) -> str:
    lines = [
        "# V1 Hardness Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Samples analyzed: `{summary['samples']}`.",
        f"- Hard candidates: `{summary['candidate_pool']['hard_candidates']}`.",
        f"- All curated candidates: `{summary['candidate_pool']['all_candidates']}`.",
        f"- Dropped candidates: `{summary['candidate_pool']['drops']}`.",
        "",
        "## V1 Target Gaps",
        "",
        "| Task | Target | Current | Hard Current | Hard Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, item in summary["v1_targets"].items():
        lines.append(
            f"| `{task}` | {item['target']} | {item['all_curated_current']} | {item['hard_current']} | {item['hard_gap']} |"
        )
    lines.extend(
        [
            "",
            "## Split Metrics",
            "",
            "| Split | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, item in sorted(summary["splits"].items()):
        metrics = item["metrics"]
        lines.append(
            f"| `{split}` | {item['samples']} | {metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | "
            f"{metrics['Recall@20']:.4f} | {metrics['MRR']:.4f} | {metrics['gold_coverage@8k']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Task Metrics",
            "",
            "| Task | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, metrics in summary["metrics_by_task"].items():
        lines.append(
            f"| `{task}` | {metrics['samples']} | {metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | "
            f"{metrics['Recall@20']:.4f} | {metrics['MRR']:.4f} | {metrics['gold_coverage@8k']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Hint Checks",
            "",
            f"- Direct gold path hints: `{summary['hint_counts']['direct_path_hint']}`.",
            f"- Gold basename hints: `{summary['hint_counts']['basename_hint']}`.",
            f"- Gold module token overlap: `{summary['hint_counts']['module_overlap']}`.",
            f"- Same-directory gold: `{summary['hint_counts']['same_directory_gold']}`.",
            "",
            "| Task | Direct Path | Basename | Module Token | Same Directory |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, counts in summary["hint_counts"]["by_task"].items():
        lines.append(
            f"| `{task}` | {counts.get('direct_path_hint', 0)} | {counts.get('basename_hint', 0)} | "
            f"{counts.get('module_overlap', 0)} | {counts.get('same_directory_gold', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Gates",
            "",
        ]
    )
    for key, value in summary["acceptance_gates"].items():
        if key.endswith("_metrics"):
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Top Hard Candidates",
            "",
            "| Sample | Task | Repo | Score | Recall@20 | MRR | Hints | Gold |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    hard_rows = [row for row in pool if row["recommended_split"] == "hard_curated"]
    for row in hard_rows[:40]:
        hints = []
        if row["direct_path_hint"]:
            hints.append("path")
        if row["basename_hint"]:
            hints.append("basename")
        if row["module_overlap"]:
            hints.append("module")
        if row["same_directory_gold"]:
            hints.append("same_dir")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['sample_id']}`",
                    f"`{row['task_type']}`",
                    f"`{row['repo']}`",
                    f"{row['hardness_score']:.4f}",
                    f"{row['metrics']['Recall@20']:.4f}",
                    f"{row['metrics']['MRR']:.4f}",
                    ", ".join(hints) or "none",
                    markdown_list(row["gold_files"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `hardness_report.json` contains aggregate hint, split, task, and V1 target-gap statistics.",
            "- `hardness_samples.jsonl` contains one diagnostic row per sample.",
            "- `candidate_keep_pool.jsonl` sorts hard candidates first for manual V1 audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def metrics_for_detail(detail: dict[str, Any]) -> dict[str, float]:
    metrics = detail.get("metrics") or {}
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def markdown_list(values: list[str]) -> str:
    if not values:
        return ""
    escaped = [value.replace("|", "\\|") for value in values]
    return "<br>".join(f"`{value}`" for value in escaped)


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def filter_hard_pool(
    pool_path: Path,
    out_path: Path,
    summary_path: Path,
    audit_path: Path | None = None,
    audit_out_path: Path | None = None,
    audit_csv_path: Path | None = None,
    audit_limit: int = 120,
    min_score: float = 0.0,
    include_unaudited: bool = True,
) -> dict[str, Any]:
    pool = read_jsonl(pool_path)
    audit_by_id = load_seed_audit(audit_path)
    reviewed: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for row in pool:
        candidate = annotate_seed_candidate(row, audit_by_id.get(str(row.get("sample_id")), {}), min_score, include_unaudited)
        if candidate["seed_decision"] == "keep":
            reviewed.append(candidate)
        else:
            drops.append(candidate)

    selected, duplicate_drops = dedupe_seed_candidates(reviewed)
    selected.sort(key=seed_sort_key)
    for index, row in enumerate(selected, start=1):
        row["seed_rank"] = index
    drops.extend(duplicate_drops)
    drops.sort(key=lambda row: (row.get("task_type", ""), row.get("repo", ""), row.get("sample_id", "")))

    summary = summarize_seed_pool(pool, selected, drops, audit_path, min_score, include_unaudited)
    write_jsonl(out_path, selected)
    write_json(summary_path, summary)
    audit_outputs: dict[str, str] = {}
    audit_rows = seed_audit_rows(selected[: max(0, audit_limit)])
    if audit_out_path:
        write_jsonl(audit_out_path, audit_rows)
        audit_outputs["jsonl"] = str(audit_out_path)
    if audit_csv_path:
        write_seed_audit_csv(audit_csv_path, audit_rows)
        audit_outputs["csv"] = str(audit_csv_path)
    return {
        "input_pool": str(pool_path),
        "audit": str(audit_path) if audit_path else None,
        "seed_candidates": str(out_path),
        "summary": str(summary_path),
        "audit_outputs": audit_outputs,
        "input_rows": len(pool),
        "kept": len(selected),
        "dropped": len(drops),
        "audit_rows": len(audit_rows),
        "counts_by_task": summary["kept_by_task"],
        "drop_reasons": summary["drop_reasons"],
    }


def annotate_seed_candidate(
    row: dict[str, Any],
    audit: dict[str, Any],
    min_score: float,
    include_unaudited: bool,
) -> dict[str, Any]:
    candidate = dict(row)
    verdict = normalize_seed_verdict(audit.get("verdict"))
    keep_flag = audit_keep_flag(audit)
    reasons = automatic_drop_reasons(row)
    score = float(row.get("hardness_score") or 0.0)
    if verdict in KEEP_AUDIT_VERDICTS or keep_flag is True:
        decision = "keep"
        reasons = []
        source = "audit_keep"
    elif verdict in DROP_AUDIT_VERDICTS or keep_flag is False:
        decision = "drop"
        reasons = [f"audit_{verdict}" if verdict else "audit_drop"]
        source = "audit_drop"
    elif reasons:
        decision = "drop"
        source = "auto_drop"
    elif include_unaudited and score >= min_score:
        decision = "keep"
        source = "auto_keep"
    else:
        decision = "drop"
        reasons = ["below_min_score"]
        source = "auto_drop"

    candidate.update(
        {
            "seed_decision": decision,
            "seed_source": source,
            "filter_reasons": reasons,
            "audit_verdict": verdict or "",
            "audit_keep": keep_flag,
            "cluster_key": cluster_key(row),
        }
    )
    return candidate


def automatic_drop_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    query_excerpt = str(row.get("query_excerpt") or "")
    gold_count = int(row.get("gold_count") or len(row.get("gold_files") or []))
    if row.get("recommended_split") != "hard_curated":
        reasons.append("not_hard_curated")
    if not row.get("gold_in_corpus", True):
        reasons.append("missing_gold")
    if row.get("direct_path_hint") or row.get("basename_hint"):
        reasons.append("direct_hint")
    if has_generated_summary(query_excerpt):
        reasons.append("generated_summary")
    if has_pr_template_boilerplate(query_excerpt):
        reasons.append("pr_template")
    if has_generic_test_template(query_excerpt):
        reasons.append("generic_test_template")
    if row.get("same_directory_gold") and not has_behavior_signal(query_excerpt):
        reasons.append("weak_same_directory")
    if gold_count > 3:
        reasons.append("broad_gold")
    return dedupe(reasons)


def dedupe_seed_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster_key"]].append(row)

    selected: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for cluster_rows in by_cluster.values():
        ordered = sorted(cluster_rows, key=seed_sort_key)
        selected.append(ordered[0])
        for duplicate in ordered[1:]:
            duplicate = dict(duplicate)
            duplicate["seed_decision"] = "drop"
            duplicate["seed_source"] = "cluster_dedupe"
            duplicate["filter_reasons"] = dedupe(list(duplicate.get("filter_reasons") or []) + ["duplicate_cluster"])
            drops.append(duplicate)
    return selected, drops


def summarize_seed_pool(
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    audit_path: Path | None,
    min_score: float,
    include_unaudited: bool,
) -> dict[str, Any]:
    metrics = average_metrics([row["metrics"] for row in selected if row.get("metrics")])
    kept_by_task = Counter(row.get("task_type", "") for row in selected)
    drop_reasons = Counter(reason for row in drops for reason in row.get("filter_reasons", []))
    verdicts = Counter(row.get("audit_verdict", "") or "unaudited" for row in selected + drops)
    return {
        "generated_at": utc_now(),
        "inputs": {
            "pool_rows": len(pool),
            "audit": str(audit_path) if audit_path else None,
            "min_score": min_score,
            "include_unaudited": include_unaudited,
        },
        "kept": len(selected),
        "dropped": len(drops),
        "kept_by_task": dict(sorted(kept_by_task.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "audit_verdicts": dict(sorted(verdicts.items())),
        "metrics": metrics,
        "quality_gates": {
            "kept_ge_15": len(selected) >= 15,
            "mrr_le_0_15": metrics["MRR"] <= 0.15,
            "recall20_le_0_65": metrics["Recall@20"] <= 0.65,
        },
    }


def seed_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        output.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "task_type": str(row.get("task_type", "")),
                "repo": str(row.get("repo", "")),
                "query_excerpt": str(row.get("query_excerpt", "")),
                "gold_files": "; ".join(str(path) for path in row.get("gold_files") or []),
                "verdict": "",
                "reason": "",
                "keep": "",
                "notes": "",
            }
        )
    return output


def load_seed_audit(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        rows = read_jsonl(path)
    return {str(row.get("sample_id")): row for row in rows if row.get("sample_id")}


def normalize_seed_verdict(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized


def audit_keep_flag(audit: dict[str, Any]) -> bool | None:
    if not audit:
        return None
    value = audit.get("keep_for_v1_hard", audit.get("keep"))
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "keep"}:
        return True
    if normalized in {"0", "false", "no", "n", "drop"}:
        return False
    return None


def cluster_key(row: dict[str, Any]) -> str:
    gold_files = tuple(sorted(str(path) for path in row.get("gold_files") or []))
    pr_url = str(row.get("pr_url") or row.get("sample_id") or "")
    return json.dumps([row.get("repo", ""), pr_url, gold_files], ensure_ascii=False, sort_keys=True)


def seed_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source_rank = {"audit_keep": 0, "auto_keep": 1, "cluster_dedupe": 2, "auto_drop": 3, "audit_drop": 4}.get(
        str(row.get("seed_source", "")),
        5,
    )
    lexical_bucket_rank = {"miss@20": 0, "partial@20": 1, "deep@20": 2, "top20": 3, "top5": 4}.get(
        str(row.get("lexical_rank_bucket", "")),
        5,
    )
    return (
        source_rank,
        int(bool(row.get("module_overlap"))),
        int(bool(row.get("same_directory_gold"))),
        lexical_bucket_rank,
        -float(row.get("hardness_score") or 0.0),
        row.get("task_type", ""),
        row.get("repo", ""),
        row.get("sample_id", ""),
    )


def has_generated_summary(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "coderabbit",
            "summary by coderabbit",
            "auto-generated comment",
            "auto generated comment",
            "release notes by coderabbit",
        )
    )


def has_pr_template_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "what does this pr do?",
            "congratulations! you've made it this far",
            "please replace this with a description",
            "once merged, your pr is going to appear in the release notes",
            "open your pull request against",
            "your pull request should have no more than two commits",
        )
    )


def has_generic_test_template(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    template_markers = (
        "you should add/modify tests to cover your proposed code changes",
        "it should pass all tests in the available continuous integration systems",
    )
    if any(marker in lowered for marker in template_markers):
        return True
    return bool(re.fullmatch(r"(please\s+)?(also\s+)?add tests?\.?", lowered))


def has_behavior_signal(text: str) -> bool:
    lowered = text.lower()
    behavior_markers = (
        "behavior",
        "case",
        "config",
        "error",
        "exception",
        "fail",
        "flag",
        "invalid",
        "order",
        "panic",
        "performance",
        "regress",
        "return",
        "when ",
        "if ",
        "should ",
        "could ",
        "can we",
        "does ",
        "missing",
    )
    return any(marker in lowered for marker in behavior_markers)


def write_seed_audit_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SEED_AUDIT_FIELDS})
            count += 1
    return count
