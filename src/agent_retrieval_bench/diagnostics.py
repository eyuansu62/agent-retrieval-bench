from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .baseline import average_metrics, query_text_for_eval, target_gold_files
from .io import ensure_parent, read_jsonl, utc_now

METRIC_KEYS = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")


def diagnose_benchmark(
    samples_path: Path,
    corpus_manifest_path: Path,
    details_path: Path,
    out_dir: Path,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    allowed_tasks = set(tasks or [])
    samples = [
        sample
        for sample in read_jsonl(samples_path)
        if not allowed_tasks or str(sample.get("task_type", "")) in allowed_tasks
    ]
    details_by_id = {str(record.get("sample_id")): record for record in read_jsonl(details_path)}
    corpus_records = read_jsonl(corpus_manifest_path)
    corpus_by_pair = {
        (record.get("repo"), record.get("base_commit")): Path(str(record.get("chunks_path", "")))
        for record in corpus_records
        if record.get("status") == "ok"
    }
    path_cache: dict[tuple[str, str], set[str]] = {}

    diagnostics: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        detail = details_by_id.get(sample_id, {})
        repo = str(sample.get("repo", ""))
        base_commit = str(sample.get("base_commit", ""))
        task_type = str(sample.get("task_type", ""))
        gold_files = target_gold_files(sample)
        corpus_paths = paths_for_pair(repo, base_commit, corpus_by_pair, path_cache)
        missing_gold = [path for path in gold_files if path not in corpus_paths]
        hints = query_gold_hints(query_text_for_eval(sample), gold_files)
        metrics = metrics_for_detail(detail)
        bucket = bucket_sample(metrics, hints, missing_gold)
        diagnostics.append(
            {
                "sample_id": sample_id,
                "task_type": task_type,
                "repo": repo,
                "base_commit": base_commit,
                "gold_files": gold_files,
                "gold_in_corpus": not missing_gold,
                "missing_gold_files": missing_gold,
                "query_hints": hints,
                "metrics": metrics,
                "bucket": bucket,
                "recommendation": recommendation_for_sample(task_type, bucket, hints, missing_gold),
                "top_files": list(detail.get("top_files") or [])[:20],
                "pr_url": ((sample.get("metadata") or {}).get("pr_url") or ""),
            }
        )

    summary = summarize_diagnostics(diagnostics, samples_path, corpus_manifest_path, details_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "diagnostic_summary.json"
    samples_out_path = out_dir / "sample_diagnostics.jsonl"
    report_path = out_dir / "report.md"
    write_json(summary_path, summary)
    write_jsonl(samples_out_path, diagnostics)
    report_path.write_text(render_markdown_report(summary, diagnostics), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "sample_diagnostics": str(samples_out_path),
        "report": str(report_path),
        "samples": summary["samples"],
        "tasks": summary["distribution"]["by_task"],
        "buckets": summary["buckets"],
    }


def query_gold_hints(query_text: str, gold_files: list[str]) -> dict[str, Any]:
    lowered_query = query_text.lower()
    full_path_hits: list[str] = []
    basename_hits: list[str] = []
    for path in gold_files:
        lowered_path = path.lower()
        basename = Path(path).name.lower()
        if lowered_path and lowered_path in lowered_query:
            full_path_hits.append(path)
        elif basename and basename in lowered_query:
            basename_hits.append(path)
    return {
        "has_gold_path_hint": bool(full_path_hits),
        "has_gold_basename_hint": bool(basename_hits),
        "gold_path_hits": full_path_hits,
        "gold_basename_hits": basename_hits,
    }


def bucket_sample(metrics: dict[str, float], hints: dict[str, Any], missing_gold: list[str]) -> str:
    if missing_gold:
        return "invalid_missing_gold"
    if hints.get("has_gold_path_hint") or hints.get("has_gold_basename_hint"):
        return "too_easy_direct_hint"
    if metrics.get("Recall@5", 0.0) >= 1.0:
        return "easy_lexical"
    if metrics.get("Recall@20", 0.0) <= 0.0:
        return "hard_lexical_miss"
    if metrics.get("Recall@20", 0.0) < 1.0:
        return "partial_lexical"
    return "medium_lexical"


def paths_for_pair(
    repo: str,
    base_commit: str,
    corpus_by_pair: dict[tuple[str, str], Path],
    path_cache: dict[tuple[str, str], set[str]],
) -> set[str]:
    key = (repo, base_commit)
    if key in path_cache:
        return path_cache[key]
    chunks_path = corpus_by_pair.get(key)
    if not chunks_path or not chunks_path.exists():
        path_cache[key] = set()
        return path_cache[key]
    paths = {str(chunk.get("path", "")) for chunk in read_jsonl(chunks_path) if chunk.get("path")}
    path_cache[key] = paths
    return paths


def summarize_diagnostics(
    diagnostics: list[dict[str, Any]],
    samples_path: Path,
    corpus_manifest_path: Path,
    details_path: Path,
) -> dict[str, Any]:
    by_task = Counter(item["task_type"] for item in diagnostics)
    by_repo = Counter(item["repo"] for item in diagnostics)
    by_task_repo: dict[str, Counter[str]] = defaultdict(Counter)
    metrics_by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    hint_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    missing_by_task: dict[str, int] = defaultdict(int)
    for item in diagnostics:
        task_type = item["task_type"]
        by_task_repo[task_type][item["repo"]] += 1
        metrics_by_task[task_type].append(item["metrics"])
        if item["query_hints"]["has_gold_path_hint"]:
            hint_by_task[task_type]["gold_path_hint"] += 1
        if item["query_hints"]["has_gold_basename_hint"]:
            hint_by_task[task_type]["gold_basename_hint"] += 1
        if not item["gold_in_corpus"]:
            missing_by_task[task_type] += 1
    metrics_by_task["overall"] = [item["metrics"] for item in diagnostics]
    failures = failure_samples(diagnostics)
    return {
        "generated_at": utc_now(),
        "inputs": {
            "samples": str(samples_path),
            "corpus_manifest": str(corpus_manifest_path),
            "baseline_details": str(details_path),
        },
        "samples": len(diagnostics),
        "distribution": {
            "by_task": dict(sorted(by_task.items())),
            "by_repo": dict(sorted(by_repo.items())),
            "by_task_repo": {task: dict(sorted(repos.items())) for task, repos in sorted(by_task_repo.items())},
        },
        "metrics_by_task": {
            task: average_metrics(rows) for task, rows in sorted(metrics_by_task.items())
        },
        "gold_corpus": {
            "samples_with_all_gold": sum(1 for item in diagnostics if item["gold_in_corpus"]),
            "samples_with_missing_gold": sum(1 for item in diagnostics if not item["gold_in_corpus"]),
            "missing_by_task": dict(sorted(missing_by_task.items())),
        },
        "query_hints": {
            "samples_with_gold_path_hint": sum(1 for item in diagnostics if item["query_hints"]["has_gold_path_hint"]),
            "samples_with_gold_basename_hint": sum(
                1 for item in diagnostics if item["query_hints"]["has_gold_basename_hint"]
            ),
            "by_task": {task: dict(counts) for task, counts in sorted(hint_by_task.items())},
        },
        "buckets": dict(sorted(Counter(item["bucket"] for item in diagnostics).items())),
        "recommendations": dict(sorted(Counter(item["recommendation"] for item in diagnostics).items())),
        "failure_samples": failures,
        "conclusions": conclusions(by_task, metrics_by_task, failures),
    }


def failure_samples(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        item
        for item in diagnostics
        if item["metrics"].get("Recall@20", 0.0) < 1.0 or not item["gold_in_corpus"]
    ]
    failures.sort(
        key=lambda item: (
            item["task_type"],
            item["metrics"].get("Recall@20", 0.0),
            item["metrics"].get("MRR", 0.0),
            item["repo"],
            item["sample_id"],
        )
    )
    return [
        {
            "sample_id": item["sample_id"],
            "task_type": item["task_type"],
            "repo": item["repo"],
            "Recall@20": item["metrics"].get("Recall@20", 0.0),
            "MRR": item["metrics"].get("MRR", 0.0),
            "gold_files": item["gold_files"],
            "top_files": item["top_files"][:5],
            "bucket": item["bucket"],
            "recommendation": item["recommendation"],
            "missing_gold_files": item["missing_gold_files"],
        }
        for item in failures
    ]


def recommendation_for_sample(
    task_type: str,
    bucket: str,
    hints: dict[str, Any],
    missing_gold: list[str],
) -> str:
    if missing_gold:
        return "drop_missing_gold"
    if task_type == "trace2code":
        return "smoke_only"
    if hints.get("has_gold_path_hint") or hints.get("has_gold_basename_hint"):
        return "downweight_direct_hint"
    if task_type == "code2test" and bucket in {"hard_lexical_miss", "partial_lexical"}:
        return "keep_hard_code2test"
    return "keep"


def conclusions(
    by_task: Counter[str],
    metrics_by_task: dict[str, list[dict[str, float]]],
    failures: list[dict[str, Any]],
) -> list[str]:
    averaged = {task: average_metrics(rows) for task, rows in metrics_by_task.items()}
    output = [
        "V0.1 can be used as a closed-loop smoke benchmark, but not yet as a final model-ranking benchmark.",
    ]
    code2test = averaged.get("code2test")
    if code2test:
        output.append(
            "Keep code2test as the main hard slice for V0.2; lexical Recall@20 is low enough to expose retrieval weakness."
        )
    comment2context = averaged.get("comment2context")
    if comment2context and comment2context.get("Recall@20", 0.0) >= 0.95:
        output.append(
            "Keep comment2context, but downweight or separately report direct-hint samples because lexical retrieval is near ceiling."
        )
    if by_task.get("trace2code", 0) < 10:
        output.append("Treat trace2code as smoke-only until it has enough validated samples for stable task-level metrics.")
    if failures:
        output.append("Use the failure table to inspect code2test weak labels before scaling V0.2.")
    if not by_task.get("testlog2code"):
        output.append("Keep testlog2code excluded until the audited valid rate reaches the planned threshold.")
    return output


def metrics_for_detail(detail: dict[str, Any]) -> dict[str, float]:
    metrics = detail.get("metrics") or {}
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS}


def render_markdown_report(summary: dict[str, Any], diagnostics: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark V0.1 Diagnostic Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Executive Conclusions",
    ]
    lines.extend(f"- {item}" for item in summary["conclusions"])
    lines.extend(
        [
            "",
            "## Dataset Distribution",
            "",
            "| Task | Samples |",
            "| --- | ---: |",
        ]
    )
    for task, count in summary["distribution"]["by_task"].items():
        lines.append(f"| `{task}` | {count} |")
    lines.extend(["", "| Repo | Samples |", "| --- | ---: |"])
    for repo, count in summary["distribution"]["by_repo"].items():
        lines.append(f"| `{repo}` | {count} |")

    lines.extend(["", "## Baseline Metrics", "", "| Task | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for task, metrics in summary["metrics_by_task"].items():
        lines.append(
            f"| `{task}` | {metrics['samples']} | {metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | "
            f"{metrics['Recall@20']:.4f} | {metrics['MRR']:.4f} | {metrics['gold_coverage@8k']:.4f} |"
        )

    gold = summary["gold_corpus"]
    hints = summary["query_hints"]
    lines.extend(
        [
            "",
            "## Corpus And Query Hint Checks",
            "",
            f"- Gold fully present in corpus: `{gold['samples_with_all_gold']}/{summary['samples']}`.",
            f"- Samples with missing gold files: `{gold['samples_with_missing_gold']}`.",
            f"- Samples with direct gold path hint in query: `{hints['samples_with_gold_path_hint']}`.",
            f"- Samples with direct gold basename hint in query: `{hints['samples_with_gold_basename_hint']}`.",
            "",
            "## Hard/Easy Buckets",
            "",
            "| Bucket | Samples |",
            "| --- | ---: |",
        ]
    )
    for bucket, count in summary["buckets"].items():
        lines.append(f"| `{bucket}` | {count} |")

    lines.extend(["", "## Sample Recommendations", "", "| Recommendation | Samples |", "| --- | ---: |"])
    for recommendation, count in summary["recommendations"].items():
        lines.append(f"| `{recommendation}` | {count} |")
    lines.extend(["", "| Task | Gold path hints | Gold basename hints |", "| --- | ---: | ---: |"])
    for task, counts in summary["query_hints"]["by_task"].items():
        lines.append(
            f"| `{task}` | {counts.get('gold_path_hint', 0)} | {counts.get('gold_basename_hint', 0)} |"
        )
    drop_ids = [item["sample_id"] for item in diagnostics if item["recommendation"] == "drop_missing_gold"]
    downweight_ids = [item["sample_id"] for item in diagnostics if item["recommendation"] == "downweight_direct_hint"]
    if drop_ids:
        lines.extend(["", f"- Drop candidates due to missing gold in corpus: {markdown_list_inline(drop_ids)}."])
    if downweight_ids:
        lines.extend(["", f"- Downweight candidates due to direct query hints: {markdown_list_inline(downweight_ids)}."])

    lines.extend(
        [
            "",
            "## Failure Samples",
            "",
            "| Sample | Task | Repo | Recall@20 | MRR | Bucket | Recommendation | Gold | Top 5 |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for item in summary["failure_samples"][:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['sample_id']}`",
                    f"`{item['task_type']}`",
                    f"`{item['repo']}`",
                    f"{item['Recall@20']:.4f}",
                    f"{item['MRR']:.4f}",
                    f"`{item['bucket']}`",
                    f"`{item['recommendation']}`",
                    markdown_list(item["gold_files"]),
                    markdown_list(item["top_files"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## V0.2 Decisions",
            "",
            "- Prioritize expanding and auditing `code2test`; it is the only V0.1 slice with enough hard lexical misses.",
            "- Keep `comment2context`, but report direct-hint and no-hint subsets separately to avoid ceiling-effect metrics.",
            "- Keep `trace2code` as a smoke slice until the validated count is large enough for stable metrics.",
            "- Do not add `testlog2code` to V0.2 until the cleaned audit valid rate reaches at least 50%.",
            "",
            "## Output Files",
            "",
            "- `diagnostic_summary.json` contains aggregate counts, metrics, buckets, and conclusions.",
            "- `sample_diagnostics.jsonl` contains one row per evaluated sample with hints, corpus coverage, bucket, and recommendation.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_list(values: list[str]) -> str:
    if not values:
        return ""
    return "<br>".join(f"`{escape_markdown(value)}`" for value in values)


def markdown_list_inline(values: list[str]) -> str:
    return ", ".join(f"`{escape_markdown(value)}`" for value in values)


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_json(path: Path, value: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
