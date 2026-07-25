from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def gold_file_paths(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    explicit = path_values(gold.get("files") or [])
    if explicit:
        return dedupe(explicit)
    task_type = sample.get("task_type")
    if task_type == "code2test":
        return dedupe(path_values(gold.get("related_tests") or []))
    if task_type == "comment2context":
        context = path_values(gold.get("must_context_files") or gold.get("context_files") or [])
        if context:
            return dedupe(context)
    root_cause = path_values(gold.get("root_cause_files") or [])
    return dedupe(root_cause or path_values(gold.get("related_tests") or []))


def path_values(values: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return paths


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def interval_line_count(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if start > 0 and end >= start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    return total + current_end - current_start + 1


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "min": min(data) if data else None,
        "median": percentile(data, 0.5),
        "p75": percentile(data, 0.75),
        "p90": percentile(data, 0.9),
        "p95": percentile(data, 0.95),
        "max": max(data) if data else None,
    }


def threshold_rates(values: Iterable[float], thresholds: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {str(threshold): 0.0 for threshold in thresholds}
    return {
        str(threshold): sum(value > threshold for value in data) / len(data)
        for threshold in thresholds
    }


def load_manifest_index(manifest_paths: Iterable[Path], repo_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for manifest_path in manifest_paths:
        for row in iter_jsonl(manifest_path):
            if row.get("status") != "ok":
                continue
            repo = str(row.get("repo") or "")
            base_commit = str(row.get("base_commit") or "")
            chunks_value = str(row.get("chunks_path") or "")
            if not repo or not base_commit or not chunks_value:
                continue
            chunks_path = Path(chunks_value)
            if not chunks_path.is_absolute():
                chunks_path = repo_root / chunks_path
            index[(repo, base_commit)] = chunks_path
    return index


def collect_file_lengths(
    samples: Iterable[dict[str, Any]],
    manifest_index: dict[tuple[str, str], Path],
) -> tuple[dict[tuple[str, str, str], int], list[dict[str, str]]]:
    needed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        needed[key].update(gold_file_paths(sample))

    lengths: dict[tuple[str, str, str], int] = {}
    missing: list[dict[str, str]] = []
    for (repo, base_commit), paths in needed.items():
        chunks_path = manifest_index.get((repo, base_commit))
        if chunks_path is None or not chunks_path.exists():
            for path in sorted(paths):
                missing.append({"repo": repo, "base_commit": base_commit, "path": path, "reason": "missing_manifest"})
            continue
        maxima = {path: 0 for path in paths}
        for chunk in iter_jsonl(chunks_path):
            path = str(chunk.get("path") or "")
            if path not in maxima:
                continue
            try:
                end_line = int(chunk.get("end_line") or 0)
            except (TypeError, ValueError):
                continue
            maxima[path] = max(maxima[path], end_line)
        for path, line_count in maxima.items():
            if line_count > 0:
                lengths[(repo, base_commit, path)] = line_count
            else:
                missing.append({"repo": repo, "base_commit": base_commit, "path": path, "reason": "missing_path"})
    return lengths, missing


def file_size_records(
    samples: Iterable[dict[str, Any]],
    lengths: dict[tuple[str, str, str], int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        repo = str(sample.get("repo") or "")
        base_commit = str(sample.get("base_commit") or "")
        for path in gold_file_paths(sample):
            line_count = lengths.get((repo, base_commit, path))
            if line_count is None:
                continue
            records.append(
                {
                    "sample_id": str(sample.get("id") or ""),
                    "task_type": str(sample.get("task_type") or "unknown"),
                    "repo": repo,
                    "base_commit": base_commit,
                    "path": path,
                    "file_lines": line_count,
                }
            )
    return records


def summarize_file_sizes(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["task_type"]].append(record)
        grouped["overall"].append(record)
    sample_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        sample_counts[str(sample.get("task_type") or "unknown")] += 1
        sample_counts["overall"] += 1
    result: dict[str, Any] = {}
    for task_type, rows in grouped.items():
        values = [row["file_lines"] for row in rows]
        result[task_type] = {
            "samples": sample_counts[task_type],
            "gold_file_occurrences": len(rows),
            "file_lines": distribution(values),
            "fraction_over_lines": threshold_rates(values, (500, 1000, 2000, 5000)),
        }
    return result


def summarize_span_evidence(
    samples: Iterable[dict[str, Any]],
    lengths: dict[tuple[str, str, str], int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_counts: dict[str, int] = defaultdict(int)
    samples_with_spans: dict[str, int] = defaultdict(int)
    for sample in samples:
        task_type = str(sample.get("task_type") or "unknown")
        sample_counts[task_type] += 1
        sample_counts["overall"] += 1
        intervals_by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for span in sample.get("gold_spans") or []:
            if not isinstance(span, dict):
                continue
            path = str(span.get("path") or "")
            try:
                start_line = int(span.get("start_line") or 0)
                end_line = int(span.get("end_line") or 0)
            except (TypeError, ValueError):
                continue
            if path and start_line > 0 and end_line >= start_line:
                intervals_by_path[path].append((start_line, end_line))
        if intervals_by_path:
            samples_with_spans[task_type] += 1
            samples_with_spans["overall"] += 1
        repo = str(sample.get("repo") or "")
        base_commit = str(sample.get("base_commit") or "")
        for path, intervals in intervals_by_path.items():
            file_lines = lengths.get((repo, base_commit, path))
            if not file_lines:
                continue
            evidence_lines = interval_line_count(intervals)
            row = {
                "sample_id": str(sample.get("id") or ""),
                "task_type": task_type,
                "path": path,
                "file_lines": file_lines,
                "evidence_lines": evidence_lines,
                "evidence_to_file_ratio": evidence_lines / file_lines,
            }
            grouped[task_type].append(row)
            grouped["overall"].append(row)

    result: dict[str, Any] = {}
    for task_type, rows in grouped.items():
        ratios = [row["evidence_to_file_ratio"] for row in rows]
        evidence_lines = [row["evidence_lines"] for row in rows]
        result[task_type] = {
            "samples": sample_counts[task_type],
            "samples_with_spans": samples_with_spans[task_type],
            "span_file_occurrences": len(rows),
            "evidence_lines": distribution(evidence_lines),
            "evidence_to_file_ratio": distribution(ratios),
            "fraction_ratio_at_most": {
                "0.10": sum(value <= 0.10 for value in ratios) / len(ratios) if ratios else 0.0,
                "0.25": sum(value <= 0.25 for value in ratios) / len(ratios) if ratios else 0.0,
            },
        }
    return result


def mean_metric(rows: list[dict[str, Any]], container: str, key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        metrics = row.get(container)
        if isinstance(metrics, dict) and isinstance(metrics.get(key), (int, float)):
            values.append(float(metrics[key]))
    return sum(values) / len(values) if values else None


def summarize_details(method_paths: dict[str, Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method, path in method_paths.items():
        rows = read_jsonl(path)
        predicted_lines = mean_metric(rows, "line_metrics", "line_predicted_count")
        span_capable = bool(predicted_lines and predicted_lines > 0)
        summaries.append(
            {
                "method": method,
                "samples": len(rows),
                "file_recall@20": mean_metric(rows, "metrics", "Recall@20"),
                "file_exposure@8k_chars": mean_metric(rows, "metrics", "gold_coverage@8k"),
                "span_capable": span_capable,
                "line_recall@8k_chars": mean_metric(rows, "line_metrics", "line_recall@8k") if span_capable else None,
                "line_precision@8k_chars": mean_metric(rows, "line_metrics", "line_precision@8k") if span_capable else None,
                "line_f1@8k_chars": mean_metric(rows, "line_metrics", "line_f1@8k") if span_capable else None,
                "mean_predicted_lines": predicted_lines if span_capable else None,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["line_f1@8k_chars"] is not None, row["line_f1@8k_chars"] or -1.0),
        reverse=True,
    )


def build_report(
    positive_samples: list[dict[str, Any]],
    core_samples: list[dict[str, Any]],
    manifest_paths: list[Path],
    method_paths: dict[str, Path],
    repo_root: Path,
) -> dict[str, Any]:
    manifest_index = load_manifest_index(manifest_paths, repo_root)
    lengths, missing = collect_file_lengths(positive_samples, manifest_index)
    records = file_size_records(positive_samples, lengths)
    return {
        "protocol": {
            "file_size_unit": "physical lines inferred as max end_line across released corpus chunks",
            "file_size_population": "sample-gold-file occurrences; repeated files at different samples/commits remain separate",
            "span_population": "V1.3 reviewed positive core with gold_spans",
            "span_sensitivity_budget": "legacy evaluator chunk packing by 8,000 characters; not canonical token BCY",
        },
        "positive_samples": len(positive_samples),
        "core_samples": len(core_samples),
        "manifest_snapshots": len(manifest_index),
        "missing_gold_file_lengths": missing,
        "file_sizes": summarize_file_sizes(records, positive_samples),
        "span_evidence": summarize_span_evidence(core_samples, lengths),
        "method_sensitivity": summarize_details(method_paths),
    }


def fmt_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_rate(value: float | int | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.1f}%"


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# File-versus-span granularity analysis",
        "",
        "File sizes are measured on sample-gold-file occurrences in the released base-commit corpus. "
        "The span sensitivity table uses the existing V1.3 chunk-ranking artifacts under their legacy "
        "8,000-character packing rule; it is diagnostic and is not numerically interchangeable with canonical token BCY.",
        "",
        "## Gold file size distribution",
        "",
        "| Task | Samples | Gold files | Median lines | P90 | P95 | Max | >500 | >1k | >2k | >5k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    task_order = ("code2test", "comment2context", "trace2code", "edit2ripple", "overall")
    for task_type in task_order:
        row = report["file_sizes"].get(task_type)
        if not row:
            continue
        dist = row["file_lines"]
        rates = row["fraction_over_lines"]
        lines.append(
            f"| {task_type} | {row['samples']} | {row['gold_file_occurrences']} | "
            f"{fmt_number(dist['median'], 0)} | {fmt_number(dist['p90'], 0)} | "
            f"{fmt_number(dist['p95'], 0)} | {fmt_number(dist['max'], 0)} | "
            f"{fmt_rate(rates['500'])} | {fmt_rate(rates['1000'])} | "
            f"{fmt_rate(rates['2000'])} | {fmt_rate(rates['5000'])} |"
        )

    lines.extend(
        [
            "",
            "## Span evidence density on the 287-sample core",
            "",
            "| Task | Samples with spans | Span-file pairs | Median evidence lines | P90 | Median evidence/file | <=10% of file | <=25% of file |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_type in ("code2test", "comment2context", "trace2code", "overall"):
        row = report["span_evidence"].get(task_type)
        if not row:
            continue
        evidence = row["evidence_lines"]
        ratios = row["evidence_to_file_ratio"]
        lines.append(
            f"| {task_type} | {row['samples_with_spans']}/{row['samples']} | {row['span_file_occurrences']} | "
            f"{fmt_number(evidence['median'], 0)} | {fmt_number(evidence['p90'], 0)} | "
            f"{fmt_rate(ratios['median'])} | {fmt_rate(row['fraction_ratio_at_most']['0.10'])} | "
            f"{fmt_rate(row['fraction_ratio_at_most']['0.25'])} |"
        )

    lines.extend(
        [
            "",
            "## File exposure versus line overlap",
            "",
            "| Method | n | File R@20 | File exposure@8k chars | Line recall | Line precision | Line F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["method_sensitivity"]:
        lines.append(
            f"| {row['method']} | {row['samples']} | {fmt_number(row['file_recall@20'], 4)} | "
            f"{fmt_number(row['file_exposure@8k_chars'], 4)} | {fmt_number(row['line_recall@8k_chars'], 4)} | "
            f"{fmt_number(row['line_precision@8k_chars'], 4)} | {fmt_number(row['line_f1@8k_chars'], 4)} |"
        )
    if report["missing_gold_file_lengths"]:
        lines.extend(
            [
                "",
                f"Warning: {len(report['missing_gold_file_lengths'])} gold file occurrences could not be resolved in the supplied corpus manifests.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_method_paths(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected METHOD=PATH, got {value!r}")
        method, path = value.split("=", 1)
        result[method] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Report file-level versus span-level benchmark granularity.")
    parser.add_argument("--samples", type=Path, action="append", required=True, help="Positive benchmark JSONL; repeatable.")
    parser.add_argument("--core-samples", type=Path, required=True, help="Span-annotated positive-core JSONL.")
    parser.add_argument("--corpus-manifest", type=Path, action="append", required=True, help="Corpus manifest JSONL; repeatable.")
    parser.add_argument("--details", action="append", default=[], metavar="METHOD=PATH", help="Evaluation details JSONL; repeatable.")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Root used to resolve manifest chunks_path entries.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    positive_samples = [row for path in args.samples for row in read_jsonl(path)]
    core_samples = read_jsonl(args.core_samples)
    report = build_report(
        positive_samples,
        core_samples,
        args.corpus_manifest,
        parse_method_paths(args.details),
        args.repo_root,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(report_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
