from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .baseline import (
    gold_spans,
    hard_negative_files,
    query_provenance,
    target_gold_files,
)
from .corpus import sample_paths_from_derived
from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .model_report import model_label, normalize_summary, task_sort_key
from .quality import validate_sample

SPAN_METRIC_KEYS = (
    "span_recall@8k",
    "span_precision@8k",
    "span_f0.5@8k",
    "line_overlap_f0.5",
    "gold_lines",
    "predicted_lines",
    "overlap_lines",
)
CONTEXT_REPORT_METRIC_KEYS = (
    "Precision@5",
    "Precision@10",
    "Precision@20",
    "F0.5@5",
    "F0.5@10",
    "F0.5@20",
    "irrelevant_files@20",
    "hard_negative_hits@20",
    "context_pollution_tokens@8k",
    "gold_token_ratio@8k",
)


def merge_manual_annotations(
    base_derived: Path,
    annotations_path: Path,
    out_dir: Path,
    report_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    samples = [row for path in sample_paths_from_derived(base_derived) for row in read_jsonl(path)]
    annotations = load_manual_annotations(annotations_path)
    sample_ids = {str(sample.get("id") or "") for sample in samples}
    unknown_ids = sorted(set(annotations) - sample_ids)
    if unknown_ids:
        raise ValueError(f"manual annotations reference unknown sample ids: {', '.join(unknown_ids)}")

    merged: list[dict[str, Any]] = []
    annotated_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        annotation = annotations.get(sample_id)
        if annotation:
            sample = apply_manual_annotation(sample, annotation)
            annotated_counts[str(sample.get("task_type") or "unknown")] += 1
        merged.append(sample)

    write_split_benchmark(out_dir, merged)
    manifest = write_v1_2_manifest(base_derived, annotations_path, out_dir, merged, annotated_counts)
    report = {
        "generated_at": utc_now(),
        "base_derived": str(base_derived),
        "annotations": str(annotations_path),
        "out_dir": str(out_dir),
        "samples": len(merged),
        "annotation_count": len(annotations),
        "annotated_counts_by_task": dict(sorted(annotated_counts.items())),
        "span_samples": sum(1 for sample in merged if gold_spans(sample)),
        "hard_negative_samples": sum(1 for sample in merged if hard_negative_files(sample)),
        "sample_ids_preserved": [str(sample.get("id") or "") for sample in merged] == [str(sample.get("id") or "") for sample in samples],
        "manifest": manifest,
    }
    if report_path:
        write_json(report_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_manual_annotation_markdown(report), encoding="utf-8")
    return report


def load_manual_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if not sample_id:
            raise ValueError(f"{path}:{line_number} missing sample_id")
        if sample_id in annotations:
            raise ValueError(f"{path}:{line_number} duplicates sample_id {sample_id}")
        annotations[sample_id] = row
    return annotations


def apply_manual_annotation(sample: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(sample))
    gold_files = target_gold_files(merged)
    if "gold_spans" in annotation:
        merged["gold_spans"] = normalize_annotation_spans(annotation.get("gold_spans"), gold_files, str(merged.get("id") or ""))
    if "hard_negative_files" in annotation:
        merged["hard_negative_files"] = normalize_annotation_hard_negatives(
            annotation.get("hard_negative_files"),
            gold_files,
            str(merged.get("id") or ""),
        )
    if annotation.get("query_provenance"):
        merged["query_provenance"] = str(annotation["query_provenance"])
    return merged


def normalize_annotation_spans(values: Any, gold_files: list[str], sample_id: str) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{sample_id}: gold_spans must be a list")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{sample_id}: gold_spans[{index}] must be an object")
        path = str(value.get("path") or "")
        if not path:
            raise ValueError(f"{sample_id}: gold_spans[{index}] missing path")
        if path not in gold_files:
            raise ValueError(f"{sample_id}: gold_spans[{index}] path is not in gold_files: {path}")
        try:
            start_line = int(value.get("start_line"))
            end_line = int(value.get("end_line"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id}: gold_spans[{index}] has non-integer line range") from exc
        if start_line <= 0 or end_line < start_line:
            raise ValueError(f"{sample_id}: gold_spans[{index}] has invalid line range: {start_line}-{end_line}")
        normalized.append(
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "reason": str(value.get("reason") or ""),
            }
        )
    return normalized


def normalize_annotation_hard_negatives(values: Any, gold_files: list[str], sample_id: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{sample_id}: hard_negative_files must be a list")
    normalized = []
    seen = set()
    for value in values:
        path = str(value or "")
        if not path or path in seen:
            continue
        normalized.append(path)
        seen.add(path)
    overlap = sorted(set(normalized) & set(gold_files))
    if overlap:
        raise ValueError(f"{sample_id}: hard_negative_files overlap gold_files: {', '.join(overlap)}")
    return normalized


def write_split_benchmark(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_task[str(sample.get("task_type") or "unknown")].append(sample)
    write_jsonl(out_dir / "samples.jsonl", samples)
    for task_type, rows in sorted(by_task.items()):
        write_jsonl(out_dir / f"{task_type}.jsonl", rows)


def write_v1_2_manifest(
    base_derived: Path,
    annotations_path: Path,
    out_dir: Path,
    samples: list[dict[str, Any]],
    annotated_counts: dict[str, int],
) -> dict[str, Any]:
    base_manifest = read_json(base_derived / "manifest.json", {})
    manifest: dict[str, Any] = dict(base_manifest) if isinstance(base_manifest, dict) else {}
    counts_by_task: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts_by_task[str(sample.get("task_type") or "unknown")] += 1
    manifest.update(
        {
            "generated_at": utc_now(),
            "version": "v1_2",
            "base_derived": str(base_derived),
            "manual_annotations": str(annotations_path),
            "total": len(samples),
            "counts_by_task": dict(sorted(counts_by_task.items())),
            "manual_annotation_count": sum(annotated_counts.values()),
            "manual_annotation_counts_by_task": dict(sorted(annotated_counts.items())),
            "span_samples": sum(1 for sample in samples if gold_spans(sample)),
            "hard_negative_samples": sum(1 for sample in samples if hard_negative_files(sample)),
            "outputs": {
                "samples": str(out_dir / "samples.jsonl"),
                "code2test": str(out_dir / "code2test.jsonl"),
                "comment2context": str(out_dir / "comment2context.jsonl"),
                "trace2code": str(out_dir / "trace2code.jsonl"),
            },
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def validate_v1_2_benchmark(
    derived: Path,
    corpus_manifest_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    samples = [row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)]
    required_paths = required_corpus_paths_for_v1_2(samples)
    corpus = load_corpus_file_bounds(corpus_manifest_path, required_paths) if corpus_manifest_path else {}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        errors = validate_sample(sample)
        gold_files = target_gold_files(sample)
        raw_spans = raw_gold_spans(sample)
        span_rows = gold_spans(sample)
        hard_negatives = hard_negative_files(sample)
        if raw_spans is not None and len(span_rows) != len(raw_spans):
            errors.append("gold_spans contains malformed span rows")
        if span_rows:
            errors.extend(validate_gold_spans(span_rows, gold_files, sample, corpus))
        if hard_negatives:
            errors.extend(validate_hard_negatives(hard_negatives, gold_files, sample, corpus))
        rows.append(
            {
                "sample_id": sample_id,
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "gold_files": gold_files,
                "gold_span_count": len(span_rows),
                "hard_negative_count": len(hard_negatives),
                "query_provenance": query_provenance(sample),
                "errors": errors,
            }
        )
    invalid_rows = [row for row in rows if row["errors"]]
    report = {
        "generated_at": utc_now(),
        "derived": str(derived),
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "samples": len(rows),
        "invalid": len(invalid_rows),
        "ready": not invalid_rows,
        "span_samples": sum(1 for row in rows if row["gold_span_count"]),
        "hard_negative_samples": sum(1 for row in rows if row["hard_negative_count"]),
        "query_provenance_samples": sum(1 for row in rows if row["query_provenance"]),
        "missing_query_provenance": [row["sample_id"] for row in rows if not row["query_provenance"]][:50],
        "rows": rows,
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_2_validation_markdown(report), encoding="utf-8")
    return report


def validate_gold_spans(
    spans: list[dict[str, Any]],
    gold_files: list[str],
    sample: dict[str, Any],
    corpus: dict[tuple[str, str], dict[str, int]],
) -> list[str]:
    errors: list[str] = []
    gold = set(gold_files)
    corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
    file_bounds = corpus.get(corpus_key, {})
    if corpus and corpus_key not in corpus:
        errors.append(f"sample repo/base_commit missing from corpus manifest: {corpus_key[0]} {corpus_key[1]}")
    for index, span in enumerate(spans):
        path = str(span.get("path") or "")
        try:
            start_line = int(span.get("start_line") or 0)
            end_line = int(span.get("end_line") or 0)
        except (TypeError, ValueError):
            errors.append(f"gold_spans[{index}] has non-integer line range")
            continue
        if path not in gold:
            errors.append(f"gold_spans[{index}] path is not in gold_files: {path}")
        if start_line <= 0 or end_line < start_line:
            errors.append(f"gold_spans[{index}] has invalid line range: {start_line}-{end_line}")
        max_line = file_bounds.get(path)
        if file_bounds and path not in file_bounds:
            errors.append(f"gold_spans[{index}] path missing from corpus: {path}")
        elif max_line and end_line > max_line:
            errors.append(f"gold_spans[{index}] ends after corpus file: {path}:{end_line}>{max_line}")
    return errors


def raw_gold_spans(sample: dict[str, Any]) -> list[Any] | None:
    values = sample.get("gold_spans")
    if values is None:
        values = (sample.get("gold") or {}).get("gold_spans")
    if values is None:
        return None
    return values if isinstance(values, list) else [values]


def validate_hard_negatives(
    hard_negatives: list[str],
    gold_files: list[str],
    sample: dict[str, Any],
    corpus: dict[tuple[str, str], dict[str, int]],
) -> list[str]:
    errors: list[str] = []
    overlap = sorted(set(hard_negatives) & set(gold_files))
    if overlap:
        errors.append(f"hard_negative_files overlap gold_files: {', '.join(overlap)}")
    corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
    file_bounds = corpus.get(corpus_key, {})
    if corpus and corpus_key not in corpus:
        errors.append(f"sample repo/base_commit missing from corpus manifest: {corpus_key[0]} {corpus_key[1]}")
    if file_bounds:
        missing = sorted(path for path in hard_negatives if path not in file_bounds)
        if missing:
            errors.append(f"hard_negative_files missing from corpus: {', '.join(missing)}")
    return errors


def required_corpus_paths_for_v1_2(samples: Iterable[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    required: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        paths = {span["path"] for span in gold_spans(sample)}
        paths.update(hard_negative_files(sample))
        if not paths:
            continue
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        required[key].update(paths)
    return required


def load_corpus_file_bounds(
    corpus_manifest_path: Path | None,
    required_paths: dict[tuple[str, str], set[str]] | None = None,
) -> dict[tuple[str, str], dict[str, int]]:
    if not corpus_manifest_path or not corpus_manifest_path.exists():
        return {}
    required_paths = required_paths or {}
    if not required_paths:
        return {}
    bounds: dict[tuple[str, str], dict[str, int]] = {}
    for record in read_jsonl(corpus_manifest_path):
        if record.get("status") != "ok":
            continue
        key = (str(record.get("repo") or ""), str(record.get("base_commit") or ""))
        if required_paths and key not in required_paths:
            continue
        chunks_path = Path(str(record.get("chunks_path") or ""))
        if not chunks_path.exists():
            continue
        file_bounds: dict[str, int] = {}
        needed = required_paths.get(key, set())
        with chunks_path.open("r", encoding="utf-8") as handle:
            chunk_rows = (json.loads(line) for line in handle if line.strip())
            for chunk in chunk_rows:
                path = str(chunk.get("path") or "")
                if not path:
                    continue
                if needed and path not in needed:
                    continue
                try:
                    end_line = int(chunk.get("end_line") or 0)
                except (TypeError, ValueError):
                    end_line = 0
                file_bounds[path] = max(file_bounds.get(path, 0), end_line)
        bounds[key] = file_bounds
    return bounds


def report_context_pollution(
    eval_dir: Path,
    out_path: Path,
    json_out_path: Path | None = None,
    candidate_filter: str = "all_files",
) -> dict[str, Any]:
    runs = load_detail_runs(eval_dir, candidate_filter=candidate_filter)
    rows: list[dict[str, Any]] = []
    for run in runs:
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for detail in run["details"]:
            grouped["overall"].append(detail.get("metrics") or {})
            grouped[str(detail.get("task_type") or "unknown")].append(detail.get("metrics") or {})
        for task, metrics_rows in sorted(grouped.items(), key=lambda item: task_sort_key(item[0])):
            rows.append(
                {
                    "model": run["label"],
                    "mode": run["mode"],
                    "task": task,
                    "samples": len(metrics_rows),
                    "metric_samples": {
                        key: sum(1 for metrics in metrics_rows if key in metrics)
                        for key in CONTEXT_REPORT_METRIC_KEYS
                    },
                    **average_metric_rows(metrics_rows, CONTEXT_REPORT_METRIC_KEYS),
                }
            )
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "candidate_filter": candidate_filter,
        "runs": len(runs),
        "rows": rows,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_context_pollution_markdown(report), encoding="utf-8")
    return report


def report_span_subset(
    eval_dir: Path,
    out_path: Path,
    json_out_path: Path | None = None,
    candidate_filter: str = "all_files",
) -> dict[str, Any]:
    runs = load_detail_runs(eval_dir, candidate_filter=candidate_filter)
    rows: list[dict[str, Any]] = []
    for run in runs:
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for detail in run["details"]:
            span_metrics = detail.get("span_metrics")
            if not isinstance(span_metrics, dict):
                continue
            grouped["overall"].append(span_metrics)
            grouped[str(detail.get("task_type") or "unknown")].append(span_metrics)
        for task, metrics_rows in sorted(grouped.items(), key=lambda item: task_sort_key(item[0])):
            rows.append(
                {
                    "model": run["label"],
                    "mode": run["mode"],
                    "task": task,
                    "samples": len(metrics_rows),
                    **average_metric_rows(metrics_rows, SPAN_METRIC_KEYS),
                }
            )
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "candidate_filter": candidate_filter,
        "runs": len(runs),
        "rows": rows,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_span_subset_markdown(report), encoding="utf-8")
    return report


def report_runtime_cache(
    eval_dir: Path,
    out_path: Path,
    json_out_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(eval_dir.glob("*_summary.json")):
        summary = read_json(summary_path, {})
        if not isinstance(summary, dict):
            continue
        normalized = normalize_summary(summary_path, summary) if isinstance(summary.get("metrics"), dict) else None
        runtime = summary.get("runtime") or {}
        rows.append(
            {
                "model": normalized["model_label"] if normalized else model_label(str(summary.get("model") or "lexical"), str(summary.get("mode") or "lexical")),
                "mode": str(summary.get("mode") or ""),
                "candidate_filter": str(summary.get("candidate_filter") or ""),
                "evaluated": int(summary.get("evaluated") or 0),
                "wall_time_seconds": runtime.get("wall_time_seconds"),
                "device": runtime.get("device"),
                "batch_size": runtime.get("batch_size"),
                "query_batch_size": runtime.get("query_batch_size"),
                "cache_size_bytes": runtime.get("cache_size_bytes"),
                "shared_text_cache_size_bytes": runtime.get("shared_text_cache_size_bytes"),
                "source": str(summary_path),
            }
        )
    report = {
        "generated_at": utc_now(),
        "eval_dir": str(eval_dir),
        "rows": rows,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_runtime_cache_markdown(report), encoding="utf-8")
    return report


def load_detail_runs(eval_dir: Path, candidate_filter: str = "all_files") -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for details_path in sorted(eval_dir.glob("*_details.jsonl")):
        rows = [row for row in read_jsonl(details_path) if str(row.get("candidate_filter") or "all_files") == candidate_filter]
        if not rows:
            continue
        summary_path = details_path.with_name(details_path.name.replace("_details.jsonl", "_summary.json"))
        summary = read_json(summary_path, {})
        if isinstance(summary, dict) and isinstance(summary.get("metrics"), dict):
            normalized = normalize_summary(summary_path, summary)
            label = normalized["model_label"]
            mode = normalized["mode"]
        else:
            label = details_path.name.replace("_details.jsonl", "")
            mode = "unknown"
        runs.append({"details_path": details_path, "label": label, "mode": mode, "details": rows})
    return runs


def average_metric_rows(rows: list[dict[str, float]], keys: Iterable[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows) for key in keys}


def render_v1_2_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.2 Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Derived: `{report['derived']}`",
        f"- Corpus manifest: `{report.get('corpus_manifest')}`",
        f"- Samples: `{report['samples']}`",
        f"- Span samples: `{report['span_samples']}`",
        f"- Hard-negative samples: `{report['hard_negative_samples']}`",
        f"- Query-provenance samples: `{report['query_provenance_samples']}`",
        f"- Ready: `{str(report['ready']).lower()}`",
        "",
    ]
    invalid = [row for row in report["rows"] if row["errors"]]
    if invalid:
        lines.extend(["## Invalid Rows", "", "| Sample | Task | Errors |", "| --- | --- | --- |"])
        for row in invalid[:50]:
            lines.append(f"| `{row['sample_id']}` | `{row['task_type']}` | {escape_cell('; '.join(row['errors']))} |")
    else:
        lines.append("No invalid V1.2 rows found.")
    return "\n".join(lines).rstrip() + "\n"


def render_context_pollution_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Context Pollution Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Candidate filter: `{report['candidate_filter']}`",
        "",
        "| Model | Task | Samples | P@20 | F0.5@20 | Irrelevant@20 | HardNeg@20 | Pollution@8k | Gold Token Ratio@8k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {model} | `{task}` | {samples} | {p20} | {f20} | {irr} | {hard} | {pollution} | {ratio} |".format(
                model=escape_cell(row["model"]),
                task=row["task"],
                samples=row["samples"],
                p20=format_metric(row["Precision@20"]),
                f20=format_metric(row["F0.5@20"]),
                irr=format_metric(row["irrelevant_files@20"]),
                hard=format_metric(row["hard_negative_hits@20"]),
                pollution=format_metric(row["context_pollution_tokens@8k"]),
                ratio=format_metric(row["gold_token_ratio@8k"]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_span_subset_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Span Subset Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Candidate filter: `{report['candidate_filter']}`",
        "",
        "| Model | Task | Span Samples | Span Recall@8k | Span Precision@8k | Span F0.5@8k | Overlap Lines |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {model} | `{task}` | {samples} | {recall} | {precision} | {f05} | {overlap} |".format(
                model=escape_cell(row["model"]),
                task=row["task"],
                samples=row["samples"],
                recall=format_metric(row["span_recall@8k"]),
                precision=format_metric(row["span_precision@8k"]),
                f05=format_metric(row["span_f0.5@8k"]),
                overlap=format_metric(row["overlap_lines"]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_runtime_cache_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Runtime And Cache Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval dir: `{report['eval_dir']}`",
        "",
        "| Model | Mode | Evaluated | Wall Time (s) | Device | Batch | Query Batch | Cache Size | Shared Text Cache |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {model} | `{mode}` | {evaluated} | {wall} | {device} | {batch} | {query_batch} | {cache} | {shared} |".format(
                model=escape_cell(row["model"]),
                mode=row["mode"],
                evaluated=row["evaluated"],
                wall=format_optional(row.get("wall_time_seconds")),
                device=escape_cell(str(row.get("device") or "")),
                batch=format_optional(row.get("batch_size"), decimals=0),
                query_batch=format_optional(row.get("query_batch_size"), decimals=0),
                cache=format_optional(row.get("cache_size_bytes"), decimals=0),
                shared=format_optional(row.get("shared_text_cache_size_bytes"), decimals=0),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_manual_annotation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.2 Manual Annotation Merge",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Base derived: `{report['base_derived']}`",
        f"- Annotations: `{report['annotations']}`",
        f"- Output: `{report['out_dir']}`",
        f"- Samples: `{report['samples']}`",
        f"- Annotated samples: `{report['annotation_count']}`",
        f"- Span samples: `{report['span_samples']}`",
        f"- Hard-negative samples: `{report['hard_negative_samples']}`",
        f"- Sample ids preserved: `{str(report['sample_ids_preserved']).lower()}`",
        "",
        "| Task | Annotated Samples |",
        "| --- | ---: |",
    ]
    for task, count in report["annotated_counts_by_task"].items():
        lines.append(f"| `{task}` | {count} |")
    return "\n".join(lines).rstrip() + "\n"


def format_metric(value: Any) -> str:
    return f"{float(value or 0.0):.4f}"


def format_optional(value: Any, decimals: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|")
