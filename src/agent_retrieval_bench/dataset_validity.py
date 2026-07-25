from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Iterable

from .audit import AUDIT_FIELDS, normalize_verdict, query_excerpt, should_keep
from .baseline import gold_blocks, gold_spans, query_provenance, query_text_for_eval, target_gold_files
from .filters import contains_raw_patch_marker, contains_review_leakage, extract_repo_trace_paths, is_test_file
from .io import ensure_parent, read_json, read_jsonl, truncate_text, utc_now, write_json, write_jsonl

TASK_TYPES = ("code2test", "comment2context", "trace2code")
DEFAULT_AUDIT_PATHS = (
    Path("data/reports/v1_3/manual_annotations_code2test_codex_reviewed.jsonl"),
    Path("data/reports/v1_3/manual_annotations_comment2context_codex_reviewed.jsonl"),
    Path("data/reports/v1_3/manual_annotations_trace2code_codex_reviewed.jsonl"),
)
DEFAULT_EVAL_DIRS = (
    Path("data/eval/v1_3_reviewed"),
    Path("data/eval/v1_4/context_selection"),
    Path("data/eval/v1_4/context_budget"),
    Path("data/eval/v1_4"),
)
DEFAULT_SPLIT_DETAILS_DIRS = (
    Path("data/eval/v1_3_reviewed"),
    Path("data/eval/v1_4/context_selection"),
    Path("data/eval/v1_4/context_budget"),
)
FATAL_LEAKAGE_TYPES = {"exact_gold_path", "raw_patch_marker", "review_leakage", "fix_commit_hash"}
LEAKAGE_TYPES = (
    "exact_gold_path",
    "gold_basename",
    "gold_stem",
    "test_name",
    "gold_symbol",
    "test_symbol",
    "raw_patch_marker",
    "review_leakage",
    "fix_commit_hash",
    "direct_stack_hint",
)
REVIEW_REASON_MARKERS = ("manual", "human_verified", "codex review", "reviewed", "accepted")
METRIC_KEYS = (
    "Recall@20",
    "MRR",
    "nDCG@20",
    "gold_coverage@8k",
    "context_efficiency@8k",
    "final_file_f1",
    "final_file_precision",
    "final_file_recall",
    "retrieved_file_f1",
)
DIVERSITY_METRICS = ("Recall@20", "nDCG@20", "gold_coverage@8k", "final_file_f1")
COMMON_STEMS = {
    "test",
    "tests",
    "index",
    "main",
    "init",
    "utils",
    "util",
    "common",
    "helper",
    "helpers",
    "types",
    "mod",
    "lib",
    "app",
}
COMMON_SYMBOLS = {
    "test",
    "tests",
    "main",
    "init",
    "setup",
    "teardown",
    "some",
    "none",
    "cmd",
    "args",
    "expected",
    "actual",
    "value",
    "error",
}


def report_dataset_validity(
    samples_path: Path,
    corpus_manifest_path: Path,
    out_dir: Path,
    validation_path: Path | None = None,
    audit_paths: Iterable[Path] | None = None,
    eval_dirs: Iterable[Path] | None = None,
    split_details_dirs: Iterable[Path] | None = None,
    audit_packet_size: int = 90,
    valid_threshold: float = 0.90,
    min_task_spread: float = 0.05,
) -> dict[str, Any]:
    samples = read_jsonl(samples_path)
    samples_by_id = {str(sample.get("id")): sample for sample in samples}
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_result = build_gold_audit_summary(
        samples=samples,
        validation_path=validation_path,
        audit_paths=list(audit_paths) if audit_paths is not None else list(DEFAULT_AUDIT_PATHS),
        out_dir=out_dir,
        audit_packet_size=audit_packet_size,
        valid_threshold=valid_threshold,
    )
    leakage_result = build_leakage_diagnostic(
        samples=samples,
        out_dir=out_dir,
        split_details_dirs=list(split_details_dirs) if split_details_dirs is not None else list(DEFAULT_SPLIT_DETAILS_DIRS),
    )
    diversity_result = build_task_diversity_report(
        samples=samples,
        samples_path=samples_path,
        corpus_manifest_path=corpus_manifest_path,
        eval_dirs=list(eval_dirs) if eval_dirs is not None else list(DEFAULT_EVAL_DIRS),
        out_dir=out_dir,
        min_task_spread=min_task_spread,
    )

    combined = {
        "generated_at": utc_now(),
        "inputs": {
            "samples": str(samples_path),
            "corpus_manifest": str(corpus_manifest_path),
            "validation": str(validation_path) if validation_path else None,
        },
        "sample_count": len(samples),
        "task_counts": dict(sorted(Counter(str(sample.get("task_type")) for sample in samples).items())),
        "gold_audit": audit_result["summary"],
        "leakage": leakage_result["summary"],
        "task_diversity": diversity_result["summary"],
        "sample_ids_loaded": len(samples_by_id),
        "artifacts": {
            "gold_audit_json": audit_result["json"],
            "gold_audit_markdown": audit_result["markdown"],
            "leakage_json": leakage_result["json"],
            "leakage_markdown": leakage_result["markdown"],
            "task_diversity_json": diversity_result["json"],
            "task_diversity_markdown": diversity_result["markdown"],
            "combined_markdown": str(out_dir / "validity_report.md"),
        },
    }
    write_json(out_dir / "validity_report.json", combined)
    (out_dir / "validity_report.md").write_text(
        render_combined_validity_markdown(combined, audit_result, leakage_result, diversity_result),
        encoding="utf-8",
    )
    return {
        "samples": len(samples),
        "task_counts": combined["task_counts"],
        "out_dir": str(out_dir),
        "validity_report": str(out_dir / "validity_report.md"),
        "validity_json": str(out_dir / "validity_report.json"),
        "gold_audit": audit_result["json"],
        "leakage": leakage_result["json"],
        "task_diversity": diversity_result["json"],
    }


def build_gold_audit_summary(
    samples: list[dict[str, Any]],
    validation_path: Path | None,
    audit_paths: list[Path],
    out_dir: Path,
    audit_packet_size: int,
    valid_threshold: float,
) -> dict[str, Any]:
    validation = read_validation_rows(validation_path)
    evidence_by_id = read_review_evidence(audit_paths)
    sample_rows: list[dict[str, Any]] = []
    pending_samples: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = str(sample.get("id"))
        direct_evidence = evidence_by_id.get(sample_id, [])
        validation_row = validation.get(sample_id, {})
        validation_clean = not validation_row.get("errors") if validation_row else None
        reviewed_annotation = has_reviewed_gold_annotation(sample)
        verdict, basis = gold_verdict_for_sample(direct_evidence, validation_clean, reviewed_annotation)
        gold_files = target_gold_files(sample)
        row = {
            "sample_id": sample_id,
            "task_type": str(sample.get("task_type") or ""),
            "repo": str(sample.get("repo") or ""),
            "query_provenance": query_provenance(sample),
            "gold_files": gold_files,
            "gold_span_count": len(gold_spans(sample)),
            "gold_block_count": len(gold_blocks(sample)),
            "validation_clean": validation_clean,
            "reviewed_gold_annotation": reviewed_annotation,
            "direct_evidence_count": len(direct_evidence),
            "direct_evidence_sources": sorted({item["source"] for item in direct_evidence}),
            "verdict": verdict,
            "basis": basis,
        }
        sample_rows.append(row)
        if verdict == "pending":
            pending_samples.append(sample)
        elif verdict != "valid":
            drop_rows.append({"sample_id": sample_id, "task_type": row["task_type"], "repo": row["repo"], "verdict": verdict, "basis": basis})

    packet_rows = stratified_audit_packet(pending_samples, audit_packet_size)
    packet_jsonl = out_dir / "gold_audit_packet.jsonl"
    packet_csv = out_dir / "gold_audit_packet.csv"
    write_jsonl(packet_jsonl, packet_rows)
    write_audit_packet_csv(packet_csv, packet_rows)
    drop_path = out_dir / "drop_list.jsonl"
    write_jsonl(drop_path, drop_rows)

    summary = summarize_gold_audit_rows(
        sample_rows=sample_rows,
        validation_path=validation_path,
        audit_paths=audit_paths,
        audit_packet_jsonl=packet_jsonl,
        audit_packet_csv=packet_csv,
        drop_path=drop_path,
        valid_threshold=valid_threshold,
    )
    report = {"summary": summary, "samples": sample_rows, "pending_packet": packet_rows, "drop_list": drop_rows}
    json_path = out_dir / "gold_audit_summary.json"
    md_path = out_dir / "gold_audit_summary.md"
    write_json(json_path, report)
    md_path.write_text(render_gold_audit_markdown(summary, sample_rows, packet_rows, drop_rows), encoding="utf-8")
    return {"summary": summary, "json": str(json_path), "markdown": str(md_path), "samples": sample_rows}


def read_validation_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = read_json(path, {})
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    return {str(row.get("sample_id")): row for row in rows if isinstance(row, dict) and row.get("sample_id")}


def read_review_evidence(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for record in audit_records_from_path(path):
            item = review_evidence_from_record(record, path)
            if item:
                evidence[item["sample_id"]].append(item)
    return evidence


def audit_records_from_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = read_json(path, {})
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("rows", "adjustments", "annotations", "samples"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def review_evidence_from_record(record: dict[str, Any], path: Path) -> dict[str, Any] | None:
    sample_id = str(record.get("sample_id") or record.get("id") or "")
    if not sample_id:
        return None
    raw_status = str(record.get("verdict") or record.get("review_status") or record.get("status") or "").strip().lower()
    keep_value = str(record.get("keep") or "").strip().lower()
    if raw_status in {"accepted", "accept", "valid", "kept", "keep"} or keep_value in {"1", "true", "yes", "y", "keep"}:
        verdict = "valid"
    elif raw_status in {"leaked", "leak"}:
        verdict = "leaked"
    elif raw_status in {"ambiguous", "unclear"}:
        verdict = "ambiguous"
    elif raw_status in {"noisy", "rejected", "reject", "drop", "dropped", "invalid"} or keep_value in {"0", "false", "no", "n", "drop"}:
        verdict = "noisy"
    else:
        verdict = normalize_verdict(raw_status)
    return {
        "sample_id": sample_id,
        "task_type": str(record.get("task_type") or ""),
        "repo": str(record.get("repo") or ""),
        "verdict": verdict,
        "status": raw_status,
        "reviewer": str(record.get("reviewer") or ""),
        "source": str(path),
        "reason": str(record.get("reason") or record.get("annotation_note") or record.get("notes") or ""),
    }


def has_reviewed_gold_annotation(sample: dict[str, Any]) -> bool:
    if not target_gold_files(sample):
        return False
    span_rows = gold_spans(sample)
    block_rows = gold_blocks(sample)
    if not span_rows or not block_rows:
        return False
    reasons = "\n".join(str(row.get("reason") or "") for row in [*span_rows, *block_rows]).lower()
    return any(marker in reasons for marker in REVIEW_REASON_MARKERS)


def gold_verdict_for_sample(
    direct_evidence: list[dict[str, Any]],
    validation_clean: bool | None,
    reviewed_annotation: bool,
) -> tuple[str, str]:
    direct_verdicts = [str(item.get("verdict") or "pending") for item in direct_evidence]
    for verdict in ("leaked", "noisy", "ambiguous", "other"):
        if verdict in direct_verdicts:
            return verdict, "direct_audit"
    if "valid" in direct_verdicts:
        return "valid", "direct_audit"
    if reviewed_annotation and validation_clean is not False:
        return "valid", "reviewed_gold_annotation"
    return "pending", "needs_manual_audit"


def summarize_gold_audit_rows(
    sample_rows: list[dict[str, Any]],
    validation_path: Path | None,
    audit_paths: list[Path],
    audit_packet_jsonl: Path,
    audit_packet_csv: Path,
    drop_path: Path,
    valid_threshold: float,
) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for task in sorted({row["task_type"] for row in sample_rows}):
        rows = [row for row in sample_rows if row["task_type"] == task]
        by_task[task] = summarize_gold_audit_group(rows)
    overall = summarize_gold_audit_group(sample_rows)
    return {
        "generated_at": utc_now(),
        "validation_path": str(validation_path) if validation_path else None,
        "audit_paths": [str(path) for path in audit_paths],
        "total": len(sample_rows),
        "overall": overall,
        "by_task": by_task,
        "valid_threshold": valid_threshold,
        "passes_valid_threshold": overall["valid_rate_reviewed"] >= valid_threshold if overall["reviewed"] else False,
        "audit_packet": {"jsonl": str(audit_packet_jsonl), "csv": str(audit_packet_csv), "rows": overall["pending"]},
        "drop_list": {"path": str(drop_path), "rows": overall["non_valid_reviewed"]},
        "basis_counts": dict(sorted(Counter(row["basis"] for row in sample_rows).items())),
    }


def summarize_gold_audit_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["verdict"] for row in rows)
    reviewed = len(rows) - counts.get("pending", 0)
    valid = counts.get("valid", 0)
    non_valid = reviewed - valid
    total = len(rows)
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": counts.get("pending", 0),
        "coverage": reviewed / total if total else 0.0,
        "valid": valid,
        "valid_rate_reviewed": valid / reviewed if reviewed else 0.0,
        "valid_rate_total": valid / total if total else 0.0,
        "non_valid_reviewed": non_valid,
        "counts": {key: counts.get(key, 0) for key in ("valid", "noisy", "leaked", "ambiguous", "other", "pending")},
        "direct_evidence": sum(1 for row in rows if row["direct_evidence_count"]),
        "reviewed_gold_annotation": sum(1 for row in rows if row["reviewed_gold_annotation"]),
        "validation_clean": sum(1 for row in rows if row["validation_clean"] is True),
    }


def stratified_audit_packet(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not samples:
        return []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (str(sample.get("task_type") or ""), str(sample.get("repo") or ""), str(query_provenance(sample) or ""))
        groups[key].append(sample)
    for group in groups.values():
        group.sort(key=lambda sample: str(sample.get("id") or ""))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < limit and any(groups.values()):
        key = keys[cursor % len(keys)]
        cursor += 1
        if groups[key]:
            selected.append(groups[key].pop(0))
    return [audit_packet_row(sample) for sample in selected]


def audit_packet_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(sample.get("id") or ""),
        "task_type": str(sample.get("task_type") or ""),
        "repo": str(sample.get("repo") or ""),
        "query_provenance": query_provenance(sample) or "",
        "query_excerpt": query_excerpt(sample),
        "gold_files": target_gold_files(sample),
        "gold_spans": gold_spans(sample),
        "gold_blocks": gold_blocks(sample),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "",
    }


def write_audit_packet_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [*AUDIT_FIELDS, "query_provenance", "gold_span_count", "gold_block_count"]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "task_type": row["task_type"],
                    "repo": row["repo"],
                    "query_excerpt": row["query_excerpt"],
                    "gold_files": "; ".join(row["gold_files"]),
                    "verdict": row["verdict"],
                    "reason": row["reason"],
                    "keep": row["keep"],
                    "notes": row["notes"],
                    "query_provenance": row["query_provenance"],
                    "gold_span_count": len(row["gold_spans"]),
                    "gold_block_count": len(row["gold_blocks"]),
                }
            )


def build_leakage_diagnostic(
    samples: list[dict[str, Any]],
    out_dir: Path,
    split_details_dirs: list[Path],
) -> dict[str, Any]:
    rows = [detect_query_leakage(sample) for sample in samples]
    clean_splits = summarize_clean_splits(split_details_dirs, rows, expected_samples=len(samples))
    summary = summarize_leakage_rows(rows, clean_splits)
    report = {"summary": summary, "samples": rows, "clean_splits": clean_splits}
    json_path = out_dir / "leakage_diagnostic.json"
    md_path = out_dir / "leakage_diagnostic.md"
    write_json(json_path, report)
    md_path.write_text(render_leakage_markdown(summary, rows, clean_splits), encoding="utf-8")
    return {"summary": summary, "json": str(json_path), "markdown": str(md_path), "samples": rows}


def detect_query_leakage(sample: dict[str, Any]) -> dict[str, Any]:
    query_text = query_text_for_eval(sample)
    normalized_query = query_text.replace("\\n", "\n")
    lowered_query = normalized_query.lower()
    task_type = str(sample.get("task_type") or "")
    gold_files = target_gold_files(sample)
    flags: list[dict[str, Any]] = []

    trace_hits = trace_gold_path_hits(normalized_query, gold_files) if task_type == "trace2code" else []
    if trace_hits:
        flags.append(flag("direct_stack_hint", trace_hits, fatal=False))

    exact_hits = [path for path in gold_files if path and path.lower() in lowered_query and path not in trace_hits]
    if exact_hits:
        flags.append(flag("exact_gold_path", exact_hits, fatal=True))

    basename_hits: list[str] = []
    stem_hits: list[str] = []
    test_name_hits: list[str] = []
    for path in gold_files:
        basename = PurePosixPath(path).name
        stem = PurePosixPath(path).stem
        if meaningful_basename(basename) and tokenish_contains(lowered_query, basename.lower()) and path not in exact_hits:
            basename_hits.append(path)
            if task_type == "code2test" and is_test_file(path):
                test_name_hits.append(basename)
        if meaningful_stem(stem) and tokenish_contains(lowered_query, stem.lower()) and path not in basename_hits and path not in exact_hits:
            stem_hits.append(path)
            if task_type == "code2test" and is_test_file(path):
                test_name_hits.append(stem)
    if basename_hits:
        flags.append(flag("gold_basename", basename_hits, fatal=False))
    if stem_hits:
        flags.append(flag("gold_stem", stem_hits, fatal=False))

    symbol_hits, test_symbol_hits = query_symbol_hits(sample, normalized_query)
    if symbol_hits:
        flags.append(flag("gold_symbol", symbol_hits, fatal=False))
    if test_symbol_hits:
        flags.append(flag("test_symbol", test_symbol_hits, fatal=False))
    if test_name_hits:
        flags.append(flag("test_name", sorted(set(test_name_hits)), fatal=False))

    if contains_raw_patch_marker(normalized_query):
        flags.append(flag("raw_patch_marker", ["diff/patch marker"], fatal=True))
    if contains_review_leakage(normalized_query):
        flags.append(flag("review_leakage", ["review leakage marker"], fatal=True))
    commit_hits = fix_commit_hits(sample, normalized_query)
    if commit_hits:
        flags.append(flag("fix_commit_hash", commit_hits, fatal=True))

    flag_types = [item["type"] for item in flags]
    return {
        "sample_id": str(sample.get("id") or ""),
        "task_type": task_type,
        "repo": str(sample.get("repo") or ""),
        "query_provenance": query_provenance(sample),
        "gold_files": gold_files,
        "flags": flags,
        "flag_types": flag_types,
        "flagged": bool(flags),
        "fatal": any(item["fatal"] for item in flags),
        "query_excerpt": truncate_text(normalized_query, 700),
    }


def flag(flag_type: str, hits: list[str], fatal: bool) -> dict[str, Any]:
    return {"type": flag_type, "hits": sorted(set(hits)), "fatal": fatal}


def trace_gold_path_hits(query_text: str, gold_files: list[str]) -> list[str]:
    trace_text = query_text.replace('\\"', '"').replace('\\/', '/')
    trace_paths = extract_repo_trace_paths(trace_text)
    hits: list[str] = []
    for gold_file in gold_files:
        for trace_path in trace_paths:
            if path_matches(gold_file, trace_path):
                hits.append(gold_file)
                break
    return sorted(set(hits))


def path_matches(gold_path: str, observed_path: str) -> bool:
    gold = gold_path.replace("\\", "/").strip("/").lower()
    observed = observed_path.replace("\\", "/").strip("/").lower()
    if not gold or not observed:
        return False
    return observed == gold or observed.endswith("/" + gold) or PurePosixPath(observed).name == PurePosixPath(gold).name


def meaningful_basename(value: str) -> bool:
    lowered = value.lower()
    return len(lowered) >= 6 and lowered not in {"__init__.py", "index.js", "index.ts", "main.py"}


def meaningful_stem(value: str) -> bool:
    lowered = value.lower().strip("_")
    return len(lowered) >= 5 and lowered not in COMMON_STEMS and not lowered.isdigit()


def tokenish_contains(lowered_text: str, lowered_term: str) -> bool:
    escaped = re.escape(lowered_term)
    return bool(re.search(rf"(?<![a-z0-9_./-]){escaped}(?![a-z0-9_./-])", lowered_text))


def query_symbol_hits(sample: dict[str, Any], query_text: str) -> tuple[list[str], list[str]]:
    lowered = query_text.lower()
    hits: list[str] = []
    test_hits: list[str] = []
    for symbol, path in gold_symbols(sample):
        if not meaningful_symbol(symbol):
            continue
        if symbol_in_query(symbol, lowered):
            hits.append(symbol)
            if path and is_test_file(path):
                test_hits.append(symbol)
    return sorted(set(hits)), sorted(set(test_hits))


def gold_symbols(sample: dict[str, Any]) -> list[tuple[str, str]]:
    gold = sample.get("gold") or {}
    pairs: list[tuple[str, str]] = []
    for key in ("root_cause_symbols", "symbols", "related_symbols", "target_symbols"):
        for value in gold.get(key) or []:
            if isinstance(value, str):
                pairs.append((value, ""))
            elif isinstance(value, dict):
                pairs.append((str(value.get("symbol") or value.get("name") or ""), str(value.get("path") or "")))
    for block in gold_blocks(sample):
        if block.get("symbol"):
            pairs.append((str(block.get("symbol")), str(block.get("path") or "")))
    return pairs


def meaningful_symbol(symbol: str) -> bool:
    lowered = symbol.lower().strip("_")
    if len(lowered) < 5 or lowered in COMMON_SYMBOLS:
        return False
    return any(char.isalpha() for char in lowered)


def symbol_in_query(symbol: str, lowered_query: str) -> bool:
    lowered = symbol.lower()
    if not re.match(r"^[a-z_][a-z0-9_]*$", lowered):
        return lowered in lowered_query
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])", lowered_query))


def fix_commit_hits(sample: dict[str, Any], query_text: str) -> list[str]:
    fix_commit = str((sample.get("gold") or {}).get("fix_commit") or "").strip()
    if len(fix_commit) < 7:
        return []
    lowered_query = query_text.lower()
    hits = []
    if fix_commit.lower() in lowered_query:
        hits.append(fix_commit)
    elif fix_commit[:12].lower() in lowered_query:
        hits.append(fix_commit[:12])
    elif fix_commit[:7].lower() in lowered_query:
        hits.append(fix_commit[:7])
    return hits


def summarize_leakage_rows(rows: list[dict[str, Any]], clean_splits: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for task in sorted({row["task_type"] for row in rows}):
        task_rows = [row for row in rows if row["task_type"] == task]
        by_task[task] = summarize_leakage_group(task_rows)
    by_type = Counter(flag["type"] for row in rows for flag in row["flags"])
    fatal_by_type = Counter(flag["type"] for row in rows for flag in row["flags"] if flag["fatal"])
    return {
        "generated_at": utc_now(),
        "total": len(rows),
        "flagged_samples": sum(1 for row in rows if row["flagged"]),
        "fatal_flagged_samples": sum(1 for row in rows if row["fatal"]),
        "clean_samples": sum(1 for row in rows if not row["flagged"]),
        "fatal_clean_samples": sum(1 for row in rows if not row["fatal"]),
        "by_task": by_task,
        "by_type": {key: by_type.get(key, 0) for key in LEAKAGE_TYPES},
        "fatal_by_type": dict(sorted(fatal_by_type.items())),
        "examples_by_type": leakage_examples_by_type(rows),
        "clean_split_files": len(clean_splits),
    }


def summarize_leakage_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(flag["type"] for row in rows for flag in row["flags"])
    total = len(rows)
    return {
        "total": total,
        "flagged_samples": sum(1 for row in rows if row["flagged"]),
        "fatal_flagged_samples": sum(1 for row in rows if row["fatal"]),
        "flagged_rate": sum(1 for row in rows if row["flagged"]) / total if total else 0.0,
        "fatal_rate": sum(1 for row in rows if row["fatal"]) / total if total else 0.0,
        "by_type": {key: by_type.get(key, 0) for key in LEAKAGE_TYPES},
    }


def leakage_examples_by_type(rows: list[dict[str, Any]], limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {key: [] for key in LEAKAGE_TYPES}
    for row in rows:
        for item in row["flags"]:
            bucket = examples[item["type"]]
            if len(bucket) < limit:
                bucket.append(
                    {
                        "sample_id": row["sample_id"],
                        "task_type": row["task_type"],
                        "repo": row["repo"],
                        "hits": item["hits"][:5],
                        "fatal": item["fatal"],
                    }
                )
    return {key: value for key, value in examples.items() if value}


def summarize_clean_splits(details_dirs: list[Path], leakage_rows: list[dict[str, Any]], expected_samples: int) -> list[dict[str, Any]]:
    leakage_by_id = {row["sample_id"]: row for row in leakage_rows}
    summaries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for directory in details_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_details.jsonl")):
            if path in seen:
                continue
            seen.add(path)
            rows = read_jsonl(path)
            if len(rows) < max(1, int(expected_samples * 0.5)):
                continue
            split_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
            for row in rows:
                sample_id = str(row.get("sample_id") or "")
                leakage = leakage_by_id.get(sample_id)
                if leakage is None:
                    continue
                metrics = metric_payload(row)
                if not metrics:
                    continue
                if leakage["fatal"]:
                    split = "fatal_flagged"
                elif leakage["flagged"]:
                    split = "nonfatal_flagged"
                else:
                    split = "clean"
                split_rows[split].append(metrics)
            if split_rows:
                summaries.append(
                    {
                        "details": str(path),
                        "method": method_label_from_path(path),
                        "splits": {split: average_metric_payloads(values) for split, values in sorted(split_rows.items())},
                    }
                )
    return summaries


def metric_payload(row: dict[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    payload: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in metrics:
            payload[key] = float(metrics[key] or 0.0)
    return payload


def average_metric_payloads(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {"samples": len(rows), **{key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}}


def build_task_diversity_report(
    samples: list[dict[str, Any]],
    samples_path: Path,
    corpus_manifest_path: Path,
    eval_dirs: list[Path],
    out_dir: Path,
    min_task_spread: float,
) -> dict[str, Any]:
    rows = load_eval_rows(eval_dirs, expected_samples=len(samples))
    sample_distribution = sample_distribution_summary(samples)
    macro = macro_average_rows(rows)
    spread = metric_spread(rows)
    correlations = task_rank_correlations(rows)
    contributions = task_contributions(rows, sample_distribution["by_task"])
    summary = {
        "generated_at": utc_now(),
        "inputs": {"samples": str(samples_path), "corpus_manifest": str(corpus_manifest_path), "eval_dirs": [str(path) for path in eval_dirs]},
        "sample_distribution": sample_distribution,
        "row_count": len(rows),
        "method_count": len({row["method_key"] for row in rows}),
        "macro_average_rows": len(macro),
        "spread_checks": spread_checks(spread, min_task_spread),
        "min_task_spread": min_task_spread,
    }
    report = {
        "summary": summary,
        "leaderboard_rows": rows,
        "macro_average": macro,
        "spread": spread,
        "task_rank_correlations": correlations,
        "task_contributions": contributions,
    }
    json_path = out_dir / "task_diversity_report.json"
    md_path = out_dir / "task_diversity_report.md"
    write_json(json_path, report)
    md_path.write_text(render_task_diversity_markdown(report), encoding="utf-8")
    return {"summary": summary, "json": str(json_path), "markdown": str(md_path), "rows": rows}


def sample_distribution_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = Counter(str(sample.get("task_type") or "") for sample in samples)
    by_repo = Counter(str(sample.get("repo") or "") for sample in samples)
    by_task_repo: dict[str, Counter[str]] = defaultdict(Counter)
    by_task_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        task = str(sample.get("task_type") or "")
        by_task_repo[task][str(sample.get("repo") or "")] += 1
        by_task_provenance[task][str(query_provenance(sample) or "unknown")] += 1
    return {
        "total": len(samples),
        "by_task": dict(sorted(by_task.items())),
        "by_repo": dict(sorted(by_repo.items())),
        "by_task_repo": {task: dict(sorted(counts.items())) for task, counts in sorted(by_task_repo.items())},
        "by_task_provenance": {task: dict(sorted(counts.items())) for task, counts in sorted(by_task_provenance.items())},
    }


def load_eval_rows(eval_dirs: list[Path], expected_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for directory in eval_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_summary.json")):
            if path in seen:
                continue
            seen.add(path)
            summary = read_json(path, {})
            if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
                continue
            if not full_benchmark_summary(summary, expected_samples):
                continue
            metrics_by_task = dict(summary["metrics"])
            add_ndcg_from_details(metrics_by_task, details_path_for_summary(path))
            method = method_label(summary, path)
            top_k = summary.get("top_k") or parse_top_k(method, path)
            method_type = infer_method_type(method, path, summary)
            context = str(summary.get("context") or summary.get("mode") or "")
            top_k_label = str(top_k) if top_k is not None else ""
            method_key = f"{method}|{context}|top{top_k_label}|{path.parent.name}"
            for task, metrics in metrics_by_task.items():
                if not isinstance(metrics, dict):
                    continue
                row = {
                    "method": method,
                    "method_key": method_key,
                    "method_type": method_type,
                    "task": str(task),
                    "top_k": top_k,
                    "samples": int(metrics.get("samples") or summary.get("evaluated") or 0),
                    "source": str(path),
                }
                for key in METRIC_KEYS:
                    if key in metrics:
                        row[key] = float(metrics.get(key) or 0.0)
                rows.append(row)
    rows.sort(key=lambda row: (task_sort_key(row["task"]), row["method_type"], -(row.get("final_file_f1") or row.get("gold_coverage@8k") or row.get("Recall@20") or 0.0), row["method"]))
    return rows


def full_benchmark_summary(summary: dict[str, Any], expected_samples: int) -> bool:
    metrics = summary.get("metrics") or {}
    overall = metrics.get("overall") if isinstance(metrics, dict) else None
    samples = 0
    if isinstance(overall, dict):
        samples = int(overall.get("samples") or 0)
    evaluated = int(summary.get("evaluated") or 0)
    return samples == expected_samples or evaluated == expected_samples


def details_path_for_summary(path: Path) -> Path:
    name = path.name.replace("_summary.json", "_details.jsonl")
    return path.with_name(name)


def add_ndcg_from_details(metrics_by_task: dict[str, Any], details_path: Path, cutoff: int = 20) -> None:
    if not details_path.exists():
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(details_path):
        ranks = row.get("gold_ranks") or {}
        if not isinstance(ranks, dict) or not ranks:
            continue
        score = ndcg_from_ranks(ranks, cutoff=cutoff)
        task = str(row.get("task_type") or "overall")
        grouped[task].append(score)
        grouped["overall"].append(score)
    for task, values in grouped.items():
        if task in metrics_by_task and isinstance(metrics_by_task[task], dict) and values:
            metrics_by_task[task][f"nDCG@{cutoff}"] = sum(values) / len(values)


def ndcg_from_ranks(ranks: dict[str, Any], cutoff: int = 20) -> float:
    rank_values = [int(rank) for rank in ranks.values() if isinstance(rank, int) and rank > 0 and rank <= cutoff]
    total_relevant = max(1, len(ranks))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in rank_values)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(total_relevant, cutoff) + 1))
    return dcg / ideal if ideal else 0.0


def method_label(summary: dict[str, Any], path: Path) -> str:
    value = str(summary.get("model_label") or summary.get("model") or "").strip()
    if value:
        return value
    stem = path.stem.replace("_summary", "")
    return stem.replace("_", " ")


def parse_top_k(method: str, path: Path) -> int | None:
    text = f"{method} {path.stem}"
    match = re.search(r"(?:@|top)(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_method_type(method: str, path: Path, summary: dict[str, Any]) -> str:
    mode = str(summary.get("mode") or "")
    text = f"{method} {path} {mode}".lower()
    if any(marker in text for marker in ("codex", "openai", "gpt")) and "ranked_context" not in text:
        return "agent"
    return "retriever"


def macro_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task"] in TASK_TYPES:
            grouped[row["method_key"]].append(row)
    output: list[dict[str, Any]] = []
    for method_key, method_rows in grouped.items():
        tasks = {row["task"] for row in method_rows}
        if not set(TASK_TYPES).issubset(tasks):
            continue
        base = method_rows[0]
        macro = {
            "method": base["method"],
            "method_key": method_key,
            "method_type": base["method_type"],
            "top_k": base.get("top_k"),
            "task_count": len(TASK_TYPES),
        }
        for key in METRIC_KEYS:
            values = [row[key] for row in method_rows if row["task"] in TASK_TYPES and key in row]
            if len(values) == len(TASK_TYPES):
                macro[f"macro_{key}"] = sum(values) / len(values)
        overall = next((row for row in rows if row["method_key"] == method_key and row["task"] == "overall"), None)
        if overall:
            for key in METRIC_KEYS:
                if key in overall:
                    macro[f"overall_{key}"] = overall[key]
                    macro[f"macro_minus_overall_{key}"] = macro.get(f"macro_{key}", overall[key]) - overall[key]
        output.append(macro)
    output.sort(key=lambda row: -(row.get("macro_final_file_f1") or row.get("macro_gold_coverage@8k") or row.get("macro_Recall@20") or 0.0))
    return output


def metric_spread(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in DIVERSITY_METRICS:
        per_task: dict[str, Any] = {}
        for task in TASK_TYPES:
            values = [(row["method"], float(row[metric])) for row in rows if row["task"] == task and metric in row]
            if not values:
                continue
            values.sort(key=lambda item: item[1], reverse=True)
            nums = [value for _, value in values]
            per_task[task] = {
                "methods": len(values),
                "best_method": values[0][0],
                "best": values[0][1],
                "median": median(nums),
                "worst": values[-1][1],
                "spread": values[0][1] - values[-1][1],
                "best_minus_median": values[0][1] - median(nums),
            }
        if per_task:
            output[metric] = per_task
    return output


def spread_checks(spread: dict[str, dict[str, Any]], min_task_spread: float) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for metric, per_task in spread.items():
        checks[metric] = {
            task: {
                "spread": row["spread"],
                "passes": row["spread"] >= min_task_spread,
            }
            for task, row in per_task.items()
        }
    return checks


def task_rank_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pairs = (("code2test", "comment2context"), ("code2test", "trace2code"), ("comment2context", "trace2code"))
    for metric in DIVERSITY_METRICS:
        for left, right in pairs:
            left_values = {row["method_key"]: row[metric] for row in rows if row["task"] == left and metric in row}
            right_values = {row["method_key"]: row[metric] for row in rows if row["task"] == right and metric in row}
            shared = sorted(set(left_values) & set(right_values))
            if len(shared) < 3:
                continue
            output.append(
                {
                    "metric": metric,
                    "task_a": left,
                    "task_b": right,
                    "shared_methods": len(shared),
                    "spearman": spearman([left_values[key] for key in shared], [right_values[key] for key in shared]),
                }
            )
    return output


def task_contributions(rows: list[dict[str, Any]], task_counts: dict[str, int]) -> list[dict[str, Any]]:
    total = sum(task_counts.get(task, 0) for task in TASK_TYPES)
    output: list[dict[str, Any]] = []
    for metric in DIVERSITY_METRICS:
        overall_rows = [row for row in rows if row["task"] == "overall" and metric in row]
        if len(overall_rows) < 2:
            continue
        overall_rows.sort(key=lambda row: row[metric], reverse=True)
        first, second = overall_rows[0], overall_rows[1]
        overall_delta = first[metric] - second[metric]
        if abs(overall_delta) < 1e-12:
            continue
        task_items: list[dict[str, Any]] = []
        for task in TASK_TYPES:
            first_task = next((row for row in rows if row["method_key"] == first["method_key"] and row["task"] == task and metric in row), None)
            second_task = next((row for row in rows if row["method_key"] == second["method_key"] and row["task"] == task and metric in row), None)
            if not first_task or not second_task:
                continue
            weighted_delta = (task_counts.get(task, 0) / total) * (first_task[metric] - second_task[metric]) if total else 0.0
            task_items.append(
                {
                    "task": task,
                    "weighted_delta": weighted_delta,
                    "share_of_overall_delta": weighted_delta / overall_delta,
                    "top_value": first_task[metric],
                    "runner_up_value": second_task[metric],
                }
            )
        output.append(
            {
                "metric": metric,
                "top_method": first["method"],
                "runner_up_method": second["method"],
                "overall_delta": overall_delta,
                "tasks": task_items,
            }
        )
    return output


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(ranks(left), ranks(right))


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _value in ordered[index:end]:
            result[original_index] = average_rank
        index = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if denom_left <= 0.0 or denom_right <= 0.0:
        return 0.0
    return numerator / (denom_left * denom_right)


def method_label_from_path(path: Path) -> str:
    return path.stem.replace("_details", "").replace("_", " ")


def task_sort_key(task: str) -> tuple[int, str]:
    order = ("overall", *TASK_TYPES)
    return (order.index(task), task) if task in order else (len(order), task)


def render_gold_audit_markdown(
    summary: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Gold Correctness Audit Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Samples: `{summary['total']}`",
        f"- Reviewed/evidence coverage: `{summary['overall']['coverage']:.3f}`",
        f"- Valid rate among reviewed: `{summary['overall']['valid_rate_reviewed']:.3f}`",
        f"- Valid-threshold pass: `{summary['passes_valid_threshold']}`",
        f"- Pending audit packet rows: `{len(packet_rows)}`",
        f"- Non-valid reviewed drop-list rows: `{len(drop_rows)}`",
        "",
        "## By Task",
        "",
        "| Task | Total | Reviewed | Coverage | Valid | Valid Rate | Noisy | Leaked | Ambiguous | Pending |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, row in summary["by_task"].items():
        counts = row["counts"]
        lines.append(
            f"| `{task}` | {row['total']} | {row['reviewed']} | {row['coverage']:.3f} | {row['valid']} | "
            f"{row['valid_rate_reviewed']:.3f} | {counts['noisy']} | {counts['leaked']} | {counts['ambiguous']} | {counts['pending']} |"
        )
    lines.extend(["", "## Evidence Basis", "", "| Basis | Samples |", "| --- | ---: |"])
    for basis, count in summary["basis_counts"].items():
        lines.append(f"| `{basis}` | {count} |")
    if packet_rows:
        lines.extend(["", "## Pending Audit Examples", "", "| Sample | Task | Repo | Gold Files |", "| --- | --- | --- | --- |"])
        for row in packet_rows[:12]:
            lines.append(
                f"| `{row['sample_id']}` | `{row['task_type']}` | `{row['repo']}` | "
                f"{escape_cell(', '.join(row['gold_files'][:3]))} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def render_leakage_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], clean_splits: list[dict[str, Any]]) -> str:
    lines = [
        "# Leakage Diagnostic",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Samples: `{summary['total']}`",
        f"- Flagged samples: `{summary['flagged_samples']}`",
        f"- Fatal flagged samples: `{summary['fatal_flagged_samples']}`",
        f"- Fatal-clean samples: `{summary['fatal_clean_samples']}`",
        "",
        "## By Task",
        "",
        "| Task | Total | Flagged | Fatal | Flagged Rate | Fatal Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, row in summary["by_task"].items():
        lines.append(
            f"| `{task}` | {row['total']} | {row['flagged_samples']} | {row['fatal_flagged_samples']} | "
            f"{row['flagged_rate']:.3f} | {row['fatal_rate']:.3f} |"
        )
    lines.extend(["", "## By Leakage Type", "", "| Type | Samples | Fatal Samples |", "| --- | ---: | ---: |"])
    for kind in LEAKAGE_TYPES:
        lines.append(f"| `{kind}` | {summary['by_type'].get(kind, 0)} | {summary['fatal_by_type'].get(kind, 0)} |")
    if summary["examples_by_type"]:
        lines.extend(["", "## Examples", ""])
        for kind, examples in summary["examples_by_type"].items():
            lines.extend([f"### `{kind}`", "", "| Sample | Task | Repo | Hits | Fatal |", "| --- | --- | --- | --- | ---: |"])
            for example in examples[:6]:
                lines.append(
                    f"| `{example['sample_id']}` | `{example['task_type']}` | `{example['repo']}` | "
                    f"{escape_cell(', '.join(example['hits']))} | `{example['fatal']}` |"
                )
            lines.append("")
    if clean_splits:
        lines.extend(["## Flagged vs Clean Split Files", "", "| Method | Clean n | Nonfatal n | Fatal n |", "| --- | ---: | ---: | ---: |"])
        for split in clean_splits[:20]:
            splits = split["splits"]
            lines.append(
                f"| {escape_cell(split['method'])} | {splits.get('clean', {}).get('samples', 0)} | "
                f"{splits.get('nonfatal_flagged', {}).get('samples', 0)} | {splits.get('fatal_flagged', {}).get('samples', 0)} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def render_task_diversity_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = report["leaderboard_rows"]
    macro = report["macro_average"]
    spread = report["spread"]
    lines = [
        "# Task Diversity Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Samples: `{summary['sample_distribution']['total']}`",
        f"- Leaderboard rows: `{summary['row_count']}`",
        f"- Methods: `{summary['method_count']}`",
        "",
        "## Sample Distribution",
        "",
        "| Task | Samples |",
        "| --- | ---: |",
    ]
    for task, count in summary["sample_distribution"]["by_task"].items():
        lines.append(f"| `{task}` | {count} |")
    lines.extend(["", "## Per-Task Spread", ""])
    for metric, per_task in spread.items():
        lines.extend([f"### `{metric}`", "", "| Task | Methods | Best | Best Method | Median | Worst | Spread | Best-Median |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"])
        for task, row in per_task.items():
            lines.append(
                f"| `{task}` | {row['methods']} | {row['best']:.4f} | {escape_cell(row['best_method'])} | "
                f"{row['median']:.4f} | {row['worst']:.4f} | {row['spread']:.4f} | {row['best_minus_median']:.4f} |"
            )
        lines.append("")
    if macro:
        lines.extend(["## Task Macro-Average", "", "| Method | Type | Top-k | Macro Final F1 | Overall Final F1 | Macro Gold@8k | Overall Gold@8k | Macro R@20 | Overall R@20 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in macro[:20]:
            lines.append(
                f"| {escape_cell(row['method'])} | `{row['method_type']}` | {row.get('top_k') or ''} | "
                f"{fmt(row.get('macro_final_file_f1'))} | {fmt(row.get('overall_final_file_f1'))} | "
                f"{fmt(row.get('macro_gold_coverage@8k'))} | {fmt(row.get('overall_gold_coverage@8k'))} | "
                f"{fmt(row.get('macro_Recall@20'))} | {fmt(row.get('overall_Recall@20'))} |"
            )
    lines.extend(["", "## Top Overall Rows", "", "| Task | Method | Type | Top-k | R@20 | nDCG@20 | Gold@8k | Final File F1 | Source |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in [item for item in rows if item["task"] == "overall"][:30]:
        lines.append(
            f"| `{row['task']}` | {escape_cell(row['method'])} | `{row['method_type']}` | {row.get('top_k') or ''} | "
            f"{fmt(row.get('Recall@20'))} | {fmt(row.get('nDCG@20'))} | {fmt(row.get('gold_coverage@8k'))} | "
            f"{fmt(row.get('final_file_f1'))} | `{Path(row['source']).name}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_combined_validity_markdown(
    combined: dict[str, Any],
    audit_result: dict[str, Any],
    leakage_result: dict[str, Any],
    diversity_result: dict[str, Any],
) -> str:
    audit = combined["gold_audit"]["overall"]
    leakage = combined["leakage"]
    diversity = combined["task_diversity"]
    lines = [
        "# Dataset Validity Report",
        "",
        f"- Generated at: `{combined['generated_at']}`",
        f"- Samples: `{combined['sample_count']}`",
        f"- Task counts: `{combined['task_counts']}`",
        "",
        "## Headline Checks",
        "",
        f"- Gold correctness evidence coverage: `{audit['coverage']:.3f}`; valid rate among reviewed/evidence-backed samples: `{audit['valid_rate_reviewed']:.3f}`.",
        f"- Leakage control: `{leakage['fatal_flagged_samples']}` fatal flagged samples; `{leakage['flagged_samples']}` total shortcut/hint flagged samples.",
        f"- Task diversity: `{diversity['row_count']}` per-task leaderboard rows across `{diversity['method_count']}` method variants; macro-average rows: `{diversity['macro_average_rows']}`.",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact_path in combined["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.extend([
        "",
        "## Paper-Ready Wording",
        "",
        "ARB reports gold-correctness evidence coverage and valid-rate statistics separately from schema/corpus validation, so human or review evidence is not conflated with automatic checks.",
        "Leakage diagnostics flag exact gold paths, test names, symbols, raw patches, fix commits, and trace stack hints; fatal shortcut samples can be excluded or reported as a clean-only robustness slice.",
        "Task diversity is reported with both sample-weighted overall metrics and task-macro averages over code2test, comment2context, and trace2code, plus per-task spread checks to show the benchmark is not driven by a single task family.",
    ])
    return "\n".join(lines).rstrip() + "\n"

def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
