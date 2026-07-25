from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .audit import query_excerpt, read_audit_rows, should_keep
from .baseline import given_files, is_test_path, query_has_leakage, query_text_for_eval, summarize_details, target_gold_files
from .corpus import sample_paths_from_derived
from .diagnostics import paths_for_pair, query_gold_hints
from .hardness import explicit_query_files, same_directory_gold
from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json
from .model_report import load_eval_summaries, normalize_summary, report_model_leaderboard
from .quality import validate_sample

V1_FROZEN_COUNTS = {"code2test": 106, "comment2context": 51, "trace2code": 68}
V1_FROZEN_FILE_FINGERPRINTS = {
    "data/benchmark/v1/code2test.jsonl": {
        "exists": True,
        "type": "file",
        "size_bytes": 236818,
        "sha256": "91bea4a2cbdeecbcd75d5a6558e3170b4be927784ca1ea1badcf0d93a4010cd8",
        "line_count": 106,
    },
    "data/benchmark/v1/comment2context.jsonl": {
        "exists": True,
        "type": "file",
        "size_bytes": 132036,
        "sha256": "77fd96d3fb805731b50679f0e80e6763ac3e214d9b1c697205c806996b2a45c0",
        "line_count": 51,
    },
    "data/benchmark/v1/samples.jsonl": {
        "exists": True,
        "type": "file",
        "size_bytes": 553215,
        "sha256": "7949ac26dc5fd1d626f2f4a51348c62df0cc9fc2dca14dc81fe7d3044986b01e",
        "line_count": 225,
    },
    "data/benchmark/v1/trace2code.jsonl": {
        "exists": True,
        "type": "file",
        "size_bytes": 184361,
        "sha256": "d9d1b4fa429a34e365f578632bd9a68cb2150affb29abeaf8702f7fb27a29b64",
        "line_count": 68,
    },
    "data/benchmark/v1/manifest.json": {
        "exists": True,
        "type": "file",
        "size_bytes": 954,
        "sha256": "9c0285c3bf174b3248ed8b12514b89616b803aefa86ab4de6cff9e1900413ce9",
        "line_count": 33,
    },
}
V1_1_TARGETS = {
    "comment2context": {"min": 80, "max": 100},
    "trace2code": {"min": 100},
}
V1_1_EXPANSION_TASKS = {"comment2context", "trace2code"}
REQUIRED_V1_1_BASELINES = (
    "lexical",
    "aider-style-repomap",
    "jina-code-embeddings-0.5b",
    "qwen3-embedding-4b",
)
V1_1_COMPLETION_OBJECTIVE = (
    "Build Agent Retrieval Bench V1.1 as a targeted expansion of frozen V1 with comment2context 80-100, "
    "trace2code 100+, quality/corpus/audit gates, required open-source baselines, regenerated reports, and docs."
)
V1_1_RELEASE_DOCS = (Path("README.md"), Path("PLAN.md"), Path("docs/v1_1_completion_audit.md"))
V1_1_DOC_CONTENT_MARKERS = (
    {
        "name": "names_v1_1",
        "description": "Docs name V1.1 explicitly.",
        "all_of": ("v1.1",),
    },
    {
        "name": "focused_improvement_over_v1",
        "description": "Docs present V1.1 as a focused improvement or expansion over V1.",
        "all_of": ("focused",),
        "any_of": ("improvement", "expansion"),
    },
    {
        "name": "frozen_v1_unchanged",
        "description": "Docs state that frozen benchmark/v1 stays unchanged.",
        "all_of": ("benchmark/v1",),
        "any_of": ("frozen v1", "keep `benchmark/v1` unchanged", "keep benchmark/v1 unchanged"),
    },
    {
        "name": "comment2context_expansion",
        "description": "Docs cover the comment2context expansion target.",
        "all_of": ("comment2context", "80"),
    },
    {
        "name": "trace2code_expansion",
        "description": "Docs cover the trace2code expansion target.",
        "all_of": ("trace2code", "100"),
    },
)
BASELINE_DETAIL_METRICS = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")
V1_1_TASKS = ("code2test", "comment2context", "trace2code")
COMMENT_SIGNAL_KEYS = {"review_comment", "comment"}
TRACE_SIGNAL_KEYS = {
    "error",
    "failure",
    "failure_excerpt",
    "failure_log",
    "failure_trace",
    "log_excerpt",
    "stderr",
    "trace",
    "traceback",
}
MODULE_ANCHOR_DIRS = {
    "app",
    "apps",
    "cmd",
    "core",
    "crates",
    "internal",
    "lib",
    "libs",
    "modules",
    "pkg",
    "packages",
    "server",
    "src",
    "test",
    "tests",
}
V1_1_AUDIT_FIELDS = (
    "sample_id",
    "task_type",
    "repo",
    "pr_url",
    "query_excerpt",
    "given_files",
    "gold_files",
    "gold_extensions",
    "trace_failure_types",
    "preflight_status",
    "preflight_reason",
    "verdict",
    "reason",
    "keep",
    "notes",
)


def assemble_v1_1_benchmark(
    base_sample_paths: Iterable[Path],
    expansion_sources: Iterable[Path],
    out_dir: Path,
    corpus_manifest_path: Path | None = None,
    require_corpus: bool = False,
    audit_paths: Iterable[Path] | None = None,
    require_audit_keep: bool = False,
) -> dict[str, Any]:
    if is_benchmark_v1_path(out_dir):
        raise ValueError("Refusing to write V1.1 outputs into benchmark/v1.")
    base_sample_paths = list(base_sample_paths)
    expansion_sources = list(expansion_sources)
    audit_paths = list(audit_paths or [])
    base_samples = load_samples(base_sample_paths)
    base_ids = {str(sample.get("id")) for sample in base_samples if sample.get("id")}
    corpus_by_pair = load_corpus_manifest_paths(corpus_manifest_path) if corpus_manifest_path else {}
    path_cache: dict[tuple[str, str], set[str]] = {}
    audit_keep_ids = load_audit_keep_ids(audit_paths)

    selected_expansion: list[dict[str, Any]] = []
    seen_expansion_ids: set[str] = set()
    dropped = Counter()
    source_counts = Counter()
    for source in expansion_sources:
        for sample_path in source_sample_paths(source):
            for sample in read_jsonl(sample_path):
                sample_id = str(sample.get("id", ""))
                drop_reason = expansion_drop_reason(
                    sample=sample,
                    base_ids=base_ids,
                    seen_expansion_ids=seen_expansion_ids,
                    corpus_by_pair=corpus_by_pair,
                    path_cache=path_cache,
                    require_corpus=require_corpus,
                    audit_keep_ids=audit_keep_ids,
                    require_audit_keep=require_audit_keep,
                )
                if sample_id:
                    seen_expansion_ids.add(sample_id)
                if drop_reason:
                    dropped[drop_reason] += 1
                    continue
                record = dict(sample)
                record.setdefault("metadata", {})
                if isinstance(record["metadata"], dict):
                    record["metadata"] = {**record["metadata"], "v1_1_expansion_source": str(sample_path)}
                selected_expansion.append(record)
                source_counts[str(sample_path)] += 1

    all_samples = sorted(
        base_samples + selected_expansion,
        key=lambda sample: (str(sample.get("task_type", "")), str(sample.get("repo", "")), str(sample.get("id", ""))),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"samples": str(out_dir / "samples.jsonl")}
    write_jsonl(out_dir / "samples.jsonl", all_samples)
    by_task = Counter(str(sample.get("task_type", "")) for sample in all_samples)
    new_by_task = Counter(str(sample.get("task_type", "")) for sample in selected_expansion)
    for task in V1_1_TASKS:
        task_rows = [sample for sample in all_samples if sample.get("task_type") == task]
        task_path = out_dir / f"{task}.jsonl"
        write_jsonl(task_path, task_rows)
        outputs[task] = str(task_path)

    manifest = {
        "generated_at": utc_now(),
        "base_sample_paths": [str(path) for path in base_sample_paths],
        "expansion_sources": [str(path) for path in expansion_sources],
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "require_corpus": require_corpus,
        "audit_paths": [str(path) for path in audit_paths],
        "require_audit_keep": require_audit_keep,
        "audit_keep_ids": len(audit_keep_ids),
        "total": len(all_samples),
        "base_total": len(base_samples),
        "accepted_expansion": len(selected_expansion),
        "counts_by_task": dict(sorted(by_task.items())),
        "new_counts_by_task": dict(sorted(new_by_task.items())),
        "dropped": dict(sorted(dropped.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "outputs": outputs,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def expansion_drop_reason(
    sample: dict[str, Any],
    base_ids: set[str],
    seen_expansion_ids: set[str],
    corpus_by_pair: dict[tuple[str, str], Path],
    path_cache: dict[tuple[str, str], set[str]],
    require_corpus: bool,
    audit_keep_ids: set[str] | None = None,
    require_audit_keep: bool = False,
) -> str | None:
    sample_id = str(sample.get("id", ""))
    if not sample_id:
        return "missing_id"
    if sample_id in base_ids:
        return "duplicate_base_id"
    if sample_id in seen_expansion_ids:
        return "duplicate_expansion_id"
    task_type = str(sample.get("task_type", ""))
    if task_type not in V1_1_EXPANSION_TASKS:
        return "excluded_task"
    if require_audit_keep and sample_id not in (audit_keep_ids or set()):
        return "audit_not_kept"
    gold_files = target_gold_files(sample)
    if not gold_files:
        return "missing_gold"
    validation_errors = validate_sample(sample)
    if validation_errors:
        return "schema_invalid"
    overlap_issues = sample_path_role_overlap_issues(sample)
    if overlap_issues:
        return sorted(overlap_issues)[0]
    if task_semantics_issues(sample):
        return "unclear_task_semantics"
    query_text = query_text_for_eval(sample)
    if query_has_leakage(sample, query_text):
        return "query_leakage"
    hints = query_gold_hints(query_text, gold_files)
    if hints["has_gold_path_hint"] or hints["has_gold_basename_hint"]:
        return "direct_gold_hint"
    if not sample_has_audit_evidence(sample):
        return "missing_audit_evidence"
    if task_type == "comment2context":
        if not given_files(sample):
            return "missing_given_files"
        reference_files = given_files(sample) + explicit_query_files(sample)
        if same_directory_gold(gold_files, reference_files):
            return "same_directory_gold"
    if task_type == "trace2code" and any(is_test_path(path) for path in gold_files):
        return "trace_gold_is_test"
    if require_corpus:
        repo = str(sample.get("repo", ""))
        base_commit = str(sample.get("base_commit", ""))
        corpus_paths = paths_for_pair(repo, base_commit, corpus_by_pair, path_cache)
        if not corpus_paths:
            return "missing_corpus_pair"
        if any(path not in corpus_paths for path in gold_files):
            return "missing_gold_in_corpus"
    return None


def write_v1_1_audit_packet(
    candidate_sources: Iterable[Path],
    out_dir: Path,
    base_sample_paths: Iterable[Path] | None = None,
    corpus_manifest_path: Path | None = None,
    require_corpus: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    candidate_sources = list(candidate_sources)
    base_samples = load_samples(base_sample_paths or [])
    base_ids = {str(sample.get("id")) for sample in base_samples if sample.get("id")}
    corpus_by_pair = load_corpus_manifest_paths(corpus_manifest_path) if corpus_manifest_path else {}
    path_cache: dict[tuple[str, str], set[str]] = {}
    seen_expansion_ids: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    preflight_counts = Counter()
    for source in candidate_sources:
        for sample_path in source_sample_paths(source):
            for sample in read_jsonl(sample_path):
                sample_id = str(sample.get("id", ""))
                reason = expansion_drop_reason(
                    sample=sample,
                    base_ids=base_ids,
                    seen_expansion_ids=seen_expansion_ids,
                    corpus_by_pair=corpus_by_pair,
                    path_cache=path_cache,
                    require_corpus=require_corpus,
                )
                if sample_id:
                    seen_expansion_ids.add(sample_id)
                status = "candidate" if reason is None else "drop"
                preflight_counts[reason or "candidate"] += 1
                row = audit_packet_row(sample, status=status, reason=reason or "", source_path=sample_path)
                diagnostics.append(row)
                if reason is None:
                    audit_rows.append({field: str(row.get(field, "")) for field in V1_1_AUDIT_FIELDS})
                    if limit and len(audit_rows) >= limit:
                        break
            if limit and len(audit_rows) >= limit:
                break
        if limit and len(audit_rows) >= limit:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "candidate_diagnostics.jsonl"
    audit_jsonl_path = out_dir / "audit_samples.jsonl"
    audit_csv_path = out_dir / "audit_samples.csv"
    write_jsonl(diagnostics_path, diagnostics)
    write_jsonl(audit_jsonl_path, audit_rows)
    write_audit_csv(audit_csv_path, audit_rows)
    summary = {
        "generated_at": utc_now(),
        "candidate_sources": [str(path) for path in candidate_sources],
        "base_sample_paths": [str(path) for path in (base_sample_paths or [])],
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "require_corpus": require_corpus,
        "diagnostics": len(diagnostics),
        "audit_rows": len(audit_rows),
        "preflight_counts": dict(sorted(preflight_counts.items())),
        "outputs": {
            "diagnostics": str(diagnostics_path),
            "jsonl": str(audit_jsonl_path),
            "csv": str(audit_csv_path),
        },
    }
    write_json(out_dir / "audit_packet_summary.json", summary)
    return summary


def audit_packet_row(sample: dict[str, Any], status: str, reason: str, source_path: Path) -> dict[str, Any]:
    gold_files = target_gold_files(sample)
    query_text = query_text_for_eval(sample)
    task_type = str(sample.get("task_type", ""))
    trace_text = trace_signal_text(sample) if task_type == "trace2code" else ""
    return {
        "sample_id": str(sample.get("id", "")),
        "task_type": task_type,
        "repo": str(sample.get("repo", "")),
        "base_commit": str(sample.get("base_commit", "")),
        "source_path": str(source_path),
        "pr_url": ((sample.get("metadata") or {}).get("pr_url") or ""),
        "query_excerpt": query_excerpt(sample, limit=1600),
        "given_files": "; ".join(given_files(sample)),
        "gold_files": "; ".join(gold_files),
        "gold_extensions": "; ".join(sorted(path_extension(path) for path in gold_files)),
        "trace_failure_types": "; ".join(trace_failure_types(trace_text)) if task_type == "trace2code" else "",
        "preflight_status": status,
        "preflight_reason": reason,
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "",
    }


def check_v1_1_readiness(
    sample_paths: Iterable[Path],
    corpus_manifest_path: Path,
    base_sample_paths: Iterable[Path] | None = None,
    assembly_manifest_path: Path | None = None,
    eval_dir: Path | None = None,
    leaderboard_path: Path | None = None,
    leaderboard_json_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    min_comment2context: int = 80,
    max_comment2context: int = 100,
    min_comment_cross_module: int = 1,
    min_trace2code: int = 100,
    min_trace_non_go_repos: int = 1,
    min_trace_languages: int = 2,
    min_trace_failure_types: int = 2,
) -> dict[str, Any]:
    sample_paths = list(sample_paths)
    base_sample_paths = list(base_sample_paths or [])
    samples = load_samples(sample_paths)
    base_samples = load_samples(base_sample_paths)
    base_ids = {str(sample.get("id")) for sample in base_samples if sample.get("id")}
    corpus_by_pair = load_corpus_manifest_paths(corpus_manifest_path)
    path_cache: dict[tuple[str, str], set[str]] = {}

    rows = [
        readiness_row(
            sample=sample,
            base_ids=base_ids,
            corpus_by_pair=corpus_by_pair,
            path_cache=path_cache,
        )
        for sample in samples
    ]
    summary = summarize_readiness(
        rows=rows,
        samples=samples,
        base_samples=base_samples,
        sample_paths=sample_paths,
        base_sample_paths=base_sample_paths,
        corpus_manifest_path=corpus_manifest_path,
        corpus_manifest_exists=corpus_manifest_path.exists(),
        assembly_manifest_path=assembly_manifest_path,
        min_comment2context=min_comment2context,
        max_comment2context=max_comment2context,
        min_comment_cross_module=min_comment_cross_module,
        min_trace2code=min_trace2code,
        min_trace_non_go_repos=min_trace_non_go_repos,
        min_trace_languages=min_trace_languages,
        min_trace_failure_types=min_trace_failure_types,
        eval_dir=eval_dir,
        leaderboard_path=leaderboard_path,
        leaderboard_json_path=leaderboard_json_path,
    )
    result = {
        "summary": summary,
        "sample_diagnostics": rows,
        "ready": summary["ready"],
        "blocking_gates": [name for name, passed in summary["gates"].items() if not passed],
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_readiness_markdown(summary, rows), encoding="utf-8")
    return result


def readiness_row(
    sample: dict[str, Any],
    base_ids: set[str],
    corpus_by_pair: dict[tuple[str, str], Path],
    path_cache: dict[tuple[str, str], set[str]],
) -> dict[str, Any]:
    sample_id = str(sample.get("id", ""))
    task_type = str(sample.get("task_type", ""))
    repo = str(sample.get("repo", ""))
    base_commit = str(sample.get("base_commit", ""))
    gold_files = target_gold_files(sample)
    reference_files = given_files(sample) + explicit_query_files(sample)
    query_text = query_text_for_eval(sample)
    trace_text = trace_signal_text(sample) if task_type == "trace2code" else ""
    corpus_paths = paths_for_pair(repo, base_commit, corpus_by_pair, path_cache)
    missing_gold_files = [path for path in gold_files if path not in corpus_paths]
    hints = query_gold_hints(query_text, gold_files)
    validation_errors = validate_sample(sample)
    semantic_issues = task_semantics_issues(sample)
    trace_gold_test_files = [path for path in gold_files if task_type == "trace2code" and is_test_path(path)]
    same_directory = same_directory_gold(gold_files, reference_files)
    cross_module_files = cross_module_gold_files(gold_files, reference_files) if task_type == "comment2context" else []
    path_role_overlap_issues = sample_path_role_overlap_issues(sample)

    issues = []
    if validation_errors:
        issues.append("schema")
    issues.extend(path_role_overlap_issues)
    if semantic_issues:
        issues.append("unclear_task_semantics")
    if missing_gold_files:
        issues.append("missing_gold")
    if query_has_leakage(sample, query_text):
        issues.append("query_leakage")
    if hints["has_gold_path_hint"]:
        issues.append("direct_gold_path_hint")
    if hints["has_gold_basename_hint"]:
        issues.append("gold_basename_hint")
    if not sample_has_audit_evidence(sample):
        issues.append("missing_audit_evidence")
    if task_type == "comment2context":
        if not given_files(sample):
            issues.append("missing_given_files")
        if same_directory:
            issues.append("same_directory_gold")
    if trace_gold_test_files:
        issues.append("trace_gold_is_test")

    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "repo": repo,
        "base_commit": base_commit,
        "is_new": sample_id not in base_ids if base_ids else True,
        "gold_files": gold_files,
        "given_files": given_files(sample),
        "query_files": explicit_query_files(sample),
        "gold_extensions": sorted(path_extension(path) for path in gold_files),
        "validation_errors": validation_errors,
        "task_semantics_issues": semantic_issues,
        "path_role_overlap_issues": path_role_overlap_issues,
        "gold_in_corpus": not missing_gold_files,
        "missing_gold_files": missing_gold_files,
        "query_leakage": query_has_leakage(sample, query_text),
        "direct_path_hint": hints["has_gold_path_hint"],
        "basename_hint": hints["has_gold_basename_hint"],
        "same_directory_gold": bool(same_directory),
        "same_directory_gold_files": same_directory,
        "cross_module_gold_files": cross_module_files,
        "trace_gold_test_files": trace_gold_test_files,
        "trace_failure_types": trace_failure_types(trace_text) if task_type == "trace2code" else [],
        "has_audit_evidence": sample_has_audit_evidence(sample),
        "issues": issues,
        "pr_url": ((sample.get("metadata") or {}).get("pr_url") or ""),
    }


def summarize_readiness(
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    base_samples: list[dict[str, Any]],
    sample_paths: list[Path],
    base_sample_paths: list[Path],
    corpus_manifest_path: Path,
    corpus_manifest_exists: bool,
    assembly_manifest_path: Path | None,
    min_comment2context: int,
    max_comment2context: int,
    min_comment_cross_module: int,
    min_trace2code: int,
    min_trace_non_go_repos: int,
    min_trace_languages: int,
    min_trace_failure_types: int,
    eval_dir: Path | None = None,
    leaderboard_path: Path | None = None,
    leaderboard_json_path: Path | None = None,
) -> dict[str, Any]:
    counts = Counter(row["task_type"] for row in rows)
    base_counts = Counter(str(sample.get("task_type", "")) for sample in base_samples)
    base_ids = {str(sample.get("id")) for sample in base_samples if sample.get("id")}
    sample_ids = {str(sample.get("id")) for sample in samples if sample.get("id")}
    sample_id_counts = Counter(str(sample.get("id")) for sample in samples if sample.get("id"))
    duplicate_sample_ids = sorted(sample_id for sample_id, count in sample_id_counts.items() if count > 1)
    missing_base_ids = sorted(base_ids - sample_ids)
    new_rows = [row for row in rows if row["is_new"]]
    new_by_task = Counter(row["task_type"] for row in new_rows)
    new_comment_rows = [row for row in new_rows if row["task_type"] == "comment2context"]
    new_comment_cross_module_rows = [row for row in new_comment_rows if row["cross_module_gold_files"]]
    new_trace_rows = [row for row in new_rows if row["task_type"] == "trace2code"]
    new_trace_non_go_rows = [
        row for row in new_trace_rows if any(extension != ".go" for extension in row["gold_extensions"])
    ]
    new_trace_extensions = sorted({extension for row in new_trace_rows for extension in row["gold_extensions"]})
    new_trace_non_go_repos = sorted({row["repo"] for row in new_trace_non_go_rows})
    new_trace_failure_types = sorted(
        {signal for row in new_trace_rows for signal in row["trace_failure_types"] if signal != "unknown"}
    )
    new_trace_unknown_failure_rows = [row for row in new_trace_rows if "unknown" in row["trace_failure_types"]]
    target_gaps = {
        "comment2context_samples": max(0, min_comment2context - counts.get("comment2context", 0)),
        "comment2context_cross_module_samples": max(0, min_comment_cross_module - len(new_comment_cross_module_rows)),
        "trace2code_samples": max(0, min_trace2code - counts.get("trace2code", 0)),
        "trace2code_non_go_repos": max(0, min_trace_non_go_repos - len(new_trace_non_go_repos)),
        "trace2code_languages": max(0, min_trace_languages - len(new_trace_extensions)),
        "trace2code_failure_types": max(0, min_trace_failure_types - len(new_trace_failure_types)),
    }
    issues_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for issue in row["issues"]:
            issues_by_type[issue][row["task_type"]] += 1
    baseline_status = baseline_readiness(eval_dir, len(rows), sample_ids) if eval_dir else []
    assembly_manifest = read_json(assembly_manifest_path, {}) if assembly_manifest_path and assembly_manifest_path.exists() else {}
    leaderboard_status = leaderboard_readiness(
        leaderboard_path=leaderboard_path,
        leaderboard_json_path=leaderboard_json_path,
        expected_baselines=REQUIRED_V1_1_BASELINES,
    )

    full_v1_compatible = bool(base_samples)
    if not full_v1_compatible:
        base_counts = Counter(V1_FROZEN_COUNTS)

    gates = {
        "corpus_manifest_exists": corpus_manifest_exists,
        "sample_paths_do_not_point_at_benchmark_v1": all(not is_benchmark_v1_path(path) for path in sample_paths),
        "sample_ids_unique": not duplicate_sample_ids,
        "contains_all_base_v1_ids": not full_v1_compatible or not missing_base_ids,
        "code2test_count_preserved": counts.get("code2test", 0) == base_counts.get("code2test", 0),
        "comment2context_count_ge_target": counts.get("comment2context", 0) >= min_comment2context,
        "comment2context_count_le_target_band": counts.get("comment2context", 0) <= max_comment2context,
        "trace2code_count_ge_target": counts.get("trace2code", 0) >= min_trace2code,
        "new_samples_are_target_tasks": all(row["task_type"] in V1_1_EXPANSION_TASKS for row in new_rows),
        "new_samples_schema_valid": all(not row["validation_errors"] for row in new_rows),
        "new_samples_no_path_role_overlap": all(not row["path_role_overlap_issues"] for row in new_rows),
        "new_samples_have_clear_task_semantics": all(not row["task_semantics_issues"] for row in new_rows),
        "new_samples_gold_in_corpus": all(row["gold_in_corpus"] for row in new_rows),
        "new_samples_no_query_leakage": all(not row["query_leakage"] for row in new_rows),
        "new_samples_no_direct_gold_hints": all(not row["direct_path_hint"] and not row["basename_hint"] for row in new_rows),
        "new_samples_have_audit_evidence": all(row["has_audit_evidence"] for row in new_rows),
        "new_comment2context_have_given_files": all(
            row["given_files"] for row in new_rows if row["task_type"] == "comment2context"
        ),
        "new_comment2context_no_same_directory_gold": all(
            not row["same_directory_gold"] for row in new_rows if row["task_type"] == "comment2context"
        ),
        "new_comment2context_cross_module_count_ge_min": len(new_comment_cross_module_rows) >= min_comment_cross_module,
        "new_trace2code_gold_not_tests": all(
            not row["trace_gold_test_files"] for row in new_rows if row["task_type"] == "trace2code"
        ),
        "new_trace2code_has_non_go_gold": bool(new_trace_non_go_rows),
        "new_trace2code_non_go_repo_count_ge_min": len(new_trace_non_go_repos) >= min_trace_non_go_repos,
        "new_trace2code_language_count_ge_min": len(new_trace_extensions) >= min_trace_languages,
        "new_trace2code_failure_type_count_ge_min": len(new_trace_failure_types) >= min_trace_failure_types,
    }
    if eval_dir:
        gates["eval_dir_exists"] = eval_dir.exists()
        gates["required_baseline_summaries_complete"] = all(item["complete"] for item in baseline_status)
    if assembly_manifest_path:
        gates["assembly_manifest_exists"] = assembly_manifest_path.exists()
        gates["assembly_manifest_requires_audit_keep"] = bool(assembly_manifest.get("require_audit_keep"))
        gates["assembly_manifest_has_audit_paths"] = bool(assembly_manifest.get("audit_paths"))
    if leaderboard_path or leaderboard_json_path:
        gates["leaderboard_reports_exist"] = leaderboard_status["reports_exist"]
        gates["leaderboard_reports_contain_required_baselines"] = leaderboard_status["contains_required_baselines"]
    ready = all(gates.values())

    return {
        "generated_at": utc_now(),
        "ready": ready,
        "inputs": {
            "samples": [str(path) for path in sample_paths],
            "base_samples": [str(path) for path in base_sample_paths],
            "corpus_manifest": str(corpus_manifest_path),
            "assembly_manifest": str(assembly_manifest_path) if assembly_manifest_path else None,
            "eval_dir": str(eval_dir) if eval_dir else None,
            "leaderboard": str(leaderboard_path) if leaderboard_path else None,
            "leaderboard_json": str(leaderboard_json_path) if leaderboard_json_path else None,
        },
        "targets": {
            "comment2context": {
                "min": min_comment2context,
                "max": max_comment2context,
                "min_new_cross_module": min_comment_cross_module,
            },
            "trace2code": {
                "min": min_trace2code,
                "min_non_go_repos": min_trace_non_go_repos,
                "min_languages": min_trace_languages,
                "min_failure_types": min_trace_failure_types,
            },
        },
        "samples": len(rows),
        "new_samples": len(new_rows),
        "counts_by_task": dict(sorted(counts.items())),
        "base_counts_by_task": dict(sorted(base_counts.items())),
        "new_counts_by_task": dict(sorted(new_by_task.items())),
        "comment2context_new_cross_module_samples": len(new_comment_cross_module_rows),
        "missing_base_ids": missing_base_ids,
        "duplicate_sample_ids": duplicate_sample_ids,
        "issues_by_type": {issue: dict(sorted(counts.items())) for issue, counts in sorted(issues_by_type.items())},
        "trace2code_new_gold_extensions": new_trace_extensions,
        "trace2code_new_non_go_samples": len(new_trace_non_go_rows),
        "trace2code_new_non_go_repos": new_trace_non_go_repos,
        "trace2code_new_failure_types": new_trace_failure_types,
        "trace2code_new_unknown_failure_samples": len(new_trace_unknown_failure_rows),
        "target_gaps": target_gaps,
        "assembly_manifest": {
            "path": str(assembly_manifest_path) if assembly_manifest_path else None,
            "exists": assembly_manifest_path.exists() if assembly_manifest_path else None,
            "require_audit_keep": assembly_manifest.get("require_audit_keep"),
            "audit_paths": assembly_manifest.get("audit_paths") or [],
            "accepted_expansion": assembly_manifest.get("accepted_expansion"),
        },
        "required_baselines": baseline_status,
        "leaderboard": leaderboard_status,
        "gates": gates,
    }


def baseline_readiness(
    eval_dir: Path,
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    summaries = load_eval_summaries(eval_dir) if eval_dir.exists() else []
    return baseline_status_from_summaries(summaries, expected_samples, expected_sample_ids=expected_sample_ids)


def baseline_status_from_summaries(
    summaries: Iterable[dict[str, Any]],
    expected_samples: int,
    expected_baselines: Iterable[str] | None = REQUIRED_V1_1_BASELINES,
    require_details: bool = True,
    expected_sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    summaries = list(summaries)
    status = []
    for expected in expected_baselines or REQUIRED_V1_1_BASELINES:
        candidates = [
            summary
            for summary in summaries
            if summary.get("candidate_filter") == "all_files" and baseline_label_matches(str(summary.get("model_label", "")), expected)
        ]
        summary = candidates[0] if candidates else None
        skipped = summary.get("skipped", {}) if summary else {}
        overall = (summary.get("metrics", {}) if summary else {}).get("overall", {}) if summary else {}
        evaluated = int(summary.get("evaluated") or 0) if summary else 0
        overall_samples = int(overall.get("samples") or 0) if overall else 0
        complete = bool(summary) and evaluated == expected_samples and overall_samples == expected_samples and not skipped
        details = (
            baseline_details_status(
                Path(str(summary["path"])),
                expected_samples,
                expected_sample_ids=expected_sample_ids,
                summary_metrics=summary.get("metrics") or {},
            )
            if require_details and summary
            else empty_baseline_details_status()
        )
        if require_details:
            complete = bool(complete and details["complete"])
        item = {
            "baseline": expected,
            "found": bool(summary),
            "source": summary.get("path") if summary else None,
            "evaluated": evaluated,
            "overall_samples": overall_samples,
            "skipped": skipped,
            "complete": complete,
        }
        if require_details:
            item["details"] = details
        status.append(item)
    return status


def check_required_baseline_summaries(
    summary_paths: Iterable[Path],
    expected_samples: int,
    expected_baselines: Iterable[str] | None = REQUIRED_V1_1_BASELINES,
    require_details: bool = True,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    invalid_summaries: list[dict[str, Any]] = []
    paths = list(summary_paths)
    for path in paths:
        if not path.exists():
            invalid_summaries.append({"path": str(path), "reason": "missing_file"})
            continue
        payload = read_json(path, {})
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            invalid_summaries.append({"path": str(path), "reason": "missing_metrics"})
            continue
        summaries.append(normalize_summary(path, payload))
    baseline_status = baseline_status_from_summaries(
        summaries,
        expected_samples=expected_samples,
        expected_baselines=expected_baselines,
        require_details=require_details,
        expected_sample_ids=expected_sample_ids,
    )
    blocking = [item["baseline"] for item in baseline_status if not item["complete"]]
    return {
        "generated_at": utc_now(),
        "expected_samples": expected_samples,
        "expected_sample_ids_count": len(expected_sample_ids) if expected_sample_ids is not None else None,
        "summary_paths": [str(path) for path in paths],
        "invalid_summaries": invalid_summaries,
        "require_details": require_details,
        "required_baselines": baseline_status,
        "complete": not invalid_summaries and not blocking,
        "blocking_baselines": blocking,
    }


def check_v1_1_baseline_status(
    summary_paths: Iterable[Path],
    expected_samples: int,
    expected_baselines: Iterable[str] | None = REQUIRED_V1_1_BASELINES,
    require_details: bool = True,
    expected_sample_ids: set[str] | None = None,
    runtime_status: dict[str, Any] | None = None,
    eval_dirs: Iterable[Path] | None = None,
    shard_commands_path: Path | None = None,
) -> dict[str, Any]:
    preflight = check_required_baseline_summaries(
        summary_paths=summary_paths,
        expected_samples=expected_samples,
        expected_baselines=expected_baselines,
        require_details=require_details,
        expected_sample_ids=expected_sample_ids,
    )
    runtime = runtime_status or embedding_runtime_status()
    partial_details = baseline_partial_details_by_label(
        expected_baselines or REQUIRED_V1_1_BASELINES,
        eval_dirs or [],
        expected_samples=expected_samples,
        expected_sample_ids=expected_sample_ids,
    )
    partial_details = merge_preflight_details_into_partial_details(preflight["required_baselines"], partial_details)
    shard_details = baseline_shard_details_by_label(
        shard_commands_path,
        expected_samples=expected_samples,
        expected_sample_ids=expected_sample_ids,
    )
    blockers = classify_required_baseline_blockers(preflight["required_baselines"], runtime, partial_details, shard_details)
    return {
        "generated_at": utc_now(),
        "expected_samples": expected_samples,
        "expected_sample_ids_count": len(expected_sample_ids) if expected_sample_ids is not None else None,
        "complete": bool(preflight["complete"]),
        "preflight": preflight,
        "runtime": runtime,
        "partial_details": partial_details,
        "shard_commands": str(shard_commands_path) if shard_commands_path else None,
        "shard_details": shard_details,
        "baseline_blockers": blockers,
        "blocking_baselines": [item["baseline"] for item in blockers if not item["complete"]],
    }


def write_v1_1_baseline_status_report(
    summary_paths: Iterable[Path],
    expected_samples: int,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    expected_baselines: Iterable[str] | None = REQUIRED_V1_1_BASELINES,
    require_details: bool = True,
    expected_sample_ids: set[str] | None = None,
    eval_dirs: Iterable[Path] | None = None,
    shard_commands_path: Path | None = None,
) -> dict[str, Any]:
    report = check_v1_1_baseline_status(
        summary_paths=summary_paths,
        expected_samples=expected_samples,
        expected_baselines=expected_baselines,
        require_details=require_details,
        expected_sample_ids=expected_sample_ids,
        eval_dirs=eval_dirs,
        shard_commands_path=shard_commands_path,
    )
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_status_markdown(report), encoding="utf-8")
    return report


def embedding_runtime_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env if env is not None else dict(os.environ)
    numpy_installed = importlib.util.find_spec("numpy") is not None
    sentence_transformers_installed = importlib.util.find_spec("sentence_transformers") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    torch_status: dict[str, Any] = {
        "installed": torch_installed,
        "version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "error": None,
    }
    if torch_installed:
        try:
            import torch  # type: ignore[import-not-found]

            torch_status.update(
                {
                    "version": getattr(torch, "__version__", None),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                }
            )
        except Exception as exc:  # pragma: no cover - depends on optional local runtime
            torch_status["error"] = str(exc)

    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_status: dict[str, Any] = {"available": bool(nvidia_smi_path), "path": nvidia_smi_path, "gpus": [], "error": None}
    if nvidia_smi_path:
        try:
            result = subprocess.run(
                [nvidia_smi_path, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            if result.returncode == 0:
                nvidia_status["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            else:
                nvidia_status["error"] = result.stderr.strip() or f"exit {result.returncode}"
        except Exception as exc:  # pragma: no cover - depends on optional local runtime
            nvidia_status["error"] = str(exc)

    return {
        "numpy": {"installed": numpy_installed},
        "sentence_transformers": {"installed": sentence_transformers_installed},
        "torch": torch_status,
        "nvidia_smi": nvidia_status,
        "cuda_available": bool(torch_status.get("cuda_available")) or bool(nvidia_status.get("gpus")),
        "voyage_api_key_set": bool(env.get("VOYAGE_API_KEY")),
    }


def baseline_partial_details_by_label(
    baselines: Iterable[str],
    eval_dirs: Iterable[Path],
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    eval_dirs = list(eval_dirs)
    return {
        baseline: partial_details_status_for_baseline(
            baseline,
            eval_dirs,
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
        )
        for baseline in baselines
    }


def merge_preflight_details_into_partial_details(
    baseline_statuses: Iterable[dict[str, Any]],
    partial_details: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(partial_details)
    for status in baseline_statuses:
        baseline = str(status.get("baseline") or "")
        details = status.get("details") or {}
        if not baseline or not details.get("path"):
            continue
        if details.get("exists"):
            merged[baseline] = preflight_details_as_partial_details(details)
    return merged


def preflight_details_as_partial_details(details: dict[str, Any]) -> dict[str, Any]:
    missing_sample_ids = list(details.get("missing_sample_ids") or [])
    unexpected_sample_ids = list(details.get("unexpected_sample_ids") or [])
    max_reported_ids = 20
    return {
        "path": details.get("path"),
        "exists": bool(details.get("exists")),
        "rows": int(details.get("rows") or 0),
        "unique_sample_ids": int(details.get("unique_sample_ids") or 0),
        "duplicate_sample_ids": list(details.get("duplicate_sample_ids") or []),
        "missing_sample_id_rows": int(details.get("missing_sample_id_rows") or 0),
        "missing_sample_ids_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids[:max_reported_ids],
        "unexpected_sample_ids_count": len(unexpected_sample_ids),
        "unexpected_sample_ids": unexpected_sample_ids[:max_reported_ids],
        "sample_id_lists_truncated": len(missing_sample_ids) > max_reported_ids
        or len(unexpected_sample_ids) > max_reported_ids,
        "candidate_filters": list(details.get("candidate_filters") or []),
        "complete_by_rows": bool(details.get("complete")),
    }


def partial_details_status_for_baseline(
    baseline: str,
    eval_dirs: list[Path],
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    paths = [candidate_details_path(eval_dir, baseline) for eval_dir in eval_dirs]
    existing = [path for path in paths if path.exists()]
    path = existing[0] if existing else (paths[0] if paths else None)
    rows = read_jsonl(path) if path and path.exists() else []
    sample_ids = [str(row.get("sample_id")) for row in rows if row.get("sample_id")]
    actual_sample_ids = set(sample_ids)
    duplicate_sample_ids = sorted({sample_id for sample_id, count in Counter(sample_ids).items() if count > 1})
    candidate_filters = sorted({str(row.get("candidate_filter") or "") for row in rows if row.get("candidate_filter")})
    missing_sample_ids = sorted((expected_sample_ids or set()) - actual_sample_ids) if expected_sample_ids is not None else []
    unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids) if expected_sample_ids is not None else []
    max_reported_ids = 20
    return {
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "rows": len(rows),
        "unique_sample_ids": len(actual_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "missing_sample_id_rows": len(rows) - len(sample_ids),
        "missing_sample_ids_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids[:max_reported_ids],
        "unexpected_sample_ids_count": len(unexpected_sample_ids),
        "unexpected_sample_ids": unexpected_sample_ids[:max_reported_ids],
        "sample_id_lists_truncated": len(missing_sample_ids) > max_reported_ids or len(unexpected_sample_ids) > max_reported_ids,
        "candidate_filters": candidate_filters,
        "complete_by_rows": bool(
            path
            and path.exists()
            and len(rows) == expected_samples
            and len(actual_sample_ids) == expected_samples
            and len(sample_ids) == len(rows)
            and not duplicate_sample_ids
            and not missing_sample_ids
            and not unexpected_sample_ids
            and candidate_filters == ["all_files"]
        ),
    }


def baseline_shard_details_by_label(
    shard_commands_path: Path | None,
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not shard_commands_path or not shard_commands_path.exists():
        return {}
    payload = read_json(shard_commands_path, {})
    by_baseline: dict[str, dict[str, Any]] = {}
    for baseline_report in payload.get("baselines") or []:
        baseline = str(baseline_report.get("baseline") or "")
        if not baseline:
            continue
        detail_paths = [
            Path(str(((shard.get("artifacts") or {}).get("details"))))
            for shard in baseline_report.get("shard_commands") or []
            if (shard.get("artifacts") or {}).get("details")
        ]
        by_baseline[baseline] = shard_details_status_for_baseline(
            detail_paths,
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
        )
    return by_baseline


def shard_details_status_for_baseline(
    details_paths: Iterable[Path],
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    paths = list(details_paths)
    shards: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    candidate_filters: set[str] = set()
    missing_sample_id_rows = 0
    missing_candidate_filter_rows = 0
    malformed_metric_rows = 0
    for path in paths:
        rows = read_jsonl(path) if path.exists() else []
        shard_sample_ids = [str(row.get("sample_id")) for row in rows if row.get("sample_id")]
        sample_ids.extend(shard_sample_ids)
        missing_sample_id_rows += len(rows) - len(shard_sample_ids)
        candidate_filters.update(str(row.get("candidate_filter") or "") for row in rows if row.get("candidate_filter"))
        missing_candidate_filter_rows += sum(1 for row in rows if not row.get("candidate_filter"))
        for row in rows:
            metrics = row.get("metrics")
            missing_metric_keys = (
                list(BASELINE_DETAIL_METRICS)
                if not isinstance(metrics, dict)
                else [key for key in BASELINE_DETAIL_METRICS if key not in metrics]
            )
            if not row.get("task_type") or missing_metric_keys:
                malformed_metric_rows += 1
        shards.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "rows": len(rows),
                "unique_sample_ids": len(set(shard_sample_ids)),
                "missing_sample_id_rows": len(rows) - len(shard_sample_ids),
                "missing_candidate_filter_rows": sum(1 for row in rows if not row.get("candidate_filter")),
            }
        )

    actual_sample_ids = set(sample_ids)
    duplicate_sample_ids = sorted({sample_id for sample_id, count in Counter(sample_ids).items() if count > 1})
    missing_sample_ids = sorted((expected_sample_ids or set()) - actual_sample_ids) if expected_sample_ids is not None else []
    unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids) if expected_sample_ids is not None else []
    missing_shard_files = [str(path) for path in paths if not path.exists()]
    max_reported_ids = 20
    complete_by_rows = bool(
        paths
        and not missing_shard_files
        and len(sample_ids) == expected_samples
        and len(actual_sample_ids) == expected_samples
        and not duplicate_sample_ids
        and not missing_sample_ids
        and not unexpected_sample_ids
        and not missing_sample_id_rows
        and not missing_candidate_filter_rows
        and not malformed_metric_rows
        and sorted(candidate_filters) == ["all_files"]
    )
    return {
        "paths": [str(path) for path in paths],
        "shard_count": len(paths),
        "shards": shards,
        "missing_shard_files": missing_shard_files,
        "rows": len(sample_ids) + missing_sample_id_rows,
        "unique_sample_ids": len(actual_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "missing_sample_id_rows": missing_sample_id_rows,
        "missing_candidate_filter_rows": missing_candidate_filter_rows,
        "malformed_metric_rows": malformed_metric_rows,
        "missing_sample_ids_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids[:max_reported_ids],
        "unexpected_sample_ids_count": len(unexpected_sample_ids),
        "unexpected_sample_ids": unexpected_sample_ids[:max_reported_ids],
        "sample_id_lists_truncated": len(missing_sample_ids) > max_reported_ids or len(unexpected_sample_ids) > max_reported_ids,
        "candidate_filters": sorted(candidate_filters),
        "complete_by_rows": complete_by_rows,
    }


def candidate_details_path(eval_dir: Path, baseline: str) -> Path:
    stems = {
        "lexical": "lexical",
        "aider-style-repomap": "repomap",
        "jina-code-embeddings-0.5b": "jina-code-embeddings-0.5b",
        "qwen3-embedding-4b": "qwen3-embedding-4b",
        "voyage-code-3": "voyage-code-3",
    }
    return eval_dir / f"{stems.get(baseline, baseline)}_details.jsonl"


def classify_required_baseline_blockers(
    required_baselines: Iterable[dict[str, Any]],
    runtime: dict[str, Any],
    partial_details: dict[str, dict[str, Any]] | None = None,
    shard_details: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for item in required_baselines:
        baseline = str(item.get("baseline") or "")
        complete = bool(item.get("complete"))
        reason = None
        action = None
        partial = (partial_details or {}).get(baseline, {})
        shards = (shard_details or {}).get(baseline, {})
        if not complete:
            reason, action = baseline_blocker_reason(baseline, item, runtime, partial, shards)
        blockers.append(
            {
                "baseline": baseline,
                "complete": complete,
                "found": bool(item.get("found")),
                "details_complete": bool((item.get("details") or {}).get("complete")),
                "partial_details": partial,
                "shard_details": shards,
                "reason": reason,
                "next_action": action,
            }
        )
    return blockers


def baseline_blocker_reason(
    baseline: str,
    status: dict[str, Any],
    runtime: dict[str, Any],
    partial_details: dict[str, Any] | None = None,
    shard_details: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if not status.get("found"):
        artifact_reason = "missing_summary"
    elif not (status.get("details") or {}).get("complete"):
        artifact_reason = "missing_or_invalid_details"
    else:
        artifact_reason = "incomplete_summary"

    if not status.get("found") and (partial_details or {}).get("complete_by_rows"):
        details_path = (partial_details or {}).get("path")
        return (
            "missing_summary_from_complete_details",
            f"Regenerate the summary from complete details with v1-1-summary-from-details --details {details_path} --model {baseline}.",
        )
    if (shard_details or {}).get("complete_by_rows"):
        return (
            "ready_to_merge_shards",
            f"Run v1-1-merge-details for {baseline}, then v1-1-summary-from-details and rerun v1-1-baseline-status.",
        )
    if (shard_details or {}).get("rows"):
        return (
            "partial_shards_incomplete",
            f"Finish or repair the remaining shard details for {baseline}, then merge shards and rebuild the summary.",
        )

    if baseline == "voyage-code-3":
        if not runtime.get("voyage_api_key_set"):
            return (
                "blocked_missing_voyage_api_key",
                "Set VOYAGE_API_KEY on an API-approved machine and run the eval-voyage command from the V1.1 embedding runbook.",
            )
        if not (runtime.get("numpy") or {}).get("installed"):
            return (
                "blocked_missing_numpy",
                "Install numpy or the embedding extra, then run or resume the Voyage command from the V1.1 embedding runbook.",
            )
        return (
            artifact_reason,
            "Run or resume the voyage-code-3 all-files baseline, then rerun check-baseline-summaries.",
        )

    if baseline in {"jina-code-embeddings-0.5b", "qwen3-embedding-4b"}:
        if not (runtime.get("numpy") or {}).get("installed"):
            return (
                "blocked_missing_numpy",
                "Install numpy or the embedding extra, then run the eval-embedding command from the V1.1 embedding runbook.",
            )
        if not (runtime.get("sentence_transformers") or {}).get("installed"):
            return (
                "blocked_missing_sentence_transformers",
                "Install the embedding extra or run on a prepared baseline machine, then run the eval-embedding command from the V1.1 embedding runbook.",
            )
        if not (runtime.get("torch") or {}).get("installed"):
            return (
                "blocked_missing_torch",
                "Install torch with the appropriate accelerator support, then run the eval-embedding command from the V1.1 embedding runbook.",
            )
        if not runtime.get("cuda_available"):
            return (
                "blocked_no_cuda_or_precomputed_cache",
                "Run on a CUDA/precomputed-cache machine; CPU-only full V1.1 Jina/Qwen baselines are impractical for the current corpus.",
            )
        return (
            artifact_reason,
            f"Run or resume the {baseline} all-files baseline, then rerun check-baseline-summaries.",
        )

    return (
        artifact_reason,
        "Regenerate the required all-files summary/details artifacts, then rerun check-baseline-summaries.",
    )


def render_v1_1_baseline_status_markdown(report: dict[str, Any]) -> str:
    runtime = report.get("runtime") or {}
    torch_status = runtime.get("torch") or {}
    nvidia_status = runtime.get("nvidia_smi") or {}
    lines = [
        "# V1.1 Baseline Status",
        "",
        f"- Complete: `{bool(report.get('complete'))}`",
        f"- Blocking baselines: `{json.dumps(report.get('blocking_baselines') or [])}`",
        f"- Expected samples: `{report.get('expected_samples')}`",
        f"- Shard commands: `{report.get('shard_commands')}`",
        "",
        "## Runtime",
        "",
        f"- `numpy` installed: `{bool((runtime.get('numpy') or {}).get('installed'))}`",
        f"- `sentence_transformers` installed: `{bool((runtime.get('sentence_transformers') or {}).get('installed'))}`",
        f"- `torch` installed: `{bool(torch_status.get('installed'))}`",
        f"- `torch` version: `{torch_status.get('version')}`",
        f"- CUDA available: `{bool(runtime.get('cuda_available'))}`",
        f"- CUDA device count: `{int(torch_status.get('cuda_device_count') or 0)}`",
        f"- `nvidia-smi` available: `{bool(nvidia_status.get('available'))}`",
        f"- `VOYAGE_API_KEY` set: `{bool(runtime.get('voyage_api_key_set'))}`",
        "",
        "## Baselines",
        "",
        "| Baseline | Complete | Found | Details complete | Partial rows | Shard rows | Reason | Next action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.get("baseline_blockers") or []:
        partial = item.get("partial_details") or {}
        shards = item.get("shard_details") or {}
        lines.append(
            "| {baseline} | `{complete}` | `{found}` | `{details_complete}` | `{partial_rows}` | `{shard_rows}` | `{reason}` | {next_action} |".format(
                baseline=item.get("baseline"),
                complete=bool(item.get("complete")),
                found=bool(item.get("found")),
                details_complete=bool(item.get("details_complete")),
                partial_rows=partial.get("rows"),
                shard_rows=shards.get("rows"),
                reason=item.get("reason"),
                next_action=str(item.get("next_action") or "").replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_v1_1_external_runner_preflight_report(
    baseline_status_path: Path,
    return_acceptance_path: Path,
    return_manifest_path: Path,
    transfer_manifest_verify_path: Path | None = None,
    handoff_verify_path: Path | None = None,
    transfer_bundle_verify_path: Path | None = None,
    copy_packet_path: Path | None = None,
    full_runner_path: Path | None = None,
    gpu_runner_path: Path | None = None,
    voyage_runner_path: Path | None = None,
    return_bundle_script_path: Path | None = None,
    gpu_return_bundle_script_path: Path | None = None,
    voyage_return_bundle_script_path: Path | None = None,
    gpu_return_manifest_path: Path | None = None,
    voyage_return_manifest_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    baseline_status = read_json(baseline_status_path, {})
    return_acceptance = read_json(return_acceptance_path, {})
    return_manifest = read_json(return_manifest_path, {})
    baseline_status_source = "artifact" if baseline_status_path.is_file() and isinstance(baseline_status, dict) else "live_runtime_fallback"
    if not (baseline_status_path.is_file() and isinstance(baseline_status, dict)):
        baseline_status = preflight_live_baseline_status_fallback(return_manifest)
    gpu_return_manifest = read_json(gpu_return_manifest_path, {}) if gpu_return_manifest_path and gpu_return_manifest_path.is_file() else return_manifest
    voyage_return_manifest = (
        read_json(voyage_return_manifest_path, {}) if voyage_return_manifest_path and voyage_return_manifest_path.is_file() else return_manifest
    )
    runtime = baseline_status.get("runtime") if isinstance(baseline_status, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}

    transfer_bundle_verify_required = bool(
        copy_packet_path is not None or (transfer_bundle_verify_path is not None and transfer_bundle_verify_path.is_file())
    )
    artifact_checks = {
        "baseline_status": preflight_json_artifact_status(baseline_status_path, require_complete=False),
        "return_acceptance": preflight_return_acceptance_status(return_acceptance_path, return_manifest_path),
        "return_manifest": preflight_json_artifact_status(return_manifest_path, require_complete=False),
        "transfer_manifest_verify": preflight_transfer_manifest_verify_status(transfer_manifest_verify_path),
        "handoff_verify": preflight_handoff_verify_status(handoff_verify_path),
        "transfer_bundle_verify": preflight_transfer_bundle_verify_status(
            transfer_bundle_verify_path,
            require_complete=transfer_bundle_verify_required,
        ),
        "gpu_return_manifest": preflight_json_artifact_status(gpu_return_manifest_path, require_complete=False),
        "voyage_return_manifest": preflight_json_artifact_status(voyage_return_manifest_path, require_complete=False),
    }
    if copy_packet_path is None and not artifact_checks["return_acceptance"].get("exists"):
        artifact_checks["return_acceptance"].update(
            {
                "complete": True,
                "required": False,
                "skipped": True,
                "reason": "sender_side_gate_snapshot_not_transferred",
            }
        )
    if copy_packet_path is not None:
        artifact_checks["copy_packet"] = preflight_copy_packet_status(
            copy_packet_path,
            transfer_bundle_verify_path=transfer_bundle_verify_path,
        )
    script_checks = {
        "full": external_runner_script_status(
            full_runner_path,
            runtime=runtime,
            requires_cuda=True,
            required_env=("VOYAGE_API_KEY",),
            requires_numpy=True,
            requires_local_embedding_deps=True,
        ),
        "gpu": external_runner_script_status(
            gpu_runner_path,
            runtime=runtime,
            requires_cuda=True,
            required_env=(),
            requires_numpy=True,
            requires_local_embedding_deps=True,
        ),
        "voyage": external_runner_script_status(
            voyage_runner_path,
            runtime=runtime,
            requires_cuda=False,
            required_env=("VOYAGE_API_KEY",),
            requires_numpy=True,
            requires_local_embedding_deps=False,
        ),
    }
    return_packaging = return_packaging_preflight_status(return_bundle_script_path, return_manifest)
    return_packaging_scripts = {
        "full": return_packaging,
        "gpu": return_packaging_preflight_status(
            gpu_return_bundle_script_path,
            gpu_return_manifest,
            baseline_filters=("jina-code-embeddings-0.5b", "qwen3-embedding-4b"),
        ),
        "voyage": return_packaging_preflight_status(
            voyage_return_bundle_script_path,
            voyage_return_manifest,
            baseline_filters=("voyage-code-3",),
        ),
    }
    transfer_checks = [
        artifact_checks["transfer_manifest_verify"],
        artifact_checks["handoff_verify"],
    ]
    if copy_packet_path is not None or artifact_checks["transfer_bundle_verify"].get("exists"):
        transfer_checks.append(artifact_checks["transfer_bundle_verify"])
    transfer_ready = all(check.get("complete") for check in transfer_checks if check.get("path"))
    copy_packet_check = artifact_checks.get("copy_packet")
    copy_packet_checked = copy_packet_check is not None
    copy_packet_gate_ready = bool(copy_packet_check.get("complete")) if copy_packet_check is not None else True
    return_acceptance_gate_ready = bool(artifact_checks["return_acceptance"].get("complete"))
    local_runner_ready = all(check.get("ready") for check in script_checks.values() if check.get("path"))
    return_packaging_ready = all(bool(check.get("ready")) for check in return_packaging_scripts.values() if check.get("path"))
    report = {
        "generated_at": utc_now(),
        "complete": bool(
            transfer_ready
            and copy_packet_gate_ready
            and return_acceptance_gate_ready
            and local_runner_ready
            and return_packaging_ready
            and return_acceptance.get("complete")
        ),
        "handoff_ready": bool(transfer_ready and copy_packet_gate_ready),
        "copy_packet_checked": copy_packet_checked,
        "copy_packet_ready": bool(copy_packet_check.get("complete")) if copy_packet_check is not None else None,
        "local_runner_ready": bool(local_runner_ready),
        "return_packaging_ready": return_packaging_ready,
        "baseline_status_complete": bool(baseline_status.get("complete")) if isinstance(baseline_status, dict) else False,
        "baseline_status_source": baseline_status_source,
        "return_acceptance_ready": return_acceptance_gate_ready,
        "return_acceptance_complete": bool(return_acceptance.get("complete")) if isinstance(return_acceptance, dict) else False,
        "blocking_baselines": baseline_status.get("blocking_baselines") if isinstance(baseline_status, dict) else [],
        "runtime": {
            "cuda_available": bool(runtime.get("cuda_available")),
            "nvidia_smi_available": bool(((runtime.get("nvidia_smi") or {}) if isinstance(runtime.get("nvidia_smi"), dict) else {}).get("available")),
            "numpy_installed": runtime_dependency_installed(runtime, "numpy"),
            "sentence_transformers_installed": runtime_dependency_installed(runtime, "sentence_transformers"),
            "torch_installed": runtime_dependency_installed(runtime, "torch"),
            "voyage_api_key_set": bool(runtime.get("voyage_api_key_set")),
        },
        "artifact_checks": artifact_checks,
        "runner_scripts": script_checks,
        "return_packaging": return_packaging,
        "return_packaging_scripts": return_packaging_scripts,
        "next_required_action": (
            "Run missing open-source embedding baselines on CUDA-capable machines, package a verified return bundle, "
            "apply it locally, and rerun v1-1-finalize-baselines."
        ),
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_external_runner_preflight_markdown(report), encoding="utf-8")
    return report


def preflight_json_artifact_status(path: Path | None, require_complete: bool = False) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "complete": not require_complete, "required": require_complete}
    payload = read_json(path, {})
    exists = path.is_file()
    complete_value = payload.get("complete") if isinstance(payload, dict) else None
    failed_checks = payload.get("failed_checks") if isinstance(payload, dict) else None
    status = {
        "path": str(path),
        "exists": exists,
        "required": require_complete,
        "complete": bool(exists and (not require_complete or complete_value is True)),
        "reported_complete": complete_value,
        "failed_checks": failed_checks if isinstance(failed_checks, list) else [],
    }
    if isinstance(payload, dict) and payload.get("generated_at"):
        status["generated_at"] = payload.get("generated_at")
    return status


def preflight_live_baseline_status_fallback(return_manifest: dict[str, Any]) -> dict[str, Any]:
    missing_required = [str(path) for path in (return_manifest.get("missing_required_files") or [])] if isinstance(return_manifest, dict) else []
    required = [str(path) for path in (return_manifest.get("required_files") or [])] if isinstance(return_manifest, dict) else []
    if required and not missing_required:
        missing_required = [path for path in required if not Path(path).is_file()]
    missing_by_baseline = group_return_files_by_baseline(missing_required, REQUIRED_V1_1_BASELINES)
    blocking_baselines = [baseline for baseline in REQUIRED_V1_1_BASELINES if missing_by_baseline.get(baseline)]
    return {
        "complete": False,
        "runtime": embedding_runtime_status(),
        "blocking_baselines": blocking_baselines,
        "missing_required_files": missing_required,
        "missing_required_files_by_baseline": missing_by_baseline,
        "source": "live_runtime_return_manifest_fallback",
    }


def unpack_smoke_fallback_path(path: Path | None, fallback_name: str) -> Path | None:
    if path is None or path.is_file():
        return path
    candidate = path.with_name(fallback_name)
    return candidate if candidate.is_file() else path


def preflight_transfer_manifest_verify_status(path: Path | None) -> dict[str, Any]:
    original_path = path
    path = unpack_smoke_fallback_path(path, "baseline_transfer_unpack_smoke_v19.json")
    status = preflight_json_artifact_status(path, require_complete=True)
    if original_path is not None and path is not None and path != original_path:
        status["primary_path"] = str(original_path)
        status["used_fallback"] = True
    payload = read_json(path, {}) if path else {}
    freshness = {
        "checked": False,
        "complete": True,
        "mismatches": [],
        "transfer_manifest_path": None,
    }
    status["freshness"] = freshness
    status["freshness_mismatches"] = freshness["mismatches"]
    if not status.get("exists") or not isinstance(payload, dict):
        return status

    manifest_value = payload.get("manifest")
    if not manifest_value:
        return status

    freshness["checked"] = True
    manifest_path = Path(str(manifest_value))
    freshness["transfer_manifest_path"] = str(manifest_path)
    manifest_payload = read_json(manifest_path, {}) if manifest_path.is_file() else {}
    verify_generated_at = payload.get("generated_at")
    manifest_generated_at = manifest_payload.get("generated_at") if isinstance(manifest_payload, dict) else None
    freshness["transfer_manifest_verify_generated_at"] = verify_generated_at
    freshness["transfer_manifest_generated_at"] = manifest_generated_at
    if not manifest_path.is_file():
        freshness["mismatches"].append("missing_transfer_manifest")
    elif manifest_generated_at and not verify_generated_at:
        freshness["mismatches"].append("missing_transfer_manifest_verify_generated_at")
    elif verify_generated_at and manifest_generated_at and str(verify_generated_at) < str(manifest_generated_at):
        freshness["mismatches"].append("transfer_manifest_verify_stale")

    if freshness["mismatches"]:
        freshness["complete"] = False
        status["complete"] = False
        failed_checks = status.get("failed_checks")
        if not isinstance(failed_checks, list):
            failed_checks = []
        failed_checks.append("transfer_manifest_verify_freshness")
        status["failed_checks"] = failed_checks
    return status


def preflight_handoff_verify_status(path: Path | None) -> dict[str, Any]:
    original_path = path
    path = unpack_smoke_fallback_path(path, "baseline_handoff_unpack_smoke_v19.json")
    status = preflight_json_artifact_status(path, require_complete=True)
    if original_path is not None and path is not None and path != original_path:
        status["primary_path"] = str(original_path)
        status["used_fallback"] = True
    payload = read_json(path, {}) if path else {}
    freshness = {
        "checked": False,
        "complete": True,
        "mismatches": [],
        "handoff_path": None,
    }
    status["freshness"] = freshness
    status["freshness_mismatches"] = freshness["mismatches"]
    if not status.get("exists") or not isinstance(payload, dict):
        return status

    handoff_value = payload.get("handoff")
    if not handoff_value:
        return status

    freshness["checked"] = True
    handoff_path = Path(str(handoff_value))
    freshness["handoff_path"] = str(handoff_path)
    handoff_payload = read_json(handoff_path, {}) if handoff_path.is_file() else {}
    verify_generated_at = payload.get("generated_at")
    handoff_generated_at = handoff_payload.get("generated_at") if isinstance(handoff_payload, dict) else None
    freshness["handoff_verify_generated_at"] = verify_generated_at
    freshness["handoff_generated_at"] = handoff_generated_at
    if not handoff_path.is_file():
        freshness["mismatches"].append("missing_handoff")
    elif handoff_generated_at and not verify_generated_at:
        freshness["mismatches"].append("missing_handoff_verify_generated_at")
    elif verify_generated_at and handoff_generated_at and str(verify_generated_at) < str(handoff_generated_at):
        freshness["mismatches"].append("handoff_verify_stale")

    if freshness["mismatches"]:
        freshness["complete"] = False
        status["complete"] = False
        failed_checks = status.get("failed_checks")
        if not isinstance(failed_checks, list):
            failed_checks = []
        failed_checks.append("handoff_verify_freshness")
        status["failed_checks"] = failed_checks
    return status


def preflight_transfer_bundle_verify_status(path: Path | None, *, require_complete: bool = True) -> dict[str, Any]:
    status = preflight_json_artifact_status(path, require_complete=require_complete)
    if not require_complete and not status.get("exists"):
        status["complete"] = True
    payload = read_json(path, {}) if path else {}
    freshness = {
        "checked": False,
        "complete": True,
        "mismatches": [],
        "transfer_manifest_path": None,
        "transfer_bundle_path": None,
    }
    status["freshness"] = freshness
    status["freshness_mismatches"] = freshness["mismatches"]
    if not status.get("exists") or not isinstance(payload, dict):
        return status

    verify_generated_at = payload.get("generated_at")
    manifest_value = payload.get("manifest")
    if manifest_value:
        freshness["checked"] = True
        manifest_path = Path(str(manifest_value))
        freshness["transfer_manifest_path"] = str(manifest_path)
        manifest_payload = read_json(manifest_path, {}) if manifest_path.is_file() else {}
        manifest_generated_at = manifest_payload.get("generated_at") if isinstance(manifest_payload, dict) else None
        freshness["transfer_bundle_verify_generated_at"] = verify_generated_at
        freshness["transfer_manifest_generated_at"] = manifest_generated_at
        if not manifest_path.is_file():
            freshness["mismatches"].append("missing_transfer_manifest")
        elif manifest_generated_at and not verify_generated_at:
            freshness["mismatches"].append("missing_transfer_bundle_verify_generated_at")
        elif verify_generated_at and manifest_generated_at and str(verify_generated_at) < str(manifest_generated_at):
            freshness["mismatches"].append("transfer_bundle_verify_stale")

    bundle_value = payload.get("bundle")
    if bundle_value:
        freshness["checked"] = True
        bundle_path = Path(str(bundle_value))
        freshness["transfer_bundle_path"] = str(bundle_path)
        expected_size = payload.get("size_bytes")
        expected_sha256 = payload.get("sha256")
        bundle_fingerprint = file_fingerprint(bundle_path) if bundle_path.is_file() else {}
        actual_size = bundle_fingerprint.get("size_bytes") if bundle_fingerprint else None
        actual_sha256 = bundle_fingerprint.get("sha256") if bundle_fingerprint else None
        freshness["transfer_bundle_declared_size_bytes"] = expected_size
        freshness["transfer_bundle_actual_size_bytes"] = actual_size
        freshness["transfer_bundle_declared_sha256"] = expected_sha256
        freshness["transfer_bundle_actual_sha256"] = actual_sha256
        if not bundle_path.is_file():
            freshness["mismatches"].append("missing_transfer_bundle")
        elif isinstance(expected_size, int) and actual_size != expected_size:
            freshness["mismatches"].append("transfer_bundle_size_mismatch")
        if bundle_path.is_file() and expected_sha256 and actual_sha256 != expected_sha256:
            freshness["mismatches"].append("transfer_bundle_sha256_mismatch")

    if freshness["mismatches"]:
        freshness["complete"] = False
        status["complete"] = False
        failed_checks = status.get("failed_checks")
        if not isinstance(failed_checks, list):
            failed_checks = []
        failed_checks.append("transfer_bundle_verify_freshness")
        status["failed_checks"] = failed_checks
    return status


def preflight_return_acceptance_status(return_acceptance_path: Path, return_manifest_path: Path) -> dict[str, Any]:
    status = preflight_json_artifact_status(return_acceptance_path, require_complete=False)
    payload = read_json(return_acceptance_path, {})
    freshness = {
        "checked": False,
        "complete": True,
        "mismatches": [],
        "return_manifest_path": str(return_manifest_path),
        "completion_json_path": None,
    }
    status["freshness"] = freshness
    status["freshness_mismatches"] = freshness["mismatches"]
    if not status.get("exists") or not isinstance(payload, dict):
        return status

    current_status = payload.get("current_status") if isinstance(payload.get("current_status"), dict) else {}
    return_manifest_payload = read_json(return_manifest_path, {}) if return_manifest_path else {}

    declared_return_manifest_path = payload.get("return_manifest")
    declared_return_manifest_generated_at = payload.get("return_manifest_generated_at") or current_status.get("return_manifest_generated_at")
    should_check_return_manifest = bool(
        declared_return_manifest_path
        or declared_return_manifest_generated_at
        or payload.get("required_return_file_count") is not None
        or "missing_required_file_count" in payload
    )
    if should_check_return_manifest:
        freshness["checked"] = True
        actual_return_manifest_generated_at = (
            return_manifest_payload.get("generated_at") if isinstance(return_manifest_payload, dict) else None
        )
        freshness["return_manifest_declared_path"] = declared_return_manifest_path
        freshness["return_manifest_declared_generated_at"] = declared_return_manifest_generated_at
        freshness["return_manifest_actual_generated_at"] = actual_return_manifest_generated_at
        if declared_return_manifest_path and str(declared_return_manifest_path) != str(return_manifest_path):
            freshness["mismatches"].append("return_manifest_path_mismatch")
        if actual_return_manifest_generated_at and not declared_return_manifest_generated_at:
            freshness["mismatches"].append("missing_return_manifest_generated_at")
        elif declared_return_manifest_generated_at and actual_return_manifest_generated_at != declared_return_manifest_generated_at:
            freshness["mismatches"].append("return_manifest_generated_at_mismatch")

    completion_json_path = Path(str(payload.get("completion_json"))) if payload.get("completion_json") else None
    declared_completion_generated_at = payload.get("completion_audit_generated_at") or current_status.get("completion_audit_generated_at")
    should_check_completion = bool(completion_json_path or declared_completion_generated_at)
    if should_check_completion:
        freshness["checked"] = True
        freshness["completion_json_path"] = str(completion_json_path) if completion_json_path else None
        completion_payload = read_json(completion_json_path, {}) if completion_json_path else {}
        actual_completion_generated_at = completion_payload.get("generated_at") if isinstance(completion_payload, dict) else None
        freshness["completion_audit_declared_generated_at"] = declared_completion_generated_at
        freshness["completion_audit_actual_generated_at"] = actual_completion_generated_at
        if not completion_json_path:
            freshness["mismatches"].append("missing_completion_json_path")
        elif not completion_json_path.is_file():
            freshness["mismatches"].append("missing_completion_json")
        elif actual_completion_generated_at and not declared_completion_generated_at:
            freshness["mismatches"].append("missing_completion_audit_generated_at")
        elif declared_completion_generated_at and actual_completion_generated_at != declared_completion_generated_at:
            freshness["mismatches"].append("completion_audit_generated_at_mismatch")

    if freshness["mismatches"]:
        freshness["complete"] = False
        status["complete"] = False
        failed_checks = status.get("failed_checks")
        if not isinstance(failed_checks, list):
            failed_checks = []
        failed_checks.append("return_acceptance_freshness")
        status["failed_checks"] = failed_checks
    return status


def preflight_copy_packet_status(
    copy_packet_path: Path,
    transfer_bundle_verify_path: Path | None = None,
) -> dict[str, Any]:
    status = preflight_json_artifact_status(copy_packet_path, require_complete=True)
    payload = read_json(copy_packet_path, {})
    mismatches: list[str] = []
    comparison: dict[str, Any] = {}

    status["matches_transfer_bundle"] = False
    status["mismatches"] = mismatches
    status["comparison"] = comparison

    if not status.get("exists"):
        mismatches.append("missing_copy_packet")
        status["complete"] = False
        return status
    if not isinstance(payload, dict):
        mismatches.append("invalid_copy_packet_json")
        status["complete"] = False
        return status
    if payload.get("complete") is not True:
        mismatches.append("copy_packet_not_complete")
    if status.get("failed_checks"):
        mismatches.append("copy_packet_failed_checks")

    copy_files = [str(path) for path in (payload.get("copy_to_external_runner") or [])]
    copy_file_statuses: list[dict[str, Any]] = []
    status["copy_files"] = copy_file_statuses
    if not copy_files:
        mismatches.append("copy_packet_missing_copy_file_list")
    for path in copy_files:
        fingerprint = file_fingerprint(Path(path))
        copy_file_statuses.append(
            {
                "path": path,
                "exists": bool(fingerprint.get("exists")),
                "type": fingerprint.get("type"),
                "size_bytes": fingerprint.get("size_bytes"),
                "sha256": fingerprint.get("sha256"),
            }
        )
        if not (fingerprint.get("exists") and fingerprint.get("type") == "file"):
            mismatches.append("copy_packet_copy_file_missing")

    bundle_path_value = payload.get("bundle_path")
    if not bundle_path_value:
        mismatches.append("copy_packet_missing_bundle_path")
    else:
        bundle_fingerprint = file_fingerprint(Path(str(bundle_path_value)))
        status["bundle_file_fingerprint"] = bundle_fingerprint
        comparison["copy_packet_bundle_file"] = {
            "expected_bundle_sha256": payload.get("bundle_sha256"),
            "actual_bundle_sha256": bundle_fingerprint.get("sha256"),
            "expected_bundle_size_bytes": payload.get("bundle_size_bytes"),
            "actual_bundle_size_bytes": bundle_fingerprint.get("size_bytes"),
        }
        if not (bundle_fingerprint.get("exists") and bundle_fingerprint.get("type") == "file"):
            mismatches.append("copy_packet_bundle_file_missing")
        else:
            if payload.get("bundle_sha256") and payload.get("bundle_sha256") != bundle_fingerprint.get("sha256"):
                mismatches.append("copy_packet_bundle_file_sha256_mismatch")
            if payload.get("bundle_size_bytes") is not None and payload.get("bundle_size_bytes") != bundle_fingerprint.get("size_bytes"):
                mismatches.append("copy_packet_bundle_file_size_mismatch")

    checksum_path_value = payload.get("checksum_path")
    if not checksum_path_value:
        mismatches.append("copy_packet_missing_checksum_path")
    else:
        checksum_path = Path(str(checksum_path_value))
        checksum_fingerprint = file_fingerprint(checksum_path)
        status["checksum_file_fingerprint"] = checksum_fingerprint
        if not (checksum_fingerprint.get("exists") and checksum_fingerprint.get("type") == "file"):
            mismatches.append("copy_packet_checksum_file_missing")
        else:
            try:
                checksum_token = checksum_path.read_text(encoding="utf-8").split()[0]
            except (IndexError, OSError, UnicodeDecodeError):
                checksum_token = ""
            status["checksum_file_sha256_value"] = checksum_token
            comparison["copy_packet_checksum_file"] = {
                "expected_bundle_sha256": payload.get("bundle_sha256"),
                "actual_checksum_sha256": checksum_token,
            }
            if payload.get("bundle_sha256") and checksum_token != payload.get("bundle_sha256"):
                mismatches.append("copy_packet_checksum_sha256_mismatch")

    verify_payload: dict[str, Any] = {}
    if transfer_bundle_verify_path is not None:
        status["transfer_bundle_verify_path"] = str(transfer_bundle_verify_path)
        status["transfer_bundle_verify_exists"] = transfer_bundle_verify_path.is_file()
        verify_value = read_json(transfer_bundle_verify_path, {})
        if not transfer_bundle_verify_path.is_file():
            mismatches.append("missing_transfer_bundle_verify")
        elif not isinstance(verify_value, dict):
            mismatches.append("invalid_transfer_bundle_verify_json")
        else:
            verify_payload = verify_value
            status["transfer_bundle_verify_complete"] = bool(verify_payload.get("complete"))
            if verify_payload.get("complete") is not True:
                mismatches.append("transfer_bundle_verify_not_complete")
            verify_fingerprint = verify_payload.get("bundle_fingerprint")
            if not isinstance(verify_fingerprint, dict):
                verify_fingerprint = {}
            expected_sha = str(verify_payload.get("sha256") or verify_fingerprint.get("sha256") or "")
            expected_size = (
                verify_payload.get("size_bytes")
                if verify_payload.get("size_bytes") is not None
                else verify_fingerprint.get("size_bytes")
            )
            comparison["transfer_bundle_verify"] = {
                "expected_bundle_sha256": expected_sha,
                "actual_bundle_sha256": payload.get("bundle_sha256"),
                "expected_bundle_size_bytes": expected_size,
                "actual_bundle_size_bytes": payload.get("bundle_size_bytes"),
            }
            verify_copy_packet_check = named_report_check(verify_payload, "bundle_excludes_external_runner_copy_packet")
            status["transfer_bundle_verify_copy_packet_exclusion"] = verify_copy_packet_check
            if not verify_copy_packet_check.get("present"):
                mismatches.append("transfer_bundle_verify_missing_copy_packet_exclusion_check")
            elif verify_copy_packet_check.get("status") != "pass":
                mismatches.append("transfer_bundle_verify_copy_packet_exclusion_failed")
            if expected_sha:
                if payload.get("bundle_sha256") != expected_sha:
                    mismatches.append("bundle_sha256_mismatch")
            else:
                mismatches.append("transfer_bundle_verify_sha256_missing")
            if expected_size is not None:
                if payload.get("bundle_size_bytes") != expected_size:
                    mismatches.append("bundle_size_bytes_mismatch")
            else:
                mismatches.append("transfer_bundle_verify_size_missing")

    transfer_report_path = payload.get("transfer_bundle_report")
    if not transfer_report_path:
        mismatches.append("copy_packet_missing_transfer_bundle_report")
    else:
        report_path = Path(str(transfer_report_path))
        status["transfer_bundle_report_path"] = str(report_path)
        status["transfer_bundle_report_exists"] = report_path.is_file()
        report_value = read_json(report_path, {})
        if not report_path.is_file():
            mismatches.append("missing_transfer_bundle_report")
        elif not isinstance(report_value, dict):
            mismatches.append("invalid_transfer_bundle_report_json")
        else:
            status["transfer_bundle_report_complete"] = bool(report_value.get("complete"))
            if report_value.get("complete") is not True:
                mismatches.append("transfer_bundle_report_not_complete")
            report_fingerprint = report_value.get("bundle_fingerprint")
            if not isinstance(report_fingerprint, dict):
                report_fingerprint = {}
            expected_report_sha = str(report_value.get("sha256") or report_fingerprint.get("sha256") or "")
            expected_report_size = (
                report_value.get("size_bytes")
                if report_value.get("size_bytes") is not None
                else report_fingerprint.get("size_bytes")
            )
            expected_file_count = report_value.get("file_count")
            expected_generated_at = report_value.get("generated_at")
            external_acceptance = (
                report_value.get("external_acceptance") if isinstance(report_value.get("external_acceptance"), dict) else {}
            )
            expected_return_file_count = external_acceptance.get("required_return_file_count")
            expected_return_files = [str(path) for path in (external_acceptance.get("required_return_files") or [])]
            expected_external_baselines = [str(name) for name in (external_acceptance.get("external_baselines") or [])]
            expected_return_files_by_baseline_payload = external_acceptance.get("required_return_files_by_baseline")
            if isinstance(expected_return_files_by_baseline_payload, dict) and expected_return_files_by_baseline_payload:
                expected_return_files_by_baseline = {
                    str(baseline): [str(path) for path in paths]
                    for baseline, paths in expected_return_files_by_baseline_payload.items()
                    if isinstance(paths, list)
                }
            else:
                expected_return_files_by_baseline = (
                    group_return_files_by_baseline(expected_return_files, expected_external_baselines)
                    if expected_return_files
                    else {}
                )
            actual_return_files = [str(path) for path in (payload.get("required_return_files") or [])]
            actual_return_files_by_baseline = payload.get("required_return_files_by_baseline")
            if not isinstance(actual_return_files_by_baseline, dict):
                actual_return_files_by_baseline = {}
            comparison["transfer_bundle_report"] = {
                "expected_generated_at": expected_generated_at,
                "actual_generated_at": payload.get("bundle_generated_at"),
                "expected_bundle_sha256": expected_report_sha,
                "actual_bundle_sha256": payload.get("bundle_sha256"),
                "expected_bundle_size_bytes": expected_report_size,
                "actual_bundle_size_bytes": payload.get("bundle_size_bytes"),
                "expected_transfer_file_count": expected_file_count,
                "actual_transfer_file_count": payload.get("transfer_file_count"),
                "expected_required_return_file_count": expected_return_file_count,
                "actual_required_return_file_count": payload.get("required_return_file_count"),
                "expected_required_return_files": expected_return_files,
                "actual_required_return_files": actual_return_files,
            }
            if expected_return_files_by_baseline:
                comparison["transfer_bundle_report"]["expected_required_return_files_by_baseline"] = (
                    expected_return_files_by_baseline
                )
                comparison["transfer_bundle_report"]["actual_required_return_files_by_baseline"] = (
                    actual_return_files_by_baseline
                )
            report_manifest_copy_packet_check = named_report_check(report_value, "manifest_excludes_external_runner_copy_packet")
            status["transfer_bundle_report_manifest_copy_packet_exclusion"] = report_manifest_copy_packet_check
            if not report_manifest_copy_packet_check.get("present"):
                mismatches.append("transfer_bundle_report_missing_copy_packet_exclusion_check")
            elif report_manifest_copy_packet_check.get("status") != "pass":
                mismatches.append("transfer_bundle_report_copy_packet_exclusion_failed")
            report_verify_value = report_value.get("verification") if isinstance(report_value.get("verification"), dict) else {}
            report_verify_copy_packet_check = named_report_check(report_verify_value, "bundle_excludes_external_runner_copy_packet")
            status["transfer_bundle_report_verify_copy_packet_exclusion"] = report_verify_copy_packet_check
            if not report_verify_copy_packet_check.get("present"):
                mismatches.append("transfer_bundle_report_missing_verify_copy_packet_exclusion_check")
            elif report_verify_copy_packet_check.get("status") != "pass":
                mismatches.append("transfer_bundle_report_verify_copy_packet_exclusion_failed")
            if expected_generated_at and payload.get("bundle_generated_at") != expected_generated_at:
                mismatches.append("bundle_generated_at_mismatch")
            if expected_report_sha and payload.get("bundle_sha256") != expected_report_sha:
                mismatches.append("bundle_report_sha256_mismatch")
            if expected_report_size is not None and payload.get("bundle_size_bytes") != expected_report_size:
                mismatches.append("bundle_report_size_bytes_mismatch")
            if expected_file_count is not None and payload.get("transfer_file_count") != expected_file_count:
                mismatches.append("transfer_file_count_mismatch")
            if expected_return_file_count is not None and payload.get("required_return_file_count") != expected_return_file_count:
                mismatches.append("required_return_file_count_mismatch")
            if expected_return_files and actual_return_files != expected_return_files:
                mismatches.append("required_return_files_mismatch")
            if expected_return_files_by_baseline and actual_return_files_by_baseline != expected_return_files_by_baseline:
                mismatches.append("required_return_files_by_baseline_mismatch")

    status["matches_transfer_bundle"] = not mismatches
    status["complete"] = bool(status.get("exists") and payload.get("complete") is True and not mismatches)
    return status


def named_report_check(report: dict[str, Any], name: str) -> dict[str, Any]:
    checks = report.get("checks") if isinstance(report, dict) else []
    if not isinstance(checks, list):
        checks = []
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return {
                "present": True,
                "status": check.get("status"),
                "name": name,
            }
    return {"present": False, "status": None, "name": name}


def external_runner_script_status(
    path: Path | None,
    runtime: dict[str, Any],
    requires_cuda: bool,
    required_env: Iterable[str],
    requires_numpy: bool = False,
    requires_local_embedding_deps: bool = False,
) -> dict[str, Any]:
    required_env_values = [str(value) for value in required_env]
    required_dependencies: list[str] = []
    missing_dependencies: list[str] = []
    blockers: list[str] = []
    if "VOYAGE_API_KEY" in required_env_values and not runtime.get("voyage_api_key_set"):
        blockers.append("missing_voyage_api_key")
    if requires_numpy:
        required_dependencies.append("numpy")
    if requires_local_embedding_deps:
        required_dependencies.extend(["sentence_transformers", "torch"])
    for dependency in required_dependencies:
        if not runtime_dependency_installed(runtime, dependency):
            missing_dependencies.append(dependency)
    if missing_dependencies:
        blockers.append("missing_dependency")
    if requires_cuda and not runtime.get("cuda_available"):
        blockers.append("missing_cuda")
    exists = bool(path and path.is_file())
    executable = bool(path and path.is_file() and os.access(path, os.X_OK))
    if path and not exists:
        blockers.append("missing_script")
    return {
        "path": str(path) if path else None,
        "exists": exists,
        "executable": executable,
        "requires_cuda": requires_cuda,
        "required_env": required_env_values,
        "requires_numpy": bool(requires_numpy),
        "requires_local_embedding_deps": bool(requires_local_embedding_deps),
        "required_dependencies": required_dependencies,
        "missing_dependencies": missing_dependencies,
        "ready": bool(exists and not blockers),
        "expected_local_status": "ready" if exists and not blockers else "blocked",
        "expected_blockers": blockers,
        "fail_fast_before_long_run": bool(blockers),
    }


def runtime_dependency_installed(runtime: dict[str, Any], name: str) -> bool:
    status = runtime.get(name)
    if not isinstance(status, dict):
        return False
    return bool(status.get("installed"))


def return_packaging_preflight_status(
    script_path: Path | None,
    return_manifest: dict[str, Any],
    baseline_filters: Iterable[str] | None = None,
) -> dict[str, Any]:
    baseline_filter_values = [str(baseline) for baseline in (baseline_filters or [])]
    manifest_required = [str(path) for path in (return_manifest.get("required_files") or [])] if isinstance(return_manifest, dict) else []
    manifest_missing = [str(path) for path in (return_manifest.get("missing_required_files") or [])] if isinstance(return_manifest, dict) else []
    if baseline_filter_values:
        manifest_required = filter_return_files_by_baselines(manifest_required, baseline_filter_values)
        manifest_missing = filter_return_files_by_baselines(manifest_missing, baseline_filter_values)
    if manifest_required:
        missing_required_files = [path for path in manifest_required if not Path(path).is_file()]
    else:
        missing_required_files = manifest_missing
    artifacts_complete = bool(manifest_required) and not missing_required_files
    if not manifest_required and isinstance(return_manifest, dict):
        artifacts_complete = bool(return_manifest.get("artifacts_complete")) and not missing_required_files
    exists = bool(script_path and script_path.is_file())
    return {
        "path": str(script_path) if script_path else None,
        "exists": exists,
        "executable": bool(script_path and script_path.is_file() and os.access(script_path, os.X_OK)),
        "ready": bool(exists and artifacts_complete and not missing_required_files),
        "expected_local_status": "ready" if exists and artifacts_complete and not missing_required_files else "blocked",
        "expected_blocker": "missing_required_return_files" if missing_required_files else (None if artifacts_complete else "return_manifest_not_artifact_complete"),
        "missing_required_file_count": len(missing_required_files),
        "missing_required_files": missing_required_files,
        "manifest_missing_required_files": manifest_missing,
        "required_files": manifest_required,
        "requested_baselines": baseline_filter_values,
    }


def filter_return_files_by_baselines(paths: Iterable[str], baselines: Iterable[str]) -> list[str]:
    baseline_values = [str(baseline) for baseline in baselines]
    grouped = group_return_files_by_baseline(paths, baseline_values)
    return [path for baseline in baseline_values for path in grouped.get(baseline, [])]


def render_v1_1_external_runner_preflight_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 External Runner Preflight",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Handoff ready: `{bool(report.get('handoff_ready'))}`",
        f"Copy packet checked: `{bool(report.get('copy_packet_checked'))}`",
        f"Copy packet ready: `{report.get('copy_packet_ready')}`",
        f"Local runner ready: `{bool(report.get('local_runner_ready'))}`",
        f"Return packaging ready: `{bool(report.get('return_packaging_ready'))}`",
        f"Baseline status complete: `{bool(report.get('baseline_status_complete'))}`",
        f"Baseline status source: `{report.get('baseline_status_source')}`",
        f"Return acceptance ready: `{bool(report.get('return_acceptance_ready'))}`",
        f"Return acceptance complete: `{bool(report.get('return_acceptance_complete'))}`",
        f"Blocking baselines: `{json.dumps(report.get('blocking_baselines') or [])}`",
        "",
        "## Runtime",
        "",
    ]
    for key, value in (report.get("runtime") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Artifact Checks", ""])
    for name, check in (report.get("artifact_checks") or {}).items():
        reported_complete = check.get("reported_complete")
        reported_value = "n/a" if reported_complete is None else str(bool(reported_complete))
        lines.append(
            f"- `{name}`: exists=`{bool(check.get('exists'))}`, artifact_ready=`{bool(check.get('complete'))}`, "
            f"reported_complete=`{reported_value}`, path=`{check.get('path')}`"
        )
        if name == "copy_packet" and check.get("mismatches"):
            lines.append(f"  mismatches=`{json.dumps(check.get('mismatches') or [])}`")
        if check.get("freshness_mismatches"):
            lines.append(f"  freshness_mismatches=`{json.dumps(check.get('freshness_mismatches') or [])}`")
    lines.extend(["", "## Runner Scripts", ""])
    for name, check in (report.get("runner_scripts") or {}).items():
        lines.append(
            f"- `{name}`: status=`{check.get('expected_local_status')}`, "
            f"blockers=`{json.dumps(check.get('expected_blockers') or [])}`, "
            f"missing_dependencies=`{json.dumps(check.get('missing_dependencies') or [])}`, "
            f"path=`{check.get('path')}`"
        )
    packaging = report.get("return_packaging") or {}
    lines.extend(
        [
            "",
            "## Return Packaging",
            "",
            f"- Status: `{packaging.get('expected_local_status')}`",
            f"- Blocker: `{packaging.get('expected_blocker')}`",
            f"- Missing required files: `{packaging.get('missing_required_file_count')}`",
            "",
            "| Script | Status | Blocker | Missing required files | Baselines | Path |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, check in (report.get("return_packaging_scripts") or {}).items():
        lines.append(
            "| {name} | `{status}` | `{blocker}` | `{missing}` | `{baselines}` | `{path}` |".format(
                name=name,
                status=check.get("expected_local_status"),
                blocker=check.get("expected_blocker"),
                missing=check.get("missing_required_file_count"),
                baselines=json.dumps(check.get("requested_baselines") or []),
                path=check.get("path"),
            )
        )
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            str(report.get("next_required_action") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def write_v1_1_external_runner_failfast_smoke_report(
    preflight_path: Path,
    full_runner_path: Path | None = None,
    gpu_runner_path: Path | None = None,
    voyage_runner_path: Path | None = None,
    return_bundle_script_path: Path | None = None,
    gpu_return_bundle_script_path: Path | None = None,
    voyage_return_bundle_script_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    timeout_seconds: int = 60,
    allow_ready_runs: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    preflight = read_json(preflight_path, {})
    runner_scripts = preflight.get("runner_scripts") if isinstance(preflight, dict) else {}
    return_packaging = preflight.get("return_packaging") if isinstance(preflight, dict) else {}
    return_packaging_scripts = preflight.get("return_packaging_scripts") if isinstance(preflight, dict) else {}
    runtime = preflight.get("runtime") if isinstance(preflight, dict) else {}
    if not isinstance(runner_scripts, dict):
        runner_scripts = {}
    if not isinstance(return_packaging, dict):
        return_packaging = {}
    if not isinstance(return_packaging_scripts, dict):
        return_packaging_scripts = {}
    if not isinstance(runtime, dict):
        runtime = {}

    script_paths = {
        "full": full_runner_path,
        "gpu": gpu_runner_path,
        "voyage": voyage_runner_path,
    }
    steps: dict[str, Any] = {}
    for name, path in script_paths.items():
        expected = runner_scripts.get(name) if isinstance(runner_scripts.get(name), dict) else {}
        expected_blockers = [str(value) for value in expected.get("expected_blockers") or []]
        steps[name] = run_v1_1_failfast_smoke_step(
            name=name,
            path=path,
            expected_blockers=expected_blockers,
            long_run_markers=("== Baseline:",),
            timeout_seconds=timeout_seconds,
            allow_ready_runs=allow_ready_runs,
            cwd=cwd,
        )
        if should_probe_full_runner_cuda_after_voyage_blocker(name, expected_blockers, steps[name], runtime):
            steps["full_cuda_probe"] = run_v1_1_failfast_smoke_step(
                name="full_cuda_probe",
                path=path,
                expected_blockers=["missing_cuda"],
                long_run_markers=("== Baseline:",),
                timeout_seconds=timeout_seconds,
                allow_ready_runs=allow_ready_runs,
                cwd=cwd,
                env_overrides={"VOYAGE_API_KEY": "ARB_FAILFAST_PLACEHOLDER"},
            )

    package_expected = str(return_packaging.get("expected_blocker") or "")
    return_package_scripts = {
        "return_packaging": return_bundle_script_path,
        "gpu_return_packaging": gpu_return_bundle_script_path,
        "voyage_return_packaging": voyage_return_bundle_script_path,
    }
    for name, path in return_package_scripts.items():
        if path is None and name != "return_packaging":
            continue
        preflight_key = {"return_packaging": "full", "gpu_return_packaging": "gpu", "voyage_return_packaging": "voyage"}[name]
        package_status = return_packaging_scripts.get(preflight_key) if isinstance(return_packaging_scripts.get(preflight_key), dict) else {}
        expected_blocker = str(package_status.get("expected_blocker") or package_expected)
        steps[name] = run_v1_1_failfast_smoke_step(
            name=name,
            path=path,
            expected_blockers=[expected_blocker] if expected_blocker else [],
            long_run_markers=("-- create return artifact bundle --", "V1.1 return artifact bundle completed."),
            timeout_seconds=timeout_seconds,
            allow_ready_runs=allow_ready_runs,
            cwd=cwd,
        )

    report = {
        "generated_at": utc_now(),
        "complete": all(bool(step.get("complete")) for step in steps.values()),
        "preflight": str(preflight_path),
        "allow_ready_runs": bool(allow_ready_runs),
        "timeout_seconds": int(timeout_seconds),
        "steps": steps,
        "next_required_action": (
            "Run missing open-source embedding baselines on CUDA-capable machines, package a verified return bundle, "
            "apply it locally, and rerun v1-1-finalize-baselines."
        ),
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_external_runner_failfast_smoke_markdown(report), encoding="utf-8")
    return report


def run_v1_1_failfast_smoke_step(
    name: str,
    path: Path | None,
    expected_blockers: list[str],
    long_run_markers: Iterable[str],
    timeout_seconds: int,
    allow_ready_runs: bool,
    cwd: Path | None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    exists = bool(path and path.is_file())
    executable = bool(path and path.is_file() and os.access(path, os.X_OK))
    if not exists:
        return {
            "path": str(path) if path else None,
            "exists": exists,
            "executable": executable,
            "expected_blockers": expected_blockers,
            "status": "missing_script",
            "complete": False,
        }
    if not expected_blockers and not allow_ready_runs:
        return {
            "path": str(path),
            "exists": exists,
            "executable": executable,
            "expected_blockers": expected_blockers,
            "status": "skipped_ready",
            "complete": True,
            "note": "Skipped because preflight did not report an expected local blocker.",
        }

    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        completed = subprocess.run(
            ["bash", str(path)],
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = redact_sensitive_output(completed.stdout)
        stderr = redact_sensitive_output(completed.stderr)
        combined = f"{stdout}\n{stderr}"
        observed_blockers = classify_v1_1_failfast_blockers(combined)
        long_run_started = any(marker in combined for marker in long_run_markers)
        expected_match = bool(set(expected_blockers) & set(observed_blockers)) if expected_blockers else True
        blocked_before_long_run = completed.returncode != 0 and not long_run_started
        complete = bool(blocked_before_long_run and expected_match)
        return {
            "path": str(path),
            "exists": exists,
            "executable": executable,
            "expected_blockers": expected_blockers,
            "observed_blockers": observed_blockers,
            "exit_code": completed.returncode,
            "timed_out": False,
            "long_run_started": long_run_started,
            "blocked_before_long_run": blocked_before_long_run,
            "expected_blocker_observed": expected_match,
            "status": "blocked_as_expected" if complete else "unexpected_result",
            "complete": complete,
            "stdout_tail": tail_text(stdout),
            "stderr_tail": tail_text(stderr),
            "env_overrides": sorted(env_overrides) if env_overrides else [],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = redact_sensitive_output(exc.stdout or "")
        stderr = redact_sensitive_output(exc.stderr or "")
        return {
            "path": str(path),
            "exists": exists,
            "executable": executable,
            "expected_blockers": expected_blockers,
            "observed_blockers": classify_v1_1_failfast_blockers(f"{stdout}\n{stderr}"),
            "exit_code": None,
            "timed_out": True,
            "long_run_started": False,
            "blocked_before_long_run": False,
            "expected_blocker_observed": False,
            "status": "timed_out",
            "complete": False,
            "stdout_tail": tail_text(stdout),
            "stderr_tail": tail_text(stderr),
            "env_overrides": sorted(env_overrides) if env_overrides else [],
        }


def should_probe_full_runner_cuda_after_voyage_blocker(
    name: str,
    expected_blockers: list[str],
    primary_step: dict[str, Any],
    runtime: dict[str, Any],
) -> bool:
    if name != "full":
        return False
    expected = set(expected_blockers)
    if not {"missing_voyage_api_key", "missing_cuda"}.issubset(expected):
        return False
    if "missing_dependency" in expected:
        return False
    if runtime.get("cuda_available") is not False:
        return False
    observed = set(primary_step.get("observed_blockers") or [])
    return bool(
        primary_step.get("complete")
        and "missing_voyage_api_key" in observed
        and "missing_cuda" not in observed
        and not primary_step.get("long_run_started")
    )


def classify_v1_1_failfast_blockers(output: str) -> list[str]:
    blockers: list[str] = []
    checks = [
        ("missing_voyage_api_key", ("Set VOYAGE_API_KEY", "VOYAGE_API_KEY:")),
        ("missing_cuda", ("CUDA is not available", "missing_cuda")),
        ("missing_required_return_files", ("missing_required_files", "Missing required return artifact")),
        ("missing_dependency", ("Missing optional embedding dependencies", "Missing numpy")),
        ("missing_repo_source_files", ("Missing repo source files",)),
    ]
    for blocker, needles in checks:
        if any(needle in output for needle in needles):
            blockers.append(blocker)
    return blockers


def redact_sensitive_output(text: str) -> str:
    redacted = re.sub(r"ghp_[A-Za-z0-9_]+", "ghp_[REDACTED]", text)
    redacted = re.sub(r"(GITHUB_TOKEN=)[^\s]+", r"\1[REDACTED]", redacted)
    redacted = re.sub(r"(VOYAGE_API_KEY=)[^\s]+", r"\1[REDACTED]", redacted)
    return redacted


def tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def render_v1_1_external_runner_failfast_smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 External Runner Fail-Fast Smoke",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Preflight: `{report.get('preflight')}`",
        f"Timeout seconds: `{report.get('timeout_seconds')}`",
        "",
        "| Step | Status | Exit | Expected blockers | Observed blockers | Long run started |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, step in (report.get("steps") or {}).items():
        lines.append(
            "| {name} | `{status}` | `{exit_code}` | `{expected}` | `{observed}` | `{long_run}` |".format(
                name=name,
                status=step.get("status"),
                exit_code=step.get("exit_code"),
                expected=json.dumps(step.get("expected_blockers") or []),
                observed=json.dumps(step.get("observed_blockers") or []),
                long_run=bool(step.get("long_run_started")),
            )
        )
    lines.extend(["", "## Next Action", "", str(report.get("next_required_action") or ""), ""])
    return "\n".join(lines)


def write_v1_1_summary_from_details(
    details_path: Path,
    out_path: Path,
    model: str,
    mode: str = "embedding",
    candidate_filter: str | None = None,
    expected_samples: int | None = None,
    expected_sample_ids: set[str] | None = None,
    cache_dir: Path | None = None,
    shared_text_cache: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(details_path)
    if not rows:
        raise RuntimeError(f"No detail rows found: {details_path}")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    missing_sample_id_rows = sum(1 for sample_id in sample_ids if not sample_id)
    if missing_sample_id_rows:
        raise RuntimeError(f"Cannot summarize details with {missing_sample_id_rows} missing sample_id rows: {details_path}")
    duplicate_sample_ids = sorted({sample_id for sample_id, count in Counter(sample_ids).items() if count > 1})
    if duplicate_sample_ids:
        raise RuntimeError(f"Cannot summarize details with duplicate sample IDs: {duplicate_sample_ids[:20]}")
    if expected_samples is not None and len(rows) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} detail rows, found {len(rows)}: {details_path}")
    if expected_sample_ids is not None:
        actual_sample_ids = set(sample_ids)
        missing = sorted(expected_sample_ids - actual_sample_ids)
        unexpected = sorted(actual_sample_ids - expected_sample_ids)
        if missing or unexpected:
            raise RuntimeError(
                "Detail sample IDs do not match expected benchmark IDs: "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )

    candidate_filters = sorted({str(row.get("candidate_filter") or "") for row in rows if row.get("candidate_filter")})
    if candidate_filter is None:
        if len(candidate_filters) != 1:
            raise RuntimeError(f"Expected exactly one candidate_filter in details, found {candidate_filters}: {details_path}")
        candidate_filter = candidate_filters[0]
    elif candidate_filters and candidate_filters != [candidate_filter]:
        raise RuntimeError(f"Details candidate_filter mismatch: expected {candidate_filter!r}, found {candidate_filters}")

    malformed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        missing_metric_keys = list(BASELINE_DETAIL_METRICS) if not isinstance(metrics, dict) else [key for key in BASELINE_DETAIL_METRICS if key not in metrics]
        if not row.get("task_type") or missing_metric_keys:
            malformed_rows.append(
                {
                    "row": index,
                    "sample_id": row.get("sample_id"),
                    "missing_task_type": not bool(row.get("task_type")),
                    "missing_metric_keys": missing_metric_keys,
                }
            )
    if malformed_rows:
        raise RuntimeError(f"Cannot summarize malformed detail rows: {malformed_rows[:20]}")

    metrics = summarize_details(rows)
    payload: dict[str, Any] = {
        "mode": mode,
        "model": model,
        "candidate_filter": candidate_filter,
        "evaluated": len(rows),
        "skipped": {},
        "metrics": metrics,
    }
    if cache_dir is not None:
        payload["cache_dir"] = str(cache_dir)
    if shared_text_cache is not None:
        payload["shared_text_cache"] = str(shared_text_cache)
    write_json(out_path, payload)
    return {
        "summary": str(out_path),
        "details": str(details_path),
        "model": model,
        "mode": mode,
        "candidate_filter": candidate_filter,
        "evaluated": len(rows),
        "metrics": metrics,
    }


def write_v1_1_merged_details(
    details_paths: Iterable[Path],
    out_path: Path,
    expected_samples: int | None = None,
    expected_sample_ids: set[str] | None = None,
    candidate_filter: str | None = None,
    allow_incomplete: bool = False,
    report_out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    paths = list(details_paths)
    input_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    row_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_sample_id_rows: list[dict[str, Any]] = []
    missing_candidate_filter_rows: list[dict[str, Any]] = []
    malformed_metric_rows: list[dict[str, Any]] = []
    candidate_filters: set[str] = set()

    for path in paths:
        input_rows = read_jsonl(path)
        input_reports.append({"path": str(path), "rows": len(input_rows)})
        for index, row in enumerate(input_rows):
            rows.append(row)
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                row_sources[sample_id].append({"path": str(path), "row": index})
            else:
                missing_sample_id_rows.append({"path": str(path), "row": index})

            row_candidate_filter = str(row.get("candidate_filter") or "")
            if row_candidate_filter:
                candidate_filters.add(row_candidate_filter)
            else:
                missing_candidate_filter_rows.append({"path": str(path), "row": index, "sample_id": sample_id or None})

            metrics = row.get("metrics")
            missing_metric_keys = (
                list(BASELINE_DETAIL_METRICS)
                if not isinstance(metrics, dict)
                else [key for key in BASELINE_DETAIL_METRICS if key not in metrics]
            )
            if not row.get("task_type") or missing_metric_keys:
                malformed_metric_rows.append(
                    {
                        "path": str(path),
                        "row": index,
                        "sample_id": sample_id or None,
                        "missing_task_type": not bool(row.get("task_type")),
                        "missing_metric_keys": missing_metric_keys,
                    }
                )

    sample_counts = Counter({sample_id: len(sources) for sample_id, sources in row_sources.items()})
    duplicate_sample_ids = sorted(sample_id for sample_id, count in sample_counts.items() if count > 1)
    actual_sample_ids = set(row_sources)
    missing_sample_ids = sorted((expected_sample_ids or set()) - actual_sample_ids) if expected_sample_ids is not None else []
    unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids) if expected_sample_ids is not None else []
    expected_count_mismatch = expected_samples is not None and len(rows) != expected_samples

    candidate_filter_mismatch = False
    if candidate_filter is None:
        candidate_filter_mismatch = len(candidate_filters) != 1 or bool(missing_candidate_filter_rows)
        resolved_candidate_filter = sorted(candidate_filters)[0] if len(candidate_filters) == 1 else None
    else:
        candidate_filter_mismatch = candidate_filters != {candidate_filter} or bool(missing_candidate_filter_rows)
        resolved_candidate_filter = candidate_filter

    complete = (
        bool(paths)
        and bool(rows)
        and not missing_sample_id_rows
        and not duplicate_sample_ids
        and not expected_count_mismatch
        and not missing_sample_ids
        and not unexpected_sample_ids
        and not candidate_filter_mismatch
        and not malformed_metric_rows
    )
    wrote_output = False
    if complete or allow_incomplete:
        ensure_parent(out_path)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        wrote_output = True

    report: dict[str, Any] = {
        "complete": complete,
        "output": str(out_path),
        "wrote_output": wrote_output,
        "inputs": input_reports,
        "rows": len(rows),
        "unique_sample_ids": len(actual_sample_ids),
        "expected_samples": expected_samples,
        "candidate_filter": resolved_candidate_filter,
        "candidate_filters": sorted(candidate_filters),
        "missing_candidate_filter_rows": missing_candidate_filter_rows,
        "missing_sample_id_rows": missing_sample_id_rows,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_sample_sources": {sample_id: row_sources[sample_id] for sample_id in duplicate_sample_ids},
        "missing_sample_ids": missing_sample_ids,
        "unexpected_sample_ids": unexpected_sample_ids,
        "malformed_metric_rows": malformed_metric_rows,
        "allow_incomplete": allow_incomplete,
    }
    if report_out_path is not None:
        write_json(report_out_path, report)
    if markdown_out_path is not None:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_merged_details_markdown(report), encoding="utf-8")
    return report


def render_v1_1_merged_details_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Merged Details",
        "",
        f"- Complete: `{bool(report.get('complete'))}`",
        f"- Output: `{report.get('output')}`",
        f"- Wrote output: `{bool(report.get('wrote_output'))}`",
        f"- Rows: `{report.get('rows')}`",
        f"- Unique sample IDs: `{report.get('unique_sample_ids')}`",
        f"- Expected samples: `{report.get('expected_samples')}`",
        f"- Candidate filters: `{', '.join(report.get('candidate_filters') or [])}`",
        "",
        "## Inputs",
        "",
    ]
    for item in report.get("inputs") or []:
        lines.append(f"- `{item.get('path')}`: `{item.get('rows')}` rows")
    blockers = [
        ("Missing sample-id rows", report.get("missing_sample_id_rows") or []),
        ("Duplicate sample IDs", report.get("duplicate_sample_ids") or []),
        ("Missing expected sample IDs", report.get("missing_sample_ids") or []),
        ("Unexpected sample IDs", report.get("unexpected_sample_ids") or []),
        ("Missing candidate-filter rows", report.get("missing_candidate_filter_rows") or []),
        ("Malformed metric rows", report.get("malformed_metric_rows") or []),
    ]
    lines.extend(["", "## Blockers", ""])
    any_blocker = False
    for label, values in blockers:
        if values:
            any_blocker = True
            lines.append(f"- {label}: `{json.dumps(values[:20], ensure_ascii=False, sort_keys=True)}`")
    if not any_blocker:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_sample_id_shards(
    derived: Path,
    out_dir: Path,
    shard_count: int,
    prefix: str = "sample_ids",
    manifest_out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    allow_empty_shards: bool = False,
    assignment_strategy: str = "input_order_modulo",
    corpus_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not prefix:
        raise ValueError("prefix must not be empty.")
    allowed_strategies = {"input_order_modulo", "corpus_balanced"}
    if assignment_strategy not in allowed_strategies:
        raise ValueError(f"assignment_strategy must be one of {sorted(allowed_strategies)}.")

    corpus_weights: dict[tuple[str, str], int] = {}
    corpus_manifest_rows = 0
    if corpus_manifest_path is not None and corpus_manifest_path.exists():
        for row in read_jsonl(corpus_manifest_path):
            repo = str(row.get("repo") or "")
            base_commit = str(row.get("base_commit") or "")
            if not repo or not base_commit:
                continue
            try:
                weight = int(row.get("chunk_count") or 0)
            except (TypeError, ValueError):
                weight = 0
            corpus_weights[(repo, base_commit)] = max(1, weight)
            corpus_manifest_rows += 1

    sample_entries: list[dict[str, Any]] = []
    missing_sample_id_rows: list[dict[str, Any]] = []
    missing_corpus_key_rows: list[dict[str, Any]] = []
    for path in sample_paths_from_derived(derived):
        for row_index, row in enumerate(read_jsonl(path)):
            sample_id = str(row.get("id") or "")
            if not sample_id:
                missing_sample_id_rows.append({"path": str(path), "row": row_index})
                continue
            repo = str(row.get("repo") or "")
            base_commit = str(row.get("base_commit") or "")
            if assignment_strategy == "corpus_balanced" and (not repo or not base_commit):
                missing_corpus_key_rows.append({"path": str(path), "row": row_index, "sample_id": sample_id})
            sample_entries.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "row": row_index,
                    "repo": repo,
                    "base_commit": base_commit,
                }
            )

    sample_ids = [entry["sample_id"] for entry in sample_entries]
    duplicate_sample_ids = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    shard_weights = [0 for _ in range(shard_count)]
    shard_corpus_keys: list[set[tuple[str, str]]] = [set() for _ in range(shard_count)]
    corpus_groups_without_manifest_weight: list[dict[str, Any]] = []
    corpus_group_count = 0
    if assignment_strategy == "input_order_modulo":
        for index, sample_id in enumerate(sample_ids):
            shard_index = index % shard_count
            shards[shard_index].append(sample_id)
            shard_weights[shard_index] += 1
    else:
        grouped_entries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        fallback_prefix = "__sample__"
        for entry in sample_entries:
            repo = str(entry.get("repo") or "")
            base_commit = str(entry.get("base_commit") or "")
            key = (repo, base_commit) if repo and base_commit else (fallback_prefix, str(entry.get("sample_id") or ""))
            grouped_entries[key].append(entry)
        corpus_group_count = len(grouped_entries)
        groups: list[dict[str, Any]] = []
        for key, entries in grouped_entries.items():
            weight = corpus_weights.get(key)
            if weight is None:
                weight = len(entries)
                if key[0] != fallback_prefix:
                    corpus_groups_without_manifest_weight.append(
                        {
                            "repo": key[0],
                            "base_commit": key[1],
                            "samples": len(entries),
                        }
                    )
            groups.append(
                {
                    "key": key,
                    "entries": entries,
                    "sample_ids": [str(entry["sample_id"]) for entry in entries],
                    "first_row": min(int(entry["row"]) for entry in entries),
                    "weight": max(1, int(weight)),
                }
            )
        for group in sorted(groups, key=lambda item: (-int(item["weight"]), int(item["first_row"]), item["key"])):
            shard_index = min(range(shard_count), key=lambda index: (shard_weights[index], len(shards[index]), index))
            shards[shard_index].extend(group["sample_ids"])
            shard_weights[shard_index] += int(group["weight"])
            key = group["key"]
            if key[0] != fallback_prefix:
                shard_corpus_keys[shard_index].add(key)
    empty_shards = [index for index, ids in enumerate(shards) if not ids]
    complete = (
        bool(sample_ids)
        and not missing_sample_id_rows
        and not missing_corpus_key_rows
        and not duplicate_sample_ids
        and (allow_empty_shards or not empty_shards)
    )
    width = max(2, len(str(shard_count - 1)))
    files: list[dict[str, Any]] = []
    wrote_files = False
    if complete:
        out_dir.mkdir(parents=True, exist_ok=True)
        wrote_files = True
    for index, ids in enumerate(shards):
        shard_path = out_dir / f"{prefix}_shard{index:0{width}d}.txt"
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if complete:
            shard_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        files.append(
            {
                "index": index,
                "path": str(shard_path),
                "rows": len(ids),
                "estimated_weight": shard_weights[index],
                "corpus_group_count": len(shard_corpus_keys[index]),
                "sha256": digest,
                "first_sample_id": ids[0] if ids else None,
                "last_sample_id": ids[-1] if ids else None,
            }
        )

    manifest: dict[str, Any] = {
        "generated_at": utc_now(),
        "complete": complete,
        "wrote_files": wrote_files,
        "derived": str(derived),
        "out_dir": str(out_dir),
        "prefix": prefix,
        "shard_count": shard_count,
        "sample_count": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": duplicate_sample_ids,
        "missing_sample_id_rows": missing_sample_id_rows,
        "missing_corpus_key_rows": missing_corpus_key_rows,
        "empty_shards": empty_shards,
        "allow_empty_shards": allow_empty_shards,
        "selection_strategy": assignment_strategy,
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "corpus_manifest_rows": corpus_manifest_rows,
        "corpus_group_count": corpus_group_count,
        "corpus_groups_without_manifest_weight": corpus_groups_without_manifest_weight,
        "shard_estimated_weights": shard_weights,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "files": files,
        "notes": [
            "Pass a shard path to eval-embedding/eval-voyage with --sample-id-file.",
            "The input_order_modulo strategy uses the same assignment as --shard-count/--shard-index.",
            "The corpus_balanced strategy keeps each (repo, base_commit) corpus in one shard and balances by corpus_manifest chunk_count when provided.",
            "Run v1-1-merge-details after all shard details files are complete.",
        ],
    }
    if manifest_out_path is not None:
        write_json(manifest_out_path, manifest)
    if markdown_out_path is not None:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_sample_id_shards_markdown(manifest), encoding="utf-8")
    return manifest


def render_v1_1_sample_id_shards_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Sample ID Shards",
        "",
        f"Generated at: `{manifest.get('generated_at')}`",
        f"Complete: `{bool(manifest.get('complete'))}`",
        f"Derived: `{manifest.get('derived')}`",
        f"Shard count: `{manifest.get('shard_count')}`",
        f"Sample count: `{manifest.get('sample_count')}`",
        f"Unique sample IDs: `{manifest.get('unique_sample_ids')}`",
        f"Selection strategy: `{manifest.get('selection_strategy')}`",
        f"Corpus manifest: `{manifest.get('corpus_manifest')}`",
        f"Corpus groups: `{manifest.get('corpus_group_count')}`",
        f"Sample ID digest: `{manifest.get('sample_ids_sha256')}`",
        "",
        "## Files",
        "",
    ]
    for item in manifest.get("files") or []:
        lines.append(
            f"- shard `{item.get('index')}`: `{item.get('path')}` rows `{item.get('rows')}` "
            f"weight `{item.get('estimated_weight')}` corpora `{item.get('corpus_group_count')}` sha256 `{item.get('sha256')}`"
        )
    blockers = [
        ("Duplicate sample IDs", manifest.get("duplicate_sample_ids") or []),
        ("Missing sample-id rows", manifest.get("missing_sample_id_rows") or []),
        ("Missing corpus-key rows", manifest.get("missing_corpus_key_rows") or []),
        ("Empty shards", manifest.get("empty_shards") or []),
    ]
    lines.extend(["", "## Blockers", ""])
    any_blocker = False
    for label, values in blockers:
        if values:
            any_blocker = True
            lines.append(f"- {label}: `{json.dumps(values[:20], ensure_ascii=False, sort_keys=True)}`")
    if not any_blocker:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_handoff(
    derived: Path,
    corpus: Path,
    eval_dir: Path,
    cache_root: Path,
    report_dir: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    base_derived: Path = Path("data/benchmark/v1"),
    assembly_manifest: Path | None = None,
    corpus_manifest: Path | None = None,
    leaderboard_path: Path | None = None,
    leaderboard_json_path: Path | None = None,
    readiness_path: Path | None = None,
    readiness_markdown_path: Path | None = None,
    release_path: Path | None = None,
    release_json_path: Path | None = None,
    completion_path: Path | None = None,
    completion_json_path: Path | None = None,
    baseline_status_path: Path | None = None,
    baseline_status_markdown_path: Path | None = None,
    baseline_preflight_path: Path | None = None,
    handoff_verify_path: Path | None = None,
    handoff_verify_markdown_path: Path | None = None,
    finalization_path: Path | None = None,
    finalization_markdown_path: Path | None = None,
    shard_commands_path: Path | None = None,
    return_manifest_path: Path | None = None,
    return_manifest_markdown_path: Path | None = None,
    return_files_path: Path | None = None,
    return_acceptance_path: Path | None = None,
    return_acceptance_markdown_path: Path | None = None,
    include_shard_artifacts: bool = False,
    include_caches: bool = False,
    auto_merge_shards: bool = False,
    workflow_evidence_paths: Iterable[Path] | None = None,
    transfer_manifest_path: Path | None = None,
    transfer_manifest_markdown_path: Path | None = None,
    transfer_manifest_verify_path: Path | None = None,
    transfer_manifest_verify_markdown_path: Path | None = None,
    transfer_files_path: Path | None = None,
    transfer_bundle_path: Path | None = None,
    transfer_bundle_checksum_path: Path | None = None,
    transfer_bundle_archive_members_path: Path | None = None,
    transfer_bundle_report_path: Path | None = None,
    transfer_bundle_markdown_path: Path | None = None,
    transfer_bundle_verify_path: Path | None = None,
    transfer_bundle_verify_markdown_path: Path | None = None,
    transfer_unpack_script_path: Path | None = None,
    transfer_unpack_script_markdown_path: Path | None = None,
    transfer_unpack_destination: str | None = None,
    transfer_unpack_transfer_verify_path: Path | None = None,
    transfer_unpack_transfer_verify_markdown_path: Path | None = None,
    transfer_unpack_handoff_verify_path: Path | None = None,
    transfer_unpack_handoff_verify_markdown_path: Path | None = None,
    transfer_include_paths: Iterable[Path] | None = None,
    base_leaderboard_json_path: Path = Path("data/reports/v1/model_leaderboard.json"),
) -> dict[str, Any]:
    resolved_transfer_bundle_path = transfer_bundle_path or report_dir / "baseline_transfer_bundle.tar.zst"
    resolved_completion_path = completion_path or report_dir / "completion_audit.md"
    resolved_completion_json_path = completion_json_path or report_dir / "completion_audit.json"
    resolved_transfer_includes = list(transfer_include_paths) if transfer_include_paths is not None else default_v1_1_transfer_includes()
    manifest = build_v1_1_baseline_handoff(
        derived=derived,
        corpus=corpus,
        eval_dir=eval_dir,
        cache_root=cache_root,
        report_dir=report_dir,
        base_derived=base_derived,
        assembly_manifest=assembly_manifest or derived / "manifest.json",
        corpus_manifest=corpus_manifest or corpus / "corpus_manifest.jsonl",
        leaderboard_path=leaderboard_path or report_dir / "model_leaderboard.md",
        leaderboard_json_path=leaderboard_json_path or report_dir / "model_leaderboard.json",
        readiness_path=readiness_path or report_dir / "readiness.json",
        readiness_markdown_path=readiness_markdown_path or report_dir / "readiness.md",
        release_path=release_path or report_dir / "release_report.md",
        release_json_path=release_json_path or report_dir / "release_report.json",
        completion_path=resolved_completion_path,
        completion_json_path=resolved_completion_json_path,
        baseline_status_path=baseline_status_path or report_dir / "baseline_status.json",
        baseline_status_markdown_path=baseline_status_markdown_path or report_dir / "baseline_status.md",
        baseline_preflight_path=baseline_preflight_path or report_dir / "baseline_summary_preflight.json",
        handoff_verify_path=handoff_verify_path or report_dir / "baseline_handoff_verify.json",
        handoff_verify_markdown_path=handoff_verify_markdown_path or report_dir / "baseline_handoff_verify.md",
        finalization_path=finalization_path or report_dir / "baseline_finalization.json",
        finalization_markdown_path=finalization_markdown_path or report_dir / "baseline_finalization.md",
        shard_commands_path=shard_commands_path,
        return_manifest_path=return_manifest_path,
        return_manifest_markdown_path=return_manifest_markdown_path,
        return_files_path=return_files_path,
        return_acceptance_path=return_acceptance_path,
        return_acceptance_markdown_path=return_acceptance_markdown_path,
        include_shard_artifacts=include_shard_artifacts,
        include_caches=include_caches,
        auto_merge_shards=auto_merge_shards,
        workflow_evidence_paths=workflow_evidence_paths,
        transfer_manifest_path=transfer_manifest_path or report_dir / "baseline_transfer_manifest.json",
        transfer_manifest_markdown_path=transfer_manifest_markdown_path or report_dir / "baseline_transfer_manifest.md",
        transfer_manifest_verify_path=transfer_manifest_verify_path or report_dir / "baseline_transfer_manifest_verify.json",
        transfer_manifest_verify_markdown_path=transfer_manifest_verify_markdown_path or report_dir / "baseline_transfer_manifest_verify.md",
        transfer_files_path=transfer_files_path or report_dir / "baseline_transfer_manifest.files",
        transfer_bundle_path=resolved_transfer_bundle_path,
        transfer_bundle_checksum_path=transfer_bundle_checksum_path or Path(f"{resolved_transfer_bundle_path}.sha256"),
        transfer_bundle_archive_members_path=transfer_bundle_archive_members_path or Path(f"{resolved_transfer_bundle_path}.members"),
        transfer_bundle_report_path=transfer_bundle_report_path or report_dir / "baseline_transfer_bundle.json",
        transfer_bundle_markdown_path=transfer_bundle_markdown_path or report_dir / "baseline_transfer_bundle.md",
        transfer_bundle_verify_path=transfer_bundle_verify_path or report_dir / "baseline_transfer_bundle_verify.json",
        transfer_bundle_verify_markdown_path=transfer_bundle_verify_markdown_path or report_dir / "baseline_transfer_bundle_verify.md",
        transfer_unpack_script_path=transfer_unpack_script_path,
        transfer_unpack_script_markdown_path=transfer_unpack_script_markdown_path,
        transfer_unpack_destination=transfer_unpack_destination,
        transfer_unpack_transfer_verify_path=transfer_unpack_transfer_verify_path,
        transfer_unpack_transfer_verify_markdown_path=transfer_unpack_transfer_verify_markdown_path,
        transfer_unpack_handoff_verify_path=transfer_unpack_handoff_verify_path,
        transfer_unpack_handoff_verify_markdown_path=transfer_unpack_handoff_verify_markdown_path,
        transfer_include_paths=resolved_transfer_includes,
        base_leaderboard_json_path=base_leaderboard_json_path,
        handoff_path=out_path,
    )
    write_json(out_path, manifest)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_handoff_markdown(manifest), encoding="utf-8")
    return manifest


def build_v1_1_baseline_handoff(
    derived: Path,
    corpus: Path,
    eval_dir: Path,
    cache_root: Path,
    report_dir: Path,
    base_derived: Path,
    assembly_manifest: Path,
    corpus_manifest: Path,
    leaderboard_path: Path,
    leaderboard_json_path: Path,
    readiness_path: Path,
    readiness_markdown_path: Path,
    release_path: Path,
    release_json_path: Path,
    completion_path: Path,
    completion_json_path: Path,
    baseline_status_path: Path,
    baseline_status_markdown_path: Path,
    baseline_preflight_path: Path,
    handoff_verify_path: Path,
    handoff_verify_markdown_path: Path,
    finalization_path: Path,
    finalization_markdown_path: Path,
    shard_commands_path: Path | None,
    return_manifest_path: Path | None,
    return_manifest_markdown_path: Path | None,
    return_files_path: Path | None,
    return_acceptance_path: Path | None,
    return_acceptance_markdown_path: Path | None,
    include_shard_artifacts: bool,
    include_caches: bool,
    auto_merge_shards: bool,
    workflow_evidence_paths: Iterable[Path] | None,
    transfer_manifest_path: Path,
    transfer_manifest_markdown_path: Path,
    transfer_manifest_verify_path: Path,
    transfer_manifest_verify_markdown_path: Path,
    transfer_files_path: Path,
    transfer_bundle_path: Path,
    transfer_bundle_checksum_path: Path,
    transfer_bundle_archive_members_path: Path,
    transfer_bundle_report_path: Path,
    transfer_bundle_markdown_path: Path,
    transfer_bundle_verify_path: Path,
    transfer_bundle_verify_markdown_path: Path,
    transfer_unpack_script_path: Path | None,
    transfer_unpack_script_markdown_path: Path | None,
    transfer_unpack_destination: str | None,
    transfer_unpack_transfer_verify_path: Path | None,
    transfer_unpack_transfer_verify_markdown_path: Path | None,
    transfer_unpack_handoff_verify_path: Path | None,
    transfer_unpack_handoff_verify_markdown_path: Path | None,
    transfer_include_paths: Iterable[Path],
    base_leaderboard_json_path: Path,
    handoff_path: Path,
) -> dict[str, Any]:
    jobs = [
        embedding_job(
            label="jina-code-embeddings-0.5b",
            command="eval-embedding",
            model="jinaai/jina-code-embeddings-0.5b",
            derived=derived,
            corpus=corpus,
            eval_dir=eval_dir,
            cache_root=cache_root,
            batch_size=64,
            extra_args=["--device", "cuda", "--trust-remote-code"],
        ),
        embedding_job(
            label="qwen3-embedding-4b",
            command="eval-embedding",
            model="Qwen/Qwen3-Embedding-4B",
            derived=derived,
            corpus=corpus,
            eval_dir=eval_dir,
            cache_root=cache_root,
            batch_size=8,
            extra_args=["--device", "cuda", "--trust-remote-code"],
        ),
    ]
    transfer_manifest_args: list[Any] = [
        "v1-1-baseline-transfer-manifest",
        "--handoff",
        handoff_path,
        "--out",
        transfer_manifest_path,
        "--markdown-out",
        transfer_manifest_markdown_path,
        "--files-out",
        transfer_files_path,
    ]
    for include_path in transfer_include_paths:
        transfer_manifest_args.extend(["--include", include_path])
    transfer_manifest_verify_args: list[Any] = [
        "v1-1-verify-transfer-manifest",
        "--manifest",
        transfer_manifest_path,
        "--out",
        transfer_manifest_verify_path,
        "--markdown-out",
        transfer_manifest_verify_markdown_path,
    ]
    setup_commands = []
    transfer_unpack_script: dict[str, Any] | None = None
    workflow_transfer_unpack_verify_path: Path | None = None
    workflow_handoff_unpack_verify_path: Path | None = None
    if transfer_unpack_script_path:
        resolved_transfer_unpack_transfer_verify_path = (
            transfer_unpack_transfer_verify_path or report_dir / "baseline_transfer_unpack_smoke.json"
        )
        resolved_transfer_unpack_handoff_verify_path = (
            transfer_unpack_handoff_verify_path or report_dir / "baseline_handoff_unpack_smoke.json"
        )
        workflow_transfer_unpack_verify_path = resolved_transfer_unpack_transfer_verify_path
        workflow_handoff_unpack_verify_path = resolved_transfer_unpack_handoff_verify_path
        transfer_unpack_args: list[Any] = [
            "v1-1-baseline-transfer-unpack-script",
            "--bundle",
            transfer_bundle_path,
            "--checksum",
            transfer_bundle_checksum_path,
            "--manifest",
            transfer_manifest_path,
            "--handoff",
            handoff_path,
            "--out",
            transfer_unpack_script_path,
            "--transfer-verify",
            resolved_transfer_unpack_transfer_verify_path,
            "--handoff-verify",
            resolved_transfer_unpack_handoff_verify_path,
        ]
        if transfer_unpack_script_markdown_path:
            transfer_unpack_args.extend(["--markdown-out", transfer_unpack_script_markdown_path])
        if transfer_unpack_destination:
            transfer_unpack_args.extend(["--destination", transfer_unpack_destination])
        if transfer_unpack_transfer_verify_markdown_path:
            transfer_unpack_args.extend(["--transfer-verify-markdown", transfer_unpack_transfer_verify_markdown_path])
        if transfer_unpack_handoff_verify_markdown_path:
            transfer_unpack_args.extend(["--handoff-verify-markdown", transfer_unpack_handoff_verify_markdown_path])
        setup_commands.append(cli_command(*transfer_unpack_args))
        transfer_unpack_script = {
            "script": str(transfer_unpack_script_path),
            "markdown": str(transfer_unpack_script_markdown_path) if transfer_unpack_script_markdown_path else None,
            "destination": transfer_unpack_destination,
            "transfer_verify": str(resolved_transfer_unpack_transfer_verify_path),
            "transfer_verify_markdown": str(transfer_unpack_transfer_verify_markdown_path)
            if transfer_unpack_transfer_verify_markdown_path
            else str(Path(f"{resolved_transfer_unpack_transfer_verify_path}.md")),
            "handoff_verify": str(resolved_transfer_unpack_handoff_verify_path),
            "handoff_verify_markdown": str(transfer_unpack_handoff_verify_markdown_path)
            if transfer_unpack_handoff_verify_markdown_path
            else str(Path(f"{resolved_transfer_unpack_handoff_verify_path}.md")),
        }
    if return_acceptance_path:
        acceptance_args: list[Any] = [
            "v1-1-baseline-return-acceptance",
            "--handoff",
            handoff_path,
            "--out",
            return_acceptance_path,
            "--completion-json",
            completion_json_path,
        ]
        if return_manifest_path:
            acceptance_args.extend(["--return-manifest", return_manifest_path])
        if return_acceptance_markdown_path:
            acceptance_args.extend(["--markdown-out", return_acceptance_markdown_path])
        setup_commands.append(cli_command(*acceptance_args))
    setup_commands.extend(
        [
            cli_command(*transfer_manifest_args),
            cli_command(
            "v1-1-baseline-transfer-bundle",
            "--manifest",
            transfer_manifest_path,
            "--bundle",
            transfer_bundle_path,
            "--checksum",
            transfer_bundle_checksum_path,
            "--archive-members",
            transfer_bundle_archive_members_path,
            "--bundle-files",
            transfer_files_path,
            "--out",
            transfer_bundle_report_path,
            "--markdown-out",
            transfer_bundle_markdown_path,
            "--verify-out",
            transfer_bundle_verify_path,
            "--verify-markdown-out",
            transfer_bundle_verify_markdown_path,
            ),
            cli_command(*transfer_manifest_verify_args),
            cli_command(
            "v1-1-verify-handoff",
            "--handoff",
            handoff_path,
            "--out",
            handoff_verify_path,
            "--markdown-out",
            handoff_verify_markdown_path,
            ),
        ]
    )
    finalization_args: list[Any] = [
        "v1-1-finalize-baselines",
        "--handoff",
        handoff_path,
        "--out",
        finalization_path,
        "--markdown-out",
        finalization_markdown_path,
    ]
    if shard_commands_path:
        finalization_args.extend(["--shard-commands", shard_commands_path])
    if return_manifest_path:
        finalization_args.extend(["--return-manifest", return_manifest_path])
    if return_manifest_markdown_path:
        finalization_args.extend(["--return-manifest-markdown", return_manifest_markdown_path])
    if return_files_path:
        finalization_args.extend(["--return-files", return_files_path])
    if include_shard_artifacts:
        finalization_args.append("--include-shard-artifacts")
    if include_caches:
        finalization_args.append("--include-caches")
    if auto_merge_shards:
        finalization_args.append("--auto-merge-shards")
    finalization_args.extend(completion_doc_cli_args(V1_1_RELEASE_DOCS))
    workflow_evidence: list[Path] = []
    seen_workflow_evidence: set[str] = set()
    handoff_payload: dict[str, Any] = {}

    def add_workflow_evidence(path: Path | str | None) -> None:
        if path is None:
            return
        evidence_path = Path(str(path))
        key = str(evidence_path)
        if key in seen_workflow_evidence:
            return
        seen_workflow_evidence.add(key)
        workflow_evidence.append(evidence_path)

    add_workflow_evidence(handoff_path)
    add_workflow_evidence(handoff_verify_path)
    add_workflow_evidence(transfer_bundle_verify_path)
    add_workflow_evidence(workflow_transfer_unpack_verify_path)
    add_workflow_evidence(workflow_handoff_unpack_verify_path)
    for evidence_path in workflow_evidence_paths or []:
        add_workflow_evidence(evidence_path)
    if return_manifest_path:
        add_workflow_evidence(return_manifest_path)
    for evidence_path in workflow_evidence:
        finalization_args.extend(["--workflow-evidence", evidence_path])

    completion_audit_args: list[Any] = [
        "report-v1-1-completion-audit",
        "--readiness",
        readiness_path,
        "--release-json",
        release_json_path,
        "--baseline-status",
        baseline_status_path,
        "--leaderboard-json",
        leaderboard_json_path,
        "--out",
        completion_path,
        "--json-out",
        completion_json_path,
        *completion_doc_cli_args(V1_1_RELEASE_DOCS),
    ]
    for evidence_path in workflow_evidence:
        completion_audit_args.extend(["--workflow-evidence", evidence_path])

    verification_commands = [
        cli_command(
            "v1-1-baseline-status",
            "--derived",
            derived,
            "--eval-dir",
            eval_dir,
            "--out",
            baseline_status_path,
            "--markdown-out",
            baseline_status_markdown_path,
        ),
        cli_command("check-baseline-summaries", "--derived", derived, "--eval-dir", eval_dir, "--out", baseline_preflight_path),
        cli_command(
            "report-models",
            "--eval-dir",
            eval_dir,
            "--out",
            leaderboard_path,
            "--json-out",
            leaderboard_json_path,
            "--required-baseline",
            "lexical",
            "--required-baseline",
            "aider-style-repomap",
            "--required-baseline",
            "jina-code-embeddings-0.5b",
            "--required-baseline",
            "qwen3-embedding-4b",
        ),
        cli_command(
            "v1-1-readiness",
            "--derived",
            derived,
            "--base-derived",
            base_derived,
            "--manifest",
            assembly_manifest,
            "--corpus-manifest",
            corpus_manifest,
            "--eval-dir",
            eval_dir,
            "--leaderboard",
            leaderboard_path,
            "--leaderboard-json",
            leaderboard_json_path,
            "--out",
            readiness_path,
            "--markdown-out",
            readiness_markdown_path,
        ),
        cli_command(
            "report-v1-1",
            "--readiness",
            readiness_path,
            "--leaderboard-json",
            leaderboard_json_path,
            "--base-leaderboard-json",
            base_leaderboard_json_path,
            "--out",
            release_path,
            "--json-out",
            release_json_path,
        ),
        cli_command(*completion_audit_args),
        cli_command(*finalization_args),
    ]
    return {
        "generated_at": utc_now(),
        "inputs": {
            "derived": str(derived),
            "base_derived": str(base_derived),
            "corpus": str(corpus),
            "corpus_manifest": str(corpus_manifest),
            "assembly_manifest": str(assembly_manifest),
            "eval_dir": str(eval_dir),
            "cache_root": str(cache_root),
            "report_dir": str(report_dir),
            "handoff": str(handoff_path),
        },
        "transfer_bundle": {
            "bundle": str(transfer_bundle_path),
            "checksum": str(transfer_bundle_checksum_path),
            "archive_members": str(transfer_bundle_archive_members_path),
            "bundle_files": str(transfer_files_path),
            "report": str(transfer_bundle_report_path),
            "markdown": str(transfer_bundle_markdown_path),
            "verify_report": str(transfer_bundle_verify_path),
            "verify_markdown": str(transfer_bundle_verify_markdown_path),
        },
        "transfer_unpack_script": transfer_unpack_script,
        "handoff_verification": {
            "report": str(handoff_verify_path),
            "markdown": str(handoff_verify_markdown_path),
        },
        "transfer_manifest_verification": {
            "report": str(transfer_manifest_verify_path),
            "markdown": str(transfer_manifest_verify_markdown_path),
        },
        "finalization": {
            "report": str(finalization_path),
            "markdown": str(finalization_markdown_path),
            "shard_commands": str(shard_commands_path) if shard_commands_path else None,
            "return_manifest": str(return_manifest_path) if return_manifest_path else None,
            "return_manifest_markdown": str(return_manifest_markdown_path) if return_manifest_markdown_path else None,
            "return_files": str(return_files_path) if return_files_path else None,
            "include_shard_artifacts": include_shard_artifacts,
            "include_caches": include_caches,
            "auto_merge_shards": auto_merge_shards,
            "workflow_evidence": [str(path) for path in workflow_evidence],
        },
        "return_acceptance": {
            "report": str(return_acceptance_path) if return_acceptance_path else None,
            "markdown": str(return_acceptance_markdown_path) if return_acceptance_markdown_path else None,
        },
        "transfer_includes": [str(path) for path in transfer_include_paths],
        "setup_commands": setup_commands,
        "input_fingerprints": {
            "algorithm": "sha256",
            "derived_samples": sample_collection_fingerprint(derived),
            "base_samples": sample_collection_fingerprint(base_derived),
            "assembly_manifest": file_fingerprint(assembly_manifest),
            "corpus_manifest": file_fingerprint(corpus_manifest),
        },
        "jobs": jobs,
        "external_acceptance": build_v1_1_handoff_external_acceptance(jobs, transfer_include_paths, return_manifest_path),
        "verification_commands": verification_commands,
        "expected_required_baselines": list(REQUIRED_V1_1_BASELINES),
        "notes": [
            "Run all jobs with candidate_filter=all_files and no keep list.",
            "Use --resume-details after interruptions; completed details rows are skipped on retry.",
            "Use v1-1-write-sample-shards if external workers should receive explicit --sample-id-file inputs instead of modulo shard arguments.",
            "Use v1-1-baseline-shard-commands with a sample shard manifest to produce ready-to-run per-shard baseline commands.",
            "Pass the shard command report to v1-1-baseline-status with --shard-commands to inspect per-shard details progress.",
            "When transferring helper artifacts or source files, pass them to v1-1-baseline-transfer-manifest with --include so they appear in the rsync/tar file list.",
            "For prepared V19-style runs, prefer the generated transfer bundle and embedding runbook when present; they include the source/helper files, transfer-unpack bootstrap script, shard scripts, return packaging script, and local apply script.",
            "Minimal external-runner path: verify the transfer bundle checksum, unpack the bundle, verify transfer and handoff fingerprints, install the local package plus embedding dependencies including numpy, run the full or split shard scripts, package a verified return bundle, apply it on the local reporting checkout, and accept completion only when the audit reports overall_status=complete.",
            "For parallel external runs, add --shard-count and --shard-index with shard-specific --details/--out paths, then run v1-1-merge-details and v1-1-summary-from-details.",
            "After external baselines finish, package artifacts on the external runner with the generated return-bundle script, then apply the returned bundle on the local reporting checkout before finalization. The full serial shard runner can invoke the packaging script before finalization when generated with --return-bundle-script.",
            "The full return-bundle verifier report is required workflow evidence for completion; copying raw summaries/details without applying a verified bundle is not enough to pass the final audit.",
            "If a job produces a complete details file but no summary, use v1-1-summary-from-details before rerunning embeddings.",
            "Only publish V1.1 when report-v1-1-completion-audit reports overall_status=complete.",
        ],
    }


def build_v1_1_handoff_external_acceptance(
    jobs: Iterable[dict[str, Any]],
    transfer_include_paths: Iterable[Path | str],
    return_manifest_path: Path | str | None,
) -> dict[str, Any]:
    external_baselines: list[str] = []
    required_return_files: list[str] = []
    required_return_files_by_baseline: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        baseline = str(job.get("baseline") or "")
        if baseline:
            external_baselines.append(baseline)
        artifacts = job.get("artifacts") if isinstance(job.get("artifacts"), dict) else {}
        for artifact_name in ("details", "summary"):
            artifact_path = artifacts.get(artifact_name)
            if artifact_path:
                artifact = str(artifact_path)
                required_return_files.append(artifact)
                required_return_files_by_baseline[baseline or "unassigned"].append(artifact)
    transfer_includes = [str(path) for path in transfer_include_paths]
    return {
        "external_baselines": external_baselines,
        "required_return_files": required_return_files,
        "required_return_files_by_baseline": {
            baseline: paths for baseline, paths in required_return_files_by_baseline.items() if paths
        },
        "required_return_file_count": len(required_return_files),
        "run_scripts": sorted(path for path in transfer_includes if Path(path).name.startswith("run_v19_") and Path(path).suffix == ".sh"),
        "return_packaging_scripts": sorted(path for path in transfer_includes if Path(path).name.startswith("package_v19_") and Path(path).suffix == ".sh"),
        "return_apply_scripts": sorted(path for path in transfer_includes if Path(path).name.startswith("apply_v19_") and Path(path).suffix == ".sh"),
        "return_manifest": str(return_manifest_path) if return_manifest_path else None,
        "completion_gate": (
            "Apply a verified return bundle and rerun v1-1-finalize-baselines until completion_audit_v19.json "
            "reports overall_status=complete. The full generated run/apply scripts refresh the compact return acceptance "
            "report with --require-complete, so final acceptance exits nonzero until this gate is complete."
        ),
    }


def default_v1_1_transfer_includes() -> list[Path]:
    return [
        Path("README.md"),
        Path("PLAN.md"),
        Path("docs/v1_1_completion_audit.md"),
        Path("pyproject.toml"),
        Path("src/agent_retrieval_bench"),
    ]


def verify_v1_1_baseline_handoff(
    handoff_path: Path,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(handoff_path, {})
    fingerprints = manifest.get("input_fingerprints") if isinstance(manifest, dict) else None
    checks: list[dict[str, Any]] = []
    if not isinstance(fingerprints, dict):
        checks.append(
            {
                "name": "input_fingerprints_present",
                "status": "fail",
                "mismatches": ["handoff manifest has no input_fingerprints object"],
                "expected": {"present": True},
                "actual": {"present": False},
            }
        )
    else:
        algorithm = fingerprints.get("algorithm")
        checks.append(
            {
                "name": "fingerprint_algorithm",
                "status": "pass" if algorithm == "sha256" else "fail",
                "mismatches": [] if algorithm == "sha256" else [f"expected sha256, found {algorithm!r}"],
                "expected": {"algorithm": "sha256"},
                "actual": {"algorithm": algorithm},
            }
        )
        for key in ("derived_samples", "base_samples"):
            checks.append(verify_sample_collection_fingerprint(key, fingerprints.get(key) or {}))
        for key in ("assembly_manifest", "corpus_manifest"):
            checks.append(verify_file_fingerprint(key, fingerprints.get(key) or {}))
    report = {
        "generated_at": utc_now(),
        "handoff": str(handoff_path),
        "complete": all(check.get("status") == "pass" for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_handoff_verification_markdown(report), encoding="utf-8")
    return report


def write_v1_1_baseline_transfer_manifest(
    handoff_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    files_out_path: Path | None = None,
    include_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    manifest = read_json(handoff_path, {})
    inputs = manifest.get("inputs") if isinstance(manifest, dict) else {}
    fingerprints = manifest.get("input_fingerprints") if isinstance(manifest, dict) else {}
    file_paths: list[str] = [str(handoff_path)]
    for collection_key in ("derived_samples", "base_samples"):
        for item in ((fingerprints or {}).get(collection_key) or {}).get("files") or []:
            path_value = item.get("path")
            if path_value:
                file_paths.append(str(path_value))
    for key in ("assembly_manifest", "corpus_manifest"):
        path_value = ((fingerprints or {}).get(key) or {}).get("path") or (inputs or {}).get(key)
        if path_value:
            file_paths.append(str(path_value))

    benchmark_split_paths = benchmark_task_split_paths(inputs)
    file_paths.extend(str(path) for path in benchmark_split_paths)
    benchmark_split_file_fingerprints = [file_fingerprint(path) for path in benchmark_split_paths]

    corpus_manifest_value = (inputs or {}).get("corpus_manifest")
    corpus_manifest_path = Path(str(corpus_manifest_value)) if corpus_manifest_value else None
    corpus_rows = read_jsonl(corpus_manifest_path) if corpus_manifest_path and corpus_manifest_path.is_file() else []
    chunk_entries: list[dict[str, Any]] = []
    for row in corpus_rows:
        chunks_path_value = row.get("chunks_path")
        if not chunks_path_value:
            continue
        chunks_path = Path(str(chunks_path_value))
        exists = chunks_path.exists()
        stat = chunks_path.stat() if exists and chunks_path.is_file() else None
        chunk_entries.append(
            {
                "path": str(chunks_path),
                "exists": exists,
                "size_bytes": stat.st_size if stat else 0,
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "status": row.get("status"),
                "chunk_count": int(row.get("chunk_count") or 0),
                "file_count": int(row.get("file_count") or 0),
            }
        )
        file_paths.append(str(chunks_path))
    generated_output_paths = {
        str(path)
        for path in (out_path, markdown_out_path, files_out_path)
        if path is not None
    }
    file_paths.extend(sorted(generated_output_paths))
    included_files, include_entries = expand_transfer_include_paths(include_paths or [])
    file_paths.extend(included_files)
    unique_files = sorted(dict.fromkeys(file_paths))
    missing_files = [path for path in unique_files if path not in generated_output_paths and not Path(path).exists()]
    chunk_missing = [entry["path"] for entry in chunk_entries if not entry["exists"]]
    included_file_fingerprints = [
        file_fingerprint(Path(path))
        for path in included_files
        if str(path) not in generated_output_paths
    ]
    unfingerprinted_included_files = [path for path in included_files if str(path) in generated_output_paths]
    report = {
        "generated_at": utc_now(),
        "handoff": str(handoff_path),
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "complete": not missing_files and bool(corpus_rows),
        "file_count": len(unique_files),
        "files": unique_files,
        "generated_output_files": sorted(generated_output_paths),
        "missing_files": missing_files,
        "corpus_rows": len(corpus_rows),
        "chunk_files": len(chunk_entries),
        "chunk_files_missing": chunk_missing,
        "chunk_count": sum(entry["chunk_count"] for entry in chunk_entries),
        "chunk_file_size_bytes": sum(entry["size_bytes"] for entry in chunk_entries),
        "chunk_entries": chunk_entries,
        "benchmark_split_files": [str(path) for path in benchmark_split_paths],
        "benchmark_split_file_fingerprints": benchmark_split_file_fingerprints,
        "include_entries": include_entries,
        "included_files": included_files,
        "included_file_count": len(included_files),
        "included_file_fingerprints": included_file_fingerprints,
        "unfingerprinted_included_files": unfingerprinted_included_files,
        "notes": [
            "Copy every path in files before running external baselines.",
            "Run v1-1-verify-handoff after transfer and before starting GPU/API embedding jobs.",
            "Use --include for helper/source files such as README.md, pyproject.toml, src/agent_retrieval_bench, sample shard manifests, shard command reports, and runbooks.",
            "Included source/helper files are fingerprinted for transfer auditing except the generated manifest output files themselves.",
            "Existing per-task benchmark split files are included and fingerprinted when present; eval commands still use samples.jsonl as the canonical sample source.",
            "external_runner_copy_packet*.json/.md files are intentionally excluded from the transfer manifest because they record the current bundle hash.",
            "Chunk file contents are not rehashed here by default; corpus_manifest records the expected chunk counts.",
        ],
    }
    report = preserve_generated_at_if_report_unchanged(report, out_path)
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_transfer_manifest_markdown(report), encoding="utf-8")
    if files_out_path:
        ensure_parent(files_out_path)
        files_out_path.write_text("\n".join(unique_files) + "\n", encoding="utf-8")
    return report


def benchmark_task_split_paths(inputs: dict[str, Any] | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for key in ("derived", "base_derived"):
        root_value = (inputs or {}).get(key)
        if not root_value:
            continue
        root = Path(str(root_value))
        if not root.is_dir():
            continue
        for task in V1_1_TASKS:
            split_path = root / f"{task}.jsonl"
            if not split_path.is_file():
                continue
            split_key = str(split_path)
            if split_key in seen:
                continue
            seen.add(split_key)
            paths.append(split_path)
    return sorted(paths, key=str)


def expand_transfer_include_paths(include_paths: Iterable[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    files: list[str] = []
    entries: list[dict[str, Any]] = []
    for include_path in include_paths:
        path = Path(include_path)
        if ignored_transfer_include_path(path):
            entries.append({"path": str(path), "exists": path.exists(), "type": "ignored", "files": []})
        elif path.is_file():
            files.append(str(path))
            entries.append({"path": str(path), "exists": True, "type": "file", "files": [str(path)]})
        elif path.is_dir():
            child_files = sorted(str(child) for child in path.rglob("*") if child.is_file() and not ignored_transfer_include_path(child))
            files.extend(child_files)
            entries.append({"path": str(path), "exists": True, "type": "directory", "files": child_files})
        else:
            files.append(str(path))
            entries.append({"path": str(path), "exists": False, "type": "missing", "files": [str(path)]})
    return sorted(dict.fromkeys(files)), entries


def ignored_transfer_include_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"} or external_runner_copy_packet_path(path)


def external_runner_copy_packet_path(path: Path | str) -> bool:
    candidate = Path(str(path))
    return candidate.name.startswith("external_runner_copy_packet") and candidate.suffix in {".json", ".md"}


def verify_v1_1_baseline_transfer_manifest(
    manifest_path: Path,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path, {})
    files = [str(path) for path in (manifest.get("files") if isinstance(manifest, dict) else []) or []]
    missing_files = [path for path in files if not Path(path).exists()]
    checks: list[dict[str, Any]] = [
        {
            "name": "transfer_files_exist",
            "status": "pass" if files and not missing_files else "fail",
            "file_count": len(files),
            "missing_file_count": len(missing_files),
            "missing_files": missing_files[:100],
            "missing_files_truncated": len(missing_files) > 100,
        }
    ]

    chunk_mismatches: list[dict[str, Any]] = []
    chunk_entries = (manifest.get("chunk_entries") if isinstance(manifest, dict) else []) or []
    for entry in chunk_entries:
        path = Path(str(entry.get("path") or ""))
        expected_size = entry.get("size_bytes")
        if not path.is_file():
            chunk_mismatches.append({"path": str(path), "reason": "missing_or_not_file", "expected_size_bytes": expected_size})
            continue
        actual_size = path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            chunk_mismatches.append(
                {
                    "path": str(path),
                    "reason": "size_mismatch",
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                }
            )
    checks.append(
        {
            "name": "chunk_files_exist_with_recorded_size",
            "status": "pass" if not chunk_mismatches else "fail",
            "chunk_file_count": len(chunk_entries),
            "mismatch_count": len(chunk_mismatches),
            "mismatches": chunk_mismatches[:100],
            "mismatches_truncated": len(chunk_mismatches) > 100,
        }
    )

    included_files = (manifest.get("included_files") if isinstance(manifest, dict) else []) or []
    helper_fingerprints = (manifest.get("included_file_fingerprints") if isinstance(manifest, dict) else []) or []
    helper_mismatches: list[dict[str, Any]] = []
    if included_files and not helper_fingerprints:
        helper_mismatches.append({"path": None, "mismatches": ["included files exist but included_file_fingerprints is empty"]})
    for expected in helper_fingerprints:
        path_value = expected.get("path")
        actual = file_fingerprint(Path(str(path_value))) if path_value else {"path": path_value, "exists": False, "type": "missing"}
        mismatches = compare_fields(expected, actual, ("exists", "type", "size_bytes", "sha256", "line_count"))
        if mismatches:
            helper_mismatches.append({"path": path_value, "mismatches": mismatches, "expected": expected, "actual": actual})
    checks.append(
        {
            "name": "included_file_fingerprints_match",
            "status": "pass" if not helper_mismatches else "fail",
            "included_file_count": len(included_files),
            "fingerprint_count": len(helper_fingerprints),
            "mismatch_count": len(helper_mismatches),
            "mismatches": helper_mismatches[:100],
            "mismatches_truncated": len(helper_mismatches) > 100,
        }
    )

    split_fingerprints = (manifest.get("benchmark_split_file_fingerprints") if isinstance(manifest, dict) else []) or []
    split_mismatches: list[dict[str, Any]] = []
    for expected in split_fingerprints:
        path_value = expected.get("path")
        actual = file_fingerprint(Path(str(path_value))) if path_value else {"path": path_value, "exists": False, "type": "missing"}
        mismatches = compare_fields(expected, actual, ("exists", "type", "size_bytes", "sha256", "line_count"))
        if mismatches:
            split_mismatches.append({"path": path_value, "mismatches": mismatches, "expected": expected, "actual": actual})
    checks.append(
        {
            "name": "benchmark_split_file_fingerprints_match",
            "status": "pass" if not split_mismatches else "fail",
            "benchmark_split_file_count": len(split_fingerprints),
            "mismatch_count": len(split_mismatches),
            "mismatches": split_mismatches[:100],
            "mismatches_truncated": len(split_mismatches) > 100,
        }
    )

    report = {
        "generated_at": utc_now(),
        "manifest": str(manifest_path),
        "complete": all(check.get("status") == "pass" for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
        "notes": [
            "This verifier checks path presence, chunk file sizes, and included source/helper-file fingerprints.",
            "It checks per-task benchmark split-file fingerprints when the transfer manifest records them.",
            "It does not hash corpus chunk contents; use v1-1-verify-handoff to verify benchmark input fingerprints.",
        ],
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_transfer_manifest_verification_markdown(report), encoding="utf-8")
    return report


def create_v1_1_baseline_transfer_bundle(
    manifest_path: Path,
    bundle_path: Path,
    checksum_path: Path | None = None,
    archive_members_path: Path | None = None,
    bundle_files_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    verify_out_path: Path | None = None,
    verify_markdown_out_path: Path | None = None,
    compression: str = "zstd -T0 -3",
    test_compression: bool = True,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    root = work_dir or Path.cwd()
    manifest = read_json(resolve_work_path(manifest_path, root), {})
    expected_files = sorted(dict.fromkeys(str(path) for path in (manifest.get("files") if isinstance(manifest, dict) else []) or []))
    transfer_unpack_scripts = [
        path
        for path in expected_files
        if Path(path).name.startswith("unpack_") and Path(path).name.endswith("_transfer_bundle.sh")
    ]
    checksum = checksum_path or Path(f"{bundle_path}.sha256")
    archive_members = archive_members_path or Path(f"{bundle_path}.members")
    bundle_files = bundle_files_path or Path(f"{bundle_path}.files")

    missing_files = [path for path in expected_files if not resolve_manifest_member_path(path, root).is_file()]
    unsafe_members = unsafe_transfer_member_paths(expected_files)
    copy_packet_members = [path for path in expected_files if external_runner_copy_packet_path(path)]
    non_regular_files = [
        path
        for path in expected_files
        if resolve_manifest_member_path(path, root).exists() and resolve_manifest_member_path(path, root).is_symlink()
    ]
    checks: list[dict[str, Any]] = [
        {
            "name": "manifest_files_present",
            "status": "pass" if expected_files and not missing_files else "fail",
            "manifest": str(manifest_path),
            "file_count": len(expected_files),
            "missing_file_count": len(missing_files),
            "missing_files": missing_files[:100],
            "missing_files_truncated": len(missing_files) > 100,
        },
        {
            "name": "manifest_members_are_safe_paths",
            "status": "pass" if expected_files and not unsafe_members else "fail",
            "unsafe_member_count": len(unsafe_members),
            "unsafe_members": unsafe_members[:100],
            "members_truncated": len(unsafe_members) > 100,
        },
        {
            "name": "manifest_excludes_external_runner_copy_packet",
            "status": "pass" if not copy_packet_members else "fail",
            "copy_packet_member_count": len(copy_packet_members),
            "copy_packet_members": copy_packet_members[:100],
            "members_truncated": len(copy_packet_members) > 100,
        },
        {
            "name": "manifest_members_are_regular_files",
            "status": "pass" if expected_files and not non_regular_files else "fail",
            "non_regular_member_count": len(non_regular_files),
            "non_regular_members": non_regular_files[:100],
            "members_truncated": len(non_regular_files) > 100,
        },
    ]

    bundle_created = False
    bundle_error = None
    tar_stdout = ""
    tar_stderr = ""
    if all(check["status"] == "pass" for check in checks):
        resolved_bundle_files = resolve_work_path(bundle_files, root)
        ensure_parent(resolved_bundle_files)
        resolved_bundle_files.write_text("\n".join(expected_files) + "\n", encoding="utf-8")
        resolved_bundle = resolve_work_path(bundle_path, root)
        ensure_parent(resolved_bundle)
        try:
            tar_result = subprocess.run(
                [
                    "tar",
                    f"--files-from={str(resolved_bundle_files)}",
                    f"--use-compress-program={compression}",
                    "-cf",
                    str(resolved_bundle),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            bundle_created = tar_result.returncode == 0 and resolved_bundle.is_file()
            tar_stdout = tar_result.stdout.strip()
            tar_stderr = tar_result.stderr.strip()
            if not bundle_created:
                bundle_error = tar_stderr or tar_stdout or f"tar exited with status {tar_result.returncode}"
        except FileNotFoundError as error:
            bundle_error = f"missing_{error.filename}_executable"
    else:
        bundle_error = "preflight_failed"
    checks.append(
        {
            "name": "bundle_created",
            "status": "pass" if bundle_created else "fail",
            "bundle": str(bundle_path),
            "bundle_files": str(bundle_files),
            "compression": compression,
            "error": bundle_error,
            "stdout": tar_stdout,
            "stderr": tar_stderr,
        }
    )

    checksum_written = False
    checksum_error = None
    expected_sha256 = None
    if bundle_created:
        resolved_bundle = resolve_work_path(bundle_path, root)
        expected_sha256 = file_fingerprint(resolved_bundle).get("sha256")
        if expected_sha256:
            resolved_checksum = resolve_work_path(checksum, root)
            ensure_parent(resolved_checksum)
            resolved_checksum.write_text(f"{expected_sha256}  {bundle_path}\n", encoding="utf-8")
            checksum_written = True
        else:
            checksum_error = "bundle_sha256_unavailable"
    else:
        checksum_error = "bundle_not_created"
    checks.append(
        {
            "name": "bundle_checksum_written",
            "status": "pass" if checksum_written else "fail",
            "checksum": str(checksum),
            "sha256": expected_sha256,
            "error": checksum_error,
        }
    )

    verification: dict[str, Any] | None = None
    if checksum_written:
        verification = verify_v1_1_baseline_transfer_bundle(
            bundle_path=verify_path_for_work_dir(bundle_path, root),
            manifest_path=verify_path_for_work_dir(manifest_path, root),
            checksum_path=verify_path_for_work_dir(checksum, root),
            archive_members_path=verify_path_for_work_dir(archive_members, root),
            out_path=verify_path_for_work_dir(verify_out_path, root) if verify_out_path else None,
            markdown_out_path=verify_path_for_work_dir(verify_markdown_out_path, root) if verify_markdown_out_path else None,
            test_compression=test_compression,
        )
    checks.append(
        {
            "name": "bundle_verification_complete",
            "status": "pass" if verification and verification.get("complete") else "fail",
            "verify_out": str(verify_out_path) if verify_out_path else None,
            "verify_markdown_out": str(verify_markdown_out_path) if verify_markdown_out_path else None,
            "failed_checks": verification.get("failed_checks") if verification else ["bundle_not_verified"],
        }
    )

    bundle_fingerprint = file_fingerprint(resolve_work_path(bundle_path, root))
    external_acceptance = build_v1_1_transfer_bundle_external_acceptance(manifest, root)
    report = {
        "generated_at": utc_now(),
        "complete": all(check.get("status") == "pass" for check in checks),
        "manifest": str(manifest_path),
        "handoff": manifest.get("handoff") if isinstance(manifest, dict) else None,
        "work_dir": str(root),
        "bundle": str(bundle_path),
        "checksum": str(checksum),
        "archive_members": str(archive_members),
        "bundle_files": str(bundle_files),
        "report": str(out_path) if out_path else None,
        "markdown": str(markdown_out_path) if markdown_out_path else None,
        "verify_report": str(verify_out_path) if verify_out_path else None,
        "verify_markdown": str(verify_markdown_out_path) if verify_markdown_out_path else None,
        "compression": compression,
        "test_compression": test_compression,
        "file_count": len(expected_files),
        "sha256": bundle_fingerprint.get("sha256"),
        "size_bytes": bundle_fingerprint.get("size_bytes"),
        "missing_files": missing_files,
        "unsafe_members": unsafe_members,
        "copy_packet_members": copy_packet_members,
        "transfer_unpack_scripts": transfer_unpack_scripts,
        "external_acceptance": external_acceptance,
        "bundle_fingerprint": bundle_fingerprint,
        "verification": verification,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
        "notes": [
            "This command creates the transfer bundle directly from the transfer manifest file list.",
            "external_runner_copy_packet*.json/.md files are rejected because they record the current bundle hash and would create self-referential drift.",
            "The generated checksum, archive member list, and archive member types are verified by v1-1-verify-transfer-bundle before the command is complete.",
            "Run v1-1-verify-transfer-manifest after unpacking on the external runner to check local file presence, chunk sizes, and helper/source fingerprints.",
        ],
    }
    if out_path:
        write_json(resolve_work_path(out_path, root), report)
    if markdown_out_path:
        resolved_markdown = resolve_work_path(markdown_out_path, root)
        ensure_parent(resolved_markdown)
        resolved_markdown.write_text(render_v1_1_baseline_transfer_bundle_markdown(report), encoding="utf-8")
    return report


def build_v1_1_transfer_bundle_external_acceptance(manifest: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    handoff_path = str(manifest.get("handoff") or "")
    handoff_payload: dict[str, Any] = {}
    if handoff_path:
        handoff_payload = read_json(resolve_manifest_member_path(handoff_path, work_dir), {})
    finalization = handoff_payload.get("finalization") if isinstance(handoff_payload, dict) else {}
    return_manifest_path = str(finalization.get("return_manifest") or "") if isinstance(finalization, dict) else ""
    return_manifest: dict[str, Any] = {}
    if return_manifest_path:
        return_manifest = read_json(resolve_manifest_member_path(return_manifest_path, work_dir), {})

    jobs = handoff_payload.get("jobs") if isinstance(handoff_payload, dict) else []
    external_baselines: list[str] = []
    required_return_files: list[str] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        baseline = str(job.get("baseline") or "")
        if baseline:
            external_baselines.append(baseline)
        artifacts = job.get("artifacts") if isinstance(job.get("artifacts"), dict) else {}
        for artifact_name in ("details", "summary"):
            artifact_path = artifacts.get(artifact_name)
            if artifact_path:
                required_return_files.append(str(artifact_path))
    manifest_required = return_manifest.get("required_files") if isinstance(return_manifest, dict) else []
    if manifest_required:
        required_return_files = [str(path) for path in manifest_required]
    manifest_required_by_baseline = (
        return_manifest.get("required_files_by_baseline") if isinstance(return_manifest, dict) else None
    )
    if isinstance(manifest_required_by_baseline, dict) and manifest_required_by_baseline:
        required_return_files_by_baseline = {
            str(baseline): [str(path) for path in paths]
            for baseline, paths in manifest_required_by_baseline.items()
            if isinstance(paths, list)
        }
    else:
        required_return_files_by_baseline = group_return_files_by_baseline(required_return_files, external_baselines)

    files = [str(path) for path in (manifest.get("files") if isinstance(manifest, dict) else []) or []]
    run_scripts = sorted(path for path in files if Path(path).name.startswith("run_v19_") and Path(path).suffix == ".sh")
    package_scripts = sorted(path for path in files if Path(path).name.startswith("package_v19_") and Path(path).suffix == ".sh")
    apply_scripts = sorted(path for path in files if Path(path).name.startswith("apply_v19_") and Path(path).suffix == ".sh")
    return {
        "handoff": handoff_path or None,
        "return_manifest": return_manifest_path or None,
        "external_baselines": external_baselines,
        "required_return_files": required_return_files,
        "required_return_files_by_baseline": required_return_files_by_baseline,
        "required_return_file_count": len(required_return_files),
        "run_scripts": run_scripts,
        "return_packaging_scripts": package_scripts,
        "return_apply_scripts": apply_scripts,
        "completion_gate": (
            "Apply a verified return bundle and rerun v1-1-finalize-baselines until completion_audit_v19.json "
            "reports overall_status=complete. The full generated run/apply scripts refresh the compact return acceptance "
            "report with --require-complete, so final acceptance exits nonzero until this gate is complete."
        ),
    }


def resolve_work_path(path: Path, work_dir: Path) -> Path:
    return path if path.is_absolute() else work_dir / path


def verify_path_for_work_dir(path: Path, work_dir: Path) -> Path:
    return path if path.is_absolute() or work_dir == Path.cwd() else work_dir / path


def resolve_manifest_member_path(path: str, work_dir: Path) -> Path:
    member = Path(path)
    return member if member.is_absolute() else work_dir / member


def unsafe_transfer_member_paths(paths: Iterable[str]) -> list[str]:
    unsafe = []
    for path in paths:
        posix = PurePosixPath(path)
        if posix.is_absolute() or ".." in posix.parts:
            unsafe.append(path)
    return unsafe


def render_v1_1_baseline_transfer_bundle_markdown(report: dict[str, Any]) -> str:
    fingerprint = report.get("bundle_fingerprint") or {}
    bundle = str(report.get("bundle") or "")
    checksum = str(report.get("checksum") or "")
    manifest = str(report.get("manifest") or "")
    handoff = str(report.get("handoff") or "")
    local_bundle = Path(bundle).name if bundle else "baseline_transfer_bundle.tar.zst"
    local_checksum = Path(checksum).name if checksum else "baseline_transfer_bundle.tar.zst.sha256"
    lines = [
        "# V1.1 Baseline Transfer Bundle",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Manifest: `{manifest}`",
        f"Handoff: `{handoff}`",
        f"Bundle: `{bundle}`",
        f"Checksum: `{checksum}`",
        f"Bundle files: `{report.get('bundle_files')}`",
        f"Archive members: `{report.get('archive_members')}`",
        f"SHA256: `{fingerprint.get('sha256')}`",
        f"Size bytes: `{fingerprint.get('size_bytes')}`",
        f"Compression: `{report.get('compression')}`",
        f"File count: `{report.get('file_count')}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        status = "PASS" if check.get("status") == "pass" else "FAIL"
        lines.append(f"### {status} - {check.get('name')}")
        lines.append("")
        for key, value in check.items():
            if key in {"name", "status"}:
                continue
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    lines.extend(
        [
            "## External Runner",
            "",
        ]
    )
    acceptance = report.get("external_acceptance") or {}
    if acceptance:
        lines.extend(
            [
                "### Acceptance Checklist",
                "",
                f"- External baselines: `{json.dumps(acceptance.get('external_baselines') or [])}`",
                f"- Required final return files: `{int(acceptance.get('required_return_file_count') or 0)}`",
                f"- Return manifest: `{acceptance.get('return_manifest')}`",
                f"- Completion gate: `{acceptance.get('completion_gate')}`",
                "",
            ]
        )
        required_return_files = acceptance.get("required_return_files") or []
        if required_return_files:
            lines.extend(["Required files:", ""])
            for path in required_return_files:
                lines.append(f"- `{path}`")
            lines.append("")
        required_by_baseline = acceptance.get("required_return_files_by_baseline")
        if isinstance(required_by_baseline, dict) and required_by_baseline:
            lines.extend(["Required files by baseline:", ""])
            for baseline, paths in sorted(required_by_baseline.items()):
                path_list = list(paths or [])
                lines.append(f"- `{baseline}`: `{len(path_list)}`")
                for path in path_list:
                    lines.append(f"  - `{path}`")
            lines.append("")
        run_scripts = acceptance.get("run_scripts") or []
        if run_scripts:
            lines.extend(["Run scripts:", ""])
            for path in run_scripts:
                lines.append(f"- `{path}`")
            lines.append("")
        package_scripts = acceptance.get("return_packaging_scripts") or []
        apply_scripts = acceptance.get("return_apply_scripts") or []
        if package_scripts or apply_scripts:
            lines.extend(["Return bundle scripts:", ""])
            for path in package_scripts:
                lines.append(f"- `{path}`")
            for path in apply_scripts:
                lines.append(f"- `{path}`")
            lines.append("")
    unpack_scripts = report.get("transfer_unpack_scripts") or []
    lines.extend(["### Bootstrap", ""])
    if unpack_scripts:
        lines.extend(
            [
                "Preferred bootstrap:",
                "",
                "```bash",
                f"bash {unpack_scripts[0]} {bundle} {checksum} agent-retrieval-bench-v1_1-transfer",
                "```",
                "",
                "If the bootstrap script is copied beside the bundle and checksum, pass local basenames instead:",
                "",
                "```bash",
                f"bash {Path(str(unpack_scripts[0])).name} {local_bundle} {local_checksum} agent-retrieval-bench-v1_1-transfer",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "Manual fallback, preserving the generated bundle path recorded inside the checksum sidecar:",
            "",
            "```bash",
            f"sha256sum -c {checksum}",
            f"tar -xf {bundle}",
            f"PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-verify-transfer-manifest --manifest {manifest}",
        ]
    )
    if handoff:
        lines.append(f"PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-verify-handoff --handoff {handoff}")
    lines.extend(
        [
            "```",
            "",
            "If only basename files were copied, do not run `sha256sum -c` against the generated checksum sidecar directly; use the preferred bootstrap or rewrite the checked path before manual verification.",
            "",
        ]
    )
    return "\n".join(lines)


def write_v1_1_external_runner_copy_packet(
    transfer_bundle_report_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    unpack_script_path: Path = Path("data/reports/v1_1/unpack_v19_transfer_bundle.sh"),
    bundle_path: Path | None = None,
    checksum_path: Path | None = None,
    bundle_report_markdown_path: Path | None = None,
    destination: str = "agent-retrieval-bench-v1_1-v19-transfer",
    full_runner_path: Path = Path("data/reports/v1_1/run_v19_baseline_shards.sh"),
    gpu_runner_path: Path = Path("data/reports/v1_1/run_v19_gpu_baseline_shards.sh"),
    voyage_runner_path: Path = Path("data/reports/v1_1/run_v19_voyage_baseline_shards.sh"),
    local_apply_script_path: Path = Path("data/reports/v1_1/apply_v19_return_artifacts.sh"),
    completion_json_path: Path = Path("data/reports/v1_1/completion_audit_v19.json"),
) -> dict[str, Any]:
    bundle_report = read_json(transfer_bundle_report_path, {})
    if not isinstance(bundle_report, dict):
        bundle_report = {}
    fingerprint = bundle_report.get("bundle_fingerprint") if isinstance(bundle_report.get("bundle_fingerprint"), dict) else {}
    resolved_bundle = bundle_path or Path(str(bundle_report.get("bundle") or "data/reports/v1_1/baseline_transfer_bundle_v19.tar.zst"))
    resolved_checksum = checksum_path or Path(str(bundle_report.get("checksum") or f"{resolved_bundle}.sha256"))
    resolved_report_markdown = bundle_report_markdown_path or Path(
        str(bundle_report.get("markdown") or f"{transfer_bundle_report_path.with_suffix('.md')}")
    )
    sha256 = str(bundle_report.get("sha256") or fingerprint.get("sha256") or "")
    size_bytes = bundle_report.get("size_bytes") if bundle_report.get("size_bytes") is not None else fingerprint.get("size_bytes")
    file_count = bundle_report.get("file_count")
    external_acceptance = bundle_report.get("external_acceptance") if isinstance(bundle_report.get("external_acceptance"), dict) else {}
    required_baselines = [str(item) for item in (external_acceptance.get("external_baselines") or [])] or list(REQUIRED_V1_1_BASELINES[2:])
    required_return_files = [str(item) for item in (external_acceptance.get("required_return_files") or [])]
    required_return_files_by_baseline_payload = external_acceptance.get("required_return_files_by_baseline")
    if isinstance(required_return_files_by_baseline_payload, dict) and required_return_files_by_baseline_payload:
        required_return_files_by_baseline = {
            str(baseline): [str(path) for path in paths]
            for baseline, paths in required_return_files_by_baseline_payload.items()
            if isinstance(paths, list)
        }
    else:
        required_return_files_by_baseline = group_return_files_by_baseline(required_return_files, required_baselines)
    copy_files = [str(unpack_script_path), str(resolved_bundle), str(resolved_checksum)]
    first_command = (
        f"bash {shlex.quote(unpack_script_path.name)} "
        f"{shlex.quote(resolved_bundle.name)} {shlex.quote(resolved_checksum.name)} {shlex.quote(destination)}"
    )
    sender_preflight_command = (
        "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-external-runner-preflight "
        f"--copy-packet {shlex.quote(str(out_path))}"
    )
    split_runner_scripts: dict[str, str] = {}
    if any(baseline in required_baselines for baseline in ("jina-code-embeddings-0.5b", "qwen3-embedding-4b")):
        split_runner_scripts["cuda_jina_qwen"] = str(gpu_runner_path)
    if "voyage-code-3" in required_baselines:
        split_runner_scripts["api_voyage"] = str(voyage_runner_path)
    checks = [
        {"name": "transfer_bundle_report_complete", "status": "pass" if bundle_report.get("complete") else "fail"},
        {"name": "bundle_sha256_recorded", "status": "pass" if re.fullmatch(r"[0-9a-f]{64}", sha256) else "fail"},
        {"name": "bundle_size_recorded", "status": "pass" if isinstance(size_bytes, int) and size_bytes > 0 else "fail"},
        {"name": "transfer_file_count_recorded", "status": "pass" if isinstance(file_count, int) and file_count > 0 else "fail"},
        {
            "name": "copy_files_exist",
            "status": "pass" if all(Path(path).exists() for path in copy_files) else "fail",
            "missing_files": [path for path in copy_files if not Path(path).exists()],
        },
    ]
    report = {
        "generated_at": utc_now(),
        "complete": all(check.get("status") == "pass" for check in checks),
        "transfer_bundle_report": str(transfer_bundle_report_path),
        "bundle_generated_at": bundle_report.get("generated_at"),
        "bundle_path": str(resolved_bundle),
        "checksum_path": str(resolved_checksum),
        "unpack_script_path": str(unpack_script_path),
        "bundle_report_path": str(resolved_report_markdown),
        "bundle_sha256": sha256,
        "bundle_size_bytes": size_bytes,
        "transfer_file_count": file_count,
        "copy_to_external_runner": copy_files,
        "sender_preflight_command": sender_preflight_command,
        "external_runner_first_command": first_command,
        "external_runner_after_unpack": [
            f"cd {destination}",
            "python -m venv .venv && . .venv/bin/activate",
            "pip install -e '.[embedding]'",
            "PYTHONPATH=src python -m agent_retrieval_bench.cli v1-1-external-runner-preflight",
            str(full_runner_path),
        ],
        "split_runner_scripts": split_runner_scripts,
        "required_external_baselines": required_baselines,
        "required_return_file_count": external_acceptance.get("required_return_file_count") or len(required_return_files),
        "required_return_files": required_return_files,
        "required_return_files_by_baseline": required_return_files_by_baseline,
        "local_apply_after_return": str(local_apply_script_path),
        "completion_json": str(completion_json_path),
        "completion_required_status": "overall_status=complete",
        "completion_gate": f"{completion_json_path} must report overall_status=complete after applying the verified return bundle.",
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
        "note": (
            "This copy packet is intentionally not included in the transfer bundle, so recording the current bundle hash here "
            "does not create self-referential hash drift."
        ),
    }
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_external_runner_copy_packet_markdown(report), encoding="utf-8")
    return report


def render_v1_1_external_runner_copy_packet_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 V19 External Runner Copy Packet",
        "",
        str(report.get("note") or ""),
        "",
        "## Copy These Files",
        "",
    ]
    for path in report.get("copy_to_external_runner") or []:
        lines.append(f"- `{path}`")
    required_return_files = report.get("required_return_files") or []
    if required_return_files:
        lines.extend(["", "## Required Return Files", ""])
        for path in required_return_files:
            lines.append(f"- `{path}`")
    required_by_baseline = report.get("required_return_files_by_baseline")
    if isinstance(required_by_baseline, dict) and required_by_baseline:
        lines.extend(["", "## Required Return Files By Baseline", ""])
        for baseline, paths in sorted(required_by_baseline.items()):
            lines.append(f"- `{baseline}`: `{len(paths)}`")
            for path in paths:
                lines.append(f"  - `{path}`")
    lines.extend(
        [
            "",
            "## Sender-Side Check",
            "",
            "Run this before copying the transfer files:",
            "",
            "```bash",
            str(report.get("sender_preflight_command") or ""),
            "```",
            "",
            "## Current Bundle",
            "",
            f"- Bundle report: `{report.get('bundle_report_path')}`",
            f"- Bundle generated at: `{report.get('bundle_generated_at')}`",
            f"- SHA256: `{report.get('bundle_sha256')}`",
            f"- Size bytes: `{report.get('bundle_size_bytes')}`",
            f"- Transfer files: `{report.get('transfer_file_count')}`",
            "",
            "## First External Command",
            "",
            "```bash",
            str(report.get("external_runner_first_command") or ""),
            "```",
            "",
            "## Then Run",
            "",
            "```bash",
        ]
    )
    for command in report.get("external_runner_after_unpack") or []:
        if "&&" in str(command):
            lines.extend(part.strip() for part in str(command).split("&&") if part.strip())
        else:
            lines.append(str(command))
    lines.extend(["```", ""])
    split = report.get("split_runner_scripts") if isinstance(report.get("split_runner_scripts"), dict) else {}
    if split:
        if split.get("cuda_jina_qwen"):
            lines.append(f"Use `{split.get('cuda_jina_qwen')}` for required Jina/Qwen GPU runs.")
        if split.get("api_voyage"):
            lines.append(f"Use `{split.get('api_voyage')}` only for an explicit optional Voyage run.")
        lines.append("")
    lines.append(
        f"After the return bundle is copied back to the reporting checkout, run `{report.get('local_apply_after_return')}`. "
        f"The final gate is `{report.get('completion_json')}` reporting `{report.get('completion_required_status')}` "
        "after applying the verified return bundle."
    )
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_transfer_unpack_script(
    out_path: Path,
    bundle_path: Path = Path("data/reports/v1_1/baseline_transfer_bundle.tar.zst"),
    checksum_path: Path | None = None,
    manifest_path: Path = Path("data/reports/v1_1/baseline_transfer_manifest.json"),
    handoff_path: Path = Path("data/reports/v1_1/baseline_handoff.json"),
    markdown_out_path: Path | None = None,
    destination: str = "agent-retrieval-bench-v1_1-transfer",
    transfer_verify_path: Path = Path("data/reports/v1_1/baseline_transfer_unpack_smoke.json"),
    transfer_verify_markdown_path: Path | None = None,
    handoff_verify_path: Path = Path("data/reports/v1_1/baseline_handoff_unpack_smoke.json"),
    handoff_verify_markdown_path: Path | None = None,
) -> dict[str, Any]:
    checksum = checksum_path or Path(f"{bundle_path}.sha256")
    transfer_verify_markdown = transfer_verify_markdown_path or Path(f"{transfer_verify_path}.md")
    handoff_verify_markdown = handoff_verify_markdown_path or Path(f"{handoff_verify_path}.md")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by v1-1-baseline-transfer-unpack-script at {utc_now()}",
        f"default_bundle={shlex.quote(str(bundle_path))}",
        f"default_checksum={shlex.quote(str(checksum))}",
        f"default_dest={shlex.quote(destination)}",
        'bundle="${1:-$default_bundle}"',
        'checksum="${2:-$default_checksum}"',
        'dest="${3:-$default_dest}"',
        "",
        'if [[ ! -f "$bundle" ]]; then',
        '  echo "Missing transfer bundle: $bundle" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ ! -f "$checksum" ]]; then',
        '  echo "Missing transfer bundle checksum: $checksum" >&2',
        "  exit 1",
        "fi",
        "",
        "for tool in sha256sum tar zstd python3; do",
        '  if ! command -v "$tool" >/dev/null 2>&1; then',
        '    echo "Missing required tool: $tool" >&2',
        "    exit 1",
        "  fi",
        "done",
        "",
        'expected_sha256="$(awk \'NF {print $1; exit}\' "$checksum")"',
        'actual_sha256="$(sha256sum "$bundle" | awk \'{print $1}\')"',
        'if [[ -z "$expected_sha256" || "$actual_sha256" != "$expected_sha256" ]]; then',
        '  echo "Transfer bundle checksum mismatch" >&2',
        '  echo "expected: $expected_sha256" >&2',
        '  echo "actual:   $actual_sha256" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ -e "$dest" ]] && [[ -n "$(find "$dest" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then',
        '  echo "Destination exists and is not empty: $dest" >&2',
        "  exit 1",
        "fi",
        "",
        'member_list="$(mktemp)"',
        'member_type_list="$(mktemp)"',
        "cleanup() {",
        '  rm -f "$member_list" "$member_type_list"',
        "}",
        "trap cleanup EXIT",
        "",
        'if ! tar -tf "$bundle" > "$member_list" 2>/dev/null; then',
        '  tar --use-compress-program=zstd -tf "$bundle" > "$member_list"',
        "fi",
        'if ! tar -tvf "$bundle" > "$member_type_list" 2>/dev/null; then',
        '  tar --use-compress-program=zstd -tvf "$bundle" > "$member_type_list"',
        "fi",
        "python3 - \"$member_list\" \"$member_type_list\" <<'PY'",
        "from pathlib import Path, PurePosixPath",
        "import sys",
        "members = [line for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]",
        "unsafe = []",
        "for member in members:",
        "    posix = PurePosixPath(member)",
        "    if posix.is_absolute() or '..' in posix.parts:",
        "        unsafe.append(member)",
        "if unsafe:",
        "    print('Transfer bundle contains unsafe tar member paths:', file=sys.stderr)",
        "    for member in unsafe:",
        "        print(member, file=sys.stderr)",
        "    raise SystemExit(1)",
        "if not members:",
        "    raise SystemExit('Transfer bundle contains no members.')",
        "print(f'Validated {len(members)} safe transfer bundle member paths.')",
        "non_regular = []",
        "for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines():",
        "    if line and line[0] != '-':",
        "        non_regular.append(line)",
        "if non_regular:",
        "    print('Transfer bundle contains non-regular file members:', file=sys.stderr)",
        "    for line in non_regular:",
        "        print(line, file=sys.stderr)",
        "    raise SystemExit(1)",
        "print(f'Validated {len(members)} regular file transfer bundle members.')",
        "PY",
        "",
        'mkdir -p "$dest"',
        'if ! tar -xf "$bundle" -C "$dest" 2>/dev/null; then',
        '  tar --use-compress-program=zstd -xf "$bundle" -C "$dest"',
        "fi",
        "",
        'cd "$dest"',
        cli_command(
            "v1-1-verify-transfer-manifest",
            "--manifest",
            manifest_path,
            "--out",
            transfer_verify_path,
            "--markdown-out",
            transfer_verify_markdown,
        ),
        cli_command(
            "v1-1-verify-handoff",
            "--handoff",
            handoff_path,
            "--out",
            handoff_verify_path,
            "--markdown-out",
            handoff_verify_markdown,
        ),
        "",
        'echo "Transfer bundle verified and unpacked at: $dest"',
        "",
    ]
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o755)
    report = {
        "generated_at": utc_now(),
        "complete": True,
        "script": str(out_path),
        "markdown": str(markdown_out_path) if markdown_out_path else None,
        "bundle": str(bundle_path),
        "checksum": str(checksum),
        "destination": destination,
        "manifest": str(manifest_path),
        "handoff": str(handoff_path),
        "transfer_verify": str(transfer_verify_path),
        "transfer_verify_markdown": str(transfer_verify_markdown),
        "handoff_verify": str(handoff_verify_path),
        "handoff_verify_markdown": str(handoff_verify_markdown),
        "command_count": 5,
        "notes": [
            "Copy this script beside the prepared transfer bundle and checksum on the external runner.",
            "The checksum sidecar may record the generated repo-relative bundle path; this script reads the expected hash from the first checksum column and verifies the supplied bundle path, so copied basenames work.",
            "The script verifies the checksum, rejects unsafe tar member paths and non-regular tar members, unpacks into a clean directory, and runs transfer-manifest plus handoff verification before GPU/API work.",
            "Run the generated shard script from the printed unpacked checkout path after this script succeeds.",
        ],
    }
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_transfer_unpack_script_markdown(report), encoding="utf-8")
    return report


def render_v1_1_baseline_transfer_unpack_script_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Transfer Unpack Script",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Script: `{report.get('script')}`",
        f"Bundle: `{report.get('bundle')}`",
        f"Checksum: `{report.get('checksum')}`",
        f"Destination: `{report.get('destination')}`",
        f"Manifest: `{report.get('manifest')}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Transfer verify: `{report.get('transfer_verify')}`",
        f"Handoff verify: `{report.get('handoff_verify')}`",
        f"Command count: `{report.get('command_count')}`",
        "",
        "## Usage",
        "",
        "```bash",
        f"bash {report.get('script')} {report.get('bundle')} {report.get('checksum')} {report.get('destination')}",
        "```",
        "",
        "If only the script, bundle, and checksum were copied into one directory, pass local basenames instead.",
        "The script verifies the supplied bundle path against the first hash in the checksum file, so it works even when the checksum sidecar records a repo-relative bundle path.",
        "",
        "## Notes",
        "",
    ]
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def run_tar_read(bundle_path: Path, operation: str) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    attempts = (
        ["tar", operation, str(bundle_path)],
        ["tar", "--use-compress-program=zstd", operation, str(bundle_path)],
    )
    errors = []
    for command in attempts:
        try:
            result = subprocess.run(command, text=True, capture_output=True)
        except FileNotFoundError as error:
            return None, f"missing_{error.filename}_executable"
        if result.returncode == 0:
            return result, None
        errors.append((result.stderr or result.stdout or f"tar exited with status {result.returncode}").strip())
    return result, " | ".join(error for error in errors if error)


def verify_v1_1_baseline_transfer_bundle(
    bundle_path: Path,
    manifest_path: Path,
    checksum_path: Path | None = None,
    archive_members_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    test_compression: bool = True,
) -> dict[str, Any]:
    checksum = checksum_path or Path(f"{bundle_path}.sha256")
    archive_members = archive_members_path or Path(f"{bundle_path}.members")
    manifest = read_json(manifest_path, {})
    expected_files = sorted(str(path) for path in (manifest.get("files") if isinstance(manifest, dict) else []) or [])
    checks: list[dict[str, Any]] = []

    bundle_fingerprint = file_fingerprint(bundle_path)
    expected_checksum = None
    checksum_error = None
    if checksum.is_file():
        parts = checksum.read_text(encoding="utf-8").split()
        expected_checksum = parts[0] if parts else None
    else:
        checksum_error = "missing_checksum_file"
    checksum_matches = bool(
        expected_checksum
        and bundle_fingerprint.get("exists")
        and bundle_fingerprint.get("type") == "file"
        and bundle_fingerprint.get("sha256") == expected_checksum
    )
    checks.append(
        {
            "name": "bundle_checksum_matches",
            "status": "pass" if checksum_matches else "fail",
            "bundle": str(bundle_path),
            "checksum": str(checksum),
            "expected_sha256": expected_checksum,
            "actual_sha256": bundle_fingerprint.get("sha256"),
            "bundle_exists": bool(bundle_fingerprint.get("exists")),
            "bundle_size_bytes": bundle_fingerprint.get("size_bytes"),
            "error": checksum_error,
        }
    )

    if test_compression:
        compression_stdout = ""
        compression_stderr = ""
        if bundle_path.is_file():
            try:
                zstd_result = subprocess.run(["zstd", "-t", str(bundle_path)], text=True, capture_output=True)
                compression_passed = zstd_result.returncode == 0
                compression_stdout = (zstd_result.stdout or "").strip()
                compression_stderr = (zstd_result.stderr or "").strip()
                compression_error = None if compression_passed else (compression_stderr or compression_stdout or "zstd_integrity_test_failed")
            except FileNotFoundError:
                compression_passed = False
                compression_error = "missing_zstd_executable"
        else:
            compression_passed = False
            compression_error = "missing_bundle"
        checks.append(
            {
                "name": "bundle_compression_valid",
                "status": "pass" if compression_passed else "fail",
                "tested": True,
                "stdout": compression_stdout,
                "stderr": compression_stderr,
                "error": compression_error,
            }
        )
    else:
        checks.append({"name": "bundle_compression_valid", "status": "pass", "tested": False, "error": None})

    actual_members: list[str] = []
    member_error = None
    if bundle_path.is_file():
        tar_result, member_error = run_tar_read(bundle_path, "-tf")
        if tar_result is not None and tar_result.returncode == 0:
            actual_members = sorted(line for line in tar_result.stdout.splitlines() if line)
            ensure_parent(archive_members)
            archive_members.write_text("\n".join(actual_members) + ("\n" if actual_members else ""), encoding="utf-8")
    else:
        member_error = "missing_bundle"
    unsafe_members = []
    for member in actual_members:
        posix = PurePosixPath(member)
        if posix.is_absolute() or ".." in posix.parts:
            unsafe_members.append(member)
    checks.append(
        {
            "name": "bundle_members_are_safe_paths",
            "status": "pass" if not unsafe_members and actual_members else "fail",
            "unsafe_member_count": len(unsafe_members),
            "unsafe_members": unsafe_members[:100],
            "members_truncated": len(unsafe_members) > 100,
            "member_count": len(actual_members),
            "error": member_error,
        }
    )
    non_regular_members: list[dict[str, str]] = []
    member_type_error = None
    if bundle_path.is_file():
        tar_types_result, member_type_error = run_tar_read(bundle_path, "-tvf")
        if tar_types_result is not None and tar_types_result.returncode == 0:
            for line in tar_types_result.stdout.splitlines():
                if line and line[0] != "-":
                    non_regular_members.append({"type": line[0], "listing": line})
    else:
        member_type_error = "missing_bundle"
    checks.append(
        {
            "name": "bundle_members_are_regular_files",
            "status": "pass" if actual_members and not non_regular_members and not member_type_error else "fail",
            "non_regular_member_count": len(non_regular_members),
            "non_regular_members": non_regular_members[:100],
            "members_truncated": len(non_regular_members) > 100,
            "member_count": len(actual_members),
            "error": member_type_error,
        }
    )
    copy_packet_members = [member for member in actual_members if external_runner_copy_packet_path(member)]
    checks.append(
        {
            "name": "bundle_excludes_external_runner_copy_packet",
            "status": "pass" if not copy_packet_members else "fail",
            "copy_packet_member_count": len(copy_packet_members),
            "copy_packet_members": copy_packet_members[:100],
            "members_truncated": len(copy_packet_members) > 100,
        }
    )
    missing_members = sorted(set(expected_files) - set(actual_members))
    extra_members = sorted(set(actual_members) - set(expected_files))
    member_list_matches = bool(expected_files) and actual_members == expected_files and not missing_members and not extra_members
    checks.append(
        {
            "name": "bundle_members_match_manifest",
            "status": "pass" if member_list_matches else "fail",
            "manifest": str(manifest_path),
            "archive_members": str(archive_members),
            "expected_member_count": len(expected_files),
            "actual_member_count": len(actual_members),
            "missing_member_count": len(missing_members),
            "extra_member_count": len(extra_members),
            "missing_members": missing_members[:100],
            "extra_members": extra_members[:100],
            "members_truncated": len(missing_members) > 100 or len(extra_members) > 100,
            "error": member_error,
        }
    )

    report = {
        "generated_at": utc_now(),
        "complete": all(check.get("status") == "pass" for check in checks),
        "bundle": str(bundle_path),
        "manifest": str(manifest_path),
        "checksum": str(checksum),
        "archive_members": str(archive_members),
        "sha256": bundle_fingerprint.get("sha256"),
        "size_bytes": bundle_fingerprint.get("size_bytes"),
        "bundle_fingerprint": bundle_fingerprint,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
        "notes": [
            "This verifier checks the transfer bundle checksum, compressed archive integrity, regular-file member types, and exact tar member list against the transfer manifest file list.",
            "external_runner_copy_packet*.json/.md members are rejected because copy packets are intentionally outside the bundle.",
            "Run v1-1-verify-transfer-manifest after unpacking to check local file presence, chunk sizes, and helper/source fingerprints.",
        ],
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_transfer_bundle_verification_markdown(report), encoding="utf-8")
    return report


def render_v1_1_transfer_bundle_verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Transfer Bundle Verification",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Bundle: `{report.get('bundle')}`",
        f"Manifest: `{report.get('manifest')}`",
        f"SHA256: `{report.get('sha256')}`",
        f"Size bytes: `{report.get('size_bytes')}`",
        f"Overall status: `{'complete' if report.get('complete') else 'not_complete'}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        status = "PASS" if check.get("status") == "pass" else "FAIL"
        lines.append(f"### {status} - {check.get('name')}")
        lines.append("")
        for key, value in check.items():
            if key in {"name", "status"}:
                continue
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def render_v1_1_transfer_manifest_verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Transfer Manifest Verification",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Manifest: `{report.get('manifest')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
    ]
    for check in report.get("checks") or []:
        detail = {key: value for key, value in check.items() if key not in {"name", "status", "mismatches", "missing_files"}}
        lines.append(f"| `{check.get('name')}` | `{check.get('status')}` | `{json.dumps(detail, ensure_ascii=False, sort_keys=True)}` |")
    failed = report.get("failed_checks") or []
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for name in failed:
            lines.append(f"- `{name}`")
    lines.extend(["", "## Notes", ""])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_shard_commands(
    handoff_path: Path,
    sample_shards_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    report_dir: Path | None = None,
    use_shard_caches: bool = True,
) -> dict[str, Any]:
    handoff = read_json(handoff_path, {})
    sample_shards = read_json(sample_shards_path, {})
    jobs = handoff.get("jobs") if isinstance(handoff, dict) else []
    shard_files = sample_shards.get("files") if isinstance(sample_shards, dict) else []
    inputs = handoff.get("inputs") if isinstance(handoff, dict) else {}
    derived = Path(str((inputs or {}).get("derived") or "data/benchmark/v1_1"))
    command_report_dir = report_dir or Path(str((inputs or {}).get("report_dir") or out_path.parent))
    width = max(2, len(str(max((int(item.get("index") or 0) for item in shard_files), default=0))))
    baselines: list[dict[str, Any]] = []

    for job in jobs or []:
        artifacts = job.get("artifacts") or {}
        baseline = str(job.get("baseline") or "")
        model = str(job.get("model") or baseline)
        summary_path = Path(str(artifacts.get("summary") or f"{baseline}_summary.json"))
        details_path = Path(str(artifacts.get("details") or f"{baseline}_details.jsonl"))
        cache_path = Path(str(artifacts.get("cache") or f"data/embeddings/v1_1/{baseline}"))
        shared_text_cache_path = Path(str(artifacts.get("shared_text_cache") or f"data/embeddings/v1_1/{baseline}_texts.sqlite"))
        shard_commands: list[dict[str, Any]] = []
        for item in shard_files or []:
            shard_index = int(item.get("index") or 0)
            shard_name = f"shard{shard_index:0{width}d}"
            shard_summary = shard_eval_artifact_path(summary_path, shard_name, "summary")
            shard_details = shard_eval_artifact_path(details_path, shard_name, "details")
            shard_cache = shard_cache_artifact_path(cache_path, shard_name) if use_shard_caches else cache_path
            shard_shared_text_cache = (
                shard_shared_text_cache_path(shared_text_cache_path, shard_name)
                if use_shard_caches
                else shared_text_cache_path
            )
            command = replace_or_append_cli_options(
                str(job.get("command") or ""),
                {
                    "--out": shard_summary,
                    "--details": shard_details,
                    "--cache": shard_cache,
                    "--shared-text-cache": shard_shared_text_cache,
                    "--sample-id-file": item.get("path"),
                },
            )
            shard_commands.append(
                {
                    "index": shard_index,
                    "sample_id_file": item.get("path"),
                    "sample_count": item.get("rows"),
                    "command": command,
                    "artifacts": {
                        "summary": str(shard_summary),
                        "details": str(shard_details),
                        "cache": str(shard_cache),
                        "shared_text_cache": str(shard_shared_text_cache),
                    },
                }
            )

        merge_args: list[Any] = ["v1-1-merge-details", "--derived", derived]
        for shard in shard_commands:
            merge_args.extend(["--details", (shard.get("artifacts") or {}).get("details")])
        merge_report = command_report_dir / f"{baseline}_merge_report.json"
        merge_markdown = command_report_dir / f"{baseline}_merge_report.md"
        merge_args.extend(
            [
                "--out",
                details_path,
                "--candidate-filter",
                "all_files",
                "--report-out",
                merge_report,
                "--markdown-out",
                merge_markdown,
            ]
        )
        summary_args: list[Any] = [
            "v1-1-summary-from-details",
            "--derived",
            derived,
            "--details",
            details_path,
            "--out",
            summary_path,
            "--model",
            model,
            "--candidate-filter",
            "all_files",
        ]
        baselines.append(
            {
                "baseline": baseline,
                "model": model,
                "shard_count": len(shard_commands),
                "shard_commands": shard_commands,
                "merge_command": cli_command(*merge_args),
                "summary_from_details_command": cli_command(*summary_args),
                "final_artifacts": {
                    "summary": str(summary_path),
                    "details": str(details_path),
                    "merge_report": str(merge_report),
                    "merge_markdown": str(merge_markdown),
                },
            }
        )

    missing_shard_files = [str(item.get("path")) for item in shard_files or [] if item.get("path") and not Path(str(item.get("path"))).exists()]
    complete = (
        bool(jobs)
        and bool(shard_files)
        and bool(sample_shards.get("complete"))
        and not missing_shard_files
        and all(item.get("shard_commands") for item in baselines)
    )
    report = {
        "generated_at": utc_now(),
        "complete": complete,
        "handoff": str(handoff_path),
        "sample_shards": str(sample_shards_path),
        "sample_shards_complete": bool(sample_shards.get("complete")),
        "sample_count": sample_shards.get("sample_count"),
        "shard_count": len(shard_files or []),
        "missing_shard_files": missing_shard_files,
        "use_shard_caches": use_shard_caches,
        "baselines": baselines,
        "notes": [
            "Run every shard command for each baseline.",
            "After all shard details are present, run merge_command and then summary_from_details_command for that baseline.",
            "Shard cache paths are separate by default to avoid concurrent writes on shared filesystems.",
        ],
    }
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_shard_commands_markdown(report), encoding="utf-8")
    return report


def write_v1_1_baseline_run_script(
    shard_commands_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    include_runtime_checks: bool = True,
    baseline_filters: Iterable[str] | None = None,
    transfer_manifest_path: Path | None = None,
    return_manifest_path: Path | None = None,
    return_manifest_markdown_path: Path | None = None,
    return_files_path: Path | None = None,
    return_bundle_script_path: Path | None = None,
    include_return_shard_artifacts: bool = False,
    include_return_caches: bool = False,
    finalization_path: Path | None = None,
    finalization_markdown_path: Path | None = None,
    return_acceptance_path: Path | None = None,
    return_acceptance_markdown_path: Path | None = None,
    completion_json_path: Path | None = None,
    workflow_evidence_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    shard_report = read_json(shard_commands_path, {})
    all_baselines = shard_report.get("baselines") if isinstance(shard_report, dict) else []
    requested_baselines = list(baseline_filters or [])
    baselines = [
        baseline
        for baseline in all_baselines or []
        if not requested_baselines or str(baseline.get("baseline") or "") in requested_baselines
    ]
    selected_baseline_names = [str(baseline.get("baseline") or "") for baseline in baselines]
    all_baseline_names = [str(baseline.get("baseline") or "") for baseline in all_baselines or []]
    missing_requested_baselines = sorted(set(requested_baselines) - set(selected_baseline_names))
    handoff_value = shard_report.get("handoff") if isinstance(shard_report, dict) else None
    handoff_path = Path(str(handoff_value)) if handoff_value else None
    includes_all_baselines = bool(all_baseline_names) and set(selected_baseline_names) == set(all_baseline_names)
    include_return_manifest_check = bool(return_manifest_path and handoff_path and includes_all_baselines)
    include_return_bundle_script = bool(return_bundle_script_path and includes_all_baselines)
    include_finalization_check = bool(finalization_path and handoff_path and includes_all_baselines)
    include_return_acceptance_refresh = bool(return_acceptance_path and handoff_path and includes_all_baselines and include_finalization_check)
    completion_json = completion_json_path
    workflow_evidence: list[Path] = []
    seen_workflow_evidence: set[str] = set()

    def add_workflow_evidence(path: Path | str | None) -> None:
        if path is None:
            return
        evidence_path = Path(str(path))
        key = str(evidence_path)
        if key in seen_workflow_evidence:
            return
        seen_workflow_evidence.add(key)
        workflow_evidence.append(evidence_path)

    if handoff_path:
        add_workflow_evidence(handoff_path)
        handoff_payload = read_json(handoff_path, {})
        if not isinstance(handoff_payload, dict):
            handoff_payload = {}
        if completion_json is None:
            completion_json = handoff_completion_json_path(handoff_payload)
        for evidence_path in handoff_workflow_evidence_paths(handoff_payload):
            add_workflow_evidence(evidence_path)
        transfer_bundle_verify = ((handoff_payload.get("transfer_bundle") if isinstance(handoff_payload, dict) else {}) or {}).get("verify_report")
        if transfer_bundle_verify:
            add_workflow_evidence(transfer_bundle_verify)
    handoff_verify_args: list[Any] | None = None
    if handoff_path:
        handoff_verify_args = ["v1-1-verify-handoff", "--handoff", handoff_path]
        handoff_verification = handoff_payload.get("handoff_verification") if isinstance(handoff_payload, dict) else {}
        if isinstance(handoff_verification, dict):
            handoff_verify_report = handoff_verification.get("report")
            handoff_verify_markdown = handoff_verification.get("markdown")
            if handoff_verify_report:
                handoff_verify_args.extend(["--out", handoff_verify_report])
            if handoff_verify_markdown:
                handoff_verify_args.extend(["--markdown-out", handoff_verify_markdown])
    transfer_verify_args: list[Any] | None = None
    transfer_verify_report: str | None = None
    transfer_verify_markdown: str | None = None
    if transfer_manifest_path:
        transfer_verify_args = ["v1-1-verify-transfer-manifest", "--manifest", transfer_manifest_path]
        transfer_manifest_verification = (
            handoff_payload.get("transfer_manifest_verification") if isinstance(handoff_payload, dict) else {}
        )
        if isinstance(transfer_manifest_verification, dict):
            transfer_verify_report = (
                str(transfer_manifest_verification.get("report")) if transfer_manifest_verification.get("report") else None
            )
            transfer_verify_markdown = (
                str(transfer_manifest_verification.get("markdown")) if transfer_manifest_verification.get("markdown") else None
            )
            if transfer_verify_report:
                transfer_verify_args.extend(["--out", transfer_verify_report])
            if transfer_verify_markdown:
                transfer_verify_args.extend(["--markdown-out", transfer_verify_markdown])
    for evidence_path in workflow_evidence_paths or []:
        add_workflow_evidence(evidence_path)
    if return_manifest_path:
        add_workflow_evidence(return_manifest_path)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by v1-1-baseline-run-script at {utc_now()}",
        f"# Source shard command report: {shard_commands_path}",
        'cd "${ARB_REPO_ROOT:-$(pwd)}"',
        "",
        "if [ ! -f pyproject.toml ] || [ ! -f src/agent_retrieval_bench/cli.py ]; then",
        "  echo 'Missing repo source files; copy pyproject.toml and src/agent_retrieval_bench before running this script.' >&2",
        "  exit 1",
        "fi",
        "",
    ]
    command_count = 0
    if transfer_manifest_path:
        lines.extend(
            [
                "echo '-- verify transfer manifest --'",
                cli_command(*(transfer_verify_args or ["v1-1-verify-transfer-manifest", "--manifest", transfer_manifest_path])),
                "",
            ]
        )
        command_count += 1
    if handoff_path:
        lines.extend(
            [
                "echo '-- verify handoff fingerprints --'",
                cli_command(*(handoff_verify_args or ["v1-1-verify-handoff", "--handoff", handoff_path])),
                "",
            ]
        )
        command_count += 1
    required_env = sorted(
        {
            "VOYAGE_API_KEY"
            for baseline in baselines
            for shard in baseline.get("shard_commands") or []
            if " eval-voyage " in f" {shard.get('command') or ''} "
        }
    )
    requires_embedding_runtime = any(
        " eval-embedding " in f" {shard.get('command') or ''} "
        for baseline in baselines
        for shard in baseline.get("shard_commands") or []
    )
    requires_numpy = any(
        any(command_name in f" {shard.get('command') or ''} " for command_name in (" eval-embedding ", " eval-voyage "))
        for baseline in baselines
        for shard in baseline.get("shard_commands") or []
    )
    requires_cuda = any(
        "--device cuda" in str(shard.get("command") or "")
        for baseline in baselines
        for shard in baseline.get("shard_commands") or []
    )
    preflight_dirs = v1_1_baseline_artifact_preflight_dirs(baselines)
    if include_runtime_checks and required_env:
        lines.append("# Required API credentials.")
        for env_name in required_env:
            lines.append(f': "${{{env_name}:?Set {env_name} before running this script.}}"')
        lines.append("")
    if include_runtime_checks and requires_numpy:
        lines.extend(
            [
                "# Required vector/ranking dependency for embedding baselines.",
                "PYTHONPATH=src python3 - <<'PY'",
                "import importlib.util",
                "if importlib.util.find_spec('numpy') is None:",
                "    raise SystemExit(\"Missing numpy; install with: pip install -e '.[embedding]' (or pip install numpy for Voyage-only runners).\")",
                "PY",
                "",
            ]
        )
    if include_runtime_checks and requires_embedding_runtime:
        lines.extend(
            [
                "# Required optional embedding dependencies for local model baselines.",
                "PYTHONPATH=src python3 - <<'PY'",
                "import importlib.util",
                "missing = [name for name in ('sentence_transformers', 'torch') if importlib.util.find_spec(name) is None]",
                "if missing:",
                "    raise SystemExit('Missing optional embedding dependencies: ' + ', '.join(missing) + \"; install with: pip install -e '.[embedding]' and a CUDA-capable torch build.\")",
                "PY",
                "",
            ]
        )
    if include_runtime_checks and requires_cuda:
        lines.extend(
            [
                "# Required CUDA runtime for local embedding baselines.",
                "PYTHONPATH=src python3 - <<'PY'",
                "import torch",
                "if not torch.cuda.is_available():",
                "    raise SystemExit('CUDA is not available to torch; run Jina/Qwen shards on a CUDA-capable machine.')",
                "print(f'CUDA devices: {torch.cuda.device_count()}')",
                "PY",
                "",
            ]
        )
    if preflight_dirs:
        quoted_dirs = " ".join(shlex.quote(path) for path in preflight_dirs)
        lines.extend(
            [
                "# Prepare output/cache locations before long-running shard jobs.",
                "echo '-- prepare output and cache directories --'",
                f"mkdir -p {quoted_dirs}",
                "echo '-- disk space for output and cache directories --'",
                f"df -h {quoted_dirs}",
                "",
            ]
        )

    baseline_reports: list[dict[str, Any]] = []
    for baseline in baselines:
        baseline_name = str(baseline.get("baseline") or "")
        shard_commands = baseline.get("shard_commands") or []
        baseline_command_count = 0
        lines.extend([f"echo {shlex.quote(f'== Baseline: {baseline_name} ==')}", ""])
        for shard in shard_commands:
            command = str(shard.get("command") or "")
            if not command:
                continue
            shard_index = shard.get("index")
            lines.append(f"echo {shlex.quote(f'-- {baseline_name} shard {shard_index} --')}")
            lines.append(command)
            lines.append("")
            command_count += 1
            baseline_command_count += 1
        for label, key in (("merge", "merge_command"), ("summary", "summary_from_details_command")):
            command = str(baseline.get(key) or "")
            if not command:
                continue
            lines.append(f"echo {shlex.quote(f'-- {baseline_name} {label} --')}")
            lines.append(command)
            lines.append("")
            command_count += 1
            baseline_command_count += 1
        baseline_reports.append(
            {
                "baseline": baseline_name,
                "shard_count": len(shard_commands),
                "command_count": baseline_command_count,
                "has_merge_command": bool(baseline.get("merge_command")),
                "has_summary_from_details_command": bool(baseline.get("summary_from_details_command")),
            }
        )
    if include_return_manifest_check:
        return_manifest_args: list[Any] = [
            "v1-1-baseline-return-manifest",
            "--handoff",
            handoff_path,
            "--shard-commands",
            shard_commands_path,
            "--out",
            return_manifest_path,
        ]
        if return_manifest_markdown_path:
            return_manifest_args.extend(["--markdown-out", return_manifest_markdown_path])
        if return_files_path:
            return_manifest_args.extend(["--files-out", return_files_path])
        if include_return_shard_artifacts:
            return_manifest_args.append("--include-shard-artifacts")
        if include_return_caches:
            return_manifest_args.append("--include-caches")
        return_manifest_args.append("--require-existing")
        lines.extend(
            [
                "echo '-- verify required return artifacts --'",
                cli_command(*return_manifest_args),
                "",
            ]
        )
        command_count += 1
    if include_return_bundle_script:
        lines.extend(
            [
                "echo '-- package verified return bundle --'",
                f"bash {shlex.quote(str(return_bundle_script_path))}",
                "",
            ]
        )
        command_count += 1
    if include_finalization_check:
        finalization_args: list[Any] = [
            "v1-1-finalize-baselines",
            "--handoff",
            handoff_path,
            "--shard-commands",
            shard_commands_path,
            "--out",
            finalization_path,
            "--auto-merge-shards",
        ]
        if finalization_markdown_path:
            finalization_args.extend(["--markdown-out", finalization_markdown_path])
        if return_manifest_path:
            finalization_args.extend(["--return-manifest", return_manifest_path])
        if return_manifest_markdown_path:
            finalization_args.extend(["--return-manifest-markdown", return_manifest_markdown_path])
        if return_files_path:
            finalization_args.extend(["--return-files", return_files_path])
        if include_return_shard_artifacts:
            finalization_args.append("--include-shard-artifacts")
        if include_return_caches:
            finalization_args.append("--include-caches")
        finalization_args.extend(completion_doc_cli_args(V1_1_RELEASE_DOCS))
        for evidence_path in workflow_evidence:
            finalization_args.extend(["--workflow-evidence", evidence_path])
        lines.extend(
            [
                "echo '-- finalize V1.1 baseline gates --'",
                cli_command(*finalization_args),
                "",
            ]
        )
        command_count += 1
    if include_return_acceptance_refresh:
        return_acceptance_args: list[Any] = [
            "v1-1-baseline-return-acceptance",
            "--handoff",
            handoff_path,
            "--out",
            return_acceptance_path,
        ]
        if return_manifest_path:
            return_acceptance_args.extend(["--return-manifest", return_manifest_path])
        if completion_json:
            return_acceptance_args.extend(["--completion-json", completion_json])
        if return_acceptance_markdown_path:
            return_acceptance_args.extend(["--markdown-out", return_acceptance_markdown_path])
        return_acceptance_args.append("--require-complete")
        lines.extend(
            [
                "echo '-- refresh return acceptance report --'",
                cli_command(*return_acceptance_args),
                "",
            ]
        )
        command_count += 1
    lines.append("echo 'V1.1 baseline shard script completed.'")
    lines.append("")

    complete = bool(shard_report.get("complete")) and not missing_requested_baselines and bool(baseline_reports) and all(
        item["shard_count"] > 0 and item["has_merge_command"] and item["has_summary_from_details_command"]
        for item in baseline_reports
    )
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o755)
    report = {
        "generated_at": utc_now(),
        "complete": complete,
        "shard_commands": str(shard_commands_path),
        "handoff": str(handoff_path) if handoff_path else None,
        "script": str(out_path),
        "markdown": str(markdown_out_path) if markdown_out_path else None,
        "transfer_manifest": str(transfer_manifest_path) if transfer_manifest_path else None,
        "include_transfer_manifest_check": bool(transfer_manifest_path),
        "transfer_manifest_verify": transfer_verify_report,
        "transfer_manifest_verify_markdown": transfer_verify_markdown,
        "return_manifest": str(return_manifest_path) if return_manifest_path else None,
        "return_manifest_markdown": str(return_manifest_markdown_path) if return_manifest_markdown_path else None,
        "return_files": str(return_files_path) if return_files_path else None,
        "return_bundle_script": str(return_bundle_script_path) if return_bundle_script_path else None,
        "finalization": str(finalization_path) if finalization_path else None,
        "finalization_markdown": str(finalization_markdown_path) if finalization_markdown_path else None,
        "return_acceptance": str(return_acceptance_path) if return_acceptance_path else None,
        "return_acceptance_markdown": str(return_acceptance_markdown_path) if return_acceptance_markdown_path else None,
        "completion_json": str(completion_json) if completion_json else None,
        "include_runtime_checks": include_runtime_checks,
        "include_return_manifest_check": include_return_manifest_check,
        "include_return_bundle_script": include_return_bundle_script,
        "include_finalization_check": include_finalization_check,
        "include_return_acceptance_refresh": include_return_acceptance_refresh,
        "include_return_shard_artifacts": include_return_shard_artifacts,
        "include_return_caches": include_return_caches,
        "workflow_evidence": [str(path) for path in workflow_evidence],
        "requested_baselines": requested_baselines,
        "selected_baselines": selected_baseline_names,
        "missing_requested_baselines": missing_requested_baselines,
        "required_env": required_env,
        "requires_numpy": requires_numpy,
        "requires_cuda": requires_cuda,
        "preflight_dirs": preflight_dirs,
        "preflight_dir_count": len(preflight_dirs),
        "baselines": baseline_reports,
        "command_count": command_count,
        "notes": [
            "Run this script from the repository root on the external GPU/API baseline runner.",
            "The script verifies the transfer manifest before starting shard jobs when --transfer-manifest is provided and writes the configured verifier report when the handoff records transfer_manifest_verification paths.",
            "The script verifies handoff input fingerprints before starting shard jobs when the shard command report records a handoff path.",
            "The script creates output/cache directories and prints df -h for those locations before long-running shard jobs.",
            "Voyage-only runners need numpy plus VOYAGE_API_KEY; Jina/Qwen runners additionally need sentence-transformers and a CUDA-capable torch build.",
            "Shard commands run serially; use the JSON/Markdown shard command report directly if parallel scheduling is preferred.",
            "The script runs merge and summary recovery after each baseline's shard commands finish.",
            "When a full-baseline return manifest path is provided, the script refreshes it with --require-existing after summaries are rebuilt.",
            "When a full-baseline return-bundle script is provided, the script packages and verifies the return bundle before finalization so return-bundle workflow evidence exists.",
            "When a full-baseline finalization path is provided, the script runs v1-1-finalize-baselines after return artifacts are checked and passes workflow evidence to the completion audit.",
            "When a full-baseline return acceptance path is provided, the script refreshes that compact gate report with --require-complete after finalization succeeds.",
        ],
    }
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_run_script_markdown(report), encoding="utf-8")
    return report


def v1_1_baseline_artifact_preflight_dirs(baselines: Iterable[dict[str, Any]]) -> list[str]:
    dirs: set[str] = set()
    for baseline in baselines:
        artifact_maps: list[Any] = [baseline.get("final_artifacts")]
        artifact_maps.extend(shard.get("artifacts") for shard in baseline.get("shard_commands") or [])
        for artifacts in artifact_maps:
            if not isinstance(artifacts, dict):
                continue
            for key, value in artifacts.items():
                if not value:
                    continue
                path = Path(str(value))
                if key == "cache":
                    directory = path
                else:
                    directory = path.parent
                if str(directory) != ".":
                    dirs.add(str(directory))
    return sorted(dirs)


def render_v1_1_baseline_run_script_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Run Script",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Shard commands: `{report.get('shard_commands')}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Script: `{report.get('script')}`",
        f"Transfer manifest check: `{bool(report.get('include_transfer_manifest_check'))}`",
        f"Transfer manifest: `{report.get('transfer_manifest')}`",
        f"Transfer manifest verify: `{report.get('transfer_manifest_verify')}`",
        f"Transfer manifest verify Markdown: `{report.get('transfer_manifest_verify_markdown')}`",
        f"Return manifest check: `{bool(report.get('include_return_manifest_check'))}`",
        f"Return manifest: `{report.get('return_manifest')}`",
        f"Return bundle packaging: `{bool(report.get('include_return_bundle_script'))}`",
        f"Return bundle script: `{report.get('return_bundle_script')}`",
        f"Finalization check: `{bool(report.get('include_finalization_check'))}`",
        f"Finalization: `{report.get('finalization')}`",
        f"Return acceptance refresh: `{bool(report.get('include_return_acceptance_refresh'))}`",
        f"Return acceptance: `{report.get('return_acceptance')}`",
        f"Completion audit: `{report.get('completion_json')}`",
        f"Workflow evidence: `{json.dumps(report.get('workflow_evidence') or [], ensure_ascii=False)}`",
        f"Command count: `{report.get('command_count')}`",
        f"Requested baselines: `{json.dumps(report.get('requested_baselines') or [], ensure_ascii=False)}`",
        f"Missing requested baselines: `{json.dumps(report.get('missing_requested_baselines') or [], ensure_ascii=False)}`",
        f"Requires numpy: `{bool(report.get('requires_numpy'))}`",
        f"Requires CUDA: `{bool(report.get('requires_cuda'))}`",
        f"Required env: `{json.dumps(report.get('required_env') or [], ensure_ascii=False)}`",
        f"Preflight directories: `{int(report.get('preflight_dir_count') or 0)}`",
        "",
        "## Baselines",
        "",
        "| Baseline | Shards | Commands | Merge | Summary |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in report.get("baselines") or []:
        lines.append(
            "| {baseline} | {shards} | {commands} | `{merge}` | `{summary}` |".format(
                baseline=item.get("baseline"),
                shards=item.get("shard_count"),
                commands=item.get("command_count"),
                merge=bool(item.get("has_merge_command")),
                summary=bool(item.get("has_summary_from_details_command")),
            )
        )
    preflight_dirs = report.get("preflight_dirs") or []
    if preflight_dirs:
        lines.extend(["", "## Preflight Directories", ""])
        for path in preflight_dirs:
            lines.append(f"- `{path}`")
    lines.extend(["", "## Notes", ""])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_return_bundle_script(
    handoff_path: Path,
    out_path: Path,
    bundle_path: Path,
    return_manifest_path: Path,
    markdown_out_path: Path | None = None,
    return_manifest_markdown_path: Path | None = None,
    return_files_path: Path | None = None,
    bundle_files_path: Path | None = None,
    shard_commands_path: Path | None = None,
    include_shard_artifacts: bool = False,
    include_caches: bool = False,
    compression: str = "zstd -T0 -3",
    test_bundle: bool = True,
    baseline_filters: Iterable[str] | None = None,
) -> dict[str, Any]:
    handoff = read_json(handoff_path, {})
    handoff_inputs = handoff.get("inputs") if isinstance(handoff.get("inputs"), dict) else {}
    derived = Path(str(handoff_inputs.get("derived") or "data/benchmark/v1_1"))
    eval_dir = Path(str(handoff_inputs.get("eval_dir") or "data/eval/v1_1"))
    bundle_files = bundle_files_path or Path(f"{bundle_path}.files")
    bundle_archive_files = Path(f"{bundle_files}.archive")
    checksum_path = Path(f"{bundle_path}.sha256")
    bundle_verify_path = Path(f"{bundle_path}.verify.json")
    bundle_verify_markdown_path = Path(f"{bundle_path}.verify.md")
    bundle_preflight_path = Path(f"{bundle_path}.preflight.json")
    return_manifest_args: list[Any] = [
        "v1-1-baseline-return-manifest",
        "--handoff",
        handoff_path,
        "--out",
        return_manifest_path,
    ]
    requested_baselines = list(baseline_filters or [])
    for baseline in requested_baselines:
        return_manifest_args.extend(["--baseline", baseline])
    if shard_commands_path:
        return_manifest_args.extend(["--shard-commands", shard_commands_path])
    if return_manifest_markdown_path:
        return_manifest_args.extend(["--markdown-out", return_manifest_markdown_path])
    if return_files_path:
        return_manifest_args.extend(["--files-out", return_files_path])
    if include_shard_artifacts:
        return_manifest_args.append("--include-shard-artifacts")
    if include_caches:
        return_manifest_args.append("--include-caches")
    return_manifest_args.append("--require-existing")
    preflight_args: list[Any] = [
        "check-baseline-summaries",
        "--derived",
        derived,
        "--eval-dir",
        eval_dir,
        "--out",
        bundle_preflight_path,
    ]
    for baseline in requested_baselines:
        preflight_args.extend(["--required-baseline", baseline])

    bundle_parent = str(bundle_path.parent)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by v1-1-baseline-return-bundle-script at {utc_now()}",
        'cd "${ARB_REPO_ROOT:-$(pwd)}"',
        "",
        "if [ ! -f pyproject.toml ] || [ ! -f src/agent_retrieval_bench/cli.py ]; then",
        "  echo 'Missing repo source files; run this script from the repository root after unpacking the transfer bundle.' >&2",
        "  exit 1",
        "fi",
        "",
        "echo '-- refresh and require return manifest --'",
        cli_command(*return_manifest_args),
        "",
        "echo '-- validate returned baseline summaries and details --'",
        cli_command(*preflight_args),
        "",
        "echo '-- write existing return artifact file list --'",
        "PYTHONPATH=src python3 - <<'PY'",
        "import json",
        "from pathlib import Path",
        "from pathlib import PurePosixPath",
        f"manifest_path = Path({json.dumps(str(return_manifest_path))})",
        f"files_path = Path({json.dumps(str(bundle_files))})",
        "payload = json.loads(manifest_path.read_text(encoding='utf-8'))",
        "missing_required = payload.get('missing_required_files') or []",
        "if missing_required:",
        "    for path in missing_required:",
        "        print(f'Missing required return artifact: {path}')",
        "    raise SystemExit(1)",
        "existing_files = payload.get('existing_files')",
        "if existing_files is None:",
        "    existing_files = [item.get('path') for item in payload.get('artifacts', []) if item.get('exists')]",
        "existing_files = [path for path in existing_files if path]",
        "if not existing_files:",
        "    raise SystemExit('No existing return artifacts to bundle.')",
        "unsafe_paths = []",
        "for path in existing_files:",
        "    posix = PurePosixPath(path)",
        "    if posix.is_absolute() or '..' in posix.parts:",
        "        unsafe_paths.append(path)",
        "if unsafe_paths:",
        "    print('Unsafe return artifact paths:')",
        "    for path in unsafe_paths:",
        "        print(path)",
        "    raise SystemExit(1)",
        "symlink_paths = [path for path in existing_files if Path(path).is_symlink()]",
        "if symlink_paths:",
        "    print('Symlink return artifact paths are not allowed:')",
        "    for path in symlink_paths:",
        "        print(path)",
        "    raise SystemExit(1)",
        "files_path.parent.mkdir(parents=True, exist_ok=True)",
        "files_path.write_text('\\n'.join(existing_files) + '\\n', encoding='utf-8')",
        "print(f'Wrote {len(existing_files)} return artifact paths to {files_path}')",
        "PY",
        "",
    ]
    if bundle_parent and bundle_parent != ".":
        lines.append(f"mkdir -p {shlex.quote(bundle_parent)}")
    lines.extend(
        [
            "echo '-- create return artifact bundle --'",
            (
                f"tar --files-from={shlex.quote(str(bundle_files))} "
                f"--use-compress-program={shlex.quote(compression)} "
                f"-cf {shlex.quote(str(bundle_path))}"
            ),
            f"sha256sum {shlex.quote(str(bundle_path))} > {shlex.quote(str(checksum_path))}",
        ]
    )
    if test_bundle:
        lines.extend(["echo '-- verify return artifact bundle --'", f"zstd -t {shlex.quote(str(bundle_path))}"])
    lines.extend(
        [
            "echo '-- verify return artifact bundle contents --'",
            (
                f"if ! tar -tf {shlex.quote(str(bundle_path))} > {shlex.quote(str(bundle_archive_files))} 2>/dev/null; then "
                f"tar --use-compress-program=zstd -tf {shlex.quote(str(bundle_path))} > "
                f"{shlex.quote(str(bundle_archive_files))}; fi"
            ),
            "PYTHONPATH=src python3 - <<'PY'",
            "from pathlib import Path",
            f"expected_path = Path({json.dumps(str(bundle_files))})",
            f"archive_path = Path({json.dumps(str(bundle_archive_files))})",
            "expected = sorted(path for path in expected_path.read_text(encoding='utf-8').splitlines() if path)",
            "actual = sorted(path for path in archive_path.read_text(encoding='utf-8').splitlines() if path)",
            "if expected != actual:",
            "    missing = sorted(set(expected) - set(actual))",
            "    extra = sorted(set(actual) - set(expected))",
            "    if missing:",
            "        print('Bundle is missing expected files:')",
            "        for path in missing:",
            "            print(path)",
            "    if extra:",
            "        print('Bundle contains unexpected files:')",
            "        for path in extra:",
            "            print(path)",
            "    raise SystemExit(1)",
            "print(f'Verified {len(actual)} return artifact bundle members.')",
            "PY",
            "echo '-- write return artifact bundle verification report --'",
            cli_command(
                "v1-1-verify-return-bundle",
                "--bundle",
                bundle_path,
                "--return-manifest",
                return_manifest_path,
                "--checksum",
                checksum_path,
                "--archive-members",
                bundle_archive_files,
                "--bundle-files",
                bundle_files,
                "--out",
                bundle_verify_path,
                "--markdown-out",
                bundle_verify_markdown_path,
            ),
        ]
    )
    lines.append("echo 'V1.1 return artifact bundle completed.'")
    lines.append("")

    ensure_parent(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o755)
    report = {
        "generated_at": utc_now(),
        "complete": handoff_path.exists(),
        "handoff": str(handoff_path),
        "shard_commands": str(shard_commands_path) if shard_commands_path else None,
        "script": str(out_path),
        "markdown": str(markdown_out_path) if markdown_out_path else None,
        "return_manifest": str(return_manifest_path),
        "return_manifest_markdown": str(return_manifest_markdown_path) if return_manifest_markdown_path else None,
        "return_files": str(return_files_path) if return_files_path else None,
        "bundle": str(bundle_path),
        "bundle_files": str(bundle_files),
        "bundle_archive_files": str(bundle_archive_files),
        "bundle_verify": str(bundle_verify_path),
        "bundle_verify_markdown": str(bundle_verify_markdown_path),
        "bundle_preflight": str(bundle_preflight_path),
        "checksum": str(checksum_path),
        "derived": str(derived),
        "eval_dir": str(eval_dir),
        "include_shard_artifacts": include_shard_artifacts,
        "include_caches": include_caches,
        "requested_baselines": requested_baselines,
        "compression": compression,
        "test_bundle": test_bundle,
        "command_count": 7 if test_bundle else 6,
        "notes": [
            "Run this script from the repository root on the external runner after all required baseline artifacts exist.",
            "The script refreshes the return manifest with --require-existing before creating a bundle.",
            "The script runs check-baseline-summaries before packaging so stale, partial, or metric-inconsistent summaries/details fail before transfer.",
            "Only existing return artifacts are placed in the bundle file list; missing optional artifacts are omitted.",
            "The script rejects absolute or parent-traversal return artifact paths before tar receives the file list.",
            "The script rejects symlink return artifact paths before tar receives the file list.",
            "The script verifies that the compressed tar member list exactly matches the generated return-artifact file list.",
            "Copy the bundle and checksum back to the local reporting machine, then unpack before running v1-1-finalize-baselines.",
        ],
    }
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_return_bundle_script_markdown(report), encoding="utf-8")
    return report


def render_v1_1_baseline_return_bundle_script_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Return Bundle Script",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Shard commands: `{report.get('shard_commands')}`",
        f"Script: `{report.get('script')}`",
        f"Return manifest: `{report.get('return_manifest')}`",
        f"Bundle: `{report.get('bundle')}`",
        f"Bundle files: `{report.get('bundle_files')}`",
        f"Bundle archive files: `{report.get('bundle_archive_files')}`",
        f"Bundle verification: `{report.get('bundle_verify')}`",
        f"Bundle preflight: `{report.get('bundle_preflight')}`",
        f"Checksum: `{report.get('checksum')}`",
        f"Derived: `{report.get('derived')}`",
        f"Eval dir: `{report.get('eval_dir')}`",
        f"Include shard artifacts: `{bool(report.get('include_shard_artifacts'))}`",
        f"Include caches: `{bool(report.get('include_caches'))}`",
        f"Requested baselines: `{json.dumps(report.get('requested_baselines') or [], ensure_ascii=False)}`",
        f"Compression: `{report.get('compression')}`",
        f"Command count: `{report.get('command_count')}`",
        "",
        "## Notes",
        "",
    ]
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_apply_return_bundle_script(
    handoff_path: Path,
    out_path: Path,
    bundle_path: Path,
    finalization_path: Path,
    markdown_out_path: Path | None = None,
    checksum_path: Path | None = None,
    shard_commands_path: Path | None = None,
    return_manifest_path: Path | None = None,
    return_manifest_markdown_path: Path | None = None,
    return_files_path: Path | None = None,
    finalization_markdown_path: Path | None = None,
    return_acceptance_path: Path | None = None,
    return_acceptance_markdown_path: Path | None = None,
    completion_json_path: Path | None = None,
    include_shard_artifacts: bool = False,
    include_caches: bool = False,
    test_bundle: bool = True,
    auto_merge_shards: bool = True,
    run_finalization: bool = True,
    baseline_filters: Iterable[str] | None = None,
    workflow_evidence_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    requested_baselines = [str(item) for item in baseline_filters or []]
    checksum = checksum_path or Path(f"{bundle_path}.sha256")
    bundle_members_path = Path(f"{bundle_path}.members")
    bundle_member_types_path = Path(f"{bundle_path}.members.types")
    bundle_verify_path = Path(f"{bundle_path}.verify.json")
    bundle_verify_markdown_path = Path(f"{bundle_path}.verify.md")
    precheck_return_manifest_args: list[Any] | None = None
    return_manifest_args: list[Any] | None = None
    if return_manifest_path:
        base_return_manifest_args: list[Any] = [
            "v1-1-baseline-return-manifest",
            "--handoff",
            handoff_path,
            "--out",
            return_manifest_path,
        ]
        if shard_commands_path:
            base_return_manifest_args.extend(["--shard-commands", shard_commands_path])
        for baseline in requested_baselines:
            base_return_manifest_args.extend(["--baseline", baseline])
        if return_manifest_markdown_path:
            base_return_manifest_args.extend(["--markdown-out", return_manifest_markdown_path])
        if return_files_path:
            base_return_manifest_args.extend(["--files-out", return_files_path])
        if include_shard_artifacts:
            base_return_manifest_args.append("--include-shard-artifacts")
        if include_caches:
            base_return_manifest_args.append("--include-caches")
        if requested_baselines:
            precheck_return_manifest_args = list(base_return_manifest_args)
        return_manifest_args = list(base_return_manifest_args)
        return_manifest_args.append("--require-existing")

    workflow_evidence: list[Path] = []
    finalization_args: list[Any] = []
    include_return_acceptance_refresh = bool(return_acceptance_path and run_finalization and not requested_baselines)
    completion_json = completion_json_path
    if run_finalization:
        finalization_args = [
            "v1-1-finalize-baselines",
            "--handoff",
            handoff_path,
            "--out",
            finalization_path,
        ]
        if shard_commands_path:
            finalization_args.extend(["--shard-commands", shard_commands_path])
        if finalization_markdown_path:
            finalization_args.extend(["--markdown-out", finalization_markdown_path])
        if return_manifest_path:
            finalization_args.extend(["--return-manifest", return_manifest_path])
        if return_manifest_markdown_path:
            finalization_args.extend(["--return-manifest-markdown", return_manifest_markdown_path])
        if return_files_path:
            finalization_args.extend(["--return-files", return_files_path])
        if include_shard_artifacts:
            finalization_args.append("--include-shard-artifacts")
        if include_caches:
            finalization_args.append("--include-caches")
        if auto_merge_shards:
            finalization_args.append("--auto-merge-shards")
        finalization_args.extend(completion_doc_cli_args(V1_1_RELEASE_DOCS))
        seen_workflow_evidence: set[str] = set()

        def add_workflow_evidence(path: Path | str | None) -> None:
            if path is None:
                return
            evidence_path = Path(str(path))
            key = str(evidence_path)
            if key in seen_workflow_evidence:
                return
            seen_workflow_evidence.add(key)
            workflow_evidence.append(evidence_path)

        add_workflow_evidence(handoff_path)
        handoff_payload = read_json(handoff_path, {})
        if completion_json is None:
            completion_json = handoff_completion_json_path(handoff_payload)
        for evidence_path in handoff_workflow_evidence_paths(handoff_payload):
            add_workflow_evidence(evidence_path)
        transfer_bundle_verify = ((handoff_payload.get("transfer_bundle") if isinstance(handoff_payload, dict) else {}) or {}).get("verify_report")
        if transfer_bundle_verify:
            add_workflow_evidence(transfer_bundle_verify)
        for evidence_path in workflow_evidence_paths or []:
            add_workflow_evidence(evidence_path)
        if return_manifest_path:
            add_workflow_evidence(return_manifest_path)
        add_workflow_evidence(bundle_verify_path)
        for evidence_path in workflow_evidence:
            finalization_args.extend(["--workflow-evidence", evidence_path])

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by v1-1-baseline-apply-return-bundle-script at {utc_now()}",
        f"default_bundle_path={shlex.quote(str(bundle_path))}",
        f"default_checksum_path={shlex.quote(str(checksum))}",
        'bundle_path="${ARB_RETURN_BUNDLE:-$default_bundle_path}"',
        'checksum_path="${ARB_RETURN_CHECKSUM:-$default_checksum_path}"',
        'cd "${ARB_REPO_ROOT:-$(pwd)}"',
        "",
        "if [ ! -f pyproject.toml ] || [ ! -f src/agent_retrieval_bench/cli.py ]; then",
        "  echo 'Missing repo source files; run this script from the repository root.' >&2",
        "  exit 1",
        "fi",
        "",
        "echo '-- verify return bundle checksum --'",
        'if [[ ! -f "$bundle_path" ]]; then',
        '  echo "Missing return bundle: $bundle_path" >&2',
        "  exit 1",
        "fi",
        'if [[ ! -f "$checksum_path" ]]; then',
        '  echo "Missing return bundle checksum: $checksum_path" >&2',
        "  exit 1",
        "fi",
        'expected_sha256="$(awk \'NF {print $1; exit}\' "$checksum_path")"',
        'actual_sha256="$(sha256sum "$bundle_path" | awk \'{print $1}\')"',
        'if [[ -z "$expected_sha256" || "$actual_sha256" != "$expected_sha256" ]]; then',
        '  echo "Return bundle checksum mismatch" >&2',
        '  echo "expected: $expected_sha256" >&2',
        '  echo "actual:   $actual_sha256" >&2',
        "  exit 1",
        "fi",
    ]
    if test_bundle:
        lines.extend(["", "echo '-- verify return bundle compression --'", 'zstd -t "$bundle_path"'])
    lines.extend(
        [
            "",
            "echo '-- inspect return bundle members --'",
            f'if ! tar -tf "$bundle_path" > {shlex.quote(str(bundle_members_path))} 2>/dev/null; then',
            f'  tar --use-compress-program=zstd -tf "$bundle_path" > {shlex.quote(str(bundle_members_path))}',
            "fi",
            f'if ! tar -tvf "$bundle_path" > {shlex.quote(str(bundle_member_types_path))} 2>/dev/null; then',
            f'  tar --use-compress-program=zstd -tvf "$bundle_path" > {shlex.quote(str(bundle_member_types_path))}',
            "fi",
            "PYTHONPATH=src python3 - <<'PY'",
            "from pathlib import Path, PurePosixPath",
            f"members_path = Path({json.dumps(str(bundle_members_path))})",
            f"types_path = Path({json.dumps(str(bundle_member_types_path))})",
            "members = [line for line in members_path.read_text(encoding='utf-8').splitlines() if line]",
            "unsafe = []",
            "for member in members:",
            "    posix = PurePosixPath(member)",
            "    if posix.is_absolute() or '..' in posix.parts:",
            "        unsafe.append(member)",
            "if unsafe:",
            "    print('Unsafe return bundle member paths:')",
            "    for member in unsafe:",
            "        print(member)",
            "    raise SystemExit(1)",
            "if not members:",
            "    raise SystemExit('Return bundle contains no members.')",
            "print(f'Validated {len(members)} safe return bundle member paths.')",
            "non_regular = []",
            "for line in types_path.read_text(encoding='utf-8').splitlines():",
            "    if line and line[0] != '-':",
            "        non_regular.append(line)",
            "if non_regular:",
            "    print('Return bundle contains non-regular file members:')",
            "    for line in non_regular:",
            "        print(line)",
            "    raise SystemExit(1)",
            "print(f'Validated {len(members)} regular file return bundle members.')",
            "PY",
            "",
        ]
    )
    command_count = 4 if test_bundle else 3
    if precheck_return_manifest_args:
        lines.extend(
            [
                "echo '-- write return manifest for bundle verification --'",
                cli_command(*precheck_return_manifest_args),
                "",
            ]
        )
        command_count += 1
    if return_manifest_path:
        lines.extend(
            [
                "echo '-- verify return bundle members against manifest --'",
                "PYTHONPATH=src python3 - <<'PY'",
                "import json",
                "from pathlib import Path",
                f"members_path = Path({json.dumps(str(bundle_members_path))})",
                f"manifest_path = Path({json.dumps(str(return_manifest_path))})",
                "members = sorted(line for line in members_path.read_text(encoding='utf-8').splitlines() if line)",
                "payload = json.loads(manifest_path.read_text(encoding='utf-8'))",
                "allowed = set(payload.get('files') or [])",
                "required = set(payload.get('required_files') or [])",
                "if not allowed:",
                "    raise SystemExit('Return manifest does not list allowed files.')",
                "extra = sorted(set(members) - allowed)",
                "missing_required = sorted(required - set(members))",
                "if extra:",
                "    print('Return bundle contains files not listed in return manifest:')",
                "    for path in extra:",
                "        print(path)",
                "if missing_required:",
                "    print('Return bundle is missing required return files:')",
                "    for path in missing_required:",
                "        print(path)",
                "if extra or missing_required:",
                "    raise SystemExit(1)",
                "print(f'Validated {len(members)} return bundle members against {manifest_path}.')",
                "PY",
                "echo '-- write return bundle verification report --'",
                " ".join(
                    [
                        "PYTHONPATH=src",
                        "python",
                        "-m",
                        "agent_retrieval_bench.cli",
                        "v1-1-verify-return-bundle",
                        "--bundle",
                        '"$bundle_path"',
                        "--return-manifest",
                        shlex.quote(str(return_manifest_path)),
                        "--checksum",
                        '"$checksum_path"',
                        "--archive-members",
                        shlex.quote(str(bundle_members_path)),
                        "--out",
                        shlex.quote(str(bundle_verify_path)),
                        "--markdown-out",
                        shlex.quote(str(bundle_verify_markdown_path)),
                    ]
                ),
                "",
            ]
        )
        command_count += 2
    lines.extend(
        [
            "echo '-- unpack return artifacts --'",
            'if ! tar -xf "$bundle_path" 2>/dev/null; then',
            '  tar --use-compress-program=zstd -xf "$bundle_path"',
            "fi",
            "",
        ]
    )
    if return_manifest_args:
        lines.extend(["echo '-- verify returned required artifacts --'", cli_command(*return_manifest_args), ""])
        command_count += 1
    if run_finalization:
        lines.extend(["echo '-- finalize V1.1 baseline gates --'", cli_command(*finalization_args), ""])
        command_count += 1
        if include_return_acceptance_refresh:
            return_acceptance_args: list[Any] = [
                "v1-1-baseline-return-acceptance",
                "--handoff",
                handoff_path,
                "--out",
                return_acceptance_path,
            ]
            if return_manifest_path:
                return_acceptance_args.extend(["--return-manifest", return_manifest_path])
            if completion_json:
                return_acceptance_args.extend(["--completion-json", completion_json])
            if return_acceptance_markdown_path:
                return_acceptance_args.extend(["--markdown-out", return_acceptance_markdown_path])
            return_acceptance_args.append("--require-complete")
            lines.extend(["echo '-- refresh return acceptance report --'", cli_command(*return_acceptance_args), ""])
            command_count += 1
        lines.extend(["echo 'V1.1 return bundle applied.'", ""])
    else:
        lines.extend(
            [
                "echo '-- finalization disabled --'",
                "echo 'V1.1 return bundle verified and unpacked; finalization was not run.'",
                "",
            ]
        )

    ensure_parent(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o755)
    report = {
        "generated_at": utc_now(),
        "complete": handoff_path.exists(),
        "handoff": str(handoff_path),
        "shard_commands": str(shard_commands_path) if shard_commands_path else None,
        "script": str(out_path),
        "markdown": str(markdown_out_path) if markdown_out_path else None,
        "bundle": str(bundle_path),
        "bundle_members": str(bundle_members_path),
        "bundle_member_types": str(bundle_member_types_path),
        "bundle_verify": str(bundle_verify_path),
        "bundle_verify_markdown": str(bundle_verify_markdown_path),
        "checksum": str(checksum),
        "return_manifest": str(return_manifest_path) if return_manifest_path else None,
        "return_manifest_markdown": str(return_manifest_markdown_path) if return_manifest_markdown_path else None,
        "return_files": str(return_files_path) if return_files_path else None,
        "finalization": str(finalization_path) if run_finalization else None,
        "finalization_markdown": str(finalization_markdown_path) if run_finalization and finalization_markdown_path else None,
        "return_acceptance": str(return_acceptance_path) if return_acceptance_path else None,
        "return_acceptance_markdown": str(return_acceptance_markdown_path) if return_acceptance_markdown_path else None,
        "completion_json": str(completion_json) if completion_json else None,
        "include_return_acceptance_refresh": include_return_acceptance_refresh,
        "include_shard_artifacts": include_shard_artifacts,
        "include_caches": include_caches,
        "test_bundle": test_bundle,
        "auto_merge_shards": auto_merge_shards,
        "run_finalization": run_finalization,
        "requested_baselines": requested_baselines,
        "precheck_return_manifest": bool(precheck_return_manifest_args),
        "workflow_evidence": [str(path) for path in workflow_evidence],
        "command_count": command_count,
        "notes": [
            "Run this script from the local reporting checkout after copying back the return bundle and checksum.",
            "The checksum sidecar may record the external runner's bundle path; this script reads the expected hash from the first checksum column and verifies the configured bundle path.",
            "Set ARB_RETURN_BUNDLE and ARB_RETURN_CHECKSUM to override the generated bundle and checksum paths.",
            "The script verifies the checksum, tests the compressed bundle, rejects unsafe tar member paths and non-regular tar members, verifies members against the return manifest when provided, unpacks returned artifacts, and checks required files.",
            "Finalization regenerates baseline status, summary preflight, leaderboard, readiness, release, and completion-audit reports when enabled.",
            "When a full return acceptance path is provided, the script refreshes that compact gate report with --require-complete after finalization succeeds.",
        ],
    }
    if not run_finalization:
        report["notes"].append("Finalization is disabled for this script; use it for partial GPU/API return bundles before all required baselines are present.")
    if requested_baselines:
        report["notes"].append("The script writes a filtered return manifest before member verification, then reruns it with --require-existing after unpacking.")
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_apply_return_bundle_script_markdown(report), encoding="utf-8")
    return report


def render_v1_1_baseline_apply_return_bundle_script_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Apply Return Bundle Script",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Shard commands: `{report.get('shard_commands')}`",
        f"Script: `{report.get('script')}`",
        f"Bundle: `{report.get('bundle')}`",
        f"Bundle members: `{report.get('bundle_members')}`",
        f"Bundle verification: `{report.get('bundle_verify')}`",
        f"Checksum: `{report.get('checksum')}`",
        f"Return manifest: `{report.get('return_manifest')}`",
        f"Requested baselines: `{json.dumps(report.get('requested_baselines') or [], ensure_ascii=False)}`",
        f"Run finalization: `{bool(report.get('run_finalization'))}`",
        f"Finalization: `{report.get('finalization')}`",
        f"Return acceptance refresh: `{bool(report.get('include_return_acceptance_refresh'))}`",
        f"Return acceptance: `{report.get('return_acceptance')}`",
        f"Completion audit: `{report.get('completion_json')}`",
        f"Include shard artifacts: `{bool(report.get('include_shard_artifacts'))}`",
        f"Include caches: `{bool(report.get('include_caches'))}`",
        f"Test bundle: `{bool(report.get('test_bundle'))}`",
        f"Auto-merge shards: `{bool(report.get('auto_merge_shards'))}`",
        f"Workflow evidence: `{json.dumps(report.get('workflow_evidence') or [], ensure_ascii=False)}`",
        f"Command count: `{report.get('command_count')}`",
        "",
        "## Notes",
        "",
    ]
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def verify_v1_1_baseline_return_bundle(
    bundle_path: Path,
    return_manifest_path: Path,
    checksum_path: Path | None = None,
    archive_members_path: Path | None = None,
    bundle_files_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    test_compression: bool = True,
) -> dict[str, Any]:
    checksum = checksum_path or Path(f"{bundle_path}.sha256")
    archive_members = archive_members_path or Path(f"{bundle_path}.members")
    manifest = read_json(return_manifest_path, {})
    allowed_files = sorted(str(path) for path in (manifest.get("files") if isinstance(manifest, dict) else []) or [])
    required_files = sorted(str(path) for path in (manifest.get("required_files") if isinstance(manifest, dict) else []) or [])
    checks: list[dict[str, Any]] = []

    bundle_fingerprint = file_fingerprint(bundle_path)
    expected_checksum = None
    checksum_error = None
    if checksum.is_file():
        parts = checksum.read_text(encoding="utf-8").split()
        expected_checksum = parts[0] if parts else None
    else:
        checksum_error = "missing_checksum_file"
    checksum_matches = bool(
        expected_checksum
        and bundle_fingerprint.get("exists")
        and bundle_fingerprint.get("type") == "file"
        and bundle_fingerprint.get("sha256") == expected_checksum
    )
    checks.append(
        {
            "name": "return_bundle_checksum_matches",
            "status": "pass" if checksum_matches else "fail",
            "bundle": str(bundle_path),
            "checksum": str(checksum),
            "expected_sha256": expected_checksum,
            "actual_sha256": bundle_fingerprint.get("sha256"),
            "bundle_exists": bool(bundle_fingerprint.get("exists")),
            "bundle_size_bytes": bundle_fingerprint.get("size_bytes"),
            "error": checksum_error,
        }
    )

    if test_compression:
        compression_stdout = ""
        compression_stderr = ""
        if bundle_path.is_file():
            try:
                zstd_result = subprocess.run(["zstd", "-t", str(bundle_path)], text=True, capture_output=True)
                compression_passed = zstd_result.returncode == 0
                compression_stdout = (zstd_result.stdout or "").strip()
                compression_stderr = (zstd_result.stderr or "").strip()
                compression_error = None if compression_passed else (compression_stderr or compression_stdout or "zstd_integrity_test_failed")
            except FileNotFoundError:
                compression_passed = False
                compression_error = "missing_zstd_executable"
        else:
            compression_passed = False
            compression_error = "missing_bundle"
        checks.append(
            {
                "name": "return_bundle_compression_valid",
                "status": "pass" if compression_passed else "fail",
                "tested": True,
                "stdout": compression_stdout,
                "stderr": compression_stderr,
                "error": compression_error,
            }
        )
    else:
        checks.append({"name": "return_bundle_compression_valid", "status": "pass", "tested": False, "error": None})

    actual_members: list[str] = []
    member_error = None
    if bundle_path.is_file():
        tar_result, member_error = run_tar_read(bundle_path, "-tf")
        if tar_result is not None and tar_result.returncode == 0:
            actual_members = sorted(line for line in tar_result.stdout.splitlines() if line)
            ensure_parent(archive_members)
            archive_members.write_text("\n".join(actual_members) + ("\n" if actual_members else ""), encoding="utf-8")
    else:
        member_error = "missing_bundle"
    unsafe_members = unsafe_transfer_member_paths(actual_members)
    checks.append(
        {
            "name": "return_bundle_members_are_safe_paths",
            "status": "pass" if not unsafe_members and actual_members else "fail",
            "unsafe_member_count": len(unsafe_members),
            "unsafe_members": unsafe_members[:100],
            "members_truncated": len(unsafe_members) > 100,
            "member_count": len(actual_members),
            "error": member_error,
        }
    )
    non_regular_members: list[dict[str, str]] = []
    member_type_error = None
    if bundle_path.is_file():
        tar_types_result, member_type_error = run_tar_read(bundle_path, "-tvf")
        if tar_types_result is not None and tar_types_result.returncode == 0:
            for line in tar_types_result.stdout.splitlines():
                if line and line[0] != "-":
                    non_regular_members.append({"type": line[0], "listing": line})
    else:
        member_type_error = "missing_bundle"
    checks.append(
        {
            "name": "return_bundle_members_are_regular_files",
            "status": "pass" if actual_members and not non_regular_members and not member_type_error else "fail",
            "non_regular_member_count": len(non_regular_members),
            "non_regular_members": non_regular_members[:100],
            "members_truncated": len(non_regular_members) > 100,
            "member_count": len(actual_members),
            "error": member_type_error,
        }
    )

    extra_members = sorted(set(actual_members) - set(allowed_files))
    missing_required_members = sorted(set(required_files) - set(actual_members))
    manifest_matches = bool(allowed_files) and bool(required_files) and actual_members and not extra_members and not missing_required_members
    checks.append(
        {
            "name": "return_bundle_members_match_return_manifest",
            "status": "pass" if manifest_matches else "fail",
            "return_manifest": str(return_manifest_path),
            "allowed_file_count": len(allowed_files),
            "required_file_count": len(required_files),
            "actual_member_count": len(actual_members),
            "extra_member_count": len(extra_members),
            "missing_required_member_count": len(missing_required_members),
            "extra_members": extra_members[:100],
            "missing_required_members": missing_required_members[:100],
            "members_truncated": len(extra_members) > 100 or len(missing_required_members) > 100,
            "error": member_error,
        }
    )

    expected_bundle_files: list[str] = []
    if bundle_files_path:
        expected_bundle_files = sorted(line for line in bundle_files_path.read_text(encoding="utf-8").splitlines() if line) if bundle_files_path.is_file() else []
        missing_from_bundle = sorted(set(expected_bundle_files) - set(actual_members))
        extra_from_bundle_list = sorted(set(actual_members) - set(expected_bundle_files))
        exact_bundle_list = bool(expected_bundle_files) and actual_members == expected_bundle_files and not missing_from_bundle and not extra_from_bundle_list
        checks.append(
            {
                "name": "return_bundle_members_match_bundle_file_list",
                "status": "pass" if exact_bundle_list else "fail",
                "bundle_files": str(bundle_files_path),
                "expected_member_count": len(expected_bundle_files),
                "actual_member_count": len(actual_members),
                "missing_member_count": len(missing_from_bundle),
                "extra_member_count": len(extra_from_bundle_list),
                "missing_members": missing_from_bundle[:100],
                "extra_members": extra_from_bundle_list[:100],
                "members_truncated": len(missing_from_bundle) > 100 or len(extra_from_bundle_list) > 100,
                "error": None if bundle_files_path.is_file() else "missing_bundle_files",
            }
        )

    report = {
        "generated_at": utc_now(),
        "complete": all(check.get("status") == "pass" for check in checks),
        "bundle": str(bundle_path),
        "return_manifest": str(return_manifest_path),
        "checksum": str(checksum),
        "archive_members": str(archive_members),
        "bundle_files": str(bundle_files_path) if bundle_files_path else None,
        "sha256": bundle_fingerprint.get("sha256"),
        "size_bytes": bundle_fingerprint.get("size_bytes"),
        "bundle_fingerprint": bundle_fingerprint,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check.get("status") != "pass"],
        "notes": [
            "This verifier checks the returned baseline bundle checksum, compressed archive integrity, safe member paths, regular-file member types, required return artifacts, and unexpected members.",
            "When --bundle-files is provided, it also checks the archive member list exactly matches the generated return bundle file list.",
            "Run this before unpacking returned artifacts on the local reporting checkout.",
        ],
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_return_bundle_verification_markdown(report), encoding="utf-8")
    return report


def render_v1_1_return_bundle_verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Return Bundle Verification",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Bundle: `{report.get('bundle')}`",
        f"Return manifest: `{report.get('return_manifest')}`",
        f"SHA256: `{report.get('sha256')}`",
        f"Size bytes: `{report.get('size_bytes')}`",
        f"Overall status: `{'complete' if report.get('complete') else 'not_complete'}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        status = "PASS" if check.get("status") == "pass" else "FAIL"
        lines.append(f"### {status} - {check.get('name')}")
        lines.append("")
        for key, value in check.items():
            if key in {"name", "status"}:
                continue
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_return_manifest(
    handoff_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    files_out_path: Path | None = None,
    shard_commands_path: Path | None = None,
    include_shard_artifacts: bool = False,
    include_caches: bool = False,
    baseline_filters: Iterable[str] | None = None,
) -> dict[str, Any]:
    handoff = read_json(handoff_path, {})
    jobs = handoff.get("jobs") if isinstance(handoff, dict) else []
    shard_commands = read_json(shard_commands_path, {}) if shard_commands_path else {}
    requested_baselines = list(baseline_filters or [])
    requested_baseline_set = set(requested_baselines)
    available_baselines = {
        str(job.get("baseline") or "")
        for job in jobs or []
        if str(job.get("baseline") or "")
    }
    available_baselines.update(
        str(baseline_report.get("baseline") or "")
        for baseline_report in shard_commands.get("baselines") or []
        if str(baseline_report.get("baseline") or "")
    )
    selected_baselines = sorted(
        available_baselines if not requested_baseline_set else available_baselines & requested_baseline_set
    )
    missing_requested_baselines = sorted(requested_baseline_set - available_baselines)
    entries: list[dict[str, Any]] = []

    for job in jobs or []:
        baseline = str(job.get("baseline") or "")
        if requested_baseline_set and baseline not in requested_baseline_set:
            continue
        artifacts = job.get("artifacts") or {}
        for role in ("summary", "details"):
            path_value = artifacts.get(role)
            if path_value:
                entries.append(return_artifact_entry(path_value, role=role, baseline=baseline, required=True))
        if include_caches:
            for role in ("cache", "shared_text_cache"):
                path_value = artifacts.get(role)
                if path_value:
                    entries.append(return_artifact_entry(path_value, role=role, baseline=baseline, required=False))

    for baseline_report in shard_commands.get("baselines") or []:
        baseline = str(baseline_report.get("baseline") or "")
        if requested_baseline_set and baseline not in requested_baseline_set:
            continue
        final_artifacts = baseline_report.get("final_artifacts") or {}
        for role in ("summary", "details"):
            path_value = final_artifacts.get(role)
            if path_value:
                entries.append(return_artifact_entry(path_value, role=role, baseline=baseline, required=True))
        for role in ("merge_report", "merge_markdown"):
            path_value = final_artifacts.get(role)
            if path_value:
                entries.append(return_artifact_entry(path_value, role=role, baseline=baseline, required=False))
        if include_shard_artifacts:
            for shard in baseline_report.get("shard_commands") or []:
                shard_index = shard.get("index")
                artifacts = shard.get("artifacts") or {}
                for role in ("summary", "details"):
                    path_value = artifacts.get(role)
                    if path_value:
                        entries.append(
                            return_artifact_entry(
                                path_value,
                                role=f"shard_{role}",
                                baseline=baseline,
                                required=False,
                                shard_index=shard_index,
                            )
                        )
                if include_caches:
                    for role in ("cache", "shared_text_cache"):
                        path_value = artifacts.get(role)
                        if path_value:
                            entries.append(
                                return_artifact_entry(
                                    path_value,
                                    role=f"shard_{role}",
                                    baseline=baseline,
                                    required=False,
                                    shard_index=shard_index,
                                )
                            )

    deduped = dedupe_return_artifact_entries(entries)
    required_entries = [entry for entry in deduped if entry.get("required")]
    optional_entries = [entry for entry in deduped if not entry.get("required")]
    required_files = [entry["path"] for entry in required_entries]
    optional_files = [entry["path"] for entry in optional_entries]
    existing_files = [entry["path"] for entry in deduped if entry.get("exists")]
    missing_files = [entry["path"] for entry in deduped if not entry.get("exists")]
    missing_required = [entry["path"] for entry in required_entries if not entry.get("exists")]
    missing_optional = [entry["path"] for entry in optional_entries if not entry.get("exists")]
    required_files_by_baseline = group_return_entry_paths_by_baseline(required_entries)
    optional_files_by_baseline = group_return_entry_paths_by_baseline(optional_entries)
    existing_files_by_baseline = group_return_entry_paths_by_baseline(
        [entry for entry in deduped if entry.get("exists")]
    )
    missing_files_by_baseline = group_return_entry_paths_by_baseline(
        [entry for entry in deduped if not entry.get("exists")]
    )
    missing_required_files_by_baseline = group_return_entry_paths_by_baseline(
        [entry for entry in required_entries if not entry.get("exists")]
    )
    missing_optional_files_by_baseline = group_return_entry_paths_by_baseline(
        [entry for entry in optional_entries if not entry.get("exists")]
    )
    report = {
        "generated_at": utc_now(),
        "complete": bool(required_entries) and not missing_requested_baselines,
        "artifacts_complete": bool(required_entries) and not missing_required,
        "handoff": str(handoff_path),
        "shard_commands": str(shard_commands_path) if shard_commands_path else None,
        "include_shard_artifacts": include_shard_artifacts,
        "include_caches": include_caches,
        "requested_baselines": requested_baselines,
        "selected_baselines": selected_baselines,
        "missing_requested_baselines": missing_requested_baselines,
        "file_count": len(deduped),
        "required_file_count": len(required_entries),
        "optional_file_count": len(optional_entries),
        "existing_file_count": len(existing_files),
        "missing_file_count": len(missing_files),
        "required_files": required_files,
        "required_files_by_baseline": required_files_by_baseline,
        "optional_files": optional_files,
        "optional_files_by_baseline": optional_files_by_baseline,
        "existing_files": existing_files,
        "existing_files_by_baseline": existing_files_by_baseline,
        "missing_files": missing_files,
        "missing_files_by_baseline": missing_files_by_baseline,
        "missing_required_files": missing_required,
        "missing_required_files_by_baseline": missing_required_files_by_baseline,
        "missing_optional_files": missing_optional,
        "missing_optional_files_by_baseline": missing_optional_files_by_baseline,
        "files": [entry["path"] for entry in deduped],
        "artifacts": deduped,
        "notes": [
            "Copy required files back from the external runner before final local preflight/report regeneration.",
            "Final summary/details files are required; shard details and merge reports are useful audit/debug artifacts when sharded runs are used.",
            "After copying results back, run check-baseline-summaries and report-models with required baselines.",
        ],
    }
    report = preserve_generated_at_if_report_unchanged(report, out_path)
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_return_manifest_markdown(report), encoding="utf-8")
    if files_out_path:
        ensure_parent(files_out_path)
        files_out_path.write_text("\n".join(report["files"]) + "\n", encoding="utf-8")
    return report


def preserve_generated_at_if_report_unchanged(report: dict[str, Any], existing_path: Path) -> dict[str, Any]:
    existing = read_json(existing_path, {})
    if not isinstance(existing, dict) or not existing.get("generated_at"):
        return report
    existing_without_time = {key: value for key, value in existing.items() if key != "generated_at"}
    report_without_time = {key: value for key, value in report.items() if key != "generated_at"}
    if existing_without_time != report_without_time:
        return report
    preserved = dict(report)
    preserved["generated_at"] = existing["generated_at"]
    return preserved


def return_artifact_entry(
    path_value: Any,
    role: str,
    baseline: str,
    required: bool,
    shard_index: Any | None = None,
) -> dict[str, Any]:
    path = Path(str(path_value))
    fingerprint = file_fingerprint(path)
    return {
        "path": str(path),
        "baseline": baseline,
        "role": role,
        "required": required,
        "shard_index": shard_index,
        "exists": bool(fingerprint.get("exists")),
        "type": fingerprint.get("type"),
        "size_bytes": fingerprint.get("size_bytes"),
        "sha256": fingerprint.get("sha256"),
        "line_count": fingerprint.get("line_count"),
    }


def dedupe_return_artifact_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = str(entry.get("path") or "")
        if not path:
            continue
        existing = deduped.get(path)
        if existing is None:
            deduped[path] = dict(entry)
            continue
        existing["required"] = bool(existing.get("required") or entry.get("required"))
        roles = sorted(set(str(role) for role in [existing.get("role"), entry.get("role")] if role))
        existing["role"] = ",".join(roles)
        baselines = sorted(set(str(item) for item in [existing.get("baseline"), entry.get("baseline")] if item))
        existing["baseline"] = ",".join(baselines)
    return [deduped[path] for path in sorted(deduped)]


def group_return_entry_paths_by_baseline(entries: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        path = str(entry.get("path") or "")
        baselines = [baseline.strip() for baseline in str(entry.get("baseline") or "").split(",") if baseline.strip()]
        if not baselines:
            baselines = ["unassigned"]
        for baseline in baselines:
            grouped[baseline].append(path)
    return {baseline: sorted(paths) for baseline, paths in sorted(grouped.items()) if paths}


def render_v1_1_baseline_return_manifest_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Return Manifest",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Manifest complete: `{bool(report.get('complete'))}`",
        f"Artifacts complete: `{bool(report.get('artifacts_complete'))}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Shard commands: `{report.get('shard_commands')}`",
        f"Requested baselines: `{json.dumps(report.get('requested_baselines') or [], ensure_ascii=False)}`",
        f"Selected baselines: `{json.dumps(report.get('selected_baselines') or [], ensure_ascii=False)}`",
        f"Missing requested baselines: `{json.dumps(report.get('missing_requested_baselines') or [], ensure_ascii=False)}`",
        "",
        "## Summary",
        "",
        f"- `file_count`: `{report.get('file_count')}`",
        f"- `required_file_count`: `{report.get('required_file_count')}`",
        f"- `missing_required_files`: `{len(report.get('missing_required_files') or [])}`",
        f"- `missing_optional_files`: `{len(report.get('missing_optional_files') or [])}`",
        "",
    ]
    missing_required = report.get("missing_required_files") or []
    if missing_required:
        lines.extend(["## Missing Required Files", ""])
        for path in missing_required:
            lines.append(f"- `{path}`")
        lines.append("")
    missing_required_by_baseline = report.get("missing_required_files_by_baseline") or {}
    if missing_required_by_baseline:
        lines.extend(["## Missing Required Files By Baseline", ""])
        for baseline, paths in sorted(missing_required_by_baseline.items()):
            lines.append(f"- `{baseline}`: `{len(paths)}`")
            for path in paths:
                lines.append(f"  - `{path}`")
        lines.append("")
    lines.extend(["## Files", ""])
    for entry in report.get("artifacts") or []:
        required = "required" if entry.get("required") else "optional"
        exists = "exists" if entry.get("exists") else "missing"
        lines.append(f"- `{entry.get('path')}` ({required}, `{entry.get('baseline')}`, `{entry.get('role')}`, {exists})")
    lines.append("")
    return "\n".join(lines)


def write_v1_1_baseline_return_acceptance(
    handoff_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    return_manifest_path: Path | None = None,
    completion_json_path: Path | None = None,
) -> dict[str, Any]:
    handoff = read_json(handoff_path, {})
    acceptance = handoff.get("external_acceptance") if isinstance(handoff, dict) else {}
    if not isinstance(acceptance, dict):
        acceptance = {}
    if not acceptance and isinstance(handoff, dict):
        acceptance = build_v1_1_handoff_external_acceptance(
            handoff.get("jobs") or [],
            handoff.get("transfer_includes") or [],
            return_manifest_path or ((handoff.get("finalization") or {}).get("return_manifest") if isinstance(handoff.get("finalization"), dict) else None),
        )

    finalization = handoff.get("finalization") if isinstance(handoff, dict) else {}
    return_manifest = return_manifest_path
    if return_manifest is None and acceptance.get("return_manifest"):
        return_manifest = Path(str(acceptance.get("return_manifest")))
    if return_manifest is None and isinstance(finalization, dict) and finalization.get("return_manifest"):
        return_manifest = Path(str(finalization.get("return_manifest")))
    if return_manifest is None and isinstance(handoff, dict):
        finalize_cmd = handoff_command_options(handoff, "v1-1-finalize-baselines")
        return_manifest = command_path_optional(finalize_cmd, "--return-manifest")

    completion_json = completion_json_path
    if completion_json is None and isinstance(handoff, dict):
        completion_cmd = handoff_command_options(handoff, "report-v1-1-completion-audit")
        completion_json = command_path_optional(completion_cmd, "--json-out")

    return_manifest_payload = read_json(return_manifest, {}) if return_manifest else {}
    completion_payload = read_json(completion_json, {}) if completion_json else {}
    required_return_files = [str(path) for path in (acceptance.get("required_return_files") or [])]
    manifest_required_files = return_manifest_payload.get("required_files") if isinstance(return_manifest_payload, dict) else []
    if manifest_required_files:
        required_return_files = [str(path) for path in manifest_required_files]
    missing_required_files = (
        [str(path) for path in (return_manifest_payload.get("missing_required_files") or [])]
        if isinstance(return_manifest_payload, dict) and "missing_required_files" in return_manifest_payload
        else [path for path in required_return_files if not Path(path).is_file()]
    )
    artifacts_complete = bool(required_return_files) and not missing_required_files
    if isinstance(return_manifest_payload, dict) and "artifacts_complete" in return_manifest_payload:
        artifacts_complete = bool(return_manifest_payload.get("artifacts_complete"))
    completion_overall_status = (
        str(completion_payload.get("overall_status"))
        if isinstance(completion_payload, dict) and completion_payload.get("overall_status")
        else None
    )
    return_manifest_generated_at = (
        str(return_manifest_payload.get("generated_at"))
        if isinstance(return_manifest_payload, dict) and return_manifest_payload.get("generated_at")
        else None
    )
    completion_audit_generated_at = (
        str(completion_payload.get("generated_at"))
        if isinstance(completion_payload, dict) and completion_payload.get("generated_at")
        else None
    )
    completion_gate_path = str(completion_json) if completion_json else "completion_audit_v19.json"
    completion_gate = (
        f"Apply a verified return bundle and rerun v1-1-finalize-baselines until {completion_gate_path} "
        "reports overall_status=complete. The full generated run/apply scripts refresh this report with "
        "--require-complete, so the final acceptance command exits nonzero until this gate is complete."
    )
    external_baselines = [str(baseline) for baseline in (acceptance.get("external_baselines") or [])]
    required_return_files_by_baseline = group_return_files_by_baseline(required_return_files, external_baselines)
    missing_required_files_by_baseline = group_return_files_by_baseline(missing_required_files, external_baselines)
    current_status = {
        "artifacts_complete": artifacts_complete,
        "return_manifest_exists": bool(return_manifest and return_manifest.is_file()),
        "return_manifest_generated_at": return_manifest_generated_at,
        "completion_audit_exists": bool(completion_json and completion_json.is_file()),
        "completion_audit_generated_at": completion_audit_generated_at,
        "completion_audit_overall_status": completion_overall_status,
        "missing_required_file_count": len(missing_required_files),
        "missing_required_files": missing_required_files,
        "missing_required_files_by_baseline": missing_required_files_by_baseline,
    }
    report = {
        "generated_at": utc_now(),
        "complete": bool(artifacts_complete and completion_overall_status == "complete"),
        "handoff": str(handoff_path),
        "return_manifest": str(return_manifest) if return_manifest else None,
        "completion_json": str(completion_json) if completion_json else None,
        "external_baselines": external_baselines,
        "artifacts_complete": current_status["artifacts_complete"],
        "return_manifest_exists": current_status["return_manifest_exists"],
        "return_manifest_generated_at": current_status["return_manifest_generated_at"],
        "completion_audit_exists": current_status["completion_audit_exists"],
        "completion_audit_generated_at": current_status["completion_audit_generated_at"],
        "completion_audit_overall_status": current_status["completion_audit_overall_status"],
        "required_return_file_count": len(required_return_files),
        "required_return_files": required_return_files,
        "required_return_files_by_baseline": required_return_files_by_baseline,
        "missing_required_file_count": current_status["missing_required_file_count"],
        "missing_required_files": current_status["missing_required_files"],
        "missing_required_files_by_baseline": current_status["missing_required_files_by_baseline"],
        "run_scripts": [str(path) for path in (acceptance.get("run_scripts") or [])],
        "return_packaging_scripts": [str(path) for path in (acceptance.get("return_packaging_scripts") or [])],
        "return_apply_scripts": [str(path) for path in (acceptance.get("return_apply_scripts") or [])],
        "completion_gate": completion_gate,
        "current_status": current_status,
        "notes": [
            "This is the compact acceptance checklist for the external required baseline runner.",
            "Completion requires all final required return files, a verified return bundle/apply workflow, and a completion audit with overall_status=complete.",
        ],
    }
    report = preserve_generated_at_if_report_unchanged(report, out_path)
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_return_acceptance_markdown(report), encoding="utf-8")
    return report


def group_return_files_by_baseline(paths: Iterable[str], baselines: Iterable[str]) -> dict[str, list[str]]:
    baseline_values = [str(baseline) for baseline in baselines]
    grouped: dict[str, list[str]] = {baseline: [] for baseline in baseline_values}
    unassigned: list[str] = []
    for path_value in paths:
        path = str(path_value)
        name = Path(path).name
        matched_baseline = next(
            (baseline for baseline in baseline_values if name.startswith(f"{baseline}_") or baseline in name),
            None,
        )
        if matched_baseline:
            grouped.setdefault(matched_baseline, []).append(path)
        else:
            unassigned.append(path)
    if unassigned:
        grouped["unassigned"] = unassigned
    return {baseline: values for baseline, values in grouped.items() if values}


def render_v1_1_baseline_return_acceptance_markdown(report: dict[str, Any]) -> str:
    status = report.get("current_status") or {}
    lines = [
        "# V1.1 Baseline Return Acceptance",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Overall status: `{'complete' if report.get('complete') else 'not_complete'}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Return manifest: `{report.get('return_manifest')}`",
        f"Completion audit: `{report.get('completion_json')}`",
        "",
        "## Required Baselines",
        "",
    ]
    for baseline in report.get("external_baselines") or []:
        lines.append(f"- `{baseline}`")
    required_by_baseline = report.get("required_return_files_by_baseline") or {}
    if required_by_baseline:
        lines.extend(["", "## Required Return Files By Baseline", ""])
        for baseline, paths in required_by_baseline.items():
            lines.append(f"### {baseline}")
            lines.append("")
            for path in paths or []:
                lines.append(f"- `{path}`")
            lines.append("")
    lines.extend(["", "## Required Return Files", ""])
    for path in report.get("required_return_files") or []:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Scripts",
            "",
        ]
    )
    for path in report.get("run_scripts") or []:
        lines.append(f"- Run: `{path}`")
    for path in report.get("return_packaging_scripts") or []:
        lines.append(f"- Package: `{path}`")
    for path in report.get("return_apply_scripts") or []:
        lines.append(f"- Apply: `{path}`")
    lines.extend(
        [
            "",
            "## Completion Gate",
            "",
            str(report.get("completion_gate") or ""),
            "",
            "## Current Status",
            "",
            f"- Artifacts complete: `{bool(status.get('artifacts_complete'))}`",
            f"- Missing required files: `{int(status.get('missing_required_file_count') or 0)}`",
            f"- Completion audit status: `{status.get('completion_audit_overall_status')}`",
            f"- Return manifest generated at: `{status.get('return_manifest_generated_at')}`",
            f"- Completion audit generated at: `{status.get('completion_audit_generated_at')}`",
            "",
        ]
    )
    missing_files = status.get("missing_required_files") or []
    if missing_files:
        missing_by_baseline = status.get("missing_required_files_by_baseline") or {}
        if missing_by_baseline:
            lines.extend(["Missing required files by baseline:", ""])
            for baseline, paths in missing_by_baseline.items():
                lines.append(f"- `{baseline}`: `{len(paths or [])}`")
                for path in paths or []:
                    lines.append(f"  - `{path}`")
            lines.append("")
        lines.extend(["Missing required files:", ""])
        for path in missing_files:
            lines.append(f"- `{path}`")
        lines.append("")
    return "\n".join(lines)


def auto_merge_v1_1_baseline_shards(
    shard_commands_path: Path,
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    shard_report = read_json(shard_commands_path, {})
    baselines = shard_report.get("baselines") if isinstance(shard_report, dict) else []
    attempted: list[str] = []
    merged: list[str] = []
    blocked: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for baseline_report in baselines or []:
        baseline = str(baseline_report.get("baseline") or "")
        final_artifacts = baseline_report.get("final_artifacts") or {}
        details_path_value = final_artifacts.get("details")
        summary_path_value = final_artifacts.get("summary")
        shard_details = [
            Path(str((shard.get("artifacts") or {}).get("details")))
            for shard in baseline_report.get("shard_commands") or []
            if (shard.get("artifacts") or {}).get("details")
        ]
        if not baseline or not details_path_value or not summary_path_value or not shard_details:
            blocked.append({"baseline": baseline, "reason": "missing_shard_command_metadata"})
            continue
        attempted.append(baseline)
        missing_shard_details = [str(path) for path in shard_details if not path.is_file()]
        if missing_shard_details:
            merge_report = {
                "complete": False,
                "output": str(details_path_value),
                "wrote_output": False,
                "inputs": [{"path": str(path), "exists": path.is_file()} for path in shard_details],
                "rows": 0,
                "unique_sample_ids": 0,
                "expected_samples": expected_samples,
                "candidate_filter": "all_files",
                "missing_shard_details": missing_shard_details,
            }
            if final_artifacts.get("merge_report"):
                write_json(Path(str(final_artifacts["merge_report"])), merge_report)
            if final_artifacts.get("merge_markdown"):
                merge_markdown_path = Path(str(final_artifacts["merge_markdown"]))
                ensure_parent(merge_markdown_path)
                merge_markdown_path.write_text(
                    "# V1.1 Merged Details\n\n"
                    f"- Complete: `False`\n"
                    f"- Output: `{details_path_value}`\n"
                    f"- Wrote output: `False`\n"
                    f"- Rows: `0`\n"
                    f"- Expected samples: `{expected_samples}`\n\n"
                    "## Missing Shard Details\n\n"
                    + "\n".join(f"- `{path}`" for path in missing_shard_details)
                    + "\n",
                    encoding="utf-8",
                )
            blocked.append(
                {
                    "baseline": baseline,
                    "reason": "missing_shard_details",
                    "missing_shard_details": missing_shard_details,
                    "missing_shard_detail_count": len(missing_shard_details),
                    "merge": merge_report,
                }
            )
            results.append({"baseline": baseline, "merge": merge_report, "summary": None})
            continue
        merge_result = write_v1_1_merged_details(
            details_paths=shard_details,
            out_path=Path(str(details_path_value)),
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
            candidate_filter="all_files",
            report_out_path=Path(str(final_artifacts.get("merge_report"))) if final_artifacts.get("merge_report") else None,
            markdown_out_path=Path(str(final_artifacts.get("merge_markdown"))) if final_artifacts.get("merge_markdown") else None,
        )
        if not merge_result.get("complete"):
            blocked.append({"baseline": baseline, "reason": "merge_incomplete", "merge": merge_result})
            results.append({"baseline": baseline, "merge": merge_result, "summary": None})
            continue
        summary_result = write_v1_1_summary_from_details(
            details_path=Path(str(details_path_value)),
            out_path=Path(str(summary_path_value)),
            model=str(baseline_report.get("model") or baseline),
            mode="embedding",
            candidate_filter="all_files",
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
        )
        merged.append(baseline)
        results.append({"baseline": baseline, "merge": merge_result, "summary": summary_result})
    return {
        "generated_at": utc_now(),
        "complete": bool(attempted) and not blocked,
        "shard_commands": str(shard_commands_path),
        "attempted": attempted,
        "merged": merged,
        "blocked": blocked,
        "results": results,
    }


def write_v1_1_baseline_finalization(
    handoff_path: Path,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    shard_commands_path: Path | None = None,
    return_manifest_path: Path | None = None,
    return_manifest_markdown_path: Path | None = None,
    return_files_path: Path | None = None,
    include_shard_artifacts: bool = False,
    include_caches: bool = False,
    auto_merge_shards: bool = False,
    docs: Iterable[Path] | None = None,
    workflow_evidence_paths: Iterable[Path] | None = None,
    objective: str | None = V1_1_COMPLETION_OBJECTIVE,
) -> dict[str, Any]:
    handoff = read_json(handoff_path, {})
    inputs = handoff.get("inputs") if isinstance(handoff, dict) else {}
    report_dir = Path(str((inputs or {}).get("report_dir") or "data/reports/v1_1"))
    status_cmd = handoff_command_options(handoff, "v1-1-baseline-status")
    preflight_cmd = handoff_command_options(handoff, "check-baseline-summaries")
    leaderboard_cmd = handoff_command_options(handoff, "report-models")
    readiness_cmd = handoff_command_options(handoff, "v1-1-readiness")
    release_cmd = handoff_command_options(handoff, "report-v1-1")
    completion_cmd = handoff_command_options(handoff, "report-v1-1-completion-audit")

    derived = command_path(readiness_cmd, "--derived", command_path(status_cmd, "--derived", Path(str((inputs or {}).get("derived") or "data/benchmark/v1_1"))))
    base_derived = command_path(readiness_cmd, "--base-derived", Path(str((inputs or {}).get("base_derived") or "data/benchmark/v1")))
    assembly_manifest = command_path(readiness_cmd, "--manifest", Path(str((inputs or {}).get("assembly_manifest") or derived / "manifest.json")))
    corpus_manifest = command_path(readiness_cmd, "--corpus-manifest", Path(str((inputs or {}).get("corpus_manifest") or "data/corpus/v1_1/corpus_manifest.jsonl")))
    eval_dir = command_path(
        status_cmd,
        "--eval-dir",
        command_path(readiness_cmd, "--eval-dir", command_path(leaderboard_cmd, "--eval-dir", Path(str((inputs or {}).get("eval_dir") or "data/eval/v1_1")))),
    )
    baseline_status_path = command_path(status_cmd, "--out", report_dir / "baseline_status.json")
    baseline_status_markdown_path = command_path(status_cmd, "--markdown-out", report_dir / "baseline_status.md")
    baseline_preflight_path = command_path(preflight_cmd, "--out", report_dir / "baseline_summary_preflight.json")
    leaderboard_path = command_path(
        leaderboard_cmd,
        "--out",
        command_path(readiness_cmd, "--leaderboard", report_dir / "model_leaderboard.md"),
    )
    leaderboard_json_path = command_path(
        leaderboard_cmd,
        "--json-out",
        command_path(readiness_cmd, "--leaderboard-json", report_dir / "model_leaderboard.json"),
    )
    readiness_path = command_path(
        readiness_cmd,
        "--out",
        command_path(release_cmd, "--readiness", command_path(completion_cmd, "--readiness", report_dir / "readiness.json")),
    )
    readiness_markdown_path = command_path(readiness_cmd, "--markdown-out", report_dir / "readiness.md")
    release_path = command_path(release_cmd, "--out", report_dir / "release_report.md")
    release_json_path = command_path(
        release_cmd,
        "--json-out",
        command_path(completion_cmd, "--release-json", report_dir / "release_report.json"),
    )
    completion_path = command_path(completion_cmd, "--out", report_dir / "completion_audit.md")
    completion_json_path = command_path(completion_cmd, "--json-out", report_dir / "completion_audit.json")
    base_leaderboard_json_path = command_path_optional(release_cmd, "--base-leaderboard-json")
    required_baselines = command_values(leaderboard_cmd, "--required-baseline") or list(REQUIRED_V1_1_BASELINES)
    derived_rows = [row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)]
    expected_samples = len(derived_rows)
    expected_sample_ids = {str(row.get("id")) for row in derived_rows if row.get("id")}

    handoff_verification = verify_v1_1_baseline_handoff(handoff_path)
    auto_merge_report = (
        auto_merge_v1_1_baseline_shards(
            shard_commands_path=shard_commands_path,
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
        )
        if auto_merge_shards and shard_commands_path is not None
        else None
    )
    return_manifest = None
    if return_manifest_path is not None:
        return_manifest = write_v1_1_baseline_return_manifest(
            handoff_path=handoff_path,
            shard_commands_path=shard_commands_path,
            out_path=return_manifest_path,
            markdown_out_path=return_manifest_markdown_path,
            files_out_path=return_files_path,
            include_shard_artifacts=include_shard_artifacts,
            include_caches=include_caches,
        )

    summary_paths = sorted(eval_dir.glob("*_summary.json")) if eval_dir.exists() else []
    baseline_status = write_v1_1_baseline_status_report(
        summary_paths=summary_paths,
        expected_samples=expected_samples,
        out_path=baseline_status_path,
        markdown_out_path=baseline_status_markdown_path,
        expected_baselines=required_baselines,
        expected_sample_ids=expected_sample_ids,
        eval_dirs=[eval_dir],
        shard_commands_path=shard_commands_path,
    )
    baseline_preflight = check_required_baseline_summaries(
        summary_paths=summary_paths,
        expected_samples=expected_samples,
        expected_baselines=required_baselines,
        expected_sample_ids=expected_sample_ids,
    )
    write_json(baseline_preflight_path, baseline_preflight)
    leaderboard = report_model_leaderboard(
        eval_dir=eval_dir,
        out_path=leaderboard_path,
        json_out_path=leaderboard_json_path,
        required_baselines=required_baselines,
    )
    readiness = check_v1_1_readiness(
        sample_paths=sample_paths_from_derived(derived),
        base_sample_paths=sample_paths_from_derived(base_derived) if base_derived.exists() else [],
        corpus_manifest_path=corpus_manifest,
        assembly_manifest_path=assembly_manifest,
        eval_dir=eval_dir,
        leaderboard_path=leaderboard_path,
        leaderboard_json_path=leaderboard_json_path,
        out_path=readiness_path,
        markdown_out_path=readiness_markdown_path,
        min_comment2context=command_int(readiness_cmd, "--min-comment2context", 80),
        max_comment2context=command_int(readiness_cmd, "--max-comment2context", 100),
        min_comment_cross_module=command_int(readiness_cmd, "--min-comment-cross-module", 1),
        min_trace2code=command_int(readiness_cmd, "--min-trace2code", 100),
        min_trace_non_go_repos=command_int(readiness_cmd, "--min-trace-non-go-repos", 1),
        min_trace_languages=command_int(readiness_cmd, "--min-trace-languages", 2),
        min_trace_failure_types=command_int(readiness_cmd, "--min-trace-failure-types", 2),
    )
    release = report_v1_1_release(
        readiness_path=readiness_path,
        leaderboard_json_path=leaderboard_json_path,
        base_leaderboard_json_path=base_leaderboard_json_path,
        out_path=release_path,
        json_out_path=release_json_path,
    )
    completion = report_v1_1_completion_audit(
        readiness_path=readiness_path,
        release_json_path=release_json_path,
        baseline_status_path=baseline_status_path,
        leaderboard_json_path=leaderboard_json_path,
        out_path=completion_path,
        json_out_path=completion_json_path,
        docs=docs,
        workflow_evidence_paths=workflow_evidence_paths,
        objective=objective,
    )
    complete = bool(
        handoff_verification.get("complete")
        and (return_manifest is None or return_manifest.get("artifacts_complete"))
        and baseline_status.get("complete")
        and baseline_preflight.get("complete")
        and leaderboard.get("contains_required_baselines")
        and readiness.get("ready")
        and release.get("status") == "ready"
        and completion.get("overall_status") == "complete"
    )
    next_required_action = None if complete else (completion.get("summary") or {}).get("next_required_action")
    report = {
        "generated_at": utc_now(),
        "complete": complete,
        "next_required_action": next_required_action,
        "handoff": str(handoff_path),
        "paths": {
            "derived": str(derived),
            "base_derived": str(base_derived),
            "eval_dir": str(eval_dir),
            "assembly_manifest": str(assembly_manifest),
            "corpus_manifest": str(corpus_manifest),
            "baseline_status": str(baseline_status_path),
            "baseline_status_markdown": str(baseline_status_markdown_path),
            "baseline_preflight": str(baseline_preflight_path),
            "leaderboard": str(leaderboard_path),
            "leaderboard_json": str(leaderboard_json_path),
            "readiness": str(readiness_path),
            "readiness_markdown": str(readiness_markdown_path),
            "release": str(release_path),
            "release_json": str(release_json_path),
            "completion": str(completion_path),
            "completion_json": str(completion_json_path),
            "shard_commands": str(shard_commands_path) if shard_commands_path else None,
            "return_manifest": str(return_manifest_path) if return_manifest_path else None,
        },
        "steps": {
            "handoff_verification": {
                "complete": bool(handoff_verification.get("complete")),
                "failed_checks": handoff_verification.get("failed_checks") or [],
            },
            "return_manifest": (
                {
                    "complete": bool(return_manifest.get("complete")),
                    "artifacts_complete": bool(return_manifest.get("artifacts_complete")),
                    "missing_required_files": return_manifest.get("missing_required_files") or [],
                }
                if return_manifest is not None
                else None
            ),
            "auto_merge_shards": (
                {
                    "complete": bool(auto_merge_report.get("complete")),
                    "attempted": auto_merge_report.get("attempted") or [],
                    "merged": auto_merge_report.get("merged") or [],
                    "blocked": auto_merge_report.get("blocked") or [],
                }
                if auto_merge_report is not None
                else None
            ),
            "baseline_status": {
                "complete": bool(baseline_status.get("complete")),
                "blocking_baselines": baseline_status.get("blocking_baselines") or [],
            },
            "baseline_preflight": {
                "complete": bool(baseline_preflight.get("complete")),
                "blocking_baselines": baseline_preflight.get("blocking_baselines") or [],
            },
            "leaderboard": {
                "contains_required_baselines": bool(leaderboard.get("contains_required_baselines")),
                "missing_required_baselines": leaderboard.get("missing_required_baselines") or [],
                "rows": leaderboard.get("rows"),
            },
            "readiness": {
                "ready": bool(readiness.get("ready")),
                "blocking_gates": readiness.get("blocking_gates") or [],
            },
            "release": {
                "status": release.get("status"),
                "blocking_gates": release.get("blocking_gates") or [],
            },
            "completion_audit": {
                "overall_status": completion.get("overall_status"),
                "blocked_or_partial_requirements": len(completion.get("blocked_or_partial_requirements") or []),
                "next_required_action": (completion.get("summary") or {}).get("next_required_action"),
            },
        },
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_1_baseline_finalization_markdown(report), encoding="utf-8")
    return report


def handoff_command_options(manifest: dict[str, Any], command_name: str) -> dict[str, Any]:
    for command in manifest.get("verification_commands") or []:
        parts = shlex.split(str(command))
        if command_name not in parts:
            continue
        index = parts.index(command_name) + 1
        repeated: dict[str, list[Any]] = defaultdict(list)
        positionals: list[str] = []
        while index < len(parts):
            token = parts[index]
            if token.startswith("--"):
                if index + 1 < len(parts) and not parts[index + 1].startswith("--"):
                    repeated[token].append(parts[index + 1])
                    index += 2
                else:
                    repeated[token].append(True)
                    index += 1
            else:
                positionals.append(token)
                index += 1
        options = {key: values[-1] for key, values in repeated.items() if values}
        return {"found": True, "command": str(command), "options": options, "repeated": dict(repeated), "positionals": positionals}
    return {"found": False, "command": None, "options": {}, "repeated": {}, "positionals": []}


def handoff_completion_json_path(manifest: dict[str, Any]) -> Path | None:
    completion_cmd = handoff_command_options(manifest, "report-v1-1-completion-audit")
    return command_path_optional(completion_cmd, "--json-out")


def command_values(command: dict[str, Any], option: str) -> list[str]:
    return [str(value) for value in (command.get("repeated") or {}).get(option, []) if value is not True]


def command_path(command: dict[str, Any], option: str, default: Path) -> Path:
    value = (command.get("options") or {}).get(option)
    return Path(str(value)) if value not in (None, True) else default


def command_path_optional(command: dict[str, Any], option: str) -> Path | None:
    value = (command.get("options") or {}).get(option)
    return Path(str(value)) if value not in (None, True) else None


def command_int(command: dict[str, Any], option: str, default: int) -> int:
    value = (command.get("options") or {}).get(option)
    if value in (None, True):
        return default
    return int(value)


def render_v1_1_baseline_finalization_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Finalization",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Handoff: `{report.get('handoff')}`",
    ]
    if report.get("next_required_action"):
        lines.extend(["", "## Next Action", "", str(report.get("next_required_action"))])
    lines.extend(
        [
            "",
            "## Steps",
            "",
            "| Step | Complete | Details |",
            "| --- | --- | --- |",
        ]
    )
    for name, details in (report.get("steps") or {}).items():
        if details is None:
            lines.append(f"| `{name}` | `n/a` | Not requested. |")
            continue
        complete = details.get("complete")
        if complete is None:
            complete = details.get("ready")
        if complete is None:
            complete = details.get("status") == "ready" if details.get("status") is not None else details.get("overall_status") == "complete"
        summary = summarize_v1_1_baseline_finalization_step(name, details)
        lines.append(
            "| `{name}` | `{complete}` | {details} |".format(
                name=name,
                complete=bool(complete),
                details=summary.replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Paths", ""])
    for key, value in (report.get("paths") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def summarize_v1_1_baseline_finalization_step(name: str, details: dict[str, Any]) -> str:
    if name == "return_manifest":
        missing = details.get("missing_required_files") or []
        return "artifacts_complete={}, missing_required_files={}".format(bool(details.get("artifacts_complete")), len(missing))
    if name == "auto_merge_shards":
        blocked = details.get("blocked") or []
        blocked_summary = ", ".join(
            "{}:{}{}".format(
                item.get("baseline"),
                item.get("reason"),
                f"({item['missing_shard_detail_count']})" if item.get("missing_shard_detail_count") is not None else "",
            )
            for item in blocked
        )
        return "attempted={}, merged={}, blocked={}{}".format(
            len(details.get("attempted") or []),
            len(details.get("merged") or []),
            len(blocked),
            f"; {blocked_summary}" if blocked_summary else "",
        )
    if name in {"baseline_status", "baseline_preflight"}:
        blocking = details.get("blocking_baselines") or []
        return "blocking_baselines={}{}".format(len(blocking), f"; {', '.join(blocking)}" if blocking else "")
    if name == "leaderboard":
        missing = details.get("missing_required_baselines") or []
        return "rows={}, missing_required_baselines={}{}".format(
            details.get("rows"),
            len(missing),
            f"; {', '.join(missing)}" if missing else "",
        )
    if name in {"readiness", "release"}:
        blockers = details.get("blocking_gates") or []
        status = details.get("status") if name == "release" else details.get("ready")
        return "status={}, blocking_gates={}{}".format(status, len(blockers), f"; {', '.join(blockers)}" if blockers else "")
    if name == "completion_audit":
        return "overall_status={}, blocked_or_partial_requirements={}, next_required_action={}".format(
            details.get("overall_status"),
            details.get("blocked_or_partial_requirements"),
            details.get("next_required_action"),
        )
    if name == "handoff_verification":
        failed = details.get("failed_checks") or []
        return "failed_checks={}".format(len(failed))
    return json.dumps(details, ensure_ascii=False, sort_keys=True)


def replace_or_append_cli_options(command: str, option_values: dict[str, Any]) -> str:
    parts = shlex.split(command)
    result: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(parts):
        part = parts[index]
        if part in option_values:
            value = option_values[part]
            if value is not None:
                result.extend([part, str(value)])
            seen.add(part)
            index += 2
            continue
        result.append(part)
        index += 1
    for option, value in option_values.items():
        if option not in seen and value is not None:
            result.extend([option, str(value)])
    return shlex.join(result)


def shard_eval_artifact_path(path: Path, shard_name: str, artifact_kind: str) -> Path:
    stem = path.stem
    if artifact_kind == "summary" and stem.endswith("_summary"):
        return path.with_name(f"{stem.removesuffix('_summary')}_{shard_name}_summary{path.suffix}")
    if artifact_kind == "details" and stem.endswith("_details"):
        return path.with_name(f"{stem.removesuffix('_details')}_{shard_name}_details{path.suffix}")
    return path.with_name(f"{stem}_{shard_name}{path.suffix}")


def shard_cache_artifact_path(path: Path, shard_name: str) -> Path:
    return path.with_name(f"{path.name}_{shard_name}")


def shard_shared_text_cache_path(path: Path, shard_name: str) -> Path:
    stem = path.stem
    if stem.endswith("_texts"):
        return path.with_name(f"{stem.removesuffix('_texts')}_{shard_name}_texts{path.suffix}")
    return path.with_name(f"{stem}_{shard_name}{path.suffix}")


def render_v1_1_baseline_shard_commands_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Shard Commands",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Complete: `{bool(report.get('complete'))}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Sample shards: `{report.get('sample_shards')}`",
        f"Shard count: `{report.get('shard_count')}`",
        f"Sample count: `{report.get('sample_count')}`",
        "",
    ]
    missing = report.get("missing_shard_files") or []
    if missing:
        lines.extend(["## Missing Shard Files", ""])
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    for baseline in report.get("baselines") or []:
        lines.extend(["## " + str(baseline.get("baseline")), ""])
        for shard in baseline.get("shard_commands") or []:
            lines.append(f"### shard {shard.get('index')}")
            lines.append("")
            lines.append("```bash")
            lines.append(str(shard.get("command")))
            lines.append("```")
            lines.append("")
        lines.append("### Merge")
        lines.append("")
        lines.append("```bash")
        lines.append(str(baseline.get("merge_command")))
        lines.append("```")
        lines.append("")
        lines.append("### Summary")
        lines.append("")
        lines.append("```bash")
        lines.append(str(baseline.get("summary_from_details_command")))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_v1_1_baseline_transfer_manifest_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Baseline Transfer Manifest",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Overall status: `{'complete' if report.get('complete') else 'not_complete'}`",
        "",
        "## Summary",
        "",
        f"- `file_count`: `{report.get('file_count')}`",
        f"- `corpus_rows`: `{report.get('corpus_rows')}`",
        f"- `chunk_files`: `{report.get('chunk_files')}`",
        f"- `chunk_count`: `{report.get('chunk_count')}`",
        f"- `chunk_file_size_bytes`: `{report.get('chunk_file_size_bytes')}`",
        f"- `included_file_count`: `{report.get('included_file_count')}`",
        f"- `included_file_fingerprints`: `{len(report.get('included_file_fingerprints') or [])}`",
        f"- `unfingerprinted_included_files`: `{len(report.get('unfingerprinted_included_files') or [])}`",
        f"- `generated_output_files`: `{len(report.get('generated_output_files') or [])}`",
        f"- `missing_files`: `{len(report.get('missing_files') or [])}`",
        "",
    ]
    missing = report.get("missing_files") or []
    if missing:
        lines.extend(["## Missing Files", ""])
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    fingerprints = report.get("included_file_fingerprints") or []
    if fingerprints:
        lines.extend(["## Included File Fingerprints", ""])
        lines.append("| Path | SHA256 | Size |")
        lines.append("| --- | --- | ---: |")
        for item in fingerprints:
            lines.append(f"| `{item.get('path')}` | `{item.get('sha256')}` | `{item.get('size_bytes')}` |")
        lines.append("")
    lines.extend(["## Transfer Files", ""])
    for path in report.get("files") or []:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def verify_sample_collection_fingerprint(name: str, expected: dict[str, Any]) -> dict[str, Any]:
    path_value = expected.get("path")
    actual = sample_collection_fingerprint(Path(path_value)) if path_value else {"path": path_value, "exists": False, "files": []}
    mismatches = compare_fields(
        expected,
        actual,
        ("exists", "rows", "unique_ids", "duplicate_ids", "duplicate_ids_truncated", "task_counts", "errors"),
    )
    expected_files = {item.get("path"): item for item in expected.get("files") or []}
    actual_files = {item.get("path"): item for item in actual.get("files") or []}
    missing_files = sorted(path for path in expected_files if path not in actual_files)
    unexpected_files = sorted(path for path in actual_files if path not in expected_files)
    if missing_files:
        mismatches.append(f"missing files: {missing_files}")
    if unexpected_files:
        mismatches.append(f"unexpected files: {unexpected_files}")
    for file_path, expected_file in sorted(expected_files.items()):
        actual_file = actual_files.get(file_path)
        if actual_file is None:
            continue
        for mismatch in compare_fields(expected_file, actual_file, ("exists", "type", "size_bytes", "sha256", "line_count")):
            mismatches.append(f"{file_path}: {mismatch}")
    return {
        "name": name,
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def verify_file_fingerprint(name: str, expected: dict[str, Any]) -> dict[str, Any]:
    path_value = expected.get("path")
    actual = file_fingerprint(Path(path_value)) if path_value else {"path": path_value, "exists": False, "type": "missing"}
    mismatches = compare_fields(expected, actual, ("exists", "type", "size_bytes", "sha256", "line_count"))
    return {
        "name": name,
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def compare_fields(expected: dict[str, Any], actual: dict[str, Any], fields: Iterable[str]) -> list[str]:
    mismatches: list[str] = []
    for field in fields:
        if expected.get(field) != actual.get(field):
            mismatches.append(f"{field}: expected {expected.get(field)!r}, found {actual.get(field)!r}")
    return mismatches


def sample_collection_fingerprint(path: Path) -> dict[str, Any]:
    files = sample_paths_from_derived(path) if path.exists() else []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for sample_path in files:
        try:
            rows.extend(read_jsonl(sample_path))
        except Exception as exc:  # pragma: no cover - defensive report path
            errors.append(f"{sample_path}: {exc}")
    ids = [str(row.get("id")) for row in rows if row.get("id")]
    id_counts = Counter(ids)
    duplicate_ids = sorted(sample_id for sample_id, count in id_counts.items() if count > 1)
    return {
        "path": str(path),
        "exists": path.exists(),
        "files": [file_fingerprint(sample_path) for sample_path in files],
        "rows": len(rows),
        "unique_ids": len(id_counts),
        "duplicate_ids": duplicate_ids[:20],
        "duplicate_ids_truncated": len(duplicate_ids) > 20,
        "task_counts": dict(sorted(Counter(str(row.get("task_type")) for row in rows if row.get("task_type")).items())),
        "errors": errors,
    }


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "type": "missing", "size_bytes": 0, "sha256": None, "line_count": 0}
    if not path.is_file():
        return {"path": str(path), "exists": True, "type": "directory", "size_bytes": None, "sha256": None, "line_count": None}
    digest = hashlib.sha256()
    line_count = 0
    size_bytes = 0
    last_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size_bytes += len(chunk)
            line_count += chunk.count(b"\n")
            digest.update(chunk)
            last_byte = chunk[-1:]
    if size_bytes and last_byte != b"\n":
        line_count += 1
    return {
        "path": str(path),
        "exists": True,
        "type": "file",
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "line_count": line_count,
    }


def frozen_v1_fingerprint_status(readiness: dict[str, Any]) -> dict[str, Any]:
    inputs = readiness.get("inputs") if isinstance(readiness, dict) else {}
    base_samples = [Path(str(path)) for path in (inputs.get("base_samples") if isinstance(inputs, dict) else []) or []]
    canonical_base_samples = Path("data/benchmark/v1/samples.jsonl")
    canonical_base_samples_resolved = canonical_base_samples.resolve()
    applicable = any(
        path == canonical_base_samples or (path.is_absolute() and path.resolve() == canonical_base_samples_resolved)
        for path in base_samples
    )
    checks: list[dict[str, Any]] = []
    if applicable:
        for path_value, expected in V1_FROZEN_FILE_FINGERPRINTS.items():
            actual = file_fingerprint(Path(path_value))
            mismatches = []
            for key in ("exists", "type", "size_bytes", "sha256", "line_count"):
                if actual.get(key) != expected.get(key):
                    mismatches.append(f"{key}: expected {expected.get(key)!r}, found {actual.get(key)!r}")
            checks.append(
                {
                    "path": path_value,
                    "expected": expected,
                    "actual": actual,
                    "mismatches": mismatches,
                    "complete": not mismatches,
                }
            )
    return {
        "applicable": applicable,
        "complete": (all(bool(check.get("complete")) for check in checks) if applicable else True),
        "expected_paths": list(V1_FROZEN_FILE_FINGERPRINTS),
        "base_sample_paths": [str(path) for path in base_samples],
        "checks": checks,
        "reason": None if applicable else "canonical_data_benchmark_v1_samples_not_used",
    }


def embedding_job(
    label: str,
    command: str,
    model: str,
    derived: Path,
    corpus: Path,
    eval_dir: Path,
    cache_root: Path,
    batch_size: int,
    extra_args: list[str],
    required_env: list[str] | None = None,
) -> dict[str, Any]:
    summary = eval_dir / f"{label}_summary.json"
    details = eval_dir / f"{label}_details.jsonl"
    cache = cache_root / label
    shared_text_cache = cache_root / f"{label}_texts.sqlite"
    argv: list[Any] = [
        command,
        "--model",
        model,
        "--derived",
        derived,
        "--corpus",
        corpus,
        "--out",
        summary,
        "--details",
        details,
        "--cache",
        cache,
        "--shared-text-cache",
        shared_text_cache,
        "--candidate-filter",
        "all_files",
        "--batch-size",
        str(batch_size),
        *extra_args,
        "--resume-details",
        "--no-keep-list",
    ]
    return {
        "baseline": label,
        "model": model,
        "command": cli_command(*argv),
        "required_env": required_env or [],
        "artifacts": {
            "summary": str(summary),
            "details": str(details),
            "cache": str(cache),
            "shared_text_cache": str(shared_text_cache),
        },
    }


def cli_command(*parts: Any) -> str:
    return " ".join(["PYTHONPATH=src", "python3", "-m", "agent_retrieval_bench.cli", *[shlex.quote(str(part)) for part in parts]])


def handoff_workflow_evidence_paths(handoff_payload: dict[str, Any]) -> list[Path]:
    """Return finalization evidence paths that do not depend on completion audit output.

    The return-acceptance report is deliberately excluded: its `complete` field
    depends on the completion audit already being complete, so using it as
    workflow evidence would create a circular finalization gate.
    """
    evidence: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        evidence_path = Path(str(path))
        key = str(evidence_path)
        if key in seen:
            return
        seen.add(key)
        evidence.append(evidence_path)

    handoff_verification = handoff_payload.get("handoff_verification") if isinstance(handoff_payload, dict) else {}
    if isinstance(handoff_verification, dict) and handoff_verification.get("report"):
        add(handoff_verification["report"])
    transfer_unpack_script = handoff_payload.get("transfer_unpack_script") if isinstance(handoff_payload, dict) else {}
    if isinstance(transfer_unpack_script, dict):
        add(transfer_unpack_script.get("transfer_verify"))
        add(transfer_unpack_script.get("handoff_verify"))
    finalization = handoff_payload.get("finalization") if isinstance(handoff_payload, dict) else {}
    for path in (finalization.get("workflow_evidence") if isinstance(finalization, dict) else []) or []:
        add(path)
    return evidence


def render_v1_1_baseline_handoff_markdown(manifest: dict[str, Any]) -> str:
    transfer_bundle = manifest.get("transfer_bundle") or {}
    transfer_unpack_script = manifest.get("transfer_unpack_script") or {}
    bundle = str(transfer_bundle.get("bundle") or "")
    checksum = str(transfer_bundle.get("checksum") or "")
    unpack_script = str(transfer_unpack_script.get("script") or "")
    unpack_destination = str(transfer_unpack_script.get("destination") or "agent-retrieval-bench-v1_1-transfer")
    lines = [
        "# V1.1 Baseline Handoff",
        "",
        f"Generated at: `{manifest.get('generated_at')}`",
        "",
        "## Minimal External Runner Path",
        "",
        "1. Verify the transfer bundle checksum, then unpack it before any GPU/API work.",
        "2. Run `v1-1-verify-transfer-manifest` and `v1-1-verify-handoff` in the unpacked checkout.",
        "3. Install the local package plus embedding dependencies, including `numpy`, `sentence-transformers`, and a CUDA-capable Torch build, in the unpacked checkout.",
        "4. Run the generated shard script or schedule the shard commands directly for the required open-source embedding baselines.",
        "5. Package returned artifacts with the generated return-bundle script, then apply that verified bundle on the local reporting checkout.",
        "6. Publish only after the final completion audit reports `overall_status=complete`.",
        "",
        "The full return-bundle verifier report is required workflow evidence; copying raw summaries/details is not enough for completion.",
        "",
    ]
    if bundle and checksum and unpack_script:
        lines.extend(
            [
                "Current prepared transfer bootstrap:",
                "",
                "```bash",
                f"bash {Path(unpack_script).name} {Path(bundle).name} {Path(checksum).name} {unpack_destination}",
                "```",
                "",
                "Copy these files beside each other on the external runner before running that command:",
                f"- `{unpack_script}`",
                f"- `{bundle}`",
                f"- `{checksum}`",
                "",
                "The bootstrap verifies the supplied bundle path against the checksum hash, so basename copies work even when the checksum sidecar records a generated repo-relative path.",
                "",
            ]
        )
    acceptance = manifest.get("external_acceptance") or {}
    if acceptance:
        lines.extend(
            [
                "## Acceptance Checklist",
                "",
                f"- External baselines: `{json.dumps(acceptance.get('external_baselines') or [])}`",
                f"- Required final return files: `{int(acceptance.get('required_return_file_count') or 0)}`",
                f"- Return manifest: `{acceptance.get('return_manifest')}`",
                f"- Completion gate: `{acceptance.get('completion_gate')}`",
                "",
            ]
        )
        required_return_files = acceptance.get("required_return_files") or []
        if required_return_files:
            lines.extend(["Required files:", ""])
            for path in required_return_files:
                lines.append(f"- `{path}`")
            lines.append("")
        run_scripts = acceptance.get("run_scripts") or []
        if run_scripts:
            lines.extend(["Run scripts:", ""])
            for path in run_scripts:
                lines.append(f"- `{path}`")
            lines.append("")
        package_scripts = acceptance.get("return_packaging_scripts") or []
        apply_scripts = acceptance.get("return_apply_scripts") or []
        if package_scripts or apply_scripts:
            lines.extend(["Return bundle scripts:", ""])
            for path in package_scripts:
                lines.append(f"- `{path}`")
            for path in apply_scripts:
                lines.append(f"- `{path}`")
            lines.append("")
    lines.extend(["## Inputs", ""])
    for key, value in (manifest.get("inputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if transfer_bundle:
        lines.extend(["", "## Transfer Bundle", ""])
        for key, value in transfer_bundle.items():
            lines.append(f"- `{key}`: `{value}`")
    if transfer_unpack_script:
        lines.extend(["", "## Transfer Unpack Script", ""])
        for key, value in transfer_unpack_script.items():
            lines.append(f"- `{key}`: `{value}`")
    transfer_includes = manifest.get("transfer_includes") or []
    if transfer_includes:
        lines.extend(["", "## Transfer Includes", ""])
        for path in transfer_includes:
            lines.append(f"- `{path}`")
    fingerprints = manifest.get("input_fingerprints") or {}
    if fingerprints:
        lines.extend(["", "## Input Fingerprints", ""])
        algorithm = fingerprints.get("algorithm") or "sha256"
        lines.append(f"- Algorithm: `{algorithm}`")
        for key in ("derived_samples", "base_samples"):
            collection = fingerprints.get(key) or {}
            lines.append(
                f"- `{key}`: rows `{collection.get('rows')}`, unique IDs `{collection.get('unique_ids')}`, files `{len(collection.get('files') or [])}`"
            )
            for item in collection.get("files") or []:
                lines.append(f"  - `{item.get('path')}`: `{item.get('sha256')}`")
        for key in ("assembly_manifest", "corpus_manifest"):
            item = fingerprints.get(key) or {}
            lines.append(f"- `{key}`: `{item.get('sha256')}` (`{item.get('path')}`)")
    setup_commands = manifest.get("setup_commands") or []
    if setup_commands:
        lines.extend(["", "## Prepare Transfer", ""])
        for command in setup_commands:
            lines.append("```bash")
            lines.append(str(command))
            lines.append("```")
            lines.append("")
    lines.extend(["", "## Jobs", ""])
    for job in manifest.get("jobs") or []:
        lines.append(f"### {job.get('baseline')}")
        lines.append("")
        if job.get("required_env"):
            lines.append(f"- Required env: `{', '.join(job.get('required_env') or [])}`")
            lines.append("")
        lines.append("```bash")
        lines.append(str(job.get("command")))
        lines.append("```")
        lines.append("")
    lines.extend(["## Verify And Publish", ""])
    for command in manifest.get("verification_commands") or []:
        lines.append("```bash")
        lines.append(str(command))
        lines.append("```")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in manifest.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def render_v1_1_handoff_verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Handoff Verification",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Handoff: `{report.get('handoff')}`",
        f"Overall status: `{'complete' if report.get('complete') else 'not_complete'}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        status = "PASS" if check.get("status") == "pass" else "FAIL"
        lines.append(f"### {status} - {check.get('name')}")
        lines.append("")
        mismatches = check.get("mismatches") or []
        if mismatches:
            for mismatch in mismatches:
                lines.append(f"- {mismatch}")
        else:
            lines.append("- No mismatches.")
        lines.append("")
    return "\n".join(lines)


def report_v1_1_completion_audit(
    readiness_path: Path,
    release_json_path: Path,
    baseline_status_path: Path,
    leaderboard_json_path: Path,
    out_path: Path,
    json_out_path: Path,
    docs: Iterable[Path] | None = None,
    workflow_evidence_paths: Iterable[Path] | None = None,
    objective: str | None = V1_1_COMPLETION_OBJECTIVE,
) -> dict[str, Any]:
    readiness_payload = read_json(readiness_path, {})
    readiness = readiness_payload.get("summary") if isinstance(readiness_payload.get("summary"), dict) else readiness_payload
    if isinstance(readiness, dict) and not readiness.get("blocking_gates"):
        readiness = {**readiness, "blocking_gates": readiness_payload.get("blocking_gates") or []}
    release = read_json(release_json_path, {})
    baseline_status = read_json(baseline_status_path, {})
    leaderboard = read_json(leaderboard_json_path, {})
    doc_paths = list(docs or V1_1_RELEASE_DOCS)
    checklist = completion_checklist(
        readiness=readiness,
        release=release,
        baseline_status=baseline_status,
        leaderboard=leaderboard,
        readiness_path=readiness_path,
        release_json_path=release_json_path,
        baseline_status_path=baseline_status_path,
        leaderboard_json_path=leaderboard_json_path,
        docs=doc_paths,
        workflow_evidence_paths=list(workflow_evidence_paths or []),
    )
    blocked_or_partial = [item for item in checklist if item["status"] != "pass"]
    next_required_action = completion_next_action(blocked_or_partial)
    report = {
        "generated_at": utc_now(),
        "objective": objective or V1_1_COMPLETION_OBJECTIVE,
        "overall_status": "complete" if not blocked_or_partial else "not_complete",
        "success_criteria": [str(item.get("requirement")) for item in checklist],
        "requirement_count": len(checklist),
        "passed_requirement_count": len(checklist) - len(blocked_or_partial),
        "blocked_or_partial_requirement_count": len(blocked_or_partial),
        "next_required_action": next_required_action,
        "summary": {
            "samples": readiness.get("samples"),
            "new_samples": readiness.get("new_samples"),
            "counts_by_task": readiness.get("counts_by_task") or {},
            "new_counts_by_task": readiness.get("new_counts_by_task") or {},
            "target_gaps": readiness.get("target_gaps") or {},
            "readiness_ready": bool(readiness.get("ready")),
            "readiness_blocking_gates": readiness.get("blocking_gates") or [],
            "release_status": release.get("status"),
            "baseline_status_complete": bool(baseline_status.get("complete")),
            "missing_required_baselines": (
                (leaderboard.get("missing_required_baselines") or [])
                or ((readiness.get("leaderboard") or {}).get("missing_baselines") or [])
            ),
            "next_required_action": next_required_action,
        },
        "checklist": checklist,
        "blocked_or_partial_requirements": blocked_or_partial,
    }
    write_json(json_out_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_v1_1_completion_audit_markdown(report), encoding="utf-8")
    return report


def completion_checklist(
    readiness: dict[str, Any],
    release: dict[str, Any],
    baseline_status: dict[str, Any],
    leaderboard: dict[str, Any],
    readiness_path: Path,
    release_json_path: Path,
    baseline_status_path: Path,
    leaderboard_json_path: Path,
    docs: list[Path],
    workflow_evidence_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    gates = readiness.get("gates") or {}
    target_gaps = readiness.get("target_gaps") or {}
    counts = readiness.get("counts_by_task") or {}
    base_counts = readiness.get("base_counts_by_task") or {}
    new_counts = readiness.get("new_counts_by_task") or {}
    evidence_readiness = [str(readiness_path)]
    frozen_v1_status = frozen_v1_fingerprint_status(readiness)
    frozen_v1_evidence = evidence_readiness + (
        list(V1_FROZEN_FILE_FINGERPRINTS) if frozen_v1_status.get("applicable") else []
    )
    checklist = [
        completion_item(
            "Keep frozen V1 unchanged and preserve all V1 IDs",
            all(
                bool(gates.get(name))
                for name in (
                    "sample_paths_do_not_point_at_benchmark_v1",
                    "contains_all_base_v1_ids",
                    "code2test_count_preserved",
                )
            )
            and bool(frozen_v1_status.get("complete")),
            frozen_v1_evidence,
            {
                "sample_paths_do_not_point_at_benchmark_v1": gates.get("sample_paths_do_not_point_at_benchmark_v1"),
                "contains_all_base_v1_ids": gates.get("contains_all_base_v1_ids"),
                "code2test_count_preserved": gates.get("code2test_count_preserved"),
                "missing_base_ids": readiness.get("missing_base_ids") or [],
                "frozen_v1_fingerprints": frozen_v1_status,
            },
        ),
        completion_item(
            "Expand comment2context from 51 to 80-100 high-quality samples",
            bool(gates.get("comment2context_count_ge_target") and gates.get("comment2context_count_le_target_band")),
            evidence_readiness,
            {
                "base": base_counts.get("comment2context"),
                "new": new_counts.get("comment2context"),
                "v1_1": counts.get("comment2context"),
                "target_gap": target_gaps.get("comment2context_samples"),
            },
        ),
        completion_item(
            "Expand trace2code from 68 to 100+ samples",
            bool(gates.get("trace2code_count_ge_target")),
            evidence_readiness,
            {
                "base": base_counts.get("trace2code"),
                "new": new_counts.get("trace2code"),
                "v1_1": counts.get("trace2code"),
                "target_gap": target_gaps.get("trace2code_samples"),
            },
        ),
        completion_item(
            "Do not expand code2test",
            bool(gates.get("code2test_count_preserved")),
            evidence_readiness,
            {"v1_1": counts.get("code2test"), "base": base_counts.get("code2test")},
        ),
        completion_item(
            "Quality gates: no leakage, direct hints, path-role overlap, or unclear semantics",
            all(
                bool(gates.get(name))
                for name in (
                    "new_samples_schema_valid",
                    "new_samples_have_clear_task_semantics",
                    "new_samples_no_path_role_overlap",
                    "new_samples_no_query_leakage",
                    "new_samples_no_direct_gold_hints",
                )
            ),
            evidence_readiness,
            {
                "new_samples_schema_valid": gates.get("new_samples_schema_valid"),
                "new_samples_have_clear_task_semantics": gates.get("new_samples_have_clear_task_semantics"),
                "new_samples_no_path_role_overlap": gates.get("new_samples_no_path_role_overlap"),
                "new_samples_no_query_leakage": gates.get("new_samples_no_query_leakage"),
                "new_samples_no_direct_gold_hints": gates.get("new_samples_no_direct_gold_hints"),
            },
        ),
        completion_item(
            "Comment2context adds cross-module context and avoids same-directory shortcuts",
            all(
                bool(gates.get(name))
                for name in (
                    "new_comment2context_have_given_files",
                    "new_comment2context_no_same_directory_gold",
                    "new_comment2context_cross_module_count_ge_min",
                )
            ),
            evidence_readiness,
            {
                "new_cross_module_samples": readiness.get("comment2context_new_cross_module_samples"),
                "target_gap": target_gaps.get("comment2context_cross_module_samples"),
            },
        ),
        completion_item(
            "Trace2code adds non-Go repositories, language diversity, and real failure types",
            all(
                bool(gates.get(name))
                for name in (
                    "new_trace2code_gold_not_tests",
                    "new_trace2code_has_non_go_gold",
                    "new_trace2code_non_go_repo_count_ge_min",
                    "new_trace2code_language_count_ge_min",
                    "new_trace2code_failure_type_count_ge_min",
                )
            ),
            evidence_readiness,
            {
                "non_go_repos": readiness.get("trace2code_new_non_go_repos") or [],
                "extensions": readiness.get("trace2code_new_gold_extensions") or [],
                "failure_types": readiness.get("trace2code_new_failure_types") or [],
                "unknown_failure_samples": readiness.get("trace2code_new_unknown_failure_samples"),
            },
        ),
        completion_item(
            "Gold files are present in base-commit corpus and assembly requires manual audit evidence",
            all(
                bool(gates.get(name))
                for name in (
                    "corpus_manifest_exists",
                    "new_samples_gold_in_corpus",
                    "new_samples_have_audit_evidence",
                    "assembly_manifest_exists",
                    "assembly_manifest_requires_audit_keep",
                    "assembly_manifest_has_audit_paths",
                )
            ),
            [
                str(readiness_path),
                str((readiness.get("inputs") or {}).get("corpus_manifest"))
                if (readiness.get("inputs") or {}).get("corpus_manifest")
                else None,
            ],
            {
                "accepted_expansion": (readiness.get("assembly_manifest") or {}).get("accepted_expansion"),
                "corpus_manifest": (readiness.get("inputs") or {}).get("corpus_manifest"),
            },
        ),
    ]

    baseline_blockers = {item.get("baseline"): item for item in baseline_status.get("baseline_blockers") or []}
    for baseline in REQUIRED_V1_1_BASELINES:
        readiness_item = next((item for item in readiness.get("required_baselines") or [] if item.get("baseline") == baseline), {})
        blocker = baseline_blockers.get(baseline, {})
        source = readiness_item.get("source")
        baseline_evidence = baseline_completion_evidence_paths(baseline_status_path, str(source) if source else None, blocker)
        checklist.append(
            completion_item(
                f"Run {baseline} all-files baseline with no skipped samples and metric-consistent details",
                bool(readiness_item.get("complete") and blocker.get("complete", readiness_item.get("complete"))),
                baseline_evidence,
                {
                    "baseline": baseline,
                    "found": readiness_item.get("found"),
                    "evaluated": readiness_item.get("evaluated"),
                    "overall_samples": readiness_item.get("overall_samples"),
                    "skipped": readiness_item.get("skipped"),
                    "details": readiness_item.get("details") or {},
                    "partial_details": blocker.get("partial_details") or {},
                    "shard_details": blocker.get("shard_details") or {},
                    "blocker_reason": blocker.get("reason"),
                    "next_action": blocker.get("next_action"),
                },
            )
        )

    leaderboard_missing = (leaderboard.get("missing_required_baselines") or []) or (
        (readiness.get("leaderboard") or {}).get("missing_baselines") or []
    )
    checklist.append(
        completion_item(
            "Regenerate leaderboard and include every required baseline",
            bool(gates.get("leaderboard_reports_contain_required_baselines") and not leaderboard_missing),
            [
                str(leaderboard_json_path),
                str((readiness.get("leaderboard") or {}).get("markdown"))
                if (readiness.get("leaderboard") or {}).get("markdown")
                else None,
            ],
            {
                "missing_baselines": leaderboard_missing,
                "row_count": leaderboard.get("row_count"),
                "next_action": "Finish missing required baseline summaries/details, then rerun report-models with all required baselines."
                if leaderboard_missing
                else None,
            },
        )
    )
    checklist.append(
        completion_item(
            "Regenerate V1.1 release report and mark release ready",
            bool(readiness.get("ready") and release.get("status") == "ready"),
            [str(release_json_path), str(readiness_path)],
            {
                "status": release.get("status"),
                "blocking_gates": release.get("blocking_gates") or readiness.get("blocking_gates") or [],
                "next_action": "Clear readiness blocking gates, then rerun the V1.1 release report."
                if not (readiness.get("ready") and release.get("status") == "ready")
                else None,
            },
        )
    )
    workflow_paths = list(workflow_evidence_paths or [])
    if workflow_paths:
        workflow_status = completion_evidence_status([str(path) for path in workflow_paths])
        workflow_report_checks = workflow_report_completion_status(workflow_paths)
        workflow_pass = all(bool(item.get("exists")) for item in workflow_status) and all(
            bool(item.get("complete")) for item in workflow_report_checks
        )
        workflow_next_action = (
            "Run the external baselines, package returned artifacts, and apply a verified return bundle so the return manifest is artifact-complete."
        )
        checklist.append(
            {
                "name": "External baseline handoff, transfer, and return-bundle workflow evidence",
                "requirement": "External baseline handoff, transfer, and return-bundle workflow evidence",
                "status": "pass" if workflow_pass else "blocked",
                "evidence": [str(path) for path in workflow_paths],
                "evidence_status": workflow_status,
                "evidence_all_present": all(bool(item.get("exists")) for item in workflow_status) if workflow_status else False,
                "details": {
                    "workflow_report_checks": workflow_report_checks,
                    "next_action": workflow_next_action if not workflow_pass else None,
                },
                "next_action": workflow_next_action if not workflow_pass else None,
            }
        )
    existing_docs = [str(path) for path in docs if path.exists()]
    doc_content = completion_doc_content_status(docs)
    docs_files_complete = len(existing_docs) == len(docs)
    docs_content_complete = bool(doc_content.get("complete"))
    docs_pass = docs_files_complete and docs_content_complete and release.get("status") == "ready"
    docs_status = "pass" if docs_pass else ("partial" if docs_files_complete and docs_content_complete else "blocked")
    docs_next_action = None
    if not docs_content_complete:
        docs_next_action = "Update V1.1 docs to cover all required release-positioning markers."
    elif release.get("status") != "ready":
        docs_next_action = "Finalize public V1.1 docs after the release report is ready."
    docs_evidence_status = completion_evidence_status([str(path) for path in docs])
    checklist.append(
        {
            "name": "Update docs to present V1.1 as focused benchmark improvement over V1",
            "requirement": "Update docs to present V1.1 as focused benchmark improvement over V1",
            "status": docs_status,
            "evidence": existing_docs,
            "evidence_status": docs_evidence_status,
            "evidence_all_present": all(bool(item.get("exists")) for item in docs_evidence_status) if docs_evidence_status else False,
            "details": {
                "docs_expected": [str(path) for path in docs],
                "docs_found": existing_docs,
                "content_markers_complete": docs_content_complete,
                "content_markers": doc_content.get("markers") or [],
                "missing_content_markers": doc_content.get("missing_markers") or [],
                "final_public_release_docs_blocked_until_release_ready": release.get("status") != "ready",
                "next_action": docs_next_action,
            },
            "next_action": docs_next_action,
        }
    )
    return checklist


def baseline_completion_evidence_paths(
    baseline_status_path: Path,
    source: str | None,
    blocker: dict[str, Any],
) -> list[str]:
    paths: list[str] = [str(baseline_status_path)]
    if source:
        paths.append(source)
    partial_path = (blocker.get("partial_details") or {}).get("path")
    if partial_path:
        paths.append(str(partial_path))
        summary_path = final_summary_path_from_details_path(str(partial_path))
        if summary_path:
            paths.append(summary_path)
    shard_paths = (blocker.get("shard_details") or {}).get("paths") or []
    paths.extend(str(path) for path in shard_paths if path)
    return sorted(dict.fromkeys(paths))


def final_summary_path_from_details_path(details_path: str) -> str | None:
    suffix = "_details.jsonl"
    if details_path.endswith(suffix):
        return f"{details_path[:-len(suffix)]}_summary.json"
    return None


def completion_doc_content_status(docs: Iterable[Path]) -> dict[str, Any]:
    texts: list[str] = []
    existing_docs: list[str] = []
    missing_docs: list[str] = []
    for doc in docs:
        if not doc.exists():
            missing_docs.append(str(doc))
            continue
        existing_docs.append(str(doc))
        try:
            texts.append(doc.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            missing_docs.append(str(doc))
            texts.append(f"read_error:{error}")
    combined = "\n".join(texts).casefold()
    markers: list[dict[str, Any]] = []
    for marker in V1_1_DOC_CONTENT_MARKERS:
        required = tuple(str(item).casefold() for item in marker.get("all_of", ()))
        alternatives = tuple(str(item).casefold() for item in marker.get("any_of", ()))
        missing_required = [phrase for phrase in required if phrase not in combined]
        any_match = any(phrase in combined for phrase in alternatives) if alternatives else True
        passed = not missing_required and any_match
        markers.append(
            {
                "name": marker["name"],
                "description": marker["description"],
                "pass": passed,
                "missing_required_phrases": missing_required,
                "matched_any_of": any_match,
                "any_of": list(alternatives),
            }
        )
    missing_markers = [marker["name"] for marker in markers if not marker.get("pass")]
    return {
        "complete": not missing_docs and not missing_markers,
        "docs_found": existing_docs,
        "docs_missing": missing_docs,
        "markers": markers,
        "missing_markers": missing_markers,
    }


def completion_doc_cli_args(docs: Iterable[Path]) -> list[Any]:
    args: list[Any] = []
    for doc in docs:
        args.extend(["--doc", doc])
    return args


def completion_item(requirement: str, passed: bool, evidence: Iterable[str | None], details: dict[str, Any]) -> dict[str, Any]:
    evidence_paths = [item for item in evidence if item]
    evidence_status = completion_evidence_status(evidence_paths)
    evidence_all_present = all(bool(item.get("exists")) for item in evidence_status) if evidence_status else False
    next_action = details.get("next_action")
    return {
        "name": requirement,
        "requirement": requirement,
        "status": "pass" if passed and evidence_all_present else "blocked",
        "evidence": evidence_paths,
        "evidence_status": evidence_status,
        "evidence_all_present": evidence_all_present,
        "details": details,
        "next_action": str(next_action) if next_action else None,
    }


def completion_evidence_status(evidence: Iterable[str]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for item in evidence:
        path = Path(item)
        exists = path.exists()
        status: dict[str, Any] = {
            "path": item,
            "exists": exists,
            "kind": "missing",
        }
        if exists:
            if path.is_dir():
                status["kind"] = "directory"
            elif path.is_file():
                status["kind"] = "file"
                status["bytes"] = path.stat().st_size
            else:
                status["kind"] = "other"
        statuses.append(status)
    return statuses


def workflow_report_completion_status(paths: Iterable[Path]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "complete": path.exists(),
            "checked_fields": [],
        }
        if not path.exists():
            item["reason"] = "missing"
            checks.append(item)
            continue
        if path.is_file() and path.suffix == ".json":
            try:
                payload = read_json(path, {})
            except json.JSONDecodeError as error:
                item.update({"complete": False, "error": f"invalid_json: {error}"})
                checks.append(item)
                continue
            if isinstance(payload, dict):
                for metadata_key in ("generated_at", "handoff", "manifest", "bundle", "checksum"):
                    if payload.get(metadata_key) is not None:
                        item[metadata_key] = payload.get(metadata_key)
                if payload.get("sha256") is not None:
                    item["sha256"] = payload.get("sha256")
                if payload.get("size_bytes") is not None:
                    item["size_bytes"] = payload.get("size_bytes")
                bundle_fingerprint = payload.get("bundle_fingerprint")
                if isinstance(bundle_fingerprint, dict):
                    if bundle_fingerprint.get("sha256") is not None:
                        item["bundle_sha256"] = bundle_fingerprint.get("sha256")
                    if bundle_fingerprint.get("size_bytes") is not None:
                        item["bundle_size_bytes"] = bundle_fingerprint.get("size_bytes")
                if isinstance(payload.get("failed_checks"), list):
                    item["failed_checks"] = payload.get("failed_checks") or []
                if isinstance(payload.get("missing_required_files"), list):
                    item["missing_required_file_count"] = len(payload.get("missing_required_files") or [])
                field_results: list[dict[str, Any]] = []
                if "complete" in payload:
                    field_results.append({"field": "complete", "value": payload.get("complete"), "pass": bool(payload.get("complete"))})
                if "artifacts_complete" in payload:
                    field_results.append(
                        {
                            "field": "artifacts_complete",
                            "value": payload.get("artifacts_complete"),
                            "pass": bool(payload.get("artifacts_complete")),
                        }
                    )
                if "overall_status" in payload:
                    field_results.append(
                        {
                            "field": "overall_status",
                            "value": payload.get("overall_status"),
                            "pass": payload.get("overall_status") == "complete",
                        }
                    )
                if field_results:
                    item["checked_fields"] = field_results
                    item["complete"] = all(bool(result.get("pass")) for result in field_results)
                elif item.get("failed_checks"):
                    item["complete"] = False
        checks.append(item)
    return checks


def completion_next_action(blocked_or_partial: list[dict[str, Any]]) -> str | None:
    missing_required_baselines = [
        baseline
        for baseline in REQUIRED_V1_1_BASELINES
        if any(
            str(item.get("requirement") or "").startswith(f"Run {baseline} all-files baseline")
            for item in blocked_or_partial
        )
    ]
    missing_embedding_baselines = [
        baseline
        for baseline in missing_required_baselines
        if baseline not in {"lexical", "aider-style-repomap"}
    ]
    if missing_embedding_baselines:
        return (
            "Run missing embedding baselines "
            f"({', '.join(missing_embedding_baselines)}) on CUDA/API-capable machines, "
            "package and apply a verified return bundle, then rerun v1-1-finalize-baselines."
        )
    for item in blocked_or_partial:
        details = item.get("details") or {}
        if details.get("next_action"):
            return str(details["next_action"])
    if blocked_or_partial:
        return str(blocked_or_partial[0].get("requirement"))
    return None


def render_v1_1_completion_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.1 Completion Audit",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        "",
        f"Overall status: `{report.get('overall_status')}`",
        "",
        "## Objective",
        "",
        str(report.get("objective") or ""),
        "",
        "## Summary",
        "",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False)}`")
    lines.extend(["", "## Checklist", ""])
    for item in report.get("checklist") or []:
        lines.append(f"### {str(item.get('status')).upper()} - {item.get('requirement')}")
        lines.append("")
        lines.append(f"- Evidence: `{json.dumps(item.get('evidence') or [], ensure_ascii=False)}`")
        lines.append("")
        lines.append(f"- Evidence status: `{json.dumps(item.get('evidence_status') or [], ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
        if item.get("next_action"):
            lines.append(f"- Next action: {item.get('next_action')}")
            lines.append("")
        lines.append(f"- Details: `{json.dumps(item.get('details') or {}, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if report.get("overall_status") == "complete":
        lines.append("V1.1 is complete according to the current prompt-to-artifact checklist.")
    else:
        lines.append("V1.1 is not complete. Blocked or partial requirements remain in the checklist above.")
    lines.append("")
    return "\n".join(lines)


def empty_baseline_details_status() -> dict[str, Any]:
    return {
        "path": None,
        "exists": False,
        "rows": 0,
        "unique_sample_ids": 0,
        "duplicate_sample_ids": [],
        "missing_sample_id_rows": 0,
        "missing_sample_ids": [],
        "unexpected_sample_ids": [],
        "metrics_match": False,
        "metrics_mismatches": [],
        "missing_metric_tasks": [],
        "extra_metric_tasks": [],
        "malformed_metric_rows": [],
        "candidate_filters": [],
        "complete": False,
    }


def baseline_details_status(
    summary_path: Path,
    expected_samples: int,
    expected_sample_ids: set[str] | None = None,
    summary_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details_path = details_path_for_summary(summary_path)
    rows = read_jsonl(details_path) if details_path.exists() else []
    sample_ids = [str(row.get("sample_id")) for row in rows if row.get("sample_id")]
    actual_sample_ids = set(sample_ids)
    duplicate_sample_ids = sorted({sample_id for sample_id, count in Counter(sample_ids).items() if count > 1})
    candidate_filters = sorted({str(row.get("candidate_filter") or "") for row in rows if row.get("candidate_filter")})
    missing_sample_ids = sorted((expected_sample_ids or set()) - actual_sample_ids) if expected_sample_ids is not None else []
    unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids) if expected_sample_ids is not None else []
    metrics_status = details_metrics_status(rows, summary_metrics or {})
    complete = (
        details_path.exists()
        and len(rows) == expected_samples
        and len(set(sample_ids)) == expected_samples
        and not duplicate_sample_ids
        and len(sample_ids) == len(rows)
        and candidate_filters == ["all_files"]
        and not missing_sample_ids
        and not unexpected_sample_ids
        and metrics_status["metrics_match"]
    )
    return {
        "path": str(details_path),
        "exists": details_path.exists(),
        "rows": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": duplicate_sample_ids,
        "missing_sample_id_rows": len(rows) - len(sample_ids),
        "missing_sample_ids": missing_sample_ids,
        "unexpected_sample_ids": unexpected_sample_ids,
        **metrics_status,
        "candidate_filters": candidate_filters,
        "complete": complete,
    }


def details_metrics_status(rows: list[dict[str, Any]], summary_metrics: dict[str, Any]) -> dict[str, Any]:
    malformed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        missing_metric_keys = []
        if not isinstance(metrics, dict):
            missing_metric_keys = list(BASELINE_DETAIL_METRICS)
        else:
            missing_metric_keys = [key for key in BASELINE_DETAIL_METRICS if key not in metrics]
        if not row.get("task_type") or missing_metric_keys:
            malformed_rows.append(
                {
                    "row": index,
                    "sample_id": row.get("sample_id"),
                    "missing_task_type": not bool(row.get("task_type")),
                    "missing_metric_keys": missing_metric_keys,
                }
            )
    if malformed_rows:
        return {
            "metrics_match": False,
            "metrics_mismatches": [],
            "missing_metric_tasks": [],
            "extra_metric_tasks": [],
            "malformed_metric_rows": malformed_rows,
        }

    recomputed = summarize_details(rows)
    missing_tasks = sorted(set(recomputed) - set(summary_metrics))
    extra_tasks = sorted(set(summary_metrics) - set(recomputed))
    mismatches: list[dict[str, Any]] = []
    for task in sorted(set(recomputed) & set(summary_metrics)):
        actual_task_metrics = summary_metrics.get(task) or {}
        expected_task_metrics = recomputed[task]
        for key in ("samples", *BASELINE_DETAIL_METRICS):
            actual = actual_task_metrics.get(key)
            expected = expected_task_metrics.get(key)
            if key == "samples":
                if int(actual or 0) != int(expected or 0):
                    mismatches.append({"task": task, "metric": key, "summary": actual, "details": expected})
            else:
                try:
                    matches = abs(float(actual) - float(expected)) <= 1e-9
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    mismatches.append({"task": task, "metric": key, "summary": actual, "details": expected})
    return {
        "metrics_match": not missing_tasks and not extra_tasks and not mismatches,
        "metrics_mismatches": mismatches,
        "missing_metric_tasks": missing_tasks,
        "extra_metric_tasks": extra_tasks,
        "malformed_metric_rows": [],
    }


def details_path_for_summary(summary_path: Path) -> Path:
    stem = summary_path.stem
    if stem.endswith("_summary"):
        return summary_path.with_name(f"{stem.removesuffix('_summary')}_details.jsonl")
    return summary_path.with_suffix(".details.jsonl")


def leaderboard_readiness(
    leaderboard_path: Path | None,
    leaderboard_json_path: Path | None,
    expected_baselines: Iterable[str],
) -> dict[str, Any]:
    markdown_exists = bool(leaderboard_path and leaderboard_path.exists())
    json_exists = bool(leaderboard_json_path and leaderboard_json_path.exists())
    labels: set[str] = set()
    if json_exists and leaderboard_json_path:
        report = read_json(leaderboard_json_path, {})
        for row in report.get("rows") or []:
            if isinstance(row, dict):
                labels.add(str(row.get("model_label", "")))
    if markdown_exists and leaderboard_path:
        text = leaderboard_path.read_text(encoding="utf-8").lower()
        for expected in expected_baselines:
            if expected.lower() in text:
                labels.add(expected)
    missing = [expected for expected in expected_baselines if not any(baseline_label_matches(label, expected) for label in labels)]
    requested_paths = [path for path in (leaderboard_path, leaderboard_json_path) if path is not None]
    return {
        "markdown": str(leaderboard_path) if leaderboard_path else None,
        "json": str(leaderboard_json_path) if leaderboard_json_path else None,
        "markdown_exists": markdown_exists,
        "json_exists": json_exists,
        "reports_exist": all(path.exists() for path in requested_paths) if requested_paths else True,
        "contains_required_baselines": not missing,
        "missing_baselines": missing,
    }


def baseline_label_matches(label: str, expected: str) -> bool:
    normalized_label = label.lower()
    normalized_expected = expected.lower()
    return normalized_label == normalized_expected or normalized_expected in normalized_label


def load_audit_keep_ids(audit_paths: Iterable[Path]) -> set[str]:
    keep_ids: set[str] = set()
    for path in audit_paths:
        if not path.exists():
            continue
        for row in read_audit_rows(path):
            if should_keep(row) and row.get("sample_id"):
                keep_ids.add(str(row["sample_id"]))
    return keep_ids


def report_v1_1_release(
    readiness_path: Path,
    leaderboard_json_path: Path,
    out_path: Path,
    json_out_path: Path,
    base_leaderboard_json_path: Path | None = None,
) -> dict[str, Any]:
    readiness = read_json(readiness_path, {}) or {}
    readiness_summary = readiness.get("summary") if isinstance(readiness.get("summary"), dict) else readiness
    leaderboard = read_json(leaderboard_json_path, {}) or {}
    base_leaderboard = read_json(base_leaderboard_json_path, {}) if base_leaderboard_json_path else {}
    best_rows = best_leaderboard_rows(leaderboard.get("rows") or [])
    base_best_rows = best_leaderboard_rows(base_leaderboard.get("rows") or [])
    gates = readiness_summary.get("gates") or {}
    blocking_gates = [name for name, passed in gates.items() if not passed]
    report = {
        "generated_at": utc_now(),
        "status": "ready" if readiness_summary.get("ready") else "not_ready",
        "inputs": {
            "readiness": str(readiness_path),
            "leaderboard_json": str(leaderboard_json_path),
            "base_leaderboard_json": str(base_leaderboard_json_path) if base_leaderboard_json_path else None,
        },
        "counts_by_task": readiness_summary.get("counts_by_task") or {},
        "base_counts_by_task": readiness_summary.get("base_counts_by_task") or {},
        "new_counts_by_task": readiness_summary.get("new_counts_by_task") or {},
        "comment_context": {
            "new_cross_module_samples": int(readiness_summary.get("comment2context_new_cross_module_samples") or 0),
        },
        "trace_diversity": {
            "new_gold_extensions": readiness_summary.get("trace2code_new_gold_extensions") or [],
            "new_non_go_repos": readiness_summary.get("trace2code_new_non_go_repos") or [],
            "new_failure_types": readiness_summary.get("trace2code_new_failure_types") or [],
            "new_unknown_failure_samples": int(readiness_summary.get("trace2code_new_unknown_failure_samples") or 0),
        },
        "target_gaps": readiness_summary.get("target_gaps") or {},
        "required_baselines": readiness_summary.get("required_baselines") or [],
        "blocking_gates": blocking_gates,
        "leaderboard_best": best_rows,
        "base_leaderboard_best": base_best_rows,
    }
    write_json(json_out_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_v1_1_release_report(report), encoding="utf-8")
    return {
        "report": str(out_path),
        "json": str(json_out_path),
        "status": report["status"],
        "blocking_gates": blocking_gates,
    }


def best_leaderboard_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("candidate_filter") or "all_files") != "all_files":
            continue
        task = str(row.get("task", ""))
        if task:
            grouped[task].append(row)
    best: dict[str, dict[str, Any]] = {}
    for task, task_rows in grouped.items():
        by_mrr = max(task_rows, key=lambda row: (float(row.get("MRR") or 0.0), float(row.get("Recall@20") or 0.0), str(row.get("model_label", ""))))
        by_recall20 = max(task_rows, key=lambda row: (float(row.get("Recall@20") or 0.0), float(row.get("MRR") or 0.0), str(row.get("model_label", ""))))
        best[task] = {
            "best_mrr_model": by_mrr.get("model_label") or by_mrr.get("model"),
            "best_mrr": float(by_mrr.get("MRR") or 0.0),
            "best_recall20_model": by_recall20.get("model_label") or by_recall20.get("model"),
            "best_recall20": float(by_recall20.get("Recall@20") or 0.0),
            "samples": int(by_mrr.get("samples") or 0),
        }
    return dict(sorted(best.items()))


def render_v1_1_release_report(report: dict[str, Any]) -> str:
    lines = [
        "# Agent Retrieval Bench V1.1 Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        "",
        "## Focus",
        "",
        "V1.1 is a targeted expansion of frozen V1. It keeps `code2test` stable and expands the weaker, more diagnostic `comment2context` and `trace2code` tracks.",
        "",
        "## Samples",
        "",
        "| Task | V1 | V1.1 | New |",
        "| --- | ---: | ---: | ---: |",
    ]
    tasks = sorted(set(report["base_counts_by_task"]) | set(report["counts_by_task"]) | set(report["new_counts_by_task"]))
    for task in tasks:
        lines.append(
            f"| `{task}` | {report['base_counts_by_task'].get(task, 0)} | "
            f"{report['counts_by_task'].get(task, 0)} | {report['new_counts_by_task'].get(task, 0)} |"
        )
    comment_context = report["comment_context"]
    trace = report["trace_diversity"]
    lines.extend(
        [
            "",
            "## Target Gaps",
            "",
            "| Gap | Remaining |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(report.get("target_gaps", {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Review Context",
            "",
            f"- New cross-module `comment2context` samples: `{comment_context['new_cross_module_samples']}`",
            "",
            "## Trace Diversity",
            "",
            f"- New gold extensions: `{', '.join(trace['new_gold_extensions']) or 'none'}`",
            f"- New non-Go repos: `{', '.join(trace['new_non_go_repos']) or 'none'}`",
            f"- New failure types: `{', '.join(trace['new_failure_types']) or 'none'}`",
            f"- New unknown failure-type samples: `{trace['new_unknown_failure_samples']}`",
            "",
            "## Leaderboard Highlights",
            "",
            "| Task | Best MRR Model | MRR | Best Recall@20 Model | Recall@20 |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for task, row in sorted(report["leaderboard_best"].items()):
        lines.append(
            f"| `{task}` | {row['best_mrr_model']} | {row['best_mrr']:.4f} | "
            f"{row['best_recall20_model']} | {row['best_recall20']:.4f} |"
        )
    lines.extend(["", "## Baselines", ""])
    for item in report["required_baselines"]:
        status = "complete" if item.get("complete") else "missing/incomplete"
        lines.append(
            f"- `{item.get('baseline')}`: {status}, evaluated=`{item.get('evaluated', 0)}`, skipped=`{json.dumps(item.get('skipped', {}), sort_keys=True)}`"
        )
    lines.extend(["", "## Release Gates", ""])
    if report["blocking_gates"]:
        lines.append(f"- Blocking gates: `{', '.join(report['blocking_gates'])}`")
    else:
        lines.append("- Blocking gates: `none`")
    return "\n".join(lines) + "\n"


def task_semantics_issues(sample: dict[str, Any]) -> list[str]:
    task_type = str(sample.get("task_type", ""))
    query = sample.get("query") or {}
    issues: list[str] = []
    if task_type == "comment2context":
        if not has_meaningful_query_value(query, COMMENT_SIGNAL_KEYS):
            issues.append("missing_review_comment")
        if not (given_files(sample) or has_meaningful_query_value(query, {"given_file", "path", "file"})):
            issues.append("missing_reviewed_file_context")
    elif task_type == "trace2code":
        if not has_meaningful_query_value(query, TRACE_SIGNAL_KEYS):
            issues.append("missing_failure_signal")
    return issues


def trace_signal_text(sample: dict[str, Any]) -> str:
    values = query_values_for_keys(sample.get("query") or {}, TRACE_SIGNAL_KEYS)
    if not values:
        return query_text_for_eval(sample)
    return "\n".join(text_from_query_value(value) for value in values if text_from_query_value(value).strip())


def query_values_for_keys(value: Any, keys: set[str]) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                matches.append(item)
            if isinstance(item, (dict, list)):
                matches.extend(query_values_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            matches.extend(query_values_for_keys(item, keys))
    return matches


def text_from_query_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_from_query_value(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(text_from_query_value(item) for item in value.values())
    return str(value)


def has_meaningful_query_value(value: Any, keys: set[str]) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys and meaningful_text(item):
                return True
            if isinstance(item, (dict, list)) and has_meaningful_query_value(item, keys):
                return True
        return False
    if isinstance(value, list):
        return any(has_meaningful_query_value(item, keys) for item in value)
    return False


def meaningful_text(value: Any) -> bool:
    if isinstance(value, str):
        return len(value.strip()) >= 10
    if isinstance(value, list):
        return any(meaningful_text(item) for item in value)
    if isinstance(value, dict):
        return any(meaningful_text(item) for item in value.values())
    return False


def sample_has_audit_evidence(sample: dict[str, Any]) -> bool:
    metadata = sample.get("metadata") or {}
    for key in ("evidence", "audit", "audit_evidence", "manual_audit"):
        if has_evidence_value(metadata.get(key)):
            return True
    gold = sample.get("gold") or {}
    context_files = gold.get("must_context_files") or gold.get("context_files") or []
    for value in context_files:
        if isinstance(value, dict) and value.get("path") and has_evidence_value(value.get("evidence")):
            return True
    return False


def sample_path_role_overlap_issues(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    target_paths = set(target_gold_files(sample))
    supporting_paths = set(path_values(gold.get("supporting_files") or []))
    distractor_paths = set(path_values(gold.get("negative_distractors") or []))
    issues: list[str] = []
    if target_paths & supporting_paths:
        issues.append("gold_supporting_overlap")
    if target_paths & distractor_paths:
        issues.append("gold_negative_distractor_overlap")
    if supporting_paths & distractor_paths:
        issues.append("support_negative_distractor_overlap")
    return issues


def path_values(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return sorted(set(paths))


def has_evidence_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return any(has_evidence_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(has_evidence_value(item) for item in value)
    return True


def trace_failure_types(query_text: str) -> list[str]:
    lowered = query_text.lower()
    signals: set[str] = set()
    if "traceback" in lowered or ".py\", line" in lowered or ".py', line" in lowered:
        signals.add("traceback")
    if "panic" in lowered:
        signals.add("panic")
    if "assert" in lowered or "expected" in lowered or "got" in lowered:
        signals.add("assertion")
    if "exception" in lowered:
        signals.add("exception")
    if "timeout" in lowered or "timed out" in lowered:
        signals.add("timeout")
    if (
        "compile" in lowered
        or "compilation" in lowered
        or "syntaxerror" in lowered
        or "cannot find symbol" in lowered
        or "undefined:" in lowered
        or "error[" in lowered
    ):
        signals.add("compile_error")
    if "--- fail" in lowered or "failed" in lowered or "failure" in lowered:
        signals.add("test_failure")
    if not signals:
        signals.add("unknown")
    return sorted(signals)


def path_extension(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix or "<none>"


def cross_module_gold_files(gold_files: Iterable[str], reference_files: Iterable[str]) -> list[str]:
    reference_modules = {module_key(path) for path in reference_files if module_key(path)}
    if not reference_modules:
        return []
    return [path for path in gold_files if module_key(path) and module_key(path) not in reference_modules]


def module_key(path: str) -> str:
    parts = [part for part in PurePosixPath(path).parts if part not in {"", "."}]
    if not parts:
        return ""
    if len(parts) == 1:
        return "<root>"
    first = parts[0]
    if first in MODULE_ANCHOR_DIRS:
        return "/".join(parts[:2])
    return first


def is_benchmark_v1_path(path: Path) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "benchmark" and parts[index + 1] == "v1":
            return True
    return False


def load_corpus_manifest_paths(corpus_manifest_path: Path) -> dict[tuple[str, str], Path]:
    if not corpus_manifest_path.exists():
        return {}
    records = read_jsonl(corpus_manifest_path)
    return {
        (record.get("repo"), record.get("base_commit")): Path(str(record.get("chunks_path", "")))
        for record in records
        if record.get("status") == "ok"
    }


def load_samples(paths: Iterable[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        samples.extend(read_jsonl(path))
    return samples


def source_sample_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        return []
    curated = source / "samples.jsonl"
    if curated.exists():
        return [curated]
    return [source / f"{task}.jsonl" for task in V1_1_EXPANSION_TASKS if (source / f"{task}.jsonl").exists()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_audit_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V1_1_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in V1_1_AUDIT_FIELDS})
            count += 1
    return count


def render_readiness_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# V1.1 Readiness Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Ready: `{summary['ready']}`.",
        f"- Samples: `{summary['samples']}`.",
        f"- New samples: `{summary['new_samples']}`.",
        "",
        "## Counts",
        "",
        "| Task | Base | V1.1 | New |",
        "| --- | ---: | ---: | ---: |",
    ]
    tasks = sorted(set(summary["base_counts_by_task"]) | set(summary["counts_by_task"]) | set(summary["new_counts_by_task"]))
    for task in tasks:
        lines.append(
            f"| `{task}` | {summary['base_counts_by_task'].get(task, 0)} | "
            f"{summary['counts_by_task'].get(task, 0)} | {summary['new_counts_by_task'].get(task, 0)} |"
        )
    if summary.get("target_gaps"):
        lines.extend(["", "## Target Gaps", "", "| Gap | Remaining |", "| --- | ---: |"])
        for key, value in sorted(summary["target_gaps"].items()):
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Gates", "", "| Gate | Status |", "| --- | --- |"])
    for gate, passed in summary["gates"].items():
        lines.append(f"| `{gate}` | {'pass' if passed else 'FAIL'} |")
    if summary["issues_by_type"]:
        lines.extend(["", "## Issues", "", "| Issue | Counts by task |", "| --- | --- |"])
        for issue, counts in summary["issues_by_type"].items():
            lines.append(f"| `{issue}` | `{json.dumps(counts, sort_keys=True)}` |")
    failing_rows = [row for row in rows if row["is_new"] and row["issues"]]
    if failing_rows:
        lines.extend(["", "## New Sample Failures", "", "| Sample | Task | Issues |", "| --- | --- | --- |"])
        for row in failing_rows[:50]:
            lines.append(f"| `{row['sample_id']}` | `{row['task_type']}` | `{', '.join(row['issues'])}` |")
    lines.append("")
    return "\n".join(lines)
