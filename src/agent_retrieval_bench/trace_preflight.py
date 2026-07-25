from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .audit import audit_row
from .derive import _extract_test_names, _failure_excerpt, _read_log_excerpt
from .filters import (
    contains_review_leakage,
    extract_repo_trace_paths,
    has_failure_or_trace_signal,
    is_ignored_check_signal,
    is_job_name_only_signal,
    is_runner_setup_noise,
    is_source_file,
    is_test_file,
    split_changed_files,
)
from .io import ensure_parent, read_jsonl, repo_slug, stable_id, truncate_text, utc_now, write_json
from .quality import validate_sample

TRACE_AUDIT_VERDICTS = ("valid", "noisy", "leaked", "ambiguous", "too_easy", "duplicate", "not_root_cause")
TRACE_DEBUG_VERDICTS = ("valid_root_cause", "test_only_but_mappable", "infra_noise", "third_party_only", "too_broad", "leaked", "ambiguous")
TRACE_DEBUG_RECOVERABLE_VERDICTS = {"valid_root_cause", "test_only_but_mappable"}
TRACE_SOURCE_VERDICTS = ("usable_trace_source", "infra_noise", "third_party_only", "too_broad", "ambiguous")
TRACE_SOURCE_PRIORITY = {"job_log": 0, "check_run": 1, "review_comment": 2}
STRONG_FAILURE_RE = re.compile(
    r"("
    r"\bTraceback\b|\bAssertionError\b|\bRuntimeError\b|\bException\b|\bpanic\b|"
    r"\bstack backtrace\b|\bCaused by\b|\bError Trace:\b|\bFAIL(?:ED)?\b|"
    r"\bfailed\b|##\[error\]|\berror:|Assertion failed"
    r")",
    re.IGNORECASE,
)
REAL_FAILURE_CONTEXT_RE = re.compile(
    r"("
    r"\bTraceback\b|\bAssertionError\b|\bRuntimeError\b|\bException\b|\bpanic\b|"
    r"\bstack backtrace\b|\bCaused by\b|\bError Trace:\b|"
    r"\bFAILED?\s+[^\s]+::[A-Za-z0-9_.$:-]+|"
    r"\bFAIL\s+[^\s]+\.(?:py|js|jsx|ts|tsx|java|go|rs|kt|scala)\b|"
    r"##\[error\]|\berror:\s+[A-Za-z]"
    r")",
    re.IGNORECASE,
)
POST_HOC_REVIEW_RE = re.compile(
    r"("
    r"review_comment_addressed|resolved in|fixed in|thanks for (?:the )?(?:fix|update)|"
    r"confirmed|looks good|lgtm|duly noted|moving on|addressed by|"
    r"this is now fixed|the fix looks correct"
    r")",
    re.IGNORECASE,
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GITHUB_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*", re.MULTILINE)
WORKSPACE_PREFIX_RE = re.compile(r"^/?(?:home/runner/work|Users/runner/work|__w|runner/work)/([^/]+)/\1/", re.IGNORECASE)
LOW_VALUE_TRACE_PARTS = (
    "site-packages/",
    ".venv/",
    "node_modules/",
    "/usr/local/go/",
    "go/pkg/mod/",
    ".cargo/",
    "rustc/",
    "<frozen ",
)
INFRA_FAILURE_RE = re.compile(
    r"("
    r"actions/cache|fail-on-cache-miss|cache hit for:|setup-python|setup-pnpm|"
    r"pip install uv|uv pip install|prepared \d+ packages|building [^\\n]+ @ file://|"
    r"download(?:ing|ed) (?:ruff|black|grpcio|tensorboard|timm)|"
    r"version \d+(?:\.\d+)? was not found in the local cache|"
    r"goreleaser|go: downloading|git log -1 --format|git config --local|"
    r"No module named ['\"]hypothesis['\"]"
    r")",
    re.IGNORECASE,
)
DOWNSTREAM_CHECK_RE = re.compile(r"\b(?:Dify|Polar|Semantic Kernel)\b", re.IGNORECASE)


def trace_preflight(
    raw_dir: Path,
    out_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
) -> dict[str, Any]:
    candidates, scanned, dropped = collect_trace_candidates(
        raw_dir=raw_dir,
        repos=repos,
        max_changed_files=max_changed_files,
        include_review_comments=True,
    )
    candidates = dedupe_candidates(candidates)
    candidates.sort(key=lambda row: (row.get("repo", ""), row.get("pr_number", 0), row.get("source", ""), row.get("source_id", "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    summary = trace_summary(
        raw_dir=raw_dir,
        out_dir=out_dir,
        candidates=candidates,
        scanned=scanned,
        dropped=dropped,
        max_changed_files=max_changed_files,
        outputs={"candidates": str(out_dir / "candidates.jsonl"), "summary": str(out_dir / "summary.json")},
    )
    write_json(out_dir / "summary.json", summary)
    return summary


def mine_trace2code(
    raw_dir: Path,
    out_dir: Path,
    report_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    audit_limit: int = 120,
    include_review_comments: bool = True,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    candidates, scanned, dropped = collect_trace_candidates(
        raw_dir=raw_dir,
        repos=repos,
        max_changed_files=max_changed_files,
        include_review_comments=include_review_comments,
    )
    candidates = dedupe_candidates(candidates)
    candidates.sort(key=trace_candidate_sort_key)

    samples: list[dict[str, Any]] = []
    invalid_by_id: dict[str, list[str]] = {}
    for candidate in candidates:
        sample = trace_candidate_to_sample(candidate)
        errors = validate_sample(sample)
        if errors:
            invalid_by_id[str(sample.get("id", ""))] = errors
            dropped["invalid_sample"] += 1
            continue
        samples.append(sample)
        if limit_samples and len(samples) >= limit_samples:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "trace2code.jsonl", samples)
    write_jsonl(out_dir / "code2test.jsonl", [])
    write_jsonl(out_dir / "comment2context.jsonl", [])

    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "candidates.jsonl", candidates)
    audit_rows = [trace_audit_row(sample) for sample in samples[: max(0, audit_limit)]]
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(
        report_dir / "audit_samples.csv",
        audit_rows,
        ("sample_id", "task_type", "repo", "query_excerpt", "gold_files", "verdict", "reason", "keep", "notes"),
    )

    summary = trace_summary(
        raw_dir=raw_dir,
        out_dir=report_dir,
        candidates=candidates,
        scanned=scanned,
        dropped=dropped,
        max_changed_files=max_changed_files,
        outputs={
            "benchmark_samples": str(out_dir / "samples.jsonl"),
            "benchmark_trace2code": str(out_dir / "trace2code.jsonl"),
            "candidates": str(report_dir / "candidates.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
            "preflight_summary": str(report_dir / "preflight_summary.json"),
        },
    )
    summary.update(
        {
            "benchmark_out_dir": str(out_dir),
            "report_dir": str(report_dir),
            "include_review_comments": include_review_comments,
            "samples": len(samples),
            "counts_by_task": {"trace2code": len(samples)} if samples else {},
            "unique_pairs": len({(sample["repo"], sample["base_commit"]) for sample in samples}),
            "audit_rows": len(audit_rows),
            "valid_audit_verdicts": list(TRACE_AUDIT_VERDICTS),
            "invalid_samples": invalid_by_id,
            "quality_gate": {
                **summary["quality_gate"],
                "audit_batch_ge_80": len(audit_rows) >= 80,
                "ready_for_trace_audit": len(audit_rows) >= 80,
                "full_v1_export_ready": False,
            },
        }
    )
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "preflight_summary.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def trace_debug_drops(
    raw_dir: Path,
    out_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    audit_limit: int = 120,
) -> dict[str, Any]:
    rows, scanned, dropped = collect_trace_debug_drops(raw_dir=raw_dir, repos=repos, max_changed_files=max_changed_files)
    rows = dedupe_debug_rows(rows)
    rows.sort(key=trace_debug_sort_key)
    audit_rows = [trace_debug_audit_row(row) for row in rows[: max(0, audit_limit)]]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "weak_signals.jsonl", rows)
    write_jsonl(out_dir / "audit_samples.jsonl", audit_rows)
    write_csv(
        out_dir / "audit_samples.csv",
        audit_rows,
        (
            "sample_id",
            "repo",
            "pr_number",
            "source",
            "check_name",
            "drop_reason",
            "failure_excerpt",
            "trace_paths",
            "implementation_files",
            "test_files",
            "candidate_root_files",
            "verdict",
            "reason",
            "keep",
            "notes",
        ),
    )
    summary = {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "max_changed_files": max_changed_files,
        "scanned": dict(sorted(scanned.items())),
        "weak_signals": len(rows),
        "audit_rows": len(audit_rows),
        "by_source": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "by_drop_reason": dict(sorted(Counter(str(row.get("drop_reason", "")) for row in rows).items())),
        "dropped": dict(sorted(dropped.items())),
        "quality_gate": {"audit_rows_ge_120": len(audit_rows) >= 120},
        "outputs": {
            "weak_signals": str(out_dir / "weak_signals.jsonl"),
            "audit_jsonl": str(out_dir / "audit_samples.jsonl"),
            "audit_csv": str(out_dir / "audit_samples.csv"),
            "summary": str(out_dir / "summary.json"),
        },
        "valid_audit_verdicts": list(TRACE_DEBUG_VERDICTS),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def trace_debug_summary(audit_path: Path, out_path: Path, recoverable_out: Path) -> dict[str, Any]:
    rows = read_trace_debug_audit(audit_path)
    pending = [row for row in rows if not str(row.get("verdict") or "").strip()]
    invalid_verdicts = Counter(
        str(row.get("verdict") or "").strip()
        for row in rows
        if str(row.get("verdict") or "").strip() and str(row.get("verdict") or "").strip() not in TRACE_DEBUG_VERDICTS
    )
    recoverable_rows = [
        normalize_trace_debug_audit_row(row)
        for row in rows
        if str(row.get("verdict") or "").strip() in TRACE_DEBUG_RECOVERABLE_VERDICTS and truthy(row.get("keep"))
    ]
    failure_modes = Counter(
        (
            str(row.get("verdict") or "pending").strip() or "pending",
            str(row.get("drop_reason") or ""),
            str(row.get("source") or ""),
            str(row.get("check_name") or "")[:120],
        )
        for row in rows
    )
    summary = {
        "generated_at": utc_now(),
        "audit": str(audit_path),
        "total": len(rows),
        "pending": len(pending),
        "invalid_verdicts": dict(sorted(invalid_verdicts.items())),
        "verdicts": dict(sorted(Counter(str(row.get("verdict") or "pending").strip() or "pending" for row in rows).items())),
        "by_source": dict(sorted(Counter(str(row.get("source") or "") for row in rows).items())),
        "by_repo": dict(sorted(Counter(str(row.get("repo") or "") for row in rows).items())),
        "by_drop_reason": dict(sorted(Counter(str(row.get("drop_reason") or "") for row in rows).items())),
        "recoverable": len(recoverable_rows),
        "recoverable_rate": (len(recoverable_rows) / len(rows)) if rows else 0.0,
        "top_failure_modes": [
            {
                "verdict": verdict,
                "drop_reason": drop_reason,
                "source": source,
                "check_name": check_name,
                "count": count,
            }
            for (verdict, drop_reason, source, check_name), count in failure_modes.most_common(20)
        ],
        "quality_gate": {
            "pending_zero": not pending,
            "invalid_verdicts_zero": not invalid_verdicts,
            "has_recoverable_signals": bool(recoverable_rows),
        },
        "outputs": {"summary": str(out_path), "recoverable_signals": str(recoverable_out)},
        "valid_audit_verdicts": list(TRACE_DEBUG_VERDICTS),
    }
    write_jsonl(recoverable_out, recoverable_rows)
    write_json(out_path, summary)
    return summary


def trace_source_scan(
    raw_dir: Path,
    out_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    audit_limit: int = 50,
    min_score: int = 4,
) -> dict[str, Any]:
    rows, scanned, rejected, repo_stats = collect_trace_source_candidates(
        raw_dir=raw_dir,
        repos=repos,
        max_changed_files=max_changed_files,
        min_score=min_score,
    )
    rows = dedupe_trace_source_rows(rows)
    rows.sort(key=trace_source_sort_key)
    audit_rows = [trace_source_audit_row(row) for row in rows[: max(0, audit_limit)]]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "source_candidates.jsonl", rows)
    write_jsonl(out_dir / "audit_samples.jsonl", audit_rows)
    write_csv(
        out_dir / "audit_samples.csv",
        audit_rows,
        (
            "sample_id",
            "repo",
            "pr_number",
            "source",
            "check_name",
            "score",
            "tags",
            "failure_excerpt",
            "trace_paths",
            "implementation_files",
            "test_files",
            "verdict",
            "reason",
            "keep",
            "notes",
        ),
    )
    for repo, stats in repo_stats.items():
        stats["usable_ratio"] = (stats.get("candidates", 0) / stats.get("scanned", 1)) if stats.get("scanned") else 0.0
    summary = {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "max_changed_files": max_changed_files,
        "min_score": min_score,
        "scanned": dict(sorted(scanned.items())),
        "candidates": len(rows),
        "audit_rows": len(audit_rows),
        "by_repo": dict(sorted(Counter(str(row.get("repo", "")) for row in rows).items())),
        "by_source": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "rejected": dict(sorted(rejected.items())),
        "repo_stats": {repo: dict(sorted(stats.items())) for repo, stats in sorted(repo_stats.items())},
        "quality_gate": {
            "source_candidates_ge_80": len(rows) >= 80,
            "audit_rows_ge_50": len(audit_rows) >= 50,
            "ready_for_source_audit": len(audit_rows) >= 50,
        },
        "outputs": {
            "source_candidates": str(out_dir / "source_candidates.jsonl"),
            "audit_jsonl": str(out_dir / "audit_samples.jsonl"),
            "audit_csv": str(out_dir / "audit_samples.csv"),
            "summary": str(out_dir / "summary.json"),
        },
        "valid_audit_verdicts": list(TRACE_SOURCE_VERDICTS),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def collect_trace_candidates(
    raw_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    include_review_comments: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    repo_names = list(repos or repos_from_raw(raw_dir))
    candidates: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    for repo in repo_names:
        repo_raw = raw_dir / repo_slug(repo)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        comments_by_pr = latest_by_pr(repo_raw / "review_comments.jsonl")
        checks_by_pr = group_by_pr(repo_raw / "check_runs.jsonl")
        logs_by_pr = group_by_pr(repo_raw / "job_logs.jsonl")
        for pr_number, pr_record in pr_by_number.items():
            pr = pr_record.get("data") or {}
            changed_paths = changed_paths_for(files_by_pr.get(pr_number))
            if not usable_pr(pr, changed_paths, max_changed_files):
                dropped["unusable_pr"] += 1
                continue
            implementation, tests, _ignored = split_changed_files(changed_paths)
            implementation = [path for path in implementation if is_source_file(path) and not is_test_file(path)]
            if not implementation:
                dropped["no_implementation"] += 1
                continue
            if include_review_comments:
                for comment in (comments_by_pr.get(pr_number) or {}).get("data", []):
                    scanned["review_comment"] += 1
                    candidate = trace_candidate_from_signal(
                        repo=repo,
                        pr=pr,
                        source="review_comment",
                        source_id=comment.get("id"),
                        raw_signal=comment.get("body") or "",
                        implementation=implementation,
                        tests=tests,
                        metadata={"path": comment.get("path"), "line": comment.get("line") or comment.get("original_line")},
                    )
                    if candidate:
                        candidates.append(candidate)
                    else:
                        dropped["weak_review_comment_trace"] += 1
            for check_record in checks_by_pr.get(pr_number, []):
                for check in check_record.get("data", []):
                    scanned["check_run"] += 1
                    if check.get("conclusion") not in {"failure", "timed_out", "cancelled", "action_required"}:
                        dropped["non_failed_check"] += 1
                        continue
                    output = check.get("output") or {}
                    raw_signal = "\n".join(
                        part for part in [check.get("name"), output.get("title"), output.get("summary"), output.get("text")] if part
                    )
                    candidate = trace_candidate_from_signal(
                        repo=repo,
                        pr=pr,
                        source="check_run",
                        source_id=check.get("id"),
                        raw_signal=raw_signal,
                        implementation=implementation,
                        tests=tests,
                        metadata={"check_name": check.get("name"), "check_url": check.get("html_url")},
                    )
                    if candidate:
                        candidates.append(candidate)
                    else:
                        dropped["weak_check_trace"] += 1
            for log_record in logs_by_pr.get(pr_number, []):
                scanned["job_log"] += 1
                if log_record.get("type") != "job_log":
                    dropped["non_job_log"] += 1
                    continue
                raw_signal = _read_log_excerpt(repo_raw, log_record) or log_record.get("excerpt") or ""
                candidate = trace_candidate_from_signal(
                    repo=repo,
                    pr=pr,
                    source="job_log",
                    source_id=log_record.get("job_id"),
                    raw_signal=raw_signal,
                    implementation=implementation,
                    tests=tests,
                    metadata={
                        "check_name": log_record.get("check_name"),
                        "check_url": log_record.get("html_url"),
                        "log_path": log_record.get("log_path"),
                    },
                )
                if candidate:
                    candidates.append(candidate)
                else:
                    dropped["weak_job_log_trace"] += 1

    return candidates, scanned, dropped


def collect_trace_debug_drops(
    raw_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    repo_names = list(repos or repos_from_raw(raw_dir))
    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    for repo in repo_names:
        repo_raw = raw_dir / repo_slug(repo)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        checks_by_pr = group_by_pr(repo_raw / "check_runs.jsonl")
        logs_by_pr = group_by_pr(repo_raw / "job_logs.jsonl")
        for pr_number, pr_record in pr_by_number.items():
            pr = pr_record.get("data") or {}
            changed_paths = changed_paths_for(files_by_pr.get(pr_number))
            if not usable_pr(pr, changed_paths, max_changed_files):
                dropped["unusable_pr"] += 1
                continue
            implementation, tests, _ignored = split_changed_files(changed_paths)
            implementation = [path for path in implementation if is_source_file(path) and not is_test_file(path)]
            if not implementation:
                dropped["no_implementation"] += 1
                continue
            for check_record in checks_by_pr.get(pr_number, []):
                for check in check_record.get("data", []):
                    scanned["check_run"] += 1
                    if check.get("conclusion") not in {"failure", "timed_out", "cancelled", "action_required"}:
                        dropped["non_failed_check"] += 1
                        continue
                    output = check.get("output") or {}
                    raw_signal = "\n".join(
                        part for part in [check.get("name"), output.get("title"), output.get("summary"), output.get("text")] if part
                    )
                    candidate, debug = analyze_trace_signal(
                        repo=repo,
                        pr=pr,
                        source="check_run",
                        source_id=check.get("id"),
                        raw_signal=raw_signal,
                        implementation=implementation,
                        tests=tests,
                        metadata={"check_name": check.get("name"), "check_url": check.get("html_url")},
                    )
                    if candidate:
                        dropped["candidate"] += 1
                    elif should_emit_debug_drop(debug):
                        rows.append(debug)
                    else:
                        dropped[debug.get("drop_reason") or "weak_check_trace"] += 1
            for log_record in logs_by_pr.get(pr_number, []):
                scanned["job_log"] += 1
                if log_record.get("type") != "job_log":
                    dropped["non_job_log"] += 1
                    continue
                raw_signal = _read_log_excerpt(repo_raw, log_record) or log_record.get("excerpt") or ""
                candidate, debug = analyze_trace_signal(
                    repo=repo,
                    pr=pr,
                    source="job_log",
                    source_id=log_record.get("job_id"),
                    raw_signal=raw_signal,
                    implementation=implementation,
                    tests=tests,
                    metadata={
                        "check_name": log_record.get("check_name"),
                        "check_url": log_record.get("html_url"),
                        "log_path": log_record.get("log_path"),
                    },
                )
                if candidate:
                    dropped["candidate"] += 1
                elif should_emit_debug_drop(debug):
                    rows.append(debug)
                else:
                    dropped[debug.get("drop_reason") or "weak_job_log_trace"] += 1
    return rows, scanned, dropped


def collect_trace_source_candidates(
    raw_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    min_score: int = 4,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], dict[str, Counter[str]]]:
    repo_names = list(repos or repos_from_raw(raw_dir))
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    repo_stats: dict[str, Counter[str]] = {}
    for repo in repo_names:
        repo_stats.setdefault(repo, Counter())
        repo_raw = raw_dir / repo_slug(repo)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        checks_by_pr = group_by_pr(repo_raw / "check_runs.jsonl")
        logs_by_pr = group_by_pr(repo_raw / "job_logs.jsonl")
        for pr_number, pr_record in pr_by_number.items():
            pr = pr_record.get("data") or {}
            changed_paths = changed_paths_for(files_by_pr.get(pr_number))
            if not usable_pr(pr, changed_paths, max_changed_files):
                rejected["unusable_pr"] += 1
                repo_stats[repo]["unusable_pr"] += 1
                continue
            implementation, tests, _ignored = split_changed_files(changed_paths)
            implementation = [path for path in implementation if is_source_file(path) and not is_test_file(path)]
            if not implementation:
                rejected["no_implementation"] += 1
                repo_stats[repo]["no_implementation"] += 1
                continue
            for check_record in checks_by_pr.get(pr_number, []):
                for check in check_record.get("data", []):
                    scanned["check_run"] += 1
                    repo_stats[repo]["scanned"] += 1
                    if check.get("conclusion") not in {"failure", "timed_out", "cancelled", "action_required"}:
                        rejected["non_failed_check"] += 1
                        repo_stats[repo]["non_failed_check"] += 1
                        continue
                    output = check.get("output") or {}
                    raw_signal = "\n".join(
                        part for part in [check.get("name"), output.get("title"), output.get("summary"), output.get("text")] if part
                    )
                    row, reason = trace_source_row_from_signal(
                        repo=repo,
                        pr=pr,
                        source="check_run",
                        source_id=check.get("id"),
                        raw_signal=raw_signal,
                        implementation=implementation,
                        tests=tests,
                        metadata={"check_name": check.get("name"), "check_url": check.get("html_url")},
                        min_score=min_score,
                    )
                    if row:
                        rows.append(row)
                        repo_stats[repo]["candidates"] += 1
                    else:
                        rejected[reason] += 1
                        repo_stats[repo][reason] += 1
            for log_record in logs_by_pr.get(pr_number, []):
                scanned["job_log"] += 1
                repo_stats[repo]["scanned"] += 1
                if log_record.get("type") != "job_log":
                    rejected["non_job_log"] += 1
                    repo_stats[repo]["non_job_log"] += 1
                    continue
                raw_signal = _read_log_excerpt(repo_raw, log_record) or log_record.get("excerpt") or ""
                row, reason = trace_source_row_from_signal(
                    repo=repo,
                    pr=pr,
                    source="job_log",
                    source_id=log_record.get("job_id"),
                    raw_signal=raw_signal,
                    implementation=implementation,
                    tests=tests,
                    metadata={
                        "check_name": log_record.get("check_name"),
                        "check_url": log_record.get("html_url"),
                        "log_path": log_record.get("log_path"),
                    },
                    min_score=min_score,
                )
                if row:
                    rows.append(row)
                    repo_stats[repo]["candidates"] += 1
                else:
                    rejected[reason] += 1
                    repo_stats[repo][reason] += 1
    return rows, scanned, rejected, repo_stats


def trace_summary(
    raw_dir: Path,
    out_dir: Path,
    candidates: list[dict[str, Any]],
    scanned: Counter[str],
    dropped: Counter[str],
    max_changed_files: int,
    outputs: dict[str, str],
) -> dict[str, Any]:
    by_repo = Counter(str(row.get("repo", "")) for row in candidates)
    by_source = Counter(str(row.get("source", "")) for row in candidates)
    return {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "max_changed_files": max_changed_files,
        "scanned": dict(sorted(scanned.items())),
        "candidates": len(candidates),
        "by_repo": dict(sorted(by_repo.items())),
        "by_source": dict(sorted(by_source.items())),
        "dropped": dict(sorted(dropped.items())),
        "quality_gate": {
            "candidate_count_ge_80": len(candidates) >= 80,
            "ready_for_trace_audit": len(candidates) >= 80,
        },
        "outputs": outputs,
    }


def trace_candidate_from_signal(
    repo: str,
    pr: dict[str, Any],
    source: str,
    source_id: Any,
    raw_signal: str,
    implementation: list[str],
    tests: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    candidate, _debug = analyze_trace_signal(repo, pr, source, source_id, raw_signal, implementation, tests, metadata)
    return candidate


def analyze_trace_signal(
    repo: str,
    pr: dict[str, Any],
    source: str,
    source_id: Any,
    raw_signal: str,
    implementation: list[str],
    tests: list[str],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    original_signal = raw_signal or ""
    check_name = str(metadata.get("check_name") or "")
    debug = trace_debug_record(
        repo=repo,
        pr=pr,
        source=source,
        source_id=source_id,
        raw_signal=original_signal,
        implementation=implementation,
        tests=tests,
        metadata=metadata,
        drop_reason="",
        trace_paths=[],
        candidate_root_files=[],
    )
    if should_drop_trace_signal(source, check_name, original_signal):
        debug["drop_reason"] = "signal_noise"
        return None, debug
    raw_signal = clean_failure_text(truncate_text(_failure_excerpt(original_signal), 5000))
    if source != "review_comment" and is_infra_failure_excerpt(raw_signal):
        debug["drop_reason"] = "signal_noise"
        debug["failure_excerpt"] = raw_signal
        return None, debug
    if not raw_signal or not has_failure_or_trace_signal(raw_signal):
        debug["drop_reason"] = "no_failure_signal"
        debug["failure_excerpt"] = raw_signal
        return None, debug
    trace_paths = normalize_trace_paths(extract_repo_trace_paths(raw_signal), repo)
    debug["failure_excerpt"] = raw_signal
    debug["trace_paths"] = trace_paths
    if not is_real_failure_trace(raw_signal, trace_paths):
        debug["drop_reason"] = "no_trace_paths" if not trace_paths else "not_real_failure_trace"
        return None, debug
    inference = infer_trace_roots(
        source=source,
        raw_signal=raw_signal,
        trace_paths=trace_paths,
        implementation=implementation,
        tests=tests,
        test_names=_extract_test_names(raw_signal),
    )
    root_files = inference["root_files"]
    root_files = [path for path in root_files if is_source_file(path) and not is_test_file(path)][:3]
    if not root_files:
        debug["drop_reason"] = "no_root_candidate"
        debug["candidate_root_files"] = inference.get("candidate_root_files") or []
        return None, debug
    test_names = _extract_test_names(raw_signal)
    evidence = trace_evidence(source, raw_signal, trace_paths, root_files, inference.get("evidence") or [])
    candidate = {
        "id": stable_id(repo, "trace_preflight", pr.get("number"), source, source_id, *root_files),
        "repo": repo,
        "pr_number": pr.get("number"),
        "pr_url": pr.get("url"),
        "base_commit": pr.get("baseRefOid"),
        "fix_commit": (pr.get("mergeCommit") or {}).get("oid"),
        "source": source,
        "source_id": source_id,
        "raw_signal": raw_signal,
        "trace_paths": trace_paths,
        "root_cause_files": root_files,
        "related_tests": tests[:10],
        "supporting_files": [path for path in implementation if path not in root_files][:10],
        "test_names": test_names,
        "evidence": evidence,
        "metadata": metadata,
    }
    debug["drop_reason"] = "candidate"
    debug["candidate_root_files"] = root_files
    return candidate, debug


def should_drop_trace_signal(source: str, check_name: str, raw_signal: str) -> bool:
    if not raw_signal.strip():
        return True
    if contains_review_leakage(raw_signal):
        return True
    if source == "review_comment" and POST_HOC_REVIEW_RE.search(raw_signal):
        return True
    if source != "review_comment" and (
        is_ignored_check_signal(check_name, raw_signal)
        or is_job_name_only_signal(check_name, raw_signal)
        or is_runner_setup_noise(raw_signal)
    ):
        return True
    return False


def is_infra_failure_excerpt(raw_signal: str) -> bool:
    lowered = raw_signal.lower()
    if not raw_signal.strip():
        return True
    if "no module named 'hypothesis'" in lowered or 'no module named "hypothesis"' in lowered:
        return True
    if INFRA_FAILURE_RE.search(raw_signal) and not REAL_FAILURE_CONTEXT_RE.search(raw_signal):
        return True
    return False


def is_real_failure_trace(raw_signal: str, trace_paths: list[str]) -> bool:
    if not trace_paths:
        return False
    return bool(STRONG_FAILURE_RE.search(raw_signal))


def fallback_root_files_from_failure_context(
    source: str,
    trace_paths: list[str],
    implementation: list[str],
    tests: list[str],
    test_names: list[str],
) -> list[str]:
    if source == "review_comment":
        return []
    if not implementation or len(implementation) > 3:
        return []
    if match_trace_files(trace_paths, tests) or test_names:
        return implementation[:3]
    return []


def infer_trace_roots(
    source: str,
    raw_signal: str,
    trace_paths: list[str],
    implementation: list[str],
    tests: list[str],
    test_names: list[str],
) -> dict[str, Any]:
    source_hits = match_trace_files(trace_paths, implementation)
    if source_hits:
        return {"root_files": source_hits, "candidate_root_files": source_hits, "evidence": ["source_trace_frame"]}
    test_hits = match_trace_files(trace_paths, tests)
    mapped = map_test_trace_to_implementation(
        test_hits=test_hits,
        test_names=test_names,
        implementation=implementation,
        raw_signal=raw_signal,
    )
    if source != "review_comment" and mapped:
        evidence = ["test_trace_to_impl_mapping"]
        if len(implementation) <= 5:
            evidence.append("small_fix_diff_root_source")
        if any(path_relation_score(test_file, impl_file, raw_signal) >= 2 for test_file in test_hits for impl_file in mapped):
            evidence.append("package_overlap")
        if test_names:
            evidence.append("explicit_test_failure_name")
        return {"root_files": mapped, "candidate_root_files": mapped, "evidence": evidence}
    fallback = fallback_root_files_from_failure_context(source, trace_paths, implementation, tests, test_names)
    return {"root_files": fallback, "candidate_root_files": fallback, "evidence": ["small_fix_diff_root_source"] if fallback else []}


def map_test_trace_to_implementation(
    test_hits: list[str],
    test_names: list[str],
    implementation: list[str],
    raw_signal: str,
) -> list[str]:
    if not implementation or len(implementation) > 5:
        return []
    if not test_hits and not test_names:
        return []
    scored: list[tuple[int, str]] = []
    relation_inputs = test_hits or test_names
    for impl_file in implementation:
        score = 0
        for value in relation_inputs:
            score = max(score, path_relation_score(str(value), impl_file, raw_signal))
        scored.append((score, impl_file))
    positive = [(score, path) for score, path in scored if score > 0]
    if positive:
        positive.sort(key=lambda item: (-item[0], implementation.index(item[1])))
        return [path for _score, path in positive[:3]]
    if len(implementation) <= 3 and (test_hits or test_names):
        return implementation[:3]
    return []


def path_relation_score(test_value: str, impl_file: str, raw_signal: str = "") -> int:
    test_tokens = path_tokens(test_value)
    impl_tokens = path_tokens(impl_file)
    if not impl_tokens:
        return 0
    score = 0
    if test_tokens & impl_tokens:
        score += len(test_tokens & impl_tokens)
    if stripped_stem(test_value) and stripped_stem(test_value) == stripped_stem(impl_file):
        score += 4
    impl_text = impl_file.replace("\\", "/").lower()
    if stripped_stem(impl_file) and re.search(rf"\b{re.escape(stripped_stem(impl_file))}\b", raw_signal, re.IGNORECASE):
        score += 1
    if common_directory_depth(test_value, impl_file) >= 2:
        score += 2
    if impl_text.endswith("/__init__.py") and test_tokens & set(Path(impl_file).parts):
        score += 1
    return score


def match_trace_files(trace_paths: list[str], files: list[str]) -> list[str]:
    normalized_files = {normalize_repo_path(path): path for path in files}
    hits: list[str] = []
    for trace_path in trace_paths:
        normalized_trace = normalize_repo_path(trace_path)
        for file_path, original in normalized_files.items():
            if not normalized_trace or not file_path:
                continue
            if normalized_trace.endswith(file_path) or file_path.endswith(normalized_trace):
                hits.append(original)
                continue
            if Path(normalized_trace).name == Path(file_path).name and path_relation_score(normalized_trace, file_path) >= 2:
                hits.append(original)
    return dedupe(hits)


def clean_failure_text(text: str) -> str:
    without_ansi = ANSI_RE.sub("", text.replace("\r\n", "\n"))
    return GITHUB_TIMESTAMP_RE.sub("", without_ansi)


def normalize_trace_paths(trace_paths: list[str], repo: str) -> list[str]:
    return dedupe(normalize_trace_path(path, repo) for path in trace_paths)


def normalize_trace_path(path: str, repo: str) -> str:
    owner, name = repo.split("/", 1)
    value = ANSI_RE.sub("", path).replace("\\", "/").strip("'\"`()[]{}.,")
    value = re.sub(r"^[A-Z]:/", "", value)
    value = value.lstrip("./")
    if any(part in value for part in LOW_VALUE_TRACE_PARTS):
        return value
    value = WORKSPACE_PREFIX_RE.sub("", value)
    for prefix in (
        f"github.com/{owner}/{name}/",
        f"{owner}/{name}/",
        f"{name}/{name}/",
    ):
        if prefix in value:
            value = value.split(prefix, 1)[1]
    marker = f"/{name}/"
    if marker in value:
        value = value.rsplit(marker, 1)[1]
    module_marker = f".{name}/"
    if module_marker in value:
        value = value.split(module_marker, 1)[1]
    if "/" not in value and value.endswith((".java", ".kt", ".scala")) and "." in value:
        value = value.replace(".", "/").replace("/java", ".java").replace("/kt", ".kt").replace("/scala", ".scala")
    return normalize_repo_path(value)


def normalize_repo_path(path: str) -> str:
    value = path.replace("\\", "/").strip("'\"` .")
    value = re.sub(r"^\./", "", value)
    return value


def path_tokens(path: str) -> set[str]:
    normalized = normalize_repo_path(path).lower()
    parts = re.split(r"[/_.\\-]+", normalized)
    ignored = {
        "",
        "src",
        "source",
        "lib",
        "libs",
        "pkg",
        "packages",
        "test",
        "tests",
        "testing",
        "spec",
        "specs",
        "__tests__",
        "unit",
        "integration",
        "py",
        "ts",
        "tsx",
        "js",
        "jsx",
        "java",
        "go",
        "rs",
        "kt",
        "scala",
    }
    return {strip_test_marker(part) for part in parts if strip_test_marker(part) not in ignored}


def stripped_stem(path: str) -> str:
    return strip_test_marker(Path(normalize_repo_path(path)).stem.lower())


def strip_test_marker(value: str) -> str:
    output = value.lower()
    output = re.sub(r"^(test|spec)[_-]?", "", output)
    output = re.sub(r"[_-]?(test|spec)$", "", output)
    return output


def common_directory_depth(left: str, right: str) -> int:
    left_parts = [part for part in Path(normalize_repo_path(left)).parts[:-1] if part not in {"src", "test", "tests", "__tests__", "spec"}]
    right_parts = [part for part in Path(normalize_repo_path(right)).parts[:-1] if part not in {"src", "test", "tests", "__tests__", "spec"}]
    count = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        count += 1
    return count


def trace_debug_record(
    repo: str,
    pr: dict[str, Any],
    source: str,
    source_id: Any,
    raw_signal: str,
    implementation: list[str],
    tests: list[str],
    metadata: dict[str, Any],
    drop_reason: str,
    trace_paths: list[str],
    candidate_root_files: list[str],
) -> dict[str, Any]:
    failure_excerpt = clean_failure_text(truncate_text(_failure_excerpt(raw_signal or ""), 5000))
    return {
        "id": stable_id(repo, "trace_debug", pr.get("number"), source, source_id, drop_reason),
        "repo": repo,
        "pr_number": pr.get("number"),
        "pr_url": pr.get("url"),
        "source": source,
        "source_id": source_id,
        "check_name": metadata.get("check_name"),
        "drop_reason": drop_reason,
        "failure_excerpt": failure_excerpt,
        "trace_paths": trace_paths,
        "implementation_files": implementation,
        "test_files": tests,
        "candidate_root_files": candidate_root_files,
        "test_names": _extract_test_names(failure_excerpt),
        "metadata": metadata,
    }


def should_emit_debug_drop(row: dict[str, Any]) -> bool:
    reason = str(row.get("drop_reason") or "")
    if reason in {"signal_noise", "no_failure_signal"}:
        return False
    return bool(row.get("failure_excerpt")) and (
        bool(row.get("trace_paths")) or has_failure_or_trace_signal(str(row.get("failure_excerpt") or ""))
    )


def trace_debug_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    implementation_count = len(row.get("implementation_files") or [])
    trace_count = len(row.get("trace_paths") or [])
    return (
        TRACE_SOURCE_PRIORITY.get(str(row.get("source")), 9),
        0 if 1 <= implementation_count <= 5 else 1,
        0 if trace_count else 1,
        str(row.get("drop_reason", "")),
        str(row.get("repo", "")),
        -int(row.get("pr_number") or 0),
        str(row.get("id", "")),
    )


def dedupe_debug_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("repo"), row.get("pr_number"), row.get("source"), row.get("source_id"), row.get("drop_reason"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def trace_debug_audit_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": str(row.get("id") or ""),
        "repo": str(row.get("repo") or ""),
        "pr_number": str(row.get("pr_number") or ""),
        "source": str(row.get("source") or ""),
        "check_name": str(row.get("check_name") or ""),
        "drop_reason": str(row.get("drop_reason") or ""),
        "failure_excerpt": str(row.get("failure_excerpt") or ""),
        "trace_paths": "; ".join(row.get("trace_paths") or []),
        "implementation_files": "; ".join(row.get("implementation_files") or []),
        "test_files": "; ".join(row.get("test_files") or []),
        "candidate_root_files": "; ".join(row.get("candidate_root_files") or []),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "verdict options: " + "/".join(TRACE_DEBUG_VERDICTS),
    }


def read_trace_debug_audit(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_trace_debug_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for field in ("trace_paths", "implementation_files", "test_files", "candidate_root_files"):
        output[field] = split_semicolon_field(row.get(field))
    output["keep"] = truthy(row.get("keep"))
    return output


def split_semicolon_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "keep"}


def trace_source_row_from_signal(
    repo: str,
    pr: dict[str, Any],
    source: str,
    source_id: Any,
    raw_signal: str,
    implementation: list[str],
    tests: list[str],
    metadata: dict[str, Any],
    min_score: int,
) -> tuple[dict[str, Any] | None, str]:
    check_name = str(metadata.get("check_name") or "")
    if should_drop_trace_signal(source, check_name, raw_signal):
        return None, "signal_noise"
    failure_excerpt = clean_failure_text(truncate_text(_failure_excerpt(raw_signal or ""), 5000))
    trace_paths = normalize_trace_paths(extract_repo_trace_paths(failure_excerpt), repo)
    test_names = _extract_test_names(failure_excerpt)
    score, tags, reject_reason = score_trace_source(
        source=source,
        check_name=check_name,
        failure_excerpt=failure_excerpt,
        trace_paths=trace_paths,
        implementation=implementation,
        tests=tests,
        test_names=test_names,
    )
    if reject_reason:
        return None, reject_reason
    if score < min_score:
        return None, "low_score"
    row = {
        "id": stable_id(repo, "trace_source", pr.get("number"), source, source_id, *trace_paths[:3], *implementation[:3]),
        "repo": repo,
        "pr_number": pr.get("number"),
        "pr_url": pr.get("url"),
        "source": source,
        "source_id": source_id,
        "check_name": check_name,
        "score": score,
        "tags": tags,
        "failure_excerpt": failure_excerpt,
        "trace_paths": trace_paths,
        "implementation_files": implementation,
        "test_files": tests,
        "test_names": test_names,
        "metadata": metadata,
    }
    return row, ""


def score_trace_source(
    source: str,
    check_name: str,
    failure_excerpt: str,
    trace_paths: list[str],
    implementation: list[str],
    tests: list[str],
    test_names: list[str],
) -> tuple[int, list[str], str]:
    if not failure_excerpt or not has_failure_or_trace_signal(failure_excerpt):
        return 0, [], "no_failure_signal"
    if DOWNSTREAM_CHECK_RE.search(check_name):
        return 0, ["downstream_matrix"], "downstream_check"
    if is_infra_failure_excerpt(failure_excerpt):
        return 0, ["infra_setup_or_dependency"], "infra_noise"
    if not REAL_FAILURE_CONTEXT_RE.search(failure_excerpt):
        return 0, [], "no_real_failure_context"
    if not implementation:
        return 0, [], "no_implementation"
    if len(implementation) > 5:
        return 0, ["too_many_changed_sources"], "too_broad"

    score = 0
    tags: list[str] = []
    if source == "job_log":
        score += 1
        tags.append("job_log")
    if STRONG_FAILURE_RE.search(failure_excerpt):
        score += 1
        tags.append("strong_failure_signal")
    if trace_paths:
        score += 2
        tags.append("repo_owned_trace_path")
    if match_trace_files(trace_paths, implementation):
        score += 5
        tags.append("source_trace_frame")
    if match_trace_files(trace_paths, tests):
        score += 3
        tags.append("test_trace_frame")
    if test_names:
        score += 2
        tags.append("explicit_test_failure_name")
    if len(implementation) <= 3:
        score += 1
        tags.append("small_source_diff")
    if len(tests) <= 3 and tests:
        score += 1
        tags.append("small_test_diff")
    if not trace_paths and not test_names:
        return score, tags, "no_trace_context"
    if trace_paths and all(is_low_value_trace_path(path) for path in trace_paths):
        return score, tags, "third_party_only"
    return score, dedupe(tags), ""


def is_low_value_trace_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    lowered = normalized.lower()
    if any(part in lowered for part in LOW_VALUE_TRACE_PARTS):
        return True
    return lowered.startswith(
        (
            "server/migrations/",
            "api/tests/",
            "api/providers/",
            "python/semantic_kernel/",
        )
    )


def dedupe_trace_source_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("repo"), row.get("pr_number"), row.get("source"), row.get("source_id"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def trace_source_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row.get("score") or 0),
        TRACE_SOURCE_PRIORITY.get(str(row.get("source")), 9),
        str(row.get("repo", "")),
        -int(row.get("pr_number") or 0),
        str(row.get("id", "")),
    )


def trace_source_audit_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": str(row.get("id") or ""),
        "repo": str(row.get("repo") or ""),
        "pr_number": str(row.get("pr_number") or ""),
        "source": str(row.get("source") or ""),
        "check_name": str(row.get("check_name") or ""),
        "score": str(row.get("score") or 0),
        "tags": "; ".join(row.get("tags") or []),
        "failure_excerpt": str(row.get("failure_excerpt") or ""),
        "trace_paths": "; ".join(row.get("trace_paths") or []),
        "implementation_files": "; ".join(row.get("implementation_files") or []),
        "test_files": "; ".join(row.get("test_files") or []),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "verdict options: " + "/".join(TRACE_SOURCE_VERDICTS),
    }


def trace_evidence(source: str, raw_signal: str, trace_paths: list[str], root_files: list[str], inference_evidence: list[str]) -> list[str]:
    evidence = ["modified_in_fix_pr"]
    evidence.extend(inference_evidence)
    if source == "job_log":
        evidence.append("github_actions_job_log_failure")
    elif source == "check_run":
        evidence.append("github_check_output_failure")
    else:
        evidence.append("review_failure_snippet")
    if _extract_test_names(raw_signal):
        evidence.append("explicit_test_failure_name")
    if len(trace_paths) > len(root_files):
        evidence.append("trace_has_context_frames")
    return dedupe(evidence)


def trace_candidate_to_sample(candidate: dict[str, Any]) -> dict[str, Any]:
    root_files = dedupe(candidate.get("root_cause_files") or [])[:3]
    fix_commit = candidate.get("fix_commit")
    query = {
        "failure_excerpt": candidate.get("raw_signal", ""),
        "source": candidate.get("source", ""),
        "check_name": (candidate.get("metadata") or {}).get("check_name"),
        "test_names": candidate.get("test_names") or [],
        "trace_paths": candidate.get("trace_paths") or [],
    }
    sample = {
        "id": stable_id(candidate.get("repo"), "trace2code", candidate.get("pr_number"), candidate.get("source"), candidate.get("source_id"), *root_files),
        "version": 2,
        "task_type": "trace2code",
        "repo": candidate.get("repo"),
        "base_commit": candidate.get("base_commit"),
        "query": redact_value(query, fix_commit),
        "gold": {
            "root_cause_files": root_files,
            "root_cause_symbols": [],
            "related_tests": dedupe(candidate.get("related_tests") or []),
            "supporting_files": dedupe(candidate.get("supporting_files") or []),
            "negative_distractors": [],
            "fix_commit": fix_commit,
        },
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": candidate.get("base_commit")},
        "metadata": {
            "pr": candidate.get("pr_number"),
            "pr_url": candidate.get("pr_url"),
            "source": candidate.get("source"),
            "source_id": candidate.get("source_id"),
            "confidence": "weak",
            "evidence": {
                "signals": candidate.get("evidence") or [],
                "trace_paths": candidate.get("trace_paths") or [],
                "source_metadata": candidate.get("metadata") or {},
            },
            "generated_at": utc_now(),
        },
    }
    return sample


def trace_audit_row(sample: dict[str, Any]) -> dict[str, str]:
    row = audit_row(sample)
    row["notes"] = "verdict options: " + "/".join(TRACE_AUDIT_VERDICTS)
    return row


def trace_candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        TRACE_SOURCE_PRIORITY.get(str(candidate.get("source")), 9),
        len(candidate.get("root_cause_files") or []),
        str(candidate.get("repo", "")),
        -int(candidate.get("pr_number") or 0),
        str(candidate.get("id", "")),
    )


def usable_pr(pr: dict[str, Any], changed_paths: list[str], max_changed_files: int) -> bool:
    if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
        return False
    if not changed_paths or len(changed_paths) > max_changed_files:
        return False
    implementation, tests, _ignored = split_changed_files(changed_paths)
    return bool(implementation or tests)


def changed_paths_for(files_record: dict[str, Any] | None) -> list[str]:
    if not files_record:
        return []
    return [file.get("filename") for file in files_record.get("data", []) if file.get("filename")]


def latest_by_pr(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        pr_number = record.get("pr_number") or (record.get("data") or {}).get("number")
        if pr_number is not None:
            records[int(pr_number)] = record
    return records


def group_by_pr(path: Path) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in read_jsonl(path):
        pr_number = record.get("pr_number")
        if pr_number is not None:
            groups.setdefault(int(pr_number), []).append(record)
    return groups


def repos_from_raw(raw_dir: Path) -> list[str]:
    repos: list[str] = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else []):
        if path.is_dir() and "__" in path.name:
            repos.append(path.name.replace("__", "/", 1))
    return repos


def dedupe_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            candidate.get("repo"),
            candidate.get("pr_number"),
            candidate.get("source"),
            candidate.get("source_id"),
            tuple(candidate.get("root_cause_files") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def redact_value(value: Any, fix_commit: str | None) -> Any:
    if not fix_commit:
        return value
    if isinstance(value, str):
        return value.replace(fix_commit, "[fix_commit]")
    if isinstance(value, list):
        return [redact_value(item, fix_commit) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, fix_commit) for key, item in value.items()}
    return value


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
