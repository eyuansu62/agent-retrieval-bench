from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .derive import _extract_test_names, _failure_excerpt, _matching_trace_files, _read_log_excerpt
from .filters import (
    extract_repo_trace_paths,
    has_failure_or_trace_signal,
    is_source_file,
    is_test_file,
    split_changed_files,
)
from .io import read_jsonl, repo_slug, stable_id, truncate_text, utc_now, write_json


def trace_preflight(
    raw_dir: Path,
    out_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
) -> dict[str, Any]:
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

    candidates = dedupe_candidates(candidates)
    candidates.sort(key=lambda row: (row.get("repo", ""), row.get("pr_number", 0), row.get("source", ""), row.get("source_id", "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    by_repo = Counter(str(row.get("repo", "")) for row in candidates)
    by_source = Counter(str(row.get("source", "")) for row in candidates)
    summary = {
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
        "outputs": {"candidates": str(out_dir / "candidates.jsonl"), "summary": str(out_dir / "summary.json")},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


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
    raw_signal = truncate_text(_failure_excerpt(raw_signal), 5000)
    if not raw_signal or not has_failure_or_trace_signal(raw_signal):
        return None
    trace_paths = extract_repo_trace_paths(raw_signal)
    root_files = _matching_trace_files(trace_paths, implementation)
    if not root_files:
        return None
    test_names = _extract_test_names(raw_signal)
    return {
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
        "test_names": test_names,
        "metadata": metadata,
    }


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
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
