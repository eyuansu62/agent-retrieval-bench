from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .filters import (
    contains_review_leakage,
    extract_repo_trace_paths,
    has_failure_or_trace_signal,
    is_ignored_check_signal,
    is_job_name_only_signal,
    is_runner_setup_noise,
    sanitize_diff_hunk,
    sanitize_review_body,
    split_changed_files,
)
from .io import append_jsonl, read_jsonl, repo_slug, stable_id, truncate_text, utc_now

TEST_NAME_RE = re.compile(
    r"(?P<name>(?:test[_:][A-Za-z0-9_:\\[\\]\\.\\-/]+|[A-Za-z0-9_$.]+Test\\.[A-Za-z0-9_$]+|[A-Za-z0-9_$.]+::[A-Za-z0-9_]+))"
)


def derive_repo(raw_dir: Path, repo: str, out_dir: Path, max_changed_files: int = 20) -> dict[str, int]:
    slug = repo_slug(repo)
    repo_raw = raw_dir / slug
    pr_by_number = _latest_by_pr(repo_raw / "pull_requests.jsonl")
    files_by_pr = _latest_by_pr(repo_raw / "pull_files.jsonl")
    comments_by_pr = _latest_by_pr(repo_raw / "review_comments.jsonl")
    checks_by_pr = _group_by_pr(repo_raw / "check_runs.jsonl")
    logs_by_pr = _group_by_pr(repo_raw / "job_logs.jsonl")

    outputs = {
        "comment2context": list(_comment2context(repo, pr_by_number, files_by_pr, comments_by_pr, max_changed_files)),
        "trace2code": list(_review_trace2code(repo, pr_by_number, files_by_pr, comments_by_pr, max_changed_files)),
        "code2test": list(_code2test(repo, pr_by_number, files_by_pr, max_changed_files)),
        "testlog2code": [],
    }
    for sample in _log_samples(repo, repo_raw, pr_by_number, files_by_pr, logs_by_pr, max_changed_files):
        outputs[sample["task_type"]].append(sample)
    for sample in _failure_samples(repo, pr_by_number, files_by_pr, checks_by_pr, max_changed_files):
        outputs[sample["task_type"]].append(sample)

    counts: dict[str, int] = {}
    for task, samples in outputs.items():
        if samples:
            counts[task] = append_jsonl(out_dir / f"{task}.jsonl", samples)
        else:
            counts[task] = 0
    return counts


def _latest_by_pr(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        pr_number = record.get("pr_number") or (record.get("data") or {}).get("number")
        if pr_number is not None:
            records[int(pr_number)] = record
    return records


def _group_by_pr(path: Path) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in read_jsonl(path):
        pr_number = record.get("pr_number")
        if pr_number is not None:
            groups.setdefault(int(pr_number), []).append(record)
    return groups


def _comment2context(
    repo: str,
    pr_by_number: dict[int, dict[str, Any]],
    files_by_pr: dict[int, dict[str, Any]],
    comments_by_pr: dict[int, dict[str, Any]],
    max_changed_files: int,
) -> Iterable[dict[str, Any]]:
    for pr_number, comments_record in comments_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_by_pr.get(pr_number))
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        for comment in comments_record.get("data", []):
            raw_body = comment.get("body") or ""
            if contains_review_leakage(raw_body):
                continue
            body = sanitize_review_body(raw_body, 2200)
            path = comment.get("path")
            if not body or not path or path not in changed_paths:
                continue
            query = {
                "review_comment": body,
                "path": path,
                "line": comment.get("line") or comment.get("original_line"),
                "diff_hunk_context": sanitize_diff_hunk(comment.get("diff_hunk")),
                "pr_title": truncate_text(pr.get("title"), 300),
            }
            yield _sample(
                repo=repo,
                pr=pr,
                task_type="comment2context",
                seed=("comment", pr_number, comment.get("id")),
                query=query,
                root_files=[path],
                tests=tests,
                supporting=[candidate for candidate in implementation if candidate != path][:8],
                distractors=[candidate for candidate in changed_paths if candidate != path][:8],
                evidence={"review_comment_id": comment.get("id"), "comment_commit_id": comment.get("commit_id")},
                confidence="weak",
            )


def _code2test(
    repo: str,
    pr_by_number: dict[int, dict[str, Any]],
    files_by_pr: dict[int, dict[str, Any]],
    max_changed_files: int,
) -> Iterable[dict[str, Any]]:
    for pr_number, files_record in files_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_record)
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        if not implementation or not tests:
            continue
        for path in implementation[:6]:
            query = {
                "changed_file": path,
                "pr_title": truncate_text(pr.get("title"), 300),
                "pr_body": truncate_text(pr.get("body"), 1400),
            }
            yield _sample(
                repo=repo,
                pr=pr,
                task_type="code2test",
                seed=("code2test", pr_number, path),
                query=query,
                root_files=[path],
                tests=tests[:10],
                supporting=[candidate for candidate in implementation if candidate != path][:8],
                distractors=[candidate for candidate in implementation if candidate != path][:8],
                evidence={"source": "same_pr_changed_tests"},
                confidence="weak",
            )


def _review_trace2code(
    repo: str,
    pr_by_number: dict[int, dict[str, Any]],
    files_by_pr: dict[int, dict[str, Any]],
    comments_by_pr: dict[int, dict[str, Any]],
    max_changed_files: int,
) -> Iterable[dict[str, Any]]:
    for pr_number, comments_record in comments_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_by_pr.get(pr_number))
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        searchable_paths = implementation + tests
        for comment in comments_record.get("data", []):
            raw_body = comment.get("body") or ""
            if contains_review_leakage(raw_body):
                continue
            body = sanitize_review_body(raw_body, 3200)
            if not body:
                continue
            trace_paths = extract_repo_trace_paths(body)
            matched_files = _matching_trace_files(trace_paths, searchable_paths)
            if not matched_files or not has_failure_or_trace_signal(body):
                continue
            query = {
                "raw_signal": body,
                "path": comment.get("path"),
                "line": comment.get("line") or comment.get("original_line"),
                "trace_paths": trace_paths,
                "test_names": _extract_test_names(body),
                "pr_title": truncate_text(pr.get("title"), 300),
            }
            yield _sample(
                repo=repo,
                pr=pr,
                task_type="trace2code",
                seed=("review_trace", pr_number, comment.get("id")),
                query=query,
                root_files=matched_files,
                tests=[path for path in matched_files if path in tests] or tests[:10],
                supporting=[path for path in implementation if path not in matched_files][:8],
                distractors=[path for path in changed_paths if path not in matched_files][:8],
                evidence={"review_comment_id": comment.get("id"), "comment_commit_id": comment.get("commit_id")},
                confidence="weak",
            )


def _failure_samples(
    repo: str,
    pr_by_number: dict[int, dict[str, Any]],
    files_by_pr: dict[int, dict[str, Any]],
    checks_by_pr: dict[int, list[dict[str, Any]]],
    max_changed_files: int,
) -> Iterable[dict[str, Any]]:
    for pr_number, check_records in checks_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_by_pr.get(pr_number))
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        if not implementation:
            continue
        for record in check_records:
            for check in record.get("data", []):
                if check.get("conclusion") not in {"failure", "timed_out", "cancelled", "action_required"}:
                    continue
                output = check.get("output") or {}
                raw_signal = "\n".join(
                    part for part in [check.get("name"), output.get("title"), output.get("summary"), output.get("text")] if part
                )
                raw_signal = truncate_text(raw_signal, 4000)
                if not raw_signal:
                    continue
                if _skip_failure_signal(check.get("name"), raw_signal):
                    continue
                trace_paths = extract_repo_trace_paths(raw_signal)
                matched_trace_files = _matching_trace_files(trace_paths, implementation)
                task_type = "trace2code" if matched_trace_files else "testlog2code"
                if not _is_actionable_task_signal(task_type, raw_signal, trace_paths, matched_trace_files):
                    continue
                query = {
                    "raw_signal": raw_signal,
                    "check_name": check.get("name"),
                    "check_url": check.get("html_url"),
                    "trace_paths": trace_paths,
                    "test_names": _extract_test_names(raw_signal),
                }
                yield _sample(
                    repo=repo,
                    pr=pr,
                    task_type=task_type,
                    seed=(task_type, pr_number, record.get("sha"), check.get("id")),
                    query=query,
                    root_files=matched_trace_files or implementation[:5],
                    tests=tests[:10],
                    supporting=implementation[:10],
                    distractors=[path for path in changed_paths if path not in implementation[:3]][:8],
                    evidence={"check_run_id": check.get("id"), "sha": record.get("sha"), "ref_type": record.get("ref_type")},
                    confidence="weak",
                )


def _log_samples(
    repo: str,
    repo_raw: Path,
    pr_by_number: dict[int, dict[str, Any]],
    files_by_pr: dict[int, dict[str, Any]],
    logs_by_pr: dict[int, list[dict[str, Any]]],
    max_changed_files: int,
) -> Iterable[dict[str, Any]]:
    for pr_number, log_records in logs_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_by_pr.get(pr_number))
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        if not implementation:
            continue
        for record in log_records:
            if record.get("type") != "job_log":
                continue
            log_text = _read_log_excerpt(repo_raw, record)
            raw_signal = truncate_text(log_text or record.get("excerpt"), 5000)
            if not raw_signal:
                continue
            if _skip_failure_signal(record.get("check_name"), raw_signal):
                continue
            trace_paths = extract_repo_trace_paths(raw_signal)
            matched_trace_files = _matching_trace_files(trace_paths, implementation)
            task_type = "trace2code" if matched_trace_files else "testlog2code"
            if not _is_actionable_task_signal(task_type, raw_signal, trace_paths, matched_trace_files):
                continue
            query = {
                "raw_signal": raw_signal,
                "check_name": record.get("check_name"),
                "check_url": record.get("html_url"),
                "trace_paths": trace_paths,
                "test_names": _extract_test_names(raw_signal),
            }
            yield _sample(
                repo=repo,
                pr=pr,
                task_type=task_type,
                seed=("job_log", task_type, pr_number, record.get("job_id")),
                query=query,
                root_files=matched_trace_files or implementation[:5],
                tests=tests[:10],
                supporting=implementation[:10],
                distractors=[path for path in changed_paths if path not in implementation[:3]][:8],
                evidence={
                    "job_id": record.get("job_id"),
                    "sha": record.get("sha"),
                    "ref_type": record.get("ref_type"),
                    "log_path": record.get("log_path"),
                },
                confidence="weak",
            )


def _read_log_excerpt(repo_raw: Path, record: dict[str, Any]) -> str:
    log_path = record.get("log_path")
    if log_path:
        path = repo_raw / log_path
        if path.exists():
            return _failure_excerpt(path.read_text(encoding="utf-8", errors="replace"))
    return record.get("excerpt") or ""


def _failure_excerpt(text: str, max_lines: int = 80) -> str:
    lines = text.replace("\r\n", "\n").splitlines()
    hit_indexes = [
        index
        for index, line in enumerate(lines)
        if re.search(r"(traceback|error:|failed|failure|panic|exception|assertion|FAIL|FAILED|Caused by|stack backtrace)", line, re.I)
    ]
    if not hit_indexes:
        return truncate_text("\n".join(lines[-max_lines:]), 5000)
    selected: list[str] = []
    seen: set[int] = set()
    for hit in hit_indexes[:8]:
        for index in range(max(0, hit - 8), min(len(lines), hit + 9)):
            if index not in seen:
                selected.append(lines[index])
                seen.add(index)
        if len(selected) >= max_lines:
            break
    return truncate_text("\n".join(selected[:max_lines]), 5000)


def _skip_failure_signal(check_name: str | None, raw_signal: str) -> bool:
    return (
        is_ignored_check_signal(check_name, raw_signal)
        or is_job_name_only_signal(check_name, raw_signal)
        or is_runner_setup_noise(raw_signal)
    )


def _is_actionable_task_signal(
    task_type: str,
    raw_signal: str,
    trace_paths: list[str],
    matched_trace_files: list[str],
) -> bool:
    if task_type == "trace2code":
        return bool(matched_trace_files) and has_failure_or_trace_signal(raw_signal)
    return bool(trace_paths or _extract_test_names(raw_signal) or has_failure_or_trace_signal(raw_signal))


def _sample(
    repo: str,
    pr: dict[str, Any],
    task_type: str,
    seed: tuple[Any, ...],
    query: dict[str, Any],
    root_files: list[str],
    tests: list[str],
    supporting: list[str],
    distractors: list[str],
    evidence: dict[str, Any],
    confidence: str,
) -> dict[str, Any]:
    base_commit = pr.get("baseRefOid")
    fix_commit = (pr.get("mergeCommit") or {}).get("oid")
    query = _redact_query(query, fix_commit)
    return {
        "id": stable_id(repo, *seed),
        "version": 1,
        "task_type": task_type,
        "repo": repo,
        "base_commit": base_commit,
        "query": query,
        "gold": {
            "root_cause_files": _dedupe(root_files),
            "root_cause_symbols": [],
            "related_tests": _dedupe(tests),
            "supporting_files": _dedupe(supporting),
            "negative_distractors": _dedupe(distractors),
            "fix_commit": fix_commit,
        },
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": base_commit},
        "metadata": {
            "pr": pr.get("number"),
            "pr_url": pr.get("url"),
            "created_at": pr.get("createdAt"),
            "merged_at": pr.get("mergedAt"),
            "confidence": confidence,
            "evidence": evidence,
            "generated_at": utc_now(),
        },
    }


def _changed_paths(files_record: dict[str, Any] | None) -> list[str]:
    if not files_record:
        return []
    return [file.get("filename") for file in files_record.get("data", []) if file.get("filename")]


def _usable_pr(pr: dict[str, Any], changed_paths: list[str], max_changed_files: int) -> bool:
    if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
        return False
    if len(changed_paths) > max_changed_files:
        return False
    implementation, tests, _ignored = split_changed_files(changed_paths)
    return bool(implementation or tests)


def _extract_test_names(text: str) -> list[str]:
    return _dedupe(match.group("name") for match in TEST_NAME_RE.finditer(text))[:20]


def _prioritize_trace_files(trace_paths: list[str], implementation: list[str]) -> list[str]:
    return _matching_trace_files(trace_paths, implementation) or _dedupe(implementation[:5])


def _matching_trace_files(trace_paths: list[str], implementation: list[str]) -> list[str]:
    normalized_impl = {path.replace("\\", "/"): path for path in implementation}
    hits: list[str] = []
    for trace_path in trace_paths:
        if "/" not in trace_path and "\\" not in trace_path:
            continue
        for impl_path, original in normalized_impl.items():
            if trace_path.endswith(impl_path) or impl_path.endswith(trace_path):
                hits.append(original)
    return _dedupe(hits)


def _redact_query(value: Any, fix_commit: str | None) -> Any:
    if not fix_commit:
        return value
    if isinstance(value, str):
        return value.replace(fix_commit, "[fix_commit]")
    if isinstance(value, list):
        return [_redact_query(item, fix_commit) for item in value]
    if isinstance(value, dict):
        return {key: _redact_query(item, fix_commit) for key, item in value.items()}
    return value


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
