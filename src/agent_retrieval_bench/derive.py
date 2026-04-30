from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .filters import (
    contains_raw_patch_marker,
    contains_review_leakage,
    extract_repo_trace_paths,
    has_failure_or_trace_signal,
    is_ignored_check_signal,
    is_job_name_only_signal,
    is_runner_setup_noise,
    is_source_file,
    is_test_file,
    sanitize_diff_hunk,
    sanitize_review_body,
    split_changed_files,
)
from .io import append_jsonl, read_jsonl, repo_slug, stable_id, truncate_text, utc_now

TEST_NAME_RE = re.compile(
    r"(?P<name>(?:test[_:][A-Za-z0-9_:\\[\\]\\.\\-/]+|[A-Za-z0-9_$.]+Test\\.[A-Za-z0-9_$]+|[A-Za-z0-9_$.]+::[A-Za-z0-9_]+))"
)
ACTIONABLE_REVIEW_RE = re.compile(
    r"\b("
    r"should|shouldn't|could|can we|please|why|does this|do we|would you|missing|add tests?|"
    r"duplicate|duplicated|pull out|move|rename|return|prefer|instead|not sure|wdyt|what do you think"
    r")\b|\?",
    re.IGNORECASE,
)
POST_HOC_REVIEW_RE = re.compile(
    r"^\s*(fixed|addressed|added|updated|corrected|done|resolved|changed|thanks|thank you|lgtm|separated|good call)\b",
    re.IGNORECASE,
)
LOW_VALUE_PATH_PARTS = {
    ".github",
    "changelog",
    "changelogs",
    "doc",
    "docs",
    "documentation",
}
LOW_VALUE_FILENAMES = {
    "changelog.md",
    "contributing.md",
    "license",
    "license.md",
    "readme.md",
}
COMMENT_CONTEXT_RESPONSE_WINDOW = timedelta(hours=72)
MAX_COMMENT_CONTEXT_COMMIT_FILES = 8
MAX_COMMENT_CONTEXT_GOLD_FILES = 2
LOW_VALUE_REVIEW_RE = re.compile(
    r"\b("
    r"formatting|formatting changes?|style nit|nit(?:pick)?|header|copyright year|update the year|"
    r"changelog|change ?note|release note|pr description|linter|lint|type annotations?|"
    r"named return|parametri[sz]e the test case|move these imports|"
    r"documentation|docs?|javadocs?|section|note|mention|description|"
    r"readable|easier to follow|colorless|informational|make fix-copies|look through all"
    r")\b",
    re.IGNORECASE,
)
POST_HOC_REPLY_RE = re.compile(
    r"\b("
    r"i (have )?(updated|changed|adjusted|added|removed|fixed)|"
    r"i'll|i will|let me change|works locally|it works locally|"
    r"i misunderstood|i opted|i think that's the one we want|"
    r"as suggested|updated as suggested"
    r")\b",
    re.IGNORECASE,
)
CONTEXT_BEHAVIOR_RE = re.compile(
    r"\b("
    r"bug|fail(?:s|ing)?|failure|regression|coverage|test|assert|expected|behavior|behaviour|"
    r"api|interface|type|config|configuration|metadata|cache|runtime|scheduler|binding|"
    r"validation|validate|error|panic|exception|compat(?:ibility)?|refactor|support|"
    r"argument|latents?|pipeline|transformer|model|output|input|method|class"
    r")\b",
    re.IGNORECASE,
)
GENERIC_PATH_TOKENS = {
    "test",
    "tests",
    "spec",
    "index",
    "utils",
    "util",
    "common",
    "helper",
    "helpers",
    "main",
    "mod",
    "lib",
    "src",
}


def derive_repo(raw_dir: Path, repo: str, out_dir: Path, max_changed_files: int = 20) -> dict[str, int]:
    slug = repo_slug(repo)
    repo_raw = raw_dir / slug
    pr_by_number = _latest_by_pr(repo_raw / "pull_requests.jsonl")
    files_by_pr = _latest_by_pr(repo_raw / "pull_files.jsonl")
    commits_by_pr = _latest_by_pr(repo_raw / "pull_commits.jsonl")
    details_by_pr = _latest_by_pr(repo_raw / "commit_details.jsonl")
    comments_by_pr = _latest_by_pr(repo_raw / "review_comments.jsonl")
    checks_by_pr = _group_by_pr(repo_raw / "check_runs.jsonl")
    logs_by_pr = _group_by_pr(repo_raw / "job_logs.jsonl")

    outputs = {
        "comment2context": list(
            _comment2context(
                repo,
                pr_by_number,
                files_by_pr,
                comments_by_pr,
                max_changed_files,
                commits_by_pr,
                details_by_pr,
            )
        ),
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
    commits_by_pr: dict[int, dict[str, Any]] | None = None,
    details_by_pr: dict[int, dict[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    commits_by_pr = commits_by_pr or {}
    details_by_pr = details_by_pr or {}
    for pr_number, comments_record in comments_by_pr.items():
        pr = (pr_by_number.get(pr_number) or {}).get("data", {})
        changed_paths = _changed_paths(files_by_pr.get(pr_number))
        changed_statuses = _changed_statuses(files_by_pr.get(pr_number))
        base_existing_paths = {path for path, status in changed_statuses.items() if status != "added"}
        if not _usable_pr(pr, changed_paths, max_changed_files):
            continue
        implementation, tests, _ignored = split_changed_files(changed_paths)
        for comment in comments_record.get("data", []):
            if _skip_comment2context_comment(comment):
                continue
            raw_body = comment.get("body") or ""
            if contains_review_leakage(raw_body):
                continue
            body = sanitize_review_body(raw_body, 2200)
            path = comment.get("path")
            if not body or not path or path not in changed_paths:
                continue
            if changed_statuses.get(path) == "added":
                continue
            if _skip_comment2context_body(body) or _skip_comment2context_path(path, body):
                continue
            post_comment_commits = _post_comment_commits(
                commits_by_pr.get(pr_number),
                comment.get("created_at"),
                details_by_pr.get(pr_number),
                COMMENT_CONTEXT_RESPONSE_WINDOW,
            )
            response_commit = _first_valid_comment_response_commit(post_comment_commits)
            if not response_commit:
                continue
            post_changed_paths = _post_comment_changed_paths([response_commit])
            context_candidates = _comment_context_candidates(path, post_changed_paths, base_existing_paths)
            if not context_candidates or _query_mentions_any_context(body, context_candidates):
                continue
            must_context_files = [
                {"path": candidate, "evidence": _context_evidence(candidate, path, body)}
                for candidate in context_candidates[:MAX_COMMENT_CONTEXT_GOLD_FILES]
            ]
            must_context_files = [item for item in must_context_files if _has_required_context_evidence(item["evidence"])]
            if not must_context_files:
                continue
            query = {
                "review_comment": body,
                "given_file": path,
                "path": path,
                "line": comment.get("line") or comment.get("original_line"),
                "diff_hunk_context": sanitize_diff_hunk(comment.get("diff_hunk")),
                "pr_title": truncate_text(pr.get("title"), 300),
            }
            query = _redact_paths_from_query(query, [item["path"] for item in must_context_files])
            sample = _sample(
                repo=repo,
                pr=pr,
                task_type="comment2context",
                seed=("comment_v2", pr_number, comment.get("id")),
                query=query,
                root_files=[item["path"] for item in must_context_files],
                tests=tests,
                supporting=[
                    candidate
                    for candidate in post_changed_paths
                    if candidate != path and candidate not in [item["path"] for item in must_context_files]
                ][:8],
                distractors=[candidate for candidate in changed_paths if candidate != path][:8],
                evidence={
                    "review_comment_id": comment.get("id"),
                    "comment_commit_id": comment.get("commit_id"),
                    "comment_created_at": comment.get("created_at"),
                    "response_commit": response_commit.get("sha"),
                    "post_comment_commits": [commit["sha"] for commit in post_comment_commits[:8]],
                    "gold_definition": "must_context_files_excluding_given_file",
                },
                confidence="weak",
            )
            sample["version"] = 2
            sample["gold"]["given_files"] = [path]
            sample["gold"]["must_context_files"] = must_context_files
            yield sample


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


def _post_comment_commits(
    commits_record: dict[str, Any] | None,
    comment_created_at: str | None,
    details_record: dict[str, Any] | None = None,
    max_age: timedelta | None = None,
) -> list[dict[str, Any]]:
    comment_time = _parse_github_time(comment_created_at)
    if not comment_time:
        return []
    detail_by_sha = {
        detail.get("sha"): detail
        for detail in (details_record or {}).get("data", [])
        if detail.get("sha") and not detail.get("error")
    }
    commit_rows = (details_record or {}).get("data") or (commits_record or {}).get("data", [])
    commits: list[tuple[datetime, dict[str, Any]]] = []
    for commit in commit_rows:
        sha = commit.get("sha")
        detail = detail_by_sha.get(sha, commit)
        commit_data = commit.get("commit") or {}
        commit_time = _parse_github_time(
            ((commit_data.get("committer") or {}).get("date"))
            or ((commit_data.get("author") or {}).get("date"))
        )
        if sha and commit_time and commit_time > comment_time:
            if max_age is not None and commit_time - comment_time > max_age:
                continue
            commits.append((commit_time, detail))
    commits.sort(key=lambda item: item[0])
    return [detail for _time, detail in commits if detail.get("files")]


def _first_valid_comment_response_commit(post_comment_commits: list[dict[str, Any]]) -> dict[str, Any] | None:
    for commit in post_comment_commits:
        changed_paths = _post_comment_changed_paths([commit])
        if not changed_paths or len(changed_paths) > MAX_COMMENT_CONTEXT_COMMIT_FILES:
            continue
        if _low_value_change_majority(changed_paths):
            continue
        context_paths = [path for path in changed_paths if _is_allowed_context_gold_path(path)]
        if not context_paths:
            continue
        return commit
    return None


def _post_comment_changed_paths(post_comment_commits: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for commit in post_comment_commits:
        for file in commit.get("files", []):
            path = file.get("filename")
            status = file.get("status")
            if path and status not in {"added", "removed", "renamed"}:
                paths.append(path)
    return _dedupe(paths)


def _comment_context_candidates(
    given_path: str,
    post_changed_paths: list[str],
    base_existing_paths: set[str] | None = None,
) -> list[str]:
    base_existing_paths = base_existing_paths or set(post_changed_paths)
    implementation, tests, _ignored = split_changed_files(post_changed_paths)
    if is_test_file(given_path):
        primary = implementation
        secondary = tests
    else:
        primary = tests
        secondary = implementation
    return [
        path
        for path in _dedupe(primary + secondary)
        if path != given_path and path in base_existing_paths and _is_allowed_context_gold_path(path)
    ]


def _context_evidence(candidate_path: str, given_path: str, review_body: str) -> list[str]:
    evidence = ["modified_after_review_comment"]
    lowered_body = review_body.lower()
    if candidate_path.lower() in lowered_body:
        evidence.append("review_mentions_context_path")
    if Path(candidate_path).name.lower() in lowered_body:
        evidence.append("review_mentions_context_basename")
    shared_tokens = _path_stem_tokens(candidate_path) & _path_stem_tokens(given_path)
    if shared_tokens:
        evidence.append("symbol_or_path_overlap")
    if _same_module_context(candidate_path, given_path):
        evidence.append("same_module_context")
    if _looks_like_test_context(candidate_path, given_path) and (_asks_for_tests(review_body) or shared_tokens):
        evidence.append("behavior_test_for_reviewed_change")
    elif _looks_like_implementation_context(candidate_path, given_path) and (
        shared_tokens or _same_module_context(candidate_path, given_path) or CONTEXT_BEHAVIOR_RE.search(review_body)
    ):
        evidence.append("implementation_context_for_reviewed_change")
    if _asks_for_tests(review_body) and is_test_file(candidate_path):
        evidence.append("review_requests_tests")
    if CONTEXT_BEHAVIOR_RE.search(review_body) and not _is_low_value_review_body(review_body):
        evidence.append("explicit_behavior_or_api_dependency")
    return evidence


def _looks_like_test_context(candidate_path: str, given_path: str) -> bool:
    _implementation, candidate_tests, _ignored = split_changed_files([candidate_path])
    implementation, _tests, _ignored_given = split_changed_files([given_path])
    return bool(candidate_tests and implementation)


def _looks_like_implementation_context(candidate_path: str, given_path: str) -> bool:
    implementation, _tests, _ignored = split_changed_files([candidate_path])
    _given_implementation, given_tests, _ignored_given = split_changed_files([given_path])
    return bool(implementation and given_tests)


def _has_required_context_evidence(evidence: list[str]) -> bool:
    return "modified_after_review_comment" in evidence and bool(
        set(evidence)
        & {
            "review_requests_tests",
            "explicit_behavior_or_api_dependency",
            "behavior_test_for_reviewed_change",
            "implementation_context_for_reviewed_change",
            "review_mentions_context_path",
            "review_mentions_context_basename",
        }
    )


def _skip_comment2context_body(body: str) -> bool:
    normalized = body.strip()
    if not normalized:
        return True
    if _word_count(normalized) < 6:
        return True
    if POST_HOC_REVIEW_RE.search(normalized):
        return True
    if normalized.startswith(">") and POST_HOC_REPLY_RE.search(normalized):
        return True
    if POST_HOC_REPLY_RE.search(normalized) and not re.search(r"\b(can we|could|should|please|why|does this|do we)\b", normalized, re.I):
        return True
    if _is_low_value_review_body(normalized):
        return True
    if contains_raw_patch_marker(normalized) or _has_solution_code_block(normalized):
        return True
    return not ACTIONABLE_REVIEW_RE.search(normalized)


def _skip_comment2context_path(path: str, body: str) -> bool:
    normalized = path.replace("\\", "/")
    lowered_parts = {part.lower() for part in normalized.split("/")}
    basename = Path(normalized).name.lower()
    suffix = Path(normalized).suffix.lower()
    if (lowered_parts & LOW_VALUE_PATH_PARTS or basename in LOW_VALUE_FILENAMES) and not _asks_for_tests(body):
        return True
    if (normalized.startswith(".github/") or suffix in {".yml", ".yaml", ".toml", ".ini", ".cfg"}) and not _asks_for_tests(body):
        return True
    if _is_top_level_config_path(normalized) and not _asks_for_tests(body):
        return True
    return False


def _is_top_level_config_path(path: str) -> bool:
    basename = Path(path).name.lower()
    return "/" not in path and (
        basename.startswith(("eslint.", "prettier.", "tsconfig."))
        or ".config." in basename
        or basename in {"ruff.toml", "pyproject.toml", "package.json"}
    )


def _asks_for_tests(body: str) -> bool:
    return bool(re.search(r"\b(test|tests|coverage|regression)\b", body, re.IGNORECASE))


def _has_large_solution_code_block(body: str, max_lines: int = 10, max_chars: int = 500) -> bool:
    for match in re.finditer(r"```[^\n]*\n(?P<code>.*?)```", body, re.DOTALL):
        code = match.group("code")
        if len(code) >= max_chars or len(code.splitlines()) >= max_lines:
            return True
    return False


def _has_solution_code_block(body: str) -> bool:
    if _has_large_solution_code_block(body):
        return True
    for match in re.finditer(r"```[^\n]*\n(?P<code>.*?)```", body, re.DOTALL):
        code = match.group("code").strip()
        if not code:
            continue
        code_lines = [line.strip() for line in code.splitlines() if line.strip()]
        if len(code_lines) >= 2:
            return True
        if code_lines and re.search(
            r"\b(return|if|for|while|def|class|const|let|var|import|from|func|fn|public|private|try|except|catch)\b|[=;{}()]",
            code_lines[0],
            re.IGNORECASE,
        ):
            return True
    return False


def _query_mentions_any_context(body: str, context_candidates: list[str]) -> bool:
    lowered_body = body.lower()
    for path in context_candidates:
        lowered_path = path.lower()
        basename = Path(path).name.lower()
        if lowered_path in lowered_body or (basename and basename in lowered_body):
            return True
    return False


def _path_stem_tokens(path: str) -> set[str]:
    stem = Path(path).stem.lower()
    return {
        token
        for token in re.split(r"[^a-z0-9]+", stem)
        if len(token) >= 3 and token not in GENERIC_PATH_TOKENS
    }


def _skip_comment2context_comment(comment: dict[str, Any]) -> bool:
    return bool(comment.get("in_reply_to_id"))


def _is_low_value_review_body(body: str) -> bool:
    normalized = re.sub(r"\s+", " ", body).strip().lower()
    if not normalized:
        return True
    if CONTEXT_BEHAVIOR_RE.search(normalized) and "configuration property metadata" in normalized:
        return False
    if re.search(r"\b(add|update|write|cover).{0,40}\btest", normalized):
        return False
    if re.search(r"\b(what'?s|what is) the difference\b", normalized):
        return True
    if re.search(r"\bwhy (is|are|do|does|we)\b", normalized) and not CONTEXT_BEHAVIOR_RE.search(normalized):
        return True
    return bool(LOW_VALUE_REVIEW_RE.search(normalized))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def _is_allowed_context_gold_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not (is_source_file(normalized) or is_test_file(normalized)):
        return False
    if _is_low_value_data_path(normalized):
        return False
    return True


def _is_low_value_data_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    lowered_parts = {part.lower() for part in normalized.split("/")}
    basename = Path(normalized).name.lower()
    suffix = Path(normalized).suffix.lower()
    if lowered_parts & LOW_VALUE_PATH_PARTS:
        return True
    if basename in LOW_VALUE_FILENAMES:
        return True
    if normalized.startswith(".github/"):
        return True
    if suffix in {".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg"}:
        return True
    if _is_top_level_config_path(normalized):
        return True
    return False


def _low_value_change_majority(paths: list[str]) -> bool:
    if not paths:
        return True
    low_value = sum(1 for path in paths if _is_low_value_data_path(path))
    return low_value > len(paths) / 2


def _same_module_context(candidate_path: str, given_path: str) -> bool:
    candidate_parts = [part for part in candidate_path.replace("\\", "/").split("/") if part]
    given_parts = [part for part in given_path.replace("\\", "/").split("/") if part]
    if not candidate_parts or not given_parts:
        return False
    shared_prefix = 0
    for left, right in zip(candidate_parts, given_parts):
        if left != right:
            break
        shared_prefix += 1
    if shared_prefix >= 2:
        return True
    return bool(_path_stem_tokens(candidate_path) & _path_stem_tokens(given_path))


def _parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _redact_paths_from_query(query: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    redacted = query
    for path in sorted(_dedupe(paths), key=len, reverse=True):
        redacted = _replace_string_value(redacted, path, "[context_file]")
    return redacted


def _replace_string_value(value: Any, needle: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(needle, replacement)
    if isinstance(value, list):
        return [_replace_string_value(item, needle, replacement) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string_value(item, needle, replacement) for key, item in value.items()}
    return value


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


def _changed_statuses(files_record: dict[str, Any] | None) -> dict[str, str | None]:
    if not files_record:
        return {}
    return {
        file["filename"]: file.get("status")
        for file in files_record.get("data", [])
        if file.get("filename")
    }


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
