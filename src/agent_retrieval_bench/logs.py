from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .filters import is_ignored_check_signal
from .github_api import GitHubAPI
from .io import ensure_parent, read_jsonl, repo_slug, truncate_text, utc_now

DEFAULT_FAILURE_CONCLUSIONS = {"failure", "timed_out", "action_required"}
CONCLUSION_PRIORITY = {"failure": 0, "timed_out": 1, "action_required": 2, "cancelled": 3}
ERROR_HINT_RE = re.compile(
    r"(traceback|error:|failed|failure|panic|exception|assertion|expected|received|FAIL|FAILED|Caused by|stack backtrace)",
    re.IGNORECASE,
)
TEST_JOB_RE = re.compile(r"\b(test|tests|pytest|unit|integration|e2e|vitest|junit|cargo test|go test)\b", re.IGNORECASE)


def crawl_job_logs(
    api: GitHubAPI,
    raw_dir: Path,
    repo: str,
    max_jobs: int | None = None,
    max_new_jobs: int = 25,
    max_bytes: int = 2_000_000,
    conclusions: set[str] | None = None,
) -> dict[str, Any]:
    conclusions = conclusions or DEFAULT_FAILURE_CONCLUSIONS
    owner, name = repo.split("/", 1)
    repo_dir = raw_dir / repo_slug(repo)
    metadata_path = repo_dir / "job_logs.jsonl"
    existing_records = dedupe_job_log_records(read_jsonl(metadata_path))
    records = _candidate_jobs(repo_dir, conclusions)
    considered = 0
    new_downloaded = 0
    skipped_existing = 0
    errors = 0
    metadata_records: list[dict[str, Any]] = []
    last_rate_limit: dict[str, str | None] = {}
    for record in records:
        if max_jobs is not None and considered >= max_jobs:
            break
        if new_downloaded >= max_new_jobs:
            break
        considered += 1
        job_id = record["job_id"]
        log_relpath = Path("job_logs") / f"{job_id}.txt"
        log_path = repo_dir / log_relpath
        if log_path.exists():
            skipped_existing += 1
            metadata_records.append(_metadata_from_existing(record, log_relpath, log_path, max_bytes))
            continue
        try:
            response = api.get_bytes(
                f"/repos/{owner}/{name}/actions/jobs/{job_id}/logs",
                accept="application/vnd.github+json",
                max_bytes=max_bytes,
            )
            last_rate_limit = rate_limit_from_headers(response.headers)
            truncated = len(response.body) > max_bytes
            body = response.body[:max_bytes]
            text = body.decode("utf-8", errors="replace")
            ensure_parent(log_path)
            log_path.write_text(text, encoding="utf-8")
            metadata_records.append(_metadata_from_text(record, log_relpath, text, truncated, response.headers))
            new_downloaded += 1
        except Exception as error:
            errors += 1
            metadata_records.append(
                {
                    "type": "job_log_error",
                    **record,
                    "fetched_at": utc_now(),
                    "error": str(error),
                }
            )
            new_downloaded += 1
    if metadata_records:
        write_jsonl(metadata_path, dedupe_job_log_records([*existing_records, *metadata_records]))
    return {
        "repo": repo,
        "candidate_jobs": len(records),
        "candidates": len(records),
        "considered": considered,
        "new_downloaded": new_downloaded,
        "downloaded_or_existing": new_downloaded + skipped_existing,
        "skipped_existing": skipped_existing,
        "existing_skipped": skipped_existing,
        "errors": errors,
        "rate_limit": last_rate_limit,
        "metadata_path": str(metadata_path),
    }


def _candidate_jobs(repo_dir: Path, conclusions: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for check_record in read_jsonl(repo_dir / "check_runs.jsonl"):
        for run in check_record.get("data") or []:
            app = run.get("app") or {}
            job_id = run.get("id")
            if not job_id or job_id in seen:
                continue
            if app.get("slug") != "github-actions":
                continue
            if run.get("conclusion") not in conclusions:
                continue
            if is_ignored_check_signal(run.get("name"), json.dumps(run.get("output") or {}, ensure_ascii=False)):
                continue
            seen.add(job_id)
            candidates.append(
                {
                    "repo": check_record.get("repo"),
                    "pr_number": check_record.get("pr_number"),
                    "ref_type": check_record.get("ref_type"),
                    "sha": check_record.get("sha"),
                    "job_id": job_id,
                    "check_name": run.get("name"),
                    "conclusion": run.get("conclusion"),
                    "html_url": run.get("html_url"),
                    "details_url": run.get("details_url"),
                }
            )
    return sorted(candidates, key=job_sort_key)


def job_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    name = str(record.get("check_name") or "")
    return (
        CONCLUSION_PRIORITY.get(str(record.get("conclusion")), 99),
        0 if TEST_JOB_RE.search(name) else 1,
        str(record.get("repo") or ""),
        int(record.get("pr_number") or 0) * -1,
        int(record.get("job_id") or 0) * -1,
    )


def dedupe_job_log_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_job: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for record in records:
        job_id = record.get("job_id")
        if job_id is None:
            continue
        job_id = int(job_id)
        if job_id not in by_job:
            order.append(job_id)
        by_job[job_id] = preferred_job_record(by_job.get(job_id), record)
    return [by_job[job_id] for job_id in order]


def preferred_job_record(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return new
    if existing.get("type") == "job_log_error" and new.get("type") == "job_log":
        return new
    if existing.get("type") == "job_log" and new.get("type") == "job_log_error":
        return existing
    return new


def _metadata_from_existing(record: dict[str, Any], log_relpath: Path, log_path: Path, max_bytes: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return _metadata_from_text(record, log_relpath, text[:max_bytes], len(text.encode("utf-8")) > max_bytes, {})


def _metadata_from_text(
    record: dict[str, Any],
    log_relpath: Path,
    text: str,
    truncated: bool,
    headers: dict[str, str],
) -> dict[str, Any]:
    return {
        "type": "job_log",
        **record,
        "fetched_at": utc_now(),
        "log_path": str(log_relpath),
        "bytes": len(text.encode("utf-8")),
        "truncated": truncated,
        "content_type": headers.get("content-type"),
        "excerpt": _failure_excerpt(text),
    }


def rate_limit_from_headers(headers: dict[str, str]) -> dict[str, str | None]:
    return {
        "limit": headers.get("x-ratelimit-limit"),
        "remaining": headers.get("x-ratelimit-remaining"),
        "reset": headers.get("x-ratelimit-reset"),
        "used": headers.get("x-ratelimit-used"),
        "resource": headers.get("x-ratelimit-resource"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _failure_excerpt(text: str, max_lines: int = 80, window: int = 8) -> str:
    lines = text.replace("\r\n", "\n").splitlines()
    hit_indexes = [index for index, line in enumerate(lines) if ERROR_HINT_RE.search(line)]
    if not hit_indexes:
        return truncate_text("\n".join(lines[-max_lines:]), 6000)
    selected: list[str] = []
    seen: set[int] = set()
    for hit in hit_indexes[:8]:
        start = max(0, hit - window)
        end = min(len(lines), hit + window + 1)
        for index in range(start, end):
            if index not in seen:
                selected.append(lines[index])
                seen.add(index)
        if len(selected) >= max_lines:
            break
    return truncate_text("\n".join(selected[:max_lines]), 6000)
