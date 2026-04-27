from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .github_api import GitHubAPI
from .io import append_jsonl, ensure_parent, read_jsonl, repo_slug, stable_id, truncate_text, utc_now

DEFAULT_FAILURE_CONCLUSIONS = {"failure", "timed_out", "action_required"}
ERROR_HINT_RE = re.compile(
    r"(traceback|error:|failed|failure|panic|exception|assertion|expected|received|FAIL|FAILED|Caused by|stack backtrace)",
    re.IGNORECASE,
)


def crawl_job_logs(
    api: GitHubAPI,
    raw_dir: Path,
    repo: str,
    max_jobs: int = 25,
    max_bytes: int = 2_000_000,
    conclusions: set[str] | None = None,
) -> dict[str, Any]:
    conclusions = conclusions or DEFAULT_FAILURE_CONCLUSIONS
    owner, name = repo.split("/", 1)
    repo_dir = raw_dir / repo_slug(repo)
    records = _candidate_jobs(repo_dir, conclusions)
    downloaded = 0
    skipped_existing = 0
    errors = 0
    metadata_records: list[dict[str, Any]] = []
    for record in records:
        if downloaded >= max_jobs:
            break
        job_id = record["job_id"]
        log_relpath = Path("job_logs") / f"{job_id}.txt"
        log_path = repo_dir / log_relpath
        if log_path.exists():
            skipped_existing += 1
            metadata_records.append(_metadata_from_existing(record, log_relpath, log_path, max_bytes))
            downloaded += 1
            continue
        try:
            response = api.get_bytes(
                f"/repos/{owner}/{name}/actions/jobs/{job_id}/logs",
                accept="application/vnd.github+json",
                max_bytes=max_bytes,
            )
            truncated = len(response.body) > max_bytes
            body = response.body[:max_bytes]
            text = body.decode("utf-8", errors="replace")
            ensure_parent(log_path)
            log_path.write_text(text, encoding="utf-8")
            metadata_records.append(_metadata_from_text(record, log_relpath, text, truncated, response.headers))
            downloaded += 1
        except RuntimeError as error:
            errors += 1
            metadata_records.append(
                {
                    "type": "job_log_error",
                    **record,
                    "fetched_at": utc_now(),
                    "error": str(error),
                }
            )
    if metadata_records:
        append_jsonl(repo_dir / "job_logs.jsonl", metadata_records)
    return {
        "repo": repo,
        "candidates": len(records),
        "downloaded_or_existing": downloaded,
        "skipped_existing": skipped_existing,
        "errors": errors,
        "metadata_path": str(repo_dir / "job_logs.jsonl"),
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
    return candidates


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
