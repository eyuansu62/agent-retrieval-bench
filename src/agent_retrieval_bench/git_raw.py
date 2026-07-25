from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .crawler import _pr_record_key, _pull_request_key, _upsert_jsonl
from .filters import split_changed_files
from .io import ensure_parent, read_jsonl, repo_slug, utc_now, write_json


def backfill_git_raw(
    raw_dir: Path,
    repo: str,
    repos_dir: Path,
    limit_prs: int | None = None,
    repo_url_template: str = "https://github.com/{repo}.git",
    timeout_seconds: int = 600,
    infer_missing_base: bool = False,
    max_inferred_commits: int = 80,
) -> dict[str, Any]:
    repo_raw = raw_dir / repo_slug(repo)
    prs = latest_pull_requests(repo_raw / "pull_requests.jsonl")
    worktree = repos_dir / repo_slug(repo)
    remote_url = repo_url_template.format(repo=repo)
    setup = ensure_git_repo(worktree, remote_url, timeout_seconds=timeout_seconds)
    if setup["returncode"] != 0:
        summary = {
            "repo": repo,
            "raw_dir": str(repo_raw),
            "repos_dir": str(repos_dir),
            "status": "setup_failed",
            "setup": setup,
            "processed": 0,
            "errors": 1,
        }
        write_json(repo_raw / "git_backfill_summary.json", summary)
        return summary

    existing = raw_coverage(repo_raw)
    processed = 0
    skipped = Counter()
    errors = Counter()
    written = Counter()
    for pr_number, record in sorted(prs.items(), reverse=True):
        if limit_prs is not None and processed >= limit_prs:
            break
        if pr_number in existing["pull_files"] and pr_number in existing["pull_commits"] and pr_number in existing["commit_details"]:
            skipped["already_complete"] += 1
            continue
        pr = record.get("data") or {}
        base = str(pr.get("baseRefOid") or "")
        head = str(pr.get("headRefOid") or (pr.get("mergeCommit") or {}).get("oid") or "")
        if not head:
            skipped["missing_base_or_head"] += 1
            continue
        fetch_head = fetch_ref(worktree, "origin", head, fallback_ref=f"pull/{pr_number}/head", timeout_seconds=timeout_seconds)
        if fetch_head.returncode != 0:
            errors["fetch_failed"] += 1
            continue
        inferred_base = False
        if not base and infer_missing_base:
            base = infer_base_ref(worktree, pr, head, timeout_seconds=timeout_seconds)
            if base:
                inferred_base = True
                updated_pr = {**pr, "baseRefOid": base}
                record = {**record, "data": updated_pr}
                _upsert_jsonl(repo_raw / "pull_requests.jsonl", [record], _pull_request_key)
        if not base:
            skipped["missing_base_or_head"] += 1
            continue
        fetch_base = fetch_ref(worktree, "origin", base, timeout_seconds=timeout_seconds)
        if fetch_base.returncode != 0:
            errors["fetch_failed"] += 1
            continue

        files = changed_files(worktree, base, head, timeout_seconds=timeout_seconds)
        if not files:
            skipped["no_changed_files"] += 1
            continue
        commits = commit_records(worktree, base, head, timeout_seconds=timeout_seconds)
        if inferred_base and len(commits) > max_inferred_commits:
            skipped["inferred_base_too_many_commits"] += 1
            continue
        details = commit_details(worktree, commits, timeout_seconds=timeout_seconds)

        common = {"repo": repo, "pr_number": pr_number, "fetched_at": utc_now()}
        implementation, tests, ignored = split_changed_files([file["filename"] for file in files])
        if pr_number not in existing["pull_files"]:
            _upsert_jsonl(repo_raw / "pull_files.jsonl", [{**common, "type": "pull_files", "data": files}], _pr_record_key)
            existing["pull_files"].add(pr_number)
            written["pull_files"] += 1
        if pr_number not in existing["pull_file_summary"]:
            _upsert_jsonl(
                repo_raw / "pull_file_summary.jsonl",
                [{**common, "type": "pull_file_summary", "implementation": implementation, "tests": tests, "ignored": ignored}],
                _pr_record_key,
            )
            existing["pull_file_summary"].add(pr_number)
            written["pull_file_summary"] += 1
        if pr_number not in existing["pull_commits"]:
            _upsert_jsonl(repo_raw / "pull_commits.jsonl", [{**common, "type": "pull_commits", "data": commits}], _pr_record_key)
            existing["pull_commits"].add(pr_number)
            written["pull_commits"] += 1
        if pr_number not in existing["commit_details"]:
            _upsert_jsonl(repo_raw / "commit_details.jsonl", [{**common, "type": "commit_details", "data": details}], _pr_record_key)
            existing["commit_details"].add(pr_number)
            written["commit_details"] += 1
        processed += 1

    summary = {
        "generated_at": utc_now(),
        "repo": repo,
        "raw_dir": str(repo_raw),
        "repos_dir": str(repos_dir),
        "status": "ok",
        "processed": processed,
        "pull_requests": len(prs),
        "skipped": dict(sorted(skipped.items())),
        "errors": dict(sorted(errors.items())),
        "written": dict(sorted(written.items())),
        "outputs": {
            "pull_files": str(repo_raw / "pull_files.jsonl"),
            "pull_file_summary": str(repo_raw / "pull_file_summary.jsonl"),
            "pull_commits": str(repo_raw / "pull_commits.jsonl"),
            "commit_details": str(repo_raw / "commit_details.jsonl"),
        },
    }
    write_json(repo_raw / "git_backfill_summary.json", summary)
    return summary


def infer_base_ref(worktree: Path, pr: dict[str, Any], head: str, timeout_seconds: int) -> str:
    base_ref = str(pr.get("baseRefName") or "")
    if not base_ref:
        return ""
    remote_ref = f"refs/remotes/origin/{base_ref}"
    fetched = run_git(["fetch", "--depth", "500", "origin", f"refs/heads/{base_ref}:{remote_ref}"], worktree, timeout_seconds=timeout_seconds)
    if fetched.returncode != 0:
        fetched = run_git(["fetch", "origin", f"refs/heads/{base_ref}:{remote_ref}"], worktree, timeout_seconds=timeout_seconds)
    if fetched.returncode != 0:
        return ""
    result = run_git(["merge-base", head, remote_ref], worktree, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def latest_pull_requests(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        number = (record.get("data") or {}).get("number")
        if number is not None:
            records[int(number)] = record
    return records


def raw_coverage(repo_raw: Path) -> dict[str, set[int]]:
    return {
        "pull_files": prs_with_record(repo_raw / "pull_files.jsonl", "pull_files"),
        "pull_file_summary": prs_with_record(repo_raw / "pull_file_summary.jsonl", "pull_file_summary"),
        "pull_commits": prs_with_record(repo_raw / "pull_commits.jsonl", "pull_commits"),
        "commit_details": prs_with_record(repo_raw / "commit_details.jsonl", "commit_details"),
    }


def prs_with_record(path: Path, record_type: str) -> set[int]:
    return {
        int(record["pr_number"])
        for record in read_jsonl(path)
        if record.get("pr_number") is not None and record.get("type") == record_type
    }


def ensure_git_repo(worktree: Path, remote_url: str, timeout_seconds: int) -> dict[str, Any]:
    if worktree.exists() and not (worktree / ".git").exists():
        shutil.rmtree(worktree)
    if worktree.exists():
        remote = run_git(["remote", "set-url", "origin", remote_url], worktree, timeout_seconds=timeout_seconds)
        return process_record(remote)
    ensure_parent(worktree)
    clone = run_process(["git", "clone", "--no-checkout", "--filter=blob:none", remote_url, str(worktree)], timeout_seconds=timeout_seconds)
    if clone.returncode != 0 and worktree.exists():
        shutil.rmtree(worktree)
        clone = run_process(["git", "clone", "--no-checkout", remote_url, str(worktree)], timeout_seconds=timeout_seconds)
    return process_record(clone)


def fetch_ref(worktree: Path, remote: str, ref: str, fallback_ref: str | None = None, timeout_seconds: int = 600) -> subprocess.CompletedProcess[str]:
    result = run_git(["fetch", "--depth", "50", remote, ref], worktree, timeout_seconds=timeout_seconds)
    if result.returncode == 0:
        return result
    if fallback_ref:
        result = run_git(["fetch", "--depth", "50", remote, fallback_ref], worktree, timeout_seconds=timeout_seconds)
        if result.returncode == 0:
            return result
    return run_git(["fetch", remote, ref], worktree, timeout_seconds=timeout_seconds)


def changed_files(worktree: Path, base: str, head: str, timeout_seconds: int) -> list[dict[str, Any]]:
    status = run_git(["diff", "--name-status", "--find-renames", base, head], worktree, timeout_seconds=timeout_seconds)
    if status.returncode != 0:
        return []
    files: list[dict[str, Any]] = []
    for line in status.stdout.splitlines():
        parsed = parse_name_status(line)
        if not parsed:
            continue
        filename, file_status, previous_filename = parsed
        additions, deletions = file_numstat(worktree, base, head, filename, timeout_seconds=timeout_seconds)
        record: dict[str, Any] = {
            "filename": filename,
            "status": file_status,
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "patch": file_patch(worktree, base, head, filename, timeout_seconds=timeout_seconds),
        }
        if previous_filename:
            record["previous_filename"] = previous_filename
        files.append(record)
    return files


def parse_name_status(line: str) -> tuple[str, str, str | None] | None:
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    code = parts[0]
    if code.startswith("R") and len(parts) >= 3:
        return parts[2], "renamed", parts[1]
    status_map = {"A": "added", "D": "removed", "M": "modified", "C": "modified", "T": "modified"}
    return parts[1], status_map.get(code[:1], "modified"), None


def file_numstat(worktree: Path, base: str, head: str, filename: str, timeout_seconds: int) -> tuple[int, int]:
    result = run_git(["diff", "--numstat", base, head, "--", filename], worktree, timeout_seconds=timeout_seconds)
    if result.returncode != 0 or not result.stdout.strip():
        return 0, 0
    parts = result.stdout.splitlines()[0].split("\t")
    if len(parts) < 2:
        return 0, 0
    return parse_numstat(parts[0]), parse_numstat(parts[1])


def parse_numstat(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def file_patch(worktree: Path, base: str, head: str, filename: str, timeout_seconds: int) -> str:
    result = run_git(["diff", "--unified=3", "--no-ext-diff", base, head, "--", filename], worktree, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        return ""
    return extract_hunks(result.stdout)


def extract_hunks(diff_text: str) -> str:
    lines = []
    in_hunk = False
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            in_hunk = True
        if in_hunk:
            lines.append(line)
    return "\n".join(lines)


def commit_records(worktree: Path, base: str, head: str, timeout_seconds: int) -> list[dict[str, Any]]:
    result = run_git(["log", "--reverse", "--format=%H%x00%cI%x00%s", f"{base}..{head}"], worktree, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        return []
    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("\x00", 2)
        if len(parts) != 3:
            continue
        sha, date, subject = parts
        commits.append({"sha": sha, "commit": {"committer": {"date": date}, "message": subject}})
    return commits


def commit_details(worktree: Path, commits: Iterable[dict[str, Any]], timeout_seconds: int) -> list[dict[str, Any]]:
    details = []
    for commit in commits:
        sha = str(commit.get("sha") or "")
        if not sha:
            continue
        status = run_git(["diff-tree", "--no-commit-id", "--name-status", "-r", sha], worktree, timeout_seconds=timeout_seconds)
        files = []
        if status.returncode == 0:
            for line in status.stdout.splitlines():
                parsed = parse_name_status(line)
                if parsed:
                    filename, file_status, _previous = parsed
                    files.append({"filename": filename, "status": file_status, "additions": 0, "deletions": 0, "changes": 0})
        details.append({"sha": sha, "commit": commit.get("commit") or {}, "files": files})
    return details


def run_git(args: list[str], worktree: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return run_process(["git", *args], cwd=worktree, timeout_seconds=timeout_seconds)


def run_process(args: list[str], cwd: Path | None = None, timeout_seconds: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(args, 124, text_or_empty(error.stdout), text_or_empty(error.stderr) or f"Timed out after {timeout_seconds}s")


def process_record(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "args": process.args,
        "returncode": process.returncode,
        "stdout": text_or_empty(process.stdout),
        "stderr": text_or_empty(process.stderr),
    }


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
