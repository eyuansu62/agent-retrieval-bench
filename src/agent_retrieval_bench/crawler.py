from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .filters import is_ignored_check_signal, should_skip_pr, split_changed_files
from .github_api import GitHubAPI
from .io import append_jsonl, ensure_parent, read_json, read_jsonl, repo_slug, utc_now, write_json

MERGED_PR_QUERY = """
query MergedPullRequests($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    defaultBranchRef { name target { oid } }
    licenseInfo { spdxId }
    stargazerCount
    primaryLanguage { name }
    pullRequests(states: MERGED, first: $pageSize, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        url
        createdAt
        updatedAt
        mergedAt
        baseRefName
        headRefName
        baseRefOid
        headRefOid
        additions
        deletions
        changedFiles
        authorAssociation
        mergeCommit { oid }
        labels(first: 30) { nodes { name } }
        reviews(first: 1) { totalCount }
        comments(first: 1) { totalCount }
        closingIssuesReferences(first: 10) { nodes { number title url } }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


def fetch_repo_manifest(api: GitHubAPI, repo: str, configured_language: str | None = None, tasks: list[str] | None = None) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    response = api.get(f"/repos/{owner}/{name}")
    data = response.body
    return {
        "repo": repo,
        "default_branch": data.get("default_branch"),
        "license": (data.get("license") or {}).get("spdx_id"),
        "stars": data.get("stargazers_count"),
        "language": configured_language or data.get("language"),
        "tasks": tasks or [],
        "archived": data.get("archived"),
        "has_issues": data.get("has_issues"),
        "size_kb": data.get("size"),
        "fetched_at": utc_now(),
    }


def write_manifest(api: GitHubAPI, targets: list[dict[str, Any]], output: Path) -> int:
    records = [
        fetch_repo_manifest(api, target["repo"], target.get("language"), target.get("tasks", []))
        for target in targets
    ]
    return append_jsonl(output, records)


def crawl_repo(
    api: GitHubAPI,
    repo: str,
    out_dir: Path,
    limit_prs: int = 20,
    page_size: int = 25,
    max_changed_files: int = 20,
    include_checks: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    slug = repo_slug(repo)
    repo_dir = out_dir / slug
    state_path = repo_dir / "crawl_state.json"
    state = read_json(state_path, {}) or {}
    fetched_at = utc_now()
    if api.authenticated:
        fetched = _fetch_merged_prs_graphql(api, owner, name, state.get("graphql_cursor"), limit_prs, page_size, max_changed_files)
    else:
        fetched = _fetch_merged_prs_rest(api, owner, name, limit_prs, page_size, max_changed_files)
    accepted = fetched["accepted"]
    skipped = fetched["skipped"]
    cursor = fetched.get("cursor")

    summary = {
        "repo": repo,
        "source": fetched["source"],
        "seen_prs": fetched["seen"],
        "accepted_prs": len(accepted),
        "skipped_prs": skipped,
        "next_cursor": cursor,
        "rate_limit": fetched.get("rate_limit"),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    append_jsonl(
        repo_dir / "pull_requests.jsonl",
        [{"type": "pull_request", "repo": repo, "fetched_at": fetched_at, "data": pr} for pr in accepted],
    )

    for pr in accepted:
        pr_number = int(pr["number"])
        files = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/files")
        changed_paths = [file.get("filename", "") for file in files if file.get("filename")]
        if should_skip_pr(changed_paths, max_changed_files):
            skipped += 1
            continue
        implementation, tests, ignored = split_changed_files(changed_paths)
        common = {"repo": repo, "pr_number": pr_number, "fetched_at": utc_now()}
        append_jsonl(repo_dir / "pull_files.jsonl", [{**common, "type": "pull_files", "data": files}])
        append_jsonl(
            repo_dir / "pull_file_summary.jsonl",
            [{**common, "type": "pull_file_summary", "implementation": implementation, "tests": tests, "ignored": ignored}],
        )
        comments = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/comments")
        append_jsonl(repo_dir / "review_comments.jsonl", [{**common, "type": "review_comments", "data": comments}])
        commits = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/commits")
        append_jsonl(repo_dir / "pull_commits.jsonl", [{**common, "type": "pull_commits", "data": commits}])
        commit_details = fetch_commit_details(api, repo, commits)
        append_jsonl(repo_dir / "commit_details.jsonl", [{**common, "type": "commit_details", "data": commit_details}])
        if include_checks:
            check_records = fetch_check_runs(api, repo, pr, pr_number)
            append_jsonl(repo_dir / "check_runs.jsonl", check_records)

    write_json(
        state_path,
        {
            "repo": repo,
            "graphql_cursor": cursor,
            "updated_at": utc_now(),
            "last_summary": summary,
        },
    )
    summary["raw_dir"] = str(repo_dir)
    summary["skipped_prs"] = skipped
    return summary


def crawl_pr_checks(
    api: GitHubAPI,
    repo: str,
    out_dir: Path,
    limit_prs: int = 300,
    page_size: int = 50,
    max_changed_files: int = 30,
    include_review_comments: bool = False,
    refresh_existing_checks: bool = False,
    repair_empty_state: bool = False,
    max_pages: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not api.authenticated:
        raise RuntimeError("crawl-pr-checks requires GITHUB_TOKEN or GH_TOKEN for GraphQL pagination.")
    owner, name = repo.split("/", 1)
    slug = repo_slug(repo)
    repo_dir = out_dir / slug
    state_path = repo_dir / "crawl_state.json"
    state = read_json(state_path, {}) or {}
    cursor = state.get("graphql_cursor")
    state_repaired = False
    if repair_empty_state and _state_has_empty_crawl(state) and read_jsonl(repo_dir / "pull_requests.jsonl"):
        cursor = None
        state_repaired = True

    index = _raw_coverage_index(repo_dir)
    fetched_at = utc_now()
    seen = 0
    accepted = 0
    skipped = 0
    existing_skipped = 0
    new_pull_files = 0
    new_pull_commits = 0
    new_commit_details = 0
    new_check_runs = 0
    new_review_comments = 0
    failed_github_actions_jobs = 0
    errors = 0
    pages = 0
    rate_limit: dict[str, Any] | None = None

    while accepted < limit_prs:
        if max_pages is not None and pages >= max_pages:
            break
        response = api.graphql(
            MERGED_PR_QUERY,
            {"owner": owner, "name": name, "cursor": cursor, "pageSize": min(page_size, 100)},
        )
        pages += 1
        payload = response.body
        repository = payload["data"]["repository"]
        page = repository["pullRequests"]
        rate_limit = payload["data"].get("rateLimit")
        for pr in page["nodes"]:
            seen += 1
            pr_number = int(pr["number"])
            if pr.get("changedFiles", 0) > max_changed_files:
                skipped += 1
                continue
            if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
                skipped += 1
                continue
            if _has_complete_pr_check_coverage(index, pr_number, include_review_comments) and not refresh_existing_checks:
                existing_skipped += 1
                continue
            files = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/files")
            changed_paths = [file.get("filename", "") for file in files if file.get("filename")]
            if should_skip_pr(changed_paths, max_changed_files):
                skipped += 1
                continue
            if dry_run:
                accepted += 1
                continue

            common = {"repo": repo, "pr_number": pr_number, "fetched_at": utc_now()}
            _upsert_jsonl(
                repo_dir / "pull_requests.jsonl",
                [{"type": "pull_request", "repo": repo, "fetched_at": fetched_at, "data": pr}],
                _pull_request_key,
            )
            if refresh_existing_checks or pr_number not in index["pull_files_prs"]:
                implementation, tests, ignored = split_changed_files(changed_paths)
                _upsert_jsonl(repo_dir / "pull_files.jsonl", [{**common, "type": "pull_files", "data": files}], _pr_record_key)
                _upsert_jsonl(
                    repo_dir / "pull_file_summary.jsonl",
                    [{**common, "type": "pull_file_summary", "implementation": implementation, "tests": tests, "ignored": ignored}],
                    _pr_record_key,
                )
                index["pull_files_prs"].add(pr_number)
                new_pull_files += 1

            if include_review_comments and (refresh_existing_checks or pr_number not in index["review_comments_prs"]):
                comments = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/comments")
                _upsert_jsonl(repo_dir / "review_comments.jsonl", [{**common, "type": "review_comments", "data": comments}], _pr_record_key)
                index["review_comments_prs"].add(pr_number)
                new_review_comments += 1

            commits = index["pull_commits_records"].get(pr_number, {}).get("data") or []
            if refresh_existing_checks or pr_number not in index["pull_commits_prs"]:
                commits = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/commits")
                _upsert_jsonl(repo_dir / "pull_commits.jsonl", [{**common, "type": "pull_commits", "data": commits}], _pr_record_key)
                index["pull_commits_prs"].add(pr_number)
                index["pull_commits_records"][pr_number] = {**common, "type": "pull_commits", "data": commits}
                new_pull_commits += 1

            if refresh_existing_checks or pr_number not in index["commit_details_prs"]:
                commit_details = fetch_commit_details(api, repo, commits)
                errors += sum(1 for detail in commit_details if detail.get("error"))
                _upsert_jsonl(repo_dir / "commit_details.jsonl", [{**common, "type": "commit_details", "data": commit_details}], _pr_record_key)
                index["commit_details_prs"].add(pr_number)
                new_commit_details += 1

            if refresh_existing_checks or pr_number not in index["check_runs_prs"]:
                check_records = fetch_check_runs(api, repo, pr, pr_number)
                errors += sum(1 for record in check_records if record.get("type") == "check_runs_error")
                failed_github_actions_jobs += _failed_github_actions_jobs(check_records)
                _upsert_jsonl(repo_dir / "check_runs.jsonl", check_records, _check_run_key)
                index["check_runs_prs"].add(pr_number)
                new_check_runs += len(check_records)

            accepted += 1
            if accepted >= limit_prs:
                break
        cursor = page["pageInfo"].get("endCursor")
        if not page["pageInfo"].get("hasNextPage") or not cursor:
            break

    summary = {
        "repo": repo,
        "source": "graphql",
        "seen_prs": seen,
        "pages": pages,
        "accepted_prs": accepted,
        "skipped_prs": skipped,
        "existing_skipped": existing_skipped,
        "new_pull_files": new_pull_files,
        "new_pull_commits": new_pull_commits,
        "new_commit_details": new_commit_details,
        "new_review_comments": new_review_comments,
        "new_check_runs": new_check_runs,
        "failed_github_actions_jobs": failed_github_actions_jobs,
        "errors": errors,
        "next_cursor": cursor,
        "rate_limit": rate_limit,
        "state_repaired": state_repaired,
        "dry_run": dry_run,
        "raw_dir": str(repo_dir),
    }
    if not dry_run:
        write_json(
            state_path,
            {
                **state,
                "repo": repo,
                "graphql_cursor": cursor,
                "updated_at": utc_now(),
                "last_pr_check_summary": summary,
            },
        )
    return summary


def _fetch_merged_prs_graphql(
    api: GitHubAPI,
    owner: str,
    name: str,
    cursor: str | None,
    limit_prs: int,
    page_size: int,
    max_changed_files: int,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    seen = 0
    skipped = 0
    rate_limit: dict[str, Any] | None = None
    while len(accepted) < limit_prs:
        response = api.graphql(
            MERGED_PR_QUERY,
            {"owner": owner, "name": name, "cursor": cursor, "pageSize": min(page_size, limit_prs)},
        )
        payload = response.body
        repository = payload["data"]["repository"]
        page = repository["pullRequests"]
        rate_limit = payload["data"].get("rateLimit")
        for pr in page["nodes"]:
            seen += 1
            if pr.get("changedFiles", 0) > max_changed_files:
                skipped += 1
                continue
            if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
                skipped += 1
                continue
            accepted.append(pr)
            if len(accepted) >= limit_prs:
                break
        cursor = page["pageInfo"].get("endCursor")
        if not page["pageInfo"].get("hasNextPage") or not cursor:
            break
    return {
        "source": "graphql",
        "accepted": accepted,
        "seen": seen,
        "skipped": skipped,
        "cursor": cursor,
        "rate_limit": rate_limit,
    }


def _fetch_merged_prs_rest(
    api: GitHubAPI,
    owner: str,
    name: str,
    limit_prs: int,
    page_size: int,
    max_changed_files: int,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    seen = 0
    skipped = 0
    page = 1
    while len(accepted) < limit_prs:
        response = api.get(
            f"/repos/{owner}/{name}/pulls",
            {"state": "closed", "sort": "updated", "direction": "desc", "per_page": min(page_size, 100), "page": page},
        )
        pulls = response.body
        if not pulls:
            break
        for pull in pulls:
            seen += 1
            if not pull.get("merged_at"):
                skipped += 1
                continue
            detail = api.get(f"/repos/{owner}/{name}/pulls/{pull['number']}").body
            if detail.get("changed_files", 0) > max_changed_files:
                skipped += 1
                continue
            pr = _rest_pull_to_pr(detail)
            if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
                skipped += 1
                continue
            accepted.append(pr)
            if len(accepted) >= limit_prs:
                break
        if len(pulls) < min(page_size, 100):
            break
        page += 1
    return {"source": "rest", "accepted": accepted, "seen": seen, "skipped": skipped, "cursor": None, "rate_limit": None}


def _rest_pull_to_pr(pull: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pull.get("number"),
        "title": pull.get("title"),
        "body": pull.get("body"),
        "url": pull.get("html_url"),
        "createdAt": pull.get("created_at"),
        "updatedAt": pull.get("updated_at"),
        "mergedAt": pull.get("merged_at"),
        "baseRefName": (pull.get("base") or {}).get("ref"),
        "headRefName": (pull.get("head") or {}).get("ref"),
        "baseRefOid": (pull.get("base") or {}).get("sha"),
        "headRefOid": (pull.get("head") or {}).get("sha"),
        "additions": pull.get("additions"),
        "deletions": pull.get("deletions"),
        "changedFiles": pull.get("changed_files"),
        "authorAssociation": pull.get("author_association"),
        "mergeCommit": {"oid": pull.get("merge_commit_sha")},
        "labels": {"nodes": [{"name": label.get("name")} for label in pull.get("labels", [])]},
        "reviews": {"totalCount": None},
        "comments": {"totalCount": pull.get("comments")},
        "closingIssuesReferences": {"nodes": []},
    }


def fetch_check_runs(api: GitHubAPI, repo: str, pr: dict[str, Any], pr_number: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    refs = {
        "head": pr.get("headRefOid"),
        "merge": (pr.get("mergeCommit") or {}).get("oid"),
    }
    records: list[dict[str, Any]] = []
    for ref_type, sha in refs.items():
        if not sha:
            continue
        try:
            response = api.get(
                f"/repos/{owner}/{name}/commits/{sha}/check-runs",
                {"per_page": 100},
                accept="application/vnd.github+json",
            )
            records.append(
                {
                    "type": "check_runs",
                    "repo": repo,
                    "pr_number": pr_number,
                    "ref_type": ref_type,
                    "sha": sha,
                    "fetched_at": utc_now(),
                    "data": response.body.get("check_runs", []) if isinstance(response.body, dict) else [],
                }
            )
        except RuntimeError as error:
            records.append(
                {
                    "type": "check_runs_error",
                    "repo": repo,
                    "pr_number": pr_number,
                    "ref_type": ref_type,
                    "sha": sha,
                    "fetched_at": utc_now(),
                    "error": str(error),
                }
            )
    return records


def fetch_commit_details(api: GitHubAPI, repo: str, commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    details: list[dict[str, Any]] = []
    for commit in commits:
        sha = commit.get("sha")
        if not sha:
            continue
        try:
            response = api.get(f"/repos/{owner}/{name}/commits/{sha}")
            data = response.body if isinstance(response.body, dict) else {}
            details.append(
                {
                    "sha": sha,
                    "commit": data.get("commit") or commit.get("commit") or {},
                    "files": [
                        {
                            "filename": file.get("filename"),
                            "status": file.get("status"),
                            "additions": file.get("additions"),
                            "deletions": file.get("deletions"),
                            "changes": file.get("changes"),
                        }
                        for file in data.get("files", [])
                        if file.get("filename")
                    ],
                }
            )
        except RuntimeError as error:
            details.append({"sha": sha, "error": str(error), "commit": commit.get("commit") or {}, "files": []})
    return details


def crawl_commit_details_for_raw(
    api: GitHubAPI,
    raw_dir: Path,
    repo: str,
    limit_prs: int | None = None,
    max_commits_per_pr: int | None = None,
) -> dict[str, Any]:
    slug = repo_slug(repo)
    repo_dir = raw_dir / slug
    pull_commits_path = repo_dir / "pull_commits.jsonl"
    commit_details_path = repo_dir / "commit_details.jsonl"
    existing_prs = {
        int(record["pr_number"])
        for record in read_jsonl(commit_details_path)
        if record.get("pr_number") is not None and record.get("type") == "commit_details"
    }
    latest_pull_commits: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(pull_commits_path):
        if record.get("pr_number") is not None:
            latest_pull_commits[int(record["pr_number"])] = record

    fetched = 0
    skipped_existing = 0
    skipped_empty = 0
    errors = 0
    for pr_number, record in sorted(latest_pull_commits.items(), reverse=True):
        if limit_prs is not None and fetched >= limit_prs:
            break
        if pr_number in existing_prs:
            skipped_existing += 1
            continue
        commits = list(record.get("data") or [])
        if max_commits_per_pr is not None:
            commits = commits[:max_commits_per_pr]
        if not commits:
            skipped_empty += 1
            continue
        details = fetch_commit_details(api, repo, commits)
        errors += sum(1 for detail in details if detail.get("error"))
        append_jsonl(
            commit_details_path,
            [
                {
                    "repo": repo,
                    "pr_number": pr_number,
                    "fetched_at": utc_now(),
                    "type": "commit_details",
                    "data": details,
                }
            ],
        )
        fetched += 1
    return {
        "repo": repo,
        "raw_dir": str(repo_dir),
        "fetched_prs": fetched,
        "skipped_existing": skipped_existing,
        "skipped_empty": skipped_empty,
        "errors": errors,
        "authenticated": api.authenticated,
    }


def _state_has_empty_crawl(state: dict[str, Any]) -> bool:
    summary = state.get("last_summary") or state.get("last_pr_check_summary") or {}
    return not state.get("graphql_cursor") and summary.get("accepted_prs") == 0


def _raw_coverage_index(repo_dir: Path) -> dict[str, Any]:
    pull_commits_records: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(repo_dir / "pull_commits.jsonl"):
        pr_number = record.get("pr_number")
        if pr_number is not None and record.get("type") == "pull_commits":
            pull_commits_records[int(pr_number)] = record
    return {
        "pull_files_prs": _prs_with_record(repo_dir / "pull_files.jsonl", "pull_files"),
        "pull_commits_prs": set(pull_commits_records),
        "pull_commits_records": pull_commits_records,
        "commit_details_prs": _prs_with_record(repo_dir / "commit_details.jsonl", "commit_details"),
        "check_runs_prs": _prs_with_record(repo_dir / "check_runs.jsonl", {"check_runs", "check_runs_error"}),
        "review_comments_prs": _prs_with_record(repo_dir / "review_comments.jsonl", "review_comments"),
    }


def _prs_with_record(path: Path, record_type: str | set[str]) -> set[int]:
    types = {record_type} if isinstance(record_type, str) else record_type
    return {
        int(record["pr_number"])
        for record in read_jsonl(path)
        if record.get("pr_number") is not None and record.get("type") in types
    }


def _has_complete_pr_check_coverage(index: dict[str, Any], pr_number: int, include_review_comments: bool) -> bool:
    required_sets = (
        index["pull_files_prs"],
        index["pull_commits_prs"],
        index["commit_details_prs"],
        index["check_runs_prs"],
    )
    if any(pr_number not in values for values in required_sets):
        return False
    return not include_review_comments or pr_number in index["review_comments_prs"]


def _failed_github_actions_jobs(check_records: list[dict[str, Any]]) -> int:
    failures = {"failure", "timed_out", "action_required"}
    count = 0
    for record in check_records:
        for run in record.get("data") or []:
            app = run.get("app") or {}
            if app.get("slug") != "github-actions":
                continue
            if run.get("conclusion") not in failures:
                continue
            if is_ignored_check_signal(run.get("name"), json.dumps(run.get("output") or {}, ensure_ascii=False)):
                continue
            count += 1
    return count


def _upsert_jsonl(path: Path, records: list[dict[str, Any]], key_fn) -> int:
    if not records:
        return 0
    existing = read_jsonl(path)
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for record in [*existing, *records]:
        key = key_fn(record)
        if key not in by_key:
            order.append(key)
        by_key[key] = _preferred_record(by_key.get(key), record)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for key in order:
            handle.write(json.dumps(by_key[key], ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(records)


def _pull_request_key(record: dict[str, Any]) -> tuple[Any, ...]:
    data = record.get("data") or {}
    return (record.get("type"), record.get("repo"), int(data.get("number") or 0))


def _pr_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("type"), record.get("repo"), int(record.get("pr_number") or 0))


def _check_run_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        "check_runs",
        record.get("repo"),
        int(record.get("pr_number") or 0),
        record.get("ref_type"),
        record.get("sha"),
    )


def _preferred_record(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return new
    if str(existing.get("type", "")).endswith("_error") and not str(new.get("type", "")).endswith("_error"):
        return new
    if not str(existing.get("type", "")).endswith("_error") and str(new.get("type", "")).endswith("_error"):
        return existing
    return new
