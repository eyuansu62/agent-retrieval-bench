from __future__ import annotations

from pathlib import Path
from typing import Any

from .filters import should_skip_pr, split_changed_files
from .github_api import GitHubAPI
from .io import append_jsonl, read_json, repo_slug, utc_now, write_json

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
