from __future__ import annotations

import subprocess
from pathlib import Path

from .io import repo_slug


def verify_base_commits(
    repo: str,
    commits: list[str],
    repos_dir: Path,
    remote_base: str = "https://github.com",
    blob_filter: bool = True,
) -> dict[str, object]:
    repos_dir.mkdir(parents=True, exist_ok=True)
    cache_path = repos_dir / f"{repo_slug(repo)}.git"
    url = f"{remote_base.rstrip('/')}/{repo}.git"
    if not cache_path.exists():
        clone_args = ["git", "clone", "--bare", url, str(cache_path)]
        if blob_filter:
            clone_args.insert(3, "--filter=blob:none")
        _run(clone_args)
    elif not blob_filter:
        _disable_partial_clone_filter(cache_path)
    existing: list[str] = []
    missing: list[str] = []
    for commit in sorted(set(commits)):
        if not commit:
            continue
        if _has_commit(cache_path, commit):
            existing.append(commit)
            continue
        fetch_args = ["git", "-C", str(cache_path), "fetch", "--depth=1"]
        if blob_filter:
            fetch_args.append("--filter=blob:none")
        fetch_args.extend(["origin", commit])
        _run(fetch_args, check=False)
        if _has_commit(cache_path, commit):
            existing.append(commit)
        else:
            missing.append(commit)
    return {"repo": repo, "cache_path": str(cache_path), "existing": existing, "missing": missing}


def _has_commit(repo_path: Path, commit: str) -> bool:
    result = _run(["git", "-C", str(repo_path), "cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    return result.returncode == 0


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def _disable_partial_clone_filter(repo_path: Path) -> None:
    for key in ("remote.origin.promisor", "remote.origin.partialclonefilter", "extensions.partialClone"):
        _run(["git", "-C", str(repo_path), "config", "--unset-all", key], check=False)
