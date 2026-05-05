from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .clone import verify_base_commits
from .curate import load_keep_ids
from .filters import is_generated_or_lockfile
from .io import ensure_parent, read_jsonl, repo_slug, stable_id, truncate_text, utc_now

SKIP_EXTENSIONS = {
    ".7z",
    ".avif",
    ".bin",
    ".bmp",
    ".bz2",
    ".class",
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".o",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".svg",
    ".tar",
    ".tgz",
    ".ttf",
    ".wasm",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
}

SYMBOL_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"),
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(?"),
    re.compile(r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b"),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(
        r"^\s*(?:public|private|protected|static|final|abstract|synchronized|\s)+[\w<>\[\], ?]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
    re.compile(r"^\s*(?:public\s+)?(?:class|interface|enum|record)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"),
)


def sample_paths_from_derived(derived_dir: Path) -> list[Path]:
    curated_samples = derived_dir / "samples.jsonl"
    if curated_samples.exists():
        return [curated_samples]
    return sorted(path for path in derived_dir.glob("*.jsonl") if path.is_file())


def load_sample_refs(
    sample_paths: Iterable[Path],
    repos: set[str] | None = None,
    keep_ids: set[str] | None = None,
    limit_samples: int | None = None,
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    count = 0
    for path in sample_paths:
        for sample in read_jsonl(path):
            if keep_ids is not None and sample.get("id") not in keep_ids:
                continue
            repo = sample.get("repo")
            base_commit = sample.get("base_commit")
            if not repo or not base_commit:
                continue
            if repos and repo not in repos:
                continue
            count += 1
            key = (repo, base_commit)
            if key not in seen:
                refs.append({"repo": repo, "base_commit": base_commit})
                seen.add(key)
            if limit_samples and count >= limit_samples:
                return refs
    return refs


def build_candidate_corpus(
    sample_paths: Iterable[Path],
    out_dir: Path,
    repos_dir: Path,
    repos: set[str] | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    limit_pairs: int | None = None,
    max_file_bytes: int = 400_000,
    max_chunk_chars: int = 8_000,
    max_files_per_commit: int = 20_000,
    remote_base: str = "https://github.com",
) -> dict[str, Any]:
    keep_ids = load_keep_ids(keep_list)
    refs = load_sample_refs(sample_paths, repos=repos, keep_ids=keep_ids, limit_samples=limit_samples)
    if limit_pairs:
        refs = refs[:limit_pairs]
    by_repo: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        by_repo[ref["repo"]].append(ref["base_commit"])

    out_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for repo, commits in sorted(by_repo.items()):
        verification = verify_base_commits(repo, commits, repos_dir, remote_base=remote_base, blob_filter=False)
        missing = set(verification.get("missing", []))
        for commit in sorted(set(commits)):
            if commit in missing:
                manifests.append({"repo": repo, "base_commit": commit, "status": "missing_commit", "chunk_count": 0})
                continue
            manifests.append(
                build_commit_chunks(
                    repo=repo,
                    base_commit=commit,
                    bare_repo=Path(str(verification["cache_path"])),
                    out_dir=out_dir,
                    max_file_bytes=max_file_bytes,
                    max_chunk_chars=max_chunk_chars,
                    max_files_per_commit=max_files_per_commit,
                )
            )
    manifest_path = out_dir / "corpus_manifest.jsonl"
    _write_jsonl(manifest_path, manifests)
    return {
        "pairs": len(refs),
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "chunks": sum(int(item.get("chunk_count", 0)) for item in manifests),
        "manifest": str(manifest_path),
        "outputs": [item for item in manifests if item.get("status") == "ok"],
        "missing": [item for item in manifests if item.get("status") != "ok"],
    }


def build_commit_chunks(
    repo: str,
    base_commit: str,
    bare_repo: Path,
    out_dir: Path,
    max_file_bytes: int = 400_000,
    max_chunk_chars: int = 8_000,
    max_files_per_commit: int = 20_000,
) -> dict[str, Any]:
    chunks_path = out_dir / repo_slug(repo) / f"{base_commit}.chunks.jsonl"
    existing = existing_chunks_manifest(repo, base_commit, chunks_path)
    if existing is not None:
        return existing

    tree_entries = _git_tree_entries(bare_repo, base_commit)
    candidate_paths = [path for path in tree_entries if is_candidate_path(path)]
    if max_files_per_commit:
        candidate_paths = candidate_paths[:max_files_per_commit]

    ensure_parent(chunks_path)
    file_count = 0
    symbol_count = 0
    chunk_count = 0
    skipped_large = 0
    with chunks_path.open("w", encoding="utf-8") as handle:
        for path in candidate_paths:
            size = tree_entries.get(path)
            if size is None or size > max_file_bytes:
                skipped_large += 1
                continue
            content = _git_blob(bare_repo, base_commit, path)
            if not content:
                continue
            file_count += 1
            for chunk in chunks_for_file(repo, base_commit, path, content, max_chunk_chars=max_chunk_chars):
                handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                chunk_count += 1
                if chunk["kind"] == "symbol":
                    symbol_count += 1

    return {
        "repo": repo,
        "base_commit": base_commit,
        "status": "ok",
        "chunks_path": str(chunks_path),
        "chunk_count": chunk_count,
        "file_count": file_count,
        "symbol_count": symbol_count,
        "skipped_large": skipped_large,
        "generated_at": utc_now(),
    }


def existing_chunks_manifest(repo: str, base_commit: str, chunks_path: Path) -> dict[str, Any] | None:
    if not chunks_path.exists() or chunks_path.stat().st_size == 0:
        return None

    chunk_count = 0
    file_count = 0
    symbol_count = 0
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_repo = chunk.get("repo")
            chunk_commit = chunk.get("base_commit")
            if chunk_repo and chunk_repo != repo:
                return None
            if chunk_commit and chunk_commit != base_commit:
                return None
            chunk_count += 1
            if chunk.get("kind") == "file":
                file_count += 1
            elif chunk.get("kind") == "symbol":
                symbol_count += 1

    if chunk_count == 0:
        return None

    return {
        "repo": repo,
        "base_commit": base_commit,
        "status": "ok",
        "chunks_path": str(chunks_path),
        "chunk_count": chunk_count,
        "file_count": file_count,
        "symbol_count": symbol_count,
        "skipped_large": None,
        "reused": True,
        "resumed_at": utc_now(),
    }


def chunks_for_file(repo: str, base_commit: str, path: str, content: str, max_chunk_chars: int = 8_000) -> list[dict[str, Any]]:
    lines = content.splitlines()
    chunks = [
        _chunk(
            repo=repo,
            base_commit=base_commit,
            path=path,
            kind="file",
            symbol="",
            start_line=1,
            end_line=len(lines),
            text=truncate_text(content, max_chunk_chars),
        )
    ]
    for symbol, start_line, end_line in extract_symbols(content):
        text = "\n".join(lines[start_line - 1 : end_line])
        chunks.append(
            _chunk(
                repo=repo,
                base_commit=base_commit,
                path=path,
                kind="symbol",
                symbol=symbol,
                start_line=start_line,
                end_line=end_line,
                text=truncate_text(text, max_chunk_chars),
            )
        )
    return chunks


def extract_symbols(content: str) -> list[tuple[str, int, int]]:
    lines = content.splitlines()
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines, start=1):
        for pattern in SYMBOL_PATTERNS:
            match = pattern.match(line)
            if match:
                starts.append((match.group("name"), index))
                break
    symbols: list[tuple[str, int, int]] = []
    for index, (symbol, start_line) in enumerate(starts):
        next_start = starts[index + 1][1] if index + 1 < len(starts) else len(lines) + 1
        symbols.append((symbol, start_line, max(start_line, next_start - 1)))
    return symbols


def is_candidate_path(path: str) -> bool:
    if is_generated_or_lockfile(path):
        return False
    suffix = PurePosixPath(path).suffix.lower()
    return suffix not in SKIP_EXTENSIONS


def _chunk(
    repo: str,
    base_commit: str,
    path: str,
    kind: str,
    symbol: str,
    start_line: int,
    end_line: int,
    text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": stable_id(repo, base_commit, path, kind, symbol, start_line, end_line),
        "repo": repo,
        "base_commit": base_commit,
        "path": path,
        "kind": kind,
        "symbol": symbol,
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
    }


def _git_lines(repo_path: Path, args: list[str]) -> list[str]:
    result = _run(["git", "-C", str(repo_path), *args])
    return [line for line in result.stdout.splitlines() if line]


def _git_size(repo_path: Path, commit: str, path: str) -> int | None:
    result = _run(["git", "-C", str(repo_path), "cat-file", "-s", f"{commit}:{path}"], check=False)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _git_tree_entries(repo_path: Path, commit: str) -> dict[str, int | None]:
    entries: dict[str, int | None] = {}
    for line in _git_lines(repo_path, ["ls-tree", "-r", "-l", commit]):
        metadata, separator, path = line.partition("\t")
        if not separator or not path:
            continue
        parts = metadata.split()
        if len(parts) < 4:
            continue
        size_text = parts[3]
        entries[path] = int(size_text) if size_text.isdigit() else None
    return entries


def _git_blob(repo_path: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "show", f"{commit}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or b"\x00" in result.stdout:
        return ""
    return result.stdout.decode("utf-8", errors="replace")


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
