from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .filters import (
    contains_raw_patch_marker,
    contains_review_leakage,
    is_source_file,
    is_test_file,
    split_changed_files,
)
from .hardmine import load_corpus_pairs
from .hardness import seed_audit_rows
from .io import ensure_parent, read_jsonl, repo_slug, stable_id, truncate_text, utc_now, write_json

PR_TEMPLATE_MARKERS = (
    "congratulations! you've made it this far",
    "please replace this with a description",
    "pull request checklist",
    "your pull request should have no more than two commits",
    "you should add/modify tests to cover your proposed code changes",
    "it should pass all tests in the available continuous integration systems",
    "before submitting",
    "did you read the contributor guideline",
)
CODERABBIT_MARKERS = (
    "summary by coderabbit",
    "release notes by coderabbit",
    "auto-generated comment: release notes by coderabbit",
    "auto-generated reply by coderabbit",
)
BEHAVIOR_MARKERS = re.compile(
    r"\b("
    r"bug|fix(?:es|ed)?|fail(?:s|ed|ing)?|regression|reproducer|assert|error|exception|panic|"
    r"behavior|behaviour|support|compat(?:ibility)?|validation|performance|race|runtime|"
    r"api|config|configuration|pipeline|model|tokenizer|scheduler|binding|cache|serve|loading"
    r")\b",
    re.IGNORECASE,
)
TEST_REQUEST_MARKERS = re.compile(r"\b(test|tests|coverage|regression|pytest|vitest|junit)\b", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
PATH_RE = re.compile(r"(?P<path>[\w./-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|kt|scala))")
GENERIC_PATH_TOKENS = {
    "src",
    "test",
    "tests",
    "spec",
    "utils",
    "util",
    "common",
    "helper",
    "helpers",
    "index",
    "main",
    "mod",
    "lib",
}


def mine_code2test_prs(
    raw_dir: Path,
    out_dir: Path,
    report_dir: Path,
    audit_path: Path | None = None,
    audited_pool_path: Path | None = None,
    corpus_manifest: Path | None = None,
    require_corpus: bool = False,
    require_gold_in_corpus: bool = False,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 20,
    max_tests: int = 3,
    audit_limit: int = 120,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    corpus_pairs = load_corpus_pairs(corpus_manifest) if corpus_manifest else None
    corpus_paths = load_corpus_paths(corpus_manifest) if require_gold_in_corpus and corpus_manifest else {}
    audited_ids, audited_clusters = load_audited_exclusions(audit_path, audited_pool_path)
    selected: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    seen_clusters: set[str] = set()
    source_counts: Counter[str] = Counter()
    repo_names = list(repos or repos_from_raw(raw_dir))

    for repo in repo_names:
        repo_raw = raw_dir / repo_slug(repo)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        details_by_pr = latest_by_pr(repo_raw / "commit_details.jsonl")
        for pr_number, files_record in sorted(files_by_pr.items(), reverse=True):
            pr = (pr_by_number.get(pr_number) or {}).get("data") or {}
            candidate, reason = build_pr_code2test_sample(
                repo=repo,
                pr_number=pr_number,
                pr=pr,
                files_record=files_record,
                details_record=details_by_pr.get(pr_number),
                max_changed_files=max_changed_files,
                max_tests=max_tests,
            )
            if reason:
                dropped[reason] += 1
                continue
            assert candidate is not None
            sample_id = str(candidate["id"])
            cluster = pr_code2test_cluster(candidate)
            if sample_id in audited_ids or cluster in audited_clusters:
                dropped["already_audited"] += 1
                continue
            if cluster in seen_clusters:
                dropped["duplicate_cluster"] += 1
                continue
            if require_corpus and corpus_pairs is not None and (repo, candidate["base_commit"]) not in corpus_pairs:
                dropped["missing_corpus_pair"] += 1
                continue
            if require_gold_in_corpus and corpus_manifest:
                paths = corpus_paths.get((repo, candidate["base_commit"]), set())
                missing = [path for path in candidate["gold"]["related_tests"] if path not in paths]
                if missing:
                    dropped["gold_missing_from_corpus"] += 1
                    continue
            seen_clusters.add(cluster)
            selected.append(candidate)
            source_counts[repo] += 1
            if limit_samples and len(selected) >= limit_samples:
                break
        if limit_samples and len(selected) >= limit_samples:
            break

    selected.sort(key=sample_sort_key)
    if limit_samples:
        selected = selected[:limit_samples]
    write_candidate_outputs(out_dir, selected)
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = seed_audit_rows([sample_to_pool_like_row(sample) for sample in selected[: max(0, audit_limit)]])
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(report_dir / "audit_samples.csv", audit_rows, ("sample_id", "task_type", "repo", "query_excerpt", "gold_files", "verdict", "reason", "keep", "notes"))
    summary = {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "audit": str(audit_path) if audit_path else None,
        "audited_pool": str(audited_pool_path) if audited_pool_path else None,
        "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
        "require_corpus": require_corpus,
        "require_gold_in_corpus": require_gold_in_corpus,
        "max_changed_files": max_changed_files,
        "max_tests": max_tests,
        "total": len(selected),
        "counts_by_task": {"code2test": len(selected)} if selected else {},
        "unique_pairs": len({(sample["repo"], sample["base_commit"]) for sample in selected}),
        "repos": dict(sorted(source_counts.items())),
        "dropped": dict(sorted(dropped.items())),
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "code2test": str(out_dir / "code2test.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def build_pr_code2test_sample(
    repo: str,
    pr_number: int,
    pr: dict[str, Any],
    files_record: dict[str, Any],
    details_record: dict[str, Any] | None = None,
    max_changed_files: int = 20,
    max_tests: int = 3,
) -> tuple[dict[str, Any] | None, str | None]:
    changed_files = [file for file in files_record.get("data", []) if file.get("filename")]
    changed_paths = [file["filename"] for file in changed_files]
    if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
        return None, "missing_base_or_merge"
    if not changed_paths or len(changed_paths) > max_changed_files:
        return None, "changed_file_limit"
    if low_value_change_majority(changed_paths):
        return None, "low_value_pr"
    implementation = [
        file["filename"]
        for file in changed_files
        if file.get("status") not in {"removed", "renamed"}
        and is_source_file(file["filename"])
        and not is_test_file(file["filename"])
    ]
    tests = [
        file["filename"]
        for file in changed_files
        if file.get("status") not in {"added", "removed", "renamed"} and is_test_file(file["filename"])
    ]
    if not implementation or not tests:
        return None, "missing_impl_or_existing_tests"
    if len(tests) > max_tests:
        return None, "too_many_tests"
    body = clean_pr_body(pr.get("body") or "")
    title = truncate_text(pr.get("title"), 300)
    query_text = f"{title}\n{body}\n{' '.join(implementation)}"
    if not meaningful_query(title, body):
        return None, "weak_query"
    if has_query_noise(query_text):
        return None, "query_noise"
    if leaks_gold_path(query_text, tests):
        return None, "test_path_leak"
    evidence = code2test_evidence(implementation, tests, title, body, details_record)
    if len(evidence) < 2:
        return None, "weak_evidence"
    query = {
        "pr_title": title,
        "pr_body": truncate_text(body, 1800),
        "implementation_files": implementation[:8],
        "implementation_file_count": len(implementation),
        "changed_file_summary": summarize_changed_files(implementation, tests, changed_paths),
    }
    sample_id = stable_id(repo, "code2test_pr", pr_number, *sorted(tests))
    fix_commit = (pr.get("mergeCommit") or {}).get("oid")
    sample = {
        "id": sample_id,
        "version": 2,
        "task_type": "code2test",
        "repo": repo,
        "base_commit": pr.get("baseRefOid"),
        "query": redact_fix_commit(query, fix_commit),
        "gold": {
            "root_cause_files": dedupe(implementation[:8]),
            "root_cause_symbols": [],
            "related_tests": dedupe(tests[:max_tests]),
            "supporting_files": dedupe(implementation[8:] + [path for path in changed_paths if path not in implementation and path not in tests])[:10],
            "negative_distractors": dedupe([path for path in implementation if path not in implementation[:3]])[:8],
            "fix_commit": fix_commit,
        },
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": pr.get("baseRefOid")},
        "metadata": {
            "pr": pr.get("number") or pr_number,
            "pr_url": pr.get("url"),
            "created_at": pr.get("createdAt"),
            "merged_at": pr.get("mergedAt"),
            "confidence": "weak",
            "evidence": {
                "source": "pr_level_changed_implementation_and_existing_tests",
                "signals": evidence,
                "implementation_files": implementation,
                "related_tests": tests,
            },
            "generated_at": utc_now(),
        },
    }
    return sample, None


def clean_pr_body(body: str) -> str:
    body = HTML_COMMENT_RE.sub("", body.replace("\r\n", "\n"))
    kept_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("diff --git", "index ", "--- ", "+++ ", "@@ ")):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    return truncate_text(cleaned, 3000)


def meaningful_query(title: str, body: str) -> bool:
    text = f"{title}\n{body}".strip()
    if len(re.findall(r"[A-Za-z0-9_]+", text)) < 8:
        return False
    return bool(BEHAVIOR_MARKERS.search(text) or TEST_REQUEST_MARKERS.search(text))


def has_query_noise(text: str) -> bool:
    lowered = text.lower()
    if contains_raw_patch_marker(text) or contains_review_leakage(text):
        return True
    if any(marker in lowered for marker in CODERABBIT_MARKERS):
        return True
    if any(marker in lowered for marker in PR_TEMPLATE_MARKERS) and not BEHAVIOR_MARKERS.search(text):
        return True
    return False


def leaks_gold_path(query_text: str, tests: Iterable[str]) -> bool:
    lowered = query_text.lower()
    mentioned_paths = {match.group("path").lower() for match in PATH_RE.finditer(query_text)}
    for path in tests:
        normalized = path.lower()
        basename = PurePosixPath(path).name.lower()
        if normalized in lowered or basename in lowered:
            return True
        if normalized in mentioned_paths or basename in {PurePosixPath(path).name for path in mentioned_paths}:
            return True
    return False


def code2test_evidence(
    implementation: list[str],
    tests: list[str],
    title: str,
    body: str,
    details_record: dict[str, Any] | None,
) -> list[str]:
    evidence = ["same_pr_changed_implementation_and_tests"]
    text = f"{title}\n{body}"
    if BEHAVIOR_MARKERS.search(text):
        evidence.append("behavior_or_bug_signal")
    if TEST_REQUEST_MARKERS.search(text):
        evidence.append("test_or_regression_signal")
    if any(path_token_overlap(src, test) for src in implementation for test in tests):
        evidence.append("source_test_path_overlap")
    if any(not same_directory(src, test) for src in implementation for test in tests):
        evidence.append("cross_directory_test")
    if details_record and commit_details_touch_tests_and_sources(details_record):
        evidence.append("commit_detail_confirms_source_and_test_changes")
    return dedupe(evidence)


def commit_details_touch_tests_and_sources(details_record: dict[str, Any]) -> bool:
    for detail in details_record.get("data", []):
        paths = [file.get("filename", "") for file in detail.get("files", [])]
        implementation, tests, _ignored = split_changed_files(paths)
        if implementation and tests:
            return True
    return False


def summarize_changed_files(implementation: list[str], tests: list[str], changed_paths: list[str]) -> str:
    return (
        f"{len(implementation)} implementation files and {len(tests)} existing test files changed "
        f"within {len(changed_paths)} total files."
    )


def low_value_change_majority(paths: list[str]) -> bool:
    low_value = sum(1 for path in paths if is_low_value_path(path))
    return low_value > len(paths) / 2


def is_low_value_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    lowered_parts = {part.lower() for part in normalized.split("/")}
    suffix = PurePosixPath(normalized).suffix.lower()
    basename = PurePosixPath(normalized).name.lower()
    if lowered_parts & {"docs", "doc", "documentation", ".github", "changelog", "changelogs", "snapshot", "snapshots"}:
        return True
    if suffix in {".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".json"} and not is_source_file(normalized):
        return True
    if basename in {"readme.md", "changelog.md", "license", "license.md", "contributing.md"}:
        return True
    return False


def path_token_overlap(left: str, right: str) -> bool:
    return bool(path_tokens(left) & path_tokens(right))


def path_tokens(path: str) -> set[str]:
    pure = PurePosixPath(path.replace("\\", "/"))
    parts = list(pure.parts)
    if pure.suffix:
        parts.append(pure.stem)
    tokens: set[str] = set()
    for part in parts:
        tokens.update(token.lower() for token in re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part)))
    return {token for token in tokens if len(token) >= 3 and token not in GENERIC_PATH_TOKENS}


def same_directory(left: str, right: str) -> bool:
    return str(PurePosixPath(left).parent) == str(PurePosixPath(right).parent)


def pr_code2test_cluster(sample: dict[str, Any]) -> str:
    gold = sample.get("gold") or {}
    metadata = sample.get("metadata") or {}
    return json.dumps(
        [sample.get("repo", ""), metadata.get("pr_url") or metadata.get("pr") or sample.get("id", ""), sorted(gold.get("related_tests") or [])],
        ensure_ascii=False,
        sort_keys=True,
    )


def sample_sort_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    query = json.dumps(sample.get("query") or {}, ensure_ascii=False)
    implementation = (sample.get("metadata") or {}).get("evidence", {}).get("implementation_files") or []
    tests = (sample.get("gold") or {}).get("related_tests") or []
    evidence = (sample.get("metadata") or {}).get("evidence", {}).get("signals") or []
    same_dir = any(same_directory(src, test) for src in implementation for test in tests)
    return (
        int(same_dir),
        len(tests),
        -len(evidence),
        -int(bool(BEHAVIOR_MARKERS.search(query))),
        sample.get("repo", ""),
        -int((sample.get("metadata") or {}).get("pr") or 0),
        sample.get("id", ""),
    )


def sample_to_pool_like_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.get("id", ""),
        "task_type": sample.get("task_type", ""),
        "repo": sample.get("repo", ""),
        "query_excerpt": json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True),
        "gold_files": (sample.get("gold") or {}).get("related_tests") or [],
    }


def write_candidate_outputs(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "code2test.jsonl", samples)
    write_jsonl(out_dir / "comment2context.jsonl", [])
    write_jsonl(out_dir / "trace2code.jsonl", [])


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def latest_by_pr(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        pr_number = record.get("pr_number") or (record.get("data") or {}).get("number")
        if pr_number is not None:
            records[int(pr_number)] = record
    return records


def repos_from_raw(raw_dir: Path) -> list[str]:
    repos: list[str] = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else []):
        if path.is_dir() and "__" in path.name:
            repos.append(path.name.replace("__", "/", 1))
    return repos


def load_audited_exclusions(audit_path: Path | None, audited_pool_path: Path | None) -> tuple[set[str], set[str]]:
    audited_ids: set[str] = set()
    if audit_path and audit_path.exists():
        if audit_path.suffix.lower() == ".csv":
            with audit_path.open("r", encoding="utf-8", newline="") as handle:
                audited_ids.update(str(row.get("sample_id", "")) for row in csv.DictReader(handle) if row.get("sample_id"))
        else:
            audited_ids.update(str(row.get("sample_id", "")) for row in read_jsonl(audit_path) if row.get("sample_id"))
    clusters: set[str] = set()
    if audited_pool_path and audited_pool_path.exists():
        for row in read_jsonl(audited_pool_path):
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if sample_id in audited_ids:
                repo = row.get("repo", "")
                pr_url = row.get("pr_url") or ((row.get("metadata") or {}).get("pr_url")) or sample_id
                gold = row.get("gold_files") or ((row.get("gold") or {}).get("related_tests")) or []
                clusters.add(json.dumps([repo, pr_url, sorted(str(path) for path in gold)], ensure_ascii=False, sort_keys=True))
    return audited_ids, clusters


def load_corpus_paths(corpus_manifest: Path | None) -> dict[tuple[str, str], set[str]]:
    paths_by_pair: dict[tuple[str, str], set[str]] = {}
    if not corpus_manifest:
        return paths_by_pair
    for record in read_jsonl(corpus_manifest):
        if record.get("status") != "ok":
            continue
        chunks_path = Path(str(record.get("chunks_path", "")))
        if not chunks_path.exists():
            continue
        paths = {str(chunk.get("path")) for chunk in read_jsonl(chunks_path) if chunk.get("path")}
        paths_by_pair[(str(record.get("repo")), str(record.get("base_commit")))] = paths
    return paths_by_pair


def redact_fix_commit(value: Any, fix_commit: str | None) -> Any:
    if not fix_commit:
        return value
    if isinstance(value, str):
        return value.replace(fix_commit, "[fix_commit]")
    if isinstance(value, list):
        return [redact_fix_commit(item, fix_commit) for item in value]
    if isinstance(value, dict):
        return {key: redact_fix_commit(item, fix_commit) for key, item in value.items()}
    return value


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
