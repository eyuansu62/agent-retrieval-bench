from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .code2test_pr import (
    clean_pr_body,
    dedupe,
    latest_by_pr,
    load_corpus_paths,
    low_value_change_majority,
    path_tokens,
    repos_from_raw,
    same_directory,
    write_csv,
    write_jsonl,
)
from .filters import contains_raw_patch_marker, is_generated_or_lockfile, is_source_file, is_test_file
from .hardmine import load_corpus_pairs
from .io import read_jsonl, repo_slug, stable_id, truncate_text, utc_now, write_json

EDIT2RIPPLE_AUDIT_FIELDS = (
    "sample_id",
    "task_type",
    "repo",
    "source_pr",
    "anchor_file",
    "query_excerpt",
    "gold_files",
    "changed_files",
    "ripple_relation",
    "leakage_fatal",
    "leakage_nonfatal",
    "verdict",
    "reason",
    "keep",
    "notes",
)
SCHEMA_EXTENSIONS = {".proto", ".graphql", ".graphqls", ".thrift", ".avsc"}
CONFIG_EXTENSIONS = {".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"}
DOC_EXTENSIONS = {".md", ".rst"}
LOW_VALUE_PARTS = {
    ".github",
    "changelog",
    "changelogs",
    "dist",
    "docs",
    "doc",
    "generated",
    "node_modules",
    "snapshots",
    "target",
    "third_party",
    "vendor",
}
LOW_VALUE_BASENAMES = {
    "changelog",
    "changelog.md",
    "license",
    "license.md",
    "readme",
    "readme.md",
}
COMMON_DIFF_TOKENS = {
    "and",
    "args",
    "bool",
    "class",
    "const",
    "def",
    "else",
    "false",
    "for",
    "from",
    "func",
    "function",
    "if",
    "import",
    "int",
    "let",
    "none",
    "null",
    "return",
    "self",
    "str",
    "string",
    "struct",
    "the",
    "this",
    "true",
}
GENERIC_PATH_TOKENS = {
    "api",
    "app",
    "base",
    "common",
    "config",
    "core",
    "file",
    "helper",
    "helpers",
    "index",
    "init",
    "lib",
    "main",
    "mod",
    "module",
    "options",
    "package",
    "src",
    "test",
    "tests",
    "util",
    "utils",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def mine_edit2ripple(
    raw_dir: Path,
    out_dir: Path,
    report_dir: Path,
    audit_path: Path | None = None,
    corpus_manifest: Path | None = None,
    require_corpus: bool = False,
    require_gold_in_corpus: bool = False,
    repos: Iterable[str] | None = None,
    min_changed_files: int = 2,
    max_changed_files: int = 8,
    max_gold_files: int = 4,
    max_candidates_per_pr: int = 2,
    audit_limit: int = 120,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    corpus_pairs = load_corpus_pairs(corpus_manifest) if corpus_manifest else set()
    corpus_paths = load_corpus_paths(corpus_manifest) if require_gold_in_corpus and corpus_manifest else {}
    audited_ids = load_audited_ids(audit_path)
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    dropped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for repo in list(repos or repos_from_raw(raw_dir)):
        repo_raw = raw_dir / repo.replace("/", "__", 1)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        details_by_pr = latest_by_pr(repo_raw / "commit_details.jsonl")
        for pr_number, files_record in sorted(files_by_pr.items(), reverse=True):
            pr = (pr_by_number.get(pr_number) or {}).get("data") or {}
            candidates, reason = build_pr_edit2ripple_samples(
                repo=repo,
                pr_number=pr_number,
                pr=pr,
                files_record=files_record,
                details_record=details_by_pr.get(pr_number),
                min_changed_files=min_changed_files,
                max_changed_files=max_changed_files,
                max_gold_files=max_gold_files,
                max_candidates_per_pr=max_candidates_per_pr,
            )
            if reason:
                dropped[reason] += 1
                continue
            for candidate in candidates:
                sample_id = str(candidate["id"])
                cluster = edit2ripple_cluster(candidate)
                if sample_id in audited_ids:
                    dropped["already_audited"] += 1
                    continue
                if cluster in seen_clusters:
                    dropped["duplicate_cluster"] += 1
                    continue
                pair = (repo, str(candidate["base_commit"]))
                if require_corpus and corpus_manifest and pair not in corpus_pairs:
                    dropped["missing_corpus_pair"] += 1
                    continue
                if require_gold_in_corpus and corpus_manifest:
                    paths = corpus_paths.get(pair, set())
                    missing = [path for path in candidate["gold"]["files"] if path not in paths]
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
        if limit_samples and len(selected) >= limit_samples:
            break

    selected.sort(key=sample_sort_key)
    write_candidate_outputs(out_dir, selected)
    audit_rows = [audit_row(sample) for sample in selected[: max(0, audit_limit)]]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(report_dir / "audit_samples.csv", audit_rows, EDIT2RIPPLE_AUDIT_FIELDS)
    summary = {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "audit": str(audit_path) if audit_path else None,
        "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
        "require_corpus": require_corpus,
        "require_gold_in_corpus": require_gold_in_corpus,
        "min_changed_files": min_changed_files,
        "max_changed_files": max_changed_files,
        "max_gold_files": max_gold_files,
        "max_candidates_per_pr": max_candidates_per_pr,
        "total": len(selected),
        "counts_by_task": {"edit2ripple": len(selected)} if selected else {},
        "unique_pairs": len({(sample["repo"], sample["base_commit"]) for sample in selected}),
        "repos": dict(sorted(source_counts.items())),
        "dropped": dict(sorted(dropped.items())),
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "edit2ripple": str(out_dir / "edit2ripple.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def mine_edit2ripple_from_samples(
    api: Any,
    sample_paths: Iterable[Path],
    out_dir: Path,
    report_dir: Path,
    audit_path: Path | None = None,
    corpus_manifest: Path | None = None,
    require_corpus: bool = False,
    require_gold_in_corpus: bool = False,
    repos: Iterable[str] | None = None,
    max_gold_files: int = 4,
    max_candidates_per_pr: int = 2,
    audit_limit: int = 120,
    limit_prs: int | None = None,
    limit_samples: int | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Mine edit2ripple candidates by refetching PR metadata from existing sample PR URLs."""

    allowed_repos = set(repos or [])
    records = [
        record
        for record in sample_pr_records(sample_paths)
        if not allowed_repos or record["repo"] in allowed_repos
    ]
    if limit_prs:
        records = records[:limit_prs]

    corpus_pairs = load_corpus_pairs(corpus_manifest) if corpus_manifest else set()
    corpus_paths = load_corpus_paths(corpus_manifest) if require_gold_in_corpus and corpus_manifest else {}
    audited_ids = load_audited_ids(audit_path)
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    dropped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    fetched_prs = 0

    for record in records:
        repo = str(record["repo"])
        pr_number = int(record["pr_number"])
        base_commit = str(record["base_commit"])
        pair = (repo, base_commit)
        if require_corpus and corpus_manifest and pair not in corpus_pairs:
            dropped["missing_corpus_pair"] += 1
            continue
        try:
            pr, files_record, details_record = fetch_pr_inputs_from_github(api, repo, pr_number, base_commit)
            fetched_prs += 1
        except RuntimeError as error:
            errors.append({"repo": repo, "pr_number": pr_number, "error": str(error)})
            dropped["fetch_error"] += 1
            if continue_on_error:
                continue
            raise

        candidates, reason = build_pr_edit2ripple_samples(
            repo=repo,
            pr_number=pr_number,
            pr=pr,
            files_record=files_record,
            details_record=details_record,
            max_gold_files=max_gold_files,
            max_candidates_per_pr=max_candidates_per_pr,
        )
        if reason:
            dropped[reason] += 1
            continue
        for candidate in candidates:
            sample_id = str(candidate["id"])
            cluster = edit2ripple_cluster(candidate)
            if sample_id in audited_ids:
                dropped["already_audited"] += 1
                continue
            if cluster in seen_clusters:
                dropped["duplicate_cluster"] += 1
                continue
            if require_gold_in_corpus and corpus_manifest:
                paths = corpus_paths.get(pair, set())
                missing = [path for path in candidate["gold"]["files"] if path not in paths]
                if missing:
                    dropped["gold_missing_from_corpus"] += 1
                    continue
            candidate["metadata"]["source_samples"] = record["sample_ids"]
            seen_clusters.add(cluster)
            selected.append(candidate)
            source_counts[repo] += 1
            if limit_samples and len(selected) >= limit_samples:
                break
        if limit_samples and len(selected) >= limit_samples:
            break

    selected.sort(key=sample_sort_key)
    write_candidate_outputs(out_dir, selected)
    audit_rows = [audit_row(sample) for sample in selected[: max(0, audit_limit)]]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(report_dir / "audit_samples.csv", audit_rows, EDIT2RIPPLE_AUDIT_FIELDS)
    summary = {
        "generated_at": utc_now(),
        "source": "sample_pr_url_backfill",
        "sample_paths": [str(path) for path in sample_paths],
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "audit": str(audit_path) if audit_path else None,
        "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
        "require_corpus": require_corpus,
        "require_gold_in_corpus": require_gold_in_corpus,
        "input_prs": len(records),
        "fetched_prs": fetched_prs,
        "total": len(selected),
        "counts_by_task": {"edit2ripple": len(selected)} if selected else {},
        "unique_pairs": len({(sample["repo"], sample["base_commit"]) for sample in selected}),
        "repos": dict(sorted(source_counts.items())),
        "dropped": dict(sorted(dropped.items())),
        "errors": errors[:20],
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "edit2ripple": str(out_dir / "edit2ripple.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def mine_edit2ripple_from_sample_commits(
    sample_paths: Iterable[Path],
    out_dir: Path,
    report_dir: Path,
    repos_dir: Path,
    audit_path: Path | None = None,
    corpus_manifest: Path | None = None,
    require_corpus: bool = False,
    require_gold_in_corpus: bool = False,
    repos: Iterable[str] | None = None,
    remote_base: str = "https://github.com",
    blob_filter: bool = True,
    max_gold_files: int = 4,
    max_candidates_per_pr: int = 2,
    audit_limit: int = 120,
    limit_commits: int | None = None,
    limit_samples: int | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    allowed_repos = set(repos or [])
    records = [
        record
        for record in sample_commit_records(sample_paths)
        if not allowed_repos or record["repo"] in allowed_repos
    ]
    if limit_commits:
        records = records[:limit_commits]

    corpus_pairs = load_corpus_pairs(corpus_manifest) if corpus_manifest else set()
    corpus_paths = load_corpus_paths(corpus_manifest) if require_gold_in_corpus and corpus_manifest else {}
    audited_ids = load_audited_ids(audit_path)
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    dropped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    prepared_repos: dict[str, Path] = {}
    processed_commits = 0

    for record in records:
        repo = str(record["repo"])
        base_commit = str(record["base_commit"])
        fix_commit = str(record["fix_commit"])
        pair = (repo, base_commit)
        if require_corpus and corpus_manifest and pair not in corpus_pairs:
            dropped["missing_corpus_pair"] += 1
            continue
        try:
            bare_repo = prepared_repos.get(repo)
            if bare_repo is None:
                bare_repo = ensure_git_repo(repo, repos_dir, remote_base=remote_base, blob_filter=blob_filter)
                prepared_repos[repo] = bare_repo
            ensure_git_commits(bare_repo, repo, [base_commit, fix_commit], blob_filter=blob_filter)
            pr, files_record, details_record = build_git_diff_inputs(bare_repo, record)
            processed_commits += 1
        except RuntimeError as error:
            errors.append({"repo": repo, "base_commit": base_commit, "fix_commit": fix_commit, "error": str(error)})
            dropped["git_error"] += 1
            if continue_on_error:
                continue
            raise

        candidates, reason = build_pr_edit2ripple_samples(
            repo=repo,
            pr_number=int(record.get("pr_number") or 0),
            pr=pr,
            files_record=files_record,
            details_record=details_record,
            max_gold_files=max_gold_files,
            max_candidates_per_pr=max_candidates_per_pr,
        )
        if reason:
            dropped[reason] += 1
            continue
        for candidate in candidates:
            sample_id = str(candidate["id"])
            cluster = edit2ripple_cluster(candidate)
            if sample_id in audited_ids:
                dropped["already_audited"] += 1
                continue
            if cluster in seen_clusters:
                dropped["duplicate_cluster"] += 1
                continue
            if require_gold_in_corpus and corpus_manifest:
                paths = corpus_paths.get(pair, set())
                missing = [path for path in candidate["gold"]["files"] if path not in paths]
                if missing:
                    dropped["gold_missing_from_corpus"] += 1
                    continue
            candidate["metadata"]["source_samples"] = record["sample_ids"]
            candidate["metadata"]["source"] = "git_diff"
            seen_clusters.add(cluster)
            selected.append(candidate)
            source_counts[repo] += 1
            if limit_samples and len(selected) >= limit_samples:
                break
        if limit_samples and len(selected) >= limit_samples:
            break

    selected.sort(key=sample_sort_key)
    write_candidate_outputs(out_dir, selected)
    audit_rows = [audit_row(sample) for sample in selected[: max(0, audit_limit)]]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(report_dir / "audit_samples.csv", audit_rows, EDIT2RIPPLE_AUDIT_FIELDS)
    summary = {
        "generated_at": utc_now(),
        "source": "sample_fix_commit_git_diff",
        "sample_paths": [str(path) for path in sample_paths],
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "repos_dir": str(repos_dir),
        "audit": str(audit_path) if audit_path else None,
        "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
        "require_corpus": require_corpus,
        "require_gold_in_corpus": require_gold_in_corpus,
        "input_commits": len(records),
        "processed_commits": processed_commits,
        "total": len(selected),
        "counts_by_task": {"edit2ripple": len(selected)} if selected else {},
        "unique_pairs": len({(sample["repo"], sample["base_commit"]) for sample in selected}),
        "repos": dict(sorted(source_counts.items())),
        "dropped": dict(sorted(dropped.items())),
        "errors": errors[:20],
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "edit2ripple": str(out_dir / "edit2ripple.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def report_edit2ripple_pilot(
    sample_paths: Iterable[Path],
    out_path: Path | None = None,
    json_out_path: Path | None = None,
    corpus_manifest: Path | None = None,
    audit_path: Path | None = None,
    min_samples: int = 50,
    min_valid_rate: float = 0.90,
    max_test_only_ratio: float = 0.35,
    max_candidates_per_pr: int = 2,
) -> dict[str, Any]:
    samples = [sample for path in sample_paths for sample in read_jsonl(path) if sample.get("task_type") == "edit2ripple"]
    corpus_paths = load_corpus_paths(corpus_manifest) if corpus_manifest else {}
    audit_verdicts = load_audit_verdicts(audit_path)
    rows: list[dict[str, Any]] = []
    pr_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()

    for sample in samples:
        repo = str(sample.get("repo") or "")
        base_commit = str(sample.get("base_commit") or "")
        metadata = sample.get("metadata") or {}
        query = sample.get("query") or {}
        gold_files = [str(path) for path in (sample.get("gold") or {}).get("files") or []]
        leakage = metadata.get("leakage") or diagnose_edit2ripple_leakage(sample)
        pair_paths = corpus_paths.get((repo, base_commit), set()) if corpus_manifest else set()
        missing_gold = [path for path in gold_files if corpus_manifest and path not in pair_paths]
        pr_key = edit2ripple_pr_key(sample)
        pr_counts[pr_key] += 1
        verdict = normalized_verdict(audit_verdicts.get(str(sample.get("id"))) or ((sample.get("audit") or {}).get("verdict")))
        if verdict:
            verdict_counts[verdict] += 1
        relations = [str(item) for item in metadata.get("ripple_relation") or []]
        relation_counts.update(relations)
        strata = relation_strata(gold_files, relations)
        stratum_counts.update(strata)
        rows.append(
            {
                "sample_id": sample.get("id"),
                "repo": repo,
                "base_commit": base_commit,
                "source_pr": metadata.get("source_pr") or metadata.get("pr") or "",
                "anchor_file": query.get("anchor_file") or "",
                "anchor_is_test": is_test_file(str(query.get("anchor_file") or "")),
                "gold_files": gold_files,
                "gold_count": len(gold_files),
                "missing_gold_files": missing_gold,
                "fatal_leakage": leakage.get("fatal") or [],
                "nonfatal_hints": leakage.get("nonfatal") or [],
                "changed_file_count": len(metadata.get("changed_files") or []),
                "changed_file_range_ok": 2 <= len(metadata.get("changed_files") or []) <= 8,
                "test_only_gold": bool(gold_files) and all(is_test_file(path) for path in gold_files),
                "has_test_gold": any(is_test_file(path) for path in gold_files),
                "relations": relations,
                "strata": strata,
                "verdict": verdict or "pending",
            }
        )

    audited_total = sum(verdict_counts[verdict] for verdict in ("valid", "noisy", "leaked", "ambiguous"))
    valid_rate = (verdict_counts["valid"] / audited_total) if audited_total else None
    pr_over_limit = {key: count for key, count in sorted(pr_counts.items()) if count > max_candidates_per_pr}
    test_only_ratio = (sum(1 for row in rows if row["test_only_gold"]) / len(rows)) if rows else 0.0
    gates = {
        "sample_count_ok": len(rows) >= min_samples,
        "audit_complete": audited_total >= len(rows) and len(rows) > 0,
        "anchor_not_test": all(not row["anchor_is_test"] for row in rows),
        "fatal_leakage_zero": all(not row["fatal_leakage"] for row in rows),
        "every_sample_has_gold": all(row["gold_count"] >= 1 for row in rows),
        "every_gold_in_corpus": None if not corpus_manifest else all(not row["missing_gold_files"] for row in rows),
        "changed_file_range_ok": all(row["changed_file_range_ok"] for row in rows),
        "per_pr_candidate_limit_ok": not pr_over_limit,
        "test_only_gold_controlled": test_only_ratio <= max_test_only_ratio,
        "valid_rate_ok": None if valid_rate is None else valid_rate >= min_valid_rate,
    }
    status = edit2ripple_pilot_status(gates, audited_total)
    report = {
        "generated_at": utc_now(),
        "status": status,
        "inputs": {
            "samples": [str(path) for path in sample_paths],
            "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
            "audit": str(audit_path) if audit_path else None,
        },
        "thresholds": {
            "min_samples": min_samples,
            "min_valid_rate": min_valid_rate,
            "max_test_only_ratio": max_test_only_ratio,
            "max_candidates_per_pr": max_candidates_per_pr,
        },
        "total": len(rows),
        "audited_total": audited_total,
        "valid_rate": valid_rate,
        "test_only_gold_ratio": test_only_ratio,
        "verdicts": dict(sorted(verdict_counts.items())),
        "relations": dict(sorted(relation_counts.items())),
        "strata": dict(sorted(stratum_counts.items())),
        "gates": gates,
        "failures": {
            "fatal_leakage": [row for row in rows if row["fatal_leakage"]],
            "test_anchor": [row for row in rows if row["anchor_is_test"]],
            "missing_gold": [row for row in rows if row["missing_gold_files"]],
            "no_gold": [row for row in rows if row["gold_count"] < 1],
            "changed_file_range": [row for row in rows if not row["changed_file_range_ok"]],
            "pr_over_limit": pr_over_limit,
            "test_only_gold": [row for row in rows if row["test_only_gold"]],
        },
        "rows": rows,
    }
    if json_out_path:
        write_json(json_out_path, report)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_edit2ripple_pilot_markdown(report), encoding="utf-8")
    return report


def build_pr_edit2ripple_samples(
    repo: str,
    pr_number: int,
    pr: dict[str, Any],
    files_record: dict[str, Any],
    details_record: dict[str, Any] | None = None,
    min_changed_files: int = 2,
    max_changed_files: int = 8,
    max_gold_files: int = 4,
    max_candidates_per_pr: int = 2,
) -> tuple[list[dict[str, Any]], str | None]:
    changed_files = [file for file in files_record.get("data", []) if file.get("filename")]
    changed_paths = [str(file["filename"]) for file in changed_files]
    if not pr.get("baseRefOid") or not (pr.get("mergeCommit") or {}).get("oid"):
        return [], "missing_base_or_merge"
    if len(changed_paths) < min_changed_files or len(changed_paths) > max_changed_files:
        return [], "changed_file_limit"
    if low_value_change_majority(changed_paths):
        return [], "low_value_pr"
    if mass_rename_or_generated_only(changed_files):
        return [], "rename_or_generated_only"

    anchors = [file for file in changed_files if is_anchor_file(file)]
    if not anchors:
        return [], "missing_anchor"

    patches = {str(file["filename"]): str(file.get("patch") or "") for file in changed_files}
    intent = build_intent(pr, details_record)
    if not intent:
        return [], "weak_intent"

    scored_samples: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for anchor in anchors:
        anchor_path = str(anchor["filename"])
        related = related_gold_files(anchor_path, changed_files, patches, max_gold_files=max_gold_files)
        if not related:
            continue
        gold_files = [item["path"] for item in related]
        relation = dedupe(signal for item in related for signal in item["signals"])
        fix_commit = (pr.get("mergeCommit") or {}).get("oid")
        query = {
            "intent": truncate_text(intent, 1800),
            "anchor_file": anchor_path,
            "anchor_diff": sanitize_anchor_diff(patches.get(anchor_path, "")),
        }
        if not query["anchor_diff"]:
            continue
        sample = {
            "id": stable_id(repo, "edit2ripple", pr_number, anchor_path, *gold_files),
            "version": 1,
            "task_type": "edit2ripple",
            "repo": repo,
            "base_commit": pr.get("baseRefOid"),
            "query": redact_fix_commit(query, fix_commit),
            "gold": {
                "files": gold_files,
                "given_files": [anchor_path],
            },
            "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": pr.get("baseRefOid")},
            "metadata": {
                "source_pr": pr.get("url"),
                "pr": pr.get("number") or pr_number,
                "fix_commit": fix_commit,
                "changed_files": changed_paths,
                "ripple_relation": relation,
                "gold_evidence": related,
                "created_at": pr.get("createdAt"),
                "merged_at": pr.get("mergedAt"),
                "generated_at": utc_now(),
            },
            "audit": {
                "verdict": "pending",
                "notes": "",
            },
        }
        leakage = diagnose_edit2ripple_leakage(sample)
        sample["metadata"]["leakage"] = leakage
        if leakage["fatal"]:
            continue
        scored_samples.append((anchor_score(anchor, related), sample))

    if not scored_samples:
        return [], "no_related_gold_or_query_leakage"
    scored_samples.sort(key=lambda item: item[0])
    return [sample for _score, sample in scored_samples[:max_candidates_per_pr]], None


def is_anchor_file(file_record: dict[str, Any]) -> bool:
    path = str(file_record.get("filename") or "")
    status = str(file_record.get("status") or "")
    if status in {"added", "removed", "renamed"}:
        return False
    if is_format_only_patch(str(file_record.get("patch") or "")):
        return False
    return is_source_file(path) and not is_test_file(path) and not is_config_only_path(path)


def is_config_only_path(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in CONFIG_EXTENSIONS or suffix in DOC_EXTENSIONS


def is_allowed_gold_file(file_record: dict[str, Any]) -> bool:
    path = str(file_record.get("filename") or "")
    status = str(file_record.get("status") or "")
    if status in {"added", "removed", "renamed"}:
        return False
    if is_generated_or_lockfile(path) or is_low_value_path(path):
        return False
    if is_format_only_patch(str(file_record.get("patch") or "")):
        return False
    suffix = PurePosixPath(path).suffix.lower()
    return is_source_file(path) or suffix in SCHEMA_EXTENSIONS or suffix in CONFIG_EXTENSIONS or is_api_contract_doc(path)


def is_format_only_patch(patch: str) -> bool:
    if not patch:
        return False
    added: list[str] = []
    removed: list[str] = []
    for line in patch.replace("\r\n", "\n").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(normalize_format_line(line[1:]))
        elif line.startswith("-"):
            removed.append(normalize_format_line(line[1:]))
    if not added and not removed:
        return True
    return Counter(added) == Counter(removed)


def normalize_format_line(line: str) -> str:
    return re.sub(r"\s+", "", line)


def is_low_value_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = {part.lower() for part in PurePosixPath(normalized).parts}
    basename = PurePosixPath(normalized).name.lower()
    stem = PurePosixPath(normalized).stem.lower()
    if basename in LOW_VALUE_BASENAMES or stem in LOW_VALUE_BASENAMES:
        return True
    if parts & (LOW_VALUE_PARTS - {"docs", "doc"}):
        return True
    suffix = PurePosixPath(normalized).suffix.lower()
    if suffix in DOC_EXTENSIONS and not is_api_contract_doc(normalized):
        return True
    return False


def is_api_contract_doc(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if PurePosixPath(normalized).suffix not in DOC_EXTENSIONS:
        return False
    return any(token in normalized for token in ("api", "config", "configuration", "option", "reference", "schema"))


def mass_rename_or_generated_only(files: list[dict[str, Any]]) -> bool:
    if files and all(str(file.get("status") or "") == "renamed" for file in files):
        return True
    return bool(files) and all(is_generated_or_lockfile(str(file.get("filename") or "")) for file in files)


def related_gold_files(
    anchor_path: str,
    changed_files: list[dict[str, Any]],
    patches: dict[str, str],
    max_gold_files: int,
) -> list[dict[str, Any]]:
    related: list[dict[str, Any]] = []
    for file_record in changed_files:
        gold_path = str(file_record.get("filename") or "")
        if gold_path == anchor_path or not is_allowed_gold_file(file_record):
            continue
        signals = ripple_signals(anchor_path, gold_path, patches.get(anchor_path, ""), patches.get(gold_path, ""))
        if not signals:
            continue
        related.append(
            {
                "path": gold_path,
                "signals": signals,
                "additions": int(file_record.get("additions") or 0),
                "deletions": int(file_record.get("deletions") or 0),
                "is_test": is_test_file(gold_path),
            }
        )
    related.sort(key=gold_sort_key)
    return related[:max_gold_files]


def ripple_signals(anchor_path: str, gold_path: str, anchor_patch: str, gold_patch: str) -> list[str]:
    signals: list[str] = []
    anchor_tokens = meaningful_path_tokens(anchor_path)
    gold_tokens = meaningful_path_tokens(gold_path)
    overlap = anchor_tokens & gold_tokens
    if same_directory(anchor_path, gold_path) and is_source_file(gold_path) and not is_test_file(gold_path):
        signals.append("same_component")
    if overlap:
        signals.append("path_token_overlap")
    if patch_mentions_module(anchor_patch, gold_path):
        signals.append("anchor_diff_mentions_gold_module")
    if patch_mentions_module(gold_patch, anchor_path):
        signals.append("gold_diff_mentions_anchor_module")
    diff_overlap = diff_tokens(anchor_patch) & diff_tokens(gold_patch)
    if diff_overlap:
        signals.append("shared_changed_symbol")
    if is_test_file(gold_path) and (overlap or diff_overlap):
        signals.append("optional_test_ripple")
    if is_schema_or_config(gold_path) and (overlap or diff_overlap or patch_mentions_module(anchor_patch, gold_path)):
        signals.append("shared_config_or_schema")
    return dedupe(signals)


def meaningful_path_tokens(path: str) -> set[str]:
    return {token for token in path_tokens(path) if token not in GENERIC_PATH_TOKENS}


def diff_tokens(patch: str) -> set[str]:
    tokens: set[str] = set()
    for line in patch.replace("\r\n", "\n").splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        tokens.update(token.lower() for token in TOKEN_RE.findall(line))
    return {token for token in tokens if len(token) >= 4 and token not in COMMON_DIFF_TOKENS}


def patch_mentions_module(patch: str, path: str) -> bool:
    lowered = patch.lower()
    variants = module_variants(path)
    return any(variant and variant in lowered for variant in variants)


def module_variants(path: str) -> list[str]:
    pure = PurePosixPath(path.replace("\\", "/"))
    stem_parts = [part for part in pure.with_suffix("").parts if part not in {".", ""}]
    dotted = ".".join(stem_parts)
    variants = [dotted, "/".join(stem_parts)]
    if len(stem_parts) > 1:
        variants.extend([".".join(stem_parts[1:]), "/".join(stem_parts[1:])])
    return [variant.lower() for variant in variants if len(variant) >= 3]


def is_schema_or_config(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in SCHEMA_EXTENSIONS | CONFIG_EXTENSIONS


def gold_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item.get("is_test", False)),
        -len(item.get("signals") or []),
        int(item.get("additions") or 0) + int(item.get("deletions") or 0),
        item.get("path", ""),
    )


def anchor_score(anchor: dict[str, Any], related: list[dict[str, Any]]) -> tuple[Any, ...]:
    changes = int(anchor.get("additions") or 0) + int(anchor.get("deletions") or 0)
    non_test_gold = sum(1 for item in related if not item.get("is_test"))
    signal_count = sum(len(item.get("signals") or []) for item in related)
    return (
        int(non_test_gold == 0),
        abs(changes - 20),
        -signal_count,
        str(anchor.get("filename") or ""),
    )


def build_intent(pr: dict[str, Any], details_record: dict[str, Any] | None) -> str:
    parts = [truncate_text(pr.get("title"), 300), clean_pr_body(pr.get("body") or "")]
    for message in commit_messages(details_record):
        parts.append(message)
    text = "\n\n".join(part for part in parts if part).strip()
    return truncate_text(text, 2400)


def commit_messages(details_record: dict[str, Any] | None) -> list[str]:
    messages: list[str] = []
    if not details_record:
        return messages
    for detail in details_record.get("data", []):
        commit = detail.get("commit") or {}
        message = truncate_text(commit.get("message"), 500)
        if message:
            messages.append(message)
    return dedupe(messages)


def sanitize_anchor_diff(patch: str | None, max_lines: int = 180, max_chars: int = 8000) -> str:
    if not patch:
        return ""
    kept: list[str] = []
    for raw_line in patch.replace("\r\n", "\n").splitlines():
        if raw_line.startswith(("diff --git", "index ", "--- ", "+++ ")):
            continue
        if raw_line.startswith(("@@", "+", "-", " ")):
            kept.append(raw_line[:320])
        if len(kept) >= max_lines:
            kept.append("...[truncated]")
            break
    sanitized = "\n".join(kept).strip()
    if len(sanitized) > max_chars:
        return sanitized[:max_chars].rstrip() + "\n...[truncated]"
    return sanitized


def diagnose_edit2ripple_leakage(sample: dict[str, Any]) -> dict[str, Any]:
    query_text = json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True)
    normalized_query = query_text.replace("\\n", "\n")
    lowered = normalized_query.lower()
    query_tokens = {token.lower() for token in TOKEN_RE.findall(normalized_query)}
    gold_files = [str(path) for path in (sample.get("gold") or {}).get("files") or []]
    fix_commit = str((sample.get("metadata") or {}).get("fix_commit") or "")
    fatal: list[str] = []
    nonfatal: list[str] = []
    fatal_hits: dict[str, list[str]] = {}
    nonfatal_hits: dict[str, list[str]] = {}

    if contains_raw_patch_marker(normalized_query):
        fatal.append("raw_full_patch_marker")
    if fix_commit and fix_commit in normalized_query:
        fatal.append("fix_commit")
        fatal_hits["fix_commit"] = [fix_commit]

    for path in gold_files:
        lowered_path = path.lower()
        pure = PurePosixPath(path)
        basename = pure.name.lower()
        stem = pure.stem.lower()
        if lowered_path and lowered_path in lowered:
            fatal.append("gold_path")
            fatal_hits.setdefault("gold_path", []).append(path)
        if basename and basename in lowered:
            fatal.append("gold_basename")
            fatal_hits.setdefault("gold_basename", []).append(path)
        if stem and len(stem) >= 3 and re.search(rf"(?<![A-Za-z0-9_]){re.escape(stem)}(?![A-Za-z0-9_])", lowered):
            nonfatal.append("gold_stem_hint")
            nonfatal_hits.setdefault(path, []).append(stem)
        import_hits = [variant for variant in module_variants(path) if "." in variant and variant in lowered]
        if import_hits:
            fatal.append("gold_import_path")
            fatal_hits.setdefault("gold_import_path", []).extend(import_hits)
        module_hits = sorted((meaningful_path_tokens(path) - {stem}) & query_tokens)
        if module_hits:
            nonfatal.append("gold_module_token_hint")
            nonfatal_hits.setdefault(path, []).extend(module_hits)

    return {
        "fatal": sorted(set(fatal)),
        "nonfatal": sorted(set(nonfatal)),
        "fatal_hits": {key: dedupe(values) for key, values in sorted(fatal_hits.items())},
        "nonfatal_hits": {key: dedupe(values) for key, values in sorted(nonfatal_hits.items())},
    }


def edit2ripple_has_fatal_leakage(sample: dict[str, Any], query_text: str | None = None) -> bool:
    if sample.get("task_type") != "edit2ripple":
        return False
    payload = dict(sample)
    if query_text is not None:
        try:
            payload["query"] = json.loads(query_text) if query_text.strip().startswith("{") else {"text": query_text}
        except json.JSONDecodeError:
            payload["query"] = {"text": query_text}
    return bool(diagnose_edit2ripple_leakage(payload)["fatal"])


def audit_row(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    leakage = metadata.get("leakage") or diagnose_edit2ripple_leakage(sample)
    return {
        "sample_id": sample.get("id", ""),
        "task_type": sample.get("task_type", ""),
        "repo": sample.get("repo", ""),
        "source_pr": metadata.get("source_pr") or metadata.get("pr") or "",
        "anchor_file": ((sample.get("query") or {}).get("anchor_file") or ""),
        "query_excerpt": truncate_text(json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True), 2400),
        "gold_files": "; ".join((sample.get("gold") or {}).get("files") or []),
        "changed_files": "; ".join(metadata.get("changed_files") or []),
        "ripple_relation": "; ".join(metadata.get("ripple_relation") or []),
        "leakage_fatal": "; ".join(leakage.get("fatal") or []),
        "leakage_nonfatal": "; ".join(leakage.get("nonfatal") or []),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "",
    }


def edit2ripple_cluster(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    gold = sample.get("gold") or {}
    query = sample.get("query") or {}
    return json.dumps(
        [
            sample.get("repo", ""),
            metadata.get("source_pr") or metadata.get("pr") or sample.get("id", ""),
            query.get("anchor_file") or "",
            sorted(str(path) for path in gold.get("files") or []),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def sample_sort_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    metadata = sample.get("metadata") or {}
    gold = sample.get("gold") or {}
    return (
        -sum(1 for path in gold.get("files") or [] if not is_test_file(str(path))),
        len(gold.get("files") or []),
        sample.get("repo", ""),
        -int(metadata.get("pr") or 0),
        (sample.get("query") or {}).get("anchor_file") or "",
        sample.get("id", ""),
    )


def write_candidate_outputs(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "edit2ripple.jsonl", samples)


def sample_pr_records(sample_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in sample_paths:
        for sample in read_jsonl(path):
            repo = str(sample.get("repo") or "")
            base_commit = str(sample.get("base_commit") or "")
            metadata = sample.get("metadata") or {}
            pr_url = str(metadata.get("pr_url") or metadata.get("source_pr") or "")
            parsed = parse_github_pr_url(pr_url)
            if parsed:
                parsed_repo, pr_number = parsed
                repo = repo or parsed_repo
            else:
                pr_number = metadata.get("pr")
            if not repo or not base_commit or not pr_number:
                continue
            key = (repo, int(pr_number), base_commit)
            record = records_by_key.setdefault(
                key,
                {
                    "repo": repo,
                    "pr_number": int(pr_number),
                    "base_commit": base_commit,
                    "pr_url": pr_url or f"https://github.com/{repo}/pull/{int(pr_number)}",
                    "sample_ids": [],
                },
            )
            sample_id = str(sample.get("id") or "")
            if sample_id:
                record["sample_ids"].append(sample_id)
    return sorted(records_by_key.values(), key=lambda item: (item["repo"], item["pr_number"], item["base_commit"]))


def sample_commit_records(sample_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sample_paths:
        for sample in read_jsonl(path):
            repo = str(sample.get("repo") or "")
            base_commit = str(sample.get("base_commit") or "")
            gold = sample.get("gold") or {}
            fix_commit = str(gold.get("fix_commit") or "")
            if not repo or not base_commit or not fix_commit:
                continue
            metadata = sample.get("metadata") or {}
            pr_url = str(metadata.get("pr_url") or metadata.get("source_pr") or "")
            parsed = parse_github_pr_url(pr_url)
            pr_number = int(parsed[1]) if parsed else int(metadata.get("pr") or 0)
            query = sample.get("query") or {}
            key = (repo, base_commit, fix_commit)
            record = records_by_key.setdefault(
                key,
                {
                    "repo": repo,
                    "base_commit": base_commit,
                    "fix_commit": fix_commit,
                    "pr_number": pr_number,
                    "pr_url": pr_url or (f"https://github.com/{repo}/pull/{pr_number}" if pr_number else ""),
                    "title": query.get("pr_title") or query.get("title") or "",
                    "body": query.get("pr_body") or query.get("body") or query.get("raw_signal") or "",
                    "created_at": metadata.get("created_at") or "",
                    "merged_at": metadata.get("merged_at") or "",
                    "sample_ids": [],
                },
            )
            if not record.get("title") and (query.get("pr_title") or query.get("title")):
                record["title"] = query.get("pr_title") or query.get("title")
            if not record.get("body") and (query.get("pr_body") or query.get("body")):
                record["body"] = query.get("pr_body") or query.get("body")
            sample_id = str(sample.get("id") or "")
            if sample_id:
                record["sample_ids"].append(sample_id)
    return sorted(records_by_key.values(), key=lambda item: (item["repo"], item["pr_number"], item["base_commit"], item["fix_commit"]))


def parse_github_pr_url(url: str) -> tuple[str, int] | None:
    match = re.search(r"github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)", url)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def fetch_pr_inputs_from_github(
    api: Any,
    repo: str,
    pr_number: int,
    base_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    owner, name = repo.split("/", 1)
    detail = api.get(f"/repos/{owner}/{name}/pulls/{pr_number}").body
    pr = rest_pull_to_pr(detail)
    pr["baseRefOid"] = base_commit
    files = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/files")
    commits = api.paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/commits")
    files_record = {"repo": repo, "pr_number": pr_number, "type": "pull_files", "data": files}
    details_record = {
        "repo": repo,
        "pr_number": pr_number,
        "type": "commit_details",
        "data": [
            {
                "sha": commit.get("sha"),
                "commit": commit.get("commit") or {},
                "files": [],
            }
            for commit in commits
        ],
    }
    return pr, files_record, details_record


def rest_pull_to_pr(pull: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pull.get("number"),
        "title": pull.get("title"),
        "body": pull.get("body"),
        "url": pull.get("html_url") or pull.get("url"),
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


def ensure_git_repo(repo: str, repos_dir: Path, remote_base: str = "https://github.com", blob_filter: bool = True) -> Path:
    repos_dir.mkdir(parents=True, exist_ok=True)
    bare_repo = repos_dir / f"{repo_slug(repo)}.git"
    if bare_repo.exists():
        return bare_repo
    url = f"{remote_base.rstrip('/')}/{repo}.git"
    args = ["git", "clone", "--bare"]
    if blob_filter:
        args.append("--filter=blob:none")
    args.extend([url, str(bare_repo)])
    run_git(args, cwd=None)
    return bare_repo


def ensure_git_commits(bare_repo: Path, repo: str, commits: Iterable[str], blob_filter: bool = True) -> None:
    missing = [commit for commit in sorted(set(commits)) if commit and not git_has_commit(bare_repo, commit)]
    if not missing:
        return
    for commit in missing:
        args = ["git", "-C", str(bare_repo), "fetch", "--depth=1"]
        if blob_filter:
            args.append("--filter=blob:none")
        args.extend(["origin", commit])
        run_git(args, cwd=None)
    still_missing = [commit for commit in missing if not git_has_commit(bare_repo, commit)]
    if still_missing:
        raise RuntimeError(f"Missing commits in {repo}: {', '.join(still_missing)}")


def git_has_commit(bare_repo: Path, commit: str) -> bool:
    result = run_git(["git", "-C", str(bare_repo), "cat-file", "-e", f"{commit}^{{commit}}"], cwd=None, check=False)
    return result.returncode == 0


def build_git_diff_inputs(bare_repo: Path, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = str(record["repo"])
    base_commit = str(record["base_commit"])
    fix_commit = str(record["fix_commit"])
    pr_number = int(record.get("pr_number") or 0)
    changed_files = git_changed_files(bare_repo, base_commit, fix_commit)
    files_record = {"repo": repo, "pr_number": pr_number, "type": "git_diff_files", "data": changed_files}
    commit_message = git_commit_message(bare_repo, fix_commit)
    details_record = {
        "repo": repo,
        "pr_number": pr_number,
        "type": "git_diff_commit_details",
        "data": [{"sha": fix_commit, "commit": {"message": commit_message}, "files": changed_files}],
    }
    pr = {
        "number": pr_number,
        "title": record.get("title") or first_line(commit_message),
        "body": record.get("body") or commit_message,
        "url": record.get("pr_url") or "",
        "createdAt": record.get("created_at") or "",
        "mergedAt": record.get("merged_at") or "",
        "baseRefOid": base_commit,
        "mergeCommit": {"oid": fix_commit},
        "changedFiles": len(changed_files),
    }
    return pr, files_record, details_record


def git_changed_files(bare_repo: Path, base_commit: str, fix_commit: str) -> list[dict[str, Any]]:
    name_status = run_git(
        ["git", "-C", str(bare_repo), "diff", "--name-status", "--find-renames", base_commit, fix_commit],
        cwd=None,
    ).stdout.splitlines()
    numstat = git_numstat(bare_repo, base_commit, fix_commit)
    files: list[dict[str, Any]] = []
    for line in name_status:
        if not line.strip():
            continue
        parts = line.split("\t")
        status_code = parts[0]
        status = git_status_name(status_code)
        if status == "renamed" and len(parts) >= 3:
            filename = parts[2]
            previous_filename = parts[1]
        elif len(parts) >= 2:
            filename = parts[1]
            previous_filename = None
        else:
            continue
        additions, deletions = numstat.get(filename, (0, 0))
        patch = "" if status in {"removed", "renamed"} else git_file_patch(bare_repo, base_commit, fix_commit, filename)
        files.append(
            {
                "filename": filename,
                "previous_filename": previous_filename,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "changes": additions + deletions,
                "patch": patch,
            }
        )
    return files


def git_numstat(bare_repo: Path, base_commit: str, fix_commit: str) -> dict[str, tuple[int, int]]:
    output = run_git(["git", "-C", str(bare_repo), "diff", "--numstat", base_commit, fix_commit], cwd=None).stdout
    stats: dict[str, tuple[int, int]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        additions = int(parts[0]) if parts[0].isdigit() else 0
        deletions = int(parts[1]) if parts[1].isdigit() else 0
        path = parts[-1]
        if " => " in path:
            path = path.split(" => ", 1)[1].strip("{}")
        stats[path] = (additions, deletions)
    return stats


def git_file_patch(bare_repo: Path, base_commit: str, fix_commit: str, path: str) -> str:
    return run_git(
        ["git", "-C", str(bare_repo), "diff", "--unified=80", base_commit, fix_commit, "--", path],
        cwd=None,
    ).stdout


def git_commit_message(bare_repo: Path, commit: str) -> str:
    return run_git(["git", "-C", str(bare_repo), "log", "-1", "--format=%B", commit], cwd=None).stdout.strip()


def git_status_name(status_code: str) -> str:
    if status_code.startswith("A"):
        return "added"
    if status_code.startswith("D"):
        return "removed"
    if status_code.startswith("R"):
        return "renamed"
    if status_code.startswith("C"):
        return "copied"
    return "modified"


def first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def run_git(args: list[str], cwd: Path | None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result


def load_audited_ids(audit_path: Path | None) -> set[str]:
    if not audit_path or not audit_path.exists():
        return set()
    if audit_path.suffix.lower() == ".csv":
        with audit_path.open("r", encoding="utf-8", newline="") as handle:
            return {str(row.get("sample_id")) for row in csv.DictReader(handle) if row.get("sample_id")}
    return {str(row.get("sample_id")) for row in read_jsonl(audit_path) if row.get("sample_id")}


def load_audit_verdicts(audit_path: Path | None) -> dict[str, str]:
    if not audit_path or not audit_path.exists():
        return {}
    rows: Iterable[dict[str, Any]]
    if audit_path.suffix.lower() == ".csv":
        with audit_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = read_jsonl(audit_path)
    verdicts: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        verdict = normalized_verdict(row.get("verdict"))
        if sample_id and verdict:
            verdicts[sample_id] = verdict
    return verdicts


def normalized_verdict(value: Any) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in {"valid", "noisy", "leaked", "ambiguous"}:
        return verdict
    if verdict in {"", "pending", "todo", "tbd"}:
        return ""
    return verdict


def edit2ripple_pr_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    return json.dumps(
        [sample.get("repo", ""), metadata.get("source_pr") or metadata.get("pr") or sample.get("id", "")],
        ensure_ascii=False,
        sort_keys=True,
    )


def relation_strata(gold_files: list[str], relations: list[str]) -> list[str]:
    strata: list[str] = []
    non_test_gold = [path for path in gold_files if not is_test_file(path)]
    if non_test_gold and "same_component" in relations:
        strata.append("source_to_source")
    if "shared_config_or_schema" in relations:
        strata.append("config_registry")
    if "shared_changed_symbol" in relations:
        strata.append("interface_type")
    if "anchor_diff_mentions_gold_module" in relations or "gold_diff_mentions_anchor_module" in relations:
        strata.append("caller_callee")
    if any(is_test_file(path) for path in gold_files):
        strata.append("optional_test")
    if not strata:
        strata.append("other")
    return dedupe(strata)


def edit2ripple_pilot_status(gates: dict[str, Any], audited_total: int) -> str:
    hard_gate_values = [
        gates["fatal_leakage_zero"],
        gates["anchor_not_test"],
        gates["every_sample_has_gold"],
        gates["every_gold_in_corpus"],
        gates["changed_file_range_ok"],
        gates["per_pr_candidate_limit_ok"],
        gates["test_only_gold_controlled"],
    ]
    if any(value is False for value in hard_gate_values):
        return "failed"
    if audited_total == 0 or gates["valid_rate_ok"] is None or not gates["audit_complete"]:
        return "needs_audit"
    if not gates["valid_rate_ok"]:
        return "failed"
    if not gates["sample_count_ok"]:
        return "needs_more_samples"
    return "ready"


def render_edit2ripple_pilot_markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# edit2ripple Pilot Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Samples: `{report['total']}`",
        f"- Audited samples: `{report['audited_total']}`",
        f"- Valid rate: `{format_optional_rate(report['valid_rate'])}`",
        f"- Test-only gold ratio: `{format_optional_rate(report['test_only_gold_ratio'])}`",
        "",
        "## Gates",
    ]
    for key, value in gates.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Distribution",
            f"- Verdicts: `{json.dumps(report['verdicts'], sort_keys=True)}`",
            f"- Relations: `{json.dumps(report['relations'], sort_keys=True)}`",
            f"- Strata: `{json.dumps(report['strata'], sort_keys=True)}`",
            "",
            "## Failures",
        ]
    )
    failures = report["failures"]
    lines.append(f"- Fatal leakage samples: `{len(failures['fatal_leakage'])}`")
    lines.append(f"- Test-anchor samples: `{len(failures.get('test_anchor', []))}`")
    lines.append(f"- Missing-gold samples: `{len(failures['missing_gold'])}`")
    lines.append(f"- No-gold samples: `{len(failures['no_gold'])}`")
    lines.append(f"- Changed-file range violations: `{len(failures['changed_file_range'])}`")
    lines.append(f"- PRs over candidate limit: `{len(failures['pr_over_limit'])}`")
    lines.append(f"- Test-only gold samples: `{len(failures['test_only_gold'])}`")
    lines.append("")
    return "\n".join(lines)


def format_optional_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


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
