from __future__ import annotations

import csv
import hashlib
import html as html_lib
import json
import re
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .github_api import GitHubAPI
from .io import ensure_parent, read_jsonl, stable_id, truncate_text, utc_now, write_json, write_jsonl

ABSTENTION_REASONS = {
    "flaky_ci",
    "upstream_dependency",
    "external_service",
    "user_error",
    "counterfactual_wrong_repo",
}
ABSTENTION_VERDICTS = {
    "valid_no_gold",
    "has_local_gold",
    "ambiguous",
    "too_easy_irrelevant",
    "misconstructed",
}
COUNTERFACTUAL_AUDIT_FIELDS = (
    "id",
    "repo",
    "base_commit",
    "query_source",
    "query_text",
    "proposed_no_gold_reason",
    "source_url",
    "evidence_summary",
    "source_sample_id",
    "source_repo",
    "source_task_type",
    "pairing_profile",
    "verdict",
    "notes",
    "has_local_gold_files",
)
ORGANIC_CANDIDATE_FIELDS = (
    "id",
    "repo",
    "source_type",
    "event_url",
    "query_text",
    "matched_keywords",
    "candidate_reason",
    "possible_base_sha",
    "linked_commits",
    "linked_prs",
    "evidence_snippets",
    "verdict",
    "notes",
    "has_local_gold_files",
)
ABSTENTION_AUDIT_PACKET_CSV_FIELDS = (
    "id",
    "repo",
    "base_commit",
    "query_source",
    "query_text",
    "proposed_no_gold_reason",
    "source_url",
    "resolution_comments",
    "rerun_status",
    "linked_commits",
    "why_no_local_fix",
    "evidence_snippets",
    "source_sample_id",
    "source_repo",
    "source_task_type",
    "pairing_profile",
    "verdict",
    "notes",
    "has_local_gold_files",
)
ABSTENTION_AUDIT_WORKLIST_FIELDS = (
    "audit_order",
    "audit_priority",
    "pool",
    "review_focus",
    "allowed_verdicts",
    *ABSTENTION_AUDIT_PACKET_CSV_FIELDS,
)
ORGANIC_KEYWORDS = (
    "flaky",
    "rerun",
    "infra",
    "ci failure",
    "transient",
    "timeout",
    "timed out",
    "network",
    "github actions",
    "rate limit",
)
RESOLUTION_ANSWER_HINTS: dict[str, tuple[str, ...]] = {
    "flaky_ci": (
        "same head_sha later passed",
        "same sha later passed",
        "same check later passed",
        "rerun passed",
        "re-run passed",
        "transient failure",
        "flaky failure",
        "infra issue",
        "no code change",
    ),
    "external_service": (
        "external service issue",
        "provider issue",
        "credential issue",
        "environment issue",
        "local setup issue",
        "cannot reproduce",
        "can't reproduce",
        "works as intended",
    ),
    "user_error": (
        "not a bug",
        "user error",
        "works as intended",
        "expected behavior",
    ),
    "counterfactual_wrong_repo": (
        "counterfactual",
        "wrong repo",
        "wrong repository",
        "different repository",
    ),
}
ISSUE_REASON_KEYWORDS: dict[str, tuple[str, ...]] = {
    "upstream_dependency": (
        "upstream bug",
        "fixed upstream",
        "resolved upstream",
        "upstream issue",
        "upstream dependency",
        "dependency bug",
        "not our bug",
        "third-party",
        "third party",
        "library issue",
        "waiting on upstream",
        "blocked by https://github.com",
    ),
    "external_service": (
        "credential",
        "credentials",
        "token",
        "permission",
        "network",
        "dns",
        "outage",
        "service unavailable",
        "rate limit",
        "environment",
        "local setup",
        "cannot reproduce",
        "can't reproduce",
    ),
    "user_error": (
        "invalid",
        "not a bug",
        "user error",
        "misconfiguration",
        "usage",
        "works as intended",
        "expected behavior",
    ),
}
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
LOCAL_FIX_PATH_PARTS = {
    ".github",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
}


@dataclass(frozen=True)
class CorpusPair:
    repo: str
    base_commit: str
    chunks_path: str
    language: str
    domain: str

    @property
    def profile(self) -> str:
        return f"{self.language}:{self.domain}"


REPO_PROFILES: dict[str, tuple[str, str]] = {
    "HypothesisWorks/hypothesis": ("python", "testing"),
    "agronholm/anyio": ("python", "async"),
    "astral-sh/ruff": ("rust", "linting"),
    "caddyserver/caddy": ("go", "web"),
    "clap-rs/clap": ("rust", "cli"),
    "encode/uvicorn": ("python", "web"),
    "eslint/eslint": ("javascript", "linting"),
    "etcd-io/etcd": ("go", "database"),
    "fastapi/fastapi": ("python", "web"),
    "gin-gonic/gin": ("go", "web"),
    "huggingface/diffusers": ("python", "ml"),
    "huggingface/transformers": ("python", "ml"),
    "ipython/ipython": ("python", "tooling"),
    "microsoft/playwright": ("javascript", "testing"),
    "mockito/mockito": ("java", "testing"),
    "numpy/numpy": ("python", "data"),
    "pallets/click": ("python", "cli"),
    "pre-commit/pre-commit": ("python", "tooling"),
    "pydantic/pydantic": ("python", "data"),
    "pypa/pip": ("python", "packaging"),
    "pytest-dev/pytest": ("python", "testing"),
    "python-trio/trio": ("python", "async"),
    "python/mypy": ("python", "typing"),
    "scrapy/scrapy": ("python", "web"),
    "spring-projects/spring-boot": ("java", "web"),
    "tokio-rs/tokio": ("rust", "async"),
    "tox-dev/tox": ("python", "testing"),
    "vitejs/vite": ("javascript", "frontend"),
    "vuejs/core": ("javascript", "frontend"),
}

PATH_FIELD_NAMES = {
    "changed_file",
    "changed_files",
    "diff_hunk_context",
    "given_file",
    "implementation_files",
    "path",
    "paths",
    "trace_paths",
}
TEXT_FIELD_NAMES = (
    "pr_title",
    "title",
    "issue_title",
    "review_comment",
    "comment",
    "raw_signal",
    "failure_excerpt",
    "error_excerpt",
    "test_names",
    "pr_body",
    "issue_body",
    "changed_file_summary",
)
PATH_LIKE_RE = re.compile(r"(?:^|[\s\"'`])(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")


def mine_abstention_counterfactuals(
    sample_paths: Iterable[Path],
    corpus_manifest_path: Path,
    out_dir: Path,
    audit_dir: Path,
    limit: int = 80,
    max_per_wrong_repo: int = 8,
    max_per_source_repo: int = 8,
    pairs_per_sample: int = 1,
) -> dict[str, Any]:
    corpus_pairs = load_corpus_pairs(corpus_manifest_path)
    samples = [sample for path in sample_paths for sample in read_jsonl(path)]
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dropped = Counter()
    per_wrong_repo: Counter[str] = Counter()
    per_source_repo: Counter[str] = Counter()
    selected_ids: set[str] = set()

    for sample in samples:
        if len(selected) >= limit:
            break
        source_repo = str(sample.get("repo") or "")
        if per_source_repo[source_repo] >= max_per_source_repo:
            dropped["source_repo_cap"] += 1
            continue
        source_profile = profile_for_sample(sample)
        query_text, query_drop_reason = counterfactual_query_text(sample)
        if query_drop_reason:
            dropped[query_drop_reason] += 1
            continue
        if query_mentions_repo_identity(query_text, source_repo):
            dropped["source_identity_in_query"] += 1
            continue
        wrong_pairs = choose_wrong_pairs(
            sample=sample,
            source_profile=source_profile,
            corpus_pairs=corpus_pairs,
            per_wrong_repo=per_wrong_repo,
            max_per_wrong_repo=max_per_wrong_repo,
            limit=max(1, pairs_per_sample),
        )
        if not wrong_pairs:
            dropped["no_plausible_wrong_repo"] += 1
            continue

        sample_selected = 0
        for wrong_pair in wrong_pairs:
            if len(selected) >= limit or per_source_repo[source_repo] >= max_per_source_repo:
                break
            candidate = build_counterfactual_sample(sample, wrong_pair, query_text, source_profile)
            if candidate["id"] in selected_ids:
                dropped["duplicate_candidate_id"] += 1
                continue
            selected_ids.add(candidate["id"])
            selected.append(candidate)
            audit_rows.append(counterfactual_audit_row(candidate))
            per_wrong_repo[wrong_pair.repo] += 1
            per_source_repo[source_repo] += 1
            sample_selected += 1
        if not sample_selected:
            dropped["duplicate_or_capped_pairs"] += 1

    candidate_path = out_dir / "counterfactual_wrong_repo_candidates.jsonl"
    audit_jsonl_path = audit_dir / "abstention_audit_packet.jsonl"
    audit_csv_path = audit_dir / "abstention_audit_packet.csv"
    report_json_path = audit_dir / "counterfactual_wrong_repo_report.json"
    report_md_path = audit_dir / "counterfactual_wrong_repo_report.md"
    write_jsonl(candidate_path, selected)
    write_jsonl(audit_jsonl_path, [audit_packet_from_sample(sample) for sample in selected])
    write_csv(audit_csv_path, audit_rows, COUNTERFACTUAL_AUDIT_FIELDS)

    report = {
        "generated_at": utc_now(),
        "inputs": {
            "samples": [str(path) for path in sample_paths],
            "corpus_manifest": str(corpus_manifest_path),
            "pairs_per_sample": pairs_per_sample,
        },
        "outputs": {
            "candidates": str(candidate_path),
            "audit_jsonl": str(audit_jsonl_path),
            "audit_csv": str(audit_csv_path),
            "json": str(report_json_path),
            "markdown": str(report_md_path),
        },
        "source_samples": len(samples),
        "corpus_pairs": len(corpus_pairs),
        "selected": len(selected),
        "dropped": dict(sorted(dropped.items())),
        "by_source_repo": dict(sorted(Counter(sample["metadata"]["source_repo"] for sample in selected).items())),
        "by_wrong_repo": dict(sorted(Counter(sample["repo"] for sample in selected).items())),
        "by_profile": dict(sorted(Counter(sample["metadata"]["pairing_profile"] for sample in selected).items())),
        "ready_for_manual_audit": len(audit_rows) >= 80,
    }
    write_json(report_json_path, report)
    report_md_path.write_text(render_counterfactual_report(report), encoding="utf-8")
    return report


def mine_abstention_organic_candidates(
    raw_dirs: Iterable[Path],
    out_dir: Path,
    audit_dir: Path,
    repos: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    repo_filter = set(repos or [])
    raw_candidates: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    dropped = Counter()
    scanned_check_groups = 0

    for repo_dir in discover_raw_repo_dirs(raw_dirs):
        repo = repo_from_raw_dir(repo_dir)
        if repo_filter and repo not in repo_filter:
            continue
        pull_requests = load_pull_requests_by_number(repo_dir)
        check_runs = load_check_runs(repo_dir, repo)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for run in check_runs:
            sha = str(run.get("head_sha") or run.get("sha") or "")
            name = str(run.get("name") or run.get("check_name") or "")
            if sha and name:
                grouped[(sha, name)].append(run)
        for (sha, name), runs in sorted(grouped.items()):
            scanned_check_groups += 1
            candidate = organic_candidate_from_check_group(repo, sha, name, runs, pull_requests)
            if not candidate:
                continue
            if candidate["id"] in seen_candidate_ids:
                dropped["duplicate_candidate_id"] += 1
                continue
            seen_candidate_ids.add(candidate["id"])
            raw_candidates.append(candidate)
            drop_reason = organic_prefilter_drop_reason(candidate)
            if drop_reason:
                dropped[drop_reason] += 1
                continue
            prefiltered.append(candidate)
            if limit and len(prefiltered) >= limit:
                break
        if not (limit and len(prefiltered) >= limit):
            issue_rows = load_issues(repo_dir)
            comments_by_issue = load_issue_related_rows(repo_dir / "issue_comments.jsonl")
            events_by_issue = load_issue_related_rows(repo_dir / "issue_events.jsonl")
            for issue_row in issue_rows:
                candidate = organic_candidate_from_issue_row(
                    repo=repo,
                    issue_row=issue_row,
                    comments=comments_by_issue.get(int(issue_row.get("issue_number") or -1), []),
                    events=events_by_issue.get(int(issue_row.get("issue_number") or -1), []),
                )
                if not candidate:
                    continue
                if candidate["id"] in seen_candidate_ids:
                    dropped["duplicate_candidate_id"] += 1
                    continue
                seen_candidate_ids.add(candidate["id"])
                raw_candidates.append(candidate)
                drop_reason = organic_prefilter_drop_reason(candidate)
                if drop_reason:
                    dropped[drop_reason] += 1
                    continue
                prefiltered.append(candidate)
                if limit and len(prefiltered) >= limit:
                    break
        if limit and len(prefiltered) >= limit:
            break

    candidates_path = out_dir / "organic_candidates.jsonl"
    prefiltered_path = out_dir / "organic_prefiltered.jsonl"
    audit_jsonl_path = audit_dir / "organic_abstention_audit_packet.jsonl"
    audit_csv_path = audit_dir / "organic_abstention_audit_packet.csv"
    report_json_path = audit_dir / "organic_candidate_report.json"
    report_md_path = audit_dir / "organic_candidate_report.md"
    write_jsonl(candidates_path, raw_candidates)
    write_jsonl(prefiltered_path, prefiltered)
    packets = [organic_audit_packet(candidate) for candidate in prefiltered]
    write_jsonl(audit_jsonl_path, packets)
    write_csv(audit_csv_path, [organic_audit_row(candidate) for candidate in prefiltered], ORGANIC_CANDIDATE_FIELDS)
    report = {
        "generated_at": utc_now(),
        "inputs": {"raw_dirs": [str(path) for path in raw_dirs], "repos": sorted(repo_filter)},
        "outputs": {
            "candidates": str(candidates_path),
            "prefiltered": str(prefiltered_path),
            "audit_jsonl": str(audit_jsonl_path),
            "audit_csv": str(audit_csv_path),
            "json": str(report_json_path),
            "markdown": str(report_md_path),
        },
        "scanned_check_groups": scanned_check_groups,
        "raw_candidates": len(raw_candidates),
        "prefiltered": len(prefiltered),
        "dropped": dict(sorted(dropped.items())),
        "by_repo": dict(sorted(Counter(candidate["repo"] for candidate in prefiltered).items())),
        "by_reason": dict(sorted(Counter(candidate["candidate_reason"] for candidate in prefiltered).items())),
        "ready_for_manual_audit": len(prefiltered) >= 50,
    }
    write_json(report_json_path, report)
    report_md_path.write_text(render_organic_report(report), encoding="utf-8")
    return report


def merge_abstention_audit_packets(
    packet_paths: Iterable[Path],
    out_jsonl: Path,
    out_csv: Path | None = None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: list[dict[str, str]] = []
    missing_id_rows: list[dict[str, Any]] = []
    inputs = list(packet_paths)

    for path in inputs:
        for index, packet in enumerate(read_jsonl(path), start=1):
            packet_id = str(packet.get("id") or "")
            if not packet_id:
                missing_id_rows.append({"path": str(path), "row": index})
                continue
            if packet_id in seen:
                duplicates.append({"path": str(path), "row": str(index), "id": packet_id})
                continue
            seen.add(packet_id)
            merged.append(packet)

    write_jsonl(out_jsonl, merged)
    csv_path = out_csv or out_jsonl.with_suffix(".csv")
    write_csv(csv_path, [audit_packet_csv_row(packet) for packet in merged], ABSTENTION_AUDIT_PACKET_CSV_FIELDS)
    invalid_packets = validate_abstention_audit_packets(merged)
    report = {
        "generated_at": utc_now(),
        "inputs": [str(path) for path in inputs],
        "outputs": {"jsonl": str(out_jsonl), "csv": str(csv_path)},
        "total": len(merged),
        "by_query_source": dict(sorted(Counter((packet.get("query") or {}).get("source") or "missing" for packet in merged).items())),
        "by_reason": dict(sorted(Counter(packet.get("proposed_no_gold_reason") or "missing" for packet in merged).items())),
        "duplicates": duplicates,
        "missing_id_rows": missing_id_rows,
        "invalid_packets": invalid_packets,
        "ready_for_manual_audit": len(merged) >= 80 and not missing_id_rows and not invalid_packets,
    }
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_audit_packet_merge_report(report), encoding="utf-8")
    return report


def report_abstention_crawling_status(
    prefiltered_paths: Iterable[Path],
    audit_packet_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    prefiltered_by_path: dict[str, int] = {}
    prefiltered_rows: list[dict[str, Any]] = []
    for path in prefiltered_paths:
        rows = read_jsonl(path)
        prefiltered_by_path[str(path)] = len(rows)
        prefiltered_rows.extend(rows)

    packet_rows = read_jsonl(audit_packet_path) if audit_packet_path.exists() else []
    invalid_packets = validate_abstention_audit_packets(packet_rows)
    pool_counts = Counter(pool_for_candidate(row) for row in prefiltered_rows)
    packet_pool_counts = Counter(pool_for_packet(row) for row in packet_rows)
    counterfactual_diagnostics = counterfactual_packet_diagnostics(packet_rows)
    packet_ids = {str(packet.get("id") or "") for packet in packet_rows if packet.get("id")}
    prefiltered_ids = {str(row.get("id") or "") for row in prefiltered_rows if row.get("id")}
    missing_packet_for_prefiltered = sorted(prefiltered_ids - packet_ids)
    gates = {
        "prefiltered_candidates_ge_150": len(prefiltered_rows) >= 150,
        "audit_packet_candidates_ge_80": len(packet_rows) >= 80,
        "every_packet_has_resolution_evidence": all(packet_has_resolution_evidence(packet) for packet in packet_rows),
        "no_packet_query_contains_resolution_answer": not any(packet_query_contains_resolution_answer(packet) for packet in packet_rows),
        "counterfactual_queries_avoid_source_identity": not counterfactual_diagnostics["source_identity_leaks"],
        "counterfactual_pairing_metadata_present": not counterfactual_diagnostics["missing_pairing_metadata"],
        "prefiltered_ids_have_audit_packets": not missing_packet_for_prefiltered,
        "organic_and_counterfactual_reported_separately": bool(pool_counts or packet_pool_counts),
        "audit_packets_schema_valid": not invalid_packets,
    }
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "prefiltered": prefiltered_by_path,
            "audit_packet": str(audit_packet_path),
        },
        "prefiltered_candidates": len(prefiltered_rows),
        "audit_packet_candidates": len(packet_rows),
        "prefiltered_by_pool": dict(sorted(pool_counts.items())),
        "audit_packets_by_pool": dict(sorted(packet_pool_counts.items())),
        "audit_packets_by_reason": dict(sorted(Counter(packet.get("proposed_no_gold_reason") or "missing" for packet in packet_rows).items())),
        "counterfactual_diagnostics": counterfactual_diagnostics,
        "missing_packet_for_prefiltered": missing_packet_for_prefiltered,
        "invalid_packets": invalid_packets,
        "gates": gates,
        "status": "ready_for_manual_audit" if all(gates.values()) else "not_ready",
    }
    if out_path:
        write_json(out_path, report)
        out_path.with_suffix(".md").write_text(render_crawling_status_report(report), encoding="utf-8")
    return report


def export_abstention_clean(
    audit_packet_path: Path,
    audit_path: Path,
    out_dir: Path,
    report_out: Path | None = None,
    max_samples: int | None = None,
    max_counterfactual_share: float | None = None,
) -> dict[str, Any]:
    packets = {row["id"]: row for row in read_jsonl(audit_packet_path) if row.get("id")}
    audit_rows = read_audit_verdict_rows(audit_path)
    selected: list[dict[str, Any]] = []
    verdicts = Counter()
    missing_packet_ids: list[str] = []

    for row in audit_rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        verdict = normalize_abstention_verdict(row.get("verdict"))
        if not sample_id:
            continue
        verdicts[verdict or "missing"] += 1
        packet = packets.get(sample_id)
        if not packet:
            missing_packet_ids.append(sample_id)
            continue
        if verdict != "valid_no_gold":
            continue
        sample = clean_sample_from_packet(packet, row)
        selected.append(sample)

    selected_before_balancing = len(selected)
    selected = balance_clean_selection(selected, max_samples=max_samples, max_counterfactual_share=max_counterfactual_share)
    sample_path = out_dir / "abstention.jsonl"
    write_jsonl(sample_path, selected)
    report = report_abstention_pilot(
        sample_paths=[sample_path],
        out_path=report_out,
        audit_packet_path=audit_packet_path,
        extra={
            "audit_path": str(audit_path),
            "audit_verdicts": dict(sorted(verdicts.items())),
            "missing_packet_ids": missing_packet_ids,
            "selected_before_balancing": selected_before_balancing,
            "selection": {
                "max_samples": max_samples,
                "max_counterfactual_share": max_counterfactual_share,
            },
        },
    )
    return report


def report_abstention_audit(
    audit_packet_path: Path,
    audit_path: Path,
    out_path: Path | None = None,
    max_samples: int | None = None,
    max_counterfactual_share: float | None = None,
) -> dict[str, Any]:
    packets = {row["id"]: row for row in read_jsonl(audit_packet_path) if row.get("id")}
    audit_rows = read_audit_verdict_rows(audit_path)
    verdicts = Counter()
    reviewed_ids: set[str] = set()
    missing_packet_ids: list[str] = []
    invalid_verdicts: Counter[str] = Counter()
    valid_samples: list[dict[str, Any]] = []
    by_pool: dict[str, Counter[str]] = defaultdict(Counter)
    by_reason: dict[str, Counter[str]] = defaultdict(Counter)

    for row in audit_rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            continue
        verdict = normalize_abstention_verdict(row.get("verdict"))
        if not verdict:
            verdict = "missing"
        verdicts[verdict] += 1
        packet = packets.get(sample_id)
        if not packet:
            missing_packet_ids.append(sample_id)
            continue
        if verdict != "missing":
            reviewed_ids.add(sample_id)
        if verdict not in ABSTENTION_VERDICTS and verdict != "missing":
            invalid_verdicts[verdict] += 1
        pool = pool_for_packet(packet)
        reason = str(packet.get("proposed_no_gold_reason") or "missing")
        by_pool[pool][verdict] += 1
        by_reason[reason][verdict] += 1
        if verdict == "valid_no_gold":
            valid_samples.append(clean_sample_from_packet(packet, row))

    balanced = balance_clean_selection(valid_samples, max_samples=max_samples, max_counterfactual_share=max_counterfactual_share)
    reviewed_total = sum(verdicts.get(verdict, 0) for verdict in ABSTENTION_VERDICTS)
    valid_total = verdicts.get("valid_no_gold", 0)
    valid_rate = valid_total / reviewed_total if reviewed_total else None
    balanced_counterfactual = sum(1 for sample in balanced if (sample.get("gold") or {}).get("reason") == "counterfactual_wrong_repo")
    balanced_organic = len(balanced) - balanced_counterfactual
    balanced_counterfactual_share = balanced_counterfactual / len(balanced) if balanced else None
    pending_packet_ids = sorted(set(packets) - reviewed_ids)
    gates = {
        "all_packets_reviewed": not pending_packet_ids,
        "valid_rate_ge_90": bool(valid_rate is not None and valid_rate >= 0.90),
        "invalid_verdicts_zero": not invalid_verdicts,
        "missing_packet_ids_zero": not missing_packet_ids,
        "balanced_clean_ge_50": len(balanced) >= 50,
        "balanced_counterfactual_no_more_than_half": bool(
            balanced_counterfactual_share is not None and balanced_counterfactual_share <= 0.5
        ),
    }
    report = {
        "generated_at": utc_now(),
        "inputs": {"audit_packet": str(audit_packet_path), "audit": str(audit_path)},
        "audit_packet_candidates": len(packets),
        "audit_rows": len(audit_rows),
        "reviewed": reviewed_total,
        "pending": len(pending_packet_ids),
        "pending_packet_ids": pending_packet_ids[:50],
        "pending_packet_ids_truncated": max(0, len(pending_packet_ids) - 50),
        "verdicts": dict(sorted(verdicts.items())),
        "invalid_verdicts": dict(sorted(invalid_verdicts.items())),
        "missing_packet_ids": missing_packet_ids,
        "valid_rate": valid_rate,
        "by_pool": {key: dict(sorted(value.items())) for key, value in sorted(by_pool.items())},
        "by_reason": {key: dict(sorted(value.items())) for key, value in sorted(by_reason.items())},
        "balanced_clean_preview": {
            "total": len(balanced),
            "organic": balanced_organic,
            "counterfactual": balanced_counterfactual,
            "counterfactual_share": balanced_counterfactual_share,
            "max_samples": max_samples,
            "max_counterfactual_share": max_counterfactual_share,
        },
        "gates": gates,
        "status": "ready_to_export" if all(gates.values()) else "not_ready",
    }
    if out_path:
        write_json(out_path, report)
        out_path.with_suffix(".md").write_text(render_abstention_audit_report(report), encoding="utf-8")
    return report


def apply_abstention_audit_verdicts(
    source_audit_path: Path | Iterable[Path],
    target_audit_path: Path,
    out_path: Path,
    report_out: Path | None = None,
    overwrite: bool = False,
    require_complete: bool = False,
) -> dict[str, Any]:
    source_paths = list(source_audit_path) if not isinstance(source_audit_path, Path) else [source_audit_path]
    source_rows = [row for path in source_paths for row in read_audit_verdict_rows(path)]
    target_rows, target_fields = read_csv_rows_with_fields(target_audit_path)
    target_by_id: dict[str, dict[str, Any]] = {}
    duplicate_target_ids: list[str] = []
    for row in target_rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in target_by_id:
            duplicate_target_ids.append(sample_id)
            continue
        target_by_id[sample_id] = row

    seen_source_ids: set[str] = set()
    duplicate_source_ids: list[str] = []
    invalid_verdicts: list[dict[str, str]] = []
    unknown_ids: list[str] = []
    conflicts: list[dict[str, str]] = []
    applied_ids: list[str] = []
    ignored_blank = 0

    for row in source_rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in seen_source_ids:
            duplicate_source_ids.append(sample_id)
            continue
        seen_source_ids.add(sample_id)
        verdict = normalize_abstention_verdict(row.get("verdict"))
        if not verdict:
            ignored_blank += 1
            continue
        if verdict not in ABSTENTION_VERDICTS:
            invalid_verdicts.append({"id": sample_id, "verdict": verdict})
            continue
        target = target_by_id.get(sample_id)
        if target is None:
            unknown_ids.append(sample_id)
            continue
        existing_verdict = normalize_abstention_verdict(target.get("verdict"))
        if existing_verdict and existing_verdict != verdict and not overwrite:
            conflicts.append({"id": sample_id, "existing": existing_verdict, "incoming": verdict})
            continue
        target["verdict"] = verdict
        target["notes"] = str(row.get("notes") or "")
        target["has_local_gold_files"] = str(row.get("has_local_gold_files") or "")
        applied_ids.append(sample_id)

    write_csv(out_path, target_rows, tuple(target_fields))
    remaining_missing = sum(1 for row in target_rows if not normalize_abstention_verdict(row.get("verdict")))
    verdicts_after = Counter(normalize_abstention_verdict(row.get("verdict")) or "missing" for row in target_rows)
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "source_audits": [str(path) for path in source_paths],
            "source_audit": str(source_paths[0]) if len(source_paths) == 1 else None,
            "target_audit": str(target_audit_path),
        },
        "outputs": {
            "audit": str(out_path),
            "report": str(report_out) if report_out else None,
        },
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "applied": len(applied_ids),
        "applied_ids": applied_ids[:50],
        "applied_ids_truncated": max(0, len(applied_ids) - 50),
        "ignored_blank": ignored_blank,
        "remaining_missing": remaining_missing,
        "verdicts_after": dict(sorted(verdicts_after.items())),
        "invalid_verdicts": invalid_verdicts,
        "unknown_ids": unknown_ids,
        "duplicate_source_ids": duplicate_source_ids,
        "duplicate_target_ids": duplicate_target_ids,
        "conflicts": conflicts,
        "overwrite": overwrite,
        "require_complete": require_complete,
        "gates": {
            "invalid_verdicts_zero": not invalid_verdicts,
            "unknown_ids_zero": not unknown_ids,
            "duplicate_source_ids_zero": not duplicate_source_ids,
            "duplicate_target_ids_zero": not duplicate_target_ids,
            "conflicts_zero": not conflicts,
            "complete_if_required": (remaining_missing == 0) if require_complete else True,
        },
    }
    report["status"] = "ready" if all(report["gates"].values()) else "not_ready"
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_abstention_verdict_apply_report(report), encoding="utf-8")
    return report


def prepare_abstention_audit_worklist(
    audit_packet_path: Path,
    out_csv: Path,
    out_jsonl: Path | None = None,
    report_out: Path | None = None,
    target_size: int = 100,
    max_counterfactual_share: float = 0.5,
) -> dict[str, Any]:
    packets = [packet for packet in read_jsonl(audit_packet_path) if packet.get("id")]
    invalid_packets = validate_abstention_audit_packets(packets)
    organic = stratified_packet_order([packet for packet in packets if pool_for_packet(packet) == "organic_no_gold"])
    counterfactual = stratified_packet_order([packet for packet in packets if pool_for_packet(packet) == "counterfactual_wrong_repo"])

    core_organic_count = min(len(organic), max(0, target_size))
    if max_counterfactual_share <= 0:
        counterfactual_allowed_by_share = 0
    elif max_counterfactual_share >= 1:
        counterfactual_allowed_by_share = len(counterfactual)
    else:
        counterfactual_allowed_by_share = int(
            (max_counterfactual_share / (1.0 - max_counterfactual_share)) * core_organic_count
        )
    core_counterfactual_count = min(
        len(counterfactual),
        max(0, target_size - core_organic_count),
        counterfactual_allowed_by_share,
    )
    core_packets = organic[:core_organic_count] + counterfactual[:core_counterfactual_count]
    core_ids = {str(packet.get("id")) for packet in core_packets}
    reserve_packets = [packet for packet in stratified_packet_order(packets) if str(packet.get("id")) not in core_ids]

    rows = [
        audit_worklist_row(packet, index, "core")
        for index, packet in enumerate(core_packets, start=1)
    ]
    rows.extend(
        audit_worklist_row(packet, index, "reserve")
        for index, packet in enumerate(reserve_packets, start=len(rows) + 1)
    )
    write_csv(out_csv, rows, ABSTENTION_AUDIT_WORKLIST_FIELDS)
    if out_jsonl:
        write_jsonl(out_jsonl, rows)

    core_counterfactual_share = core_counterfactual_count / len(core_packets) if core_packets else None
    report = {
        "generated_at": utc_now(),
        "inputs": {"audit_packet": str(audit_packet_path)},
        "outputs": {
            "csv": str(out_csv),
            "jsonl": str(out_jsonl) if out_jsonl else None,
            "report": str(report_out) if report_out else None,
        },
        "total_packets": len(packets),
        "invalid_packets": invalid_packets,
        "target_size": target_size,
        "max_counterfactual_share": max_counterfactual_share,
        "core": {
            "total": len(core_packets),
            "organic": core_organic_count,
            "counterfactual": core_counterfactual_count,
            "counterfactual_share": core_counterfactual_share,
            "by_reason": dict(sorted(Counter(packet.get("proposed_no_gold_reason") or "missing" for packet in core_packets).items())),
            "by_repo": dict(sorted(Counter(packet.get("repo") or "missing" for packet in core_packets).items())),
        },
        "reserve": {
            "total": len(reserve_packets),
            "organic": sum(1 for packet in reserve_packets if pool_for_packet(packet) == "organic_no_gold"),
            "counterfactual": sum(1 for packet in reserve_packets if pool_for_packet(packet) == "counterfactual_wrong_repo"),
            "by_reason": dict(sorted(Counter(packet.get("proposed_no_gold_reason") or "missing" for packet in reserve_packets).items())),
        },
        "all_packets": {
            "by_pool": dict(sorted(Counter(pool_for_packet(packet) for packet in packets).items())),
            "by_reason": dict(sorted(Counter(packet.get("proposed_no_gold_reason") or "missing" for packet in packets).items())),
        },
        "audit_instructions": {
            "allowed_verdicts": sorted(ABSTENTION_VERDICTS),
            "manual_verdicts_required_for_complete_packet_audit": len(packets),
            "clean_export_verdict": "valid_no_gold",
            "do_not_fill_automatically": True,
        },
        "gates": {
            "packet_schema_valid": not invalid_packets,
            "core_target_ge_80": len(core_packets) >= 80,
            "core_counterfactual_no_more_than_half": bool(
                core_counterfactual_share is not None and core_counterfactual_share <= max_counterfactual_share
            ),
            "core_has_organic_and_counterfactual": core_organic_count > 0 and core_counterfactual_count > 0,
        },
        "status": "ready_for_manual_audit" if not invalid_packets and packets else "not_ready",
    }
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_abstention_worklist_report(report), encoding="utf-8")
    return report


def report_abstention_pilot(
    sample_paths: Iterable[Path],
    out_path: Path | None = None,
    audit_packet_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = [sample for path in sample_paths for sample in read_jsonl(path)]
    verdicts = Counter(normalize_abstention_verdict((sample.get("audit") or {}).get("verdict")) or "missing" for sample in samples)
    reasons = Counter((sample.get("gold") or {}).get("reason") or "missing" for sample in samples)
    organic_count = sum(1 for sample in samples if (sample.get("metadata") or {}).get("organic") is True)
    counterfactual_count = sum(1 for sample in samples if (sample.get("gold") or {}).get("reason") == "counterfactual_wrong_repo")
    invalid_rows = validate_abstention_rows(samples)
    has_local_gold_final = sum(1 for sample in samples if normalize_abstention_verdict((sample.get("audit") or {}).get("verdict")) == "has_local_gold")
    valid_total = verdicts.get("valid_no_gold", 0)
    reviewed_total = sum(verdicts.get(verdict, 0) for verdict in ABSTENTION_VERDICTS)
    valid_rate = valid_total / reviewed_total if reviewed_total else None
    packet_rows = read_jsonl(audit_packet_path) if audit_packet_path else []
    counterfactual_share = counterfactual_count / len(samples) if samples else None
    gates = {
        "audit_packet_candidates_ge_80": len(packet_rows) >= 80 if audit_packet_path else None,
        "final_clean_ge_50": len(samples) >= 50,
        "valid_rate_ge_90": bool(valid_rate is not None and valid_rate >= 0.90),
        "has_local_gold_zero": has_local_gold_final == 0,
        "every_sample_has_resolution_evidence": all(bool((sample.get("metadata") or {}).get("evidence_summary")) for sample in samples),
        "no_query_contains_resolution_answer": not any(sample_query_contains_resolution_answer(sample) for sample in samples),
        "organic_and_counterfactual_reported_separately": organic_count > 0 and counterfactual_count > 0,
        "counterfactual_no_more_than_half": bool(counterfactual_share is not None and counterfactual_share <= 0.5),
        "schema_valid": not invalid_rows,
    }
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "samples": [str(path) for path in sample_paths],
            "audit_packet": str(audit_packet_path) if audit_packet_path else None,
        },
        "total": len(samples),
        "audit_packet_candidates": len(packet_rows) if audit_packet_path else None,
        "organic": organic_count,
        "counterfactual": counterfactual_count,
        "counterfactual_share": counterfactual_share,
        "verdicts": dict(sorted(verdicts.items())),
        "reasons": dict(sorted(reasons.items())),
        "valid_rate": valid_rate,
        "has_local_gold_final": has_local_gold_final,
        "invalid_rows": invalid_rows,
        "gates": gates,
        "status": "ready" if all(value is True or value is None for value in gates.values()) else "not_ready",
    }
    if extra:
        report.update(extra)
    if out_path:
        write_json(out_path, report)
        out_path.with_suffix(".md").write_text(render_abstention_pilot_report(report), encoding="utf-8")
    return report


def report_abstention_completion_audit(
    prefiltered_paths: Iterable[Path],
    audit_packet_path: Path,
    audit_path: Path,
    clean_sample_paths: Iterable[Path] | None = None,
    worklist_report_path: Path | None = None,
    crawling_report_path: Path | None = None,
    out_path: Path | None = None,
    max_samples: int = 100,
    max_counterfactual_share: float = 0.5,
) -> dict[str, Any]:
    prefiltered_list = list(prefiltered_paths)
    clean_paths = [path for path in (clean_sample_paths or []) if path.exists()]
    crawling = report_abstention_crawling_status(prefiltered_list, audit_packet_path)
    crawling_stage = (
        json.loads(crawling_report_path.read_text(encoding="utf-8"))
        if crawling_report_path and crawling_report_path.exists()
        else crawling
    )
    audit = report_abstention_audit(
        audit_packet_path,
        audit_path,
        max_samples=max_samples,
        max_counterfactual_share=max_counterfactual_share,
    )
    pilot = report_abstention_pilot(clean_paths, audit_packet_path=audit_packet_path)
    worklist = json.loads(worklist_report_path.read_text(encoding="utf-8")) if worklist_report_path and worklist_report_path.exists() else None
    final_clean_exists = bool(clean_paths)
    final_clean_pending = not final_clean_exists
    reviewed_total = int(audit.get("reviewed") or 0)
    reviewed_pending = int(audit.get("pending") or 0)
    organic_final = int(pilot.get("organic") or 0)
    counterfactual_final = int(pilot.get("counterfactual") or 0)
    total_final = int(pilot.get("total") or 0)

    requirements = [
        completion_requirement(
            "two_candidate_pools_built",
            crawling_stage.get("prefiltered_by_pool", {}).get("organic_no_gold", 0) > 0
            and crawling_stage.get("prefiltered_by_pool", {}).get("counterfactual_wrong_repo", 0) > 0,
            f"prefiltered_by_pool={crawling_stage.get('prefiltered_by_pool')}",
        ),
        completion_requirement(
            "prefiltered_candidates_ge_150",
            bool(crawling_stage.get("gates", {}).get("prefiltered_candidates_ge_150")),
            f"prefiltered_candidates={crawling_stage.get('prefiltered_candidates')}",
        ),
        completion_requirement(
            "audit_packet_candidates_ge_80",
            bool(crawling.get("gates", {}).get("audit_packet_candidates_ge_80")),
            f"audit_packet_candidates={crawling.get('audit_packet_candidates')}",
        ),
        completion_requirement(
            "every_packet_has_resolution_evidence",
            bool(crawling.get("gates", {}).get("every_packet_has_resolution_evidence")),
            "crawling status validates resolution evidence for every packet",
        ),
        completion_requirement(
            "no_packet_query_contains_resolution_answer",
            bool(crawling.get("gates", {}).get("no_packet_query_contains_resolution_answer")),
            "crawling status validates query/evidence separation",
        ),
        completion_requirement(
            "counterfactual_queries_avoid_source_identity",
            bool(crawling.get("gates", {}).get("counterfactual_queries_avoid_source_identity")),
            f"source_identity_leaks={len((crawling.get('counterfactual_diagnostics') or {}).get('source_identity_leaks') or [])}",
        ),
        completion_requirement(
            "counterfactual_pairing_metadata_present",
            bool(crawling.get("gates", {}).get("counterfactual_pairing_metadata_present")),
            f"missing_pairing_metadata={len((crawling.get('counterfactual_diagnostics') or {}).get('missing_pairing_metadata') or [])}",
        ),
        completion_requirement(
            "organic_and_counterfactual_reported_separately",
            bool(crawling.get("gates", {}).get("organic_and_counterfactual_reported_separately")),
            f"audit_packets_by_pool={crawling.get('audit_packets_by_pool')}",
        ),
        completion_requirement(
            "manual_audit_all_packets_reviewed",
            bool(audit.get("gates", {}).get("all_packets_reviewed")),
            f"reviewed={reviewed_total}, pending={reviewed_pending}",
            pending=reviewed_pending > 0,
        ),
        completion_requirement(
            "valid_no_gold_rate_after_audit_ge_90",
            bool(audit.get("gates", {}).get("valid_rate_ge_90")),
            f"valid_rate={audit.get('valid_rate')}",
            pending=reviewed_total == 0,
        ),
        completion_requirement(
            "final_clean_no_gold_samples_ge_50",
            bool(pilot.get("gates", {}).get("final_clean_ge_50")),
            f"final_clean_samples={total_final}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "recommended_final_total_60_to_100",
            60 <= total_final <= 100,
            f"final_clean_samples={total_final}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "recommended_organic_30_to_50",
            30 <= organic_final <= 50,
            f"organic_final={organic_final}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "recommended_counterfactual_30_to_50",
            30 <= counterfactual_final <= 50,
            f"counterfactual_final={counterfactual_final}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "has_local_gold_zero_in_final_clean",
            bool(pilot.get("gates", {}).get("has_local_gold_zero")) and final_clean_exists,
            f"has_local_gold_final={pilot.get('has_local_gold_final')}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "final_every_sample_has_resolution_evidence",
            bool(pilot.get("gates", {}).get("every_sample_has_resolution_evidence")) and final_clean_exists,
            "final pilot validates evidence_summary for every clean sample",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "final_no_query_contains_resolution_answer",
            bool(pilot.get("gates", {}).get("no_query_contains_resolution_answer")) and final_clean_exists,
            "final pilot validates query/evidence separation",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "final_counterfactual_no_more_than_half",
            bool(pilot.get("gates", {}).get("counterfactual_no_more_than_half")),
            f"counterfactual_share={pilot.get('counterfactual_share')}",
            pending=final_clean_pending,
        ),
        completion_requirement(
            "final_schema_valid",
            bool(pilot.get("gates", {}).get("schema_valid")) and final_clean_exists,
            f"invalid_rows={len(pilot.get('invalid_rows') or [])}",
            pending=final_clean_pending,
        ),
    ]
    if worklist is not None:
        requirements.append(
            completion_requirement(
                "manual_audit_worklist_ready",
                worklist.get("status") == "ready_for_manual_audit",
                f"core={worklist.get('core')}",
            )
        )

    statuses = Counter(item["status"] for item in requirements)
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "prefiltered": [str(path) for path in prefiltered_list],
            "audit_packet": str(audit_packet_path),
            "audit": str(audit_path),
            "clean_samples": [str(path) for path in clean_paths],
            "worklist_report": str(worklist_report_path) if worklist_report_path else None,
            "crawling_report": str(crawling_report_path) if crawling_report_path else None,
        },
        "status_counts": dict(sorted(statuses.items())),
        "requirements": requirements,
        "crawling_status": {
            "status": crawling.get("status"),
            "prefiltered_candidates": crawling.get("prefiltered_candidates"),
            "audit_packet_candidates": crawling.get("audit_packet_candidates"),
            "gates": crawling.get("gates"),
            "audit_packets_by_pool": crawling.get("audit_packets_by_pool"),
            "audit_packets_by_reason": crawling.get("audit_packets_by_reason"),
            "counterfactual_diagnostics": crawling.get("counterfactual_diagnostics"),
        },
        "crawling_stage_status": None
        if crawling_stage is crawling
        else {
            "status": crawling_stage.get("status"),
            "prefiltered_candidates": crawling_stage.get("prefiltered_candidates"),
            "audit_packet_candidates": crawling_stage.get("audit_packet_candidates"),
            "gates": crawling_stage.get("gates"),
            "prefiltered_by_pool": crawling_stage.get("prefiltered_by_pool"),
            "audit_packets_by_pool": crawling_stage.get("audit_packets_by_pool"),
            "audit_packets_by_reason": crawling_stage.get("audit_packets_by_reason"),
            "counterfactual_diagnostics": crawling_stage.get("counterfactual_diagnostics"),
        },
        "audit_status": {
            "status": audit.get("status"),
            "reviewed": audit.get("reviewed"),
            "pending": audit.get("pending"),
            "valid_rate": audit.get("valid_rate"),
            "verdicts": audit.get("verdicts"),
            "gates": audit.get("gates"),
            "balanced_clean_preview": audit.get("balanced_clean_preview"),
        },
        "final_clean_status": {
            "exists": final_clean_exists,
            "total": total_final,
            "organic": organic_final,
            "counterfactual": counterfactual_final,
            "counterfactual_share": pilot.get("counterfactual_share"),
            "gates": pilot.get("gates"),
        },
        "status": completion_status(requirements, crawling),
    }
    if out_path:
        write_json(out_path, report)
        out_path.with_suffix(".md").write_text(render_abstention_completion_audit_report(report), encoding="utf-8")
    return report


def finalize_abstention_pilot(
    prefiltered_paths: Iterable[Path],
    audit_packet_path: Path,
    audit_path: Path,
    out_dir: Path,
    report_out: Path,
    audit_report_out: Path,
    pilot_report_out: Path,
    completion_report_out: Path,
    worklist_report_path: Path | None = None,
    crawling_report_path: Path | None = None,
    max_samples: int = 100,
    max_counterfactual_share: float = 0.5,
) -> dict[str, Any]:
    prefiltered_list = list(prefiltered_paths)
    audit_report = report_abstention_audit(
        audit_packet_path=audit_packet_path,
        audit_path=audit_path,
        out_path=audit_report_out,
        max_samples=max_samples,
        max_counterfactual_share=max_counterfactual_share,
    )
    exported = False
    pilot_report: dict[str, Any] | None = None
    clean_sample_paths: list[Path] = []
    if audit_report.get("status") == "ready_to_export":
        pilot_report = export_abstention_clean(
            audit_packet_path=audit_packet_path,
            audit_path=audit_path,
            out_dir=out_dir,
            report_out=pilot_report_out,
            max_samples=max_samples,
            max_counterfactual_share=max_counterfactual_share,
        )
        exported = True
        clean_sample_paths = [out_dir / "abstention.jsonl"]

    completion_report = report_abstention_completion_audit(
        prefiltered_paths=prefiltered_list,
        audit_packet_path=audit_packet_path,
        audit_path=audit_path,
        clean_sample_paths=clean_sample_paths,
        worklist_report_path=worklist_report_path,
        crawling_report_path=crawling_report_path,
        out_path=completion_report_out,
        max_samples=max_samples,
        max_counterfactual_share=max_counterfactual_share,
    )
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "prefiltered": [str(path) for path in prefiltered_list],
            "audit_packet": str(audit_packet_path),
            "audit": str(audit_path),
            "crawling_report": str(crawling_report_path) if crawling_report_path else None,
        },
        "outputs": {
            "clean_dir": str(out_dir),
            "clean_samples": str(out_dir / "abstention.jsonl") if exported else None,
            "audit_report": str(audit_report_out),
            "pilot_report": str(pilot_report_out) if exported else None,
            "completion_report": str(completion_report_out),
            "finalization_report": str(report_out),
        },
        "exported": exported,
        "audit_status": audit_report.get("status"),
        "pilot_status": pilot_report.get("status") if pilot_report else None,
        "completion_status": completion_report.get("status"),
        "audit_gates": audit_report.get("gates"),
        "pilot_gates": pilot_report.get("gates") if pilot_report else None,
        "completion_status_counts": completion_report.get("status_counts"),
        "next_action": finalization_next_action(audit_report, completion_report, exported),
        "status": "complete" if exported and completion_report.get("status") == "complete" else "not_ready",
    }
    write_json(report_out, report)
    report_out.with_suffix(".md").write_text(render_abstention_finalization_report(report), encoding="utf-8")
    return report


def shard_abstention_audit_worklist(
    worklist_path: Path,
    out_dir: Path,
    report_out: Path | None = None,
    shard_size: int = 25,
    priority: str = "core",
) -> dict[str, Any]:
    rows, fields = read_csv_rows_with_fields(worklist_path)
    if priority == "all":
        selected = rows
    else:
        selected = [row for row in rows if str(row.get("audit_priority") or "") == priority]
    selected.sort(key=lambda row: int(str(row.get("audit_order") or "0") or "0"))
    shard_size = max(1, shard_size)
    shards: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, start in enumerate(range(0, len(selected), shard_size), start=1):
        shard_rows = selected[start : start + shard_size]
        shard_name = f"abstention_audit_{priority}_shard_{shard_index:02d}"
        csv_path = out_dir / f"{shard_name}.csv"
        jsonl_path = out_dir / f"{shard_name}.jsonl"
        write_csv(csv_path, shard_rows, tuple(fields))
        write_jsonl(jsonl_path, shard_rows)
        shards.append(
            {
                "index": shard_index,
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "rows": len(shard_rows),
                "first_audit_order": shard_rows[0].get("audit_order") if shard_rows else None,
                "last_audit_order": shard_rows[-1].get("audit_order") if shard_rows else None,
                "by_pool": dict(sorted(Counter(row.get("pool") or "missing" for row in shard_rows).items())),
                "by_reason": dict(sorted(Counter(row.get("proposed_no_gold_reason") or "missing" for row in shard_rows).items())),
            }
        )

    report = {
        "generated_at": utc_now(),
        "inputs": {
            "worklist": str(worklist_path),
            "priority": priority,
            "shard_size": shard_size,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "report": str(report_out) if report_out else None,
        },
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "shards": shards,
        "by_priority": dict(sorted(Counter(row.get("audit_priority") or "missing" for row in rows).items())),
        "selected_by_pool": dict(sorted(Counter(row.get("pool") or "missing" for row in selected).items())),
        "selected_by_reason": dict(sorted(Counter(row.get("proposed_no_gold_reason") or "missing" for row in selected).items())),
        "allowed_verdicts": sorted(ABSTENTION_VERDICTS),
        "gates": {
            "selected_rows_nonzero": bool(selected),
            "all_shards_nonempty": all(shard["rows"] > 0 for shard in shards),
            "priority_present": priority == "all" or any(row.get("audit_priority") == priority for row in rows),
        },
        "status": "ready_for_manual_audit" if selected else "not_ready",
    }
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_abstention_worklist_shard_report(report), encoding="utf-8")
    return report


def report_abstention_shard_progress(
    shard_paths: Iterable[Path],
    out_path: Path | None = None,
) -> dict[str, Any]:
    paths = list(shard_paths)
    shard_reports: list[dict[str, Any]] = []
    seen_verdicts: dict[str, tuple[str, str]] = {}
    duplicate_ids: list[dict[str, str]] = []
    conflicting_verdicts: list[dict[str, str]] = []
    total_rows = 0
    total_reviewed = 0
    total_pending = 0
    all_invalid_verdicts: list[dict[str, str]] = []
    verdicts = Counter()

    for path in paths:
        rows = read_audit_verdict_rows(path)
        shard_invalid: list[dict[str, str]] = []
        shard_verdicts = Counter()
        reviewed = 0
        pending = 0
        for row in rows:
            sample_id = str(row.get("id") or row.get("sample_id") or "")
            verdict = normalize_abstention_verdict(row.get("verdict"))
            if not verdict:
                pending += 1
                shard_verdicts["missing"] += 1
                verdicts["missing"] += 1
            elif verdict not in ABSTENTION_VERDICTS:
                shard_invalid.append({"id": sample_id, "verdict": verdict, "path": str(path)})
                all_invalid_verdicts.append({"id": sample_id, "verdict": verdict, "path": str(path)})
                shard_verdicts[verdict] += 1
                verdicts[verdict] += 1
            else:
                reviewed += 1
                shard_verdicts[verdict] += 1
                verdicts[verdict] += 1
                if sample_id in seen_verdicts:
                    previous_verdict, previous_path = seen_verdicts[sample_id]
                    duplicate_ids.append({"id": sample_id, "first_path": previous_path, "duplicate_path": str(path)})
                    if previous_verdict != verdict:
                        conflicting_verdicts.append(
                            {
                                "id": sample_id,
                                "first_path": previous_path,
                                "first_verdict": previous_verdict,
                                "conflict_path": str(path),
                                "conflict_verdict": verdict,
                            }
                        )
                else:
                    seen_verdicts[sample_id] = (verdict, str(path))
        total_rows += len(rows)
        total_reviewed += reviewed
        total_pending += pending
        shard_reports.append(
            {
                "path": str(path),
                "rows": len(rows),
                "reviewed": reviewed,
                "pending": pending,
                "invalid_verdicts": shard_invalid,
                "verdicts": dict(sorted(shard_verdicts.items())),
                "complete": bool(rows and pending == 0 and not shard_invalid),
            }
        )

    gates = {
        "shards_present": bool(paths),
        "invalid_verdicts_zero": not all_invalid_verdicts,
        "duplicate_ids_zero": not duplicate_ids,
        "conflicting_verdicts_zero": not conflicting_verdicts,
        "all_rows_reviewed": total_rows > 0 and total_pending == 0,
    }
    if all(gates.values()):
        status = "complete"
    elif gates["shards_present"] and gates["invalid_verdicts_zero"] and gates["duplicate_ids_zero"] and gates["conflicting_verdicts_zero"]:
        status = "in_progress"
    else:
        status = "not_ready"
    report = {
        "generated_at": utc_now(),
        "inputs": {"shards": [str(path) for path in paths]},
        "total_rows": total_rows,
        "reviewed": total_reviewed,
        "pending": total_pending,
        "verdicts": dict(sorted(verdicts.items())),
        "shards": shard_reports,
        "invalid_verdicts": all_invalid_verdicts,
        "duplicate_ids": duplicate_ids,
        "conflicting_verdicts": conflicting_verdicts,
        "gates": gates,
        "status": status,
    }
    if out_path:
        write_json(out_path, report)
        out_path.with_suffix(".md").write_text(render_abstention_shard_progress_report(report), encoding="utf-8")
    return report


def render_abstention_review_packets(
    shard_paths: Iterable[Path],
    out_dir: Path,
    report_out: Path | None = None,
    query_limit: int = 1800,
    evidence_limit: int = 1200,
) -> dict[str, Any]:
    paths = list(shard_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    for path in paths:
        rows = read_audit_verdict_rows(path)
        markdown_path = out_dir / f"{path.stem}_review.md"
        markdown_path.write_text(
            render_abstention_review_packet_markdown(path, rows, query_limit=query_limit, evidence_limit=evidence_limit),
            encoding="utf-8",
        )
        packets.append(
            {
                "source": str(path),
                "markdown": str(markdown_path),
                "rows": len(rows),
                "by_pool": dict(sorted(Counter(row.get("pool") or "missing" for row in rows).items())),
                "by_reason": dict(sorted(Counter(row.get("proposed_no_gold_reason") or "missing" for row in rows).items())),
                "pending": sum(1 for row in rows if not normalize_abstention_verdict(row.get("verdict"))),
            }
        )
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "shards": [str(path) for path in paths],
            "query_limit": query_limit,
            "evidence_limit": evidence_limit,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "report": str(report_out) if report_out else None,
        },
        "packets": packets,
        "total_rows": sum(packet["rows"] for packet in packets),
        "total_pending": sum(packet["pending"] for packet in packets),
        "status": "ready_for_manual_audit" if packets else "not_ready",
    }
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_abstention_review_packets_report(report), encoding="utf-8")
    return report


def write_abstention_audit_handoff_manifest(
    audit_dir: Path,
    out_path: Path,
    finalization_report_path: Path | None = None,
) -> dict[str, Any]:
    shard_dir = audit_dir / "shards"
    review_dir = audit_dir / "review_packets"
    core_files = [
        ("handoff", audit_dir / "abstention_manual_audit_handoff.md", True),
        ("canonical_audit_jsonl", audit_dir / "abstention_audit_packet.jsonl", True),
        ("canonical_audit_csv", audit_dir / "abstention_audit_packet.csv", True),
        ("worklist_csv", audit_dir / "abstention_manual_audit_worklist.csv", True),
        ("worklist_jsonl", audit_dir / "abstention_manual_audit_worklist.jsonl", True),
        ("worklist_report_json", audit_dir / "abstention_manual_audit_worklist_report.json", True),
        ("shard_progress_json", audit_dir / "abstention_shard_progress_report.json", True),
        ("review_packets_report_json", audit_dir / "abstention_review_packets_report.json", True),
        ("crawling_status_json", audit_dir / "abstention_crawling_status.json", True),
        ("completion_audit_json", audit_dir / "abstention_completion_audit.json", True),
        ("canonical_apply_report_json", audit_dir / "abstention_audit_verdict_apply_report.json", False),
    ]
    dynamic_files: list[tuple[str, Path, bool]] = []
    for path in sorted(shard_dir.glob("*")):
        if path.suffix.lower() in {".csv", ".jsonl"}:
            dynamic_files.append(("reviewer_shard", path, True))
    for path in sorted(review_dir.glob("*.md")):
        dynamic_files.append(("review_packet_markdown", path, True))
    if finalization_report_path:
        dynamic_files.append(("finalization_report_json", finalization_report_path, False))
        finalization_markdown = finalization_report_path.with_suffix(".md")
        if finalization_markdown.exists():
            dynamic_files.append(("finalization_report_markdown", finalization_markdown, False))

    file_entries = [handoff_file_entry(role, path, required=required) for role, path, required in [*core_files, *dynamic_files]]
    missing_required = [entry for entry in file_entries if entry["required"] and not entry["exists"]]
    shard_entries = [entry for entry in file_entries if entry["role"] == "reviewer_shard" and entry["path"].endswith(".csv")]
    review_entries = [entry for entry in file_entries if entry["role"] == "review_packet_markdown"]
    progress = load_optional_json(audit_dir / "abstention_shard_progress_report.json")
    review_report = load_optional_json(audit_dir / "abstention_review_packets_report.json")
    completion = load_optional_json(audit_dir / "abstention_completion_audit.json")
    finalization = load_optional_json(finalization_report_path) if finalization_report_path else None

    gates = {
        "required_files_exist": not missing_required,
        "reviewer_shards_present": bool(shard_entries),
        "review_packets_present": bool(review_entries),
        "shard_progress_has_no_invalid_verdicts": bool((progress.get("gates") or {}).get("invalid_verdicts_zero")) if progress else False,
        "shard_progress_has_no_duplicate_ids": bool((progress.get("gates") or {}).get("duplicate_ids_zero")) if progress else False,
        "shard_progress_has_no_conflicts": bool((progress.get("gates") or {}).get("conflicting_verdicts_zero")) if progress else False,
        "review_packets_cover_all_rows": bool(
            progress
            and review_report
            and int(progress.get("total_rows") or 0) == int(review_report.get("total_rows") or -1)
        ),
    }
    status = "ready_for_manual_audit" if all(gates.values()) else "not_ready"
    shard_csv_paths = [Path(entry["path"]) for entry in shard_entries]
    report = {
        "generated_at": utc_now(),
        "inputs": {
            "audit_dir": str(audit_dir),
            "finalization_report": str(finalization_report_path) if finalization_report_path else None,
        },
        "outputs": {"manifest": str(out_path), "markdown": str(out_path.with_suffix(".md"))},
        "files": file_entries,
        "file_count": len(file_entries),
        "missing_required": missing_required,
        "reviewer_shards": {
            "csv": sum(1 for entry in file_entries if entry["role"] == "reviewer_shard" and entry["path"].endswith(".csv")),
            "jsonl": sum(1 for entry in file_entries if entry["role"] == "reviewer_shard" and entry["path"].endswith(".jsonl")),
        },
        "review_packets": len(review_entries),
        "shard_progress": {
            "status": progress.get("status") if progress else None,
            "total_rows": progress.get("total_rows") if progress else None,
            "reviewed": progress.get("reviewed") if progress else None,
            "pending": progress.get("pending") if progress else None,
            "gates": progress.get("gates") if progress else None,
        },
        "completion_audit": {
            "status": completion.get("status") if completion else None,
            "status_counts": completion.get("status_counts") if completion else None,
        },
        "finalization": {
            "status": finalization.get("status") if finalization else None,
            "next_action": finalization.get("next_action") if finalization else None,
        },
        "next_commands": [
            shell_command(
                [
                    "python",
                    "-m",
                    "agent_retrieval_bench.cli",
                    "report-abstention-shard-progress",
                    "--out",
                    audit_dir / "abstention_shard_progress_report.json",
                ]
            ),
            abstention_apply_verdicts_command(
                shard_csv_paths,
                target_audit=audit_dir / "abstention_audit_packet.csv",
                out=audit_dir / "abstention_audit_packet.csv",
                report_out=audit_dir / "abstention_audit_verdict_apply_report.json",
                require_complete=True,
            ),
            "python -m agent_retrieval_bench.cli finalize-abstention-pilot",
        ],
        "gates": gates,
        "status": status,
    }
    write_json(out_path, report)
    out_path.with_suffix(".md").write_text(render_abstention_handoff_manifest_report(report), encoding="utf-8")
    return report


def shell_command(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def abstention_apply_verdicts_command(
    source_audit_paths: Iterable[Path],
    target_audit: Path,
    out: Path,
    report_out: Path,
    require_complete: bool = False,
) -> str:
    lines = ["python -m agent_retrieval_bench.cli apply-abstention-audit-verdicts"]
    for path in source_audit_paths:
        lines.append(f"  --source-audit {shlex.quote(str(path))}")
    lines.extend(
        [
            f"  --target-audit {shlex.quote(str(target_audit))}",
            f"  --out {shlex.quote(str(out))}",
            f"  --report-out {shlex.quote(str(report_out))}",
        ]
    )
    if require_complete:
        lines.append("  --require-complete")
    return " \\\n".join(lines)


def handoff_file_entry(role: str, path: Path, required: bool) -> dict[str, Any]:
    exists = path.exists()
    entry: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "required": required,
        "exists": exists,
    }
    if not exists:
        return entry
    entry["size_bytes"] = path.stat().st_size
    entry["sha256"] = sha256_file(path)
    if path.suffix.lower() == ".jsonl":
        entry["rows"] = len(read_jsonl(path))
    elif path.suffix.lower() == ".csv":
        entry["rows"] = count_csv_rows(path)
    return entry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def crawl_abstention_issues(
    api: GitHubAPI,
    repos: Iterable[str],
    out_dir: Path,
    limit_per_repo: int = 60,
    comments_per_issue: int = 100,
    events_per_issue: int = 100,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for repo in repos:
        owner, name = repo.split("/", 1)
        repo_dir = out_dir / repo.replace("/", "__")
        seen_numbers: set[int] = set()
        issue_rows: list[dict[str, Any]] = []
        comment_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        rate_limited = False
        for keyword in issue_search_keywords():
            if len(seen_numbers) >= limit_per_repo or rate_limited:
                break
            try:
                items = search_closed_issues(api, repo, keyword, limit=max(1, limit_per_repo - len(seen_numbers)))
            except RuntimeError as error:
                errors.append({"keyword": keyword, "error": str(error)})
                rate_limited = is_rate_limit_wait_error(str(error))
                if rate_limited:
                    break
                continue
            for issue in items:
                number = issue.get("number")
                if not isinstance(number, int) or number in seen_numbers or issue.get("pull_request"):
                    continue
                seen_numbers.add(number)
                comments = fetch_issue_page(api, owner, name, number, "comments", comments_per_issue, errors)
                if last_error_is_rate_limit(errors):
                    rate_limited = True
                    break
                events = fetch_issue_page(api, owner, name, number, "events", events_per_issue, errors)
                if last_error_is_rate_limit(errors):
                    rate_limited = True
                    break
                evidence_comment = first_maintainer_reason_comment(comments)
                base_commit = ""
                base_commit_source = ""
                if evidence_comment and evidence_comment.get("created_at"):
                    base_commit = fetch_default_branch_commit_before(api, owner, name, str(evidence_comment["created_at"]), errors)
                    if last_error_is_rate_limit(errors):
                        rate_limited = True
                        break
                    base_commit_source = "default_branch_before_maintainer_evidence" if base_commit else ""
                common = {"repo": repo, "issue_number": number, "fetched_at": utc_now()}
                issue_rows.append(
                    {
                        **common,
                        "type": "issue",
                        "matched_search_keyword": keyword,
                        "candidate_base_commit": base_commit,
                        "candidate_base_commit_source": base_commit_source,
                        "data": issue,
                    }
                )
                comment_rows.append({**common, "type": "issue_comments", "data": comments})
                event_rows.append({**common, "type": "issue_events", "data": events})
                if len(seen_numbers) >= limit_per_repo:
                    break
        write_jsonl(repo_dir / "issues.jsonl", issue_rows)
        write_jsonl(repo_dir / "issue_comments.jsonl", comment_rows)
        write_jsonl(repo_dir / "issue_events.jsonl", event_rows)
        summaries.append(
            {
                "repo": repo,
                "issues": len(issue_rows),
                "issue_comments": len(comment_rows),
                "issue_events": len(event_rows),
                "errors": errors,
                "out_dir": str(repo_dir),
            }
        )
    report = {
        "generated_at": utc_now(),
        "out_dir": str(out_dir),
        "repos": summaries,
        "total_issues": sum(row["issues"] for row in summaries),
        "total_errors": sum(len(row["errors"]) for row in summaries),
    }
    write_json(out_dir / "issue_crawl_summary.json", report)
    return report


def crawl_abstention_issue_html(
    repos: Iterable[str],
    out_dir: Path,
    git_repos_dir: Path = Path("data/git_raw_repos"),
    limit_per_repo: int = 60,
    pages_per_keyword: int = 2,
    request_delay_seconds: float = 0.4,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    cache_dir = out_dir / "_html_cache"
    for repo in repos:
        repo_dir = out_dir / repo.replace("/", "__")
        seen_numbers: set[int] = set()
        issue_rows: list[dict[str, Any]] = []
        comment_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for keyword in issue_search_keywords():
            if len(seen_numbers) >= limit_per_repo:
                break
            try:
                issue_refs = search_closed_issue_html(repo, keyword, pages_per_keyword, limit_per_repo - len(seen_numbers), cache_dir, request_delay_seconds)
            except RuntimeError as error:
                errors.append({"keyword": keyword, "error": str(error)})
                continue
            for issue_ref in issue_refs:
                number = int(issue_ref["number"])
                if number in seen_numbers or len(seen_numbers) >= limit_per_repo:
                    continue
                seen_numbers.add(number)
                try:
                    html = fetch_url_text(issue_ref["url"], cache_dir, request_delay_seconds)
                    parsed = parse_github_issue_html(html, repo, number)
                except RuntimeError as error:
                    errors.append({"issue": str(number), "error": str(error)})
                    continue
                comments = parsed["comments"]
                events = parsed["events"]
                evidence_comment = first_maintainer_reason_comment(comments)
                base_commit = ""
                base_commit_source = ""
                if evidence_comment and evidence_comment.get("created_at"):
                    base_commit = local_git_commit_before(repo, git_repos_dir, str(evidence_comment["created_at"]))
                    base_commit_source = "local_default_branch_before_maintainer_evidence" if base_commit else ""
                common = {"repo": repo, "issue_number": number, "fetched_at": utc_now()}
                issue_rows.append(
                    {
                        **common,
                        "type": "issue",
                        "matched_search_keyword": keyword,
                        "candidate_base_commit": base_commit,
                        "candidate_base_commit_source": base_commit_source,
                        "data": parsed["issue"],
                    }
                )
                comment_rows.append({**common, "type": "issue_comments", "data": comments})
                event_rows.append({**common, "type": "issue_events", "data": events})
        write_jsonl(repo_dir / "issues.jsonl", issue_rows)
        write_jsonl(repo_dir / "issue_comments.jsonl", comment_rows)
        write_jsonl(repo_dir / "issue_events.jsonl", event_rows)
        summaries.append(
            {
                "repo": repo,
                "issues": len(issue_rows),
                "issue_comments": len(comment_rows),
                "issue_events": len(event_rows),
                "errors": errors,
                "out_dir": str(repo_dir),
            }
        )
    report = {
        "generated_at": utc_now(),
        "out_dir": str(out_dir),
        "git_repos_dir": str(git_repos_dir),
        "repos": summaries,
        "total_issues": sum(row["issues"] for row in summaries),
        "total_errors": sum(len(row["errors"]) for row in summaries),
    }
    write_json(out_dir / "issue_html_crawl_summary.json", report)
    return report


def backfill_abstention_issue_base_commits(
    raw_dirs: Iterable[Path],
    git_repos_dir: Path,
    api: GitHubAPI | None = None,
    api_fallback: bool = False,
    report_out: Path | None = None,
) -> dict[str, Any]:
    raw_dir_list = list(raw_dirs)
    repo_reports: list[dict[str, Any]] = []
    total_issues = 0
    total_missing_before = 0
    total_backfilled = 0
    total_missing_after = 0
    all_errors: list[dict[str, str]] = []

    for raw_dir in raw_dir_list:
        if not raw_dir.exists():
            continue
        for repo_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir() and path.name != "_html_cache"):
            issues_path = repo_dir / "issues.jsonl"
            comments_path = repo_dir / "issue_comments.jsonl"
            if not issues_path.exists() or not comments_path.exists():
                continue
            issues = read_jsonl(issues_path)
            comments_by_issue = load_issue_related_rows(comments_path)
            repo = str((issues[0] if issues else {}).get("repo") or repo_dir.name.replace("__", "/"))
            owner, _, name = repo.partition("/")
            repo_errors: list[dict[str, str]] = []
            changed = False
            missing_before = 0
            backfilled = 0
            missing_after = 0

            for issue_row in issues:
                if issue_row.get("type") != "issue":
                    continue
                total_issues += 1
                if issue_row.get("candidate_base_commit"):
                    continue
                missing_before += 1
                issue_number = issue_row.get("issue_number")
                try:
                    issue_number_int = int(issue_number)
                except (TypeError, ValueError):
                    missing_after += 1
                    continue
                comments = comments_by_issue.get(issue_number_int, [])
                evidence_comment = first_maintainer_reason_comment(comments)
                evidence_created_at = str((evidence_comment or {}).get("created_at") or "")
                if not evidence_created_at:
                    missing_after += 1
                    continue
                base_commit = local_git_commit_before(repo, git_repos_dir, evidence_created_at)
                base_commit_source = "local_default_branch_before_maintainer_evidence" if base_commit else ""
                if not base_commit and api_fallback and api and owner and name:
                    api_errors: list[dict[str, str]] = []
                    base_commit = fetch_default_branch_commit_before(api, owner, name, evidence_created_at, api_errors)
                    if base_commit:
                        base_commit_source = "github_default_branch_before_maintainer_evidence"
                    for error in api_errors:
                        issue_error = {"repo": repo, "issue": str(issue_number), **error}
                        repo_errors.append(issue_error)
                        all_errors.append(issue_error)
                if base_commit:
                    issue_row["candidate_base_commit"] = base_commit
                    issue_row["candidate_base_commit_source"] = base_commit_source
                    changed = True
                    backfilled += 1
                else:
                    missing_after += 1

            total_missing_before += missing_before
            total_backfilled += backfilled
            total_missing_after += missing_after
            if changed:
                write_jsonl(issues_path, issues)
            repo_reports.append(
                {
                    "raw_dir": str(raw_dir),
                    "repo_dir": str(repo_dir),
                    "repo": repo,
                    "issues": sum(1 for row in issues if row.get("type") == "issue"),
                    "missing_before": missing_before,
                    "backfilled": backfilled,
                    "missing_after": missing_after,
                    "errors": repo_errors,
                }
            )

    report = {
        "generated_at": utc_now(),
        "inputs": {
            "raw_dirs": [str(path) for path in raw_dir_list],
            "git_repos_dir": str(git_repos_dir),
            "api_fallback": api_fallback,
        },
        "repos": repo_reports,
        "total_issues": total_issues,
        "missing_before": total_missing_before,
        "backfilled": total_backfilled,
        "missing_after": total_missing_after,
        "errors": all_errors,
        "status": "complete" if total_missing_before == 0 or total_backfilled > 0 else "no_updates",
    }
    if report_out:
        write_json(report_out, report)
        report_out.with_suffix(".md").write_text(render_abstention_issue_base_backfill_report(report), encoding="utf-8")
    return report


def issue_search_keywords() -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for values in ISSUE_REASON_KEYWORDS.values():
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
    return output


def last_error_is_rate_limit(errors: list[dict[str, str]]) -> bool:
    return bool(errors and is_rate_limit_wait_error(errors[-1].get("error", "")))


def is_rate_limit_wait_error(message: str) -> bool:
    lowered = message.lower()
    return "rate limit wait exceeds configured maximum" in lowered or "x-ratelimit" in lowered


def search_closed_issue_html(
    repo: str,
    keyword: str,
    pages_per_keyword: int,
    limit: int,
    cache_dir: Path,
    request_delay_seconds: float,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for page in range(1, pages_per_keyword + 1):
        if len(refs) >= limit:
            break
        query = f'is:issue is:closed "{keyword}"'
        url = f"https://github.com/{repo}/issues?" + urllib.parse.urlencode({"q": query, "page": page})
        html = fetch_url_text(url, cache_dir, request_delay_seconds)
        for ref in parse_issue_search_html(html, repo):
            number = int(ref["number"])
            if number in seen:
                continue
            seen.add(number)
            refs.append(ref)
            if len(refs) >= limit:
                break
    return refs


def parse_issue_search_html(html: str, repo: str) -> list[dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError("BeautifulSoup (bs4) is required for HTML issue crawling") from error
    soup = BeautifulSoup(html, "html.parser")
    refs: list[dict[str, Any]] = []
    prefix = f"/{repo}/issues/"
    seen: set[int] = set()
    for link in soup.select('[data-testid="issue-pr-title-link"]'):
        href = str(link.get("href") or "")
        if not href.startswith(prefix):
            continue
        tail = href.removeprefix(prefix).split("/", 1)[0].split("#", 1)[0]
        try:
            number = int(tail)
        except ValueError:
            continue
        if number in seen:
            continue
        seen.add(number)
        refs.append({"number": number, "url": f"https://github.com{href}", "title": link.get_text(" ", strip=True)})
    return refs


def fetch_url_text(url: str, cache_dir: Path, request_delay_seconds: float) -> str:
    ensure_parent(cache_dir / "placeholder")
    cache_path = cache_dir / f"{stable_id(url)}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")
    time.sleep(max(0.0, request_delay_seconds))
    request = urllib.request.Request(url, headers={"User-Agent": "agent-retrieval-bench/0.1"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except Exception as error:  # pragma: no cover - network-specific failure detail
        raise RuntimeError(f"failed to fetch {url}: {error}") from error
    text = raw.decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_github_issue_html(html: str, repo: str, number: int) -> dict[str, Any]:
    issue = extract_issue_viewer_issue(html, repo, number)
    comments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for collection_name in ("frontTimelineItems", "backTimelineItems"):
        for edge in ((issue.get(collection_name) or {}).get("edges") or []):
            node = edge.get("node") or {}
            typename = str(node.get("__typename") or "")
            if typename == "IssueComment" or (node.get("body") and node.get("authorAssociation")):
                comments.append(issue_comment_from_html_node(node))
            elif typename.endswith("Event"):
                event = issue_event_from_html_node(node)
                if event:
                    events.append(event)
    if str(issue.get("state") or "").upper() == "CLOSED" and not any(event.get("event") == "closed" for event in events):
        events.append({"event": "closed", "created_at": issue.get("closedAt") or ""})
    return {"issue": issue_from_html_node(issue, repo, number), "comments": comments, "events": events}


def extract_issue_viewer_issue(html: str, repo: str, number: int) -> dict[str, Any]:
    for text in iter_json_script_texts(html):
        if "IssueViewerViewQuery" not in text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        for query in ((data.get("payload") or {}).get("preloadedQueries") or []):
            result = query.get("result") or {}
            repository = ((result.get("data") or {}).get("repository") or {})
            issue = repository.get("issue") or {}
            if int(issue.get("number") or -1) == number:
                return issue
    raise RuntimeError(f"could not parse issue preload data for {repo}#{number}")


def iter_json_script_texts(html: str) -> Iterable[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        for match in re.finditer(r"""<script\b[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>""", html, re.IGNORECASE | re.DOTALL):
            yield html_lib.unescape(match.group(1))
        return
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", {"type": "application/json"}):
        yield script.string or script.get_text() or ""


def issue_from_html_node(issue: dict[str, Any], repo: str, number: int) -> dict[str, Any]:
    linked_prs: list[str] = []
    for key in ("linkedPullRequests", "linkedPullRequestsIncludingClosed"):
        for node in (((issue.get(key) or {}).get("nodes")) or []):
            url = str(node.get("url") or "")
            if url:
                linked_prs.append(url)
    return {
        "number": number,
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "html_url": issue.get("url") or f"https://github.com/{repo}/issues/{number}",
        "url": issue.get("url") or f"https://github.com/{repo}/issues/{number}",
        "created_at": issue.get("createdAt") or "",
        "closed_at": issue.get("closedAt") or "",
        "state": issue.get("state") or "",
        "linked_pull_requests": sorted(set(linked_prs)),
        "user": {"login": ((issue.get("author") or {}).get("login") or "")},
    }


def issue_comment_from_html_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "author_association": node.get("authorAssociation") or "",
        "created_at": node.get("createdAt") or "",
        "body": node.get("body") or "",
        "html_url": node.get("url") or "",
        "user": {"login": ((node.get("author") or {}).get("login") or "")},
    }


def issue_event_from_html_node(node: dict[str, Any]) -> dict[str, Any]:
    typename = str(node.get("__typename") or "")
    if typename == "ClosedEvent":
        return {"event": "closed", "created_at": node.get("createdAt") or "", "commit_id": node.get("closer", {}).get("oid") if isinstance(node.get("closer"), dict) else ""}
    if typename == "ReopenedEvent":
        return {"event": "reopened", "created_at": node.get("createdAt") or ""}
    return {"event": typename, "created_at": node.get("createdAt") or ""} if typename else {}


def local_git_commit_before(repo: str, git_repos_dir: Path, until: str) -> str:
    repo_dir = git_repos_dir / repo.replace("/", "__")
    if not (repo_dir / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", f"--before={until}", "--format=%H"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else ""


def search_closed_issues(api: GitHubAPI, repo: str, keyword: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while len(items) < limit:
        query = f'repo:{repo} is:issue is:closed "{keyword}"'
        response = api.get(
            "/search/issues",
            {
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": min(100, limit - len(items)),
                "page": page,
            },
        )
        body = response.body or {}
        page_items = body.get("items") or []
        if not page_items:
            break
        items.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < min(100, limit - len(items) + len(page_items)):
            break
        page += 1
    return items[:limit]


def fetch_issue_page(
    api: GitHubAPI,
    owner: str,
    name: str,
    number: int,
    kind: str,
    limit: int,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    try:
        while len(rows) < limit:
            response = api.get(
                f"/repos/{owner}/{name}/issues/{number}/{kind}",
                {"per_page": min(100, limit - len(rows)), "page": page},
            )
            if not isinstance(response.body, list):
                raise RuntimeError(f"Expected list from issue {kind}, got {type(response.body).__name__}")
            rows.extend(row for row in response.body if isinstance(row, dict))
            if len(response.body) < min(100, limit - len(rows) + len(response.body)):
                break
            if 'rel="next"' not in response.headers.get("link", ""):
                break
            page += 1
    except RuntimeError as error:
        errors.append({"issue": str(number), "kind": kind, "error": str(error)})
        return rows[:limit]
    return rows[:limit]


def fetch_default_branch_commit_before(
    api: GitHubAPI,
    owner: str,
    name: str,
    until: str,
    errors: list[dict[str, str]],
) -> str:
    try:
        repo = api.get(f"/repos/{owner}/{name}").body or {}
        branch = repo.get("default_branch")
        commits = api.get(
            f"/repos/{owner}/{name}/commits",
            {"sha": branch, "until": until, "per_page": 1},
        ).body
    except RuntimeError as error:
        errors.append({"kind": "base_commit", "error": str(error)})
        return ""
    if isinstance(commits, list) and commits:
        return str(commits[0].get("sha") or "")
    return ""


def first_maintainer_reason_comment(comments: list[dict[str, Any]]) -> dict[str, Any] | None:
    for comment in sorted(comments, key=lambda row: str(row.get("created_at") or "")):
        if not is_maintainer_comment(comment):
            continue
        if classify_issue_reason(str(comment.get("body") or "")):
            return comment
    return None


def discover_raw_repo_dirs(raw_dirs: Iterable[Path]) -> list[Path]:
    repo_dirs: list[Path] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        for child in sorted(raw_dir.iterdir()):
            if child.is_dir() and ((child / "check_runs.jsonl").exists() or (child / "issues.jsonl").exists()):
                repo_dirs.append(child)
    return repo_dirs


def repo_from_raw_dir(repo_dir: Path) -> str:
    return repo_dir.name.replace("__", "/")


def load_pull_requests_by_number(repo_dir: Path) -> dict[int, dict[str, Any]]:
    pulls: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(repo_dir / "pull_requests.jsonl"):
        data = row.get("data") or {}
        number = data.get("number") or row.get("pr_number")
        if number is None:
            continue
        try:
            pulls[int(number)] = data
        except (TypeError, ValueError):
            continue
    return pulls


def load_check_runs(repo_dir: Path, repo: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in read_jsonl(repo_dir / "check_runs.jsonl"):
        pr_number = row.get("pr_number")
        row_sha = row.get("sha")
        for run in row.get("data") or []:
            if not isinstance(run, dict):
                continue
            item = dict(run)
            item.setdefault("repo", repo)
            item.setdefault("pr_number", pr_number)
            item.setdefault("sha", row_sha)
            runs.append(item)
    return runs


def organic_candidate_from_check_group(
    repo: str,
    sha: str,
    name: str,
    runs: list[dict[str, Any]],
    pull_requests: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    failures = [
        run
        for run in runs
        if str(run.get("conclusion") or "").lower() in {"failure", "timed_out", "action_required", "cancelled"}
    ]
    successes = [run for run in runs if str(run.get("conclusion") or "").lower() == "success"]
    if not failures or not successes:
        return None
    failure = sorted(failures, key=lambda run: str(run.get("completed_at") or run.get("started_at") or ""))[0]
    success_after = [
        run
        for run in successes
        if str(run.get("completed_at") or "") >= str(failure.get("completed_at") or failure.get("started_at") or "")
    ]
    if not success_after:
        return None
    pr_number = first_pr_number(failure, runs)
    pr = pull_requests.get(pr_number or -1, {})
    query_text = ci_query_text(failure, pr)
    matched = matched_keywords(" ".join([query_text, name, str(failure.get("conclusion") or "")]))
    if not matched and str(failure.get("conclusion") or "").lower() == "timed_out":
        matched = ["timeout"]
    candidate_id = "abstention_candidate__organic__" + stable_id(repo, sha, name, pr_number or "", failure.get("id") or "")
    source_url = str(failure.get("html_url") or failure.get("details_url") or pr.get("url") or "")
    return {
        "id": candidate_id,
        "repo": repo,
        "source_type": "ci_log",
        "event_url": source_url,
        "query_text": query_text,
        "matched_keywords": matched,
        "candidate_reason": "flaky_ci_same_sha_rerun_passed",
        "possible_base_sha": sha,
        "linked_commits": [],
        "linked_prs": [pr.get("url")] if pr.get("url") else [],
        "evidence_snippets": [
            f"Check `{name}` failed with conclusion `{failure.get('conclusion')}` on head_sha {sha}.",
            f"The same check later had {len(success_after)} success run(s) on the same head_sha.",
        ],
        "evidence": {
            "source_url": source_url,
            "resolution_comments": [],
            "rerun_status": "same_head_sha_later_passed",
            "linked_commits": [],
            "why_no_local_fix": "Same check name later passed on the same head_sha, so no repository-local code change was required between failure and success.",
        },
    }


def first_pr_number(failure: dict[str, Any], runs: list[dict[str, Any]]) -> int | None:
    for value in [failure.get("pr_number"), *(run.get("pr_number") for run in runs)]:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    pull_requests = failure.get("pull_requests") or []
    if pull_requests and isinstance(pull_requests[0], dict):
        try:
            return int(pull_requests[0].get("number"))
        except (TypeError, ValueError):
            return None
    return None


def ci_query_text(failure: dict[str, Any], pr: dict[str, Any]) -> str:
    output = failure.get("output") or {}
    excerpt_parts = [
        output.get("title"),
        output.get("summary"),
        output.get("text"),
        failure.get("details_url"),
    ]
    excerpt = truncate_text("\n".join(str(part) for part in excerpt_parts if part), 1200)
    lines = [f"Check name: {failure.get('name') or failure.get('check_name')}", f"Conclusion: {failure.get('conclusion')}"]
    if excerpt:
        lines.append(f"Failure excerpt: {excerpt}")
    if pr.get("title"):
        lines.append(f"PR title: {pr.get('title')}")
    return truncate_text("\n".join(lines), 2400)


def matched_keywords(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in ORGANIC_KEYWORDS if keyword in lowered]


def organic_prefilter_drop_reason(candidate: dict[str, Any]) -> str | None:
    if candidate.get("source_type") == "issue":
        if not candidate.get("possible_base_sha"):
            return "missing_base_sha"
        if not candidate.get("evidence_snippets"):
            return "missing_resolution_evidence"
        if (candidate.get("local_fix_signals") or {}).get("has_local_fix_signal"):
            return "local_fix_signal"
        if candidate.get("query_resolution_leakage"):
            return "query_contains_resolution_answer"
        if len(str(candidate.get("query_text") or "")) < 80:
            return "query_too_short"
        return None
    if candidate.get("source_type") != "ci_log":
        return "unsupported_source_type"
    if not candidate.get("possible_base_sha"):
        return "missing_base_sha"
    if not (candidate.get("evidence") or {}).get("rerun_status"):
        return "missing_resolution_evidence"
    if not candidate.get("matched_keywords"):
        return "no_recall_keyword"
    if len(str(candidate.get("query_text") or "")) < 80:
        return "query_too_short"
    return None


def organic_audit_packet(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence") or {}
    return {
        "id": candidate["id"],
        "repo": candidate["repo"],
        "base_commit": candidate["possible_base_sha"],
        "query": {"source": candidate["source_type"], "text": candidate["query_text"]},
        "proposed_no_gold_reason": candidate.get("no_gold_reason") or "flaky_ci",
        "evidence": {
            "source_url": evidence.get("source_url") or candidate.get("event_url") or "",
            "resolution_comments": evidence.get("resolution_comments") or [],
            "rerun_status": evidence.get("rerun_status") or "",
            "linked_commits": evidence.get("linked_commits") or [],
            "why_no_local_fix": evidence.get("why_no_local_fix") or "",
            "evidence_snippets": candidate.get("evidence_snippets") or [],
        },
        "audit_fields": {"verdict": "", "notes": "", "has_local_gold_files": []},
    }


def load_issues(repo_dir: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(repo_dir / "issues.jsonl") if row.get("type") == "issue"]


def load_issue_related_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        try:
            issue_number = int(row.get("issue_number"))
        except (TypeError, ValueError):
            continue
        rows[issue_number] = [item for item in row.get("data") or [] if isinstance(item, dict)]
    return rows


def organic_candidate_from_issue_row(
    repo: str,
    issue_row: dict[str, Any],
    comments: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    issue = issue_row.get("data") or {}
    if not isinstance(issue, dict) or issue.get("pull_request"):
        return None
    evidence_comment = first_maintainer_reason_comment(comments)
    if not evidence_comment:
        return None
    reason = classify_issue_reason(str(evidence_comment.get("body") or ""))
    if not reason:
        return None
    issue_number = issue.get("number") or issue_row.get("issue_number")
    query_text = issue_query_text(issue)
    resolution_comment = truncate_text(str(evidence_comment.get("body") or ""), 1000)
    local_fix_signals = issue_local_fix_signals(issue, comments, events)
    query_leakage = issue_query_contains_resolution_answer(query_text, reason)
    candidate_id = "abstention_candidate__organic_issue__" + stable_id(repo, issue_number, reason)
    source_url = str(issue.get("html_url") or issue.get("url") or "")
    base_commit = str(issue_row.get("candidate_base_commit") or "")
    return {
        "id": candidate_id,
        "repo": repo,
        "source_type": "issue",
        "event_url": source_url,
        "query_text": query_text,
        "matched_keywords": matched_issue_keywords(" ".join([str(issue.get("title") or ""), str(issue.get("body") or ""), resolution_comment])),
        "candidate_reason": reason,
        "no_gold_reason": reason,
        "possible_base_sha": base_commit,
        "base_commit_source": issue_row.get("candidate_base_commit_source") or "",
        "linked_commits": local_fix_signals.get("commit_ids", []),
        "linked_prs": local_fix_signals.get("linked_prs", []),
        "local_fix_signals": local_fix_signals,
        "query_resolution_leakage": query_leakage,
        "evidence_snippets": [
            f"Maintainer comment by {((evidence_comment.get('user') or {}).get('login') or 'unknown')} classified as {reason}.",
            resolution_comment,
        ],
        "evidence": {
            "source_url": source_url,
            "resolution_comments": [resolution_comment],
            "rerun_status": "",
            "linked_commits": local_fix_signals.get("commit_ids", []),
            "why_no_local_fix": "Maintainer resolution evidence classifies the issue as no repository-local fix; automatic checks found no issue closing commit event.",
        },
    }


def issue_query_text(issue: dict[str, Any]) -> str:
    title = str(issue.get("title") or "").strip()
    body = truncate_text(str(issue.get("body") or ""), 1800)
    lines = [f"Issue title: {title}"]
    if body:
        lines.append(f"Issue body excerpt: {body}")
    return truncate_text("\n".join(lines), 2400)


def classify_issue_reason(text: str) -> str:
    lowered = text.lower()
    for reason, keywords in ISSUE_REASON_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return reason
    return ""


def matched_issue_keywords(text: str) -> list[str]:
    lowered = text.lower()
    output: list[str] = []
    for keywords in ISSUE_REASON_KEYWORDS.values():
        for keyword in keywords:
            if keyword in lowered and keyword not in output:
                output.append(keyword)
    return output


def is_maintainer_comment(comment: dict[str, Any]) -> bool:
    association = str(comment.get("author_association") or "").upper()
    return association in MAINTAINER_ASSOCIATIONS


def issue_local_fix_signals(
    issue: dict[str, Any],
    comments: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    commit_ids: list[str] = []
    linked_prs: list[str] = []
    event_types: list[str] = []
    for event in events:
        event_type = str(event.get("event") or "")
        if event_type:
            event_types.append(event_type)
        commit_id = event.get("commit_id")
        if commit_id:
            commit_ids.append(str(commit_id))
        commit_url = str(event.get("commit_url") or "")
        if commit_url and commit_url not in commit_ids:
            commit_ids.append(commit_url)
    text = "\n".join([str(issue.get("body") or ""), *(str(comment.get("body") or "") for comment in comments)])
    for match in re.finditer(r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+", text):
        linked_prs.append(match.group(0))
    for url in issue.get("linked_pull_requests") or []:
        linked_prs.append(str(url))
    local_fix_paths = [path for path in LOCAL_FIX_PATH_PARTS if path.lower() in text.lower()]
    return {
        "commit_ids": sorted(set(commit_ids)),
        "linked_prs": sorted(set(linked_prs)),
        "event_types": sorted(set(event_types)),
        "local_fix_paths": sorted(set(local_fix_paths)),
        "has_local_fix_signal": bool(commit_ids or linked_prs or local_fix_paths),
    }


def issue_query_contains_resolution_answer(query_text: str, reason: str) -> bool:
    lowered = query_text.lower()
    if reason == "upstream_dependency":
        return any(keyword in lowered for keyword in ISSUE_REASON_KEYWORDS[reason])
    if reason == "user_error":
        shortcut_keywords = ("not a bug", "user error", "works as intended", "expected behavior")
        return any(keyword in lowered for keyword in shortcut_keywords)
    return False


def organic_audit_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "repo": candidate.get("repo"),
        "source_type": candidate.get("source_type"),
        "event_url": candidate.get("event_url"),
        "query_text": candidate.get("query_text"),
        "matched_keywords": ",".join(candidate.get("matched_keywords") or []),
        "candidate_reason": candidate.get("candidate_reason"),
        "possible_base_sha": candidate.get("possible_base_sha"),
        "linked_commits": json.dumps(candidate.get("linked_commits") or []),
        "linked_prs": json.dumps(candidate.get("linked_prs") or []),
        "evidence_snippets": json.dumps(candidate.get("evidence_snippets") or []),
        "verdict": "",
        "notes": "",
        "has_local_gold_files": "",
    }


def load_corpus_pairs(corpus_manifest_path: Path) -> list[CorpusPair]:
    pairs: list[CorpusPair] = []
    for row in read_jsonl(corpus_manifest_path):
        if row.get("status") != "ok":
            continue
        repo = str(row.get("repo") or "")
        base_commit = str(row.get("base_commit") or "")
        chunks_path = str(row.get("chunks_path") or "")
        if not repo or not base_commit or not chunks_path:
            continue
        language, domain = REPO_PROFILES.get(repo, ("unknown", "general"))
        pairs.append(CorpusPair(repo=repo, base_commit=base_commit, chunks_path=chunks_path, language=language, domain=domain))
    pairs.sort(key=lambda pair: (pair.repo, pair.base_commit))
    return pairs


def profile_for_sample(sample: dict[str, Any]) -> tuple[str, str]:
    repo = str(sample.get("repo") or "")
    if repo in REPO_PROFILES:
        return REPO_PROFILES[repo]
    query_text = json.dumps(sample.get("query") or {}, ensure_ascii=False)
    if re.search(r"\.(rs|toml)\b|cargo\b|rust\b", query_text, re.IGNORECASE):
        return ("rust", "general")
    if re.search(r"\.(go|mod)\b|go test\b", query_text, re.IGNORECASE):
        return ("go", "general")
    if re.search(r"\.(java|kt|kts)\b|gradle|maven", query_text, re.IGNORECASE):
        return ("java", "general")
    if re.search(r"\.(ts|tsx|js|jsx|vue)\b|npm|vite|eslint", query_text, re.IGNORECASE):
        return ("javascript", "general")
    return ("python", "general")


def counterfactual_query_text(sample: dict[str, Any]) -> tuple[str, str | None]:
    query = sample.get("query") or {}
    pieces: list[str] = []
    for key in TEXT_FIELD_NAMES:
        if key not in query:
            continue
        value = query.get(key)
        if key in PATH_FIELD_NAMES:
            continue
        rendered = render_query_value(value)
        if not rendered:
            continue
        label = key.replace("_", " ").title()
        pieces.append(f"{label}: {rendered}")
    if not pieces:
        return "", "empty_query_after_sanitization"
    text = scrub_path_like_text("\n".join(pieces))
    text = truncate_text(text, 5000)
    if len(text) < 80:
        return text, "query_too_short"
    if not coding_signal_like(text):
        return text, "not_coding_signal"
    return text, None


def render_query_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        rendered = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(rendered[:8])
    return str(value).strip()


def scrub_path_like_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if PATH_LIKE_RE.search(line):
            line = PATH_LIKE_RE.sub(" [path redacted]", line)
        lines.append(line)
    return "\n".join(lines).strip()


def coding_signal_like(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "error",
        "fail",
        "test",
        "traceback",
        "assert",
        "exception",
        "compile",
        "build",
        "import",
        "type",
        "api",
        "config",
        "dependency",
        "warning",
        "regression",
        "pr title",
        "review comment",
    )
    return any(marker in lowered for marker in markers)


def query_mentions_repo_identity(query_text: str, source_repo: str) -> bool:
    owner, _, name = source_repo.partition("/")
    tokens = source_identity_tokens(owner, name)
    lowered = query_text.lower()
    if source_repo.lower() in lowered or f"github.com/{source_repo.lower()}" in lowered:
        return True
    return any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered) for token in tokens)


def source_identity_tokens(owner: str, name: str) -> set[str]:
    tokens: set[str] = set()
    for raw in (owner, name):
        normalized = raw.lower()
        for token in re.split(r"[^a-z0-9]+", normalized):
            if len(token) >= 4 and token not in {"core", "main", "test", "tests", "project"}:
                tokens.add(token)
    return tokens


def choose_wrong_pair(
    sample: dict[str, Any],
    source_profile: tuple[str, str],
    corpus_pairs: list[CorpusPair],
    per_wrong_repo: Counter[str],
    max_per_wrong_repo: int,
) -> CorpusPair | None:
    pairs = choose_wrong_pairs(
        sample=sample,
        source_profile=source_profile,
        corpus_pairs=corpus_pairs,
        per_wrong_repo=per_wrong_repo,
        max_per_wrong_repo=max_per_wrong_repo,
        limit=1,
    )
    return pairs[0] if pairs else None


def choose_wrong_pairs(
    sample: dict[str, Any],
    source_profile: tuple[str, str],
    corpus_pairs: list[CorpusPair],
    per_wrong_repo: Counter[str],
    max_per_wrong_repo: int,
    limit: int,
) -> list[CorpusPair]:
    source_repo = str(sample.get("repo") or "")
    language, domain = source_profile
    candidates = [
        pair
        for pair in corpus_pairs
        if pair.repo != source_repo
        and pair.language == language
        and per_wrong_repo[pair.repo] < max_per_wrong_repo
    ]
    if not candidates:
        return []
    same_domain = [pair for pair in candidates if pair.domain == domain]
    pool = same_domain or candidates
    key = stable_id(sample.get("id"), source_repo, language, domain)
    offset = int(key[:8], 16) % len(pool)
    rotated = sorted(pool, key=lambda pair: (pair.repo, pair.base_commit))
    rotated = rotated[offset:] + rotated[:offset]
    selected: list[CorpusPair] = []
    prospective_counts: Counter[str] = Counter()
    for pair in rotated:
        if len(selected) >= limit:
            break
        if per_wrong_repo[pair.repo] + prospective_counts[pair.repo] >= max_per_wrong_repo:
            continue
        selected.append(pair)
        prospective_counts[pair.repo] += 1
    return selected


def build_counterfactual_sample(
    source_sample: dict[str, Any],
    wrong_pair: CorpusPair,
    query_text: str,
    source_profile: tuple[str, str],
) -> dict[str, Any]:
    source_repo = str(source_sample.get("repo") or "")
    source_id = str(source_sample.get("id") or "")
    source_url = source_url_for_sample(source_sample)
    sample_id = "abstention__counterfactual__" + stable_id(source_id, source_repo, wrong_pair.repo, wrong_pair.base_commit)
    profile = f"{source_profile[0]}:{source_profile[1]}"
    evidence_summary = (
        "Counterfactual wrong-repo sample: this query is derived from a real positive ARB sample "
        f"({source_id}) in {source_repo}, but the candidate corpus is {wrong_pair.repo} at {wrong_pair.base_commit}. "
        f"The paired repository shares profile {profile}; source repository paths and repository identity tokens were excluded from the query."
    )
    return {
        "id": sample_id,
        "task_type": "abstention",
        "repo": wrong_pair.repo,
        "base_commit": wrong_pair.base_commit,
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": wrong_pair.base_commit},
        "query": {"source": "counterfactual_wrong_repo", "text": query_text},
        "gold": {"files": [], "no_gold": True, "reason": "counterfactual_wrong_repo"},
        "metadata": {
            "source_url": source_url,
            "evidence_urls": [source_url] if source_url else [],
            "evidence_summary": evidence_summary,
            "organic": False,
            "source_sample_id": source_id,
            "source_repo": source_repo,
            "source_base_commit": source_sample.get("base_commit"),
            "source_task_type": source_sample.get("task_type"),
            "pairing_profile": profile,
            "wrong_repo_chunks_path": wrong_pair.chunks_path,
        },
        "audit": {"verdict": "", "notes": ""},
    }


def source_url_for_sample(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    return str(metadata.get("pr_url") or metadata.get("source_url") or metadata.get("url") or "")


def audit_packet_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    return {
        "id": sample["id"],
        "repo": sample["repo"],
        "base_commit": sample["base_commit"],
        "query": sample["query"],
        "proposed_no_gold_reason": (sample.get("gold") or {}).get("reason"),
        "evidence": {
            "source_url": metadata.get("source_url") or "",
            "resolution_comments": [],
            "rerun_status": "",
            "linked_commits": [],
            "why_no_local_fix": metadata.get("evidence_summary") or "",
            "counterfactual": {
                "source_sample_id": metadata.get("source_sample_id"),
                "source_repo": metadata.get("source_repo"),
                "source_task_type": metadata.get("source_task_type"),
                "pairing_profile": metadata.get("pairing_profile"),
            },
        },
        "audit_fields": {"verdict": "", "notes": "", "has_local_gold_files": []},
    }


def counterfactual_audit_row(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    query = sample.get("query") or {}
    return {
        "id": sample.get("id"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "query_source": query.get("source"),
        "query_text": query.get("text"),
        "proposed_no_gold_reason": (sample.get("gold") or {}).get("reason"),
        "source_url": metadata.get("source_url"),
        "evidence_summary": metadata.get("evidence_summary"),
        "source_sample_id": metadata.get("source_sample_id"),
        "source_repo": metadata.get("source_repo"),
        "source_task_type": metadata.get("source_task_type"),
        "pairing_profile": metadata.get("pairing_profile"),
        "verdict": "",
        "notes": "",
        "has_local_gold_files": "",
    }


def audit_packet_csv_row(packet: dict[str, Any]) -> dict[str, Any]:
    query = packet.get("query") or {}
    evidence = packet.get("evidence") or {}
    counterfactual = evidence.get("counterfactual") or {}
    audit_fields = packet.get("audit_fields") or {}
    return {
        "id": packet.get("id"),
        "repo": packet.get("repo"),
        "base_commit": packet.get("base_commit"),
        "query_source": query.get("source"),
        "query_text": query.get("text"),
        "proposed_no_gold_reason": packet.get("proposed_no_gold_reason"),
        "source_url": evidence.get("source_url"),
        "resolution_comments": json.dumps(evidence.get("resolution_comments") or [], ensure_ascii=False),
        "rerun_status": evidence.get("rerun_status") or "",
        "linked_commits": json.dumps(evidence.get("linked_commits") or [], ensure_ascii=False),
        "why_no_local_fix": evidence.get("why_no_local_fix") or "",
        "evidence_snippets": json.dumps(evidence.get("evidence_snippets") or [], ensure_ascii=False),
        "source_sample_id": counterfactual.get("source_sample_id"),
        "source_repo": counterfactual.get("source_repo"),
        "source_task_type": counterfactual.get("source_task_type"),
        "pairing_profile": counterfactual.get("pairing_profile"),
        "verdict": audit_fields.get("verdict") or "",
        "notes": audit_fields.get("notes") or "",
        "has_local_gold_files": json.dumps(audit_fields.get("has_local_gold_files") or [], ensure_ascii=False),
    }


def validate_abstention_audit_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, packet in enumerate(packets, start=1):
        errors: list[str] = []
        packet_id = str(packet.get("id") or "")
        if not packet_id:
            errors.append("missing id")
        elif packet_id in seen:
            errors.append("duplicate id")
        seen.add(packet_id)
        if not str(packet.get("repo") or ""):
            errors.append("missing repo")
        if not str(packet.get("base_commit") or ""):
            errors.append("missing base_commit")
        reason = str(packet.get("proposed_no_gold_reason") or "")
        if reason not in ABSTENTION_REASONS:
            errors.append("invalid proposed_no_gold_reason")
        query = packet.get("query") or {}
        if query.get("source") not in {"ci_log", "issue", "counterfactual_wrong_repo"}:
            errors.append("invalid query source")
        if not str(query.get("text") or "").strip():
            errors.append("missing query text")
        if not packet_has_resolution_evidence(packet):
            errors.append("missing resolution evidence")
        if packet_query_contains_resolution_answer(packet):
            errors.append("query contains resolution answer")
        if errors:
            invalid.append({"row": index, "id": packet_id or f"row-{index}", "errors": errors})
    return invalid


def packet_has_resolution_evidence(packet: dict[str, Any]) -> bool:
    evidence = packet.get("evidence") or {}
    if evidence.get("counterfactual"):
        return bool(evidence.get("why_no_local_fix"))
    evidence_values = [
        evidence.get("why_no_local_fix"),
        evidence.get("rerun_status"),
        *(evidence.get("resolution_comments") or []),
        *(evidence.get("evidence_snippets") or []),
    ]
    return any(str(value or "").strip() for value in evidence_values)


def packet_query_contains_resolution_answer(packet: dict[str, Any]) -> bool:
    reason = str(packet.get("proposed_no_gold_reason") or "")
    query = packet.get("query") or {}
    return query_contains_resolution_answer(str(query.get("text") or ""), reason, packet)


def sample_query_contains_resolution_answer(sample: dict[str, Any]) -> bool:
    reason = str((sample.get("gold") or {}).get("reason") or "")
    return query_contains_resolution_answer(str((sample.get("query") or {}).get("text") or ""), reason, sample)


def query_contains_resolution_answer(text: str, reason: str, container: dict[str, Any] | None = None) -> bool:
    lowered = text.lower()
    if reason == "upstream_dependency":
        return issue_query_contains_resolution_answer(text, reason)
    hints = RESOLUTION_ANSWER_HINTS.get(reason, ())
    if any(hint in lowered for hint in hints):
        return True
    if reason == "counterfactual_wrong_repo" and container:
        evidence = container.get("evidence") or {}
        metadata = container.get("metadata") or {}
        counterfactual = evidence.get("counterfactual") or {}
        source_repo = str(counterfactual.get("source_repo") or metadata.get("source_repo") or "")
        if source_repo and query_mentions_repo_identity(text, source_repo):
            return True
    return False


def counterfactual_packet_diagnostics(packets: list[dict[str, Any]]) -> dict[str, Any]:
    counterfactual_packets = [
        packet
        for packet in packets
        if str(packet.get("proposed_no_gold_reason") or "") == "counterfactual_wrong_repo"
    ]
    source_identity_leaks: list[dict[str, str]] = []
    missing_pairing_metadata: list[dict[str, str]] = []
    by_pairing_profile: Counter[str] = Counter()
    by_source_repo: Counter[str] = Counter()
    by_wrong_repo: Counter[str] = Counter()

    for packet in counterfactual_packets:
        evidence = packet.get("evidence") or {}
        counterfactual = evidence.get("counterfactual") or {}
        packet_id = str(packet.get("id") or "")
        wrong_repo = str(packet.get("repo") or "")
        source_repo = str(counterfactual.get("source_repo") or "")
        pairing_profile = str(counterfactual.get("pairing_profile") or "")
        query_text = str((packet.get("query") or {}).get("text") or "")

        by_wrong_repo[wrong_repo or "missing"] += 1
        by_source_repo[source_repo or "missing"] += 1
        by_pairing_profile[pairing_profile or "missing"] += 1

        if source_repo and query_mentions_repo_identity(query_text, source_repo):
            source_identity_leaks.append({"id": packet_id, "source_repo": source_repo, "wrong_repo": wrong_repo})
        if not source_repo or not pairing_profile:
            missing_pairing_metadata.append(
                {
                    "id": packet_id,
                    "source_repo": source_repo,
                    "wrong_repo": wrong_repo,
                    "pairing_profile": pairing_profile,
                }
            )

    return {
        "count": len(counterfactual_packets),
        "source_identity_leaks": source_identity_leaks,
        "missing_pairing_metadata": missing_pairing_metadata,
        "by_pairing_profile": dict(sorted(by_pairing_profile.items())),
        "by_source_repo": dict(sorted(by_source_repo.items())),
        "by_wrong_repo": dict(sorted(by_wrong_repo.items())),
    }


def pool_for_candidate(row: dict[str, Any]) -> str:
    reason = str((row.get("gold") or {}).get("reason") or row.get("candidate_reason") or row.get("no_gold_reason") or "")
    if reason == "counterfactual_wrong_repo" or row.get("task_type") == "abstention":
        return "counterfactual_wrong_repo"
    return "organic_no_gold"


def pool_for_packet(packet: dict[str, Any]) -> str:
    if packet.get("proposed_no_gold_reason") == "counterfactual_wrong_repo":
        return "counterfactual_wrong_repo"
    return "organic_no_gold"


def read_audit_verdict_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        rows, _fields = read_csv_rows_with_fields(path)
        return rows
    return read_jsonl(path)


def read_csv_rows_with_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def normalize_abstention_verdict(value: Any) -> str:
    verdict = str(value or "").strip().lower()
    aliases = {
        "valid": "valid_no_gold",
        "keep": "valid_no_gold",
        "no_gold": "valid_no_gold",
        "local_gold": "has_local_gold",
        "invalid": "misconstructed",
        "too_easy": "too_easy_irrelevant",
    }
    return aliases.get(verdict, verdict)


def clean_sample_from_packet(packet: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    reason = str(packet.get("proposed_no_gold_reason") or "")
    evidence = packet.get("evidence") or {}
    counterfactual = evidence.get("counterfactual") or {}
    source_url = str(evidence.get("source_url") or "")
    sample = {
        "id": str(packet.get("id") or ""),
        "task_type": "abstention",
        "repo": str(packet.get("repo") or ""),
        "base_commit": str(packet.get("base_commit") or ""),
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": str(packet.get("base_commit") or "")},
        "query": packet.get("query") or {},
        "gold": {"files": [], "no_gold": True, "reason": reason},
        "metadata": {
            "source_url": source_url,
            "evidence_urls": [source_url] if source_url else [],
            "evidence_summary": evidence.get("why_no_local_fix") or "",
            "organic": reason != "counterfactual_wrong_repo",
        },
        "audit": {
            "verdict": "valid_no_gold",
            "notes": str(audit_row.get("notes") or ""),
        },
    }
    if counterfactual:
        sample["metadata"].update(
            {
                "source_sample_id": counterfactual.get("source_sample_id"),
                "source_repo": counterfactual.get("source_repo"),
                "source_task_type": counterfactual.get("source_task_type"),
                "pairing_profile": counterfactual.get("pairing_profile"),
            }
        )
    return sample


def stratified_packet_order(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        buckets[audit_worklist_bucket_key(packet)].append(packet)
    for bucket in buckets.values():
        bucket.sort(key=lambda packet: str(packet.get("id") or ""))
    ordered: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def audit_worklist_bucket_key(packet: dict[str, Any]) -> tuple[str, ...]:
    reason = str(packet.get("proposed_no_gold_reason") or "")
    repo = str(packet.get("repo") or "")
    if reason == "counterfactual_wrong_repo":
        counterfactual = (packet.get("evidence") or {}).get("counterfactual") or {}
        return (
            "counterfactual_wrong_repo",
            str(counterfactual.get("pairing_profile") or "missing_profile"),
            repo,
            str(counterfactual.get("source_repo") or "missing_source_repo"),
        )
    return ("organic_no_gold", reason or "missing_reason", repo)


def audit_worklist_row(packet: dict[str, Any], order: int, priority: str) -> dict[str, Any]:
    row = audit_packet_csv_row(packet)
    reason = str(packet.get("proposed_no_gold_reason") or "")
    row.update(
        {
            "audit_order": order,
            "audit_priority": priority,
            "pool": pool_for_packet(packet),
            "review_focus": review_focus_for_reason(reason),
            "allowed_verdicts": "|".join(sorted(ABSTENTION_VERDICTS)),
        }
    )
    return row


def review_focus_for_reason(reason: str) -> str:
    if reason == "counterfactual_wrong_repo":
        return "Check wrong repo plausibility, source-repo leakage, and whether this repo has any needed local files."
    if reason == "flaky_ci":
        return "Verify same-SHA rerun or infra evidence, no local fix, and no flaky diagnosis leaked into query."
    if reason == "upstream_dependency":
        return "Verify maintainer upstream attribution, no local workaround, and resolution evidence stays out of query."
    if reason == "external_service":
        return "Verify external service, credential, provider, or environment cause with no local config or source fix."
    if reason == "user_error":
        return "Verify maintainer explanation, plausible coding-workflow query, and no local file is needed."
    return "Verify no-gold evidence, query construction, and local-gold absence."


def completion_requirement(name: str, passed: bool, evidence: str, pending: bool = False) -> dict[str, str]:
    if passed:
        status = "passed"
    elif pending:
        status = "pending"
    else:
        status = "failed"
    return {"name": name, "status": status, "evidence": evidence}


def completion_status(requirements: list[dict[str, str]], crawling: dict[str, Any]) -> str:
    if all(item["status"] == "passed" for item in requirements):
        return "complete"
    if any(item["status"] == "failed" for item in requirements):
        return "not_ready"
    if crawling.get("status") == "ready_for_manual_audit":
        return "ready_for_manual_audit"
    return "not_ready"


def finalization_next_action(audit_report: dict[str, Any], completion_report: dict[str, Any], exported: bool) -> str:
    if not exported:
        pending = audit_report.get("pending")
        if pending:
            return f"Complete manual audit verdicts for {pending} pending packet(s), then rerun finalization."
        return "Fix audit gates before export; inspect audit_report.gates for the failing requirement."
    if completion_report.get("status") == "complete":
        return "No required action remains; final clean pilot satisfies the plan gates."
    return "Inspect completion_report requirements and fix any pending or failed final clean gate."


def balance_clean_selection(
    samples: list[dict[str, Any]],
    max_samples: int | None = None,
    max_counterfactual_share: float | None = None,
) -> list[dict[str, Any]]:
    if max_samples is None and max_counterfactual_share is None:
        return samples
    limit = max_samples if max_samples is not None else len(samples)
    if limit <= 0:
        return []
    organic = [sample for sample in samples if (sample.get("gold") or {}).get("reason") != "counterfactual_wrong_repo"]
    counterfactual = [sample for sample in samples if (sample.get("gold") or {}).get("reason") == "counterfactual_wrong_repo"]
    if max_counterfactual_share is None:
        return samples[:limit]
    share = max(0.0, min(1.0, max_counterfactual_share))
    best_org = 0
    best_counterfactual = 0
    for organic_count in range(0, min(len(organic), limit) + 1):
        remaining = limit - organic_count
        if share >= 1.0:
            counterfactual_allowed_by_share = len(counterfactual)
        elif share <= 0.0:
            counterfactual_allowed_by_share = 0
        else:
            counterfactual_allowed_by_share = int((share / (1.0 - share)) * organic_count)
        counterfactual_count = min(len(counterfactual), remaining, counterfactual_allowed_by_share)
        total = organic_count + counterfactual_count
        best_total = best_org + best_counterfactual
        if total > best_total or (total == best_total and counterfactual_count > best_counterfactual):
            best_org = organic_count
            best_counterfactual = counterfactual_count
    selected = organic[:best_org] + counterfactual[:best_counterfactual]
    original_order = {id(sample): index for index, sample in enumerate(samples)}
    selected.sort(key=lambda sample: original_order[id(sample)])
    return selected


def validate_abstention_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        errors: list[str] = []
        sample_id = str(sample.get("id") or "")
        if not sample_id:
            errors.append("missing id")
        elif sample_id in seen:
            errors.append("duplicate id")
        seen.add(sample_id)
        if sample.get("task_type") != "abstention":
            errors.append("task_type must be abstention")
        gold = sample.get("gold") or {}
        if gold.get("files") != []:
            errors.append("gold.files must be empty")
        if gold.get("no_gold") is not True:
            errors.append("gold.no_gold must be true")
        if gold.get("reason") not in ABSTENTION_REASONS:
            errors.append("invalid no_gold reason")
        query = sample.get("query") or {}
        if query.get("source") not in {"ci_log", "issue", "counterfactual_wrong_repo"}:
            errors.append("invalid query source")
        if not str(query.get("text") or "").strip():
            errors.append("missing query text")
        if errors:
            invalid.append({"row": index, "id": sample_id or f"row-{index}", "errors": errors})
    return invalid


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_counterfactual_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Counterfactual Wrong-Repo Report",
        "",
        f"- Selected candidates: `{report['selected']}`",
        f"- Source samples scanned: `{report['source_samples']}`",
        f"- Corpus pairs available: `{report['corpus_pairs']}`",
        f"- Ready for manual audit: `{report['ready_for_manual_audit']}`",
        "",
        "## Drops",
        "",
    ]
    for reason, count in report.get("dropped", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Pairing Profiles", ""])
    for profile, count in report.get("by_profile", {}).items():
        lines.append(f"- `{profile}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_organic_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Organic Candidate Report",
        "",
        f"- Scanned check groups: `{report['scanned_check_groups']}`",
        f"- Raw candidates: `{report['raw_candidates']}`",
        f"- Prefiltered candidates: `{report['prefiltered']}`",
        f"- Ready for manual audit: `{report['ready_for_manual_audit']}`",
        "",
        "## Drops",
        "",
    ]
    for reason, count in report.get("dropped", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Prefiltered By Repo", ""])
    for repo, count in report.get("by_repo", {}).items():
        lines.append(f"- `{repo}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_audit_packet_merge_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Audit Packet Merge Report",
        "",
        f"- Total packets: `{report['total']}`",
        f"- Ready for manual audit: `{report['ready_for_manual_audit']}`",
        f"- Duplicate rows: `{len(report.get('duplicates', []))}`",
        f"- Missing-id rows: `{len(report.get('missing_id_rows', []))}`",
        f"- Invalid packets: `{len(report.get('invalid_packets', []))}`",
        "",
        "## By Query Source",
        "",
    ]
    for source, count in report.get("by_query_source", {}).items():
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(["", "## By Reason", ""])
    for reason, count in report.get("by_reason", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_crawling_status_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Crawling Status Report",
        "",
        f"- Prefiltered candidates: `{report['prefiltered_candidates']}`",
        f"- Audit packet candidates: `{report['audit_packet_candidates']}`",
        f"- Status: `{report['status']}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Prefiltered By Pool", ""])
    for pool, count in report.get("prefiltered_by_pool", {}).items():
        lines.append(f"- `{pool}`: `{count}`")
    lines.extend(["", "## Audit Packets By Pool", ""])
    for pool, count in report.get("audit_packets_by_pool", {}).items():
        lines.append(f"- `{pool}`: `{count}`")
    diagnostics = report.get("counterfactual_diagnostics") or {}
    lines.extend(
        [
            "",
            "## Counterfactual Diagnostics",
            "",
            f"- Counterfactual packets: `{diagnostics.get('count')}`",
            f"- Source identity leaks: `{len(diagnostics.get('source_identity_leaks') or [])}`",
            f"- Missing pairing metadata: `{len(diagnostics.get('missing_pairing_metadata') or [])}`",
            "",
            "### Pairing Profiles",
            "",
        ]
    )
    for profile, count in (diagnostics.get("by_pairing_profile") or {}).items():
        lines.append(f"- `{profile}`: `{count}`")
    return "\n".join(lines) + "\n"


def render_abstention_audit_report(report: dict[str, Any]) -> str:
    preview = report.get("balanced_clean_preview", {})
    lines = [
        "# Abstention Audit Verdict Report",
        "",
        f"- Audit packet candidates: `{report['audit_packet_candidates']}`",
        f"- Audit rows: `{report['audit_rows']}`",
        f"- Reviewed: `{report['reviewed']}`",
        f"- Pending: `{report['pending']}`",
        f"- Valid rate: `{report['valid_rate']}`",
        f"- Status: `{report['status']}`",
        "",
        "## Balanced Clean Preview",
        "",
        f"- Total: `{preview.get('total')}`",
        f"- Organic: `{preview.get('organic')}`",
        f"- Counterfactual: `{preview.get('counterfactual')}`",
        f"- Counterfactual share: `{preview.get('counterfactual_share')}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Verdicts", ""])
    for verdict, count in report.get("verdicts", {}).items():
        lines.append(f"- `{verdict}`: `{count}`")
    lines.extend(["", "## By Pool", ""])
    for pool, counts in report.get("by_pool", {}).items():
        lines.append(f"- `{pool}`: `{json.dumps(counts, sort_keys=True)}`")
    lines.extend(["", "## By Reason", ""])
    for reason, counts in report.get("by_reason", {}).items():
        lines.append(f"- `{reason}`: `{json.dumps(counts, sort_keys=True)}`")
    return "\n".join(lines) + "\n"


def render_abstention_verdict_apply_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Audit Verdict Apply Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Source rows: `{report['source_rows']}`",
        f"- Target rows: `{report['target_rows']}`",
        f"- Applied verdicts: `{report['applied']}`",
        f"- Ignored blank verdicts: `{report['ignored_blank']}`",
        f"- Remaining missing verdicts: `{report['remaining_missing']}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Verdicts After Apply", ""])
    for verdict, count in report.get("verdicts_after", {}).items():
        lines.append(f"- `{verdict}`: `{count}`")
    lines.extend(["", "## Problems", ""])
    problem_counts = {
        "invalid_verdicts": len(report.get("invalid_verdicts", [])),
        "unknown_ids": len(report.get("unknown_ids", [])),
        "duplicate_source_ids": len(report.get("duplicate_source_ids", [])),
        "duplicate_target_ids": len(report.get("duplicate_target_ids", [])),
        "conflicts": len(report.get("conflicts", [])),
    }
    for key, value in problem_counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_abstention_worklist_report(report: dict[str, Any]) -> str:
    core = report.get("core", {})
    reserve = report.get("reserve", {})
    instructions = report.get("audit_instructions", {})
    lines = [
        "# Abstention Manual Audit Worklist",
        "",
        f"- Status: `{report['status']}`",
        f"- Total packets: `{report['total_packets']}`",
        f"- Core audit rows: `{core.get('total')}`",
        f"- Reserve rows: `{reserve.get('total')}`",
        f"- Invalid packets: `{len(report.get('invalid_packets', []))}`",
        "",
        "## Core Balance",
        "",
        f"- Organic: `{core.get('organic')}`",
        f"- Counterfactual: `{core.get('counterfactual')}`",
        f"- Counterfactual share: `{core.get('counterfactual_share')}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(
        [
            "",
            "## Audit Rules",
            "",
            f"- Allowed verdicts: `{', '.join(instructions.get('allowed_verdicts', []))}`",
            f"- Complete packet audit requires manual verdicts for: `{instructions.get('manual_verdicts_required_for_complete_packet_audit')}` rows",
            "- Only `valid_no_gold` rows may enter the clean export.",
            "- Do not infer labels from this worklist ordering; `core` is only the first-pass balanced review target.",
            "",
            "## Core By Reason",
            "",
        ]
    )
    for reason, count in core.get("by_reason", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Reserve By Reason", ""])
    for reason, count in reserve.get("by_reason", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_abstention_completion_audit_report(report: dict[str, Any]) -> str:
    crawling = report.get("crawling_status", {})
    audit = report.get("audit_status", {})
    final = report.get("final_clean_status", {})
    lines = [
        "# Abstention Completion Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Requirement status counts: `{json.dumps(report.get('status_counts', {}), sort_keys=True)}`",
        "",
        "## Current State",
        "",
        f"- Prefiltered candidates: `{crawling.get('prefiltered_candidates')}`",
        f"- Audit packet candidates: `{crawling.get('audit_packet_candidates')}`",
        f"- Audit reviewed: `{audit.get('reviewed')}`",
        f"- Audit pending: `{audit.get('pending')}`",
        f"- Audit valid rate: `{audit.get('valid_rate')}`",
        f"- Final clean exists: `{final.get('exists')}`",
        f"- Final clean total: `{final.get('total')}`",
        f"- Final organic/counterfactual: `{final.get('organic')}` / `{final.get('counterfactual')}`",
        "",
        "## Requirements",
        "",
    ]
    for item in report.get("requirements", []):
        lines.append(f"- `{item['status']}` `{item['name']}`: {item['evidence']}")
    lines.extend(["", "## Next Required Action", ""])
    if report["status"] == "complete":
        lines.append("- No required action remains; the plan gates are satisfied.")
    elif audit.get("pending"):
        lines.append("- Fill manual audit verdicts, then rerun `report-abstention-audit` and `export-abstention-clean`.")
    elif not final.get("exists"):
        lines.append("- Export the clean set from reviewed `valid_no_gold` rows and rerun this completion audit.")
    else:
        lines.append("- Resolve failed requirements listed above and rerun this completion audit.")
    return "\n".join(lines) + "\n"


def render_abstention_finalization_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Finalization Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Exported clean samples: `{report['exported']}`",
        f"- Audit status: `{report['audit_status']}`",
        f"- Pilot status: `{report['pilot_status']}`",
        f"- Completion status: `{report['completion_status']}`",
        f"- Completion status counts: `{json.dumps(report.get('completion_status_counts', {}), sort_keys=True)}`",
        "",
        "## Next Action",
        "",
        f"- {report['next_action']}",
        "",
        "## Audit Gates",
        "",
    ]
    for gate, value in (report.get("audit_gates") or {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    if report.get("pilot_gates"):
        lines.extend(["", "## Pilot Gates", ""])
        for gate, value in report.get("pilot_gates", {}).items():
            lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_abstention_worklist_shard_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Audit Worklist Shards",
        "",
        f"- Status: `{report['status']}`",
        f"- Source rows: `{report['source_rows']}`",
        f"- Selected rows: `{report['selected_rows']}`",
        f"- Shards: `{len(report.get('shards', []))}`",
        f"- Priority: `{report['inputs']['priority']}`",
        f"- Shard size: `{report['inputs']['shard_size']}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Selected By Pool", ""])
    for pool, count in report.get("selected_by_pool", {}).items():
        lines.append(f"- `{pool}`: `{count}`")
    lines.extend(["", "## Selected By Reason", ""])
    for reason, count in report.get("selected_by_reason", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Shards", ""])
    for shard in report.get("shards", []):
        lines.append(
            f"- `#{shard['index']}` rows `{shard['rows']}` orders `{shard['first_audit_order']}`-`{shard['last_audit_order']}`: `{shard['csv']}`"
        )
    lines.extend(["", "## Audit Rules", ""])
    lines.append(f"- Allowed verdicts: `{', '.join(report.get('allowed_verdicts', []))}`")
    lines.append("- Leave `verdict` blank until manually reviewed; only `valid_no_gold` can enter the clean export.")
    return "\n".join(lines) + "\n"


def render_abstention_shard_progress_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Shard Audit Progress",
        "",
        f"- Status: `{report['status']}`",
        f"- Total rows: `{report['total_rows']}`",
        f"- Reviewed: `{report['reviewed']}`",
        f"- Pending: `{report['pending']}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Verdicts", ""])
    for verdict, count in report.get("verdicts", {}).items():
        lines.append(f"- `{verdict}`: `{count}`")
    lines.extend(["", "## Shards", ""])
    for shard in report.get("shards", []):
        lines.append(
            f"- `{shard['path']}`: reviewed `{shard['reviewed']}`, pending `{shard['pending']}`, complete `{shard['complete']}`"
        )
    lines.extend(["", "## Problems", ""])
    lines.append(f"- Invalid verdicts: `{len(report.get('invalid_verdicts', []))}`")
    lines.append(f"- Duplicate IDs: `{len(report.get('duplicate_ids', []))}`")
    lines.append(f"- Conflicting verdicts: `{len(report.get('conflicting_verdicts', []))}`")
    return "\n".join(lines) + "\n"


def render_abstention_review_packets_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Review Packet Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Packets: `{len(report.get('packets', []))}`",
        f"- Total rows: `{report['total_rows']}`",
        f"- Total pending: `{report['total_pending']}`",
        "",
        "## Packets",
        "",
    ]
    for packet in report.get("packets", []):
        lines.append(f"- `{packet['markdown']}` from `{packet['source']}`: rows `{packet['rows']}`, pending `{packet['pending']}`")
    return "\n".join(lines) + "\n"


def render_abstention_handoff_manifest_report(report: dict[str, Any]) -> str:
    progress = report.get("shard_progress") or {}
    completion = report.get("completion_audit") or {}
    finalization = report.get("finalization") or {}
    reviewer_shards = report.get("reviewer_shards") or {}
    lines = [
        "# Abstention Audit Handoff Manifest",
        "",
        f"- Status: `{report['status']}`",
        f"- Files listed: `{report['file_count']}`",
        f"- Missing required files: `{len(report.get('missing_required') or [])}`",
        f"- Reviewer shard CSVs: `{reviewer_shards.get('csv')}`",
        f"- Reviewer shard JSONLs: `{reviewer_shards.get('jsonl')}`",
        f"- Review packet Markdown files: `{report.get('review_packets')}`",
        f"- Shard progress: `{progress.get('reviewed')}` reviewed / `{progress.get('pending')}` pending",
        f"- Completion audit: `{completion.get('status')}` `{json.dumps(completion.get('status_counts') or {}, sort_keys=True)}`",
        f"- Finalization: `{finalization.get('status')}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Next Commands", ""])
    for command in report.get("next_commands", []):
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["", "## Files", ""])
    for entry in report.get("files", []):
        rows = f", rows `{entry.get('rows')}`" if "rows" in entry else ""
        sha = f", sha256 `{entry.get('sha256')}`" if entry.get("sha256") else ""
        lines.append(
            f"- `{entry['path']}`: role `{entry['role']}`, required `{entry['required']}`, exists `{entry['exists']}`{rows}{sha}"
        )
    return "\n".join(lines) + "\n"


def render_abstention_issue_base_backfill_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Issue Base Commit Backfill",
        "",
        f"- Status: `{report['status']}`",
        f"- Total issues: `{report['total_issues']}`",
        f"- Missing before: `{report['missing_before']}`",
        f"- Backfilled: `{report['backfilled']}`",
        f"- Missing after: `{report['missing_after']}`",
        f"- Errors: `{len(report.get('errors') or [])}`",
        "",
        "## Repositories",
        "",
    ]
    for repo in report.get("repos", []):
        lines.append(
            f"- `{repo['repo']}`: issues `{repo['issues']}`, missing before `{repo['missing_before']}`, "
            f"backfilled `{repo['backfilled']}`, missing after `{repo['missing_after']}`"
        )
    return "\n".join(lines) + "\n"


def render_abstention_review_packet_markdown(
    shard_path: Path,
    rows: list[dict[str, Any]],
    query_limit: int = 1800,
    evidence_limit: int = 1200,
) -> str:
    lines = [
        f"# Abstention Review Packet: {shard_path.name}",
        "",
        "Fill verdicts in the source CSV, not this Markdown file.",
        "",
        "Allowed verdicts: `valid_no_gold`, `has_local_gold`, `ambiguous`, `too_easy_irrelevant`, `misconstructed`.",
        "",
        "Only `valid_no_gold` rows can enter the final clean set.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        order = row.get("audit_order") or index
        lines.extend(
            [
                f"## {order}. `{row.get('id')}`",
                "",
                f"- Repo: `{row.get('repo')}`",
                f"- Base commit: `{row.get('base_commit')}`",
                f"- Pool: `{row.get('pool')}`",
                f"- Reason: `{row.get('proposed_no_gold_reason')}`",
                f"- Query source: `{row.get('query_source')}`",
                f"- Source URL: {row.get('source_url') or ''}",
                f"- Review focus: {row.get('review_focus') or ''}",
                "",
                "### Query",
                "",
                "```text",
                truncate_text(str(row.get("query_text") or ""), query_limit),
                "```",
                "",
                "### Evidence",
                "",
            ]
        )
        if row.get("why_no_local_fix"):
            lines.extend(["Why no local fix:", "", "```text", truncate_text(str(row.get("why_no_local_fix") or ""), evidence_limit), "```", ""])
        if row.get("rerun_status"):
            lines.append(f"- Rerun status: `{row.get('rerun_status')}`")
        for label, key in [
            ("Resolution comments", "resolution_comments"),
            ("Evidence snippets", "evidence_snippets"),
            ("Linked commits", "linked_commits"),
        ]:
            values = parse_json_cell_list(row.get(key))
            if values:
                lines.extend(["", f"{label}:", ""])
                for value in values:
                    lines.extend(["```text", truncate_text(value, evidence_limit), "```", ""])
        if row.get("source_repo") or row.get("source_sample_id"):
            lines.extend(
                [
                    "Counterfactual source:",
                    "",
                    f"- Source repo: `{row.get('source_repo') or ''}`",
                    f"- Source sample: `{row.get('source_sample_id') or ''}`",
                    f"- Source task type: `{row.get('source_task_type') or ''}`",
                    f"- Pairing profile: `{row.get('pairing_profile') or ''}`",
                    "",
                ]
            )
        lines.extend(
            [
                "### Audit Fields To Fill In CSV",
                "",
                "- `verdict`: ",
                "- `notes`: ",
                "- `has_local_gold_files`: ",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_json_cell_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if parsed:
        return [str(parsed)]
    return []


def render_abstention_pilot_report(report: dict[str, Any]) -> str:
    lines = [
        "# Abstention Pilot Report",
        "",
        f"- Total clean samples: `{report['total']}`",
        f"- Organic samples: `{report['organic']}`",
        f"- Counterfactual samples: `{report['counterfactual']}`",
        f"- Valid rate: `{report['valid_rate']}`",
        f"- Status: `{report['status']}`",
        "",
        "## Gates",
        "",
    ]
    for gate, value in report.get("gates", {}).items():
        lines.append(f"- `{gate}`: `{value}`")
    lines.extend(["", "## Verdicts", ""])
    for verdict, count in report.get("verdicts", {}).items():
        lines.append(f"- `{verdict}`: `{count}`")
    lines.extend(["", "## Reasons", ""])
    for reason, count in report.get("reasons", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    return "\n".join(lines) + "\n"
