from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from .baseline import (
    iter_samples,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    supporting_context_files,
    target_gold_files,
)
from .bcy_curve import TOKEN_RE, count_tokens
from .curate import filter_samples, load_keep_ids
from .indexed_tools import INDEXED_TOOLS, TOOL_DESCRIPTIONS, TOOL_EXAMPLES, IndexedTools
from .io import ensure_parent, read_jsonl, utc_now, write_json
from .trajectory import summarize_trajectory_details, trajectory_detail, write_jsonl

DEFAULT_SAMPLES = (Path("data/benchmark/v1_3_reviewed/samples.jsonl"),)
DEFAULT_CORPUS_DIR = Path("data/corpus/v1_2")
DEFAULT_OUT = Path("data/eval/v1_4/closed_tool_grep_summary.json")
DEFAULT_DETAILS = Path("data/eval/v1_4/closed_tool_grep_details.jsonl")
DEFAULT_REPORT = Path("data/reports/v1_4/closed_tool_grep.md")
DEFAULT_LLM_OUT = Path("data/eval/v1_4/closed_tool_llm_summary.json")
DEFAULT_LLM_DETAILS = Path("data/eval/v1_4/closed_tool_llm_details.jsonl")
DEFAULT_LLM_REPORT = Path("data/reports/v1_4/closed_tool_llm.md")
DEFAULT_CODEX_OUT = Path("data/eval/v1_4/closed_tool_codex_summary.json")
DEFAULT_CODEX_DETAILS = Path("data/eval/v1_4/closed_tool_codex_details.jsonl")
DEFAULT_CODEX_REPORT = Path("data/reports/v1_4/closed_tool_codex.md")
DEFAULT_BUDGET_CURVE_OUT = Path("data/reports/v1_4/closed_tool_budget_curve.json")
DEFAULT_BUDGET_CURVE_REPORT = Path("data/reports/v1_4/closed_tool_budget_curve.md")
DEFAULT_SEED_INTERVENTION_OUT = Path("data/reports/v1_4/closed_tool_seed_intervention.json")
DEFAULT_SEED_INTERVENTION_REPORT = Path("data/reports/v1_4/closed_tool_seed_intervention.md")
DEFAULT_CODEX_BIN = Path(os.environ.get("CODEX_BIN", "/home/ubuntu/.nvm/versions/node/v22.22.0/bin/codex"))
DEFAULT_CODEX_WORK_ROOT = Path("/tmp/arb-codex-closed-tool")
DEFAULT_SELECTION_OUT = Path("data/eval/indexed_interactive/pilot_samples.jsonl")
DEFAULT_SELECTION_MANIFEST = Path("data/eval/indexed_interactive/pilot_samples_manifest.json")

MINIMAL_TOOLS = ("list_dir", "grep", "read_file", "submit")
CLOSED_TOOL_MODES = ("closed_tool", "indexed_interactive")
# Sampling and output limits sent with every OpenAI policy turn.
POLICY_TEMPERATURE = 0
POLICY_MAX_OUTPUT_TOKENS = 700

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
STOP_WORDS = {
    "about",
    "actual",
    "after",
    "also",
    "before",
    "being",
    "build",
    "case",
    "change",
    "changed",
    "class",
    "code",
    "could",
    "error",
    "expected",
    "failed",
    "file",
    "files",
    "from",
    "function",
    "have",
    "here",
    "into",
    "make",
    "method",
    "module",
    "only",
    "path",
    "should",
    "source",
    "test",
    "tests",
    "that",
    "this",
    "using",
    "value",
    "when",
    "with",
}

ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["list_dir", "grep", "read_file", "submit"]},
        "path": {"type": ["string", "null"]},
        "pattern": {"type": ["string", "null"]},
        "query": {"type": ["string", "null"]},
        "top_k": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
        "files": {"type": ["array", "null"], "items": {"type": "string"}},
        "context_files": {"type": ["array", "null"], "items": {"type": "string"}},
    },
    "required": ["action", "path", "pattern", "query", "top_k", "files", "context_files"],
}


def validate_sample_ids(samples: Iterable[dict[str, Any]]) -> None:
    """Nonempty IDs address resume entries and must identify one selected row."""
    seen: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        if sample_id in seen:
            raise ValueError(f"duplicate sample id in selected inputs: {sample_id!r}")
        if sample_id:
            seen.add(sample_id)


def evaluate_closed_tool_grep(
    *,
    sample_paths: Iterable[Path] = DEFAULT_SAMPLES,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    out_path: Path = DEFAULT_OUT,
    details_path: Path = DEFAULT_DETAILS,
    markdown_out_path: Path = DEFAULT_REPORT,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    max_tool_calls: int = 16,
    max_read_tokens: int = 8_000,
    max_read_tokens_per_file: int = 1_200,
    max_grep_calls: int = 8,
    final_k: int = 3,
    grep_top_k: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if max_tool_calls < 2:
        raise ValueError("max_tool_calls must be at least 2")
    if max_read_tokens <= 0:
        raise ValueError("max_read_tokens must be positive")
    if final_k <= 0:
        raise ValueError("final_k must be positive")

    manifest = load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        samples.append(sample)
        if limit_samples and len(samples) >= limit_samples:
            break

    validate_sample_ids(samples)
    existing_details, details_handle = prepare_details_resume(details_path, force=force)
    details_by_id = dict(existing_details)
    ordered_ids: list[str] = []
    skipped = Counter()
    file_cache: dict[Path, dict[str, str]] = {}
    with details_handle:
        for sample in samples:
            sample_id = str(sample.get("id") or "")
            if sample_id:
                ordered_ids.append(sample_id)
            if sample_id in details_by_id:
                skipped["existing"] += 1
                continue
            gold_files = target_gold_files(sample)
            if not gold_files:
                skipped["no_gold"] += 1
                continue
            query_text = query_text_for_eval(sample)
            if query_has_leakage(sample, query_text):
                skipped["query_leakage"] += 1
                continue
            corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
            chunks_path = manifest.get(corpus_key)
            if not chunks_path:
                skipped["missing_corpus"] += 1
                continue
            if chunks_path not in file_cache:
                file_cache[chunks_path] = load_file_texts(chunks_path)
            files = file_cache[chunks_path]
            if not files:
                skipped["empty_corpus"] += 1
                continue
            detail = run_closed_tool_sample(
                sample=sample,
                gold_files=gold_files,
                query_text=query_text,
                files=files,
                max_tool_calls=max_tool_calls,
                max_read_tokens=max_read_tokens,
                max_read_tokens_per_file=max_read_tokens_per_file,
                max_grep_calls=max_grep_calls,
                final_k=final_k,
                grep_top_k=grep_top_k,
            )
            if sample_id:
                details_by_id[sample_id] = detail
            append_detail_jsonl(details_handle, detail)
    details = [details_by_id[sample_id] for sample_id in ordered_ids if sample_id in details_by_id]

    summary = {
        "mode": "closed_tool_context_acquisition",
        "policy": "scripted_grep_read_submit",
        "setting": "closed_tool",
        "evaluated": len(details),
        "skipped": dict(skipped),
        "budget": {
            "max_tool_calls": max_tool_calls,
            "max_read_tokens": max_read_tokens,
            "max_read_tokens_per_file": max_read_tokens_per_file,
            "max_grep_calls": max_grep_calls,
            "final_k": final_k,
            "grep_top_k": grep_top_k,
            "allowed_tools": ["list_dir", "grep", "read_file", "submit"],
            "forbidden": ["embedding_index", "repo_map_index", "edit", "test_execution", "network", "fix_commit"],
        },
        "metrics": summarize_trajectory_details(details),
        "closed_tool_metrics": summarize_closed_tool(details),
        "paths": {
            "summary": str(out_path),
            "details": str(details_path),
            "markdown": str(markdown_out_path),
        },
        "generated_at": utc_now(),
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(out_path, summary)
    write_jsonl(details_path, details)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def evaluate_closed_tool_openai(
    *,
    sample_paths: Iterable[Path] = DEFAULT_SAMPLES,
    corpus_dir: Path | list[Path] = DEFAULT_CORPUS_DIR,
    out_path: Path = DEFAULT_LLM_OUT,
    details_path: Path = DEFAULT_LLM_DETAILS,
    markdown_out_path: Path = DEFAULT_LLM_REPORT,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    model: str = "gpt-5.4-mini",
    max_tool_calls: int = 16,
    max_read_tokens: int = 8_000,
    max_read_tokens_per_file: int = 1_200,
    final_k: int = 3,
    grep_top_k: int = 12,
    max_model_turns: int | None = None,
    force: bool = False,
    mode: str = "closed_tool",
    indexed_tools: IndexedTools | None = None,
    reasoning_effort: str | None = None,
    retain_failures: bool = False,
    require_read_submit: bool = False,
) -> dict[str, Any]:
    if mode not in CLOSED_TOOL_MODES:
        raise ValueError(f"mode must be one of {CLOSED_TOOL_MODES}")
    if (mode == "indexed_interactive") != (indexed_tools is not None):
        raise ValueError("indexed tools are required in indexed_interactive mode and only there")
    started = time.monotonic()
    sample_paths = list(sample_paths)
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample, manifest in filter_sample_manifests(sample_manifests(sample_paths, corpus_dir), keep_ids):
        samples.append((sample, manifest))
        if limit_samples and len(samples) >= limit_samples:
            break

    validate_sample_ids(sample for sample, _manifest in samples)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set on this host.")
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        client = OpenAIResponsesHTTPClient(
            api_key=str(os.environ["OPENAI_API_KEY"]),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    existing_details, details_handle = prepare_details_resume(details_path, force=force)
    details_by_id = dict(existing_details)
    ordered_ids: list[str] = []
    skipped = Counter()
    failures: list[dict[str, Any]] = []
    file_cache: dict[Path, dict[str, str]] = {}
    with details_handle:
        for sample, manifest in samples:
            sample_id = str(sample.get("id") or "")
            if sample_id:
                ordered_ids.append(sample_id)
            previous = details_by_id.get(sample_id)
            # A retained failure is retried; a completed trajectory is kept.
            if previous is not None and "failure" not in previous.get("closed_tool", {}):
                skipped["existing"] += 1
                continue
            gold_files = target_gold_files(sample)
            if not gold_files:
                skipped["no_gold"] += 1
                continue
            query_text = query_text_for_eval(sample)
            if query_has_leakage(sample, query_text):
                skipped["query_leakage"] += 1
                continue
            corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
            chunks_path = manifest.get(corpus_key)
            if not chunks_path:
                skipped["missing_corpus"] += 1
                continue
            if chunks_path not in file_cache:
                file_cache[chunks_path] = load_file_texts(chunks_path)
            files = file_cache[chunks_path]
            index_status = None
            if indexed_tools is not None:
                index_status = indexed_tools.prepare(repo=corpus_key[0], base_commit=corpus_key[1], files=files)
            try:
                detail = run_closed_tool_llm_sample(
                    sample=sample,
                    gold_files=gold_files,
                    query_text=query_text,
                    files=files,
                    model=model,
                    client=client,
                    max_tool_calls=max_tool_calls,
                    max_read_tokens=max_read_tokens,
                    max_read_tokens_per_file=max_read_tokens_per_file,
                    final_k=final_k,
                    grep_top_k=grep_top_k,
                    max_model_turns=max_model_turns or max_tool_calls + 4,
                    policy=f"openai_{mode}",
                    indexed_tools=indexed_tools,
                    index_status=index_status,
                    reasoning_effort=reasoning_effort,
                    require_read_submit=require_read_submit,
                )
            except Exception as exc:
                failure = {"sample_id": sample_id, "error": type(exc).__name__, "message": str(exc)}
                failures.append(failure)
                if not retain_failures:
                    continue
                # The failed attempt scores as empty context but keeps its costs.
                detail = getattr(exc, "partial_detail", None) or failed_closed_tool_detail(
                    sample, gold_files, failure, policy=f"openai_{mode}", index_status=index_status
                )
                detail["closed_tool"]["failure"] = {**detail["closed_tool"].get("failure", {}), **failure}
            if previous is not None:
                detail["closed_tool"]["prior_attempts"] = prior_attempts(previous)
            if sample_id:
                details_by_id[sample_id] = detail
            append_detail_jsonl(details_handle, detail)
    details = [details_by_id[sample_id] for sample_id in ordered_ids if sample_id in details_by_id]
    completed = [detail for detail in details if "failure" not in detail.get("closed_tool", {})]
    # Retained failures score zero in the summary and stay in the details
    # file, so later analyses keep the denominator; resuming retries them.
    scored = details if retain_failures else completed

    allowed_tools = list(MINIMAL_TOOLS) + (list(indexed_tools.tools) if indexed_tools else [])
    summary = {
        "mode": "closed_tool_context_acquisition",
        "policy": f"openai_{mode}",
        "model": model,
        "setting": mode,
        "evaluated": len(completed),
        "scored": len(scored),
        "retain_failures": retain_failures,
        "skipped": dict(skipped),
        "failures": failures,
        "policy_settings": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "temperature": POLICY_TEMPERATURE,
            "max_output_tokens": POLICY_MAX_OUTPUT_TOKENS,
        },
        "budget": {
            "max_tool_calls": max_tool_calls,
            "max_model_turns": max_model_turns or max_tool_calls + 4,
            "max_read_tokens": max_read_tokens,
            "max_read_tokens_per_file": max_read_tokens_per_file,
            "final_k": final_k,
            "grep_top_k": grep_top_k,
            "require_read_submit": require_read_submit,
            "allowed_tools": allowed_tools,
            "forbidden": ["embedding_index", "repo_map_index", "edit", "test_execution", "network", "fix_commit"],
        },
        "metrics": summarize_trajectory_details(scored),
        "closed_tool_metrics": summarize_closed_tool(scored),
        "harness": {"git_commit": harness_commit()},
        "paths": {
            "summary": str(out_path),
            "details": str(details_path),
            "markdown": str(markdown_out_path),
        },
        "generated_at": utc_now(),
        "runtime_seconds": time.monotonic() - started,
    }
    if indexed_tools is not None:
        summary["indexed_tools"] = indexed_tools.describe()
        # Index builds are a per-snapshot cost, reported apart from the policy run.
        summary["index_builds"] = index_builds(details, indexed_tools.snapshots)
    write_json(out_path, summary)
    write_jsonl(details_path, scored)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def sample_manifests(
    sample_paths: list[Path], corpus_dir: Path | list[Path]
) -> Iterable[tuple[dict[str, Any], dict[tuple[str, str], Path]]]:
    """Pair samples with a corpus: one corpus for every file, or one per file in order."""
    corpus_dirs = [corpus_dir] if isinstance(corpus_dir, Path) else list(corpus_dir)
    if len(corpus_dirs) not in {1, len(sample_paths)}:
        raise ValueError("give one corpus for all sample files or one corpus per sample file")
    manifests = [load_corpus_manifest(path) for path in corpus_dirs]
    for index, path in enumerate(sample_paths):
        manifest = manifests[index if len(manifests) > 1 else 0]
        for sample in iter_samples([path]):
            yield sample, manifest


def filter_sample_manifests(
    pairs: Iterable[tuple[dict[str, Any], dict[tuple[str, str], Path]]], keep_ids: set[str] | None
) -> Iterable[tuple[dict[str, Any], dict[tuple[str, str], Path]]]:
    for sample, manifest in pairs:
        if keep_ids is None or sample.get("id") in keep_ids:
            yield sample, manifest


def failed_closed_tool_detail(
    sample: dict[str, Any],
    gold_files: list[str],
    failure: dict[str, Any],
    *,
    policy: str,
    index_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A zero-context trajectory for a sample that failed before any policy turn."""
    detail = trajectory_detail(sample, gold_files, [], supporting_files=supporting_context_files(sample))
    detail["closed_tool"] = {
        "policy": policy,
        "failure": failure,
        "tool_calls": 0,
        "grep_calls": 0,
        "read_calls": 0,
        "read_tokens": 0,
        "final_files": [],
        "first_gold_step": None,
        "trace": [],
    }
    if index_status is not None:
        detail["closed_tool"].update(
            index_status={key: value for key, value in index_status.items() if key not in {"repo", "base_commit"}},
            index_calls=0,
            index_calls_by_tool={},
            index_observation_tokens=0,
            index_provider_errors=0,
        )
    return detail


ATTEMPT_FIELDS = (
    "failure",
    "tool_calls",
    "grep_calls",
    "read_calls",
    "read_tokens",
    "observation_tokens",
    "model_usage",
    "rejected_submits",
    "rejected_submit_model_usage",
    "index_calls",
    "index_calls_by_tool",
    "index_observation_tokens",
    "index_provider_errors",
    "index_status",
    "latency_seconds",
    "trace",
)


def prior_attempts(previous: dict[str, Any]) -> list[dict[str, Any]]:
    """Earlier failed attempts of a retried sample, with their costs and traces."""
    policy = previous.get("closed_tool", {})
    attempt = {key: policy[key] for key in ATTEMPT_FIELDS if key in policy}
    return [*policy.get("prior_attempts", []), attempt]


def index_builds(details: list[dict[str, Any]], snapshots: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per snapshot: its construction, else its first preparation.

    Records come from each stored trajectory and its earlier attempts, then
    from this run's preparations. The construction is the first record not
    served from a cache, so a resume or a cached retry keeps the original
    build cost, once.
    """
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for detail in details:
        policy = detail.get("closed_tool", {})
        key = (str(detail.get("repo") or ""), str(detail.get("base_commit") or ""))
        attempts = [attempt.get("index_status") for attempt in policy.get("prior_attempts", [])]
        for status in [*attempts, policy.get("index_status")]:
            if isinstance(status, dict):
                records.setdefault(key, []).append({"repo": key[0], "base_commit": key[1], **status})
    for key, status in snapshots.items():
        records.setdefault(key, []).append(dict(status))
    return [next((row for row in rows if row.get("cached") is not True), rows[0]) for rows in records.values()]


def select_closed_tool_samples(
    *,
    sample_paths: Iterable[Path],
    corpus_dir: Path | list[Path],
    per_task: int,
    seed: str,
    out_path: Path = DEFAULT_SELECTION_OUT,
    manifest_out_path: Path = DEFAULT_SELECTION_MANIFEST,
) -> dict[str, Any]:
    """Freeze a per-task sample set before any policy runs.

    Eligible samples are those the closed-tool runners would evaluate: they
    have gold files, no query leakage, and a released corpus snapshot. Within
    each task, repositories take turns in ``sha256(seed, task, repo)`` order,
    and each turn takes the repository's next sample in ``sha256(seed, id)``
    order, so every repository gets one sample before any gets two. The
    selection reads no method scores and builds no index.
    """
    if per_task <= 0:
        raise ValueError("per_task must be positive")
    if not seed:
        raise ValueError("a non-empty seed is required")
    sample_paths = list(sample_paths)
    eligible: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    excluded: Counter = Counter()
    for sample, manifest in sample_manifests(sample_paths, corpus_dir):
        task = str(sample.get("task_type") or "")
        reason = closed_tool_exclusion(sample, manifest)
        if reason:
            excluded[f"{task}:{reason}"] += 1
            continue
        eligible[task][str(sample.get("repo") or "")].append(sample)

    def order(*parts: str) -> str:
        return hashlib.sha256("\n".join((seed, *parts)).encode("utf-8")).hexdigest()

    selected: list[dict[str, Any]] = []
    tasks = {}
    for task in sorted(eligible):
        queues = {
            repo: sorted(rows, key=lambda row: order(str(row.get("id"))))
            for repo, rows in eligible[task].items()
        }
        repos = sorted(queues, key=lambda repo: order(task, repo))
        available = sum(len(rows) for rows in queues.values())
        if available < per_task:
            raise ValueError(f"{task} has {available} eligible samples, fewer than {per_task}")
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per_task:
            for repo in repos:
                if queues[repo] and len(chosen) < per_task:
                    chosen.append(queues[repo].pop(0))
        selected.extend(chosen)
        tasks[task] = {
            "eligible": available,
            "eligible_repos": len(repos),
            "selected": len(chosen),
            "selected_repos": dict(sorted(Counter(str(row.get("repo")) for row in chosen).items())),
        }

    rows = [
        {"sample_id": row.get("id"), "task_type": row.get("task_type"), "repo": row.get("repo"), "base_commit": row.get("base_commit")}
        for row in selected
    ]
    ids = sorted(str(row["sample_id"]) for row in rows)
    manifest = {
        "mode": "closed_tool_sample_selection",
        "seed": seed,
        "per_task": per_task,
        "sample_count": len(rows),
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "definition": (
            "Eligible: gold files, no query leakage, released corpus snapshot. Per task, repositories take "
            "turns in sha256(seed, task, repo) order; each turn takes that repository's next sample in "
            "sha256(seed, id) order. No method scores or index builds are read."
        ),
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sample_paths
        ],
        "tasks": tasks,
        "excluded": dict(sorted(excluded.items())),
        "paths": {"samples": str(out_path), "manifest": str(manifest_out_path)},
        "generated_at": utc_now(),
    }
    write_jsonl(out_path, rows)
    write_json(manifest_out_path, manifest)
    return manifest


def closed_tool_exclusion(sample: dict[str, Any], manifest: dict[tuple[str, str], Path]) -> str:
    """Why the closed-tool runners would skip ``sample``, or an empty string."""
    if not target_gold_files(sample):
        return "no_gold"
    if query_has_leakage(sample, query_text_for_eval(sample)):
        return "query_leakage"
    if (str(sample.get("repo") or ""), str(sample.get("base_commit") or "")) not in manifest:
        return "missing_corpus"
    return ""


def harness_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def evaluate_closed_tool_codex(
    *,
    sample_paths: Iterable[Path] = DEFAULT_SAMPLES,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    out_path: Path = DEFAULT_CODEX_OUT,
    details_path: Path = DEFAULT_CODEX_DETAILS,
    markdown_out_path: Path = DEFAULT_CODEX_REPORT,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    model: str = "gpt-5.5",
    codex_bin: Path = DEFAULT_CODEX_BIN,
    work_root: Path = DEFAULT_CODEX_WORK_ROOT,
    timeout_seconds: int = 180,
    max_tool_calls: int = 16,
    max_read_tokens: int = 8_000,
    max_read_tokens_per_file: int = 1_200,
    final_k: int = 3,
    grep_top_k: int = 12,
    max_model_turns: int | None = None,
    force: bool = False,
    seed_details: Path | None = None,
    seed_label: str = "",
    seed_top_k: int = 3,
    max_seed_tokens: int = 0,
    max_seed_tokens_per_file: int = 1_200,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        samples.append(sample)
        if limit_samples and len(samples) >= limit_samples:
            break

    validate_sample_ids(samples)
    codex_bin = resolve_codex_bin(codex_bin)
    seed_files_by_sample = load_seed_files(seed_details, top_k=seed_top_k)
    existing_details, details_handle = prepare_details_resume(details_path, force=force)
    details_by_id = dict(existing_details)
    ordered_ids: list[str] = []
    skipped = Counter()
    failures: list[dict[str, Any]] = []
    file_cache: dict[Path, dict[str, str]] = {}
    with details_handle:
        for sample in samples:
            sample_id = str(sample.get("id") or "")
            if sample_id:
                ordered_ids.append(sample_id)
            if sample_id in details_by_id:
                skipped["existing"] += 1
                continue
            gold_files = target_gold_files(sample)
            if not gold_files:
                skipped["no_gold"] += 1
                continue
            query_text = query_text_for_eval(sample)
            if query_has_leakage(sample, query_text):
                skipped["query_leakage"] += 1
                continue
            corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
            chunks_path = manifest.get(corpus_key)
            if not chunks_path:
                skipped["missing_corpus"] += 1
                continue
            if chunks_path not in file_cache:
                file_cache[chunks_path] = load_file_texts(chunks_path)
            files = file_cache[chunks_path]
            try:
                detail = run_closed_tool_codex_sample(
                    sample=sample,
                    gold_files=gold_files,
                    query_text=query_text,
                    files=files,
                    model=model,
                    codex_bin=codex_bin,
                    work_root=work_root,
                    timeout_seconds=timeout_seconds,
                    max_tool_calls=max_tool_calls,
                    max_read_tokens=max_read_tokens,
                        max_read_tokens_per_file=max_read_tokens_per_file,
                        final_k=final_k,
                        grep_top_k=grep_top_k,
                        max_model_turns=max_model_turns or max_tool_calls + 4,
                        seed_files=seed_files_by_sample.get(sample_id, []),
                        seed_label=seed_label or (seed_details.stem if seed_details else ""),
                        max_seed_tokens=max_seed_tokens,
                        max_seed_tokens_per_file=max_seed_tokens_per_file,
                )
                if sample_id:
                    details_by_id[sample_id] = detail
                append_detail_jsonl(details_handle, detail)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": type(exc).__name__, "message": str(exc)})
    details = [details_by_id[sample_id] for sample_id in ordered_ids if sample_id in details_by_id]

    protocol_violations = [
        {
            "sample_id": detail.get("sample_id"),
            "violations": detail.get("closed_tool", {}).get("protocol_violations", []),
        }
        for detail in details
        if detail.get("closed_tool", {}).get("protocol_violations")
    ]
    summary = {
        "mode": "closed_tool_context_acquisition",
        "policy": "codex_closed_tool",
        "model": model,
        "codex_bin": str(codex_bin),
        "setting": "closed_tool",
        "evaluated": len(details),
        "skipped": dict(skipped),
        "failures": failures,
        "protocol_violation_samples": len(protocol_violations),
        "protocol_violations": protocol_violations[:25],
        "budget": {
            "max_tool_calls": max_tool_calls,
            "max_model_turns": max_model_turns or max_tool_calls + 4,
            "max_read_tokens": max_read_tokens,
            "max_read_tokens_per_file": max_read_tokens_per_file,
            "final_k": final_k,
            "grep_top_k": grep_top_k,
            "codex_timeout_seconds": timeout_seconds,
            "codex_work_root": str(work_root),
            "seed_details": str(seed_details) if seed_details else "",
            "seed_label": seed_label or (seed_details.stem if seed_details else ""),
            "seed_top_k": seed_top_k,
            "max_seed_tokens": max_seed_tokens,
            "max_seed_tokens_per_file": max_seed_tokens_per_file,
            "allowed_tools": ["list_dir", "grep", "read_file", "submit"],
            "forbidden": ["embedding_index", "repo_map_index", "edit", "test_execution", "network", "fix_commit", "codex_shell"],
        },
        "metrics": summarize_trajectory_details(details),
        "closed_tool_metrics": summarize_closed_tool(details),
        "paths": {
            "summary": str(out_path),
            "details": str(details_path),
            "markdown": str(markdown_out_path),
        },
        "generated_at": utc_now(),
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(out_path, summary)
    write_jsonl(details_path, details)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def run_closed_tool_sample(
    *,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    files: dict[str, str],
    max_tool_calls: int,
    max_read_tokens: int,
    max_read_tokens_per_file: int,
    max_grep_calls: int,
    final_k: int,
    grep_top_k: int,
) -> dict[str, Any]:
    start = time.monotonic()
    query_payload = parse_query_payload(query_text)
    reserved_read_calls = max(1, min(max_tool_calls - 2, final_k + 3))
    pattern_budget = max(1, min(max_grep_calls, max_tool_calls - reserved_read_calls - 2))
    patterns = choose_grep_patterns(query_payload, max_patterns=pattern_budget)
    trace: list[dict[str, Any]] = []
    tool_calls = 0
    candidate_scores: dict[str, float] = defaultdict(float)
    candidate_hits: dict[str, list[str]] = defaultdict(list)

    roots = list_dir(files, "")
    tool_calls += 1
    trace.append({"tool": "list_dir", "path": "", "entries": roots[:20]})

    for pattern in patterns:
        if tool_calls >= max_tool_calls - 1:
            break
        hits = grep_files(files, pattern, top_k=grep_top_k)
        tool_calls += 1
        trace.append(
            {
                "tool": "grep",
                "pattern": pattern,
                "hits": [{"path": hit["path"], "matches": hit["matches"], "snippet": hit["snippet"][:240]} for hit in hits[:5]],
            }
        )
        pattern_weight = pattern_score(pattern)
        for rank, hit in enumerate(hits, start=1):
            path = str(hit["path"])
            candidate_scores[path] += pattern_weight * float(hit["matches"]) / math.sqrt(rank)
            candidate_hits[path].append(pattern)

    anchor_paths = query_anchor_paths(query_payload)
    for path in anchor_paths:
        if path in files:
            candidate_scores[path] += 2.0
            candidate_hits[path].append("query_path")
    for path in files:
        path_score = path_token_score(path, patterns)
        if path_score > 0:
            candidate_scores[path] += path_score

    read_steps: list[dict[str, Any]] = []
    read_tokens = 0
    read_paths: list[str] = []
    query_tokens = query_token_set(query_payload)
    for path, _score in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0])):
        if tool_calls >= max_tool_calls - 1:
            break
        if read_tokens >= max_read_tokens:
            break
        text = files.get(path)
        if text is None:
            continue
        token_count = count_tokens(text)
        remaining = max_read_tokens - read_tokens
        included_tokens = min(token_count, remaining, max_read_tokens_per_file)
        if included_tokens <= 0:
            break
        read_tokens += included_tokens
        tool_calls += 1
        read_paths.append(path)
        read_steps.append(
            {
                "step": tool_calls,
                "tool": "read_file",
                "path": path,
                "start_line": None,
                "end_line": None,
                "kind": "file",
                "symbol": "",
                "content_hash": "",
                "is_final_context": False,
                "is_utilized_context": False,
            }
        )
        visible_text = token_prefix(text, included_tokens)
        content_bonus = len(query_tokens & tokenize(visible_text[:20_000])) / max(1, len(query_tokens))
        candidate_scores[path] += content_bonus
        trace.append(
            {
                "tool": "read_file",
                "path": path,
                "tokens": included_tokens,
                "truncated": token_count > included_tokens,
                "matched_patterns": candidate_hits.get(path, [])[:8],
            }
        )

    if read_paths:
        final_files = [path for path, _ in sorted(((path, candidate_scores[path]) for path in read_paths), key=lambda item: (-item[1], item[0]))[:final_k]]
    else:
        final_files = [path for path, _ in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[:final_k]]
    tool_calls += 1
    trace.append({"tool": "submit", "files": final_files})

    for step in read_steps:
        if step["path"] in set(final_files):
            step["is_final_context"] = True
            step["is_utilized_context"] = True
    if not read_steps:
        for index, path in enumerate(final_files, start=1):
            read_steps.append(
                {
                    "step": index,
                    "tool": "submit",
                    "path": path,
                    "start_line": None,
                    "end_line": None,
                    "kind": "file",
                    "symbol": "",
                    "content_hash": "",
                    "is_final_context": True,
                    "is_utilized_context": True,
                }
            )

    detail = trajectory_detail(
        sample,
        gold_files,
        read_steps,
        supporting_files=supporting_context_files(sample),
    )
    detail["closed_tool"] = {
        "policy": "scripted_grep_read_submit",
        "allowed_tools": ["list_dir", "grep", "read_file", "submit"],
        "max_tool_calls": max_tool_calls,
        "max_read_tokens": max_read_tokens,
        "max_read_tokens_per_file": max_read_tokens_per_file,
        "max_grep_calls": max_grep_calls,
        "final_k": final_k,
        "grep_patterns": patterns,
        "tool_calls": tool_calls,
        "grep_calls": sum(1 for item in trace if item["tool"] == "grep"),
        "read_calls": sum(1 for item in trace if item["tool"] == "read_file"),
        "read_tokens": read_tokens,
        "final_files": final_files,
        "first_gold_step": first_gold_step(read_steps, gold_files),
        "latency_seconds": time.monotonic() - start,
        "trace": trace,
    }
    return detail


def run_closed_tool_codex_sample(
    *,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    files: dict[str, str],
    model: str,
    codex_bin: Path,
    work_root: Path,
    timeout_seconds: int,
    max_tool_calls: int,
    max_read_tokens: int,
    max_read_tokens_per_file: int,
    final_k: int,
    grep_top_k: int,
    max_model_turns: int,
    seed_files: list[str] | None = None,
    seed_label: str = "",
    max_seed_tokens: int = 0,
    max_seed_tokens_per_file: int = 1_200,
) -> dict[str, Any]:
    client = CodexActionClient(
        codex_bin=codex_bin,
        work_root=work_root,
        timeout_seconds=timeout_seconds,
        sample_id=str(sample.get("id") or "sample"),
    )
    detail = run_closed_tool_llm_sample(
        sample=sample,
        gold_files=gold_files,
        query_text=query_text,
        files=files,
        model=model,
        client=client,
        max_tool_calls=max_tool_calls,
        max_read_tokens=max_read_tokens,
        max_read_tokens_per_file=max_read_tokens_per_file,
        final_k=final_k,
        grep_top_k=grep_top_k,
        max_model_turns=max_model_turns,
        policy="codex_closed_tool",
        seed_files=seed_files,
        seed_label=seed_label,
        max_seed_tokens=max_seed_tokens,
        max_seed_tokens_per_file=max_seed_tokens_per_file,
    )
    detail["closed_tool"]["codex_bin"] = str(codex_bin)
    detail["closed_tool"]["codex_exec_calls"] = len(client.call_records)
    detail["closed_tool"]["codex_exec_returncodes"] = [record["returncode"] for record in client.call_records]
    detail["closed_tool"]["protocol_violations"] = [
        violation
        for record in client.call_records
        for violation in record.get("protocol_violations", [])
    ]
    detail["closed_tool"]["codex_exec_events"] = [
        {
            "turn": record["turn"],
            "returncode": record["returncode"],
            "json_events": record["json_events"],
            "stderr_excerpt": record.get("stderr_excerpt", ""),
        }
        for record in client.call_records
    ]
    return detail


def run_closed_tool_llm_sample(
    *,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    files: dict[str, str],
    model: str,
    client: Any,
    max_tool_calls: int,
    max_read_tokens: int,
    max_read_tokens_per_file: int,
    final_k: int,
    grep_top_k: int,
    max_model_turns: int,
    policy: str = "openai_closed_tool",
    seed_files: list[str] | None = None,
    seed_label: str = "",
    max_seed_tokens: int = 0,
    max_seed_tokens_per_file: int = 1_200,
    indexed_tools: IndexedTools | None = None,
    index_status: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    require_read_submit: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    query_payload = parse_query_payload(query_text)
    observations: list[str] = []
    trace: list[dict[str, Any]] = []
    read_steps: list[dict[str, Any]] = []
    read_paths: list[str] = []
    read_set: set[str] = set()
    read_tokens = 0
    tool_calls = 0
    final_files: list[str] = []
    last_error = ""
    usage_totals: Counter = Counter()
    rejected_usage: Counter = Counter()
    error: Exception | None = None
    failure: dict[str, Any] | None = None
    turn = 0
    # Indexed tools are offered only when the provider built this snapshot.
    offered_index = tuple(indexed_tools.tools) if indexed_tools and (index_status or {}).get("available") else ()
    tools = MINIMAL_TOOLS + offered_index
    seed_context = build_seed_context(
        files=files,
        seed_files=seed_files or [],
        seed_label=seed_label,
        max_seed_tokens=max_seed_tokens,
        max_seed_tokens_per_file=max_seed_tokens_per_file,
        max_read_tokens=max_read_tokens,
    )
    if seed_context["observation"]:
        observations.append(seed_context["observation"])
    trace.extend(seed_context["trace"])
    read_steps.extend(seed_context["steps"])
    read_paths.extend(seed_context["paths"])
    read_set.update(seed_context["paths"])
    read_tokens += int(seed_context["tokens"])

    try:
        for turn in range(1, max_model_turns + 1):
            prompt = build_closed_tool_prompt(
                sample=sample,
                query_text=query_text,
                observations=observations,
                read_paths=read_paths,
                tool_calls=tool_calls,
                max_tool_calls=max_tool_calls,
                read_tokens=read_tokens,
                max_read_tokens=max_read_tokens,
                final_k=final_k,
                last_error=last_error,
                tools=tools,
                index_unavailable=indexed_tools is not None and not offered_index,
            )
            response = create_policy_response(
                client=client, model=model, prompt=prompt, reasoning_effort=reasoning_effort
            )
            response_text = response_output_text(response)
            usage = response_usage(response)
            usage_totals.update(usage)
            action = parse_tool_action(response_text)
            event = {"model_turn": turn, "tool": "model_action", "raw": response_text, "parsed": action}
            if usage:
                event["usage"] = usage
            trace.append(event)
            last_error = ""
            kind = str(action.get("action") or "").lower()
            if kind not in tools:
                last_error = f"Action must be one of {', '.join(tools)}."
                observations.append(f"ERROR: {last_error}")
                continue
            if tool_calls >= max_tool_calls:
                last_error = "Tool-call budget exhausted; submit from already read files."
                observations.append(f"ERROR: {last_error}")
                continue

            if kind == "list_dir":
                path = str(action.get("path") or "")
                entries = list_dir(files, path)
                tool_calls += 1
                observation = format_list_dir_observation(path, entries)
                observations.append(observation)
                trace.append({"tool": "list_dir", "path": path, "entries": entries[:40]})
                continue

            if kind == "grep":
                pattern = str(action.get("pattern") or action.get("query") or "").strip()
                if not pattern:
                    last_error = "grep requires a non-empty pattern."
                    observations.append(f"ERROR: {last_error}")
                    continue
                top_k = clamp_int(action.get("top_k"), default=grep_top_k, minimum=1, maximum=grep_top_k)
                hits = grep_files(files, pattern, top_k=top_k)
                tool_calls += 1
                observation = format_grep_observation(pattern, hits)
                observations.append(observation)
                trace.append({"tool": "grep", "pattern": pattern, "hits": compact_hits(hits)})
                continue

            if kind in INDEXED_TOOLS:
                outcome = indexed_tools.lookup(
                    kind,
                    action,
                    repo=str(sample.get("repo") or ""),
                    base_commit=str(sample.get("base_commit") or ""),
                    files=files,
                )
                if outcome["error"]:
                    last_error = outcome["error"]
                    observations.append(f"ERROR: {last_error}")
                    continue
                tool_calls += 1
                observations.append(outcome["observation"])
                trace.append(outcome["trace"])
                continue

            if kind == "read_file":
                path = str(action.get("path") or "")
                if path not in files:
                    last_error = f"read_file path not found: {path}"
                    observations.append(f"ERROR: {last_error}")
                    continue
                if path in read_set:
                    last_error = f"file already read: {path}"
                    observations.append(f"ERROR: {last_error}")
                    continue
                if read_tokens >= max_read_tokens:
                    last_error = "read token budget exhausted; submit from already read files."
                    observations.append(f"ERROR: {last_error}")
                    continue
                text = files[path]
                token_count = count_tokens(text)
                included_tokens = min(token_count, max_read_tokens - read_tokens, max_read_tokens_per_file)
                if included_tokens <= 0:
                    last_error = "read token budget exhausted; submit from already read files."
                    observations.append(f"ERROR: {last_error}")
                    continue
                read_tokens += included_tokens
                tool_calls += 1
                read_set.add(path)
                read_paths.append(path)
                read_steps.append(
                    {
                        "step": tool_calls,
                        "tool": "read_file",
                        "path": path,
                        "start_line": None,
                        "end_line": None,
                        "kind": "file",
                        "symbol": "",
                        "content_hash": "",
                        "is_final_context": False,
                        "is_utilized_context": False,
                    }
                )
                observation = format_read_file_observation(path, text, included_tokens)
                observations.append(observation)
                trace.append({"tool": "read_file", "path": path, "tokens": included_tokens, "truncated": token_count > included_tokens})
                continue

            requested = dedupe(str(path) for path in (action.get("files") or action.get("context_files") or []) if path)
            final_files = [path for path in requested if path in read_set][:final_k]
            if require_read_submit and not final_files:
                last_error = "submit files must already be read with read_file; read the files you want to submit first."
                observations.append(f"ERROR: {last_error}")
                rejected_usage.update(usage)
                trace.append({"tool": "rejected_submit", "raw_files": requested, "model_turn": turn, "usage": usage})
                continue
            if not final_files and read_paths:
                final_files = read_paths[:final_k]
            tool_calls += 1
            trace.append({"tool": "submit", "files": final_files, "raw_files": requested})
            break
    except Exception as exc:  # The attempt still consumed what it used so far.
        error = exc
        failure = {"error": type(exc).__name__, "message": str(exc)[:500], "model_turn": turn}

    if failure is not None:
        # A failed attempt scores as empty context; its costs stay recorded.
        final_files, read_steps = [], []
    elif not final_files:
        final_files = read_paths[:final_k]
        trace.append({"tool": "submit", "files": final_files, "fallback": True})
    final_set = set(final_files)
    for step in read_steps:
        if step["path"] in final_set:
            step["is_final_context"] = True
            step["is_utilized_context"] = True
    if not read_steps:
        read_steps = [
            {
                "step": index,
                "tool": "submit",
                "path": path,
                "start_line": None,
                "end_line": None,
                "kind": "file",
                "symbol": "",
                "content_hash": "",
                "is_final_context": True,
                "is_utilized_context": True,
            }
            for index, path in enumerate(final_files, start=1)
        ]
    detail = trajectory_detail(
        sample,
        gold_files,
        read_steps,
        supporting_files=supporting_context_files(sample),
    )
    detail["closed_tool"] = {
        "policy": policy,
        "model": model,
        "allowed_tools": list(tools),
        "max_tool_calls": max_tool_calls,
        "max_model_turns": max_model_turns,
        "max_read_tokens": max_read_tokens,
        "max_read_tokens_per_file": max_read_tokens_per_file,
        "final_k": final_k,
        "require_read_submit": require_read_submit,
        "rejected_submits": sum(1 for item in trace if item.get("tool") == "rejected_submit"),
        "rejected_submit_model_usage": dict(rejected_usage),
        "tool_calls": tool_calls,
        "grep_calls": sum(1 for item in trace if item.get("tool") == "grep"),
        "read_calls": sum(1 for item in trace if item.get("tool") == "read_file"),
        "read_tokens": read_tokens,
        "seed_label": seed_label,
        "seed_files": seed_context["paths"],
        "seed_tokens": seed_context["tokens"],
        "seed_gold_files": sorted(set(seed_context["paths"]) & set(gold_files)),
        "seed_any_gold": bool(set(seed_context["paths"]) & set(gold_files)),
        "max_seed_tokens": max_seed_tokens,
        "max_seed_tokens_per_file": max_seed_tokens_per_file,
        "final_files": final_files,
        "first_gold_step": first_gold_step(read_steps, gold_files),
        "observation_tokens": sum(count_tokens(observation) for observation in observations),
        "model_usage": dict(usage_totals),
        "latency_seconds": time.monotonic() - start,
        "trace": trace,
    }
    if indexed_tools is not None:
        index_events = [item for item in trace if item.get("tool") in INDEXED_TOOLS]
        detail["closed_tool"].update(
            index_status={key: value for key, value in (index_status or {}).items() if key not in {"repo", "base_commit"}},
            index_calls=len(index_events),
            index_calls_by_tool=dict(Counter(str(item["tool"]) for item in index_events)),
            index_observation_tokens=sum(int(item.get("observation_tokens") or 0) for item in index_events),
            index_provider_errors=sum(1 for item in index_events if item.get("provider_error")),
        )
    if error is not None:
        detail["closed_tool"]["failure"] = failure
        # Callers see the original exception, carrying what the attempt consumed.
        error.partial_detail = detail
        raise error
    return detail


def load_existing_details(path: Path) -> dict[str, dict[str, Any]]:
    details = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if sample_id:
            details[sample_id] = row
    return details


def prepare_details_resume(path: Path, *, force: bool) -> tuple[dict[str, dict[str, Any]], Any]:
    if force and path.exists():
        path.unlink()
    existing = load_existing_details(path) if path.exists() and not force else {}
    ensure_parent(path)
    return existing, path.open("a", encoding="utf-8")


def append_detail_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def load_seed_files(path: Path | None, *, top_k: int) -> dict[str, list[str]]:
    if path is None or top_k <= 0:
        return {}
    seeds: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if not sample_id:
            continue
        seeds[sample_id] = dedupe(str(value) for value in (row.get("top_files") or [])[:top_k] if value)
    return seeds


def build_seed_context(
    *,
    files: dict[str, str],
    seed_files: list[str],
    seed_label: str,
    max_seed_tokens: int,
    max_seed_tokens_per_file: int,
    max_read_tokens: int,
) -> dict[str, Any]:
    if not seed_files or max_seed_tokens <= 0 or max_read_tokens <= 0:
        return {"paths": [], "tokens": 0, "steps": [], "trace": [], "observation": ""}
    paths: list[str] = []
    steps: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    observation_parts = [f"OBS seed_context source={seed_label or 'seed'}"]
    seed_tokens = 0
    for path in dedupe(seed_files):
        text = files.get(path)
        if text is None:
            trace.append({"tool": "seed_context", "path": path, "missing": True, "source": seed_label})
            continue
        remaining = min(max_seed_tokens - seed_tokens, max_read_tokens - seed_tokens)
        if remaining <= 0:
            break
        included_tokens = min(count_tokens(text), remaining, max_seed_tokens_per_file)
        if included_tokens <= 0:
            continue
        paths.append(path)
        seed_tokens += included_tokens
        steps.append(
            {
                "step": 0,
                "tool": "seed_context",
                "path": path,
                "start_line": None,
                "end_line": None,
                "kind": "file",
                "symbol": "",
                "content_hash": "",
                "is_final_context": False,
                "is_utilized_context": False,
            }
        )
        trace.append({"tool": "seed_context", "path": path, "tokens": included_tokens, "source": seed_label})
        observation_parts.append(f"\nSEED_FILE path={path} tokens={included_tokens}\n{token_prefix(text, included_tokens)}")
    if not paths:
        return {"paths": [], "tokens": 0, "steps": [], "trace": trace, "observation": ""}
    observation = "\n".join(observation_parts)
    return {"paths": paths, "tokens": seed_tokens, "steps": steps, "trace": trace, "observation": observation}


def build_closed_tool_prompt(
    *,
    sample: dict[str, Any],
    query_text: str,
    observations: list[str],
    read_paths: list[str],
    tool_calls: int,
    max_tool_calls: int,
    read_tokens: int,
    max_read_tokens: int,
    final_k: int,
    last_error: str,
    tools: tuple[str, ...] = MINIMAL_TOOLS,
    index_unavailable: bool = False,
) -> str:
    indexed = [tool for tool in tools if tool in INDEXED_TOOLS]
    status = {
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "tool_calls_used": tool_calls,
        "tool_calls_remaining": max(0, max_tool_calls - tool_calls),
        "read_tokens_used": read_tokens,
        "read_tokens_remaining": max(0, max_read_tokens - read_tokens),
        "already_read_files": read_paths,
        "final_k": final_k,
    }
    recent_observations = observations[-10:]
    instructions = [
        "You are evaluating a closed-tool context-acquisition policy for an agentic coding benchmark.",
        "Goal: identify the repository files an agent should read next to solve the sample.",
        f"You may only use these tools: {', '.join(tools)}.",
        "Forbidden: embedding index, repository map index, editing files, running tests, network access, fix commits, or using hidden corpus content.",
        "If you are running inside Codex CLI, do not call shell commands or inspect local files; the working directory is intentionally empty.",
        "The only repository information available to you is the QUERY and RECENT OBSERVATIONS below.",
        "Return exactly one JSON object and no prose.",
        "All fields must be present; use null for unused scalar fields and [] for unused file lists.",
        "Schemas:",
        '{"action":"list_dir","path":"","pattern":null,"query":null,"top_k":null,"files":[],"context_files":[]}',
        '{"action":"grep","path":null,"pattern":"timeout panic","query":null,"top_k":12,"files":[],"context_files":[]}',
        '{"action":"read_file","path":"src/example.py","pattern":null,"query":null,"top_k":null,"files":[],"context_files":[]}',
        '{"action":"submit","path":null,"pattern":null,"query":null,"top_k":null,"files":["src/example.py","tests/test_example.py"],"context_files":[]}',
        "submit files must be selected from already_read_files. Prefer submit once you have enough evidence or the budget is nearly exhausted.",
    ]
    if indexed:
        instructions.append(
            "Indexed tools return file paths and line ranges only, cost one tool call each, and need a symbol field:"
        )
        instructions.extend(TOOL_DESCRIPTIONS[tool] for tool in indexed)
        instructions.extend(TOOL_EXAMPLES[tool] for tool in indexed)
    elif index_unavailable:
        instructions.append("Indexed tools are unavailable for this repository snapshot.")
    if last_error:
        instructions.append(f"Previous action error: {last_error}")
    return "\n".join(
        [
            "\n".join(instructions),
            "",
            "STATUS:",
            json.dumps(status, ensure_ascii=False, indent=2),
            "",
            "QUERY:",
            query_text,
            "",
            "RECENT OBSERVATIONS:",
            "\n\n".join(recent_observations) if recent_observations else "(none)",
        ]
    )


def call_openai(*, client: Any, model: str, prompt: str, reasoning_effort: str | None = None) -> str:
    response = create_policy_response(client=client, model=model, prompt=prompt, reasoning_effort=reasoning_effort)
    return response_output_text(response)


def create_policy_response(*, client: Any, model: str, prompt: str, reasoning_effort: str | None = None) -> Any:
    create = client.responses.create
    options: dict[str, Any] = {"model": model, "input": prompt, "max_output_tokens": POLICY_MAX_OUTPUT_TOKENS}
    if reasoning_effort:
        options["reasoning"] = {"effort": reasoning_effort}
    try:
        return create(temperature=POLICY_TEMPERATURE, **options)
    except TypeError:
        return create(**options)


def response_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def response_usage(response: Any) -> dict[str, int]:
    """Model token counts reported by the Responses API, if any."""
    payload = getattr(response, "payload", None)
    usage = payload.get("usage") if isinstance(payload, dict) else getattr(response, "usage", None)
    if usage is None:
        return {}
    if not isinstance(usage, dict):
        dump = getattr(usage, "model_dump", None)
        usage = dump() if callable(dump) else {}
    counts = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
    }
    return {key: value for key, value in counts.items() if isinstance(value, int)}


class CodexActionClient:
    def __init__(self, *, codex_bin: Path, work_root: Path, timeout_seconds: int, sample_id: str) -> None:
        self.codex_bin = codex_bin
        self.work_root = work_root
        self.timeout_seconds = timeout_seconds
        self.sample_id = safe_name(sample_id)
        self.responses = self
        self.call_records: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        turn = len(self.call_records) + 1
        record = run_codex_exec_action(
            codex_bin=self.codex_bin,
            model=str(kwargs.get("model") or "gpt-5.5"),
            prompt=str(kwargs.get("input") or ""),
            work_root=self.work_root,
            timeout_seconds=self.timeout_seconds,
            sample_id=self.sample_id,
            turn=turn,
        )
        self.call_records.append(record)
        return SimpleTextResponse(record["output_text"])


class SimpleTextResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


def resolve_codex_bin(path: Path) -> Path:
    if path.exists():
        return path
    resolved = shutil.which(str(path))
    if resolved:
        return Path(resolved)
    raise RuntimeError(f"Codex CLI binary not found: {path}")


def run_codex_exec_action(
    *,
    codex_bin: Path,
    model: str,
    prompt: str,
    work_root: Path,
    timeout_seconds: int,
    sample_id: str,
    turn: int,
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{safe_name(sample_id)}-turn{turn}-", dir=work_root) as temp_dir:
        work_dir = Path(temp_dir)
        schema_path = work_dir / "action_schema.json"
        output_path = work_dir / "action.json"
        schema_path.write_text(json.dumps(ACTION_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [
            str(codex_bin),
            "--ask-for-approval",
            "never",
            "exec",
            "-C",
            str(work_dir),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--json",
            "-m",
            model,
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codex exec timed out after {timeout_seconds}s on turn {turn}") from exc
        output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else result.stdout
        events = parse_codex_json_events(result.stdout)
        record = {
            "turn": turn,
            "returncode": result.returncode,
            "output_text": output_text,
            "json_events": len(events),
            "protocol_violations": detect_codex_protocol_violations(events),
            "stderr_excerpt": result.stderr[-1000:],
            "stdout_excerpt": result.stdout[-1000:],
        }
        if result.returncode != 0:
            raise RuntimeError(
                "codex exec failed on turn {turn}: stderr={stderr!r} stdout={stdout!r} output={output!r}".format(
                    turn=turn,
                    stderr=result.stderr[-500:],
                    stdout=result.stdout[-500:],
                    output=output_text[-500:],
                )
            )
        return record


def parse_codex_json_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def detect_codex_protocol_violations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for index, event in enumerate(events):
        markers = forbidden_tool_markers(event)
        if markers:
            violations.append({"event_index": index, "markers": sorted(markers), "event_type": str(event.get("type") or "")})
    return violations


def forbidden_tool_markers(value: Any) -> set[str]:
    markers: set[str] = set()
    forbidden_names = {
        "apply_patch",
        "exec",
        "exec_command",
        "functions.exec_command",
        "shell",
        "shell_command",
        "write_stdin",
    }

    def walk(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                walk(child_value, str(child_key))
            return
        if isinstance(item, list):
            for child in item:
                walk(child, key)
            return
        if not isinstance(item, str):
            return
        lowered = item.lower()
        key_lower = key.lower()
        if key_lower in {"tool", "tool_name", "name", "recipient", "recipient_name", "function", "type"}:
            for forbidden in forbidden_names:
                if lowered == forbidden or lowered.endswith("." + forbidden) or forbidden in lowered:
                    markers.add(forbidden)

    walk(value)
    return markers


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "sample"


class OpenAIResponsesHTTPClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.responses = self

    def create(self, **kwargs: Any) -> Any:
        import urllib.error
        import urllib.request

        payload = {key: value for key, value in kwargs.items() if value is not None}
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API HTTP {exc.code}: {message[:500]}") from exc
        return SimpleOpenAIResponse(body)


class SimpleOpenAIResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.output_text = extract_openai_output_text(payload)


def extract_openai_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    chunks = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def parse_tool_action(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"action": "", "parse_error": "No JSON object found in model response."}


def clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def format_list_dir_observation(path: str, entries: list[str]) -> str:
    shown = entries[:60]
    suffix = "" if len(entries) <= len(shown) else f"\n... {len(entries) - len(shown)} more entries"
    return "OBS list_dir path={path}\n{entries}{suffix}".format(
        path=path or ".",
        entries="\n".join(f"- {entry}" for entry in shown) if shown else "(empty)",
        suffix=suffix,
    )


def format_grep_observation(pattern: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return f"OBS grep pattern={pattern!r}\n(no hits)"
    lines = [f"OBS grep pattern={pattern!r}"]
    for index, hit in enumerate(hits[:10], start=1):
        snippet = str(hit.get("snippet") or "")[:260]
        lines.append(f"{index}. {hit['path']} matches={hit['matches']} snippet={snippet}")
    if len(hits) > 10:
        lines.append(f"... {len(hits) - 10} more hits omitted")
    return "\n".join(lines)


def compact_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": hit.get("path"),
            "matches": hit.get("matches"),
            "snippet": str(hit.get("snippet") or "")[:240],
        }
        for hit in hits[:10]
    ]


def format_read_file_observation(path: str, text: str, included_tokens: int) -> str:
    snippet = token_prefix(text, included_tokens)
    return f"OBS read_file path={path} tokens={included_tokens}\n{snippet}"


def token_prefix(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    matches = list(islice(TOKEN_RE.finditer(text), min(max_tokens + 1, len(text))))
    if len(matches) <= max_tokens:
        return text
    end = matches[max_tokens - 1].end()
    return text[:end] + "\n[TRUNCATED]"


def load_file_texts(chunks_path: Path) -> dict[str, str]:
    by_file: dict[str, list[str]] = defaultdict(list)
    file_rows: dict[str, str] = {}
    for row in read_jsonl(chunks_path):
        path = str(row.get("path") or "")
        if not path:
            continue
        text = str(row.get("text") or "")
        if row.get("kind") == "file":
            file_rows[path] = text
        else:
            by_file[path].append(text)
    output = dict(file_rows)
    for path, parts in by_file.items():
        output.setdefault(path, "\n".join(parts))
    return output


def grep_files(files: dict[str, str], pattern: str, *, top_k: int) -> list[dict[str, Any]]:
    if not pattern:
        return []
    pattern_lower = pattern.lower()
    results = []
    for path, text in files.items():
        haystack = f"{path}\n{text}".lower()
        matches = haystack.count(pattern_lower)
        if matches <= 0:
            continue
        results.append(
            {
                "path": path,
                "matches": matches,
                "snippet": first_snippet(text, pattern_lower),
                "path_match": pattern_lower in path.lower(),
            }
        )
    results.sort(key=lambda row: (-int(row["path_match"]), -int(row["matches"]), len(str(row["path"])), str(row["path"])))
    return results[:top_k]


def list_dir(files: dict[str, str], path: str) -> list[str]:
    prefix = path.strip("/")
    children = set()
    for file_path in files:
        if prefix:
            if not file_path.startswith(prefix + "/"):
                continue
            rest = file_path[len(prefix) + 1 :]
        else:
            rest = file_path
        if rest:
            children.add(rest.split("/", 1)[0])
    return sorted(children)


def choose_grep_patterns(query: dict[str, Any], *, max_patterns: int) -> list[str]:
    weighted = Counter()
    for key, value in query.items():
        weight = 1
        if key in {"pr_title", "review_comment", "failure_excerpt", "command"}:
            weight = 3
        elif key in {"changed_file", "given_file", "path", "implementation_files"}:
            weight = 2
        for token in tokenize(" ".join(flatten_text(value))):
            if token not in STOP_WORDS and len(token) >= 3:
                weighted[token] += weight
    for phrase in quoted_phrases(query):
        if 3 <= len(phrase) <= 64:
            weighted[phrase.lower()] += 4
    patterns = [pattern for pattern, _count in sorted(weighted.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))]
    return patterns[:max_patterns]


def quoted_phrases(query: dict[str, Any]) -> list[str]:
    text = "\n".join(flatten_text(query))
    phrases = re.findall(r"`([^`]{3,80})`|\"([^\"]{3,80})\"|'([^']{3,80})'", text)
    output = []
    for groups in phrases:
        phrase = next((value for value in groups if value), "")
        if phrase:
            output.append(phrase.strip())
    return output


def query_anchor_paths(query: dict[str, Any]) -> list[str]:
    anchors = []
    for key in ("changed_file", "given_file", "path"):
        value = query.get(key)
        if isinstance(value, str) and "/" in value:
            anchors.append(value)
    for value in query.get("implementation_files") or []:
        if isinstance(value, str) and "/" in value:
            anchors.append(value)
    return dedupe(anchors)


def parse_query_payload(query_text: str) -> dict[str, Any]:
    try:
        value = json.loads(query_text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {"query": query_text}


def query_token_set(query: dict[str, Any]) -> set[str]:
    return tokenize(" ".join(flatten_text(query)))


def tokenize(text: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    normalized = normalized.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    return {token.lower() for token in WORD_RE.findall(normalized) if token.lower() not in STOP_WORDS}


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten_text(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(flatten_text(item))
        return output
    return [str(value)]


def first_snippet(text: str, pattern_lower: str) -> str:
    lower = text.lower()
    index = lower.find(pattern_lower)
    if index < 0:
        return ""
    if len(lower) != len(text):
        # Lowercasing can expand a character, so map back before slicing text.
        lowered_offset = 0
        for original_index, char in enumerate(text):
            lowered_offset += len(char.lower())
            if lowered_offset > index:
                index = original_index
                break
    start = max(0, index - 80)
    end = min(len(text), index + len(pattern_lower) + 160)
    return text[start:end].replace("\n", "\\n")


def pattern_score(pattern: str) -> float:
    return 1.0 + min(2.0, len(pattern) / 20)


def path_token_score(path: str, patterns: list[str]) -> float:
    path_tokens = tokenize(path)
    return 0.25 * sum(1 for pattern in patterns if pattern in path_tokens or pattern in path.lower())


def first_gold_step(steps: list[dict[str, Any]], gold_files: list[str]) -> int | None:
    gold = set(gold_files)
    values = [int(step["step"]) for step in steps if step.get("path") in gold and step.get("step") is not None]
    return min(values) if values else None


def summarize_closed_tool(details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped["overall"].append(detail)
        grouped[str(detail.get("task_type") or "unknown")].append(detail)
    return {key: summarize_closed_tool_rows(rows) for key, rows in sorted(grouped.items())}


def summarize_closed_tool_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    first_hits = [row["closed_tool"]["first_gold_step"] for row in rows if row["closed_tool"].get("first_gold_step") is not None]
    final_hits = [1.0 if any(path in set(row["gold_files"]) for path in row["closed_tool"].get("final_files", [])) else 0.0 for row in rows]
    summary = {
        "samples": len(rows),
        "any_gold_rate": sum(1 for row in rows if row["closed_tool"].get("first_gold_step") is not None) / len(rows),
        "final_any_gold_rate": sum(final_hits) / len(rows),
        "median_first_gold_step": median(first_hits),
        "mean_tool_calls": mean(row["closed_tool"].get("tool_calls") for row in rows),
        "mean_grep_calls": mean(row["closed_tool"].get("grep_calls") for row in rows),
        "mean_read_calls": mean(row["closed_tool"].get("read_calls") for row in rows),
        "mean_read_tokens": mean(row["closed_tool"].get("read_tokens") for row in rows),
    }
    if any("model_usage" in row["closed_tool"] for row in rows):
        summary["mean_observation_tokens"] = mean(row["closed_tool"].get("observation_tokens") or 0 for row in rows)
        for key in ("input_tokens", "output_tokens"):
            summary[f"mean_model_{key}"] = mean((row["closed_tool"].get("model_usage") or {}).get(key) or 0 for row in rows)
    if any("index_calls" in row["closed_tool"] for row in rows):
        summary["mean_index_calls"] = mean(row["closed_tool"].get("index_calls") or 0 for row in rows)
        summary["index_available_rate"] = mean(
            1.0 if (row["closed_tool"].get("index_status") or {}).get("available") else 0.0 for row in rows
        )
    if any("failure" in row["closed_tool"] for row in rows):
        summary["failed_samples"] = sum(1 for row in rows if "failure" in row["closed_tool"])
    if any(row["closed_tool"].get("require_read_submit") for row in rows):
        summary["mean_rejected_submits"] = mean(row["closed_tool"].get("rejected_submits") or 0 for row in rows)
        summary["rejected_submit_rate"] = mean(1.0 if row["closed_tool"].get("rejected_submits") else 0.0 for row in rows)
        for key in ("input_tokens", "output_tokens"):
            summary[f"mean_rejected_submit_model_{key}"] = mean(
                (row["closed_tool"].get("rejected_submit_model_usage") or {}).get(key) or 0 for row in rows
            )
    attempts = [attempt for row in rows for attempt in row["closed_tool"].get("prior_attempts", [])]
    if attempts:
        # Retried samples report their final attempt; earlier attempts are totalled here.
        summary["retried_samples"] = sum(1 for row in rows if row["closed_tool"].get("prior_attempts"))
        summary["prior_attempts"] = len(attempts)
        summary["prior_attempt_tool_calls"] = sum(attempt.get("tool_calls") or 0 for attempt in attempts)
        for key in ("input_tokens", "output_tokens"):
            summary[f"prior_attempt_model_{key}"] = sum(
                (attempt.get("model_usage") or {}).get(key) or 0 for attempt in attempts
            )
    return summary


def report_closed_tool_budget_curve(
    *,
    details_path: Path,
    out_path: Path = DEFAULT_BUDGET_CURVE_OUT,
    markdown_out_path: Path = DEFAULT_BUDGET_CURVE_REPORT,
    budgets: Iterable[int] = (2, 4, 6, 8),
    keep_list: Path | None = None,
) -> dict[str, Any]:
    details = [row for row in read_jsonl(details_path) if isinstance(row, dict)]
    missing: list[str] = []
    if keep_list is not None:
        details, missing = selected_details(details, keep_list)
    budget_values = sorted({int(value) for value in budgets if int(value) > 0})
    if not budget_values:
        raise ValueError("at least one positive budget is required")

    rows = []
    for budget in budget_values:
        prefix_details = [closed_tool_budget_prefix_detail(detail, budget) for detail in details]
        rows.append(
            {
                "budget": budget,
                "evaluated": len(prefix_details),
                "metrics": summarize_trajectory_details(prefix_details),
                "closed_tool_metrics": summarize_closed_tool(prefix_details),
            }
        )

    report = {
        "mode": "closed_tool_budget_prefix_curve",
        "details_path": str(details_path),
        "budgets": budget_values,
        "keep_list": str(keep_list) if keep_list is not None else None,
        "missing_selected": missing,
        "definition": (
            "Post-hoc prefix diagnostic: for each closed-tool trajectory, "
            "score files read within the first B tool calls as the acquired context. "
            "A failed attempt, or a selected sample without a trajectory, counts as empty context. "
            "This is not a counterfactual rerun with a smaller budget."
        ),
        "rows": rows,
        "paths": {"summary": str(out_path), "markdown": str(markdown_out_path)},
        "generated_at": utc_now(),
    }
    write_json(out_path, report)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_budget_curve_markdown(report), encoding="utf-8")
    return report


def selected_details(details: list[dict[str, Any]], keep_list: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Details for the frozen selection, in its order, with empty rows for missing samples."""
    by_id = {str(row.get("sample_id") or row.get("id") or ""): row for row in details}
    rows, missing = [], []
    for record in read_jsonl(keep_list):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in by_id:
            rows.append(by_id[sample_id])
            continue
        missing.append(sample_id)
        sample = {"id": sample_id, **{key: record.get(key) for key in ("task_type", "repo", "base_commit")}}
        rows.append(
            failed_closed_tool_detail(sample, [], {"sample_id": sample_id, "error": "missing"}, policy="")
        )
    return rows, missing


def closed_tool_budget_prefix_detail(detail: dict[str, Any], budget: int) -> dict[str, Any]:
    gold_files = [str(path) for path in detail.get("gold_files", []) if path]
    prefix_steps: list[dict[str, Any]] = []
    for step in detail.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        step_no = optional_int(step.get("step"))
        if step_no is None or step_no > budget:
            continue
        prefix_step = dict(step)
        prefix_step["is_final_context"] = True
        prefix_step["is_utilized_context"] = True
        prefix_steps.append(prefix_step)

    sample = {
        "id": detail.get("sample_id") or detail.get("id"),
        "task_type": detail.get("task_type"),
        "repo": detail.get("repo"),
        "base_commit": detail.get("base_commit"),
        "gold_spans": detail.get("gold_spans") or [],
        "gold_blocks": detail.get("gold_blocks") or [],
    }
    prefix = trajectory_detail(
        sample,
        gold_files,
        prefix_steps,
        supporting_files=[str(path) for path in detail.get("supporting_context_files", []) if path],
    )
    counts = closed_tool_prefix_counts(detail, budget)
    final_files = dedupe(str(step.get("path") or "") for step in prefix_steps)
    original = detail.get("closed_tool", {}) if isinstance(detail.get("closed_tool"), dict) else {}
    prefix["closed_tool"] = {
        "policy": original.get("policy"),
        "model": original.get("model"),
        "budget_prefix": budget,
        "max_tool_calls": original.get("max_tool_calls"),
        "tool_calls": counts["tool_calls"],
        "grep_calls": counts["grep_calls"],
        "read_calls": counts["read_calls"],
        "read_tokens": counts["read_tokens"],
        "final_files": final_files,
        "first_gold_step": first_gold_step(prefix_steps, gold_files),
        "prefix_definition": "files read within the first B tool calls",
    }
    return prefix


def closed_tool_prefix_counts(detail: dict[str, Any], budget: int) -> dict[str, int]:
    trace = detail.get("closed_tool", {}).get("trace", []) if isinstance(detail.get("closed_tool"), dict) else []
    tool_calls = 0
    grep_calls = 0
    read_calls = 0
    read_tokens = 0
    for event in trace:
        if not isinstance(event, dict):
            continue
        tool = str(event.get("tool") or "")
        if tool == "model_action":
            continue
        if tool not in MINIMAL_TOOLS and tool not in INDEXED_TOOLS:
            continue
        if tool_calls >= budget:
            break
        tool_calls += 1
        if tool == "grep":
            grep_calls += 1
        elif tool == "read_file":
            read_calls += 1
            read_tokens += optional_int(event.get("tokens")) or 0
        elif tool == "submit":
            break
    return {
        "tool_calls": tool_calls,
        "grep_calls": grep_calls,
        "read_calls": read_calls,
        "read_tokens": read_tokens,
    }


def render_budget_curve_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Closed-Tool Budget Prefix Curve",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Details: `{report['details_path']}`",
        f"- Definition: {report['definition']}",
        *(
            [f"- Selection: `{report['keep_list']}`; samples without a trajectory: `{len(report['missing_selected'])}`"]
            if report.get("keep_list")
            else []
        ),
        "",
        "## Overall",
        "",
        "| Tool budget | n | Prefix File F1 | Prefix Recall | Prefix Precision | Any-gold read | Tool calls | Grep calls | Read calls | Read tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        metrics = row["metrics"].get("overall", {})
        cost = row["closed_tool_metrics"].get("overall", {})
        lines.append(
            "| {budget} | {n} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {any_gold:.4f} | {calls:.2f} | {grep:.2f} | {reads:.2f} | {tokens:.1f} |".format(
                budget=int(row["budget"]),
                n=int(metrics.get("samples", 0)),
                f1=float(metrics.get("final_file_f1", 0.0)),
                recall=float(metrics.get("final_file_recall", 0.0)),
                precision=float(metrics.get("final_file_precision", 0.0)),
                any_gold=float(cost.get("any_gold_rate", 0.0)),
                calls=float(cost.get("mean_tool_calls", 0.0)),
                grep=float(cost.get("mean_grep_calls", 0.0)),
                reads=float(cost.get("mean_read_calls", 0.0)),
                tokens=float(cost.get("mean_read_tokens", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Tool budget | Task | n | Prefix File F1 | Prefix Recall | Prefix Precision | Any-gold read |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        for task, metrics in sorted(row["metrics"].items()):
            if task == "overall":
                continue
            cost = row["closed_tool_metrics"].get(task, {})
            lines.append(
                "| {budget} | {task} | {n} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {any_gold:.4f} |".format(
                    budget=int(row["budget"]),
                    task=task,
                    n=int(metrics.get("samples", 0)),
                    f1=float(metrics.get("final_file_f1", 0.0)),
                    recall=float(metrics.get("final_file_recall", 0.0)),
                    precision=float(metrics.get("final_file_precision", 0.0)),
                    any_gold=float(cost.get("any_gold_rate", 0.0)),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def report_closed_tool_seed_intervention(
    *,
    control_details: Path,
    arms: dict[str, Path],
    keep_list: Path | None = None,
    out_path: Path = DEFAULT_SEED_INTERVENTION_OUT,
    markdown_out_path: Path = DEFAULT_SEED_INTERVENTION_REPORT,
) -> dict[str, Any]:
    if not arms:
        raise ValueError("at least one intervention arm is required")
    keep_ids = load_keep_ids(keep_list)
    loaded: dict[str, dict[str, dict[str, Any]]] = {"no_seed": load_existing_details(control_details)}
    for label, path in arms.items():
        loaded[label] = load_existing_details(path)
    common_ids: set[str] | None = None
    for rows in loaded.values():
        ids = set(rows)
        if keep_ids is not None:
            ids &= keep_ids
        common_ids = ids if common_ids is None else common_ids & ids
    sample_ids = sorted(common_ids or set())

    rows = []
    control_by_sample = loaded["no_seed"]
    for label, by_sample in loaded.items():
        details = [by_sample[sample_id] for sample_id in sample_ids]
        rows.append(
            {
                "label": label,
                "details_path": str(control_details if label == "no_seed" else arms[label]),
                "evaluated": len(details),
                "metrics": summarize_trajectory_details(details),
                "closed_tool_metrics": summarize_closed_tool(details),
                "seed_metrics": summarize_seed_rows(details),
                "paired_vs_no_seed": summarize_seed_pairwise(control_by_sample, by_sample, sample_ids),
            }
        )

    report = {
        "mode": "closed_tool_seed_intervention",
        "generated_at": utc_now(),
        "definition": (
            "Context-acquisition intervention over a fixed sample intersection. "
            "The no_seed arm is loaded from an existing closed-tool run; seeded arms "
            "preload top ranked files as initial read context before the same policy acts."
        ),
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "control_details": str(control_details),
        "arms": {label: str(path) for label, path in arms.items()},
        "keep_list": str(keep_list) if keep_list else "",
        "rows": rows,
        "paths": {"summary": str(out_path), "markdown": str(markdown_out_path)},
    }
    write_json(out_path, report)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_seed_intervention_markdown(report), encoding="utf-8")
    return report


def summarize_seed_rows(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        return {"samples": 0}
    seed_tokens = [float(row.get("closed_tool", {}).get("seed_tokens") or 0.0) for row in details]
    read_tokens = [float(row.get("closed_tool", {}).get("read_tokens") or 0.0) for row in details]
    return {
        "samples": len(details),
        "seed_any_gold_rate": mean(1.0 if row.get("closed_tool", {}).get("seed_any_gold") else 0.0 for row in details),
        "mean_seed_tokens": mean(seed_tokens),
        "mean_post_seed_read_tokens": mean(max(0.0, total - seed) for total, seed in zip(read_tokens, seed_tokens)),
        "mean_seed_files": mean(len(row.get("closed_tool", {}).get("seed_files") or []) for row in details),
    }


def summarize_seed_pairwise(
    control_by_sample: dict[str, dict[str, Any]],
    arm_by_sample: dict[str, dict[str, Any]],
    sample_ids: list[str],
) -> dict[str, Any]:
    if not sample_ids:
        return {"samples": 0}
    f1_deltas: list[float] = []
    hit_deltas: list[float] = []
    miss_deltas: list[float] = []
    first_deltas: list[float] = []
    rescued = 0
    lost = 0
    no_seed_miss = 0
    improved = 0
    worsened = 0
    unchanged = 0
    for sample_id in sample_ids:
        control = control_by_sample[sample_id]
        arm = arm_by_sample[sample_id]
        delta = detail_final_f1(arm) - detail_final_f1(control)
        f1_deltas.append(delta)
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
            worsened += 1
        else:
            unchanged += 1
        control_any = detail_any_gold(control)
        arm_any = detail_any_gold(arm)
        if not control_any:
            no_seed_miss += 1
            if arm_any:
                rescued += 1
        elif not arm_any:
            lost += 1
        if bool(arm.get("closed_tool", {}).get("seed_any_gold")):
            hit_deltas.append(delta)
        else:
            miss_deltas.append(delta)
        control_first = detail_first_gold(control)
        arm_first = detail_first_gold(arm)
        if control_first is not None and arm_first is not None:
            first_deltas.append(float(arm_first) - float(control_first))
    total = len(sample_ids)
    return {
        "samples": total,
        "mean_final_f1_delta": mean(f1_deltas),
        "improved_f1_count": improved,
        "worsened_f1_count": worsened,
        "unchanged_f1_count": unchanged,
        "improved_f1_rate": improved / total,
        "worsened_f1_rate": worsened / total,
        "rescued_any_gold_count": rescued,
        "lost_any_gold_count": lost,
        "no_seed_miss_count": no_seed_miss,
        "rescued_any_gold_rate": rescued / total,
        "lost_any_gold_rate": lost / total,
        "rescue_rate_given_no_seed_miss": rescued / no_seed_miss if no_seed_miss else 0.0,
        "mean_first_gold_step_delta_when_both_hit": mean(first_deltas),
        "seed_hit_samples": len(hit_deltas),
        "seed_miss_samples": len(miss_deltas),
        "seed_hit_mean_final_f1_delta": mean(hit_deltas),
        "seed_miss_mean_final_f1_delta": mean(miss_deltas),
    }


def detail_final_f1(detail: dict[str, Any]) -> float:
    return float(detail.get("metrics", {}).get("final_file_f1") or 0.0)


def detail_any_gold(detail: dict[str, Any]) -> bool:
    return detail.get("closed_tool", {}).get("first_gold_step") is not None


def detail_first_gold(detail: dict[str, Any]) -> float | None:
    value = detail.get("closed_tool", {}).get("first_gold_step")
    return None if value is None else float(value)


def render_seed_intervention_markdown(report: dict[str, Any]) -> str:
    control = next((row for row in report["rows"] if row["label"] == "no_seed"), None)
    control_metrics = (control or {}).get("metrics", {}).get("overall", {})
    control_cost = (control or {}).get("closed_tool_metrics", {}).get("overall", {})
    lines = [
        "# Closed-Tool Seed Intervention Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Samples in common intersection: `{report['sample_count']}`",
        f"- Definition: {report['definition']}",
        "",
        "## Overall",
        "",
        "| Arm | n | Final F1 | Delta F1 | Final R | Final P | Any-gold | Delta Any | Median first gold | Tool calls | Read tokens | Post-seed tokens | Seed any | Seed tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        metrics = row["metrics"].get("overall", {})
        cost = row["closed_tool_metrics"].get("overall", {})
        seed = row["seed_metrics"]
        lines.append(
            "| {label} | {n} | {f1:.4f} | {delta_f1:+.4f} | {recall:.4f} | {precision:.4f} | {any_gold:.4f} | {delta_any:+.4f} | {first} | {calls:.2f} | {tokens:.1f} | {post_tokens:.1f} | {seed_any:.4f} | {seed_tokens:.1f} |".format(
                label=row["label"],
                n=int(metrics.get("samples", 0)),
                f1=float(metrics.get("final_file_f1", 0.0)),
                delta_f1=float(metrics.get("final_file_f1", 0.0)) - float(control_metrics.get("final_file_f1", 0.0)),
                recall=float(metrics.get("final_file_recall", 0.0)),
                precision=float(metrics.get("final_file_precision", 0.0)),
                any_gold=float(cost.get("any_gold_rate", 0.0)),
                delta_any=float(cost.get("any_gold_rate", 0.0)) - float(control_cost.get("any_gold_rate", 0.0)),
                first="" if cost.get("median_first_gold_step") is None else f"{float(cost['median_first_gold_step']):.1f}",
                calls=float(cost.get("mean_tool_calls", 0.0)),
                tokens=float(cost.get("mean_read_tokens", 0.0)),
                post_tokens=float(seed.get("mean_post_seed_read_tokens", 0.0)),
                seed_any=float(seed.get("seed_any_gold_rate", 0.0)),
                seed_tokens=float(seed.get("mean_seed_tokens", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Delta vs No Seed",
            "",
            "| Arm | n | Mean ΔF1 | Improved | Worsened | Rescued any-gold | Lost any-gold | Rescue / no-seed miss | Seed-hit ΔF1 | Seed-miss ΔF1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        paired = row.get("paired_vs_no_seed", {})
        lines.append(
            "| {label} | {n} | {delta:+.4f} | {improved:.4f} | {worsened:.4f} | {rescued:.4f} | {lost:.4f} | {rescue_miss:.4f} | {hit_delta:+.4f} | {miss_delta:+.4f} |".format(
                label=row["label"],
                n=int(paired.get("samples", 0)),
                delta=float(paired.get("mean_final_f1_delta", 0.0)),
                improved=float(paired.get("improved_f1_rate", 0.0)),
                worsened=float(paired.get("worsened_f1_rate", 0.0)),
                rescued=float(paired.get("rescued_any_gold_rate", 0.0)),
                lost=float(paired.get("lost_any_gold_rate", 0.0)),
                rescue_miss=float(paired.get("rescue_rate_given_no_seed_miss", 0.0)),
                hit_delta=float(paired.get("seed_hit_mean_final_f1_delta", 0.0)),
                miss_delta=float(paired.get("seed_miss_mean_final_f1_delta", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Arm | Task | n | Final F1 | Final R | Final P | Any-gold | Median first gold | Tool calls | Read tokens |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        for task, metrics in sorted(row["metrics"].items()):
            if task == "overall":
                continue
            cost = row["closed_tool_metrics"].get(task, {})
            lines.append(
                "| {label} | {task} | {n} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {any_gold:.4f} | {first} | {calls:.2f} | {tokens:.1f} |".format(
                    label=row["label"],
                    task=task,
                    n=int(metrics.get("samples", 0)),
                    f1=float(metrics.get("final_file_f1", 0.0)),
                    recall=float(metrics.get("final_file_recall", 0.0)),
                    precision=float(metrics.get("final_file_precision", 0.0)),
                    any_gold=float(cost.get("any_gold_rate", 0.0)),
                    first="" if cost.get("median_first_gold_step") is None else f"{float(cost['median_first_gold_step']):.1f}",
                    calls=float(cost.get("mean_tool_calls", 0.0)),
                    tokens=float(cost.get("mean_read_tokens", 0.0)),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(summary: dict[str, Any]) -> str:
    budget = summary["budget"]
    if "max_grep_calls" in budget:
        policy_split = (
            f"- Policy split: at most `{budget['max_grep_calls']}` grep calls and "
            f"`{budget['max_read_tokens_per_file']}` tokens per read."
        )
    else:
        policy_split = (
            f"- Policy split: at most `{budget.get('max_model_turns', '')}` model turns, "
            f"grep top-`{budget.get('grep_top_k', '')}`, and `{budget['max_read_tokens_per_file']}` tokens per read."
        )
    lines = [
        "# Closed-Tool Context Acquisition Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Evaluated samples: `{summary['evaluated']}`",
        f"- Policy: `{summary['policy']}`",
        f"- Allowed tools: `{', '.join(budget['allowed_tools'])}`",
        f"- Budget: `{budget['max_tool_calls']}` tool calls, `{budget['max_read_tokens']}` read tokens, final top-`{budget['final_k']}` files",
        policy_split,
        (
            "- Indexed tools come from an external provider and return locations only; "
            "no embedding, RepoMap, edit, test execution, network, or fix-commit access is used."
            if summary.get("indexed_tools")
            else "- No embedding, RepoMap, edit, test execution, network, or fix-commit access is used."
        ),
        f"- Protocol violation samples: `{summary.get('protocol_violation_samples', 0)}`",
        "",
        "## Final Context Metrics",
        "",
        "| Slice | n | Final File F1 | Final Recall | Final Precision | Retrieved F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slice_name, row in sorted(summary["metrics"].items()):
        lines.append(
            "| {slice} | {n} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {retrieved:.4f} |".format(
                slice=slice_name,
                n=int(row.get("samples", 0)),
                f1=float(row.get("final_file_f1", 0.0)),
                recall=float(row.get("final_file_recall", 0.0)),
                precision=float(row.get("final_file_precision", 0.0)),
                retrieved=float(row.get("retrieved_file_f1", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Tool Cost",
            "",
            "| Slice | n | Any-gold read | Final any-gold | Median first gold step | Tool calls | Grep calls | Read calls | Read tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slice_name, row in sorted(summary["closed_tool_metrics"].items()):
        lines.append(
            "| {slice} | {n} | {any:.4f} | {final:.4f} | {first} | {tools:.2f} | {grep:.2f} | {reads:.2f} | {tokens:.1f} |".format(
                slice=slice_name,
                n=int(row.get("samples", 0)),
                any=float(row.get("any_gold_rate", 0.0)),
                final=float(row.get("final_any_gold_rate", 0.0)),
                first="" if row.get("median_first_gold_step") is None else f"{float(row['median_first_gold_step']):.1f}",
                tools=float(row.get("mean_tool_calls", 0.0)),
                grep=float(row.get("mean_grep_calls", 0.0)),
                reads=float(row.get("mean_read_calls", 0.0)),
                tokens=float(row.get("mean_read_tokens", 0.0)),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def median(values: list[int | float]) -> float | None:
    if not values:
        return None
    values = sorted(float(value) for value in values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def mean(values: Iterable[Any]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
