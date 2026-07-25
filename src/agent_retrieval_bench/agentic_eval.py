from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO

from .baseline import (
    block_metrics_at_budget,
    filter_candidate_chunks,
    gold_blocks,
    gold_file_ranks,
    gold_spans,
    hard_negative_files,
    iter_samples,
    line_metrics_at_budget,
    load_corpus_manifest,
    query_provenance,
    query_has_leakage,
    query_text_for_eval,
    rank_chunks,
    sample_metrics,
    span_metrics_from_line_metrics,
    summarize_details,
    target_gold_files,
    unique_ranked_paths,
    validate_candidate_filter,
)
from .curate import filter_samples, load_keep_ids
from .io import ensure_parent, read_jsonl
from .progress import ProgressReporter


def evaluate_agentic_search(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    candidate_filter: str = "all_files",
    max_turns: int = 4,
    max_tool_calls_per_turn: int = 8,
    max_read_chars: int = 8_000,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    validate_candidate_filter(candidate_filter)
    if max_turns <= 0:
        raise ValueError("max_turns must be positive.")
    if max_tool_calls_per_turn <= 0:
        raise ValueError("max_tool_calls_per_turn must be positive.")
    reporter = ProgressReporter(progress, progress_stream)
    reporter.message(f"loading corpus manifest: {corpus_dir / 'corpus_manifest.jsonl'}")
    manifest = load_corpus_manifest(corpus_dir)
    reporter.message(f"loaded corpus manifest: {len(manifest)} commit corpora")
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        if limit_samples and len(samples) >= limit_samples:
            break
        samples.append(sample)
    reporter.message(f"loaded benchmark samples: {len(samples)}")

    details: list[dict[str, Any]] = []
    skipped = Counter()
    chunks_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    sample_bar = reporter.bar("evaluating samples", len(samples))
    for sample in samples:
        gold_files = target_gold_files(sample)
        if not gold_files:
            skipped["no_gold"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        query_text = query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        chunks_path = manifest.get(key)
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        filtered_key = (key[0], key[1], candidate_filter)
        chunks = chunks_cache.get(filtered_key)
        if chunks is None:
            chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
            chunks_cache[filtered_key] = chunks
        if not chunks:
            skipped["empty_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        detail = run_scripted_search_agent(
            sample=sample,
            gold_files=gold_files,
            query_text=query_text,
            chunks=chunks,
            candidate_filter=candidate_filter,
            max_turns=max_turns,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            max_read_chars=max_read_chars,
        )
        details.append(detail)
        sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")

    result = {
        "mode": "agentic_search",
        "model": "scripted-search-read",
        "candidate_filter": candidate_filter,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": len(details),
        "skipped": dict(skipped),
        "agentic_budget": {
            "max_turns": max_turns,
            "max_tool_calls_per_turn": max_tool_calls_per_turn,
            "max_read_chars": max_read_chars,
        },
        "metrics": summarize_details(details),
        "runtime": {
            "wall_time_seconds": time.monotonic() - started_at,
            "progress": bool(progress),
        },
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        write_jsonl(details_path, details)
    return result


def run_scripted_search_agent(
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    chunks: list[dict[str, Any]],
    candidate_filter: str,
    max_turns: int,
    max_tool_calls_per_turn: int,
    max_read_chars: int,
) -> dict[str, Any]:
    started_at = time.monotonic()
    ranked = rank_chunks(query_text, chunks)
    ranked_paths = unique_ranked_paths(ranked)
    tool_calls: list[dict[str, Any]] = []
    search_hits = ranked_paths[: max(1, max_tool_calls_per_turn)]
    tool_calls.append({"turn": 1, "tool": "search", "query_chars": len(query_text), "hits": search_hits})
    read_budget = max(0, max_tool_calls_per_turn - 1)
    chars_read = 0
    for path in search_hits[:read_budget]:
        text = file_text_for_path(chunks, path)
        clipped = text[:max_read_chars]
        chars_read += len(clipped)
        tool_calls.append({"turn": 1, "tool": "read", "path": path, "chars": len(clipped)})
    hard_negatives = hard_negative_files(sample)
    span_rows = gold_spans(sample)
    block_rows = gold_blocks(sample)
    metrics = sample_metrics(gold_files, ranked, hard_negative_files=hard_negatives)
    detail = {
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "candidate_filter": candidate_filter,
        "gold_files": gold_files,
        "gold_spans": span_rows,
        "gold_blocks": block_rows,
        "hard_negative_files": hard_negatives,
        "query_provenance": query_provenance(sample),
        "gold_ranks": gold_file_ranks(gold_files, ranked),
        "hard_negative_ranks": gold_file_ranks(hard_negatives, ranked),
        "top_files": ranked_paths[:20],
        "metrics": metrics,
        "agentic": {
            "max_turns": max_turns,
            "max_tool_calls_per_turn": max_tool_calls_per_turn,
            "turns_used": 1,
            "tool_calls": len(tool_calls),
            "chars_read": chars_read,
            "latency_seconds": time.monotonic() - started_at,
            "trace": tool_calls,
        },
    }
    if span_rows:
        line_metrics = line_metrics_at_budget(span_rows, ranked)
        detail["line_metrics"] = line_metrics
        detail["span_metrics"] = span_metrics_from_line_metrics(line_metrics)
    if block_rows:
        detail["block_metrics"] = block_metrics_at_budget(block_rows, ranked)
    return detail


def file_text_for_path(chunks: list[dict[str, Any]], path: str) -> str:
    file_chunks = [chunk for chunk in chunks if str(chunk.get("path") or "") == path]
    full_file = [chunk for chunk in file_chunks if str(chunk.get("kind") or "") == "file"]
    rows = full_file or file_chunks
    return "\n".join(str(chunk.get("text") or "") for chunk in rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
