from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from .baseline import (
    CANDIDATE_FILTERS,
    filter_candidate_chunks,
    gold_file_ranks,
    hard_negative_files,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    sample_metrics,
    target_gold_files,
    unique_ranked_paths,
)
from .curate import filter_samples, load_keep_ids
from .embedding_eval import (
    SentenceTransformerEmbedder,
    TextEmbedder,
    TextEmbeddingCache,
    directory_size,
    embedding_cache_metadata,
    encode_query_texts_resilient,
    load_or_encode_chunk_vectors,
    select_samples_for_embedding_run,
    vector_scores,
)
from .io import ensure_parent, read_jsonl
from .progress import ProgressReporter
from .selective_eval import render_selective_report, summarize_selective, threshold_sweep


def evaluate_selective_embedding_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    model_name: str,
    out_path: Path | None = None,
    details_path: Path | None = None,
    sweep_path: Path | None = None,
    report_path: Path | None = None,
    keep_list: Path | None = None,
    cache_dir: Path | None = None,
    shared_text_cache: Path | None = None,
    sample_ids: set[str] | None = None,
    shard_count: int | None = None,
    shard_index: int | None = None,
    limit_samples: int | None = None,
    batch_size: int = 32,
    query_batch_size: int | None = None,
    device: str | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    query_input_type: str | None = None,
    passage_input_type: str | None = None,
    normalize_embeddings: bool = True,
    trust_remote_code: bool = False,
    embedder: TextEmbedder | None = None,
    progress: bool = False,
    progress_stream: TextIO | None = None,
    candidate_filter: str = "all_files",
) -> dict[str, Any]:
    started_at = time.monotonic()
    if candidate_filter not in CANDIDATE_FILTERS:
        raise ValueError(f"unknown candidate_filter: {candidate_filter}")
    reporter = ProgressReporter(progress, progress_stream)
    keep_ids = load_keep_ids(keep_list)
    if keep_list:
        if keep_ids is None:
            reporter.message(f"keep list not found, evaluating all samples: {keep_list}")
        else:
            reporter.message(f"loaded keep list: {len(keep_ids)} ids")
    resolved_sample_paths = list(sample_paths)
    raw_samples = list(_iter_samples(resolved_sample_paths))
    if not raw_samples:
        rendered_paths = ", ".join(str(path) for path in resolved_sample_paths) or "<none>"
        raise ValueError(
            "No benchmark samples found. Download the requested benchmark release "
            f"or pass valid sample JSONL files. Checked: {rendered_paths}"
        )
    loaded_samples = list(filter_samples(raw_samples, keep_ids))
    if not loaded_samples:
        raise ValueError("No benchmark samples remain after applying the keep-list filter.")
    samples, selection = select_samples_for_embedding_run(
        loaded_samples,
        sample_ids=sample_ids,
        shard_count=shard_count,
        shard_index=shard_index,
        limit_samples=limit_samples,
    )
    reporter.message(f"loaded benchmark samples: {len(samples)}")

    reporter.message(f"loading corpus manifest: {corpus_dir / 'corpus_manifest.jsonl'}")
    manifest = load_corpus_manifest(corpus_dir)
    reporter.message(f"loaded corpus manifest: {len(manifest)} commit corpora")
    reporter.message(f"loading embedding model: {model_name}")
    actual_embedder = embedder or SentenceTransformerEmbedder(
        model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        trust_remote_code=trust_remote_code,
    )
    reporter.message("embedding model loaded")
    passage_embedding_options = embedding_cache_metadata(actual_embedder, input_type=passage_input_type)
    shared_cache = (
        TextEmbeddingCache(
            shared_text_cache,
            {
                "model": model_name,
                "normalize_embeddings": normalize_embeddings,
                "passage_prefix": passage_prefix,
                "embedding_options": passage_embedding_options,
            },
        )
        if shared_text_cache
        else None
    )

    effective_query_batch_size = max(1, query_batch_size if query_batch_size is not None else batch_size)
    skipped = Counter()
    pending_by_chunks_path: dict[Path, list[tuple[int, dict[str, Any], list[str], str, bool]]] = defaultdict(list)
    sample_bar = reporter.bar("preparing selective embedding samples", len(samples))
    for sample_index, sample in enumerate(samples):
        gold_files = target_gold_files(sample)
        is_no_gold = bool((sample.get("gold") or {}).get("no_gold") is True)
        if not gold_files and not is_no_gold:
            skipped["no_gold_unlabeled"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        query_text = query_prefix + query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        chunks_path = manifest.get(key)
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        pending_by_chunks_path[chunks_path].append((sample_index, sample, gold_files, query_text, is_no_gold))
        sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")

    details: list[dict[str, Any]] = []
    vector_cache: dict[tuple[str, str, str], Any] = {}
    chunk_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    pending_queries: list[dict[str, Any]] = []
    eval_bar = reporter.bar("ranking selective embedding samples", sum(len(v) for v in pending_by_chunks_path.values()))

    def flush_pending_queries() -> None:
        nonlocal pending_queries
        if not pending_queries:
            return
        query_vectors = encode_query_texts_resilient(
            actual_embedder,
            [str(item["query_text"]) for item in pending_queries],
            batch_size=effective_query_batch_size,
            show_progress_bar=False,
            input_type=query_input_type,
            reporter=reporter,
        )
        for item, query_vector in zip(pending_queries, query_vectors):
            detail = selective_embedding_detail(
                sample_index=int(item["sample_index"]),
                sample=item["sample"],
                gold_files=item["gold_files"],
                chunks=item["chunks"],
                chunk_vectors=item["vectors"],
                query_vector=query_vector,
                candidate_filter=candidate_filter,
                is_no_gold=bool(item["is_no_gold"]),
            )
            details.append(detail)
            eval_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
        pending_queries = []

    try:
        for corpus_index, (chunks_path, pending) in enumerate(pending_by_chunks_path.items(), start=1):
            repo = str(pending[0][1].get("repo") or "")
            base_commit = str(pending[0][1].get("base_commit") or "")
            filtered_key = (repo, base_commit, candidate_filter)
            chunks = chunk_cache.get(filtered_key)
            if chunks is None:
                reporter.message(f"loading corpus {corpus_index}/{len(pending_by_chunks_path)}: {chunks_path}")
                chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
                chunk_cache[filtered_key] = chunks
            if not chunks:
                skipped["empty_corpus"] += len(pending)
                eval_bar.update(step=len(pending), suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
                continue
            vectors = vector_cache.get(filtered_key)
            if vectors is None:
                vectors = load_or_encode_chunk_vectors(
                    chunks=chunks,
                    chunks_path=chunks_path,
                    embedder=actual_embedder,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    shared_text_cache=shared_cache,
                    batch_size=batch_size,
                    passage_prefix=passage_prefix,
                    input_type=passage_input_type,
                    normalize_embeddings=normalize_embeddings,
                    embedding_options=passage_embedding_options,
                    candidate_filter=candidate_filter,
                    progress_reporter=reporter,
                )
                vector_cache[filtered_key] = vectors
            for sample_index, sample, gold_files, query_text, is_no_gold in pending:
                pending_queries.append(
                    {
                        "sample_index": sample_index,
                        "sample": sample,
                        "gold_files": gold_files,
                        "query_text": query_text,
                        "is_no_gold": is_no_gold,
                        "chunks": chunks,
                        "vectors": vectors,
                    }
                )
                if len(pending_queries) >= effective_query_batch_size:
                    flush_pending_queries()
        flush_pending_queries()
        eval_bar.finish(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    finally:
        if shared_cache is not None:
            shared_cache.close()

    details.sort(key=lambda item: item.pop("_sample_index", 0))
    sweep = threshold_sweep(details)
    summary = summarize_selective(
        details,
        sweep,
        dict(skipped),
        time.monotonic() - started_at,
        candidate_filter,
        keep_list,
        ranker="embedding",
    )
    summary.update(
        {
            "mode": "selective_embedding",
            "model": model_name,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "shared_text_cache": str(shared_text_cache) if shared_text_cache else None,
            "selection": selection,
            "runtime": {
                "wall_time_seconds": time.monotonic() - started_at,
                "device": device,
                "batch_size": batch_size,
                "query_batch_size": effective_query_batch_size,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "cache_size_bytes": directory_size(cache_dir) if cache_dir else None,
                "shared_text_cache": str(shared_text_cache) if shared_text_cache else None,
                "shared_text_cache_size_bytes": shared_text_cache.stat().st_size if shared_text_cache and shared_text_cache.exists() else None,
            },
        }
    )
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        write_jsonl(details_path, details)
    if sweep_path:
        write_jsonl(sweep_path, sweep)
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(render_selective_report(summary), encoding="utf-8")
    return summary


def selective_embedding_detail(
    sample_index: int,
    sample: dict[str, Any],
    gold_files: list[str],
    chunks: list[dict[str, Any]],
    chunk_vectors: Any,
    query_vector: Sequence[float],
    candidate_filter: str,
    is_no_gold: bool,
) -> dict[str, Any]:
    scored = rank_chunks_by_vector_scores(query_vector, chunk_vectors, chunks)
    ranked = [chunk for _, chunk in scored]
    top_score = float(scored[0][0]) if scored else 0.0
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    top_files = unique_ranked_paths(ranked)[:20]
    detail: dict[str, Any] = {
        "_sample_index": sample_index,
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "candidate_filter": candidate_filter,
        "ranker": "embedding",
        "label": "no_gold" if is_no_gold else "positive",
        "no_gold_reason": (sample.get("gold") or {}).get("reason") if is_no_gold else None,
        "gold_files": gold_files,
        "confidence": top_score,
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": top_score - second_score,
        "top_files": top_files,
    }
    if gold_files:
        hard_negatives = hard_negative_files(sample)
        metrics = sample_metrics(gold_files, ranked, hard_negative_files=hard_negatives)
        detail.update(
            {
                "metrics": metrics,
                "gold_ranks": gold_file_ranks(gold_files, ranked),
                "hard_negative_files": hard_negatives,
                "hard_negative_ranks": gold_file_ranks(hard_negatives, ranked),
            }
        )
    return detail


def rank_chunks_by_vector_scores(
    query_vector: Sequence[float],
    chunk_vectors: Any,
    chunks: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    scores = vector_scores(query_vector, chunk_vectors)
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for score, chunk in zip(scores, chunks):
        scored.append((float(score), str(chunk.get("path", "")), str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(score, chunk) for score, _, _, chunk in scored]


def _iter_samples(sample_paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sample_paths:
        yield from read_jsonl(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
