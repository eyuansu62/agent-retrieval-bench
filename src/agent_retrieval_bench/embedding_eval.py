from __future__ import annotations

import json
import math
import os
import re
import hashlib
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, TextIO

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
    sample_metrics,
    span_metrics_from_line_metrics,
    summarize_details,
    target_gold_files,
    unique_ranked_paths,
    validate_candidate_filter,
)
from .curate import filter_samples, load_keep_ids
from .io import ensure_parent, read_jsonl, repo_slug
from .progress import ProgressBar as ProgressBar, ProgressReporter as ProgressReporter

SHARED_TEXT_CACHE_GET_BATCH_SIZE = 500
SHARED_TEXT_CACHE_PUT_BATCH_SIZE = 1_000
SHARED_TEXT_CACHE_ENCODE_WINDOW_MIN = 256
SHARED_TEXT_CACHE_ENCODE_WINDOW_MAX = 4_096
SHARED_TEXT_CACHE_ENCODE_WINDOW_BATCH_MULTIPLIER = 128


class TextEmbedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Sequence[Sequence[float]]:
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding evaluation requires optional dependencies. Install with: pip install -e '.[embedding]'"
            ) from exc
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = False
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
        )


class VoyageAPIEmbedder:
    def __init__(
        self,
        model_name: str = "voyage-code-3",
        api_key: str | None = None,
        api_base: str = "https://api.voyageai.com/v1",
        output_dimension: int | None = None,
        output_dtype: str = "float",
        truncation: bool = True,
        normalize_embeddings: bool = True,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        retry_base_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.0,
        max_request_chars: int | None = None,
        request_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.truncation = truncation
        self.normalize_embeddings = normalize_embeddings
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.max_request_chars = max_request_chars if max_request_chars and max_request_chars > 0 else None
        self.request_func = request_func
        self._last_request_at = 0.0
        if self.request_func is None and not self.api_key:
            raise RuntimeError("Voyage evaluation requires VOYAGE_API_KEY or --api-key.")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        input_type: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in voyage_text_batches(texts, batch_size=batch_size, max_request_chars=self.max_request_chars):
            payload: dict[str, Any] = {
                "input": batch,
                "model": self.model_name,
                "truncation": self.truncation,
                "output_dtype": self.output_dtype,
            }
            if input_type:
                payload["input_type"] = input_type
            if self.output_dimension:
                payload["output_dimension"] = self.output_dimension
            response = self._request_embeddings(payload)
            batch_vectors = self._extract_embeddings(response, expected_count=len(batch))
            if self.normalize_embeddings:
                batch_vectors = [normalize_vector(vector) for vector in batch_vectors]
            vectors.extend(batch_vectors)
        return vectors

    def cache_metadata(self) -> dict[str, Any]:
        return {
            "provider": "voyage",
            "api_base": self.api_base,
            "output_dimension": self.output_dimension,
            "output_dtype": self.output_dtype,
            "truncation": self.truncation,
            "normalize_embeddings": self.normalize_embeddings,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "max_request_chars": self.max_request_chars,
        }

    def _request_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        if self.request_func is not None:
            try:
                return self.request_func(payload)
            finally:
                self._last_request_at = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post_json(payload)
                self._last_request_at = time.monotonic()
                return response
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                retry_after = parse_retry_after(exc.headers.get("Retry-After"))
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    message = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Voyage embedding request failed with HTTP {exc.code}: {message}") from exc
            except urllib.error.URLError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                retry_after = None
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Voyage embedding request failed: {exc}") from exc
            sleep_seconds = retry_after
            if sleep_seconds is None and self.retry_base_seconds:
                sleep_seconds = self.retry_base_seconds * (2**attempt)
            if sleep_seconds:
                time.sleep(sleep_seconds)
        raise RuntimeError(f"Voyage embedding request failed: {last_error}")

    def _throttle(self) -> None:
        if not self.min_request_interval_seconds or not self._last_request_at:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_embeddings(self, response: dict[str, Any], expected_count: int) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Voyage embedding response is missing a data list.")
        ordered = sorted(data, key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0)
        embeddings: list[list[float]] = []
        for row in ordered:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise RuntimeError("Voyage embedding response contains an invalid embedding row.")
            embeddings.append([float(value) for value in row["embedding"]])
        if len(embeddings) != expected_count:
            raise RuntimeError(f"Voyage returned {len(embeddings)} embeddings for {expected_count} inputs.")
        return embeddings


def voyage_text_batches(
    texts: Sequence[str],
    batch_size: int = 32,
    max_request_chars: int | None = None,
) -> Iterable[list[str]]:
    effective_batch_size = max(1, min(batch_size, 1000))
    max_chars = max_request_chars if max_request_chars and max_request_chars > 0 else None
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        text_chars = len(text)
        should_flush_for_count = len(batch) >= effective_batch_size
        should_flush_for_chars = bool(batch and max_chars is not None and batch_chars + text_chars > max_chars)
        if should_flush_for_count or should_flush_for_chars:
            yield batch
            batch = []
            batch_chars = 0
        batch.append(text)
        batch_chars += text_chars
    if batch:
        yield batch


def evaluate_embedding_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    model_name: str,
    out_path: Path | None = None,
    details_path: Path | None = None,
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
    resume_details: bool = False,
) -> dict[str, Any]:
    started_at = time.monotonic()
    validate_candidate_filter(candidate_filter)
    reporter = ProgressReporter(progress, progress_stream)
    keep_ids = load_keep_ids(keep_list)
    if keep_list:
        if keep_ids is None:
            reporter.message(f"keep list not found, evaluating all samples: {keep_list}")
        else:
            reporter.message(f"loaded keep list: {len(keep_ids)} ids")
    loaded_samples = list(filter_samples(iter_samples(sample_paths), keep_ids))
    samples, selection = select_samples_for_embedding_run(
        loaded_samples,
        sample_ids=sample_ids,
        shard_count=shard_count,
        shard_index=shard_index,
        limit_samples=limit_samples,
    )
    reporter.message(f"loaded benchmark samples: {len(samples)}")
    sample_ids = {str(sample.get("id")) for sample in samples if sample.get("id")}
    effective_query_batch_size = max(1, query_batch_size if query_batch_size is not None else batch_size)
    details: list[dict[str, Any]] = []
    resumed_ids: set[str] = set()
    if resume_details and details_path and details_path.exists():
        details, resumed_ids = load_resumable_details(
            details_path=details_path,
            sample_ids=sample_ids,
            candidate_filter=candidate_filter,
        )
        reporter.message(f"resumed completed details: {len(resumed_ids)} samples from {details_path}")
    if sample_ids and resumed_ids == sample_ids:
        reporter.message("all selected samples already have details; rebuilding summary without loading model or corpus")
        result = {
            "mode": "embedding",
            "model": model_name,
            "candidate_filter": candidate_filter,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "shared_text_cache": str(shared_text_cache) if shared_text_cache else None,
            "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
            "selection": selection,
            "evaluated": len(details),
            "skipped": {},
            "metrics": summarize_details(details),
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
        if out_path:
            ensure_parent(out_path)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

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
    vector_cache: dict[tuple[str, str, str], Any] = {}
    chunk_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    skipped = Counter()
    evaluated = len(details)
    pending_queries: list[dict[str, Any]] = []
    sample_bar = reporter.bar("evaluating samples", len(samples))
    details_handle = None

    def flush_pending_queries() -> None:
        nonlocal evaluated, pending_queries
        if not pending_queries:
            return
        query_vectors = encode_query_texts_resilient(
            actual_embedder,
            [str(item["query_text"]) for item in pending_queries],
            batch_size=batch_size,
            show_progress_bar=False,
            input_type=query_input_type,
            reporter=reporter,
        )
        for item, query_vector in zip(pending_queries, query_vectors):
            sample = item["sample"]
            gold_files = item["gold_files"]
            hard_negatives = hard_negative_files(sample)
            span_rows = gold_spans(sample)
            block_rows = gold_blocks(sample)
            chunks = item["chunks"]
            ranked = rank_chunks_by_vectors(query_vector, item["vectors"], chunks)
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
                "top_files": unique_ranked_paths(ranked)[:20],
                "metrics": metrics,
            }
            if span_rows:
                line_metrics = line_metrics_at_budget(span_rows, ranked)
                detail["line_metrics"] = line_metrics
                detail["span_metrics"] = span_metrics_from_line_metrics(line_metrics)
            if block_rows:
                detail["block_metrics"] = block_metrics_at_budget(block_rows, ranked)
            details.append(detail)
            if details_handle:
                write_jsonl_row(details_handle, detail)
            evaluated += 1
            sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
        pending_queries = []

    try:
        if details_path:
            ensure_parent(details_path)
            details_handle = details_path.open("a" if resume_details and details_path.exists() else "w", encoding="utf-8")
        for sample in samples:
            sample_id = str(sample.get("id")) if sample.get("id") else ""
            if sample_id and sample_id in resumed_ids:
                flush_pending_queries()
                sample_bar.update(suffix=f"evaluated={evaluated} resumed={len(resumed_ids)} skipped={sum(skipped.values())}")
                continue
            gold_files = target_gold_files(sample)
            if not gold_files:
                flush_pending_queries()
                skipped["no_gold"] += 1
                sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
                continue
            query_text = query_prefix + query_text_for_eval(sample)
            if query_has_leakage(sample, query_text):
                flush_pending_queries()
                skipped["query_leakage"] += 1
                sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
                continue
            key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
            chunks_path = manifest.get(key)
            if not chunks_path:
                flush_pending_queries()
                skipped["missing_corpus"] += 1
                sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
                continue
            filtered_key = (key[0], key[1], candidate_filter)
            chunks = chunk_cache.get(filtered_key)
            if chunks is None:
                reporter.message(f"loading chunks: {sample.get('repo')} {str(sample.get('base_commit', ''))[:12]}")
                chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
                chunk_cache[filtered_key] = chunks
                reporter.message(f"loaded candidate chunks: {len(chunks)} from {chunks_path}")
            if not chunks:
                flush_pending_queries()
                skipped["empty_corpus"] += 1
                sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
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
            pending_queries.append(
                {
                    "sample": sample,
                    "gold_files": gold_files,
                    "query_text": query_text,
                    "chunks": chunks,
                    "vectors": vectors,
                }
            )
            if len(pending_queries) >= effective_query_batch_size:
                flush_pending_queries()
        flush_pending_queries()
        sample_bar.finish(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
    finally:
        if details_handle:
            details_handle.close()
        if shared_cache is not None:
            shared_cache.close()

    summary = summarize_details(details)
    result = {
        "mode": "embedding",
        "model": model_name,
        "candidate_filter": candidate_filter,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "shared_text_cache": str(shared_text_cache) if shared_text_cache else None,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "selection": selection,
        "evaluated": evaluated,
        "skipped": dict(skipped),
        "metrics": summary,
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
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def select_samples_for_embedding_run(
    samples: list[dict[str, Any]],
    sample_ids: set[str] | None = None,
    shard_count: int | None = None,
    shard_index: int | None = None,
    limit_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_count = len(samples)
    selected = samples
    missing_sample_ids: list[str] = []
    if sample_ids is not None:
        seen_ids = {str(sample.get("id")) for sample in selected if sample.get("id")}
        missing_sample_ids = sorted(sample_ids - seen_ids)
        selected = [sample for sample in selected if str(sample.get("id")) in sample_ids]
    if shard_count is not None or shard_index is not None:
        if shard_count is None or shard_index is None:
            raise ValueError("shard_count and shard_index must be provided together.")
        if shard_count <= 0:
            raise ValueError("shard_count must be positive.")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")
        selected = [sample for index, sample in enumerate(selected) if index % shard_count == shard_index]
    if limit_samples:
        selected = selected[:limit_samples]
    return selected, {
        "input_samples": original_count,
        "sample_id_filter_count": len(sample_ids) if sample_ids is not None else None,
        "missing_sample_ids": missing_sample_ids[:20],
        "missing_sample_ids_count": len(missing_sample_ids),
        "missing_sample_ids_truncated": len(missing_sample_ids) > 20,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "limit_samples": limit_samples,
        "selected_samples": len(selected),
    }


def load_sample_id_file(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                row = json.loads(stripped)
                sample_id = row.get("sample_id") or row.get("id")
                if sample_id:
                    ids.add(str(sample_id))
            else:
                ids.add(stripped)
    return ids


def load_or_encode_chunk_vectors(
    chunks: list[dict[str, Any]],
    chunks_path: Path,
    embedder: TextEmbedder,
    model_name: str,
    cache_dir: Path | None,
    shared_text_cache: "TextEmbeddingCache | None" = None,
    batch_size: int = 32,
    passage_prefix: str = "",
    input_type: str | None = None,
    normalize_embeddings: bool = True,
    embedding_options: dict[str, Any] | None = None,
    candidate_filter: str = "all_files",
    progress_reporter: "ProgressReporter | None" = None,
) -> Any:
    reporter = progress_reporter or ProgressReporter(False)
    if cache_dir is None:
        reporter.message(f"encoding chunks without cache: {len(chunks)} chunks")
        return encode_texts(
            embedder,
            chunk_texts_for_embedding(chunks, passage_prefix=passage_prefix),
            batch_size=batch_size,
            show_progress_bar=reporter.enabled,
            input_type=input_type,
        )

    np = import_numpy()
    embedding_options = embedding_options or {}
    repo = str(chunks[0].get("repo", "")) if chunks else "unknown"
    base_commit = str(chunks[0].get("base_commit", "")) if chunks else chunks_path.stem.split(".")[0]
    pair_dir = cache_dir / repo_slug(repo)
    cache_stem = f"{base_commit}.embeddings" if candidate_filter == "all_files" else f"{base_commit}.{candidate_filter}.embeddings"
    vectors_path = pair_dir / f"{cache_stem}.npy"
    meta_path = pair_dir / f"{cache_stem}.meta.json"
    if vectors_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if chunk_embedding_cache_meta_matches(
                meta=meta,
                model_name=model_name,
                chunk_count=len(chunks),
                chunks_path=chunks_path,
                passage_prefix=passage_prefix,
                normalize_embeddings=normalize_embeddings,
                embedding_options=embedding_options,
                candidate_filter=candidate_filter,
            ):
                vectors = np.load(vectors_path)
                if is_valid_chunk_vector_cache(vectors, chunk_count=len(chunks), meta=meta):
                    reporter.message(f"embedding cache hit: {vectors_path}")
                    return vectors
                reporter.message(f"embedding cache invalid shape, rebuilding: {vectors_path}")
        except Exception as exc:
            reporter.message(f"embedding cache unreadable, rebuilding: {vectors_path} ({exc})")

    texts = chunk_texts_for_embedding(chunks, passage_prefix=passage_prefix)
    reporter.message(f"encoding chunks: {len(texts)} chunks -> {vectors_path}")
    if shared_text_cache is not None:
        vectors = encode_texts_with_shared_cache(
            embedder=embedder,
            texts=texts,
            shared_text_cache=shared_text_cache,
            batch_size=batch_size,
            show_progress_bar=reporter.enabled,
            input_type=input_type,
            reporter=reporter,
        )
    else:
        vectors = np.asarray(
            encode_texts(
                embedder,
                texts,
                batch_size=batch_size,
                show_progress_bar=reporter.enabled,
                input_type=input_type,
            ),
            dtype="float32",
        )
    write_chunk_embedding_cache(
        vectors_path=vectors_path,
        meta_path=meta_path,
        vectors=vectors,
        meta={
            "model": model_name,
            "chunk_count": len(chunks),
            "chunks_path": str(chunks_path),
            "embedding_dim": int(vectors.shape[1]) if len(vectors.shape) == 2 else 0,
            "normalize_embeddings": normalize_embeddings,
            "passage_prefix": passage_prefix,
            "embedding_options": embedding_options,
            "candidate_filter": candidate_filter,
        },
    )
    return vectors


def chunk_embedding_cache_meta_matches(
    *,
    meta: dict[str, Any],
    model_name: str,
    chunk_count: int,
    chunks_path: Path,
    passage_prefix: str,
    normalize_embeddings: bool,
    embedding_options: dict[str, Any],
    candidate_filter: str,
) -> bool:
    return (
        meta.get("model") == model_name
        and meta.get("chunk_count") == chunk_count
        and meta.get("chunks_path") == str(chunks_path)
        and meta.get("passage_prefix") == passage_prefix
        and meta.get("normalize_embeddings") == normalize_embeddings
        and meta.get("embedding_options", {}) == embedding_options
        and meta.get("candidate_filter", "all_files") == candidate_filter
    )


def is_valid_chunk_vector_cache(vectors: Any, *, chunk_count: int, meta: dict[str, Any]) -> bool:
    shape = getattr(vectors, "shape", ())
    if len(shape) != 2 or int(shape[0]) != chunk_count:
        return False
    embedding_dim = int(meta.get("embedding_dim") or 0)
    return embedding_dim <= 0 or int(shape[1]) == embedding_dim


def write_chunk_embedding_cache(
    *,
    vectors_path: Path,
    meta_path: Path,
    vectors: Any,
    meta: dict[str, Any],
) -> None:
    ensure_parent(vectors_path)
    unique = f"{os.getpid()}.{time.monotonic_ns()}"
    vectors_tmp = vectors_path.with_name(f".{vectors_path.name}.{unique}.tmp.npy")
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{unique}.tmp")
    try:
        np = import_numpy()
        np.save(vectors_tmp, vectors)
        meta_tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(vectors_tmp, vectors_path)
        os.replace(meta_tmp, meta_path)
    finally:
        for path in (vectors_tmp, meta_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class TextEmbeddingCache:
    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = path
        self.metadata = metadata
        ensure_parent(path)
        self.connection = sqlite3.connect(path, timeout=60.0)
        self._closed = False
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA busy_timeout=60000")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (digest BLOB PRIMARY KEY, vector BLOB NOT NULL, dim INTEGER NOT NULL)"
            )
            self._validate_metadata()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "TextEmbeddingCache":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def _validate_metadata(self) -> None:
        expected = {key: json.dumps(value, sort_keys=True) for key, value in self.metadata.items()}
        rows = dict(self.connection.execute("SELECT key, value FROM metadata").fetchall())
        if not rows:
            self.connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()
            return
        mismatches = {
            key: {"expected": value, "actual": rows.get(key)}
            for key, value in expected.items()
            if rows.get(key) != value
        }
        extra = sorted(set(rows) - set(expected))
        if mismatches or extra:
            raise RuntimeError(
                f"Shared embedding text cache metadata mismatch for {self.path}: "
                f"mismatches={mismatches}, extra_keys={extra}"
            )

    def get_many(self, digests: Sequence[bytes]) -> dict[bytes, Any]:
        if not digests:
            return {}
        np = import_numpy()
        hits: dict[bytes, Any] = {}
        for start in range(0, len(digests), SHARED_TEXT_CACHE_GET_BATCH_SIZE):
            batch = list(digests[start : start + SHARED_TEXT_CACHE_GET_BATCH_SIZE])
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT digest, vector, dim FROM embeddings WHERE digest IN ({placeholders})",
                [sqlite3.Binary(digest) for digest in batch],
            )
            for digest, vector_bytes, dim in rows:
                hits[bytes(digest)] = np.frombuffer(vector_bytes, dtype="float32", count=int(dim)).copy()
        return hits

    def put_many(self, rows: Iterable[tuple[bytes, Sequence[float]]]) -> int:
        np = import_numpy()
        encoded_rows = []
        for digest, vector in rows:
            array = np.asarray(vector, dtype="float32")
            encoded_rows.append((sqlite3.Binary(digest), sqlite3.Binary(array.tobytes()), int(array.shape[0])))
        if not encoded_rows:
            return 0
        self.connection.executemany(
            "INSERT OR IGNORE INTO embeddings (digest, vector, dim) VALUES (?, ?, ?)",
            encoded_rows,
        )
        self.connection.commit()
        return len(encoded_rows)


def encode_texts_with_shared_cache(
    embedder: TextEmbedder,
    texts: Sequence[str],
    shared_text_cache: TextEmbeddingCache,
    batch_size: int = 32,
    show_progress_bar: bool = False,
    input_type: str | None = None,
    reporter: "ProgressReporter | None" = None,
) -> Any:
    np = import_numpy()
    digests = [embedding_text_digest(text) for text in texts]
    text_by_digest: dict[bytes, str] = {}
    unique_digests: list[bytes] = []
    for digest, text in zip(digests, texts):
        if digest not in text_by_digest:
            text_by_digest[digest] = text
            unique_digests.append(digest)
    cached = shared_text_cache.get_many(unique_digests)
    missing = [digest for digest in unique_digests if digest not in cached]
    if reporter:
        reporter.message(
            f"shared embedding text cache: unique={len(unique_digests)} hit={len(cached)} miss={len(missing)}"
        )
    if missing:
        encode_window_size = shared_text_cache_encode_window_size(batch_size)
        if reporter:
            reporter.message(
                f"encoding shared text cache misses: {len(missing)} texts, encode_window={encode_window_size}, model_batch_size={batch_size}"
            )
        for start in range(0, len(missing), encode_window_size):
            batch_digests = missing[start : start + encode_window_size]
            encoded = np.asarray(
                encode_texts(
                    embedder,
                    [text_by_digest[digest] for digest in batch_digests],
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    input_type=input_type,
                ),
                dtype="float32",
            )
            for put_start in range(0, len(batch_digests), SHARED_TEXT_CACHE_PUT_BATCH_SIZE):
                shared_text_cache.put_many(
                    zip(
                        batch_digests[put_start : put_start + SHARED_TEXT_CACHE_PUT_BATCH_SIZE],
                        encoded[put_start : put_start + SHARED_TEXT_CACHE_PUT_BATCH_SIZE],
                    )
                )
            for digest, vector in zip(batch_digests, encoded):
                cached[digest] = vector
    return np.asarray([cached[digest] for digest in digests], dtype="float32")


def shared_text_cache_encode_window_size(batch_size: int) -> int:
    model_batch_size = max(1, batch_size)
    return min(
        SHARED_TEXT_CACHE_ENCODE_WINDOW_MAX,
        max(SHARED_TEXT_CACHE_ENCODE_WINDOW_MIN, model_batch_size * SHARED_TEXT_CACHE_ENCODE_WINDOW_BATCH_MULTIPLIER),
    )


def embedding_text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def encode_texts(
    embedder: TextEmbedder,
    texts: Sequence[str],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    input_type: str | None = None,
) -> Sequence[Sequence[float]]:
    previous_progress = getattr(embedder, "show_progress_bar", None)
    if previous_progress is not None:
        setattr(embedder, "show_progress_bar", show_progress_bar)
    try:
        if input_type is not None:
            try:
                return embedder.encode(texts, batch_size=batch_size, input_type=input_type)  # type: ignore[call-arg]
            except TypeError:
                return embedder.encode(texts, batch_size=batch_size)
        try:
            return embedder.encode(texts, batch_size=batch_size)
        except TypeError:
            return embedder.encode(texts)  # type: ignore[call-arg]
    finally:
        if previous_progress is not None:
            setattr(embedder, "show_progress_bar", previous_progress)


def encode_query_texts_resilient(
    embedder: TextEmbedder,
    texts: Sequence[str],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    input_type: str | None = None,
    reporter: "ProgressReporter | None" = None,
) -> Sequence[Sequence[float]]:
    try:
        return encode_texts(
            embedder,
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            input_type=input_type,
        )
    except Exception as exc:
        if len(texts) <= 1 or not is_embedding_out_of_memory_error(exc):
            raise
        clear_accelerator_memory()
        midpoint = len(texts) // 2
        if reporter:
            reporter.message(
                f"query embedding batch hit OOM; retrying as {midpoint} + {len(texts) - midpoint} texts"
            )
        left = encode_query_texts_resilient(
            embedder,
            texts[:midpoint],
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            input_type=input_type,
            reporter=reporter,
        )
        right = encode_query_texts_resilient(
            embedder,
            texts[midpoint:],
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            input_type=input_type,
            reporter=reporter,
        )
        return [*left, *right]


def is_embedding_out_of_memory_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "outofmemory",
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "mps backend out of memory",
        )
    )


def clear_accelerator_memory() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        return


def rank_chunks_by_vectors(query_vector: Sequence[float], chunk_vectors: Any, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = vector_scores(query_vector, chunk_vectors)
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for score, chunk in zip(scores, chunks):
        scored.append((float(score), str(chunk.get("path", "")), str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def vector_scores(query_vector: Sequence[float], chunk_vectors: Any) -> list[float]:
    try:
        np = import_numpy()
        chunk_array = np.asarray(chunk_vectors, dtype="float32")
        query_array = np.asarray(query_vector, dtype="float32")
        if chunk_array.ndim == 2:
            return [float(value) for value in chunk_array @ query_array]
    except ImportError:
        pass
    return [dot_product(query_vector, vector) for vector in chunk_vectors]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def embedding_cache_metadata(embedder: TextEmbedder, input_type: str | None = None) -> dict[str, Any]:
    metadata_func = getattr(embedder, "cache_metadata", None)
    metadata = dict(metadata_func()) if callable(metadata_func) else {}
    if input_type is not None:
        metadata["input_type"] = input_type
    return metadata


def chunk_texts_for_embedding(chunks: list[dict[str, Any]], passage_prefix: str = "") -> list[str]:
    return [passage_prefix + chunk_text_for_embedding(chunk) for chunk in chunks]


def chunk_text_for_embedding(chunk: dict[str, Any]) -> str:
    parts = [
        f"path: {chunk.get('path', '')}",
        f"kind: {chunk.get('kind', '')}",
    ]
    if chunk.get("symbol"):
        parts.append(f"symbol: {chunk.get('symbol')}")
    parts.append("content:")
    parts.append(str(chunk.get("text", "")))
    return "\n".join(parts)


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    return slug.strip("-") or "embedding-model"


def default_embedding_summary_path(
    model_name: str,
    root: Path = Path("data/eval/v0_1"),
    candidate_filter: str = "all_files",
) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"{model_slug(model_name)}{suffix}.json"


def default_embedding_cache_dir(model_name: str, root: Path = Path("data/embeddings/v0_1")) -> Path:
    return root / model_slug(model_name)


def directory_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Embedding cache/ranking requires numpy. Install with: pip install -e '.[embedding]'"
        ) from exc
    return np


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_row(handle, row)
            count += 1
    return count


def write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def load_resumable_details(
    details_path: Path,
    sample_ids: set[str],
    candidate_filter: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    details = read_jsonl(details_path)
    resumed_ids: set[str] = set()
    for row in details:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError(f"Cannot resume details with missing sample_id: {details_path}")
        if sample_id in resumed_ids:
            raise RuntimeError(f"Cannot resume details with duplicate sample_id {sample_id}: {details_path}")
        if sample_id not in sample_ids:
            raise RuntimeError(f"Cannot resume details for sample not in current run {sample_id}: {details_path}")
        row_filter = str(row.get("candidate_filter") or "")
        if row_filter != candidate_filter:
            raise RuntimeError(
                f"Cannot resume details with candidate_filter={row_filter!r}; expected {candidate_filter!r}: {details_path}"
            )
        resumed_ids.add(sample_id)
    return details, resumed_ids
