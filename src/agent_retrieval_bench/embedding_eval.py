from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, TextIO

from .baseline import (
    filter_candidate_chunks,
    gold_file_ranks,
    iter_samples,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    sample_metrics,
    summarize_details,
    target_gold_files,
    unique_ranked_paths,
    validate_candidate_filter,
)
from .curate import filter_samples, load_keep_ids
from .io import ensure_parent, read_jsonl, repo_slug


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
        effective_batch_size = max(1, min(batch_size, 1000))
        for start in range(0, len(texts), effective_batch_size):
            batch = list(texts[start : start + effective_batch_size])
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


def evaluate_embedding_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    model_name: str,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    cache_dir: Path | None = None,
    limit_samples: int | None = None,
    batch_size: int = 32,
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
    validate_candidate_filter(candidate_filter)
    reporter = ProgressReporter(progress, progress_stream)
    reporter.message(f"loading embedding model: {model_name}")
    actual_embedder = embedder or SentenceTransformerEmbedder(
        model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        trust_remote_code=trust_remote_code,
    )
    reporter.message("embedding model loaded")
    reporter.message(f"loading corpus manifest: {corpus_dir / 'corpus_manifest.jsonl'}")
    manifest = load_corpus_manifest(corpus_dir)
    reporter.message(f"loaded corpus manifest: {len(manifest)} commit corpora")
    keep_ids = load_keep_ids(keep_list)
    if keep_list:
        if keep_ids is None:
            reporter.message(f"keep list not found, evaluating all samples: {keep_list}")
        else:
            reporter.message(f"loaded keep list: {len(keep_ids)} ids")
    samples = list(filter_samples(iter_samples(sample_paths), keep_ids))
    if limit_samples:
        samples = samples[:limit_samples]
    reporter.message(f"loaded benchmark samples: {len(samples)}")
    vector_cache: dict[tuple[str, str], Any] = {}
    chunk_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    skipped = Counter()
    evaluated = 0
    sample_bar = reporter.bar("evaluating samples", len(samples))
    for sample in samples:
        gold_files = target_gold_files(sample)
        if not gold_files:
            skipped["no_gold"] += 1
            sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
            continue
        query_text = query_prefix + query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
            continue
        key = (sample.get("repo"), sample.get("base_commit"))
        chunks_path = manifest.get(key)
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
            continue
        chunks = chunk_cache.get(key)
        if chunks is None:
            reporter.message(f"loading chunks: {sample.get('repo')} {str(sample.get('base_commit', ''))[:12]}")
            chunks = read_jsonl(chunks_path)
            chunk_cache[key] = chunks
            reporter.message(f"loaded chunks: {len(chunks)} from {chunks_path}")
        chunks = filter_candidate_chunks(chunks, candidate_filter)
        if not chunks:
            skipped["empty_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
            continue
        vectors = vector_cache.get(key)
        if vectors is None:
            vectors = load_or_encode_chunk_vectors(
                chunks=chunks,
                chunks_path=chunks_path,
                embedder=actual_embedder,
                model_name=model_name,
                cache_dir=cache_dir,
                batch_size=batch_size,
                passage_prefix=passage_prefix,
                input_type=passage_input_type,
                normalize_embeddings=normalize_embeddings,
                embedding_options=embedding_cache_metadata(actual_embedder, input_type=passage_input_type),
                candidate_filter=candidate_filter,
                progress_reporter=reporter,
            )
            vector_cache[key] = vectors
        query_vector = encode_texts(
            actual_embedder,
            [query_text],
            batch_size=batch_size,
            show_progress_bar=False,
            input_type=query_input_type,
        )[0]
        ranked = rank_chunks_by_vectors(query_vector, vectors, chunks)
        metrics = sample_metrics(gold_files, ranked)
        details.append(
            {
                "sample_id": sample.get("id"),
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "candidate_filter": candidate_filter,
                "gold_files": gold_files,
                "gold_ranks": gold_file_ranks(gold_files, ranked),
                "top_files": unique_ranked_paths(ranked)[:20],
                "metrics": metrics,
            }
        )
        evaluated += 1
        sample_bar.update(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"evaluated={evaluated} skipped={sum(skipped.values())}")

    summary = summarize_details(details)
    result = {
        "mode": "embedding",
        "model": model_name,
        "candidate_filter": candidate_filter,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": evaluated,
        "skipped": dict(skipped),
        "metrics": summary,
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if details_path:
        write_jsonl(details_path, details)
    return result


def load_or_encode_chunk_vectors(
    chunks: list[dict[str, Any]],
    chunks_path: Path,
    embedder: TextEmbedder,
    model_name: str,
    cache_dir: Path | None,
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
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model") == model_name
            and meta.get("chunk_count") == len(chunks)
            and meta.get("chunks_path") == str(chunks_path)
            and meta.get("passage_prefix") == passage_prefix
            and meta.get("normalize_embeddings") == normalize_embeddings
            and meta.get("embedding_options", {}) == embedding_options
            and meta.get("candidate_filter", "all_files") == candidate_filter
        ):
            reporter.message(f"embedding cache hit: {vectors_path}")
            return np.load(vectors_path)

    texts = chunk_texts_for_embedding(chunks, passage_prefix=passage_prefix)
    reporter.message(f"encoding chunks: {len(texts)} chunks -> {vectors_path}")
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
    ensure_parent(vectors_path)
    np.save(vectors_path, vectors)
    meta_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "chunk_count": len(chunks),
                "chunks_path": str(chunks_path),
                "embedding_dim": int(vectors.shape[1]) if len(vectors.shape) == 2 else 0,
                "normalize_embeddings": normalize_embeddings,
                "passage_prefix": passage_prefix,
                "embedding_options": embedding_options,
                "candidate_filter": candidate_filter,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return vectors


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


class ProgressReporter:
    def __init__(self, enabled: bool = False, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.line_open = False

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        if self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False
        print(f"[arb] {text}", file=self.stream, flush=True)

    def bar(self, label: str, total: int) -> "ProgressBar":
        return ProgressBar(label=label, total=total, reporter=self)


class ProgressBar:
    def __init__(self, label: str, total: int, reporter: ProgressReporter) -> None:
        self.label = label
        self.total = max(0, total)
        self.reporter = reporter
        self.current = 0
        self.rendered = False
        if self.reporter.enabled:
            self._render()

    def update(self, step: int = 1, suffix: str = "") -> None:
        if not self.reporter.enabled:
            return
        self.current = min(self.total, self.current + step)
        self._render(suffix=suffix)

    def finish(self, suffix: str = "") -> None:
        if not self.reporter.enabled or not self.rendered:
            return
        self.current = self.total
        self._render(suffix=suffix)
        print(file=self.reporter.stream, flush=True)
        self.reporter.line_open = False

    def _render(self, suffix: str = "") -> None:
        width = 28
        if self.total:
            filled = int(width * self.current / self.total)
            percent = int(100 * self.current / self.total)
        else:
            filled = width
            percent = 100
        bar = "#" * filled + "-" * (width - filled)
        suffix_text = f" {suffix}" if suffix else ""
        print(
            f"\r[arb] {self.label}: [{bar}] {self.current}/{self.total} {percent:3d}%{suffix_text}",
            end="",
            file=self.reporter.stream,
            flush=True,
        )
        self.reporter.line_open = True
        self.rendered = True


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
