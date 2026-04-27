from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .baseline import (
    iter_samples,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    sample_metrics,
    summarize_details,
    target_gold_files,
    unique_ranked_paths,
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
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )


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
    normalize_embeddings: bool = True,
    trust_remote_code: bool = False,
    embedder: TextEmbedder | None = None,
) -> dict[str, Any]:
    actual_embedder = embedder or SentenceTransformerEmbedder(
        model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        trust_remote_code=trust_remote_code,
    )
    manifest = load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
    vector_cache: dict[tuple[str, str], Any] = {}
    chunk_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    skipped = Counter()
    evaluated = 0
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        if limit_samples and evaluated + sum(skipped.values()) >= limit_samples:
            break
        gold_files = target_gold_files(sample)
        if not gold_files:
            skipped["no_gold"] += 1
            continue
        query_text = query_prefix + query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            continue
        key = (sample.get("repo"), sample.get("base_commit"))
        chunks_path = manifest.get(key)
        if not chunks_path:
            skipped["missing_corpus"] += 1
            continue
        chunks = chunk_cache.get(key)
        if chunks is None:
            chunks = read_jsonl(chunks_path)
            chunk_cache[key] = chunks
        if not chunks:
            skipped["empty_corpus"] += 1
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
                normalize_embeddings=normalize_embeddings,
            )
            vector_cache[key] = vectors
        query_vector = encode_texts(actual_embedder, [query_text], batch_size=batch_size)[0]
        ranked = rank_chunks_by_vectors(query_vector, vectors, chunks)
        metrics = sample_metrics(gold_files, ranked)
        details.append(
            {
                "sample_id": sample.get("id"),
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "gold_files": gold_files,
                "top_files": unique_ranked_paths(ranked)[:20],
                "metrics": metrics,
            }
        )
        evaluated += 1

    summary = summarize_details(details)
    result = {
        "mode": "embedding",
        "model": model_name,
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
    normalize_embeddings: bool = True,
) -> Any:
    if cache_dir is None:
        return encode_texts(
            embedder,
            chunk_texts_for_embedding(chunks, passage_prefix=passage_prefix),
            batch_size=batch_size,
        )

    np = import_numpy()
    repo = str(chunks[0].get("repo", "")) if chunks else "unknown"
    base_commit = str(chunks[0].get("base_commit", "")) if chunks else chunks_path.stem.split(".")[0]
    pair_dir = cache_dir / repo_slug(repo)
    vectors_path = pair_dir / f"{base_commit}.embeddings.npy"
    meta_path = pair_dir / f"{base_commit}.embeddings.meta.json"
    if vectors_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model") == model_name
            and meta.get("chunk_count") == len(chunks)
            and meta.get("chunks_path") == str(chunks_path)
            and meta.get("passage_prefix") == passage_prefix
            and meta.get("normalize_embeddings") == normalize_embeddings
        ):
            return np.load(vectors_path)

    texts = chunk_texts_for_embedding(chunks, passage_prefix=passage_prefix)
    vectors = np.asarray(encode_texts(embedder, texts, batch_size=batch_size), dtype="float32")
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
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return vectors


def encode_texts(embedder: TextEmbedder, texts: Sequence[str], batch_size: int = 32) -> Sequence[Sequence[float]]:
    try:
        return embedder.encode(texts, batch_size=batch_size)
    except TypeError:
        return embedder.encode(texts)  # type: ignore[call-arg]


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


def default_embedding_summary_path(model_name: str, root: Path = Path("data/eval/v0_1")) -> Path:
    return root / f"{model_slug(model_name)}_summary.json"


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
