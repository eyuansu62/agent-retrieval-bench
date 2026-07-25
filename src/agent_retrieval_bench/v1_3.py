from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .baseline import (
    given_files,
    gold_blocks,
    gold_spans,
    hard_negative_files,
    query_provenance,
    query_text_for_eval,
    target_gold_files,
)
from .corpus import sample_paths_from_derived
from .io import ensure_parent, read_json, read_jsonl, truncate_text, utc_now, write_json, write_jsonl
from .quality import validate_sample
from .v1_2 import (
    load_corpus_file_bounds,
    raw_gold_spans,
    render_v1_2_validation_markdown,
    validate_gold_spans,
    validate_hard_negatives,
    write_split_benchmark,
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+")
STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "any",
    "are",
    "but",
    "can",
    "could",
    "does",
    "done",
    "for",
    "from",
    "get",
    "has",
    "have",
    "here",
    "how",
    "into",
    "its",
    "let",
    "like",
    "make",
    "more",
    "move",
    "none",
    "not",
    "now",
    "one",
    "only",
    "other",
    "our",
    "out",
    "should",
    "some",
    "that",
    "the",
    "then",
    "this",
    "too",
    "top",
    "use",
    "via",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "won",
    "would",
}


def derive_v1_3_blocks(
    base_derived: Path,
    corpus_manifest_path: Path,
    out_dir: Path,
    report_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    samples = [row for path in sample_paths_from_derived(base_derived) for row in read_jsonl(path)]
    corpus_paths = load_corpus_manifest_paths(corpus_manifest_path)
    chunks_cache: dict[Path, list[dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        spans = gold_spans(sample)
        existing_blocks = gold_blocks(sample)
        derived_blocks: list[dict[str, Any]] = existing_blocks
        source = "existing"
        if spans and not existing_blocks:
            key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
            chunks_path = corpus_paths.get(key)
            chunks: list[dict[str, Any]] = []
            if chunks_path:
                chunks = chunks_cache.setdefault(chunks_path, read_jsonl(chunks_path))
            derived_blocks = derive_blocks_for_spans(spans, chunks)
            source = "derived"
        merged = json.loads(json.dumps(sample))
        if derived_blocks:
            merged["gold_blocks"] = derived_blocks
        output.append(merged)
        rows.append(
            {
                "sample_id": sample_id,
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "gold_span_count": len(spans),
                "gold_block_count": len(derived_blocks),
                "source": source if derived_blocks else "none",
            }
        )
    write_split_benchmark(out_dir, output)
    manifest = write_v1_3_manifest(base_derived, corpus_manifest_path, out_dir, output)
    report = {
        "generated_at": utc_now(),
        "base_derived": str(base_derived),
        "corpus_manifest": str(corpus_manifest_path),
        "out_dir": str(out_dir),
        "samples": len(output),
        "span_samples": sum(1 for sample in output if gold_spans(sample)),
        "block_samples": sum(1 for sample in output if gold_blocks(sample)),
        "manifest": manifest,
        "rows": rows,
    }
    if report_path:
        write_json(report_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_3_block_derivation_markdown(report), encoding="utf-8")
    return report


def derive_blocks_for_spans(spans: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        path = str(chunk.get("path") or "")
        if path and str(chunk.get("kind") or "") != "file":
            chunks_by_path[path].append(chunk)
    blocks: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    for span in spans:
        path = str(span.get("path") or "")
        start_line = int(span.get("start_line") or 0)
        end_line = int(span.get("end_line") or 0)
        matches = [
            chunk
            for chunk in chunks_by_path.get(path, [])
            if line_ranges_overlap(
                start_line,
                end_line,
                int(chunk.get("start_line") or 0),
                int(chunk.get("end_line") or 0),
            )
        ]
        if not matches:
            fallback = {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "kind": "span_fallback",
                "symbol": "",
                "source": "gold_span_fallback",
                "reason": str(span.get("reason") or "derived from gold span"),
            }
            key = block_key(fallback)
            if key not in seen:
                blocks.append(fallback)
                seen.add(key)
            continue
        for chunk in sorted(matches, key=lambda item: (int(item.get("start_line") or 0), int(item.get("end_line") or 0), str(item.get("symbol") or ""))):
            block = {
                "path": path,
                "start_line": int(chunk.get("start_line") or 0),
                "end_line": int(chunk.get("end_line") or 0),
                "kind": str(chunk.get("kind") or "block"),
                "symbol": str(chunk.get("symbol") or ""),
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "source": "corpus_symbol_overlap",
                "reason": str(span.get("reason") or "derived from gold span"),
            }
            key = block_key(block)
            if key not in seen:
                blocks.append(block)
                seen.add(key)
    return blocks


def write_v1_3_span_worklist(
    derived: Path,
    corpus_manifest_path: Path,
    out_path: Path,
    markdown_out_path: Path | None = None,
    jsonl_out_path: Path | None = None,
    task_type: str = "comment2context",
    max_candidates_per_file: int = 3,
) -> dict[str, Any]:
    samples = [row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)]
    corpus_paths = load_corpus_manifest_paths(corpus_manifest_path)
    chunks_cache: dict[Path, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for sample in samples:
        if task_type and str(sample.get("task_type") or "") != task_type:
            continue
        if gold_spans(sample):
            continue
        sample_id = str(sample.get("id") or "")
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        chunks_path = corpus_paths.get(key)
        chunks = chunks_cache.setdefault(chunks_path, read_jsonl(chunks_path)) if chunks_path else []
        target_files = target_gold_files(sample)
        candidate_spans = suggest_gold_span_candidates(sample, chunks, target_files, max_candidates_per_file=max_candidates_per_file)
        row = {
            "sample_id": sample_id,
            "task_type": sample.get("task_type"),
            "repo": sample.get("repo"),
            "base_commit": sample.get("base_commit"),
            "query_provenance": query_provenance(sample),
            "given_files": given_files(sample),
            "target_gold_files": target_files,
            "corpus_chunks_path": str(chunks_path) if chunks_path else None,
            "review_required": True,
            "candidate_span_count": len(candidate_spans),
            "candidate_spans": candidate_spans,
            "query": worklist_query_payload(sample),
            "existing_hard_negative_files": hard_negative_files(sample),
            "annotation_note": "V1.3 span candidate worklist; candidates require human review before merging as gold.",
        }
        rows.append(row)
        suggestions.append(
            {
                "sample_id": sample_id,
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "query_provenance": query_provenance(sample),
                "target_gold_files": target_files,
                "candidate_gold_spans": candidate_spans,
                "hard_negative_files": hard_negative_files(sample),
                "review_required": True,
                "annotation_note": "V1.3 candidate spans only; do not merge without manual review.",
            }
        )
    report = {
        "generated_at": utc_now(),
        "derived": str(derived),
        "corpus_manifest": str(corpus_manifest_path),
        "task_type": task_type,
        "max_candidates_per_file": max_candidates_per_file,
        "missing_span_samples": len(rows),
        "with_candidates": sum(1 for row in rows if row["candidate_span_count"]),
        "without_candidates": [row["sample_id"] for row in rows if not row["candidate_span_count"]],
        "rows": rows,
    }
    write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_3_span_worklist_markdown(report), encoding="utf-8")
    if jsonl_out_path:
        write_jsonl(jsonl_out_path, suggestions)
    return report


def suggest_gold_span_candidates(
    sample: dict[str, Any],
    chunks: list[dict[str, Any]],
    target_files: list[str],
    max_candidates_per_file: int = 3,
) -> list[dict[str, Any]]:
    query_text = worklist_query_text(sample)
    query_tokens = tokenize_for_similarity(query_text)
    query = sample.get("query") or {}
    query_path = str(query.get("path") or query.get("given_file") or "")
    try:
        query_line = int(query.get("line") or 0)
    except (TypeError, ValueError):
        query_line = 0
    chunks_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_chunks_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        path = str(chunk.get("path") or "")
        if not path:
            continue
        if str(chunk.get("kind") or "") == "file":
            file_chunks_by_path[path].append(chunk)
        else:
            chunks_by_path[path].append(chunk)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    for path in target_files:
        path_chunks = chunks_by_path.get(path) or file_chunks_by_path.get(path) or []
        scored = [
            (score_span_candidate(chunk, query_tokens, query_text, query_path, query_line), chunk)
            for chunk in path_chunks
            if chunk_has_line_range(chunk)
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                int(item[1].get("end_line") or 0) - int(item[1].get("start_line") or 0),
                int(item[1].get("start_line") or 0),
                str(item[1].get("symbol") or ""),
            )
        )
        for score, chunk in scored[: max(1, max_candidates_per_file)]:
            candidate = span_candidate_from_chunk(chunk, score)
            key = block_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def score_span_candidate(
    chunk: dict[str, Any],
    query_tokens: set[str],
    query_text: str,
    query_path: str,
    query_line: int,
) -> float:
    symbol = str(chunk.get("symbol") or "")
    path = str(chunk.get("path") or "")
    chunk_text = "\n".join([path, symbol, str(chunk.get("text") or "")])
    chunk_tokens = tokenize_for_similarity(chunk_text)
    overlap = query_tokens & chunk_tokens
    score = float(len(overlap))
    lowered_query = query_text.lower()
    lowered_symbol = symbol.lower()
    if lowered_symbol and lowered_symbol in lowered_query:
        score += 8.0
    for token in tokenize_for_similarity(symbol):
        if token in query_tokens:
            score += 2.0
    if path == query_path and query_line > 0:
        start_line = int(chunk.get("start_line") or 0)
        end_line = int(chunk.get("end_line") or 0)
        if start_line <= query_line <= end_line:
            score += 6.0
        elif start_line > 0:
            distance = min(abs(query_line - start_line), abs(query_line - end_line))
            if distance <= 20:
                score += 3.0
            elif distance <= 80:
                score += 1.0
    return round(score, 4)


def span_candidate_from_chunk(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    symbol = str(chunk.get("symbol") or "")
    kind = str(chunk.get("kind") or "block")
    candidate = {
        "path": str(chunk.get("path") or ""),
        "start_line": int(chunk.get("start_line") or 0),
        "end_line": int(chunk.get("end_line") or 0),
        "reason": "candidate selected from target file by query/corpus token overlap; human review required",
        "source": "v1_3_span_worklist_candidate",
        "kind": kind,
        "symbol": symbol,
        "score": score,
        "text_preview": truncate_text(str(chunk.get("text") or ""), 420),
    }
    if chunk.get("chunk_id"):
        candidate["chunk_id"] = str(chunk["chunk_id"])
    return candidate


def worklist_query_payload(sample: dict[str, Any]) -> dict[str, Any]:
    query = sample.get("query") or {}
    return {
        "pr_title": query.get("pr_title"),
        "review_comment": query.get("review_comment"),
        "diff_hunk_context": truncate_text(str(query.get("diff_hunk_context") or ""), 800),
        "path": query.get("path") or query.get("given_file"),
        "line": query.get("line"),
    }


def worklist_query_text(sample: dict[str, Any]) -> str:
    query = sample.get("query") or {}
    values = [
        query.get("pr_title"),
        query.get("review_comment"),
        query.get("diff_hunk_context"),
        query.get("path"),
        query.get("given_file"),
        query_text_for_eval(sample),
    ]
    return "\n".join(str(value) for value in values if value)


def tokenize_for_similarity(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in TOKEN_RE.findall(text or ""):
        raw_lower = raw.lower()
        if len(raw_lower) > 2 and raw_lower not in STOPWORDS:
            tokens.add(raw_lower)
        for part in re.split(r"[_\W]+", raw):
            part_lower = part.lower()
            if len(part_lower) > 2 and part_lower not in STOPWORDS:
                tokens.add(part_lower)
        for part in CAMEL_RE.findall(raw):
            part_lower = part.lower()
            if len(part_lower) > 2 and part_lower not in STOPWORDS:
                tokens.add(part_lower)
    return tokens


def chunk_has_line_range(chunk: dict[str, Any]) -> bool:
    try:
        start_line = int(chunk.get("start_line") or 0)
        end_line = int(chunk.get("end_line") or 0)
    except (TypeError, ValueError):
        return False
    return start_line > 0 and end_line >= start_line


def validate_v1_3_benchmark(
    derived: Path,
    corpus_manifest_path: Path | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
    require_full_spans: bool = True,
    require_full_blocks: bool = True,
) -> dict[str, Any]:
    samples = [row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)]
    required_paths = required_corpus_paths_for_v1_3(samples)
    file_bounds = load_corpus_file_bounds(corpus_manifest_path, required_paths) if corpus_manifest_path else {}
    block_index = load_corpus_block_index(corpus_manifest_path, required_paths) if corpus_manifest_path else {}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        errors = validate_sample(sample)
        gold_files = target_gold_files(sample)
        span_rows = gold_spans(sample)
        block_rows = gold_blocks(sample)
        raw_spans = raw_gold_spans(sample)
        raw_blocks = raw_gold_blocks(sample)
        hard_negatives = hard_negative_files(sample)
        if require_full_spans and not span_rows:
            errors.append("missing gold_spans for V1.3 full coverage")
        if require_full_blocks and not block_rows:
            errors.append("missing gold_blocks for V1.3 full coverage")
        if raw_spans is not None and len(span_rows) != len(raw_spans):
            errors.append("gold_spans contains malformed span rows")
        if raw_blocks is not None and len(block_rows) != len(raw_blocks):
            errors.append("gold_blocks contains malformed block rows")
        if span_rows:
            errors.extend(validate_gold_spans(span_rows, gold_files, sample, file_bounds))
        if block_rows:
            errors.extend(validate_gold_blocks(block_rows, gold_files, sample, file_bounds, block_index))
        if hard_negatives:
            errors.extend(validate_hard_negatives(hard_negatives, gold_files, sample, file_bounds))
        rows.append(
            {
                "sample_id": sample_id,
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "gold_files": gold_files,
                "gold_span_count": len(span_rows),
                "gold_block_count": len(block_rows),
                "hard_negative_count": len(hard_negatives),
                "query_provenance": query_provenance(sample),
                "errors": errors,
            }
        )
    invalid_rows = [row for row in rows if row["errors"]]
    report = {
        "generated_at": utc_now(),
        "derived": str(derived),
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "require_full_spans": require_full_spans,
        "require_full_blocks": require_full_blocks,
        "samples": len(rows),
        "invalid": len(invalid_rows),
        "ready": not invalid_rows,
        "span_samples": sum(1 for row in rows if row["gold_span_count"]),
        "block_samples": sum(1 for row in rows if row["gold_block_count"]),
        "hard_negative_samples": sum(1 for row in rows if row["hard_negative_count"]),
        "query_provenance_samples": sum(1 for row in rows if row["query_provenance"]),
        "missing_query_provenance": [row["sample_id"] for row in rows if not row["query_provenance"]][:50],
        "rows": rows,
    }
    if out_path:
        write_json(out_path, report)
    if markdown_out_path:
        ensure_parent(markdown_out_path)
        markdown_out_path.write_text(render_v1_3_validation_markdown(report), encoding="utf-8")
    return report


def validate_gold_blocks(
    blocks: list[dict[str, Any]],
    gold_files: list[str],
    sample: dict[str, Any],
    file_bounds_by_key: dict[tuple[str, str], dict[str, int]],
    block_index: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    gold = set(gold_files)
    corpus_key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
    file_bounds = file_bounds_by_key.get(corpus_key, {})
    corpus_blocks = block_index.get(corpus_key, {})
    block_ids = corpus_blocks.get("chunk_ids", set()) if isinstance(corpus_blocks, dict) else set()
    blocks_by_path = corpus_blocks.get("by_path", {}) if isinstance(corpus_blocks, dict) else {}
    if file_bounds_by_key and corpus_key not in file_bounds_by_key:
        errors.append(f"sample repo/base_commit missing from corpus manifest: {corpus_key[0]} {corpus_key[1]}")
    for index, block in enumerate(blocks):
        path = str(block.get("path") or "")
        try:
            start_line = int(block.get("start_line") or 0)
            end_line = int(block.get("end_line") or 0)
        except (TypeError, ValueError):
            errors.append(f"gold_blocks[{index}] has non-integer line range")
            continue
        if path not in gold:
            errors.append(f"gold_blocks[{index}] path is not in gold_files: {path}")
        if start_line <= 0 or end_line < start_line:
            errors.append(f"gold_blocks[{index}] has invalid line range: {start_line}-{end_line}")
        max_line = file_bounds.get(path)
        if file_bounds and path not in file_bounds:
            errors.append(f"gold_blocks[{index}] path missing from corpus: {path}")
        elif max_line and end_line > max_line:
            errors.append(f"gold_blocks[{index}] ends after corpus file: {path}:{end_line}>{max_line}")
        chunk_id = str(block.get("chunk_id") or "")
        if chunk_id and block_ids and chunk_id not in block_ids:
            errors.append(f"gold_blocks[{index}] chunk_id missing from corpus: {chunk_id}")
        kind = str(block.get("kind") or "")
        if not chunk_id and kind != "span_fallback" and blocks_by_path and not block_overlaps_corpus(block, blocks_by_path.get(path, [])):
            errors.append(f"gold_blocks[{index}] does not overlap a corpus block: {path}:{start_line}-{end_line}")
    return errors


def load_corpus_manifest_paths(corpus_manifest_path: Path | None) -> dict[tuple[str, str], Path]:
    if not corpus_manifest_path or not corpus_manifest_path.exists():
        return {}
    paths: dict[tuple[str, str], Path] = {}
    for record in read_jsonl(corpus_manifest_path):
        if record.get("status") != "ok":
            continue
        key = (str(record.get("repo") or ""), str(record.get("base_commit") or ""))
        chunks_path = Path(str(record.get("chunks_path") or ""))
        if key[0] and key[1] and chunks_path.exists():
            paths[key] = chunks_path
    return paths


def load_corpus_block_index(
    corpus_manifest_path: Path | None,
    required_paths: dict[tuple[str, str], set[str]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_paths = load_corpus_manifest_paths(corpus_manifest_path)
    required_paths = required_paths or {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, chunks_path in manifest_paths.items():
        if required_paths and key not in required_paths:
            continue
        needed = required_paths.get(key, set())
        chunk_ids: set[str] = set()
        by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in read_jsonl(chunks_path):
            path = str(chunk.get("path") or "")
            if not path or (needed and path not in needed):
                continue
            if str(chunk.get("kind") or "") == "file":
                continue
            if chunk.get("chunk_id"):
                chunk_ids.add(str(chunk["chunk_id"]))
            by_path[path].append(chunk)
        index[key] = {"chunk_ids": chunk_ids, "by_path": dict(by_path)}
    return index


def required_corpus_paths_for_v1_3(samples: Iterable[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    required: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        paths = {span["path"] for span in gold_spans(sample)}
        paths.update(block["path"] for block in gold_blocks(sample))
        paths.update(hard_negative_files(sample))
        if not paths:
            continue
        key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
        required[key].update(paths)
    return required


def write_v1_3_manifest(base_derived: Path, corpus_manifest_path: Path, out_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    base_manifest = read_json(base_derived / "manifest.json", {})
    manifest: dict[str, Any] = dict(base_manifest) if isinstance(base_manifest, dict) else {}
    counts_by_task: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts_by_task[str(sample.get("task_type") or "unknown")] += 1
    manifest.update(
        {
            "generated_at": utc_now(),
            "version": "v1_3",
            "base_derived": str(base_derived),
            "corpus_manifest": str(corpus_manifest_path),
            "total": len(samples),
            "counts_by_task": dict(sorted(counts_by_task.items())),
            "span_samples": sum(1 for sample in samples if gold_spans(sample)),
            "block_samples": sum(1 for sample in samples if gold_blocks(sample)),
            "hard_negative_samples": sum(1 for sample in samples if hard_negative_files(sample)),
            "outputs": {
                "samples": str(out_dir / "samples.jsonl"),
                "code2test": str(out_dir / "code2test.jsonl"),
                "comment2context": str(out_dir / "comment2context.jsonl"),
                "trace2code": str(out_dir / "trace2code.jsonl"),
            },
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def raw_gold_blocks(sample: dict[str, Any]) -> list[Any] | None:
    values = sample.get("gold_blocks")
    if values is None:
        values = (sample.get("gold") or {}).get("gold_blocks")
    if values is None:
        return None
    return values if isinstance(values, list) else [values]


def block_key(block: dict[str, Any]) -> tuple[str, int, int, str, str]:
    return (
        str(block.get("path") or ""),
        int(block.get("start_line") or 0),
        int(block.get("end_line") or 0),
        str(block.get("kind") or ""),
        str(block.get("symbol") or ""),
    )


def line_ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start > 0 and right_start > 0 and min(left_end, right_end) >= max(left_start, right_start)


def block_overlaps_corpus(block: dict[str, Any], corpus_blocks: list[dict[str, Any]]) -> bool:
    path = str(block.get("path") or "")
    start_line = int(block.get("start_line") or 0)
    end_line = int(block.get("end_line") or 0)
    symbol = str(block.get("symbol") or "")
    for corpus_block in corpus_blocks:
        if path != str(corpus_block.get("path") or ""):
            continue
        if symbol and str(corpus_block.get("symbol") or "") and symbol != str(corpus_block.get("symbol") or ""):
            continue
        if line_ranges_overlap(start_line, end_line, int(corpus_block.get("start_line") or 0), int(corpus_block.get("end_line") or 0)):
            return True
    return False


def render_v1_3_block_derivation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.3 Gold Block Derivation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Base derived: `{report['base_derived']}`",
        f"- Corpus manifest: `{report['corpus_manifest']}`",
        f"- Output: `{report['out_dir']}`",
        f"- Samples: `{report['samples']}`",
        f"- Span samples: `{report['span_samples']}`",
        f"- Block samples: `{report['block_samples']}`",
        "",
        "| Sample | Task | Spans | Blocks | Source |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"][:100]:
        lines.append(
            f"| `{row['sample_id']}` | `{row['task_type']}` | {row['gold_span_count']} | {row['gold_block_count']} | `{row['source']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_v1_3_span_worklist_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V1.3 Span Worklist",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Derived: `{report['derived']}`",
        f"- Corpus manifest: `{report['corpus_manifest']}`",
        f"- Task type: `{report['task_type']}`",
        f"- Missing span samples: `{report['missing_span_samples']}`",
        f"- Samples with candidates: `{report['with_candidates']}`",
        f"- Samples without candidates: `{len(report['without_candidates'])}`",
        "",
        "Candidate spans are machine-generated from target-file corpus chunks and require human review before merging.",
        "",
        "| Sample | Repo | Target files | Top candidates | Review comment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        target_files = "<br>".join(f"`{markdown_cell(path)}`" for path in row["target_gold_files"])
        top_candidates = "<br>".join(format_candidate_for_markdown(candidate) for candidate in row["candidate_spans"][:5])
        query = row.get("query") or {}
        review_comment = markdown_cell(truncate_text(str(query.get("review_comment") or ""), 220))
        lines.append(
            f"| `{row['sample_id']}` | `{markdown_cell(str(row['repo']))}` | {target_files} | {top_candidates} | {review_comment} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def format_candidate_for_markdown(candidate: dict[str, Any]) -> str:
    symbol = markdown_cell(str(candidate.get("symbol") or candidate.get("kind") or "candidate"))
    path = markdown_cell(str(candidate.get("path") or ""))
    start_line = int(candidate.get("start_line") or 0)
    end_line = int(candidate.get("end_line") or 0)
    score = candidate.get("score", 0)
    return f"`{path}:{start_line}-{end_line}` `{symbol}` score={score}"


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_v1_3_validation_markdown(report: dict[str, Any]) -> str:
    base = render_v1_2_validation_markdown({**report, "rows": report["rows"]})
    return base.replace("# V1.2 Validation", "# V1.3 Validation", 1).replace(
        f"- Span samples: `{report['span_samples']}`",
        f"- Span samples: `{report['span_samples']}`\n- Block samples: `{report['block_samples']}`",
        1,
    )
