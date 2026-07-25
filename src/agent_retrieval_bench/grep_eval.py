from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .baseline import (
    CANDIDATE_FILTERS,
    filter_candidate_chunks,
    gold_file_ranks,
    iter_samples,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    sample_metrics,
    summarize_details,
    target_gold_files,
    tokenize,
    unique_ranked_paths,
    validate_candidate_filter,
)
from .curate import filter_samples, load_keep_ids
from .io import ensure_parent, read_jsonl

GREP_PATTERN_MODES = ("strict", "expanded")
CODE_SPAN_RE = re.compile(r"`([^`\n]{2,200})`")
FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]+\."
    r"(?:bash|c|cc|cpp|cs|cxx|fish|go|h|hpp|java|js|jsx|kt|kts|m|mm|php|proto|py|pyi|rb|rs|scala|sh|swift|ts|tsx|zsh|json|ya?ml|toml|md|rst|txt)\b"
)
BARE_FILE_RE = re.compile(
    r"\b[A-Za-z0-9_.@+-]+\."
    r"(?:bash|c|cc|cpp|cs|cxx|fish|go|h|hpp|java|js|jsx|kt|kts|m|mm|php|proto|py|pyi|rb|rs|scala|sh|swift|ts|tsx|zsh)\b"
)
FLAG_RE = re.compile(r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9_-]{1,80}\b")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.|/)[A-Za-z_][A-Za-z0-9_]*)*\b")
ERROR_SUFFIX_RE = re.compile(r"(?:Error|Exception|Failure|Warning|Panic|Timeout|Traceback)$")

STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "arg",
    "args",
    "base",
    "body",
    "bug",
    "call",
    "case",
    "change",
    "changed",
    "changed_file",
    "class",
    "code",
    "comment",
    "config",
    "context",
    "could",
    "data",
    "default",
    "does",
    "error",
    "expected",
    "fail",
    "failed",
    "failure",
    "failure_excerpt",
    "file",
    "files",
    "fix",
    "from",
    "function",
    "given_file",
    "gold",
    "has",
    "have",
    "issue",
    "kind",
    "line",
    "log",
    "message",
    "method",
    "must",
    "name",
    "not",
    "path",
    "pr",
    "pr_body",
    "pr_title",
    "query",
    "raw_signal",
    "review",
    "review_comment",
    "run",
    "should",
    "signal",
    "source",
    "stack",
    "test",
    "tests",
    "text",
    "that",
    "the",
    "this",
    "title",
    "trace",
    "type",
    "update",
    "value",
    "when",
    "with",
}


@dataclass(frozen=True)
class SearchPattern:
    term: str
    kind: str
    weight: float

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.term.lower()}"


def evaluate_grep_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    candidate_filter: str = "all_files",
    pattern_mode: str = "strict",
) -> dict[str, Any]:
    validate_candidate_filter(candidate_filter)
    validate_pattern_mode(pattern_mode)
    manifest = load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        if limit_samples and len(samples) >= limit_samples:
            break
        samples.append(sample)

    details: list[dict[str, Any]] = []
    skipped = Counter()
    pending_by_chunks_path: dict[Path, list[tuple[int, dict[str, Any], list[str], str]]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        gold_files = target_gold_files(sample)
        if not gold_files:
            skipped["no_gold"] += 1
            continue
        query_text = query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            continue
        chunks_path = manifest.get((sample.get("repo"), sample.get("base_commit")))
        if not chunks_path:
            skipped["missing_corpus"] += 1
            continue
        pending_by_chunks_path[chunks_path].append((sample_index, sample, gold_files, query_text))

    for chunks_path, pending in pending_by_chunks_path.items():
        chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
        if not chunks:
            skipped["empty_corpus"] += len(pending)
            continue
        for sample_index, sample, gold_files, query_text in pending:
            ranked, patterns, file_scores = rank_chunks_by_grep(query_text, chunks, pattern_mode=pattern_mode)
            metrics = sample_metrics(gold_files, ranked)
            details.append(
                {
                    "_sample_index": sample_index,
                    "sample_id": sample.get("id"),
                    "task_type": sample.get("task_type"),
                    "repo": sample.get("repo"),
                    "base_commit": sample.get("base_commit"),
                    "candidate_filter": candidate_filter,
                    "pattern_mode": pattern_mode,
                    "patterns": [format_pattern(pattern) for pattern in patterns],
                    "gold_files": gold_files,
                    "gold_ranks": gold_file_ranks(gold_files, ranked),
                    "top_files": unique_ranked_paths(ranked)[:20],
                    "top_file_scores": file_scores[:20],
                    "metrics": metrics,
                }
            )

    details.sort(key=lambda item: item.pop("_sample_index", 0))
    result = {
        "mode": "grep",
        "model": f"grep-{pattern_mode}",
        "candidate_filter": candidate_filter,
        "pattern_mode": pattern_mode,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": len(details),
        "skipped": dict(skipped),
        "metrics": summarize_details(details),
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        write_jsonl(details_path, details)
    return result


def validate_pattern_mode(pattern_mode: str) -> None:
    if pattern_mode not in GREP_PATTERN_MODES:
        raise ValueError(f"Unknown grep pattern mode {pattern_mode!r}. Expected one of: {', '.join(GREP_PATTERN_MODES)}")


def extract_search_patterns(query_text: str, pattern_mode: str = "strict") -> list[SearchPattern]:
    validate_pattern_mode(pattern_mode)
    normalized = query_text.replace("\\n", "\n")
    patterns: dict[str, SearchPattern] = {}

    def add(term: str, kind: str, weight: float) -> None:
        cleaned = normalize_search_term(term)
        if not is_usable_term(cleaned):
            return
        pattern = SearchPattern(cleaned, kind, weight)
        existing = patterns.get(pattern.term.lower())
        if existing is None or existing.weight < pattern.weight:
            patterns[pattern.term.lower()] = pattern

    for match in CODE_SPAN_RE.finditer(normalized):
        term = match.group(1).strip()
        add(term, classify_code_span(term), 12.0)

    for regex in (FILE_PATH_RE, BARE_FILE_RE):
        for match in regex.finditer(normalized):
            add(match.group(0), "file_path", 18.0)

    for match in FLAG_RE.finditer(normalized):
        add(match.group(0), "flag", 10.0)

    for match in IDENTIFIER_RE.finditer(normalized):
        term = match.group(0)
        if is_high_signal_identifier(term):
            add(term, classify_identifier(term), 7.0)

    if pattern_mode == "expanded":
        for token in tokenize(normalized):
            if len(token) >= 4 and token.lower() not in STOP_TERMS:
                add(token, "token", 1.0)

    return sorted(patterns.values(), key=lambda pattern: (-pattern.weight, pattern.kind, pattern.term.lower()))


def rank_chunks_by_grep(
    query_text: str,
    chunks: list[dict[str, Any]],
    pattern_mode: str = "strict",
) -> tuple[list[dict[str, Any]], list[SearchPattern], list[dict[str, Any]]]:
    patterns = extract_search_patterns(query_text, pattern_mode=pattern_mode)
    if not patterns:
        return [], patterns, []

    file_pattern_scores: dict[str, dict[str, float]] = defaultdict(dict)
    file_best_chunk_score: dict[str, float] = defaultdict(float)
    file_best_chunk: dict[str, dict[str, Any]] = {}
    file_hit_counts: Counter[str] = Counter()

    for chunk in chunks:
        path = str(chunk.get("path", ""))
        if not path:
            continue
        matches = score_chunk_patterns(chunk, patterns)
        if not matches:
            continue
        chunk_score = sum(matches.values())
        file_hit_counts[path] += 1
        if chunk_score > file_best_chunk_score[path]:
            file_best_chunk_score[path] = chunk_score
            file_best_chunk[path] = chunk
        for key, score in matches.items():
            previous = file_pattern_scores[path].get(key, 0.0)
            if score > previous:
                file_pattern_scores[path][key] = score

    ranked_files: list[tuple[str, float, int, float]] = []
    for path, scores in file_pattern_scores.items():
        score = sum(scores.values())
        score += min(2.0, math.log1p(file_hit_counts[path]) * 0.25)
        ranked_files.append((path, score, len(scores), file_best_chunk_score[path]))
    ranked_files.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))

    ranked_chunks = [representative_chunk(path, file_best_chunk[path]) for path, _, _, _ in ranked_files]
    file_scores = [
        {
            "path": path,
            "score": score,
            "matched_patterns": matched_patterns,
            "best_chunk_score": best_chunk_score,
            "hit_chunks": file_hit_counts[path],
        }
        for path, score, matched_patterns, best_chunk_score in ranked_files
    ]
    return ranked_chunks, patterns, file_scores


def score_chunk_patterns(chunk: dict[str, Any], patterns: list[SearchPattern]) -> dict[str, float]:
    path = str(chunk.get("path", ""))
    basename = PurePosixPath(path).name
    symbol = str(chunk.get("symbol", ""))
    text = str(chunk.get("text", ""))
    lowered_path = path.lower()
    lowered_basename = basename.lower()
    lowered_symbol = symbol.lower()
    lowered_text = text.lower()
    matches: dict[str, float] = {}

    for pattern in patterns:
        term = pattern.term
        lowered_term = term.lower()
        score = 0.0
        if lowered_term and lowered_term in lowered_path:
            score += pattern.weight * (2.0 if pattern.kind == "file_path" else 1.35)
        if lowered_term and lowered_term in lowered_basename:
            score += pattern.weight * 1.1
        if lowered_term and lowered_term in lowered_symbol:
            score += pattern.weight * 1.2
        if pattern.kind == "token":
            if token_occurs(lowered_text, lowered_term):
                score += pattern.weight
        elif lowered_term and lowered_term in lowered_text:
            score += pattern.weight
        if score > 0:
            matches[pattern.key] = score
    return matches


def token_occurs(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text))


def representative_chunk(path: str, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "path": path,
        "kind": chunk.get("kind", "file"),
        "symbol": chunk.get("symbol", ""),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "text": chunk.get("text") or path,
    }


def normalize_search_term(term: str) -> str:
    return term.strip().strip("'\"“”‘’.,;:()[]{}")


def is_usable_term(term: str) -> bool:
    lowered = term.lower()
    if len(term) < 2 or lowered in STOP_TERMS:
        return False
    return any(char.isalnum() for char in term)


def classify_code_span(term: str) -> str:
    if FILE_PATH_RE.fullmatch(term) or BARE_FILE_RE.fullmatch(term):
        return "file_path"
    if term.startswith("--"):
        return "flag"
    if ERROR_SUFFIX_RE.search(term):
        return "error"
    return "code"


def classify_identifier(term: str) -> str:
    if ERROR_SUFFIX_RE.search(term):
        return "error"
    return "identifier"


def is_high_signal_identifier(term: str) -> bool:
    if not is_usable_term(term):
        return False
    lowered = term.lower()
    if lowered in STOP_TERMS:
        return False
    if "/" in term or "." in term or "::" in term or "_" in term:
        return True
    if any(char.isdigit() for char in term):
        return True
    if ERROR_SUFFIX_RE.search(term):
        return True
    if len(term) >= 3 and term.isupper():
        return True
    return any(char.isupper() for char in term[1:]) and any(char.islower() for char in term)


def format_pattern(pattern: SearchPattern) -> dict[str, Any]:
    return {"term": pattern.term, "kind": pattern.kind, "weight": pattern.weight}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
