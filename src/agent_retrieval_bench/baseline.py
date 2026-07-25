from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

from .curate import filter_samples, load_keep_ids
from .filters import contains_raw_patch_marker
from .io import ensure_parent, read_jsonl, stable_id
from .progress import ProgressReporter

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CANDIDATE_FILTERS = ("all_files", "code_only", "tests_only")
RANKERS = ("lexical", "bm25")
BM25_K1 = 1.5
BM25_B = 0.75
CODE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".fish",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".m",
    ".mm",
    ".php",
    ".proto",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".zsh",
}
NOISE_PATH_PARTS = {
    ".github",
    ".idea",
    ".vscode",
    "benchmark",
    "benchmarks",
    "changelog",
    "changelogs",
    "doc",
    "docs",
    "documentation",
    "examples",
    "licenses",
    "node_modules",
    "snapshot",
    "snapshots",
    "template",
    "templates",
    "vendor",
}
NOISE_FILENAMES = {
    "changelog",
    "changelog.md",
    "contributing.md",
    "license",
    "license.md",
    "readme",
    "readme.md",
}

BASE_METRIC_KEYS = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")
V1_2_FILE_METRIC_KEYS = (
    "Precision@5",
    "Precision@10",
    "Precision@20",
    "F0.5@5",
    "F0.5@10",
    "F0.5@20",
    "irrelevant_files@5",
    "irrelevant_files@10",
    "irrelevant_files@20",
    "hard_negative_hits@5",
    "hard_negative_hits@10",
    "hard_negative_hits@20",
    "context_pollution_tokens@8k",
    "gold_token_ratio@8k",
)
V1_3_FILE_METRIC_KEYS = (
    "coverage_auc@20",
    "redundancy@8k",
    "context_efficiency@8k",
)
LINE_METRIC_KEYS = (
    "line_samples",
    "line_recall@8k",
    "line_precision@8k",
    "line_f1@8k",
    "line_f0.5@8k",
    "line_gold_count",
    "line_predicted_count",
    "line_overlap_count",
)
BLOCK_METRIC_KEYS = (
    "block_samples",
    "block_recall@8k",
    "block_precision@8k",
    "block_f1@8k",
    "block_f0.5@8k",
    "gold_blocks",
    "predicted_blocks",
    "matched_blocks",
)
SUMMARY_METRIC_KEYS = (*BASE_METRIC_KEYS, *V1_2_FILE_METRIC_KEYS, *V1_3_FILE_METRIC_KEYS, *LINE_METRIC_KEYS, *BLOCK_METRIC_KEYS)


def evaluate_lexical_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    dry_run: bool = False,
    candidate_filter: str = "all_files",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    return evaluate_corpus_baseline(
        sample_paths=sample_paths,
        corpus_dir=corpus_dir,
        out_path=out_path,
        details_path=details_path,
        keep_list=keep_list,
        limit_samples=limit_samples,
        dry_run=dry_run,
        candidate_filter=candidate_filter,
        ranker="lexical",
        progress=progress,
        progress_stream=progress_stream,
    )


def evaluate_bm25_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    dry_run: bool = False,
    candidate_filter: str = "all_files",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    return evaluate_corpus_baseline(
        sample_paths=sample_paths,
        corpus_dir=corpus_dir,
        out_path=out_path,
        details_path=details_path,
        keep_list=keep_list,
        limit_samples=limit_samples,
        dry_run=dry_run,
        candidate_filter=candidate_filter,
        ranker="bm25",
        progress=progress,
        progress_stream=progress_stream,
    )


def evaluate_corpus_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    dry_run: bool = False,
    candidate_filter: str = "all_files",
    ranker: str = "lexical",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    validate_candidate_filter(candidate_filter)
    validate_ranker(ranker)
    reporter = ProgressReporter(progress, progress_stream)
    if dry_run:
        reporter.message("using synthetic dry-run corpus")
        manifest = {}
    else:
        reporter.message(f"loading corpus manifest: {corpus_dir / 'corpus_manifest.jsonl'}")
        manifest = load_corpus_manifest(corpus_dir)
        reporter.message(f"loaded corpus manifest: {len(manifest)} commit corpora")
    keep_ids = load_keep_ids(keep_list)
    if keep_list:
        if keep_ids is None:
            reporter.message(f"keep list not found, evaluating all samples: {keep_list}")
        else:
            reporter.message(f"loaded keep list: {len(keep_ids)} ids")
    samples = []
    for sample in filter_samples(iter_samples(sample_paths), keep_ids):
        if limit_samples and len(samples) >= limit_samples:
            break
        samples.append(sample)
    reporter.message(f"loaded benchmark samples: {len(samples)}")

    details: list[dict[str, Any]] = []
    skipped = Counter()
    pending_by_chunks_path: dict[Path, list[tuple[int, dict[str, Any], list[str], str]]] = defaultdict(list)
    sample_bar = reporter.bar("evaluating samples", len(samples))
    for sample_index, sample in enumerate(samples):
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
        if dry_run:
            chunks = synthetic_chunks(sample)
            append_ranker_detail(details, sample_index, sample, gold_files, query_text, chunks, candidate_filter, skipped, ranker=ranker)
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        chunks_path = manifest.get((sample.get("repo"), sample.get("base_commit")))
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        pending_by_chunks_path[chunks_path].append((sample_index, sample, gold_files, query_text))

    if pending_by_chunks_path:
        reporter.message(f"grouped samples by corpus: {len(pending_by_chunks_path)} commit corpora")
    for corpus_index, (chunks_path, pending) in enumerate(pending_by_chunks_path.items(), start=1):
        reporter.message(f"loading corpus {corpus_index}/{len(pending_by_chunks_path)}: {chunks_path}")
        chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
        if not chunks:
            skipped["empty_corpus"] += len(pending)
            sample_bar.update(step=len(pending), suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        for sample_index, sample, gold_files, query_text in pending:
            append_ranker_detail(details, sample_index, sample, gold_files, query_text, chunks, candidate_filter, skipped, ranker=ranker, pre_filtered=True)
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")

    details.sort(key=lambda item: item.pop("_sample_index", 0))
    evaluated = len(details)

    summary = summarize_details(details)
    result = {
        "mode": "dry_run" if dry_run else "corpus",
        "ranker": ranker,
        "candidate_filter": candidate_filter,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": evaluated,
        "skipped": dict(skipped),
        "metrics": summary,
        "runtime": {
            "wall_time_seconds": time.monotonic() - started_at,
            "progress": bool(progress),
        },
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        _write_jsonl(details_path, details)
    return result


def append_lexical_detail(
    details: list[dict[str, Any]],
    sample_index: int,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    chunks: list[dict[str, Any]],
    candidate_filter: str,
    skipped: Counter[str],
    pre_filtered: bool = False,
) -> None:
    append_ranker_detail(
        details=details,
        sample_index=sample_index,
        sample=sample,
        gold_files=gold_files,
        query_text=query_text,
        chunks=chunks,
        candidate_filter=candidate_filter,
        skipped=skipped,
        ranker="lexical",
        pre_filtered=pre_filtered,
    )


def append_ranker_detail(
    details: list[dict[str, Any]],
    sample_index: int,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    chunks: list[dict[str, Any]],
    candidate_filter: str,
    skipped: Counter[str],
    ranker: str = "lexical",
    pre_filtered: bool = False,
) -> None:
    if not pre_filtered:
        chunks = filter_candidate_chunks(chunks, candidate_filter)
    if not chunks:
        skipped["empty_corpus"] += 1
        return
    ranked = rank_chunks_for_ranker(query_text, chunks, ranker)
    hard_negatives = hard_negative_files(sample)
    span_rows = gold_spans(sample)
    block_rows = gold_blocks(sample)
    metrics = sample_metrics(gold_files, ranked, hard_negative_files=hard_negatives)
    detail = {
        "_sample_index": sample_index,
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "candidate_filter": candidate_filter,
        "ranker": ranker,
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


def load_corpus_manifest(corpus_dir: Path) -> dict[tuple[str, str], Path]:
    manifest: dict[tuple[str, str], Path] = {}
    for record in read_jsonl(corpus_dir / "corpus_manifest.jsonl"):
        if record.get("status") != "ok":
            continue
        repo = record.get("repo")
        base_commit = record.get("base_commit")
        chunks_path = Path(record.get("chunks_path", ""))
        if repo and base_commit and chunks_path.exists():
            manifest[(repo, base_commit)] = chunks_path
    return manifest


def iter_samples(sample_paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sample_paths:
        yield from read_jsonl(path)


def target_gold_files(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    if gold.get("no_gold") is True:
        return []
    explicit_files = _gold_paths(gold.get("files") or [])
    if explicit_files:
        return explicit_files
    if sample.get("task_type") == "code2test":
        return _dedupe(gold.get("related_tests") or [])
    if sample.get("task_type") == "comment2context":
        context_files = _gold_paths(gold.get("must_context_files") or gold.get("context_files") or [])
        if context_files:
            return context_files
    root_files = _dedupe(gold.get("root_cause_files") or [])
    return root_files or _dedupe(gold.get("related_tests") or [])


def supporting_context_files(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    values: list[Any] = []
    for key in ("supporting_context_files", "supporting_files"):
        if sample.get(key):
            values.extend(sample.get(key) or [])
        if gold.get(key):
            values.extend(gold.get(key) or [])
    return _dedupe(path_values(values))


def hard_negative_files(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    values: list[Any] = []
    for key in ("hard_negative_files",):
        if sample.get(key):
            values.extend(sample.get(key) or [])
        if gold.get(key):
            values.extend(gold.get(key) or [])
    return _dedupe(path_values(values))


def gold_spans(sample: dict[str, Any]) -> list[dict[str, Any]]:
    values = sample.get("gold_spans")
    if values is None:
        values = (sample.get("gold") or {}).get("gold_spans")
    spans: list[dict[str, Any]] = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        path = str(value.get("path") or "")
        if not path:
            continue
        try:
            start_line = int(value.get("start_line"))
            end_line = int(value.get("end_line"))
        except (TypeError, ValueError):
            continue
        spans.append(
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "reason": str(value.get("reason") or ""),
            }
        )
    return spans


def gold_blocks(sample: dict[str, Any]) -> list[dict[str, Any]]:
    values = sample.get("gold_blocks")
    if values is None:
        values = (sample.get("gold") or {}).get("gold_blocks")
    blocks: list[dict[str, Any]] = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        path = str(value.get("path") or "")
        if not path:
            continue
        try:
            start_line = int(value.get("start_line"))
            end_line = int(value.get("end_line"))
        except (TypeError, ValueError):
            continue
        if start_line <= 0 or end_line < start_line:
            continue
        block: dict[str, Any] = {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "kind": str(value.get("kind") or "block"),
            "symbol": str(value.get("symbol") or ""),
            "reason": str(value.get("reason") or ""),
        }
        if value.get("chunk_id"):
            block["chunk_id"] = str(value["chunk_id"])
        if value.get("source"):
            block["source"] = str(value["source"])
        blocks.append(block)
    return blocks


def query_provenance(sample: dict[str, Any]) -> str | None:
    value = sample.get("query_provenance")
    if value:
        return str(value)
    metadata = sample.get("metadata") or {}
    for key in ("query_provenance", "source_signal", "signal_type"):
        if metadata.get(key):
            return str(metadata[key])
    task_type = str(sample.get("task_type") or "")
    if task_type == "code2test":
        return "pr_summary"
    if task_type == "comment2context":
        return "review_comment"
    if task_type in {"trace2code", "testlog2code"}:
        return "failure_trace"
    return None


def given_files(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    query = sample.get("query") or {}
    values = gold.get("given_files") or []
    if not values and sample.get("task_type") == "comment2context":
        values = [query.get("given_file") or query.get("path")]
    return _dedupe(str(value) for value in values if value)


def _gold_paths(values: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return _dedupe(paths)


def query_text_for_eval(sample: dict[str, Any]) -> str:
    return json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True)


def query_has_leakage(sample: dict[str, Any], query_text: str) -> bool:
    if sample.get("task_type") == "edit2ripple":
        from .edit2ripple import edit2ripple_has_fatal_leakage

        if edit2ripple_has_fatal_leakage(sample, query_text):
            return True
    fix_commit = ((sample.get("gold") or {}).get("fix_commit") or "").strip()
    normalized_query = query_text.replace("\\n", "\n")
    return contains_raw_patch_marker(normalized_query) or bool(fix_commit and fix_commit in normalized_query)


def synthetic_chunks(sample: dict[str, Any]) -> list[dict[str, Any]]:
    gold = sample.get("gold") or {}
    paths = _dedupe(
        (gold.get("root_cause_files") or [])
        + (gold.get("related_tests") or [])
        + (gold.get("supporting_files") or [])
        + (gold.get("negative_distractors") or [])
        + _gold_paths(gold.get("must_context_files") or [])
        + _gold_paths(gold.get("context_files") or [])
        + (gold.get("given_files") or [])
        + path_values(gold.get("negative_distractors") or [])
        + hard_negative_files(sample)
    )
    chunks: list[dict[str, Any]] = []
    for path in paths:
        chunks.append(
            {
                "chunk_id": stable_id(sample.get("repo"), sample.get("base_commit"), path, "dry_run"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "path": path,
                "kind": "file",
                "symbol": "",
                "start_line": 1,
                "end_line": 1,
                "text": path.replace("/", " "),
            }
        )
    return chunks


def rank_chunks(query_text: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk for _, chunk in rank_chunks_with_scores(query_text, chunks)]


def rank_chunks_for_ranker(query_text: str, chunks: list[dict[str, Any]], ranker: str) -> list[dict[str, Any]]:
    if ranker == "lexical":
        return rank_chunks(query_text, chunks)
    if ranker == "bm25":
        return [chunk for _, chunk in rank_chunks_bm25_with_scores(query_text, chunks)]
    raise ValueError(f"Unknown ranker {ranker!r}. Expected one of: {', '.join(RANKERS)}")


def rank_chunks_with_scores(query_text: str, chunks: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return [(0.0, chunk) for chunk in chunks]
    doc_tokens: list[set[str]] = []
    document_frequency = Counter()
    for chunk in chunks:
        tokens = set(tokenize(chunk_text(chunk)))
        doc_tokens.append(tokens)
        document_frequency.update(tokens)
    total_docs = max(1, len(chunks))
    query_counts = Counter(query_tokens)
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    lowered_query = query_text.lower()
    for chunk, tokens in zip(chunks, doc_tokens):
        score = 0.0
        for token, count in query_counts.items():
            if token in tokens:
                score += (1.0 + math.log(count)) * math.log((total_docs + 1) / (1 + document_frequency[token]) + 1.0)
        path = str(chunk.get("path", ""))
        lowered_path = path.lower()
        basename = Path(path).name.lower()
        if lowered_path and lowered_path in lowered_query:
            score += 25.0
        if basename and basename in lowered_query:
            score += 8.0
        if chunk.get("symbol") and str(chunk["symbol"]).lower() in lowered_query:
            score += 5.0
        normalized = score / max(1.0, math.sqrt(len(tokens)))
        scored.append((normalized, path, str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(item[0], item[3]) for item in scored]


def rank_chunks_bm25_with_scores(
    query_text: str,
    chunks: list[dict[str, Any]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[tuple[float, dict[str, Any]]]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return [(0.0, chunk) for chunk in chunks]
    query_counts = Counter(query_tokens)
    document_tokens = [tokenize(chunk_text(chunk)) for chunk in chunks]
    document_lengths = [len(tokens) for tokens in document_tokens]
    average_document_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    document_frequency = Counter()
    for tokens in document_tokens:
        document_frequency.update(set(tokens))

    total_docs = max(1, len(chunks))
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for chunk, tokens, doc_length in zip(chunks, document_tokens, document_lengths):
        term_counts = Counter(tokens)
        score = 0.0
        for token, query_count in query_counts.items():
            tf = term_counts.get(token, 0)
            if not tf:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            length_norm = 1.0 - b
            if average_document_length > 0:
                length_norm += b * (doc_length / average_document_length)
            denominator = tf + k1 * length_norm
            score += query_count * idf * ((tf * (k1 + 1.0)) / denominator)
        path = str(chunk.get("path", ""))
        scored.append((score, path, str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(item[0], item[3]) for item in scored]


def sample_metrics(
    gold_files: list[str],
    ranked_chunks: list[dict[str, Any]],
    context_budget: int = 8_000,
    hard_negative_files: list[str] | None = None,
) -> dict[str, float]:
    gold = set(gold_files)
    hard_negatives = set(hard_negative_files or [])
    ranked_paths = unique_ranked_paths(ranked_chunks)
    metrics = {
        "Recall@5": recall_at(gold, ranked_paths, 5),
        "Recall@10": recall_at(gold, ranked_paths, 10),
        "Recall@20": recall_at(gold, ranked_paths, 20),
        "MRR": reciprocal_rank(gold, ranked_paths),
        "gold_coverage@8k": gold_coverage_at_budget(gold, ranked_chunks, context_budget),
    }
    for k in (5, 10, 20):
        precision = precision_at(gold, ranked_paths, k)
        recall = metrics[f"Recall@{k}"]
        metrics[f"Precision@{k}"] = precision
        metrics[f"F0.5@{k}"] = f_score(precision, recall, beta=0.5)
        metrics[f"irrelevant_files@{k}"] = float(irrelevant_files_at(gold, ranked_paths, k))
        metrics[f"hard_negative_hits@{k}"] = float(hard_negative_hits_at(hard_negatives, ranked_paths, k))
    budget_stats = context_budget_stats(gold, ranked_chunks, context_budget)
    metrics["context_pollution_tokens@8k"] = float(budget_stats["pollution_tokens"])
    metrics["gold_token_ratio@8k"] = float(budget_stats["gold_token_ratio"])
    metrics["coverage_auc@20"] = coverage_auc_at(gold, ranked_paths, 20)
    metrics["redundancy@8k"] = context_redundancy_at_budget(ranked_chunks, context_budget)
    metrics["context_efficiency@8k"] = float(budget_stats["gold_token_ratio"])
    return metrics


def gold_file_ranks(gold_files: list[str], ranked_chunks: list[dict[str, Any]]) -> dict[str, int | None]:
    ranked_paths = unique_ranked_paths(ranked_chunks)
    ranks = {path: index for index, path in enumerate(ranked_paths, start=1)}
    return {path: ranks.get(path) for path in gold_files}


def filter_candidate_chunks(chunks: list[dict[str, Any]], candidate_filter: str = "all_files") -> list[dict[str, Any]]:
    validate_candidate_filter(candidate_filter)
    if candidate_filter == "all_files":
        return chunks
    if candidate_filter == "tests_only":
        return [chunk for chunk in chunks if is_test_path(str(chunk.get("path", "")))]
    return [chunk for chunk in chunks if is_code_path(str(chunk.get("path", "")))]


def validate_candidate_filter(candidate_filter: str) -> None:
    if candidate_filter not in CANDIDATE_FILTERS:
        raise ValueError(f"Unknown candidate filter {candidate_filter!r}. Expected one of: {', '.join(CANDIDATE_FILTERS)}")


def validate_ranker(ranker: str) -> None:
    if ranker not in RANKERS:
        raise ValueError(f"Unknown ranker {ranker!r}. Expected one of: {', '.join(RANKERS)}")


def is_code_path(path: str) -> bool:
    if not path or is_noise_path(path):
        return False
    return Path(path).suffix.lower() in CODE_EXTENSIONS


def is_test_path(path: str) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/")
    lowered = normalized.lower()
    parts = {part.lower() for part in normalized.split("/")}
    basename = Path(lowered).name
    in_test_location = bool(parts & {"__tests__", "test", "tests", "testing", "testdata", "testsuite", "testsuites"})
    test_suffixes = (
        "_test.go",
        "_test.py",
        "_test.rs",
        "_test.cc",
        "_test.cpp",
        ".spec.js",
        ".spec.jsx",
        ".spec.ts",
        ".spec.tsx",
        ".test.js",
        ".test.jsx",
        ".test.ts",
        ".test.tsx",
    )
    test_name = basename.startswith("test_") or basename.endswith(test_suffixes) or basename.endswith(("test.java", "tests.java"))
    return (in_test_location or test_name) and Path(path).suffix.lower() in CODE_EXTENSIONS


def is_noise_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    lowered_parts = [part.lower() for part in normalized.split("/")]
    basename = Path(normalized).name.lower()
    stem = Path(normalized).stem.lower()
    return bool(set(lowered_parts) & NOISE_PATH_PARTS) or basename in NOISE_FILENAMES or stem in NOISE_FILENAMES


def recall_at(gold: set[str], ranked_paths: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked_paths[:k])) / len(gold)


def precision_at(gold: set[str], ranked_paths: list[str], k: int) -> float:
    if not ranked_paths or k <= 0:
        return 0.0
    denominator = min(k, len(ranked_paths))
    return len(gold & set(ranked_paths[:k])) / denominator if denominator else 0.0


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / max((beta_sq * precision) + recall, 1e-12)


def irrelevant_files_at(gold: set[str], ranked_paths: list[str], k: int) -> int:
    return sum(1 for path in ranked_paths[:k] if path not in gold)


def hard_negative_hits_at(hard_negatives: set[str], ranked_paths: list[str], k: int) -> int:
    if not hard_negatives:
        return 0
    return sum(1 for path in ranked_paths[:k] if path in hard_negatives)


def reciprocal_rank(gold: set[str], ranked_paths: list[str]) -> float:
    for index, path in enumerate(ranked_paths, start=1):
        if path in gold:
            return 1.0 / index
    return 0.0


def gold_coverage_at_budget(gold: set[str], ranked_chunks: list[dict[str, Any]], context_budget: int) -> float:
    if not gold:
        return 0.0
    used = 0
    covered: set[str] = set()
    for chunk in ranked_chunks:
        text = str(chunk.get("text", ""))
        if used + len(text) > context_budget and used > 0:
            break
        used += len(text)
        path = str(chunk.get("path", ""))
        if path in gold:
            covered.add(path)
    return len(covered) / len(gold)


def context_budget_stats(gold: set[str], ranked_chunks: list[dict[str, Any]], context_budget: int) -> dict[str, float]:
    used = 0
    gold_tokens = 0
    pollution_tokens = 0
    for chunk in ranked_chunks:
        text = str(chunk.get("text", ""))
        tokens = len(text)
        if used + tokens > context_budget and used > 0:
            break
        used += tokens
        path = str(chunk.get("path", ""))
        if path in gold:
            gold_tokens += tokens
        else:
            pollution_tokens += tokens
    return {
        "used_tokens": float(used),
        "gold_tokens": float(gold_tokens),
        "pollution_tokens": float(pollution_tokens),
        "gold_token_ratio": (gold_tokens / used) if used else 0.0,
    }


def coverage_auc_at(gold: set[str], ranked_paths: list[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    return sum(recall_at(gold, ranked_paths, cutoff) for cutoff in range(1, k + 1)) / k


def context_redundancy_at_budget(ranked_chunks: list[dict[str, Any]], context_budget: int) -> float:
    selected = chunks_within_budget(ranked_chunks, context_budget)
    if not selected:
        return 0.0
    unique_paths = {str(chunk.get("path") or "") for chunk in selected if chunk.get("path")}
    if not unique_paths:
        return 0.0
    redundant = max(0, len(selected) - len(unique_paths))
    return redundant / len(selected)


def chunks_within_budget(ranked_chunks: list[dict[str, Any]], context_budget: int) -> list[dict[str, Any]]:
    used = 0
    selected: list[dict[str, Any]] = []
    for chunk in ranked_chunks:
        text = str(chunk.get("text", ""))
        tokens = len(text)
        if used + tokens > context_budget and used > 0:
            break
        used += tokens
        selected.append(chunk)
    return selected


def span_metrics_at_budget(
    gold_span_rows: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    context_budget: int = 8_000,
) -> dict[str, float]:
    return span_metrics_from_line_metrics(line_metrics_at_budget(gold_span_rows, ranked_chunks, context_budget=context_budget))


def line_metrics_at_budget(
    gold_span_rows: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    context_budget: int = 8_000,
) -> dict[str, float]:
    gold_intervals = span_intervals(gold_span_rows)
    if not gold_intervals:
        return {
            "line_samples": 0,
            "line_recall@8k": 0.0,
            "line_precision@8k": 0.0,
            "line_f1@8k": 0.0,
            "line_f0.5@8k": 0.0,
            "line_gold_count": 0.0,
            "line_predicted_count": 0.0,
            "line_overlap_count": 0.0,
        }
    predicted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk in chunks_within_budget(ranked_chunks, context_budget):
        path = str(chunk.get("path") or "")
        try:
            start_line = int(chunk.get("start_line") or 0)
            end_line = int(chunk.get("end_line") or 0)
        except (TypeError, ValueError):
            continue
        if path and start_line > 0 and end_line >= start_line:
            predicted[path].append((start_line, end_line))
    gold_lines = interval_line_count(gold_intervals)
    predicted_lines = interval_line_count(predicted)
    overlap = interval_overlap_lines(gold_intervals, predicted)
    recall = overlap / gold_lines if gold_lines else 0.0
    precision = overlap / predicted_lines if predicted_lines else 0.0
    return {
        "line_samples": 1,
        "line_recall@8k": recall,
        "line_precision@8k": precision,
        "line_f1@8k": f_score(precision, recall),
        "line_f0.5@8k": f_score(precision, recall, beta=0.5),
        "line_gold_count": float(gold_lines),
        "line_predicted_count": float(predicted_lines),
        "line_overlap_count": float(overlap),
    }


def span_metrics_from_line_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "span_samples": metrics.get("line_samples", 0.0),
        "span_recall@8k": metrics.get("line_recall@8k", 0.0),
        "span_precision@8k": metrics.get("line_precision@8k", 0.0),
        "span_f0.5@8k": metrics.get("line_f0.5@8k", 0.0),
        "line_overlap_f0.5": metrics.get("line_f0.5@8k", 0.0),
        "gold_lines": metrics.get("line_gold_count", 0.0),
        "predicted_lines": metrics.get("line_predicted_count", 0.0),
        "overlap_lines": metrics.get("line_overlap_count", 0.0),
    }


def block_metrics_at_budget(
    gold_block_rows: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    context_budget: int = 8_000,
) -> dict[str, float]:
    gold_rows = [row for row in (normalize_block_row(row) for row in gold_block_rows) if row]
    if not gold_rows:
        return {
            "block_samples": 0,
            "block_recall@8k": 0.0,
            "block_precision@8k": 0.0,
            "block_f1@8k": 0.0,
            "block_f0.5@8k": 0.0,
            "gold_blocks": 0.0,
            "predicted_blocks": 0.0,
            "matched_blocks": 0.0,
        }
    predicted_rows = [
        row
        for row in (normalize_block_row(chunk) for chunk in chunks_within_budget(ranked_chunks, context_budget))
        if row and row.get("kind") != "file"
    ]
    predicted_rows = dedupe_blocks(predicted_rows)
    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()
    for pred_index, predicted in enumerate(predicted_rows):
        for gold_index, gold in enumerate(gold_rows):
            if blocks_overlap(gold, predicted):
                matched_gold.add(gold_index)
                matched_predicted.add(pred_index)
    recall = len(matched_gold) / len(gold_rows) if gold_rows else 0.0
    precision = len(matched_predicted) / len(predicted_rows) if predicted_rows else 0.0
    return {
        "block_samples": 1,
        "block_recall@8k": recall,
        "block_precision@8k": precision,
        "block_f1@8k": f_score(precision, recall),
        "block_f0.5@8k": f_score(precision, recall, beta=0.5),
        "gold_blocks": float(len(gold_rows)),
        "predicted_blocks": float(len(predicted_rows)),
        "matched_blocks": float(len(matched_gold)),
    }


def normalize_block_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    path = str(row.get("path") or "")
    if not path:
        return None
    try:
        start_line = int(row.get("start_line") or 0)
        end_line = int(row.get("end_line") or 0)
    except (TypeError, ValueError):
        return None
    if start_line <= 0 or end_line < start_line:
        return None
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "kind": str(row.get("kind") or "block"),
        "symbol": str(row.get("symbol") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
    }


def dedupe_blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("path") or ""),
            int(row.get("start_line") or 0),
            int(row.get("end_line") or 0),
            str(row.get("kind") or ""),
            str(row.get("symbol") or ""),
            str(row.get("chunk_id") or ""),
        )
        if key in seen:
            continue
        output.append(row)
        seen.add(key)
    return output


def blocks_overlap(gold: dict[str, Any], predicted: dict[str, Any]) -> bool:
    if str(gold.get("path") or "") != str(predicted.get("path") or ""):
        return False
    if gold.get("chunk_id") and predicted.get("chunk_id") and gold.get("chunk_id") == predicted.get("chunk_id"):
        return True
    gold_start = int(gold.get("start_line") or 0)
    gold_end = int(gold.get("end_line") or 0)
    pred_start = int(predicted.get("start_line") or 0)
    pred_end = int(predicted.get("end_line") or 0)
    return min(gold_end, pred_end) >= max(gold_start, pred_start)


def span_intervals(span_rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in span_rows:
        path = str(span.get("path") or "")
        try:
            start_line = int(span.get("start_line") or 0)
            end_line = int(span.get("end_line") or 0)
        except (TypeError, ValueError):
            continue
        if path and start_line > 0 and end_line >= start_line:
            intervals[path].append((start_line, end_line))
    return {path: merge_intervals(rows) for path, rows in intervals.items()}


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_line_count(intervals_by_path: dict[str, list[tuple[int, int]]]) -> int:
    return sum(end - start + 1 for intervals in intervals_by_path.values() for start, end in merge_intervals(intervals))


def interval_overlap_lines(
    gold_intervals: dict[str, list[tuple[int, int]]],
    predicted_intervals: dict[str, list[tuple[int, int]]],
) -> int:
    overlap = 0
    for path, gold_rows in gold_intervals.items():
        pred_rows = merge_intervals(predicted_intervals.get(path, []))
        for gold_start, gold_end in gold_rows:
            for pred_start, pred_end in pred_rows:
                start = max(gold_start, pred_start)
                end = min(gold_end, pred_end)
                if end >= start:
                    overlap += end - start + 1
    return overlap


def unique_ranked_paths(ranked_chunks: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for chunk in ranked_chunks:
        path = str(chunk.get("path", ""))
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    line_grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    block_grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for detail in details:
        task_type = str(detail.get("task_type") or "unknown")
        grouped["overall"].append(detail["metrics"])
        grouped[task_type].append(detail["metrics"])
        if isinstance(detail.get("line_metrics"), dict):
            line_grouped["overall"].append(detail["line_metrics"])
            line_grouped[task_type].append(detail["line_metrics"])
        if isinstance(detail.get("block_metrics"), dict):
            block_grouped["overall"].append(detail["block_metrics"])
            block_grouped[task_type].append(detail["block_metrics"])
    summary: dict[str, Any] = {}
    for task_type, metrics in sorted(grouped.items()):
        row = average_metrics(metrics)
        if line_grouped.get(task_type):
            row.update(average_named_metrics(line_grouped[task_type], LINE_METRIC_KEYS))
        if block_grouped.get(task_type):
            row.update(average_named_metrics(block_grouped[task_type], BLOCK_METRIC_KEYS))
        summary[task_type] = row
    return summary


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0, "Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
    keys = [key for key in SUMMARY_METRIC_KEYS if key in BASE_METRIC_KEYS or any(key in row for row in rows)]
    return {"samples": len(rows), **{key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows) for key in keys}}


def average_named_metrics(rows: list[dict[str, float]], keys: Iterable[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows) for key in keys if any(key in row for row in rows)}


def chunk_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in [chunk.get("path"), chunk.get("symbol"), chunk.get("kind"), chunk.get("text")]
        if part
    )


def tokenize(text: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in TOKEN_RE.findall(spaced)]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def path_values(values: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return paths
