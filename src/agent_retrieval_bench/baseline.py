from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .curate import filter_samples, load_keep_ids
from .filters import contains_raw_patch_marker
from .io import ensure_parent, read_jsonl, stable_id

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CANDIDATE_FILTERS = ("all_files", "code_only", "tests_only")
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


def evaluate_lexical_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    dry_run: bool = False,
    candidate_filter: str = "all_files",
) -> dict[str, Any]:
    validate_candidate_filter(candidate_filter)
    manifest = {} if dry_run else load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
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
        query_text = query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            continue
        if dry_run:
            chunks = synthetic_chunks(sample)
        else:
            chunks_path = manifest.get((sample.get("repo"), sample.get("base_commit")))
            if not chunks_path:
                skipped["missing_corpus"] += 1
                continue
            chunks = read_jsonl(chunks_path)
        chunks = filter_candidate_chunks(chunks, candidate_filter)
        if not chunks:
            skipped["empty_corpus"] += 1
            continue
        ranked = rank_chunks(query_text, chunks)
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

    summary = summarize_details(details)
    result = {
        "mode": "dry_run" if dry_run else "corpus",
        "candidate_filter": candidate_filter,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": evaluated,
        "skipped": dict(skipped),
        "metrics": summary,
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        _write_jsonl(details_path, details)
    return result


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
    if sample.get("task_type") == "code2test":
        return _dedupe(gold.get("related_tests") or [])
    if sample.get("task_type") == "comment2context":
        context_files = _gold_paths(gold.get("must_context_files") or gold.get("context_files") or [])
        if context_files:
            return context_files
    root_files = _dedupe(gold.get("root_cause_files") or [])
    return root_files or _dedupe(gold.get("related_tests") or [])


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
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return chunks[:]
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
    return [item[3] for item in scored]


def sample_metrics(gold_files: list[str], ranked_chunks: list[dict[str, Any]], context_budget: int = 8_000) -> dict[str, float]:
    gold = set(gold_files)
    ranked_paths = unique_ranked_paths(ranked_chunks)
    metrics = {
        "Recall@5": recall_at(gold, ranked_paths, 5),
        "Recall@10": recall_at(gold, ranked_paths, 10),
        "Recall@20": recall_at(gold, ranked_paths, 20),
        "MRR": reciprocal_rank(gold, ranked_paths),
        "gold_coverage@8k": gold_coverage_at_budget(gold, ranked_chunks, context_budget),
    }
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
    for detail in details:
        grouped["overall"].append(detail["metrics"])
        grouped[detail["task_type"]].append(detail["metrics"])
    return {task_type: average_metrics(metrics) for task_type, metrics in sorted(grouped.items())}


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0, "Recall@5": 0.0, "Recall@10": 0.0, "Recall@20": 0.0, "MRR": 0.0, "gold_coverage@8k": 0.0}
    keys = ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k")
    return {"samples": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


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
