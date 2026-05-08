from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, TextIO

from .baseline import (
    CANDIDATE_FILTERS,
    filter_candidate_chunks,
    given_files,
    gold_file_ranks,
    iter_samples,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    sample_metrics,
    summarize_details,
    target_gold_files,
    tokenize,
    validate_candidate_filter,
)
from .curate import filter_samples, load_keep_ids
from .embedding_eval import ProgressReporter
from .io import ensure_parent, read_jsonl

IMPORT_RE = re.compile(
    r"(?:^|\s)(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\./:-]+)|"
    r"require\(\s*['\"]([^'\"]+)['\"]\s*\)|from\s+['\"]([^'\"]+)['\"]|"
    r"import\s+['\"]([^'\"]+)['\"]|#include\s+[<\"]([^>\"]+)[>\"]|"
    r"use\s+(?:crate::|super::)?([A-Za-z0-9_:]+))"
)
GENERIC_SYMBOLS = {
    "add",
    "app",
    "args",
    "build",
    "call",
    "case",
    "check",
    "class",
    "close",
    "config",
    "context",
    "data",
    "default",
    "delete",
    "error",
    "event",
    "file",
    "find",
    "get",
    "handle",
    "helper",
    "init",
    "item",
    "list",
    "load",
    "main",
    "make",
    "mock",
    "model",
    "name",
    "node",
    "option",
    "options",
    "parse",
    "path",
    "read",
    "render",
    "request",
    "response",
    "result",
    "run",
    "save",
    "server",
    "set",
    "state",
    "string",
    "test",
    "tests",
    "type",
    "update",
    "value",
    "write",
}
TEST_MARKERS = ("test", "tests", "testing", "__tests__", "testsuite", "testsuites")
MAX_SYMBOL_REFERENCE_SCAN_FILES = 3_000


@dataclass
class FileNode:
    path: str
    text: str = ""
    symbols: set[str] = field(default_factory=set)
    kind: str = "other"
    token_set: set[str] = field(default_factory=set)
    path_tokens: set[str] = field(default_factory=set)
    symbol_tokens: set[str] = field(default_factory=set)


@dataclass
class RepoMapIndex:
    nodes: dict[str, FileNode]
    graph: dict[str, dict[str, float]]
    idf: dict[str, float]
    stats: dict[str, Any]


def evaluate_repomap_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    candidate_filter: str = "all_files",
    query_weight: float = 0.65,
    pagerank_weight: float = 0.25,
    affinity_weight: float = 0.10,
    max_symbol_refs_per_file: int = 80,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    validate_candidate_filter(candidate_filter)
    reporter = ProgressReporter(progress, progress_stream)
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
    index_cache: dict[Path, RepoMapIndex] = {}
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
        chunks_path = manifest.get((sample.get("repo"), sample.get("base_commit")))
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        pending_by_chunks_path[chunks_path].append((sample_index, sample, gold_files, query_text))

    for corpus_index, (chunks_path, pending) in enumerate(pending_by_chunks_path.items(), start=1):
        index = index_cache.get(chunks_path)
        if index is None:
            reporter.message(f"building repo map {corpus_index}/{len(pending_by_chunks_path)}: {chunks_path}")
            chunks = read_jsonl(chunks_path)
            if not chunks:
                skipped["empty_corpus"] += len(pending)
                sample_bar.update(step=len(pending), suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
                continue
            index = build_repomap_index(chunks, max_symbol_refs_per_file=max_symbol_refs_per_file)
            index_cache[chunks_path] = index
        candidate_paths = candidate_paths_for_filter(index.nodes, candidate_filter)
        if not candidate_paths:
            skipped["empty_corpus"] += len(pending)
            sample_bar.update(step=len(pending), suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        for sample_index, sample, gold_files, query_text in pending:
            ranked = rank_files_for_sample(
                sample=sample,
                query_text=query_text,
                index=index,
                candidate_paths=candidate_paths,
                query_weight=query_weight,
                pagerank_weight=pagerank_weight,
                affinity_weight=affinity_weight,
            )
            if not ranked:
                skipped["empty_corpus"] += 1
                sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
                continue
            ranked_chunks = ranked_file_chunks(ranked, index.nodes)
            metrics = sample_metrics(gold_files, ranked_chunks)
            details.append(
                {
                    "_sample_index": sample_index,
                    "sample_id": sample.get("id"),
                    "task_type": sample.get("task_type"),
                    "repo": sample.get("repo"),
                    "base_commit": sample.get("base_commit"),
                    "candidate_filter": candidate_filter,
                    "gold_files": gold_files,
                    "gold_ranks": gold_file_ranks(gold_files, ranked_chunks),
                    "top_files": [item[0] for item in ranked[:20]],
                    "top_file_scores": [format_file_score(item) for item in ranked[:20]],
                    "repo_map_stats": index.stats,
                    "metrics": metrics,
                }
            )
            sample_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")

    details.sort(key=lambda item: item.pop("_sample_index", 0))
    result = {
        "mode": "repomap",
        "model": "aider-style-repomap",
        "candidate_filter": candidate_filter,
        "weights": {
            "query": query_weight,
            "pagerank": pagerank_weight,
            "affinity": affinity_weight,
        },
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


def build_repomap_index(chunks: list[dict[str, Any]], max_symbol_refs_per_file: int = 80) -> RepoMapIndex:
    nodes = aggregate_file_nodes(chunks)
    compute_tokens_and_idf(nodes)
    graph: dict[str, dict[str, float]] = {path: {} for path in nodes}
    add_import_edges(nodes, graph)
    add_symbol_reference_edges(nodes, graph, max_symbol_refs_per_file=max_symbol_refs_per_file)
    add_source_test_edges(nodes, graph)
    edge_count = sum(len(neighbors) for neighbors in graph.values())
    searchable_file_count = sum(1 for node in nodes.values() if node.kind != "other")
    stats = {
        "file_count": len(nodes),
        "symbol_count": sum(len(node.symbols) for node in nodes.values()),
        "edge_count": edge_count,
        "source_count": sum(1 for node in nodes.values() if node.kind == "source"),
        "test_count": sum(1 for node in nodes.values() if node.kind == "test"),
        "symbol_reference_scan_skipped": searchable_file_count > MAX_SYMBOL_REFERENCE_SCAN_FILES,
    }
    idf = build_idf(nodes)
    return RepoMapIndex(nodes=nodes, graph=graph, idf=idf, stats=stats)


def aggregate_file_nodes(chunks: list[dict[str, Any]]) -> dict[str, FileNode]:
    nodes: dict[str, FileNode] = {}
    for chunk in chunks:
        path = str(chunk.get("path") or "")
        if not path:
            continue
        node = nodes.setdefault(path, FileNode(path=path, kind=classify_file_kind(path)))
        if chunk.get("kind") == "file" and not node.text:
            node.text = str(chunk.get("text") or "")
        symbol = str(chunk.get("symbol") or "").strip()
        if symbol and is_informative_symbol(symbol):
            node.symbols.add(symbol)
    return nodes


def compute_tokens_and_idf(nodes: dict[str, FileNode]) -> None:
    for node in nodes.values():
        node.path_tokens = set(tokenize(node.path.replace("/", " ").replace(".", " ").replace("_", " ")))
        symbol_text = " ".join(node.symbols)
        node.symbol_tokens = set(tokenize(symbol_text))
        content_tokens = set(tokenize(node.text[:20_000]))
        node.token_set = node.path_tokens | node.symbol_tokens | content_tokens


def build_idf(nodes: dict[str, FileNode]) -> dict[str, float]:
    df = Counter()
    for node in nodes.values():
        df.update(node.token_set)
    total = max(1, len(nodes))
    return {token: math.log((total + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}


def add_import_edges(nodes: dict[str, FileNode], graph: dict[str, dict[str, float]]) -> None:
    path_index = build_path_index(nodes)
    for path, node in nodes.items():
        for spec in extract_import_specs(node.text):
            for target in resolve_import_spec(spec, path, path_index):
                if target != path:
                    add_edge(graph, path, target, 1.5)


def add_symbol_reference_edges(nodes: dict[str, FileNode], graph: dict[str, dict[str, float]], max_symbol_refs_per_file: int = 80) -> None:
    searchable_file_count = sum(1 for node in nodes.values() if node.kind != "other")
    if searchable_file_count > MAX_SYMBOL_REFERENCE_SCAN_FILES:
        return
    symbol_index: dict[str, set[str]] = defaultdict(set)
    symbol_parts: dict[str, set[str]] = {}
    token_to_symbol_keys: dict[str, set[str]] = defaultdict(set)
    for path, node in nodes.items():
        if node.kind == "other":
            continue
        for symbol in node.symbols:
            key = normalize_identifier(symbol)
            if key:
                symbol_index[key].add(path)
    for path, node in nodes.items():
        if node.kind == "other":
            continue
        for symbol in node.symbols:
            key = normalize_identifier(symbol)
            if not key or len(symbol_index[key]) > 8:
                continue
            parts = {token for token in tokenize(symbol) if len(token) >= 3 and token not in GENERIC_SYMBOLS}
            if not parts or (len(parts) == 1 and len(next(iter(parts))) < 8):
                continue
            symbol_parts[key] = parts
            for token in parts:
                token_to_symbol_keys[token].add(key)

    for path, node in nodes.items():
        if node.kind == "other" or not node.token_set:
            continue
        refs = 0
        seen_targets: set[str] = set()
        candidate_keys = Counter()
        for token in node.token_set:
            for key in token_to_symbol_keys.get(token, set()):
                candidate_keys[key] += 1
        for key, matched_parts in candidate_keys.most_common(max_symbol_refs_per_file * 4):
            parts = symbol_parts.get(key)
            if not parts or matched_parts < len(parts):
                continue
            for target in sorted(symbol_index[key]):
                if target == path or target in seen_targets:
                    continue
                add_edge(graph, path, target, 1.0)
                seen_targets.add(target)
                refs += 1
                if refs >= max_symbol_refs_per_file:
                    break
            if refs >= max_symbol_refs_per_file:
                break


def add_source_test_edges(nodes: dict[str, FileNode], graph: dict[str, dict[str, float]]) -> None:
    sources_by_key: dict[str, list[str]] = defaultdict(list)
    tests_by_key: dict[str, list[str]] = defaultdict(list)
    for path, node in nodes.items():
        keys = canonical_test_source_keys(path)
        if not keys:
            continue
        for key in keys:
            if node.kind == "test":
                tests_by_key[key].append(path)
            elif node.kind == "source":
                sources_by_key[key].append(path)
    for key, tests in tests_by_key.items():
        sources = sources_by_key.get(key, [])
        for source in sources[:20]:
            for test in tests[:20]:
                add_edge(graph, source, test, 2.0)


def rank_files_for_sample(
    sample: dict[str, Any],
    query_text: str,
    index: RepoMapIndex,
    candidate_paths: set[str],
    query_weight: float,
    pagerank_weight: float,
    affinity_weight: float,
) -> list[tuple[str, float, dict[str, float]]]:
    query_scores = compute_query_scores(query_text, index)
    affinity_scores = compute_task_affinity(sample, query_text, index)
    personalization = merge_seed_scores(query_scores, affinity_scores)
    pagerank_scores = personalized_pagerank(index.graph, personalization)
    max_query = max(query_scores.values() or [0.0]) or 1.0
    max_pr = max(pagerank_scores.values() or [0.0]) or 1.0
    max_affinity = max((abs(value) for value in affinity_scores.values()), default=0.0) or 1.0
    ranked: list[tuple[str, float, dict[str, float]]] = []
    for path in candidate_paths:
        query_component = query_scores.get(path, 0.0) / max_query
        pagerank_component = pagerank_scores.get(path, 0.0) / max_pr
        affinity_component = affinity_scores.get(path, 0.0) / max_affinity
        final_score = query_weight * query_component + pagerank_weight * pagerank_component + affinity_weight * affinity_component
        ranked.append(
            (
                path,
                final_score,
                {
                    "query": query_component,
                    "pagerank": pagerank_component,
                    "affinity": affinity_component,
                },
            )
        )
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def compute_query_scores(query_text: str, index: RepoMapIndex) -> dict[str, float]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return {path: 0.0 for path in index.nodes}
    query_counts = Counter(query_tokens)
    lowered_query = query_text.lower()
    scores: dict[str, float] = {}
    for path, node in index.nodes.items():
        score = 0.0
        for token, count in query_counts.items():
            if token in node.token_set:
                weight = index.idf.get(token, 1.0)
                if token in node.path_tokens:
                    weight *= 3.0
                elif token in node.symbol_tokens:
                    weight *= 2.0
                score += (1.0 + math.log(count)) * weight
        lowered_path = path.lower()
        basename = PurePosixPath(path).name.lower()
        if lowered_path and lowered_path in lowered_query:
            score += 20.0
        if basename and basename in lowered_query:
            score += 7.0
        scores[path] = score / max(1.0, math.sqrt(len(node.token_set)))
    return scores


def compute_task_affinity(sample: dict[str, Any], query_text: str, index: RepoMapIndex) -> dict[str, float]:
    task_type = str(sample.get("task_type") or "")
    scores = {path: 0.0 for path in index.nodes}
    query_paths = paths_mentioned_by_query(sample, query_text, index.nodes.keys())
    for path in query_paths:
        scores[path] += 0.4
        for neighbor in index.graph.get(path, {}):
            scores[neighbor] += 1.0
    if task_type == "code2test":
        for path, node in index.nodes.items():
            if node.kind == "test":
                scores[path] += 1.0
    elif task_type == "comment2context":
        for path in given_files(sample):
            if path in scores:
                scores[path] -= 1.0
                for neighbor in index.graph.get(path, {}):
                    scores[neighbor] += 0.8
    elif task_type == "trace2code":
        for path, node in index.nodes.items():
            if node.kind == "source":
                scores[path] += 0.5
            elif node.kind == "test":
                scores[path] -= 0.2
    return scores


def paths_mentioned_by_query(sample: dict[str, Any], query_text: str, paths: Iterable[str]) -> set[str]:
    lowered_query = query_text.lower()
    path_set = set(paths)
    explicit: set[str] = set()
    query = sample.get("query") or {}
    for key in ("changed_file", "given_file", "path"):
        value = query.get(key)
        if isinstance(value, str):
            explicit.add(value)
    for key in ("implementation_files", "changed_files"):
        values = query.get(key)
        if isinstance(values, list):
            explicit.update(str(value) for value in values if value)
    mentioned = {path for path in explicit if path in path_set}
    for path in path_set:
        lowered_path = path.lower()
        if lowered_path in lowered_query:
            mentioned.add(path)
    return mentioned


def merge_seed_scores(query_scores: dict[str, float], affinity_scores: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for path, score in query_scores.items():
        merged[path] = max(0.0, score) + max(0.0, affinity_scores.get(path, 0.0))
    if not any(value > 0 for value in merged.values()):
        return {path: 1.0 for path in query_scores}
    return merged


def personalized_pagerank(
    graph: dict[str, dict[str, float]],
    personalization: dict[str, float],
    damping: float = 0.85,
    iterations: int = 24,
) -> dict[str, float]:
    nodes = list(graph)
    if not nodes:
        return {}
    total_personalization = sum(max(0.0, personalization.get(node, 0.0)) for node in nodes)
    if total_personalization <= 0:
        base = {node: 1.0 / len(nodes) for node in nodes}
    else:
        base = {node: max(0.0, personalization.get(node, 0.0)) / total_personalization for node in nodes}
    ranks = dict(base)
    out_weight = {node: sum(neighbors.values()) for node, neighbors in graph.items()}
    for _ in range(iterations):
        next_ranks = {node: (1.0 - damping) * base[node] for node in nodes}
        dangling = sum(ranks[node] for node in nodes if out_weight.get(node, 0.0) <= 0)
        if dangling:
            for node in nodes:
                next_ranks[node] += damping * dangling * base[node]
        for node, neighbors in graph.items():
            total = out_weight.get(node, 0.0)
            if total <= 0:
                continue
            share = damping * ranks.get(node, 0.0) / total
            for neighbor, weight in neighbors.items():
                next_ranks[neighbor] += share * weight
        ranks = next_ranks
    return ranks


def ranked_file_chunks(ranked: list[tuple[str, float, dict[str, float]]], nodes: dict[str, FileNode]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "kind": "file",
            "symbol": "",
            "text": nodes[path].text or path,
        }
        for path, _, _ in ranked
        if path in nodes
    ]


def format_file_score(item: tuple[str, float, dict[str, float]]) -> dict[str, Any]:
    path, score, components = item
    return {
        "path": path,
        "score": score,
        "query": components["query"],
        "pagerank": components["pagerank"],
        "affinity": components["affinity"],
    }


def candidate_paths_for_filter(nodes: dict[str, FileNode], candidate_filter: str) -> set[str]:
    validate_candidate_filter(candidate_filter)
    if candidate_filter == "all_files":
        return set(nodes)
    filtered = filter_candidate_chunks([{"path": path} for path in nodes], candidate_filter)
    return {str(item["path"]) for item in filtered}


def extract_import_specs(text: str) -> list[str]:
    specs: list[str] = []
    for line in text.splitlines()[:2000]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "# ", "*")):
            continue
        for match in IMPORT_RE.finditer(stripped):
            for group in match.groups():
                if group:
                    specs.append(group)
                    break
    return specs[:200]


def build_path_index(nodes: dict[str, FileNode]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for path in nodes:
        pure = PurePosixPath(path)
        no_ext = str(pure.with_suffix(""))
        parts = no_ext.split("/")
        variants = {
            no_ext,
            no_ext.replace("/", "."),
            pure.name,
            pure.stem,
        }
        for width in range(2, min(5, len(parts)) + 1):
            suffix = parts[-width:]
            variants.add("/".join(suffix))
            variants.add(".".join(suffix))
        for variant in variants:
            if variant:
                index[variant.lower()].add(path)
    return index


def resolve_import_spec(spec: str, from_path: str, path_index: dict[str, set[str]]) -> list[str]:
    cleaned = spec.strip().strip(";,").replace("::", "/").replace(".", "/")
    candidates: list[str] = []
    if spec.startswith("."):
        base = PurePosixPath(from_path).parent
        relative = base / spec
        if not relative.name:
            return []
        normalized = str(relative.with_suffix(""))
        keys = [normalized.lower(), f"{normalized}/index".lower(), f"{normalized}/mod".lower()]
    else:
        slash = cleaned.strip("/")
        keys = [slash.lower(), slash.replace("/", ".").lower()]
    for key in keys:
        candidates.extend(sorted(path_index.get(key, [])))
    return sorted(set(candidates))[:8]


def add_edge(graph: dict[str, dict[str, float]], left: str, right: str, weight: float) -> None:
    if left not in graph or right not in graph or left == right:
        return
    graph[left][right] = graph[left].get(right, 0.0) + weight
    graph[right][left] = graph[right].get(left, 0.0) + weight


def classify_file_kind(path: str) -> str:
    fake = {"path": path}
    if filter_candidate_chunks([fake], "tests_only"):
        return "test"
    if filter_candidate_chunks([fake], "code_only"):
        return "source"
    return "other"


def is_informative_symbol(symbol: str) -> bool:
    key = normalize_identifier(symbol)
    return bool(key and len(key) >= 4 and key not in GENERIC_SYMBOLS and not key.isdigit())


def normalize_identifier(value: str) -> str:
    compact = "".join(tokenize(value))
    if len(compact) < 4 or compact in GENERIC_SYMBOLS or compact.isdigit():
        return ""
    return compact


def canonical_test_source_keys(path: str) -> set[str]:
    pure = PurePosixPath(path)
    stem = pure.stem.lower()
    for suffix in ("_test", "test", "tests", ".test", ".spec", "_spec"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for prefix in ("test_", "tests_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    if len(stem) < 3:
        return set()
    parent_parts = [part.lower() for part in pure.parent.parts if part.lower() not in TEST_MARKERS]
    module = parent_parts[-1] if parent_parts else ""
    keys = {stem}
    if module:
        keys.add(f"{module}:{stem}")
    return keys


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
