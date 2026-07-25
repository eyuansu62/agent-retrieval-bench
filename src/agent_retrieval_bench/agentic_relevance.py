from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .bcy_curve import CorpusFileCache, DEFAULT_RUNS, evaluate_sample, load_corpus_manifest
from .io import ensure_parent, read_jsonl, utc_now, write_json
from .model_report import format_metric, task_sort_key

DEFAULT_SAMPLES = Path("data/benchmark/v1_3_reviewed/samples.jsonl")
DEFAULT_CORPUS_MANIFEST = Path("data/corpus/v1_2/corpus_manifest.jsonl")
DEFAULT_OUT_DIR = Path("data/reports/v1_4/agentic_relevance")
DEFAULT_BUDGET = 8000

LABELS = {
    "semantic_direct": "The query text directly names or describes terms in the gold file, so text/code similarity can plausibly retrieve it.",
    "structural_indirect": "The gold file is reached through repository structure such as neighboring modules, imports, symbols, or given/changed-file adjacency.",
    "workflow_conventional": "The gold file is relevant because of software workflow convention, such as implementation-to-test or review-to-test/context expectations.",
    "causal_indirect": "The query shows a symptom, failure, or visible frame, while the gold file is the root-cause implementation context.",
}

STOP_TOKENS = {
    "about",
    "after",
    "also",
    "before",
    "being",
    "build",
    "change",
    "changed",
    "class",
    "code",
    "could",
    "error",
    "file",
    "files",
    "from",
    "func",
    "function",
    "gold",
    "have",
    "here",
    "impl",
    "into",
    "make",
    "method",
    "module",
    "only",
    "path",
    "should",
    "source",
    "test",
    "tests",
    "that",
    "this",
    "using",
    "when",
    "with",
}

PATH_QUERY_KEYS = {"changed_file", "given_file", "path", "implementation_files"}
NON_PATH_QUERY_KEYS = {
    "pr_title",
    "pr_body",
    "review_comment",
    "diff_hunk_context",
    "failure_excerpt",
    "command",
    "changed_file_summary",
}


def report_agentic_relevance_taxonomy(
    *,
    samples_path: Path = DEFAULT_SAMPLES,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
    budget: int = DEFAULT_BUDGET,
    runs: Iterable[tuple[str, str, Path]] = DEFAULT_RUNS,
) -> dict[str, Any]:
    samples = read_jsonl(samples_path)
    annotations = [annotate_sample(sample) for sample in samples]
    annotation_by_id = {row["sample_id"]: row for row in annotations}

    leaderboard = compute_type_leaderboard(annotation_by_id, corpus_manifest_path, budget, runs)
    result = {
        "mode": "agentic_relevance_taxonomy",
        "generated_at": utc_now(),
        "samples_path": str(samples_path),
        "corpus_manifest": str(corpus_manifest_path),
        "budget": budget,
        "rubric": LABELS,
        "annotation_policy": {
            "annotator": "Codex rubric-based annotation requested by the authors",
            "primary_label": "dominant relation between the workflow query and the labeled gold files",
            "secondary_labels": "additional relation modes detected by the rubric",
            "caveat": "This is a qualitative mechanism analysis, not an independent human gold-correctness audit.",
        },
        "counts": summarize_counts(annotations),
        "leaderboard": leaderboard,
        "examples": select_examples(annotations),
        "paths": {
            "annotations_jsonl": str(out_dir / "annotations.jsonl"),
            "json": str(out_dir / "relevance_taxonomy_report.json"),
            "markdown": str(out_dir / "relevance_taxonomy_report.md"),
        },
    }

    write_annotations(out_dir / "annotations.jsonl", annotations)
    write_json(out_dir / "relevance_taxonomy_report.json", result)
    ensure_parent(out_dir / "relevance_taxonomy_report.md")
    (out_dir / "relevance_taxonomy_report.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def annotate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    task_type = str(sample.get("task_type") or "unknown")
    query = sample.get("query") or {}
    gold_files = extract_gold_files(sample)
    gold_evidence = extract_gold_evidence(sample)
    non_path_text = query_text(query, NON_PATH_QUERY_KEYS)
    path_text = query_text(query, PATH_QUERY_KEYS)
    query_tokens = tokenize(non_path_text)
    path_query_tokens = tokenize(path_text)
    gold_tokens = tokenize(" ".join(gold_files + gold_symbols(sample)))
    overlap = sorted((query_tokens & gold_tokens) - STOP_TOKENS)
    path_overlap = sorted((path_query_tokens & gold_tokens) - STOP_TOKENS)
    semantic_score = semantic_overlap_score(query_tokens, gold_tokens)
    structural_signals = structural_signal_count(sample, gold_files)
    workflow_signals = workflow_signal_count(sample, gold_files)

    labels: list[str] = []
    evidence: list[str] = []
    confidence = "medium"

    if task_type == "trace2code":
        labels.append("causal_indirect")
        evidence.append("trace2code maps visible failure output or test frames to root-cause source files")
        if structural_signals:
            labels.append("structural_indirect")
            evidence.append("root-cause paths are structurally adjacent to visible test/package paths")
        if overlap:
            labels.append("semantic_direct")
            evidence.append(f"query/gold token overlap: {', '.join(overlap[:8])}")
    elif task_type == "code2test":
        if semantic_score >= 0.10 or len(overlap) >= 2:
            labels.append("semantic_direct")
            evidence.append(f"PR text overlaps target test terms: {', '.join(overlap[:8])}")
        labels.append("workflow_conventional")
        evidence.append("code2test relevance follows implementation/PR signal to related tests")
        if structural_signals or path_overlap:
            labels.append("structural_indirect")
            evidence.append("test path shares repository module/path structure with changed implementation context")
    elif task_type == "comment2context":
        if semantic_score >= 0.12 or len(overlap) >= 2 or "symbol_or_path_overlap" in gold_evidence:
            labels.append("semantic_direct")
            if overlap:
                evidence.append(f"review/query text overlaps target context terms: {', '.join(overlap[:8])}")
            else:
                evidence.append("gold evidence records symbol/path overlap")
        if workflow_signals or any(is_test_path(path) for path in gold_files):
            labels.append("workflow_conventional")
            evidence.append("review comment requests tests or context implied by review workflow")
        if structural_signals or not labels:
            labels.append("structural_indirect")
            evidence.append("reviewed file points to additional same-module or dependency context")
    else:
        labels.append("semantic_direct" if overlap else "structural_indirect")
        confidence = "low"
        evidence.append("fallback label for unknown task type")

    labels = dedupe(labels)
    primary = choose_primary_label(task_type, labels, semantic_score, structural_signals, workflow_signals)
    secondary = [label for label in labels if label != primary]
    if primary == "semantic_direct" and len(overlap) < 2 and semantic_score < 0.10:
        confidence = "low"
    elif task_type == "trace2code" and primary == "causal_indirect":
        confidence = "high"
    elif task_type in {"code2test", "comment2context"} and len(labels) > 1:
        confidence = "medium"

    return {
        "sample_id": str(sample.get("id") or sample.get("sample_id") or ""),
        "task_type": task_type,
        "repo": str(sample.get("repo") or ""),
        "primary_relevance_type": primary,
        "secondary_relevance_types": secondary,
        "confidence": confidence,
        "gold_files": gold_files,
        "query_token_overlap": overlap[:12],
        "path_token_overlap": path_overlap[:12],
        "semantic_overlap_score": semantic_score,
        "structural_signal_count": structural_signals,
        "workflow_signal_count": workflow_signals,
        "evidence": evidence,
    }


def choose_primary_label(
    task_type: str,
    labels: list[str],
    semantic_score: float,
    structural_signals: int,
    workflow_signals: int,
) -> str:
    if task_type == "trace2code":
        return "causal_indirect"
    if task_type == "code2test":
        if "semantic_direct" in labels and semantic_score >= 0.30:
            return "semantic_direct"
        return "workflow_conventional"
    if task_type == "comment2context":
        if "semantic_direct" in labels and semantic_score >= 0.16:
            return "semantic_direct"
        if "workflow_conventional" in labels and workflow_signals > structural_signals:
            return "workflow_conventional"
        if "structural_indirect" in labels:
            return "structural_indirect"
        return labels[0]
    return labels[0]


def extract_gold_files(sample: dict[str, Any]) -> list[str]:
    task_type = str(sample.get("task_type") or "")
    paths = []
    for span in sample.get("gold_spans") or []:
        if span.get("path"):
            paths.append(str(span["path"]))
    gold = sample.get("gold") or {}
    if gold.get("no_gold") is True:
        return []
    for path in gold.get("files") or []:
        if path:
            paths.append(str(path))
    if task_type == "code2test":
        keys = ("related_tests",)
    elif task_type == "trace2code":
        keys = ("root_cause_files",)
    elif task_type == "comment2context":
        keys = ("root_cause_files", "supporting_files")
    else:
        keys = ("related_tests", "root_cause_files", "supporting_files")
    for key in keys:
        for path in gold.get(key) or []:
            paths.append(str(path))
    if task_type in {"comment2context", ""}:
        for item in gold.get("must_context_files") or []:
            if item.get("path"):
                paths.append(str(item["path"]))
    return dedupe(paths)


def extract_gold_evidence(sample: dict[str, Any]) -> set[str]:
    evidence = set()
    gold = sample.get("gold") or {}
    for item in gold.get("must_context_files") or []:
        evidence.update(str(value) for value in item.get("evidence") or [])
    metadata_evidence = (sample.get("metadata") or {}).get("evidence") or {}
    if isinstance(metadata_evidence, dict):
        for value in metadata_evidence.get("signals") or []:
            evidence.add(str(value))
        if metadata_evidence.get("source"):
            evidence.add(str(metadata_evidence["source"]))
    return evidence


def structural_signal_count(sample: dict[str, Any], gold_files: list[str]) -> int:
    query = sample.get("query") or {}
    evidence = extract_gold_evidence(sample)
    count = 0
    count += sum(1 for key in ("same_module_context", "symbol_or_path_overlap", "source_test_path_overlap") if key in evidence)
    anchors = []
    for key in ("changed_file", "given_file", "path"):
        value = query.get(key)
        if isinstance(value, str):
            anchors.append(value)
    for value in query.get("implementation_files") or []:
        anchors.append(str(value))
    for anchor in anchors:
        for gold in gold_files:
            if same_module_family(anchor, gold):
                count += 1
    return count


def workflow_signal_count(sample: dict[str, Any], gold_files: list[str]) -> int:
    task_type = str(sample.get("task_type") or "")
    evidence = extract_gold_evidence(sample)
    count = 0
    if task_type == "code2test":
        count += 2
    count += sum(
        1
        for key in (
            "same_pr_changed_tests",
            "pr_level_changed_implementation_and_existing_tests",
            "behavior_test_for_reviewed_change",
            "review_requests_tests",
            "implementation_context_for_reviewed_change",
        )
        if key in evidence
    )
    if any(is_test_path(path) for path in gold_files):
        count += 1
    return count


def compute_type_leaderboard(
    annotations: dict[str, dict[str, Any]],
    corpus_manifest_path: Path,
    budget: int,
    runs: Iterable[tuple[str, str, Path]],
) -> list[dict[str, Any]]:
    corpus_cache = CorpusFileCache(load_corpus_manifest(corpus_manifest_path))
    output = []
    for label, family, details_path in runs:
        if not details_path.exists():
            continue
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detail in read_jsonl(details_path):
            sample_id = str(detail.get("sample_id") or detail.get("id") or "")
            annotation = annotations.get(sample_id)
            if not annotation:
                continue
            gold_files = [str(path) for path in detail.get("gold_files") or [] if path]
            if not gold_files:
                continue
            bcy = evaluate_sample(detail, gold_files, corpus_cache, (budget,))["packed"][budget]["bcy"]
            groups[annotation["primary_relevance_type"]].append(
                {
                    "sample_id": sample_id,
                    "task_type": annotation["task_type"],
                    "recall20": optional_float((detail.get("metrics") or {}).get("Recall@20")),
                    "mrr": optional_float((detail.get("metrics") or {}).get("MRR")),
                    "bcy": optional_float(bcy),
                }
            )
        for relevance_type, rows in sorted(groups.items()):
            output.append(
                {
                    "method": label,
                    "family": family,
                    "relevance_type": relevance_type,
                    "samples": len(rows),
                    "R@20": mean(row["recall20"] for row in rows),
                    "MRR": mean(row["mrr"] for row in rows),
                    f"BCY@{budget}": mean(row["bcy"] for row in rows),
                    "by_task": summarize_rows_by_task(rows, budget),
                }
            )
    return output


def summarize_rows_by_task(rows: list[dict[str, Any]], budget: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    return {
        task: {
            "samples": len(task_rows),
            "R@20": mean(row["recall20"] for row in task_rows),
            "MRR": mean(row["mrr"] for row in task_rows),
            f"BCY@{budget}": mean(row["bcy"] for row in task_rows),
        }
        for task, task_rows in sorted(grouped.items(), key=lambda item: task_sort_key(item[0]))
    }


def summarize_counts(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(row["primary_relevance_type"] for row in annotations)
    by_task_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in annotations:
        by_task_type[row["task_type"]][row["primary_relevance_type"]] += 1
    return {
        "samples": len(annotations),
        "by_primary_relevance_type": dict(sorted(by_type.items())),
        "by_task_and_primary_relevance_type": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(by_task_type.items(), key=lambda item: task_sort_key(item[0]))
        },
        "by_confidence": dict(sorted(Counter(row["confidence"] for row in annotations).items())),
    }


def select_examples(annotations: list[dict[str, Any]], per_type: int = 4) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        grouped[row["primary_relevance_type"]].append(row)
    for relevance_type, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (row["confidence"] != "high", -row["semantic_overlap_score"], row["sample_id"]))
        output[relevance_type] = [
            {
                "sample_id": row["sample_id"],
                "task_type": row["task_type"],
                "repo": row["repo"],
                "gold_files": row["gold_files"][:3],
                "evidence": row["evidence"][:2],
            }
            for row in rows[:per_type]
        ]
    return output


def write_annotations(path: Path, annotations: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in annotations:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Agentic Relevance Taxonomy Report",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Samples: `{result['counts']['samples']}`",
        f"- Annotation policy: {result['annotation_policy']['annotator']}.",
        f"- Caveat: {result['annotation_policy']['caveat']}",
        "",
        "## Rubric",
        "",
    ]
    for label, definition in result["rubric"].items():
        lines.append(f"- `{label}`: {definition}")
    lines.extend(["", "## Distribution", "", "| Primary type | Samples |", "| --- | ---: |"])
    for label, count in sorted(result["counts"]["by_primary_relevance_type"].items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "### By Task", "", "| Task | " + " | ".join(f"`{label}`" for label in LABELS) + " |", "| --- |" + "|".join("---:" for _ in LABELS) + "|"])
    for task, counts in result["counts"]["by_task_and_primary_relevance_type"].items():
        values = " | ".join(str(counts.get(label, 0)) for label in LABELS)
        lines.append(f"| `{task}` | {values} |")

    lines.extend(["", "## Leaderboard by Relevance Type", ""])
    for relevance_type in LABELS:
        rows = [row for row in result["leaderboard"] if row["relevance_type"] == relevance_type]
        if not rows:
            continue
        rows.sort(key=lambda row: (-float(row["MRR"] or 0.0), -float(row["R@20"] or 0.0), row["method"]))
        lines.extend(
            [
                f"### `{relevance_type}`",
                "",
                "| Rank | Method | Family | n | R@20 | MRR | BCY@8k |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(rows[:8], start=1):
            lines.append(
                "| {rank} | {method} | {family} | {n} | {r20} | {mrr} | {bcy} |".format(
                    rank=rank,
                    method=row["method"],
                    family=row["family"],
                    n=row["samples"],
                    r20=format_metric(row["R@20"]),
                    mrr=format_metric(row["MRR"]),
                    bcy=format_metric(row.get("BCY@8000")),
                )
            )
        lines.append("")

    lines.extend(["## Examples", ""])
    for relevance_type, examples in result["examples"].items():
        lines.append(f"### `{relevance_type}`")
        lines.append("")
        for row in examples:
            lines.append(
                "- `{sample_id}` ({task}, {repo}) -> {gold}: {evidence}".format(
                    sample_id=row["sample_id"],
                    task=row["task_type"],
                    repo=row["repo"],
                    gold=", ".join(row["gold_files"]),
                    evidence="; ".join(row["evidence"]),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def query_text(query: dict[str, Any], keys: set[str]) -> str:
    pieces = []
    for key, value in query.items():
        if key not in keys:
            continue
        pieces.extend(flatten_text(value))
    return "\n".join(pieces)


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten_text(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(flatten_text(item))
        return output
    return [str(value)]


def gold_symbols(sample: dict[str, Any]) -> list[str]:
    symbols = []
    for block in sample.get("gold_blocks") or []:
        if block.get("symbol"):
            symbols.append(str(block["symbol"]))
    for value in (sample.get("gold") or {}).get("root_cause_symbols") or []:
        symbols.append(str(value))
    return symbols


def semantic_overlap_score(query_tokens: set[str], gold_tokens: set[str]) -> float:
    query_clean = query_tokens - STOP_TOKENS
    gold_clean = gold_tokens - STOP_TOKENS
    if not query_clean or not gold_clean:
        return 0.0
    overlap = query_clean & gold_clean
    return len(overlap) / max(1, min(len(query_clean), len(gold_clean)))


def same_module_family(anchor: str, gold: str) -> bool:
    anchor_parts = [part for part in Path(anchor).parts if part not in {"src", "test", "tests"}]
    gold_parts = [part for part in Path(gold).parts if part not in {"src", "test", "tests"}]
    if not anchor_parts or not gold_parts:
        return False
    if anchor_parts[0] == gold_parts[0]:
        return True
    anchor_stems = set(tokenize(Path(anchor).stem))
    gold_stems = set(tokenize(Path(gold).stem))
    return bool((anchor_stems & gold_stems) - STOP_TOKENS)


def is_test_path(path: str) -> bool:
    parts = set(Path(path).parts)
    stem = Path(path).stem.lower()
    return "test" in parts or "tests" in parts or stem.endswith("_test") or stem.startswith("test_")


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    normalized = normalized.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", normalized)
        if token.lower() not in STOP_TOKENS
    }


def dedupe(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
