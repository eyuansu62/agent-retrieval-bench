from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json
from .model_report import format_metric, task_sort_key

DEFAULT_BUDGETS = (4_000, 8_000, 16_000, 32_000)
DEFAULT_COVERAGE_THRESHOLDS = (1, 16, 32, 64, 128)
TOKENIZER_NAME = "regex_code_tokenizer_v1"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^A-Za-z0-9_\s]", re.UNICODE)

DEFAULT_RUNS = (
    ("Qwen3-Embedding-4B", "embedding", Path("data/eval/v1_3_reviewed/Qwen3-Embedding-4B_details.jsonl")),
    ("Qwen3-Embedding-8B", "embedding", Path("data/eval/v1_3_reviewed/Qwen3-Embedding-8B_details.jsonl")),
    ("pplx-embed-v1-4b", "embedding", Path("data/eval/v1_3_reviewed/pplx-embed-v1-4b_details.jsonl")),
    ("RepoMap", "repo map", Path("data/eval/v1_3_reviewed/repomap_all_files_details.jsonl")),
    ("jina-code-embeddings-0.5b", "embedding", Path("data/eval/v1_3_reviewed/jina-code-embeddings-0.5b_details.jsonl")),
    ("nomic-embed-code", "embedding", Path("data/eval/v1_3_reviewed/nomic-embed-code_details.jsonl")),
    ("Lexical", "lexical", Path("data/eval/v1_3_reviewed/lexical_all_files_details.jsonl")),
    ("BM25", "lexical", Path("data/eval/v1_3_reviewed/bm25_all_files_details.jsonl")),
    ("Grep strict", "grep", Path("data/eval/v1_4/v1_3_grep_strict_details.jsonl")),
    ("RRF(Qwen3-8B + RepoMap)", "rank fusion", Path("data/eval/v1_4/rank_fusion/rrf_qwen8b_repomap_details.jsonl")),
    (
        "RRF(Qwen3-8B + Qwen3-4B + RepoMap)",
        "rank fusion",
        Path("data/eval/v1_4/rank_fusion/rrf_qwen8b_qwen4b_repomap_details.jsonl"),
    ),
    ("RRF(Qwen3-4B + RepoMap)", "rank fusion", Path("data/eval/v1_4/rank_fusion/rrf_qwen4b_repomap_details.jsonl")),
)


def report_bcy_budget_curve(
    corpus_manifest_path: Path = Path("data/corpus/v1_2/corpus_manifest.jsonl"),
    out_path: Path = Path("data/reports/v1_4/bcy_budget_curve.json"),
    markdown_out_path: Path = Path("data/reports/v1_4/bcy_budget_curve.md"),
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    runs: Iterable[tuple[str, str, Path]] = DEFAULT_RUNS,
    coverage_thresholds: Iterable[int] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    budgets = tuple(sorted({int(budget) for budget in budgets if int(budget) > 0}))
    if not budgets:
        raise ValueError("At least one positive budget is required")
    coverage_thresholds = tuple(sorted({int(value) for value in coverage_thresholds if int(value) > 0}))
    if not coverage_thresholds:
        raise ValueError("At least one positive coverage threshold is required")
    if 1 not in coverage_thresholds:
        coverage_thresholds = (1, *coverage_thresholds)
    manifest = load_corpus_manifest(corpus_manifest_path)
    corpus_cache = CorpusFileCache(manifest)
    run_results = []
    for label, family, details_path in runs:
        if not details_path.exists():
            continue
        details = read_jsonl(details_path)
        run_results.append(
            evaluate_run(label, family, details_path, details, corpus_cache, budgets, coverage_thresholds)
        )

    report = {
        "generated_at": utc_now(),
        "corpus_manifest": str(corpus_manifest_path),
        "tokenizer": TOKENIZER_NAME,
        "budgets": list(budgets),
        "coverage_thresholds": list(coverage_thresholds),
        "packing_protocol": {
            "rank_source": "stored top_files from each details row; chunk-level systems are file-deduplicated before packing",
            "canonical_renderer": "### {path}\\n{corpus file text}\\n",
            "file_text_source": "kind=file rows in the released corpus chunk files",
            "boundary_rule": "greedy by file rank; if a file exceeds the remaining budget, include a token-boundary prefix and stop",
            "coverage_rule": (
                "for threshold tau, gold file g is covered when at least min(tau, L_g) non-header content "
                "tokens are packed, where L_g is its available corpus-text length; canonical BCY uses tau=1"
            ),
            "known_limitations": [
                "Released corpus file chunks are truncated by the corpus builder, so long files use the available canonical corpus text.",
                "Current details artifacts store top-20 file lists, so BCY at larger budgets is computed from the stored top-20 prefix.",
                "BCY is recall-like over labeled gold files and does not require treating unjudged files as waste.",
            ],
        },
        "runs": run_results,
        "paths": {"json": str(out_path), "markdown": str(markdown_out_path)},
    }
    write_json(out_path, report)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_bcy_markdown(report), encoding="utf-8")
    return report


def load_corpus_manifest(path: Path) -> dict[tuple[str, str], Path]:
    manifest = {}
    for row in read_jsonl(path):
        if row.get("status") and row.get("status") != "ok":
            continue
        repo = str(row.get("repo") or "")
        base_commit = str(row.get("base_commit") or "")
        chunks_path = row.get("chunks_path")
        if repo and base_commit and chunks_path:
            manifest[(repo, base_commit)] = Path(str(chunks_path))
    return manifest


class CorpusFileCache:
    def __init__(self, manifest: dict[tuple[str, str], Path]) -> None:
        self.manifest = manifest
        self._cache: dict[Path, dict[str, str]] = {}

    def file_text(self, repo: str, base_commit: str, path: str) -> str | None:
        chunks_path = self.manifest.get((repo, base_commit))
        if chunks_path is None:
            return None
        if chunks_path not in self._cache:
            self._cache[chunks_path] = load_file_texts(chunks_path)
        return self._cache[chunks_path].get(path)


def load_file_texts(chunks_path: Path) -> dict[str, str]:
    file_texts = {}
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "file" and row.get("path"):
                file_texts[str(row["path"])] = str(row.get("text") or "")
    return file_texts


def evaluate_run(
    label: str,
    family: str,
    details_path: Path,
    details: list[dict[str, Any]],
    corpus_cache: CorpusFileCache,
    budgets: tuple[int, ...],
    coverage_thresholds: tuple[int, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    rows = []
    for detail in details:
        gold_files = [str(path) for path in detail.get("gold_files") or [] if path]
        if not gold_files:
            continue
        rows.append(evaluate_sample(detail, gold_files, corpus_cache, budgets, coverage_thresholds))
    return {
        "label": label,
        "family": family,
        "details_path": str(details_path),
        "samples": len(rows),
        "overall": summarize_rows(rows, budgets, coverage_thresholds),
        "by_task": {
            task: summarize_rows(task_rows, budgets, coverage_thresholds)
            for task, task_rows in sorted(group_rows_by_task(rows).items(), key=lambda item: task_sort_key(item[0]))
        },
    }


def evaluate_sample(
    detail: dict[str, Any],
    gold_files: list[str],
    corpus_cache: CorpusFileCache,
    budgets: tuple[int, ...],
    coverage_thresholds: tuple[int, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    repo = str(detail.get("repo") or "")
    base_commit = str(detail.get("base_commit") or "")
    ranked_files = dedupe_paths(str(path) for path in detail.get("top_files") or [])
    packed = {
        budget: pack_files(
            repo,
            base_commit,
            ranked_files,
            set(gold_files),
            corpus_cache,
            budget,
            coverage_thresholds,
        )
        for budget in budgets
    }
    return {
        "sample_id": str(detail.get("sample_id") or ""),
        "task_type": str(detail.get("task_type") or "unknown"),
        "repo": repo,
        "base_commit": base_commit,
        "gold_files": gold_files,
        "ranked_files": ranked_files,
        "packed": packed,
    }


def pack_files(
    repo: str,
    base_commit: str,
    ranked_files: list[str],
    gold_files: set[str],
    corpus_cache: CorpusFileCache,
    budget: int,
    coverage_thresholds: tuple[int, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    used = 0
    covered_gold = {threshold: set() for threshold in coverage_thresholds}
    packed_files = 0
    partial_files = 0
    missing_ranked_files = 0
    for path in ranked_files:
        text = corpus_cache.file_text(repo, base_commit, path)
        if text is None:
            missing_ranked_files += 1
            continue
        header_tokens = count_tokens(f"### {path}\n")
        content_tokens = count_tokens(text)
        separator_tokens = 1 if text else 0
        total_tokens = header_tokens + content_tokens + separator_tokens
        if total_tokens <= 0:
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        packed_files += 1
        if total_tokens <= remaining:
            used += total_tokens
            included_content_tokens = content_tokens
        else:
            partial_files += 1
            included_after_header = max(0, remaining - header_tokens)
            included_content_tokens = min(content_tokens, included_after_header)
            used = budget
        if path in gold_files and content_tokens > 0:
            for threshold in coverage_thresholds:
                if included_content_tokens >= min(threshold, content_tokens):
                    covered_gold[threshold].add(path)
        if used >= budget:
            break
    return {
        "budget": budget,
        "used_tokens": used,
        "bcy": (len(covered_gold[1]) / len(gold_files)) if gold_files else 0.0,
        "covered_gold_count": len(covered_gold[1]),
        "bcy_by_min_content_tokens": {
            str(threshold): (len(covered_gold[threshold]) / len(gold_files)) if gold_files else 0.0
            for threshold in coverage_thresholds
        },
        "covered_gold_count_by_min_content_tokens": {
            str(threshold): len(covered_gold[threshold]) for threshold in coverage_thresholds
        },
        "gold_count": len(gold_files),
        "packed_files": packed_files,
        "partial_files": partial_files,
        "missing_ranked_files": missing_ranked_files,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    budgets: tuple[int, ...],
    coverage_thresholds: tuple[int, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, **{f"BCY@{budget}": 0.0 for budget in budgets}}
    summary: dict[str, Any] = {"samples": len(rows)}
    sensitivity: dict[str, dict[str, float]] = {}
    for budget in budgets:
        packed_rows = [row["packed"][budget] for row in rows]
        summary[f"BCY@{budget}"] = mean(float(row["bcy"]) for row in packed_rows)
        summary[f"used_tokens@{budget}"] = mean(float(row["used_tokens"]) for row in packed_rows)
        summary[f"packed_files@{budget}"] = mean(float(row["packed_files"]) for row in packed_rows)
        summary[f"partial_files@{budget}"] = mean(float(row["partial_files"]) for row in packed_rows)
        summary[f"missing_ranked_files@{budget}"] = mean(float(row["missing_ranked_files"]) for row in packed_rows)
        sensitivity[str(budget)] = {}
        for threshold in coverage_thresholds:
            sensitivity[str(budget)][str(threshold)] = mean(
                float(row["bcy_by_min_content_tokens"][str(threshold)]) for row in packed_rows
            )
    summary["coverage_sensitivity"] = sensitivity
    return summary


def group_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_type") or "unknown")].append(row)
    return grouped


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def dedupe_paths(paths: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            output.append(path)
            seen.add(path)
    return output


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def render_bcy_markdown(report: dict[str, Any]) -> str:
    budgets = [int(budget) for budget in report["budgets"]]
    coverage_thresholds = [int(value) for value in report.get("coverage_thresholds") or [1]]
    lines = [
        "# BCY Budget Curve",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Corpus manifest: `{report['corpus_manifest']}`",
        f"- Tokenizer: `{report['tokenizer']}`",
        "- Protocol: file-deduplicate ranked results, render canonical corpus file text with a path header, greedily pack by rank, and prefix-truncate at the budget boundary.",
        "- Canonical coverage: a gold file counts when at least one non-header content token is packed.",
        "- Sensitivity coverage: at threshold tau, require at least min(tau, available file-content tokens).",
        "- Caveat: this report uses released corpus `kind=file` text and stored top-20 file lists.",
        "",
        "## Overall Curve",
        "",
        "| Rank | Method | Family | Samples | " + " | ".join(f"BCY@{budget_label(budget)}" for budget in budgets) + " |",
        "|---:|---|---|---:|" + "|".join("---:" for _ in budgets) + "|",
    ]
    ranked = sorted(report["runs"], key=lambda run: (-float(run["overall"].get("BCY@8000", 0.0)), str(run["label"])))
    for index, run in enumerate(ranked, start=1):
        values = " | ".join(format_metric(run["overall"].get(f"BCY@{budget}", 0.0)) for budget in budgets)
        lines.append(f"| {index} | {run['label']} | {run['family']} | {run['samples']} | {values} |")

    for task in sorted({task for run in report["runs"] for task in (run.get("by_task") or {})}, key=task_sort_key):
        lines.extend(
            [
                "",
                f"## {task}",
                "",
                "| Rank | Method | Family | Samples | " + " | ".join(f"BCY@{budget_label(budget)}" for budget in budgets) + " |",
                "|---:|---|---|---:|" + "|".join("---:" for _ in budgets) + "|",
            ]
        )
        task_runs = [
            run
            for run in report["runs"]
            if task in (run.get("by_task") or {})
        ]
        task_runs.sort(key=lambda run: (-float(run["by_task"][task].get("BCY@8000", 0.0)), str(run["label"])))
        for index, run in enumerate(task_runs, start=1):
            summary = run["by_task"][task]
            values = " | ".join(format_metric(summary.get(f"BCY@{budget}", 0.0)) for budget in budgets)
            lines.append(f"| {index} | {run['label']} | {run['family']} | {summary['samples']} | {values} |")

    lines.extend(
        [
            "",
            "## Packing Diagnostics",
            "",
            "| Method | Used@8k | Packed files@8k | Partial files@8k | Missing ranked files@8k |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run in ranked:
        overall = run["overall"]
        lines.append(
            "| {label} | {used} | {files} | {partial} | {missing} |".format(
                label=run["label"],
                used=format_metric(overall.get("used_tokens@8000", 0.0)),
                files=format_metric(overall.get("packed_files@8000", 0.0)),
                partial=format_metric(overall.get("partial_files@8000", 0.0)),
                missing=format_metric(overall.get("missing_ranked_files@8000", 0.0)),
            )
        )
    if 8000 in budgets:
        lines.extend(
            [
                "",
                "## Minimum-content Threshold Sensitivity at 8k",
                "",
                "| Method | " + " | ".join(f"tau={threshold}" for threshold in coverage_thresholds) + " |",
                "|---|" + "|".join("---:" for _ in coverage_thresholds) + "|",
            ]
        )
        for run in ranked:
            overall = run["overall"]
            values = " | ".join(
                format_metric((overall.get("coverage_sensitivity") or {}).get("8000", {}).get(str(threshold), 0.0))
                for threshold in coverage_thresholds
            )
            lines.append(f"| {run['label']} | {values} |")
    return "\n".join(lines).rstrip() + "\n"


def budget_label(budget: int) -> str:
    if budget >= 1000 and budget % 1000 == 0:
        return f"{budget // 1000}k"
    return str(budget)
