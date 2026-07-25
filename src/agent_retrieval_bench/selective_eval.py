from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, TextIO

from .baseline import (
    CANDIDATE_FILTERS,
    RANKERS,
    filter_candidate_chunks,
    gold_file_ranks,
    hard_negative_files,
    load_corpus_manifest,
    query_has_leakage,
    query_text_for_eval,
    rank_chunks_bm25_with_scores,
    rank_chunks_with_scores,
    sample_metrics,
    target_gold_files,
    unique_ranked_paths,
    validate_ranker,
)
from .curate import filter_samples, load_keep_ids
from .io import ensure_parent, read_jsonl
from .progress import ProgressReporter


def evaluate_selective_lexical_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    sweep_path: Path | None = None,
    report_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    candidate_filter: str = "all_files",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    return evaluate_selective_baseline(
        sample_paths=sample_paths,
        corpus_dir=corpus_dir,
        out_path=out_path,
        details_path=details_path,
        sweep_path=sweep_path,
        report_path=report_path,
        keep_list=keep_list,
        limit_samples=limit_samples,
        candidate_filter=candidate_filter,
        ranker="lexical",
        progress=progress,
        progress_stream=progress_stream,
    )


def evaluate_selective_baseline(
    sample_paths: Iterable[Path],
    corpus_dir: Path,
    out_path: Path | None = None,
    details_path: Path | None = None,
    sweep_path: Path | None = None,
    report_path: Path | None = None,
    keep_list: Path | None = None,
    limit_samples: int | None = None,
    candidate_filter: str = "all_files",
    ranker: str = "lexical",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    if candidate_filter not in CANDIDATE_FILTERS:
        raise ValueError(f"unknown candidate_filter: {candidate_filter}")
    validate_ranker(ranker)
    reporter = ProgressReporter(progress, progress_stream)
    reporter.message(f"loading corpus manifest: {corpus_dir / 'corpus_manifest.jsonl'}")
    manifest = load_corpus_manifest(corpus_dir)
    keep_ids = load_keep_ids(keep_list)
    samples = []
    for sample in filter_samples(_iter_samples(sample_paths), keep_ids):
        if limit_samples and len(samples) >= limit_samples:
            break
        samples.append(sample)
    reporter.message(f"loaded benchmark samples: {len(samples)}")

    skipped = Counter()
    pending_by_chunks_path: dict[Path, list[tuple[int, dict[str, Any], list[str], str, bool]]] = defaultdict(list)
    sample_bar = reporter.bar("preparing selective samples", len(samples))
    for sample_index, sample in enumerate(samples):
        gold_files = target_gold_files(sample)
        is_no_gold = bool((sample.get("gold") or {}).get("no_gold") is True)
        if not gold_files and not is_no_gold:
            skipped["no_gold_unlabeled"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        query_text = query_text_for_eval(sample)
        if query_has_leakage(sample, query_text):
            skipped["query_leakage"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        chunks_path = manifest.get((sample.get("repo"), sample.get("base_commit")))
        if not chunks_path:
            skipped["missing_corpus"] += 1
            sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
            continue
        pending_by_chunks_path[chunks_path].append((sample_index, sample, gold_files, query_text, is_no_gold))
        sample_bar.update(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")
    sample_bar.finish(suffix=f"pending={sum(len(v) for v in pending_by_chunks_path.values())} skipped={sum(skipped.values())}")

    details: list[dict[str, Any]] = []
    eval_bar = reporter.bar("ranking selective samples", sum(len(v) for v in pending_by_chunks_path.values()))
    for corpus_index, (chunks_path, pending) in enumerate(pending_by_chunks_path.items(), start=1):
        reporter.message(f"loading corpus {corpus_index}/{len(pending_by_chunks_path)}: {chunks_path}")
        chunks = filter_candidate_chunks(read_jsonl(chunks_path), candidate_filter)
        if not chunks:
            skipped["empty_corpus"] += len(pending)
            eval_bar.update(step=len(pending), suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
            continue
        for sample_index, sample, gold_files, query_text, is_no_gold in pending:
            detail = selective_detail(sample_index, sample, gold_files, query_text, chunks, candidate_filter, is_no_gold, ranker=ranker)
            details.append(detail)
            eval_bar.update(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    eval_bar.finish(suffix=f"evaluated={len(details)} skipped={sum(skipped.values())}")
    details.sort(key=lambda item: item.pop("_sample_index", 0))

    sweep = threshold_sweep(details)
    summary = summarize_selective(details, sweep, dict(skipped), time.monotonic() - started_at, candidate_filter, keep_list, ranker=ranker)
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if details_path:
        _write_jsonl(details_path, details)
    if sweep_path:
        _write_jsonl(sweep_path, sweep)
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(render_selective_report(summary), encoding="utf-8")
    return summary


def selective_detail(
    sample_index: int,
    sample: dict[str, Any],
    gold_files: list[str],
    query_text: str,
    chunks: list[dict[str, Any]],
    candidate_filter: str,
    is_no_gold: bool,
    ranker: str = "lexical",
) -> dict[str, Any]:
    scored = rank_chunks_with_scores_for_ranker(query_text, chunks, ranker)
    ranked = [chunk for _, chunk in scored]
    top_score = float(scored[0][0]) if scored else 0.0
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    top_files = unique_ranked_paths(ranked)[:20]
    detail: dict[str, Any] = {
        "_sample_index": sample_index,
        "sample_id": sample.get("id"),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "candidate_filter": candidate_filter,
        "ranker": ranker,
        "label": "no_gold" if is_no_gold else "positive",
        "no_gold_reason": (sample.get("gold") or {}).get("reason") if is_no_gold else None,
        "gold_files": gold_files,
        "confidence": top_score,
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": top_score - second_score,
        "top_files": top_files,
    }
    if gold_files:
        hard_negatives = hard_negative_files(sample)
        metrics = sample_metrics(gold_files, ranked, hard_negative_files=hard_negatives)
        detail.update(
            {
                "metrics": metrics,
                "gold_ranks": gold_file_ranks(gold_files, ranked),
                "hard_negative_files": hard_negatives,
                "hard_negative_ranks": gold_file_ranks(hard_negatives, ranked),
            }
        )
    return detail


def rank_chunks_with_scores_for_ranker(query_text: str, chunks: list[dict[str, Any]], ranker: str) -> list[tuple[float, dict[str, Any]]]:
    if ranker == "lexical":
        return rank_chunks_with_scores(query_text, chunks)
    if ranker == "bm25":
        return rank_chunks_bm25_with_scores(query_text, chunks)
    raise ValueError(f"Unknown ranker {ranker!r}. Expected one of: {', '.join(RANKERS)}")


def threshold_sweep(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not details:
        return []
    scores = sorted({float(row.get("confidence") or 0.0) for row in details})
    thresholds = [-math.inf]
    thresholds.extend((a + b) / 2.0 for a, b in zip(scores, scores[1:]))
    thresholds.append(scores[-1] + max(1e-9, abs(scores[-1]) * 1e-9))
    rows = [threshold_metrics(details, threshold) for threshold in thresholds]
    for index, row in enumerate(rows):
        row["threshold_index"] = index
    return rows


def threshold_metrics(details: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    cm = Counter()
    accepted_positive_recalls: list[float] = []
    accepted_positive_hits = 0
    abstained = 0
    accepted = 0
    for row in details:
        is_no_gold = row.get("label") == "no_gold"
        is_positive = not is_no_gold
        predict_abstain = float(row.get("confidence") or 0.0) < threshold
        if predict_abstain:
            abstained += 1
        else:
            accepted += 1
        if is_no_gold and predict_abstain:
            cm["no_gold_abstained"] += 1
        elif is_no_gold and not predict_abstain:
            cm["no_gold_returned"] += 1
        elif is_positive and predict_abstain:
            cm["positive_abstained"] += 1
        else:
            cm["positive_returned"] += 1
            recall20 = float((row.get("metrics") or {}).get("Recall@20") or 0.0)
            accepted_positive_recalls.append(recall20)
            if recall20 > 0.0:
                accepted_positive_hits += 1
    positive_total = cm["positive_abstained"] + cm["positive_returned"]
    no_gold_total = cm["no_gold_abstained"] + cm["no_gold_returned"]
    total = positive_total + no_gold_total
    positive_pass_rate = cm["positive_returned"] / positive_total if positive_total else 0.0
    no_gold_abstain_rate = cm["no_gold_abstained"] / no_gold_total if no_gold_total else 0.0
    abstain_precision = cm["no_gold_abstained"] / (cm["no_gold_abstained"] + cm["positive_abstained"]) if (cm["no_gold_abstained"] + cm["positive_abstained"]) else 0.0
    abstain_f1 = _f1(abstain_precision, no_gold_abstain_rate)
    selective_accuracy = (cm["no_gold_abstained"] + cm["positive_returned"]) / total if total else 0.0
    balanced_accuracy = (positive_pass_rate + no_gold_abstain_rate) / 2.0 if positive_total and no_gold_total else selective_accuracy
    selective_success_at20 = (cm["no_gold_abstained"] + accepted_positive_hits) / total if total else 0.0
    return {
        "threshold": threshold,
        "threshold_display": "-inf" if threshold == -math.inf else threshold,
        "total": total,
        "positive_total": positive_total,
        "no_gold_total": no_gold_total,
        "accepted": accepted,
        "abstained": abstained,
        "coverage": accepted / total if total else 0.0,
        "positive_pass_rate": positive_pass_rate,
        "positive_false_abstain_rate": 1.0 - positive_pass_rate if positive_total else 0.0,
        "no_gold_abstain_rate": no_gold_abstain_rate,
        "no_gold_false_return_rate": 1.0 - no_gold_abstain_rate if no_gold_total else 0.0,
        "abstain_precision": abstain_precision,
        "abstain_f1": abstain_f1,
        "selective_accuracy": selective_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "selective_success@20": selective_success_at20,
        "accepted_positive_recall@20": mean(accepted_positive_recalls) if accepted_positive_recalls else 0.0,
        "accepted_positive_hit@20": accepted_positive_hits / cm["positive_returned"] if cm["positive_returned"] else 0.0,
        "confusion": dict(cm),
    }


def summarize_selective(
    details: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
    skipped: dict[str, int],
    wall_time_seconds: float,
    candidate_filter: str,
    keep_list: Path | None,
    ranker: str = "lexical",
) -> dict[str, Any]:
    positive = [row for row in details if row.get("label") == "positive"]
    no_gold = [row for row in details if row.get("label") == "no_gold"]
    best_accuracy = max(sweep, key=lambda row: (row["selective_accuracy"], row["balanced_accuracy"], row["selective_success@20"])) if sweep else {}
    best_balanced = max(sweep, key=lambda row: (row["balanced_accuracy"], row["selective_accuracy"], row["selective_success@20"])) if sweep else {}
    best_success = max(sweep, key=lambda row: (row["selective_success@20"], row["balanced_accuracy"], row["selective_accuracy"])) if sweep else {}
    return {
        "mode": f"selective_{ranker}",
        "ranker": ranker,
        "candidate_filter": candidate_filter,
        "keep_list": str(keep_list) if keep_list and keep_list.exists() else None,
        "evaluated": len(details),
        "positive_evaluated": len(positive),
        "no_gold_evaluated": len(no_gold),
        "skipped": skipped,
        "confidence_summary": {
            "positive": confidence_summary(positive),
            "no_gold": confidence_summary(no_gold),
        },
        "positive_retrieval_metrics": average_positive_metrics(positive),
        "operating_points": {
            "no_abstain": sweep[0] if sweep else {},
            "best_selective_accuracy": compact_threshold_row(best_accuracy),
            "best_balanced_accuracy": compact_threshold_row(best_balanced),
            "best_selective_success@20": compact_threshold_row(best_success),
            "target_no_gold_abstain_80": compact_threshold_row(best_with_constraint(sweep, lambda row: row["no_gold_abstain_rate"] >= 0.80, "positive_pass_rate")),
            "target_positive_pass_90": compact_threshold_row(best_with_constraint(sweep, lambda row: row["positive_pass_rate"] >= 0.90, "no_gold_abstain_rate")),
        },
        "threshold_sweep_count": len(sweep),
        "runtime": {"wall_time_seconds": wall_time_seconds},
    }


def compact_threshold_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    keys = (
        "threshold",
        "threshold_display",
        "threshold_index",
        "coverage",
        "positive_pass_rate",
        "no_gold_abstain_rate",
        "abstain_precision",
        "abstain_f1",
        "selective_accuracy",
        "balanced_accuracy",
        "selective_success@20",
        "accepted_positive_recall@20",
        "accepted_positive_hit@20",
        "confusion",
    )
    return {key: row.get(key) for key in keys if key in row}


def best_with_constraint(sweep: list[dict[str, Any]], predicate: Any, optimize_key: str) -> dict[str, Any]:
    candidates = [row for row in sweep if predicate(row)]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: (row.get(optimize_key) or 0.0, row["balanced_accuracy"], row["selective_accuracy"]))


def confidence_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = sorted(float(row.get("confidence") or 0.0) for row in rows)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "p10": percentile(values, 0.10),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "max": values[-1],
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def average_positive_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0.0}
    keys = sorted({key for row in rows for key in (row.get("metrics") or {})})
    output = {"samples": float(len(rows))}
    for key in keys:
        output[key] = mean(float((row.get("metrics") or {}).get(key) or 0.0) for row in rows)
    return output


def render_selective_report(summary: dict[str, Any]) -> str:
    ranker = str(summary.get("ranker") or "lexical")
    lines = [f"# Selective {ranker} Threshold Report", ""]
    lines.append(f"Evaluated: {summary['evaluated']} ({summary['positive_evaluated']} positive, {summary['no_gold_evaluated']} no-gold)")
    lines.append(f"Skipped: `{summary.get('skipped', {})}`")
    lines.append("")
    lines.append("## Confidence Summary")
    lines.append("")
    for label, row in summary["confidence_summary"].items():
        if not row.get("count"):
            lines.append(f"- {label}: count 0")
        else:
            lines.append(
                f"- {label}: count {row['count']}, median {row['median']:.4f}, p10 {row['p10']:.4f}, p90 {row['p90']:.4f}"
            )
    lines.append("")
    lines.append("## Operating Points")
    lines.append("")
    lines.append("| point | threshold | selective acc | balanced acc | no-gold abstain | positive pass | success@20 | coverage |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, row in summary["operating_points"].items():
        if not row:
            continue
        threshold = row.get("threshold_display", row.get("threshold"))
        if threshold == "-inf":
            threshold_text = "-inf"
        else:
            threshold_text = f"{float(threshold):.6g}"
        lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                name,
                threshold_text,
                float(row.get("selective_accuracy") or 0.0),
                float(row.get("balanced_accuracy") or 0.0),
                float(row.get("no_gold_abstain_rate") or 0.0),
                float(row.get("positive_pass_rate") or 0.0),
                float(row.get("selective_success@20") or 0.0),
                float(row.get("coverage") or 0.0),
            )
        )
    lines.append("")
    lines.append("## Positive Retrieval Without Abstention")
    lines.append("")
    metrics = summary.get("positive_retrieval_metrics") or {}
    for key in ("Recall@5", "Recall@10", "Recall@20", "MRR", "gold_coverage@8k", "Precision@20", "context_efficiency@8k"):
        if key in metrics:
            lines.append(f"- {key}: {float(metrics[key]):.4f}")
    lines.append("")
    return "\n".join(lines)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _iter_samples(sample_paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sample_paths:
        yield from read_jsonl(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
