from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .bcy_curve import CorpusFileCache, evaluate_sample, load_corpus_manifest
from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json

DEFAULT_BCY_REPORT = Path("data/reports/v1_4/bcy_budget_curve.json")
DEFAULT_CONTEXT_COST_REPORT = Path("data/reports/v1_4/context_acquisition_cost.json")
DEFAULT_PES_CALIBRATION_REPORT = Path("data/reports/v1_4/pes_calibration/calibration_report.json")
DEFAULT_OUT = Path("data/reports/v1_4/cae_validity_correlation.json")
DEFAULT_MARKDOWN_OUT = Path("data/reports/v1_4/cae_validity_correlation.md")
DEFAULT_BUDGET = 8000
PREFERRED_RETRIEVERS = {
    "Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B",
    "RepoMap",
    "Grep strict",
    "RRF(Qwen3-8B + RepoMap)",
    "RRF(Qwen3-8B + Qwen3-4B + RepoMap)",
    "RRF(Qwen3-4B + RepoMap)",
}


def report_cae_validity(
    *,
    bcy_report_path: Path = DEFAULT_BCY_REPORT,
    context_cost_path: Path = DEFAULT_CONTEXT_COST_REPORT,
    pes_calibration_path: Path = DEFAULT_PES_CALIBRATION_REPORT,
    out_path: Path = DEFAULT_OUT,
    markdown_out_path: Path = DEFAULT_MARKDOWN_OUT,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    bcy_report = read_json(bcy_report_path)
    context_cost = read_json(context_cost_path)
    pes_report = read_json(pes_calibration_path) if pes_calibration_path.exists() else None

    bcy_samples = compute_bcy_samples(bcy_report, budget)
    agent_samples = compute_agent_samples(context_cost)
    correlation_rows = []
    bucket_rows_output = []
    for retriever_label, retriever_samples in sorted(bcy_samples.items()):
        for agent_label, samples in sorted(agent_samples.items()):
            common_ids = sorted(set(retriever_samples) & set(samples))
            joined = [join_sample(retriever_samples[sample_id], samples[sample_id]) for sample_id in common_ids]
            correlation_rows.append(summarize_correlations(retriever_label, agent_label, joined))
            bucket_rows_output.append(summarize_buckets(retriever_label, agent_label, joined))

    result = {
        "mode": "cae_validity_correlation",
        "generated_at": utc_now(),
        "budget": budget,
        "paths": {
            "bcy_report": str(bcy_report_path),
            "context_cost_report": str(context_cost_path),
            "pes_calibration_report": str(pes_calibration_path),
            "json": str(out_path),
            "markdown": str(markdown_out_path),
        },
        "notes": [
            "BCY correlations are observational joins between a static retriever and logged agent trajectories; they support proxy validity, not causality.",
            "BCY uses the canonical token packing protocol from the BCY budget-curve report.",
            "PES calibration is a seeded intervention diagnostic and remains an upper-bound proxy for localization delay.",
            "Bootstrap confidence intervals are intentionally omitted in this report.",
        ],
        "bcy_correlations": correlation_rows,
        "bcy_bucket_sanity": bucket_rows_output,
        "pes_calibration": summarize_pes_calibration(pes_report),
    }
    write_json(out_path, result)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_markdown(result), encoding="utf-8")
    return result


def compute_bcy_samples(bcy_report: dict[str, Any], budget: int) -> dict[str, dict[str, dict[str, Any]]]:
    corpus_manifest = Path(str(bcy_report["corpus_manifest"]))
    corpus_cache = CorpusFileCache(load_corpus_manifest(corpus_manifest))
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for run in bcy_report.get("runs") or []:
        label = str(run.get("label") or "")
        details_path = Path(str(run.get("details_path") or ""))
        if not label or not details_path.exists():
            continue
        rows: dict[str, dict[str, Any]] = {}
        for detail in read_jsonl(details_path):
            gold_files = [str(path) for path in detail.get("gold_files") or [] if path]
            sample_id = str(detail.get("sample_id") or "")
            if not sample_id or not gold_files:
                continue
            sample = evaluate_sample(detail, gold_files, corpus_cache, (budget,))
            packed = sample["packed"][budget]
            rows[sample_id] = {
                "sample_id": sample_id,
                "task_type": sample["task_type"],
                "repo": sample["repo"],
                "bcy": optional_float(packed.get("bcy")),
                "covered_gold_count": optional_int(packed.get("covered_gold_count")),
                "gold_count": optional_int(packed.get("gold_count")),
                "used_tokens": optional_float(packed.get("used_tokens")),
            }
        output[label] = rows
    return output


def compute_agent_samples(context_cost: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for agent in context_cost.get("agents") or []:
        label = str(agent.get("label") or "")
        path = Path(str(agent.get("path") or ""))
        if not label or not path.exists():
            continue
        rows: dict[str, dict[str, Any]] = {}
        for detail in read_jsonl(path):
            sample_id = str(detail.get("sample_id") or "")
            if not sample_id:
                continue
            gold_files = set(str(path) for path in detail.get("gold_files") or [])
            steps = detail.get("steps") or []
            first_step = first_gold_step(steps, gold_files)
            final_hit = any(step.get("path") in gold_files and step.get("is_final_context") for step in steps)
            utilized_hit = any(step.get("path") in gold_files and step.get("is_utilized_context") for step in steps)
            metrics = detail.get("metrics") or {}
            rows[sample_id] = {
                "sample_id": sample_id,
                "task_type": str(detail.get("task_type") or "unknown"),
                "repo": str(detail.get("repo") or ""),
                "event_count": len(steps),
                "any_gold": 1.0 if first_step is not None else 0.0,
                "first_gold_step": first_step,
                "final_gold": 1.0 if final_hit else 0.0,
                "utilized_gold": 1.0 if utilized_hit else 0.0,
                "final_file_f1": optional_float(metrics.get("final_file_f1")),
                "retrieved_file_f1": optional_float(metrics.get("retrieved_file_f1")),
                "utilized_file_f1": optional_float(metrics.get("utilized_file_f1")),
            }
        output[label] = rows
    return output


def join_sample(retriever: dict[str, Any], agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": retriever["sample_id"],
        "task_type": retriever.get("task_type") or agent.get("task_type") or "unknown",
        "repo": retriever.get("repo") or agent.get("repo") or "",
        "bcy": retriever.get("bcy"),
        "covered_gold_count": retriever.get("covered_gold_count"),
        "gold_count": retriever.get("gold_count"),
        "used_tokens": retriever.get("used_tokens"),
        "any_gold": agent.get("any_gold"),
        "first_gold_step": agent.get("first_gold_step"),
        "event_count": agent.get("event_count"),
        "final_gold": agent.get("final_gold"),
        "utilized_gold": agent.get("utilized_gold"),
        "final_file_f1": agent.get("final_file_f1"),
        "retrieved_file_f1": agent.get("retrieved_file_f1"),
        "utilized_file_f1": agent.get("utilized_file_f1"),
    }


def summarize_correlations(retriever_label: str, agent_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "any_gold": "positive",
        "final_gold": "positive",
        "utilized_gold": "positive",
        "final_file_f1": "positive",
        "retrieved_file_f1": "positive",
        "utilized_file_f1": "positive",
        "first_gold_step_hit_only": "negative",
        "event_count": "negative",
    }
    pairs_by_metric: dict[str, list[tuple[float, float]]] = {}
    for metric in metrics:
        pairs_by_metric[metric] = bcy_pairs(rows, metric)
    return {
        "retriever": retriever_label,
        "agent": agent_label,
        "samples": len(rows),
        "mean_bcy": mean([row.get("bcy") for row in rows]),
        "agent_any_gold_rate": mean([row.get("any_gold") for row in rows]),
        "agent_final_gold_rate": mean([row.get("final_gold") for row in rows]),
        "agent_final_file_f1_mean": mean([row.get("final_file_f1") for row in rows]),
        "correlations": {
            metric: {
                "expected_direction": direction,
                "n": len(pairs),
                "pearson": pearson(pairs),
                "spearman": spearman(pairs),
            }
            for metric, direction in metrics.items()
            for pairs in [pairs_by_metric[metric]]
        },
    }


def bcy_pairs(rows: list[dict[str, Any]], metric: str) -> list[tuple[float, float]]:
    pairs = []
    for row in rows:
        bcy = optional_float(row.get("bcy"))
        if bcy is None:
            continue
        if metric == "first_gold_step_hit_only":
            value = optional_float(row.get("first_gold_step"))
        else:
            value = optional_float(row.get(metric))
        if value is not None:
            pairs.append((bcy, value))
    return pairs


def summarize_buckets(retriever_label: str, agent_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = []
    for name, bucket in bucket_by_metric(rows, "bcy"):
        first_steps = [row.get("first_gold_step") for row in bucket if row.get("first_gold_step") is not None]
        buckets.append(
            {
                "bucket": name,
                "samples": len(bucket),
                "mean_bcy": mean([row.get("bcy") for row in bucket]),
                "any_gold_rate": mean([row.get("any_gold") for row in bucket]),
                "final_gold_rate": mean([row.get("final_gold") for row in bucket]),
                "final_file_f1_mean": mean([row.get("final_file_f1") for row in bucket]),
                "event_count_mean": mean([row.get("event_count") for row in bucket]),
                "median_first_gold_step_hit_only": safe_median(first_steps),
            }
        )
    return {"retriever": retriever_label, "agent": agent_label, "buckets": buckets}


def bucket_by_metric(rows: list[dict[str, Any]], metric: str) -> list[tuple[str, list[dict[str, Any]]]]:
    clean = [row for row in rows if optional_float(row.get(metric)) is not None]
    clean.sort(key=lambda row: (float(row[metric]), str(row.get("sample_id") or "")))
    if not clean:
        return [("low", []), ("mid", []), ("high", [])]
    n = len(clean)
    first = n // 3
    second = (2 * n) // 3
    return [
        ("low", clean[:first]),
        ("mid", clean[first:second]),
        ("high", clean[second:]),
    ]


def summarize_pes_calibration(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    rows = [row for row in report.get("samples") or [] if row.get("has_control") and row.get("has_seeded")]
    pes_step_pairs = [
        (float(row["predicted_pes@20"]), float(row["first_hit_delta"]))
        for row in rows
        if row.get("predicted_pes@20") is not None and row.get("first_hit_delta") is not None
    ]
    pes_event_pairs = [
        (float(row["predicted_pes@20"]), float(row["event_saving"]))
        for row in rows
        if row.get("predicted_pes@20") is not None and row.get("event_saving") is not None
    ]
    pes_f1_pairs = [
        (float(row["predicted_pes@20"]), float(row["final_file_f1_delta"]))
        for row in rows
        if row.get("predicted_pes@20") is not None and row.get("final_file_f1_delta") is not None
    ]
    overall = report.get("overall") or {}
    return {
        "paired_samples": (report.get("counts") or {}).get("paired_samples"),
        "mean_predicted_pes": overall.get("predicted_pes_mean"),
        "mean_first_hit_delta_both_hit": overall.get("first_hit_delta_mean_both_hit"),
        "mean_event_saving": overall.get("event_saving_mean"),
        "mean_final_file_f1_delta": overall.get("final_file_f1_delta_mean"),
        "rescue_rate": overall.get("rescue_rate"),
        "lost_rate": overall.get("lost_rate"),
        "pes_vs_first_hit_delta": correlation_summary(pes_step_pairs, "positive"),
        "pes_vs_event_saving": correlation_summary(pes_event_pairs, "positive"),
        "pes_vs_final_file_f1_delta": correlation_summary(pes_f1_pairs, "positive"),
        "by_group": report.get("by_group") or {},
    }


def correlation_summary(pairs: list[tuple[float, float]], expected_direction: str) -> dict[str, Any]:
    return {
        "expected_direction": expected_direction,
        "n": len(pairs),
        "pearson": pearson(pairs),
        "spearman": spearman(pairs),
    }


def first_gold_step(steps: list[dict[str, Any]], gold_files: set[str]) -> int | None:
    for step in steps:
        if step.get("path") in gold_files and step.get("step") is not None:
            try:
                return int(step["step"])
            except (TypeError, ValueError):
                return None
    return None


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(x_var * y_var)


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    return pearson(list(zip(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))))


def ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        for _, index in ordered[i:j]:
            output[index] = rank
        i = j
    return output


def mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def safe_median(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(median(clean))


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CAE Validity Correlation Report",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Budget: `{result['budget']}` tokens",
        "- Interpretation: correlations support proxy validity, not causal attribution.",
        "- Bootstrap confidence intervals are intentionally omitted.",
        "",
        "## BCY vs Logged Agent Quantities",
        "",
        "Positive correlations are expected for gold-touch and File F1 metrics; negative correlations are expected for first-gold delay and event count.",
        "",
        "| Retriever | Agent | n | Mean BCY | Any-gold | Final F1 | rho(BCY, any-gold) | rho(BCY, final F1) | rho(BCY, first-step) | rho(BCY, events) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in preferred_correlation_rows(result["bcy_correlations"]):
        corr = row["correlations"]
        lines.append(
            "| {retriever} | {agent} | {n} | {mean_bcy} | {any_gold} | {f1} | {rho_any} | {rho_f1} | {rho_step} | {rho_events} |".format(
                retriever=row["retriever"],
                agent=row["agent"],
                n=row["samples"],
                mean_bcy=fmt(row["mean_bcy"]),
                any_gold=fmt(row["agent_any_gold_rate"]),
                f1=fmt(row["agent_final_file_f1_mean"]),
                rho_any=fmt(corr["any_gold"]["spearman"]),
                rho_f1=fmt(corr["final_file_f1"]["spearman"]),
                rho_step=fmt(corr["first_gold_step_hit_only"]["spearman"]),
                rho_events=fmt(corr["event_count"]["spearman"]),
            )
        )
    lines.extend(
        [
            "",
            "## BCY Bucket Sanity Check",
            "",
            "Rows split each `(retriever, agent)` pair into equal-count low/mid/high BCY buckets. A useful proxy should generally improve gold-touch or File F1 as BCY rises, though logged agent behavior can be noisy.",
            "",
            "| Retriever | Agent | Bucket | n | Mean BCY | Any-gold | Final F1 | Events | Median first step |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in preferred_bucket_rows(result["bcy_bucket_sanity"]):
        for bucket in row["buckets"]:
            lines.append(
                "| {retriever} | {agent} | {bucket} | {n} | {bcy} | {any_gold} | {f1} | {events} | {first} |".format(
                    retriever=row["retriever"],
                    agent=row["agent"],
                    bucket=bucket["bucket"],
                    n=bucket["samples"],
                    bcy=fmt(bucket["mean_bcy"]),
                    any_gold=fmt(bucket["any_gold_rate"]),
                    f1=fmt(bucket["final_file_f1_mean"]),
                    events=fmt(bucket["event_count_mean"]),
                    first=fmt(bucket["median_first_gold_step_hit_only"], 1),
                )
            )
    lines.extend(["", "## PES Calibration", ""])
    pes = result.get("pes_calibration")
    if pes:
        lines.extend(
            [
                f"- Paired samples: `{pes['paired_samples']}`",
                f"- Mean predicted PES@20: `{fmt(pes['mean_predicted_pes'])}`",
                f"- Mean first-hit delta among both-hit samples: `{fmt(pes['mean_first_hit_delta_both_hit'])}`",
                f"- Mean event saving: `{fmt(pes['mean_event_saving'])}`",
                f"- Mean final File F1 delta: `{fmt(pes['mean_final_file_f1_delta'])}`",
                f"- Rescue rate: `{fmt(pes['rescue_rate'])}`; lost rate: `{fmt(pes['lost_rate'])}`",
                "",
                "| Pair | n | Pearson | Spearman | Expected |",
                "| --- | ---: | ---: | ---: | --- |",
                pes_correlation_line("PES vs first-hit delta", pes["pes_vs_first_hit_delta"]),
                pes_correlation_line("PES vs event saving", pes["pes_vs_event_saving"]),
                pes_correlation_line("PES vs final File F1 delta", pes["pes_vs_final_file_f1_delta"]),
            ]
        )
    else:
        lines.append("_No PES calibration report found._")
    lines.extend(["", "## Paper-ready Summary", ""])
    lines.append(paper_ready_summary(result))
    return "\n".join(lines).rstrip() + "\n"


def preferred_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [row for row in rows if row["retriever"] in PREFERRED_RETRIEVERS]
    return sorted(preferred, key=lambda row: (row["agent"], row["retriever"]))


def preferred_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [row for row in rows if row["retriever"] in {"RRF(Qwen3-8B + RepoMap)", "Qwen3-Embedding-4B", "RepoMap"}]
    return sorted(preferred, key=lambda row: (row["agent"], row["retriever"]))


def pes_correlation_line(label: str, row: dict[str, Any]) -> str:
    return "| {label} | {n} | {pearson} | {spearman} | {direction} |".format(
        label=label,
        n=row["n"],
        pearson=fmt(row["pearson"]),
        spearman=fmt(row["spearman"]),
        direction=row["expected_direction"],
    )


def paper_ready_summary(result: dict[str, Any]) -> str:
    pes = result.get("pes_calibration") or {}
    pes_step = (pes.get("pes_vs_first_hit_delta") or {}).get("spearman")
    pes_event = (pes.get("pes_vs_event_saving") or {}).get("spearman")
    return (
        "We validate CAE as a proxy rather than a causal claim by correlating its components with logged agent-side quantities. "
        f"In the seeded PES calibration, PES@20 correlates with actual first-gold-step reduction (Spearman rho={fmt(pes_step)}) "
        f"and event savings (rho={fmt(pes_event)}). "
        "For BCY, we report observational correlations with gold-touch, final File F1, first-gold delay, and event count, plus tertile bucket checks."
    )
