from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .bcy_curve import DEFAULT_RUNS, budget_label
from .io import ensure_parent, read_json, utc_now, write_json
from .model_report import format_metric

DEFAULT_BCY_REPORT_PATH = Path("data/reports/v1_4/bcy_budget_curve.json")
DEFAULT_CONTEXT_SELECTION_PATH = Path("data/eval/v1_4/context_selection_leaderboard_top3.json")
DEFAULT_CONTEXT_COST_PATH = Path("data/reports/v1_4/context_acquisition_cost.json")
DEFAULT_OUT_PATH = Path("data/reports/v1_4/layered_leaderboard.json")
DEFAULT_MARKDOWN_OUT_PATH = Path("data/reports/v1_4/layered_leaderboard.md")


def report_layered_leaderboard(
    bcy_report_path: Path = DEFAULT_BCY_REPORT_PATH,
    context_selection_path: Path = DEFAULT_CONTEXT_SELECTION_PATH,
    context_cost_path: Path = DEFAULT_CONTEXT_COST_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    markdown_out_path: Path = DEFAULT_MARKDOWN_OUT_PATH,
    runs: Iterable[tuple[str, str, Path]] = DEFAULT_RUNS,
) -> dict[str, Any]:
    bcy_report = read_json(bcy_report_path, {})
    context_selection = read_json(context_selection_path, {"rows": []})
    context_cost = read_json(context_cost_path, {"agents": [], "joins": []})

    rank_rows = build_rank_rows(runs, bcy_report, context_selection)
    bcy_rows = sorted(
        rank_rows,
        key=lambda row: (none_last(row.get("BCY@8000")), str(row["method"])),
        reverse=True,
    )
    context_selection_rows = list(context_selection.get("rows") or [])
    complementarity_rows = build_complementarity_rows(context_cost)

    report = {
        "generated_at": utc_now(),
        "source_reports": {
            "bcy_budget_curve": str(bcy_report_path),
            "context_selection_at3": str(context_selection_path),
            "context_acquisition_cost": str(context_cost_path),
        },
        "metric_notes": {
            "R@20": "File-level Recall@20 from stored ranked lists.",
            "MRR": "Mean reciprocal rank of the first labeled gold file.",
            "BCY@B": "Budgeted Context Yield under the canonical packing protocol in bcy_budget_curve.json.",
            "ContextSelect-F1@3": "Compact file-selection F1 for top-3 retrieved files or logged agent final context.",
            "Complementarity@20": "A trajectory-conditioned 2x2 matrix for a specific (retriever, agent) pair: both hit, retriever-only, agent-only, and both miss.",
            "PES@20": "Potential Exploration Savings upper-bound proxy: max(0, first_gold_step - 1) when the retriever hits gold in top 20. It is not a causal saved-step estimate.",
            "Waste": "Intentionally excluded from primary tables because unjudged useful files would be counted as waste under incomplete gold labels.",
        },
        "rank_leaderboard": sorted(
            rank_rows,
            key=lambda row: (none_last(row.get("MRR")), none_last(row.get("Recall@20")), str(row["method"])),
            reverse=True,
        ),
        "bcy_leaderboard": bcy_rows,
        "context_selection_at3": context_selection_rows,
        "agent_cost": list(context_cost.get("agents") or []),
        "trajectory_complementarity": complementarity_rows,
        "paths": {
            "json": str(out_path),
            "markdown": str(markdown_out_path),
        },
    }
    write_json(out_path, report)
    ensure_parent(markdown_out_path)
    markdown_out_path.write_text(render_layered_leaderboard_markdown(report), encoding="utf-8")
    return report


def build_rank_rows(
    runs: Iterable[tuple[str, str, Path]],
    bcy_report: dict[str, Any],
    context_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    bcy_by_label = {str(run.get("label")): run for run in bcy_report.get("runs") or []}
    context_f1_by_method = {
        str(row.get("method")): row.get("file_f1")
        for row in context_selection.get("rows") or []
        if row.get("type") == "retriever"
    }
    rows = []
    for label, family, details_path in runs:
        summary_path = infer_summary_path(details_path)
        summary = read_json(summary_path, {})
        metrics = (summary.get("metrics") or {}).get("overall") or {}
        bcy_overall = (bcy_by_label.get(label) or {}).get("overall") or {}
        if not metrics and not bcy_overall:
            continue
        rows.append(
            {
                "method": label,
                "type": family,
                "samples": int(summary.get("evaluated") or metrics.get("samples") or bcy_overall.get("samples") or 0),
                "summary_path": str(summary_path),
                "details_path": str(details_path),
                "Recall@20": metrics.get("Recall@20"),
                "MRR": metrics.get("MRR"),
                "ContextSelect-F1@3": context_f1_by_method.get(label),
                **{
                    key: bcy_overall.get(key)
                    for key in sorted(bcy_overall)
                    if key.startswith("BCY@")
                },
            }
        )
    return rows


def build_complementarity_rows(context_cost: dict[str, Any], depth: int = 20) -> list[dict[str, Any]]:
    key = f"@{depth}"
    rows = []
    for join in context_cost.get("joins") or []:
        comp = (join.get("complementarity") or {}).get(key) or {}
        if not comp:
            continue
        rows.append(
            {
                "agent": join.get("agent"),
                "retriever": join.get("retriever"),
                "samples": comp.get("samples") or join.get("samples"),
                "depth": depth,
                "both_hit_rate": comp.get("both_hit_rate"),
                "retriever_only_rate": comp.get("retriever_only_rate"),
                "agent_only_rate": comp.get("agent_only_rate"),
                "both_miss_rate": comp.get("both_miss_rate"),
                "both_hit_count": comp.get("both_hit_count"),
                "retriever_only_count": comp.get("retriever_only_count"),
                "agent_only_count": comp.get("agent_only_count"),
                "both_miss_count": comp.get("both_miss_count"),
                "potential_exploration_savings": comp.get("potential_exploration_savings"),
                "retriever_only_given_agent_miss": comp.get("retriever_only_given_agent_miss"),
                "agent_only_given_retriever_miss": comp.get("agent_only_given_retriever_miss"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("agent") or ""),
            none_last(row.get("potential_exploration_savings")),
            none_last(row.get("retriever_only_given_agent_miss")),
            str(row.get("retriever") or ""),
        ),
        reverse=True,
    )
    return rows


def render_layered_leaderboard_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Layered Leaderboard",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "- Primary columns: `R@20`, `MRR`, `BCY@8k`, and `ContextSelect-F1@3`.",
        "- `BCY@B` uses the canonical token packing protocol in `bcy_budget_curve.*`.",
        "- `PES@20` is a trajectory-conditioned upper-bound proxy for gold-localization delay, not a causal saved-step estimate.",
        "- Waste-style precision diagnostics are intentionally excluded from primary tables because gold labels are not exhaustive over every useful file.",
        "",
        "## Main Leaderboard (ranked by MRR)",
        "",
        "| Rank | Method | Type | R@20 | MRR | BCY@8k | ContextSelect-F1@3 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(report["rank_leaderboard"], start=1):
        lines.append(
            "| {rank} | {method} | {type} | {recall} | {mrr} | {bcy} | {context_f1} |".format(
                rank=index,
                method=row["method"],
                type=row["type"],
                recall=fmt(row.get("Recall@20")),
                mrr=fmt(row.get("MRR")),
                bcy=fmt(row.get("BCY@8000")),
                context_f1=fmt(row.get("ContextSelect-F1@3")),
            )
        )

    budgets = sorted(
        {
            int(key.removeprefix("BCY@"))
            for row in report["bcy_leaderboard"]
            for key in row
            if key.startswith("BCY@")
        }
    )
    lines.extend(
        [
            "",
            "## BCY Budget Curve (ranked by BCY@8k)",
            "",
            "| Rank | Method | Type | " + " | ".join(f"BCY@{budget_label(budget)}" for budget in budgets) + " |",
            "|---:|---|---|" + "|".join("---:" for _ in budgets) + "|",
        ]
    )
    for index, row in enumerate(report["bcy_leaderboard"], start=1):
        values = " | ".join(fmt(row.get(f"BCY@{budget}")) for budget in budgets)
        lines.append(f"| {index} | {row['method']} | {row['type']} | {values} |")

    lines.extend(
        [
            "",
            "## Context Selection @3 (ranked by File F1)",
            "",
            "| Rank | Method | Type | Context | Avg files | File R | File P | File F1 |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    context_rows = sorted(
        report["context_selection_at3"],
        key=lambda row: (none_last(row.get("file_f1")), str(row.get("method") or "")),
        reverse=True,
    )
    for index, row in enumerate(context_rows, start=1):
        lines.append(
            "| {rank} | {method} | {type} | {context} | {avg_files} | {recall} | {precision} | {f1} |".format(
                rank=index,
                method=row.get("method"),
                type=row.get("type"),
                context=row.get("context"),
                avg_files=fmt(row.get("avg_final_files")),
                recall=fmt(row.get("file_recall")),
                precision=fmt(row.get("file_precision")),
                f1=fmt(row.get("file_f1")),
            )
        )

    lines.extend(
        [
            "",
            "## Agent Trajectory Cost",
            "",
            "| Agent | Events/sample | Any-gold | Miss rate | Late-hit rate | Median first hit |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["agent_cost"]:
        lines.append(
            "| {agent} | {events} | {any_gold} | {miss} | {late} | {median} |".format(
                agent=row.get("label"),
                events=fmt(row.get("events_per_sample_mean")),
                any_gold=fmt(row.get("any_gold_rate")),
                miss=fmt(row.get("miss_rate")),
                late=fmt(row.get("late_hit_rate")),
                median=fmt(row.get("median_first_hit_step")),
            )
        )

    lines.extend(
        [
            "",
            "## Trajectory-Conditioned Complementarity @20",
            "",
            "Rows are defined for a specific `(retriever, agent)` pair. The four hit/miss columns sum to one over the shared sample set.",
            "",
        ]
    )
    for agent in sorted({str(row.get("agent")) for row in report["trajectory_complementarity"]}):
        agent_rows = [
            row for row in report["trajectory_complementarity"]
            if str(row.get("agent")) == agent
        ]
        agent_rows.sort(
            key=lambda row: (none_last(row.get("potential_exploration_savings")), str(row.get("retriever") or "")),
            reverse=True,
        )
        lines.extend(
            [
                f"### {agent}",
                "",
                "| Rank | Retriever | Both hit | Retriever only | Agent only | Both miss | PES@20 | Retriever-only / agent-miss |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for index, row in enumerate(agent_rows, start=1):
            lines.append(
                "| {rank} | {retriever} | {both} | {retriever_only} | {agent_only} | {miss} | {pes} | {conditional} |".format(
                    rank=index,
                    retriever=row.get("retriever"),
                    both=fmt(row.get("both_hit_rate")),
                    retriever_only=fmt(row.get("retriever_only_rate")),
                    agent_only=fmt(row.get("agent_only_rate")),
                    miss=fmt(row.get("both_miss_rate")),
                    pes=fmt(row.get("potential_exploration_savings")),
                    conditional=fmt(row.get("retriever_only_given_agent_miss")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def infer_summary_path(details_path: Path) -> Path:
    name = details_path.name
    if name.endswith("_details.jsonl"):
        return details_path.with_name(name[: -len("_details.jsonl")] + "_summary.json")
    return details_path.with_suffix(".json")


def none_last(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return format_metric(float(value))
