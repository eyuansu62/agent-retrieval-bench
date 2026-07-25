from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .io import read_jsonl, utc_now, write_json

DEFAULT_AGENT_DETAILS = (
    ("OpenAI GPT-5.4-mini strict", Path("data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl")),
    ("Codex CLI GPT-5.4", Path("data/eval/v1_4/codex_cli_gpt54_corpus_full_budgeted_answer_recovered_all_details.jsonl")),
    ("Codex CLI GPT-5.5", Path("data/eval/v1_4/codex_cli_gpt55_corpus_full_budgeted_answer_recovered_all_details.jsonl")),
)
DEFAULT_RETRIEVER_DETAILS = (
    ("Qwen3-Embedding-4B", Path("data/eval/v1_3_reviewed/Qwen3-Embedding-4B_details.jsonl")),
    ("Qwen3-Embedding-8B", Path("data/eval/v1_3_reviewed/Qwen3-Embedding-8B_details.jsonl")),
    ("pplx-embed-v1-4b", Path("data/eval/v1_3_reviewed/pplx-embed-v1-4b_details.jsonl")),
    ("RepoMap", Path("data/eval/v1_3_reviewed/repomap_all_files_details.jsonl")),
    ("jina-code-embeddings-0.5b", Path("data/eval/v1_3_reviewed/jina-code-embeddings-0.5b_details.jsonl")),
    ("nomic-embed-code", Path("data/eval/v1_3_reviewed/nomic-embed-code_details.jsonl")),
    ("Lexical", Path("data/eval/v1_3_reviewed/lexical_all_files_details.jsonl")),
    ("Grep strict", Path("data/eval/v1_4/v1_3_grep_strict_details.jsonl")),
    ("RRF(Qwen3-8B + RepoMap)", Path("data/eval/v1_4/rank_fusion/rrf_qwen8b_repomap_details.jsonl")),
    (
        "RRF(Qwen3-8B + Qwen3-4B + RepoMap)",
        Path("data/eval/v1_4/rank_fusion/rrf_qwen8b_qwen4b_repomap_details.jsonl"),
    ),
    ("RRF(Qwen3-4B + RepoMap)", Path("data/eval/v1_4/rank_fusion/rrf_qwen4b_repomap_details.jsonl")),
)
DEFAULT_DEPTHS = (3, 5, 10, 20)


def report_context_acquisition_cost(
    out_path: Path = Path("data/reports/v1_4/context_acquisition_cost.json"),
    markdown_out_path: Path = Path("data/reports/v1_4/context_acquisition_cost.md"),
    agent_details: Iterable[tuple[str, Path]] | None = None,
    retriever_details: Iterable[tuple[str, Path]] | None = None,
    late_hit_threshold: int = 3,
    depths: Iterable[int] = DEFAULT_DEPTHS,
) -> dict[str, Any]:
    agent_rows = [(label, path, read_jsonl(path)) for label, path in (agent_details or DEFAULT_AGENT_DETAILS)]
    retriever_rows = [(label, path, read_jsonl(path)) for label, path in (retriever_details or DEFAULT_RETRIEVER_DETAILS)]
    depths = tuple(sorted(set(depths)))

    agents = []
    agent_by_label: dict[str, dict[str, Any]] = {}
    for label, path, rows in agent_rows:
        summary, per_sample = summarize_agent_trajectory(label, path, rows, late_hit_threshold)
        agents.append(summary)
        agent_by_label[label] = {"summary": summary, "per_sample": per_sample}

    retrievers = []
    retriever_by_label = {}
    for label, path, rows in retriever_rows:
        summary, per_sample = summarize_retriever(label, path, rows, depths)
        retrievers.append(summary)
        retriever_by_label[label] = {"summary": summary, "per_sample": per_sample}

    joins = []
    for agent_label, agent_data in agent_by_label.items():
        for retriever_label, retriever_data in retriever_by_label.items():
            joins.append(
                summarize_join(
                    agent_label=agent_label,
                    agent_samples=agent_data["per_sample"],
                    retriever_label=retriever_label,
                    retriever_samples=retriever_data["per_sample"],
                    depths=depths,
                    late_hit_threshold=late_hit_threshold,
                )
            )

    result = {
        "generated_at": utc_now(),
        "late_hit_threshold": late_hit_threshold,
        "depths": list(depths),
        "agents": agents,
        "retrievers": retrievers,
        "joins": joins,
        "paths": {
            "json": str(out_path),
            "markdown": str(markdown_out_path),
        },
    }
    write_json(out_path, result)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(render_context_cost_markdown(result), encoding="utf-8")
    return result


def summarize_agent_trajectory(label: str, path: Path, rows: list[dict[str, Any]], late_hit_threshold: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_sample: dict[str, dict[str, Any]] = {}
    first_steps: list[int] = []
    event_counts: list[int] = []
    final_hits = 0
    utilized_hits = 0
    task_counts: dict[str, list[Any]] = defaultdict(lambda: [0, 0, []])
    tool_counts: Counter[str] = Counter()

    for row in rows:
        sample_id = str(row.get("sample_id"))
        task_type = str(row.get("task_type", "unknown"))
        gold_files = set(row.get("gold_files") or [])
        steps = row.get("steps") or []
        event_count = len(steps)
        event_counts.append(event_count)
        for step in steps:
            tool_counts[str(step.get("tool", ""))] += 1

        first_hit_step = first_gold_step(steps, gold_files)
        hit = first_hit_step is not None
        if hit:
            first_steps.append(int(first_hit_step))
        final_hit = any(step.get("path") in gold_files and step.get("is_final_context") for step in steps)
        utilized_hit = any(step.get("path") in gold_files and step.get("is_utilized_context") for step in steps)
        final_hits += int(final_hit)
        utilized_hits += int(utilized_hit)
        task_counts[task_type][0] += 1
        task_counts[task_type][1] += int(hit)
        if hit:
            task_counts[task_type][2].append(int(first_hit_step))

        per_sample[sample_id] = {
            "sample_id": sample_id,
            "task_type": task_type,
            "event_count": event_count,
            "first_hit_step": first_hit_step,
            "hit": hit,
            "late_hit": bool(hit and int(first_hit_step) > late_hit_threshold),
            "miss": not hit,
            "final_hit": final_hit,
            "utilized_hit": utilized_hit,
            "gold_files": sorted(gold_files),
        }

    count = len(rows)
    summary = {
        "label": label,
        "path": str(path),
        "samples": count,
        "events_per_sample_mean": mean(event_counts),
        "events_per_sample_median": safe_median(event_counts),
        "any_gold_count": len(first_steps),
        "any_gold_rate": safe_div(len(first_steps), count),
        "miss_count": count - len(first_steps),
        "miss_rate": safe_div(count - len(first_steps), count),
        "late_hit_threshold": late_hit_threshold,
        "late_hit_count": sum(1 for step in first_steps if step > late_hit_threshold),
        "late_hit_rate": safe_div(sum(1 for step in first_steps if step > late_hit_threshold), count),
        "median_first_hit_step": safe_median(first_steps),
        "mean_first_hit_step": mean(first_steps),
        "final_gold_count": final_hits,
        "final_gold_rate": safe_div(final_hits, count),
        "utilized_gold_count": utilized_hits,
        "utilized_gold_rate": safe_div(utilized_hits, count),
        "tool_counts": dict(tool_counts),
        "by_task": {
            task: {
                "samples": values[0],
                "any_gold_count": values[1],
                "any_gold_rate": safe_div(values[1], values[0]),
                "median_first_hit_step": safe_median(values[2]),
            }
            for task, values in sorted(task_counts.items())
        },
    }
    return summary, per_sample


def summarize_retriever(label: str, path: Path, rows: list[dict[str, Any]], depths: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_sample: dict[str, dict[str, Any]] = {}
    ranks: list[int] = []
    by_task: dict[str, list[Any]] = defaultdict(lambda: [0, []])
    any_at = Counter()

    for row in rows:
        sample_id = str(row.get("sample_id"))
        task_type = str(row.get("task_type", "unknown"))
        rank = first_gold_rank(row)
        if rank is not None:
            ranks.append(rank)
            for depth in depths:
                if rank <= depth:
                    any_at[depth] += 1
            by_task[task_type][1].append(rank)
        by_task[task_type][0] += 1
        per_sample[sample_id] = {
            "sample_id": sample_id,
            "task_type": task_type,
            "first_gold_rank": rank,
            **{f"any@{depth}": bool(rank is not None and rank <= depth) for depth in depths},
        }

    count = len(rows)
    summary = {
        "label": label,
        "path": str(path),
        "samples": count,
        "median_first_gold_rank": safe_median(ranks),
        "mean_first_gold_rank": mean(ranks),
        **{f"any@{depth}": safe_div(any_at[depth], count) for depth in depths},
        "by_task": {
            task: {
                "samples": values[0],
                "median_first_gold_rank": safe_median(values[1]),
                **{f"any@{depth}": safe_div(sum(1 for rank in values[1] if rank <= depth), values[0]) for depth in depths},
            }
            for task, values in sorted(by_task.items())
        },
    }
    return summary, per_sample


def summarize_join(
    agent_label: str,
    agent_samples: dict[str, dict[str, Any]],
    retriever_label: str,
    retriever_samples: dict[str, dict[str, Any]],
    depths: tuple[int, ...],
    late_hit_threshold: int,
) -> dict[str, Any]:
    common_ids = sorted(set(agent_samples) & set(retriever_samples))
    subsets = {
        "all": common_ids,
        "agent_hit": [sid for sid in common_ids if agent_samples[sid]["hit"]],
        "agent_miss": [sid for sid in common_ids if agent_samples[sid]["miss"]],
        "agent_late_hit": [sid for sid in common_ids if agent_samples[sid]["late_hit"]],
        "agent_early_hit": [sid for sid in common_ids if agent_samples[sid]["hit"] and not agent_samples[sid]["late_hit"]],
    }
    result: dict[str, Any] = {
        "agent": agent_label,
        "retriever": retriever_label,
        "samples": len(common_ids),
        "late_hit_threshold": late_hit_threshold,
        "subsets": {},
        "complementarity": {},
    }
    for subset_name, ids in subsets.items():
        result["subsets"][subset_name] = summarize_join_subset(ids, agent_samples, retriever_samples, depths)
    for depth in depths:
        result["complementarity"][f"@{depth}"] = summarize_complementarity(
            common_ids,
            agent_samples,
            retriever_samples,
            depth,
        )
    return result


def summarize_join_subset(ids: list[str], agent_samples: dict[str, dict[str, Any]], retriever_samples: dict[str, dict[str, Any]], depths: tuple[int, ...]) -> dict[str, Any]:
    ranks = [retriever_samples[sid]["first_gold_rank"] for sid in ids if retriever_samples[sid]["first_gold_rank"] is not None]
    event_counts = [agent_samples[sid]["event_count"] for sid in ids]
    first_steps = [agent_samples[sid]["first_hit_step"] for sid in ids if agent_samples[sid]["first_hit_step"] is not None]
    return {
        "samples": len(ids),
        "events_per_sample_mean": mean(event_counts),
        "median_agent_first_hit_step": safe_median(first_steps),
        "median_retriever_first_gold_rank": safe_median(ranks),
        **{f"retriever_any@{depth}": safe_div(sum(1 for rank in ranks if rank <= depth), len(ids)) for depth in depths},
    }


def summarize_complementarity(
    ids: list[str],
    agent_samples: dict[str, dict[str, Any]],
    retriever_samples: dict[str, dict[str, Any]],
    depth: int,
) -> dict[str, Any]:
    counts = Counter()
    potential_saved_events = 0
    for sample_id in ids:
        agent_hit = bool(agent_samples[sample_id]["hit"])
        retriever_hit = bool(retriever_samples[sample_id].get(f"any@{depth}"))
        if agent_hit and retriever_hit:
            counts["both_hit"] += 1
        elif retriever_hit:
            counts["retriever_only"] += 1
        elif agent_hit:
            counts["agent_only"] += 1
        else:
            counts["both_miss"] += 1
        if retriever_hit and agent_samples[sample_id]["first_hit_step"] is not None:
            potential_saved_events += max(0, int(agent_samples[sample_id]["first_hit_step"]) - 1)

    total = len(ids)
    agent_miss = counts["retriever_only"] + counts["both_miss"]
    retriever_miss = counts["agent_only"] + counts["both_miss"]
    return {
        "samples": total,
        "depth": depth,
        "both_hit_count": counts["both_hit"],
        "retriever_only_count": counts["retriever_only"],
        "agent_only_count": counts["agent_only"],
        "both_miss_count": counts["both_miss"],
        "both_hit_rate": safe_div(counts["both_hit"], total),
        "retriever_only_rate": safe_div(counts["retriever_only"], total),
        "agent_only_rate": safe_div(counts["agent_only"], total),
        "both_miss_rate": safe_div(counts["both_miss"], total),
        "retriever_only_given_agent_miss": safe_div(counts["retriever_only"], agent_miss),
        "agent_only_given_retriever_miss": safe_div(counts["agent_only"], retriever_miss),
        "potential_exploration_savings": safe_div(potential_saved_events, total),
    }


def first_gold_step(steps: list[dict[str, Any]], gold_files: set[str]) -> int | None:
    for step in steps:
        if step.get("path") in gold_files and step.get("step") is not None:
            return int(step["step"])
    return None


def first_gold_rank(row: dict[str, Any]) -> int | None:
    ranks = []
    for gold_file in row.get("gold_files") or []:
        value = (row.get("gold_ranks") or {}).get(gold_file)
        if value is not None:
            try:
                ranks.append(int(value))
            except (TypeError, ValueError):
                pass
    return min(ranks) if ranks else None


def mean(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_median(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def safe_div(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_context_cost_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Context Acquisition Cost Report",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Late-hit threshold: `>{result['late_hit_threshold']}` events",
        "",
        "## Agent Trajectory Cost",
        "",
        "| Agent | Samples | Events/sample | Any-gold | Miss rate | Median first hit | Final gold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["agents"]:
        lines.append(
            "| {label} | {samples} | {events} | {any_gold} | {miss} | {median} | {final} |".format(
                label=row["label"],
                samples=row["samples"],
                events=fmt(row["events_per_sample_mean"]),
                any_gold=fmt(row["any_gold_rate"]),
                miss=fmt(row["miss_rate"]),
                median=fmt(row["median_first_hit_step"], 1),
                final=fmt(row["final_gold_rate"]),
            )
        )
    lines.extend([
        "",
        "## Static Retriever First-Gold Rank",
        "",
        "| Retriever | Samples | Any@3 | Any@5 | Any@10 | Any@20 | Median rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result["retrievers"]:
        lines.append(
            "| {label} | {samples} | {a3} | {a5} | {a10} | {a20} | {median} |".format(
                label=row["label"],
                samples=row["samples"],
                a3=fmt(row.get("any@3")),
                a5=fmt(row.get("any@5")),
                a10=fmt(row.get("any@10")),
                a20=fmt(row.get("any@20")),
                median=fmt(row["median_first_gold_rank"], 1),
            )
        )
    lines.extend([
        "",
        "## Join: Retriever Coverage on Agent Subsets",
        "",
        "| Agent | Retriever | Subset | Samples | Events/sample | Retriever Any@3 | Any@5 | Any@10 | Any@20 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    preferred = []
    for join in result["joins"]:
        if join["retriever"] in {
            "Qwen3-Embedding-4B",
            "Qwen3-Embedding-8B",
            "RepoMap",
            "RRF(Qwen3-8B + RepoMap)",
            "RRF(Qwen3-8B + Qwen3-4B + RepoMap)",
            "RRF(Qwen3-4B + RepoMap)",
        }:
            preferred.append(join)
    for join in preferred:
        for subset_name in ("all", "agent_hit", "agent_miss", "agent_late_hit"):
            row = join["subsets"][subset_name]
            lines.append(
                "| {agent} | {retriever} | `{subset}` | {samples} | {events} | {a3} | {a5} | {a10} | {a20} |".format(
                    agent=join["agent"],
                    retriever=join["retriever"],
                    subset=subset_name,
                    samples=row["samples"],
                    events=fmt(row["events_per_sample_mean"]),
                    a3=fmt(row.get("retriever_any@3")),
                    a5=fmt(row.get("retriever_any@5")),
                    a10=fmt(row.get("retriever_any@10")),
                    a20=fmt(row.get("retriever_any@20")),
                )
            )
    lines.append("")
    lines.extend([
        "## Trajectory-Conditioned Complementarity (@20)",
        "",
        "`PES@20` is Potential Exploration Savings: an upper-bound proxy for pre-first-gold localization delay, not a causal saved-step estimate. All rows are defined for a specific `(retriever, agent)` pair.",
        "",
        "| Agent | Retriever | Both hit | Retriever only | Agent only | Both miss | PES@20 | Retriever-only / agent-miss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for join in preferred:
        row = join.get("complementarity", {}).get("@20", {})
        lines.append(
            "| {agent} | {retriever} | {both} | {retriever_only} | {agent_only} | {miss} | {pes} | {rescue} |".format(
                agent=join["agent"],
                retriever=join["retriever"],
                both=fmt(row.get("both_hit_rate")),
                retriever_only=fmt(row.get("retriever_only_rate")),
                agent_only=fmt(row.get("agent_only_rate")),
                miss=fmt(row.get("both_miss_rate")),
                pes=fmt(row.get("potential_exploration_savings")),
                rescue=fmt(row.get("retriever_only_given_agent_miss")),
            )
        )
    lines.append("")
    return "\n".join(lines)
