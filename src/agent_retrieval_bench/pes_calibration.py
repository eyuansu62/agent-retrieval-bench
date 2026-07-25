from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .io import read_jsonl, utc_now, write_json

DEFAULT_PACKET = Path("data/reports/v1_4/pes_calibration/selected_samples.jsonl")
DEFAULT_CONTROL_DETAILS = Path("data/eval/v1_4/codex_cli_gpt54_corpus_full_budgeted_answer_recovered_all_details.jsonl")
DEFAULT_SEEDED_DETAILS = (Path("data/eval/v1_4/pes_calibration_seeded_gpt54_details.jsonl"),)


def report_pes_calibration(
    *,
    packet_path: Path = DEFAULT_PACKET,
    control_details_path: Path = DEFAULT_CONTROL_DETAILS,
    seeded_details_paths: Iterable[Path] = DEFAULT_SEEDED_DETAILS,
    out_path: Path = Path("data/reports/v1_4/pes_calibration/calibration_report.json"),
    markdown_out_path: Path = Path("data/reports/v1_4/pes_calibration/calibration_report.md"),
) -> dict[str, Any]:
    packet = {str(row.get("sample_id") or ""): row for row in read_jsonl(packet_path) if row.get("sample_id")}
    control = load_details([control_details_path])
    seeded_paths = tuple(seeded_details_paths)
    seeded = load_details(seeded_paths)

    rows = [
        compare_sample(sample_id, packet_row, control.get(sample_id), seeded.get(sample_id))
        for sample_id, packet_row in sorted(packet.items())
    ]
    evaluated = [row for row in rows if row["has_control"] and row["has_seeded"]]
    result = {
        "mode": "pes_calibration",
        "generated_at": utc_now(),
        "paths": {
            "packet": str(packet_path),
            "control_details": str(control_details_path),
            "seeded_details": [str(path) for path in seeded_paths],
            "json": str(out_path),
            "markdown": str(markdown_out_path),
        },
        "counts": {
            "packet_samples": len(packet),
            "control_samples": len(control),
            "seeded_samples": len(seeded),
            "paired_samples": len(evaluated),
            "missing_control": sum(1 for row in rows if not row["has_control"]),
            "missing_seeded": sum(1 for row in rows if not row["has_seeded"]),
        },
        "overall": summarize(evaluated),
        "by_group": summarize_by(evaluated, "group"),
        "by_task": summarize_by(evaluated, "task_type"),
        "samples": rows,
    }
    write_json(out_path, result)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(render_markdown(result), encoding="utf-8")
    return result


def load_details(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if sample_id:
                rows[sample_id] = row
    return rows


def compare_sample(
    sample_id: str,
    packet: dict[str, Any],
    control: dict[str, Any] | None,
    seeded: dict[str, Any] | None,
) -> dict[str, Any]:
    control_first = first_gold_step(control) if control else None
    seeded_first = first_gold_step(seeded) if seeded else None
    control_hit = control_first is not None
    seeded_hit = seeded_first is not None
    first_hit_delta = control_first - seeded_first if control_hit and seeded_hit else None
    control_events = event_count(control)
    seeded_events = event_count(seeded)
    event_saving = control_events - seeded_events if control_events is not None and seeded_events is not None else None
    control_f1 = metric(control, "final_file_f1")
    seeded_f1 = metric(seeded, "final_file_f1")
    f1_delta = seeded_f1 - control_f1 if control_f1 is not None and seeded_f1 is not None else None
    predicted_pes = optional_float(packet.get("PES@20"))
    return {
        "sample_id": sample_id,
        "group": str(packet.get("group") or "unknown"),
        "task_type": str(packet.get("task_type") or (control or seeded or {}).get("task_type") or "unknown"),
        "repo": str(packet.get("repo") or (control or seeded or {}).get("repo") or ""),
        "seed_retriever_hit@20": bool(packet.get("retriever_hit@20")),
        "seed_retriever_first_gold_rank": optional_int(packet.get("retriever_first_gold_rank")),
        "predicted_pes@20": predicted_pes,
        "has_control": control is not None,
        "has_seeded": seeded is not None,
        "control_first_gold_step": control_first,
        "seeded_first_gold_step": seeded_first,
        "control_hit": control_hit,
        "seeded_hit": seeded_hit,
        "rescue": bool(not control_hit and seeded_hit),
        "lost": bool(control_hit and not seeded_hit),
        "both_hit": bool(control_hit and seeded_hit),
        "first_hit_delta": first_hit_delta,
        "first_hit_improved": bool(first_hit_delta is not None and first_hit_delta > 0),
        "first_hit_regressed": bool(first_hit_delta is not None and first_hit_delta < 0),
        "control_event_count": control_events,
        "seeded_event_count": seeded_events,
        "event_saving": event_saving,
        "control_final_file_f1": control_f1,
        "seeded_final_file_f1": seeded_f1,
        "final_file_f1_delta": f1_delta,
    }


def first_gold_step(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    gold = set(row.get("gold_files") or [])
    if not gold:
        return None
    values = []
    for step in row.get("steps") or []:
        if step.get("path") in gold and step.get("step") is not None:
            try:
                values.append(int(step["step"]))
            except (TypeError, ValueError):
                pass
    return min(values) if values else None


def event_count(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    steps = row.get("steps") or []
    return len(steps)


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    return optional_float((row.get("metrics") or {}).get(key))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_summary()
    both_hit = [row for row in rows if row["both_hit"]]
    positive_pred = [row for row in both_hit if row["predicted_pes@20"] not in (None, 0)]
    step_pairs = [
        (float(row["predicted_pes@20"]), float(row["first_hit_delta"]))
        for row in both_hit
        if row["predicted_pes@20"] is not None and row["first_hit_delta"] is not None
    ]
    event_pairs = [
        (float(row["predicted_pes@20"]), float(row["event_saving"]))
        for row in rows
        if row["predicted_pes@20"] is not None and row["event_saving"] is not None
    ]
    predicted_total = sum(float(row["predicted_pes@20"]) for row in positive_pred)
    positive_actual_total = sum(max(0.0, float(row["first_hit_delta"])) for row in positive_pred if row["first_hit_delta"] is not None)
    return {
        "samples": len(rows),
        "seed_retriever_hit@20_rate": rate(sum(1 for row in rows if row["seed_retriever_hit@20"]), len(rows)),
        "control_hit_rate": rate(sum(1 for row in rows if row["control_hit"]), len(rows)),
        "seeded_hit_rate": rate(sum(1 for row in rows if row["seeded_hit"]), len(rows)),
        "rescue_count": sum(1 for row in rows if row["rescue"]),
        "rescue_rate": rate(sum(1 for row in rows if row["rescue"]), len(rows)),
        "lost_count": sum(1 for row in rows if row["lost"]),
        "lost_rate": rate(sum(1 for row in rows if row["lost"]), len(rows)),
        "both_hit_count": len(both_hit),
        "first_hit_improved_count": sum(1 for row in rows if row["first_hit_improved"]),
        "first_hit_regressed_count": sum(1 for row in rows if row["first_hit_regressed"]),
        "predicted_pes_mean": mean([row["predicted_pes@20"] for row in rows]),
        "predicted_pes_median": safe_median([row["predicted_pes@20"] for row in rows]),
        "first_hit_delta_mean_both_hit": mean([row["first_hit_delta"] for row in both_hit]),
        "first_hit_delta_median_both_hit": safe_median([row["first_hit_delta"] for row in both_hit]),
        "event_saving_mean": mean([row["event_saving"] for row in rows]),
        "event_saving_median": safe_median([row["event_saving"] for row in rows]),
        "final_file_f1_delta_mean": mean([row["final_file_f1_delta"] for row in rows]),
        "positive_actual_step_saving_over_predicted_pes": rate(positive_actual_total, predicted_total),
        "pearson_pes_vs_first_hit_delta": pearson(step_pairs),
        "spearman_pes_vs_first_hit_delta": spearman(step_pairs),
        "pearson_pes_vs_event_saving": pearson(event_pairs),
        "spearman_pes_vs_event_saving": spearman(event_pairs),
    }


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}


def empty_summary() -> dict[str, Any]:
    return {
        "samples": 0,
        "seed_retriever_hit@20_rate": None,
        "control_hit_rate": None,
        "seeded_hit_rate": None,
        "rescue_count": 0,
        "rescue_rate": None,
        "lost_count": 0,
        "lost_rate": None,
        "both_hit_count": 0,
        "first_hit_improved_count": 0,
        "first_hit_regressed_count": 0,
        "predicted_pes_mean": None,
        "predicted_pes_median": None,
        "first_hit_delta_mean_both_hit": None,
        "first_hit_delta_median_both_hit": None,
        "event_saving_mean": None,
        "event_saving_median": None,
        "final_file_f1_delta_mean": None,
        "positive_actual_step_saving_over_predicted_pes": None,
        "pearson_pes_vs_first_hit_delta": None,
        "spearman_pes_vs_first_hit_delta": None,
        "pearson_pes_vs_event_saving": None,
        "spearman_pes_vs_event_saving": None,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PES Calibration Report",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Packet samples: `{result['counts']['packet_samples']}`",
        f"- Paired control/seeded samples: `{result['counts']['paired_samples']}`",
        f"- Missing seeded samples: `{result['counts']['missing_seeded']}`",
        "",
        "This is a seeded intervention diagnostic for PES@20. PES remains a potential upper-bound proxy; this report checks whether higher predicted PES aligns with earlier first-gold hits or fewer context events when the retriever's top-20 paths are injected as starting hints.",
        "",
        "## Overall",
        "",
        summary_table([("overall", result["overall"])]),
        "",
        "## By Selection Group",
        "",
        summary_table(sorted(result["by_group"].items())),
        "",
        "## By Task",
        "",
        summary_table(sorted(result["by_task"].items())),
        "",
        "## Largest First-Hit Improvements",
        "",
        sample_table(top_samples(result["samples"], reverse=True)),
        "",
        "## Largest First-Hit Regressions",
        "",
        sample_table(top_samples(result["samples"], reverse=False)),
        "",
    ]
    return "\n".join(lines)


def summary_table(items: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Slice | n | Seed hit@20 | Control hit | Seeded hit | Rescue | Lost | Mean PES | Mean first-hit delta | Mean event saving | F1 delta | PES-step r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in items:
        lines.append(
            "| {name} | {n} | {seed_hit} | {control_hit} | {seeded_hit} | {rescue} | {lost} | {pes} | {step} | {event} | {f1} | {corr} |".format(
                name=name,
                n=row["samples"],
                seed_hit=fmt(row["seed_retriever_hit@20_rate"]),
                control_hit=fmt(row["control_hit_rate"]),
                seeded_hit=fmt(row["seeded_hit_rate"]),
                rescue=f"{row['rescue_count']} ({fmt(row['rescue_rate'])})",
                lost=f"{row['lost_count']} ({fmt(row['lost_rate'])})",
                pes=fmt(row["predicted_pes_mean"]),
                step=fmt(row["first_hit_delta_mean_both_hit"]),
                event=fmt(row["event_saving_mean"]),
                f1=fmt(row["final_file_f1_delta_mean"]),
                corr=fmt(row["spearman_pes_vs_first_hit_delta"]),
            )
        )
    return "\n".join(lines)


def top_samples(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("first_hit_delta") is not None]
    return sorted(candidates, key=lambda row: float(row["first_hit_delta"]), reverse=reverse)[:8]


def sample_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No paired both-hit samples in this slice._"
    lines = [
        "| Sample | Group | Task | PES@20 | Control step | Seeded step | Delta | Event saving | F1 delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{sid}` | {group} | {task} | {pes} | {control} | {seeded} | {delta} | {event} | {f1} |".format(
                sid=row["sample_id"],
                group=row["group"],
                task=row["task_type"],
                pes=fmt(row["predicted_pes@20"], 1),
                control=fmt(row["control_first_gold_step"], 0),
                seeded=fmt(row["seeded_first_gold_step"], 0),
                delta=fmt(row["first_hit_delta"], 0),
                event=fmt(row["event_saving"], 0),
                f1=fmt(row["final_file_f1_delta"]),
            )
        )
    return "\n".join(lines)


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


def rate(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


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
    x_ranks = ranks([x for x, _ in pairs])
    y_ranks = ranks([y for _, y in pairs])
    return pearson(list(zip(x_ranks, y_ranks)))


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


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
