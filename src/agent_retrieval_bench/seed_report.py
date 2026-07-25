from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import ensure_parent, read_jsonl, utc_now, write_json


def report_v1_seed(
    base_samples_path: Path,
    base_eval_path: Path,
    seed_samples_path: Path,
    seed_eval_path: Path,
    audit_summary_path: Path,
    out_path: Path,
    json_out_path: Path,
) -> dict[str, Any]:
    base_samples = sample_summary(base_samples_path)
    seed_samples = sample_summary(seed_samples_path)
    base_eval = read_json(base_eval_path)
    seed_eval = read_json(seed_eval_path)
    audit = read_json(audit_summary_path)
    report = {
        "generated_at": utc_now(),
        "status": seed_status(seed_samples, seed_eval, audit),
        "inputs": {
            "base_samples": str(base_samples_path),
            "base_eval": str(base_eval_path),
            "seed_samples": str(seed_samples_path),
            "seed_eval": str(seed_eval_path),
            "audit_summary": str(audit_summary_path),
        },
        "base": {"samples": base_samples, "metrics": metrics_summary(base_eval)},
        "seed": {"samples": seed_samples, "metrics": metrics_summary(seed_eval)},
        "audit": audit_summary(audit),
        "delta": sample_delta(base_samples, seed_samples),
    }
    write_json(json_out_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_seed_report(report), encoding="utf-8")
    return {
        "report": str(out_path),
        "json": str(json_out_path),
        "status": report["status"],
        "base_samples": base_samples.get("total", 0),
        "seed_samples": seed_samples.get("total", 0),
        "audit_kept": report["audit"].get("kept", 0),
    }


def sample_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "total": 0, "by_task": {}}
    rows = read_jsonl(path)
    counts = Counter(str(row.get("task_type", "")) for row in rows)
    return {"path": str(path), "exists": True, "total": len(rows), "by_task": dict(sorted(counts.items()))}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("path", str(path))
        data["exists"] = True
        return data
    return {"path": str(path), "exists": True, "value": data}


def metrics_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return {"exists": False}
    return {"exists": True, "evaluated": summary.get("evaluated"), "skipped": summary.get("skipped", {}), "metrics": metrics}


def audit_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("exists"):
        return {"exists": False}
    return {
        "exists": True,
        "total": summary.get("total", 0),
        "kept": summary.get("kept", 0),
        "dropped": summary.get("dropped", 0),
        "pending": summary.get("pending", 0),
        "verdicts": summary.get("verdicts", {}),
        "kept_by_task": summary.get("kept_by_task", {}),
        "dropped_by_task": summary.get("dropped_by_task", {}),
    }


def seed_status(seed_samples: dict[str, Any], seed_eval: dict[str, Any], audit: dict[str, Any]) -> str:
    if not audit.get("exists"):
        return "missing_audit"
    if int(audit.get("kept", 0)) < 50:
        return "audit_shortfall"
    if not seed_samples.get("exists"):
        return "not_exported"
    if not seed_eval.get("exists"):
        return "missing_eval"
    return "ready"


def sample_delta(base: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
    tasks = set(base.get("by_task", {})) | set(seed.get("by_task", {}))
    return {
        "total": int(seed.get("total", 0)) - int(base.get("total", 0)),
        "by_task": {
            task: int(seed.get("by_task", {}).get(task, 0)) - int(base.get("by_task", {}).get(task, 0))
            for task in sorted(tasks)
        },
    }


def render_seed_report(report: dict[str, Any]) -> str:
    lines = [
        "# V1 Seed Comparison",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        "",
        "## Samples",
        "",
        "| Dataset | Total | code2test | comment2context | trace2code |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("base", "seed"):
        samples = report[label]["samples"]
        by_task = samples.get("by_task", {})
        lines.append(
            f"| `{label}` | {samples.get('total', 0)} | {by_task.get('code2test', 0)} | "
            f"{by_task.get('comment2context', 0)} | {by_task.get('trace2code', 0)} |"
        )
    lines.extend(["", "## Audit", ""])
    audit = report["audit"]
    lines.extend(
        [
            f"- Total audited: `{audit.get('total', 0)}`",
            f"- Kept: `{audit.get('kept', 0)}`",
            f"- Dropped: `{audit.get('dropped', 0)}`",
            f"- Pending: `{audit.get('pending', 0)}`",
            f"- Verdicts: `{json.dumps(audit.get('verdicts', {}), sort_keys=True)}`",
            "",
            "## Metrics",
            "",
        ]
    )
    for label in ("base", "seed"):
        metrics = (report[label]["metrics"].get("metrics") or {}).get("overall", {})
        lines.append(
            f"- `{label}` overall: samples=`{metrics.get('samples', 0)}`, "
            f"Recall@20=`{metrics.get('Recall@20', 0):.4f}`, MRR=`{metrics.get('MRR', 0):.4f}`"
        )
    return "\n".join(lines) + "\n"
