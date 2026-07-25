from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import ensure_parent, read_json, read_jsonl, utc_now, write_json

TASKS = ("code2test", "comment2context", "trace2code", "edit2ripple")
TASK_SAMPLES = {
    "code2test": 106,
    "comment2context": 80,
    "trace2code": 101,
    "edit2ripple": 58,
}
RANK_METRICS = ("Recall@5", "Recall@10", "Recall@20", "MRR")
REPORT_METRICS = (*RANK_METRICS, "BCY@8k")

METHOD_SPECS: tuple[dict[str, str], ...] = (
    {
        "label": "Qwen3-Embedding-4B",
        "family": "embedding",
        "core_summary": "Qwen3-Embedding-4B_summary.json",
        "extension_summary": "qwen3-4b_summary.json",
        "core_details": "Qwen3-Embedding-4B_details.jsonl",
        "extension_details": "qwen3-4b_details.jsonl",
        "core_bcy": "Qwen3-Embedding-4B",
        "extension_bcy": "Qwen3-Embedding-4B",
    },
    {
        "label": "Qwen3-Embedding-8B",
        "family": "embedding",
        "core_summary": "Qwen3-Embedding-8B_summary.json",
        "extension_summary": "qwen3-8b_summary.json",
        "core_details": "Qwen3-Embedding-8B_details.jsonl",
        "extension_details": "qwen3-8b_details.jsonl",
        "core_bcy": "Qwen3-Embedding-8B",
        "extension_bcy": "Qwen3-Embedding-8B",
    },
    {
        "label": "pplx-embed-v1-4b",
        "family": "embedding",
        "core_summary": "pplx-embed-v1-4b_summary.json",
        "extension_summary": "pplx-4b_summary.json",
        "core_details": "pplx-embed-v1-4b_details.jsonl",
        "extension_details": "pplx-4b_details.jsonl",
        "core_bcy": "pplx-embed-v1-4b",
        "extension_bcy": "pplx-embed-v1-4b",
    },
    {
        "label": "RepoMap",
        "family": "structure",
        "core_summary": "repomap_all_files_summary.json",
        "extension_summary": "repomap_summary.json",
        "core_details": "repomap_all_files_details.jsonl",
        "extension_details": "repomap_details.jsonl",
        "core_bcy": "RepoMap",
        "extension_bcy": "RepoMap",
    },
    {
        "label": "jina-code-embeddings-0.5b",
        "family": "embedding",
        "core_summary": "jina-code-embeddings-0.5b_summary.json",
        "extension_summary": "jina-code-embeddings-0.5b_summary.json",
        "core_details": "jina-code-embeddings-0.5b_details.jsonl",
        "extension_details": "jina-code-embeddings-0.5b_details.jsonl",
        "core_bcy": "jina-code-embeddings-0.5b",
        "extension_bcy": "jina-code-embeddings-0.5b",
    },
    {
        "label": "nomic-embed-code",
        "family": "embedding",
        "core_summary": "nomic-embed-code_summary.json",
        "extension_summary": "nomic_summary.json",
        "core_details": "nomic-embed-code_details.jsonl",
        "extension_details": "nomic_details.jsonl",
        "core_bcy": "nomic-embed-code",
        "extension_bcy": "nomic-embed-code",
    },
    {
        "label": "Lexical",
        "family": "lexical",
        "core_summary": "lexical_all_files_summary.json",
        "extension_summary": "lexical_summary.json",
        "core_details": "lexical_all_files_details.jsonl",
        "extension_details": "lexical_details.jsonl",
        "core_bcy": "Lexical",
        "extension_bcy": "lexical",
    },
    {
        "label": "BM25",
        "family": "lexical",
        "core_summary": "bm25_all_files_summary.json",
        "extension_summary": "bm25_summary.json",
        "core_details": "bm25_all_files_details.jsonl",
        "extension_details": "bm25_details.jsonl",
        "core_bcy": "BM25",
        "extension_bcy": "BM25",
    },
)


def report_v2_positive_leaderboard(
    core_eval_dir: Path,
    extension_eval_dir: Path,
    core_bcy_path: Path,
    extension_bcy_path: Path,
    out_path: Path,
    json_out_path: Path | None = None,
    method_specs: Iterable[dict[str, str]] = METHOD_SPECS,
) -> dict[str, Any]:
    core_bcy = _bcy_runs(read_json(core_bcy_path, {}), core_bcy_path)
    extension_bcy = _bcy_runs(read_json(extension_bcy_path, {}), extension_bcy_path)
    rows = [
        _method_row(
            spec,
            core_eval_dir=core_eval_dir,
            extension_eval_dir=extension_eval_dir,
            core_bcy=core_bcy,
            extension_bcy=extension_bcy,
        )
        for spec in method_specs
    ]
    repo_samples = rows[0].pop("_repo_samples") if rows else {}
    for row in rows[1:]:
        method_repo_samples = row.pop("_repo_samples")
        if method_repo_samples != repo_samples:
            raise ValueError(f"Repository sample distribution differs for {row['model']}")
    rows.sort(key=lambda row: (-row["weighted"]["MRR"], -row["weighted"]["Recall@20"], row["model"]))
    sorted_repos = sorted(repo_samples.items(), key=lambda item: (-item[1], item[0]))
    top_four_samples = sum(count for _, count in sorted_repos[:4])
    report = {
        "generated_at": utc_now(),
        "sample_count": sum(TASK_SAMPLES.values()),
        "task_samples": TASK_SAMPLES,
        "method_count": len(rows),
        "aggregation": {
            "weighted": "sample-weighted mean over the four task metrics",
            "macro": "unweighted mean over code2test, comment2context, trace2code, and edit2ripple",
            "repo_macro": "mean of per-repository sample means; every positive-sample repository has equal weight",
            "bcy": "canonical token-packed BCY@8k from the supplied BCY reports",
        },
        "repository_distribution": {
            "repository_count": len(repo_samples),
            "counts": dict(sorted_repos),
            "largest_repository": {"repo": sorted_repos[0][0], "samples": sorted_repos[0][1]},
            "top_four_samples": top_four_samples,
            "top_four_share": top_four_samples / sum(repo_samples.values()),
        },
        "sources": {
            "core_eval_dir": str(core_eval_dir),
            "extension_eval_dir": str(extension_eval_dir),
            "core_bcy": str(core_bcy_path),
            "extension_bcy": str(extension_bcy_path),
        },
        "rows": rows,
    }
    json_path = json_out_path or out_path.with_suffix(".json")
    write_json(json_path, report)
    ensure_parent(out_path)
    out_path.write_text(render_v2_positive_markdown(report), encoding="utf-8")
    return {
        "samples": report["sample_count"],
        "methods": len(rows),
        "markdown": str(out_path),
        "json": str(json_path),
    }


def _method_row(
    spec: dict[str, str],
    *,
    core_eval_dir: Path,
    extension_eval_dir: Path,
    core_bcy: dict[str, dict[str, Any]],
    extension_bcy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    core_path = core_eval_dir / spec["core_summary"]
    extension_path = extension_eval_dir / spec["extension_summary"]
    core_summary = _summary(core_path)
    extension_summary = _summary(extension_path)
    details = read_jsonl(core_eval_dir / spec["core_details"]) + read_jsonl(
        extension_eval_dir / spec["extension_details"]
    )
    core_bcy_run = _required_run(core_bcy, spec["core_bcy"], "core BCY")
    extension_bcy_run = _required_run(extension_bcy, spec["extension_bcy"], "extension BCY")

    by_task: dict[str, dict[str, float | int]] = {}
    for task in TASKS:
        summary = extension_summary if task == "edit2ripple" else core_summary
        bcy_run = extension_bcy_run if task == "edit2ripple" else core_bcy_run
        metrics = (summary.get("metrics") or {}).get(task)
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing task {task!r} in summary for {spec['label']}")
        samples = int(metrics.get("samples") or 0)
        expected = TASK_SAMPLES[task]
        if samples != expected:
            raise ValueError(f"{spec['label']} {task} has {samples} samples; expected {expected}")
        bcy_metrics = (bcy_run.get("by_task") or {}).get(task)
        if not isinstance(bcy_metrics, dict) or "BCY@8000" not in bcy_metrics:
            raise ValueError(f"Missing canonical BCY@8000 for {spec['label']} {task}")
        by_task[task] = {
            "samples": samples,
            **{metric: float(metrics[metric]) for metric in RANK_METRICS},
            "BCY@8k": float(bcy_metrics["BCY@8000"]),
        }

    total = sum(int(row["samples"]) for row in by_task.values())
    weighted = {
        metric: sum(float(row[metric]) * int(row["samples"]) for row in by_task.values()) / total
        for metric in REPORT_METRICS
    }
    macro = {
        metric: sum(float(row[metric]) for row in by_task.values()) / len(TASKS)
        for metric in REPORT_METRICS
    }
    repo_macro, repo_samples = _repository_macro(details, spec["label"])
    return {
        "model": spec["label"],
        "family": spec["family"],
        "samples": total,
        "weighted": weighted,
        "macro": macro,
        "repo_macro": repo_macro,
        "by_task": by_task,
        "sources": {
            "core_summary": str(core_path),
            "extension_summary": str(extension_path),
            "core_details": str(core_eval_dir / spec["core_details"]),
            "extension_details": str(extension_eval_dir / spec["extension_details"]),
        },
        "_repo_samples": repo_samples,
    }


def _repository_macro(details: list[dict[str, Any]], label: str) -> tuple[dict[str, float], dict[str, int]]:
    expected = sum(TASK_SAMPLES.values())
    if len(details) != expected:
        raise ValueError(f"{label} details have {len(details)} rows; expected {expected}")
    sample_ids = [str(row.get("sample_id") or "") for row in details]
    if not all(sample_ids) or len(set(sample_ids)) != expected:
        raise ValueError(f"{label} details must contain {expected} unique sample ids")

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        repo = str(row.get("repo") or "")
        metrics = row.get("metrics")
        if not repo or not isinstance(metrics, dict):
            raise ValueError(f"{label} details contain a row without repo or metrics")
        if any(metric not in metrics for metric in RANK_METRICS):
            raise ValueError(f"{label} details contain incomplete rank metrics")
        by_repo.setdefault(repo, []).append(metrics)

    repo_macro = {
        metric: sum(
            sum(float(metrics[metric]) for metrics in repo_rows) / len(repo_rows)
            for repo_rows in by_repo.values()
        )
        / len(by_repo)
        for metric in RANK_METRICS
    }
    return repo_macro, {repo: len(rows) for repo, rows in by_repo.items()}


def _summary(path: Path) -> dict[str, Any]:
    summary = read_json(path, {})
    if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
        raise ValueError(f"Invalid or missing evaluation summary: {path}")
    return summary


def _bcy_runs(report: Any, path: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(report, dict) or not isinstance(report.get("runs"), list):
        raise ValueError(f"Invalid or missing BCY report: {path}")
    return {str(run["label"]): run for run in report["runs"] if isinstance(run, dict) and run.get("label")}


def _required_run(runs: dict[str, dict[str, Any]], label: str, source: str) -> dict[str, Any]:
    if label not in runs:
        raise ValueError(f"Missing {source} run {label!r}")
    return runs[label]


def render_v2_positive_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Positive Leaderboard",
        "",
        f"- Samples: **{report['sample_count']}**",
        f"- Methods: **{report['method_count']}**",
        "- Weighted: sample-weighted over all four positive tasks.",
        "- Macro: unweighted mean over the four positive tasks.",
        "- Repo macro: average within each repository, then average repositories equally.",
        "- BCY@8k: canonical token-packed file-level BCY.",
        f"- Repositories: **{report['repository_distribution']['repository_count']}**; the four largest contain "
        f"**{report['repository_distribution']['top_four_share']:.1%}** of positive samples.",
        "",
        "## Aggregate",
        "",
        "| Model | Family | R@5 | R@10 | R@20 | MRR | BCY@8k | Macro R@20 | Macro MRR | Repo R@20 | Repo MRR | Macro BCY@8k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        weighted = row["weighted"]
        macro = row["macro"]
        repo_macro = row["repo_macro"]
        lines.append(
            f"| {row['model']} | {row['family']} | {_fmt(weighted['Recall@5'])} | "
            f"{_fmt(weighted['Recall@10'])} | {_fmt(weighted['Recall@20'])} | {_fmt(weighted['MRR'])} | "
            f"{_fmt(weighted['BCY@8k'])} | {_fmt(macro['Recall@20'])} | {_fmt(macro['MRR'])} | "
            f"{_fmt(repo_macro['Recall@20'])} | {_fmt(repo_macro['MRR'])} | {_fmt(macro['BCY@8k'])} |"
        )
    for task in TASKS:
        lines.extend(["", f"## {task}", "", "| Model | n | R@5 | R@10 | R@20 | MRR | BCY@8k |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        task_rows = sorted(report["rows"], key=lambda row: (-row["by_task"][task]["MRR"], row["model"]))
        for row in task_rows:
            metrics = row["by_task"][task]
            lines.append(
                f"| {row['model']} | {metrics['samples']} | {_fmt(metrics['Recall@5'])} | "
                f"{_fmt(metrics['Recall@10'])} | {_fmt(metrics['Recall@20'])} | {_fmt(metrics['MRR'])} | "
                f"{_fmt(metrics['BCY@8k'])} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"
