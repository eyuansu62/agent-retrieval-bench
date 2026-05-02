from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .baseline import query_has_leakage, query_text_for_eval, target_gold_files
from .io import ensure_parent, read_jsonl, utc_now, write_json
from .quality import validate_sample

DEFAULT_HARDMINE_SOURCES = (
    Path("data/benchmark/v0_2"),
    Path("data/derived_v0_2_round3_more"),
    Path("data/derived_v0_2_round3"),
    Path("data/derived_v0_2_round2"),
    Path("data/derived_v0_2"),
    Path("data/derived_token_logs"),
)
DEFAULT_HARDMINE_TASKS = ("code2test", "comment2context", "trace2code")


def export_hardmine_candidates(
    sources: Iterable[Path],
    out_dir: Path,
    tasks: Iterable[str] = DEFAULT_HARDMINE_TASKS,
    corpus_manifest: Path | None = None,
    require_corpus: bool = False,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    allowed_tasks = set(tasks)
    corpus_pairs = load_corpus_pairs(corpus_manifest) if corpus_manifest else None
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dropped = Counter()
    source_counts = Counter()
    for source in sources:
        for sample_path in source_sample_paths(source):
            for sample in read_jsonl(sample_path):
                sample_id = str(sample.get("id", ""))
                if not sample_id:
                    dropped["missing_id"] += 1
                    continue
                if sample_id in seen_ids:
                    dropped["duplicate_sample_id"] += 1
                    continue
                seen_ids.add(sample_id)
                drop_reason = hardmine_drop_reason(sample, allowed_tasks, corpus_pairs, require_corpus)
                if drop_reason:
                    dropped[drop_reason] += 1
                    continue
                record = dict(sample)
                record.setdefault("metadata", {})
                if isinstance(record["metadata"], dict):
                    record["metadata"] = {**record["metadata"], "hardmine_source": str(sample_path)}
                selected.append(record)
                source_counts[str(sample_path)] += 1
                if limit_samples and len(selected) >= limit_samples:
                    return write_hardmine_outputs(out_dir, selected, sources, allowed_tasks, dropped, source_counts, corpus_manifest, require_corpus)
    return write_hardmine_outputs(out_dir, selected, sources, allowed_tasks, dropped, source_counts, corpus_manifest, require_corpus)


def source_sample_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        return []
    curated = source / "samples.jsonl"
    if curated.exists():
        return [curated]
    return [source / f"{task}.jsonl" for task in DEFAULT_HARDMINE_TASKS if (source / f"{task}.jsonl").exists()]


def hardmine_drop_reason(
    sample: dict[str, Any],
    allowed_tasks: set[str],
    corpus_pairs: set[tuple[str, str]] | None,
    require_corpus: bool,
) -> str | None:
    task_type = str(sample.get("task_type", ""))
    if task_type not in allowed_tasks:
        return "excluded_task"
    repo = str(sample.get("repo", ""))
    base_commit = str(sample.get("base_commit", ""))
    if not repo or not base_commit:
        return "missing_repo_or_base"
    if require_corpus and corpus_pairs is not None and (repo, base_commit) not in corpus_pairs:
        return "missing_corpus_pair"
    if not target_gold_files(sample):
        return "missing_gold"
    query_text = query_text_for_eval(sample)
    if query_has_leakage(sample, query_text):
        return "query_leakage"
    errors = validate_sample(sample)
    if errors:
        return "schema_invalid"
    if task_type == "trace2code" and not is_real_trace_sample(sample):
        return "weak_trace"
    return None


def is_real_trace_sample(sample: dict[str, Any]) -> bool:
    query_text = query_text_for_eval(sample).lower()
    markers = ("traceback", "stack trace", "panic", "exception", "failed", "error:", " at ", ".py\", line", ".rs:")
    return any(marker in query_text for marker in markers)


def load_corpus_pairs(path: Path | None) -> set[tuple[str, str]]:
    if not path or not path.exists():
        return set()
    pairs: set[tuple[str, str]] = set()
    for record in read_jsonl(path):
        if record.get("status") != "ok":
            continue
        repo = record.get("repo")
        base_commit = record.get("base_commit")
        if repo and base_commit:
            pairs.add((str(repo), str(base_commit)))
    return pairs


def write_hardmine_outputs(
    out_dir: Path,
    selected: list[dict[str, Any]],
    sources: Iterable[Path],
    tasks: set[str],
    dropped: Counter[str],
    source_counts: Counter[str],
    corpus_manifest: Path | None,
    require_corpus: bool,
) -> dict[str, Any]:
    selected.sort(key=lambda sample: (sample.get("task_type", ""), sample.get("repo", ""), sample.get("id", "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"samples": str(out_dir / "samples.jsonl")}
    write_jsonl(out_dir / "samples.jsonl", selected)
    by_task = Counter(str(sample.get("task_type", "")) for sample in selected)
    for task in sorted(tasks):
        task_rows = [sample for sample in selected if sample.get("task_type") == task]
        task_path = out_dir / f"{task}.jsonl"
        write_jsonl(task_path, task_rows)
        outputs[task] = str(task_path)
    pairs = sorted({(str(sample.get("repo", "")), str(sample.get("base_commit", ""))) for sample in selected})
    manifest = {
        "generated_at": utc_now(),
        "sources": [str(source) for source in sources],
        "tasks": sorted(tasks),
        "corpus_manifest": str(corpus_manifest) if corpus_manifest else None,
        "require_corpus": require_corpus,
        "total": len(selected),
        "counts_by_task": dict(sorted(by_task.items())),
        "unique_pairs": len(pairs),
        "dropped": dict(sorted(dropped.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "outputs": outputs,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
