from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .audit import TASK_TYPES
from .io import ensure_parent, read_jsonl

V0_TASK_TYPES = ("code2test", "comment2context", "trace2code")


def load_keep_ids(keep_list: Path | None, valid_only: bool = True, tasks: Iterable[str] | None = None) -> set[str] | None:
    if keep_list is None or not keep_list.exists():
        return None
    allowed_tasks = set(tasks or [])
    keep_ids: set[str] = set()
    for record in read_jsonl(keep_list):
        sample_id = record.get("sample_id")
        verdict = str(record.get("verdict", "")).lower()
        task_type = str(record.get("task_type", ""))
        if allowed_tasks and task_type and task_type not in allowed_tasks:
            continue
        if sample_id and (not valid_only or not verdict or verdict == "valid"):
            keep_ids.add(str(sample_id))
    return keep_ids


def filter_samples(samples: Iterable[dict[str, Any]], keep_ids: set[str] | None) -> Iterable[dict[str, Any]]:
    for sample in samples:
        if keep_ids is None or sample.get("id") in keep_ids:
            yield sample


def sample_paths_from_dir(derived_dir: Path, tasks: Iterable[str] = TASK_TYPES) -> list[Path]:
    return [derived_dir / f"{task}.jsonl" for task in tasks if (derived_dir / f"{task}.jsonl").exists()]


def export_curated_samples(
    derived_dir: Path,
    keep_list: Path,
    out_dir: Path,
    tasks: Iterable[str] = V0_TASK_TYPES,
    valid_only: bool = True,
) -> dict[str, Any]:
    allowed_tasks = set(tasks)
    keep_ids = load_keep_ids(keep_list, valid_only=valid_only, tasks=allowed_tasks) or set()
    selected: list[dict[str, Any]] = []
    found_ids: set[str] = set()
    for path in sample_paths_from_dir(derived_dir):
        for sample in read_jsonl(path):
            sample_id = sample.get("id")
            if sample_id in keep_ids and sample.get("task_type") in allowed_tasks:
                selected.append(sample)
                found_ids.add(sample_id)
    selected.sort(key=lambda sample: (sample.get("task_type", ""), sample.get("repo", ""), sample.get("id", "")))

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "samples.jsonl"
    _write_jsonl(all_path, selected)
    counts = Counter(sample.get("task_type", "") for sample in selected)
    outputs = {"samples": str(all_path)}
    for task in allowed_tasks:
        task_path = out_dir / f"{task}.jsonl"
        _write_jsonl(task_path, [sample for sample in selected if sample.get("task_type") == task])
        outputs[task] = str(task_path)

    manifest = {
        "derived_dir": str(derived_dir),
        "keep_list": str(keep_list),
        "valid_only": valid_only,
        "tasks": sorted(allowed_tasks),
        "total": len(selected),
        "counts_by_task": dict(counts),
        "missing_keep_ids": sorted(keep_ids - found_ids),
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
