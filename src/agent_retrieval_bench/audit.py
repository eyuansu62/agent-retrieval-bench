from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .filters import contains_raw_patch_marker
from .io import ensure_parent, read_jsonl, stable_id, truncate_text

TASK_TYPES = ("comment2context", "code2test", "testlog2code", "trace2code")
AUDIT_FIELDS = ("sample_id", "task_type", "repo", "query_excerpt", "gold_files", "verdict", "reason", "keep", "notes")
VERDICTS = ("valid", "noisy", "leaked", "ambiguous")


def write_audit_sample(
    derived_dir: Path,
    out_dir: Path,
    per_task: int = 20,
    seed: int = 13,
    tasks: Iterable[str] | None = TASK_TYPES,
    formats: Iterable[str] = ("jsonl", "csv"),
) -> dict[str, Any]:
    rows = build_audit_rows(derived_dir, per_task=per_task, seed=seed, tasks=tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    requested_formats = {fmt.lower() for fmt in formats}
    if "jsonl" in requested_formats:
        path = out_dir / "audit_samples.jsonl"
        _write_jsonl(path, rows)
        written["jsonl"] = str(path)
    if "csv" in requested_formats:
        path = out_dir / "audit_samples.csv"
        _write_csv(path, rows)
        written["csv"] = str(path)
    counts = Counter(row["task_type"] for row in rows)
    return {"rows": len(rows), "per_task": per_task, "seed": seed, "counts_by_task": dict(counts), "outputs": written}


def build_audit_rows(
    derived_dir: Path,
    per_task: int = 20,
    seed: int = 13,
    tasks: Iterable[str] | None = TASK_TYPES,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_type in tasks or TASK_TYPES:
        samples_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in read_jsonl(derived_dir / f"{task_type}.jsonl"):
            if sample.get("task_type") == task_type and sample.get("id"):
                samples_by_repo[sample.get("repo", "")].append(sample)
        selected = _balanced_sample(samples_by_repo, per_task, seed, task_type)
        rows.extend(audit_row(sample) for sample in selected)
    rows.sort(key=lambda row: (row["task_type"], row["repo"], row["sample_id"]))
    return rows


def audit_row(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": str(sample.get("id", "")),
        "task_type": str(sample.get("task_type", "")),
        "repo": str(sample.get("repo", "")),
        "query_excerpt": query_excerpt(sample),
        "gold_files": "; ".join(gold_files_for_task(sample)),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "",
    }


def summarize_audit(audit_path: Path, out_path: Path | None = None, keep_list_path: Path | None = None) -> dict[str, Any]:
    rows = read_audit_rows(audit_path)
    verdict_counts: dict[str, Counter[str]] = defaultdict(Counter)
    keep_rows: list[dict[str, str]] = []
    for row in rows:
        task_type = row.get("task_type", "")
        verdict = normalize_verdict(row.get("verdict", ""))
        verdict_counts[task_type][verdict] += 1
        if should_keep(row):
            keep_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "task_type": task_type,
                    "repo": row.get("repo", ""),
                    "verdict": verdict,
                }
            )

    by_task: dict[str, Any] = {}
    for task_type in sorted(verdict_counts):
        total = sum(verdict_counts[task_type].values())
        by_task[task_type] = {
            "total": total,
            "counts": dict(verdict_counts[task_type]),
            "ratios": {
                verdict: (verdict_counts[task_type][verdict] / total if total else 0.0)
                for verdict in (*VERDICTS, "pending", "other")
            },
        }
    summary = {"audit_path": str(audit_path), "total": len(rows), "kept": len(keep_rows), "by_task": by_task}
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if keep_list_path:
        _write_jsonl(keep_list_path, keep_rows)
    return summary


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{field: row.get(field, "") for field in AUDIT_FIELDS} for row in csv.DictReader(handle)]
    rows: list[dict[str, str]] = []
    for record in read_jsonl(path):
        rows.append({field: str(record.get(field, "")) for field in AUDIT_FIELDS})
    return rows


def normalize_verdict(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return "pending"
    if normalized in VERDICTS:
        return normalized
    return "other"


def should_keep(row: dict[str, str]) -> bool:
    keep = (row.get("keep") or "").strip().lower()
    if keep in {"1", "true", "yes", "y", "keep"}:
        return True
    if keep in {"0", "false", "no", "n", "drop"}:
        return False
    return normalize_verdict(row.get("verdict")) == "valid"


def query_excerpt(sample: dict[str, Any], limit: int = 1200) -> str:
    query = sample.get("query") or {}
    text = _flatten_query(query)
    fix_commit = ((sample.get("gold") or {}).get("fix_commit") or "").strip()
    if fix_commit:
        text = text.replace(fix_commit, "[fix_commit]")
    text = _strip_patch_markers(text)
    return truncate_text(text, limit)


def gold_files_for_task(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    if sample.get("task_type") == "code2test":
        return _dedupe(gold.get("related_tests") or [])
    if sample.get("task_type") == "comment2context":
        context_files = _gold_paths(gold.get("must_context_files") or gold.get("context_files") or [])
        if context_files:
            return context_files
    return _dedupe((gold.get("root_cause_files") or []) + (gold.get("related_tests") or []))


def _gold_paths(values: list[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return _dedupe(paths)


def has_query_leakage(sample: dict[str, Any]) -> bool:
    text = query_excerpt(sample, limit=100_000)
    fix_commit = ((sample.get("gold") or {}).get("fix_commit") or "").strip()
    return contains_raw_patch_marker(text) or bool(fix_commit and fix_commit in text)


def _balanced_sample(samples_by_repo: dict[str, list[dict[str, Any]]], limit: int, seed: int, task_type: str) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    shuffled: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for repo, samples in sorted(samples_by_repo.items()):
        ordered = sorted(samples, key=lambda sample: sample.get("id", ""))
        rng = random.Random(stable_id(seed, task_type, repo))
        rng.shuffle(ordered)
        shuffled[repo] = ordered
        total += len(ordered)
    if total <= limit:
        return sorted((sample for samples in shuffled.values() for sample in samples), key=lambda sample: sample.get("id", ""))

    selected: list[dict[str, Any]] = []
    repo_names = sorted(shuffled)
    cursor = 0
    while len(selected) < limit and any(shuffled.values()):
        repo = repo_names[cursor % len(repo_names)]
        cursor += 1
        if shuffled[repo]:
            selected.append(shuffled[repo].pop())
    return selected


def _flatten_query(query: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in query.items():
        text = _stringify(value)
        if text:
            parts.append(f"{key}: {text}")
    return "\n".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_stringify(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _strip_patch_markers(text: str) -> str:
    kept: list[str] = []
    for line in text.replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        if "diff --git" in stripped or stripped.startswith(("+++ ", "--- ")):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})
            count += 1
    return count


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
