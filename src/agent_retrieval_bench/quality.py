from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .filters import contains_raw_patch_marker
from .io import read_jsonl


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not sample.get("id"):
        errors.append("missing id")
    if sample.get("task_type") not in {"comment2context", "code2test", "testlog2code", "trace2code"}:
        errors.append("unknown task_type")
    if not sample.get("repo") or "/" not in sample.get("repo", ""):
        errors.append("invalid repo")
    if not sample.get("base_commit"):
        errors.append("missing base_commit")
    corpus = sample.get("candidate_corpus") or {}
    if corpus.get("type") != "repo_at_base_commit":
        errors.append("candidate_corpus must be repo_at_base_commit")
    if corpus.get("base_commit") != sample.get("base_commit"):
        errors.append("candidate_corpus base_commit mismatch")
    gold = sample.get("gold") or {}
    if sample.get("task_type") == "comment2context" and sample.get("version", 1) >= 2:
        if not _gold_paths(gold.get("given_files") or []):
            errors.append("comment2context missing given_files")
        if not _gold_paths(gold.get("must_context_files") or gold.get("context_files") or []):
            errors.append("comment2context missing must_context_files")
    if not gold.get("root_cause_files") and sample.get("task_type") != "code2test":
        errors.append("missing root_cause_files")
    if sample.get("task_type") == "code2test" and not gold.get("related_tests"):
        errors.append("code2test missing related_tests")
    query_text = json.dumps(sample.get("query") or {}, ensure_ascii=False).replace("\\n", "\n")
    if contains_raw_patch_marker(query_text):
        errors.append("query contains raw patch markers")
    fix_commit = gold.get("fix_commit")
    if fix_commit and fix_commit in query_text:
        errors.append("query contains fix commit")
    return errors


def _gold_paths(values: list[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return paths


def validate_samples(path: Path) -> dict[str, Any]:
    total = 0
    invalid = 0
    errors_by_id: dict[str, list[str]] = {}
    for sample in read_jsonl(path):
        total += 1
        errors = validate_sample(sample)
        if errors:
            invalid += 1
            errors_by_id[sample.get("id", f"row-{total}")] = errors
    return {"path": str(path), "total": total, "invalid": invalid, "errors_by_id": errors_by_id}
