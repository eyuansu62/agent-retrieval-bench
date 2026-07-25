from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .corpus import sample_paths_from_derived
from .io import ensure_parent, read_jsonl

STRICT_OPENAI_KEY_RE = re.compile(r"(^|[^A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}")


def audit_strict_context_run(
    *,
    base: Path,
    run_name: str,
    derived: Path,
    corpus_manifest: Path,
    out_path: Path | None = None,
    markdown_out: Path | None = None,
    min_reads: int = 3,
    max_reads: int = 9,
    extra_scan_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    logs_dir = base / f"logs_{run_name}"
    answers_dir = base / f"answers_{run_name}"
    traces_dir = base / f"traces_{run_name}"
    manifest_path = base / f"{run_name}_manifest.json"

    samples = {str(row.get("id") or ""): row for path in sample_paths_from_derived(derived) for row in read_jsonl(path)}
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(traces_dir.glob("*.json"))]
    corpus_paths_by_key, corpus_decode_errors = load_corpus_paths_for_traces(corpus_manifest, traces)

    read_step_distribution: Counter[int] = Counter()
    final_file_distribution: Counter[int] = Counter()
    task_counts: Counter[str] = Counter()
    task_read_steps: Counter[str] = Counter()
    task_final_files: Counter[str] = Counter()
    issues: dict[str, list[Any]] = {
        "empty_logs": [],
        "below_min_reads": [],
        "above_max_reads": [],
        "duplicate_read_samples": [],
        "missing_corpus_keys": [],
        "corpus_decode_errors": corpus_decode_errors,
        "missing_read_path_samples": [],
        "missing_final_path_samples": [],
        "empty_final_samples": [],
        "final_unread_samples": [],
        "raw_unread_samples": [],
        "invalid_action_samples": [],
        "recovery_action_samples": [],
        "strict_openai_key_hits": [],
    }

    read_steps_total = 0
    final_files_total = 0
    suggested_unread_total = 0
    raw_unread_total = 0
    max_final_files = {"sample_id": None, "count": 0}

    for trace in traces:
        sample_id = str(trace.get("sample_id") or "")
        sample = samples.get(sample_id, {})
        task = str(sample.get("task_type") or trace.get("query", {}).get("task_type") or "unknown")
        read_paths = [str(path) for path in trace.get("read_paths") or []]
        final_files = [str(path) for path in trace.get("final_files") or []]
        raw_final_files = [str(path) for path in trace.get("raw_final_files") or final_files]
        suggested_unread_files = [str(path) for path in trace.get("suggested_unread_files") or []]
        read_set = set(read_paths)

        read_count = len(read_paths)
        final_count = len(final_files)
        read_steps_total += read_count
        final_files_total += final_count
        suggested_unread_total += len(suggested_unread_files)
        read_step_distribution[read_count] += 1
        final_file_distribution[final_count] += 1
        task_counts[task] += 1
        task_read_steps[task] += read_count
        task_final_files[task] += final_count
        if final_count > int(max_final_files["count"]):
            max_final_files = {"sample_id": sample_id, "count": final_count}

        if read_count == 0:
            issues["empty_logs"].append(sample_id)
        if read_count < min_reads:
            issues["below_min_reads"].append({"sample_id": sample_id, "read_steps": read_count})
        if read_count > max_reads:
            issues["above_max_reads"].append({"sample_id": sample_id, "read_steps": read_count})
        duplicate_reads = [path for path, count in Counter(read_paths).items() if count > 1]
        if duplicate_reads:
            issues["duplicate_read_samples"].append({"sample_id": sample_id, "duplicates": duplicate_reads})

        final_unread = [path for path in final_files if path not in read_set]
        raw_unread = [path for path in raw_final_files if path not in read_set]
        raw_unread_total += len(raw_unread)
        if final_unread:
            issues["final_unread_samples"].append({"sample_id": sample_id, "paths": final_unread})
        if raw_unread:
            issues["raw_unread_samples"].append(
                {"sample_id": sample_id, "paths": raw_unread, "suggested_unread_files": suggested_unread_files}
            )
        if not final_files:
            issues["empty_final_samples"].append(sample_id)

        key = (str(trace.get("repo") or ""), str(trace.get("base_commit") or ""))
        corpus_paths = corpus_paths_by_key.get(key)
        if corpus_paths is None:
            issues["missing_corpus_keys"].append({"sample_id": sample_id, "repo": key[0], "base_commit": key[1]})
            corpus_paths = set()
        missing_reads = [path for path in read_paths if path not in corpus_paths]
        missing_finals = [path for path in final_files if path not in corpus_paths]
        if missing_reads:
            issues["missing_read_path_samples"].append({"sample_id": sample_id, "paths": missing_reads})
        if missing_finals:
            issues["missing_final_path_samples"].append({"sample_id": sample_id, "paths": missing_finals})

        invalid_actions = [
            action
            for action in trace.get("actions", [])
            if str((action.get("parsed") or {}).get("action") or "").lower() == "invalid"
        ]
        if invalid_actions:
            issues["invalid_action_samples"].append({"sample_id": sample_id, "count": len(invalid_actions)})
        recovery_actions = [action for action in trace.get("actions", []) if action.get("recovery")]
        if recovery_actions:
            issues["recovery_action_samples"].append({"sample_id": sample_id, "count": len(recovery_actions)})

    scan_paths = [
        *logs_dir.glob("*.jsonl"),
        *answers_dir.glob("*.md"),
        *traces_dir.glob("*.json"),
        *[path for path in [manifest_path, *extra_scan_paths] if path.exists()],
    ]
    issues["strict_openai_key_hits"] = [
        str(path) for path in scan_paths if STRICT_OPENAI_KEY_RE.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]

    log_files = sorted(logs_dir.glob("*.jsonl"))
    answer_files = sorted(answers_dir.glob("*.md"))
    trace_files = sorted(traces_dir.glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    audit = {
        "generated_at": utc_now(),
        "run_name": run_name,
        "model": manifest.get("model"),
        "paths": {
            "base": str(base),
            "logs_dir": str(logs_dir),
            "answers_dir": str(answers_dir),
            "traces_dir": str(traces_dir),
            "manifest": str(manifest_path),
            "derived": str(derived),
            "corpus_manifest": str(corpus_manifest),
        },
        "manifest": {
            "completed": manifest.get("completed"),
            "skipped_existing": manifest.get("skipped_existing"),
            "failures": manifest.get("failures"),
            "runtime_seconds": manifest.get("runtime_seconds"),
            "generated_at": manifest.get("generated_at"),
        },
        "counts": {
            "samples": len(samples),
            "logs": len(log_files),
            "answers": len(answer_files),
            "traces": len(trace_files),
            "read_steps_total": read_steps_total,
            "final_files_total": final_files_total,
            "suggested_unread_files_total": suggested_unread_total,
            "raw_unread_files_total": raw_unread_total,
        },
        "read_step_distribution": dict(sorted(read_step_distribution.items())),
        "final_file_distribution": dict(sorted(final_file_distribution.items())),
        "by_task": {
            task: {
                "samples": task_counts[task],
                "read_steps": task_read_steps[task],
                "avg_read_steps": task_read_steps[task] / task_counts[task],
                "final_files": task_final_files[task],
                "avg_final_files": task_final_files[task] / task_counts[task],
            }
            for task in sorted(task_counts)
        },
        "issues": issues,
        "max_final_files": max_final_files,
        "verdict": "pass" if strict_context_audit_passes(issues) else "fail",
    }
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out:
        ensure_parent(markdown_out)
        markdown_out.write_text(strict_context_audit_markdown(audit), encoding="utf-8")
    return audit


def strict_context_audit_passes(issues: dict[str, list[Any]]) -> bool:
    blocking = (
        "empty_logs",
        "below_min_reads",
        "duplicate_read_samples",
        "missing_corpus_keys",
        "missing_read_path_samples",
        "missing_final_path_samples",
        "empty_final_samples",
        "final_unread_samples",
        "strict_openai_key_hits",
    )
    return all(not issues.get(key) for key in blocking)


def strict_context_audit_markdown(audit: dict[str, Any]) -> str:
    issues = audit["issues"]
    counts = audit["counts"]
    lines = [
        "# Strict Context Trajectory Audit",
        "",
        f"- Generated at: {audit['generated_at']}",
        f"- Run: `{audit['run_name']}`",
        f"- Model: `{audit.get('model') or ''}`",
        f"- Verdict: `{audit['verdict']}`",
        f"- logs/answers/traces: {counts['logs']} / {counts['answers']} / {counts['traces']}",
        f"- read_steps_total: {counts['read_steps_total']}",
        f"- final_files_total: {counts['final_files_total']}",
        f"- suggested_unread_files_total: {counts['suggested_unread_files_total']}",
        f"- raw_unread_files_total: {counts['raw_unread_files_total']}",
        "",
        "## Structural Checks",
        "",
    ]
    for key in (
        "empty_logs",
        "below_min_reads",
        "above_max_reads",
        "duplicate_read_samples",
        "missing_corpus_keys",
        "corpus_decode_errors",
        "missing_read_path_samples",
        "missing_final_path_samples",
        "empty_final_samples",
        "final_unread_samples",
        "raw_unread_samples",
        "invalid_action_samples",
        "recovery_action_samples",
        "strict_openai_key_hits",
    ):
        lines.append(f"- {key}: {len(issues.get(key) or [])}")
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- read_step_distribution: {audit['read_step_distribution']}",
            f"- final_file_distribution: {audit['final_file_distribution']}",
            f"- max_final_files: {audit['max_final_files']}",
            "",
            "## Task Split",
            "",
        ]
    )
    for task, values in audit["by_task"].items():
        lines.append(
            f"- {task}: samples={values['samples']}, read_steps={values['read_steps']}, "
            f"avg_read_steps={values['avg_read_steps']:.3f}, final_files={values['final_files']}, "
            f"avg_final_files={values['avg_final_files']:.3f}"
        )
    lines.extend(["", "The strict-context invariant is `final_files - read_paths = empty` for every sample.", ""])
    return "\n".join(lines)


def load_corpus_paths_for_traces(
    corpus_manifest: Path, traces: Iterable[dict[str, Any]]
) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, Any]]]:
    needed = {(str(trace.get("repo") or ""), str(trace.get("base_commit") or "")) for trace in traces}
    paths_by_key: dict[tuple[str, str], set[str]] = {}
    decode_errors: list[dict[str, Any]] = []
    for row in read_jsonl(corpus_manifest):
        key = (str(row.get("repo") or ""), str(row.get("base_commit") or ""))
        if key not in needed:
            continue
        chunks_path = Path(str(row.get("chunks_path") or row.get("path") or ""))
        if not chunks_path.exists() and not chunks_path.is_absolute():
            chunks_path = corpus_manifest.parent / chunks_path
        paths: set[str] = set()
        for line_no, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                decode_errors.append({"path": str(chunks_path), "line": line_no, "error": str(exc)})
                continue
            path = str(chunk.get("path") or "")
            if path:
                paths.add(path)
        paths_by_key[key] = paths
    return paths_by_key, decode_errors


def package_trajectory_release(
    *,
    data_root: Path,
    release_dir: Path,
    archive_name: str,
    include_paths: Iterable[Path],
    checksum_path_in_release: str,
) -> dict[str, Any]:
    expected_release_dir = (data_root / checksum_path_in_release).resolve()
    if release_dir.resolve() != expected_release_dir:
        raise ValueError(
            f"release_dir must match data_root/checksum_path_in_release: {release_dir} != {expected_release_dir}"
        )
    release_dir.mkdir(parents=True, exist_ok=True)
    archive_path = release_dir / archive_name
    checksum_path = release_dir / f"{archive_name}.sha256"
    archive_path.unlink(missing_ok=True)
    checksum_path.unlink(missing_ok=True)

    members = sorted(to_archive_member(data_root, path) for path in include_paths)
    forbidden = [member for member in members if forbidden_release_member(member)]
    if forbidden:
        raise ValueError(f"forbidden release members: {forbidden[:10]}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        list_path = Path(handle.name)
        handle.write("\n".join(members))
        handle.write("\n")
    try:
        subprocess.run(["tar", "--zstd", "-cf", str(archive_path), "-C", str(data_root), "-T", str(list_path)], check=True)
    finally:
        list_path.unlink(missing_ok=True)

    sha = sha256_file(archive_path)
    checksum_path.write_text(f"{sha}  {checksum_path_in_release}/{archive_name}\n", encoding="utf-8")
    member_count = len(subprocess.check_output(["tar", "-tf", str(archive_path)], text=True).splitlines())
    return {
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "sha256": sha,
        "archive_bytes": archive_path.stat().st_size,
        "members": member_count,
        "checksum_entry": f"{sha}  {checksum_path_in_release}/{archive_name}",
        "generated_at": utc_now(),
    }


def v1_3_openai_strict_context_include_paths(
    *,
    data_root: Path = Path("data"),
    base: Path = Path("data/trajectory_runs/v1_4/v1_3_all_gpt54mini"),
    run_name: str = "openai_gpt54mini_v2_strict_context",
) -> list[Path]:
    return [
        *sorted((base / f"logs_{run_name}").glob("*.jsonl")),
        *sorted((base / f"answers_{run_name}").glob("*.md")),
        *sorted((base / f"traces_{run_name}").glob("*.json")),
        base / f"{run_name}_manifest.json",
        data_root / "eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json",
        data_root / "eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl",
        data_root / "reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context.md",
        data_root / "reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context.json",
        data_root / "reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.md",
        data_root / "reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.json",
    ]


def to_archive_member(data_root: Path, path: Path) -> str:
    path = path.resolve()
    data_root = data_root.resolve()
    return path.relative_to(data_root).as_posix()


def forbidden_release_member(member: str) -> bool:
    return (
        "smoke" in member
        or member.startswith("benchmark/")
        or member.startswith("corpus/")
        or "__pycache__" in member
        or member.endswith(".pyc")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
