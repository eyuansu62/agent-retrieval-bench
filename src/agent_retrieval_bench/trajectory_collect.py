from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .baseline import query_provenance, query_text_for_eval
from .corpus import sample_paths_from_derived
from .io import ensure_parent, read_jsonl, utc_now


def prepare_trajectory_runs(
    *,
    derived: Path,
    out_dir: Path,
    sample_paths: Iterable[Path] | None = None,
    sample_ids: Iterable[str] | None = None,
    limit_samples: int | None = None,
    model_label: str = "agent",
    overwrite_logs: bool = False,
) -> dict[str, Any]:
    selected_ids = {str(sample_id) for sample_id in sample_ids or [] if str(sample_id)}
    paths = list(sample_paths or sample_paths_from_derived(derived))
    samples: list[dict[str, Any]] = []
    for path in paths:
        for sample in read_jsonl(path):
            sample_id = str(sample.get("id") or "")
            if not sample_id:
                continue
            if selected_ids and sample_id not in selected_ids:
                continue
            samples.append(sample)
            if limit_samples and len(samples) >= limit_samples:
                break
        if limit_samples and len(samples) >= limit_samples:
            break

    prompts_dir = out_dir / "prompts"
    logs_dir = out_dir / "logs"
    answers_dir = out_dir / "answers"
    for directory in (prompts_dir, logs_dir, answers_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    existing_logs = 0
    for sample in samples:
        sample_id = str(sample["id"])
        prompt_path = prompts_dir / f"{sample_id}.md"
        trajectory_path = logs_dir / f"{sample_id}.jsonl"
        answer_path = answers_dir / f"{sample_id}.md"
        prompt_path.write_text(build_trajectory_prompt(sample, trajectory_path=trajectory_path, answer_path=answer_path), encoding="utf-8")
        if trajectory_path.exists():
            existing_logs += 1
            if overwrite_logs:
                trajectory_path.unlink()
        trajectory_path.touch(exist_ok=True)
        run_rows.append(
            {
                "sample_id": sample_id,
                "task_type": sample.get("task_type"),
                "repo": sample.get("repo"),
                "base_commit": sample.get("base_commit"),
                "query_provenance": query_provenance(sample),
                "prompt_path": str(prompt_path),
                "trajectory_path": str(trajectory_path),
                "answer_path": str(answer_path),
                "record_command_template": (
                    "arb record-trajectory-step "
                    f"--log {trajectory_path} --sample-id {sample_id} "
                    "--tool read --path <repo-relative-path> --start-line <n> --end-line <m>"
                ),
            }
        )

    runs_path = out_dir / "runs.jsonl"
    write_jsonl(runs_path, run_rows)
    manifest = {
        "mode": "trajectory_collection_runs",
        "derived": str(derived),
        "model_label": model_label,
        "samples": len(run_rows),
        "sample_ids": [row["sample_id"] for row in run_rows],
        "runs": str(runs_path),
        "prompts_dir": str(prompts_dir),
        "logs_dir": str(logs_dir),
        "answers_dir": str(answers_dir),
        "existing_logs": existing_logs,
        "overwrite_logs": overwrite_logs,
        "generated_at": utc_now(),
    }
    manifest_path = out_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_trajectory_prompt(sample: dict[str, Any], *, trajectory_path: Path, answer_path: Path) -> str:
    query = query_text_for_eval(sample)
    provenance = query_provenance(sample) or ""
    lines = [
        f"# ARB Trajectory Run: {sample.get('id')}",
        "",
        "Run the coding-agent task using only the query below and the repository checkout at the base commit.",
        "Do not inspect benchmark gold labels, evaluation details, or release answer files while solving the task.",
        "",
        "## Metadata",
        "",
        f"- sample_id: `{sample.get('id')}`",
        f"- task_type: `{sample.get('task_type')}`",
        f"- repo: `{sample.get('repo')}`",
        f"- base_commit: `{sample.get('base_commit')}`",
        f"- query_provenance: `{provenance}`",
        f"- trajectory_log: `{trajectory_path}`",
        f"- answer_path: `{answer_path}`",
        "",
        "## Logging Contract",
        "",
        "Record every file or line range that enters the agent context.",
        "",
        "```bash",
        "arb record-trajectory-step \\",
        f"  --log {trajectory_path} \\",
        f"  --sample-id {sample.get('id')} \\",
        "  --tool read \\",
        "  --path <repo-relative-path> \\",
        "  --start-line <n> \\",
        "  --end-line <m>",
        "```",
        "",
        "Use `--final` for context retained in the final answer and `--used` for context directly used in the answer or patch.",
        "",
    ]
    if provenance == "review_comment":
        lines.extend(
            [
                "## Review-Comment Search Guidance",
                "",
                "Review comments often point at a reviewed hunk while the needed follow-up context is elsewhere.",
                "When the comment mentions semver, error wording, regression coverage, process protocols, or conversion registries, proactively inspect likely policy, config, test, and implementation follow-up files.",
                "Record those files or line ranges when they enter context, even if they are not the file named in the comment.",
                "",
            ]
        )
    lines.extend(
        [
            "## Query",
            "",
            "```json",
            query,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def record_trajectory_step(
    *,
    log_path: Path,
    sample_id: str,
    path: str,
    step: int | None = None,
    tool: str = "read",
    start_line: int | None = None,
    end_line: int | None = None,
    kind: str = "block",
    symbol: str = "",
    content_hash: str = "",
    repo_root: Path | None = None,
    is_final_context: bool = False,
    is_utilized_context: bool = False,
    run_id: str = "",
    model_label: str = "",
) -> dict[str, Any]:
    if not sample_id:
        raise ValueError("sample_id is required.")
    if not path:
        raise ValueError("path is required.")
    if (start_line is None) ^ (end_line is None):
        raise ValueError("start_line and end_line must be provided together.")
    if start_line is not None and end_line is not None and (start_line <= 0 or end_line < start_line):
        raise ValueError("Invalid line range.")
    if step is None:
        step = next_step(log_path, sample_id)
    if not content_hash and repo_root:
        content_hash = file_content_hash(repo_root=repo_root, path=path, start_line=start_line, end_line=end_line)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "step": step,
        "tool": tool,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "kind": kind,
        "symbol": symbol,
        "content_hash": content_hash,
        "is_final_context": is_final_context,
        "is_utilized_context": is_utilized_context,
        "logged_at": utc_now(),
    }
    if run_id:
        row["run_id"] = run_id
    if model_label:
        row["model"] = model_label
    append_jsonl(log_path, row)
    return row


def next_step(log_path: Path, sample_id: str) -> int:
    if not log_path.exists():
        return 1
    max_step = 0
    for row in read_jsonl(log_path):
        if str(row.get("sample_id") or "") != sample_id:
            continue
        try:
            max_step = max(max_step, int(row.get("step") or 0))
        except (TypeError, ValueError):
            continue
    return max_step + 1


def file_content_hash(*, repo_root: Path, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else repo_root / candidate
    if not full_path.exists() or not full_path.is_file():
        return ""
    if start_line is None or end_line is None:
        data = full_path.read_bytes()
    else:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        data = "".join(lines[start_line - 1 : end_line]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
