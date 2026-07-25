#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BENCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_retrieval_bench.baseline import query_text_for_eval  # noqa: E402
from agent_retrieval_bench.io import read_jsonl  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Codex CLI trajectory collection over ARB samples.")
    parser.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3/samples.jsonl"))
    parser.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/trajectory_runs/v1_4/codex_cli_gpt54_corpus_full"))
    parser.add_argument("--run-name", default="codex_cli_gpt54_corpus_full")
    parser.add_argument("--model", default="gpt-5.4", help="Pass empty string to use Codex CLI default.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp/arb_codex_cli_full"))
    parser.add_argument("--seed-details", type=Path, help="Optional ranked-list details JSONL used to inject seed paths into the prompt.")
    parser.add_argument("--seed-top-k", type=int, default=20)
    parser.add_argument("--seed-label", default="offline retriever")
    parser.add_argument(
        "--seed-note",
        default="These candidates are ranked and unverified. They may include noise; verify any file by reading it before relying on it.",
    )
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("prompts", "logs", "answers", "events", "stderr", "status"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    samples_path = resolve_path(args.samples)
    manifest_path = resolve_path(args.corpus_manifest)
    corpus_index = load_corpus_index(manifest_path)
    samples = select_samples(samples_path, args.sample_id, args.limit)
    seed_candidates = load_seed_candidates(resolve_path(args.seed_details), args.seed_top_k) if args.seed_details else {}

    manifest = {
        "mode": "codex_cli_trajectory_collection",
        "run_name": args.run_name,
        "model": args.model or "codex-cli-default",
        "samples_path": str(samples_path),
        "corpus_manifest": str(manifest_path),
        "run_dir": str(run_dir),
        "sample_count": len(samples),
        "started_at": utc_now(),
        "materialization": "corpus_file_chunks",
        "codex_bin": args.codex_bin,
        "timeout_seconds": args.timeout_seconds,
        "temp_root": str(args.temp_root),
        "seed_details": str(resolve_path(args.seed_details)) if args.seed_details else "",
        "seed_top_k": args.seed_top_k if args.seed_details else 0,
        "seed_label": args.seed_label if args.seed_details else "",
        "seed_note": args.seed_note if args.seed_details else "",
    }
    write_json(run_dir / "manifest.json", manifest)

    completed = 0
    skipped = 0
    failed = 0
    started = time.monotonic()
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id") or "")
        if not sample_id:
            continue
        state_path = run_dir / "status" / f"{sample_id}.json"
        if not args.force and is_success_state(state_path):
            skipped += 1
            print(json.dumps({"event": "skip_existing", "sample_id": sample_id, "index": index}), flush=True)
            continue

        status = run_one_sample(
            sample=sample,
            corpus_index=corpus_index,
            run_dir=run_dir,
            run_name=args.run_name,
            model=args.model,
            codex_bin=args.codex_bin,
            timeout_seconds=args.timeout_seconds,
            temp_root=resolve_path(args.temp_root),
            index=index,
            total=len(samples),
            keep_worktree=args.keep_worktrees,
            seed_paths=seed_candidates.get(sample_id, []),
            seed_label=args.seed_label,
            seed_note=args.seed_note,
        )
        write_json(state_path, status)
        append_jsonl(run_dir / "status.jsonl", status)
        if status["status"] == "success":
            completed += 1
        else:
            failed += 1
        print(json.dumps(status_summary(status, started), sort_keys=True), flush=True)

    manifest.update(
        {
            "completed": completed,
            "skipped_existing": skipped,
            "failed": failed,
            "finished_at": utc_now(),
            "runtime_seconds": round(time.monotonic() - started, 3),
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    return 0 if failed == 0 else 1


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else BENCH_ROOT / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_corpus_index(manifest_path: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for row in read_jsonl(manifest_path):
        if row.get("status") != "ok":
            continue
        repo = str(row.get("repo") or "")
        commit = str(row.get("base_commit") or "")
        chunks_path = Path(str(row.get("chunks_path") or ""))
        if repo and commit and chunks_path:
            index[(repo, commit)] = chunks_path if chunks_path.is_absolute() else BENCH_ROOT / chunks_path
    return index


def select_samples(samples_path: Path, sample_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    selected = set(sample_ids)
    samples: list[dict[str, Any]] = []
    for sample in read_jsonl(samples_path):
        sample_id = str(sample.get("id") or "")
        if selected and sample_id not in selected:
            continue
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples


def load_seed_candidates(details_path: Path, top_k: int) -> dict[str, list[str]]:
    if top_k <= 0:
        raise ValueError("--seed-top-k must be positive")
    candidates: dict[str, list[str]] = {}
    for row in read_jsonl(details_path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        paths = []
        seen = set()
        for path in row.get("top_files") or []:
            path = str(path or "")
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
            if len(paths) >= top_k:
                break
        candidates[sample_id] = paths
    return candidates


def is_success_state(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "success"
    except json.JSONDecodeError:
        return False


def run_one_sample(
    *,
    sample: dict[str, Any],
    corpus_index: dict[tuple[str, str], Path],
    run_dir: Path,
    run_name: str,
    model: str,
    codex_bin: str,
    timeout_seconds: int,
    temp_root: Path,
    index: int,
    total: int,
    keep_worktree: bool,
    seed_paths: list[str],
    seed_label: str,
    seed_note: str,
) -> dict[str, Any]:
    sample_id = str(sample["id"])
    repo = str(sample.get("repo") or "")
    commit = str(sample.get("base_commit") or "")
    chunks_path = corpus_index.get((repo, commit))
    started_at = utc_now()
    started = time.monotonic()
    if not chunks_path:
        return base_status(sample, index, total, "missing_corpus", started_at, started, message=f"{repo}@{commit}")

    worktree = temp_root / sample_id
    log_path = run_dir / "logs" / f"{sample_id}.jsonl"
    local_log_path = worktree / ".arb_trajectory_log.jsonl"
    answer_path = run_dir / "answers" / f"{sample_id}.md"
    prompt_path = run_dir / "prompts" / f"{sample_id}.md"
    events_path = run_dir / "events" / f"{sample_id}.jsonl"
    stderr_path = run_dir / "stderr" / f"{sample_id}.txt"

    try:
        file_count = materialize_worktree(chunks_path, worktree)
        prompt = build_prompt(
            sample=sample,
            log_path=local_log_path,
            answer_path=answer_path,
            run_name=run_name,
            model_label=model or "codex-cli-default",
            file_count=file_count,
            seed_paths=seed_paths,
            seed_label=seed_label,
            seed_note=seed_note,
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        if log_path.exists():
            log_path.unlink()
        if local_log_path.exists():
            local_log_path.unlink()
        log_path.touch()

        cmd = [
            codex_bin,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(worktree),
            "--sandbox",
            "workspace-write",
            "-o",
            str(answer_path),
        ]
        if model:
            cmd.extend(["-m", model])
        cmd.append("-")
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        with events_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                stdout=stdout,
                stderr=stderr,
                cwd=str(worktree),
                env=env,
                timeout=timeout_seconds,
            )
        self_logged_rows = read_jsonl(local_log_path) if local_log_path.exists() else []
        event_rows = extract_event_read_steps(
            events_path=events_path,
            sample_id=sample_id,
            worktree=worktree,
            run_name=run_name,
            model_label=model or "codex-cli-default",
        )
        write_jsonl(log_path, merge_steps([*event_rows, *self_logged_rows]))
        log_entries = count_jsonl(log_path)
        answer_exists = answer_path.exists() and answer_path.stat().st_size > 0
        status = "success" if proc.returncode == 0 and answer_exists else "codex_failed"
        if status == "success" and log_entries == 0:
            status = "empty_log"
        return {
            **base_status(sample, index, total, status, started_at, started),
            "returncode": proc.returncode,
            "file_count": file_count,
            "log_entries": log_entries,
            "answer_exists": answer_exists,
            "worktree": str(worktree) if keep_worktree else "",
            "log_path": str(log_path),
            "local_log_path": str(local_log_path) if keep_worktree else "",
            "answer_path": str(answer_path),
            "prompt_path": str(prompt_path),
            "events_path": str(events_path),
            "stderr_path": str(stderr_path),
            "seed_path_count": len(seed_paths),
        }
    except subprocess.TimeoutExpired:
        return base_status(sample, index, total, "timeout", started_at, started, message=f">{timeout_seconds}s")
    except Exception as exc:  # noqa: BLE001
        return base_status(sample, index, total, type(exc).__name__, started_at, started, message=str(exc))
    finally:
        if not keep_worktree:
            shutil.rmtree(worktree, ignore_errors=True)


def base_status(
    sample: dict[str, Any],
    index: int,
    total: int,
    status: str,
    started_at: str,
    started: float,
    *,
    message: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": str(sample.get("id") or ""),
        "task_type": sample.get("task_type"),
        "repo": sample.get("repo"),
        "base_commit": sample.get("base_commit"),
        "index": index,
        "total": total,
        "status": status,
        "message": message,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def status_summary(status: dict[str, Any], run_started: float) -> dict[str, Any]:
    return {
        "event": "sample_done",
        "sample_id": status["sample_id"],
        "index": status["index"],
        "total": status["total"],
        "status": status["status"],
        "runtime_seconds": status["runtime_seconds"],
        "log_entries": status.get("log_entries", 0),
        "elapsed_seconds": round(time.monotonic() - run_started, 3),
    }


def materialize_worktree(chunks_path: Path, worktree: Path) -> int:
    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.mkdir(parents=True, exist_ok=True)
    file_count = 0
    seen: set[str] = set()
    for row in read_jsonl(chunks_path):
        if row.get("kind") != "file":
            continue
        rel = str(row.get("path") or "")
        if not is_safe_relative_path(rel) or rel in seen:
            continue
        seen.add(rel)
        target = worktree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(row.get("text") or ""), encoding="utf-8", errors="replace")
        file_count += 1
    return file_count


def is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    if not value or path.is_absolute():
        return False
    return ".." not in path.parts


def extract_event_read_steps(
    *,
    events_path: Path,
    sample_id: str,
    worktree: Path,
    run_name: str,
    model_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not events_path.exists():
        return rows
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        if item.get("status") != "completed" or item.get("exit_code") not in (0, "0"):
            continue
        output = str(item.get("aggregated_output") or "")
        if not output.strip():
            continue
        command = str(item.get("command") or "")
        rows.extend(command_to_steps(command, sample_id=sample_id, worktree=worktree, run_name=run_name, model_label=model_label))
    for step, row in enumerate(rows, start=1):
        row["step"] = step
    return rows


def command_to_steps(command: str, *, sample_id: str, worktree: Path, run_name: str, model_label: str) -> list[dict[str, Any]]:
    script = unwrap_shell_command(command)
    specs = read_specs_from_script(script, worktree)
    rows: list[dict[str, Any]] = []
    for path, start_line, end_line, tool in specs:
        if not should_log_target_path(path):
            continue
        rows.append(
            {
                "content_hash": file_content_hash(worktree / path, start_line, end_line),
                "end_line": end_line,
                "is_final_context": False,
                "is_utilized_context": False,
                "kind": "block",
                "logged_at": utc_now(),
                "model": model_label,
                "path": path,
                "run_id": run_name,
                "sample_id": sample_id,
                "start_line": start_line,
                "step": 0,
                "symbol": "",
                "tool": tool,
            }
        )
    return rows


def unwrap_shell_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    shell = Path(tokens[0]).name if tokens else ""
    if len(tokens) >= 3 and shell in {"bash", "sh", "zsh"} and tokens[1] in {"-c", "-lc"}:
        return tokens[2]
    return command


def read_specs_from_script(script: str, worktree: Path) -> list[tuple[str, int, int, str]]:
    specs: list[tuple[str, int, int, str]] = []
    segments = split_shell_segments(script)
    if len(segments) > 1:
        for segment in segments:
            specs.extend(read_specs_from_script(segment, worktree))
        return specs
    nl_match = re.search(r"\bnl\s+-ba\s+(.+?)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?", script)
    if nl_match:
        path = normalize_logged_path(nl_match.group(1).strip(), worktree)
        if path:
            specs.append((path, int(nl_match.group(2)), int(nl_match.group(3)), "event:nl-sed"))
        return specs
    try:
        tokens = shlex.split(script)
    except ValueError:
        return specs
    if not tokens:
        return specs
    command = Path(tokens[0]).name
    if command == "sed":
        specs.extend(parse_sed_tokens(tokens, worktree))
    elif command == "cat":
        for raw in non_flag_args(tokens[1:]):
            path = normalize_logged_path(raw, worktree)
            if path:
                end = file_line_count(worktree / path)
                specs.append((path, 1, end, "event:cat"))
    elif command == "head":
        line_count, paths = parse_head_tail_tokens(tokens[1:], default_lines=10)
        for raw in paths:
            path = normalize_logged_path(raw, worktree)
            if path:
                end = min(line_count, file_line_count(worktree / path))
                specs.append((path, 1, max(1, end), "event:head"))
    elif command == "tail":
        line_count, paths = parse_head_tail_tokens(tokens[1:], default_lines=10)
        for raw in paths:
            path = normalize_logged_path(raw, worktree)
            if path:
                total = file_line_count(worktree / path)
                start = max(1, total - line_count + 1)
                specs.append((path, start, total, "event:tail"))
    return specs


def split_shell_segments(script: str) -> list[str]:
    segments = [part.strip() for part in re.split(r"\s*&&\s*", script) if part.strip()]
    return segments or [script]


def parse_sed_tokens(tokens: list[str], worktree: Path) -> list[tuple[str, int, int, str]]:
    specs: list[tuple[str, int, int, str]] = []
    if "-n" not in tokens:
        return specs
    index = tokens.index("-n")
    if index + 1 >= len(tokens):
        return specs
    match = re.fullmatch(r"(\d+)(?:,(\d+))?(?:p|\{=;p\})", tokens[index + 1])
    if not match:
        return specs
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    for raw in non_flag_args(tokens[index + 2 :]):
        path = normalize_logged_path(raw, worktree)
        if path:
            specs.append((path, start, end, "event:sed"))
    return specs


def parse_head_tail_tokens(tokens: list[str], *, default_lines: int) -> tuple[int, list[str]]:
    line_count = default_lines
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-n" and index + 1 < len(tokens):
            line_count = parse_positive_int(tokens[index + 1], default_lines)
            index += 2
            continue
        if token.startswith("-n") and len(token) > 2:
            line_count = parse_positive_int(token[2:], default_lines)
            index += 1
            continue
        if not token.startswith("-"):
            paths.append(token)
        index += 1
    return line_count, paths


def non_flag_args(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token != "--" and not token.startswith("-")]


def normalize_logged_path(raw: str, worktree: Path) -> str:
    cleaned = raw.strip().strip("'\"")
    if not cleaned:
        return ""
    path = Path(cleaned)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(worktree.resolve())
        except ValueError:
            return ""
    value = str(path)
    if not is_safe_relative_path(value):
        return ""
    if not (worktree / value).is_file():
        return ""
    return value


def should_log_target_path(path: str) -> bool:
    return bool(path and not path.startswith(".arb_") and "/.arb_" not in path and not path.startswith(".git/"))


def file_line_count(path: Path) -> int:
    try:
        return max(1, len(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    except OSError:
        return 1


def parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def file_content_hash(path: Path, start_line: int, end_line: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    data = "".join(lines[start_line - 1 : end_line]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def merge_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("path") or "")
        if not should_log_target_path(path):
            continue
        key = (path, row.get("start_line"), row.get("end_line"))
        if key not in merged:
            merged[key] = dict(row)
        else:
            merged[key]["is_final_context"] = bool(merged[key].get("is_final_context") or row.get("is_final_context"))
            merged[key]["is_utilized_context"] = bool(merged[key].get("is_utilized_context") or row.get("is_utilized_context"))
    output = list(merged.values())
    for step, row in enumerate(output, start=1):
        row["step"] = step
    return output


def build_prompt(
    *,
    sample: dict[str, Any],
    log_path: Path,
    answer_path: Path,
    run_name: str,
    model_label: str,
    file_count: int,
    seed_paths: list[str] | None = None,
    seed_label: str = "offline retriever",
    seed_note: str = "",
) -> str:
    query = query_text_for_eval(sample)
    seed_paths = seed_paths or []
    seed_section: list[str] = []
    if seed_paths:
        seed_section = [
            "## Retriever Seed",
            "",
            f"An offline retriever ({seed_label}) produced the ranked file-path candidates below.",
            seed_note,
            "Use them as starting hints only. The seed listing itself is not counted as file context; inspect files with shell commands before relying on them.",
            "",
        ]
        seed_section.extend(f"{index}. `{path}`" for index, path in enumerate(seed_paths, start=1))
        seed_section.append("")
    return "\n".join(
        [
            f"# ARB Codex CLI Trajectory Run: {sample.get('id')}",
            "",
            "You are inside a materialized repository workspace built from ARB corpus file chunks.",
            "Use only the files in the current working directory as the target repository.",
            "Do not inspect benchmark gold labels, evaluation details, release answer files, or any external checkout.",
            "Do not modify target repository files. Your job is to identify the context needed for the query.",
            "Use a strict exploration budget: at most 12 shell commands total and at most 6 file-content reads.",
            "Avoid broad commands that dump large globbed file contents. Use search commands to locate candidates, then read only the few files/ranges that matter.",
            "Each file-content read should be at most 120 lines. Prefer narrower `sed -n` ranges over dumping hundreds of lines.",
            "Once you have 2-5 plausible context files, stop and give the final answer; do not exhaustively prove the whole repository.",
            "",
            "## Metadata",
            "",
            f"- sample_id: `{sample.get('id')}`",
            f"- task_type: `{sample.get('task_type')}`",
            f"- repo: `{sample.get('repo')}`",
            f"- base_commit: `{sample.get('base_commit')}`",
            f"- query_provenance: `{sample.get('query_provenance')}`",
            f"- materialized_file_count: `{file_count}`",
            f"- run_name: `{run_name}`",
            f"- model_label: `{model_label}`",
            f"- trajectory_log: `{log_path}`",
            f"- answer_path: `{answer_path}`",
            "",
            "## Read-Trace Contract",
            "",
            "The runner records file-reading shell events automatically.",
            "When inspecting source, prefer commands that expose explicit file paths and line ranges, such as `sed -n '40,90p' path/to/file`.",
            "Use `grep`/`find` for search and `sed`/`cat`/`head`/`tail` for reading the files that matter.",
            "Do not use `sed` over wildcard globs to dump many files; use `grep` for broad search instead.",
            "Do not inspect hidden runner files, prompt files, logs, or generated answer files.",
            "",
            *seed_section,
            "## Final Answer",
            "",
            "Write a concise answer naming the final context files and why they are relevant.",
            "Do not implement a patch.",
            "",
            "## Query",
            "",
            "```json",
            query,
            "```",
            "",
        ]
    )


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
