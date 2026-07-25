from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import load_corpus_manifest, query_text_for_eval, rank_chunks, unique_ranked_paths
from .io import ensure_parent, read_jsonl, utc_now


@dataclass
class FileView:
    path: str
    start_line: int | None
    end_line: int | None
    kind: str
    text: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arb run-openai-context-agent")
    parser.add_argument("--base", type=Path, default=Path("data/trajectory_runs/v1_4/v1_3_all_gpt54mini"))
    parser.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3/samples.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--run-name", default="openai_gpt54mini_v2_strict_context")
    parser.add_argument("--all-samples", action="store_true", help="Run every row in --samples instead of base/sample_ids.txt.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-actions", type=int, default=9)
    parser.add_argument("--min-reads", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_openai_context_agent(
        base=args.base,
        samples_path=args.samples,
        corpus_dir=args.corpus,
        model=args.model,
        run_name=args.run_name,
        all_samples=args.all_samples,
        limit=args.limit,
        sample_ids=args.sample_id,
        force=args.force,
        max_actions=args.max_actions,
        min_reads=args.min_reads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not result["failures"] else 1


def run_openai_context_agent(
    *,
    base: Path,
    samples_path: Path,
    corpus_dir: Path,
    model: str,
    run_name: str,
    all_samples: bool,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
    force: bool = False,
    max_actions: int = 9,
    min_reads: int = 3,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for run-openai-context-agent.") from exc

    samples = {str(row.get("id") or ""): row for row in read_jsonl(samples_path)}
    selected_ids = sample_id_list(base=base, samples=samples, all_samples=all_samples, sample_ids=sample_ids or [], limit=limit)
    corpus_manifest = load_corpus_manifest(corpus_dir)

    logs_dir = base / f"logs_{run_name}"
    answers_dir = base / f"answers_{run_name}"
    traces_dir = base / f"traces_{run_name}"
    for directory in (logs_dir, answers_dir, traces_dir):
        directory.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    completed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, sample_id in enumerate(selected_ids, start=1):
        sample = samples.get(sample_id)
        if not sample:
            failures.append({"sample_id": sample_id, "error": "missing_sample"})
            continue
        log_path = logs_dir / f"{sample_id}.jsonl"
        answer_path = answers_dir / f"{sample_id}.md"
        trace_path = traces_dir / f"{sample_id}.json"
        if not force and log_path.exists() and log_path.stat().st_size > 0 and answer_path.exists() and trace_path.exists():
            skipped += 1
            print(json.dumps({"event": "skip_existing", "sample_id": sample_id, "index": index}), flush=True)
            continue
        try:
            result = run_sample(
                client=client,
                model=model,
                sample=sample,
                corpus_manifest=corpus_manifest,
                max_actions=max_actions,
                min_reads=min_reads,
                run_name=run_name,
                sample_set=base.name,
            )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": type(exc).__name__, "message": str(exc)})
            print(json.dumps({"event": "failure", "sample_id": sample_id, "error": type(exc).__name__}), flush=True)
            continue
        write_outputs(result, log_path=log_path, answer_path=answer_path, trace_path=trace_path, model=model)
        completed += 1
        print(
            json.dumps(
                {
                    "event": "completed",
                    "sample_id": sample_id,
                    "index": index,
                    "read_steps": len(result["read_steps"]),
                    "final_files": len(result["final_files"]),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "mode": "openai_context_agent",
        "model": model,
        "run_name": run_name,
        "all_samples": all_samples,
        "sample_ids": selected_ids,
        "completed": completed,
        "skipped_existing": skipped,
        "failures": failures,
        "logs_dir": str(logs_dir),
        "answers_dir": str(answers_dir),
        "traces_dir": str(traces_dir),
        "generated_at": utc_now(),
        "runtime_seconds": time.monotonic() - started,
    }
    ensure_parent(base / f"{run_name}_manifest.json")
    (base / f"{run_name}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def sample_id_list(
    *, base: Path, samples: dict[str, dict[str, Any]], all_samples: bool, sample_ids: list[str], limit: int | None
) -> list[str]:
    if all_samples:
        selected = [sample_id for sample_id in samples if sample_id]
    else:
        selected = [line.strip() for line in (base / "sample_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_ids:
        wanted = set(sample_ids)
        selected = [sample_id for sample_id in selected if sample_id in wanted]
    return selected[:limit] if limit else selected


def run_sample(
    *,
    client: Any,
    model: str,
    sample: dict[str, Any],
    corpus_manifest: dict[tuple[str, str], Path],
    max_actions: int,
    min_reads: int,
    run_name: str,
    sample_set: str,
) -> dict[str, Any]:
    sample_id = str(sample["id"])
    repo = str(sample.get("repo") or "")
    base_commit = str(sample.get("base_commit") or "")
    chunks_path = corpus_manifest.get((repo, base_commit))
    if not chunks_path:
        raise RuntimeError(f"missing corpus for {repo}@{base_commit}")
    chunks = list(read_jsonl(chunks_path))
    file_views = build_file_views(chunks)
    query_json = query_text_for_eval(sample)
    query_payload = json.loads(query_json)
    given_context = format_given_context(query_payload, file_views)

    observations: list[str] = []
    read_steps: list[dict[str, Any]] = []
    read_paths: list[str] = []
    read_set: set[str] = set()
    actions: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    last_error = ""
    last_hits: list[str] = []
    must_read_from_hits = False

    for turn in range(1, max_actions + 1):
        response_text = call_model(
            client=client,
            model=model,
            prompt=build_agent_prompt(
                sample_id=sample_id,
                repo=repo,
                base_commit=base_commit,
                query_json=query_json,
                given_context=given_context,
                observations=observations,
                read_paths=read_paths,
                min_reads=min_reads,
                last_error=last_error,
                final_only=False,
                task_type=str(sample.get("task_type") or ""),
            ),
        )
        action = parse_json_action(response_text)
        actions.append({"turn": turn, "raw": response_text, "parsed": action})
        last_error = ""
        kind = str(action.get("action") or "").lower()
        if kind == "search":
            if must_read_from_hits and len(read_steps) < min_reads:
                last_error = (
                    "search blocked until you read one exact path from the latest search results; "
                    f"successful reads so far: {len(read_steps)}"
                )
                observations.append(format_search_blocked(last_hits, file_views, last_error))
                continue
            query = str(action.get("query") or "")
            top_k = clamp_int(action.get("top_k"), default=8, minimum=1, maximum=12)
            if not query:
                last_error = "search action requires a non-empty query"
                continue
            hits = search_files(query, chunks, top_k=top_k)
            last_hits = [str(hit["path"]) for hit in hits if hit.get("path")]
            must_read_from_hits = bool([hit for hit in last_hits if hit not in read_set] and len(read_steps) < min_reads)
            observations.append(format_search_observation(query, hits))
            continue
        if kind == "read":
            last_error = apply_read_action(
                action=action,
                file_views=file_views,
                read_paths=read_paths,
                read_set=read_set,
                read_steps=read_steps,
                observations=observations,
                last_hits=last_hits,
                turn=turn,
            )
            must_read_from_hits = False if not last_error else must_read_from_hits
            continue
        if kind == "final":
            if len(read_steps) < min_reads:
                last_error = f"final is not allowed until at least {min_reads} files have been read; successful reads so far: {len(read_steps)}"
                observations.append(last_error)
                continue
            requested_final_files = dedupe_paths(action.get("context_files") or action.get("files") or [])
            unread_final_files = [path for path in requested_final_files if path not in read_set]
            if unread_final_files:
                last_error = (
                    "final context_files may include only files that were successfully read. "
                    "If more context is needed, the next action must be read with one unread exact path; "
                    "otherwise move unread recommendations to suggested_unread_files and choose context_files from Files already read."
                )
                last_hits = unread_final_files
                must_read_from_hits = len(read_steps) < min_reads
                observations.append(format_final_context_error(unread_final_files, read_paths))
                continue
            final = action
            break
        last_error = "action must be one of: search, read, final"

    if final is None and len(read_steps) < min_reads:
        recover_minimum_reads(
            client=client,
            model=model,
            sample=sample,
            sample_id=sample_id,
            repo=repo,
            base_commit=base_commit,
            query_json=query_json,
            given_context=given_context,
            observations=observations,
            read_paths=read_paths,
            read_set=read_set,
            read_steps=read_steps,
            actions=actions,
            file_views=file_views,
            chunks=chunks,
            min_reads=min_reads,
            max_actions=max_actions,
            last_hits=last_hits,
        )
    if final is None and len(read_steps) < min_reads:
        raise RuntimeError(f"failed to collect minimum reads: {len(read_steps)}/{min_reads}")
    if final is None:
        response_text = call_model(
            client=client,
            model=model,
            prompt=build_agent_prompt(
                sample_id=sample_id,
                repo=repo,
                base_commit=base_commit,
                query_json=query_json,
                given_context=given_context,
                observations=observations,
                read_paths=read_paths,
                min_reads=min_reads,
                last_error="Action budget reached. Return final now.",
                final_only=True,
                task_type=str(sample.get("task_type") or ""),
            ),
        )
        final = parse_json_action(response_text)
        actions.append({"turn": max_actions + 7, "raw": response_text, "parsed": final})

    final_split = split_strict_final_files(final, read_paths)
    if not final_split["final_files"]:
        raise RuntimeError("final contains no successfully read context files")
    return {
        "sample_id": sample_id,
        "repo": repo,
        "base_commit": base_commit,
        "query": json.loads(query_json),
        "given_context": given_context,
        "read_steps": read_steps,
        "read_paths": read_paths,
        "raw_final_files": final_split["raw_final_files"],
        "final_files": final_split["final_files"],
        "suggested_unread_files": final_split["suggested_unread_files"],
        "rationale": str(final.get("rationale") or ""),
        "actions": actions,
        "run_name": run_name,
        "sample_set": sample_set,
    }


def recover_minimum_reads(
    *,
    client: Any,
    model: str,
    sample: dict[str, Any],
    sample_id: str,
    repo: str,
    base_commit: str,
    query_json: str,
    given_context: str,
    observations: list[str],
    read_paths: list[str],
    read_set: set[str],
    read_steps: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    file_views: dict[str, FileView],
    chunks: list[dict[str, Any]],
    min_reads: int,
    max_actions: int,
    last_hits: list[str],
) -> None:
    for recovery_turn in range(max_actions + 1, max_actions + 7):
        response_text = call_model(
            client=client,
            model=model,
            prompt=build_agent_prompt(
                sample_id=sample_id,
                repo=repo,
                base_commit=base_commit,
                query_json=query_json,
                given_context=given_context,
                observations=observations,
                read_paths=read_paths,
                min_reads=min_reads,
                last_error=(
                    f"Final is still blocked because only {len(read_steps)}/{min_reads} files have been read. "
                    "Return a read action for one exact unread readable path now."
                ),
                final_only=False,
                task_type=str(sample.get("task_type") or ""),
            ),
        )
        action = parse_json_action(response_text)
        actions.append({"turn": recovery_turn, "raw": response_text, "parsed": action, "recovery": True})
        kind = str(action.get("action") or "").lower()
        if kind == "search":
            query = str(action.get("query") or "")
            if not query:
                continue
            hits = search_files(query, chunks, top_k=clamp_int(action.get("top_k"), default=8, minimum=1, maximum=12))
            last_hits = [str(hit["path"]) for hit in hits if hit.get("path")]
            observations.append(format_search_observation(query, hits))
            continue
        if kind == "read":
            error = apply_read_action(
                action=action,
                file_views=file_views,
                read_paths=read_paths,
                read_set=read_set,
                read_steps=read_steps,
                observations=observations,
                last_hits=last_hits,
                turn=recovery_turn,
            )
            if not error and len(read_steps) >= min_reads:
                break
            continue
        if kind == "final":
            observations.append(f"final is still blocked until at least {min_reads} files have been read; successful reads so far: {len(read_steps)}")


def apply_read_action(
    *,
    action: dict[str, Any],
    file_views: dict[str, FileView],
    read_paths: list[str],
    read_set: set[str],
    read_steps: list[dict[str, Any]],
    observations: list[str],
    last_hits: list[str],
    turn: int,
) -> str:
    path = str(action.get("path") or "")
    view = file_views.get(path)
    if not path or not view:
        message = f"read path not found in corpus: {path}"
        observations.append(format_read_error(path, last_hits, file_views))
        return message
    if path in read_set:
        message = f"already read `{path}`; choose a different exact path from search results"
        observations.append(format_already_read(path, last_hits, read_set, file_views))
        return message
    read_paths.append(path)
    read_set.add(path)
    read_steps.append({"path": path, "start_line": view.start_line, "end_line": view.end_line, "kind": view.kind, "turn": turn})
    observations.append(format_read_observation(view))
    return ""


def split_strict_final_files(final: dict[str, Any], read_paths: list[str]) -> dict[str, list[str]]:
    raw_final_files = dedupe_paths(final.get("context_files") or final.get("files") or [])
    read_set = set(read_paths)
    final_files = [path for path in raw_final_files if path in read_set]
    suggested_unread_files = dedupe_paths(final.get("suggested_unread_files") or [])
    suggested_unread_files.extend(path for path in raw_final_files if path not in read_set and path not in suggested_unread_files)
    return {
        "raw_final_files": raw_final_files,
        "final_files": final_files,
        "suggested_unread_files": suggested_unread_files,
    }


def call_model(*, client: Any, model: str, prompt: str) -> str:
    response = client.responses.create(model=model, input=prompt, temperature=0, max_output_tokens=700)
    return response.output_text.strip()


def build_agent_prompt(
    *,
    sample_id: str,
    repo: str,
    base_commit: str,
    query_json: str,
    given_context: str,
    observations: list[str],
    read_paths: list[str],
    min_reads: int,
    last_error: str,
    final_only: bool,
    task_type: str,
) -> str:
    observations_text = "\n\n".join(observations[-12:]) if observations else "No observations yet."
    read_text = "\n".join(f"- {path}" for path in dedupe_paths(read_paths)) or "- none"
    final_schema = (
        '{"action":"final","context_files":["already/read/path.ext"],'
        '"suggested_unread_files":["optional/unread/path.ext"],"rationale":"short reason"}'
    )
    allowed_actions = final_schema if final_only else "\n".join(
        [
            '{"action":"search","query":"terms to search in the repository corpus","top_k":8}',
            '{"action":"read","path":"repo/relative/path.ext"}',
            final_schema,
        ]
    )
    error_text = f"\nPrevious error: {last_error}\n" if last_error else ""
    return f"""You are running an Agent Retrieval Bench {task_type or 'retrieval'} task.

Goal: {task_goal_text(task_type)}

Rules:
- Use only the query and tool observations below.
- Do not use benchmark gold labels, support annotation files, eval details, release answer files, or prior report metrics.
- Prefer concrete repo-relative file paths.
- Search before reading unless a path is already directly known from the query or observations.
- Read the files that look most likely to be useful context.
- You must successfully read at least {min_reads} files before returning final.
- A path is readable only if it appears verbatim in a SEARCH result or has already been successfully read.
- If a query path is unavailable, pick exact readable paths from SEARCH results instead.
- After a SEARCH, the next action should usually be READ with one exact backticked path from that SEARCH list.
- Final context_files must be a non-empty subset of Files already read.
- If you want to mention useful files that were not read, put them only in suggested_unread_files, not context_files.
- Keep final context_files focused, usually 3 to 8 files.
- Return exactly one JSON object and no markdown.

Allowed JSON actions:
{allowed_actions}

Metadata:
- sample_id: {sample_id}
- repo: {repo}
- base_commit: {base_commit}
- task_type: {task_type or 'unknown'}

Query JSON:
{query_json}

Given reviewed context:
{given_context}

Files already read:
{read_text}

Observations:
{observations_text}
{error_text}
Return the next JSON action."""


def task_goal_text(task_type: str) -> str:
    if task_type == "code2test":
        return "identify files needed to understand the code change and find or update relevant tests."
    if task_type == "trace2code":
        return "identify files needed to diagnose the failure trace and locate likely source code."
    if task_type == "comment2context":
        return "identify additional files needed to understand and act on the review comment."
    return "identify repository files a coding agent should read to solve the retrieval task."


def search_files(query: str, chunks: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    ranked = rank_chunks(query, chunks)
    paths = unique_ranked_paths(ranked)[:top_k]
    first_chunk_by_path: dict[str, dict[str, Any]] = {}
    for chunk in ranked:
        path = str(chunk.get("path") or "")
        if path in paths and path not in first_chunk_by_path:
            first_chunk_by_path[path] = chunk
    hits = []
    for path in paths:
        chunk = first_chunk_by_path.get(path, {})
        hits.append(
            {
                "path": path,
                "kind": str(chunk.get("kind") or ""),
                "symbol": str(chunk.get("symbol") or ""),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "snippet": trim_text(str(chunk.get("text") or ""), 700),
            }
        )
    return hits


def format_search_observation(query: str, hits: list[dict[str, Any]]) -> str:
    lines = [f"SEARCH query={query!r}"]
    for index, hit in enumerate(hits, start=1):
        loc = line_range(hit.get("start_line"), hit.get("end_line"))
        symbol = f" symbol={hit['symbol']!r}" if hit.get("symbol") else ""
        lines.append(f"{index}. `{hit['path']}` {loc}{symbol}\n{hit['snippet']}")
    return "\n".join(lines)


def format_read_observation(view: FileView) -> str:
    return f"READ `{view.path}` {line_range(view.start_line, view.end_line)}\n{numbered_text(view.text, max_chars=12_000)}"


def format_given_context(query: dict[str, Any], file_views: dict[str, FileView]) -> str:
    given_file = str(query.get("given_file") or query.get("path") or "")
    line = query.get("line")
    diff_hunk = str(query.get("diff_hunk_context") or "").strip()
    status = "present in readable corpus" if given_file in file_views else "not present in readable corpus"
    lines = [
        f"- given_file: `{given_file}`" if given_file else "- given_file: none",
        f"- line: `{line}`" if line else "- line: unknown",
        f"- corpus_status: {status}" if given_file else "- corpus_status: no fixed reviewed file is provided for this task.",
    ]
    if given_file:
        lines.append("- note: this reviewed file/hunk is already provided as fixed context; search and read additional files.")
    if diff_hunk:
        lines.extend(["- diff_hunk_context:", "```", diff_hunk, "```"])
    view = file_views.get(given_file)
    if view and line:
        excerpt = excerpt_around_line(view, safe_int(line), radius=40, max_chars=8_000)
        if excerpt:
            lines.extend(["- given_file_excerpt:", "```", excerpt, "```"])
    return "\n".join(lines)


def excerpt_around_line(view: FileView, line: int, *, radius: int, max_chars: int) -> str:
    if line <= 0 or not view.text:
        return ""
    file_start = view.start_line or 1
    lines = view.text.splitlines()
    relative = max(0, line - file_start)
    start = max(0, relative - radius)
    end = min(len(lines), relative + radius + 1)
    return trim_text("\n".join(f"{file_start + index:5d}: {lines[index]}" for index in range(start, end)), max_chars)


def format_already_read(path: str, last_hits: list[str], read_set: set[str], file_views: dict[str, FileView]) -> str:
    unread_hits = [hit for hit in last_hits if hit in file_views and hit not in read_set]
    lines = [f"ALREADY_READ: `{path}` has already entered context and will not be logged again."]
    if unread_hits:
        lines.append("Read a different exact path from the latest search results:")
        lines.extend(f"- `{hit}`" for hit in unread_hits[:8])
    else:
        lines.append("Search with more specific terms or read a different exact path from observations.")
    return "\n".join(lines)


def format_read_error(path: str, last_hits: list[str], file_views: dict[str, FileView]) -> str:
    readable_hits = [hit for hit in last_hits if hit in file_views]
    lines = [f"READ ERROR: `{path}` was not found in the readable corpus."]
    if readable_hits:
        lines.append("Read one of these exact paths from the latest search results instead:")
        lines.extend(f"- `{hit}`" for hit in readable_hits[:8])
    return "\n".join(lines)


def format_search_blocked(last_hits: list[str], file_views: dict[str, FileView], message: str) -> str:
    readable_hits = [hit for hit in last_hits if hit in file_views]
    lines = [f"SEARCH BLOCKED: {message}"]
    if readable_hits:
        lines.append("Use READ with one of these exact paths:")
        lines.extend(f"- `{hit}`" for hit in readable_hits[:8])
    return "\n".join(lines)


def format_final_context_error(unread_files: list[str], read_paths: list[str]) -> str:
    lines = ["FINAL CONTEXT ERROR: context_files included unread paths."]
    if unread_files:
        lines.append("Unread paths must be removed from context_files:")
        lines.extend(f"- `{path}`" for path in unread_files[:8])
    if read_paths:
        lines.append("Choose final context_files only from files already read:")
        lines.extend(f"- `{path}`" for path in dedupe_paths(read_paths)[:8])
    return "\n".join(lines)


def build_file_views(chunks: list[dict[str, Any]]) -> dict[str, FileView]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        path = str(chunk.get("path") or "")
        if path:
            grouped.setdefault(path, []).append(chunk)
    views: dict[str, FileView] = {}
    for path, rows in grouped.items():
        file_rows = [row for row in rows if str(row.get("kind") or "") == "file"]
        selected = sorted(file_rows or rows, key=lambda row: (safe_int(row.get("start_line")), safe_int(row.get("end_line"))))
        text = "\n".join(str(row.get("text") or "") for row in selected)
        starts = [safe_int(row.get("start_line")) for row in selected if safe_int(row.get("start_line")) > 0]
        ends = [safe_int(row.get("end_line")) for row in selected if safe_int(row.get("end_line")) > 0]
        views[path] = FileView(
            path=path,
            start_line=min(starts) if starts else None,
            end_line=max(ends) if ends else None,
            kind=str(selected[0].get("kind") or "file") if selected else "file",
            text=text,
        )
    return views


def write_outputs(result: dict[str, Any], *, log_path: Path, answer_path: Path, trace_path: Path, model: str) -> None:
    final_set = set(result["final_files"])
    logged_at = utc_now()
    rows = []
    for index, step in enumerate(result["read_steps"], start=1):
        rows.append(
            {
                "content_hash": "",
                "end_line": step.get("end_line"),
                "is_final_context": step["path"] in final_set,
                "is_utilized_context": step["path"] in final_set,
                "kind": step.get("kind") or "file",
                "logged_at": logged_at,
                "model": model,
                "path": step["path"],
                "run_id": f"{result.get('sample_set') or 'agent-retrieval-bench'}-{result.get('run_name') or 'openai'}",
                "sample_id": result["sample_id"],
                "start_line": step.get("start_line"),
                "step": index,
                "symbol": "",
                "tool": "read",
            }
        )
    ensure_parent(log_path)
    log_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    answer_lines = [
        "# OpenAI Context Agent Answer",
        "",
        f"Sample: `{result['sample_id']}`",
        f"Repo: `{result['repo']}`",
        f"Base commit: `{result['base_commit']}`",
        "",
        "## Final Context Files",
        "",
        *[f"- `{path}`" for path in result["final_files"]],
    ]
    if result.get("suggested_unread_files"):
        answer_lines.extend(["", "## Suggested Unread Files", "", *[f"- `{path}`" for path in result["suggested_unread_files"]]])
    answer_lines.extend(["", "## Rationale", "", str(result["rationale"]).strip() or "(none)", ""])
    ensure_parent(answer_path)
    answer_path.write_text("\n".join(answer_lines), encoding="utf-8")
    ensure_parent(trace_path)
    trace_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_action(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            value = parsed
            break
        if value is None:
            return {"action": "invalid", "raw": text}
    return value if isinstance(value, dict) else {"action": "invalid", "raw": text}


def numbered_text(text: str, *, max_chars: int) -> str:
    clipped = trim_text(text, max_chars)
    return "\n".join(f"{index:5d}: {line}" for index, line in enumerate(clipped.splitlines(), start=1))


def trim_text(text: str, limit: int) -> str:
    text = text.replace("\x00", "")
    return text if len(text) <= limit else text[: limit - 80] + "\n... [truncated] ..."


def line_range(start: Any, end: Any) -> str:
    return f"lines {start}-{end}" if start and end else "lines unknown"


def clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def dedupe_paths(values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return output
    for value in values:
        path = str(value or "")
        if path and path not in seen:
            output.append(path)
            seen.add(path)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
