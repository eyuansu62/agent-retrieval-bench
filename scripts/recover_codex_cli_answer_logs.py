#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKTREE_PREFIX_RE = re.compile(r"/tmp/arb_codex_cli_full/([A-Za-z0-9_-]+)/")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
PATHLIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"(?:[A-Za-z0-9_@.+-]+/)+[A-Za-z0-9_@.+-]+"
    r"(?:\.[A-Za-z0-9_+.-]+)?"
)
TRAILING_PUNCT = ".,;:)]}>\"'"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover file-only trajectory logs from Codex CLI final answers for empty event logs."
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="Codex CLI run directory.")
    parser.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3/samples.jsonl"))
    parser.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    parser.add_argument("--output-log-dir-name", default="logs_answer_recovered")
    parser.add_argument("--summary", type=Path, default=Path("data/eval/v1_4/codex_cli_answer_recovery_summary.json"))
    parser.add_argument("--model-label", default="", help="Override model label for recovered answer rows.")
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument(
        "--recover-all",
        action="store_true",
        help="Append answer-derived final file rows to every sample, not just empty logs.",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples)
    corpus_manifest = load_corpus_manifest(args.corpus_manifest)
    corpus_cache: dict[tuple[str, str], CorpusFiles] = {}

    run_summaries = []
    for run_dir in args.run_dir:
        run_summaries.append(
            recover_run(
                run_dir=run_dir,
                output_log_dir_name=args.output_log_dir_name,
                samples=samples,
                corpus_manifest=corpus_manifest,
                corpus_cache=corpus_cache,
                model_label=args.model_label,
                max_paths=args.max_paths,
                recover_all=args.recover_all,
            )
        )

    summary = {
        "mode": "codex_cli_answer_log_recovery",
        "generated_at": utc_now(),
        "samples_path": str(args.samples),
        "corpus_manifest": str(args.corpus_manifest),
        "output_log_dir_name": args.output_log_dir_name,
        "recover_all": args.recover_all,
        "runs": run_summaries,
        "totals": sum_run_summaries(run_summaries),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


class CorpusFiles:
    def __init__(self, chunks_path: Path):
        self.chunks_path = chunks_path
        self.text_by_path: dict[str, str] = {}
        for row in read_jsonl(chunks_path):
            if row.get("kind") != "file":
                continue
            path = str(row.get("path") or "")
            if path and is_safe_relative_path(path):
                self.text_by_path[path] = str(row.get("text") or "")
        self.paths = set(self.text_by_path)
        self._suffix_index: dict[str, str | None] = {}
        for path in self.paths:
            parts = Path(path).parts
            for index in range(1, len(parts)):
                suffix = "/".join(parts[index:])
                existing = self._suffix_index.get(suffix)
                if existing is None and suffix in self._suffix_index:
                    continue
                self._suffix_index[suffix] = path if existing is None else None

    def resolve(self, raw: str, sample_id: str) -> str:
        value = normalize_candidate(raw, sample_id)
        if not value:
            return ""
        if value in self.paths:
            return value
        suffix_match = self._suffix_index.get(value)
        return suffix_match or ""

    def content_hash(self, path: str) -> str:
        return hashlib.sha256(self.text_by_path.get(path, "").encode("utf-8")).hexdigest()


def recover_run(
    *,
    run_dir: Path,
    output_log_dir_name: str,
    samples: dict[str, dict[str, Any]],
    corpus_manifest: dict[tuple[str, str], Path],
    corpus_cache: dict[tuple[str, str], CorpusFiles],
    model_label: str,
    max_paths: int,
    recover_all: bool,
) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json", {})
    run_model = model_label or str(manifest.get("model") or "")
    output_dir = run_dir / output_log_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    statuses = load_statuses(run_dir)
    counts: Counter[str] = Counter()
    recovered_rows = 0
    recovered_samples = 0
    still_empty = 0
    copied_nonempty = 0
    examples: list[dict[str, Any]] = []

    for status in statuses:
        sample_id = str(status.get("sample_id") or "")
        if not sample_id:
            continue
        counts[str(status.get("status") or "unknown")] += 1
        source_log = run_dir / "logs" / f"{sample_id}.jsonl"
        output_log = output_dir / f"{sample_id}.jsonl"
        source_rows = read_jsonl(source_log)
        should_recover = recover_all or not source_rows or status.get("status") == "empty_log"

        rows = list(source_rows)
        recovered_paths: list[str] = []
        if should_recover:
            sample = samples.get(sample_id) or status
            corpus_files = corpus_for_sample(sample, corpus_manifest, corpus_cache)
            if corpus_files is not None:
                answer_path = Path(str(status.get("answer_path") or run_dir / "answers" / f"{sample_id}.md"))
                answer_text = answer_path.read_text(encoding="utf-8", errors="replace") if answer_path.exists() else ""
                recovered_paths = recover_paths_from_answer(answer_text, corpus_files, sample_id, max_paths=max_paths)
                rows = merge_rows(
                    [
                        *rows,
                        *answer_path_rows(
                            paths=recovered_paths,
                            corpus_files=corpus_files,
                            sample_id=sample_id,
                            run_id=f"{run_dir.name}_answer_recovered",
                            model_label=run_model or str(status.get("model") or "codex-cli"),
                        ),
                    ]
                )
        if rows:
            write_jsonl(output_log, rows)
        else:
            output_log.write_text("", encoding="utf-8")

        if source_rows:
            copied_nonempty += 1
        if recovered_paths:
            recovered_samples += 1
            recovered_rows += len(recovered_paths)
            if len(examples) < 8:
                examples.append({"sample_id": sample_id, "paths": recovered_paths})
        if not rows:
            still_empty += 1

    return {
        "run_dir": str(run_dir),
        "output_logs": str(output_dir),
        "status_counts": dict(sorted(counts.items())),
        "input_samples": len(statuses),
        "copied_nonempty_logs": copied_nonempty,
        "recovered_samples": recovered_samples,
        "recovered_rows": recovered_rows,
        "still_empty_logs": still_empty,
        "examples": examples,
    }


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or ""): row for row in read_jsonl(path) if row.get("id")}


def load_corpus_manifest(path: Path) -> dict[tuple[str, str], Path]:
    manifest: dict[tuple[str, str], Path] = {}
    root = infer_repo_root(path)
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        repo = str(row.get("repo") or "")
        commit = str(row.get("base_commit") or "")
        chunks = Path(str(row.get("chunks_path") or ""))
        if repo and commit and chunks:
            manifest[(repo, commit)] = chunks if chunks.is_absolute() else root / chunks
    return manifest


def infer_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    if (
        resolved.parent.name.startswith("v")
        and resolved.parent.parent.name == "corpus"
        and resolved.parent.parent.parent.name == "data"
    ):
        return resolved.parent.parent.parent.parent
    return Path.cwd()


def load_statuses(run_dir: Path) -> list[dict[str, Any]]:
    status_jsonl = run_dir / "status.jsonl"
    if status_jsonl.exists():
        return read_jsonl(status_jsonl)
    statuses = []
    for path in sorted((run_dir / "status").glob("*.json")):
        status = read_json(path, {})
        if status:
            statuses.append(status)
    return statuses


def corpus_for_sample(
    sample: dict[str, Any],
    manifest: dict[tuple[str, str], Path],
    cache: dict[tuple[str, str], CorpusFiles],
) -> CorpusFiles | None:
    key = (str(sample.get("repo") or ""), str(sample.get("base_commit") or ""))
    chunks = manifest.get(key)
    if not chunks or not chunks.exists():
        return None
    if key not in cache:
        cache[key] = CorpusFiles(chunks)
    return cache[key]


def recover_paths_from_answer(answer_text: str, corpus_files: CorpusFiles, sample_id: str, *, max_paths: int) -> list[str]:
    candidates: list[str] = []
    for label, target in MARKDOWN_LINK_RE.findall(answer_text):
        candidates.extend([label, target])
    candidates.extend(BACKTICK_RE.findall(answer_text))
    candidates.extend(PATHLIKE_RE.findall(answer_text))
    for path in sorted(corpus_files.paths, key=len, reverse=True):
        if "/" in path and path in answer_text:
            candidates.append(path)

    output: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        path = corpus_files.resolve(raw, sample_id)
        if path and path not in seen:
            output.append(path)
            seen.add(path)
            if len(output) >= max_paths:
                break
    return output


def normalize_candidate(raw: str, sample_id: str) -> str:
    value = raw.strip().replace("\\", "/")
    value = re.sub(r"^file://", "", value)
    value = value.strip().strip(TRAILING_PUNCT)
    if not value:
        return ""
    match = WORKTREE_PREFIX_RE.search(value)
    if match:
        value = value[match.end() :]
    marker = f"/{sample_id}/"
    if marker in value:
        value = value.split(marker, 1)[1]
    value = value.removeprefix("./")
    if value.startswith("/"):
        return ""
    value = value.strip().strip(TRAILING_PUNCT)
    return value if is_safe_relative_path(value) else ""


def answer_path_rows(
    *,
    paths: list[str],
    corpus_files: CorpusFiles,
    sample_id: str,
    run_id: str,
    model_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for step, path in enumerate(paths, start=1):
        rows.append(
            {
                "content_hash": corpus_files.content_hash(path),
                "end_line": None,
                "is_final_context": True,
                "is_utilized_context": False,
                "kind": "file",
                "logged_at": utc_now(),
                "model": model_label,
                "path": path,
                "recovery_source": "codex_final_answer",
                "run_id": run_id,
                "sample_id": sample_id,
                "start_line": None,
                "step": step,
                "symbol": "",
                "tool": "answer:path",
            }
        )
    return rows


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, Any, Any, str], dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("path") or "")
        if not path:
            continue
        key = (path, row.get("start_line"), row.get("end_line"), str(row.get("tool") or ""))
        if key not in merged:
            merged[key] = dict(row)
        else:
            merged[key]["is_final_context"] = bool(merged[key].get("is_final_context") or row.get("is_final_context"))
            merged[key]["is_utilized_context"] = bool(
                merged[key].get("is_utilized_context") or row.get("is_utilized_context")
            )
    output = list(merged.values())
    for step, row in enumerate(output, start=1):
        row["step"] = step
    return output


def sum_run_summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for run in runs:
        totals["input_samples"] += int(run.get("input_samples") or 0)
        totals["copied_nonempty_logs"] += int(run.get("copied_nonempty_logs") or 0)
        totals["recovered_samples"] += int(run.get("recovered_samples") or 0)
        totals["recovered_rows"] += int(run.get("recovered_rows") or 0)
        totals["still_empty_logs"] += int(run.get("still_empty_logs") or 0)
    return dict(totals)


def is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value and not path.is_absolute() and ".." not in path.parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
