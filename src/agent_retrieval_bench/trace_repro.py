from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .code2test_pr import clean_pr_body, has_query_noise, latest_by_pr, low_value_change_majority, path_token_overlap, repos_from_raw
from .filters import is_source_file, is_test_file, split_changed_files
from .io import ensure_parent, read_jsonl, repo_slug, stable_id, truncate_text, utc_now, write_json

REPRO_AUDIT_VERDICTS = (
    "runnable_repro_source",
    "needs_runner_work",
    "no_failure_expected",
    "too_broad",
    "leaked",
    "duplicate",
    "ambiguous",
)
TRACE_REPRO_AUDIT_VERDICTS = (
    "valid",
    "noisy",
    "leaked",
    "ambiguous",
    "too_easy",
    "duplicate",
    "not_root_cause",
)

TEST_PATCH_SIGNAL_RE = re.compile(
    r"\b("
    r"assert|assertion|expect|toEqual|toBe|raises|throws|panic|fatal|error|exception|"
    r"regression|reproducer|pytest\.raises|require\.|assert\.|@Test|func Test|it\(|test_|describe\("
    r")\b",
    re.IGNORECASE,
)
BEHAVIOR_RE = re.compile(
    r"\b("
    r"bug|fix|regression|fail|failing|error|exception|panic|assert|behavior|behaviour|"
    r"runtime|compat|validation|race|cache|api|config|pipeline|model|tokenizer"
    r")\b",
    re.IGNORECASE,
)
FAILURE_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|AssertionError|panic:|FAILED\s+[\w./:-]+|"
    r"\b(?:Error|Exception):|^\s+at\s+[\w.$<>]+\(|^\s*File [\"']|"
    r"^--- FAIL:|^[\w./-]+\.(?:go|rs):\d+:\d+:|^\w[\w./-]+\.(?:ts|tsx|js|jsx):\d+:\d+|"
    r"^error(?:\[[A-Z]\d+\])?:",
    re.IGNORECASE | re.MULTILINE,
)


def trace_repro_source(
    raw_dir: Path,
    out_dir: Path,
    repos: Iterable[str] | None = None,
    max_changed_files: int = 30,
    max_source_files: int = 5,
    max_test_files: int = 5,
    min_score: int = 5,
    audit_limit: int = 120,
    limit_candidates: int | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    by_repo: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()

    repo_names = list(repos or repos_from_raw(raw_dir))
    for repo in repo_names:
        repo_raw = raw_dir / repo_slug(repo)
        pr_by_number = latest_by_pr(repo_raw / "pull_requests.jsonl")
        files_by_pr = latest_by_pr(repo_raw / "pull_files.jsonl")
        details_by_pr = latest_by_pr(repo_raw / "commit_details.jsonl")
        for pr_number, files_record in sorted(files_by_pr.items(), reverse=True):
            pr = (pr_by_number.get(pr_number) or {}).get("data") or {}
            candidate, reason = build_trace_repro_candidate(
                repo=repo,
                pr_number=pr_number,
                pr=pr,
                files_record=files_record,
                details_record=details_by_pr.get(pr_number),
                max_changed_files=max_changed_files,
                max_source_files=max_source_files,
                max_test_files=max_test_files,
                min_score=min_score,
            )
            if reason:
                dropped[reason] += 1
                continue
            assert candidate is not None
            candidates.append(candidate)
            by_repo[repo] += 1
            by_strategy[candidate["run"]["strategy"]] += 1
            if limit_candidates and len(candidates) >= limit_candidates:
                break
        if limit_candidates and len(candidates) >= limit_candidates:
            break

    candidates.sort(key=repro_sort_key)
    if limit_candidates:
        candidates = candidates[:limit_candidates]

    audit_rows = [audit_row(candidate) for candidate in candidates[: max(0, audit_limit)]]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "repro_candidates.jsonl", candidates)
    write_jsonl(out_dir / "audit_samples.jsonl", audit_rows)
    write_csv(
        out_dir / "audit_samples.csv",
        audit_rows,
        (
            "candidate_id",
            "repo",
            "pr_number",
            "score",
            "implementation_files",
            "test_files",
            "run_strategy",
            "commands",
            "rationale",
            "verdict",
            "keep",
            "notes",
        ),
    )
    summary = {
        "generated_at": utc_now(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "source_type": "local_test_reproduction",
        "description": "Apply PR test-file patches to base commit, run focused tests, then convert real failures into trace2code samples.",
        "repos_scanned": len(repo_names),
        "candidates": len(candidates),
        "audit_rows": len(audit_rows),
        "by_repo": dict(sorted(by_repo.items())),
        "by_strategy": dict(sorted(by_strategy.items())),
        "dropped": dict(sorted(dropped.items())),
        "quality_gate": {
            "audit_rows_ge_80": len(audit_rows) >= 80,
            "requires_execution_before_benchmark": True,
            "publishable_trace2code": False,
        },
        "outputs": {
            "candidates": str(out_dir / "repro_candidates.jsonl"),
            "audit_jsonl": str(out_dir / "audit_samples.jsonl"),
            "audit_csv": str(out_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_trace_repro(
    candidate_path: Path,
    raw_dir: Path,
    repos_dir: Path,
    out_dir: Path,
    candidate_ids: Iterable[str] | None = None,
    limit: int | None = 1,
    timeout_seconds: int = 900,
    repo_url_template: str = "https://github.com/{repo}.git",
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    candidates = select_candidates(candidate_path, candidate_ids, limit)
    runs: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for candidate in candidates:
        result = run_one_trace_repro(
            candidate=candidate,
            raw_dir=raw_dir,
            repos_dir=repos_dir,
            out_dir=out_dir,
            timeout_seconds=timeout_seconds,
            repo_url_template=repo_url_template,
            dry_run=dry_run,
        )
        runs.append(result)
        status_counts[result["status"]] += 1
        if result["status"] not in {"failed_expected", "dry_run"} and not continue_on_error:
            break

    write_jsonl(out_dir / "runs.jsonl", runs)
    summary = {
        "generated_at": utc_now(),
        "candidate_path": str(candidate_path),
        "raw_dir": str(raw_dir),
        "repos_dir": str(repos_dir),
        "out_dir": str(out_dir),
        "requested_ids": list(candidate_ids or []),
        "selected": len(candidates),
        "executed": len(runs),
        "dry_run": dry_run,
        "status_counts": dict(sorted(status_counts.items())),
        "outputs": {"runs": str(out_dir / "runs.jsonl")},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def mine_trace_repro_runs(
    candidates_path: Path,
    runs_path: Path,
    out_dir: Path,
    report_dir: Path,
    max_root_files: int = 3,
    audit_limit: int = 120,
) -> dict[str, Any]:
    candidates_by_id = {str(candidate.get("id")): candidate for candidate in read_jsonl(candidates_path)}
    samples: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    by_repo: Counter[str] = Counter()
    by_failure_type: Counter[str] = Counter()

    for run in read_jsonl(runs_path):
        sample, reason = trace_sample_from_repro_run(run, candidates_by_id.get(str(run.get("id"))), max_root_files=max_root_files)
        if reason:
            dropped[reason] += 1
            continue
        assert sample is not None
        samples.append(sample)
        by_repo[str(sample.get("repo"))] += 1
        for signal in ((sample.get("metadata") or {}).get("evidence") or {}).get("signals") or []:
            if signal in {"compile_error", "assertion_failure", "panic_failure"}:
                by_failure_type[signal] += 1

    samples.sort(key=lambda sample: (sample.get("repo", ""), -int((sample.get("metadata") or {}).get("pr") or 0), sample.get("id", "")))
    write_trace_outputs(out_dir, samples)
    audit_rows = [trace_repro_audit_row(sample) for sample in samples[: max(0, audit_limit)]]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "audit_samples.jsonl", audit_rows)
    write_csv(
        report_dir / "audit_samples.csv",
        audit_rows,
        ("sample_id", "task_type", "repo", "query_excerpt", "gold_files", "verdict", "reason", "keep", "notes"),
    )
    summary = {
        "generated_at": utc_now(),
        "candidates_path": str(candidates_path),
        "runs_path": str(runs_path),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "samples": len(samples),
        "audit_rows": len(audit_rows),
        "by_repo": dict(sorted(by_repo.items())),
        "by_failure_type": dict(sorted(by_failure_type.items())),
        "dropped": dict(sorted(dropped.items())),
        "quality_gate": {
            "audit_rows_ge_80": len(audit_rows) >= 80,
            "ready_for_manual_trace_audit": len(audit_rows) >= 80,
            "publishable_trace2code": False,
        },
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "trace2code": str(out_dir / "trace2code.jsonl"),
            "audit_jsonl": str(report_dir / "audit_samples.jsonl"),
            "audit_csv": str(report_dir / "audit_samples.csv"),
        },
    }
    write_json(out_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def trace_sample_from_repro_run(
    run: dict[str, Any],
    candidate: dict[str, Any] | None,
    max_root_files: int = 3,
) -> tuple[dict[str, Any] | None, str | None]:
    status = str(run.get("status") or "")
    failure_excerpt = str(run.get("failure_excerpt") or "")
    failure_trace_found = bool(run.get("failure_trace_found")) or bool(FAILURE_TRACE_RE.search(failure_excerpt))
    if status not in {"failed_expected", "failed_without_trace"}:
        return None, f"run_status_{run.get('status') or 'missing'}"
    if not failure_trace_found:
        return None, "missing_failure_trace"
    root_files = dedupe(str(path) for path in (run.get("implementation_files") or []))
    related_tests = dedupe(str(path) for path in (run.get("test_files") or []))
    if not root_files:
        return None, "missing_root_files"
    if len(root_files) > max_root_files:
        return None, "too_broad_root_files"

    fix_commit = str((candidate or {}).get("fix_commit") or "")
    signals = ["local_test_reproduction", "test_only_patch_applied", "failure_observed", classify_failure_type(failure_excerpt)]
    sample_id = stable_id(run.get("repo"), "trace_repro_run", run.get("id"), run.get("base_commit"), *root_files)
    query = {
        "failure_excerpt": redact_value(truncate_text(str(run.get("failure_excerpt") or ""), 4000), fix_commit),
        "command": first_failed_command(run),
        "run_strategy": ((candidate or {}).get("run") or {}).get("strategy") or "",
        "source_type": "local_test_reproduction",
    }
    sample = {
        "id": sample_id,
        "version": 2,
        "task_type": "trace2code",
        "repo": run.get("repo"),
        "base_commit": run.get("base_commit"),
        "query": query,
        "gold": {
            "root_cause_files": root_files,
            "root_cause_symbols": [],
            "related_tests": related_tests,
            "supporting_files": [],
            "negative_distractors": [],
            "fix_commit": fix_commit,
        },
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": run.get("base_commit")},
        "metadata": {
            "pr": run.get("pr_number"),
            "pr_url": (candidate or {}).get("pr_url"),
            "source": "local_test_reproduction",
            "source_id": run.get("id"),
            "confidence": "weak",
            "evidence": {
                "signals": signals,
                "run_status": run.get("status"),
                "failure_trace_reclassified": status == "failed_without_trace",
                "combined_log": run.get("combined_log"),
                "test_files": related_tests,
            },
            "generated_at": utc_now(),
        },
    }
    return sample, None


def classify_failure_type(text: str) -> str:
    if re.search(r"panic:", text, re.IGNORECASE):
        return "panic_failure"
    if re.search(r"AssertionError|--- FAIL:|Error Trace:|Not equal:", text, re.IGNORECASE):
        return "assertion_failure"
    return "compile_error"


def first_failed_command(run: dict[str, Any]) -> str:
    for result in run.get("command_results") or []:
        if int(result.get("returncode") or 0) != 0:
            return str(result.get("command") or result.get("args") or "")
    commands = run.get("commands") or []
    return str(commands[0]) if commands else ""


def trace_repro_audit_row(sample: dict[str, Any]) -> dict[str, str]:
    gold = sample.get("gold") or {}
    return {
        "sample_id": str(sample.get("id") or ""),
        "task_type": str(sample.get("task_type") or ""),
        "repo": str(sample.get("repo") or ""),
        "query_excerpt": json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True),
        "gold_files": json.dumps(gold.get("root_cause_files") or [], ensure_ascii=False),
        "verdict": "",
        "reason": "",
        "keep": "",
        "notes": "verdict options: " + "/".join(TRACE_REPRO_AUDIT_VERDICTS),
    }


def write_trace_outputs(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "trace2code.jsonl", samples)
    write_jsonl(out_dir / "code2test.jsonl", [])
    write_jsonl(out_dir / "comment2context.jsonl", [])


def run_one_trace_repro(
    candidate: dict[str, Any],
    raw_dir: Path,
    repos_dir: Path,
    out_dir: Path,
    timeout_seconds: int,
    repo_url_template: str,
    dry_run: bool,
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    repo = str(candidate["repo"])
    run_dir = out_dir / "runs" / candidate_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worktree = repos_dir / repo_slug(repo) / candidate_id
    test_patch_path = (run_dir / "test_patch.diff").resolve()
    commands = list((candidate.get("run") or {}).get("commands") or [])
    result: dict[str, Any] = {
        "id": candidate_id,
        "repo": repo,
        "pr_number": candidate.get("pr_number"),
        "base_commit": candidate.get("base_commit"),
        "implementation_files": candidate.get("implementation_files") or [],
        "test_files": candidate.get("test_files") or [],
        "worktree": str(worktree),
        "run_dir": str(run_dir),
        "commands": commands,
        "started_at": utc_now(),
        "dry_run": dry_run,
    }

    files_record = latest_by_pr(raw_dir / repo_slug(repo) / "pull_files.jsonl").get(int(candidate.get("pr_number") or 0))
    test_patch = build_test_only_patch(files_record, candidate.get("test_files") or [])
    test_patch_path.write_text(test_patch, encoding="utf-8")
    result["test_patch"] = str(test_patch_path)
    if not test_patch.strip():
        result.update(status="missing_test_patch", completed_at=utc_now())
        write_json(run_dir / "run.json", result)
        return result
    if dry_run:
        result.update(status="dry_run", completed_at=utc_now())
        write_json(run_dir / "run.json", result)
        return result

    remote_url = candidate.get("repo_url") or repo_url_template.format(repo=repo)
    checkout = prepare_worktree(worktree, remote_url, str(candidate["base_commit"]), timeout_seconds)
    result["checkout"] = checkout
    if checkout["returncode"] != 0:
        result.update(status="checkout_failed", completed_at=utc_now())
        write_json(run_dir / "run.json", result)
        return result

    apply_result = run_process(["git", "apply", "--whitespace=nowarn", str(test_patch_path)], cwd=worktree, timeout_seconds=timeout_seconds)
    result["apply_patch"] = process_record(apply_result)
    if apply_result.returncode != 0:
        result.update(status="patch_failed", completed_at=utc_now())
        write_json(run_dir / "run.json", result)
        return result

    command_results: list[dict[str, Any]] = []
    combined_output: list[str] = []
    final_returncode = 0
    for index, command in enumerate(commands):
        process = run_shell(command, cwd=worktree, timeout_seconds=timeout_seconds)
        record = process_record(process)
        record["command"] = command
        record["index"] = index
        command_results.append(record)
        combined_output.append(f"$ {command}\n{record['stdout']}\n{record['stderr']}")
        final_returncode = process.returncode
        if process.returncode != 0:
            break

    combined = "\n".join(combined_output)
    (run_dir / "combined.log").write_text(combined, encoding="utf-8")
    result["command_results"] = command_results
    result["combined_log"] = str(run_dir / "combined.log")
    result["failure_trace_found"] = bool(FAILURE_TRACE_RE.search(combined))
    result["failure_excerpt"] = truncate_text(combined, 4000)
    if final_returncode == 0:
        status = "passed_no_trace"
    elif result["failure_trace_found"]:
        status = "failed_expected"
    else:
        status = "failed_without_trace"
    result.update(status=status, completed_at=utc_now())
    write_json(run_dir / "run.json", result)
    return result


def build_trace_repro_candidate(
    repo: str,
    pr_number: int,
    pr: dict[str, Any],
    files_record: dict[str, Any],
    details_record: dict[str, Any] | None = None,
    max_changed_files: int = 30,
    max_source_files: int = 5,
    max_test_files: int = 5,
    min_score: int = 5,
) -> tuple[dict[str, Any] | None, str | None]:
    changed_files = [file for file in files_record.get("data", []) if file.get("filename")]
    changed_paths = [file["filename"] for file in changed_files]
    fix_commit = (pr.get("mergeCommit") or {}).get("oid")
    base_commit = pr.get("baseRefOid")
    if not base_commit or not fix_commit:
        return None, "missing_base_or_fix"
    if not changed_paths or len(changed_paths) > max_changed_files:
        return None, "changed_file_limit"
    if low_value_change_majority(changed_paths):
        return None, "low_value_pr"

    implementation = [
        file["filename"]
        for file in changed_files
        if file.get("status") not in {"added", "removed", "renamed"}
        and is_source_file(file["filename"])
        and not is_test_file(file["filename"])
    ]
    tests = [
        file["filename"]
        for file in changed_files
        if file.get("status") not in {"removed", "renamed"} and is_test_file(file["filename"])
    ]
    if not implementation or not tests:
        return None, "missing_source_or_test"
    if len(implementation) > max_source_files:
        return None, "too_many_source_files"
    if len(tests) > max_test_files:
        return None, "too_many_test_files"

    test_files = [file for file in changed_files if file["filename"] in tests]
    test_patch_text = "\n".join(file.get("patch") or "" for file in test_files)
    if not test_patch_text.strip():
        return None, "missing_test_patch"
    if not TEST_PATCH_SIGNAL_RE.search(test_patch_text):
        text = f"{pr.get('title') or ''}\n{clean_pr_body(pr.get('body') or '')}"
        if not BEHAVIOR_RE.search(text):
            return None, "weak_failure_signal"

    title = truncate_text(pr.get("title"), 300)
    body = clean_pr_body(pr.get("body") or "")
    query_preview = f"{title}\n{body}\n{' '.join(implementation)}"
    if has_query_noise(query_preview):
        return None, "query_noise"

    run = infer_run_strategy(tests)
    if run is None:
        return None, "unsupported_test_runner"

    evidence = repro_evidence(implementation, tests, title, body, test_patch_text, details_record)
    score = score_candidate(implementation, tests, evidence, run)
    if score < min_score:
        return None, "low_score"

    sample_id = stable_id(repo, "trace_repro_source", pr_number, base_commit, *sorted(implementation), *sorted(tests))
    candidate = {
        "id": sample_id,
        "source_type": "local_test_reproduction",
        "repo": repo,
        "pr_number": pr.get("number") or pr_number,
        "pr_url": pr.get("url"),
        "base_commit": base_commit,
        "fix_commit": fix_commit,
        "implementation_files": implementation,
        "test_files": tests,
        "changed_file_count": len(changed_paths),
        "score": score,
        "evidence": evidence,
        "run": run,
        "repro_plan": {
            "checkout": base_commit,
            "apply_patches": "test_files_only",
            "expected_signal": "test failure trace before implementation fix",
            "convert_to_trace2code_gold": "implementation_files_after_manual_or_execution_confirmation",
        },
        "metadata": {
            "created_at": pr.get("createdAt"),
            "merged_at": pr.get("mergedAt"),
            "test_patch_signal": bool(TEST_PATCH_SIGNAL_RE.search(test_patch_text)),
            "generated_at": utc_now(),
        },
    }
    return candidate, None


def select_candidates(candidate_path: Path, candidate_ids: Iterable[str] | None, limit: int | None) -> list[dict[str, Any]]:
    candidates = read_jsonl(candidate_path)
    wanted = {candidate_id for candidate_id in (candidate_ids or []) if candidate_id}
    if wanted:
        candidates = [candidate for candidate in candidates if str(candidate.get("id")) in wanted]
    if limit is not None and not wanted:
        candidates = candidates[: max(0, limit)]
    return candidates


def build_test_only_patch(files_record: dict[str, Any] | None, test_files: Iterable[str]) -> str:
    if not files_record:
        return ""
    wanted = set(test_files)
    chunks: list[str] = []
    for file in files_record.get("data", []):
        path = file.get("filename")
        patch = file.get("patch")
        if not path or path not in wanted or not patch:
            continue
        chunks.append(git_patch_for_file(path, str(file.get("status") or "modified"), patch))
    return "\n".join(chunks)


def git_patch_for_file(path: str, status: str, patch: str) -> str:
    header = [f"diff --git a/{path} b/{path}"]
    if status == "added":
        header.extend(["new file mode 100644", "--- /dev/null", f"+++ b/{path}"])
    else:
        header.extend([f"--- a/{path}", f"+++ b/{path}"])
    return "\n".join(header + [patch.rstrip(), ""])


def prepare_worktree(worktree: Path, remote_url: str, base_commit: str, timeout_seconds: int) -> dict[str, Any]:
    if worktree.exists() and not (worktree / ".git").exists():
        shutil.rmtree(worktree)
    if not worktree.exists():
        ensure_parent(worktree)
        clone = run_process_with_retry(["git", "clone", "--no-checkout", "--filter=blob:none", remote_url, str(worktree)], timeout_seconds=timeout_seconds)
        if clone.returncode != 0 and worktree.exists():
            shutil.rmtree(worktree)
            clone = run_process_with_retry(["git", "clone", "--no-checkout", remote_url, str(worktree)], timeout_seconds=timeout_seconds)
        if clone.returncode != 0:
            return {"step": "clone", **process_record(clone)}
    else:
        reset = run_process(["git", "reset", "--hard"], cwd=worktree, timeout_seconds=timeout_seconds)
        clean = run_process(["git", "clean", "-fdx"], cwd=worktree, timeout_seconds=timeout_seconds)
        if reset.returncode != 0 or clean.returncode != 0:
            return {"step": "reset", "reset": process_record(reset), "clean": process_record(clean), "returncode": reset.returncode or clean.returncode}

    run_process(["git", "remote", "set-url", "origin", remote_url], cwd=worktree, timeout_seconds=timeout_seconds)
    fetch = run_process_with_retry(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=worktree, timeout_seconds=timeout_seconds)
    if fetch.returncode != 0:
        fetch = run_process_with_retry(["git", "fetch", "origin", base_commit], cwd=worktree, timeout_seconds=timeout_seconds)
    if fetch.returncode != 0:
        return {"step": "fetch", **process_record(fetch)}
    checkout = run_process(["git", "checkout", "--detach", base_commit], cwd=worktree, timeout_seconds=timeout_seconds)
    if checkout.returncode != 0:
        return {"step": "checkout", **process_record(checkout)}
    return {"step": "checkout", **process_record(checkout)}


def run_process(args: list[str], cwd: Path | None = None, timeout_seconds: int = 900) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(args, 124, ensure_text(error.stdout), ensure_text(error.stderr) or f"Timed out after {timeout_seconds}s")


def run_process_with_retry(args: list[str], cwd: Path | None = None, timeout_seconds: int = 900, attempts: int = 2) -> subprocess.CompletedProcess[str]:
    result = run_process(args, cwd=cwd, timeout_seconds=timeout_seconds)
    for _attempt in range(1, attempts):
        if result.returncode == 0 or not is_transient_git_failure(result):
            break
        result = run_process(args, cwd=cwd, timeout_seconds=timeout_seconds)
    return result


def is_transient_git_failure(process: subprocess.CompletedProcess[str]) -> bool:
    text = f"{process.stdout or ''}\n{process.stderr or ''}".lower()
    return any(marker in text for marker in ("internal server error", "http 500", "rpc failed", "expected flush after ref listing"))


def run_shell(command: str, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(command, 124, ensure_text(error.stdout), ensure_text(error.stderr) or f"Timed out after {timeout_seconds}s")


def process_record(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "args": process.args,
        "returncode": process.returncode,
        "stdout": ensure_text(process.stdout),
        "stderr": ensure_text(process.stderr),
    }


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def redact_value(value: Any, fix_commit: str | None) -> Any:
    if not fix_commit:
        return value
    if isinstance(value, str):
        return value.replace(fix_commit, "[fix_commit]")
    if isinstance(value, list):
        return [redact_value(item, fix_commit) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, fix_commit) for key, item in value.items()}
    return value


def infer_run_strategy(tests: list[str]) -> dict[str, Any] | None:
    suffixes = Counter(PurePosixPath(path).suffix.lower() for path in tests)
    quoted = " ".join(shlex.quote(path) for path in tests)
    if suffixes[".py"]:
        return {"strategy": "pytest", "commands": [f"python -m pytest {quoted}"], "confidence": "high"}
    if suffixes[".go"]:
        dirs = sorted({str(PurePosixPath(path).parent) for path in tests})
        return {"strategy": "go_test_package", "commands": [f"go test ./{shlex.quote(directory)}" for directory in dirs], "confidence": "high"}
    if suffixes[".rs"]:
        return {"strategy": "cargo_test_guess", "commands": ["cargo test"], "confidence": "medium"}
    if suffixes[".java"] or suffixes[".kt"]:
        classes = [PurePosixPath(path).stem for path in tests]
        selector = " ".join(f"--tests {shlex.quote(name)}" for name in classes)
        return {"strategy": "gradle_test_guess", "commands": [f"./gradlew test {selector}".strip()], "confidence": "medium"}
    if suffixes[".ts"] or suffixes[".tsx"] or suffixes[".js"] or suffixes[".jsx"]:
        return {"strategy": "node_test_guess", "commands": [f"pnpm test -- {quoted}"], "confidence": "medium"}
    return None


def repro_evidence(
    implementation: list[str],
    tests: list[str],
    title: str,
    body: str,
    test_patch_text: str,
    details_record: dict[str, Any] | None,
) -> list[str]:
    evidence = ["same_pr_source_and_test_change", "test_patch_available"]
    if TEST_PATCH_SIGNAL_RE.search(test_patch_text):
        evidence.append("assertion_or_failure_test_patch")
    if BEHAVIOR_RE.search(f"{title}\n{body}"):
        evidence.append("behavior_or_failure_pr_text")
    if any(path_token_overlap(src, test) for src in implementation for test in tests):
        evidence.append("source_test_module_overlap")
    if any(str(PurePosixPath(src).parent) != str(PurePosixPath(test).parent) for src in implementation for test in tests):
        evidence.append("cross_directory_test")
    if details_record and commit_details_touch_sources_and_tests(details_record):
        evidence.append("commit_detail_confirms_source_and_test_changes")
    return dedupe(evidence)


def commit_details_touch_sources_and_tests(details_record: dict[str, Any]) -> bool:
    for detail in details_record.get("data", []):
        paths = [file.get("filename", "") for file in detail.get("files", [])]
        implementation, tests, _ignored = split_changed_files(paths)
        if implementation and tests:
            return True
    return False


def score_candidate(implementation: list[str], tests: list[str], evidence: list[str], run: dict[str, Any]) -> int:
    score = len(evidence)
    if len(implementation) <= 3:
        score += 1
    if len(tests) <= 2:
        score += 1
    if run.get("confidence") == "high":
        score += 1
    if "source_test_module_overlap" in evidence:
        score += 2
    if "assertion_or_failure_test_patch" in evidence:
        score += 2
    return score


def repro_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    run = candidate.get("run") or {}
    return (
        -int(candidate.get("score") or 0),
        int(run.get("confidence") != "high"),
        len(candidate.get("implementation_files") or []),
        len(candidate.get("test_files") or []),
        candidate.get("repo", ""),
        -int(candidate.get("pr_number") or 0),
        candidate.get("id", ""),
    )


def audit_row(candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "candidate_id": str(candidate.get("id") or ""),
        "repo": str(candidate.get("repo") or ""),
        "pr_number": str(candidate.get("pr_number") or ""),
        "score": str(candidate.get("score") or 0),
        "implementation_files": json.dumps(candidate.get("implementation_files") or [], ensure_ascii=False),
        "test_files": json.dumps(candidate.get("test_files") or [], ensure_ascii=False),
        "run_strategy": str((candidate.get("run") or {}).get("strategy") or ""),
        "commands": json.dumps((candidate.get("run") or {}).get("commands") or [], ensure_ascii=False),
        "rationale": "; ".join(candidate.get("evidence") or []),
        "verdict": "",
        "keep": "",
        "notes": "verdict options: " + "/".join(REPRO_AUDIT_VERDICTS),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
