from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import summarize_audit, write_audit_sample
from .baseline import CANDIDATE_FILTERS, evaluate_lexical_baseline
from .clone import verify_base_commits
from .corpus import build_candidate_corpus, sample_paths_from_derived
from .curate import export_curated_samples
from .crawler import crawl_commit_details_for_raw, crawl_repo, write_manifest
from .derive import derive_repo
from .diagnostics import diagnose_benchmark
from .embedding_eval import (
    default_embedding_cache_dir,
    default_embedding_summary_path,
    evaluate_embedding_baseline,
)
from .github_api import GitHubAPI
from .hardmine import DEFAULT_HARDMINE_SOURCES, DEFAULT_HARDMINE_TASKS, export_hardmine_candidates
from .hardness import diagnose_hardness, filter_hard_pool, summarize_seed_audit
from .io import load_targets, read_jsonl, repo_slug
from .logs import crawl_job_logs
from .model_report import report_model_leaderboard
from .quality import validate_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arb", description="Agent Retrieval Benchmark crawler")
    parser.add_argument("--token", help="GitHub token. Defaults to GITHUB_TOKEN or GH_TOKEN.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/http"), help="ETag response cache directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Write repo_manifest.jsonl for configured targets.")
    manifest.add_argument("--targets", type=Path, default=Path("configs/crawl_targets.json"))
    manifest.add_argument("--output", type=Path, default=Path("data/repo_manifest.jsonl"))

    crawl = subparsers.add_parser("crawl", help="Crawl one GitHub repo.")
    crawl.add_argument("--repo", required=True)
    crawl.add_argument("--out", type=Path, default=Path("data/raw"))
    crawl.add_argument("--limit-prs", type=int, default=20)
    crawl.add_argument("--page-size", type=int, default=25)
    crawl.add_argument("--max-changed-files", type=int, default=20)
    crawl.add_argument("--no-checks", action="store_true")
    crawl.add_argument("--dry-run", action="store_true")

    crawl_all = subparsers.add_parser("crawl-all", help="Crawl all primary targets.")
    crawl_all.add_argument("--targets", type=Path, default=Path("configs/crawl_targets.json"))
    crawl_all.add_argument("--out", type=Path, default=Path("data/raw"))
    crawl_all.add_argument("--limit-prs", type=int, default=20)
    crawl_all.add_argument("--page-size", type=int, default=25)
    crawl_all.add_argument("--max-changed-files", type=int, default=20)
    crawl_all.add_argument("--no-checks", action="store_true")
    crawl_all.add_argument("--dry-run", action="store_true")

    derive = subparsers.add_parser("derive", help="Build weak benchmark samples from raw JSONL.")
    derive.add_argument("--raw", type=Path, default=Path("data/raw"))
    derive.add_argument("--out", type=Path, default=Path("data/derived"))
    derive.add_argument("--repo", action="append", help="Repo to derive. Can be repeated. Defaults to raw dirs.")
    derive.add_argument("--max-changed-files", type=int, default=20)

    hardmine_export = subparsers.add_parser("export-hardmine-candidates", help="Merge local samples into a V1 hard-mining candidate set.")
    hardmine_export.add_argument("--source", action="append", type=Path, help="Source file or directory. Can be repeated.")
    hardmine_export.add_argument("--out", type=Path, default=Path("data/benchmark/v1_candidate_round1"))
    hardmine_export.add_argument(
        "--tasks",
        default=",".join(DEFAULT_HARDMINE_TASKS),
        help="Comma-separated task types to export.",
    )
    hardmine_export.add_argument("--corpus-manifest", type=Path, help="Optional corpus manifest used to filter base commits.")
    hardmine_export.add_argument("--require-corpus", action="store_true", help="Drop samples whose repo/base_commit is absent from the corpus manifest.")
    hardmine_export.add_argument("--limit-samples", type=int)

    validate = subparsers.add_parser("validate", help="Validate derived sample JSONL files.")
    validate.add_argument("samples", nargs="+", type=Path)

    audit_sample = subparsers.add_parser("audit-sample", help="Create JSONL/CSV manual audit sheets from derived samples.")
    audit_sample.add_argument("--derived", type=Path, default=Path("data/derived"))
    audit_sample.add_argument("--out", type=Path, default=Path("data/audit/v0"))
    audit_sample.add_argument("--per-task", type=int, default=20)
    audit_sample.add_argument("--seed", type=int, default=13)
    audit_sample.add_argument("--task", action="append", help="Task to sample. Can be repeated. Defaults to all tasks.")
    audit_sample.add_argument("--formats", default="jsonl,csv", help="Comma-separated output formats: jsonl,csv.")

    audit_summary = subparsers.add_parser("audit-summary", help="Summarize manual audit verdicts and write keep list.")
    audit_summary.add_argument("audit", type=Path)
    audit_summary.add_argument("--out", type=Path, default=Path("data/audit/v0/summary.json"))
    audit_summary.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))

    seed_audit_summary = subparsers.add_parser("seed-audit-summary", help="Summarize V1 seed audit verdicts and write keep list.")
    seed_audit_summary.add_argument("audit", type=Path)
    seed_audit_summary.add_argument("--out", type=Path, default=Path("data/reports/v0_2/v1_seed_audit_summary.json"))
    seed_audit_summary.add_argument("--keep-list", type=Path, default=Path("data/reports/v0_2/v1_seed_keep.jsonl"))

    export_curated = subparsers.add_parser("export-curated", help="Export audited keep-list samples into benchmark JSONL files.")
    export_curated.add_argument("--derived", type=Path, default=Path("data/derived_token_logs"))
    export_curated.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))
    export_curated.add_argument("--out", type=Path, default=Path("data/benchmark/v0"))
    export_curated.add_argument(
        "--tasks",
        default="code2test,comment2context,trace2code",
        help="Comma-separated task types to export.",
    )
    export_curated.add_argument("--include-nonvalid", action="store_true", help="Include non-valid keep-list rows.")

    logs = subparsers.add_parser("crawl-logs", help="Download GitHub Actions job logs for failed check runs.")
    logs.add_argument("--raw", type=Path, default=Path("data/raw"))
    logs.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    logs.add_argument("--max-jobs", type=int, default=25)
    logs.add_argument("--max-bytes", type=int, default=2_000_000)
    logs.add_argument(
        "--conclusions",
        default="failure,timed_out,action_required",
        help="Comma-separated check conclusions to download.",
    )

    commit_details = subparsers.add_parser("crawl-commit-details", help="Fetch commit changed-file details for crawled PR commits.")
    commit_details.add_argument("--raw", type=Path, default=Path("data/raw"))
    commit_details.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    commit_details.add_argument("--limit-prs", type=int)
    commit_details.add_argument("--max-commits-per-pr", type=int)

    verify = subparsers.add_parser("verify-bases", help="Fetch and verify base commits in bare repo caches.")
    verify.add_argument("--raw", type=Path, default=Path("data/raw"))
    verify.add_argument("--repos-dir", type=Path, default=Path("data/repos"))
    verify.add_argument("--repo", action="append", help="Repo to verify. Defaults to raw dirs.")
    verify.add_argument("--limit", type=int, default=50)

    corpus = subparsers.add_parser("build-corpus", help="Build file/function candidate chunks for sample base commits.")
    corpus.add_argument("samples", nargs="*", type=Path, help="Derived sample JSONL files. Defaults to --derived/*.jsonl.")
    corpus.add_argument("--derived", type=Path, default=Path("data/derived"))
    corpus.add_argument("--repos-dir", type=Path, default=Path("data/repos"))
    corpus.add_argument("--out", type=Path, default=Path("data/corpus/v0"))
    corpus.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))
    corpus.add_argument("--no-keep-list", action="store_true")
    corpus.add_argument("--repo", action="append", help="Repo to include. Can be repeated.")
    corpus.add_argument("--limit-samples", type=int)
    corpus.add_argument("--limit-pairs", type=int)
    corpus.add_argument("--max-file-bytes", type=int, default=400_000)
    corpus.add_argument("--max-chunk-chars", type=int, default=8_000)
    corpus.add_argument("--max-files-per-commit", type=int, default=20_000)
    corpus.add_argument("--remote-base", default="https://github.com")

    baseline = subparsers.add_parser("eval-baseline", help="Run a lexical/exact retrieval baseline.")
    baseline.add_argument("samples", nargs="*", type=Path, help="Derived sample JSONL files. Defaults to --derived/*.jsonl.")
    baseline.add_argument("--derived", type=Path, default=Path("data/derived"))
    baseline.add_argument("--corpus", type=Path, default=Path("data/corpus/v0"))
    baseline.add_argument("--out", type=Path)
    baseline.add_argument("--details", type=Path)
    baseline.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))
    baseline.add_argument("--no-keep-list", action="store_true")
    baseline.add_argument("--limit-samples", type=int)
    baseline.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    baseline.add_argument("--dry-run", action="store_true", help="Use sample gold/supporting paths as a tiny synthetic corpus.")

    embedding = subparsers.add_parser("eval-embedding", help="Run an embedding retrieval baseline.")
    embedding.add_argument(
        "samples",
        nargs="*",
        type=Path,
        help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.",
    )
    embedding.add_argument("--derived", type=Path, default=Path("data/benchmark/v0_1"))
    embedding.add_argument("--corpus", type=Path, default=Path("data/corpus/v0_1"))
    embedding.add_argument("--model", required=True, help="SentenceTransformer-compatible model name or path.")
    embedding.add_argument("--out", type=Path)
    embedding.add_argument("--details", type=Path)
    embedding.add_argument("--cache", type=Path)
    embedding.add_argument("--keep-list", type=Path, default=Path("data/audit/v0_clean/keep_samples.jsonl"))
    embedding.add_argument("--no-keep-list", action="store_true")
    embedding.add_argument("--limit-samples", type=int)
    embedding.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    embedding.add_argument("--batch-size", type=int, default=32)
    embedding.add_argument("--device", help="SentenceTransformer device, e.g. cpu, cuda, mps.")
    embedding.add_argument("--query-prefix", default="")
    embedding.add_argument("--passage-prefix", default="")
    embedding.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    embedding.add_argument("--no-progress", action="store_true", help="Disable embedding evaluation progress output.")
    embedding.add_argument("--trust-remote-code", action="store_true")

    diagnose = subparsers.add_parser("diagnose", help="Diagnose benchmark difficulty and baseline quality.")
    diagnose.add_argument("--samples", type=Path, default=Path("data/benchmark/v0_1/samples.jsonl"))
    diagnose.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v0_1/corpus_manifest.jsonl"))
    diagnose.add_argument("--details", type=Path, default=Path("data/eval/v0_1/lexical_details.jsonl"))
    diagnose.add_argument("--out", type=Path, default=Path("data/reports/v0_1"))
    diagnose.add_argument(
        "--tasks",
        default="code2test,comment2context,trace2code",
        help="Comma-separated task types to include in the diagnosis.",
    )

    hardness = subparsers.add_parser("hardness", help="Diagnose hard/easy samples and build a V1 candidate pool.")
    hardness.add_argument("samples", nargs="*", type=Path, help="Sample JSONL files. Defaults to --derived/*.jsonl.")
    hardness.add_argument("--derived", type=Path, default=Path("data/benchmark/v0_2"))
    hardness.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v0_2/corpus_manifest.jsonl"))
    hardness.add_argument("--details", type=Path, default=Path("data/eval/v0_2/lexical_details.jsonl"))
    hardness.add_argument("--out", type=Path, default=Path("data/reports/v0_2"))
    hardness.add_argument("--pool-out", type=Path)
    hardness.add_argument("--keep-list", type=Path)
    hardness.add_argument(
        "--tasks",
        default="code2test,comment2context,trace2code",
        help="Comma-separated task types to include in the hardness report.",
    )
    hardness.add_argument("--hard-recall20-threshold", type=float, default=1.0)
    hardness.add_argument("--hard-mrr-threshold", type=float, default=0.25)

    hard_pool_filter = subparsers.add_parser("hard-pool-filter", help="Filter hardness candidates into a V1 seed pool.")
    hard_pool_filter.add_argument("--pool", type=Path, default=Path("data/reports/v0_2/candidate_keep_pool.jsonl"))
    hard_pool_filter.add_argument("--audit", type=Path, help="Optional manual audit JSONL/CSV for candidate verdicts.")
    hard_pool_filter.add_argument("--out", type=Path, default=Path("data/reports/v0_2/v1_seed_candidates.jsonl"))
    hard_pool_filter.add_argument("--summary", type=Path, default=Path("data/reports/v0_2/v1_seed_summary.json"))
    hard_pool_filter.add_argument("--audit-out", type=Path, default=Path("data/reports/v0_2/v1_seed_audit_samples.jsonl"))
    hard_pool_filter.add_argument("--audit-csv", type=Path, default=Path("data/reports/v0_2/v1_seed_audit_samples.csv"))
    hard_pool_filter.add_argument("--audit-limit", type=int, default=120)
    hard_pool_filter.add_argument("--min-score", type=float, default=0.0)
    hard_pool_filter.add_argument("--no-unaudited", action="store_true", help="Only keep manually audited valid candidates.")

    report_models = subparsers.add_parser("report-models", help="Build a Markdown/JSON leaderboard from eval summaries.")
    report_models.add_argument("--eval-dir", type=Path, default=Path("data/eval/v0_1"))
    report_models.add_argument("--out", type=Path, default=Path("data/reports/v0_1/model_leaderboard.md"))
    report_models.add_argument("--json-out", type=Path)

    args = parser.parse_args(argv)
    api = GitHubAPI(token=args.token, cache_dir=args.cache_dir)

    if args.command == "manifest":
        targets = load_targets(args.targets)["primary"]
        count = write_manifest(api, targets, args.output)
        print(json.dumps({"wrote": count, "output": str(args.output), "authenticated": api.authenticated}, indent=2))
        return 0
    if args.command == "crawl":
        summary = crawl_repo(
            api,
            args.repo,
            args.out,
            limit_prs=args.limit_prs,
            page_size=args.page_size,
            max_changed_files=args.max_changed_files,
            include_checks=not args.no_checks,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.command == "crawl-all":
        targets = load_targets(args.targets)["primary"]
        summaries = []
        for target in targets:
            summaries.append(
                crawl_repo(
                    api,
                    target["repo"],
                    args.out,
                    limit_prs=args.limit_prs,
                    page_size=args.page_size,
                    max_changed_files=args.max_changed_files,
                    include_checks=not args.no_checks,
                    dry_run=args.dry_run,
                )
            )
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return 0
    if args.command == "derive":
        repos = args.repo or _repos_from_raw(args.raw)
        result = {repo: derive_repo(args.raw, repo, args.out, args.max_changed_files) for repo in repos}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "export-hardmine-candidates":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        sources = args.source or list(DEFAULT_HARDMINE_SOURCES)
        result = export_hardmine_candidates(
            sources=sources,
            out_dir=args.out,
            tasks=tasks,
            corpus_manifest=args.corpus_manifest,
            require_corpus=args.require_corpus,
            limit_samples=args.limit_samples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        result = [validate_samples(path) for path in args.samples]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item["invalid"] for item in result) else 0
    if args.command == "audit-sample":
        formats = [item.strip() for item in args.formats.split(",") if item.strip()]
        result = write_audit_sample(args.derived, args.out, per_task=args.per_task, seed=args.seed, tasks=args.task or None, formats=formats)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "audit-summary":
        result = summarize_audit(args.audit, out_path=args.out, keep_list_path=args.keep_list)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "seed-audit-summary":
        result = summarize_seed_audit(args.audit, out_path=args.out, keep_list_path=args.keep_list)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["invalid_verdicts"] else 0
    if args.command == "export-curated":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        result = export_curated_samples(
            derived_dir=args.derived,
            keep_list=args.keep_list,
            out_dir=args.out,
            tasks=tasks,
            valid_only=not args.include_nonvalid,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["missing_keep_ids"] else 0
    if args.command == "crawl-logs":
        repos = args.repo or _repos_from_raw(args.raw)
        conclusions = {item.strip() for item in args.conclusions.split(",") if item.strip()}
        result = [
            crawl_job_logs(api, args.raw, repo, max_jobs=args.max_jobs, max_bytes=args.max_bytes, conclusions=conclusions)
            for repo in repos
        ]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item["errors"] for item in result) else 0
    if args.command == "crawl-commit-details":
        repos = args.repo or _repos_from_raw(args.raw)
        result = [
            crawl_commit_details_for_raw(
                api,
                args.raw,
                repo,
                limit_prs=args.limit_prs,
                max_commits_per_pr=args.max_commits_per_pr,
            )
            for repo in repos
        ]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item["errors"] for item in result) else 0
    if args.command == "verify-bases":
        repos = args.repo or _repos_from_raw(args.raw)
        result = []
        for repo in repos:
            commits = _base_commits(args.raw, repo)[: args.limit]
            result.append(verify_base_commits(repo, commits, args.repos_dir))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item["missing"] for item in result) else 0
    if args.command == "build-corpus":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        result = build_candidate_corpus(
            sample_paths=sample_paths,
            out_dir=args.out,
            repos_dir=args.repos_dir,
            repos=set(args.repo) if args.repo else None,
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            limit_pairs=args.limit_pairs,
            max_file_bytes=args.max_file_bytes,
            max_chunk_chars=args.max_chunk_chars,
            max_files_per_commit=args.max_files_per_commit,
            remote_base=args.remote_base,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["missing"] else 0
    if args.command == "eval-baseline":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_lexical_summary_path(args.candidate_filter)
        result = evaluate_lexical_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            dry_run=args.dry_run,
            candidate_filter=args.candidate_filter,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-embedding":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_embedding_summary_path(args.model, candidate_filter=args.candidate_filter)
        result = evaluate_embedding_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            model_name=args.model,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            cache_dir=args.cache or default_embedding_cache_dir(args.model),
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            device=args.device,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            normalize_embeddings=not args.no_normalize,
            trust_remote_code=args.trust_remote_code,
            progress=not args.no_progress,
            candidate_filter=args.candidate_filter,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "diagnose":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        result = diagnose_benchmark(
            samples_path=args.samples,
            corpus_manifest_path=args.corpus_manifest,
            details_path=args.details,
            out_dir=args.out,
            tasks=tasks,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "hardness":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = diagnose_hardness(
            sample_paths=sample_paths,
            corpus_manifest_path=args.corpus_manifest,
            details_path=args.details,
            out_dir=args.out,
            pool_out_path=args.pool_out,
            keep_list=args.keep_list,
            tasks=tasks,
            hard_recall20_threshold=args.hard_recall20_threshold,
            hard_mrr_threshold=args.hard_mrr_threshold,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "hard-pool-filter":
        result = filter_hard_pool(
            pool_path=args.pool,
            out_path=args.out,
            summary_path=args.summary,
            audit_path=args.audit,
            audit_out_path=args.audit_out,
            audit_csv_path=args.audit_csv,
            audit_limit=args.audit_limit,
            min_score=args.min_score,
            include_unaudited=not args.no_unaudited,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-models":
        result = report_model_leaderboard(
            eval_dir=args.eval_dir,
            out_path=args.out,
            json_out_path=args.json_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    return 2


def default_baseline_details_path(out_path: Path) -> Path:
    if out_path.stem.endswith("_summary"):
        return out_path.with_name(f"{out_path.stem.removesuffix('_summary')}_details.jsonl")
    return out_path.with_suffix(".details.jsonl")


def default_lexical_summary_path(candidate_filter: str = "all_files", root: Path = Path("data/eval/v0")) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"lexical{suffix}.json"


def _repos_from_raw(raw_dir: Path) -> list[str]:
    repos = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else []):
        if path.is_dir() and "__" in path.name:
            repos.append(path.name.replace("__", "/", 1))
    return repos


def _base_commits(raw_dir: Path, repo: str) -> list[str]:
    commits: list[str] = []
    for record in read_jsonl(raw_dir / repo_slug(repo) / "pull_requests.jsonl"):
        base = (record.get("data") or {}).get("baseRefOid")
        if base:
            commits.append(base)
    return commits


if __name__ == "__main__":
    sys.exit(main())
