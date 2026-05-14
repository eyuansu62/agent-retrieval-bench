from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import summarize_audit, write_audit_sample
from .baseline import CANDIDATE_FILTERS, evaluate_lexical_baseline
from .clone import verify_base_commits
from .code2test_pr import mine_code2test_prs
from .corpus import build_candidate_corpus, sample_paths_from_derived
from .curate import export_curated_samples
from .crawler import crawl_commit_details_for_raw, crawl_pr_checks, crawl_repo, write_manifest
from .derive import derive_repo
from .diagnostics import diagnose_benchmark
from .embedding_eval import (
    VoyageAPIEmbedder,
    default_embedding_cache_dir,
    default_embedding_summary_path,
    evaluate_embedding_baseline,
)
from .github_api import GitHubAPI
from .hardmine import DEFAULT_HARDMINE_SOURCES, DEFAULT_HARDMINE_TASKS, export_hardmine_candidates
from .hardness import diagnose_hardness, filter_hard_pool, merge_seed_audits, summarize_seed_audit
from .io import load_targets, read_jsonl, repo_slug
from .logs import crawl_job_logs
from .model_report import report_model_leaderboard
from .quality import validate_samples
from .release import DEFAULT_DATASET_REPO, download_benchmark_release
from .repomap_eval import evaluate_repomap_baseline
from .seed_report import report_v1_seed
from .trace_preflight import mine_trace2code, trace_debug_drops, trace_debug_summary, trace_preflight, trace_source_scan
from .trace_repro import mine_trace_repro_runs, run_trace_repro, trace_repro_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arb", description="Agent Retrieval Benchmark crawler")
    parser.add_argument("--token", help="GitHub token. Defaults to GITHUB_TOKEN or GH_TOKEN.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/http"), help="ETag response cache directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Write repo_manifest.jsonl for configured targets.")
    manifest.add_argument("--targets", type=Path, default=Path("configs/crawl_targets.json"))
    manifest.add_argument("--output", type=Path, default=Path("data/repo_manifest.jsonl"))

    download_release = subparsers.add_parser("download-benchmark", help="Download, verify, and extract a released benchmark bundle from Hugging Face.")
    download_release.add_argument("--version", default="v1")
    download_release.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    download_release.add_argument("--revision")
    download_release.add_argument("--local-dir", type=Path, default=Path("data"))
    download_release.add_argument("--hf-token", help="Hugging Face token. Defaults to HF_TOKEN or stored hf auth.")
    download_release.add_argument("--skip-download", action="store_true", help="Use an already downloaded release bundle.")
    download_release.add_argument("--no-extract", action="store_true", help="Only download and verify the archive.")
    download_release.add_argument("--force", action="store_true", help="Replace existing benchmark/corpus/eval/report directories for this version.")
    download_release.add_argument("--hf-bin", default="hf")
    download_release.add_argument("--zstd-bin", default="zstd")
    download_release.add_argument("--tar-bin", default="tar")

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

    crawl_checks = subparsers.add_parser("crawl-pr-checks", help="Backfill PR files, commits, commit details, and check runs for trace mining.")
    crawl_checks.add_argument("--repo", action="append", required=True, help="Repo to process. Can be repeated.")
    crawl_checks.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    crawl_checks.add_argument("--limit-prs", type=int, default=300)
    crawl_checks.add_argument("--page-size", type=int, default=50)
    crawl_checks.add_argument("--max-changed-files", type=int, default=30)
    crawl_checks.add_argument("--include-review-comments", action="store_true")
    crawl_checks.add_argument("--refresh-existing-checks", action="store_true")
    crawl_checks.add_argument("--repair-empty-state", action="store_true")
    crawl_checks.add_argument("--max-pages", type=int, help="Maximum GraphQL pages to scan for this run.")
    crawl_checks.add_argument("--dry-run", action="store_true")

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

    mine_code = subparsers.add_parser("mine-code2test-prs", help="Mine PR-level code2test candidates from raw PR files.")
    mine_code.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    mine_code.add_argument("--out", type=Path, default=Path("data/benchmark/v1_code2test_pr_candidates"))
    mine_code.add_argument("--report-out", type=Path, default=Path("data/reports/v1_code2test_pr_candidates"))
    mine_code.add_argument("--audit", type=Path, default=Path("data/reports/v1/audited_ids.csv"))
    mine_code.add_argument("--audited-pool", type=Path, default=Path("data/reports/v1_candidate_round1/candidate_keep_pool.jsonl"))
    mine_code.add_argument("--corpus-manifest", type=Path)
    mine_code.add_argument("--require-corpus", action="store_true")
    mine_code.add_argument("--require-gold-in-corpus", action="store_true")
    mine_code.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    mine_code.add_argument("--max-changed-files", type=int, default=20)
    mine_code.add_argument("--max-tests", type=int, default=3)
    mine_code.add_argument("--audit-limit", type=int, default=120)
    mine_code.add_argument("--limit-samples", type=int)

    trace_pref = subparsers.add_parser("trace-preflight", help="Count real root-cause trace candidates in raw signals.")
    trace_pref.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    trace_pref.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_preflight"))
    trace_pref.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    trace_pref.add_argument("--max-changed-files", type=int, default=20)

    mine_trace = subparsers.add_parser("mine-trace2code", help="Mine strict trace2code benchmark candidates from raw CI/check/review signals.")
    mine_trace.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    mine_trace.add_argument("--out", type=Path, default=Path("data/benchmark/v1_trace_candidate_round1"))
    mine_trace.add_argument("--report-out", type=Path, default=Path("data/reports/v1_trace_candidate_round1"))
    mine_trace.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    mine_trace.add_argument("--max-changed-files", type=int, default=20)
    mine_trace.add_argument("--audit-limit", type=int, default=120)
    mine_trace.add_argument("--limit-samples", type=int)
    mine_trace.add_argument("--no-review-comments", action="store_true", help="Only mine CI/check signals; skip review comment snippets.")

    trace_debug = subparsers.add_parser("trace-debug-drops", help="Sample weak CI/check trace signals that were dropped by the strict trace miner.")
    trace_debug.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    trace_debug.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_debug"))
    trace_debug.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    trace_debug.add_argument("--max-changed-files", type=int, default=20)
    trace_debug.add_argument("--audit-limit", type=int, default=120)

    trace_debug_summary_parser = subparsers.add_parser("trace-debug-summary", help="Summarize audited weak trace signals and export recoverable rows.")
    trace_debug_summary_parser.add_argument("audit", type=Path)
    trace_debug_summary_parser.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_debug/audit_summary.json"))
    trace_debug_summary_parser.add_argument("--recoverable-out", type=Path, default=Path("data/reports/v1_trace_debug/recoverable_signals.jsonl"))

    trace_source = subparsers.add_parser("trace-source-scan", help="Rank CI/check log sources by likelihood of yielding real trace2code samples.")
    trace_source.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    trace_source.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_source_round1"))
    trace_source.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    trace_source.add_argument("--max-changed-files", type=int, default=20)
    trace_source.add_argument("--audit-limit", type=int, default=50)
    trace_source.add_argument("--min-score", type=int, default=4)

    trace_repro = subparsers.add_parser("trace-repro-source", help="Build local test-reproduction source candidates for trace2code.")
    trace_repro.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    trace_repro.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_repro_source_round1"))
    trace_repro.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    trace_repro.add_argument("--max-changed-files", type=int, default=30)
    trace_repro.add_argument("--max-source-files", type=int, default=5)
    trace_repro.add_argument("--max-test-files", type=int, default=5)
    trace_repro.add_argument("--min-score", type=int, default=5)
    trace_repro.add_argument("--audit-limit", type=int, default=120)
    trace_repro.add_argument("--limit-candidates", type=int)

    run_repro = subparsers.add_parser("run-trace-repro", help="Checkout base commits, apply test-only patches, and run focused repro commands.")
    run_repro.add_argument("--candidate", type=Path, default=Path("data/reports/v1_trace_repro_source_round1/repro_candidates.jsonl"))
    run_repro.add_argument("--id", action="append", dest="candidate_id", help="Candidate id to run. Can be repeated.")
    run_repro.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    run_repro.add_argument("--repos-dir", type=Path, default=Path("data/repro_worktrees"))
    run_repro.add_argument("--out", type=Path, default=Path("data/reports/v1_trace_repro_runs"))
    run_repro.add_argument("--limit", type=int, default=1, help="Number of candidates to run when --id is not provided.")
    run_repro.add_argument("--timeout-seconds", type=int, default=900)
    run_repro.add_argument("--repo-url-template", default="https://github.com/{repo}.git")
    run_repro.add_argument("--dry-run", action="store_true")
    run_repro.add_argument("--continue-on-error", action="store_true")

    mine_repro_runs = subparsers.add_parser("mine-trace-repro-runs", help="Convert executed local repro failures into trace2code audit candidates.")
    mine_repro_runs.add_argument("--candidates", type=Path, default=Path("data/reports/v1_trace_repro_source_round1/repro_candidates.jsonl"))
    mine_repro_runs.add_argument("--runs", type=Path, default=Path("data/reports/v1_trace_repro_runs/runs.jsonl"))
    mine_repro_runs.add_argument("--out", type=Path, default=Path("data/benchmark/v1_trace_repro_candidate_round1"))
    mine_repro_runs.add_argument("--report-out", type=Path, default=Path("data/reports/v1_trace_repro_candidate_round1"))
    mine_repro_runs.add_argument("--max-root-files", type=int, default=3)
    mine_repro_runs.add_argument("--audit-limit", type=int, default=120)

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

    merge_seed_audit = subparsers.add_parser("merge-seed-audits", help="Merge multiple V1 seed audit files into one keep list.")
    merge_seed_audit.add_argument("--audit", action="append", type=Path, required=True, help="Audit CSV/JSONL path. Can be repeated.")
    merge_seed_audit.add_argument("--out", type=Path, default=Path("data/reports/v1_seed_round1/audit_summary.json"))
    merge_seed_audit.add_argument("--keep-list", type=Path, default=Path("data/reports/v1_seed_round1/keep_samples.jsonl"))

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
    logs.add_argument("--max-jobs", type=int, help="Legacy cap on candidate jobs considered per repo.")
    logs.add_argument("--max-new-jobs", type=int, default=25, help="Maximum newly downloaded job logs per repo; existing logs do not consume this budget.")
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

    repomap = subparsers.add_parser("eval-repomap", help="Run an Aider-style RepoMap vectorless retrieval baseline.")
    repomap.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.")
    repomap.add_argument("--derived", type=Path, default=Path("data/benchmark/v1"))
    repomap.add_argument("--corpus", type=Path, default=Path("data/corpus/v1"))
    repomap.add_argument("--out", type=Path)
    repomap.add_argument("--details", type=Path)
    repomap.add_argument("--keep-list", type=Path, default=Path("data/reports/v1/keep_samples.jsonl"))
    repomap.add_argument("--no-keep-list", action="store_true")
    repomap.add_argument("--limit-samples", type=int)
    repomap.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    repomap.add_argument("--query-weight", type=float, default=0.65)
    repomap.add_argument("--pagerank-weight", type=float, default=0.25)
    repomap.add_argument("--affinity-weight", type=float, default=0.10)
    repomap.add_argument("--max-symbol-refs-per-file", type=int, default=80)
    repomap.add_argument("--no-progress", action="store_true", help="Disable RepoMap evaluation progress output.")

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

    voyage = subparsers.add_parser("eval-voyage", help="Run a Voyage API embedding retrieval baseline.")
    voyage.add_argument(
        "samples",
        nargs="*",
        type=Path,
        help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.",
    )
    voyage.add_argument("--derived", type=Path, default=Path("data/benchmark/v1"))
    voyage.add_argument("--corpus", type=Path, default=Path("data/corpus/v1"))
    voyage.add_argument("--model", default="voyage-code-3")
    voyage.add_argument("--out", type=Path)
    voyage.add_argument("--details", type=Path)
    voyage.add_argument("--cache", type=Path)
    voyage.add_argument("--keep-list", type=Path, default=Path("data/reports/v1/keep_samples.jsonl"))
    voyage.add_argument("--no-keep-list", action="store_true")
    voyage.add_argument("--limit-samples", type=int)
    voyage.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    voyage.add_argument("--batch-size", type=int, default=32)
    voyage.add_argument("--api-key", help="Voyage API key. Defaults to VOYAGE_API_KEY.")
    voyage.add_argument("--api-base", default="https://api.voyageai.com/v1")
    voyage.add_argument("--query-input-type", default="query")
    voyage.add_argument("--passage-input-type", default="document")
    voyage.add_argument("--output-dimension", type=int, choices=[256, 512, 1024, 2048])
    voyage.add_argument(
        "--output-dtype",
        default="float",
        choices=["float", "int8", "uint8"],
    )
    voyage.add_argument("--no-truncation", action="store_true")
    voyage.add_argument("--no-normalize", action="store_true", help="Disable local L2 normalization.")
    voyage.add_argument("--timeout-seconds", type=float, default=60.0)
    voyage.add_argument("--max-retries", type=int, default=5)
    voyage.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=0.0,
        help="Sleep between Voyage requests. Use about 21 seconds for unpaid 3 RPM accounts.",
    )
    voyage.add_argument("--no-progress", action="store_true", help="Disable embedding evaluation progress output.")

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
    hard_pool_filter.add_argument("--exclude-audited", action="store_true", help="Drop any sample already present in the audit file.")
    hard_pool_filter.add_argument(
        "--task-priority",
        default="",
        help="Comma-separated task ordering for selected/audit rows, e.g. code2test,trace2code,comment2context.",
    )

    report_seed = subparsers.add_parser("report-v1-seed", help="Compare a curated V1 seed against V0.2 and audit outcomes.")
    report_seed.add_argument("--base-samples", type=Path, default=Path("data/benchmark/v0_2/samples.jsonl"))
    report_seed.add_argument("--base-eval", type=Path, default=Path("data/eval/v0_2/lexical_summary.json"))
    report_seed.add_argument("--seed-samples", type=Path, default=Path("data/benchmark/v1_seed_round1/samples.jsonl"))
    report_seed.add_argument("--seed-eval", type=Path, default=Path("data/eval/v1_seed_round1/lexical_summary.json"))
    report_seed.add_argument("--audit-summary", type=Path, default=Path("data/reports/v1_candidate_round1/v1_seed_audit_summary.json"))
    report_seed.add_argument("--out", type=Path, default=Path("data/reports/v1_candidate_round1/v1_seed_comparison.md"))
    report_seed.add_argument("--json-out", type=Path, default=Path("data/reports/v1_candidate_round1/v1_seed_comparison.json"))

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
    if args.command == "download-benchmark":
        result = download_benchmark_release(
            version=args.version,
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            hf_token=args.hf_token,
            skip_download=args.skip_download,
            no_extract=args.no_extract,
            force=args.force,
            hf_bin=args.hf_bin,
            zstd_bin=args.zstd_bin,
            tar_bin=args.tar_bin,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
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
    if args.command == "crawl-pr-checks":
        summaries = [
            crawl_pr_checks(
                api,
                repo,
                args.raw,
                limit_prs=args.limit_prs,
                page_size=args.page_size,
                max_changed_files=args.max_changed_files,
                include_review_comments=args.include_review_comments,
                refresh_existing_checks=args.refresh_existing_checks,
                repair_empty_state=args.repair_empty_state,
                max_pages=args.max_pages,
                dry_run=args.dry_run,
            )
            for repo in args.repo
        ]
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return 1 if any(item["errors"] for item in summaries) else 0
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
    if args.command == "mine-code2test-prs":
        result = mine_code2test_prs(
            raw_dir=args.raw,
            out_dir=args.out,
            report_dir=args.report_out,
            audit_path=args.audit,
            audited_pool_path=args.audited_pool,
            corpus_manifest=args.corpus_manifest,
            require_corpus=args.require_corpus,
            require_gold_in_corpus=args.require_gold_in_corpus,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
            max_tests=args.max_tests,
            audit_limit=args.audit_limit,
            limit_samples=args.limit_samples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "trace-preflight":
        result = trace_preflight(
            raw_dir=args.raw,
            out_dir=args.out,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "mine-trace2code":
        result = mine_trace2code(
            raw_dir=args.raw,
            out_dir=args.out,
            report_dir=args.report_out,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
            audit_limit=args.audit_limit,
            include_review_comments=not args.no_review_comments,
            limit_samples=args.limit_samples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "trace-debug-drops":
        result = trace_debug_drops(
            raw_dir=args.raw,
            out_dir=args.out,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
            audit_limit=args.audit_limit,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "trace-debug-summary":
        result = trace_debug_summary(args.audit, out_path=args.out, recoverable_out=args.recoverable_out)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["pending"] or result["invalid_verdicts"] else 0
    if args.command == "trace-source-scan":
        result = trace_source_scan(
            raw_dir=args.raw,
            out_dir=args.out,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
            audit_limit=args.audit_limit,
            min_score=args.min_score,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "trace-repro-source":
        result = trace_repro_source(
            raw_dir=args.raw,
            out_dir=args.out,
            repos=args.repo,
            max_changed_files=args.max_changed_files,
            max_source_files=args.max_source_files,
            max_test_files=args.max_test_files,
            min_score=args.min_score,
            audit_limit=args.audit_limit,
            limit_candidates=args.limit_candidates,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run-trace-repro":
        result = run_trace_repro(
            candidate_path=args.candidate,
            raw_dir=args.raw,
            repos_dir=args.repos_dir,
            out_dir=args.out,
            candidate_ids=args.candidate_id,
            limit=args.limit,
            timeout_seconds=args.timeout_seconds,
            repo_url_template=args.repo_url_template,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["status_counts"].get("checkout_failed") or result["status_counts"].get("patch_failed") else 0
    if args.command == "mine-trace-repro-runs":
        result = mine_trace_repro_runs(
            candidates_path=args.candidates,
            runs_path=args.runs,
            out_dir=args.out,
            report_dir=args.report_out,
            max_root_files=args.max_root_files,
            audit_limit=args.audit_limit,
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
    if args.command == "merge-seed-audits":
        result = merge_seed_audits(args.audit, out_path=args.out, keep_list_path=args.keep_list)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["invalid_verdicts"] or result["conflicts"] else 0
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
            crawl_job_logs(
                api,
                args.raw,
                repo,
                max_jobs=args.max_jobs,
                max_new_jobs=args.max_new_jobs,
                max_bytes=args.max_bytes,
                conclusions=conclusions,
            )
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
    if args.command == "eval-repomap":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_repomap_summary_path(args.candidate_filter)
        result = evaluate_repomap_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            candidate_filter=args.candidate_filter,
            query_weight=args.query_weight,
            pagerank_weight=args.pagerank_weight,
            affinity_weight=args.affinity_weight,
            max_symbol_refs_per_file=args.max_symbol_refs_per_file,
            progress=not args.no_progress,
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
    if args.command == "eval-voyage":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_embedding_summary_path(
            args.model,
            root=Path("data/eval/v1"),
            candidate_filter=args.candidate_filter,
        )
        embedder = VoyageAPIEmbedder(
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            output_dimension=args.output_dimension,
            output_dtype=args.output_dtype,
            truncation=not args.no_truncation,
            normalize_embeddings=not args.no_normalize,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            min_request_interval_seconds=args.min_request_interval_seconds,
        )
        result = evaluate_embedding_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            model_name=args.model,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            cache_dir=args.cache or default_embedding_cache_dir(args.model, root=Path("data/embeddings/v1")),
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            query_input_type=args.query_input_type or None,
            passage_input_type=args.passage_input_type or None,
            normalize_embeddings=not args.no_normalize,
            embedder=embedder,
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
        task_priority = [task.strip() for task in args.task_priority.split(",") if task.strip()]
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
            exclude_audited=args.exclude_audited,
            task_priority=task_priority or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-v1-seed":
        result = report_v1_seed(
            base_samples_path=args.base_samples,
            base_eval_path=args.base_eval,
            seed_samples_path=args.seed_samples,
            seed_eval_path=args.seed_eval,
            audit_summary_path=args.audit_summary,
            out_path=args.out,
            json_out_path=args.json_out,
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


def default_repomap_summary_path(candidate_filter: str = "all_files", root: Path = Path("data/eval/v0")) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"repomap{suffix}.json"


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
