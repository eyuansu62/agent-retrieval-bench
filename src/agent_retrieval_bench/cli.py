from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .agentic_eval import evaluate_agentic_search
from .agentic_relevance import report_agentic_relevance_taxonomy
from .abstention import (
    apply_abstention_audit_verdicts,
    backfill_abstention_issue_base_commits,
    crawl_abstention_issue_html,
    crawl_abstention_issues,
    export_abstention_clean,
    finalize_abstention_pilot,
    merge_abstention_audit_packets,
    mine_abstention_counterfactuals,
    mine_abstention_organic_candidates,
    prepare_abstention_audit_worklist,
    report_abstention_completion_audit,
    report_abstention_audit,
    report_abstention_crawling_status,
    report_abstention_pilot,
    report_abstention_shard_progress,
    render_abstention_review_packets,
    shard_abstention_audit_worklist,
    write_abstention_audit_handoff_manifest,
)
from .audit import summarize_audit, write_audit_sample
from .baseline import CANDIDATE_FILTERS, RANKERS, evaluate_corpus_baseline
from .bcy_curve import report_bcy_budget_curve
from .cae_validity import report_cae_validity
from .closed_tool_eval import (
    evaluate_closed_tool_codex,
    evaluate_closed_tool_grep,
    evaluate_closed_tool_openai,
    report_closed_tool_budget_curve,
    report_closed_tool_seed_intervention,
)
from .clone import verify_base_commits
from .code2test_pr import mine_code2test_prs
from .corpus import build_candidate_corpus, sample_paths_from_derived
from .context_cost import report_context_acquisition_cost
from .curate import export_curated_samples
from .crawler import crawl_commit_details_for_raw, crawl_pr_checks, crawl_repo, crawl_review_comment_prs, write_manifest
from .derive import derive_repo, diagnose_comment2context_repo
from .dataset_validity import report_dataset_validity
from .diagnostics import diagnose_benchmark
from .edit2ripple import (
    mine_edit2ripple,
    mine_edit2ripple_from_sample_commits,
    mine_edit2ripple_from_samples,
    report_edit2ripple_pilot,
)
from .embedding_eval import (
    VoyageAPIEmbedder,
    default_embedding_cache_dir,
    default_embedding_summary_path,
    evaluate_embedding_baseline,
    load_sample_id_file,
)
from .github_api import GitHubAPI
from .git_raw import backfill_git_raw
from .grep_eval import GREP_PATTERN_MODES, evaluate_grep_baseline
from .hardmine import DEFAULT_HARDMINE_SOURCES, DEFAULT_HARDMINE_TASKS, export_hardmine_candidates
from .hardness import diagnose_hardness, filter_hard_pool, merge_seed_audits, summarize_seed_audit
from .io import load_targets, read_jsonl, repo_slug
from .layered_leaderboard import report_layered_leaderboard
from .logs import crawl_job_logs
from .model_report import report_model_leaderboard
from .openai_context_agent import run_openai_context_agent
from .pes_calibration import report_pes_calibration
from .quality import validate_samples
from .rank_fusion import report_rank_fusion
from .rank_analysis import report_rank_analysis
from .release import (
    CURRENT_BENCHMARK_RELEASES,
    DEFAULT_DATASET_REPO,
    download_benchmark_release,
    merge_corpus_manifests,
    release_catalog,
)
from .repomap_eval import evaluate_repomap_baseline
from .seed_report import report_v1_seed
from .selective_cv import DEFAULT_SELECTIVE_DETAILS, report_selective_group_cv
from .selective_embedding_eval import evaluate_selective_embedding_baseline
from .selective_eval import evaluate_selective_baseline
from .trace_preflight import mine_trace2code, trace_debug_drops, trace_debug_summary, trace_preflight, trace_source_scan
from .trace_repro import mine_trace_repro_runs, run_trace_repro, trace_repro_source
from .trajectory import evaluate_trajectories
from .trajectory_compare import evaluate_ranked_context_as_trajectory
from .v2_positive_report import report_v2_positive_leaderboard
from .trajectory_collect import prepare_trajectory_runs, record_trajectory_step
from .trajectory_release import audit_strict_context_run, package_trajectory_release, v1_3_openai_strict_context_include_paths
from .v1_1 import (
    assemble_v1_1_benchmark,
    check_required_baseline_summaries,
    check_v1_1_readiness,
    create_v1_1_baseline_transfer_bundle,
    report_v1_1_completion_audit,
    report_v1_1_release,
    verify_v1_1_baseline_return_bundle,
    verify_v1_1_baseline_transfer_bundle,
    verify_v1_1_baseline_handoff,
    verify_v1_1_baseline_transfer_manifest,
    write_v1_1_baseline_apply_return_bundle_script,
    write_v1_1_baseline_return_bundle_script,
    write_v1_1_baseline_return_acceptance,
    write_v1_1_baseline_return_manifest,
    write_v1_1_baseline_run_script,
    write_v1_1_baseline_shard_commands,
    write_v1_1_baseline_transfer_unpack_script,
    write_v1_1_baseline_transfer_manifest,
    write_v1_1_baseline_finalization,
    write_v1_1_baseline_handoff,
    write_v1_1_external_runner_copy_packet,
    write_v1_1_external_runner_failfast_smoke_report,
    write_v1_1_external_runner_preflight_report,
    write_v1_1_merged_details,
    write_v1_1_sample_id_shards,
    write_v1_1_summary_from_details,
    write_v1_1_baseline_status_report,
    write_v1_1_audit_packet,
)
from .v1_2 import (
    merge_manual_annotations,
    report_context_pollution,
    report_runtime_cache,
    report_span_subset,
    validate_v1_2_benchmark,
)
from .v1_3 import (
    derive_v1_3_blocks,
    validate_v1_3_benchmark,
    write_v1_3_span_worklist,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arb", description="Agent Retrieval Bench evaluation toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--token", help="GitHub token. Defaults to GITHUB_TOKEN or GH_TOKEN.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/http"), help="ETag response cache directory.")
    parser.add_argument(
        "--max-rate-limit-sleep",
        type=float,
        default=60.0,
        help="Maximum seconds to sleep for a GitHub API rate-limit retry before failing fast.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Write repo_manifest.jsonl for configured targets.")
    manifest.add_argument("--targets", type=Path, default=Path("configs/crawl_targets.json"))
    manifest.add_argument("--output", type=Path, default=Path("data/repo_manifest.jsonl"))

    releases = subparsers.add_parser("releases", help="List current benchmark and auxiliary release IDs.")
    releases.add_argument("--json", action="store_true", help="Print the release catalog as JSON.")

    download_release = subparsers.add_parser(
        "download-benchmark",
        help="Download, verify, and extract one release or all five current benchmark subsets.",
    )
    download_selector = download_release.add_mutually_exclusive_group(required=True)
    download_selector.add_argument(
        "--version",
        help="Release ID to download. Run `arb releases` to list current IDs; historical IDs remain supported.",
    )
    download_selector.add_argument(
        "--all",
        dest="all_current",
        action="store_true",
        help="Download all five current benchmark subset releases.",
    )
    download_release.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    download_release.add_argument("--revision")
    download_release.add_argument("--local-dir", type=Path, default=Path("data"))
    download_release.add_argument("--hf-token", help="Hugging Face token. Defaults to HF_TOKEN or stored hf auth.")
    download_release.add_argument("--skip-download", action="store_true", help="Use an already downloaded release bundle.")
    download_release.add_argument("--no-extract", action="store_true", help="Only download and verify the archive.")
    download_release.add_argument("--force", action="store_true", help="Replace existing benchmark/corpus/eval/report directories for this release id.")
    download_release.add_argument("--hf-bin", default="hf")
    download_release.add_argument("--zstd-bin", default="zstd")
    download_release.add_argument("--tar-bin", default="tar")

    merge_corpora = subparsers.add_parser(
        "merge-corpus-manifests",
        help="Deduplicate downloaded subset corpus manifests for selective evaluation.",
    )
    merge_corpora.add_argument(
        "--version",
        action="append",
        dest="versions",
        help="Release ID to include. Can be repeated; defaults to all five current subsets.",
    )
    merge_corpora.add_argument("--local-dir", type=Path, default=Path("data"))
    merge_corpora.add_argument(
        "--out",
        type=Path,
        help="Output manifest. Defaults to data/corpus/v2_selective_mixed/corpus_manifest.jsonl under --local-dir.",
    )

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

    crawl_review_comments = subparsers.add_parser(
        "crawl-review-comments",
        help="Crawl promising PRs by starting from repository-wide review comments for comment2context mining.",
    )
    crawl_review_comments.add_argument("--repo", action="append", required=True, help="Repo to process. Can be repeated.")
    crawl_review_comments.add_argument("--out", type=Path, default=Path("data/raw_review_comments"))
    crawl_review_comments.add_argument("--comments-per-repo", type=int, default=100)
    crawl_review_comments.add_argument("--limit-prs", type=int, default=20)
    crawl_review_comments.add_argument("--max-prs-per-repo", type=int, default=4)
    crawl_review_comments.add_argument("--max-changed-files", type=int, default=45)
    crawl_review_comments.add_argument("--max-detail-commits", type=int, default=8)
    crawl_review_comments.add_argument("--response-window-hours", type=int, default=72)
    crawl_review_comments.add_argument("--include-bots", action="store_true")
    crawl_review_comments.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write selected PR metadata and review comments only; use backfill-git-raw for files and commits.",
    )
    crawl_review_comments.add_argument("--dry-run", action="store_true")

    derive = subparsers.add_parser("derive", help="Build weak benchmark samples from raw JSONL.")
    derive.add_argument("--raw", type=Path, default=Path("data/raw"))
    derive.add_argument("--out", type=Path, default=Path("data/derived"))
    derive.add_argument("--repo", action="append", help="Repo to derive. Can be repeated. Defaults to raw dirs.")
    derive.add_argument("--max-changed-files", type=int, default=20)

    comment_diag = subparsers.add_parser("diagnose-comment2context", help="Explain why raw review comments do or do not derive comment2context candidates.")
    comment_diag.add_argument("--raw", type=Path, default=Path("data/raw"))
    comment_diag.add_argument("--out", type=Path, default=Path("data/reports/comment2context_diagnostics"))
    comment_diag.add_argument("--repo", action="append", help="Repo to diagnose. Can be repeated. Defaults to raw dirs.")
    comment_diag.add_argument("--max-changed-files", type=int, default=20)

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

    mine_ripple = subparsers.add_parser("mine-edit2ripple", help="Mine edit2ripple co-change candidates from raw PR files.")
    mine_ripple.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    mine_ripple.add_argument("--out", type=Path, default=Path("data/benchmark/v1_edit2ripple_candidates"))
    mine_ripple.add_argument("--report-out", type=Path, default=Path("data/reports/v1_edit2ripple_candidates"))
    mine_ripple.add_argument("--audit", type=Path, default=Path("data/reports/v1_edit2ripple/audited_ids.csv"))
    mine_ripple.add_argument("--corpus-manifest", type=Path)
    mine_ripple.add_argument("--require-corpus", action="store_true")
    mine_ripple.add_argument("--require-gold-in-corpus", action="store_true")
    mine_ripple.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    mine_ripple.add_argument("--min-changed-files", type=int, default=2)
    mine_ripple.add_argument("--max-changed-files", type=int, default=8)
    mine_ripple.add_argument("--max-gold-files", type=int, default=4)
    mine_ripple.add_argument("--max-candidates-per-pr", type=int, default=2)
    mine_ripple.add_argument("--audit-limit", type=int, default=120)
    mine_ripple.add_argument("--limit-samples", type=int)

    mine_ripple_from_samples = subparsers.add_parser(
        "mine-edit2ripple-from-samples",
        help="Mine edit2ripple candidates by refetching PR URLs from existing benchmark samples.",
    )
    mine_ripple_from_samples.add_argument("samples", nargs="*", type=Path, help="Sample JSONL files. Defaults to --derived/*.jsonl.")
    mine_ripple_from_samples.add_argument("--derived", type=Path, default=Path("data/benchmark/v1"))
    mine_ripple_from_samples.add_argument("--out", type=Path, default=Path("data/benchmark/v1_edit2ripple_from_samples"))
    mine_ripple_from_samples.add_argument("--report-out", type=Path, default=Path("data/reports/v1_edit2ripple_from_samples"))
    mine_ripple_from_samples.add_argument("--audit", type=Path, default=Path("data/reports/v1_edit2ripple/audited_ids.csv"))
    mine_ripple_from_samples.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1/corpus_manifest.jsonl"))
    mine_ripple_from_samples.add_argument("--require-corpus", action="store_true")
    mine_ripple_from_samples.add_argument("--require-gold-in-corpus", action="store_true")
    mine_ripple_from_samples.add_argument("--repo", action="append", help="Repo to process. Defaults to all repos in sample paths.")
    mine_ripple_from_samples.add_argument("--max-gold-files", type=int, default=4)
    mine_ripple_from_samples.add_argument("--max-candidates-per-pr", type=int, default=2)
    mine_ripple_from_samples.add_argument("--audit-limit", type=int, default=120)
    mine_ripple_from_samples.add_argument("--limit-prs", type=int)
    mine_ripple_from_samples.add_argument("--limit-samples", type=int)
    mine_ripple_from_samples.add_argument("--fail-fast", action="store_true")

    mine_ripple_from_commits = subparsers.add_parser(
        "mine-edit2ripple-from-sample-commits",
        help="Mine edit2ripple candidates from existing samples by diffing base and fix commits.",
    )
    mine_ripple_from_commits.add_argument("samples", nargs="*", type=Path, help="Sample JSONL files. Defaults to --derived/*.jsonl.")
    mine_ripple_from_commits.add_argument("--derived", type=Path, default=Path("data/benchmark/v1"))
    mine_ripple_from_commits.add_argument("--out", type=Path, default=Path("data/benchmark/v1_edit2ripple_from_commits"))
    mine_ripple_from_commits.add_argument("--report-out", type=Path, default=Path("data/reports/v1_edit2ripple_from_commits"))
    mine_ripple_from_commits.add_argument("--repos-dir", type=Path, default=Path("data/repos_edit2ripple"))
    mine_ripple_from_commits.add_argument("--audit", type=Path, default=Path("data/reports/v1_edit2ripple/audited_ids.csv"))
    mine_ripple_from_commits.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1/corpus_manifest.jsonl"))
    mine_ripple_from_commits.add_argument("--require-corpus", action="store_true")
    mine_ripple_from_commits.add_argument("--require-gold-in-corpus", action="store_true")
    mine_ripple_from_commits.add_argument("--repo", action="append", help="Repo to process. Defaults to all repos in sample paths.")
    mine_ripple_from_commits.add_argument("--remote-base", default="https://github.com")
    mine_ripple_from_commits.add_argument("--no-blob-filter", action="store_true")
    mine_ripple_from_commits.add_argument("--max-gold-files", type=int, default=4)
    mine_ripple_from_commits.add_argument("--max-candidates-per-pr", type=int, default=2)
    mine_ripple_from_commits.add_argument("--audit-limit", type=int, default=120)
    mine_ripple_from_commits.add_argument("--limit-commits", type=int)
    mine_ripple_from_commits.add_argument("--limit-samples", type=int)
    mine_ripple_from_commits.add_argument("--fail-fast", action="store_true")

    report_ripple_pilot = subparsers.add_parser(
        "report-edit2ripple-pilot",
        help="Validate edit2ripple pilot gates and audit readiness.",
    )
    report_ripple_pilot.add_argument("samples", nargs="*", type=Path, help="Sample JSONL files. Defaults to --derived/*.jsonl.")
    report_ripple_pilot.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_edit2ripple_candidates"))
    report_ripple_pilot.add_argument("--corpus-manifest", type=Path)
    report_ripple_pilot.add_argument("--audit", type=Path)
    report_ripple_pilot.add_argument("--out", type=Path, default=Path("data/reports/v1_edit2ripple_candidates/pilot_report.md"))
    report_ripple_pilot.add_argument("--json-out", type=Path, default=Path("data/reports/v1_edit2ripple_candidates/pilot_report.json"))
    report_ripple_pilot.add_argument("--min-samples", type=int, default=50)
    report_ripple_pilot.add_argument("--min-valid-rate", type=float, default=0.90)
    report_ripple_pilot.add_argument("--max-test-only-ratio", type=float, default=0.35)
    report_ripple_pilot.add_argument("--max-candidates-per-pr", type=int, default=2)

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

    abstention_cf = subparsers.add_parser(
        "mine-abstention-counterfactuals",
        help="Build abstention counterfactual wrong-repo candidates from positive ARB samples.",
    )
    abstention_cf.add_argument("samples", nargs="*", type=Path, help="Positive sample JSONL files. Defaults to --derived/*.jsonl.")
    abstention_cf.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    abstention_cf.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    abstention_cf.add_argument("--out", type=Path, default=Path("data/abstention/candidates"))
    abstention_cf.add_argument("--audit-out", type=Path, default=Path("data/abstention/audit"))
    abstention_cf.add_argument("--limit", type=int, default=80)
    abstention_cf.add_argument("--max-per-wrong-repo", type=int, default=8)
    abstention_cf.add_argument("--max-per-source-repo", type=int, default=8)
    abstention_cf.add_argument("--pairs-per-sample", type=int, default=1)

    abstention_organic = subparsers.add_parser(
        "mine-abstention-organic",
        help="Mine conservative organic no-gold candidates from raw CI/check data.",
    )
    abstention_organic.add_argument("--raw", action="append", type=Path, default=[], help="Raw data dir. Can be repeated.")
    abstention_organic.add_argument("--out", type=Path, default=Path("data/abstention/candidates"))
    abstention_organic.add_argument("--audit-out", type=Path, default=Path("data/abstention/audit"))
    abstention_organic.add_argument("--repo", action="append", help="Repo to process. Defaults to raw dirs.")
    abstention_organic.add_argument("--limit", type=int)

    abstention_issue_crawl = subparsers.add_parser(
        "crawl-abstention-issues",
        help="Crawl closed issues/comments/events for organic abstention mining.",
    )
    abstention_issue_crawl.add_argument("--repo", action="append", required=True, help="Repo to crawl. Can be repeated.")
    abstention_issue_crawl.add_argument("--out", type=Path, default=Path("data/abstention/raw"))
    abstention_issue_crawl.add_argument("--limit-per-repo", type=int, default=60)
    abstention_issue_crawl.add_argument("--comments-per-issue", type=int, default=100)
    abstention_issue_crawl.add_argument("--events-per-issue", type=int, default=100)

    abstention_issue_html_crawl = subparsers.add_parser(
        "crawl-abstention-issue-html",
        help="Crawl closed issue pages through GitHub HTML for organic abstention mining.",
    )
    abstention_issue_html_crawl.add_argument("--repo", action="append", required=True, help="Repo to crawl. Can be repeated.")
    abstention_issue_html_crawl.add_argument("--out", type=Path, default=Path("data/abstention/raw_html"))
    abstention_issue_html_crawl.add_argument("--git-repos-dir", type=Path, default=Path("data/git_raw_repos"))
    abstention_issue_html_crawl.add_argument("--limit-per-repo", type=int, default=60)
    abstention_issue_html_crawl.add_argument("--pages-per-keyword", type=int, default=2)
    abstention_issue_html_crawl.add_argument("--request-delay-seconds", type=float, default=0.4)

    abstention_base_backfill = subparsers.add_parser(
        "backfill-abstention-issue-bases",
        help="Backfill missing abstention issue base commits from local git history or GitHub commits API.",
    )
    abstention_base_backfill.add_argument("--raw", action="append", type=Path, default=[], help="Raw issue dir. Can be repeated.")
    abstention_base_backfill.add_argument("--git-repos-dir", type=Path, default=Path("data/git_raw_repos"))
    abstention_base_backfill.add_argument(
        "--api-fallback",
        action="store_true",
        help="Use GitHub REST commits API when local git history cannot answer.",
    )
    abstention_base_backfill.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/abstention/audit/abstention_issue_base_backfill_report.json"),
    )

    abstention_merge = subparsers.add_parser(
        "merge-abstention-audit-packets",
        help="Merge organic and counterfactual abstention audit packets into one manual-audit packet.",
    )
    abstention_merge.add_argument(
        "--packet",
        action="append",
        type=Path,
        help="Audit packet JSONL to merge. Can be repeated.",
    )
    abstention_merge.add_argument("--out", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_merge.add_argument("--csv-out", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.csv"))
    abstention_merge.add_argument("--report-out", type=Path, default=Path("data/abstention/audit/abstention_audit_packet_merge_report.json"))

    abstention_crawling_report = subparsers.add_parser(
        "report-abstention-crawling",
        help="Validate abstention crawling-stage gates before human audit.",
    )
    abstention_crawling_report.add_argument(
        "--prefiltered",
        action="append",
        type=Path,
        help="Prefiltered candidate JSONL. Defaults to current organic and counterfactual candidate files.",
    )
    abstention_crawling_report.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_crawling_report.add_argument("--out", type=Path, default=Path("data/abstention/audit/abstention_crawling_status.json"))

    abstention_export = subparsers.add_parser(
        "export-abstention-clean",
        help="Export valid_no_gold abstention samples from a completed audit packet.",
    )
    abstention_export.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_export.add_argument("--audit", required=True, type=Path)
    abstention_export.add_argument("--out", type=Path, default=Path("data/benchmark/v2_abstention_pilot"))
    abstention_export.add_argument("--report-out", type=Path, default=Path("data/reports/v2_abstention_pilot/pilot_report.json"))
    abstention_export.add_argument("--max-samples", type=int)
    abstention_export.add_argument("--max-counterfactual-share", type=float)

    abstention_audit_report = subparsers.add_parser(
        "report-abstention-audit",
        help="Summarize abstention manual audit verdicts before clean export.",
    )
    abstention_audit_report.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_audit_report.add_argument("--audit", required=True, type=Path)
    abstention_audit_report.add_argument("--out", type=Path, default=Path("data/abstention/audit/abstention_audit_verdict_report.json"))
    abstention_audit_report.add_argument("--max-samples", type=int, default=100)
    abstention_audit_report.add_argument("--max-counterfactual-share", type=float, default=0.5)

    abstention_worklist = subparsers.add_parser(
        "prepare-abstention-audit-worklist",
        help="Write a balanced manual-audit worklist from abstention audit packets without assigning verdicts.",
    )
    abstention_worklist.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_worklist.add_argument("--out-csv", type=Path, default=Path("data/abstention/audit/abstention_manual_audit_worklist.csv"))
    abstention_worklist.add_argument("--out-jsonl", type=Path, default=Path("data/abstention/audit/abstention_manual_audit_worklist.jsonl"))
    abstention_worklist.add_argument("--report-out", type=Path, default=Path("data/abstention/audit/abstention_manual_audit_worklist_report.json"))
    abstention_worklist.add_argument("--target-size", type=int, default=100)
    abstention_worklist.add_argument("--max-counterfactual-share", type=float, default=0.5)

    abstention_completion = subparsers.add_parser(
        "report-abstention-completion",
        help="Audit current abstention artifacts against the v2 crawling plan success criteria.",
    )
    abstention_completion.add_argument(
        "--prefiltered",
        action="append",
        type=Path,
        help="Prefiltered candidate JSONL. Defaults to current organic and counterfactual candidate files.",
    )
    abstention_completion.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_completion.add_argument("--audit", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.csv"))
    abstention_completion.add_argument(
        "--clean-sample",
        action="append",
        type=Path,
        help="Final clean abstention JSONL. Defaults to data/benchmark/v2_abstention_pilot/abstention.jsonl if present.",
    )
    abstention_completion.add_argument(
        "--worklist-report",
        type=Path,
        default=Path("data/abstention/audit/abstention_manual_audit_worklist_report.json"),
    )
    abstention_completion.add_argument(
        "--crawling-report",
        type=Path,
        help="Optional crawling-stage status JSON to use as evidence for prefiltered candidate gates.",
    )
    abstention_completion.add_argument("--out", type=Path, default=Path("data/abstention/audit/abstention_completion_audit.json"))
    abstention_completion.add_argument("--max-samples", type=int, default=100)
    abstention_completion.add_argument("--max-counterfactual-share", type=float, default=0.5)

    abstention_apply_verdicts = subparsers.add_parser(
        "apply-abstention-audit-verdicts",
        help="Apply filled manual audit verdicts to the canonical abstention audit CSV.",
    )
    abstention_apply_verdicts.add_argument(
        "--source-audit",
        action="append",
        type=Path,
        help="Filled manual audit CSV/JSONL, such as a reviewer shard. Can be repeated. Defaults to the full worklist CSV.",
    )
    abstention_apply_verdicts.add_argument(
        "--target-audit",
        type=Path,
        default=Path("data/abstention/audit/abstention_audit_packet.csv"),
        help="Canonical audit CSV to update.",
    )
    abstention_apply_verdicts.add_argument("--out", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.csv"))
    abstention_apply_verdicts.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/abstention/audit/abstention_audit_verdict_apply_report.json"),
    )
    abstention_apply_verdicts.add_argument("--overwrite", action="store_true", help="Overwrite existing nonblank target verdicts.")
    abstention_apply_verdicts.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail the report gate unless every target row has a verdict after apply.",
    )

    abstention_finalize = subparsers.add_parser(
        "finalize-abstention-pilot",
        help="Export and validate the abstention pilot only after manual-audit gates pass.",
    )
    abstention_finalize.add_argument(
        "--prefiltered",
        action="append",
        type=Path,
        help="Prefiltered candidate JSONL. Defaults to current organic and counterfactual candidate files.",
    )
    abstention_finalize.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_finalize.add_argument("--audit", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.csv"))
    abstention_finalize.add_argument("--out", type=Path, default=Path("data/benchmark/v2_abstention_pilot"))
    abstention_finalize.add_argument("--report-out", type=Path, default=Path("data/reports/v2_abstention_pilot/finalization_report.json"))
    abstention_finalize.add_argument("--audit-report-out", type=Path, default=Path("data/abstention/audit/abstention_audit_verdict_report.json"))
    abstention_finalize.add_argument("--pilot-report-out", type=Path, default=Path("data/reports/v2_abstention_pilot/pilot_report.json"))
    abstention_finalize.add_argument("--completion-report-out", type=Path, default=Path("data/abstention/audit/abstention_completion_audit.json"))
    abstention_finalize.add_argument(
        "--worklist-report",
        type=Path,
        default=Path("data/abstention/audit/abstention_manual_audit_worklist_report.json"),
    )
    abstention_finalize.add_argument(
        "--crawling-report",
        type=Path,
        help="Optional crawling-stage status JSON to use as evidence for prefiltered candidate gates.",
    )
    abstention_finalize.add_argument("--max-samples", type=int, default=100)
    abstention_finalize.add_argument("--max-counterfactual-share", type=float, default=0.5)

    abstention_shard_worklist = subparsers.add_parser(
        "shard-abstention-audit-worklist",
        help="Split the abstention manual audit worklist into reviewer-sized CSV/JSONL shards.",
    )
    abstention_shard_worklist.add_argument(
        "--worklist",
        type=Path,
        default=Path("data/abstention/audit/abstention_manual_audit_worklist.csv"),
    )
    abstention_shard_worklist.add_argument("--out-dir", type=Path, default=Path("data/abstention/audit/shards"))
    abstention_shard_worklist.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/abstention/audit/abstention_manual_audit_shards_report.json"),
    )
    abstention_shard_worklist.add_argument("--shard-size", type=int, default=25)
    abstention_shard_worklist.add_argument("--priority", choices=["core", "reserve", "all"], default="core")

    abstention_shard_progress = subparsers.add_parser(
        "report-abstention-shard-progress",
        help="Summarize manual verdict progress across abstention reviewer shards.",
    )
    abstention_shard_progress.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="Reviewer shard CSV/JSONL. Can be repeated. Defaults to data/abstention/audit/shards/*.csv.",
    )
    abstention_shard_progress.add_argument(
        "--out",
        type=Path,
        default=Path("data/abstention/audit/abstention_shard_progress_report.json"),
    )

    abstention_review_packets = subparsers.add_parser(
        "render-abstention-review-packets",
        help="Render reviewer-friendly Markdown packets from abstention audit shard CSV/JSONL files.",
    )
    abstention_review_packets.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="Reviewer shard CSV/JSONL. Can be repeated. Defaults to data/abstention/audit/shards/*.csv.",
    )
    abstention_review_packets.add_argument("--out-dir", type=Path, default=Path("data/abstention/audit/review_packets"))
    abstention_review_packets.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/abstention/audit/abstention_review_packets_report.json"),
    )
    abstention_review_packets.add_argument("--query-limit", type=int, default=1800)
    abstention_review_packets.add_argument("--evidence-limit", type=int, default=1200)

    abstention_handoff_manifest = subparsers.add_parser(
        "write-abstention-audit-handoff-manifest",
        help="Write a checksumed manifest for abstention manual-audit handoff files.",
    )
    abstention_handoff_manifest.add_argument("--audit-dir", type=Path, default=Path("data/abstention/audit"))
    abstention_handoff_manifest.add_argument(
        "--out",
        type=Path,
        default=Path("data/abstention/audit/abstention_audit_handoff_manifest.json"),
    )
    abstention_handoff_manifest.add_argument(
        "--finalization-report",
        type=Path,
        default=Path("data/reports/v2_abstention_pilot/finalization_report.json"),
    )

    abstention_report = subparsers.add_parser("report-abstention-pilot", help="Validate abstention pilot gates and schema.")
    abstention_report.add_argument("samples", nargs="*", type=Path, help="Abstention sample JSONL files. Defaults to --derived/*.jsonl.")
    abstention_report.add_argument("--derived", type=Path, default=Path("data/benchmark/v2_abstention_pilot"))
    abstention_report.add_argument("--audit-packet", type=Path, default=Path("data/abstention/audit/abstention_audit_packet.jsonl"))
    abstention_report.add_argument("--out", type=Path, default=Path("data/reports/v2_abstention_pilot/pilot_report.json"))

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

    git_raw = subparsers.add_parser("backfill-git-raw", help="Backfill PR raw file/commit records from git base/head SHAs without GitHub REST.")
    git_raw.add_argument("--raw", type=Path, default=Path("data/raw_token"))
    git_raw.add_argument("--repo", action="append", required=True, help="Repo to process. Can be repeated.")
    git_raw.add_argument("--repos-dir", type=Path, default=Path("data/git_raw_repos"))
    git_raw.add_argument("--limit-prs", type=int)
    git_raw.add_argument("--repo-url-template", default="https://github.com/{repo}.git")
    git_raw.add_argument("--timeout-seconds", type=int, default=600)
    git_raw.add_argument(
        "--infer-missing-base",
        action="store_true",
        help="Infer missing baseRefOid as merge-base(headRefOid, origin/baseRefName).",
    )
    git_raw.add_argument(
        "--max-inferred-commits",
        type=int,
        default=80,
        help="Skip inferred-base PRs whose base..head range contains more than this many commits.",
    )

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

    baseline = subparsers.add_parser("eval-baseline", help="Run a lexical or BM25 retrieval baseline.")
    baseline.add_argument("samples", nargs="*", type=Path, help="Derived sample JSONL files. Defaults to --derived/*.jsonl.")
    baseline.add_argument("--derived", type=Path, default=Path("data/derived"))
    baseline.add_argument("--corpus", type=Path, default=Path("data/corpus/v0"))
    baseline.add_argument("--out", type=Path)
    baseline.add_argument("--details", type=Path)
    baseline.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))
    baseline.add_argument("--no-keep-list", action="store_true")
    baseline.add_argument("--limit-samples", type=int)
    baseline.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    baseline.add_argument("--ranker", choices=RANKERS, default="lexical")
    baseline.add_argument("--dry-run", action="store_true", help="Use sample gold/supporting paths as a tiny synthetic corpus.")
    baseline.add_argument("--no-progress", action="store_true", help="Disable baseline progress output.")

    grep = subparsers.add_parser("eval-grep", help="Run a deterministic grep-style exact-search retrieval baseline.")
    grep.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.")
    grep.add_argument("--derived", type=Path, default=Path("data/benchmark/agent_retrieval_bench"))
    grep.add_argument("--corpus", type=Path, default=Path("data/corpus/agent_retrieval_bench"))
    grep.add_argument("--out", type=Path)
    grep.add_argument("--details", type=Path)
    grep.add_argument("--keep-list", type=Path)
    grep.add_argument("--no-keep-list", action="store_true")
    grep.add_argument("--limit-samples", type=int)
    grep.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    grep.add_argument("--pattern-mode", choices=GREP_PATTERN_MODES, default="strict")

    selective_baseline = subparsers.add_parser(
        "eval-selective-baseline",
        help="Run lexical or BM25 retrieval with an abstain threshold sweep over positive and no-gold samples.",
    )
    selective_baseline.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.")
    selective_baseline.add_argument("--derived", type=Path, default=Path("data/benchmark/v2_selective_retrieval_balanced"))
    selective_baseline.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    selective_baseline.add_argument("--out", type=Path)
    selective_baseline.add_argument("--details", type=Path)
    selective_baseline.add_argument("--sweep", type=Path)
    selective_baseline.add_argument("--report", type=Path)
    selective_baseline.add_argument("--keep-list", type=Path, default=Path("data/audit/v0/keep_samples.jsonl"))
    selective_baseline.add_argument("--no-keep-list", action="store_true")
    selective_baseline.add_argument("--limit-samples", type=int)
    selective_baseline.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    selective_baseline.add_argument("--ranker", choices=RANKERS, default="lexical")
    selective_baseline.add_argument("--no-progress", action="store_true", help="Disable selective baseline progress output.")

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

    agentic = subparsers.add_parser("eval-agentic", help="Run a deterministic search/read agentic retrieval baseline.")
    agentic.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.")
    agentic.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_2"))
    agentic.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    agentic.add_argument("--out", type=Path)
    agentic.add_argument("--details", type=Path)
    agentic.add_argument("--keep-list", type=Path)
    agentic.add_argument("--no-keep-list", action="store_true")
    agentic.add_argument("--limit-samples", type=int)
    agentic.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    agentic.add_argument("--max-turns", type=int, default=4)
    agentic.add_argument("--max-tool-calls-per-turn", type=int, default=8)
    agentic.add_argument("--max-read-chars", type=int, default=8000)
    agentic.add_argument("--no-progress", action="store_true", help="Disable agentic baseline progress output.")

    closed_tool = subparsers.add_parser("eval-closed-tool", help="Run the closed-tool grep/read/submit context-acquisition baseline.")
    closed_tool.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to v1_3_reviewed/samples.jsonl.")
    closed_tool.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    closed_tool.add_argument("--out", type=Path, default=Path("data/eval/v1_4/closed_tool_grep_summary.json"))
    closed_tool.add_argument("--details", type=Path, default=Path("data/eval/v1_4/closed_tool_grep_details.jsonl"))
    closed_tool.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/closed_tool_grep.md"))
    closed_tool.add_argument("--keep-list", type=Path)
    closed_tool.add_argument("--limit-samples", type=int)
    closed_tool.add_argument("--max-tool-calls", type=int, default=16)
    closed_tool.add_argument("--max-read-tokens", type=int, default=8000)
    closed_tool.add_argument("--max-read-tokens-per-file", type=int, default=1200)
    closed_tool.add_argument("--max-grep-calls", type=int, default=8)
    closed_tool.add_argument("--final-k", type=int, default=3)
    closed_tool.add_argument("--grep-top-k", type=int, default=12)
    closed_tool.add_argument("--force", action="store_true", help="Ignore an existing details file instead of resuming completed samples.")

    closed_tool_llm = subparsers.add_parser("eval-closed-tool-llm", help="Run an OpenAI policy under the closed-tool grep/read/submit protocol.")
    closed_tool_llm.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to v1_3_reviewed/samples.jsonl.")
    closed_tool_llm.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    closed_tool_llm.add_argument("--out", type=Path, default=Path("data/eval/v1_4/closed_tool_llm_summary.json"))
    closed_tool_llm.add_argument("--details", type=Path, default=Path("data/eval/v1_4/closed_tool_llm_details.jsonl"))
    closed_tool_llm.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/closed_tool_llm.md"))
    closed_tool_llm.add_argument("--keep-list", type=Path)
    closed_tool_llm.add_argument("--limit-samples", type=int)
    closed_tool_llm.add_argument("--model", default="gpt-5.4-mini")
    closed_tool_llm.add_argument("--max-tool-calls", type=int, default=16)
    closed_tool_llm.add_argument("--max-model-turns", type=int)
    closed_tool_llm.add_argument("--max-read-tokens", type=int, default=8000)
    closed_tool_llm.add_argument("--max-read-tokens-per-file", type=int, default=1200)
    closed_tool_llm.add_argument("--final-k", type=int, default=3)
    closed_tool_llm.add_argument("--grep-top-k", type=int, default=12)
    closed_tool_llm.add_argument("--force", action="store_true", help="Ignore an existing details file instead of resuming completed samples.")

    closed_tool_codex = subparsers.add_parser("eval-closed-tool-codex", help="Run Codex CLI as a JSON action policy under the closed-tool protocol.")
    closed_tool_codex.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to v1_3_reviewed/samples.jsonl.")
    closed_tool_codex.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    closed_tool_codex.add_argument("--out", type=Path, default=Path("data/eval/v1_4/closed_tool_codex_summary.json"))
    closed_tool_codex.add_argument("--details", type=Path, default=Path("data/eval/v1_4/closed_tool_codex_details.jsonl"))
    closed_tool_codex.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/closed_tool_codex.md"))
    closed_tool_codex.add_argument("--keep-list", type=Path)
    closed_tool_codex.add_argument("--limit-samples", type=int)
    closed_tool_codex.add_argument("--model", default="gpt-5.5")
    closed_tool_codex.add_argument("--codex-bin", type=Path, default=Path("/home/ubuntu/.nvm/versions/node/v22.22.0/bin/codex"))
    closed_tool_codex.add_argument("--codex-work-root", type=Path, default=Path("/tmp/arb-codex-closed-tool"))
    closed_tool_codex.add_argument("--codex-timeout-seconds", type=int, default=180)
    closed_tool_codex.add_argument("--max-tool-calls", type=int, default=16)
    closed_tool_codex.add_argument("--max-model-turns", type=int)
    closed_tool_codex.add_argument("--max-read-tokens", type=int, default=8000)
    closed_tool_codex.add_argument("--max-read-tokens-per-file", type=int, default=1200)
    closed_tool_codex.add_argument("--final-k", type=int, default=3)
    closed_tool_codex.add_argument("--grep-top-k", type=int, default=12)
    closed_tool_codex.add_argument("--force", action="store_true", help="Ignore an existing details file instead of resuming completed samples.")
    closed_tool_codex.add_argument("--seed-details", type=Path, help="Optional ranked-detail JSONL whose top_files seed initial context.")
    closed_tool_codex.add_argument("--seed-label", default="", help="Label for the initial context seed source.")
    closed_tool_codex.add_argument("--seed-top-k", type=int, default=3)
    closed_tool_codex.add_argument("--max-seed-tokens", type=int, default=0)
    closed_tool_codex.add_argument("--max-seed-tokens-per-file", type=int, default=1200)

    closed_tool_budget = subparsers.add_parser(
        "report-closed-tool-budget-curve",
        help="Score post-hoc tool-budget prefixes from an existing closed-tool details JSONL.",
    )
    closed_tool_budget.add_argument(
        "--details",
        type=Path,
        default=Path("data/eval/v1_4/closed_tool_codex_container_full_details.jsonl"),
    )
    closed_tool_budget.add_argument("--out", type=Path, default=Path("data/reports/v1_4/closed_tool_budget_curve.json"))
    closed_tool_budget.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/closed_tool_budget_curve.md"))
    closed_tool_budget.add_argument("--budgets", default="2,4,6,8", help="Comma-separated positive tool-call budgets.")

    closed_tool_intervention = subparsers.add_parser(
        "report-closed-tool-seed-intervention",
        help="Compare no-seed and seeded closed-tool runs on their common sample intersection.",
    )
    closed_tool_intervention.add_argument(
        "--control-details",
        type=Path,
        default=Path("data/eval/v1_4/closed_tool_codex_container_full_details.jsonl"),
    )
    closed_tool_intervention.add_argument("--arm", action="append", default=[], help="Intervention arm in label=details.jsonl form. Repeatable.")
    closed_tool_intervention.add_argument("--keep-list", type=Path)
    closed_tool_intervention.add_argument("--out", type=Path, default=Path("data/reports/v1_4/closed_tool_seed_intervention.json"))
    closed_tool_intervention.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/closed_tool_seed_intervention.md"))

    trajectories = subparsers.add_parser("eval-trajectories", help="Evaluate normalized coding-agent context trajectories.")
    trajectories.add_argument("trajectories", nargs="+", type=Path, help="Trajectory JSONL files. Rows may be per-step or per-sample with a trajectory list.")
    trajectories.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    trajectories.add_argument("--out", type=Path, default=Path("data/eval/v1_4/trajectory_summary.json"))
    trajectories.add_argument("--details", type=Path, default=Path("data/eval/v1_4/trajectory_details.jsonl"))
    trajectories.add_argument("--model-label", default="trajectory-log")
    trajectories.add_argument("--supporting-context-annotations", type=Path, help="Optional JSONL overlay with sample_id and supporting_context_files.")

    ranked_context = subparsers.add_parser(
        "eval-ranked-context",
        help="Evaluate top-k ranked baseline files as same-budget final context.",
    )
    ranked_context.add_argument("--baseline-details", required=True, type=Path)
    ranked_context.add_argument("--top-k", type=int, default=3)
    ranked_context.add_argument("--out", type=Path, default=Path("data/eval/v1_4/ranked_context_summary.json"))
    ranked_context.add_argument("--details", type=Path, default=Path("data/eval/v1_4/ranked_context_details.jsonl"))
    ranked_context.add_argument("--model-label")

    run_openai_agent = subparsers.add_parser("run-openai-context-agent", help="Run the OpenAI search/read context agent with strict final context.")
    run_openai_agent.add_argument("--base", type=Path, default=Path("data/trajectory_runs/v1_4/v1_3_all_gpt54mini"))
    run_openai_agent.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3/samples.jsonl"))
    run_openai_agent.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_2"))
    run_openai_agent.add_argument("--model", default="gpt-5.4-mini")
    run_openai_agent.add_argument("--run-name", default="openai_gpt54mini_v2_strict_context")
    run_openai_agent.add_argument("--all-samples", action="store_true")
    run_openai_agent.add_argument("--limit", type=int)
    run_openai_agent.add_argument("--sample-id", action="append", default=[])
    run_openai_agent.add_argument("--force", action="store_true")
    run_openai_agent.add_argument("--max-actions", type=int, default=9)
    run_openai_agent.add_argument("--min-reads", type=int, default=3)

    audit_trajectory_release = subparsers.add_parser(
        "audit-trajectory-release", help="Audit a strict-context trajectory run for release packaging."
    )
    audit_trajectory_release.add_argument("--base", type=Path, default=Path("data/trajectory_runs/v1_4/v1_3_all_gpt54mini"))
    audit_trajectory_release.add_argument("--run-name", default="openai_gpt54mini_v2_strict_context")
    audit_trajectory_release.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    audit_trajectory_release.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    audit_trajectory_release.add_argument(
        "--out", type=Path, default=Path("data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.json")
    )
    audit_trajectory_release.add_argument(
        "--markdown-out", type=Path, default=Path("data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.md")
    )
    audit_trajectory_release.add_argument("--min-reads", type=int, default=3)
    audit_trajectory_release.add_argument("--max-reads", type=int, default=9)
    audit_trajectory_release.add_argument("--extra-scan-path", action="append", type=Path, default=[])

    package_trajectory = subparsers.add_parser("package-trajectory-release", help="Build a tar.zst bundle for a trajectory release.")
    package_trajectory.add_argument("--data-root", type=Path, default=Path("data"))
    package_trajectory.add_argument("--base", type=Path, default=Path("data/trajectory_runs/v1_4/v1_3_all_gpt54mini"))
    package_trajectory.add_argument("--run-name", default="openai_gpt54mini_v2_strict_context")
    package_trajectory.add_argument(
        "--release-dir", type=Path, default=Path("data/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context")
    )
    package_trajectory.add_argument(
        "--archive-name",
        default="agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst",
    )
    package_trajectory.add_argument(
        "--checksum-path-in-release",
        default="releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context",
        help="Path written in the .sha256 file before the archive filename.",
    )
    package_trajectory.add_argument("--include", action="append", type=Path, default=[], help="Optional explicit file to include.")

    prepare_trajectories = subparsers.add_parser("prepare-trajectory-runs", help="Write prompt packets and empty logs for real agent trajectory collection.")
    prepare_trajectories.add_argument("samples", nargs="*", type=Path, help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.")
    prepare_trajectories.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    prepare_trajectories.add_argument("--out-dir", type=Path, default=Path("data/trajectory_runs/v1_4/manual"))
    prepare_trajectories.add_argument("--sample-id", action="append", default=[])
    prepare_trajectories.add_argument("--limit-samples", type=int)
    prepare_trajectories.add_argument("--model-label", default="agent")
    prepare_trajectories.add_argument("--overwrite-logs", action="store_true", help="Reset existing per-sample trajectory logs.")

    record_trajectory = subparsers.add_parser("record-trajectory-step", help="Append one normalized file/line context access to a trajectory log.")
    record_trajectory.add_argument("--log", type=Path, help="Trajectory JSONL log path. Defaults to ARB_TRAJECTORY_LOG.")
    record_trajectory.add_argument("--sample-id", help="Sample id. Defaults to ARB_SAMPLE_ID.")
    record_trajectory.add_argument("--path", required=True, help="Repo-relative path that entered context.")
    record_trajectory.add_argument("--step", type=int)
    record_trajectory.add_argument("--tool", default="read")
    record_trajectory.add_argument("--start-line", type=int)
    record_trajectory.add_argument("--end-line", type=int)
    record_trajectory.add_argument("--kind", default="block")
    record_trajectory.add_argument("--symbol", default="")
    record_trajectory.add_argument("--content-hash", default="")
    record_trajectory.add_argument("--repo-root", type=Path, help="Optional checkout root used to compute content_hash.")
    record_trajectory.add_argument("--final", action="store_true", dest="is_final_context")
    record_trajectory.add_argument("--used", action="store_true", dest="is_utilized_context")
    record_trajectory.add_argument("--run-id", default="")
    record_trajectory.add_argument("--model-label", default="")

    embedding = subparsers.add_parser("eval-embedding", help="Run an embedding retrieval baseline.")
    embedding.add_argument(
        "samples",
        nargs="*",
        type=Path,
        help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.",
    )
    embedding.add_argument(
        "--version",
        help="Benchmark release version used to infer --derived, --corpus, --out, --cache, and --shared-text-cache.",
    )
    embedding.add_argument(
        "--model-label",
        help="Short label for output/cache paths. With --version, defaults to the model path/name basename.",
    )
    embedding.add_argument("--derived", type=Path, help="Defaults to data/benchmark/<version> or data/benchmark/v0_1.")
    embedding.add_argument("--corpus", type=Path, help="Defaults to data/corpus/<version> or data/corpus/v0_1.")
    embedding.add_argument("--model", required=True, help="SentenceTransformer-compatible model name or path.")
    embedding.add_argument("--out", type=Path)
    embedding.add_argument("--details", type=Path)
    embedding.add_argument("--cache", type=Path)
    embedding.add_argument(
        "--shared-text-cache",
        type=Path,
        help="SQLite cache for deduplicating identical chunk texts across commit corpora.",
    )
    embedding.add_argument(
        "--no-shared-text-cache",
        action="store_true",
        help="Disable the --version default shared text cache.",
    )
    embedding.add_argument(
        "--keep-list",
        type=Path,
        help="Optional sample ID allowlist. By default, embedding eval evaluates all provided samples.",
    )
    embedding.add_argument("--no-keep-list", action="store_true")
    embedding.add_argument("--sample-id-file", type=Path, help="Plain-text or JSONL file of sample IDs to evaluate.")
    embedding.add_argument("--shard-count", type=int, help="Total number of deterministic sample shards.")
    embedding.add_argument("--shard-index", type=int, help="Zero-based shard index to evaluate.")
    embedding.add_argument("--limit-samples", type=int)
    embedding.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    embedding.add_argument("--batch-size", type=int, default=32)
    embedding.add_argument("--query-batch-size", type=int, help="Query embeddings to encode together. Defaults to --batch-size.")
    embedding.add_argument("--device", help="SentenceTransformer device, e.g. cpu, cuda, mps.")
    embedding.add_argument("--query-prefix", default="")
    embedding.add_argument("--passage-prefix", default="")
    embedding.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    embedding.add_argument("--no-progress", action="store_true", help="Disable embedding evaluation progress output.")
    embedding.add_argument("--resume-details", action="store_true", help="Skip sample IDs already present in the details JSONL file and append remaining rows.")
    embedding.add_argument("--trust-remote-code", action="store_true")

    selective_embedding = subparsers.add_parser(
        "eval-selective-embedding",
        help="Run an embedding retrieval baseline with abstention threshold sweep.",
    )
    selective_embedding.add_argument(
        "samples",
        nargs="*",
        type=Path,
        help="Benchmark sample JSONL files. Defaults to --derived/*.jsonl.",
    )
    selective_embedding.add_argument(
        "--version",
        help="Benchmark release version used to infer --derived, --corpus, --out, --cache, and --shared-text-cache.",
    )
    selective_embedding.add_argument(
        "--model-label",
        help="Short label for output/cache paths. With --version, defaults to the model path/name basename.",
    )
    selective_embedding.add_argument("--derived", type=Path, help="Defaults to data/benchmark/<version> or data/benchmark/v0_1.")
    selective_embedding.add_argument("--corpus", type=Path, help="Defaults to data/corpus/<version> or data/corpus/v0_1.")
    selective_embedding.add_argument("--model", required=True, help="SentenceTransformer-compatible model name or path.")
    selective_embedding.add_argument("--out", type=Path)
    selective_embedding.add_argument("--details", type=Path)
    selective_embedding.add_argument("--sweep", type=Path)
    selective_embedding.add_argument("--report", type=Path)
    selective_embedding.add_argument("--cache", type=Path)
    selective_embedding.add_argument(
        "--shared-text-cache",
        type=Path,
        help="SQLite cache for deduplicating identical chunk texts across commit corpora.",
    )
    selective_embedding.add_argument(
        "--no-shared-text-cache",
        action="store_true",
        help="Disable the --version default shared text cache.",
    )
    selective_embedding.add_argument(
        "--keep-list",
        type=Path,
        help="Optional sample ID allowlist. By default, selective embedding eval evaluates all provided samples.",
    )
    selective_embedding.add_argument("--no-keep-list", action="store_true")
    selective_embedding.add_argument("--sample-id-file", type=Path, help="Plain-text or JSONL file of sample IDs to evaluate.")
    selective_embedding.add_argument("--shard-count", type=int, help="Total number of deterministic sample shards.")
    selective_embedding.add_argument("--shard-index", type=int, help="Zero-based shard index to evaluate.")
    selective_embedding.add_argument("--limit-samples", type=int)
    selective_embedding.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    selective_embedding.add_argument("--batch-size", type=int, default=32)
    selective_embedding.add_argument("--query-batch-size", type=int, help="Query embeddings to encode together. Defaults to --batch-size.")
    selective_embedding.add_argument("--device", help="SentenceTransformer device, e.g. cpu, cuda, mps.")
    selective_embedding.add_argument("--query-prefix", default="")
    selective_embedding.add_argument("--passage-prefix", default="")
    selective_embedding.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    selective_embedding.add_argument("--no-progress", action="store_true", help="Disable embedding evaluation progress output.")
    selective_embedding.add_argument("--trust-remote-code", action="store_true")

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
    voyage.add_argument(
        "--shared-text-cache",
        type=Path,
        help="SQLite cache for deduplicating identical chunk texts across commit corpora.",
    )
    voyage.add_argument("--keep-list", type=Path, default=Path("data/reports/v1/keep_samples.jsonl"))
    voyage.add_argument("--no-keep-list", action="store_true")
    voyage.add_argument("--sample-id-file", type=Path, help="Plain-text or JSONL file of sample IDs to evaluate.")
    voyage.add_argument("--shard-count", type=int, help="Total number of deterministic sample shards.")
    voyage.add_argument("--shard-index", type=int, help="Zero-based shard index to evaluate.")
    voyage.add_argument("--limit-samples", type=int)
    voyage.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    voyage.add_argument("--batch-size", type=int, default=32)
    voyage.add_argument("--query-batch-size", type=int, help="Query embeddings to encode together. Defaults to --batch-size.")
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
        "--max-request-chars",
        type=int,
        default=600000,
        help="Conservatively split Voyage requests by total character count; 600000 approximates 120K tokens at 5 chars/token.",
    )
    voyage.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=0.0,
        help="Sleep between Voyage requests. Use about 21 seconds for unpaid 3 RPM accounts.",
    )
    voyage.add_argument("--resume-details", action="store_true", help="Skip sample IDs already present in the details JSONL file and append remaining rows.")
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
    report_models.add_argument("--required-baseline", action="append", help="Required model label that must appear in the leaderboard. Can be repeated.")

    report_v2_positive = subparsers.add_parser(
        "report-v2-positive",
        help="Build the complete 345-sample V2 positive leaderboard.",
    )
    report_v2_positive.add_argument("--core-eval-dir", type=Path, default=Path("data/eval/v1_3_reviewed"))
    report_v2_positive.add_argument("--extension-eval-dir", type=Path, default=Path("data/eval/v2_edit2ripple"))
    report_v2_positive.add_argument("--core-bcy", type=Path, default=Path("data/reports/v1_4/bcy_budget_curve.json"))
    report_v2_positive.add_argument(
        "--extension-bcy",
        type=Path,
        default=Path("data/eval/v2_edit2ripple/edit2ripple_bcy_curves.json"),
    )
    report_v2_positive.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/v2_positive_leaderboard/leaderboard.md"),
    )
    report_v2_positive.add_argument("--json-out", type=Path)

    report_selective_cv = subparsers.add_parser(
        "report-selective-cv",
        help="Evaluate abstention with repo-grouped out-of-fold threshold calibration.",
    )
    report_selective_cv.add_argument(
        "--lexical-details",
        type=Path,
        default=DEFAULT_SELECTIVE_DETAILS["lexical"],
    )
    report_selective_cv.add_argument(
        "--bm25-details",
        type=Path,
        default=DEFAULT_SELECTIVE_DETAILS["BM25"],
    )
    report_selective_cv.add_argument(
        "--jina-details",
        type=Path,
        default=DEFAULT_SELECTIVE_DETAILS["Jina-0.5B"],
    )
    report_selective_cv.add_argument("--folds", type=int, default=5)
    report_selective_cv.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/v2_selective_group_cv/report.md"),
    )
    report_selective_cv.add_argument("--json-out", type=Path)

    dataset_validity = subparsers.add_parser("report-dataset-validity", help="Generate gold correctness, leakage, and task-diversity validity reports.")
    dataset_validity.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3_reviewed/samples.jsonl"))
    dataset_validity.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    dataset_validity.add_argument("--validation", type=Path, default=Path("data/reports/v1_3/reviewed_validation.json"))
    dataset_validity.add_argument("--audit", action="append", type=Path, help="Manual/review audit CSV, JSON, or JSONL. Defaults to V1.3 reviewed annotation files.")
    dataset_validity.add_argument("--eval-dir", action="append", type=Path, help="Eval summary directory to include. Defaults to V1.3/V1.4 full-run dirs.")
    dataset_validity.add_argument("--split-details-dir", action="append", type=Path, help="Eval details directory for flagged-vs-clean splits.")
    dataset_validity.add_argument("--out-dir", type=Path, default=Path("data/reports/v1_4/dataset_validity"))
    dataset_validity.add_argument("--audit-packet-size", type=int, default=90)
    dataset_validity.add_argument("--valid-threshold", type=float, default=0.90)
    dataset_validity.add_argument("--min-task-spread", type=float, default=0.05)

    context_cost = subparsers.add_parser("report-context-cost", help="Generate context-acquisition cost and retriever/agent join analysis.")
    context_cost.add_argument("--out", type=Path, default=Path("data/reports/v1_4/context_acquisition_cost.json"))
    context_cost.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/context_acquisition_cost.md"))
    context_cost.add_argument("--late-hit-threshold", type=int, default=3)

    rank_fusion = subparsers.add_parser("report-rank-fusion", help="Generate RRF hybrid retrieval baselines from existing ranked-list details.")
    rank_fusion.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_3_reviewed"))
    rank_fusion.add_argument("--out-dir", type=Path, default=Path("data/eval/v1_4/rank_fusion"))
    rank_fusion.add_argument("--out", type=Path, default=Path("data/reports/v1_4/rank_fusion_report.md"))
    rank_fusion.add_argument("--json-out", type=Path, default=Path("data/reports/v1_4/rank_fusion_report.json"))
    rank_fusion.add_argument("--rrf-k", type=int, default=60)

    bcy_curve = subparsers.add_parser("report-bcy-curve", help="Generate canonical BCY budget curves from stored ranked file lists.")
    bcy_curve.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    bcy_curve.add_argument("--out", type=Path, default=Path("data/reports/v1_4/bcy_budget_curve.json"))
    bcy_curve.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/bcy_budget_curve.md"))
    bcy_curve.add_argument("--budgets", default="4000,8000,16000,32000")
    bcy_curve.add_argument("--coverage-thresholds", default="1,16,32,64,128")

    pes_calibration = subparsers.add_parser("report-pes-calibration", help="Compare seeded vs control trajectories for PES calibration.")
    pes_calibration.add_argument("--packet", type=Path, default=Path("data/reports/v1_4/pes_calibration/selected_samples.jsonl"))
    pes_calibration.add_argument(
        "--control-details",
        type=Path,
        default=Path("data/eval/v1_4/codex_cli_gpt54_corpus_full_budgeted_answer_recovered_all_details.jsonl"),
    )
    pes_calibration.add_argument(
        "--seeded-details",
        action="append",
        type=Path,
        default=[],
        help="Seeded trajectory details JSONL. Can be repeated.",
    )
    pes_calibration.add_argument("--out", type=Path, default=Path("data/reports/v1_4/pes_calibration/calibration_report.json"))
    pes_calibration.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/pes_calibration/calibration_report.md"))

    layered_leaderboard = subparsers.add_parser("report-layered-leaderboard", help="Generate layered retrieval, BCY, context-selection, and trajectory complementarity tables.")
    layered_leaderboard.add_argument("--bcy-report", type=Path, default=Path("data/reports/v1_4/bcy_budget_curve.json"))
    layered_leaderboard.add_argument("--context-selection", type=Path, default=Path("data/eval/v1_4/context_selection_leaderboard_top3.json"))
    layered_leaderboard.add_argument("--context-cost", type=Path, default=Path("data/reports/v1_4/context_acquisition_cost.json"))
    layered_leaderboard.add_argument("--out", type=Path, default=Path("data/reports/v1_4/layered_leaderboard.json"))
    layered_leaderboard.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/layered_leaderboard.md"))

    cae_validity = subparsers.add_parser("report-cae-validity", help="Generate CAE proxy-validity correlations against logged agent quantities.")
    cae_validity.add_argument("--bcy-report", type=Path, default=Path("data/reports/v1_4/bcy_budget_curve.json"))
    cae_validity.add_argument("--context-cost", type=Path, default=Path("data/reports/v1_4/context_acquisition_cost.json"))
    cae_validity.add_argument("--pes-calibration", type=Path, default=Path("data/reports/v1_4/pes_calibration/calibration_report.json"))
    cae_validity.add_argument("--budget", type=int, default=8000)
    cae_validity.add_argument("--out", type=Path, default=Path("data/reports/v1_4/cae_validity_correlation.json"))
    cae_validity.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_4/cae_validity_correlation.md"))

    agentic_relevance = subparsers.add_parser("report-agentic-relevance", help="Annotate and report agentic relevance taxonomy slices.")
    agentic_relevance.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3_reviewed/samples.jsonl"))
    agentic_relevance.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    agentic_relevance.add_argument("--out-dir", type=Path, default=Path("data/reports/v1_4/agentic_relevance"))
    agentic_relevance.add_argument("--budget", type=int, default=8000)

    rank_analysis = subparsers.add_parser("report-rank-analysis", help="Build rank-depth and representative error analysis from eval details.")
    rank_analysis.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_1"))
    rank_analysis.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_1/samples.jsonl"))
    rank_analysis.add_argument("--out", type=Path, default=Path("data/reports/v1_1/rank_analysis.md"))
    rank_analysis.add_argument("--json-out", type=Path)
    rank_analysis.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")
    rank_analysis.add_argument("--top-examples", type=int, default=5)

    validate_v1_2 = subparsers.add_parser("validate-v1-2", help="Validate V1.2 optional spans, hard negatives, and provenance.")
    validate_v1_2.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_2"))
    validate_v1_2.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    validate_v1_2.add_argument("--no-corpus", action="store_true", help="Validate schema without corpus-bound line checks.")
    validate_v1_2.add_argument("--out", type=Path, default=Path("data/reports/v1_2/validation.json"))
    validate_v1_2.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_2/validation.md"))

    derive_v1_3 = subparsers.add_parser("derive-v1-3-blocks", help="Derive V1.3 gold_blocks from gold_spans and corpus symbol chunks.")
    derive_v1_3.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1_2"))
    derive_v1_3.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    derive_v1_3.add_argument("--out", type=Path, default=Path("data/benchmark/v1_3"))
    derive_v1_3.add_argument("--report-out", type=Path, default=Path("data/reports/v1_3/block_derivation.json"))
    derive_v1_3.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_3/block_derivation.md"))

    span_worklist_v1_3 = subparsers.add_parser(
        "write-v1-3-span-worklist",
        help="Write a human-review worklist for V1.3 samples that still lack gold_spans.",
    )
    span_worklist_v1_3.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    span_worklist_v1_3.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_2/corpus_manifest.jsonl"))
    span_worklist_v1_3.add_argument("--task-type", default="comment2context")
    span_worklist_v1_3.add_argument("--max-candidates-per-file", type=int, default=3)
    span_worklist_v1_3.add_argument("--out", type=Path, default=Path("data/reports/v1_3/comment2context_span_worklist.json"))
    span_worklist_v1_3.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_3/comment2context_span_worklist.md"))
    span_worklist_v1_3.add_argument("--jsonl-out", type=Path, default=Path("data/reports/v1_3/comment2context_span_suggestions.jsonl"))

    validate_v1_3 = subparsers.add_parser("validate-v1-3", help="Validate V1.3 spans, gold_blocks, hard negatives, and provenance.")
    validate_v1_3.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_3"))
    validate_v1_3.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_3/corpus_manifest.jsonl"))
    validate_v1_3.add_argument("--no-corpus", action="store_true", help="Validate schema without corpus-bound line/block checks.")
    validate_v1_3.add_argument("--allow-partial-spans", action="store_true", help="Do not require every sample to have gold_spans.")
    validate_v1_3.add_argument("--allow-partial-blocks", action="store_true", help="Do not require every sample to have gold_blocks.")
    validate_v1_3.add_argument("--out", type=Path, default=Path("data/reports/v1_3/validation.json"))
    validate_v1_3.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_3/validation.md"))

    merge_v1_2 = subparsers.add_parser("merge-v1-2-annotations", help="Apply manual V1.2 span and hard-negative annotations.")
    merge_v1_2.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1_1"))
    merge_v1_2.add_argument("--annotations", type=Path, default=Path("data/reports/v1_2/manual_annotations_round1.jsonl"))
    merge_v1_2.add_argument("--out", type=Path, default=Path("data/benchmark/v1_2"))
    merge_v1_2.add_argument("--report-out", type=Path, default=Path("data/reports/v1_2/manual_annotation_merge.json"))
    merge_v1_2.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_2/manual_annotation_merge.md"))

    context_pollution = subparsers.add_parser("report-context-pollution", help="Build V1.2 precision and context-pollution report from details.")
    context_pollution.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_2"))
    context_pollution.add_argument("--out", type=Path, default=Path("data/reports/v1_2/context_pollution.md"))
    context_pollution.add_argument("--json-out", type=Path)
    context_pollution.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")

    span_subset = subparsers.add_parser("report-span-subset", help="Build V1.2 span-level subset report from details.")
    span_subset.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_2"))
    span_subset.add_argument("--out", type=Path, default=Path("data/reports/v1_2/span_subset.md"))
    span_subset.add_argument("--json-out", type=Path)
    span_subset.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="all_files")

    runtime_cache = subparsers.add_parser("report-runtime-cache", help="Build V1.2 runtime/cache report from eval summaries.")
    runtime_cache.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_2"))
    runtime_cache.add_argument("--out", type=Path, default=Path("data/reports/v1_2/runtime_cache.md"))
    runtime_cache.add_argument("--json-out", type=Path)

    check_baselines = subparsers.add_parser(
        "check-baseline-summaries",
        help="Preflight eval summary files against required V1.1 all-files baseline gates.",
    )
    check_baselines.add_argument("summaries", nargs="*", type=Path, help="Summary JSON files to inspect.")
    check_baselines.add_argument("--eval-dir", action="append", type=Path, help="Eval dir containing *_summary.json files. Can be repeated.")
    check_baselines.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"), help="Derived benchmark used to infer expected sample count.")
    check_baselines.add_argument("--expected-samples", type=int, help="Expected evaluated and overall sample count.")
    check_baselines.add_argument("--required-baseline", action="append", help="Required baseline label. Defaults to V1.1 required baselines.")
    check_baselines.add_argument("--no-require-details", action="store_true", help="Only inspect summary JSON files; by default matching details JSONL files are required.")
    check_baselines.add_argument("--out", type=Path, help="Optional JSON report path.")

    baseline_status = subparsers.add_parser(
        "v1-1-baseline-status",
        help="Report required V1.1 baseline artifact status and local embedding runtime blockers.",
    )
    baseline_status.add_argument("summaries", nargs="*", type=Path, help="Summary JSON files to inspect.")
    baseline_status.add_argument("--eval-dir", action="append", type=Path, help="Eval dir containing *_summary.json files. Can be repeated.")
    baseline_status.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"), help="Derived benchmark used to infer expected sample count.")
    baseline_status.add_argument("--expected-samples", type=int, help="Expected evaluated and overall sample count.")
    baseline_status.add_argument("--required-baseline", action="append", help="Required baseline label. Defaults to V1.1 required baselines.")
    baseline_status.add_argument("--no-require-details", action="store_true", help="Only inspect summary JSON files; by default matching details JSONL files are required.")
    baseline_status.add_argument("--shard-commands", type=Path, help="Optional report from v1-1-baseline-shard-commands to inspect sharded details progress.")
    baseline_status.add_argument("--out", type=Path, help="Optional JSON report path.")
    baseline_status.add_argument("--markdown-out", type=Path, help="Optional Markdown report path.")

    external_runner_preflight = subparsers.add_parser(
        "v1-1-external-runner-preflight",
        help="Summarize V1.1 external runner readiness, local runtime blockers, and return packaging blockers.",
    )
    external_runner_preflight.add_argument("--baseline-status", type=Path, default=Path("data/reports/v1_1/baseline_status_v19.json"))
    external_runner_preflight.add_argument("--return-acceptance", type=Path, default=Path("data/reports/v1_1/baseline_return_acceptance_v19.json"))
    external_runner_preflight.add_argument("--return-manifest", type=Path, default=Path("data/reports/v1_1/baseline_return_manifest_v19.json"))
    external_runner_preflight.add_argument("--transfer-manifest-verify", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest_verify_v19.json"))
    external_runner_preflight.add_argument("--handoff-verify", type=Path, default=Path("data/reports/v1_1/baseline_handoff_verify_v19.json"))
    external_runner_preflight.add_argument("--transfer-bundle-verify", type=Path, default=Path("data/reports/v1_1/baseline_transfer_bundle_verify_v19.json"))
    external_runner_preflight.add_argument(
        "--copy-packet",
        type=Path,
        help="Optional sender-side copy packet JSON to check against the current transfer bundle report and verifier.",
    )
    external_runner_preflight.add_argument("--full-runner", type=Path, default=Path("data/reports/v1_1/run_v19_baseline_shards.sh"))
    external_runner_preflight.add_argument("--gpu-runner", type=Path, default=Path("data/reports/v1_1/run_v19_gpu_baseline_shards.sh"))
    external_runner_preflight.add_argument("--voyage-runner", type=Path, default=Path("data/reports/v1_1/run_v19_voyage_baseline_shards.sh"))
    external_runner_preflight.add_argument("--return-bundle-script", type=Path, default=Path("data/reports/v1_1/package_v19_return_artifacts.sh"))
    external_runner_preflight.add_argument(
        "--gpu-return-bundle-script",
        type=Path,
        default=Path("data/reports/v1_1/package_v19_gpu_return_artifacts.sh"),
    )
    external_runner_preflight.add_argument(
        "--voyage-return-bundle-script",
        type=Path,
        default=Path("data/reports/v1_1/package_v19_voyage_return_artifacts.sh"),
    )
    external_runner_preflight.add_argument(
        "--gpu-return-manifest",
        type=Path,
        help="Optional filtered Jina/Qwen return manifest. When omitted, split packaging status is derived from the full return manifest.",
    )
    external_runner_preflight.add_argument(
        "--voyage-return-manifest",
        type=Path,
        help="Optional filtered Voyage return manifest. When omitted, split packaging status is derived from the full return manifest.",
    )
    external_runner_preflight.add_argument("--out", type=Path, default=Path("data/reports/v1_1/external_runner_preflight_v19.json"))
    external_runner_preflight.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/external_runner_preflight_v19.md"))

    external_runner_smoke = subparsers.add_parser(
        "v1-1-external-runner-failfast-smoke",
        help="Execute generated V1.1 external runner scripts only when preflight expects a local fail-fast blocker.",
    )
    external_runner_smoke.add_argument(
        "--preflight",
        type=Path,
        default=Path("data/reports/v1_1/external_runner_preflight_v19.json"),
        help="Preflight JSON whose expected blockers decide which scripts are safe to smoke-run.",
    )
    external_runner_smoke.add_argument(
        "--full-runner",
        type=Path,
        default=Path("data/reports/v1_1/run_v19_baseline_shards.sh"),
        help="Generated full Jina/Qwen/Voyage runner script.",
    )
    external_runner_smoke.add_argument(
        "--gpu-runner",
        type=Path,
        default=Path("data/reports/v1_1/run_v19_gpu_baseline_shards.sh"),
        help="Generated Jina/Qwen-only runner script.",
    )
    external_runner_smoke.add_argument(
        "--voyage-runner",
        type=Path,
        default=Path("data/reports/v1_1/run_v19_voyage_baseline_shards.sh"),
        help="Generated Voyage-only runner script.",
    )
    external_runner_smoke.add_argument(
        "--return-bundle-script",
        type=Path,
        default=Path("data/reports/v1_1/package_v19_return_artifacts.sh"),
        help="Generated return-bundle packaging script to smoke-run when required files are missing.",
    )
    external_runner_smoke.add_argument(
        "--gpu-return-bundle-script",
        type=Path,
        default=Path("data/reports/v1_1/package_v19_gpu_return_artifacts.sh"),
        help="Generated Jina/Qwen partial return-bundle packaging script to smoke-run when required files are missing.",
    )
    external_runner_smoke.add_argument(
        "--voyage-return-bundle-script",
        type=Path,
        default=Path("data/reports/v1_1/package_v19_voyage_return_artifacts.sh"),
        help="Generated Voyage partial return-bundle packaging script to smoke-run when required files are missing.",
    )
    external_runner_smoke.add_argument("--timeout-seconds", type=int, default=60, help="Per-script smoke timeout in seconds.")
    external_runner_smoke.add_argument(
        "--allow-ready-runs",
        action="store_true",
        help="Also run scripts whose preflight status is ready; use only on a disposable runner.",
    )
    external_runner_smoke.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/v1_1/external_runner_failfast_smoke_v19.json"),
        help="JSON smoke report path.",
    )
    external_runner_smoke.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("data/reports/v1_1/external_runner_failfast_smoke_v19.md"),
        help="Markdown smoke report path.",
    )

    summary_from_details = subparsers.add_parser(
        "v1-1-summary-from-details",
        help="Rebuild a V1.1 baseline summary JSON from a complete details JSONL file.",
    )
    summary_from_details.add_argument("--details", type=Path, required=True)
    summary_from_details.add_argument("--out", type=Path, required=True)
    summary_from_details.add_argument("--model", required=True)
    summary_from_details.add_argument("--mode", default="embedding")
    summary_from_details.add_argument("--candidate-filter")
    summary_from_details.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"), help="Derived benchmark used to validate sample IDs.")
    summary_from_details.add_argument("--expected-samples", type=int, help="Expected details row count.")
    summary_from_details.add_argument("--cache-dir", type=Path)
    summary_from_details.add_argument("--shared-text-cache", type=Path)

    merge_details = subparsers.add_parser(
        "v1-1-merge-details",
        help="Merge sharded V1.1 baseline details JSONL files after validating sample coverage.",
    )
    merge_details.add_argument("--details", action="append", type=Path, required=True, help="Shard details JSONL path. Can be repeated.")
    merge_details.add_argument("--out", type=Path, required=True)
    merge_details.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"), help="Derived benchmark used to validate sample IDs.")
    merge_details.add_argument("--expected-samples", type=int, help="Expected merged details row count.")
    merge_details.add_argument("--candidate-filter")
    merge_details.add_argument("--allow-incomplete", action="store_true", help="Write the merged details file even when validation is incomplete.")
    merge_details.add_argument("--report-out", type=Path, help="Optional JSON merge report path.")
    merge_details.add_argument("--markdown-out", type=Path, help="Optional Markdown merge report path.")

    sample_shards = subparsers.add_parser(
        "v1-1-write-sample-shards",
        help="Write deterministic V1.1 sample-id shard files for external baseline workers.",
    )
    sample_shards.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"), help="Derived benchmark to shard.")
    sample_shards.add_argument("--out-dir", type=Path, required=True)
    sample_shards.add_argument("--shard-count", type=int, required=True)
    sample_shards.add_argument("--prefix", default="sample_ids")
    sample_shards.add_argument("--manifest-out", type=Path, help="Optional JSON shard manifest path. Defaults to OUT_DIR/manifest.json.")
    sample_shards.add_argument("--markdown-out", type=Path, help="Optional Markdown shard manifest path.")
    sample_shards.add_argument("--allow-empty-shards", action="store_true", help="Allow shard_count to exceed sample count.")
    sample_shards.add_argument(
        "--strategy",
        choices=("input_order_modulo", "corpus_balanced"),
        default="input_order_modulo",
        help="Shard assignment strategy. corpus_balanced keeps each repo/base_commit corpus in one shard.",
    )
    sample_shards.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Optional corpus_manifest.jsonl used to balance corpus shards by chunk_count.",
    )

    baseline_shard_commands = subparsers.add_parser(
        "v1-1-baseline-shard-commands",
        help="Expand V1.1 baseline handoff jobs into concrete per-sample-shard commands.",
    )
    baseline_shard_commands.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    baseline_shard_commands.add_argument("--sample-shards", type=Path, required=True, help="Manifest from v1-1-write-sample-shards.")
    baseline_shard_commands.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_shard_commands.json"))
    baseline_shard_commands.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_shard_commands.md"))
    baseline_shard_commands.add_argument("--report-dir", type=Path, help="Directory for merge reports. Defaults to handoff report_dir.")
    baseline_shard_commands.add_argument("--shared-caches", action="store_true", help="Keep original cache paths instead of writing shard-specific cache paths.")

    baseline_run_script = subparsers.add_parser(
        "v1-1-baseline-run-script",
        help="Write an executable shell script from a V1.1 baseline shard command report.",
    )
    baseline_run_script.add_argument("--shard-commands", type=Path, required=True, help="Report from v1-1-baseline-shard-commands.")
    baseline_run_script.add_argument("--baseline", action="append", help="Only include this baseline. Can be repeated.")
    baseline_run_script.add_argument("--out", type=Path, required=True)
    baseline_run_script.add_argument("--markdown-out", type=Path, help="Optional Markdown summary path.")
    baseline_run_script.add_argument("--transfer-manifest", type=Path, help="Optional transfer manifest to verify before running baseline shards.")
    baseline_run_script.add_argument("--return-manifest", type=Path, help="Optional return manifest to refresh with --require-existing after full-baseline runs.")
    baseline_run_script.add_argument("--return-manifest-markdown", type=Path, help="Optional Markdown return manifest path for the post-run check.")
    baseline_run_script.add_argument("--return-files", type=Path, help="Optional file-list path for the post-run return manifest check.")
    baseline_run_script.add_argument("--return-bundle-script", type=Path, help="Optional generated return-bundle packaging script to run before finalization for full-baseline runs.")
    baseline_run_script.add_argument("--include-return-shard-artifacts", action="store_true", help="Include per-shard artifacts in the post-run return manifest check.")
    baseline_run_script.add_argument("--include-return-caches", action="store_true", help="Include cache artifacts in the post-run return manifest check.")
    baseline_run_script.add_argument("--finalization", type=Path, help="Optional finalization report path to regenerate final V1.1 gates after full-baseline runs.")
    baseline_run_script.add_argument("--finalization-markdown", type=Path, help="Optional Markdown finalization report path.")
    baseline_run_script.add_argument("--return-acceptance", type=Path, help="Optional compact return-acceptance report to refresh after finalization.")
    baseline_run_script.add_argument("--return-acceptance-markdown", type=Path, help="Optional Markdown return-acceptance report path.")
    baseline_run_script.add_argument("--completion-json", type=Path, help="Optional completion audit JSON path to bind into the return-acceptance refresh.")
    baseline_run_script.add_argument("--workflow-evidence", action="append", type=Path, help="Workflow artifact or verifier report to pass through to finalization. Can be repeated.")
    baseline_run_script.add_argument("--no-runtime-checks", action="store_true", help="Do not emit CUDA or required-env checks at the top of the script.")

    baseline_handoff = subparsers.add_parser(
        "v1-1-baseline-handoff",
        help="Write a machine-readable handoff manifest for running the remaining V1.1 baselines externally.",
    )
    baseline_handoff.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"))
    baseline_handoff.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1"))
    baseline_handoff.add_argument("--corpus", type=Path, default=Path("data/corpus/v1_1"))
    baseline_handoff.add_argument("--corpus-manifest", type=Path)
    baseline_handoff.add_argument("--assembly-manifest", type=Path)
    baseline_handoff.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_1"))
    baseline_handoff.add_argument("--cache-root", type=Path, default=Path("data/embeddings/v1_1"))
    baseline_handoff.add_argument("--report-dir", type=Path, default=Path("data/reports/v1_1"))
    baseline_handoff.add_argument("--leaderboard", type=Path)
    baseline_handoff.add_argument("--leaderboard-json", type=Path)
    baseline_handoff.add_argument("--readiness", type=Path)
    baseline_handoff.add_argument("--readiness-markdown", type=Path)
    baseline_handoff.add_argument("--release", type=Path)
    baseline_handoff.add_argument("--release-json", type=Path)
    baseline_handoff.add_argument("--completion", type=Path)
    baseline_handoff.add_argument("--completion-json", type=Path)
    baseline_handoff.add_argument("--baseline-status", type=Path)
    baseline_handoff.add_argument("--baseline-status-markdown", type=Path)
    baseline_handoff.add_argument("--baseline-preflight", type=Path)
    baseline_handoff.add_argument("--handoff-verify", type=Path)
    baseline_handoff.add_argument("--handoff-verify-markdown", type=Path)
    baseline_handoff.add_argument("--finalization", type=Path)
    baseline_handoff.add_argument("--finalization-markdown", type=Path)
    baseline_handoff.add_argument("--shard-commands", type=Path, help="Optional shard-command report to include in generated finalization command.")
    baseline_handoff.add_argument("--return-manifest", type=Path, help="Optional return manifest to include in generated finalization command.")
    baseline_handoff.add_argument("--return-manifest-markdown", type=Path, help="Optional Markdown return manifest path for generated finalization command.")
    baseline_handoff.add_argument("--return-files", type=Path, help="Optional return-manifest file list for generated finalization command.")
    baseline_handoff.add_argument("--return-acceptance", type=Path, help="Optional sender-side compact return-acceptance report to generate outside the transfer bundle.")
    baseline_handoff.add_argument("--return-acceptance-markdown", type=Path, help="Optional Markdown return-acceptance report path.")
    baseline_handoff.add_argument("--include-shard-artifacts", action="store_true", help="Include shard artifacts in generated finalization return-manifest refresh.")
    baseline_handoff.add_argument("--include-caches", action="store_true", help="Include cache artifacts in generated finalization return-manifest refresh.")
    baseline_handoff.add_argument("--auto-merge-shards", action="store_true", help="Pass --auto-merge-shards in generated finalization command.")
    baseline_handoff.add_argument("--workflow-evidence", action="append", type=Path, help="Extra workflow artifact or verifier report for generated finalization command. Can be repeated.")
    baseline_handoff.add_argument("--transfer-manifest", type=Path)
    baseline_handoff.add_argument("--transfer-manifest-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-manifest-verify", type=Path)
    baseline_handoff.add_argument("--transfer-manifest-verify-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-files", type=Path)
    baseline_handoff.add_argument("--transfer-bundle", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-checksum", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-archive-members", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-report", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-verify", type=Path)
    baseline_handoff.add_argument("--transfer-bundle-verify-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-script", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-script-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-destination")
    baseline_handoff.add_argument("--transfer-unpack-transfer-verify", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-transfer-verify-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-handoff-verify", type=Path)
    baseline_handoff.add_argument("--transfer-unpack-handoff-verify-markdown", type=Path)
    baseline_handoff.add_argument("--transfer-include", action="append", type=Path, help="Extra helper file or directory to include in generated transfer-manifest setup commands. Can be repeated.")
    baseline_handoff.add_argument("--base-leaderboard-json", type=Path, default=Path("data/reports/v1/model_leaderboard.json"))
    baseline_handoff.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    baseline_handoff.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_handoff.md"))

    verify_handoff = subparsers.add_parser(
        "v1-1-verify-handoff",
        help="Verify a V1.1 baseline handoff manifest against local input file fingerprints.",
    )
    verify_handoff.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    verify_handoff.add_argument("--out", type=Path, help="Optional JSON verification report path.")
    verify_handoff.add_argument("--markdown-out", type=Path, help="Optional Markdown verification report path.")

    transfer_manifest = subparsers.add_parser(
        "v1-1-baseline-transfer-manifest",
        help="List local input files that must be transferred to an external V1.1 baseline runner.",
    )
    transfer_manifest.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    transfer_manifest.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.json"))
    transfer_manifest.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.md"))
    transfer_manifest.add_argument("--files-out", type=Path, help="Optional newline-delimited file list for rsync/tar.")
    transfer_manifest.add_argument("--include", action="append", type=Path, help="Extra helper file or directory to include in the transfer list. Can be repeated.")

    transfer_bundle = subparsers.add_parser(
        "v1-1-baseline-transfer-bundle",
        help="Create and verify a V1.1 baseline transfer bundle from a transfer manifest.",
    )
    transfer_bundle.add_argument("--manifest", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.json"))
    transfer_bundle.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_transfer_bundle.tar.zst"))
    transfer_bundle.add_argument("--checksum", type=Path, help="Optional SHA256 checksum path. Defaults to <bundle>.sha256.")
    transfer_bundle.add_argument("--archive-members", type=Path, help="Optional path for the extracted tar member list. Defaults to <bundle>.members.")
    transfer_bundle.add_argument("--bundle-files", type=Path, help="Optional newline-delimited file list used by tar. Defaults to <bundle>.files.")
    transfer_bundle.add_argument("--out", type=Path, help="Optional JSON bundle creation report path.")
    transfer_bundle.add_argument("--markdown-out", type=Path, help="Optional Markdown bundle creation report path.")
    transfer_bundle.add_argument("--verify-out", type=Path, help="Optional JSON verification report path.")
    transfer_bundle.add_argument("--verify-markdown-out", type=Path, help="Optional Markdown verification report path.")
    transfer_bundle.add_argument("--compression", default="zstd -T0 -3", help="Compression command passed to tar --use-compress-program.")
    transfer_bundle.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root used as tar working directory.")
    transfer_bundle.add_argument("--no-test-compression", action="store_true", help="Do not run zstd -t during verification.")
    transfer_bundle.add_argument("--copy-packet-out", type=Path, help="Optional external-runner copy-packet JSON path to refresh after bundle creation.")
    transfer_bundle.add_argument("--copy-packet-markdown-out", type=Path, help="Optional external-runner copy-packet Markdown path to refresh after bundle creation.")

    copy_packet = subparsers.add_parser(
        "v1-1-external-runner-copy-packet",
        help="Write a small non-bundled copy checklist for moving the current V1.1 transfer bundle to an external runner.",
    )
    copy_packet.add_argument("--transfer-bundle-report", type=Path, default=Path("data/reports/v1_1/baseline_transfer_bundle_v19.json"))
    copy_packet.add_argument("--bundle", type=Path, help="Optional transfer bundle path. Defaults to the path recorded in the transfer bundle report.")
    copy_packet.add_argument("--checksum", type=Path, help="Optional checksum sidecar path. Defaults to the path recorded in the transfer bundle report.")
    copy_packet.add_argument("--unpack-script", type=Path, default=Path("data/reports/v1_1/unpack_v19_transfer_bundle.sh"))
    copy_packet.add_argument("--bundle-report-markdown", type=Path, help="Optional Markdown transfer bundle report path.")
    copy_packet.add_argument("--destination", default="agent-retrieval-bench-v1_1-v19-transfer")
    copy_packet.add_argument("--full-runner", type=Path, default=Path("data/reports/v1_1/run_v19_baseline_shards.sh"))
    copy_packet.add_argument("--gpu-runner", type=Path, default=Path("data/reports/v1_1/run_v19_gpu_baseline_shards.sh"))
    copy_packet.add_argument("--voyage-runner", type=Path, default=Path("data/reports/v1_1/run_v19_voyage_baseline_shards.sh"))
    copy_packet.add_argument("--local-apply-script", type=Path, default=Path("data/reports/v1_1/apply_v19_return_artifacts.sh"))
    copy_packet.add_argument("--completion-json", type=Path, default=Path("data/reports/v1_1/completion_audit_v19.json"))
    copy_packet.add_argument("--out", type=Path, default=Path("data/reports/v1_1/external_runner_copy_packet_v19.json"))
    copy_packet.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/external_runner_copy_packet_v19.md"))

    verify_transfer = subparsers.add_parser(
        "v1-1-verify-transfer-manifest",
        help="Verify a V1.1 baseline transfer manifest after copying files to an external runner.",
    )
    verify_transfer.add_argument("--manifest", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.json"))
    verify_transfer.add_argument("--out", type=Path, help="Optional JSON verification report path.")
    verify_transfer.add_argument("--markdown-out", type=Path, help="Optional Markdown verification report path.")

    verify_transfer_bundle = subparsers.add_parser(
        "v1-1-verify-transfer-bundle",
        help="Verify a V1.1 baseline transfer bundle before copying it to an external runner.",
    )
    verify_transfer_bundle.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_transfer_bundle.tar.zst"))
    verify_transfer_bundle.add_argument("--manifest", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.json"))
    verify_transfer_bundle.add_argument("--checksum", type=Path, help="Optional SHA256 checksum path. Defaults to <bundle>.sha256.")
    verify_transfer_bundle.add_argument("--archive-members", type=Path, help="Optional path for the extracted tar member list. Defaults to <bundle>.members.")
    verify_transfer_bundle.add_argument("--out", type=Path, help="Optional JSON verification report path.")
    verify_transfer_bundle.add_argument("--markdown-out", type=Path, help="Optional Markdown verification report path.")
    verify_transfer_bundle.add_argument("--no-test-compression", action="store_true", help="Do not run zstd -t before inspecting tar members.")

    transfer_unpack_script = subparsers.add_parser(
        "v1-1-baseline-transfer-unpack-script",
        help="Write a shell script that verifies and unpacks a V1.1 baseline transfer bundle on an external runner.",
    )
    transfer_unpack_script.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_transfer_bundle.tar.zst"))
    transfer_unpack_script.add_argument("--checksum", type=Path, help="Optional SHA256 checksum path. Defaults to <bundle>.sha256.")
    transfer_unpack_script.add_argument("--manifest", type=Path, default=Path("data/reports/v1_1/baseline_transfer_manifest.json"))
    transfer_unpack_script.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    transfer_unpack_script.add_argument("--destination", default="agent-retrieval-bench-v1_1-transfer")
    transfer_unpack_script.add_argument("--transfer-verify", type=Path, default=Path("data/reports/v1_1/baseline_transfer_unpack_smoke.json"))
    transfer_unpack_script.add_argument("--transfer-verify-markdown", type=Path)
    transfer_unpack_script.add_argument("--handoff-verify", type=Path, default=Path("data/reports/v1_1/baseline_handoff_unpack_smoke.json"))
    transfer_unpack_script.add_argument("--handoff-verify-markdown", type=Path)
    transfer_unpack_script.add_argument("--out", type=Path, required=True)
    transfer_unpack_script.add_argument("--markdown-out", type=Path, help="Optional Markdown summary path.")

    verify_return_bundle = subparsers.add_parser(
        "v1-1-verify-return-bundle",
        help="Verify a returned V1.1 baseline artifact bundle before unpacking it.",
    )
    verify_return_bundle.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_return_bundle.tar.zst"))
    verify_return_bundle.add_argument("--return-manifest", type=Path, default=Path("data/reports/v1_1/baseline_return_manifest.json"))
    verify_return_bundle.add_argument("--checksum", type=Path, help="Optional SHA256 checksum path. Defaults to <bundle>.sha256.")
    verify_return_bundle.add_argument("--archive-members", type=Path, help="Optional path for the extracted tar member list. Defaults to <bundle>.members.")
    verify_return_bundle.add_argument("--bundle-files", type=Path, help="Optional generated return-bundle file list to require exact archive membership.")
    verify_return_bundle.add_argument("--out", type=Path, help="Optional JSON verification report path.")
    verify_return_bundle.add_argument("--markdown-out", type=Path, help="Optional Markdown verification report path.")
    verify_return_bundle.add_argument("--no-test-compression", action="store_true", help="Do not run zstd -t before inspecting tar members.")

    return_manifest = subparsers.add_parser(
        "v1-1-baseline-return-manifest",
        help="Write the expected files to copy back after external V1.1 baseline runs.",
    )
    return_manifest.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    return_manifest.add_argument("--shard-commands", type=Path, help="Optional baseline shard commands report.")
    return_manifest.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_return_manifest.json"))
    return_manifest.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_return_manifest.md"))
    return_manifest.add_argument("--files-out", type=Path, help="Optional newline-delimited return file list.")
    return_manifest.add_argument("--baseline", action="append", help="Only include this baseline in the return manifest. Can be repeated.")
    return_manifest.add_argument("--include-shard-artifacts", action="store_true", help="Also list per-shard summary/details artifacts.")
    return_manifest.add_argument("--include-caches", action="store_true", help="Also list embedding cache artifacts/directories.")
    return_manifest.add_argument("--require-existing", action="store_true", help="Return nonzero if required return artifacts do not exist yet.")

    return_acceptance = subparsers.add_parser(
        "v1-1-baseline-return-acceptance",
        help="Write a compact acceptance checklist for external V1.1 baseline returns.",
    )
    return_acceptance.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    return_acceptance.add_argument("--return-manifest", type=Path, help="Optional return manifest report.")
    return_acceptance.add_argument("--completion-json", type=Path, help="Optional completion audit JSON report.")
    return_acceptance.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_return_acceptance.json"))
    return_acceptance.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_return_acceptance.md"))
    return_acceptance.add_argument("--require-complete", action="store_true", help="Return nonzero unless the acceptance gate is complete.")

    return_bundle_script = subparsers.add_parser(
        "v1-1-baseline-return-bundle-script",
        help="Write a shell script that validates and bundles returned V1.1 baseline artifacts.",
    )
    return_bundle_script.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    return_bundle_script.add_argument("--shard-commands", type=Path, help="Optional baseline shard commands report.")
    return_bundle_script.add_argument("--return-manifest", type=Path, default=Path("data/reports/v1_1/baseline_return_manifest.json"))
    return_bundle_script.add_argument("--return-manifest-markdown", type=Path, help="Optional Markdown return manifest path.")
    return_bundle_script.add_argument("--return-files", type=Path, help="Optional newline-delimited return file list.")
    return_bundle_script.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_return_bundle.tar.zst"))
    return_bundle_script.add_argument("--bundle-files", type=Path, help="Optional newline-delimited file list for the return bundle.")
    return_bundle_script.add_argument("--baseline", action="append", help="Only require and bundle this baseline's artifacts. Can be repeated.")
    return_bundle_script.add_argument("--out", type=Path, required=True)
    return_bundle_script.add_argument("--markdown-out", type=Path, help="Optional Markdown summary path.")
    return_bundle_script.add_argument("--include-shard-artifacts", action="store_true", help="Also include per-shard summary/details artifacts when they exist.")
    return_bundle_script.add_argument("--include-caches", action="store_true", help="Also include embedding cache artifacts/directories when they exist.")
    return_bundle_script.add_argument("--compression", default="zstd -T0 -3", help="Compression command passed to tar --use-compress-program.")
    return_bundle_script.add_argument("--no-test-bundle", action="store_true", help="Do not add a zstd integrity test to the generated script.")

    apply_return_bundle_script = subparsers.add_parser(
        "v1-1-baseline-apply-return-bundle-script",
        help="Write a shell script that verifies, unpacks, and finalizes a returned V1.1 baseline bundle.",
    )
    apply_return_bundle_script.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    apply_return_bundle_script.add_argument("--shard-commands", type=Path, help="Optional baseline shard commands report.")
    apply_return_bundle_script.add_argument("--return-manifest", type=Path, help="Optional return manifest path to refresh with --require-existing after unpacking.")
    apply_return_bundle_script.add_argument("--return-manifest-markdown", type=Path, help="Optional Markdown return manifest path.")
    apply_return_bundle_script.add_argument("--return-files", type=Path, help="Optional newline-delimited return file list.")
    apply_return_bundle_script.add_argument("--bundle", type=Path, default=Path("data/reports/v1_1/baseline_return_bundle.tar.zst"))
    apply_return_bundle_script.add_argument("--checksum", type=Path, help="Optional SHA256 checksum path. Defaults to <bundle>.sha256.")
    apply_return_bundle_script.add_argument("--baseline", action="append", help="Only refresh/check this baseline in the post-unpack return manifest. Can be repeated.")
    apply_return_bundle_script.add_argument("--finalization", type=Path, default=Path("data/reports/v1_1/baseline_finalization.json"))
    apply_return_bundle_script.add_argument("--finalization-markdown", type=Path, help="Optional Markdown finalization report path.")
    apply_return_bundle_script.add_argument("--return-acceptance", type=Path, help="Optional compact return-acceptance report to refresh after finalization.")
    apply_return_bundle_script.add_argument("--return-acceptance-markdown", type=Path, help="Optional Markdown return-acceptance report path.")
    apply_return_bundle_script.add_argument("--completion-json", type=Path, help="Optional completion audit JSON path to bind into the return-acceptance refresh.")
    apply_return_bundle_script.add_argument("--out", type=Path, required=True)
    apply_return_bundle_script.add_argument("--markdown-out", type=Path, help="Optional Markdown summary path.")
    apply_return_bundle_script.add_argument("--include-shard-artifacts", action="store_true", help="Also check/list per-shard summary/details artifacts when they exist.")
    apply_return_bundle_script.add_argument("--include-caches", action="store_true", help="Also check/list embedding cache artifacts/directories when they exist.")
    apply_return_bundle_script.add_argument("--no-test-bundle", action="store_true", help="Do not add a zstd integrity test to the generated script.")
    apply_return_bundle_script.add_argument("--no-auto-merge-shards", action="store_true", help="Do not pass --auto-merge-shards during finalization.")
    apply_return_bundle_script.add_argument("--no-finalization", action="store_true", help="Only verify and unpack the return bundle; do not run v1-1-finalize-baselines.")
    apply_return_bundle_script.add_argument("--workflow-evidence", action="append", type=Path, help="Extra workflow artifact or verifier report to pass through to finalization. Can be repeated.")

    finalize_baselines = subparsers.add_parser(
        "v1-1-finalize-baselines",
        help="Verify returned V1.1 baseline artifacts and regenerate final leaderboard, readiness, release, and completion reports.",
    )
    finalize_baselines.add_argument("--handoff", type=Path, default=Path("data/reports/v1_1/baseline_handoff.json"))
    finalize_baselines.add_argument("--shard-commands", type=Path, help="Optional report from v1-1-baseline-shard-commands for shard progress checks.")
    finalize_baselines.add_argument("--return-manifest", type=Path, help="Optional return manifest path to refresh and require in finalization status.")
    finalize_baselines.add_argument("--return-manifest-markdown", type=Path, help="Optional Markdown return manifest path.")
    finalize_baselines.add_argument("--return-files", type=Path, help="Optional newline-delimited return file list.")
    finalize_baselines.add_argument("--include-shard-artifacts", action="store_true", help="Also list per-shard artifacts in the refreshed return manifest.")
    finalize_baselines.add_argument("--include-caches", action="store_true", help="Also list cache artifacts in the refreshed return manifest.")
    finalize_baselines.add_argument("--auto-merge-shards", action="store_true", help="Merge complete shard details and rebuild final summaries before running final gates.")
    finalize_baselines.add_argument("--doc", action="append", type=Path, help="Doc path expected for completion audit. Can be repeated.")
    finalize_baselines.add_argument("--workflow-evidence", action="append", type=Path, help="Workflow artifact or verifier report expected by completion audit. Can be repeated.")
    finalize_baselines.add_argument("--objective", help="Completion objective text to include in the audit.")
    finalize_baselines.add_argument("--out", type=Path, default=Path("data/reports/v1_1/baseline_finalization.json"))
    finalize_baselines.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/baseline_finalization.md"))

    assemble_v1_1 = subparsers.add_parser("assemble-v1-1", help="Assemble frozen V1 plus accepted V1.1 expansion samples.")
    assemble_v1_1.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1"))
    assemble_v1_1.add_argument("--expansion", action="append", type=Path, required=True, help="Expansion sample file or directory. Can be repeated.")
    assemble_v1_1.add_argument("--out", type=Path, default=Path("data/benchmark/v1_1"))
    assemble_v1_1.add_argument("--corpus-manifest", type=Path)
    assemble_v1_1.add_argument("--require-corpus", action="store_true")
    assemble_v1_1.add_argument("--audit", action="append", type=Path, help="V1.1 audit CSV/JSONL with keep/valid decisions. Can be repeated.")
    assemble_v1_1.add_argument("--require-audit-keep", action="store_true", help="Only include expansion samples marked keep/valid in --audit.")

    v1_1_audit = subparsers.add_parser("v1-1-audit-packet", help="Write V1.1 candidate diagnostics and human audit sheets.")
    v1_1_audit.add_argument("--candidate", action="append", type=Path, required=True, help="Candidate sample file or directory. Can be repeated.")
    v1_1_audit.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1"))
    v1_1_audit.add_argument("--out", type=Path, default=Path("data/reports/v1_1/audit_packet"))
    v1_1_audit.add_argument("--corpus-manifest", type=Path)
    v1_1_audit.add_argument("--require-corpus", action="store_true")
    v1_1_audit.add_argument("--limit", type=int)

    v1_1_readiness = subparsers.add_parser("v1-1-readiness", help="Check V1.1 sample expansion readiness gates.")
    v1_1_readiness.add_argument("samples", nargs="*", type=Path, help="V1.1 sample JSONL files. Defaults to --derived/*.jsonl.")
    v1_1_readiness.add_argument("--derived", type=Path, default=Path("data/benchmark/v1_1"))
    v1_1_readiness.add_argument("--base-derived", type=Path, default=Path("data/benchmark/v1"))
    v1_1_readiness.add_argument("--manifest", type=Path, help="Optional V1.1 assembly manifest to enforce release assembly gates.")
    v1_1_readiness.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/v1_1/corpus_manifest.jsonl"))
    v1_1_readiness.add_argument("--eval-dir", type=Path, default=Path("data/eval/v1_1"))
    v1_1_readiness.add_argument("--leaderboard", type=Path, default=Path("data/reports/v1_1/model_leaderboard.md"))
    v1_1_readiness.add_argument("--leaderboard-json", type=Path, default=Path("data/reports/v1_1/model_leaderboard.json"))
    v1_1_readiness.add_argument("--out", type=Path, default=Path("data/reports/v1_1/readiness.json"))
    v1_1_readiness.add_argument("--markdown-out", type=Path, default=Path("data/reports/v1_1/readiness.md"))
    v1_1_readiness.add_argument("--min-comment2context", type=int, default=80)
    v1_1_readiness.add_argument("--max-comment2context", type=int, default=100)
    v1_1_readiness.add_argument("--min-comment-cross-module", type=int, default=1)
    v1_1_readiness.add_argument("--min-trace2code", type=int, default=100)
    v1_1_readiness.add_argument("--min-trace-non-go-repos", type=int, default=1)
    v1_1_readiness.add_argument("--min-trace-languages", type=int, default=2)
    v1_1_readiness.add_argument("--min-trace-failure-types", type=int, default=2)

    report_v1_1 = subparsers.add_parser("report-v1-1", help="Generate a V1.1 release report from readiness and leaderboard artifacts.")
    report_v1_1.add_argument("--readiness", type=Path, default=Path("data/reports/v1_1/readiness.json"))
    report_v1_1.add_argument("--leaderboard-json", type=Path, default=Path("data/reports/v1_1/model_leaderboard.json"))
    report_v1_1.add_argument("--base-leaderboard-json", type=Path)
    report_v1_1.add_argument("--out", type=Path, default=Path("data/reports/v1_1/analysis.md"))
    report_v1_1.add_argument("--json-out", type=Path, default=Path("data/reports/v1_1/analysis.json"))

    completion_audit = subparsers.add_parser(
        "report-v1-1-completion-audit",
        help="Generate a prompt-to-artifact V1.1 completion audit from current release artifacts.",
    )
    completion_audit.add_argument("--readiness", type=Path, default=Path("data/reports/v1_1/readiness.json"))
    completion_audit.add_argument("--release-json", type=Path, default=Path("data/reports/v1_1/analysis.json"))
    completion_audit.add_argument("--baseline-status", type=Path, default=Path("data/reports/v1_1/baseline_status.json"))
    completion_audit.add_argument("--leaderboard-json", type=Path, default=Path("data/reports/v1_1/model_leaderboard.json"))
    completion_audit.add_argument("--doc", action="append", type=Path, help="Doc path expected for release readiness. Can be repeated.")
    completion_audit.add_argument("--workflow-evidence", action="append", type=Path, help="Workflow artifact or verifier report that must exist and, for JSON reports, be complete. Can be repeated.")
    completion_audit.add_argument("--objective", help="Completion objective text to include in the audit.")
    completion_audit.add_argument("--out", type=Path, default=Path("data/reports/v1_1/completion_audit.md"))
    completion_audit.add_argument("--json-out", type=Path, default=Path("data/reports/v1_1/completion_audit.json"))

    args = parser.parse_args(argv)
    api = GitHubAPI(token=args.token, cache_dir=args.cache_dir, max_rate_limit_sleep_seconds=args.max_rate_limit_sleep)

    if args.command == "releases":
        catalog = release_catalog()
        print(json.dumps(catalog, indent=2, ensure_ascii=False) if args.json else format_release_catalog(catalog))
        return 0
    if args.command == "manifest":
        targets = load_targets(args.targets)["primary"]
        count = write_manifest(api, targets, args.output)
        print(json.dumps({"wrote": count, "output": str(args.output), "authenticated": api.authenticated}, indent=2))
        return 0
    if args.command == "download-benchmark":
        versions = [release.id for release in CURRENT_BENCHMARK_RELEASES] if args.all_current else [args.version]
        results = [
            download_benchmark_release(
                version=version,
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
            for version in versions
        ]
        result = (
            results[0]
            if len(results) == 1
            else {
                "repo_id": args.repo_id,
                "local_dir": str(args.local_dir),
                "releases": results,
            }
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "merge-corpus-manifests":
        result = merge_corpus_manifests(
            local_dir=args.local_dir,
            versions=args.versions,
            output=args.out,
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
    if args.command == "crawl-review-comments":
        summary = crawl_review_comment_prs(
            api,
            args.repo,
            args.out,
            comments_per_repo=args.comments_per_repo,
            limit_prs=args.limit_prs,
            max_prs_per_repo=args.max_prs_per_repo,
            max_changed_files=args.max_changed_files,
            max_detail_commits=args.max_detail_commits,
            response_window_hours=args.response_window_hours,
            include_bots=args.include_bots,
            metadata_only=args.metadata_only,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.command == "derive":
        repos = args.repo or _repos_from_raw(args.raw)
        result = {repo: derive_repo(args.raw, repo, args.out, args.max_changed_files) for repo in repos}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "diagnose-comment2context":
        repos = args.repo or _repos_from_raw(args.raw)
        args.out.mkdir(parents=True, exist_ok=True)
        result = {
            repo: diagnose_comment2context_repo(
                args.raw,
                repo,
                args.out / f"{repo_slug(repo)}.jsonl",
                args.max_changed_files,
            )
            for repo in repos
        }
        (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    if args.command == "mine-edit2ripple":
        result = mine_edit2ripple(
            raw_dir=args.raw,
            out_dir=args.out,
            report_dir=args.report_out,
            audit_path=args.audit,
            corpus_manifest=args.corpus_manifest,
            require_corpus=args.require_corpus,
            require_gold_in_corpus=args.require_gold_in_corpus,
            repos=args.repo,
            min_changed_files=args.min_changed_files,
            max_changed_files=args.max_changed_files,
            max_gold_files=args.max_gold_files,
            max_candidates_per_pr=args.max_candidates_per_pr,
            audit_limit=args.audit_limit,
            limit_samples=args.limit_samples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "mine-edit2ripple-from-samples":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = mine_edit2ripple_from_samples(
            api=api,
            sample_paths=sample_paths,
            out_dir=args.out,
            report_dir=args.report_out,
            audit_path=args.audit,
            corpus_manifest=args.corpus_manifest,
            require_corpus=args.require_corpus,
            require_gold_in_corpus=args.require_gold_in_corpus,
            repos=args.repo,
            max_gold_files=args.max_gold_files,
            max_candidates_per_pr=args.max_candidates_per_pr,
            audit_limit=args.audit_limit,
            limit_prs=args.limit_prs,
            limit_samples=args.limit_samples,
            continue_on_error=not args.fail_fast,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["dropped"].get("fetch_error") and args.fail_fast else 0
    if args.command == "mine-edit2ripple-from-sample-commits":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = mine_edit2ripple_from_sample_commits(
            sample_paths=sample_paths,
            out_dir=args.out,
            report_dir=args.report_out,
            repos_dir=args.repos_dir,
            audit_path=args.audit,
            corpus_manifest=args.corpus_manifest,
            require_corpus=args.require_corpus,
            require_gold_in_corpus=args.require_gold_in_corpus,
            repos=args.repo,
            remote_base=args.remote_base,
            blob_filter=not args.no_blob_filter,
            max_gold_files=args.max_gold_files,
            max_candidates_per_pr=args.max_candidates_per_pr,
            audit_limit=args.audit_limit,
            limit_commits=args.limit_commits,
            limit_samples=args.limit_samples,
            continue_on_error=not args.fail_fast,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["dropped"].get("git_error") and args.fail_fast else 0
    if args.command == "report-edit2ripple-pilot":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = report_edit2ripple_pilot(
            sample_paths=sample_paths,
            out_path=args.out,
            json_out_path=args.json_out,
            corpus_manifest=args.corpus_manifest,
            audit_path=args.audit,
            min_samples=args.min_samples,
            min_valid_rate=args.min_valid_rate,
            max_test_only_ratio=args.max_test_only_ratio,
            max_candidates_per_pr=args.max_candidates_per_pr,
        )
        print(
            json.dumps(
                {key: result[key] for key in ("status", "total", "audited_total", "valid_rate", "gates")},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1 if result["status"] == "failed" else 0
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
    if args.command == "mine-abstention-counterfactuals":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = mine_abstention_counterfactuals(
            sample_paths=sample_paths,
            corpus_manifest_path=args.corpus_manifest,
            out_dir=args.out,
            audit_dir=args.audit_out,
            limit=args.limit,
            max_per_wrong_repo=args.max_per_wrong_repo,
            max_per_source_repo=args.max_per_source_repo,
            pairs_per_sample=args.pairs_per_sample,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["selected"] else 1
    if args.command == "mine-abstention-organic":
        raw_dirs = args.raw or [Path("data/raw_v1_1_token_probe")]
        result = mine_abstention_organic_candidates(
            raw_dirs=raw_dirs,
            out_dir=args.out,
            audit_dir=args.audit_out,
            repos=args.repo,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "crawl-abstention-issues":
        result = crawl_abstention_issues(
            api=api,
            repos=args.repo,
            out_dir=args.out,
            limit_per_repo=args.limit_per_repo,
            comments_per_issue=args.comments_per_issue,
            events_per_issue=args.events_per_issue,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["total_errors"] else 0
    if args.command == "crawl-abstention-issue-html":
        result = crawl_abstention_issue_html(
            repos=args.repo,
            out_dir=args.out,
            git_repos_dir=args.git_repos_dir,
            limit_per_repo=args.limit_per_repo,
            pages_per_keyword=args.pages_per_keyword,
            request_delay_seconds=args.request_delay_seconds,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["total_errors"] else 0
    if args.command == "backfill-abstention-issue-bases":
        raw_dirs = args.raw or [
            Path("data/abstention/raw"),
            Path("data/abstention/raw_html"),
            Path("data/abstention/raw_html2"),
        ]
        result = backfill_abstention_issue_base_commits(
            raw_dirs=raw_dirs,
            git_repos_dir=args.git_repos_dir,
            api=api,
            api_fallback=args.api_fallback,
            report_out=args.report_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] in {"complete", "no_updates"} else 1
    if args.command == "merge-abstention-audit-packets":
        packets = args.packet or [
            Path("data/abstention/audit/abstention_audit_packet.jsonl"),
            Path("data/abstention/audit/organic_abstention_audit_packet.jsonl"),
        ]
        result = merge_abstention_audit_packets(
            packet_paths=packets,
            out_jsonl=args.out,
            out_csv=args.csv_out,
            report_out=args.report_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready_for_manual_audit"] else 1
    if args.command == "report-abstention-crawling":
        prefiltered = args.prefiltered or [
            Path("data/abstention/candidates/counterfactual_wrong_repo_candidates.jsonl"),
            Path("data/abstention/candidates/organic_prefiltered.jsonl"),
        ]
        result = report_abstention_crawling_status(
            prefiltered_paths=prefiltered,
            audit_packet_path=args.audit_packet,
            out_path=args.out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["gates"]["audit_packets_schema_valid"] else 1
    if args.command == "export-abstention-clean":
        result = export_abstention_clean(
            audit_packet_path=args.audit_packet,
            audit_path=args.audit,
            out_dir=args.out,
            report_out=args.report_out,
            max_samples=args.max_samples,
            max_counterfactual_share=args.max_counterfactual_share,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["gates"]["schema_valid"] else 1
    if args.command == "report-abstention-audit":
        result = report_abstention_audit(
            audit_packet_path=args.audit_packet,
            audit_path=args.audit,
            out_path=args.out,
            max_samples=args.max_samples,
            max_counterfactual_share=args.max_counterfactual_share,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["gates"]["invalid_verdicts_zero"] and result["gates"]["missing_packet_ids_zero"] else 1
    if args.command == "prepare-abstention-audit-worklist":
        result = prepare_abstention_audit_worklist(
            audit_packet_path=args.audit_packet,
            out_csv=args.out_csv,
            out_jsonl=args.out_jsonl,
            report_out=args.report_out,
            target_size=args.target_size,
            max_counterfactual_share=args.max_counterfactual_share,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready_for_manual_audit" else 1
    if args.command == "report-abstention-completion":
        prefiltered = args.prefiltered or [
            Path("data/abstention/candidates/counterfactual_wrong_repo_candidates.jsonl"),
            Path("data/abstention/candidates/organic_prefiltered.jsonl"),
        ]
        clean_samples = args.clean_sample
        if clean_samples is None:
            default_clean = Path("data/benchmark/v2_abstention_pilot/abstention.jsonl")
            clean_samples = [default_clean] if default_clean.exists() else []
        result = report_abstention_completion_audit(
            prefiltered_paths=prefiltered,
            audit_packet_path=args.audit_packet,
            audit_path=args.audit,
            clean_sample_paths=clean_samples,
            worklist_report_path=args.worklist_report,
            crawling_report_path=args.crawling_report,
            out_path=args.out,
            max_samples=args.max_samples,
            max_counterfactual_share=args.max_counterfactual_share,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] in {"complete", "ready_for_manual_audit"} else 1
    if args.command == "apply-abstention-audit-verdicts":
        source_audits = args.source_audit or [Path("data/abstention/audit/abstention_manual_audit_worklist.csv")]
        result = apply_abstention_audit_verdicts(
            source_audit_path=source_audits,
            target_audit_path=args.target_audit,
            out_path=args.out,
            report_out=args.report_out,
            overwrite=args.overwrite,
            require_complete=args.require_complete,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready" else 1
    if args.command == "finalize-abstention-pilot":
        prefiltered = args.prefiltered or [
            Path("data/abstention/candidates/counterfactual_wrong_repo_candidates.jsonl"),
            Path("data/abstention/candidates/organic_prefiltered.jsonl"),
        ]
        result = finalize_abstention_pilot(
            prefiltered_paths=prefiltered,
            audit_packet_path=args.audit_packet,
            audit_path=args.audit,
            out_dir=args.out,
            report_out=args.report_out,
            audit_report_out=args.audit_report_out,
            pilot_report_out=args.pilot_report_out,
            completion_report_out=args.completion_report_out,
            worklist_report_path=args.worklist_report,
            crawling_report_path=args.crawling_report,
            max_samples=args.max_samples,
            max_counterfactual_share=args.max_counterfactual_share,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "complete" else 1
    if args.command == "shard-abstention-audit-worklist":
        result = shard_abstention_audit_worklist(
            worklist_path=args.worklist,
            out_dir=args.out_dir,
            report_out=args.report_out,
            shard_size=args.shard_size,
            priority=args.priority,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready_for_manual_audit" else 1
    if args.command == "report-abstention-shard-progress":
        shards = args.shard or sorted(Path("data/abstention/audit/shards").glob("*.csv"))
        result = report_abstention_shard_progress(shard_paths=shards, out_path=args.out)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] in {"complete", "in_progress"} else 1
    if args.command == "render-abstention-review-packets":
        shards = args.shard or sorted(Path("data/abstention/audit/shards").glob("*.csv"))
        result = render_abstention_review_packets(
            shard_paths=shards,
            out_dir=args.out_dir,
            report_out=args.report_out,
            query_limit=args.query_limit,
            evidence_limit=args.evidence_limit,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready_for_manual_audit" else 1
    if args.command == "write-abstention-audit-handoff-manifest":
        result = write_abstention_audit_handoff_manifest(
            audit_dir=args.audit_dir,
            out_path=args.out,
            finalization_report_path=args.finalization_report,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready_for_manual_audit" else 1
    if args.command == "report-abstention-pilot":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = report_abstention_pilot(
            sample_paths=sample_paths,
            out_path=args.out,
            audit_packet_path=args.audit_packet if args.audit_packet.exists() else None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["gates"]["schema_valid"] else 1
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
    if args.command == "backfill-git-raw":
        result = [
            backfill_git_raw(
                raw_dir=args.raw,
                repo=repo,
                repos_dir=args.repos_dir,
                limit_prs=args.limit_prs,
                repo_url_template=args.repo_url_template,
                timeout_seconds=args.timeout_seconds,
                infer_missing_base=args.infer_missing_base,
                max_inferred_commits=args.max_inferred_commits,
            )
            for repo in args.repo
        ]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item.get("status") != "ok" or item.get("errors") for item in result) else 0
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
        out_path = args.out or default_baseline_summary_path(args.ranker, args.candidate_filter)
        result = evaluate_corpus_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            dry_run=args.dry_run,
            candidate_filter=args.candidate_filter,
            ranker=args.ranker,
            progress=not args.no_progress,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-grep":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_grep_summary_path(args.pattern_mode, args.candidate_filter)
        result = evaluate_grep_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            candidate_filter=args.candidate_filter,
            pattern_mode=args.pattern_mode,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-selective-baseline":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        default_dir = Path("data/eval/v2_selective")
        default_stem = f"{args.derived.name}_{args.candidate_filter}_selective_{args.ranker}"
        out_path = args.out or default_dir / f"{default_stem}_summary.json"
        details_path = args.details or default_dir / f"{default_stem}_details.jsonl"
        sweep_path = args.sweep or default_dir / f"{default_stem}_threshold_sweep.jsonl"
        report_path = args.report or default_dir / f"{default_stem}_report.md"
        result = evaluate_selective_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=details_path,
            sweep_path=sweep_path,
            report_path=report_path,
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            candidate_filter=args.candidate_filter,
            ranker=args.ranker,
            progress=not args.no_progress,
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
    if args.command == "eval-agentic":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        keep_list = None if args.no_keep_list else args.keep_list
        out_path = args.out or default_agentic_summary_path(args.candidate_filter)
        result = evaluate_agentic_search(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            limit_samples=args.limit_samples,
            candidate_filter=args.candidate_filter,
            max_turns=args.max_turns,
            max_tool_calls_per_turn=args.max_tool_calls_per_turn,
            max_read_chars=args.max_read_chars,
            progress=not args.no_progress,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-closed-tool":
        result = evaluate_closed_tool_grep(
            sample_paths=args.samples or [Path("data/benchmark/v1_3_reviewed/samples.jsonl")],
            corpus_dir=args.corpus,
            out_path=args.out,
            details_path=args.details,
            markdown_out_path=args.markdown_out,
            keep_list=args.keep_list,
            limit_samples=args.limit_samples,
            max_tool_calls=args.max_tool_calls,
            max_read_tokens=args.max_read_tokens,
            max_read_tokens_per_file=args.max_read_tokens_per_file,
            max_grep_calls=args.max_grep_calls,
            final_k=args.final_k,
            grep_top_k=args.grep_top_k,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "evaluated": result["evaluated"],
                    "summary": result["paths"]["summary"],
                    "details": result["paths"]["details"],
                    "markdown": result["paths"]["markdown"],
                    "overall": result["metrics"].get("overall", {}),
                    "closed_tool_overall": result["closed_tool_metrics"].get("overall", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "eval-closed-tool-llm":
        result = evaluate_closed_tool_openai(
            sample_paths=args.samples or [Path("data/benchmark/v1_3_reviewed/samples.jsonl")],
            corpus_dir=args.corpus,
            out_path=args.out,
            details_path=args.details,
            markdown_out_path=args.markdown_out,
            keep_list=args.keep_list,
            limit_samples=args.limit_samples,
            model=args.model,
            max_tool_calls=args.max_tool_calls,
            max_model_turns=args.max_model_turns,
            max_read_tokens=args.max_read_tokens,
            max_read_tokens_per_file=args.max_read_tokens_per_file,
            final_k=args.final_k,
            grep_top_k=args.grep_top_k,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "evaluated": result["evaluated"],
                    "summary": result["paths"]["summary"],
                    "details": result["paths"]["details"],
                    "markdown": result["paths"]["markdown"],
                    "overall": result["metrics"].get("overall", {}),
                    "closed_tool_overall": result["closed_tool_metrics"].get("overall", {}),
                    "seed": {
                        "details": result["budget"].get("seed_details", ""),
                        "label": result["budget"].get("seed_label", ""),
                        "top_k": result["budget"].get("seed_top_k", 0),
                        "max_seed_tokens": result["budget"].get("max_seed_tokens", 0),
                    },
                    "failures": result.get("failures", []),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "eval-closed-tool-codex":
        result = evaluate_closed_tool_codex(
            sample_paths=args.samples or [Path("data/benchmark/v1_3_reviewed/samples.jsonl")],
            corpus_dir=args.corpus,
            out_path=args.out,
            details_path=args.details,
            markdown_out_path=args.markdown_out,
            keep_list=args.keep_list,
            limit_samples=args.limit_samples,
            model=args.model,
            codex_bin=args.codex_bin,
            work_root=args.codex_work_root,
            timeout_seconds=args.codex_timeout_seconds,
            max_tool_calls=args.max_tool_calls,
            max_model_turns=args.max_model_turns,
            max_read_tokens=args.max_read_tokens,
            max_read_tokens_per_file=args.max_read_tokens_per_file,
            final_k=args.final_k,
            grep_top_k=args.grep_top_k,
            force=args.force,
            seed_details=args.seed_details,
            seed_label=args.seed_label,
            seed_top_k=args.seed_top_k,
            max_seed_tokens=args.max_seed_tokens,
            max_seed_tokens_per_file=args.max_seed_tokens_per_file,
        )
        print(
            json.dumps(
                {
                    "evaluated": result["evaluated"],
                    "summary": result["paths"]["summary"],
                    "details": result["paths"]["details"],
                    "markdown": result["paths"]["markdown"],
                    "overall": result["metrics"].get("overall", {}),
                    "closed_tool_overall": result["closed_tool_metrics"].get("overall", {}),
                    "seed": {
                        "details": result["budget"].get("seed_details", ""),
                        "label": result["budget"].get("seed_label", ""),
                        "top_k": result["budget"].get("seed_top_k", 0),
                        "max_seed_tokens": result["budget"].get("max_seed_tokens", 0),
                    },
                    "failures": result.get("failures", []),
                    "protocol_violation_samples": result.get("protocol_violation_samples", 0),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "report-closed-tool-budget-curve":
        budgets = [int(item.strip()) for item in str(args.budgets).split(",") if item.strip()]
        result = report_closed_tool_budget_curve(
            details_path=args.details,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            budgets=budgets,
        )
        print(
            json.dumps(
                {
                    "budgets": result["budgets"],
                    "summary": result["paths"]["summary"],
                    "markdown": result["paths"]["markdown"],
                    "overall": [
                        {
                            "budget": row["budget"],
                            "metrics": row["metrics"].get("overall", {}),
                            "closed_tool": row["closed_tool_metrics"].get("overall", {}),
                        }
                        for row in result["rows"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "report-closed-tool-seed-intervention":
        arms = {}
        for raw_arm in args.arm:
            if "=" not in raw_arm:
                raise SystemExit(f"--arm must be label=path, got: {raw_arm}")
            label, path = raw_arm.split("=", 1)
            label = label.strip()
            if not label:
                raise SystemExit(f"--arm label is empty: {raw_arm}")
            arms[label] = Path(path)
        result = report_closed_tool_seed_intervention(
            control_details=args.control_details,
            arms=arms,
            keep_list=args.keep_list,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(
            json.dumps(
                {
                    "samples": result["sample_count"],
                    "summary": result["paths"]["summary"],
                    "markdown": result["paths"]["markdown"],
                    "rows": [
                        {
                            "label": row["label"],
                            "overall": row["metrics"].get("overall", {}),
                            "closed_tool": row["closed_tool_metrics"].get("overall", {}),
                            "seed": row["seed_metrics"],
                        }
                        for row in result["rows"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "eval-trajectories":
        result = evaluate_trajectories(
            derived=args.derived,
            trajectory_paths=args.trajectories,
            out_path=args.out,
            details_path=args.details,
            model_label=args.model_label,
            supporting_context_annotations=args.supporting_context_annotations,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-ranked-context":
        result = evaluate_ranked_context_as_trajectory(
            baseline_details=args.baseline_details,
            top_k=args.top_k,
            out_path=args.out,
            details_path=args.details,
            model_label=args.model_label,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run-openai-context-agent":
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
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if not result["failures"] else 1
    if args.command == "audit-trajectory-release":
        result = audit_strict_context_run(
            base=args.base,
            run_name=args.run_name,
            derived=args.derived,
            corpus_manifest=args.corpus_manifest,
            out_path=args.out,
            markdown_out=args.markdown_out,
            min_reads=args.min_reads,
            max_reads=args.max_reads,
            extra_scan_paths=args.extra_scan_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["verdict"] == "pass" else 1
    if args.command == "package-trajectory-release":
        include_paths = args.include or v1_3_openai_strict_context_include_paths(
            data_root=args.data_root,
            base=args.base,
            run_name=args.run_name,
        )
        result = package_trajectory_release(
            data_root=args.data_root,
            release_dir=args.release_dir,
            archive_name=args.archive_name,
            include_paths=include_paths,
            checksum_path_in_release=args.checksum_path_in_release,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "prepare-trajectory-runs":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        result = prepare_trajectory_runs(
            derived=args.derived,
            out_dir=args.out_dir,
            sample_paths=sample_paths,
            sample_ids=args.sample_id,
            limit_samples=args.limit_samples,
            model_label=args.model_label,
            overwrite_logs=args.overwrite_logs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "record-trajectory-step":
        log_path = args.log or (Path(os.environ["ARB_TRAJECTORY_LOG"]) if os.environ.get("ARB_TRAJECTORY_LOG") else None)
        sample_id = args.sample_id or os.environ.get("ARB_SAMPLE_ID", "")
        if not log_path:
            parser.error("record-trajectory-step requires --log or ARB_TRAJECTORY_LOG.")
        if not sample_id:
            parser.error("record-trajectory-step requires --sample-id or ARB_SAMPLE_ID.")
        row = record_trajectory_step(
            log_path=log_path,
            sample_id=sample_id,
            path=args.path,
            step=args.step,
            tool=args.tool,
            start_line=args.start_line,
            end_line=args.end_line,
            kind=args.kind,
            symbol=args.symbol,
            content_hash=args.content_hash,
            repo_root=args.repo_root,
            is_final_context=args.is_final_context,
            is_utilized_context=args.is_utilized_context,
            run_id=args.run_id,
            model_label=args.model_label,
        )
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-embedding":
        try:
            paths = resolve_embedding_eval_paths(
                model=args.model,
                version=args.version,
                model_label=args.model_label,
                derived=args.derived,
                corpus=args.corpus,
                out=args.out,
                details=args.details,
                cache=args.cache,
                shared_text_cache=args.shared_text_cache,
                no_shared_text_cache=args.no_shared_text_cache,
                keep_list=args.keep_list,
                no_keep_list=args.no_keep_list,
                candidate_filter=args.candidate_filter,
            )
        except ValueError as exc:
            parser.error(str(exc))
        sample_paths = args.samples or sample_paths_from_derived(paths.derived)
        result = evaluate_embedding_baseline(
            sample_paths=sample_paths,
            corpus_dir=paths.corpus,
            model_name=args.model,
            out_path=paths.out,
            details_path=paths.details,
            keep_list=paths.keep_list,
            cache_dir=paths.cache,
            shared_text_cache=paths.shared_text_cache,
            sample_ids=load_sample_id_file(args.sample_id_file) if args.sample_id_file else None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            device=args.device,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            normalize_embeddings=not args.no_normalize,
            trust_remote_code=args.trust_remote_code,
            progress=not args.no_progress,
            candidate_filter=args.candidate_filter,
            resume_details=args.resume_details,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval-selective-embedding":
        try:
            paths = resolve_embedding_eval_paths(
                model=args.model,
                version=args.version,
                model_label=args.model_label,
                derived=args.derived,
                corpus=args.corpus,
                out=args.out,
                details=args.details,
                cache=args.cache,
                shared_text_cache=args.shared_text_cache,
                no_shared_text_cache=args.no_shared_text_cache,
                keep_list=args.keep_list,
                no_keep_list=args.no_keep_list,
                candidate_filter=args.candidate_filter,
            )
        except ValueError as exc:
            parser.error(str(exc))
        out_path = args.out or selective_embedding_summary_path(paths.out)
        details_path = args.details or default_baseline_details_path(out_path)
        sweep_path = args.sweep or out_path.with_name(out_path.stem.removesuffix("_summary") + "_threshold_sweep.jsonl")
        report_path = args.report or out_path.with_name(out_path.stem.removesuffix("_summary") + "_report.md")
        sample_paths = args.samples or sample_paths_from_derived(paths.derived)
        result = evaluate_selective_embedding_baseline(
            sample_paths=sample_paths,
            corpus_dir=paths.corpus,
            model_name=args.model,
            out_path=out_path,
            details_path=details_path,
            sweep_path=sweep_path,
            report_path=report_path,
            keep_list=paths.keep_list,
            cache_dir=paths.cache,
            shared_text_cache=paths.shared_text_cache,
            sample_ids=load_sample_id_file(args.sample_id_file) if args.sample_id_file else None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
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
            max_request_chars=args.max_request_chars,
        )
        result = evaluate_embedding_baseline(
            sample_paths=sample_paths,
            corpus_dir=args.corpus,
            model_name=args.model,
            out_path=out_path,
            details_path=args.details or default_baseline_details_path(out_path),
            keep_list=keep_list,
            cache_dir=args.cache or default_embedding_cache_dir(args.model, root=Path("data/embeddings/v1")),
            shared_text_cache=args.shared_text_cache,
            sample_ids=load_sample_id_file(args.sample_id_file) if args.sample_id_file else None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            query_input_type=args.query_input_type or None,
            passage_input_type=args.passage_input_type or None,
            normalize_embeddings=not args.no_normalize,
            embedder=embedder,
            progress=not args.no_progress,
            candidate_filter=args.candidate_filter,
            resume_details=args.resume_details,
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
            required_baselines=args.required_baseline,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["contains_required_baselines"] else 1
    if args.command == "report-v2-positive":
        result = report_v2_positive_leaderboard(
            core_eval_dir=args.core_eval_dir,
            extension_eval_dir=args.extension_eval_dir,
            core_bcy_path=args.core_bcy,
            extension_bcy_path=args.extension_bcy,
            out_path=args.out,
            json_out_path=args.json_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-selective-cv":
        result = report_selective_group_cv(
            detail_paths={
                "lexical": args.lexical_details,
                "BM25": args.bm25_details,
                "Jina-0.5B": args.jina_details,
            },
            out_path=args.out,
            json_out_path=args.json_out,
            fold_count=args.folds,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "report-dataset-validity":
        result = report_dataset_validity(
            samples_path=args.samples,
            corpus_manifest_path=args.corpus_manifest,
            validation_path=args.validation,
            audit_paths=args.audit,
            eval_dirs=args.eval_dir,
            split_details_dirs=args.split_details_dir,
            out_dir=args.out_dir,
            audit_packet_size=args.audit_packet_size,
            valid_threshold=args.valid_threshold,
            min_task_spread=args.min_task_spread,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-context-cost":
        result = report_context_acquisition_cost(
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            late_hit_threshold=args.late_hit_threshold,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "report-rank-fusion":
        result = report_rank_fusion(
            eval_dir=args.eval_dir,
            out_dir=args.out_dir,
            report_out=args.out,
            json_out=args.json_out,
            rrf_k=args.rrf_k,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "report-bcy-curve":
        budgets = [int(item.strip()) for item in str(args.budgets).split(",") if item.strip()]
        coverage_thresholds = [
            int(item.strip()) for item in str(args.coverage_thresholds).split(",") if item.strip()
        ]
        result = report_bcy_budget_curve(
            corpus_manifest_path=args.corpus_manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            budgets=budgets,
            coverage_thresholds=coverage_thresholds,
        )
        print(json.dumps({"runs": len(result["runs"]), "json": str(args.out), "markdown": str(args.markdown_out)}, indent=2))
        return 0

    if args.command == "report-pes-calibration":
        seeded_details = args.seeded_details or [Path("data/eval/v1_4/pes_calibration_seeded_gpt54_details.jsonl")]
        result = report_pes_calibration(
            packet_path=args.packet,
            control_details_path=args.control_details,
            seeded_details_paths=seeded_details,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(
            json.dumps(
                {
                    "paired_samples": result["counts"]["paired_samples"],
                    "missing_seeded": result["counts"]["missing_seeded"],
                    "json": str(args.out),
                    "markdown": str(args.markdown_out),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-layered-leaderboard":
        result = report_layered_leaderboard(
            bcy_report_path=args.bcy_report,
            context_selection_path=args.context_selection,
            context_cost_path=args.context_cost,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps({"rows": len(result["rank_leaderboard"]), "json": str(args.out), "markdown": str(args.markdown_out)}, indent=2))
        return 0

    if args.command == "report-cae-validity":
        result = report_cae_validity(
            bcy_report_path=args.bcy_report,
            context_cost_path=args.context_cost,
            pes_calibration_path=args.pes_calibration,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            budget=args.budget,
        )
        print(
            json.dumps(
                {
                    "bcy_correlation_rows": len(result["bcy_correlations"]),
                    "budget": result["budget"],
                    "json": str(args.out),
                    "markdown": str(args.markdown_out),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-agentic-relevance":
        result = report_agentic_relevance_taxonomy(
            samples_path=args.samples,
            corpus_manifest_path=args.corpus_manifest,
            out_dir=args.out_dir,
            budget=args.budget,
        )
        print(
            json.dumps(
                {
                    "samples": result["counts"]["samples"],
                    "by_primary_relevance_type": result["counts"]["by_primary_relevance_type"],
                    "json": result["paths"]["json"],
                    "markdown": result["paths"]["markdown"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report-rank-analysis":
        result = report_rank_analysis(
            eval_dir=args.eval_dir,
            samples_path=args.samples,
            out_path=args.out,
            json_out_path=args.json_out,
            candidate_filter=args.candidate_filter,
            top_examples=args.top_examples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-v1-2":
        result = validate_v1_2_benchmark(
            derived=args.derived,
            corpus_manifest_path=None if args.no_corpus else args.corpus_manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 1
    if args.command == "derive-v1-3-blocks":
        result = derive_v1_3_blocks(
            base_derived=args.base_derived,
            corpus_manifest_path=args.corpus_manifest,
            out_dir=args.out,
            report_path=args.report_out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "write-v1-3-span-worklist":
        result = write_v1_3_span_worklist(
            derived=args.derived,
            corpus_manifest_path=args.corpus_manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            jsonl_out_path=args.jsonl_out,
            task_type=args.task_type,
            max_candidates_per_file=args.max_candidates_per_file,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-v1-3":
        result = validate_v1_3_benchmark(
            derived=args.derived,
            corpus_manifest_path=None if args.no_corpus else args.corpus_manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            require_full_spans=not args.allow_partial_spans,
            require_full_blocks=not args.allow_partial_blocks,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 1
    if args.command == "merge-v1-2-annotations":
        result = merge_manual_annotations(
            base_derived=args.base_derived,
            annotations_path=args.annotations,
            out_dir=args.out,
            report_path=args.report_out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-context-pollution":
        result = report_context_pollution(
            eval_dir=args.eval_dir,
            out_path=args.out,
            json_out_path=args.json_out,
            candidate_filter=args.candidate_filter,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-span-subset":
        result = report_span_subset(
            eval_dir=args.eval_dir,
            out_path=args.out,
            json_out_path=args.json_out,
            candidate_filter=args.candidate_filter,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report-runtime-cache":
        result = report_runtime_cache(
            eval_dir=args.eval_dir,
            out_path=args.out,
            json_out_path=args.json_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "check-baseline-summaries":
        summary_paths = list(args.summaries)
        for eval_dir in args.eval_dir or []:
            summary_paths.extend(sorted(eval_dir.glob("*_summary.json")))
        expected_samples = args.expected_samples
        expected_sample_ids = None
        if args.derived.exists():
            derived_rows = [row for path in sample_paths_from_derived(args.derived) for row in read_jsonl(path)]
            expected_sample_ids = {str(row.get("id")) for row in derived_rows if row.get("id")}
            if expected_samples is None:
                expected_samples = len(derived_rows)
        if expected_samples is None:
            print("--expected-samples is required when --derived does not exist", file=sys.stderr)
            return 2
        result = check_required_baseline_summaries(
            summary_paths=summary_paths,
            expected_samples=expected_samples,
            expected_baselines=args.required_baseline or None,
            require_details=not args.no_require_details,
            expected_sample_ids=expected_sample_ids,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-status":
        summary_paths = list(args.summaries)
        for eval_dir in args.eval_dir or []:
            summary_paths.extend(sorted(eval_dir.glob("*_summary.json")))
        expected_samples = args.expected_samples
        expected_sample_ids = None
        if args.derived.exists():
            derived_rows = [row for path in sample_paths_from_derived(args.derived) for row in read_jsonl(path)]
            expected_sample_ids = {str(row.get("id")) for row in derived_rows if row.get("id")}
            if expected_samples is None:
                expected_samples = len(derived_rows)
        if expected_samples is None:
            print("--expected-samples is required when --derived does not exist", file=sys.stderr)
            return 2
        result = write_v1_1_baseline_status_report(
            summary_paths=summary_paths,
            expected_samples=expected_samples,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            expected_baselines=args.required_baseline or None,
            require_details=not args.no_require_details,
            expected_sample_ids=expected_sample_ids,
            eval_dirs=args.eval_dir or [],
            shard_commands_path=args.shard_commands,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-external-runner-preflight":
        result = write_v1_1_external_runner_preflight_report(
            baseline_status_path=args.baseline_status,
            return_acceptance_path=args.return_acceptance,
            return_manifest_path=args.return_manifest,
            transfer_manifest_verify_path=args.transfer_manifest_verify,
            handoff_verify_path=args.handoff_verify,
            transfer_bundle_verify_path=args.transfer_bundle_verify,
            copy_packet_path=args.copy_packet,
            full_runner_path=args.full_runner,
            gpu_runner_path=args.gpu_runner,
            voyage_runner_path=args.voyage_runner,
            return_bundle_script_path=args.return_bundle_script,
            gpu_return_bundle_script_path=args.gpu_return_bundle_script,
            voyage_return_bundle_script_path=args.voyage_return_bundle_script,
            gpu_return_manifest_path=args.gpu_return_manifest,
            voyage_return_manifest_path=args.voyage_return_manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v1-1-external-runner-failfast-smoke":
        result = write_v1_1_external_runner_failfast_smoke_report(
            preflight_path=args.preflight,
            full_runner_path=args.full_runner,
            gpu_runner_path=args.gpu_runner,
            voyage_runner_path=args.voyage_runner,
            return_bundle_script_path=args.return_bundle_script,
            gpu_return_bundle_script_path=args.gpu_return_bundle_script,
            voyage_return_bundle_script_path=args.voyage_return_bundle_script,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            timeout_seconds=args.timeout_seconds,
            allow_ready_runs=args.allow_ready_runs,
            cwd=Path.cwd(),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-summary-from-details":
        expected_samples = args.expected_samples
        expected_sample_ids = None
        if args.derived.exists():
            derived_rows = [row for path in sample_paths_from_derived(args.derived) for row in read_jsonl(path)]
            expected_sample_ids = {str(row.get("id")) for row in derived_rows if row.get("id")}
            if expected_samples is None:
                expected_samples = len(derived_rows)
        result = write_v1_1_summary_from_details(
            details_path=args.details,
            out_path=args.out,
            model=args.model,
            mode=args.mode,
            candidate_filter=args.candidate_filter,
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
            cache_dir=args.cache_dir,
            shared_text_cache=args.shared_text_cache,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v1-1-merge-details":
        expected_samples = args.expected_samples
        expected_sample_ids = None
        if args.derived.exists():
            derived_rows = [row for path in sample_paths_from_derived(args.derived) for row in read_jsonl(path)]
            expected_sample_ids = {str(row.get("id")) for row in derived_rows if row.get("id")}
            if expected_samples is None:
                expected_samples = len(derived_rows)
        result = write_v1_1_merged_details(
            details_paths=args.details,
            out_path=args.out,
            expected_samples=expected_samples,
            expected_sample_ids=expected_sample_ids,
            candidate_filter=args.candidate_filter,
            allow_incomplete=args.allow_incomplete,
            report_out_path=args.report_out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-write-sample-shards":
        result = write_v1_1_sample_id_shards(
            derived=args.derived,
            out_dir=args.out_dir,
            shard_count=args.shard_count,
            prefix=args.prefix,
            manifest_out_path=args.manifest_out or args.out_dir / "manifest.json",
            markdown_out_path=args.markdown_out,
            allow_empty_shards=args.allow_empty_shards,
            assignment_strategy=args.strategy,
            corpus_manifest_path=args.corpus_manifest,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-shard-commands":
        result = write_v1_1_baseline_shard_commands(
            handoff_path=args.handoff,
            sample_shards_path=args.sample_shards,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            report_dir=args.report_dir,
            use_shard_caches=not args.shared_caches,
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "markdown_out": str(args.markdown_out) if args.markdown_out else None,
                    "complete": result["complete"],
                    "baselines": len(result["baselines"]),
                    "shard_count": result["shard_count"],
                    "missing_shard_files": len(result["missing_shard_files"]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-run-script":
        result = write_v1_1_baseline_run_script(
            shard_commands_path=args.shard_commands,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            include_runtime_checks=not args.no_runtime_checks,
            baseline_filters=args.baseline,
            transfer_manifest_path=args.transfer_manifest,
            return_manifest_path=args.return_manifest,
            return_manifest_markdown_path=args.return_manifest_markdown,
            return_files_path=args.return_files,
            return_bundle_script_path=args.return_bundle_script,
            include_return_shard_artifacts=args.include_return_shard_artifacts,
            include_return_caches=args.include_return_caches,
            finalization_path=args.finalization,
            finalization_markdown_path=args.finalization_markdown,
            return_acceptance_path=args.return_acceptance,
            return_acceptance_markdown_path=args.return_acceptance_markdown,
            completion_json_path=args.completion_json,
            workflow_evidence_paths=args.workflow_evidence,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-handoff":
        result = write_v1_1_baseline_handoff(
            derived=args.derived,
            base_derived=args.base_derived,
            corpus=args.corpus,
            corpus_manifest=args.corpus_manifest,
            assembly_manifest=args.assembly_manifest,
            eval_dir=args.eval_dir,
            cache_root=args.cache_root,
            report_dir=args.report_dir,
            leaderboard_path=args.leaderboard,
            leaderboard_json_path=args.leaderboard_json,
            readiness_path=args.readiness,
            readiness_markdown_path=args.readiness_markdown,
            release_path=args.release,
            release_json_path=args.release_json,
            completion_path=args.completion,
            completion_json_path=args.completion_json,
            baseline_status_path=args.baseline_status,
            baseline_status_markdown_path=args.baseline_status_markdown,
            baseline_preflight_path=args.baseline_preflight,
            handoff_verify_path=args.handoff_verify,
            handoff_verify_markdown_path=args.handoff_verify_markdown,
            finalization_path=args.finalization,
            finalization_markdown_path=args.finalization_markdown,
            shard_commands_path=args.shard_commands,
            return_manifest_path=args.return_manifest,
            return_manifest_markdown_path=args.return_manifest_markdown,
            return_files_path=args.return_files,
            return_acceptance_path=args.return_acceptance,
            return_acceptance_markdown_path=args.return_acceptance_markdown,
            include_shard_artifacts=args.include_shard_artifacts,
            include_caches=args.include_caches,
            auto_merge_shards=args.auto_merge_shards,
            workflow_evidence_paths=args.workflow_evidence,
            transfer_manifest_path=args.transfer_manifest,
            transfer_manifest_markdown_path=args.transfer_manifest_markdown,
            transfer_manifest_verify_path=args.transfer_manifest_verify,
            transfer_manifest_verify_markdown_path=args.transfer_manifest_verify_markdown,
            transfer_files_path=args.transfer_files,
            transfer_bundle_path=args.transfer_bundle,
            transfer_bundle_checksum_path=args.transfer_bundle_checksum,
            transfer_bundle_archive_members_path=args.transfer_bundle_archive_members,
            transfer_bundle_report_path=args.transfer_bundle_report,
            transfer_bundle_markdown_path=args.transfer_bundle_markdown,
            transfer_bundle_verify_path=args.transfer_bundle_verify,
            transfer_bundle_verify_markdown_path=args.transfer_bundle_verify_markdown,
            transfer_unpack_script_path=args.transfer_unpack_script,
            transfer_unpack_script_markdown_path=args.transfer_unpack_script_markdown,
            transfer_unpack_destination=args.transfer_unpack_destination,
            transfer_unpack_transfer_verify_path=args.transfer_unpack_transfer_verify,
            transfer_unpack_transfer_verify_markdown_path=args.transfer_unpack_transfer_verify_markdown,
            transfer_unpack_handoff_verify_path=args.transfer_unpack_handoff_verify,
            transfer_unpack_handoff_verify_markdown_path=args.transfer_unpack_handoff_verify_markdown,
            transfer_include_paths=args.transfer_include,
            base_leaderboard_json_path=args.base_leaderboard_json,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps({"out": str(args.out), "markdown_out": str(args.markdown_out), "jobs": len(result["jobs"])}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v1-1-verify-handoff":
        result = verify_v1_1_baseline_handoff(
            handoff_path=args.handoff,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-transfer-manifest":
        result = write_v1_1_baseline_transfer_manifest(
            handoff_path=args.handoff,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            files_out_path=args.files_out,
            include_paths=args.include or [],
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "markdown_out": str(args.markdown_out),
                    "files_out": str(args.files_out) if args.files_out else None,
                    "complete": result["complete"],
                    "file_count": result["file_count"],
                    "chunk_files": result["chunk_files"],
                    "chunk_count": result["chunk_count"],
                    "included_files": result["included_file_count"],
                    "missing_files": len(result["missing_files"]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-transfer-bundle":
        result = create_v1_1_baseline_transfer_bundle(
            manifest_path=args.manifest,
            bundle_path=args.bundle,
            checksum_path=args.checksum,
            archive_members_path=args.archive_members,
            bundle_files_path=args.bundle_files,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            verify_out_path=args.verify_out,
            verify_markdown_out_path=args.verify_markdown_out,
            compression=args.compression,
            test_compression=not args.no_test_compression,
            work_dir=args.repo_root,
        )
        copy_packet_result = None
        if args.copy_packet_out or args.copy_packet_markdown_out:
            if args.out is None:
                print("--out is required when refreshing a copy packet from v1-1-baseline-transfer-bundle", file=sys.stderr)
                return 2
            copy_packet_result = write_v1_1_external_runner_copy_packet(
                transfer_bundle_report_path=args.out,
                out_path=args.copy_packet_out or Path("data/reports/v1_1/external_runner_copy_packet_v19.json"),
                markdown_out_path=args.copy_packet_markdown_out,
                bundle_path=args.bundle,
                checksum_path=args.checksum,
            )
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "bundle": result["bundle"],
                    "checksum": result["checksum"],
                    "archive_members": result["archive_members"],
                    "bundle_files": result["bundle_files"],
                    "file_count": result["file_count"],
                    "failed_checks": result["failed_checks"],
                    "sha256": (result.get("bundle_fingerprint") or {}).get("sha256"),
                    "copy_packet": {
                        "out": str(args.copy_packet_out or Path("data/reports/v1_1/external_runner_copy_packet_v19.json")),
                        "markdown_out": str(args.copy_packet_markdown_out) if args.copy_packet_markdown_out else None,
                        "complete": copy_packet_result.get("complete") if copy_packet_result else None,
                        "failed_checks": copy_packet_result.get("failed_checks") if copy_packet_result else None,
                    }
                    if copy_packet_result
                    else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] else 1
    if args.command == "v1-1-external-runner-copy-packet":
        result = write_v1_1_external_runner_copy_packet(
            transfer_bundle_report_path=args.transfer_bundle_report,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            unpack_script_path=args.unpack_script,
            bundle_path=args.bundle,
            checksum_path=args.checksum,
            bundle_report_markdown_path=args.bundle_report_markdown,
            destination=args.destination,
            full_runner_path=args.full_runner,
            gpu_runner_path=args.gpu_runner,
            voyage_runner_path=args.voyage_runner,
            local_apply_script_path=args.local_apply_script,
            completion_json_path=args.completion_json,
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "markdown_out": str(args.markdown_out) if args.markdown_out else None,
                    "complete": result["complete"],
                    "bundle_sha256": result["bundle_sha256"],
                    "bundle_size_bytes": result["bundle_size_bytes"],
                    "transfer_file_count": result["transfer_file_count"],
                    "failed_checks": result["failed_checks"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] else 1
    if args.command == "v1-1-verify-transfer-manifest":
        result = verify_v1_1_baseline_transfer_manifest(
            manifest_path=args.manifest,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-verify-transfer-bundle":
        result = verify_v1_1_baseline_transfer_bundle(
            bundle_path=args.bundle,
            manifest_path=args.manifest,
            checksum_path=args.checksum,
            archive_members_path=args.archive_members,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            test_compression=not args.no_test_compression,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-transfer-unpack-script":
        result = write_v1_1_baseline_transfer_unpack_script(
            bundle_path=args.bundle,
            checksum_path=args.checksum,
            manifest_path=args.manifest,
            handoff_path=args.handoff,
            destination=args.destination,
            transfer_verify_path=args.transfer_verify,
            transfer_verify_markdown_path=args.transfer_verify_markdown,
            handoff_verify_path=args.handoff_verify,
            handoff_verify_markdown_path=args.handoff_verify_markdown,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-verify-return-bundle":
        result = verify_v1_1_baseline_return_bundle(
            bundle_path=args.bundle,
            return_manifest_path=args.return_manifest,
            checksum_path=args.checksum,
            archive_members_path=args.archive_members,
            bundle_files_path=args.bundle_files,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            test_compression=not args.no_test_compression,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-return-manifest":
        result = write_v1_1_baseline_return_manifest(
            handoff_path=args.handoff,
            shard_commands_path=args.shard_commands,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            files_out_path=args.files_out,
            include_shard_artifacts=args.include_shard_artifacts,
            include_caches=args.include_caches,
            baseline_filters=args.baseline,
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "markdown_out": str(args.markdown_out) if args.markdown_out else None,
                    "files_out": str(args.files_out) if args.files_out else None,
                    "complete": result["complete"],
                    "artifacts_complete": result["artifacts_complete"],
                    "file_count": result["file_count"],
                    "required_file_count": result["required_file_count"],
                    "missing_required_files": len(result["missing_required_files"]),
                    "missing_optional_files": len(result["missing_optional_files"]),
                    "requested_baselines": result["requested_baselines"],
                    "selected_baselines": result["selected_baselines"],
                    "missing_requested_baselines": result["missing_requested_baselines"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] and (result["artifacts_complete"] or not args.require_existing) else 1
    if args.command == "v1-1-baseline-return-acceptance":
        result = write_v1_1_baseline_return_acceptance(
            handoff_path=args.handoff,
            return_manifest_path=args.return_manifest,
            completion_json_path=args.completion_json,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "markdown_out": str(args.markdown_out) if args.markdown_out else None,
                    "complete": result["complete"],
                    "external_baselines": result["external_baselines"],
                    "required_return_file_count": result["required_return_file_count"],
                    "missing_required_file_count": result["missing_required_file_count"],
                    "return_manifest_generated_at": result["return_manifest_generated_at"],
                    "completion_audit_generated_at": result["completion_audit_generated_at"],
                    "completion_audit_overall_status": result["completion_audit_overall_status"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if result["complete"] or not args.require_complete else 1
    if args.command == "v1-1-baseline-return-bundle-script":
        result = write_v1_1_baseline_return_bundle_script(
            handoff_path=args.handoff,
            shard_commands_path=args.shard_commands,
            return_manifest_path=args.return_manifest,
            return_manifest_markdown_path=args.return_manifest_markdown,
            return_files_path=args.return_files,
            bundle_path=args.bundle,
            bundle_files_path=args.bundle_files,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            include_shard_artifacts=args.include_shard_artifacts,
            include_caches=args.include_caches,
            compression=args.compression,
            test_bundle=not args.no_test_bundle,
            baseline_filters=args.baseline,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-baseline-apply-return-bundle-script":
        result = write_v1_1_baseline_apply_return_bundle_script(
            handoff_path=args.handoff,
            shard_commands_path=args.shard_commands,
            return_manifest_path=args.return_manifest,
            return_manifest_markdown_path=args.return_manifest_markdown,
            return_files_path=args.return_files,
            bundle_path=args.bundle,
            checksum_path=args.checksum,
            finalization_path=args.finalization,
            finalization_markdown_path=args.finalization_markdown,
            return_acceptance_path=args.return_acceptance,
            return_acceptance_markdown_path=args.return_acceptance_markdown,
            completion_json_path=args.completion_json,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            include_shard_artifacts=args.include_shard_artifacts,
            include_caches=args.include_caches,
            test_bundle=not args.no_test_bundle,
            auto_merge_shards=not args.no_auto_merge_shards,
            run_finalization=not args.no_finalization,
            baseline_filters=args.baseline,
            workflow_evidence_paths=args.workflow_evidence,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "v1-1-finalize-baselines":
        result = write_v1_1_baseline_finalization(
            handoff_path=args.handoff,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            shard_commands_path=args.shard_commands,
            return_manifest_path=args.return_manifest,
            return_manifest_markdown_path=args.return_manifest_markdown,
            return_files_path=args.return_files,
            include_shard_artifacts=args.include_shard_artifacts,
            include_caches=args.include_caches,
            auto_merge_shards=args.auto_merge_shards,
            docs=args.doc,
            workflow_evidence_paths=args.workflow_evidence,
            objective=args.objective or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    if args.command == "assemble-v1-1":
        base_sample_paths = sample_paths_from_derived(args.base_derived)
        result = assemble_v1_1_benchmark(
            base_sample_paths=base_sample_paths,
            expansion_sources=args.expansion,
            out_dir=args.out,
            corpus_manifest_path=args.corpus_manifest,
            require_corpus=args.require_corpus,
            audit_paths=args.audit,
            require_audit_keep=args.require_audit_keep,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["dropped"] else 0
    if args.command == "v1-1-audit-packet":
        base_sample_paths = sample_paths_from_derived(args.base_derived) if args.base_derived.exists() else []
        result = write_v1_1_audit_packet(
            candidate_sources=args.candidate,
            out_dir=args.out,
            base_sample_paths=base_sample_paths,
            corpus_manifest_path=args.corpus_manifest,
            require_corpus=args.require_corpus,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v1-1-readiness":
        sample_paths = args.samples or sample_paths_from_derived(args.derived)
        base_sample_paths = sample_paths_from_derived(args.base_derived) if args.base_derived.exists() else []
        result = check_v1_1_readiness(
            sample_paths=sample_paths,
            base_sample_paths=base_sample_paths,
            corpus_manifest_path=args.corpus_manifest,
            assembly_manifest_path=args.manifest,
            eval_dir=args.eval_dir,
            leaderboard_path=args.leaderboard,
            leaderboard_json_path=args.leaderboard_json,
            out_path=args.out,
            markdown_out_path=args.markdown_out,
            min_comment2context=args.min_comment2context,
            max_comment2context=args.max_comment2context,
            min_comment_cross_module=args.min_comment_cross_module,
            min_trace2code=args.min_trace2code,
            min_trace_non_go_repos=args.min_trace_non_go_repos,
            min_trace_languages=args.min_trace_languages,
            min_trace_failure_types=args.min_trace_failure_types,
        )
        print(json.dumps({"ready": result["ready"], "blocking_gates": result["blocking_gates"], **result["summary"]}, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 1
    if args.command == "report-v1-1":
        result = report_v1_1_release(
            readiness_path=args.readiness,
            leaderboard_json_path=args.leaderboard_json,
            base_leaderboard_json_path=args.base_leaderboard_json,
            out_path=args.out,
            json_out_path=args.json_out,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready" else 1
    if args.command == "report-v1-1-completion-audit":
        result = report_v1_1_completion_audit(
            readiness_path=args.readiness,
            release_json_path=args.release_json,
            baseline_status_path=args.baseline_status,
            leaderboard_json_path=args.leaderboard_json,
            out_path=args.out,
            json_out_path=args.json_out,
            docs=args.doc,
            workflow_evidence_paths=args.workflow_evidence,
            objective=args.objective or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["overall_status"] == "complete" else 1
    return 2


def default_baseline_details_path(out_path: Path) -> Path:
    if out_path.stem.endswith("_summary"):
        return out_path.with_name(f"{out_path.stem.removesuffix('_summary')}_details.jsonl")
    return out_path.with_suffix(".details.jsonl")


def format_release_catalog(catalog: dict) -> str:
    lines = [
        f"Dataset: {catalog['dataset_repo']}",
        "",
        "Current benchmark releases (`arb download-benchmark --all`):",
    ]
    for release in catalog["current"]:
        lines.append(f"  {release['id']:<34} {release['samples']:>3} samples  {release['description']}")
    lines.extend(["", "Auxiliary selective-evaluation inputs:"])
    for release in catalog["auxiliary"]:
        lines.append(f"  {release['id']:<34}              {release['description']}")
    return "\n".join(lines)


@dataclass(frozen=True)
class EmbeddingEvalPaths:
    model_label: str
    derived: Path
    corpus: Path
    out: Path
    details: Path
    cache: Path
    shared_text_cache: Path | None
    keep_list: Path | None


def resolve_embedding_eval_paths(
    *,
    model: str,
    version: str | None,
    model_label: str | None,
    derived: Path | None,
    corpus: Path | None,
    out: Path | None,
    details: Path | None,
    cache: Path | None,
    shared_text_cache: Path | None,
    no_shared_text_cache: bool,
    keep_list: Path | None,
    no_keep_list: bool,
    candidate_filter: str,
) -> EmbeddingEvalPaths:
    if no_shared_text_cache and shared_text_cache is not None:
        raise ValueError("--no-shared-text-cache cannot be combined with --shared-text-cache")
    label = resolve_embedding_model_label(model=model, version=version, model_label=model_label)
    if version:
        version_path = Path(version)
        default_derived = Path("data/benchmark") / version_path
        default_corpus = Path("data/corpus") / version_path
        eval_root = Path("data/eval") / version_path
        embedding_root = Path("data/embeddings") / version_path
        default_keep_list = None
        default_shared_text_cache = embedding_root / f"{default_embedding_cache_dir(label).name}_texts.sqlite"
    else:
        default_derived = Path("data/benchmark/v0_1")
        default_corpus = Path("data/corpus/v0_1")
        eval_root = Path("data/eval/v0_1")
        embedding_root = Path("data/embeddings/v0_1")
        default_keep_list = None
        default_shared_text_cache = None
    resolved_out = out or default_embedding_summary_path(label, root=eval_root, candidate_filter=candidate_filter)
    resolved_cache = cache or default_embedding_cache_dir(label, root=embedding_root)
    resolved_shared_text_cache = shared_text_cache
    if resolved_shared_text_cache is None and version and not no_shared_text_cache:
        resolved_shared_text_cache = default_shared_text_cache
    return EmbeddingEvalPaths(
        model_label=label,
        derived=derived or default_derived,
        corpus=corpus or default_corpus,
        out=resolved_out,
        details=details or default_baseline_details_path(resolved_out),
        cache=resolved_cache,
        shared_text_cache=resolved_shared_text_cache,
        keep_list=None if no_keep_list else (keep_list if keep_list is not None else default_keep_list),
    )


def resolve_embedding_model_label(*, model: str, version: str | None, model_label: str | None) -> str:
    if model_label:
        return model_label
    if version and ("/" in model or "\\" in model):
        name = Path(model).name
        if name:
            return name
    return model


def selective_embedding_summary_path(base_summary_path: Path) -> Path:
    stem = base_summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    return base_summary_path.with_name(f"{stem}_selective_summary{base_summary_path.suffix}")


def default_lexical_summary_path(candidate_filter: str = "all_files", root: Path = Path("data/eval/v0")) -> Path:
    return default_baseline_summary_path("lexical", candidate_filter, root)


def default_baseline_summary_path(
    ranker: str = "lexical",
    candidate_filter: str = "all_files",
    root: Path = Path("data/eval/v0"),
) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"{ranker}{suffix}.json"


def default_grep_summary_path(
    pattern_mode: str = "strict",
    candidate_filter: str = "all_files",
    root: Path = Path("data/eval/v0"),
) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"grep_{pattern_mode}{suffix}.json"


def default_repomap_summary_path(candidate_filter: str = "all_files", root: Path = Path("data/eval/v0")) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"repomap{suffix}.json"


def default_agentic_summary_path(candidate_filter: str = "all_files", root: Path = Path("data/eval/v1_2")) -> Path:
    suffix = "_summary" if candidate_filter == "all_files" else f"_{candidate_filter}_summary"
    return root / f"scripted-search-read{suffix}.json"


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
