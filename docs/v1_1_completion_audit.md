# Agent Retrieval Bench V1.1 Completion Audit

Generated from local repo state on 2026-05-27 after applying the final external baseline return artifacts.

The latest machine-generated audit reports `overall_status=complete`. V1.1 now has a ready release report, complete required open-source baselines, and a regenerated public leaderboard.

## Final V1.1 Release Snapshot

- Completion audit: `data/reports/v1_1/completion_audit.json`, `overall_status=complete`, `16/16` requirements passed.
- Readiness: `data/reports/v1_1/readiness.json`, `ready=true`, `blocking_gates=[]`.
- Release report: `data/reports/v1_1/release_report.json`, `status=ready`.
- Leaderboard: `data/reports/v1_1/model_leaderboard.json`, `28` rows from `7` summary files.
- Dataset size: `287` samples (`code2test=106`, `comment2context=80`, `trace2code=101`), with `62` accepted V1.1 additions over frozen V1.
- Required release baselines: lexical, RepoMap, Jina, and Qwen. Hosted Voyage is deferred as an optional cost-bearing comparison and is not a V1.1 release blocker.

## Final V1.1 Leaderboard

Rows are all-files retrieval results sorted by MRR within each task.

| Task | Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | Qwen3-Embedding-4B | 287 | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 |
| overall | Qwen3-Embedding-8B | 287 | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 |
| overall | pplx-embed-v1-4b | 287 | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 |
| overall | aider-style-repomap | 287 | 0.3131 | 0.4688 | 0.6419 | 0.2125 | 0.0627 |
| overall | jina-code-embeddings-0.5b | 287 | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 |
| overall | nomic-embed-code | 287 | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 |
| overall | lexical | 287 | 0.1922 | 0.3165 | 0.4750 | 0.1392 | 0.0720 |
| code2test | Qwen3-Embedding-4B | 106 | 0.4610 | 0.5777 | 0.7230 | 0.3225 | 0.3887 |
| code2test | pplx-embed-v1-4b | 106 | 0.3525 | 0.4789 | 0.6447 | 0.2516 | 0.1934 |
| code2test | Qwen3-Embedding-8B | 106 | 0.3774 | 0.5660 | 0.6594 | 0.2406 | 0.1651 |
| code2test | nomic-embed-code | 106 | 0.2789 | 0.4462 | 0.6730 | 0.2066 | 0.0553 |
| code2test | jina-code-embeddings-0.5b | 106 | 0.2610 | 0.3868 | 0.5305 | 0.2033 | 0.2060 |
| code2test | aider-style-repomap | 106 | 0.2597 | 0.3918 | 0.5761 | 0.1962 | 0.0786 |
| code2test | lexical | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 |
| comment2context | jina-code-embeddings-0.5b | 80 | 0.3563 | 0.4375 | 0.5792 | 0.3043 | 0.2521 |
| comment2context | Qwen3-Embedding-4B | 80 | 0.3187 | 0.4771 | 0.6083 | 0.2920 | 0.3250 |
| comment2context | Qwen3-Embedding-8B | 80 | 0.3729 | 0.5021 | 0.6562 | 0.2874 | 0.2354 |
| comment2context | nomic-embed-code | 80 | 0.3521 | 0.4271 | 0.5021 | 0.2657 | 0.1875 |
| comment2context | pplx-embed-v1-4b | 80 | 0.3146 | 0.5021 | 0.5938 | 0.2623 | 0.1625 |
| comment2context | aider-style-repomap | 80 | 0.1812 | 0.2604 | 0.4708 | 0.1558 | 0.0333 |
| comment2context | lexical | 80 | 0.1667 | 0.3417 | 0.4979 | 0.1495 | 0.0563 |
| trace2code | aider-style-repomap | 101 | 0.4736 | 0.7145 | 0.8465 | 0.2745 | 0.0693 |
| trace2code | lexical | 101 | 0.3432 | 0.4818 | 0.6964 | 0.2075 | 0.1287 |
| trace2code | Qwen3-Embedding-8B | 101 | 0.2442 | 0.5281 | 0.7970 | 0.1654 | 0.0759 |
| trace2code | pplx-embed-v1-4b | 101 | 0.1601 | 0.2855 | 0.5380 | 0.1372 | 0.0545 |
| trace2code | nomic-embed-code | 101 | 0.1584 | 0.1733 | 0.3581 | 0.0871 | 0.0297 |
| trace2code | Qwen3-Embedding-4B | 101 | 0.0743 | 0.1815 | 0.5050 | 0.0827 | 0.0495 |
| trace2code | jina-code-embeddings-0.5b | 101 | 0.0693 | 0.1188 | 0.2607 | 0.0607 | 0.0297 |

Qwen3-Embedding-4B is strongest overall by MRR and on `code2test`; Jina is strongest on `comment2context`; RepoMap is strongest on `trace2code`.

## Historical V19 Gate Snapshot

The notes below preserve the working audit trail before final GPU artifacts were returned. They are retained as process history; the current authoritative status is the final release snapshot above.

## Historical Objective Checklist

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Keep frozen V1 unchanged | V1 artifacts live under `data/benchmark/v1`; V1.1 tooling refuses `benchmark/v1` output paths; the generated completion audit checks recorded SHA256, byte-size, and line-count fingerprints for the frozen V1 split files and manifest. | Covered by tooling, release still pending |
| Expand `comment2context` from 51 to 80-100 | Latest audited v19 probe has `comment2context=80`, including 29 manually accepted local additions. | Count target covered for current probe |
| Expand `trace2code` from 68 to 100+ | Latest audited v19 probe has `trace2code=101`, including 33 manually accepted local pytest, Tokio, and Click additions. | Count target covered for current probe |
| Do not expand `code2test` | Readiness requires exact preserved `code2test` count; tests cover rejection of code2test expansion. | Covered by tooling |
| Prioritize cross-module review-context cases | Readiness reports 14 new cross-module `comment2context` samples in the V19 probe, including Playwright, Ruff, Scrapy, IPython, Pydantic, NumPy, ESLint, and pytest-style cases. | Covered for current probe |
| Avoid same-directory `comment2context` shortcuts | V1.1 preflight/readiness reject same-directory new comment gold; pytest probe dropped one same-directory candidate. | Covered by tooling |
| No direct path/basename leakage | V1.1 preflight/readiness reject direct gold hints; pytest probe dropped one direct-hint candidate. | Covered by tooling |
| More non-Go trace repositories/languages | Readiness gates new non-Go trace repos and gold-file extensions; pytest, Tokio, and Click trace probes add Python and Rust candidates. | Covered for current probe |
| More diverse real trace failure types | Readiness gates classified failure types and excludes `unknown`; pytest, Tokio, and Click trace probes contribute assertion, compile-error, exception, panic, timeout, and test-failure signals. | Covered for current probe |
| Gold files present in base-commit corpus | Latest v19 assembly requires `data/corpus/v1_1_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_combined_v19/corpus_manifest.jsonl`; readiness reports all new gold files in corpus. | Covered for current probe |
| Clear task semantics | V1.1 preflight/readiness reject unclear comment/trace task signals. | Covered by tooling |
| Manually auditable evidence | V1.1 preflight requires explicit evidence, not just PR URL; audit packets are generated; current pytest+Tokio+Click+Vite+Scrapy+Ruff+IPython+Pydantic+reset6+reset7+reset8+reset9+reset10+reset14+reset18 probes have 62 local manual audit keep rows. | Covered for current probe |
| Release assembly uses manual audit decisions | `assemble-v1-1 --require-audit-keep` only includes samples marked keep/valid. Readiness enforces the assembly manifest with `--manifest`. The audited pytest+Tokio+Click+Vite+Scrapy+Ruff+IPython+Pydantic+reset6+reset7+reset8+reset9+reset10+reset14+reset18 v19 probe assembly includes 62 accepted expansion samples and passes manifest audit gates. | Covered by tooling |
| Run standard baselines | Final V1.1 lexical, RepoMap, Jina, and Qwen all-files baselines evaluate 287 samples with `skipped={}` and metric-consistent 287-row details files. Voyage is optional/deferred. | Complete |
| Regenerate V1.1 leaderboards/reports | Final `data/reports/v1_1/model_leaderboard.*`, `readiness.*`, `release_report.*`, and `completion_audit.*` exist; the release report is `ready`. | Complete |
| Update docs to present V1.1 as focused improvement | README and this audit document include the final V1.1 release counts and leaderboard. | Complete |

## Historical Probe Evidence

- `pytest-dev/pytest` authenticated crawl probe accepted 36 PRs and found 4 weak `comment2context` candidates.
- Corpus-required V1.1 audit packet accepted 1 automatic candidate and dropped 3:
  - `candidate=1`
  - `direct_gold_hint=1`
  - `duplicate_base_id=1`
  - `same_directory_gold=1`
- Probe corpus build succeeded for the accepted base commit:
  - `file_count=607`
  - `chunk_count=17033`
  - `missing=[]`
- Probe assemble without manual-audit requirement produced 226 samples:
  - `code2test=106`
  - `comment2context=52`
  - `trace2code=68`
- Release-style probe assemble with the blank audit sheet and `--require-audit-keep` produced no expansion:
  - `accepted_expansion=0`
  - `audit_keep_ids=0`
- Local manual audit accepted both current pytest probe additions:
  - audit file: `data/reports/v1_1/manual_audit_pytest_expansion_probe.jsonl`
  - audit summary: `kept=2`
  - audited assembly: `data/benchmark/v1_1_pytest_expansion_probe_audited`
  - audited assembly counts: `code2test=106`, `comment2context=52`, `trace2code=69`
  - low-threshold readiness now passes the assembly manifest audit gates and remains blocked only on baselines and leaderboard artifacts.
- Probe lexical baseline evaluated all 226 probe samples:
  - `evaluated=226`
  - `skipped={}`
- Local pytest test-reproduction probe produced one curated `trace2code` candidate:
  - candidate path: `data/benchmark/v1_1_pytest_trace_repro_candidate_v2_curated`
  - reproduction evidence: `data/reports/v1_1/trace_repro_runs_pytest_probe_installed_v2`
  - automatic fallback evidence: `data/reports/v1_1/trace_repro_runs_pytest_probe_autofallback`
  - corpus path: `data/corpus/v1_1_pytest_trace_probe`
  - corpus-required V1.1 preflight: `candidate=1`
  - failure signals: `assertion; test_failure`
- Combined pytest expansion probe contains 2 accepted local additions:
  - assembled path: `data/benchmark/v1_1_pytest_expansion_probe_assembled`
  - counts: `code2test=106`, `comment2context=52`, `trace2code=69`
  - audited assembled path: `data/benchmark/v1_1_pytest_expansion_probe_audited`
  - low-threshold readiness shows all new-sample and audit-manifest gates pass and blocks only on full baselines and leaderboard artifacts.
  - expansion-only lexical and RepoMap evaluated 2 samples with `skipped={}`.
  - expansion-only probe leaderboard: `data/reports/v1_1/pytest_expansion_probe_model_leaderboard.md` and `.json`.
- `fastapi/fastapi` local test-reproduction probe did not produce a usable trace sample:
  - missing pytest runner was environment setup noise.
  - missing pytest plugin/config internal error is now classified as `failed_environment` and dropped as `env_pytest_internal_collection_error`.
  - automatic Python fallback installs the missing pytest dependencies; after fallback, the focused test passed with no trace and is dropped as `run_status_passed_no_trace`.
- `tokio-rs/tokio` no-token REST crawl probe partially fetched 6 complete PR raw records before the unauthenticated limit was exhausted:
  - initial partial derived output: `comment2context=0`, `trace2code=0`, `code2test=12`.
  - the ordinary REST `crawl` path now skips already complete PRs, upserts raw artifacts, and writes crawl state after each successfully processed PR so future rate-limit/token windows can resume without redoing completed PR work.
  - `backfill-git-raw` then used existing PR base/head SHAs to complete 20 PR file/commit raw records without GitHub REST.
  - full backfilled derived output: `comment2context=0`, `trace2code=0`, `code2test=43`.
  - trace-repro source scan produced 4 Rust candidates under `data/reports/v1_1/trace_repro_source_tokio_git_backfill_probe_focused`.
  - the Rust toolchain is now installed locally.
  - one executed Tokio candidate reproduced a focused `cargo test` failure after automatic fallback to common Tokio features.
  - mined trace sample: `data/benchmark/v1_1_tokio_trace_repro_candidate_cargo_one_features`
  - reproduction evidence: `data/reports/v1_1/trace_repro_runs_tokio_git_backfill_cargo_one_features`
  - corpus path: `data/corpus/v1_1_tokio_trace_probe`
  - manual audit accepted 1 Tokio Rust `trace2code` sample in `data/reports/v1_1/manual_audit_tokio_trace_cargo_one_features.jsonl`.
  - remaining Tokio candidates: 2 of 3 reproduced failures; 1 was accepted after V1.1 preflight and manual audit, 1 was rejected for direct gold basename hint, and 1 passed with no trace.
  - additional accepted trace sample: `data/benchmark/v1_1_tokio_trace_repro_candidate_remaining_features_curated`
  - additional reproduction evidence: `data/reports/v1_1/trace_repro_runs_tokio_git_backfill_remaining_features`
  - additional manual audit: `data/reports/v1_1/manual_audit_tokio_trace_remaining_features.jsonl`.
- Pure git-history Tokio probe:
  - generated 230 raw-compatible commit records without GitHub API under `data/raw_v1_1_git_history/tokio-rs__tokio`.
  - trace-repro source scan produced 47 Rust candidates under `data/reports/v1_1/trace_repro_source_tokio_git_history_probe`.
  - first 20 non-duplicate candidates produced 10 real failures; 2 were manually accepted after corpus-required preflight, 6 were rejected for direct gold hints, and 2 were runner noise after Rust env-noise tightening.
  - accepted curated samples: `data/benchmark/v1_1_tokio_trace_repro_candidate_git_history_top10_curated`
  - accepted manual audit: `data/reports/v1_1/manual_audit_tokio_trace_git_history_top10.jsonl`.
  - later batches added 2 more manually accepted Rust `trace2code` samples in `data/benchmark/v1_1_tokio_trace_repro_candidate_git_history_remaining_curated`.
  - remaining-batch audit decisions rejected one duplicate/backport candidate and one non-root-cause candidate.
  - additional accepted manual audit: `data/reports/v1_1/manual_audit_tokio_trace_git_history_remaining.jsonl`.
- Pure git-history pytest probe:
  - generated 221 raw-compatible commit records without GitHub API under `data/raw_v1_1_git_history/pytest-dev__pytest`.
  - trace-repro source scan produced 5 Python candidates under `data/reports/v1_1/trace_repro_source_pytest_git_history_probe`.
  - local reproduction produced 1 expected failure, 2 failures without usable trace, and 2 passing no-trace runs.
  - mining produced one candidate, which was rejected for a direct gold hint; no accepted new pytest git-history sample was added.
- Pure git-history Click probe:
  - generated 307 raw-compatible commit records without GitHub API under `data/raw_v1_1_git_history/pallets__click`.
  - trace-repro source scan produced 80 Python pytest candidates under `data/reports/v1_1/trace_repro_source_click_git_history_probe`.
  - top-10 execution was stopped after one focused prompt test hung; the 7 completed runs were all `failed_expected`.
  - corpus-required V1.1 preflight accepted 1 candidate and rejected 6 for direct gold hints.
  - accepted curated sample: `data/benchmark/v1_1_click_trace_repro_candidate_git_history_top7_curated`
  - accepted manual audit: `data/reports/v1_1/manual_audit_click_trace_git_history_top7.jsonl`.
  - a selected clean/non-risky batch produced 7 mined samples; corpus-required V1.1 preflight accepted 6 and rejected 1 for direct gold hint.
  - additional accepted curated samples: `data/benchmark/v1_1_click_trace_repro_candidate_git_history_selected8_curated`
  - additional accepted manual audit: `data/reports/v1_1/manual_audit_click_trace_git_history_selected8.jsonl`.
  - a selected12_b batch produced 9 expected failures, 8 manually accepted samples, and 1 direct-hint reject after corpus-required preflight.
  - selected12_b accepted curated samples: `data/benchmark/v1_1_click_trace_repro_candidate_git_history_selected12_b_curated`
  - selected12_b accepted manual audit: `data/reports/v1_1/manual_audit_click_trace_git_history_selected12_b.jsonl`.
  - a selected12_c batch produced 7 mined samples; corpus-required V1.1 preflight accepted all 7, manual audit accepted 6 and rejected 1 non-root-cause candidate.
  - selected12_c accepted curated samples: `data/benchmark/v1_1_click_trace_repro_candidate_git_history_selected12_c_curated`
  - selected12_c accepted manual audit: `data/reports/v1_1/manual_audit_click_trace_git_history_selected12_c.jsonl`.
  - final selected Click batches added 5 more manually accepted trace samples:
    `data/benchmark/v1_1_click_trace_repro_candidate_git_history_selected8_e_curated`,
    `data/benchmark/v1_1_click_trace_repro_candidate_git_history_selected8_f2_curated`, and
    `data/benchmark/v1_1_click_trace_repro_candidate_git_history_remaining_g_curated`.
  - additional accepted manual audits: `data/reports/v1_1/manual_audit_click_trace_git_history_selected8_e.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected8_f2.jsonl`, and `data/reports/v1_1/manual_audit_click_trace_git_history_remaining_g.jsonl`.
- Combined pytest+Tokio+Click+Vite+Scrapy+Ruff+IPython+Pydantic+reset6+reset7+reset8+reset9+reset10+reset14+reset18 audited expansion contains 62 accepted local additions:
  - audited assembled path: `data/benchmark/v1_1_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_audited_v19`
  - counts: `code2test=106`, `comment2context=80`, `trace2code=101`
  - new counts: `comment2context=29`, `trace2code=33`
  - corpus manifest: `data/corpus/v1_1_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_combined_v19/corpus_manifest.jsonl`
  - official readiness passes both sample-count targets, all new-sample quality gates, corpus/audit-manifest gates, non-Go trace coverage, language diversity, and failure-type diversity.
  - official readiness blockers are only `required_baseline_summaries_complete` and `leaderboard_reports_contain_required_baselines`.
  - full lexical and RepoMap all-files baselines each evaluated 287 samples with `skipped={}` and 287-row details files.
  - current probe leaderboard: `data/reports/v1_1/pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_model_leaderboard_v19.md` and `.json`.
  - current not-ready release report: `data/reports/v1_1/release_report_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_audited_v19.md` and `.json`.
- V19 embedding blocker evidence:
  - report path: `data/reports/v1_1/embedding_blockers_v19.md` and `.json`
  - V19 all-files corpus size: 261 manifest rows, 7,016,525 chunks, 345,776 files.
  - local environment: `numpy` and the optional embedding runtime are installed, Torch reports no CUDA device, and `HF_TOKEN` is unset; `VOYAGE_API_KEY` is only relevant for optional hosted-model experiments.
  - public Jina/Qwen model IDs are reachable, but the model paths recorded in V1 aggregate summaries do not exist locally, and no V1 embedding details/vectors are present to reuse.
  - a Jina CPU probe did not finish one 10,142-chunk sample after 7m38s; Qwen 4B would be heavier, and no placeholder V1.1 embedding summaries were generated.
- V19 embedding deduplication evidence:
  - report path: `data/reports/v1_1/embedding_dedup_v19.md` and `.json`
  - the 287 V19 samples use 218 repo/base corpora with 6,210,965 chunks.
  - exact BLAKE2b text hashing finds 1,118,431 unique embedding texts and 5,092,534 duplicates, a duplicate fraction of 0.8199.
  - `eval-embedding` and `eval-voyage` now support optional `--shared-text-cache` SQLite storage to reuse identical chunk-text embeddings within the chosen cache scope; shared-cache rows are written after each embedding batch and details rows are flushed after each completed sample so interrupted runs can reuse completed text batches, inspect partial per-sample results, and use `--resume-details` to append only remaining sample rows on retry.
- V19 embedding resource estimate:
  - report path: `data/reports/v1_1/embedding_resource_estimate_v19.md` and `.json`
  - storage estimate is about 28 GiB for 1024-dimensional models and about 70 GiB for a 2560-dimensional Qwen run when keeping both per-commit and shared caches.
- V19 embedding runbook:
  - report path: `data/reports/v1_1/embedding_runbook_v19.md`
  - includes exact required Jina and Qwen commands, shared text-cache paths, and report-models/readiness/release-report regeneration commands; hosted Voyage commands are optional/deferred.
  - readiness label matching covers public model IDs such as `jinaai/jina-code-embeddings-0.5b` and `Qwen/Qwen3-Embedding-4B`.
  - `check-baseline-summaries` can preflight returned summary and details files with the same required-baseline rules before final report regeneration, including exact sample-ID coverage and details-to-summary metric consistency; after regenerating with the open-source-only gate, the V19 report should block only on missing required Jina/Qwen artifacts.
  - `v1-1-baseline-status` records the same artifact preflight plus local runtime blockers including `numpy`, partial final details row counts, and optional per-shard details progress when passed `--shard-commands`; the current V19 report is `data/reports/v1_1/baseline_status_v19.md` and `.json`.
  - `v1-1-external-runner-preflight` writes `data/reports/v1_1/external_runner_preflight_v19.md` and `.json` from existing status/verifier/return reports, showing whether transfer/handoff checks are ready, which generated runner scripts are locally blocked by missing dependencies or CUDA, and whether return packaging is still blocked by missing required files. In an unpacked external-runner checkout, plain preflight can use a live runtime probe when `baseline_status_v19.json` is absent, and it accepts the bootstrap-written `baseline_transfer_unpack_smoke_v19.json` and `baseline_handoff_unpack_smoke_v19.json` reports as transfer/handoff verifier fallbacks. Its Artifact Checks table separates `artifact_ready` from each report's own `reported_complete` flag, and the top-level `Baseline status complete` / `Baseline status source` / `Return acceptance complete` fields remain the gate-completion signal. Split packaging status is derived from the full return manifest unless filtered manifests are explicitly supplied. When the reporting checkout passes `--copy-packet`, it also checks that the sender-side copy packet still matches the current transfer bundle report/verifier, that the bootstrap/bundle/checksum copy-source files exist, that the bundle file hash and size match, that the checksum sidecar records the same hash, and that the required return-file list and baseline grouping are current.
  - `v1-1-external-runner-failfast-smoke` writes `data/reports/v1_1/external_runner_failfast_smoke_v19.md` and `.json` after running only preflight-blocked generated scripts, confirming they stop before expensive shard jobs or return-bundle creation on this local machine. Optional Voyage split scripts may still be checked explicitly, but they are not part of the default V1.1 release gate. This is local preflight evidence; finalization workflow evidence is limited to stable transfer, handoff, return-bundle, and return-manifest artifacts.
  - `v1-1-summary-from-details` can recover a missing summary from a complete details file after validating sample IDs and metric fields.
  - `eval-embedding` supports `--shard-count/--shard-index` and `--sample-id-file` so external workers can split Jina and Qwen runs; `eval-voyage` has the same mechanics for optional hosted experiments. `v1-1-write-sample-shards` writes deterministic explicit worker files, `v1-1-baseline-shard-commands` expands handoff jobs into concrete shard commands with shard-local shared-text caches by default, and `v1-1-merge-details` validates duplicate IDs, exact V1.1 sample coverage, `candidate_filter`, and metric fields before rebuilding final summaries from merged details.
  - current 4-way V19 sample-id shards are in `data/reports/v1_1/sample_shards_v19/manifest.json` with shard sizes 72/72/72/71 and no duplicate or empty shards.
  - current V19 shard commands are in `data/reports/v1_1/baseline_shard_commands_v19.md` and `.json`; after regenerating, the default required set should cover Jina and Qwen across 4 shards plus merge and summary recovery commands.
  - `data/reports/v1_1/embedding_runbook_v19.md` starts with a minimal external-runner path: copy `unpack_v19_transfer_bundle.sh` with the prepared transfer bundle and checksum, run that bootstrap script to verify the checksum, unpack the bundle, and verify the transfer manifest plus handoff fingerprints, run the full or split shard scripts, package a verified return bundle, apply it on the reporting checkout, and only then accept `overall_status=complete`. The reporting checkout also has `data/reports/v1_1/external_runner_copy_packet_v19.md` and `.json` as a non-bundled copy checklist that records the current prepared-archive hash and required return files without self-referential hash drift; refresh it with `arb v1-1-external-runner-copy-packet` after rebuilding the transfer bundle, and pass it to `v1-1-external-runner-preflight` before copy so the source files, bundle hash/size, checksum sidecar, and required return-file list are checked together.
  - `v1-1-external-runner-preflight` also checks whether the handoff, transfer-manifest, and transfer-bundle verifier reports are fresh against the current handoff, manifest, and bundle files, and whether `baseline_return_acceptance_v19.json` is fresh against the current return manifest and completion audit, by comparing recorded `generated_at` values plus bundle SHA256/size metadata. Plain preflight in an unpacked checkout treats the bootstrap unpack-smoke verifier reports as transfer/handoff freshness evidence; sender-side `--copy-packet` preflight still requires the transfer-bundle verifier. Stale compact handoff evidence reports `freshness_mismatches`; stale return acceptance also reports `return_acceptance_ready=false`, which prevents an old acceptance report from looking usable after finalization or return-manifest refresh.
  - current V19 serial shard runner is `data/reports/v1_1/run_v19_baseline_shards.sh`, with a Markdown summary at `data/reports/v1_1/run_v19_baseline_shards.md`; after regeneration under the open-source-only gate it should verify the transfer manifest, write `baseline_transfer_manifest_verify_v19.json/.md`, verify handoff fingerprints, check `numpy`, check `sentence-transformers` plus CUDA-capable Torch for Jina/Qwen, create output/cache directories, print `df -h` for those locations, execute shard jobs, merge, recover summaries, refresh the return manifest with `--require-existing`, package and verify the return bundle, and run `v1-1-finalize-baselines` to regenerate the final leaderboard/readiness/release/audit gates with transfer-unpack smoke, transfer-bundle, return-bundle, and return-manifest workflow evidence present. Optional Voyage split scripts remain available only for explicit hosted-model experiments.
  - current V19 baseline return manifest is in `data/reports/v1_1/baseline_return_manifest_v19.md`, `.json`, and `.files`; it records the files to copy back after external baseline execution and currently reports missing required summary/details artifacts.
  - current V19 return packaging/apply scripts are `data/reports/v1_1/package_v19_return_artifacts.sh` and `data/reports/v1_1/apply_v19_return_artifacts.sh`. The packaging script refreshes the return manifest with `--require-existing`, runs `check-baseline-summaries` so stale, partial, wrong-sample, or metric-inconsistent summary/details files fail before transfer, builds `baseline_return_bundle_v19.tar.zst`, tests the compressed archive, verifies the tar member list exactly matches the generated file list, and writes a structured `v1-1-verify-return-bundle` report. The apply script reads the expected hash from the first checksum column, verifies the configured return-bundle path, tests the compressed bundle, rejects unsafe tar member paths and non-regular tar members, checks bundle members against the return manifest before unpacking, writes a structured return-bundle verification report, checks required returned files, and runs finalization; this avoids path mismatches when the checksum sidecar records the external runner's bundle path. `ARB_RETURN_BUNDLE` and `ARB_RETURN_CHECKSUM` can override the generated return bundle/checksum paths when returned files are copied to non-default locations. Split-machine partial return paths for the required Jina/Qwen artifacts verify checksums/compression, write the filtered return manifest before member verification, reject unsafe paths and non-regular members, unpack filtered bundles, refresh the filtered manifest with `--require-existing`, and deliberately skip finalization until all required artifacts are present.
  - the hard-required V19 returned files are the final details and summary pairs for `jina-code-embeddings-0.5b` and `qwen3-embedding-4b` under `data/eval/v1_1_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_audited_v19/`; Voyage is optional/deferred.
  - `v1-1-finalize-baselines` can run the complete post-return verification sequence from the handoff paths: handoff fingerprint check, optional complete-shard merge and summary recovery, return-manifest refresh, baseline status, summary preflight, leaderboard, readiness, release report, and completion audit. The current V19 finalization report is `data/reports/v1_1/baseline_finalization_v19.md` and `.json`.
  - `v1-1-baseline-handoff` records exact external-run, transfer manifest/bundle setup, verification, and finalization commands plus sha256 input fingerprints; `v1-1-baseline-transfer-manifest` writes the concrete input file list for external machines, supports `--include` for source/helper files, and fingerprints included source/helper artifacts such as `README.md`, `PLAN.md`, `docs/v1_1_completion_audit.md`, `pyproject.toml`, `src/agent_retrieval_bench`, the V19 runbook, the return manifest, the transfer-unpack bootstrap script, and the V19 embedding blocker/dedup/resource reports for transfer auditing without rehashing the multi-GiB corpus chunks; sender-side completion-audit and compact return-acceptance snapshots stay outside the transfer bundle to avoid self-referential archive-hash drift. `v1-1-baseline-transfer-bundle` creates the prepared transfer archive from that manifest, writes its checksum, and runs the exact-member verifier; `v1-1-verify-transfer-bundle` checks a prepared transfer archive checksum, compressed archive integrity, regular-file member types, and exact tar members before copy; `v1-1-verify-transfer-manifest` checks file presence, chunk sizes, and source/helper-file hashes after transfer; `v1-1-verify-handoff` verifies benchmark-input fingerprints before expensive external runs. The current V19 handoff is `data/reports/v1_1/baseline_handoff_v19.md` and `.json`, the current transfer manifest is `data/reports/v1_1/baseline_transfer_manifest_v19.md`, `.json`, and `.files`; local reporting-checkout prepared-archive unpack-smoke evidence is in `data/reports/v1_1/baseline_transfer_unpack_smoke_v19.json` and `.md`, and handoff fingerprint smoke from the same unpacked archive is in `data/reports/v1_1/baseline_handoff_unpack_smoke_v19.json` and `.md`.
  - `report-models --required-baseline ...` can fail fast when required V1.1 leaderboard rows are missing; after regenerating, the required V19 leaderboard gate should cover lexical, RepoMap, Jina, and Qwen only.
- Prompt-to-artifact V19 completion audit:
  - report path: `data/reports/v1_1/completion_audit_v19.md` and `.json`
  - maps each explicit V1.1 goal requirement to manifest/readiness/baseline/report evidence.
  - checks frozen V1 file fingerprints for `data/benchmark/v1/code2test.jsonl`, `comment2context.jsonl`, `trace2code.jsonl`, `samples.jsonl`, and `manifest.json` whenever canonical `data/benchmark/v1/samples.jsonl` is used as the V1 base.
  - checks release-doc content markers for V1.1 naming, focused improvement/expansion positioning, frozen `benchmark/v1`, `comment2context`, and `trace2code` coverage before the docs checklist item can pass.
  - checks handoff workflow evidence and records each verifier report's `generated_at`, declared handoff/manifest/bundle/checksum paths, and transfer-bundle SHA256/size metadata where available, so the final audit is inspectable without opening every verifier JSON.
  - blocked checklist rows now expose a top-level `next_action` in both JSON and Markdown, so missing required open-source artifacts, leaderboard regeneration, release reporting, and final docs work are directly actionable from the audit.
  - can be regenerated with `arb report-v1-1-completion-audit` from readiness, release, baseline-status, and leaderboard artifacts.
  - overall status remains `not_complete` until the required open-source summaries/details and final all-baseline report gates are regenerated and complete.
- Public unauthenticated GitHub REST mining is currently blocked by exhausted API rate limit in this environment. GitHub CLI calls now expose `--max-rate-limit-sleep` and default to failing fast after 60s instead of sleeping through long reset windows.
- A high-cap derivation pass over all existing local review-comment raw artifacts produced no additional `comment2context` candidates beyond the same 4 pytest candidates already inspected:
  - `data/benchmark/v1_1_existing_raw_token_derived_max100`
  - `data/benchmark/v1_1_existing_raw_probe_derived_max100`
  - `data/benchmark/v1_1_existing_raw_rest_derived_max100`
- A first unauthenticated `crawl-review-comments`-style probe spent the reset REST budget on recent repository-wide review comments and wrote `data/raw_v1_1_comment_probe_recent`; strict derivation produced zero `comment2context` candidates.
- `diagnose-comment2context` output under `data/reports/v1_1/comment2context_diagnostics_recent` explains the zero-yield result:
  - FastAPI: `reply_comment=9`, `low_value_body=5`
  - Gin: `reply_comment=3`, `review_leakage=3`
  - Vue: `reply_comment=23`, `review_leakage=12`
- The reusable `crawl-review-comments` CLI path now exists for the next authenticated or reset window, paginates up to `--comments-per-repo`, excludes bot comments by default, filters reply comments and review-leakage text before ranking PRs, continues to bounded backup PRs when top-ranked PRs are skipped as unmerged or unsuitable, and has a metadata-only mode so selected PR files/commits can be filled later with `backfill-git-raw` instead of extra REST calls.
- Cached review-comment probes for FastAPI, pydantic, and Vite materialized local raw batches without further review-comment API calls, including a no-REST HTML+git-base-inference batch for merged pydantic/Vite PRs. Vite PR #22142 produced one manually accepted cross-template `comment2context` sample after GitHub HTML confirmed the comparison base and head SHAs; the other cached probes produced no accepted additions after strict diagnostics.
- A reset unauthenticated metadata-only review-comment crawl over pydantic, Vite, Django, DRF, Scrapy, and Flask selected 9 merged PRs. Git backfill completed for all selected PRs; derivation produced 3 accepted Scrapy `comment2context` candidates after a focused mutability-evidence derivation update. All three passed corpus-required V1.1 audit packet preflight and were manually accepted in `data/reports/v1_1/manual_audit_scrapy_comment2context_reset1.jsonl`.
- A reset unauthenticated metadata-only review-comment crawl over aiohttp, SQLAlchemy, pandas, HTTPX, Poetry, Celery, scikit-learn, and Ruff selected 4 merged PRs. Git backfill completed for all selected PRs; derivation produced 1 accepted Ruff `comment2context` candidate for inherited enum `__new__` context. It passed corpus-required V1.1 audit packet preflight and was manually accepted in `data/reports/v1_1/manual_audit_ruff_comment2context_reset4.jsonl`.
- A reset unauthenticated metadata-only review-comment crawl over Cargo, Werkzeug, Jinja, Starlette, Sphinx, JupyterLab, IPython, and Typer selected 6 merged PRs. Git backfill completed for all selected PRs; derivation produced 2 accepted IPython `comment2context` candidates around guarded evaluation completion tests. Both passed corpus-required V1.1 audit packet preflight and were manually accepted in `data/reports/v1_1/manual_audit_ipython_comment2context_reset5.jsonl`.
- A manual reset2 review over existing Pydantic raw data produced 1 accepted cross-language `comment2context` candidate around non-tagged union serialization behavior. It passed corpus-required V1.1 audit packet preflight and was manually accepted in `data/reports/v1_1/manual_audit_pydantic_comment2context_reset2_manual.jsonl`.
- A reset unauthenticated metadata-only review-comment crawl over mypy, pre-commit, tox, pluggy, Hypothesis, Trio, AnyIO, and Uvicorn selected 11 merged PRs. Git backfill completed for all selected PRs; derivation produced 4 manually accepted `comment2context` candidates from Hypothesis, mypy, and tox in `data/reports/v1_1/manual_audit_reset6_comment2context.jsonl`.
- A reset unauthenticated metadata-only review-comment crawl over Django, pandas, scikit-learn, matplotlib, Sphinx, pip, SQLAlchemy, and Starlette selected 2 merged pip PRs. Git backfill completed for both; derivation produced 1 manually accepted pip `comment2context` candidate in `data/reports/v1_1/manual_audit_reset7_comment2context.jsonl`.
- A deeper reset unauthenticated metadata-only review-comment crawl over previous high-yield repos selected 24 merged PRs. Git backfill completed for all selected PRs; derivation produced 5 new manually accepted Ruff/pip `comment2context` candidates in `data/reports/v1_1/manual_audit_reset8_comment2context.jsonl`.
- A focused reset9 metadata-only pass over recent pip/Ruff review comments selected 8 merged PRs. Git backfill completed for both repos; derivation produced 1 manually accepted pip `comment2context` candidate in `data/reports/v1_1/manual_audit_reset9_comment2context.jsonl`.
- A focused reset10 metadata-only pass over uv, Airflow, NumPy, SciPy, and Polars selected 6 merged PRs from NumPy, Polars, and uv. Git backfill completed for selected repos; derivation produced 2 NumPy `comment2context` candidates, and manual review accepted 1 DLPack BufferError/test-context sample in `data/reports/v1_1/manual_audit_reset10_comment2context.jsonl`.
- A reset12 metadata-only review-comment crawl over pandas, matplotlib, scikit-learn, Django, and Sphinx selected 7 merged PRs after 45 bounded PR-detail attempts. Git backfill completed for all selected repos; derivation produced 1 Sphinx `comment2context` candidate, but manual review rejected it as the already-known weak wording-only "formerly problematic" case. No reset12 sample was accepted.
- A reset13 metadata-only review-comment crawl over pip, Ruff, Scrapy, NumPy, and IPython selected 15 merged PRs. Git backfill, derivation, and diagnostics completed, but manual review found no clean new sample. The IPython `@property` edge case was rejected because its direct response commit only changed tests while the implementation change answered an adjacent review thread.
- A reset14 metadata-only review-comment crawl over pytest, Werkzeug, SQLAlchemy, ESLint, and Prettier selected 7 merged PRs before anonymous rate limit exhaustion. Git backfill completed for pytest, ESLint, and Prettier. Manual diagnostics review accepted 1 ESLint `comment2context` sample:
  - candidate path: `data/benchmark/v1_1_comment_candidates_reset14_manual_eslint/comment2context.jsonl`
  - corpus path: `data/corpus/v1_1_comment_candidates_reset14_manual_eslint/corpus_manifest.jsonl`
  - audit packet: `data/reports/v1_1/audit_packet_comment_candidates_reset14_manual_eslint`
  - accepted manual audit: `data/reports/v1_1/manual_audit_reset14_comment2context.jsonl`
  - sample evidence: review on `lib/rules/no-param-reassign.js` whose missing cross-directory configuration context is `eslint.config.js`.
- Reset15/reset16/reset17 review-comment mining plus reset18 manual curation added 7 accepted `comment2context` rows from Playwright, Ruff, and pip:
  - candidate path: `data/benchmark/v1_1_comment_candidates_reset18_manual/comment2context.jsonl`
  - corpus path: `data/corpus/v1_1_comment_candidates_reset18_manual_bigfile/corpus_manifest.jsonl`
  - audit packet: `data/reports/v1_1/audit_packet_comment_candidates_reset18_manual`
  - accepted manual audit: `data/reports/v1_1/manual_audit_reset18_comment2context.jsonl`
  - sample evidence includes Playwright cross-process self-destruct/stdin review context, Ruff type-builder and project-diagnostics context, and pip dead-code call-site context.

## Historical Blocking Gaps Before Final GPU Return

- Need final V1.1 baseline gates regenerated from Jina and Qwen summary/details; lexical and RepoMap are complete for the current 287-sample V19 assembly.
- Need GPU/precomputed embedding caches for Jina and Qwen; see `data/reports/v1_1/embedding_blockers_v19.md`.
- Need the missing embedding runs to use `--shared-text-cache` or equivalent deduplication within each run/cache scope; generated parallel shard commands use shard-local caches, while a serial/coordinated run can use one baseline-level cache for whole-run deduplication. See `data/reports/v1_1/embedding_dedup_v19.md`.
- Need GPU storage planning using `data/reports/v1_1/embedding_resource_estimate_v19.md`.
- Need to follow `data/reports/v1_1/embedding_runbook_v19.md` in a GPU-capable environment and then rerun readiness/release reporting.
- Need to run `arb v1-1-baseline-status` in the target environment to capture whether the remaining blocker is runtime access, partial details progress, or returned artifact validity.
- Need to run `arb v1-1-summary-from-details` if an external run leaves a complete details file but no summary.
- If the external baseline run is sharded, need to use `arb v1-1-write-sample-shards` when explicit sample-id worker files are preferred, run `arb v1-1-merge-details` on all shard details, and then `arb v1-1-summary-from-details` for each model before final preflight.
- Need to use `arb v1-1-baseline-shard-commands` or the current `data/reports/v1_1/baseline_shard_commands_v19.md` if external workers should receive ready-to-run per-shard commands.
- Need to use `arb v1-1-baseline-return-manifest --require-existing` on the external runner after baseline execution, or inspect `data/reports/v1_1/baseline_return_manifest_v19.json`, before copying results back.
- Need to follow `data/reports/v1_1/baseline_handoff_v19.md` for external GPU runs so the final verification commands write the current V19 status, preflight, leaderboard, release, and completion-audit artifacts.
- After copying returned artifacts back, `arb v1-1-finalize-baselines --auto-merge-shards` can merge complete shard details, recover summaries, and regenerate every final gate from the V19 handoff, shard-command report, and return manifest in one pass.
- Need to transfer the files listed in `data/reports/v1_1/baseline_transfer_manifest_v19.files` to the target GPU machine, or recreate the single archive with `arb v1-1-baseline-transfer-bundle --manifest data/reports/v1_1/baseline_transfer_manifest_v19.json --bundle data/reports/v1_1/baseline_transfer_bundle_v19.tar.zst --bundle-files data/reports/v1_1/baseline_transfer_manifest_v19.files`. The manifest now includes sample shard files, shard-command reports, serial/split run scripts, the transfer-unpack bootstrap script, full and filtered return packaging/apply scripts, and the runbook via transfer-manifest `--include`, with helper-file sha256 fingerprints in the JSON/Markdown manifest. The shortest reporting-checkout copy checklist is `data/reports/v1_1/external_runner_copy_packet_v19.md`, intentionally outside the transfer bundle so it can record the current prepared-archive hash and required return files; refresh it with `arb v1-1-external-runner-copy-packet` after the bundle hash changes and run `arb v1-1-external-runner-preflight --copy-packet data/reports/v1_1/external_runner_copy_packet_v19.json` before copy to confirm the bootstrap, bundle, and checksum source files exist and the bundle/checksum hashes match. Copy `data/reports/v1_1/unpack_v19_transfer_bundle.sh` next to the prepared archive and checksum on the external runner to verify/unpack/check transfer state in one step, or run `arb v1-1-verify-transfer-manifest --manifest data/reports/v1_1/baseline_transfer_manifest_v19.json` after transfer and `arb v1-1-verify-handoff --handoff data/reports/v1_1/baseline_handoff_v19.json` before expensive baseline jobs. If only basename files are copied, use the bootstrap rather than raw `sha256sum -c`; the checksum sidecar records the generated repo-relative bundle path, while the bootstrap verifies the supplied bundle path against the recorded hash. The current prepared archive's local reporting-checkout unpack-smoke report is `data/reports/v1_1/baseline_transfer_unpack_smoke_v19.json` and `.md`; handoff fingerprint smoke from the same unpacked archive is `data/reports/v1_1/baseline_handoff_unpack_smoke_v19.json` and `.md`.
- Need to run `arb v1-1-verify-handoff --handoff data/reports/v1_1/baseline_handoff_v19.json` on the target machine before starting the external GPU baselines.
- Need to run `arb check-baseline-summaries` on returned summary/details files before final report regeneration to catch sample-count, `skipped`, candidate-filter, label, stale sample ID, stale metric, or partial details-file mismatches early.
- Need to run `arb report-models` with all required `--required-baseline` labels so missing leaderboard rows fail before release reporting.
- Need `data/reports/v1_1/completion_audit_v19.json` to reach `overall_status=complete` before marking the thread goal complete.
- Need to regenerate `data/reports/v1_1/completion_audit_v19.json` with `arb report-v1-1-completion-audit` after the missing baseline artifacts are present.
- Need final V1.1 leaderboard and release report after all required baseline summaries/details exist. The current V19 report is intentionally `not_ready` because the embedding baselines are absent.
- No sample-count target gap remains. Extra `comment2context` mining is optional only if the release should carry slack above exactly 80 samples.

## Final Conclusion

V1.1 is complete according to the current prompt-to-artifact checklist. The final release has 287 samples, complete required open-source baselines, a ready release report, and a regenerated public leaderboard. The historical blocking gaps above are retained only as the pre-return audit trail.
