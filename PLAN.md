# Agent Retrieval Bench V1.1 Plan

## Goal Text for `/goal`

Build Agent Retrieval Bench V1.1 as a targeted expansion of frozen V1, focused on the weakest and most informative tracks rather than raw scale.

Keep `benchmark/v1` unchanged. Create V1.1 by expanding `comment2context` from 51 to about 80-100 high-quality samples and `trace2code` from 68 to 100+ samples. Prioritize cross-module review-context cases, non-same-directory gold files, no direct path/basename leakage, more non-Go repositories, more languages, and more diverse real failure types.

Validate that every new sample has gold files present in the base-commit corpus, no query leakage, clear task semantics, and manually auditable evidence. Then run the required open-source baselines, regenerate leaderboards/reports, and update docs to present V1.1 as a focused benchmark improvement over V1. `voyage-code-3` is deferred as an optional hosted comparison, not a V1.1 release blocker.

## Current State and Work Log

- V1 is frozen and must remain unchanged under `data/benchmark/v1` and any published `benchmark/v1` artifact.
- Current completion evidence is summarized in `docs/v1_1_completion_audit.md`.
- Local V1 release artifacts are available under `data/benchmark/v1`, `data/corpus/v1`, `data/eval/v1`, and `data/reports/v1`.
- Current V1 counts are `code2test=106`, `comment2context=51`, `trace2code=68`, total `225`.
- V1.1 is complete under the open-source release gate: the final assembly has 287 samples (`code2test=106`, `comment2context=80`, `trace2code=101`) with 62 accepted additions over frozen V1.
- Final readiness is `ready=true`, final release report is `status=ready`, and the completion audit reports `overall_status=complete` with `16/16` requirements passed.
- The final required baseline set is lexical, RepoMap, Jina, and Qwen. Hosted `voyage-code-3` is optional/deferred and is not a V1.1 release blocker.
- The final V1.1 leaderboard is published in `README.md` and `docs/v1_1_completion_audit.md`; the source artifact is `data/reports/v1_1/model_leaderboard.json` with 28 rows from 7 summary files.
- Overall V1.1 MRR leaders are Qwen3-Embedding-4B (`0.2296`), Qwen3-Embedding-8B (`0.2272`), pplx-embed-v1-4b (`0.2143`), and aider-style-repomap (`0.2125`).
- Task leaders by MRR are Qwen3-Embedding-4B for `code2test` (`0.3225`), jina-code-embeddings-0.5b for `comment2context` (`0.3043`), and aider-style-repomap for `trace2code` (`0.2745`).
- The remaining bullets are retained as the historical work log for how V1.1 was built and verified.
- A small authenticated `pytest-dev/pytest` probe produced one `comment2context` candidate that passes V1.1 automatic preflight, gold-in-corpus validation, and lexical evaluator coverage. It is stored under ignored probe artifacts (`data/benchmark/v1_1_pytest_probe*`, `data/corpus/v1_1_pytest_probe*`, `data/eval/v1_1_pytest_probe`, and `data/reports/v1_1/audit_packet_pytest_probe*`).
- A local `pytest-dev/pytest` test-reproduction probe produced one curated `trace2code` candidate after applying only the PR test patch, installing the base checkout in a local venv, and reproducing a focused pytest failure. The curated candidate passes corpus-required V1.1 preflight under `data/benchmark/v1_1_pytest_trace_repro_candidate_v2_curated` with evidence in `data/reports/v1_1/trace_repro_runs_pytest_probe_installed_v2`. The trace-repro runner now has automatic Python fallback commands for editable install + pytest/pytest-timeout, so this environment recovery no longer requires hand-edited candidate JSON.
- A local manual audit accepted those two pytest probe additions in `data/reports/v1_1/manual_audit_pytest_expansion_probe.jsonl`.
- A combined audited pytest probe expansion assembles to 227 total samples (`code2test=106`, `comment2context=52`, `trace2code=69`) under `data/benchmark/v1_1_pytest_expansion_probe_audited`. Low-threshold readiness confirms the new sample gates and assembly audit-manifest gates pass. Expansion-only lexical and RepoMap baselines both evaluate the 2 local additions with `skipped={}`, and a probe leaderboard exists under `data/reports/v1_1/pytest_expansion_probe_model_leaderboard.*`.
- These probe candidates are accepted local expansion samples, but they are not enough to constitute V1.1.
- A `fastapi/fastapi` local test-reproduction probe produced no usable trace sample: one run is now classified as `failed_environment` for pytest environment/config noise and is dropped by `mine-trace-repro-runs`; after adding the missing test plugin, the focused test passed with no trace.
- A no-token REST crawl probe for `tokio-rs/tokio` fetched partial review/commit raw data before exhausting the unauthenticated rate limit. The partial direct derived output produced no usable `comment2context`/`trace2code` additions. The generic `crawl` path now resumes more safely: it skips already complete PR raw records, upserts per-PR artifacts, and persists crawl state after each successfully processed PR.
- A new `backfill-git-raw` command can fill PR file/commit raw records from existing PR base/head SHAs using `git fetch`/`git diff`, avoiding GitHub REST. It can also infer a mining-only base from `headRefOid` plus `baseRefName`, with a commit-range cap to avoid noisy stale branches. It completed 20 Tokio PR raw records and produced 4 Rust `trace-repro-source` candidates with focused `cargo test` commands.
- The Rust toolchain is installed locally. Two Tokio REST/backfill candidates reproduced focused `cargo test` failures after automatic common-feature fallback, were mined into Rust `trace2code` samples, passed corpus-required audit packet generation, and were manually accepted in `data/reports/v1_1/manual_audit_tokio_trace_cargo_one_features.jsonl` and `data/reports/v1_1/manual_audit_tokio_trace_remaining_features.jsonl`. A third reproduced Tokio failure was rejected by V1.1 preflight because the query exposed the gold basename.
- A pure git-history Tokio raw probe generated 230 raw-compatible commit records without GitHub API access and produced 47 local reproduction candidates. The first 20 non-duplicate candidates yielded 10 real failures, 2 manually accepted Rust `trace2code` samples, 6 V1.1 preflight drops for direct gold hints, and 2 runner-noise drops after tightening Rust environment-noise detection. Later batches added 2 more accepted Rust `trace2code` samples, while duplicate/backport and non-root-cause candidates were rejected. Accepted audit files are `data/reports/v1_1/manual_audit_tokio_trace_git_history_top10.jsonl` and `data/reports/v1_1/manual_audit_tokio_trace_git_history_remaining.jsonl`.
- A pure git-history pytest raw probe generated 221 raw-compatible commit records and 5 local reproduction candidates, but produced no accepted new sample; the only mined candidate was rejected for a direct gold hint.
- A pure git-history `pallets/click` raw probe generated 307 raw-compatible commit records without GitHub API access and 80 local pytest reproduction candidates. The first completed top-10 batch produced 7 real failures; 1 Python `trace2code` sample was manually accepted after corpus-required V1.1 preflight, while 6 were rejected for direct gold hints. One additional top-10 candidate was stopped because the focused prompt test hung. Later selected batches added 45 more executed candidates; 26 mined samples passed or reached preflight, and 19 total Click samples were manually accepted. Rejects were direct gold hint, non-root-cause, broad-root-file, no-trace, passed-no-trace, patch-failed, or missing-trace cases. Accepted audit files include `data/reports/v1_1/manual_audit_click_trace_git_history_top7.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected8.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected12_b.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected12_c.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected8_e.jsonl`, `data/reports/v1_1/manual_audit_click_trace_git_history_selected8_f2.jsonl`, and `data/reports/v1_1/manual_audit_click_trace_git_history_remaining_g.jsonl`.
- A combined audited pytest+Tokio+Click+Vite+Scrapy+Ruff+IPython+Pydantic+reset6+reset7+reset8+reset9+reset10+reset14+reset18 expansion assembles to 287 total samples (`code2test=106`, `comment2context=80`, `trace2code=101`) under `data/benchmark/v1_1_pytest_tokio_click_vite_scrapy_ruff_ipython_pydantic_reset6_reset7_reset8_reset9_reset10_reset14_reset18_expansion_probe_audited_v19`, then canonicalizes to `data/benchmark/v1_1` and `data/corpus/v1_1` for release. Official readiness confirms both sample-count targets, all new-sample quality gates, corpus coverage, audit-manifest gates, non-Go trace coverage, language diversity, and failure-type diversity pass. Lexical, RepoMap, Jina, and Qwen all-files baselines evaluate all 287 samples with `skipped={}` and metric-consistent 287-row details files. The final leaderboard exists under `data/reports/v1_1/model_leaderboard.*` and is copied into public docs.
- A V19 embedding-blocker evidence report exists at `data/reports/v1_1/embedding_blockers_v19.md` and `.json`. It confirms the all-files V19 corpus has 7,016,525 chunks, no reusable local embedding vectors/details exist, the optional embedding runtime is installed, public Jina/Qwen model IDs are reachable, but this CPU-only machine cannot practically finish the local embedding baselines: a Jina CPU probe did not finish one 10,142-chunk sample after 7m38s, and Qwen 4B would be heavier. No placeholder embedding summaries were generated.
- A V19 embedding text deduplication report exists at `data/reports/v1_1/embedding_dedup_v19.md` and `.json`. The 287 V19 samples use 218 repo/base corpora with 6,210,965 chunks, but only 1,118,431 unique embedding texts; a new optional `--shared-text-cache` SQLite cache for `eval-embedding` and `eval-voyage` can avoid re-embedding duplicate chunk texts within the cache scope and writes completed embedding batches plus completed per-sample details incrementally for retryable long runs. The 5,092,534 duplicate count is the whole V19 upper bound when a run uses one coordinated text cache; generated parallel shard commands use shard-local caches to avoid SQLite writer contention, so they deduplicate within each shard rather than across all shards. `--resume-details` can skip sample IDs already present in a partial details file and append the remaining rows on retry.
- `eval-embedding` now supports `--shard-count/--shard-index` and `--sample-id-file`, so the missing V1.1 embedding baselines can be split across external workers; `eval-voyage` keeps the same mechanics for optional hosted experiments. `v1-1-write-sample-shards` writes deterministic sample-id shard files for workers that need explicit assignment, and `v1-1-baseline-shard-commands` expands the handoff into concrete per-baseline/per-shard commands with shard-specific output, details, shard-local shared-text caches, and cache paths. Each shard should write a separate details file. `v1-1-merge-details` safely merges complete shard details after checking duplicate IDs, exact V1.1 sample coverage, `candidate_filter`, and metric fields, then `v1-1-summary-from-details` builds the final all-files summary.
- A V19 embedding resource estimate exists at `data/reports/v1_1/embedding_resource_estimate_v19.md` and `.json`. Storage estimates are about 28 GiB for 1024-dimensional models and about 70 GiB for a 2560-dimensional Qwen run when keeping both per-commit and shared caches. The old Voyage token/cost estimate is retained only as optional hosted-model planning evidence, not as a V1.1 release requirement.
- Exact commands for completing the remaining required Jina and Qwen baselines and regenerating V19 reports are recorded in `data/reports/v1_1/embedding_runbook_v19.md`. The runbook starts with a minimal external-runner path: copy `unpack_v19_transfer_bundle.sh` with the prepared transfer bundle and checksum, run the bootstrap script to verify/unpack and check transfer/handoff fingerprints, install the local package plus embedding dependencies including `numpy`, run the full or split shard scripts, package a verified return bundle, apply it locally, and check that the completion audit reports `overall_status=complete`. The reporting checkout also has a tiny copy checklist at `data/reports/v1_1/external_runner_copy_packet_v19.md` and `.json`; it is intentionally outside the transfer bundle so it can record the current bundle hash without self-referential drift, and can be refreshed with `arb v1-1-external-runner-copy-packet` after rebuilding the transfer bundle. Passing the copy packet to `v1-1-external-runner-preflight` now also confirms the bootstrap, bundle, and checksum sidecar exist, hashes the bundle file itself, checks the checksum sidecar, and compares the required return-file list and baseline grouping before external copy. The full serial `run_v19_baseline_shards.sh` packages the verified return bundle before finalization so return-bundle workflow evidence exists; split GPU scripts still leave final packaging/finalization until all artifacts are in one checkout. The prepared V19 transfer bundle includes the runbook plus `embedding_blockers_v19`, `embedding_dedup_v19`, and `embedding_resource_estimate_v19` reports in both Markdown and JSON for external GPU runners. The readiness matcher accepts public Hugging Face model labels such as `jinaai/jina-code-embeddings-0.5b` and `Qwen/Qwen3-Embedding-4B` as the required baseline labels.
- A `check-baseline-summaries` command can preflight externally produced GPU summary and details files before final report regeneration, using the same V1.1 all-files/sample-count/skipped/model-label rules as readiness, requiring one details row for each current sample ID, and recomputing details metrics against the summary by default. The V19 preflight report should be regenerated after removing Voyage from the required gate so it blocks only on missing required open-source artifacts.
- A `v1-1-baseline-status` command now combines the required-baseline artifact preflight with local runtime checks for `numpy`, CUDA, `sentence_transformers`, `torch`, `nvidia-smi`, and optional `VOYAGE_API_KEY`, and reports partial `*_details.jsonl` row counts even before summaries exist. The current V19 status report should be regenerated under the open-source-only gate so lexical/RepoMap plus Jina/Qwen determine completion. A read-only `v1-1-external-runner-preflight` report summarizes the external-run handoff state: transfer/handoff verifier readiness, local runner dependency/CUDA blockers, and whether return packaging is still blocked by missing required final files. Optional Voyage split runners remain explicit, non-blocking paths.
- A `v1-1-summary-from-details` command can regenerate a missing summary JSON from a complete details JSONL after validating row count, exact V1.1 sample IDs, candidate filter, duplicate IDs, and metric fields, so a completed external embedding run does not need to be rerun if it fails after flushing details. A companion `v1-1-merge-details` command validates and merges sharded details before summary recovery.
- A deterministic 4-way V19 sample-id shard manifest now exists at `data/reports/v1_1/sample_shards_v19/manifest.json` with shard sizes 72/72/72/71, no duplicate sample IDs, and no empty shards. External workers can either use these files with `--sample-id-file` or use equivalent `--shard-count/--shard-index` arguments.
- V19 per-shard external baseline commands now exist at `data/reports/v1_1/baseline_shard_commands_v19.md` and `.json`; after regenerating under the open-source-only gate, they should expand the Jina and Qwen handoff into shard commands plus per-baseline merge and summary-from-details recovery commands.
- A V19 baseline return manifest now exists at `data/reports/v1_1/baseline_return_manifest_v19.md`, `.json`, and `.files`, listing the final summary/details artifacts that must come back from the external runner plus optional shard details/merge reports for audit. Regenerate it so the required return set is Jina and Qwen only.
- A `v1-1-finalize-baselines` command now performs the complete post-return gate sequence from the handoff paths: fingerprint verification, optional complete-shard merge and summary recovery, return-manifest refresh, baseline status, summary preflight, leaderboard, readiness, release report, and completion audit. The current V19 finalization report is `data/reports/v1_1/baseline_finalization_v19.md` and `.json`; regenerate it after applying the open-source baseline artifacts.
- A `v1-1-baseline-handoff` command writes a machine-readable and Markdown handoff for external GPU baseline runs, including exact Jina and Qwen commands, transfer manifest/bundle setup, status, preflight, leaderboard, readiness, release, completion-audit, and finalization commands. It also records sha256 input fingerprints for the V1/V1.1 sample files, assembly manifest, and corpus manifest so external runs can verify they are using the same benchmark inputs. `v1-1-baseline-transfer-manifest` writes the concrete file list for external machines and now supports `--include` for source/helper artifacts such as `README.md`, `PLAN.md`, `docs/v1_1_completion_audit.md`, `pyproject.toml`, `src/agent_retrieval_bench`, sample shards, shard command reports, generated run scripts, return manifests, filtered return packaging/apply scripts, and runbooks; included source/helper files are fingerprinted for transfer auditing without rehashing the multi-GiB corpus chunks. Sender-side gate snapshots such as the completion audit and compact return acceptance stay outside the transfer bundle to avoid self-referential archive-hash drift. `v1-1-baseline-transfer-bundle` creates a tar.zst archive from that manifest, writes the checksum, and runs the exact-member verifier; `v1-1-verify-transfer-manifest` checks file presence, chunk file sizes, and source/helper-file hashes after transfer, and `v1-1-verify-handoff` checks the benchmark-input fingerprints before expensive external runs. The current V19 handoff is `data/reports/v1_1/baseline_handoff_v19.md` and `.json`; the current transfer manifest is `data/reports/v1_1/baseline_transfer_manifest_v19.md`, `.json`, and `.files`. A prepared transfer archive is available as `data/reports/v1_1/baseline_transfer_bundle_v19.tar.zst` with checksum and unpack instructions in `data/reports/v1_1/baseline_transfer_bundle_v19.md`; local reporting-checkout unpack-smoke evidence for that prepared archive is in `data/reports/v1_1/baseline_transfer_unpack_smoke_v19.json` and `.md`, and the same unpacked archive also passes handoff fingerprint verification in `data/reports/v1_1/baseline_handoff_unpack_smoke_v19.json` and `.md`.
- `report-models --required-baseline ...` can fail fast before final leaderboard publication if any required V1.1 baseline row is missing; the required rows are lexical, RepoMap, Jina, and Qwen.
- `report-v1-1-completion-audit` regenerates the prompt-to-artifact checklist from readiness, release, baseline-status, leaderboard artifacts, docs, and handoff workflow evidence. It checks release-doc content markers for V1.1 naming, focused improvement/expansion positioning, frozen `benchmark/v1`, `comment2context`, and `trace2code` coverage, and the workflow evidence rows expose verifier `generated_at` values, declared handoff/manifest/bundle paths, and transfer-bundle hash/size metadata. The current V19 completion audit is recorded in `data/reports/v1_1/completion_audit_v19.md` and `.json` and should be regenerated after the required open-source baseline gates are complete.
- The completion audit also checks frozen V1 file fingerprints for `data/benchmark/v1/*.jsonl` and `data/benchmark/v1/manifest.json` when canonical `data/benchmark/v1/samples.jsonl` is used as the V1 base, so preserving V1 means matching the recorded SHA256, byte-size, and line-count evidence rather than only preserving IDs.
- A high-cap derivation pass over existing local review-comment raw data (`data/raw_v1_1_token_probe`, `data/raw_v1_1_probe`, and `data/raw_v1_1_rest_probe`) produced no additional `comment2context` candidates beyond the same 4 pytest candidates already inspected. FastAPI and Tokio local raw review-comment artifacts currently yield zero `comment2context` additions.
- New `crawl-review-comments` and `diagnose-comment2context` commands support focused `comment2context` mining. `crawl-review-comments` starts from repository-wide review comments, paginates up to `--comments-per-repo`, excludes bot comments by default, filters reply comments and review-leakage text before ranking PRs, continues to bounded backup PRs when top-ranked PRs are skipped as unmerged or unsuitable, and can write only PR metadata plus review comments so `backfill-git-raw` can fill file/commit records without extra REST calls. `diagnose-comment2context` reports local drop reasons so the next crawl can distinguish post-hoc replies, review leakage, missing response commits, same-path/no-context cases, and evidence failures.
- Cached review-comment probes for FastAPI, pydantic, and Vite materialized local raw batches without further review-comment API calls, including a no-REST HTML+git-base-inference batch for merged pydantic/Vite PRs. The Vite PR #22142 HTML confirmed `base_sha=a4d828f2d5ed85440bc0774eab342e6f9a5e5f62` and produced one manually accepted cross-template `comment2context` sample; the other cached probes produced no accepted additions after strict diagnostics.
- A reset unauthenticated metadata-only `crawl-review-comments` batch over pydantic, Vite, Django, DRF, Scrapy, and Flask selected 9 merged PRs. Git backfill completed without REST for all selected PRs. Strict derivation produced 3 accepted Scrapy `comment2context` candidates after a focused mutability-evidence derivation update; all three passed corpus-required V1.1 preflight and were manually accepted in `data/reports/v1_1/manual_audit_scrapy_comment2context_reset1.jsonl`.
- A later reset unauthenticated metadata-only `crawl-review-comments` batch over aiohttp, SQLAlchemy, pandas, HTTPX, Poetry, Celery, scikit-learn, and Ruff selected 4 merged PRs. Git backfill completed for all selected PRs. Strict derivation produced 1 accepted Ruff `comment2context` candidate: a review on `crates/ty_python_semantic/resources/mdtest/enums.md` asking for inherited `__new__` coverage whose missing implementation context is `crates/ty_python_semantic/src/types/enums.rs`. The sample passed corpus-required V1.1 preflight and was manually accepted in `data/reports/v1_1/manual_audit_ruff_comment2context_reset4.jsonl`.
- A subsequent reset unauthenticated metadata-only `crawl-review-comments` batch over Cargo, Werkzeug, Jinja, Starlette, Sphinx, JupyterLab, IPython, and Typer selected 6 merged PRs. Git backfill completed for all selected PRs. Strict derivation produced 2 accepted IPython `comment2context` candidates from reviews on `IPython/core/guarded_eval.py` whose missing cross-directory test context is `tests/test_completer.py`. Both samples passed corpus-required V1.1 preflight and were manually accepted in `data/reports/v1_1/manual_audit_ipython_comment2context_reset5.jsonl`.
- A manual reset2 review over existing Pydantic raw data added 1 accepted cross-language `comment2context` candidate: a review on `pydantic-core/src/serializers/type_serializers/union.rs` asking to confirm non-tagged union serialization behavior whose missing Python test context is `tests/test_discriminated_union.py`. The sample passed corpus-required V1.1 preflight and was manually accepted in `data/reports/v1_1/manual_audit_pydantic_comment2context_reset2_manual.jsonl`.
- A reset unauthenticated metadata-only `crawl-review-comments` batch over mypy, pre-commit, tox, pluggy, Hypothesis, Trio, AnyIO, and Uvicorn selected 11 merged PRs. Git backfill completed for all selected PRs. Strict derivation produced 4 accepted `comment2context` candidates from Hypothesis, mypy, and tox after manual evidence review; all passed corpus-required V1.1 preflight and were manually accepted in `data/reports/v1_1/manual_audit_reset6_comment2context.jsonl`.
- A reset unauthenticated metadata-only `crawl-review-comments` batch over Django, pandas, scikit-learn, matplotlib, Sphinx, pip, SQLAlchemy, and Starlette selected 2 merged pip PRs. Git backfill completed for both. Strict derivation produced 1 accepted pip `comment2context` candidate around unexpected post-install import warning behavior, manually accepted in `data/reports/v1_1/manual_audit_reset7_comment2context.jsonl`.
- A deeper reset unauthenticated metadata-only `crawl-review-comments` batch over previous high-yield repos selected 24 merged PRs. Git backfill completed for all selected PRs. Strict derivation produced 5 new accepted `comment2context` candidates from Ruff and pip after duplicate/reject filtering; all passed corpus-required V1.1 preflight and were manually accepted in `data/reports/v1_1/manual_audit_reset8_comment2context.jsonl`.
- A focused reset9 metadata-only pass over recent pip/Ruff review comments selected 8 merged PRs. Git backfill completed for both repos; strict derivation produced one new accepted pip `comment2context` candidate around disabling pip's self-version check during pip self-upgrade installs, manually accepted in `data/reports/v1_1/manual_audit_reset9_comment2context.jsonl`.
- A focused reset10 metadata-only pass over uv, Airflow, NumPy, SciPy, and Polars selected 6 merged PRs from NumPy, Polars, and uv. Git backfill completed for selected repos; strict derivation produced 2 NumPy `comment2context` candidates, and manual review accepted 1 cross-directory DLPack BufferError/test-context sample in `data/reports/v1_1/manual_audit_reset10_comment2context.jsonl`.
- A reset12 metadata-only pass over pandas, matplotlib, scikit-learn, Django, and Sphinx exercised the bounded backup-PR logic: it attempted 45 ranked PR details after many unmerged skips and selected 7 merged PRs. Git backfill completed for all selected repos. Strict derivation produced 1 Sphinx `comment2context` candidate, but manual review rejected it as the already-known weak wording-only "formerly problematic" case; no reset12 sample was accepted.
- A reset13 metadata-only pass over pip, Ruff, Scrapy, NumPy, and IPython selected 15 merged PRs. Git backfill and diagnostics completed, but manual review rejected all new edge candidates: several were accepted duplicates, wording-only, same-PR cross-contaminated, or otherwise weak. The tempting IPython `@property` review was rejected because its direct response commit only changed tests while the implementation change answered an adjacent review thread.
- A reset14 metadata-only pass over pytest, Werkzeug, SQLAlchemy, ESLint, and Prettier selected 7 merged PRs before anonymous rate limit exhaustion. Git backfill completed for pytest, ESLint, and Prettier. Strict derivation produced 5 pytest candidates, all duplicate or invalid. Manual diagnostics review accepted 1 ESLint cross-directory `comment2context` sample: a review on `lib/rules/no-param-reassign.js` whose missing configuration context is `eslint.config.js`. It passed corpus-required V1.1 audit packet preflight and was manually accepted in `data/reports/v1_1/manual_audit_reset14_comment2context.jsonl`.
- Reset15/reset16/reset17 mining and reset18 manual audit added 7 more accepted `comment2context` rows from Playwright, Ruff, and pip. The reset18 audit packet passed all 7 candidates with corpus-required validation using `data/corpus/v1_1_comment_candidates_reset18_manual_bigfile/corpus_manifest.jsonl`; manual decisions live in `data/reports/v1_1/manual_audit_reset18_comment2context.jsonl`. These additions close the `comment2context` lower-bound gap in the V19 assembly.
- The local V1.1 candidate pool reached the planned sample-count band, and the returned embedding artifacts now complete the final leaderboard/release report gate.
- Unauthenticated GitHub REST access is currently exhausted in this environment. CLI GitHub calls now use `--max-rate-limit-sleep` with a 60s default so exhausted rate limits fail fast instead of silently sleeping through a long reset window.
- Hosted Voyage evaluation is optional/deferred and no longer part of the V1.1 release gate.

## Non-Negotiable Invariants

- Do not modify frozen V1 samples in `benchmark/v1`.
- Assemble V1.1 into `data/benchmark/v1_1` or another explicit V1.1 output directory.
- Preserve all V1 sample IDs in V1.1.
- Do not expand `code2test` by default.
- Only accept new `comment2context` and `trace2code` samples.
- Reject any new sample with query leakage, direct gold path hints, direct gold basename hints, missing gold, missing audit evidence, or overlapping gold/supporting/distractor path roles.
- Require new gold files to exist in the base-commit corpus before release.

## Target Composition

| Track | V1 | V1.1 Target | Needed Accepted Additions |
| --- | ---: | ---: | ---: |
| `code2test` | 106 | 106 | 0 |
| `comment2context` | 51 | 80-100 | 29-49 |
| `trace2code` | 68 | 100+ | 32+ |

## Candidate Priorities

`comment2context` candidates should emphasize:

- Review comments whose missing context is outside the commented file.
- Cross-module, cross-package, or cross-layer evidence.
- Gold files outside the commented file's directory when possible.
- Clear `given_files` for the reviewed/commented file.
- Queries that describe the review problem without naming the gold path or basename.

`trace2code` candidates should emphasize:

- Real reproduced failure text rather than synthetic bug descriptions.
- Root-cause source files as gold, not only failing tests.
- More non-Go repositories.
- More gold-file languages/extensions beyond the current Go-heavy mix.
- Diverse failure signals: assertion failures, panics, tracebacks, exceptions, compile errors, timeouts, and general test failures.

## Execution Sequence

1. Mine or import V1.1 candidate samples for `comment2context` and `trace2code`.
   - For `comment2context`, prefer `arb crawl-review-comments --repo ... --out data/raw_v1_1_comment_candidates --metadata-only`, then `arb backfill-git-raw --raw data/raw_v1_1_comment_candidates --repo ...`, then `arb derive --raw ...` so REST budget is focused on PR metadata with review-comment evidence.
2. Build the matching base-commit corpus for every candidate repository/commit.
3. Generate a human audit packet:

```bash
arb v1-1-audit-packet \
  --candidate data/benchmark/v1_1_candidates \
  --base-derived data/benchmark/v1 \
  --out data/reports/v1_1/audit_packet \
  --corpus-manifest data/corpus/v1_1/corpus_manifest.jsonl \
  --require-corpus
```

4. Manually audit candidates and keep only accepted rows with defensible evidence.
5. Assemble V1.1:

```bash
arb assemble-v1-1 \
  --base-derived data/benchmark/v1 \
  --expansion data/benchmark/v1_1_candidates \
  --out data/benchmark/v1_1 \
  --corpus-manifest data/corpus/v1_1/corpus_manifest.jsonl \
  --require-corpus \
  --audit data/reports/v1_1/audit_packet/audit_samples.csv \
  --require-audit-keep
```

6. Run readiness before any release. These short paths are templates for a fresh local assembly; the current V19 release candidate uses the exact long paths recorded in `data/reports/v1_1/baseline_handoff_v19.md` and `data/reports/v1_1/embedding_runbook_v19.md`.

```bash
arb v1-1-readiness \
  --derived data/benchmark/v1_1 \
  --base-derived data/benchmark/v1 \
  --manifest data/benchmark/v1_1/manifest.json \
  --corpus-manifest data/corpus/v1_1/corpus_manifest.jsonl \
  --eval-dir data/eval/v1_1 \
  --leaderboard data/reports/v1_1/model_leaderboard.md \
  --leaderboard-json data/reports/v1_1/model_leaderboard.json
```

7. Run all required baselines with `--candidate-filter all_files` and no skipped samples:

- `lexical`
- `aider-style-repomap`
- `jina-code-embeddings-0.5b`
- `qwen3-embedding-4b`

8. Regenerate V1.1 leaderboard and release report. For the current V19 flow, prefer `arb v1-1-finalize-baselines --handoff data/reports/v1_1/baseline_handoff_v19.json --shard-commands data/reports/v1_1/baseline_shard_commands_v19.json --return-manifest data/reports/v1_1/baseline_return_manifest_v19.json --include-shard-artifacts --auto-merge-shards` after returned artifacts have been applied.

```bash
arb report-models \
  --eval-dir data/eval/v1_1 \
  --out data/reports/v1_1/model_leaderboard.md \
  --json-out data/reports/v1_1/model_leaderboard.json

arb report-v1-1 \
  --readiness data/reports/v1_1/readiness.json \
  --leaderboard-json data/reports/v1_1/model_leaderboard.json \
  --out data/reports/v1_1/analysis.md \
  --json-out data/reports/v1_1/analysis.json
```

## Release Gate

V1.1 is ready only when all of these are true:

- `comment2context` has 80-100 total samples.
- `trace2code` has at least 100 total samples.
- `code2test` remains at 106 samples.
- Sample IDs are unique.
- All frozen V1 IDs are present in V1.1.
- Every new sample passes schema, clear-semantics, leakage, direct-hint, path-role-overlap, audit-evidence, and gold-in-corpus checks.
- New `comment2context` samples include `given_files`, avoid same-directory shortcut gold, and add cross-module context coverage.
- New `trace2code` samples do not use test files as primary gold.
- New `trace2code` samples add non-Go coverage, at least two gold-file language/extension buckets, and at least two real classified failure-signal types; `unknown` does not count toward diversity.
- Required baseline summaries and per-sample details are complete for all V1.1 samples with `skipped={}`, and details recompute to the summary metrics.
- V1.1 leaderboard Markdown and JSON include every required baseline: lexical, RepoMap, Jina, and Qwen.
- Public docs describe V1.1 as a focused improvement over V1, not a raw-scale release.

## Immediate Blockers

- Need final V1.1 report artifacts regenerated from the required open-source baseline summaries/details: lexical, RepoMap, `jina-code-embeddings-0.5b`, and `qwen3-embedding-4b`. Optional additional open-source model rows can be included in the leaderboard, but Voyage is not a blocker.
- Need GPU/precomputed embedding caches for Jina and Qwen; see `data/reports/v1_1/embedding_blockers_v19.md`.
- Use `--shared-text-cache` for the missing embedding runs to reuse duplicate chunk-text embeddings within each cache scope; generated parallel shard commands use shard-local text caches, while a serial/coordinated run can use one cache for whole-run deduplication. See `data/reports/v1_1/embedding_dedup_v19.md`.
- Use `--shard-count/--shard-index` directly, or generate explicit worker files with `arb v1-1-write-sample-shards` and pass them with `--sample-id-file`, if the external GPU run needs to be split across workers; merge shard outputs with `arb v1-1-merge-details` before rebuilding final summaries.
- Use `data/reports/v1_1/embedding_resource_estimate_v19.md` for GPU storage planning.
- Follow `data/reports/v1_1/embedding_runbook_v19.md` once GPU resources are available, then rerun readiness and release reporting.
- Use `data/reports/v1_1/baseline_handoff_v19.md` or `.json` as the external-run checklist so the returned artifacts and verification reports use the current V19 paths.
- Use `data/reports/v1_1/baseline_transfer_manifest_v19.files` as the concrete input file list for rsync/tar transfer to the GPU machine; it includes the corpus chunk files plus helper files for sample shards, shard commands, executable shard scripts, transfer-unpack bootstrap script, return manifest, return-bundle packaging/apply scripts, filtered partial-return apply scripts, runbook, and the default release docs checked by finalization. The JSON/Markdown transfer manifest includes sha256 fingerprints for those helper files. If a single movable artifact is easier, regenerate `data/reports/v1_1/baseline_transfer_bundle_v19.tar.zst` with `arb v1-1-baseline-transfer-bundle`; it writes the checksum and verifies the exact tar members before copy. The non-bundled `data/reports/v1_1/external_runner_copy_packet_v19.md` is the shortest local checklist for copying the prepared archive, checksum, and bootstrap script; refresh it with `arb v1-1-external-runner-copy-packet` after the bundle hash changes. Sender-side completion-audit and compact return-acceptance files are intentionally not transferred; the copy packet and sender-side preflight are the current archive hash/size authority. Copy `data/reports/v1_1/unpack_v19_transfer_bundle.sh` beside the prepared archive and checksum on the external runner to verify, unpack, and run transfer/handoff checks in one step. If only basename files are copied, use that bootstrap instead of raw `sha256sum -c`; the checksum sidecar records the generated repo-relative bundle path, and the bootstrap verifies the supplied bundle path against the recorded hash. The current prepared archive has a local reporting-checkout unpack-smoke report at `data/reports/v1_1/baseline_transfer_unpack_smoke_v19.json` and `.md`; the unpacked copy also passes `v1-1-verify-handoff` in `data/reports/v1_1/baseline_handoff_unpack_smoke_v19.json` and `.md`.
- Use `arb v1-1-verify-transfer-bundle --bundle data/reports/v1_1/baseline_transfer_bundle_v19.tar.zst --manifest data/reports/v1_1/baseline_transfer_manifest_v19.json` before copying an already prepared transfer archive; it checks the SHA256 sidecar, `zstd -t`, exact tar members against the transfer manifest file list, and regular-file tar member types.
- Run `arb v1-1-verify-transfer-manifest --manifest data/reports/v1_1/baseline_transfer_manifest_v19.json` on the external machine after transfer and before starting GPU baselines to catch missing files, chunk-size mismatches, or stale helper scripts.
- Run `arb v1-1-external-runner-failfast-smoke` only on a local checkout where `v1-1-external-runner-preflight` reports expected blockers; it verifies the generated scripts stop at transfer/runtime/return-file gates before expensive shard work.
- Use `data/reports/v1_1/sample_shards_v19/` if the external run uses four explicit sample-id worker files; regenerate with `arb v1-1-write-sample-shards --shard-count N` if the worker count differs.
- Use `data/reports/v1_1/baseline_shard_commands_v19.md` if the external run uses those sample-id shards; it contains the shard commands and the required merge/summary recovery commands for each missing embedding baseline.
- Use `data/reports/v1_1/run_v19_baseline_shards.sh` on the external runner if a serial execution script is preferred over manually copying the shard commands; after regeneration it should verify the transfer manifest, write `baseline_transfer_manifest_verify_v19.json/.md`, verify handoff fingerprints, check `numpy`, check `sentence-transformers` plus CUDA-capable Torch for Jina/Qwen, create output/cache directories, print `df -h` for those locations, run shard jobs, merge, recover summaries, refresh the return manifest with `--require-existing`, package and verify the return bundle, and then run `v1-1-finalize-baselines` to regenerate the final leaderboard/readiness/release/audit gates with transfer-unpack smoke, transfer-bundle, return-bundle, and return-manifest workflow evidence present. Optional Voyage scripts may be run separately, but they are not required for V1.1.
- Use `data/reports/v1_1/package_v19_return_artifacts.sh` on the external runner after the shard jobs and merge/summary commands finish. It refreshes `baseline_return_manifest_v19.json` with `--require-existing`, runs `check-baseline-summaries` so stale, partial, wrong-sample, or metric-inconsistent summary/details files fail before transfer, writes `baseline_return_bundle_v19.files` from existing artifacts only, creates `baseline_return_bundle_v19.tar.zst`, writes the SHA256 checksum, tests the compressed bundle, verifies the tar member list exactly matches the generated file list, and writes a `v1-1-verify-return-bundle` report before the bundle is copied back for local preflight and final reporting.
- Use `data/reports/v1_1/apply_v19_return_artifacts.sh` on the local reporting checkout after copying back `baseline_return_bundle_v19.tar.zst` and its checksum. It verifies the checksum, tests the compressed bundle, rejects unsafe tar member paths and non-regular tar members, checks bundle members against the return manifest before unpacking, writes a structured return-bundle verification report, unpacks returned artifacts, checks required files, and runs `v1-1-finalize-baselines` with the V19 handoff, shard commands, return manifest, and finalization paths. The generated apply scripts read the expected hash from the first checksum column and verify the configured bundle path, so the checksum sidecar may record the external runner's path without breaking local apply; set `ARB_RETURN_BUNDLE` and `ARB_RETURN_CHECKSUM` if the returned files are copied to non-default paths.
- If results return from separate machines, use `package_v19_gpu_return_artifacts.sh` plus `apply_v19_gpu_return_artifacts.sh` for the required Jina/Qwen artifacts. Optional Voyage partial-return scripts may still be used explicitly, but finalization should wait only for the required open-source summary/details pairs.
- Run `arb v1-1-verify-handoff --handoff data/reports/v1_1/baseline_handoff_v19.json` on the external machine before starting GPU baselines.
- Run `arb v1-1-baseline-status --shard-commands data/reports/v1_1/baseline_shard_commands_v19.json` before, during, and after the external baseline run to capture runtime blockers, per-shard details progress, partial final details progress, and final artifact blockers.
- Run `arb v1-1-external-runner-preflight` before starting or packaging external runs when you need a compact report of transfer/handoff readiness, generated runner-script blockers, and return-packaging blockers; in the unpacked checkout it accepts the bootstrap unpack-smoke verifier reports and falls back to a live runtime probe if `baseline_status_v19.json` is absent. On the reporting checkout before copy, pass `--copy-packet data/reports/v1_1/external_runner_copy_packet_v19.json` to verify the copy checklist still matches the latest transfer bundle, the three copy-source files exist, the bundle hash/size matches, the checksum sidecar records the same hash, and the required return-file list is current.
- If `arb v1-1-baseline-status` reports complete partial details but a missing summary, run `arb v1-1-summary-from-details` before rerunning expensive embeddings.
- Run `arb check-baseline-summaries` on returned summary/details files before regenerating the final leaderboard, including the sample-ID and metric-consistency checks.
- Run `arb report-models` with all five `--required-baseline` labels so missing leaderboard rows fail before release reporting.
- Or run `arb v1-1-finalize-baselines --handoff data/reports/v1_1/baseline_handoff_v19.json --shard-commands data/reports/v1_1/baseline_shard_commands_v19.json --return-manifest data/reports/v1_1/baseline_return_manifest_v19.json --include-shard-artifacts --auto-merge-shards` after artifacts are copied back; it performs handoff verification, complete-shard merge/summary recovery, return-manifest refresh, baseline status, summary preflight, leaderboard, readiness, release, and completion-audit regeneration in one ordered gate.
- Need to regenerate the final V1.1 leaderboard and release report after the missing baseline summaries exist.
- Run `arb report-v1-1-completion-audit` after release reporting and only mark the thread goal complete if it reports `overall_status=complete`.
- Optional: mine a small extra `comment2context` buffer above exactly 80 samples if release policy wants slack against later manual drops.
