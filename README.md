# Agent Retrieval Bench

Agent Retrieval Bench is an action-oriented code retrieval benchmark for coding agents. The benchmark evaluates whether a retriever can find the repository files an agent would need for tasks such as mapping implementation changes to tests, resolving review comments, and following stack traces.

The current public release is **Benchmark V0.2**: 62 manually curated samples across `code2test`, `comment2context`, and `trace2code`. This repository is intended to be used as a benchmark/evaluation repo; raw crawling and dataset-construction workflows are intentionally not documented as the public path.

## V0.2 Contents

```text
data/benchmark/v0_2/
  manifest.json
  samples.jsonl
  code2test.jsonl
  comment2context.jsonl
  trace2code.jsonl
data/corpus/v0_2/
  corpus_manifest.jsonl
  **/*.chunks.jsonl
data/eval/v0_2/
  lexical_summary.json
  lexical_details.jsonl
data/audit/v0_2/
  keep_samples.jsonl
```

V0.2 intentionally excludes `testlog2code`: the audited cleaned slice had only 7 valid samples out of 44, so it is not reliable enough for this release. V0.2 upgrades `comment2context` to score only extra required context files instead of the commented file itself.

The prebuilt V0.2 samples and corpus are hosted on Hugging Face Datasets:

```text
https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench
```

## Setup

No runtime dependencies are required beyond Python 3.10+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Install optional embedding dependencies only when running embedding model evaluations:

```bash
pip install -e ".[embedding]"
```

No GitHub API token is required to download, inspect, validate, or evaluate the public V0.2 benchmark.

## Evaluate V0.2

Download the prebuilt V0.2 benchmark and candidate corpus from Hugging Face into the repo-local `data/` directory:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --repo-type dataset \
  --local-dir data \
  --include "benchmark/v0_2/**" \
  --include "corpus/v0_2/**" \
  --include "eval/v0_2/**" \
  --include "audit/v0_2/**" \
  --include "audit/v0_2_round3_more/summary.json"
```

Validate the benchmark samples:

```bash
arb validate data/benchmark/v0_2/*.jsonl
```

Run the lexical/exact retrieval baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v0_2 \
  --keep-list data/audit/v0_2/keep_samples.jsonl \
  --corpus data/corpus/v0_2 \
  --out data/eval/v0_2/lexical_summary.json
```

Run an embedding model evaluation:

```bash
arb eval-embedding \
  --model jinaai/jina-code-embeddings-0.5b \
  --derived data/benchmark/v0_2 \
  --keep-list data/audit/v0_2/keep_samples.jsonl \
  --corpus data/corpus/v0_2 \
  --candidate-filter code_only \
  --out data/eval/v0_2/jina-code-embeddings-0.5b_code_only_summary.json
```

Embedding evaluation prints progress by default, including model loading, corpus loading, cache hits/misses, chunk encoding, and sample evaluation. Use `--no-progress` for quiet runs. The first run for a model can be slow because it embeds full base-commit corpora; later runs reuse `data/embeddings/v0_2/`.

Use `--candidate-filter tests_only` to isolate `code2test` behavior against test files, or `--candidate-filter code_only` to exclude docs/changelogs/templates from the candidate set. Details JSONL includes `gold_ranks` for each gold file, with `null` when the gold file is not retrieved.

Generate a model leaderboard from all `*_summary.json` files in `data/eval/v0_2/`:

```bash
arb report-models \
  --eval-dir data/eval/v0_2 \
  --out data/reports/v0_2/model_leaderboard.md \
  --json-out data/reports/v0_2/model_leaderboard.json
```

Regenerate the diagnostic report:

```bash
arb diagnose \
  --samples data/benchmark/v0_2/samples.jsonl \
  --corpus-manifest data/corpus/v0_2/corpus_manifest.jsonl \
  --details data/eval/v0_2/lexical_details.jsonl \
  --out data/reports/v0_2
```

Generate a V1 hard-mining report and candidate pool:

```bash
arb hardness \
  --derived data/benchmark/v0_2 \
  --corpus-manifest data/corpus/v0_2/corpus_manifest.jsonl \
  --details data/eval/v0_2/lexical_details.jsonl \
  --keep-list data/audit/v0_2/keep_samples.jsonl \
  --out data/reports/v0_2 \
  --pool-out data/reports/v0_2/candidate_keep_pool.jsonl
```

`arb hardness` annotates each sample with direct path hints, basename hints, module-token overlap, same-directory gold, lexical rank bucket, gold count, and task-balance weight. The generated `candidate_keep_pool.jsonl` sorts hard candidates first for V1 manual audit; V0.2 remains frozen.

Filter the hard candidate pool into a V1 seed set:

```bash
arb hard-pool-filter \
  --pool data/reports/v0_2/candidate_keep_pool.jsonl \
  --audit data/reports/v0_2/top_hard_first_pass_audit.jsonl \
  --out data/reports/v0_2/v1_seed_candidates.jsonl \
  --summary data/reports/v0_2/v1_seed_summary.json \
  --audit-out data/reports/v0_2/v1_seed_audit_samples.jsonl \
  --audit-csv data/reports/v0_2/v1_seed_audit_samples.csv
```

`arb hard-pool-filter` deduplicates by repo, PR, and gold files; applies manual audit verdicts; drops obvious generated/template/generic-test noise; and writes the next V1 seed audit sheet.

Summarize a V1 seed audit sheet and write a keep list:

```bash
arb seed-audit-summary \
  data/reports/v0_2/v1_seed_audit_samples.csv \
  --out data/reports/v0_2/v1_seed_audit_summary.json \
  --keep-list data/reports/v0_2/v1_seed_keep.jsonl
```

Merge multiple V1 seed audit rounds into one freeze-gate keep list:

```bash
arb merge-seed-audits \
  --audit data/reports/v1_candidate_round1/v1_seed_audit_samples.csv \
  --audit data/reports/v1_candidate_round2/v1_seed_audit_samples.csv \
  --out data/reports/v1_seed_round1/audit_summary.json \
  --keep-list data/reports/v1_seed_round1/keep_samples.jsonl
```

`arb merge-seed-audits` accepts CSV and JSONL inputs, deduplicates repeated sample IDs, fails closed on conflicting verdicts, and keeps only `valid` rows explicitly marked `keep=true`.

Merge existing local benchmark/derived samples into a V1 hard-mining candidate set without crawling new repos:

```bash
arb export-hardmine-candidates \
  --corpus-manifest data/corpus/v0_2/corpus_manifest.jsonl \
  --require-corpus \
  --out data/benchmark/v1_candidate_round1
```

`arb export-hardmine-candidates` scans `data/benchmark/v0_2`, `data/derived_v0_2*`, and `data/derived_token_logs` by default; keeps only `code2test`, `comment2context`, and real `trace2code`; deduplicates by `sample_id`; drops leaked, schema-invalid, missing-gold, and missing-corpus samples; and writes both `samples.jsonl` and per-task JSONL files.

Compare a curated V1 seed against V0.2 and its audit summary:

```bash
arb report-v1-seed \
  --audit-summary data/reports/v1_candidate_round1/v1_seed_audit_summary.json \
  --out data/reports/v1_candidate_round1/v1_seed_comparison.md \
  --json-out data/reports/v1_candidate_round1/v1_seed_comparison.json
```

If the first audit is short, generate a second review batch from the remaining hard pool:

```bash
arb hard-pool-filter \
  --pool data/reports/v1_candidate_round1/candidate_keep_pool.jsonl \
  --audit data/reports/v1_candidate_round1/v1_seed_audit_samples.csv \
  --exclude-audited \
  --task-priority code2test,trace2code,comment2context \
  --out data/reports/v1_candidate_round2/v1_seed_candidates.jsonl \
  --summary data/reports/v1_candidate_round2/v1_seed_summary.json \
  --audit-out data/reports/v1_candidate_round2/v1_seed_audit_samples.jsonl \
  --audit-csv data/reports/v1_candidate_round2/v1_seed_audit_samples.csv
```

## Current Lexical Baseline

The published V0.2 lexical baseline evaluates all 62 samples with no skipped samples.

| Task | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code2test` | 17 | 0.2353 | 0.2941 | 0.3676 | 0.1418 | 0.1176 |
| `comment2context` | 43 | 0.4535 | 0.6512 | 0.8721 | 0.2482 | 0.1512 |
| `trace2code` | 2 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| `overall` | 62 | 0.4113 | 0.5645 | 0.7379 | 0.2352 | 0.1694 |

V0.1 remains available as a legacy 35-sample artifact under `data/benchmark/v0_1/`. New model comparisons should report V0.2 as the primary benchmark.

## Benchmark Semantics

- Every sample is evaluated against `repo_at_base_commit`; fixed code must not be indexed.
- `code2test` gold files are related tests.
- V0.1 `comment2context` gold files are root-cause files when available, otherwise related tests.
- V0.2+ `comment2context` treats the commented file as `gold.given_files` and scores only extra required context from `gold.must_context_files` / `gold.context_files`.
- `trace2code` gold files are root-cause files when available, otherwise related tests.
- Baseline queries fail closed if raw patch markers or fix commit hashes appear in the query.
- Large generated artifacts such as `data/repos/` and `data/corpus/` are local evaluation caches and are ignored by git.
