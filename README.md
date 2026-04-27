# Agent Retrieval Bench

This repository contains a crawler, curation pipeline, and baseline evaluator for an action-oriented code retrieval benchmark for coding agents. It collects GitHub pull request signals that agents actually use: review comments, changed implementation files, related tests, test failures, and stack traces.

The current checked-in artifact is **Benchmark V0.1**: 35 manually curated samples across `code2test`, `comment2context`, and `trace2code`. Large raw crawls, bare repository caches, and generated candidate corpora are intentionally not tracked in git.

## What It Builds

The crawler targets four benchmark task families:

- `comment2context`: PR review comment plus path context to the code and tests that should be inspected.
- `code2test`: changed implementation file to related test files.
- `testlog2code`: failed check output to likely root-cause implementation files and tests.
- `trace2code`: stack trace or runtime error to likely root-cause files.

All derived samples point at `repo_at_base_commit`. The raw crawler stores PR metadata, changed files, review comments, commits, and check runs as append-only JSONL so samples can be regenerated with stricter rules later.

## Setup

No runtime dependencies are required beyond Python 3.10+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Set a GitHub token for anything beyond tiny dry runs:

```bash
export GITHUB_TOKEN=...
```

## Current V0.1 Artifacts

```text
data/benchmark/v0_1/
  manifest.json
  samples.jsonl
  code2test.jsonl
  comment2context.jsonl
  trace2code.jsonl
data/eval/v0_1/
  lexical_summary.json
  lexical_details.jsonl
data/reports/v0_1/
  report.md
  diagnostic_summary.json
  sample_diagnostics.jsonl
```

V0.1 intentionally excludes `testlog2code`: the audited cleaned slice had only 7 valid samples out of 44.

## Usage

Generate the repository manifest:

```bash
arb manifest --targets configs/crawl_targets.json --output data/repo_manifest.jsonl
```

Dry-run one repo without writing raw PR artifacts:

```bash
arb crawl --repo pytest-dev/pytest --limit-prs 20 --dry-run
```

Crawl one repo:

```bash
arb crawl --repo pytest-dev/pytest --limit-prs 20 --out data/raw
```

Crawl all primary targets:

```bash
arb crawl-all --targets configs/crawl_targets.json --limit-prs 20 --out data/raw
```

Derive weak samples:

```bash
arb derive --raw data/raw --out data/derived
```

Validate derived samples:

```bash
arb validate data/derived/*.jsonl
```

Generate manual audit sheets:

```bash
arb audit-sample --derived data/derived_token_logs --out data/audit/v0 --per-task 20
```

Summarize audited verdicts and write a keep list:

```bash
arb audit-summary data/audit/v0/audit_samples.csv --out data/audit/v0/summary.json --keep-list data/audit/v0/keep_samples.jsonl
```

Export audited valid samples into benchmark V0:

```bash
arb export-curated --derived data/derived_token_logs --keep-list data/audit/v0/keep_samples.jsonl --out data/benchmark/v0
```

Export the curated V0.1 benchmark:

```bash
arb export-curated \
  --derived data/derived_v0_clean \
  --keep-list data/audit/v0_clean/keep_samples.jsonl \
  --out data/benchmark/v0_1 \
  --tasks code2test,comment2context,trace2code
```

Verify that base commits can be fetched into bare local repo caches:

```bash
arb verify-bases --raw data/raw --repos-dir data/repos --limit 50
```

Build candidate file/function chunks at each sample `base_commit`:

```bash
arb build-corpus --derived data/benchmark/v0 --keep-list data/audit/v0/keep_samples.jsonl --repos-dir data/repos --out data/corpus/v0
```

Run the lexical/exact retrieval baseline:

```bash
arb eval-baseline --derived data/benchmark/v0 --keep-list data/audit/v0/keep_samples.jsonl --corpus data/corpus/v0 --out data/eval/v0/lexical_summary.json
```

Run the V0.1 lexical/exact retrieval baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v0_1 \
  --keep-list data/audit/v0_clean/keep_samples.jsonl \
  --corpus data/corpus/v0_1 \
  --out data/eval/v0_1/lexical_summary.json
```

Diagnose V0.1 difficulty and data quality:

```bash
arb diagnose \
  --samples data/benchmark/v0_1/samples.jsonl \
  --corpus-manifest data/corpus/v0_1/corpus_manifest.jsonl \
  --details data/eval/v0_1/lexical_details.jsonl \
  --out data/reports/v0_1
```

For a quick data-plumbing smoke test without cloning repositories:

```bash
arb eval-baseline --derived data/derived_token_logs --dry-run --limit-samples 80
```

## Data Layout

```text
data/
  repo_manifest.jsonl
  raw/
    owner__repo/
      pull_requests.jsonl
      pull_files.jsonl
      pull_file_summary.jsonl
      review_comments.jsonl
      pull_commits.jsonl
      check_runs.jsonl
      crawl_state.json
  derived/
    comment2context.jsonl
    code2test.jsonl
    testlog2code.jsonl
    trace2code.jsonl
  audit/v0/
    audit_samples.jsonl
    audit_samples.csv
    summary.json
    keep_samples.jsonl
  benchmark/v0/
    manifest.json
    samples.jsonl
    code2test.jsonl
    comment2context.jsonl
    trace2code.jsonl
  corpus/v0/
    corpus_manifest.jsonl
    owner__repo/
      <base_commit>.chunks.jsonl
  eval/v0/
    lexical_summary.json
    lexical_details.jsonl
  reports/v0_1/
    report.md
    diagnostic_summary.json
    sample_diagnostics.jsonl
```

## Quality Rules

- Skip PRs with more than 20 changed files by default.
- Skip generated, vendor, build, lockfile, snapshot, and minified paths.
- Use base commits for candidate corpora; never index fixed code for a sample.
- Sanitize review diff hunks by removing added and removed patch lines.
- Treat generated samples as weak labels until manually audited.
- Audit sheets only expose `sample_id`, task/repo, query excerpt, gold files, and manual verdict fields.
- Baseline queries fail closed when raw patch markers or fix commit hashes appear in the query.
