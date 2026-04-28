# Agent Retrieval Bench

Agent Retrieval Bench is an action-oriented code retrieval benchmark for coding agents. The benchmark evaluates whether a retriever can find the repository files an agent would need for tasks such as mapping implementation changes to tests, resolving review comments, and following stack traces.

The checked-in release is **Benchmark V0.1**: 35 manually curated samples across `code2test`, `comment2context`, and `trace2code`. This repository is intended to be used as a benchmark/evaluation repo; raw crawling and dataset-construction workflows are intentionally not documented as the public path.

## V0.1 Contents

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

V0.1 intentionally excludes `testlog2code`: the audited cleaned slice had only 7 valid samples out of 44, so it is not reliable enough for this release.

The prebuilt V0.1 corpus is hosted on Hugging Face Datasets:

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

No GitHub API token is required to inspect the checked-in V0.1 benchmark, validate samples, or read the published baseline and diagnostic report.

## Evaluate V0.1

Validate the checked-in benchmark samples:

```bash
arb validate data/benchmark/v0_1/*.jsonl
```

The candidate corpus is not committed to GitHub because it is large. Download the prebuilt V0.1 corpus from Hugging Face into the repo-local `data/` directory:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --type dataset \
  --local-dir data \
  --include "corpus/v0_1/**"
```

Run the lexical/exact retrieval baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v0_1 \
  --keep-list data/audit/v0_clean/keep_samples.jsonl \
  --corpus data/corpus/v0_1 \
  --out data/eval/v0_1/lexical_summary.json
```

Run an embedding model evaluation:

```bash
arb eval-embedding \
  --model jinaai/jina-code-embeddings-0.5b \
  --derived data/benchmark/v0_1 \
  --keep-list data/audit/v0_clean/keep_samples.jsonl \
  --corpus data/corpus/v0_1 \
  --candidate-filter all_files
```

Embedding evaluation prints progress by default, including model loading, corpus loading, cache hits/misses, chunk encoding, and sample evaluation. Use `--no-progress` for quiet runs. The first run for a model can be slow because it embeds full base-commit corpora; later runs reuse `data/embeddings/v0_1/`.

Use `--candidate-filter tests_only` to isolate `code2test` behavior against test files, or `--candidate-filter code_only` to exclude docs/changelogs/templates from the candidate set. Details JSONL includes `gold_ranks` for each gold file, with `null` when the gold file is not retrieved.

Regenerate the diagnostic report:

```bash
arb diagnose \
  --samples data/benchmark/v0_1/samples.jsonl \
  --corpus-manifest data/corpus/v0_1/corpus_manifest.jsonl \
  --details data/eval/v0_1/lexical_details.jsonl \
  --out data/reports/v0_1
```

## Current Lexical Baseline

The checked-in V0.1 lexical baseline evaluates all 35 samples with no skipped samples.

| Task | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code2test` | 17 | 0.2353 | 0.2941 | 0.3676 | 0.1418 | 0.1176 |
| `comment2context` | 16 | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 1.0000 |
| `trace2code` | 2 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| `overall` | 35 | 0.6286 | 0.6571 | 0.6929 | 0.5546 | 0.5714 |

See `data/reports/v0_1/report.md` for the V0.1 diagnostic report. The main takeaway is that `code2test` is the useful hard slice, while `comment2context` and `trace2code` are currently closer to smoke/easy checks because many queries contain direct path hints or have too few samples.

## Benchmark Semantics

- Every sample is evaluated against `repo_at_base_commit`; fixed code must not be indexed.
- `code2test` gold files are related tests.
- `comment2context` and `trace2code` gold files are root-cause files when available, otherwise related tests.
- Baseline queries fail closed if raw patch markers or fix commit hashes appear in the query.
- Large generated artifacts such as `data/repos/` and `data/corpus/` are local evaluation caches and are ignored by git.
