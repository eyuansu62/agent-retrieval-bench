# Agent Retrieval Bench

Agent Retrieval Bench is an action-oriented code retrieval benchmark for coding agents. It evaluates whether a retriever can find the repository files an agent would need for real coding workflows.

The current public release candidate is **V1 Code Review RC**: 157 manually curated samples across `code2test` and `comment2context`.

Raw crawling, weak-label generation, and audit workflows are intentionally not documented as the public path. Use this repository to run evaluations against the released benchmark artifacts on Hugging Face.

## Release Contents

The prebuilt benchmark and corpus are hosted on Hugging Face Datasets:

```text
https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench
```

```text
data/benchmark/v1_code_review/
  manifest.json
  samples.jsonl
  code2test.jsonl
  comment2context.jsonl
data/corpus/v1_code_review/
  corpus_manifest.jsonl
  **/*.chunks.jsonl
data/eval/v1_code_review/
  lexical_summary.json
  lexical_details.jsonl
  jina-code-embeddings-0.5b_summary.json
  qwen3-embedding-4b_summary.json
data/reports/v1_code_review/
  model_leaderboard.md
  model_leaderboard.json
  status.md
```

V1 Code Review RC intentionally excludes `trace2code`. Full V1 will add trace/root-cause retrieval only after enough real stack-trace or CI-failure samples pass manual audit.

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Install optional embedding dependencies only when running embedding model evaluations:

```bash
pip install -e ".[embedding]"
```

No GitHub API token is required to download, validate, or evaluate the released benchmark.

## Download V1 Code Review

Download the benchmark metadata and reports:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --repo-type dataset \
  --local-dir data \
  --include "benchmark/v1_code_review/**" \
  --include "eval/v1_code_review/**" \
  --include "reports/v1_code_review/**"
```

Download and extract the prebuilt corpus archive:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --repo-type dataset \
  --local-dir data \
  --include "corpus/v1_code_review_corpus.tar.zst" \
  --include "corpus/v1_code_review_corpus.tar.zst.sha256"

cd data
sha256sum -c corpus/v1_code_review_corpus.tar.zst.sha256
rm -rf corpus/v1_code_review
zstd -dc corpus/v1_code_review_corpus.tar.zst | tar -xf - -C corpus
cd ..
```

If your `tar` prints `LIBARCHIVE.xattr.com.apple.provenance` warnings, they are macOS extended-attribute warnings and do not affect the extracted corpus.

## Evaluate

Validate the benchmark samples:

```bash
arb validate data/benchmark/v1_code_review/*.jsonl
```

Run the lexical baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v1_code_review \
  --corpus data/corpus/v1_code_review \
  --out data/eval/v1_code_review/lexical_summary.json \
  --details data/eval/v1_code_review/lexical_details.jsonl \
  --no-keep-list
```

Run Jina Code Embeddings:

```bash
arb eval-embedding \
  --model /path/to/jina-code-embeddings-0.5b \
  --derived data/benchmark/v1_code_review \
  --corpus data/corpus/v1_code_review \
  --out data/eval/v1_code_review/jina-code-embeddings-0.5b_summary.json \
  --details data/eval/v1_code_review/jina-code-embeddings-0.5b_details.jsonl \
  --cache data/embeddings/v1_code_review/jina-code-embeddings-0.5b \
  --candidate-filter all_files \
  --batch-size 8 \
  --device cuda \
  --trust-remote-code \
  --no-keep-list
```

Run Qwen3 Embedding:

```bash
arb eval-embedding \
  --model /path/to/Qwen3-Embedding-4B \
  --derived data/benchmark/v1_code_review \
  --corpus data/corpus/v1_code_review \
  --out data/eval/v1_code_review/qwen3-embedding-4b_summary.json \
  --details data/eval/v1_code_review/qwen3-embedding-4b_details.jsonl \
  --cache data/embeddings/v1_code_review/qwen3-embedding-4b \
  --candidate-filter all_files \
  --batch-size 8 \
  --device cuda \
  --trust-remote-code \
  --no-keep-list
```

Generate a leaderboard from all `*_summary.json` files:

```bash
arb report-models \
  --eval-dir data/eval/v1_code_review \
  --out data/reports/v1_code_review/model_leaderboard.md \
  --json-out data/reports/v1_code_review/model_leaderboard.json
```

The default candidate set is `all_files`. The primary metric is overall `MRR`; report `Recall@5`, `Recall@10`, `Recall@20`, and `gold_coverage@8k` alongside it.

## Current Leaderboard

All runs evaluate 157 samples with `skipped={}`.

| Task | Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | Qwen3-Embedding-4B | 157 | 0.3877 | 0.5344 | 0.6749 | 0.3179 | 0.3389 |
| overall | jina-code-embeddings-0.5b | 157 | 0.3025 | 0.4023 | 0.5227 | 0.2441 | 0.2176 |
| overall | lexical | 157 | 0.1083 | 0.2113 | 0.3535 | 0.1013 | 0.0456 |
| code2test | Qwen3-Embedding-4B | 106 | 0.4531 | 0.5965 | 0.7292 | 0.3182 | 0.3840 |
| code2test | jina-code-embeddings-0.5b | 106 | 0.2752 | 0.4009 | 0.5211 | 0.2038 | 0.2060 |
| code2test | lexical | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 |
| comment2context | jina-code-embeddings-0.5b | 51 | 0.3595 | 0.4052 | 0.5261 | 0.3280 | 0.2418 |
| comment2context | Qwen3-Embedding-4B | 51 | 0.2516 | 0.4052 | 0.5621 | 0.3173 | 0.2451 |
| comment2context | lexical | 51 | 0.1928 | 0.3595 | 0.5752 | 0.1739 | 0.0784 |

## Benchmark Semantics

- Every sample is evaluated against `repo_at_base_commit`; fixed code is not indexed.
- `code2test` queries describe implementation changes or PR intent; gold files are related tests.
- `comment2context` queries describe review comments; the commented file is treated as given context, and scoring uses only additional required context files.
- `gold_coverage@8k` measures whether gold files appear within an 8k-character retrieval budget.

Legacy V0.2 remains available under `data/benchmark/v0_2/`, but new model comparisons should report V1 Code Review RC as the primary benchmark.
