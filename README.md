# Agent Retrieval Bench

Agent Retrieval Bench is an action-oriented code retrieval benchmark for coding agents. It evaluates whether a retriever can find the repository files an agent would need for real coding workflows.

The current public release is **Agent Retrieval Bench V1**: 225 manually curated samples across `code2test`, `comment2context`, and `trace2code`.

Raw crawling, weak-label generation, and audit workflows are intentionally not documented as the public path. Use this repository to run evaluations against the released benchmark artifacts on Hugging Face.

## Release Contents

The prebuilt benchmark and corpus are hosted on Hugging Face Datasets:

```text
https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench
```

```text
data/benchmark/v1/
  manifest.json
  samples.jsonl
  code2test.jsonl
  comment2context.jsonl
  trace2code.jsonl
data/corpus/v1/
  corpus_manifest.jsonl
  **/*.chunks.jsonl
data/eval/v1/
  lexical_summary.json
  lexical_details.jsonl
data/reports/v1/
  model_leaderboard.md
  model_leaderboard.json
  status.md
data/releases/v1/
  agent_retrieval_bench_v1.tar.zst
  agent_retrieval_bench_v1.tar.zst.sha256
```

The earlier `v1_code_review` release remains available as a code-review/test-retrieval track. Full V1 adds audited `trace2code` samples from local test reproduction traces.

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

No GitHub API token is required to download, validate, or evaluate the released benchmark. If the Hugging Face dataset is private or gated in your environment, authenticate with `hf auth login` or set `HF_TOKEN`.

## Download V1

Download the single-file V1 release bundle:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --repo-type dataset \
  --local-dir data \
  --include "releases/v1/*"
```

Verify and extract it:

```bash
cd data
sha256sum -c releases/v1/agent_retrieval_bench_v1.tar.zst.sha256
rm -rf benchmark/v1 corpus/v1 eval/v1 reports/v1
zstd -dc releases/v1/agent_retrieval_bench_v1.tar.zst | tar -xf - -C .
cd ..
```

The bundle contains `benchmark/v1`, `corpus/v1`, `eval/v1`, and `reports/v1`. If your `tar` prints `LIBARCHIVE.xattr.com.apple.provenance` warnings, they are macOS extended-attribute warnings and do not affect the extracted files.

## Evaluate

Validate the benchmark samples:

```bash
arb validate data/benchmark/v1/*.jsonl
```

Run the lexical baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/lexical_summary.json \
  --details data/eval/v1/lexical_details.jsonl \
  --no-keep-list
```

Run Jina Code Embeddings:

```bash
arb eval-embedding \
  --model /path/to/jina-code-embeddings-0.5b \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/jina-code-embeddings-0.5b_summary.json \
  --details data/eval/v1/jina-code-embeddings-0.5b_details.jsonl \
  --cache data/embeddings/v1/jina-code-embeddings-0.5b \
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
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/qwen3-embedding-4b_summary.json \
  --details data/eval/v1/qwen3-embedding-4b_details.jsonl \
  --cache data/embeddings/v1/qwen3-embedding-4b \
  --candidate-filter all_files \
  --batch-size 8 \
  --device cuda \
  --trust-remote-code \
  --no-keep-list
```

Generate a leaderboard from all `*_summary.json` files:

```bash
arb report-models \
  --eval-dir data/eval/v1 \
  --out data/reports/v1/model_leaderboard.md \
  --json-out data/reports/v1/model_leaderboard.json
```

The default candidate set is `all_files`. The primary metric is overall `MRR`; report `Recall@5`, `Recall@10`, `Recall@20`, and `gold_coverage@8k` alongside it.

## Current Leaderboard

The uploaded V1 release includes a lexical baseline for all 225 samples with `skipped={}`. Embedding baselines can be generated with the commands above.

| Task | Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | lexical | 225 | 0.1970 | 0.3267 | 0.4874 | 0.1450 | 0.0785 |
| code2test | lexical | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 |
| comment2context | lexical | 51 | 0.1928 | 0.3595 | 0.5752 | 0.1739 | 0.0784 |
| trace2code | lexical | 68 | 0.4007 | 0.6201 | 0.7966 | 0.2458 | 0.1581 |

## Benchmark Semantics

- Every sample is evaluated against `repo_at_base_commit`; fixed code is not indexed.
- `code2test` queries describe implementation changes or PR intent; gold files are related tests.
- `comment2context` queries describe review comments; the commented file is treated as given context, and scoring uses only additional required context files.
- `trace2code` queries contain failure excerpts from local test reproduction; gold files are audited root-cause source files, while related tests are not scored as primary gold.
- `gold_coverage@8k` measures whether gold files appear within an 8k-character retrieval budget.

Legacy V0.2 remains available under `data/benchmark/v0_2/`, and the earlier V1 Code Review track remains available under `data/benchmark/v1_code_review/`. New model comparisons should report full V1 as the primary benchmark.
