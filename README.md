# Agent Retrieval Bench

Agent Retrieval Bench is an action-oriented code retrieval benchmark for coding agents. It evaluates whether a retriever can find the repository files an agent would need for real coding workflows.

The current public release is **Agent Retrieval Bench V1**: 225 manually curated samples across `code2test`, `comment2context`, and `trace2code`.

Project page: https://agent-retrieval-bench.github.io/

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
  repomap_summary.json
  repomap_details.jsonl
  jina-code-embeddings-0.5b_summary.json
  qwen3-embedding-4b_summary.json
data/reports/v1/
  model_leaderboard.md
  model_leaderboard.json
  analysis.md
  analysis.json
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

Use the CLI downloader as the primary path:

```bash
arb download-benchmark --version v1 --local-dir data --force
```

This creates `data/benchmark/v1`, `data/corpus/v1`, `data/eval/v1`, and `data/reports/v1`. The command wraps the Hugging Face download, checksum verification, and extraction steps. No manual `zstd` or `tar` commands are needed. If the dataset is private or gated, authenticate first with `hf auth login` or set `HF_TOKEN`.

Manual troubleshooting path:

```bash
hf download eyuansu71/agent_retrieval_bench \
  --repo-type dataset \
  --local-dir data \
  --include "releases/v1/*"

cd data
shasum -a 256 -c releases/v1/agent_retrieval_bench_v1.tar.zst.sha256
rm -rf benchmark/v1 corpus/v1 eval/v1 reports/v1
zstd -dc releases/v1/agent_retrieval_bench_v1.tar.zst | tar -xf - -C .
cd ..
```

On Linux, `sha256sum -c releases/v1/agent_retrieval_bench_v1.tar.zst.sha256` is equivalent. If your `tar` prints `LIBARCHIVE.xattr.com.apple.provenance` warnings during manual extraction, they are macOS extended-attribute warnings and do not affect the extracted files.

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

Run the deterministic Aider-style RepoMap baseline:

```bash
arb eval-repomap \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/repomap_summary.json \
  --details data/eval/v1/repomap_details.jsonl \
  --candidate-filter all_files \
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

The V1 release bundle includes lexical, RepoMap, Jina, and Qwen summaries for all 225 samples with `skipped={}`. RepoMap is a deterministic vectorless baseline inspired by Aider-style repo maps.

| Task | Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | Qwen3-Embedding-4B | 225 | 0.2883 | 0.4033 | 0.5828 | 0.2455 | 0.2542 |
| overall | aider-style-repomap | 225 | 0.3089 | 0.4705 | 0.6299 | 0.2227 | 0.0704 |
| overall | jina-code-embeddings-0.5b | 225 | 0.2230 | 0.3133 | 0.4492 | 0.1883 | 0.1556 |
| overall | lexical | 225 | 0.1970 | 0.3267 | 0.4874 | 0.1450 | 0.0785 |
| code2test | Qwen3-Embedding-4B | 106 | 0.4610 | 0.5777 | 0.7230 | 0.3225 | 0.3887 |
| code2test | jina-code-embeddings-0.5b | 106 | 0.2610 | 0.3868 | 0.5305 | 0.2033 | 0.2060 |
| code2test | aider-style-repomap | 106 | 0.2752 | 0.4107 | 0.5808 | 0.1975 | 0.0739 |
| code2test | lexical | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 |
| comment2context | jina-code-embeddings-0.5b | 51 | 0.3431 | 0.4020 | 0.5261 | 0.3282 | 0.2386 |
| comment2context | Qwen3-Embedding-4B | 51 | 0.2451 | 0.3856 | 0.5621 | 0.3113 | 0.2451 |
| comment2context | aider-style-repomap | 51 | 0.2353 | 0.3137 | 0.4967 | 0.2053 | 0.0588 |
| comment2context | lexical | 51 | 0.1928 | 0.3595 | 0.5752 | 0.1739 | 0.0784 |
| trace2code | aider-style-repomap | 68 | 0.4167 | 0.6814 | 0.8064 | 0.2750 | 0.0735 |
| trace2code | lexical | 68 | 0.4020 | 0.5931 | 0.7966 | 0.2458 | 0.1544 |
| trace2code | Qwen3-Embedding-4B | 68 | 0.0515 | 0.1446 | 0.3799 | 0.0760 | 0.0515 |
| trace2code | jina-code-embeddings-0.5b | 68 | 0.0735 | 0.1324 | 0.2647 | 0.0598 | 0.0147 |

Current results show that Qwen is strongest overall and on `code2test`, Jina is strongest on `comment2context`, and RepoMap is strongest on `trace2code`. The weak embedding scores on `trace2code` are an intended benchmark signal: that track requires structured failure-log and repo-graph retrieval, not only semantic code similarity.

## Benchmark Semantics

- Every sample is evaluated against `repo_at_base_commit`; fixed code is not indexed.
- `code2test` queries describe implementation changes or PR intent; gold files are related tests.
- `comment2context` queries describe review comments; the commented file is treated as given context, and scoring uses only additional required context files.
- `trace2code` queries contain failure excerpts from local test reproduction; gold files are audited root-cause source files, while related tests are not scored as primary gold.
- `gold_coverage@8k` measures whether gold files appear within an 8k-character retrieval budget.

Legacy V0.2 remains available under `data/benchmark/v0_2/`, and the earlier V1 Code Review track remains available under `data/benchmark/v1_code_review/`. New model comparisons should report full V1 as the primary benchmark.

Future data collection should target V1.1: expand `comment2context` to 80-100 samples and `trace2code` to 100+ samples, while leaving `benchmark/v1` frozen.
