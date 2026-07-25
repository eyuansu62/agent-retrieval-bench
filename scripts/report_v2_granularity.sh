#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHONPATH="${PYTHONPATH:-src}" python -m agent_retrieval_bench.granularity \
  --samples data/benchmark/v1_3_reviewed/samples.jsonl \
  --samples data/benchmark/v2_edit2ripple/samples.jsonl \
  --core-samples data/benchmark/v1_3_reviewed/samples.jsonl \
  --corpus-manifest data/corpus/v2_selective_mixed/corpus_manifest.jsonl \
  --corpus-manifest data/corpus/v2_comment2context/corpus_manifest.jsonl \
  --details Qwen3-Embedding-4B=data/eval/v1_3_reviewed/Qwen3-Embedding-4B_details.jsonl \
  --details Qwen3-Embedding-8B=data/eval/v1_3_reviewed/Qwen3-Embedding-8B_details.jsonl \
  --details pplx-embed-v1-4b=data/eval/v1_3_reviewed/pplx-embed-v1-4b_details.jsonl \
  --details jina-code-embeddings-0.5b=data/eval/v1_3_reviewed/jina-code-embeddings-0.5b_details.jsonl \
  --details nomic-embed-code=data/eval/v1_3_reviewed/nomic-embed-code_details.jsonl \
  --details BM25=data/eval/v1_3_reviewed/bm25_all_files_details.jsonl \
  --details lexical=data/eval/v1_3_reviewed/lexical_all_files_details.jsonl \
  --details RepoMap=data/eval/v1_3_reviewed/repomap_all_files_details.jsonl \
  --repo-root . \
  --out-json data/reports/v2_granularity/granularity_report.json \
  --out-md data/reports/v2_granularity/granularity_report.md
