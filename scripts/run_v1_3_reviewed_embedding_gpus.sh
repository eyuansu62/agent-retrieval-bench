#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run V1.3 reviewed embedding baselines on separate GPUs.

Defaults:
  models: jina,qwen4b,qwen8b
  gpus:   0,1,2
  data:   data/benchmark/v1_3_reviewed + data/corpus/v1_2

Usage:
  scripts/run_v1_3_reviewed_embedding_gpus.sh --gpus 0,1,2
  scripts/run_v1_3_reviewed_embedding_gpus.sh --smoke --gpus 0,1,2
  scripts/run_v1_3_reviewed_embedding_gpus.sh --resume --models jina,qwen4b,qwen8b,nomic --gpus 0,1,2,3

Options:
  --models LIST          Comma list: jina,qwen4b,qwen8b,nomic,pplx
  --gpus LIST            Comma list of CUDA device ids, one per model by default
  --allow-gpu-reuse      Allow cycling GPUs when there are fewer GPUs than models
  --smoke                Run a small smoke test; writes *_smoke outputs
  --limit-samples N      Limit evaluated samples. With --smoke, default is 5
  --resume               Add --resume-details to each eval-embedding job
  --dry-run              Print commands without running them
  --help                 Show this help

Per-model batch size overrides:
  JINA_BATCH_SIZE=8 QWEN4B_BATCH_SIZE=8 QWEN8B_BATCH_SIZE=4 NOMIC_BATCH_SIZE=8 PPLX_BATCH_SIZE=8
EOF
}

MODELS="jina,qwen4b,qwen8b"
GPUS="0,1,2"
BENCHMARK_DIR="data/benchmark/v1_3_reviewed"
CORPUS_DIR="data/corpus/v1_2"
EVAL_DIR="data/eval/v1_3_reviewed"
EMBED_DIR="data/embeddings/v1_3_reviewed"
LOG_DIR="data/reports/v1_3_reviewed/embedding_gpu_jobs"
SMOKE=0
LIMIT_SAMPLES=""
RESUME=0
DRY_RUN=0
ALLOW_GPU_REUSE=0
CANDIDATE_FILTER="all_files"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --allow-gpu-reuse)
      ALLOW_GPU_REUSE=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --limit-samples)
      LIMIT_SAMPLES="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$SMOKE" == "1" && -z "$LIMIT_SAMPLES" ]]; then
  LIMIT_SAMPLES="5"
fi

if [[ ! -f "$BENCHMARK_DIR/samples.jsonl" ]]; then
  echo "Missing $BENCHMARK_DIR/samples.jsonl" >&2
  echo "Download first: PYTHONPATH=src arb download-benchmark --version v1_3_reviewed --local-dir data --repo-id eyuansu71/agent_retrieval_bench-private --force" >&2
  exit 1
fi

if [[ ! -f "$CORPUS_DIR/corpus_manifest.jsonl" ]]; then
  echo "Missing $CORPUS_DIR/corpus_manifest.jsonl" >&2
  echo "Download first: PYTHONPATH=src arb download-benchmark --version v1_3_reviewed --local-dir data --repo-id eyuansu71/agent_retrieval_bench-private --force" >&2
  exit 1
fi

IFS=',' read -r -a MODEL_KEYS <<< "$MODELS"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"

if [[ "${#GPU_IDS[@]}" -eq 0 || -z "${GPU_IDS[0]}" ]]; then
  echo "At least one GPU id is required." >&2
  exit 1
fi

if [[ "$ALLOW_GPU_REUSE" != "1" && "${#MODEL_KEYS[@]}" -gt "${#GPU_IDS[@]}" ]]; then
  echo "Need at least one GPU per model, or pass --allow-gpu-reuse." >&2
  echo "models=${#MODEL_KEYS[@]} gpus=${#GPU_IDS[@]}" >&2
  exit 1
fi

resolve_model() {
  local key="${1,,}"
  case "$key" in
    jina|jina-code|jina-code-embeddings-0.5b)
      MODEL_NAME="jinaai/jina-code-embeddings-0.5b"
      MODEL_LABEL="jina-code-embeddings-0.5b"
      BATCH_SIZE="${JINA_BATCH_SIZE:-8}"
      TRUST_REMOTE_CODE=1
      ;;
    qwen4b|qwen3-4b|qwen3-embedding-4b|qwen4)
      MODEL_NAME="Qwen/Qwen3-Embedding-4B"
      MODEL_LABEL="Qwen3-Embedding-4B"
      BATCH_SIZE="${QWEN4B_BATCH_SIZE:-8}"
      TRUST_REMOTE_CODE=1
      ;;
    qwen8b|qwen3-8b|qwen3-embedding-8b|qwen8)
      MODEL_NAME="Qwen/Qwen3-Embedding-8B"
      MODEL_LABEL="Qwen3-Embedding-8B"
      BATCH_SIZE="${QWEN8B_BATCH_SIZE:-4}"
      TRUST_REMOTE_CODE=1
      ;;
    nomic|nomic-embed-code)
      MODEL_NAME="nomic-ai/nomic-embed-code"
      MODEL_LABEL="nomic-embed-code"
      BATCH_SIZE="${NOMIC_BATCH_SIZE:-8}"
      TRUST_REMOTE_CODE=1
      ;;
    pplx|perplexity|pplx-embed-v1-4b)
      MODEL_NAME="perplexity-ai/pplx-embed-v1-4b"
      MODEL_LABEL="pplx-embed-v1-4b"
      BATCH_SIZE="${PPLX_BATCH_SIZE:-8}"
      TRUST_REMOTE_CODE=1
      ;;
    *)
      echo "Unknown model alias: $1" >&2
      echo "Supported: jina,qwen4b,qwen8b,nomic,pplx" >&2
      exit 2
      ;;
  esac
}

mkdir -p "$EVAL_DIR" "$EMBED_DIR" "$LOG_DIR"

PYTHONPATH_VALUE="${PYTHONPATH:-src}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUFFIX=""
if [[ "$SMOKE" == "1" ]]; then
  SUFFIX="_smoke"
fi

declare -a PIDS=()
declare -a JOBS=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for index in "${!MODEL_KEYS[@]}"; do
  key="${MODEL_KEYS[$index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  resolve_model "$key"

  summary="$EVAL_DIR/${MODEL_LABEL}${SUFFIX}_summary.json"
  details="$EVAL_DIR/${MODEL_LABEL}${SUFFIX}_details.jsonl"
  cache="$EMBED_DIR/${MODEL_LABEL}${SUFFIX}"
  shared_text_cache="$EMBED_DIR/${MODEL_LABEL}${SUFFIX}_texts.sqlite"
  log="$LOG_DIR/${MODEL_LABEL}${SUFFIX}.gpu${gpu}.${TIMESTAMP}.log"

  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    "PYTHONPATH=$PYTHONPATH_VALUE"
    arb eval-embedding
    --model "$MODEL_NAME"
    --model-label "$MODEL_LABEL"
    --derived "$BENCHMARK_DIR"
    --corpus "$CORPUS_DIR"
    --out "$summary"
    --details "$details"
    --cache "$cache"
    --shared-text-cache "$shared_text_cache"
    --candidate-filter "$CANDIDATE_FILTER"
    --batch-size "$BATCH_SIZE"
    --device cuda
    --no-keep-list
  )

  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ -n "$LIMIT_SAMPLES" ]]; then
    cmd+=(--limit-samples "$LIMIT_SAMPLES")
  fi
  if [[ "$RESUME" == "1" ]]; then
    cmd+=(--resume-details)
  fi

  echo "[$MODEL_LABEL] gpu=$gpu batch=$BATCH_SIZE summary=$summary log=$log"
  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  (
    "${cmd[@]}"
  ) >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  JOBS+=("$pid:$MODEL_LABEL:$gpu:$log")
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
for job in "${JOBS[@]}"; do
  IFS=':' read -r pid label gpu log <<< "$job"
  if wait "$pid"; then
    echo "[$label] completed on gpu=$gpu log=$log"
  else
    echo "[$label] failed on gpu=$gpu log=$log" >&2
    status=1
  fi
done

exit "$status"
