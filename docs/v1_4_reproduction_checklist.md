# V1.4 Trajectory Reproduction Checklist

This checklist reproduces the canonical private V1.4 trajectory analysis from the private HF artifact and local benchmark inputs.

## Preconditions

- Repo: `agent-retrieval-bench-private`
- Dataset repo: `eyuansu71/agent_retrieval_bench-private`
- HF auth: `hf auth login` or `HF_TOKEN`
- Local benchmark/corpus inputs:
  - `data/benchmark/v1_3/samples.jsonl`
  - `data/corpus/v1_2/corpus_manifest.jsonl`
- Tools: `python`, `zstd`, `tar`, `sha256sum`

Do not print or commit API keys. Model reruns require `OPENAI_API_KEY`; artifact verification and evaluation do not.

## 1. Download And Verify HF Artifact

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="eyuansu71/agent_retrieval_bench-private",
    repo_type="dataset",
    allow_patterns="releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/*",
    local_dir="/tmp/arb_v1_4_repro",
)
PY

cd /tmp/arb_v1_4_repro
sha256sum -c releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst.sha256
zstd -t releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst
```

Expected checksum:

```text
612eb3595aae7c0a411a1c620346344c4998c9d685f136efce9043add752c134
```

## 2. Extract Artifact Into Local Data Tree

```bash
cd /home/ubuntu/agent-retrieval-bench-private
mkdir -p /tmp/arb_v1_4_extract
tar --zstd -xf /tmp/arb_v1_4_repro/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst -C /tmp/arb_v1_4_extract
```

The extracted bundle should contain:

- `trajectory_runs/v1_4/v1_3_all_gpt54mini/logs_openai_gpt54mini_v2_strict_context/`
- `trajectory_runs/v1_4/v1_3_all_gpt54mini/answers_openai_gpt54mini_v2_strict_context/`
- `trajectory_runs/v1_4/v1_3_all_gpt54mini/traces_openai_gpt54mini_v2_strict_context/`
- `eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json`
- `eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl`
- `reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context*`

The bundle must not contain `benchmark/`, `corpus/`, smoke artifacts, `__pycache__`, or `.pyc` files.

## 3. Re-Evaluate Trajectory Logs

```bash
cd /home/ubuntu/agent-retrieval-bench-private
mapfile -t logs < <(find data/trajectory_runs/v1_4/v1_3_all_gpt54mini/logs_openai_gpt54mini_v2_strict_context -type f -name '*.jsonl' | sort)

PYTHONPATH=src arb eval-trajectories "${logs[@]}" \
  --derived data/benchmark/v1_3 \
  --out data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json \
  --details data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl \
  --model-label openai_gpt54mini_v2_strict_context
```

Expected headline metrics:

- `evaluated=287`
- `overall.final_file_f1=0.3113323378`
- `overall.line_f1@trajectory=0.0143062552`
- `trajectory_redundancy=0.0`

## 4. Audit Strict-Context Release

```bash
PYTHONPATH=src arb audit-trajectory-release \
  --base data/trajectory_runs/v1_4/v1_3_all_gpt54mini \
  --run-name openai_gpt54mini_v2_strict_context \
  --derived data/benchmark/v1_3 \
  --corpus-manifest data/corpus/v1_2/corpus_manifest.jsonl \
  --extra-scan-path data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json \
  --extra-scan-path data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl \
  --out data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.json \
  --markdown-out data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.md
```

Expected gates:

- `verdict=pass`
- `logs=287`, `answers=287`, `traces=287`
- `read_steps_total=918`
- `final_files_total=907`
- `below_min_reads=0`
- `duplicate_read_samples=0`
- `missing_read_path_samples=0`
- `missing_final_path_samples=0`
- `empty_final_samples=0`
- `final_unread_samples=0`
- `strict_openai_key_hits=0`

## 5. Reproduce Same-Budget Strong Contrasts

```bash
PYTHONPATH=src arb eval-ranked-context \
  --baseline-details data/eval/v1_3_reviewed/lexical_all_files_details.jsonl \
  --top-k 3 \
  --out data/eval/v1_4/v1_3_reviewed_lexical_top3_context_summary.json \
  --details data/eval/v1_4/v1_3_reviewed_lexical_top3_context_details.jsonl \
  --model-label lexical@3-final-context

PYTHONPATH=src arb eval-ranked-context \
  --baseline-details data/eval/v1_3_reviewed/repomap_all_files_details.jsonl \
  --top-k 3 \
  --out data/eval/v1_4/v1_3_reviewed_repomap_top3_context_summary.json \
  --details data/eval/v1_4/v1_3_reviewed_repomap_top3_context_details.jsonl \
  --model-label repomap@3-final-context

PYTHONPATH=src arb eval-ranked-context \
  --baseline-details data/eval/v1_3_reviewed/repomap_all_files_details.jsonl \
  --top-k 4 \
  --out data/eval/v1_4/v1_3_reviewed_repomap_top4_context_summary.json \
  --details data/eval/v1_4/v1_3_reviewed_repomap_top4_context_details.jsonl \
  --model-label repomap@4-final-context
```

Expected overall final-file F1:

- `lexical@3-final-context`: `0.0721254355`
- `repomap@3-final-context`: `0.1102206736`
- `repomap@4-final-context`: `0.1198053205`
- `openai_gpt54mini_v2_strict_context`: `0.3113323378`

## 6. Regenerate Private Release Bundle

```bash
PYTHONPATH=src arb package-trajectory-release \
  --data-root data \
  --base data/trajectory_runs/v1_4/v1_3_all_gpt54mini \
  --run-name openai_gpt54mini_v2_strict_context \
  --release-dir data/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context \
  --archive-name agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst \
  --checksum-path-in-release releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context
```

## 7. Local Code Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

Release data under `data/` is ignored and should stay out of git. The tracked reproducibility surface is source code plus docs.
