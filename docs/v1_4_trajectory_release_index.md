# V1.4 Trajectory Release Index

This document tracks the private trajectory releases built on top of the V1.3 benchmark. These artifacts are intentionally separate from the public benchmark/corpus bundles: they contain model-generated trajectory logs, answers, traces, eval outputs, and reports, but not benchmark or corpus data.

## Canonical Release

Use `v1_3_openai_gpt54mini_v2_strict_context` as the canonical GPT-5.4-mini trajectory release.

- HF repo: `eyuansu71/agent_retrieval_bench-private`
- HF path: `releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/`
- HF commit: `3cb68c03528b8a28895a72547b260a85e0562a90`
- Archive: `agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst`
- SHA256: `612eb3595aae7c0a411a1c620346344c4998c9d685f136efce9043add752c134`
- Local report: `data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context.md`
- Quality audit: `data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.md`
- Release smoke: `data/reports/v1_4/release_bundle_smoke_v1_3_openai_gpt54mini_v2_strict_context.md`
- Results doc: `docs/v1_4_trajectory_results.md`
- Error analysis: `docs/v1_4_trajectory_error_analysis.md`
- Reproduction checklist: `docs/v1_4_reproduction_checklist.md`
- Paper-facing claims: `docs/v1_4_paper_claims.md`

## Release Status

| Release | Status | Use | Notes |
| --- | --- | --- | --- |
| `v1_3_openai_gpt54mini_v1` | Exploratory | Debug/reference only | 287 samples completed, but 146/287 answers had `context_files` that included files not actually read. Trajectory logs are useful, answer-level final context is not strict. |
| `v1_3_openai_gpt54mini_v2_strict_context` | Canonical | Recommended | 287/287 completed, strict final context invariant holds, private HF smoke passed. |

## Canonical Metrics

Run: `openai_gpt54mini_v2_strict_context`

- Model: `gpt-5.4-mini`
- Samples: `287`
- Logs / answers / traces: `287 / 287 / 287`
- Read steps: `918`
- Final files: `907`
- Read-step distribution: `{3: 232, 4: 53, 5: 2}`
- Final-file distribution: `{3: 241, 4: 46}`
- `retrieved_file_f1`: `0.3101708976`
- `final_file_f1`: `0.3113323378`
- `line_f1@trajectory`: `0.0143062552`
- `trajectory_redundancy`: `0.0`

Quality gate:

- `below_min_reads=0`
- `duplicate_read_samples=0`
- `missing_read_path_samples=0`
- `missing_final_path_samples=0`
- `empty_final_samples=0`
- `final_unread_samples=0`
- `strict_openai_key_hits=0`
- `raw_unread_samples=5`

`raw_unread_samples` means the model mentioned unread files in its raw final payload. The strict runner moved those paths to `suggested_unread_files`; they do not appear in persisted `final_files`.

## Reproduction Commands

Set `OPENAI_API_KEY` in the shell before running the model. Do not commit the key or print it in logs.

Run the strict-context agent:

```bash
PYTHONPATH=src arb run-openai-context-agent \
  --base data/trajectory_runs/v1_4/v1_3_all_gpt54mini \
  --samples data/benchmark/v1_3/samples.jsonl \
  --corpus data/corpus/v1_2 \
  --model gpt-5.4-mini \
  --run-name openai_gpt54mini_v2_strict_context \
  --all-samples \
  --force \
  --max-actions 9 \
  --min-reads 3
```

Evaluate trajectories:

```bash
mapfile -t logs < <(find data/trajectory_runs/v1_4/v1_3_all_gpt54mini/logs_openai_gpt54mini_v2_strict_context -type f -name '*.jsonl' | sort)

PYTHONPATH=src arb eval-trajectories "${logs[@]}" \
  --derived data/benchmark/v1_3 \
  --out data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json \
  --details data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl \
  --model-label openai_gpt54mini_v2_strict_context
```

Audit the strict-context release:

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

Package the trajectory release:

```bash
PYTHONPATH=src arb package-trajectory-release \
  --data-root data \
  --base data/trajectory_runs/v1_4/v1_3_all_gpt54mini \
  --run-name openai_gpt54mini_v2_strict_context \
  --release-dir data/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context \
  --archive-name agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst \
  --checksum-path-in-release releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context
```

Upload to the private dataset repo:

```bash
python - <<'PY'
from huggingface_hub import HfApi

HfApi().upload_folder(
    repo_id="eyuansu71/agent_retrieval_bench-private",
    repo_type="dataset",
    folder_path="data/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context",
    path_in_repo="releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context",
    commit_message="Add v1.3 gpt-5.4-mini v2 strict-context trajectories",
)
PY
```

## Smoke Verification

Local:

```bash
cd data
sha256sum -c releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst.sha256
zstd -t releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst
```

HF smoke:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="eyuansu71/agent_retrieval_bench-private",
    repo_type="dataset",
    allow_patterns="releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/*",
    local_dir="/tmp/arb_v1_3_gpt54mini_v2_smoke",
)
PY

cd /tmp/arb_v1_3_gpt54mini_v2_smoke
sha256sum -c releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst.sha256
zstd -t releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/agent_retrieval_bench_v1_3_openai_gpt54mini_v2_strict_context_trajectories.tar.zst
```

Expected extracted bundle contents:

- `trajectory_runs/v1_4/v1_3_all_gpt54mini/logs_openai_gpt54mini_v2_strict_context/`: 287 JSONL logs
- `trajectory_runs/v1_4/v1_3_all_gpt54mini/answers_openai_gpt54mini_v2_strict_context/`: 287 answers
- `trajectory_runs/v1_4/v1_3_all_gpt54mini/traces_openai_gpt54mini_v2_strict_context/`: 287 traces
- `eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json`
- `eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl`
- `reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context*`

The bundle should not contain `benchmark/`, `corpus/`, smoke artifacts, `__pycache__`, or `.pyc` files.

## Notes

The V1.4 trajectory track is a process-analysis track, not the official single-shot retrieval leaderboard. Use the V1.3 benchmark/corpus release for benchmark inputs and this trajectory release for real model read-step behavior.
