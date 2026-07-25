# V1.3/V1.4 ContextBench Alignment

ARB remains a controlled retrieval/ranking benchmark. V1.3 adds the data and metric hooks needed for ContextBench-style granularity, while V1.4 introduces a separate trajectory track for full coding-agent process analysis.

## V1.3 Gold Granularity

V1.3 samples may add `gold_blocks`, alongside existing `gold_spans` and `hard_negative_files`.

Each `gold_blocks` row uses:

- `path`: repository-relative file path.
- `start_line` / `end_line`: inclusive block bounds.
- `kind`: corpus block type, normally `symbol`; `span_fallback` means no corpus symbol chunk overlapped the span.
- `symbol`: optional function/class/method or nearest symbol name.
- `chunk_id`: optional corpus chunk id for exact corpus-backed validation.
- `source`: derivation source such as `corpus_symbol_overlap` or `gold_span_fallback`.
- `reason`: annotation or derivation note.

Use:

```bash
arb derive-v1-3-blocks \
  --base-derived data/benchmark/v1_2 \
  --corpus-manifest data/corpus/v1_2/corpus_manifest.jsonl \
  --out data/benchmark/v1_3
```

Then validate:

```bash
arb validate-v1-3 \
  --derived data/benchmark/v1_3 \
  --corpus-manifest data/corpus/v1_2/corpus_manifest.jsonl
```

By default, `validate-v1-3` requires every sample to have spans and blocks. Use `--allow-partial-spans` or `--allow-partial-blocks` only for interim annotation work.

## Reviewed Candidate

The current reviewed V1.3 candidate is:

- Derived data: `data/benchmark/v1_3_reviewed`
- Full validation report: `data/reports/v1_3/reviewed_validation.json`
- Human-review substitute annotations:
  - `data/reports/v1_3/manual_annotations_comment2context_codex_reviewed.jsonl`
  - `data/reports/v1_3/manual_annotations_trace2code_codex_reviewed.jsonl`
  - `data/reports/v1_3/manual_annotations_code2test_codex_reviewed.jsonl`

Validation status on 2026-05-30:

- `ready=true`
- `invalid=0`
- `samples=287`
- `span_samples=287`
- `block_samples=287`
- `hard_negative_samples=150`

The substitute review used fix-commit patch hunks against each sample's target gold files, checked against the corpus-derived span worklists. One `code2test` sample used a candidate chunk fallback because the newly added test landed beyond the corpus-valid line range for that file.

Release bundle prepared on 2026-05-30:

- Archive: `data/releases/v1_3/agent_retrieval_bench_v1_3.tar.zst`
- Checksum: `data/releases/v1_3/agent_retrieval_bench_v1_3.tar.zst.sha256`
- Bundle report: `data/reports/v1_3/release_bundle.json`
- Internal paths: `benchmark/v1_3`, `corpus/v1_2`, `eval/v1_3`, `reports/v1_3`
- Smoke tests: `sha256sum -c`, `zstd -t`, temp extraction, and extracted `validate-v1-3` all passed.
- Hugging Face: `https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench-private/tree/main/releases/v1_3`
- HF smoke test: `download-benchmark --version v1_3` plus extracted `validate-v1-3` passed.

The release eval directory includes all-files lexical and RepoMap baselines, each with 287 evaluated samples and no skipped samples.

## Metrics

Existing detail rows now preserve backwards-compatible `span_metrics` and also expose:

- `line_metrics`: `line_recall@8k`, `line_precision@8k`, `line_f1@8k`, `line_f0.5@8k`, line counts.
- `block_metrics`: `block_recall@8k`, `block_precision@8k`, `block_f1@8k`, `block_f0.5@8k`, block counts.
- file-level additions: `coverage_auc@20`, `redundancy@8k`, and `context_efficiency@8k`.

Summary JSON merges available line/block metrics into each task row so V1.3 leaderboards can consume them without changing the existing summary shape.

## V1.4 Trajectory Track

The trajectory evaluator accepts JSONL rows in either per-sample or per-step form. Per-sample rows contain `sample_id` and a `trajectory` list; per-step rows contain `sample_id` directly.

Each step may contain:

- `step`
- `tool`
- `path`
- `start_line`
- `end_line`
- `kind`
- `symbol`
- `content_hash`
- `is_final_context`
- `is_utilized_context`

To collect real runs, first prepare gold-free prompt packets and per-sample logs:

```bash
arb prepare-trajectory-runs \
  --derived data/benchmark/v1_3 \
  --out-dir data/trajectory_runs/v1_4/codex_pilot \
  --limit-samples 10 \
  --model-label codex
```

Each generated prompt includes the query, repo, base commit, and trajectory log path, but excludes gold labels. During the real agent run, record every file or line range that enters context:

```bash
arb record-trajectory-step \
  --log data/trajectory_runs/v1_4/codex_pilot/logs/<sample_id>.jsonl \
  --sample-id <sample_id> \
  --tool read \
  --path <repo-relative-path> \
  --start-line <n> \
  --end-line <m>
```

Use `--final` for context retained in the final answer and `--used` for context directly used in the final answer or patch. `--repo-root <checkout>` computes a stable content hash for the recorded file slice.

Run:

```bash
arb eval-trajectories data/trajectory_runs/v1_4/codex_pilot/logs/*.jsonl \
  --derived data/benchmark/v1_3 \
  --out data/eval/v1_4/trajectory_summary.json \
  --details data/eval/v1_4/trajectory_details.jsonl
```

Trajectory summaries report retrieved/final/utilized file precision/recall/F1, line/block coverage when labels exist, trajectory redundancy, and usage-drop metrics. This track is intentionally separate from the official single-shot retrieval leaderboard.
