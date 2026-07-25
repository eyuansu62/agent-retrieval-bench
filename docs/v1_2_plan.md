# Agent Retrieval Bench V1.2 Plan

V1.2 keeps V1.1 as the stable single-shot file-retrieval baseline and adds the infrastructure needed to evaluate more agent-like repository search.

## Direction

- Keep frozen V1 and V1.1 artifacts unchanged.
- Build V1.2 under `data/benchmark/v1_2`, `data/corpus/v1_2`, `data/eval/v1_2`, and `data/reports/v1_2`.
- Reuse V1.1's 287 samples as the starting point; raw scale is not the release blocker.
- Add optional span-level labels and hard negatives for a curated subset before treating V1.2 as a data release.

## New Sample Fields

- `query_provenance`: source of the retrieval query, such as `pr_summary`, `review_comment`, `failure_trace`, or `issue_report`.
- `gold_spans`: optional line ranges inside gold files, with `path`, `start_line`, `end_line`, and `reason`.
- `hard_negative_files`: optional files that look relevant but should not count as gold.

`arb validate-v1-2` validates these fields and can use a corpus manifest to check span bounds and hard-negative corpus coverage.

## New Metrics

Existing file-level metrics remain unchanged:

- `Recall@5/10/20`
- `MRR`
- `gold_coverage@8k`

V1.2 details and summaries can additionally include:

- `Precision@5/10/20`
- `F0.5@5/10/20`
- `irrelevant_files@5/10/20`
- `hard_negative_hits@5/10/20`
- `context_pollution_tokens@8k`
- `gold_token_ratio@8k`

Span-annotated samples additionally produce `span_metrics` with span recall, precision, F0.5, and line-overlap counts.

## New Commands

```bash
arb validate-v1-2 \
  --derived data/benchmark/v1_2 \
  --corpus-manifest data/corpus/v1_2/corpus_manifest.jsonl \
  --out data/reports/v1_2/validation.json \
  --markdown-out data/reports/v1_2/validation.md
```

```bash
arb eval-agentic \
  --derived data/benchmark/v1_2 \
  --corpus data/corpus/v1_2 \
  --out data/eval/v1_2/scripted-search-read_summary.json \
  --details data/eval/v1_2/scripted-search-read_details.jsonl
```

```bash
arb report-context-pollution --eval-dir data/eval/v1_2
arb report-span-subset --eval-dir data/eval/v1_2
arb report-runtime-cache --eval-dir data/eval/v1_2
```

## Release Blockers

- V1 and V1.1 fingerprints remain unchanged.
- V1.2 validation has zero invalid rows.
- At least 50 manually audited samples have `gold_spans`.
- At least 50 samples have explicit `hard_negative_files`.
- Official single-shot baselines rerun on V1.2 with no skipped rows.
- Context-pollution, span-subset, runtime/cache, rank-analysis, and model-leaderboard reports regenerate from artifacts.
- The experimental agentic-search leaderboard is clearly separated from the official single-shot leaderboard.
