# Agent Retrieval Bench V1.2 Freeze Handoff

Generated from local repo state on 2026-05-30 after the V1.2 round1-3 manual annotation freeze.

V1.2 is currently frozen as a 150-annotated-sample intermediate release candidate. The purpose of this freeze is to make span-level and context-pollution evaluation reproducible before expanding annotation coverage further.

## Freeze Snapshot

- Benchmark: `data/benchmark/v1_2`
- Corpus: `data/corpus/v1_2`
- Annotation source: `data/reports/v1_2/manual_annotations_round1_3.jsonl`
- Dataset size: `287` samples (`code2test=106`, `comment2context=80`, `trace2code=101`)
- Annotated subset: `150` samples (`code2test=30`, `comment2context=55`, `trace2code=65`)
- Span-labeled samples: `150`
- Hard-negative samples: `150`
- Validation: `ready=true`, `invalid=0`

The generated V1.2 benchmark, corpus, eval, and report artifacts live under ignored `data/` paths. This handoff document is tracked so reviewers can reproduce and verify those local artifacts without changing the ignore policy.

## Quality Audit

- Audit report: `data/reports/v1_2/quality_audit_round1_3.md`
- Audit selection: first round3 `8 trace2code`, `7 comment2context`, and `5 code2test` samples.
- Audited samples: `20`
- Corrections required: `0`

The audit checked that sampled span paths belong to the task's target gold files, hard negatives do not overlap target gold files, hard negatives exist in the corpus, and the selected span windows are not obviously over-wide or missing the core evidence.

## Evaluation Snapshot

| Model | Evaluated | Skipped | Recall@20 | Hard-negative hits@20 |
| --- | ---: | --- | ---: | ---: |
| lexical | 287 | `{}` | 0.475029 | 1.954704 |
| aider-style-repomap | 287 | `{}` | 0.640767 | 1.815331 |

Generated reports:

- `data/reports/v1_2/manual_annotation_merge.md`
- `data/reports/v1_2/validation.md`
- `data/reports/v1_2/context_pollution.md`
- `data/reports/v1_2/span_subset.md`
- `data/reports/v1_2/runtime_cache.md`
- `data/reports/v1_2/freeze_round1_3.md`

## Reproduction Commands

```bash
PYTHONPATH=src python -m agent_retrieval_bench.cli merge-v1-2-annotations \
  --base-derived data/benchmark/v1_1 \
  --annotations data/reports/v1_2/manual_annotations_round1_3.jsonl \
  --out data/benchmark/v1_2 \
  --report-out data/reports/v1_2/manual_annotation_merge.json \
  --markdown-out data/reports/v1_2/manual_annotation_merge.md
```

```bash
PYTHONPATH=src python -m agent_retrieval_bench.cli validate-v1-2 \
  --derived data/benchmark/v1_2 \
  --corpus-manifest data/corpus/v1_2/corpus_manifest.jsonl \
  --out data/reports/v1_2/validation.json \
  --markdown-out data/reports/v1_2/validation.md
```

```bash
PYTHONPATH=src python -m agent_retrieval_bench.cli eval-baseline \
  --derived data/benchmark/v1_2 \
  --corpus data/corpus/v1_2 \
  --no-keep-list --no-progress \
  --out data/eval/v1_2/lexical_summary.json \
  --details data/eval/v1_2/lexical_details.jsonl
```

```bash
PYTHONPATH=src python -m agent_retrieval_bench.cli eval-repomap \
  --derived data/benchmark/v1_2 \
  --corpus data/corpus/v1_2 \
  --no-keep-list --no-progress \
  --out data/eval/v1_2/repomap_summary.json \
  --details data/eval/v1_2/repomap_details.jsonl
```

```bash
PYTHONPATH=src python -m agent_retrieval_bench.cli report-context-pollution \
  --eval-dir data/eval/v1_2 \
  --out data/reports/v1_2/context_pollution.md

PYTHONPATH=src python -m agent_retrieval_bench.cli report-span-subset \
  --eval-dir data/eval/v1_2 \
  --out data/reports/v1_2/span_subset.md

PYTHONPATH=src python -m agent_retrieval_bench.cli report-runtime-cache \
  --eval-dir data/eval/v1_2 \
  --out data/reports/v1_2/runtime_cache.md
```

## Pre-Commit Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
git diff -- data/benchmark/v1 data/benchmark/v1_1 data/corpus/v1 data/corpus/v1_1 data/reports/v1_1
```

Expected results:

- Unit tests pass.
- `git diff --check` is clean.
- V1 and V1.1 tracked paths have no diff.
- `returns/` is unrelated untracked local state and should not be included in this freeze unless separately reviewed.

## Next Decision

Review this 150-sample freeze before expanding annotation coverage. If the freeze is accepted, the next annotation target should be a separate 200-sample round with a new audit report rather than modifying the round1-3 freeze in place.
