# V1.4 Trajectory Results and Leaderboard Draft

This document is paper/report-facing. It separates the V1.3 single-shot retrieval leaderboard from the V1.4 trajectory track because the metrics are related but not identical: single-shot methods rank candidate files/chunks, while trajectory runs measure which files actually enter an agent context.

For the full V1.3 reviewed single-shot leaderboard with embedding models, see `docs/v1_3_reviewed_model_leaderboard.md`. The compact table below keeps only lexical and RepoMap because they are the static baselines used in the same-budget trajectory contrast.

Source artifacts:

- `data/eval/v1_3_reviewed/lexical_all_files_summary.json`
- `data/eval/v1_3_reviewed/repomap_all_files_summary.json`
- `data/eval/v1_3_reviewed/jina-code-embeddings-0.5b_summary.json`
- `data/eval/v1_3_reviewed/Qwen3-Embedding-4B_summary.json`
- `data/eval/v1_3_reviewed/Qwen3-Embedding-8B_summary.json`
- `data/eval/v1_3_reviewed/nomic-embed-code_summary.json`
- `data/eval/v1_3_reviewed/pplx-embed-v1-4b_summary.json`
- `data/eval/v1_4/v1_3_reviewed_lexical_top3_context_summary.json`
- `data/eval/v1_4/v1_3_reviewed_repomap_top3_context_summary.json`
- `data/eval/v1_4/v1_3_reviewed_repomap_top4_context_summary.json`
- `data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_summary.json`
- `data/reports/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_quality_audit.json`

## Table 1: V1.3 Single-Shot Retrieval Baselines

| Task | Model | Family | Samples | R@20 | P@20 | F1@20 | MRR | Gold@8k | Redundancy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | lexical | single-shot lexical | 287 | 0.4750 | 0.0326 | 0.0610 | 0.1402 | 0.0738 | 0.8127 |
| overall | aider-style-repomap | single-shot structure | 287 | 0.6367 | 0.0422 | 0.0791 | 0.2120 | 0.0592 | 0.0000 |
| code2test | lexical | single-shot lexical | 106 | 0.2469 | 0.0179 | 0.0334 | 0.0663 | 0.0299 | 0.8040 |
| code2test | aider-style-repomap | single-shot structure | 106 | 0.5808 | 0.0392 | 0.0734 | 0.1942 | 0.0645 | 0.0000 |
| comment2context | lexical | single-shot lexical | 80 | 0.4979 | 0.0413 | 0.0762 | 0.1530 | 0.0625 | 0.8601 |
| comment2context | aider-style-repomap | single-shot structure | 80 | 0.4583 | 0.0375 | 0.0693 | 0.1571 | 0.0396 | 0.0000 |
| trace2code | lexical | single-shot lexical | 101 | 0.6964 | 0.0411 | 0.0776 | 0.2075 | 0.1287 | 0.7843 |
| trace2code | aider-style-repomap | single-shot structure | 101 | 0.8366 | 0.0490 | 0.0926 | 0.2742 | 0.0693 | 0.0000 |

Notes:

- These rows use `candidate_filter=all_files` on the reviewed V1.3 benchmark.
- `F1@20` is computed from the reported `Precision@20` and `Recall@20` for table readability.
- RepoMap has strong `trace2code` file retrieval, while lexical remains competitive on trace-like signals.

## Table 2: V1.4 Canonical Trajectory Track

| Task | Run | Samples | File R | File P | File F1 | Line F1 | Redundancy | Avg reads | Avg final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | GPT-5.4-mini strict-context | 287 | 0.5645 | 0.2236 | 0.3113 | 0.0143 | 0.0000 | 3.199 | 3.160 |
| code2test | GPT-5.4-mini strict-context | 106 | 0.4575 | 0.1792 | 0.2516 | 0.0042 | 0.0000 | 3.170 | 3.142 |
| comment2context | GPT-5.4-mini strict-context | 80 | 0.2875 | 0.1333 | 0.1732 | 0.0064 | 0.0000 | 3.325 | 3.263 |
| trace2code | GPT-5.4-mini strict-context | 101 | 0.8960 | 0.3416 | 0.4835 | 0.0312 | 0.0000 | 3.129 | 3.099 |

Notes:

- This table reports `final_file_*`, not top-k ranking metrics.
- `trace2code` is the strongest task for the GPT-5.4-mini trajectory run at file level.
- `comment2context` remains the hardest track because the reviewed file/hunk is given context and the task is to retrieve additional context.

## Table 3: Same-Budget Final-Context Contrast

This table treats the top-k files from a static ranked baseline as if they were the final context files. It is a stronger trajectory-track contrast than top-20 recall because GPT-5.4-mini strict averages only 3.16 final files per sample.

| Method | Avg final files | Overall F1 | code2test F1 | comment2context F1 | trace2code F1 | Overall line F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical@3 final context | 3.00 | 0.0721 | 0.0255 | 0.0746 | 0.1191 | 0.0000 |
| RepoMap@3 final context | 3.00 | 0.1102 | 0.1082 | 0.0633 | 0.1495 | 0.0000 |
| RepoMap@4 final context | 4.00 | 0.1198 | 0.1165 | 0.0756 | 0.1583 | 0.0000 |
| GPT-5.4-mini strict-context | 3.16 | 0.3113 | 0.2516 | 0.1732 | 0.4835 | 0.0143 |

Interpretation:

- The strict agent run is substantially stronger than same-budget static top-k selection.
- RepoMap remains useful as a candidate generator, especially on `trace2code`, but taking only the first 3-4 ranked files is much weaker than iterative search/read selection.
- Static top-k rows have zero line F1 here because ranked details store file lists, not read windows. Use this table for file-level final-context comparison only.

## Table 4: GPT-5.4-mini Trajectory Release Comparison

| Run | Status | Samples | Read steps | Final files | File F1 | Line F1 | Redundancy | Below min | Final unread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai_gpt54mini_v1 | exploratory | 287 | 979 | n/a | 0.3007 | 0.0136 | 0.0000 | 0 | 146 |
| openai_gpt54mini_v2_strict_context | canonical | 287 | 918 | 907 | 0.3113 | 0.0143 | 0.0000 | 0 | 0 |

Interpretation:

- `openai_gpt54mini_v1` is useful as an exploratory run, but it is not the recommended release because answer-level `context_files` could include unread files.
- `openai_gpt54mini_v2_strict_context` is canonical: `final_files` are a strict subset of actually read files for all 287 samples.
- The strict run has slightly better file F1 than v1 while using fewer read steps and eliminating the answer/log mismatch.

## Main Takeaways

1. Single-shot and trajectory results should be reported separately. RepoMap and lexical retrieve ranked files; GPT-5.4-mini strict produces actual read-step logs and final context files.
2. Same-budget static top-k context is much weaker than iterative agent context selection: RepoMap@4 final context reaches `overall_final_file_f1=0.1198`, while GPT-5.4-mini strict reaches `0.3113` with an average of 3.16 final files.
3. `trace2code` is comparatively easier for the agentic run at file level (`final_file_f1=0.4835`) but still has low line-level coverage, showing that identifying a file is not the same as entering the right span.
4. `comment2context` is the hardest task for trajectory retrieval (`final_file_f1=0.1732`), which supports the benchmark design: retrieving the reviewed file is not enough.
5. The strict-context release is now suitable for citation and downstream analysis because logs, traces, and answer context agree.
