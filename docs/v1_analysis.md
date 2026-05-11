# Agent Retrieval Bench V1 Analysis

Full V1 is the primary benchmark release: 225 manually curated samples across `code2test=106`, `comment2context=51`, and `trace2code=68`. The benchmark is frozen; future data work should target V1.1 rather than mutating `benchmark/v1`.

## Model Findings

- `Qwen3-Embedding-4B` is strongest overall by MRR and is clearly strongest on `code2test`.
- `jina-code-embeddings-0.5b` is strongest on `comment2context` by MRR.
- `aider-style-repomap` is strongest on `trace2code` and has the best overall `Recall@20`.
- Embedding models underperform lexical/RepoMap on `trace2code`, so the trace track is measuring structured failure-signal retrieval rather than only semantic code similarity.

## Current Leaderboard

| Task | Best MRR Model | MRR | R@20 | Best R@20 Model | Best R@20 |
| --- | --- | ---: | ---: | --- | ---: |
| overall | Qwen3-Embedding-4B | 0.2455 | 0.5828 | aider-style-repomap | 0.6299 |
| code2test | Qwen3-Embedding-4B | 0.3225 | 0.7230 | Qwen3-Embedding-4B | 0.7230 |
| comment2context | jina-code-embeddings-0.5b | 0.3282 | 0.5261 | lexical | 0.5752 |
| trace2code | aider-style-repomap | 0.2750 | 0.8064 | aider-style-repomap | 0.8064 |

## V1.1 Priorities

- Expand `comment2context` from 51 to 80-100 with cross-module required context, no direct path/basename leaks, and non-same-directory gold.
- Expand `trace2code` from 68 to 100+ with more non-Go repos and diverse real repro failure types.
- Do not expand `code2test` by default; 106 valid samples are enough for current V1.
