# V1.1 Paper Table Drafts

These tables are paper-facing drafts derived from the current V1.1 release artifacts. They are written in Markdown first so they can be copied into LaTeX later.

Source artifacts:

- `data/benchmark/v1_1/manifest.json`
- `data/corpus/v1_1/corpus_manifest.jsonl`
- `data/reports/v1_1/readiness.json`
- `data/reports/v1_1/release_report.json`
- `data/reports/v1_1/model_leaderboard.json`
- `data/reports/v1_1/embedding_dedup_v19.json`
- `data/reports/v1_1/candidate_filter_ablation.json`
- `data/reports/v1_1/rank_analysis.json`

## Table 1: Dataset Composition

| Track | Query Signal | Gold Target | V1 Samples | V1.1 Samples | V1.1 Additions |
| --- | --- | --- | ---: | ---: | ---: |
| `code2test` | PR or implementation-change intent | Related test files | 106 | 106 | 0 |
| `comment2context` | Review comment plus reviewed file | Additional context files beyond the reviewed file | 51 | 80 | 29 |
| `trace2code` | Reproduced failure log or trace | Root-cause source files | 68 | 101 | 33 |
| Total | Mixed coding-agent workflow signals | Files the agent needs to read | 225 | 287 | 62 |

Suggested caption:

> Agent Retrieval Bench V1.1 is a targeted expansion of V1. It preserves the frozen `code2test` track and expands the weaker `comment2context` and `trace2code` tracks with 62 manually audited additions.

## Table 2: Corpus and Release Scale

| Quantity | Value |
| --- | ---: |
| V1.1 samples | 287 |
| Sample repositories | 25 |
| Corpus repositories | 29 |
| Corpus repo/base rows | 261 |
| Corpus files | 345,776 |
| Corpus chunks | 7,016,525 |
| Repo/base corpora touched by samples | 218 |
| Chunks touched by sampled corpora | 6,210,965 |
| Unique embedding texts in sampled corpora | 1,118,431 |
| Duplicate embedding texts in sampled corpora | 5,092,534 |
| Duplicate fraction | 81.99% |

Suggested caption:

> V1.1 evaluates a small manually audited sample set against large real repository snapshots. The high duplicate fraction across nearby base commits motivates shared corpus-embedding caches for practical evaluation.

## Table 3: V1.1 Quality and Release Gates

| Gate | V1.1 Status |
| --- | --- |
| Frozen V1 IDs preserved | Passed |
| `code2test` count unchanged | Passed |
| New samples limited to `comment2context` and `trace2code` | Passed |
| New samples have manual audit evidence | Passed |
| New gold files present in base-commit corpus | Passed |
| No query leakage | Passed |
| No direct gold path or basename hints | Passed |
| No path-role overlap | Passed |
| New `comment2context` samples have `given_files` | Passed |
| New `comment2context` avoids same-directory shortcut gold | Passed |
| New cross-module `comment2context` samples | 14 |
| New `trace2code` gold files are not test-only gold | Passed |
| New `trace2code` non-Go samples | 33 |
| New `trace2code` gold extensions | `.py`, `.rs` |
| New `trace2code` failure types | assertion, compile error, exception, panic, test failure, timeout |
| Readiness report | `ready=true` |
| Release report | `status=ready` |
| Completion audit | 16/16 requirements passed |

Suggested caption:

> V1.1 uses explicit readiness and release gates to reject leakage, shortcut labels, missing corpus gold, and unaudited additions.

## Table 4: Overall Leaderboard

Rows are sorted by overall MRR. All rows use `candidate_filter=all_files` and evaluate all 287 V1.1 samples.

| Model | Family | R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | embedding | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 |
| Qwen3-Embedding-8B | embedding | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 |
| pplx-embed-v1-4b | embedding | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 |
| aider-style-repomap | vectorless repo map | 0.3131 | 0.4688 | 0.6419 | 0.2125 | 0.0627 |
| jina-code-embeddings-0.5b | embedding | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 |
| nomic-embed-code | embedding | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 |
| lexical | lexical | 0.1922 | 0.3165 | 0.4750 | 0.1392 | 0.0720 |

Suggested caption:

> Qwen3-Embedding-4B has the best overall MRR, but the narrow aggregate margin hides strong task-level differences.

## Table 5: Task Winners

| Task | Best MRR Model | MRR | Best R@20 Model | R@20 | Interpretation |
| --- | --- | ---: | --- | ---: | --- |
| overall | Qwen3-Embedding-4B | 0.2296 | Qwen3-Embedding-8B | 0.7070 | Strong embeddings lead aggregate performance. |
| `code2test` | Qwen3-Embedding-4B | 0.3225 | Qwen3-Embedding-4B | 0.7230 | Source-to-test retrieval benefits from broad semantic/code embeddings. |
| `comment2context` | jina-code-embeddings-0.5b | 0.3043 | Qwen3-Embedding-8B | 0.6562 | Review-context retrieval is competitive across embedding models. |
| `trace2code` | aider-style-repomap | 0.2745 | aider-style-repomap | 0.8465 | Failure trace retrieval strongly benefits from path/symbol/repo-graph signals. |

Suggested caption:

> The best retriever changes by task, showing that agentic retrieval signals require different inductive biases.

## Table 6: Full Task-Level Results

| Task | Model | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code2test` | Qwen3-Embedding-4B | 106 | 0.4610 | 0.5777 | 0.7230 | 0.3225 | 0.3887 |
| `code2test` | pplx-embed-v1-4b | 106 | 0.3525 | 0.4789 | 0.6447 | 0.2516 | 0.1934 |
| `code2test` | Qwen3-Embedding-8B | 106 | 0.3774 | 0.5660 | 0.6594 | 0.2406 | 0.1651 |
| `code2test` | nomic-embed-code | 106 | 0.2789 | 0.4462 | 0.6730 | 0.2066 | 0.0553 |
| `code2test` | jina-code-embeddings-0.5b | 106 | 0.2610 | 0.3868 | 0.5305 | 0.2033 | 0.2060 |
| `code2test` | aider-style-repomap | 106 | 0.2597 | 0.3918 | 0.5761 | 0.1962 | 0.0786 |
| `code2test` | lexical | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 |
| `comment2context` | jina-code-embeddings-0.5b | 80 | 0.3563 | 0.4375 | 0.5792 | 0.3043 | 0.2521 |
| `comment2context` | Qwen3-Embedding-4B | 80 | 0.3187 | 0.4771 | 0.6083 | 0.2920 | 0.3250 |
| `comment2context` | Qwen3-Embedding-8B | 80 | 0.3729 | 0.5021 | 0.6562 | 0.2874 | 0.2354 |
| `comment2context` | nomic-embed-code | 80 | 0.3521 | 0.4271 | 0.5021 | 0.2657 | 0.1875 |
| `comment2context` | pplx-embed-v1-4b | 80 | 0.3146 | 0.5021 | 0.5938 | 0.2623 | 0.1625 |
| `comment2context` | aider-style-repomap | 80 | 0.1812 | 0.2604 | 0.4708 | 0.1558 | 0.0333 |
| `comment2context` | lexical | 80 | 0.1667 | 0.3417 | 0.4979 | 0.1495 | 0.0563 |
| `trace2code` | aider-style-repomap | 101 | 0.4736 | 0.7145 | 0.8465 | 0.2745 | 0.0693 |
| `trace2code` | lexical | 101 | 0.3432 | 0.4818 | 0.6964 | 0.2075 | 0.1287 |
| `trace2code` | Qwen3-Embedding-8B | 101 | 0.2442 | 0.5281 | 0.7970 | 0.1654 | 0.0759 |
| `trace2code` | pplx-embed-v1-4b | 101 | 0.1601 | 0.2855 | 0.5380 | 0.1372 | 0.0545 |
| `trace2code` | nomic-embed-code | 101 | 0.1584 | 0.1733 | 0.3581 | 0.0871 | 0.0297 |
| `trace2code` | Qwen3-Embedding-4B | 101 | 0.0743 | 0.1815 | 0.5050 | 0.0827 | 0.0495 |
| `trace2code` | jina-code-embeddings-0.5b | 101 | 0.0693 | 0.1188 | 0.2607 | 0.0607 | 0.0297 |

Suggested caption:

> Task-level results show that semantic embeddings are not uniformly dominant. RepoMap and lexical retrieval are especially strong on `trace2code`, where path, symbol, and repository-structure signals matter.

## Table 7: Runtime and Cache Evidence

This table is not final until wall-clock runs are recorded consistently on the GPU machine. Fill the runtime columns from future run logs.

| Model | Device | Batch Size | Query Batch Size | Shared Text Cache | Wall Time | Corpus Cache Size | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| jina-code-embeddings-0.5b | GPU TBD | TBD | defaults to batch size | yes | TBD | TBD | Small open-source code embedding baseline. |
| Qwen3-Embedding-4B | GPU TBD | TBD | defaults to batch size | yes | TBD | TBD | Strongest overall V1.1 MRR. |
| Qwen3-Embedding-8B | GPU TBD | TBD | defaults to batch size | yes | TBD | TBD | Strongest overall Recall@20. |
| pplx-embed-v1-4b | GPU TBD | TBD | defaults to batch size | yes | TBD | TBD | Additional open-source comparison. |
| nomic-embed-code | GPU TBD | TBD | defaults to batch size | yes | TBD | TBD | Additional open-source comparison. |

Suggested caption:

> Runtime is a practical part of benchmark reproducibility. Shared text caching can avoid re-embedding duplicate corpus chunks across nearby base commits; query batching reduces per-sample model-call overhead.

## Table 8: Candidate Filter Ablation

Rows use vectorless baselines on all 287 V1.1 samples. Values are MRR; parenthesized values are deltas against the same model with `candidate_filter=all_files`.

| Model | Candidate Filter | Overall | `code2test` | `comment2context` | `trace2code` |
| --- | --- | ---: | ---: | ---: | ---: |
| lexical | `all_files` | 0.1392 | 0.0663 | 0.1495 | 0.2075 |
| lexical | `code_only` | 0.1620 (+0.0228) | 0.0789 (+0.0126) | 0.1613 (+0.0118) | 0.2498 (+0.0423) |
| lexical | `tests_only` | 0.1562 (+0.0171) | 0.2771 (+0.2107) | 0.1934 (+0.0439) | 0.0000 (-0.2075) |
| aider-style-repomap | `all_files` | 0.2125 | 0.1962 | 0.1558 | 0.2745 |
| aider-style-repomap | `code_only` | 0.2283 (+0.0158) | 0.2284 (+0.0323) | 0.1590 (+0.0032) | 0.2829 (+0.0085) |
| aider-style-repomap | `tests_only` | 0.1984 (-0.0140) | 0.3875 (+0.1913) | 0.1984 (+0.0426) | 0.0000 (-0.2745) |

Suggested caption:

> Candidate filters are diagnostic rather than leaderboard settings. `tests_only` strongly helps `code2test` but removes root-cause source files for `trace2code`; `code_only` improves vectorless retrieval overall by reducing distracting non-source candidates.

## Table 9: First-Gold Rank Depth

`Any@k` is the fraction of samples where at least one gold file appears in the top `k`. This differs from the main Recall@k metric when a sample has multiple gold files.

| Model | Overall Any@5 | Overall Any@20 | Overall Median Hit Rank | `trace2code` Any@20 | `trace2code` Median Hit Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 0.3589 | 0.6864 | 11 | 0.5347 | 20 |
| Qwen3-Embedding-8B | 0.4077 | 0.7944 | 7 | 0.8614 | 9 |
| pplx-embed-v1-4b | 0.3310 | 0.6794 | 10 | 0.5941 | 18 |
| aider-style-repomap | 0.3728 | 0.7247 | 9 | 0.9109 | 5 |
| jina-code-embeddings-0.5b | 0.2683 | 0.5226 | 19 | 0.2871 | 29 |
| nomic-embed-code | 0.3101 | 0.5784 | 15 | 0.3861 | 25 |
| lexical | 0.2474 | 0.5470 | 17 | 0.7525 | 8 |

Suggested caption:

> First-gold depth shows a ranking tradeoff hidden by aggregate MRR. Qwen3-Embedding-4B has the best overall MRR, while Qwen3-Embedding-8B retrieves at least one gold file for more samples by depth 20. On `trace2code`, RepoMap has both the strongest Any@20 and the shallowest median hit rank.

## Table 10: Cross-Model Misses

Rows count samples where none of the seven reported V1.1 `all_files` baselines retrieves any gold file in the top 20.

| Task | Samples | Any Model Hit@20 | All Models Miss@20 |
| --- | ---: | ---: | ---: |
| overall | 287 | 265 | 22 |
| `code2test` | 106 | 96 | 10 |
| `comment2context` | 80 | 71 | 9 |
| `trace2code` | 101 | 98 | 3 |

Suggested caption:

> Cross-model misses identify hard benchmark cases rather than weak individual models. Only 22 of 287 samples are missed by every baseline at depth 20, and `trace2code` has the fewest all-model misses despite large model-ranking differences.

## Result Claims to Support in Text

1. **Aggregate ranking alone is misleading.** Qwen3-Embedding-4B is best overall by MRR, but different models win the three task families.
2. **`trace2code` is the strongest evidence for structure-aware retrieval.** RepoMap outperforms all embeddings on both MRR and Recall@20 for this task.
3. **`comment2context` is not solved by finding the reviewed file.** The task treats the reviewed file as given context; the benchmark asks for additional files.
4. **Embedding models remain useful but insufficient.** Strong embeddings lead overall and on `code2test`, but their weakness on `trace2code` motivates hybrid retrieval.
5. **The benchmark is diagnostic.** The sample count is intentionally modest because samples are manually audited and leakage-controlled.

## Missing Paper Tables

These should be added before submission if time allows:

- Runtime table with measured wall time and cache hit/miss counts.
- Hybrid retrieval pilot if implemented.
