# V1.3 Reviewed Model Leaderboard

This document records the V1.3 reviewed retrieval leaderboard after rerunning embedding baselines on the reviewed annotation and gold-labeling scheme.

Source artifact on the GPU runner:

- `data/reports/v1_3/model_leaderboard.md`
- Generated at: `2026-06-06T08:47:37+00:00`
- Eval dir: `data/eval/v1_3_reviewed`
- Summary files: `7`
- Rows: `28`

The benchmark/corpus pairing is:

- Derived benchmark: `data/benchmark/v1_3_reviewed`
- Corpus: `data/corpus/v1_2`
- Candidate filter: `all_files`

## Overall

| Model | Candidate | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-Embedding-4B | `all_files` | 287 | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 | `Qwen3-Embedding-4B_summary.json` |
| Qwen3-Embedding-8B | `all_files` | 287 | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 | `Qwen3-Embedding-8B_summary.json` |
| pplx-embed-v1-4b | `all_files` | 287 | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 | `pplx-embed-v1-4b_summary.json` |
| aider-style-repomap | `all_files` | 287 | 0.3033 | 0.4728 | 0.6367 | 0.2120 | 0.0592 | `repomap_all_files_summary.json` |
| jina-code-embeddings-0.5b | `all_files` | 287 | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 | `jina-code-embeddings-0.5b_summary.json` |
| nomic-embed-code | `all_files` | 287 | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 | `nomic-embed-code_summary.json` |
| lexical | `all_files` | 287 | 0.1922 | 0.3182 | 0.4750 | 0.1402 | 0.0738 | `lexical_all_files_summary.json` |

## Code2Test

| Model | Candidate | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-Embedding-4B | `all_files` | 106 | 0.4610 | 0.5777 | 0.7230 | 0.3225 | 0.3887 | `Qwen3-Embedding-4B_summary.json` |
| pplx-embed-v1-4b | `all_files` | 106 | 0.3525 | 0.4789 | 0.6447 | 0.2516 | 0.1934 | `pplx-embed-v1-4b_summary.json` |
| Qwen3-Embedding-8B | `all_files` | 106 | 0.3774 | 0.5660 | 0.6594 | 0.2406 | 0.1651 | `Qwen3-Embedding-8B_summary.json` |
| nomic-embed-code | `all_files` | 106 | 0.2789 | 0.4462 | 0.6730 | 0.2066 | 0.0553 | `nomic-embed-code_summary.json` |
| jina-code-embeddings-0.5b | `all_files` | 106 | 0.2610 | 0.3868 | 0.5305 | 0.2033 | 0.2060 | `jina-code-embeddings-0.5b_summary.json` |
| aider-style-repomap | `all_files` | 106 | 0.2692 | 0.4013 | 0.5808 | 0.1942 | 0.0645 | `repomap_all_files_summary.json` |
| lexical | `all_files` | 106 | 0.0676 | 0.1399 | 0.2469 | 0.0663 | 0.0299 | `lexical_all_files_summary.json` |

## Comment2Context

| Model | Candidate | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| jina-code-embeddings-0.5b | `all_files` | 80 | 0.3563 | 0.4375 | 0.5792 | 0.3043 | 0.2521 | `jina-code-embeddings-0.5b_summary.json` |
| Qwen3-Embedding-4B | `all_files` | 80 | 0.3187 | 0.4771 | 0.6083 | 0.2920 | 0.3250 | `Qwen3-Embedding-4B_summary.json` |
| Qwen3-Embedding-8B | `all_files` | 80 | 0.3729 | 0.5021 | 0.6562 | 0.2874 | 0.2354 | `Qwen3-Embedding-8B_summary.json` |
| nomic-embed-code | `all_files` | 80 | 0.3521 | 0.4271 | 0.5021 | 0.2657 | 0.1875 | `nomic-embed-code_summary.json` |
| pplx-embed-v1-4b | `all_files` | 80 | 0.3146 | 0.5021 | 0.5938 | 0.2623 | 0.1625 | `pplx-embed-v1-4b_summary.json` |
| aider-style-repomap | `all_files` | 80 | 0.1646 | 0.2562 | 0.4583 | 0.1571 | 0.0396 | `repomap_all_files_summary.json` |
| lexical | `all_files` | 80 | 0.1667 | 0.3479 | 0.4979 | 0.1530 | 0.0625 | `lexical_all_files_summary.json` |

## Trace2Code

| Model | Candidate | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aider-style-repomap | `all_files` | 101 | 0.4488 | 0.7195 | 0.8366 | 0.2742 | 0.0693 | `repomap_all_files_summary.json` |
| lexical | `all_files` | 101 | 0.3432 | 0.4818 | 0.6964 | 0.2075 | 0.1287 | `lexical_all_files_summary.json` |
| Qwen3-Embedding-8B | `all_files` | 101 | 0.2442 | 0.5281 | 0.7970 | 0.1654 | 0.0759 | `Qwen3-Embedding-8B_summary.json` |
| pplx-embed-v1-4b | `all_files` | 101 | 0.1601 | 0.2855 | 0.5380 | 0.1372 | 0.0545 | `pplx-embed-v1-4b_summary.json` |
| nomic-embed-code | `all_files` | 101 | 0.1584 | 0.1733 | 0.3581 | 0.0871 | 0.0297 | `nomic-embed-code_summary.json` |
| Qwen3-Embedding-4B | `all_files` | 101 | 0.0743 | 0.1815 | 0.5050 | 0.0827 | 0.0495 | `Qwen3-Embedding-4B_summary.json` |
| jina-code-embeddings-0.5b | `all_files` | 101 | 0.0693 | 0.1188 | 0.2607 | 0.0607 | 0.0297 | `jina-code-embeddings-0.5b_summary.json` |

## Takeaways

1. `Qwen3-Embedding-4B` is strongest overall by MRR and clearly strongest on `code2test`.
2. `Qwen3-Embedding-8B` has the best overall R@20 and the best `comment2context` R@20.
3. `jina-code-embeddings-0.5b` has the best `comment2context` MRR, despite being smaller than Qwen.
4. `trace2code` remains structure-heavy: RepoMap is first and lexical is second by MRR, ahead of all embedding models.
5. `pplx-embed-v1-4b` is competitive overall and second by MRR on `code2test`, but it does not change the main task-level pattern.

## Artifact Status

The tracked repo stores this leaderboard document, but not the large `data/eval/v1_3_reviewed/*_details.jsonl` and embedding cache artifacts. The summary/details outputs should be uploaded to the private HF dataset repo so the leaderboard is reproducible from artifacts rather than only from this document.
