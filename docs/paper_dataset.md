# Agent Retrieval Bench Dataset

This is the paper-facing dataset description. It intentionally avoids internal version names. Versioned paths are implementation artifacts for provenance and compatibility only.

## Canonical Dataset

Agent Retrieval Bench contains 287 reviewed retrieval samples built from real coding workflow signals. Every sample is evaluated against a base-commit repository corpus, before the resolving patch is applied.

| Task | Samples | Query signal | Gold target |
| --- | ---: | --- | --- |
| `code2test` | 106 | PR intent or implementation-change summary | Related tests |
| `comment2context` | 80 | Review comment plus the reviewed file | Additional context files beyond the reviewed file |
| `trace2code` | 101 | Reproduced failure output | Root-cause source files |
| Total | 287 | Mixed coding-agent workflow signals | Files an agent should read |

Corpus and label coverage:

- Sample-set repositories: 25.
- Corpus repositories: 29.
- Corpus rows: 261 repository/base-commit rows.
- Corpus size: 345,776 files and 7,016,525 chunks.
- Span-labeled samples: 287/287.
- Block-labeled samples: 287/287.
- Hard-negative samples: 150/287.
- Query-provenance samples: 287/287.
- Validation status: `ready=true`, `invalid=0`.

## Canonical Public Layout

The paper and public docs should use this layout:

```text
data/benchmark/agent_retrieval_bench/
  manifest.json
  samples.jsonl
  code2test.jsonl
  comment2context.jsonl
  trace2code.jsonl
data/corpus/agent_retrieval_bench/
  corpus_manifest.jsonl
  **/*.chunks.jsonl
data/eval/agent_retrieval_bench/
  *_summary.json
  *_details.jsonl
data/reports/agent_retrieval_bench/
  model_leaderboard.md
  model_leaderboard.json
  reviewed_validation.md
  reviewed_validation.json
data/releases/agent_retrieval_bench/
  agent_retrieval_bench.tar.zst
  agent_retrieval_bench.tar.zst.sha256
```

Historical/internal paths map to the same paper-facing dataset:

| Paper-facing component | Internal source artifact |
| --- | --- |
| Benchmark samples | `data/benchmark/v1_3_reviewed` |
| Corpus | `data/corpus/v1_2` |
| Single-shot eval | `data/eval/v1_3_reviewed` |
| Single-shot reports | `data/reports/v1_3` |
| Trajectory eval | `data/eval/v1_4` |
| Trajectory release | `data/releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context` |

## Leaderboard Summary

Single-shot retrieval uses the `all_files` candidate set.

| Model | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 287 | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 |
| Qwen3-Embedding-8B | 287 | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 |
| pplx-embed-v1-4b | 287 | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 |
| aider-style-repomap | 287 | 0.3033 | 0.4728 | 0.6367 | 0.2120 | 0.0592 |
| jina-code-embeddings-0.5b | 287 | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 |
| nomic-embed-code | 287 | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 |
| lexical | 287 | 0.1922 | 0.3182 | 0.4750 | 0.1402 | 0.0738 |

Task-level winners:

- `code2test`: Qwen3-Embedding-4B, MRR `0.3225`.
- `comment2context`: jina-code-embeddings-0.5b, MRR `0.3043`.
- `trace2code`: aider-style-repomap, MRR `0.2742`.

## Trajectory Track

The trajectory track is part of the same benchmark story, but it evaluates logged context selection rather than ranked retrieval.

| Task | Run | Samples | File R | File P | File F1 | Line F1 | Avg reads | Avg final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | GPT-5.4-mini strict-context | 287 | 0.5645 | 0.2236 | 0.3113 | 0.0143 | 3.199 | 3.160 |
| `code2test` | GPT-5.4-mini strict-context | 106 | 0.4575 | 0.1792 | 0.2516 | 0.0042 | 3.170 | 3.142 |
| `comment2context` | GPT-5.4-mini strict-context | 80 | 0.2875 | 0.1333 | 0.1732 | 0.0064 | 3.325 | 3.263 |
| `trace2code` | GPT-5.4-mini strict-context | 101 | 0.8960 | 0.3416 | 0.4835 | 0.0312 | 3.129 | 3.099 |

## Wording Guidance

Use:

- "Agent Retrieval Bench contains 287 reviewed samples."
- "The canonical dataset includes file, span, and block labels."
- "Historical internal artifact paths are retained for reproducibility."

Avoid:

- Presenting the paper as a sequence of internal releases.
- Calling the labels independent human annotations unless an independent human review pass is added.
- Treating trajectory results and single-shot ranking results as the same metric family.
