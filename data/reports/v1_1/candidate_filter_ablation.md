# Candidate Filter Ablation

- Generated at: `2026-05-28T03:24:57+00:00`
- Version: `v1_1`
- Scope: lexical and aider-style RepoMap baselines.
- Main leaderboard policy: keep `candidate_filter=all_files` for official model comparison.

This ablation measures how much retrieval quality changes when the candidate universe is restricted before ranking. It is a diagnostic experiment, not a replacement for the official leaderboard, because task-specific filters can remove valid gold files for other tasks.

## Overall MRR

| Model | Candidate filter | Samples | R@5 | R@10 | R@20 | MRR | Delta MRR vs all_files | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | all_files | 287 | 0.1922 | 0.3165 | 0.4750 | 0.1392 | +0.0000 | 0.0720 |
| lexical | code_only | 287 | 0.2143 | 0.3711 | 0.5732 | 0.1620 | +0.0228 | 0.0871 |
| lexical | tests_only | 287 | 0.1754 | 0.2764 | 0.3613 | 0.1562 | +0.0171 | 0.1858 |
| aider-style-repomap | all_files | 287 | 0.3131 | 0.4688 | 0.6419 | 0.2125 | +0.0000 | 0.0627 |
| aider-style-repomap | code_only | 287 | 0.3276 | 0.5094 | 0.6686 | 0.2283 | +0.0158 | 0.0697 |
| aider-style-repomap | tests_only | 287 | 0.2295 | 0.3063 | 0.3783 | 0.1984 | -0.0140 | 0.1074 |

## Task-Level MRR

| Model | Candidate filter | code2test | comment2context | trace2code |
| --- | --- | ---: | ---: | ---: |
| lexical | all_files | 0.0663 | 0.1495 | 0.2075 |
| lexical | code_only | 0.0789 (+0.0126) | 0.1613 (+0.0118) | 0.2498 (+0.0423) |
| lexical | tests_only | 0.2771 (+0.2107) | 0.1934 (+0.0439) | 0.0000 (-0.2075) |
| aider-style-repomap | all_files | 0.1962 | 0.1558 | 0.2745 |
| aider-style-repomap | code_only | 0.2284 (+0.0323) | 0.1590 (+0.0032) | 0.2829 (+0.0085) |
| aider-style-repomap | tests_only | 0.3875 (+0.1913) | 0.1984 (+0.0426) | 0.0000 (-0.2745) |

## Interpretation

- `code_only` improves overall MRR for both lexical retrieval and RepoMap, mostly by helping `trace2code` and reducing distracting non-source candidates.
- `tests_only` sharply improves `code2test` MRR, but it collapses `trace2code` to zero because root-cause source files are excluded from the candidate set.
- `comment2context` gains MRR under `tests_only` in these two vectorless baselines, but its R@20 generally drops; this is a candidate-space effect and should not be treated as a universal model improvement.
- The official leaderboard should stay on `all_files`; candidate-filtered results are best used to show that ARB is sensitive to candidate construction and that oracle-like filters can change conclusions.

## Reproduction Commands

```bash
arb eval-baseline --derived data/benchmark/v1_1 --corpus data/corpus/v1_1 --out data/eval/v1_1_candidate_filter_ablation/lexical_code_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/lexical_code_only_details.jsonl --candidate-filter code_only --no-keep-list
arb eval-baseline --derived data/benchmark/v1_1 --corpus data/corpus/v1_1 --out data/eval/v1_1_candidate_filter_ablation/lexical_tests_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/lexical_tests_only_details.jsonl --candidate-filter tests_only --no-keep-list
arb eval-repomap --derived data/benchmark/v1_1 --corpus data/corpus/v1_1 --out data/eval/v1_1_candidate_filter_ablation/repomap_code_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/repomap_code_only_details.jsonl --candidate-filter code_only --no-keep-list
arb eval-repomap --derived data/benchmark/v1_1 --corpus data/corpus/v1_1 --out data/eval/v1_1_candidate_filter_ablation/repomap_tests_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/repomap_tests_only_details.jsonl --candidate-filter tests_only --no-keep-list
```

## Embedding Follow-Up

Embedding candidate-filter ablations were not run locally in this pass because the local workspace does not contain reusable embedding caches or a GPU. Run the same `code_only` and `tests_only` filters on the GPU machine for the strongest open embedding models if the paper needs model-level ablations beyond vectorless baselines.

Example template:

```bash
arb eval-embedding --model /path/to/model --candidate-filter code_only --out data/eval/v1_1_candidate_filter_ablation/<model>_code_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/<model>_code_only_details.jsonl --cache data/embeddings/v1_1/<model> --shared-text-cache data/embeddings/v1_1/<model>_texts.sqlite --device cuda --trust-remote-code --no-keep-list --resume-details
arb eval-embedding --model /path/to/model --candidate-filter tests_only --out data/eval/v1_1_candidate_filter_ablation/<model>_tests_only_summary.json --details data/eval/v1_1_candidate_filter_ablation/<model>_tests_only_details.jsonl --cache data/embeddings/v1_1/<model> --shared-text-cache data/embeddings/v1_1/<model>_texts.sqlite --device cuda --trust-remote-code --no-keep-list --resume-details
```
