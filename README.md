# Agent Retrieval Bench

[![Dataset](https://img.shields.io/badge/Hugging%20Face-dataset-yellow)](https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/citation-CFF-lightgrey)](CITATION.cff)

Agent Retrieval Bench (ARB) evaluates the context-acquisition layer of coding
agents: given a real workflow signal and a repository at a frozen base commit,
can a retriever find the files an agent needs to read next, or correctly decide
that the repository contains no useful local context?

ARB differs from generic semantic code search in its relevance definition.
Relevant files are selected by the next need of the coding workflow, not by
query-file text similarity alone. A failure trace may expose a test while the
needed file is its root-cause implementation; a review comment may name one
file while the missing constraint lives elsewhere.

## Benchmark

ARB contains 427 samples across 25 repositories:

| Subset | Samples | Query signal | Expected output |
| --- | ---: | --- | --- |
| `code2test` | 106 | PR intent or implementation-change signal | Related tests |
| `comment2context` | 80 | Review comment plus an already-given file | Additional context files |
| `trace2code` | 101 | Reproduced failure output | Root-cause source files |
| `edit2ripple` | 58 | Intent plus one anchored file change | Other affected files |
| Natural no-gold | 50 | Issue resolved outside the local repository | Abstain |
| Counterfactual no-gold | 32 | Plausible query paired with the wrong repository | Abstain |
| **Total** | **427** | Four positive workflows plus selective retrieval | Files or abstention |

The four positive tasks contain 345 samples. The no-gold samples are evaluated
only in the selective-retrieval track and are not mixed into positive-only MRR,
Recall, or BCY.

The combined release corpus contains 308 reusable repository/base-commit
snapshots, approximately 392K files, and 7.9M chunks. The evaluated rows use 271
of those snapshots.

### Design principles

- **Agentic relevance:** gold files represent context needed next in a coding
  workflow, including indirect structural, causal, and project-conventional
  relationships.
- **Base-commit corpora:** positive samples are evaluated before the resolving
  change, preventing fixed-code leakage.
- **All-file retrieval:** the official candidate set is `all_files`; task-aware
  filters are diagnostic ablations.
- **Context budgets:** canonical Budgeted Context Yield (BCY) uses token-based
  ranked-file packing and is reported over 4k/8k/16k/32k budgets.
- **Selective retrieval:** natural no-gold cases are analyzed separately from
  easier wrong-repository counterfactual controls.
- **Explicit scope:** the primary benchmark is file-level retrieval. Span
  diagnostics are available for `code2test`, `comment2context`, and
  `trace2code`, while `edit2ripple` does not yet have span gold.

## Results at a Glance

Positive retrieval results over all 345 samples:

| Method | Recall@20 | MRR | BCY@8k |
| --- | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 0.6306 | **0.2379** | 0.3409 |
| Qwen3-Embedding-8B | **0.7029** | 0.2336 | 0.3732 |
| pplx-embed-v1-4b* | 0.6072 | 0.2267 | 0.3549 |
| RepoMap | 0.6333 | 0.2158 | **0.3788** |
| nomic-embed-code | 0.5244 | 0.1986 | 0.2781 |
| jina-code-embeddings-0.5b | 0.4823 | 0.1914 | 0.2783 |
| Lexical | 0.4940 | 0.1574 | 0.2650 |
| BM25 | 0.4452 | 0.1520 | 0.2051 |

\* The recorded pplx run is provisional because of a tokenizer warning.

No retrieval family dominates all workflow signals:

- Qwen3-Embedding-4B leads `code2test` MRR.
- Jina leads `comment2context` MRR.
- RepoMap leads both MRR and Recall@20 on `trace2code`.
- On `edit2ripple`, Qwen3-4B has the best Recall@20, while the verified Qwen
  ordering reverses between fully packed 4k and 8k context budgets.

Selective retrieval exposes a negative result. Under repo-grouped
cross-validation on the 50 natural no-gold cases, top-score thresholding with
Lexical, BM25, or Jina does not improve selective success over always returning
a ranking. Raw retrieval confidence is not yet a reliable
repository-independent abstention signal.

## Installation

ARB requires Python 3.10 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Install embedding dependencies only when evaluating embedding models:

```bash
pip install -e ".[embedding]"
```

The dataset repository is public; authentication is not required.

## Download

The benchmark is distributed as five self-contained subset releases. Each
bundle contains its benchmark JSONL, referenced corpus snapshots, checksums, and
available evaluation/report artifacts.

```bash
arb releases
arb download-benchmark --all --local-dir data --force
```

To download only one subset, pass its release ID explicitly:

```bash
arb download-benchmark \
  --version v2_edit2ripple \
  --local-dir data \
  --force
```

`download-benchmark` requires either `--version` or `--all`; it does not
silently select a legacy unversioned bundle.

Each release extracts to:

```text
data/benchmark/<release-id>/
data/corpus/<release-id>/
data/eval/<release-id>/
data/reports/<release-id>/
```

Verify one subset after extraction:

```bash
arb validate data/benchmark/v2_edit2ripple/samples.jsonl
```

### Selective-retrieval inputs

The exact mixed sample lists are distributed separately:

```bash
arb download-benchmark \
  --version v2_selective_retrieval_balanced \
  --local-dir data \
  --force

arb download-benchmark \
  --version v2_selective_retrieval_natural \
  --local-dir data \
  --force
```

Selective evaluation shares the five subset corpora. Build a deduplicated
manifest after downloading the base releases:

```bash
arb merge-corpus-manifests --local-dir data
```

The five self-contained release corpora merge to 271 unique snapshots used by
the released samples. The larger 308-snapshot corpus inventory reported in the
paper also includes reusable snapshots not referenced by these 427 samples.

## Evaluate a Retriever

The following examples use `edit2ripple`; replace the release ID to evaluate a
different positive subset.

Lexical and BM25:

```bash
arb eval-baseline \
  --derived data/benchmark/v2_edit2ripple \
  --corpus data/corpus/v2_edit2ripple \
  --ranker lexical \
  --candidate-filter all_files \
  --no-keep-list \
  --out data/eval/v2_edit2ripple/lexical_summary.json \
  --details data/eval/v2_edit2ripple/lexical_details.jsonl

arb eval-baseline \
  --derived data/benchmark/v2_edit2ripple \
  --corpus data/corpus/v2_edit2ripple \
  --ranker bm25 \
  --candidate-filter all_files \
  --no-keep-list \
  --out data/eval/v2_edit2ripple/bm25_summary.json \
  --details data/eval/v2_edit2ripple/bm25_details.jsonl
```

RepoMap:

```bash
arb eval-repomap \
  --derived data/benchmark/v2_edit2ripple \
  --corpus data/corpus/v2_edit2ripple \
  --candidate-filter all_files \
  --no-keep-list \
  --out data/eval/v2_edit2ripple/repomap_summary.json \
  --details data/eval/v2_edit2ripple/repomap_details.jsonl
```

A SentenceTransformers-compatible embedding model:

```bash
arb eval-embedding \
  --version v2_edit2ripple \
  --model /path/to/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 8 \
  --query-batch-size 8 \
  --resume-details
```

Run a selective lexical baseline:

```bash
arb eval-selective-baseline \
  --derived data/benchmark/v2_selective_retrieval_natural \
  --corpus data/corpus/v2_selective_mixed \
  --ranker lexical \
  --candidate-filter all_files \
  --no-keep-list \
  --out data/eval/v2_selective/lexical_summary.json \
  --details data/eval/v2_selective/lexical_details.jsonl
```

Available evaluator families include Lexical, BM25, grep-style exact search,
RepoMap, embedding retrieval, selective abstention, logged trajectories, and
closed-tool context acquisition. Run `arb --help` or `arb <command> --help` for
the complete CLI.

## Metrics

- **MRR:** reciprocal rank of the first gold file.
- **Recall@k:** fraction of gold files recovered in the first `k` ranked files.
- **Precision@k / F0.5@k:** context-selection precision diagnostics.
- **BCY@B:** fraction of gold files exposed after token-based ranked-file
  packing under budget `B`.
- **Selective success@20:** a no-gold abstention is correct; an accepted
  positive is correct only when a gold file appears in the top 20.
- **File/line context metrics:** used for logged and closed-tool trajectories,
  not mixed into the static retrieval leaderboard.

Legacy JSON fields named `gold_coverage@8k` and `context_efficiency@8k` use an
older character-counted packing implementation. They are not canonical BCY.

## Repository Layout

```text
src/agent_retrieval_bench/   evaluator and reporting code
tests/                       unit and integration tests
scripts/                     reproducibility entry points
docs/                        construction and analysis documentation
data/                        lightweight reports plus downloaded releases
```

Run the test suite with:

```bash
PYTHONPATH=src pytest -q
```

## Scope

ARB isolates retrieval and context acquisition. It does not claim that a
file-level hit identifies the correct function or span, or that better
retrieval necessarily produces a test-passing patch. The current seed
intervention evaluates context-selection behavior, not end-to-end repair
success.

Corpus source files remain under their upstream repository licenses. Evaluator
code, benchmark metadata, and documentation are MIT licensed; see
[DATA_LICENSE.md](DATA_LICENSE.md).

## Citation

```bibtex
@misc{qin2026agentretrievalbench,
  title  = {Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents},
  author = {Bowen Qin and Yi Xie},
  year   = {2026},
  url    = {https://github.com/eyuansu62/agent-retrieval-bench}
}
```

A machine-readable citation record is available in [CITATION.cff](CITATION.cff).
