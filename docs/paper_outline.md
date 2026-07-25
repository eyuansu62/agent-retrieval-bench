# Agent Retrieval Bench Paper Outline

Working title: **Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents**

This is the paper-facing story. It intentionally hides internal release iteration. Version names such as `v1_3_reviewed` and `v1_4` should appear only as artifact labels in reproducibility notes, not as the conceptual structure of the paper. The current full draft is `docs/paper_draft.md`, and the paper-facing dataset snapshot is maintained in `docs/paper_dataset.md`.

## One-Sentence Thesis

Coding agents need a benchmark for the upstream step of finding actionable repository context, because PR intent, review comments, and failure traces often require indirect source/test/context reasoning rather than semantic code similarity alone.

## Abstract Draft

Modern coding agents are usually evaluated by whether they eventually produce a correct patch, but many failures begin earlier: the agent never finds the repository files it needs to read. We introduce Agent Retrieval Bench, a file-level repository context retrieval benchmark for this upstream context-finding problem. Each sample is built from a real coding workflow signal and evaluated against the repository at the base commit, before the resolving patch is applied. The benchmark covers three agentic retrieval tasks: `code2test`, where implementation intent must retrieve related tests; `comment2context`, where a review comment and given file must retrieve additional contextual files; and `trace2code`, where reproduced failure output must retrieve root-cause source files rather than merely visible test frames.

Agent Retrieval Bench contains 287 reviewed samples across 25 repositories, with corpus artifacts spanning 261 repository/base-commit rows, 345,776 files, and 7,016,525 chunks. The benchmark includes file-level gold labels plus span and block annotations for every sample, enabling both retrieval ranking and context granularity analysis. We evaluate lexical retrieval, a vectorless RepoMap-style structural retriever, and five open-source embedding models. Results show that no retrieval family dominates: Qwen3-Embedding-4B is strongest overall by MRR and on `code2test`, Jina is strongest on `comment2context` by MRR, and the vectorless RepoMap baseline is strongest on `trace2code`. A trajectory study with GPT-5.4-mini further shows that iterative context selection can outperform same-budget static top-k retrieval, but still struggles with span-level precision. These findings suggest that coding-agent retrieval needs hybrid methods combining semantic vectors with repository structure, path/symbol signals, and task-aware search.

## Paper Framing Rule

Do not tell the paper as a sequence of internal versions. The paper should have one benchmark and two evaluation tracks:

1. **Single-shot retrieval track:** rank candidate repository files/chunks from a workflow query.
2. **Trajectory context track:** evaluate which files and spans actually enter an agent's read/final context.

Internal version names are implementation details:

| Paper concept | Internal artifact |
| --- | --- |
| Benchmark samples | `data/benchmark/v1_3_reviewed` |
| Candidate corpus | `data/corpus/v1_2` |
| Single-shot eval artifacts | `data/eval/v1_3_reviewed` |
| Single-shot leaderboard | `data/reports/v1_3/model_leaderboard.md` |
| Trajectory eval artifacts | `data/eval/v1_4` |
| Trajectory release | `releases/trajectories/v1_3_openai_gpt54mini_v2_strict_context/` |

## Core Contributions

1. **A task definition for agentic repository retrieval.** The benchmark measures whether a retriever finds files an agent needs to read, not whether it retrieves semantically similar snippets.
2. **Three workflow-grounded retrieval tasks.** `code2test`, `comment2context`, and `trace2code` isolate different upstream context needs in coding-agent workflows.
3. **Reviewed file, span, and block labels.** Each sample has file-level gold labels plus reviewed span and block annotations, with 150 samples also carrying hard negatives.
4. **Leakage-resistant construction.** Queries avoid direct gold paths, gold basenames, raw patches, final diffs, and fixed-state corpus contamination.
5. **A multi-paradigm baseline analysis.** Results compare lexical retrieval, vectorless repository-structure retrieval, and embedding retrievers and show strong task-dependent behavior.
6. **A trajectory context evaluation track.** The benchmark can evaluate actual read/final context logs, not only ranked lists.
7. **Reproducible artifacts.** The release includes benchmark files, corpus manifests, eval details, leaderboards, trajectory logs, checksums, and smoke-testable bundles.

## Research Questions

RQ1. **Are semantic embeddings enough for agentic retrieval signals?**

Expected answer: no. Embeddings are strong for `code2test` and competitive for `comment2context`, but they are weak on `trace2code` compared with RepoMap and lexical retrieval.

RQ2. **Do vectorless repository-structure methods provide independent value?**

Expected answer: yes. RepoMap is strongest on `trace2code` by MRR and Recall@20, suggesting that path, symbol, stack, and repository-topology signals are central for failure-log root-cause retrieval.

RQ3. **Do different workflow signals need different retrieval biases?**

Expected answer: yes. Qwen3-Embedding-4B wins `code2test`, Jina wins `comment2context` by MRR, and RepoMap wins `trace2code`.

RQ4. **Does evaluating agent trajectories reveal a different context-selection problem than static top-k retrieval?**

Expected answer: yes. At a comparable final-context budget, GPT-5.4-mini strict-context reaches much higher file F1 than static top-k lexical or RepoMap final-context baselines, while still showing weak line-level alignment.

## Benchmark Definition

Agent Retrieval Bench evaluates file-level repository context retrieval. The corpus is chunked for ranking, but primary metrics are computed at the file level because coding agents need readable repository files as context.

The candidate corpus is fixed at the base commit. The query and candidate files are from the state before the resolving patch, which prevents leakage from the final fix.

Gold files represent files needed to complete the task:

- `code2test`: related tests for a PR or implementation-change signal.
- `comment2context`: additional files needed to understand or satisfy a review comment; the reviewed file is given context and is not enough by itself.
- `trace2code`: root-cause source files for a reproduced failure; test files are auxiliary context, not the primary target.

## Dataset Snapshot

| Task | Samples | Retrieval signal | Target context |
| --- | ---: | --- | --- |
| `code2test` | 106 | PR intent or implementation-change summary | Related regression, integration, or unit tests |
| `comment2context` | 80 | Review comment plus given reviewed file | Additional implementation, policy, config, or test context |
| `trace2code` | 101 | Reproduced failure output | Root-cause source files |
| Total | 287 | Mixed coding workflow signals | Files an agent should read |

Corpus and annotation facts:

- Sample set repositories: 25.
- Corpus manifest rows: 261 repository/base-commit rows.
- Corpus repositories: 29.
- Corpus scale: 345,776 files and 7,016,525 chunks.
- Span-labeled samples: 287/287.
- Block-labeled samples: 287/287.
- Hard-negative samples: 150/287.
- Query-provenance samples: 287/287.
- Validation status: `ready=true`, `invalid=0`.

The paper should describe labels as reviewed annotations derived from workflow evidence and fix-commit context. Do not overstate the annotation process as independent human labeling unless that extra review is added.

## Evaluation Tracks

### Single-Shot Retrieval

The single-shot track evaluates ranked retrieval over the base-commit corpus. Baselines produce ranked files/chunks from a query without interacting with the repository.

Primary metrics:

- Recall@5, Recall@10, Recall@20.
- MRR.
- Gold@8k, measuring whether gold context fits into an 8k-token budget.
- Line/block metrics where span and block labels are available.

### Trajectory Context

The trajectory track evaluates logged agent behavior. Each run records files and line ranges read by the agent and which files remain in final context.

Primary metrics:

- Final file precision, recall, and F1.
- Trajectory file precision, recall, and F1.
- Line and block F1 where read windows are available.
- Redundancy and final-context consistency checks.

The two tracks should be reported separately. Single-shot metrics evaluate ranked retrieval. Trajectory metrics evaluate actual context selection during an agent process.

## Baselines

Main single-shot baselines:

- `lexical`: traditional lexical retrieval.
- `aider-style-repomap`: deterministic vectorless repository-structure baseline using paths, symbols, and references.
- `jina-code-embeddings-0.5b`: small open-source code embedding model.
- `Qwen3-Embedding-4B`: strong open-source embedding baseline.
- `Qwen3-Embedding-8B`: larger Qwen embedding baseline.
- `nomic-embed-code`: open-source code embedding baseline.
- `pplx-embed-v1-4b`: open-source embedding baseline.

Main trajectory baseline:

- `GPT-5.4-mini strict-context`: an audited trajectory run where every final-context file must have appeared in a logged read step.

## Main Single-Shot Results

Overall leaderboard:

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

| Task | Best by MRR | MRR | Key contrast |
| --- | --- | ---: | --- |
| `code2test` | Qwen3-Embedding-4B | 0.3225 | Embedding retrieval is strong when implementation/test semantics align. |
| `comment2context` | jina-code-embeddings-0.5b | 0.3043 | Review comments reward semantic context matching, but no model dominates. |
| `trace2code` | aider-style-repomap | 0.2742 | Structure-aware retrieval beats all embedding models on failure traces. |

Central result narrative:

1. The overall leaderboard is close at the top, so aggregate MRR alone is not the story.
2. Task-level behavior diverges sharply.
3. `trace2code` reverses the embedding-heavy ranking: RepoMap is first and lexical is second by MRR, ahead of all embedding models.
4. This supports the main claim that coding-agent retrieval needs both semantic and repository-structure signals.

## Trajectory Results

Canonical trajectory result:

| Task | Run | Samples | File R | File P | File F1 | Line F1 | Avg reads | Avg final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | GPT-5.4-mini strict-context | 287 | 0.5645 | 0.2236 | 0.3113 | 0.0143 | 3.199 | 3.160 |
| `code2test` | GPT-5.4-mini strict-context | 106 | 0.4575 | 0.1792 | 0.2516 | 0.0042 | 3.170 | 3.142 |
| `comment2context` | GPT-5.4-mini strict-context | 80 | 0.2875 | 0.1333 | 0.1732 | 0.0064 | 3.325 | 3.263 |
| `trace2code` | GPT-5.4-mini strict-context | 101 | 0.8960 | 0.3416 | 0.4835 | 0.0312 | 3.129 | 3.099 |

Same-budget final-context contrast:

| Method | Avg final files | Overall file F1 | `code2test` F1 | `comment2context` F1 | `trace2code` F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical@3 final context | 3.00 | 0.0721 | 0.0255 | 0.0746 | 0.1191 |
| RepoMap@3 final context | 3.00 | 0.1102 | 0.1082 | 0.0633 | 0.1495 |
| RepoMap@4 final context | 4.00 | 0.1198 | 0.1165 | 0.0756 | 0.1583 |
| GPT-5.4-mini strict-context | 3.16 | 0.3113 | 0.2516 | 0.1732 | 0.4835 |

Trajectory interpretation:

- Iterative context selection is not equivalent to taking the first few files from a static ranked list.
- `trace2code` is easier for the agent at file level, but line-level F1 remains low.
- `comment2context` remains the hardest trajectory task because the reviewed file is known, and the target is additional context.
- The strict-context run is auditable: final files are a subset of logged reads for all samples.

## Main Story Arc

1. **Motivation:** end-to-end coding-agent benchmarks hide whether failure came from retrieval, reasoning, editing, or testing.
2. **Problem:** repository context retrieval is its own hard step, especially when queries are workflow signals rather than direct code descriptions.
3. **Benchmark:** Agent Retrieval Bench isolates this step with base-commit corpora, leakage controls, and three task families.
4. **Single-shot result:** no retrieval family dominates; different workflow signals need different retrieval biases.
5. **Trajectory result:** actual agent context selection is a process-level problem, not just top-k ranking.
6. **Implication:** future coding-agent retrieval systems should combine embeddings, repository structure, path/symbol signals, and iterative search policies.

## Paper Tables and Figures

Minimum tables:

1. Dataset composition and corpus scale.
2. Task definitions with examples.
3. Overall single-shot leaderboard.
4. Task-level winners.
5. Same-budget final-context trajectory contrast.
6. Quality/audit gates.

Useful figures:

- Figure 1: Coding-agent workflow, with retrieval before reasoning and patch generation.
- Figure 2: Three task diagrams: PR intent to tests, review comment to additional context, failure trace to root-cause source.
- Figure 3: Task-level MRR bars showing different winners.
- Figure 4: Static top-k final context versus strict trajectory final context.
- Figure 5: Error taxonomy: source-to-test mapping misses, review additional-context misses, trace file-hit/span-miss cases.

## Error Analysis Narrative

Use these as paper-facing failure modes:

- `code2test`: source-to-test mapping misses. Models read plausible implementation files or nearby tests but miss the exact regression/integration test.
- `comment2context`: support-near misses and hard additional-context misses. The model may read useful supporting files but miss the exact target, or fail when the review comment points to cross-module policy/config/test context.
- `trace2code`: file-level hits with weak span alignment. The model often finds the root-cause file but does not read the labeled region precisely.

## Claim-Evidence Map

| Claim | Evidence |
| --- | --- |
| ARB isolates upstream repository retrieval | Base-commit corpus, workflow queries, file-level gold labels |
| The benchmark is not semantic code search | Queries come from PR intent, review comments, and failure traces |
| Different tasks need different retrieval biases | Task winners differ across Qwen4B, Jina, and RepoMap |
| Structure-aware retrieval matters | RepoMap leads `trace2code` by MRR and R@20 |
| Embeddings still matter | Qwen4B leads overall MRR and `code2test`; Jina leads `comment2context` MRR |
| Trajectory evaluation adds a different lens | GPT-5.4-mini strict-context beats same-budget static top-k final context |
| Span-level context remains hard | Trajectory line F1 is low even when file-level recall is high |

## Limitations

- The benchmark has 287 reviewed samples; it is diagnostic rather than web-scale.
- Gold labels approximate files an agent should read; they are not full human reasoning traces.
- The benchmark is primarily file-level, even though span and block labels are available.
- The trajectory track currently has one canonical model run.
- Some ecosystems and languages remain underrepresented.
- Closed-source paid models are not central to the core claim.
- The benchmark isolates retrieval and context selection; it does not measure final patch correctness.

## Work Needed Before Submission

Minimum for an arXiv or workshop release:

1. Refresh the public-facing paper text so it uses this unified story instead of internal version names.
2. Refresh the final canonical release bundle so benchmark, eval artifacts, reports, and trajectory artifacts match the paper tables.
3. Run a secret/artifact scan before public release.
4. Add bootstrap confidence intervals or paired tests for close leaderboard differences.
5. Convert the dataset snapshot and leaderboard into final paper tables.

Higher-value additions for a stronger conference submission:

1. Add a simple hybrid baseline, such as embedding plus RepoMap or embedding plus lexical score fusion.
2. Add Recall@k curves or gold-rank distribution plots.
3. Add per-language or per-repository breakdowns, especially for `trace2code`.
4. Run a second trajectory model to show the trajectory finding is not single-model-specific.
5. Add one qualitative example per task in the main paper and move longer examples to an appendix.

## Recommended Paper Structure

1. Introduction
2. Related Work
3. Problem Definition
4. Benchmark Tasks
5. Dataset Construction and Quality Controls
6. Single-Shot Retrieval Evaluation
7. Trajectory Context Evaluation
8. Error Analysis
9. Reproducibility
10. Limitations and Future Work

## Writing Stance

Keep the paper narrow and defensible:

> We introduce a diagnostic benchmark for repository context retrieval in coding-agent workflows. The benchmark shows that different workflow signals require different retrieval biases, and that semantic embeddings alone are insufficient, especially for failure-trace root-cause retrieval.

Avoid overclaiming:

- Do not claim this is a complete end-to-end coding-agent benchmark.
- Do not claim one model is generally best for agent retrieval.
- Do not describe internal version history as the scientific contribution.
- Do not call annotations independent human labels unless that review is actually added.
- Do not make paid or closed-source models central to the conclusion.
