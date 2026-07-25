# Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents

## Abstract

Modern coding agents are usually evaluated by whether they eventually produce a correct patch, but many failures begin earlier: the agent never finds the repository files it needs to read. We introduce Agent Retrieval Bench, a file-level repository context retrieval benchmark for this upstream context-finding problem. Each sample is built from a real coding workflow signal and evaluated against the repository at the base commit, before the resolving patch is applied. The benchmark covers three agentic retrieval tasks: `code2test`, where implementation intent must retrieve related tests; `comment2context`, where a review comment and given file must retrieve additional contextual files; and `trace2code`, where reproduced failure output must retrieve root-cause source files rather than merely visible test frames.

Agent Retrieval Bench contains 287 reviewed samples across 25 repositories, with corpus artifacts spanning 261 repository/base-commit rows, 345,776 files, and 7,016,525 chunks. The benchmark includes file-level gold labels plus span and block annotations for every sample, enabling both retrieval ranking and context granularity analysis. We evaluate lexical retrieval, a vectorless RepoMap-style structural retriever, and five open-source embedding models. Results show that no retrieval family dominates: Qwen3-Embedding-4B is strongest overall by MRR and on `code2test`, Jina is strongest on `comment2context` by MRR, and the vectorless RepoMap baseline is strongest on `trace2code`. A trajectory study with GPT-5.4-mini further shows that iterative context selection can outperform same-budget static top-k retrieval, but still struggles with span-level precision. These findings suggest that coding-agent retrieval needs hybrid methods combining semantic vectors with repository structure, path/symbol signals, and task-aware search.

## 1. Introduction

Coding agents are increasingly evaluated as end-to-end systems: given an issue, a failing test, or a natural-language instruction, the agent must inspect a repository, edit code, and produce a patch that passes validation. End-to-end evaluation is necessary, but it hides a basic question: did the agent ever find the files it needed to read? If an agent fails to solve a task, the failure may come from bad retrieval, bad reasoning, an incorrect edit, missing test execution, or tool-use problems. The final patch outcome alone does not tell us which stage failed.

This paper isolates the upstream repository context retrieval problem. Before an agent can reason about a patch, it must construct a working context from a large repository. In realistic software workflows, the retrieval query is often not a direct natural-language description of a code snippet. A pull request may describe behavior while the relevant target is an integration test. A review comment may point at one file, while the real constraint is in another module. A failure trace may mention tests and stack frames, while the root cause is a source file not lexically obvious from the error message. These are not simply semantic code search tasks; they are repository context retrieval tasks.

We introduce Agent Retrieval Bench, a benchmark designed to measure whether retrievers can find files a coding agent should read. Each sample is grounded in a coding workflow signal and evaluated against the repository at the base commit, before the resolving patch is applied. This base-commit setup is important: the fixed code is not indexed, so retrieval cannot exploit the final patch. The benchmark evaluates three task families:

- `code2test`: from implementation intent or pull-request context to related tests.
- `comment2context`: from a review comment plus the reviewed file to additional context files.
- `trace2code`: from reproduced failure output to root-cause source files.

The benchmark is intentionally file-level. Agents typically read files, file slices, or symbol regions as context. The corpus is chunked for ranking and efficient retrieval, but the primary question is whether the resulting context includes the right repository files. To support more granular analysis, Agent Retrieval Bench also provides span and block annotations for every sample, allowing line/block coverage analysis for systems that record read windows.

Our experiments compare lexical retrieval, a vectorless RepoMap-style structural retriever, and five open-source embedding models. The results show a consistent pattern: aggregate metrics are insufficient because task-level leaders differ. Qwen3-Embedding-4B has the best overall MRR and leads `code2test`; Jina leads `comment2context` by MRR; and RepoMap leads `trace2code`, where all embedding models are weaker. This supports the central claim that agent retrieval requires more than one retrieval bias.

We also report a trajectory context track. Instead of scoring a ranked list, this track evaluates which files and spans actually enter an agent's read and final context. A GPT-5.4-mini strict-context run substantially outperforms same-budget static top-k context selection, but its low line-level F1 shows that file discovery and span discovery remain different problems.

This paper makes the following contributions:

1. We define repository context retrieval as an upstream evaluation target for coding agents.
2. We release Agent Retrieval Bench, a reviewed 287-sample benchmark across `code2test`, `comment2context`, and `trace2code`.
3. We provide file-level labels plus span and block annotations for all samples, with 150 samples carrying hard negatives.
4. We evaluate lexical, vectorless structure-aware, and embedding retrievers and show that no retrieval family dominates across workflow signals.
5. We add a trajectory context evaluation track that measures logged read/final context rather than only static ranked lists.

## 2. Related Work

**Code search and code intelligence.** CodeSearchNet [@husain2019codesearchnet] and CodeXGLUE [@lu2021codexglue] established influential benchmarks for semantic code search and broader code-intelligence tasks. These benchmarks evaluate representation learning and matching between natural language and code, often at the function or snippet level. Agent Retrieval Bench differs in both retrieval unit and query distribution. It evaluates file-level repository context retrieval from coding workflow signals, where the relevant file may be a test, a cross-module context file, or a root-cause source file that is not semantically close to the query.

**Repository-level code context.** RepoBench evaluates repository-level code auto-completion and highlights the importance of cross-file context [@liu2023repobench]. Agent Retrieval Bench is complementary. Rather than completing code at a known location, it asks which repository files an agent should inspect before editing. This shifts the problem from using context to finding context.

**End-to-end coding-agent evaluation.** SWE-bench [@jimenez2023swebench] and SWE-agent [@yang2024sweagent] evaluate realistic software engineering agents on full issue-resolution workflows. These benchmarks measure the final ability to solve software tasks. Agent Retrieval Bench targets a narrower but important intermediate stage. End-to-end patch success can fail because of retrieval, reasoning, editing, or validation. By isolating retrieval, our benchmark makes it possible to diagnose whether the agent found the necessary files before patch generation.

**Bug and fault localization.** Bug-localization benchmarks and datasets such as Defects4J study mapping bug reports or failures to source locations [@just2014defects4j], and IR-based bug localization work studies source-file ranking from textual bug evidence [@akbar2020irbuglocalization]. Our `trace2code` task is adjacent, but it is framed around agent context retrieval from reproduced failure output. The gold files are root-cause source files needed for editing; visible tests and stack frames are evidence rather than automatically counted targets.

**RAG and repository maps.** Retrieval-augmented generation motivates retrieving evidence before generation [@lewis2020rag], and coding agents apply this idea to repository context. Practical systems such as Aider's repo map use structure and symbols rather than only vector similarity [@aiderrepomap]. Agent Retrieval Bench provides an evaluation target for this setting and shows that semantic embeddings and structure-aware retrieval are complementary.

## 3. Problem Definition

We define a repository context retrieval sample as:

```text
(q, R_b, C_b, G)
```

where `q` is a workflow query, `R_b` is a repository at base commit `b`, `C_b` is a candidate corpus extracted from `R_b`, and `G` is a set of gold files an agent should read. The candidate corpus is built before the resolving patch is applied. This prevents the retriever from seeing the fixed code, final test changes, or post-hoc text that would leak the answer.

A retriever returns a ranked list of candidate chunks or files. Chunk-level ranking is allowed because many retrieval systems index chunks, but evaluation is aggregated at the file level. File-level scoring reflects how coding agents usually consume context: an agent must know which files to open before it can inspect the relevant lines.

### 3.1 Tasks

`code2test` evaluates whether a retriever can map implementation intent to related tests. The query may include a pull-request title, a description, or an implementation-change summary. The gold files are tests that should be read to understand or validate the change.

`comment2context` evaluates review-driven context retrieval. The query contains a review comment and the reviewed file. The reviewed file is treated as known context. The target is additional context, such as implementation dependencies, configuration, policy files, serializers, or tests needed to understand the comment.

`trace2code` evaluates failure-signal retrieval. The query contains reproduced failure output, such as test commands, assertion failures, compile errors, panics, or stack traces. The target is root-cause source files. Visible tests and stack frames are useful evidence, but they are not automatically gold unless reviewed as files the agent should edit or inspect.

### 3.2 Metrics

For single-shot retrieval, we report Recall@5, Recall@10, Recall@20, and MRR. We also report Gold@8k, which measures whether gold files appear within a fixed context budget. Since the benchmark includes span and block annotations, systems can also be evaluated on line and block coverage when they expose read windows or selected spans.

For trajectory evaluation, we report final file precision, recall, and F1, along with line-level F1 when read windows are available. The final-context metric asks what files remain in the agent's final context, not merely which files appeared somewhere in a ranked list. This is a different metric family and should not be merged with single-shot leaderboard scores.

## 4. Benchmark Construction

Agent Retrieval Bench contains 287 reviewed retrieval samples across 25 repositories. The corpus contains 261 repository/base-commit rows across 29 repositories, with 345,776 files and 7,016,525 chunks. Every sample has file-level gold labels, span annotations, block annotations, and query provenance. In addition, 150 samples include hard-negative files.

| Task | Samples | Query signal | Gold target |
| --- | ---: | --- | --- |
| `code2test` | 106 | PR intent or implementation-change summary | Related tests |
| `comment2context` | 80 | Review comment plus the reviewed file | Additional context beyond the reviewed file |
| `trace2code` | 101 | Reproduced failure output | Root-cause source files |
| Total | 287 | Mixed coding-agent workflow signals | Files an agent should read |

The benchmark is designed to avoid easy leakage. Queries exclude direct gold paths, gold basenames, raw patches, final diffs, and fixed-state corpus contamination. The base-commit corpus ensures that the candidate set reflects what an agent could have inspected before the fix. Gold labels are reviewed annotations derived from workflow evidence and fix-commit context. We use "reviewed" deliberately: the labels are curated and audited, but they should not be described as independent human annotations unless an additional independent review pass is performed.

### 4.1 Label Granularity

File-level gold labels identify the files an agent should read. Span labels identify relevant line intervals within those files. Block labels map spans onto corpus-backed blocks, usually symbols, when possible. This design allows the benchmark to support both current file-level agent retrieval and future span-aware systems.

Hard negatives are files that are plausible but incorrect. They help distinguish systems that merely retrieve nearby or lexically similar files from systems that retrieve the actual context required by the task.

### 4.2 Public Layout

The canonical release uses a single paper-facing dataset name:

```text
data/benchmark/agent_retrieval_bench/
data/corpus/agent_retrieval_bench/
data/eval/agent_retrieval_bench/
data/reports/agent_retrieval_bench/
```

Historical artifact paths are retained for provenance and compatibility, but the paper presents the dataset as Agent Retrieval Bench rather than as a sequence of internal releases.

## 5. Experimental Setup

We evaluate seven single-shot retrieval baselines on the `all_files` candidate set.

**Lexical retrieval** is a traditional sparse retrieval baseline. It provides a low-cost reference point and measures how much of the benchmark can be solved from direct lexical overlap.

**RepoMap** is a deterministic vectorless baseline inspired by repository maps. It uses repository structure, paths, symbols, and references rather than embedding similarity. This baseline is important because coding agents often use structural context selection rather than pure vector retrieval.

**Embedding models** include Jina code embeddings, Qwen3-Embedding-4B, Qwen3-Embedding-8B, Nomic code embeddings, and pplx-embed-v1-4b. These baselines represent semantic retrieval systems that embed queries and corpus chunks.

All baselines are evaluated on the same sample set and corpus. Rows are sorted by MRR within each task. We report task-level results because the aggregate score hides important differences between workflow signals.

## 6. Single-Shot Retrieval Results

Table 1 shows the overall single-shot leaderboard. Qwen3-Embedding-4B has the highest MRR at 0.2296, closely followed by Qwen3-Embedding-8B at 0.2272, pplx-embed-v1-4b at 0.2143, and RepoMap at 0.2120. Qwen3-Embedding-8B has the highest Recall@20 at 0.7070. The small margin between top aggregate scores suggests that task-level behavior is more informative than a single overall ranking.

| Model | Samples | R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 287 | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 |
| Qwen3-Embedding-8B | 287 | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 |
| pplx-embed-v1-4b | 287 | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 |
| aider-style-repomap | 287 | 0.3033 | 0.4728 | 0.6367 | 0.2120 | 0.0592 |
| jina-code-embeddings-0.5b | 287 | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 |
| nomic-embed-code | 287 | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 |
| lexical | 287 | 0.1922 | 0.3182 | 0.4750 | 0.1402 | 0.0738 |

### 6.1 Task-Level Divergence

The strongest result is that task-level winners differ.

| Task | Best by MRR | MRR | Interpretation |
| --- | --- | ---: | --- |
| `code2test` | Qwen3-Embedding-4B | 0.3225 | Semantic retrieval works well when implementation intent and tests align. |
| `comment2context` | jina-code-embeddings-0.5b | 0.3043 | Review comments reward semantic matching, but the task remains model-sensitive. |
| `trace2code` | aider-style-repomap | 0.2742 | Failure traces reward structure, path, symbol, and source/test signals. |

`code2test` favors embeddings. Qwen3-Embedding-4B reaches 0.3225 MRR and 0.7230 Recall@20. This suggests that implementation intent and related tests often share enough semantic signal for embedding retrieval to work well.

`comment2context` is more mixed. Jina has the highest MRR at 0.3043, while Qwen3-Embedding-8B has the highest Recall@20 at 0.6562. This task often requires interpreting a review comment as a pointer to additional policy, configuration, serializer, or test context. The reviewed file is not the target by itself.

`trace2code` reverses the aggregate embedding-heavy picture. RepoMap leads with 0.2742 MRR and 0.8366 Recall@20. Lexical retrieval is second by MRR at 0.2075, ahead of every embedding model. Qwen3-Embedding-4B, the overall MRR leader, reaches only 0.0827 MRR on `trace2code`. This result is central: failure traces often contain stack names, test names, paths, symbols, and source/test structure that vectorless methods can exploit.

### 6.2 Implications

The results do not show that embeddings are weak. They show that embeddings are incomplete. Embeddings are strong for broad semantic intent and review-context tasks, but they do not consistently capture the repository topology and failure-signal structure needed for trace root-cause retrieval. Conversely, RepoMap is not universally best; it underperforms embeddings on `code2test` and `comment2context`. A practical coding-agent retrieval system should combine semantic, lexical, path, symbol, and graph signals.

## 7. Trajectory Context Evaluation

Single-shot retrieval evaluates ranked lists. Coding agents, however, perform a process: they search, read files, update beliefs, and decide which context to keep. To measure this process, Agent Retrieval Bench includes a trajectory context track. A trajectory record logs which files and line ranges enter the agent context and which files remain in final context.

We report a canonical GPT-5.4-mini strict-context run. "Strict context" means that every final-context file must have appeared in a logged read step. This matters because an answer can mention plausible files that were never actually read. The strict run separates read evidence from suggestions.

| Task | Run | Samples | File R | File P | File F1 | Line F1 | Avg reads | Avg final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | GPT-5.4-mini strict-context | 287 | 0.5645 | 0.2236 | 0.3113 | 0.0143 | 3.199 | 3.160 |
| `code2test` | GPT-5.4-mini strict-context | 106 | 0.4575 | 0.1792 | 0.2516 | 0.0042 | 3.170 | 3.142 |
| `comment2context` | GPT-5.4-mini strict-context | 80 | 0.2875 | 0.1333 | 0.1732 | 0.0064 | 3.325 | 3.263 |
| `trace2code` | GPT-5.4-mini strict-context | 101 | 0.8960 | 0.3416 | 0.4835 | 0.0312 | 3.129 | 3.099 |

The trajectory run has a different profile from single-shot retrieval. `trace2code` is the strongest task at file level, with final file recall of 0.8960 and F1 of 0.4835. `comment2context` remains the hardest task, with final file F1 of 0.1732. This supports the benchmark design: retrieving the reviewed file is not enough; the model must infer additional context from a local review signal.

### 7.1 Same-Budget Final-Context Contrast

A fair comparison for trajectory context is not top-20 recall. The strict-context run keeps only 3.16 final files per sample on average. We therefore compare it to static baselines where the first 3 or 4 ranked files are treated as final context.

| Method | Avg final files | Overall file F1 | `code2test` F1 | `comment2context` F1 | `trace2code` F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical@3 final context | 3.00 | 0.0721 | 0.0255 | 0.0746 | 0.1191 |
| RepoMap@3 final context | 3.00 | 0.1102 | 0.1082 | 0.0633 | 0.1495 |
| RepoMap@4 final context | 4.00 | 0.1198 | 0.1165 | 0.0756 | 0.1583 |
| GPT-5.4-mini strict-context | 3.16 | 0.3113 | 0.2516 | 0.1732 | 0.4835 |

The strict agent run substantially outperforms same-budget static top-k selection. This indicates that iterative context selection is not equivalent to taking the first few files from a static ranked list. RepoMap remains valuable as a candidate generator, especially on `trace2code`, but process-level search and reading can produce a stronger final context under a small file budget.

### 7.2 File-Level Success, Span-Level Weakness

The trajectory run also reveals a limitation: file-level success does not imply span-level precision. Overall line F1 is only 0.0143. On `trace2code`, file recall is high, but line F1 is 0.0312. Many samples include at least one gold file in final context while missing the labeled spans. This suggests that future agent retrieval evaluation should measure not only which file enters context, but also which region enters context.

## 8. Error Analysis

The benchmark exposes different failure modes across tasks.

**Source-to-test mapping misses.** In `code2test`, models often read plausible implementation files or nearby tests while missing the exact regression or integration test. This suggests that source-to-test retrieval requires repository-specific knowledge of test layout, e2e test conventions, and integration boundaries.

**Additional-context misses.** In `comment2context`, failures split into near misses and hard misses. In near misses, the model reads useful supporting files but misses the exact gold file. In hard misses, the review comment is a weak anchor for a cross-module policy, configuration, serializer, or test dependency. This task measures whether a retriever can go beyond the commented file.

**Trace file-hit/span-miss cases.** In `trace2code`, the agent often finds the correct source file but reads too broadly or misses the labeled region. This is encouraging at the file level but weak at the span level. It shows why span and block annotations are useful even when the primary benchmark is file-level.

**Trace source-file misses.** Some `trace2code` failures come from reading plausible files in the failure neighborhood, such as test files or adjacent modules, while missing the root-cause source file. These errors are closely related to bug localization, but the agent context framing makes the target more specific: the goal is to read files that support editing, not merely files named in the trace.

## 9. Reproducibility

The canonical release is hosted as a Hugging Face dataset bundle. It contains the benchmark samples, base-commit corpus chunks, eval summaries and details, reports, and checksums. The public layout is:

```text
data/benchmark/agent_retrieval_bench/
data/corpus/agent_retrieval_bench/
data/eval/agent_retrieval_bench/
data/reports/agent_retrieval_bench/
```

The default downloader verifies the checksum and extracts the canonical bundle:

```bash
arb download-benchmark --local-dir data --force
```

The canonical manifest reports 287 samples, 261 corpus rows, 345,776 files, 7,016,525 chunks, and validation status `ready=true`, `invalid=0`. Historical internal paths are retained for provenance and compatibility, but the paper-facing dataset should be cited as Agent Retrieval Bench.

## 10. Limitations

Agent Retrieval Bench is diagnostic rather than web-scale. It contains 287 reviewed samples, which is enough to expose task-level contrasts but not enough to claim broad coverage of all programming languages, ecosystems, or agent workflows.

Gold labels approximate the files an agent should read. They are reviewed against workflow evidence and fix-commit context, but they are not full human reasoning traces. Some useful context files may be unannotated, especially in `comment2context`, where support-near misses may still be practically helpful.

The benchmark is primarily file-level. Span and block annotations are provided, but many current systems and trajectory logs do not expose precise read windows. The low trajectory line F1 should therefore be interpreted as both a system limitation and a logging/evaluation challenge.

The trajectory track currently reports one canonical model run. It demonstrates that process-level context selection can differ from static top-k retrieval, but more models are needed before making broad claims about agent families.

Finally, Agent Retrieval Bench isolates retrieval and context selection. It does not measure final patch correctness. This is a feature for diagnostic analysis, but it means the benchmark should complement, not replace, end-to-end coding-agent benchmarks.

## 11. Conclusion

Agent Retrieval Bench evaluates a missing piece of coding-agent performance: whether an agent can find the repository context it needs before editing. The benchmark uses real workflow signals, base-commit corpora, reviewed file/span/block labels, and three task families that stress different retrieval biases.

The results show that no retrieval family dominates. Embeddings are strong overall and on `code2test`; Jina leads `comment2context` by MRR; and RepoMap leads `trace2code`, where structure-aware retrieval beats all embedding models. The trajectory track further shows that iterative agent context selection can outperform same-budget static top-k retrieval, while still struggling to enter the right spans.

These findings support a practical direction for coding-agent retrieval: hybrid systems that combine semantic embeddings, lexical matching, repository topology, path/symbol signals, and task-aware iterative search.

## References

This draft uses citation keys from `docs/paper_references.bib`.
