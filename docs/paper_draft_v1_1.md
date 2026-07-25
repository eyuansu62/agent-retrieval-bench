# Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents

Draft status: V1.1 paper draft. This is a continuous paper-facing draft assembled from the current release artifacts. It is not yet camera-ready; citation placement, figure polish, and runtime measurements remain open.

Primary sources:

- `docs/paper_outline.md`
- `docs/paper_tables_v1_1.md`
- `docs/paper_error_analysis_v1_1.md`
- `docs/paper_related_work_v1_1.md`
- `docs/paper_references_v1_1.bib`
- `data/reports/v1_1/model_leaderboard.json`
- `data/reports/v1_1/rank_analysis.json`
- `data/reports/v1_1/candidate_filter_ablation.json`
- `data/reports/v1_1/readiness.json`
- `data/reports/v1_1/release_report.json`

## Abstract

Modern coding agents are usually evaluated by whether they eventually produce a correct patch, but patch generation failures often begin earlier: the agent never finds the repository files it needs to read. We introduce Agent Retrieval Bench, a file-level code retrieval benchmark for this upstream context-finding problem. Each sample is built from a real coding workflow signal and evaluated against the repository at the base commit, before the fix is applied. The benchmark covers three agentic retrieval tasks: `code2test`, where implementation or PR intent must retrieve related tests; `comment2context`, where a review comment and given file must retrieve additional contextual files; and `trace2code`, where reproduced failure output must retrieve root-cause source files rather than merely visible test frames.

The V1.1 release contains 287 manually curated samples across 25 repositories, preserving the frozen V1 `code2test` track while expanding the weaker `comment2context` and `trace2code` tracks with 62 audited additions. V1.1 is evaluated against 261 base-commit corpora with 7.0M chunks and 345K files. We report lexical, vectorless RepoMap, and open-source embedding baselines. Results show that no retrieval family dominates: Qwen3-Embedding-4B is strongest overall by MRR and on `code2test`, Jina is strongest on `comment2context`, while the vectorless RepoMap baseline is strongest on `trace2code`. These findings suggest that agent retrieval needs hybrid methods combining semantic vectors with repository structure, path/symbol signals, and task-aware ranking.

## 1. Introduction

Coding agents are often judged at the end of the workflow: did the generated patch pass tests, resolve the issue, or satisfy a benchmark oracle? This end-to-end measurement is useful, but it hides an upstream failure mode. Before an agent can reason about a change, it must find the files that matter. If retrieval fails, the downstream model may never inspect the implementation, tests, or configuration that would make the patch possible.

Repository context retrieval is not the same as generic semantic code search. In a real coding workflow, the query signal is often indirect. A pull request may describe an implementation change but require retrieving tests. A review comment may refer to a file already visible to the agent, while the missing evidence lives in another module. A failure trace may mention a test file or stack frame, while the root cause is an implementation file that does not look semantically similar to the error text. These settings require source/test reasoning, repository structure, symbols, paths, and task-specific priors in addition to semantic matching.

Agent Retrieval Bench (ARB) isolates this upstream problem. ARB is a file-level benchmark: given a workflow-derived query and a repository corpus at the base commit, a retriever must rank files that an agent would need to read. The benchmark is deliberately not patch-level. It does not ask whether an agent can synthesize the final edit. Instead, it measures whether the agent can find the right context before editing begins.

The V1.1 release focuses on three workflow-grounded retrieval tasks. `code2test` asks for related tests from PR or implementation-change signals. `comment2context` asks for additional context files beyond the reviewed file. `trace2code` asks for root-cause source files from reproduced failure output, while treating visible tests as auxiliary context rather than main gold. Each sample is evaluated against a corpus built from the repository state before the resolving patch, preventing fixed-code leakage.

V1.1 is a targeted expansion of the frozen V1 release. It preserves the V1 `code2test` track and expands the more diagnostic `comment2context` and `trace2code` tracks with audited samples. The release includes readiness checks, corpus manifests, model details, leaderboards, and checksums, so results can be reproduced and inspected.

This paper makes six contributions:

1. It defines agentic repository context retrieval as a file-level benchmark problem distinct from generic code search and end-to-end patch generation.
2. It introduces three workflow-grounded tasks: `code2test`, `comment2context`, and `trace2code`.
3. It releases a manually audited V1.1 benchmark with 287 samples across 25 repositories and 261 base-commit corpora.
4. It applies leakage controls that reject direct gold path hints, final-patch contamination, missing corpus gold, and shortcut labels.
5. It compares lexical, vectorless RepoMap, and open-source embedding baselines, showing strong task-dependent behavior.
6. It provides reproducible artifacts: benchmark samples, corpus chunks, baseline details, leaderboards, reports, and release checksums.

## 2. Related Work

**Code search and code intelligence.** CodeSearchNet [@husain2019codesearchnet] and CodeXGLUE [@lu2021codexglue] established influential benchmarks for semantic code search and broader code-intelligence tasks. These benchmarks evaluate representation learning and matching between natural language and code, often at the function or snippet level. ARB differs in both retrieval unit and query distribution: it evaluates file-level repository context retrieval from coding workflow signals, where the relevant file may be a test, a cross-module context file, or a root-cause source file that is not semantically close to the query.

**Repository-level code context.** RepoBench evaluates repository-level code auto-completion and highlights the importance of cross-file context [@liu2023repobench]. ARB is complementary: instead of completing code at a known location, it asks which repository files an agent should inspect before editing. This shifts the problem from using context to finding context.

**End-to-end coding-agent evaluation.** SWE-bench [@jimenez2023swebench] and SWE-agent [@yang2024sweagent] evaluate realistic software engineering agents on full issue-resolution workflows. ARB targets a narrower but important intermediate stage. End-to-end patch success can fail because of retrieval, reasoning, editing, or validation. ARB isolates retrieval, making it possible to diagnose whether the agent found the necessary files before patch generation.

**Bug and fault localization.** Bug-localization benchmarks and datasets such as Defects4J study mapping bug reports or failures to source locations [@just2014defects4j], and IR-based bug localization work studies source-file ranking from textual bug evidence [@akbar2020irbuglocalization]. ARB's `trace2code` task is adjacent, but it is framed around agent context retrieval from reproduced failure output. The gold files are root-cause source files needed for editing; visible tests and stack frames are evidence rather than automatically counted targets.

**RAG and repository maps.** Retrieval-augmented generation motivates retrieving evidence before generation [@lewis2020rag], and coding agents apply this idea to repository context. Practical systems such as Aider's repo map use structure and symbols rather than only vector similarity [@aiderrepomap]. ARB provides an evaluation target for this setting and shows that semantic embeddings and structure-aware retrieval are complementary.

Full citation notes are maintained in `docs/paper_related_work_v1_1.md`.

## 3. Problem Definition

An ARB sample consists of a repository, a base commit, a query, and a set of gold files. The candidate corpus is the repository at the base commit. A retriever receives the query and ranks candidate files. Metrics are computed at the file level.

This design follows the way coding agents consume context. A model typically needs readable source files, tests, configuration files, or related modules. Chunk-level retrieval is useful internally for scoring, but the benchmark's primary question is whether the correct file appears in the ranked file list.

### Query

The query is derived from a real coding workflow signal:

- a PR title/body or implementation-change summary,
- a review comment plus the reviewed file,
- or a reproduced failure command and failure excerpt.

Queries exclude final patches, raw fix diffs, direct gold paths, and direct gold basenames. The goal is to evaluate context discovery, not answer leakage.

### Corpus

The corpus is built from the repository at the base commit. This is the state before the resolving fix. Candidate files are chunked for indexing and scoring, but ranked output is deduplicated to file paths.

### Gold

Gold files are files an agent needs to read for the task:

- For `code2test`, gold files are related tests.
- For `comment2context`, gold files are additional context files; the reviewed file is recorded as given context and is not counted as main gold.
- For `trace2code`, gold files are root-cause source files; tests can be supporting evidence but are not main gold.

### Metrics

The main metrics are Recall@5, Recall@10, Recall@20, MRR, and `gold_coverage@8k`. Recall@k measures the fraction of gold files retrieved in the top k unique files. MRR uses the first ranked gold file. `gold_coverage@8k` estimates whether gold appears within an 8k-character ranked context budget.

We also report first-gold depth analysis. `Any@k` is the fraction of samples where at least one gold file appears in the top k. This differs from Recall@k for multi-gold samples and helps separate early precision from broader top-k coverage.

## 4. Benchmark Tasks

### 4.1 `code2test`

`code2test` simulates a coding agent receiving a PR or implementation-change signal and needing to identify relevant tests. The query may mention implementation files or behavior, but it should not directly name the test file. The gold files are manually confirmed related tests.

This task captures a common retrieval need: before editing or validating a change, the agent needs to find the tests that define expected behavior. The challenge is that the source file and the test file may use different names, reside in different directories, or be connected through project-specific conventions.

### 4.2 `comment2context`

`comment2context` simulates a code review setting. The agent sees a review comment and the reviewed file. The benchmark asks for additional files needed to understand or satisfy the comment. The reviewed file is treated as given context, not as gold.

This design avoids a common shortcut. A retriever can often rank the commented file highly because it appears in the query. That is not useful if the agent already has that file. The task instead measures whether retrieval can find cross-file implementation, test, or configuration context.

### 4.3 `trace2code`

`trace2code` simulates a reproduced failure. The query contains the command and failure excerpt. The gold files are root-cause source files. Tests and failure frames can be supporting context, but they are not counted as main gold unless they are the audited root cause.

This is the most diagnostic task in V1.1. Failure text often mentions tests, stack frames, symbols, or package paths. The implementation file that must change may be nearby in the repository graph but not semantically similar to the failure excerpt. This makes `trace2code` a strong test of structure-aware retrieval.

## 5. Dataset Construction

V1.1 is a targeted expansion of V1. It preserves the frozen V1 `code2test` track and expands `comment2context` and `trace2code`, the tracks where V1 showed the clearest need for stronger coverage.

| Track | Query Signal | Gold Target | V1 Samples | V1.1 Samples | V1.1 Additions |
| --- | --- | --- | ---: | ---: | ---: |
| `code2test` | PR or implementation-change intent | Related test files | 106 | 106 | 0 |
| `comment2context` | Review comment plus reviewed file | Additional context files beyond the reviewed file | 51 | 80 | 29 |
| `trace2code` | Reproduced failure log or trace | Root-cause source files | 68 | 101 | 33 |
| Total | Mixed coding-agent workflow signals | Files the agent needs to read | 225 | 287 | 62 |

The release contains 287 samples across 25 repositories. The canonical corpus manifest contains 261 repo/base rows across 29 repositories. The full corpus has 345,776 files and 7,016,525 chunks. The V1.1 evaluation run touches 218 repo/base corpora and 6,210,965 chunks.

The corpus scale is intentionally much larger than the sample count. ARB evaluates whether a small set of audited workflow signals can retrieve files from large real repository snapshots. This makes the benchmark diagnostic rather than web-scale: each sample is expensive to validate, but each query searches a large, realistic candidate universe.

Exact text deduplication over sampled corpora finds 1,118,431 unique embedding texts and 5,092,534 duplicate chunk texts, an 81.99% duplicate fraction. This motivates shared corpus-embedding caches for practical evaluation because nearby base commits reuse many identical chunks.

## 6. Quality and Leakage Controls

ARB's main risk is shortcut leakage. If a query contains the gold path, the gold basename, the final fix diff, or content from the resolved patch, the retrieval problem collapses into string matching. V1.1 therefore includes explicit readiness and release gates.

The release checks enforce:

- frozen V1 IDs are preserved;
- `code2test` count is unchanged in V1.1;
- new samples are limited to `comment2context` and `trace2code`;
- new samples have manual audit evidence;
- gold files exist in the base-commit corpus;
- queries avoid direct gold path or basename hints;
- path-role overlap is rejected;
- new `comment2context` samples have `given_files`;
- new `comment2context` samples avoid same-directory shortcut gold;
- new `trace2code` gold files are not test-only gold;
- new `trace2code` additions include non-Go repositories and `.py` / `.rs` gold extensions.

The latest readiness report has `ready=true`, the release report has `status=ready`, and the completion audit passes 16/16 requirements. These checks do not prove that every label is perfect, but they make the benchmark defensible as a leakage-controlled retrieval diagnostic.

## 7. Baselines

We evaluate three retrieval families.

Lexical retrieval is a traditional token-based baseline. It ranks candidate chunks and deduplicates the output to files. It is simple, deterministic, and useful for measuring how much of the benchmark can be solved by direct textual overlap.

RepoMap is a vectorless repository-structure baseline inspired by coding-agent repo maps. It ranks files using path, symbol, reference, and repository graph signals. This baseline is designed to test whether structure-aware retrieval can recover files that are not semantically similar to the query text.

Embedding baselines use open-source code/text embedding models. The V1.1 leaderboard includes Jina code embeddings, Qwen3-Embedding-4B, Qwen3-Embedding-8B, pplx-embed-v1-4b, and nomic-embed-code. Hosted paid models are intentionally not part of the main V1.1 claim.

All official leaderboard rows use `candidate_filter=all_files` and evaluate all 287 samples.

## 8. Main Results

### 8.1 Overall Leaderboard

Qwen3-Embedding-4B has the best overall MRR, but the margin is narrow and the aggregate ranking hides strong task-level differences.

| Model | Family | R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | embedding | 0.2852 | 0.4102 | 0.6143 | 0.2296 | 0.2516 |
| Qwen3-Embedding-8B | embedding | 0.3293 | 0.5348 | 0.7070 | 0.2272 | 0.1533 |
| pplx-embed-v1-4b | embedding | 0.2742 | 0.4173 | 0.5929 | 0.2143 | 0.1359 |
| aider-style-repomap | vectorless repo map | 0.3131 | 0.4688 | 0.6419 | 0.2125 | 0.0627 |
| jina-code-embeddings-0.5b | embedding | 0.2201 | 0.3066 | 0.4491 | 0.1813 | 0.1568 |
| nomic-embed-code | embedding | 0.2569 | 0.3448 | 0.5145 | 0.1810 | 0.0832 |
| lexical | lexical | 0.1922 | 0.3165 | 0.4750 | 0.1392 | 0.0720 |

The top four rows are close: Qwen3-Embedding-4B (`0.2296` MRR), Qwen3-Embedding-8B (`0.2272`), pplx-embed-v1-4b (`0.2143`), and RepoMap (`0.2125`). The narrow aggregate margin is one reason we emphasize task-level and rank-depth analysis.

### 8.2 Task Winners

The best model changes by task.

| Task | Best MRR Model | MRR | Best R@20 Model | R@20 | Interpretation |
| --- | --- | ---: | --- | ---: | --- |
| overall | Qwen3-Embedding-4B | 0.2296 | Qwen3-Embedding-8B | 0.7070 | Strong embeddings lead aggregate performance. |
| `code2test` | Qwen3-Embedding-4B | 0.3225 | Qwen3-Embedding-4B | 0.7230 | Source-to-test retrieval benefits from broad semantic/code embeddings. |
| `comment2context` | jina-code-embeddings-0.5b | 0.3043 | Qwen3-Embedding-8B | 0.6562 | Review-context retrieval is competitive across embedding models. |
| `trace2code` | aider-style-repomap | 0.2745 | aider-style-repomap | 0.8465 | Failure trace retrieval strongly benefits from path/symbol/repo-graph signals. |

These task winners support the central claim: different agentic retrieval signals require different inductive biases. A benchmark that reports only a single aggregate score would obscure this.

### 8.3 `trace2code` Reverses the Aggregate Picture

`trace2code` is the strongest evidence that semantic embeddings alone are insufficient. RepoMap leads this task by MRR and Recall@20. Lexical retrieval is also stronger than most embedding baselines on MRR. Qwen3-Embedding-4B, despite leading overall MRR, has weak `trace2code` MRR (`0.0827`).

| Model | `trace2code` R@5 | R@10 | R@20 | MRR | Gold@8k |
| --- | ---: | ---: | ---: | ---: | ---: |
| aider-style-repomap | 0.4736 | 0.7145 | 0.8465 | 0.2745 | 0.0693 |
| lexical | 0.3432 | 0.4818 | 0.6964 | 0.2075 | 0.1287 |
| Qwen3-Embedding-8B | 0.2442 | 0.5281 | 0.7970 | 0.1654 | 0.0759 |
| pplx-embed-v1-4b | 0.1601 | 0.2855 | 0.5380 | 0.1372 | 0.0545 |
| nomic-embed-code | 0.1584 | 0.1733 | 0.3581 | 0.0871 | 0.0297 |
| Qwen3-Embedding-4B | 0.0743 | 0.1815 | 0.5050 | 0.0827 | 0.0495 |
| jina-code-embeddings-0.5b | 0.0693 | 0.1188 | 0.2607 | 0.0607 | 0.0297 |

Failure traces often expose tests and local symbols, not the implementation file that should change. Structure-aware ranking can connect these visible signals to nearby source files.

## 9. Rank and Error Analysis

Aggregate MRR measures early precision, but it does not fully describe whether a model can keep a gold file within a later reranking or context-selection depth. First-gold depth analysis shows this tradeoff.

| Model | Overall Any@5 | Overall Any@20 | Overall Median Hit Rank | `trace2code` Any@20 | `trace2code` Median Hit Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 0.3589 | 0.6864 | 11 | 0.5347 | 20 |
| Qwen3-Embedding-8B | 0.4077 | 0.7944 | 7 | 0.8614 | 9 |
| pplx-embed-v1-4b | 0.3310 | 0.6794 | 10 | 0.5941 | 18 |
| aider-style-repomap | 0.3728 | 0.7247 | 9 | 0.9109 | 5 |
| jina-code-embeddings-0.5b | 0.2683 | 0.5226 | 19 | 0.2871 | 29 |
| nomic-embed-code | 0.3101 | 0.5784 | 15 | 0.3861 | 25 |
| lexical | 0.2474 | 0.5470 | 17 | 0.7525 | 8 |

Qwen3-Embedding-4B has the highest overall MRR, but Qwen3-Embedding-8B has broader top-20 coverage (`0.7944` vs. `0.6864`). This suggests that the 4B model places its successful hits earlier, while the 8B model retrieves at least one gold file for more samples by depth 20.

The cross-model miss analysis shows that most samples are solved by at least one baseline at depth 20, but not always by the same family.

| Task | Samples | Any Model Hit@20 | All Models Miss@20 |
| --- | ---: | ---: | ---: |
| overall | 287 | 265 | 22 |
| `code2test` | 106 | 96 | 10 |
| `comment2context` | 80 | 71 | 9 |
| `trace2code` | 101 | 98 | 3 |

Only 22 of 287 samples are missed by every reported baseline at top 20. This supports the interpretation that ARB exposes complementary retrieval behavior, not just uniformly impossible examples.

### 9.1 `trace2code` Error Patterns

Representative `trace2code` examples show both structure wins and embedding wins. In Gin samples such as `30fb9ae30b972db986c4365d` with gold `response_writer.go` and `9e63d7a07d201c84d7d18546` with gold `errors.go`, RepoMap retrieves the gold source file in the top 20 while Qwen3-Embedding-4B and Qwen3-Embedding-8B miss it. The failure excerpts mention tests and compile failures, so semantic models often rank visible test files or generic repository metadata above the implementation file. RepoMap can use structural proximity between tests and source files to recover the root-cause file.

The reverse also occurs. In `d4149b3f568dfe275250b62e`, a Caddy authentication failure has gold `modules/caddyhttp/caddyauth/caddyauth.go`; pplx-embed-v1-4b ranks it first while RepoMap misses it at depth 20. Similar embedding wins appear for Caddy HTTP Caddyfile files and several Gin source files. These examples prevent a simplistic "RepoMap beats embeddings" conclusion. The stronger conclusion is that semantic and structural signals are complementary.

### 9.2 `comment2context` Error Patterns

`comment2context` exposes the given-file trap. The reviewed file is already available to the agent, so returning it is not sufficient. In sample `bf47308900fafdff5ca96ecd`, the query asks about seed-printing behavior from `hypothesis-python/tests/cover/test_seed_printing.py`, but the gold context is in `hypothesis-python/src/hypothesis/core.py` and `hypothesis-python/src/hypothesis/internal/conjecture/engine.py`. RepoMap ranks the given test file first and misses the gold files at depth 20.

Other misses require configuration or cross-module context. Sample `449280b38ce700056be47fe` has a review comment in `lib/rules/no-param-reassign.js`, but the gold file is `eslint.config.js`. Sample `413524fdc2e0151ec121653c` involves Ruff project database behavior and requires `crates/ty_server/tests/e2e/configuration.rs`. These cases support the task design: the benchmark should measure additional context retrieval, not reviewed-file recovery.

### 9.3 `code2test` Error Patterns

`code2test` failures often require mapping implementation changes to project-specific test organization. Sample `6338b75970bc7489e13952b4` from etcd includes nine implementation files and two existing test files in the query summary, but all seven baselines miss the gold tests `tests/e2e/ctl_v3_test.go` and `tests/framework/e2e/cluster.go` at depth 20. Sample `210933147c231644eeebdaae` from diffusers changes `src/diffusers/utils/torch_utils.py`, but the gold is `tests/others/test_utils.py`; models tend to rank the changed utility or infrastructure files above the relevant test.

These examples show that source-to-test retrieval is not only semantic similarity. It requires knowledge of how a project organizes its tests, integration suites, fixtures, and validation utilities.

## 10. Candidate Filter Ablation

The official leaderboard uses `candidate_filter=all_files`. We also run a diagnostic candidate-filter ablation for vectorless baselines.

| Model | Candidate Filter | Overall MRR | `code2test` MRR | `comment2context` MRR | `trace2code` MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| lexical | `all_files` | 0.1392 | 0.0663 | 0.1495 | 0.2075 |
| lexical | `code_only` | 0.1620 | 0.0789 | 0.1613 | 0.2498 |
| lexical | `tests_only` | 0.1562 | 0.2771 | 0.1934 | 0.0000 |
| aider-style-repomap | `all_files` | 0.2125 | 0.1962 | 0.1558 | 0.2745 |
| aider-style-repomap | `code_only` | 0.2283 | 0.2284 | 0.1590 | 0.2829 |
| aider-style-repomap | `tests_only` | 0.1984 | 0.3875 | 0.1984 | 0.0000 |

Restricting candidates to `tests_only` sharply improves `code2test`, but it collapses `trace2code` to zero because source-file golds are removed. Conversely, `code_only` improves vectorless overall MRR, especially by reducing distracting non-source candidates. These results are useful diagnostically, but they should not replace the official `all_files` leaderboard. Task-specific filters can act like oracle information about the target file type.

## 11. Reproducibility

The V1.1 release includes:

- benchmark JSONL files for each task and a combined sample file;
- corpus chunks and corpus manifests;
- eval summaries and per-sample details;
- model leaderboards;
- readiness and release reports;
- candidate-filter and rank-analysis reports;
- checksums and downloadable release bundles.

The CLI supports downloading the benchmark release, running lexical, RepoMap, and embedding baselines, generating model leaderboards, and generating rank/error analysis reports.

Runtime remains an important practical issue. The corpus contains millions of chunks, and embedding evaluation is expensive. V1.1 includes shared text cache support and details-resume support to make evaluation recoverable. Runtime should be reported with device, batch size, query batch size, cache use, wall time, and cache size. This table is not final in the current draft because wall-clock runs still need to be recorded consistently on the GPU machine.

## 12. Limitations

V1.1 is diagnostic, not web-scale. It contains 287 manually audited samples. This is enough to expose strong task-level differences, but it is not a large statistical benchmark covering all languages and ecosystems.

ARB is file-level. It does not evaluate span-level retrieval, exact edit localization, or patch generation. This is a deliberate isolation of the retrieval stage, but it means the benchmark does not measure end-to-end agent success.

The corpus is fixed at the base commit. ARB does not evaluate dynamic tool use, iterative retrieval, or agent-driven exploration over multiple turns.

Gold labels approximate files that need to be read. They are manually audited, but they are not full human reasoning traces. Some tasks may have alternative useful files that are not labeled as gold.

The current V1.1 model set emphasizes open-source baselines. Closed-source paid embedding services are deferred and should not be part of the core claim.

## 13. Future Work

The most direct technical next step is hybrid retrieval. V1.1 shows that embeddings, lexical matching, and RepoMap have complementary strengths. A future system should combine semantic vectors, lexical/path signals, repository graphs, source-test relations, and task-aware reranking.

Another direction is a larger manually audited release with more languages, repositories, and workflow signals. Potential new tasks include issue-to-code retrieval, bug-report-to-root-cause retrieval, migration-to-affected-files retrieval, and API-usage-to-implementation retrieval.

Runtime and cache analysis should also be completed. The high duplicate fraction in sampled corpora suggests that shared corpus-embedding caches are essential for practical evaluation. A future paper version should include wall-clock evidence for cache hit rates, storage size, and recovery behavior after OOM or interrupted runs.

Finally, ARB should be integrated into end-to-end agent loops. The current benchmark isolates retrieval. A downstream study could test whether better ARB retrieval improves patch success when plugged into a coding agent.

## 14. Conclusion

Agent Retrieval Bench evaluates an upstream failure mode in coding agents: finding the repository files needed before patch generation begins. V1.1 shows that agentic retrieval is not solved by a single retrieval family. Strong embeddings lead aggregate MRR and perform well on `code2test` and `comment2context`, but vectorless structure-aware retrieval is strongest on `trace2code`. Rank-depth and error analysis show that models have complementary strengths and that official candidate-space choices matter.

The practical implication is clear: coding-agent retrieval should be hybrid and task-aware. Semantic similarity, lexical/path matching, repository structure, and source-test relations each solve different parts of the benchmark. ARB provides a controlled way to measure those differences before they are hidden inside end-to-end patch generation.

## Open TODOs Before Submission

- Convert current BibTeX draft to venue citation format.
- Convert Markdown tables to LaTeX or venue format.
- Add a figure for the three task definitions.
- Add measured runtime/cache table from GPU runs.
- Decide whether to include a small hybrid retrieval pilot or leave it as future work.
