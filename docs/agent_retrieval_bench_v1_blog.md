# Agent Retrieval Bench V1: A Code Retrieval Benchmark for Coding Agents

> This post introduces the motivation, task design, data construction process, baseline results, and V1.1 roadmap for Agent Retrieval Bench V1.

## Background: coding agents fail before they write patches

Over the last two years, most evaluations for code models and coding agents have focused on whether the system can eventually produce a correct patch. That target matters, but patch generation is rarely the first step in a real software engineering workflow. Before an agent can edit code, it has to locate the relevant context: repository structure, implementation files, existing tests, root causes behind failures, and design constraints hidden in review comments.

A coding agent's workflow therefore contains at least three stages:

1. **Context retrieval**: finding the files that need to be read in a large repository.
2. **Reasoning and planning**: understanding how those files relate and deciding what should change.
3. **Patch generation and verification**: editing code, running tests, and iterating on failures.

Many existing benchmarks emphasize the third stage, or hide the first stage inside an end-to-end task. That makes failures hard to interpret. When an agent fails, did it fail because it cannot write the patch, or because it never found the files it needed? In large real repositories, the second failure mode is common. A model can be strong at code generation and still produce plausible but useless patches if its retrieved context is wrong.

Agent Retrieval Bench isolates this upstream capability: **given a task signal from a real coding workflow, can a retriever bring the agent near the files it must read?**

## What existing code retrieval benchmarks miss

Traditional code retrieval is often framed as direct information retrieval: given a natural-language query, retrieve semantically similar code snippets; or given a code snippet, retrieve similar implementations. That setting is useful for search APIs, code QA, and snippet retrieval, but it does not fully match the context needs of coding agents.

In an agentic workflow, the query is often not “find implementation X.” It may be:

- a PR title or body that describes behavior, not the test file name;
- a review comment that points to a local issue while the required constraint lives in another module;
- a failure log containing a test name, panic, assertion, or stack trace, where the root cause is not the trace frame itself;
- a migration or compatibility task whose affected files span configuration, runtime code, tests, and documentation.

These settings share several properties.

First, **the gold file is not necessarily semantically similar to the query**. In `trace2code`, for example, the failure log may mention only a test function, while the file that must be modified is a source implementation. A pure embedding model may rank the visible test file or path above the true source file.

Second, **file relationships often matter more than single-file semantics**. Source/test mappings, package structure, module boundaries, symbol references, and dependencies between the reviewed file and its surrounding context frequently determine what should be retrieved. These are repository-graph signals, not just text-similarity signals.

Third, **leakage is easy**. If the query contains the gold path, the test basename, the fix diff, or content from the final patch, retrieval collapses into path matching. Such samples overestimate system quality and can make lexical baselines look much stronger than they are.

Fourth, **end-to-end agent success does not explain retrieval quality**. Agents can fail because of tools, test environments, patch generation, or dependency installation. They can also compensate for weak retrieval with strong reasoning. We need a smaller, more diagnostic benchmark that measures retrieval itself.

We therefore define the problem as **agentic retrieval**: not “retrieve code that looks semantically similar,” but “retrieve the repository files that an agent needs to read to complete the task.”

## Three representative examples

The following examples are simplified, but they capture patterns that repeatedly appeared while constructing V1. They are not full PR case studies; they illustrate why the task is not synthetic.

### Example 1: PR intent does not directly name the test files

The agent may see a PR-level signal such as:

> Add support for optional values in shell completion.

A query like this naturally matches completion-engine implementation files: the words “optional values” and “completion” describe implementation behavior. But if the agent needs to add or verify tests, the relevant files may be shell-specific completion tests or snapshots for bash, fish, PowerShell, or elvish.

A common retrieval failure is to return only implementation files, documentation, or changelogs. Those files help explain the feature, but they do not tell the agent which tests should be edited or inspected.

This is why `code2test` is not a simple semantic-similarity task. The test file name may never appear in the PR description, and the test may live outside the implementation directory. A retriever has to learn source-to-test mappings, module boundaries, and behavioral coverage relationships.

### Example 2: the reviewed file is not the complete context

A review comment often lands on a specific hunk. For example, a reviewer might ask near an endpoint or serializer implementation:

> Should this be sanitized consistently with the web response path?

If the task is reduced to “find the file containing the comment,” it becomes trivial. But for a coding agent, that file is already given local context. The missing context may be:

- an existing sanitization implementation in another web extension;
- a related configuration property;
- endpoint tests that define the expected behavior;
- documentation that specifies masking or sanitization semantics.

The easy mistake is to rank the commented file first and declare success. The agent still lacks the evidence needed to decide what “consistent” means.

This motivates `comment2context`: the file containing the review comment is recorded as `given_files` and is not counted as the main gold. The model must retrieve additional context files. This makes the task measure useful review-time retrieval instead of local path recovery.

### Example 3: failure traces often point to tests, not root causes

The most visible path in a failure log is frequently a test file. A Go or Python failure might look like:

```text
--- FAIL: TestOptionalCompletion
    completion_test.go:42: expected candidates [...], got [...]
```

Or the top frame of a panic/traceback may be in a test helper. Direct path matching will rank `completion_test.go` highly. But if the test is only the reproduction signal, the file to change may be the completion engine, the argument parser, runtime configuration, or another source module.

The mistake is to treat the trace frame as the root cause. For a human engineer, a test file often tells you where the failure is observed, not where the bug is. The same distinction matters for agents.

That is the core of `trace2code`: the gold files are manually audited root-cause source files; related tests are auxiliary context only. The task asks the retriever to infer the root cause from failure text, repository structure, source/test relationships, and local symbol signals, rather than performing trace-frame lookup.

Together, these examples show that agentic retrieval is hard not because the query has no information, but because the useful information is indirect. A strong retriever has to map local signals to the repository context required to finish the task.

## Research questions

Agent Retrieval Bench V1 is not designed to prove that one retrieval method is universally best. It is designed to expose which retrieval mechanisms work under different coding-agent signals. We focus on four questions.

- **Are semantic embeddings enough?** When the task signal comes from a PR, a review comment, or a failure trace, does vector retrieval still consistently beat lexical and structured methods?
- **Do vectorless repo maps have independent value?** Aider-style RepoMap methods use symbols, paths, references, and source/test relations rather than embeddings. Can this class of method outperform embeddings in some agent workflows?
- **Do different tasks require different retrieval biases?** `code2test`, `comment2context`, and `trace2code` all ask the system to “find files,” but they rely on very different signals. A single retriever may behave unevenly across tasks.
- **Can we avoid path leakage and same-name-test shortcuts?** If lexical retrieval approaches the ceiling, the benchmark cannot show whether stronger retrieval methods are useful. V1 is designed to remain solvable while resisting simple path matching.

These questions shaped the data strategy. We did not start by optimizing for thousands of samples. We first built a small, hard, manually audited, reproducible benchmark that can reveal method differences.

## Design principles

V1 follows several design principles.

**First, the evaluation target is file-level retrieval.**  
Coding agents ultimately need readable context files, not isolated chunks. The corpus is chunked to support ranking and 8k-budget diagnostics, but the primary metrics ask whether the gold file is retrieved.

**Second, the candidate corpus is fixed at the base commit.**  
Each sample is evaluated against the repository state before the task was resolved. Later fixes, PR-final diffs, and human gold evidence are not allowed into the query or candidate corpus. This prevents answer leakage through the index.

**Third, queries should resemble information an agent would actually see.**  
`code2test` uses PR intent and implementation-change summaries; `comment2context` uses the review comment and given file; `trace2code` uses failure excerpts from local reproduction. Queries do not contain fix patches, raw diffs, or gold paths.

**Fourth, gold means files needed to complete the task.**  
For `comment2context`, the commented file is given context, not the main gold. For `trace2code`, tests can be related context, but the main gold must be a root-cause source file.

**Fifth, multiple retrieval paradigms should be reported.**  
We report lexical, RepoMap, Jina, and Qwen baselines. The benchmark is not tied to a single retrieval family. A good agent retrieval benchmark should reveal the strengths and weaknesses of different inductive biases.

The core motivation is to build a retrieval layer that is close enough to real coding-agent workflows while remaining controlled, reproducible, and interpretable.

## What is in V1?

Agent Retrieval Bench V1 is the current frozen benchmark. It contains 225 manually curated samples across three tasks:

| Task | Samples | Evaluation target |
| --- | ---: | --- |
| `code2test` | 106 | Given implementation intent or PR intent, retrieve related test files |
| `comment2context` | 51 | Given a review comment and reviewed file, retrieve additional required context files |
| `trace2code` | 68 | Given a reproduced failure trace, retrieve root-cause source files |
| **Total** | **225** |  |

Every sample is evaluated on `repo_at_base_commit`: the repository state before the fix. The candidate corpus therefore does not include the later patch.

V1 ships with the benchmark files, full corpus, baseline outputs, and reports. Users can download everything with one command:

```bash
arb download-benchmark --version v1 --local-dir data --force
```

The command downloads the release bundle from Hugging Face, verifies checksums, and extracts:

```text
benchmark/v1/
corpus/v1/
eval/v1/
reports/v1/
```

## What does each task measure?

### 1. `code2test`: which tests correspond to an implementation change?

This task simulates an agent receiving a PR or implementation-change intent and needing to identify the relevant tests. The query does not directly include test paths or test basenames. The gold files are manually confirmed related tests.

The task measures structural source-to-test association, not filename matching.

### 2. `comment2context`: what else must be read beyond the review comment?

Review comments often appear on one file, but the reviewer’s intent depends on other context: API contracts, configuration behavior, existing tests, or parallel implementations.

In V1, `comment2context` treats the reviewed file as given context. It is not counted in the main metric. The model must retrieve the additional context files. Without this rule, the task degenerates into finding the commented file.

### 3. `trace2code`: where is the root cause behind a failure trace?

`trace2code` uses failure traces produced by local test reproduction, including compile errors, assertion failures, and panics. The main gold files are manually audited root-cause source files. Related tests are auxiliary and are not main gold.

The task intentionally does not reward trace-frame lookup. If the trace shows only a test file, the retriever must still use repository structure and failure signals to infer the relevant source file.

## Baselines

V1 includes four baseline families:

1. **lexical**: traditional lexical retrieval.
2. **aider-style-repomap**: a vectorless RepoMap baseline that ranks files using a file/symbol/reference graph and query-aware scoring.
3. **jina-code-embeddings-0.5b**: a code embedding model.
4. **Qwen3-Embedding-4B**: a stronger embedding model with broad code and general retrieval capacity.

The leaderboard is sorted by overall `MRR`, and also reports `Recall@5/10/20` and `gold_coverage@8k`.

## Current results

Overall results:

| Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | Gold@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 225 | 0.2883 | 0.4033 | 0.5828 | 0.2455 | 0.2542 |
| aider-style-repomap | 225 | 0.3089 | 0.4705 | 0.6299 | 0.2227 | 0.0704 |
| jina-code-embeddings-0.5b | 225 | 0.2230 | 0.3133 | 0.4492 | 0.1883 | 0.1556 |
| lexical | 225 | 0.1970 | 0.3267 | 0.4874 | 0.1450 | 0.0785 |

The task-level picture is more informative:

| Task | Best MRR model | MRR | Best Recall@20 model | Recall@20 |
| --- | --- | ---: | --- | ---: |
| overall | Qwen3-Embedding-4B | 0.2455 | aider-style-repomap | 0.6299 |
| code2test | Qwen3-Embedding-4B | 0.3225 | Qwen3-Embedding-4B | 0.7230 |
| comment2context | jina-code-embeddings-0.5b | 0.3282 | lexical | 0.5752 |
| trace2code | aider-style-repomap | 0.2750 | aider-style-repomap | 0.8064 |

Several observations stand out:

- Qwen3-Embedding-4B is strongest on overall MRR and `code2test`.
- Jina has the strongest MRR on `comment2context`.
- RepoMap is substantially stronger on `trace2code` and has the best overall Recall@20.
- `trace2code` is hard for embedding models. This is not a data bug; it is a useful signal that failure-log localization depends heavily on structure, paths, call relations, and repository graphs rather than semantic similarity alone.

## Why does RepoMap work well on `trace2code`?

The reason is tied to the task. Embedding models are good at placing natural-language queries near semantically similar code. But failure traces often contain:

- path fragments;
- function names;
- package or module names;
- test names;
- localized panic or assertion text.

These signals may not be semantically similar to the root-cause source file, but they are useful in a repository graph. The RepoMap baseline uses a file/symbol/reference graph with query-aware ranking, so it can exploit structural signals that embeddings may underweight.

This result suggests that agentic retrieval should not be evaluated only with vector baselines. In some coding-agent settings, vectorless repo maps, symbol graphs, dependency graphs, and task-aware ranking may be more aligned with the problem.

## Data construction principles

The most important rule for V1 is: fewer samples are better than easy or leaked samples.

We applied several safeguards:

- all V1 samples are manually curated;
- queries avoid direct gold paths, basenames, raw patches, and fix diffs;
- `comment2context` does not count the reviewed file as main gold;
- `trace2code` counts only root-cause source files as main gold; tests are auxiliary;
- every sample is evaluated on the base-commit corpus to avoid contamination from the fixed state.

The final V1 keeps 225 samples. Another 222 audited or mined samples were rejected or marked unsuitable for the main benchmark.

## How to reproduce

Install the evaluator:

```bash
pip install "git+https://github.com/eyuansu62/agent-retrieval-bench.git"
```

Download V1:

```bash
arb download-benchmark --version v1 --local-dir data --force
```

Validate the samples:

```bash
arb validate data/benchmark/v1/*.jsonl
```

Run the lexical baseline:

```bash
arb eval-baseline \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/lexical_summary.json \
  --details data/eval/v1/lexical_details.jsonl \
  --no-keep-list
```

Run the RepoMap baseline:

```bash
arb eval-repomap \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/repomap_summary.json \
  --details data/eval/v1/repomap_details.jsonl \
  --candidate-filter all_files \
  --no-keep-list
```

Generate a leaderboard:

```bash
arb report-models \
  --eval-dir data/eval/v1 \
  --out data/reports/v1/model_leaderboard.md \
  --json-out data/reports/v1/model_leaderboard.json
```

Dataset: `eyuansu71/agent_retrieval_bench` on Hugging Face.  
Code: `eyuansu62/agent-retrieval-bench` on GitHub.

## Next: V1.1

V1 is frozen. We will not modify `benchmark/v1`. The next release will be V1.1, focused on targeted gaps rather than raw scale.

- `comment2context`: expand from 51 to 80–100, prioritizing cross-module context without path leakage or same-directory shortcuts.
- `trace2code`: expand from 68 to 100+, adding non-Go repositories, more languages, and more failure types.
- `code2test`: currently has 106 samples, so it is not the default expansion target.

Longer term, we want Agent Retrieval Bench to cover more real coding-agent scenarios, such as issue/bug report → code, migration task → affected files, and API usage → implementation. V1 is the first step: measure code review, test retrieval, and trace root-cause retrieval clearly.

## Closing

An agent’s ceiling is not determined only by whether it can write a patch. It is also determined by whether it can find the files it needs to read in a large repository.

Agent Retrieval Bench V1 measures this upstream capability: given PR intent, a review comment, or a failure trace, can a retriever bring the agent near the right repository context?

The first results show that embeddings, lexical retrieval, and RepoMap each have distinct strengths. No method dominates every task. That points toward a hybrid future for agentic retrieval: semantic vectors, repository graphs, symbol indexes, and task-aware ranking all need to work together.
