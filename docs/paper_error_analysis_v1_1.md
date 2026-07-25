# V1.1 Error Analysis Draft

This document turns the V1.1 leaderboard, rank-depth report, and representative samples into paper-facing analysis text. It is intended to be copied into an Error Analysis or Discussion section after tightening for venue length.

Source artifacts:

- `data/reports/v1_1/model_leaderboard.json`
- `data/reports/v1_1/rank_analysis.json`
- `data/reports/v1_1/rank_analysis.md`
- `data/reports/v1_1/candidate_filter_ablation.json`

## Main Takeaway

The V1.1 results are not well explained by a single "best retriever" story. Qwen3-Embedding-4B has the highest overall MRR (`0.2296`), but Qwen3-Embedding-8B retrieves at least one gold file in the top 20 for more samples (`Any@20=0.7944` vs. `0.6864`). RepoMap is only fourth by overall MRR, but it is strongest on `trace2code` with MRR `0.2745`, Any@20 `0.9109`, and median first-gold rank `5`. These differences show that ARB measures several distinct retrieval behaviors: early precision, depth coverage, and task-specific use of repository structure.

The error analysis supports the same conclusion. Across the seven reported V1.1 `all_files` baselines, `265/287` samples have at least one model retrieve a gold file in the top 20, while `22/287` are missed by every model. The all-model misses are distributed as `10/106` for `code2test`, `9/80` for `comment2context`, and `3/101` for `trace2code`. This indicates that the benchmark is not merely separating good models from bad models; it also exposes complementary retrieval biases and a smaller set of genuinely hard cases.

## Paper-Ready Draft

Aggregate ranking alone hides important retrieval behavior. Qwen3-Embedding-4B has the best overall MRR in V1.1, but the rank-depth analysis shows that Qwen3-Embedding-8B has broader top-20 coverage. The 4B model tends to place successful hits earlier, while the 8B model retrieves at least one gold file for more samples by depth 20. This distinction matters for coding agents because a retriever can be useful either by placing the right file very early or by keeping it inside a later reranking or context-selection budget.

The strongest task-specific reversal appears in `trace2code`. On this track, RepoMap reaches MRR `0.2745` and Any@20 `0.9109`, compared with Qwen3-Embedding-4B at MRR `0.0827` and Any@20 `0.5347`. Lexical retrieval is also competitive on this task, with MRR `0.2075` and Any@20 `0.7525`. This suggests that failure-log retrieval often depends on repository-specific structure: test file names, source/test adjacency, symbols, package layout, and implementation files linked to visible failing tests. These signals are not always captured by semantic similarity between the failure text and the source file that must be changed.

Representative `trace2code` examples show both sides of this result. In several Gin samples, such as `30fb9ae30b972db986c4365d` with gold `response_writer.go` and `9e63d7a07d201c84d7d18546` with gold `errors.go`, RepoMap retrieves the gold source file in the top 20 while Qwen3-Embedding-4B and Qwen3-Embedding-8B miss it. The failure excerpts mention tests and compile failures, so semantic models often rank visible test files or generic repository metadata above the implementation file. RepoMap can use structural proximity between tests and source files to recover the root-cause file.

The opposite pattern also occurs, which is important for the paper's conclusion. In `d4149b3f568dfe275250b62e`, a Caddy authentication failure has gold `modules/caddyhttp/caddyauth/caddyauth.go`; pplx-embed-v1-4b ranks it first while RepoMap misses it at depth 20. Similar embedding wins appear for Caddy HTTP Caddyfile files and several Gin source files. These cases prevent a simplistic "RepoMap beats embeddings" conclusion. The evidence instead points toward hybrid retrieval: semantic models can recognize some behavior descriptions directly, while structure-aware methods recover files through repository topology when the textual signal is indirect.

`comment2context` exposes a different error mode. The reviewed file is already given to the agent, so retrieving it is not enough. Several failures are "given-file traps," where a retriever ranks the commented file or nearby local context first but misses the additional file needed to resolve the review. For example, sample `bf47308900fafdff5ca96ecd` asks about seed-printing behavior from `hypothesis-python/tests/cover/test_seed_printing.py`, but the gold context is in `hypothesis-python/src/hypothesis/core.py` and `hypothesis-python/src/hypothesis/internal/conjecture/engine.py`. RepoMap ranks the given test file first and misses the gold files at depth 20. This illustrates why `comment2context` treats `given_files` as context rather than gold.

Other `comment2context` misses involve configuration or cross-module behavior. Sample `449280b38ce700056be47fe` has a review comment in `lib/rules/no-param-reassign.js`, but the gold file is `eslint.config.js`. Sample `413524fdc2e0151ec121653c` involves Ruff project database behavior and requires `crates/ty_server/tests/e2e/configuration.rs`. In such cases, the relevant file may not share obvious lexical overlap with the review comment, and the reviewed file can be a distracting anchor. This supports the benchmark design choice to evaluate additional context retrieval rather than simple commented-file recovery.

`code2test` failures are often broad source-to-test mapping problems. Sample `6338b75970bc7489e13952b4` from etcd includes nine implementation files and two existing test files in the query summary, but all seven baselines miss the gold tests `tests/e2e/ctl_v3_test.go` and `tests/framework/e2e/cluster.go` at depth 20. Sample `210933147c231644eeebdaae` from diffusers changes `src/diffusers/utils/torch_utils.py`, but the gold is `tests/others/test_utils.py`; models tend to rank the changed utility or infrastructure files above the relevant test. These examples show that `code2test` is not only a semantic search problem. It also requires mapping implementation changes to project-specific test organization.

The candidate-filter ablation reinforces the need for a consistent official candidate space. Restricting candidates to `tests_only` sharply improves `code2test` MRR for vectorless baselines, but it collapses `trace2code` to zero because source-file golds are removed. Conversely, `code_only` improves vectorless overall MRR by reducing distracting non-source candidates, especially for `trace2code`. These results are useful diagnostically, but they should not replace the official `all_files` leaderboard. Task-specific filters can act like oracle information about the target file type and can make different tasks incomparable.

Overall, V1.1 suggests that agent retrieval systems should combine complementary signals. Semantic embeddings are strong for broad PR intent and some review-context queries. RepoMap and lexical signals are unusually strong on failure traces, where the query often names tests, stack frames, symbols, or package-local files rather than the implementation file itself. A practical coding-agent retriever should therefore combine semantic vectors, lexical/path matching, source-test relations, symbol graphs, and task-aware reranking rather than relying on one retrieval family.

## Error Patterns by Task

### `code2test`

Primary failure mode: source changes do not directly name the relevant tests.

Evidence:

- `10/106` `code2test` samples are missed by all seven baselines at top 20.
- Qwen3-Embedding-4B is strongest by MRR (`0.3225`), but even the best models miss broad source-to-test mappings.
- `tests_only` improves vectorless `code2test` MRR, which shows how much the task depends on candidate-space priors.

Paper wording:

> `code2test` failures often require repository-specific knowledge of test organization. A query may identify implementation files precisely while leaving the relevant regression or integration tests implicit. This makes source-to-test retrieval different from finding semantically similar code.

### `comment2context`

Primary failure mode: the retriever overfocuses on the given reviewed file.

Evidence:

- `9/80` `comment2context` samples are missed by all seven baselines at top 20.
- Jina has the best MRR (`0.3043`), while Qwen3-Embedding-8B has the best R@20 (`0.6562`).
- Given-file trap examples show top-ranked files that are already available to the agent and therefore should not count as successful context retrieval.

Paper wording:

> `comment2context` is designed to avoid a common shortcut: returning the file that contains the review comment. The reviewed file is known context. The retrieval problem is to find the additional implementation, configuration, or test files needed to understand the requested change.

### `trace2code`

Primary failure mode: the failure text is closer to tests and logs than to the root-cause source file.

Evidence:

- RepoMap leads `trace2code` by MRR (`0.2745`), R@20 (`0.8465` in the main leaderboard), Any@20 (`0.9109`), and median first-gold rank (`5`).
- Lexical retrieval is second by MRR (`0.2075`) and has Any@20 `0.7525`.
- Qwen3-Embedding-4B has strong overall MRR but weak `trace2code` MRR (`0.0827`).
- Only `3/101` `trace2code` samples are missed by every model at top 20, which suggests strong complementarity rather than uniform impossibility.

Paper wording:

> `trace2code` is the clearest evidence that agentic retrieval differs from semantic code search. Failure traces often expose tests, stack frames, error messages, and package-local symbols, but the file an agent must edit is a root-cause source file. Structure-aware retrieval can exploit the relation between the visible failure and nearby implementation files.

## Claims to Use Carefully

Strong claims:

- No retrieval family dominates all tasks.
- `trace2code` benefits strongly from structure-aware retrieval.
- `comment2context` is not solved by retrieving the reviewed file.
- Candidate filters are useful diagnostics but should not define the official leaderboard.
- Hybrid retrieval is the most plausible next direction.

Claims to avoid:

- Do not claim embeddings are generally weak. They lead overall and are strong on `code2test` and `comment2context`.
- Do not claim RepoMap is generally best. It is strongest on `trace2code`, but embeddings win many individual trace samples and lead aggregate MRR.
- Do not claim all misses are benchmark errors. The readiness gates show gold files exist in the base corpus; all-model misses are evidence of hard retrieval cases unless manually disproven.
- Do not claim V1.1 is large-scale. It is diagnostic and manually audited.

## Suggested Discussion Paragraph

The benchmark also reveals a practical design implication for coding-agent systems. A single retriever optimized for semantic similarity can miss files that are structurally related to the query but not textually similar to it. Conversely, a structure-aware retriever can miss behavior-level matches that an embedding model captures directly. The strongest future system is therefore likely to be hybrid: use semantic retrieval for broad intent, lexical and path signals for concrete names, repository graphs for source-test and symbol relations, and task-aware reranking to choose which evidence should enter the agent's context window.
