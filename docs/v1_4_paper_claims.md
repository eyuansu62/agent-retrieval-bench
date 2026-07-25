# V1.4 Paper-Facing Claims

This note turns the V1.4 trajectory results and error analysis into claims that can be used in a paper, report, or release note. It also lists the limits of each claim so that the analysis does not overstate what the run proves.

## Claim 1: Iterative Agent Context Selection Beats Same-Budget Top-K Retrieval

At a comparable final-context budget, the GPT-5.4-mini strict-context run substantially outperforms direct top-k ranked retrieval.

| Method | Avg final files | Overall file F1 | code2test F1 | comment2context F1 | trace2code F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical@3 final context | 3.00 | 0.0721 | 0.0255 | 0.0746 | 0.1191 |
| RepoMap@3 final context | 3.00 | 0.1102 | 0.1082 | 0.0633 | 0.1495 |
| RepoMap@4 final context | 4.00 | 0.1198 | 0.1165 | 0.0756 | 0.1583 |
| GPT-5.4-mini strict-context | 3.16 | 0.3113 | 0.2516 | 0.1732 | 0.4835 |

Use this claim to argue that trajectory-aware context selection is not equivalent to taking the first few files from a static ranked list. The fair comparison is not top-20 recall; it is what enters the small final context window.

Limit: this does not prove that GPT-5.4-mini is the best agent or that static retrieval is obsolete. RepoMap@20 still has strong recall as a candidate generator, especially on `trace2code`.

## Claim 2: Trace2Code Is File-Level Easier But Span-Level Hard

`trace2code` is the strongest file-level task for GPT-5.4-mini strict-context:

- `trace2code.final_file_f1=0.4835`
- `trace2code.final_file_recall=0.8960`
- `trace2code.final_file_precision=0.3416`

The same run has weak span alignment:

- `trace2code.line_f1@trajectory=0.0312`
- 36 `trace2code` samples have at least one gold file selected but zero line-level overlap.

Use this claim to motivate span-aware trajectory evaluation. Finding the root-cause file is often achievable; reading the right region is still weak.

Limit: current agent logs record whole-file reads for many actions, so line metrics are conservative and depend on logged windows.

## Claim 3: Comment2Context Is The Hardest Agentic Retrieval Track

`comment2context` remains the hardest trajectory task:

- `comment2context.final_file_f1=0.1732`
- `comment2context.exact_misses=47/80`
- `comment2context.full_hits=16/80`

The error analysis separates two failure modes:

- Support-near misses: the model reads useful supporting context but misses the exact gold file.
- Hard additional-context misses: the review comment is a weak anchor for cross-module policy, config, serializer, or test dependencies.

Use this claim to argue that the benchmark tests additional-context retrieval, not just retrieval of the file named in the review comment.

Limit: supporting-context annotations are not complete for every sample; exact gold recovery and broad usefulness should be reported separately.

## Claim 4: Code2Test Failures Are Mostly Source-To-Test Mapping Failures

For `code2test`, the agent frequently reads plausible implementation files or nearby tests but misses the exact regression/integration test target:

- `code2test.final_file_f1=0.2516`
- `code2test.exact_misses=50/106`
- `code2test.full_hits=41/106`

Representative misses include `etcd-io/etcd`, `vitejs/vite`, and `tokio-rs/tokio`, where the model selected implementation-neighborhood files while the gold target was a specific e2e, playground, or runtime test.

Use this claim to frame `code2test` as a repository-specific mapping problem: a good retriever must learn test layout and integration-test conventions, not only match changed implementation paths.

Limit: the current trajectory runner does not execute test discovery tools; it only searches and reads corpus files.

## Claim 5: The Strict-Context Release Is Auditable

The canonical release is suitable for citation because answer context, trajectory logs, traces, and release packaging agree:

- `logs=287`, `answers=287`, `traces=287`
- `read_steps_total=918`
- `final_files_total=907`
- `below_min_reads=0`
- `duplicate_read_samples=0`
- `missing_read_path_samples=0`
- `missing_final_path_samples=0`
- `empty_final_samples=0`
- `final_unread_samples=0`
- `strict_openai_key_hits=0`

The v1 exploratory run is still useful for debugging, but it should not be the cited release because 146/287 answers included unread final-context files.

Limit: raw model suggestions can still mention unread files. In the strict run these are isolated as `suggested_unread_files` and excluded from `final_files`.

## Recommended Reporting Order

1. Report V1.3 single-shot retrieval baselines separately from V1.4 trajectory metrics.
2. Use same-budget top-k final-context contrasts to show why process logs matter.
3. Use task-level error modes to explain what the aggregate numbers mean.
4. Use strict-context audit gates to establish release trustworthiness.
