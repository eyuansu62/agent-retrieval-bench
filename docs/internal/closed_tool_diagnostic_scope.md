# V1.4 Closed-Tool Diagnostic Scope Notes

Last updated: 2026-06-14

This note records scope boundaries for the closed-tool context-acquisition diagnostic. It is intended as a repo-side reminder before deciding what to move into the paper's Limitations or Appendix.

## Current Status

The closed-tool experiment is an Appendix diagnostic, not a main leaderboard track.

Implemented pieces:

- Docker-isolated Codex policy with evaluator-mediated `list_dir`, `grep`, `read_file`, and `submit` actions.
- Protocol-violation auditing for shell or unsupported tool use.
- Incremental details logging and resume support for long runs.
- Post-hoc tool-budget prefix report from existing trajectories at `B in {2,4,6,8}`.

Current headline result:

- Codex closed-tool reaches overall prefix File F1 `0.1863` and any-gold `0.2578` within 2 tool calls.
- It reaches File F1 `0.3276` and any-gold `0.5470` within 4 tool calls.
- Scores saturate after 4 calls on the current full run.

## Issues We Can Leave As Scope Boundaries

### 1. Only One Model Policy

The diagnostic currently evaluates one Codex policy. This is acceptable for an Appendix protocol check, because the claim is that ARB can score controlled interactive context acquisition, not that we have a complete interactive-retrieval leaderboard.

Do not overclaim:

- Do not call this a full closed-tool leaderboard.
- Do not compare it as a model-family ranking.

Future extension:

- Add multiple policies only if the paper wants a main interactive-retrieval track.
- Candidate policies could include Codex variants, OpenAI Responses policies, Claude-style tool policies, or scripted strategies.

### 2. Minimal Tool Interface

The current interface intentionally contains only `list_dir`, `grep`, `read_file`, and `submit`.

This is a V0 closed-tool design choice. It tests whether a policy can acquire context through basic repository exploration under evaluator control.

Do not overclaim:

- Do not present it as matching full production coding-agent tool suites.
- Do not imply that missing `symbol_search`, RepoMap, or embedding search is a flaw in the current Appendix diagnostic.

Future extension:

- Treat `symbol_search`, RepoMap seed context, and embedding seed context as an `Indexed interactive` track.
- Keep that track separate from the closed-tool minimal-tool diagnostic, because it evaluates a different system boundary.

### 3. No Patch-Success Link

The closed-tool metric evaluates context acquisition, not patch generation.

This is aligned with ARB's scope as a retrieval/context benchmark. Patch success is downstream and confounded by reasoning, editing, test execution, and validation.

Current V1.3 reviewed data does not directly support a patch-success intervention. A quick schema check on `data/benchmark/v1_3_reviewed/samples.jsonl` found `repo`, `base_commit`, `pr_url`, `query`, and `gold`, but no executable repair oracle fields:

- No `test_patch`, `FAIL_TO_PASS`, or `PASS_TO_PASS`.
- No `test_command` or `validation_command`.
- No `problem_statement`, issue text, or SWE-bench `instance_id`.
- No stored `fix_commit` or full patch payload.

Therefore, the current intervention result should be framed as evidence that retrieval seeding changes agent context acquisition, not that it improves final bug-fixing success.

Do not overclaim:

- Do not claim that closed-tool File F1 causally predicts patch success.
- Do not frame the current diagnostic as a repair benchmark.
- Do not describe the seed intervention as a downstream patch-success experiment.

Future extension:

- Add a small correlation or intervention study only if we need to strengthen the motivation claim.
- Possible study: on overlapping SWE-bench-style trajectories, compare gold-file touch / closed-tool score against patch success.
- Cleaner study: build a small executable subset with a problem statement, base checkout, validation command, and pass/fail oracle; then run the same agent scaffold with `no_seed`, random seed, retriever seed, and oracle seed.
- Lowest-risk mapping path: identify overlap with SWE-bench, Defects4J, or another executable repair benchmark and attach ARB-style retrieval seeds to those tasks instead of turning all of ARB into a patch benchmark.

## Suggested Paper Placement Later

Main Limitations, short version:

> The closed-tool diagnostic is intentionally scoped as a protocol check rather than a full interactive-retrieval leaderboard. It evaluates one Codex policy, uses a minimal `list_dir`/`grep`/`read_file`/`submit` interface, and does not establish a causal relationship with downstream patch success. Extending this to multi-policy, indexed-interactive, and patch-success-linked evaluations is future work.

Appendix, expanded version:

- Single-policy result: enough for diagnostic, not enough for leaderboard.
- Minimal tools: deliberate V0 protocol, not production-agent parity.
- Patch-success link: future work, outside current retrieval/context benchmark claim.
