# Indexed Tools in the Closed-Tool Loop

The `indexed_interactive` mode measures whether a policy uses index lookups
well when it chooses them itself. It runs the same evaluator-mediated loop as
the closed-tool diagnostic and adds one or both of these tools:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `symbol_search` | `symbol` | Definitions with that name, as path and line range |
| `references` | `path`, `symbol` | Locations that reference, or are referenced by, that symbol, as path and line range |

Lookups return locations only. Reading content still costs `read_file` calls
and read tokens, so context accounting matches the minimal-tool mode. The
minimal-tool mode (`--mode closed_tool`, the default) stays available
unchanged.

Selected sample IDs must be unique across input files. The closed-tool
evaluators reject duplicates before dispatching a policy or changing output
files, including identical repeated rows: IDs are the resume/cache keys, not
weights. Check overlapping releases or repeated `--samples` arguments instead
of silently reweighting cached trajectories. Keep-list filtering and
`--limit-samples` are applied before this validation.

## Plugging in a provider

ARB does not depend on any index library. A provider is a Python object that
`--tool-provider module:callable` creates; each `--tool-provider-option
key=value` becomes a keyword argument of the callable. An MCP-backed index
needs a small callable that forwards these calls to its server.

```python
class Provider:
    def describe(self) -> dict: ...  # identifies the provider/build for reporting and resume
    def prepare(self, *, repo, base_commit, paths) -> dict: ...
    def call(self, *, tool, arguments, repo, base_commit, limit) -> list[dict]: ...
```

`prepare` runs once per snapshot. `paths` is the released file list. It returns
`available` and may add `build_seconds`, `cached`, `size`, and `reason`. When
`available` is false, or `prepare` raises, the policy sees only the minimal
tools for that sample and the prompt says the index is unavailable.

`call` receives `{"symbol"}` for `symbol_search` and `{"path", "symbol"}` for
`references`. Each hit has `path` and may have `start_line` and `end_line`
(1-based, inclusive), `symbol`, `kind`, and `relation`.

The provider never receives the sample id, the query, or gold labels.

## What the evaluator enforces

- Only the tools named in `--indexed-tools` are offered.
- An answered lookup costs one call from the shared `--max-tool-calls` budget.
  Lookups with a missing symbol, or a `references` path that is not a released
  file, are rejected without charge, like other invalid actions.
- Hits outside the released files are dropped and counted.
- At most `--max-index-results` hits are shown (default 10), and the whole
  observation, header and status line included, fits
  `--max-index-observation-tokens` (default 400, counted with the charged code
  tokenizer). Long echoed arguments are shortened, then dropped, before the
  status line would exceed the cap; hits are then trimmed from the end. A cap
  too small for the response without echo is rejected when the run starts.
- A provider exception is shown to the policy as a failed lookup and still
  costs the call.

## What a run records

Each trajectory records index calls by tool, index observation tokens,
provider errors, and the snapshot's index status. Every OpenAI policy run also
records model input and output tokens from the Responses API and the tokens of
all observations. The summary reports the provider's `describe()` output, the
per-snapshot index builds, the policy settings, and the harness commit. Index
build time is a per-snapshot cost and is reported separately from the policy
run.

Failed attempts are always checkpointed, even when they are excluded from the
summary. With `--retain-failures`, a sample whose policy run raises scores as empty
context, so the denominator stays the selected set, while its costs are the
ones it actually incurred: model tokens, tool calls, read and observation
tokens, and its index status. The details file keeps the failed attempt with
its partial trace. Resuming retries it; the new trajectory lists the earlier
attempts, with their costs, failures, traces, and index status, under
`prior_attempts`, and the summary totals them. The summary keeps one
index-build record per snapshot: the first available cold build from the
trajectories, their earlier attempts, or this run. An unavailable preparation
without a `cached` flag does not hide a later successful build. If no cold
build is recorded, the summary uses an available cached record, otherwise the
first failed preparation. All attempt statuses remain in the details. A
resumed run, or a retry that hits the cache, therefore reports the original
build once.

## Resuming one experiment

The scripted, OpenAI, and Codex closed-tool evaluators keep cumulative records
in `--details`; each invocation's summary covers only its current selection.
Changing the keep-list, sample order, input filename, `--limit-samples`, output
paths, or `--retain-failures` does not discard earlier trajectories. Completed
samples are reused; failed samples are retried and retain their prior costs.
Use one writer per details file.

Each new record stores a resume fingerprint for the effective policy, budgets,
model/provider settings, indexed-tool configuration, and harness Python source,
plus the sample contents, released corpus texts, and selected Codex seed files.
Resume validates every saved experiment fingerprint and every selected saved
sample before dispatch or output mutation. Corpus texts are hashed once per
snapshot per invocation. Selection/reporting options are excluded from the
experiment fingerprint; effective policy or input changes require a separate
details file or an explicit `--force` reset. `--force` discards all old records,
including failed attempts and their costs; it is not a migration operation.

Legacy records without fingerprints cannot be verified and are rejected by
the evaluators unless explicitly reset. They remain readable by the report
commands. CLI provider factories/options participate in the fingerprint;
Python callers must supply a stable `describe()` or explicit
`IndexedTools(..., provider_configuration=...)` to resume an indexed run.
Provider descriptions must identify configuration/build, not mutable counters.
Fingerprints store digests, not raw options or credentials. They cannot attest
remote model weights or undeclared provider state. Codex fingerprints bind its
executable and `CODEX_HOME/config.toml`; keep other ambient Codex configuration
fixed when resuming.

Details append after each sample and compact via a sibling temporary file and
atomic replacement. An incomplete, unterminated last append is recoverable
with a warning; malformed interior or newline-terminated records are rejected
without changing the file. This protects checkpoints from an interrupted
compaction, not concurrent writers or a transaction across summary/report files.

The budget-prefix report deduplicates journals by sample ID (latest record
wins), with or without a keep-list. Its `observed_run_costs` reports actual
latest-attempt costs and recorded `prior_attempts` once, separately from the
post-hoc prefixes; prefix scores are not counterfactual rerun costs.

By default, a submit that names no file already read ends the episode with
empty context. With `--require-read-submit`, that submit is rejected without
charge and the policy continues. Each trajectory records its rejected submits,
the model turn of each, and the model tokens those turns used; the summary
reports their rate and mean tokens.

## Running arms on a frozen sample set

`select-closed-tool-samples` freezes a per-task sample set before any policy
runs. It keeps samples that have gold files, no query leakage, and a released
corpus snapshot. Within each task, repositories take turns in a seeded hash
order, so every repository gets one sample before any gets two. It reads no
method scores and builds no index.

```bash
TASKS="code2test comment2context edit2ripple trace2code"
SAMPLES=$(for t in $TASKS; do echo data/benchmark/v2_$t/samples.jsonl; done)
CORPORA=$(for t in $TASKS; do echo --corpus data/corpus/v2_$t; done)
arb select-closed-tool-samples $SAMPLES $CORPORA \
  --per-task 15 --seed <seed> --out pilot_samples.jsonl --manifest-out pilot_manifest.json

# Shared by every arm: the frozen samples, policy, and budgets.
COMMON="$SAMPLES $CORPORA --keep-list pilot_samples.jsonl \
  --model <dated snapshot> --reasoning-effort none \
  --max-tool-calls 16 --max-model-turns 20 --max-read-tokens 8000 \
  --max-read-tokens-per-file 1200 --final-k 3 --grep-top-k 12 \
  --retain-failures --require-read-submit"
PROVIDER="--tool-provider <module:callable> --tool-provider-option <key=value> \
  --max-index-results 10 --max-index-observation-tokens 400"

# A: minimal tools.
arb eval-closed-tool-llm $COMMON --mode closed_tool \
  --out summary_A.json --details details_A.jsonl --markdown-out report_A.md
# B: adds symbol_search.
arb eval-closed-tool-llm $COMMON --mode indexed_interactive $PROVIDER \
  --indexed-tools symbol_search \
  --out summary_B.json --details details_B.jsonl --markdown-out report_B.md
# C: adds symbol_search and references.
arb eval-closed-tool-llm $COMMON --mode indexed_interactive $PROVIDER \
  --indexed-tools symbol_search,references \
  --out summary_C.json --details details_C.jsonl --markdown-out report_C.md

# Tool-budget prefixes over the frozen selection, per arm.
arb report-closed-tool-budget-curve --details details_A.jsonl --keep-list pilot_samples.jsonl \
  --out curve_A.json --markdown-out curve_A.md
```

Repeat `--corpus` once per samples file to pair each release with its own
corpus. Run every arm with the same model, reasoning effort, budgets, and
sample set. `--require-read-submit` and `--retain-failures` stay off unless
given; the pilot sets both in every arm. With `--keep-list`, the prefix report
scores every selected sample, and one without a trajectory counts as empty
context.
