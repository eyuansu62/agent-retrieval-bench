# edit2ripple Data Construction

`edit2ripple` evaluates whether a retriever can start from one already-modified
anchor file and find the other files that need synchronized edits in the same
real PR or commit.

## Candidate Mining

Use raw PR crawl data with `pull_requests.jsonl`, `pull_files.jsonl`, and
optional `commit_details.jsonl`:

```bash
arb mine-edit2ripple \
  --raw data/raw_token \
  --out data/benchmark/v1_edit2ripple_candidates \
  --report-out data/reports/v1_edit2ripple_candidates \
  --corpus-manifest data/corpus/v1/corpus_manifest.jsonl \
  --require-corpus \
  --require-gold-in-corpus
```

If raw crawl data is not available but existing benchmark samples contain
GitHub PR URLs, refetch those PRs directly:

```bash
arb mine-edit2ripple-from-samples data/benchmark/v1/samples.jsonl \
  --out data/benchmark/v1_edit2ripple_from_samples \
  --report-out data/reports/v1_edit2ripple_from_samples \
  --corpus-manifest data/corpus/v1/corpus_manifest.jsonl \
  --require-corpus \
  --require-gold-in-corpus
```

For unauthenticated GitHub API runs, use small `--limit-prs` batches or set
`GITHUB_TOKEN`; the backfill path fetches PR details, files, and commits.

If existing samples include `base_commit` and `gold.fix_commit`, prefer the
git-diff path because it avoids GitHub REST rate limits:

```bash
arb mine-edit2ripple-from-sample-commits data/benchmark/v1/samples.jsonl \
  --out data/benchmark/v1_edit2ripple_from_commits \
  --report-out data/reports/v1_edit2ripple_from_commits \
  --repos-dir data/repos_edit2ripple \
  --corpus-manifest data/corpus/v1/corpus_manifest.jsonl \
  --require-corpus \
  --require-gold-in-corpus
```

This reconstructs changed files and per-file patches with
`git diff base_commit gold.fix_commit`, then applies the same anchor/gold and
leakage gates as the raw-PR miner.

The miner keeps PRs with 2-8 changed files, requires a non-test source anchor,
excludes generated/vendor/lock/snapshot/changelog-style changes, and writes:

- `samples.jsonl`
- `edit2ripple.jsonl`
- `manifest.json`
- `audit_samples.jsonl`
- `audit_samples.csv`

## Query Contract

Each sample query contains only:

- `intent`: PR title/body and commit-message text, truncated and cleaned.
- `anchor_file`: the single given file.
- `anchor_diff`: sanitized diff hunks for the anchor file only.

The query must not contain gold paths, gold basenames, full multi-file patches,
or the fix commit hash. Gold stems and shared module tokens are recorded as
nonfatal hints because they can occur naturally in issue and PR language; the
miner drops only candidates with fatal leakage.

## Gold Contract

`gold.files` contains non-anchor co-changed files that the miner can connect to
the anchor with at least one relation signal:

- same source component
- path token overlap
- anchor diff mentions the gold module
- gold diff mentions the anchor module
- shared changed symbol
- shared config or schema
- optional test ripple

The release audit must still decide whether those co-changes are necessary
context. Use verdicts `valid`, `noisy`, `leaked`, and `ambiguous`; keep only
valid rows for a clean pilot subset.

## Required Pilot Gates

Before release, verify:

- fatal leakage count is 0
- every sample has at least one gold file
- every gold file exists in the base corpus
- valid rate is at least 90%, or a clean subset can be exported
- test-only gold does not dominate the sample set
- each PR contributes at most 1-2 retained anchor candidates

Generate the gate report with:

```bash
arb report-edit2ripple-pilot data/benchmark/v1_edit2ripple_from_samples/samples.jsonl \
  --corpus-manifest data/corpus/v1/corpus_manifest.jsonl \
  --audit data/reports/v1_edit2ripple_from_samples/audit_samples.csv \
  --out data/reports/v1_edit2ripple_from_samples/pilot_report.md \
  --json-out data/reports/v1_edit2ripple_from_samples/pilot_report.json
```

`status=needs_audit` means the automatic gates pass but release verdicts are
missing. `status=needs_more_samples` means the audited quality gates pass but
the pilot is still below the default 50-sample minimum. `status=ready` requires
the configured valid-rate threshold and sample-count threshold to pass.
