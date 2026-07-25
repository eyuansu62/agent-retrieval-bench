# ARB-v2 Abstention Track Crawling Plan

## Goal

The abstention track evaluates whether a retrieval system can decide that a coding-workflow query has no necessary repository-local context. Unlike the existing positive-only tracks, these samples set `gold.files = []` with evidence that an agent should not keep searching the current repository.

The first version should optimize for precision and auditability, not scale. A clean set of 50-100 no-gold samples is more valuable than a larger ambiguous set.

## Scope

Build two candidate pools:

- `organic_no_gold`: real workflow events where the resolution indicates no repository-local fix was needed.
- `counterfactual_wrong_repo`: real positive ARB queries paired with a plausible but incorrect repository corpus.

Recommended pilot size:

- Organic no-gold: 30-50 final clean samples.
- Counterfactual wrong-repo: 30-50 final clean samples.
- Total: 60-100 final clean samples.

Organic and counterfactual samples must be reported separately.

## Repository Scope

Start with repositories already supported by ARB corpora.

Reasons:

- Base commits and corpus manifests already exist.
- Clone and evaluation infrastructure already works.
- Positive samples can be reused for paired evaluation.
- New-repository corpus construction is avoided during the pilot.

Later expansion can add adjacent repositories matched by language, framework, and domain for harder counterfactual pairs.

## Organic Source 1: Flaky or Infrastructure CI

This is the highest-priority organic source because SHA, query, and evidence are usually concrete.

### Data Sources

- GitHub Actions check runs.
- PR check suites.
- CI logs.
- PR comments and maintainer comments.

### Recall Signals

Candidate keywords:

- `flaky`
- `rerun`
- `infra`
- `CI failure`
- `transient`
- `timeout`
- `network`
- `GitHub Actions`
- `rate limit`

### Candidate Condition

Keep candidates where:

- A check run fails.
- The same `head_sha` later passes after rerun, with no new code commit.
- Maintainer comments or labels support flaky / infra / transient failure.
- There is no linked local fix or workaround.

### Query Construction

Use:

- Failed command.
- Failure excerpt.
- Failing test name or stack trace if present.
- PR title/body, if needed for workflow context.

Do not include:

- Maintainer resolution comment.
- Flaky / infra diagnosis.
- Rerun success evidence.
- Closing reason.

### Base Commit

Use the failing check run `head_sha`. This is the repository state that produced the failure signal.

### No-Gold Evidence

Accept only when evidence shows:

- Same SHA rerun passed, or maintainer explicitly marked it flaky / infra.
- No associated repository-local fix commit.
- No dependency pin, lockfile update, CI config workaround, test skip, or source change resolved the event.

## Organic Source 2: Upstream Dependency

### Data Sources

- Closed issues.
- Issue comments.
- Linked upstream issues.
- PR comments.

### Recall Signals

Candidate keywords:

- `upstream`
- `upstream bug`
- `fixed upstream`
- `dependency bug`
- `not our bug`
- `third-party`
- `library issue`
- `waiting on upstream`

### Query Construction

Use:

- Issue title/body.
- Error excerpt.
- Reproduction snippet.

Do not include:

- Maintainer comment saying it is upstream.
- Linked upstream issue URL.
- Resolution or closing reason.

### Base Commit

Prefer a stable commit near the issue creation time or the latest default-branch commit before maintainer upstream confirmation. If this cannot be determined reliably, skip the candidate in the pilot.

### No-Gold Evidence

Accept only when:

- Maintainer explicitly attributes the problem to upstream.
- Issue closes without a repository-local fix.
- No dependency pin, lockfile update, config workaround, or local compatibility patch resolves it.

If the repository changed a dependency manifest, lockfile, source file, test, or CI config to handle the issue, the sample is not no-gold.

## Organic Source 3: External Service or Environment

### Data Sources

- Issues.
- CI logs.
- PR comments.

### Recall Signals

Candidate keywords:

- `credential`
- `token`
- `permission`
- `network`
- `DNS`
- `outage`
- `service unavailable`
- `rate limit`
- `environment`
- `local setup`
- `cannot reproduce`
- `works as intended`

### No-Gold Evidence

Accept only when:

- Maintainer or CI metadata identifies an external service, credential, environment, or provider issue.
- There is no repository-local fix.
- No local config file needs to be changed to solve the issue.

## Organic Source 4: Invalid or User Error

This source is useful but should be handled conservatively.

### Recall Signals

Candidate keywords and labels:

- `invalid`
- `not a bug`
- `user error`
- `misconfiguration`
- `usage`
- `works as intended`
- `expected behavior`

### Keep Conditions

Keep only if:

- A maintainer comment explains the issue.
- The query still resembles a coding-agent workflow signal.
- There is no local file that should be read or changed.

Do not keep samples based only on labels.

## Counterfactual Wrong-Repo Construction

This source reuses existing positive ARB samples and pairs them with plausible but incorrect repositories.

### Procedure

1. Select a positive ARB sample with a realistic query.
2. Choose a wrong repository with similar language, framework, or domain.
3. Use the wrong repository corpus and a valid base commit.
4. Set `gold.files = []` and `reason = counterfactual_wrong_repo`.
5. Verify the query does not make the wrong repository obviously impossible.

### Pairing Rules

Good pairs:

- Python data-library query with another Python data-library repository.
- Rust CLI trace with another Rust CLI repository.
- JavaScript framework query with another JavaScript framework repository.

Bad pairs:

- Rust query with Python repository.
- CLI query with unrelated frontend repository.
- Query containing source repository path, project name, or unique package identifiers that make the mismatch trivial.

Counterfactual samples must be clearly marked and never presented as organic no-gold evidence.

## Crawling Pipeline

### Step 1: Raw Data Collection

For each target repository, collect:

- Closed issues.
- Issue comments.
- Pull requests.
- PR comments.
- Check suites and check runs.
- Failed CI logs.
- Labels.
- Closing events.
- Linked PR and commit references.

Suggested output:

```text
data/abstention/raw/{repo}/issues.jsonl
data/abstention/raw/{repo}/issue_comments.jsonl
data/abstention/raw/{repo}/prs.jsonl
data/abstention/raw/{repo}/check_runs.jsonl
data/abstention/raw/{repo}/ci_logs.jsonl
```

### Step 2: Keyword Candidate Mining

Use source-specific keyword lists only for recall. Keywords are not labels.

Suggested output:

```text
data/abstention/candidates/organic_candidates.jsonl
```

Each row should include:

- `repo`
- `source_type`
- `event_url`
- `query_text`
- `matched_keywords`
- `candidate_reason`
- `possible_base_sha`
- `linked_commits`
- `linked_prs`
- `evidence_snippets`

### Step 3: Automatic Exclusion

Automatically drop candidates when:

- There is a repository-local fix commit.
- A linked PR changes source, tests, config, dependency manifests, or lockfiles to solve the issue.
- The resolution is a dependency pin, lockfile update, test skip, CI config workaround, or local compatibility patch.
- Query text is too short or not a coding task.
- Base commit cannot be determined.
- Resolution evidence is missing.

Suggested output:

```text
data/abstention/candidates/organic_prefiltered.jsonl
```

### Step 4: Query Construction

Construct query text from the original workflow signal only.

For CI no-gold:

```json
{
  "source": "ci_log",
  "text": "Command: ...\nFailure excerpt: ...\nPR title: ..."
}
```

For issue no-gold:

```json
{
  "source": "issue",
  "text": "Issue title: ...\nIssue body excerpt: ...\nError excerpt: ..."
}
```

Never include the resolution evidence in the query. Evidence belongs in metadata and audit packets only.

### Step 5: Evidence Packet Generation

Generate an audit packet for every prefiltered candidate.

Suggested schema:

```json
{
  "id": "abstention_candidate__...",
  "repo": "owner/repo",
  "base_commit": "...",
  "query": {
    "source": "ci_log | issue | counterfactual_wrong_repo",
    "text": "..."
  },
  "proposed_no_gold_reason": "flaky_ci | upstream_dependency | external_service | user_error | counterfactual_wrong_repo",
  "evidence": {
    "source_url": "...",
    "resolution_comments": ["..."],
    "rerun_status": "...",
    "linked_commits": [],
    "why_no_local_fix": "..."
  },
  "audit_fields": {
    "verdict": "",
    "notes": "",
    "has_local_gold_files": []
  }
}
```

Suggested output:

```text
data/abstention/audit/abstention_audit_packet.jsonl
data/abstention/audit/abstention_audit_packet.csv
```

## Human Audit

Only `valid_no_gold` samples enter the clean set.

### Verdicts

- `valid_no_gold`: evidence supports that no repository-local context is needed.
- `has_local_gold`: at least one local file should be read or changed.
- `ambiguous`: evidence is insufficient.
- `too_easy_irrelevant`: query is obviously unrelated or not a plausible coding-agent query.
- `misconstructed`: query, base commit, repository, or evidence packet is malformed.

### Audit Questions

For every candidate, answer:

1. Is the query a real coding-workflow signal?
2. At the chosen repo/base commit, does any local file contain necessary context?
3. If an agent reads top-k files from this repo, would that likely help or waste context?
4. Does the resolution evidence clearly show the problem is not in this repository?
5. Is there any local dependency, config, CI, test, or source workaround?
6. For counterfactual samples, is the wrong repository plausible rather than trivially unrelated?

## Final Clean Schema

```json
{
  "id": "abstention__...",
  "task_type": "abstention",
  "repo": "owner/repo",
  "base_commit": "...",
  "query": {
    "source": "ci_log | issue | counterfactual_wrong_repo",
    "text": "..."
  },
  "gold": {
    "files": [],
    "no_gold": true,
    "reason": "flaky_ci | upstream_dependency | external_service | user_error | counterfactual_wrong_repo"
  },
  "metadata": {
    "source_url": "...",
    "evidence_urls": ["..."],
    "evidence_summary": "...",
    "organic": true
  },
  "audit": {
    "verdict": "valid_no_gold",
    "notes": "..."
  }
}
```

## Pilot Success Criteria

The crawling stage is successful if:

- Prefiltered candidates: at least 150.
- Audit packet candidates: at least 80.
- Final clean no-gold samples: at least 50.
- `valid_no_gold` rate after audit: at least 90%.
- `has_local_gold = 0` in the final clean subset.
- Every sample has resolution evidence.
- No query contains its resolution answer.
- Organic and counterfactual samples are reported separately.
- Counterfactual samples are no more than half of the final clean set.

## Recommended Execution Order

1. Mine flaky / infra CI candidates first because SHA and no-local-fix evidence are clearest.
2. Generate counterfactual wrong-repo candidates from existing positive ARB samples because this has low engineering cost.
3. Mine upstream dependency and user-error issues last because they require heavier manual judgment.

The guiding standard is: a sample is not no-gold because no one labeled gold; it is no-gold because there is evidence that an agent should abstain from searching for repository-local files.
