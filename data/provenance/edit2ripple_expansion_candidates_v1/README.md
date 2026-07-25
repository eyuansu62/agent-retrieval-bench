# edit2ripple expansion provenance

This directory preserves the internal candidate-selection evidence that led to
the released 58-sample `edit2ripple` subset. It is provenance, not a benchmark
release, and must not be included automatically in a public export.

## Contents

- `candidates.jsonl`: 27 expansion candidates generated in legacy commit
  `6d713d2`.
- `selection_summary.json`: the candidate-source and sampling policy recorded
  by that commit.
- `audit_samples.csv`: the completed AWS review recovered on 2026-07-25.

The recovered audit has 13 `valid`, 13 `noisy`, and 1 `ambiguous` verdict.
Thirteen rows have `keep=true`. Five of these 27 candidate IDs appear in the
final 58-sample release. The final release also contains all 50 IDs from the
earlier strict pilot and three IDs produced by later cleanup or expansion.

## Integrity

```text
032f0bc4397a78f9dee83184ec0937962c9d4d086de61bd4d3a38d481ef8624b  audit_samples.csv
b9fd6d6f0597ea08ddf7b1d3c0919cc475da94ce86ee1d74c1c4c683ca5ce015  candidates.jsonl
b0aea93ef93ea11c80cd465d3a3cd24785ee3153e490482b0e55226b7eb2ecc5  selection_summary.json
```

The released subset under `data/benchmark/v2_edit2ripple` or its Hugging Face
bundle remains the evaluation source of truth.
