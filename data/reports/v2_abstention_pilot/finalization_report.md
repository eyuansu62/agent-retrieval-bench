# Abstention Finalization Report

- Status: `complete`
- Exported clean samples: `True`
- Audit status: `ready_to_export`
- Pilot status: `ready`
- Completion status: `complete`
- Completion status counts: `{"passed": 20}`

## Next Action

- No required action remains; final clean pilot satisfies the plan gates.

## Audit Gates

- `all_packets_reviewed`: `True`
- `valid_rate_ge_90`: `True`
- `invalid_verdicts_zero`: `True`
- `missing_packet_ids_zero`: `True`
- `balanced_clean_ge_50`: `True`
- `balanced_counterfactual_no_more_than_half`: `True`

## Pilot Gates

- `audit_packet_candidates_ge_80`: `True`
- `final_clean_ge_50`: `True`
- `valid_rate_ge_90`: `True`
- `has_local_gold_zero`: `True`
- `every_sample_has_resolution_evidence`: `True`
- `no_query_contains_resolution_answer`: `True`
- `organic_and_counterfactual_reported_separately`: `True`
- `counterfactual_no_more_than_half`: `True`
- `schema_valid`: `True`

## Outputs

- `clean_dir`: `data/benchmark/v2_abstention_pilot`
- `clean_samples`: `data/benchmark/v2_abstention_pilot/abstention.jsonl`
- `audit_report`: `data/abstention/audit_clean_probe_v1/abstention_audit_verdict_report.json`
- `pilot_report`: `data/reports/v2_abstention_pilot/pilot_report.json`
- `completion_report`: `data/abstention/audit_clean_probe_v1/abstention_completion_audit.json`
- `finalization_report`: `data/reports/v2_abstention_pilot/finalization_report.json`
