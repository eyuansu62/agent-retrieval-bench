# Abstention Pilot Report

- Total clean samples: `82`
- Organic samples: `50`
- Counterfactual samples: `32`
- Valid rate: `1.0`
- Status: `ready`

## Gates

- `audit_packet_candidates_ge_80`: `True`
- `final_clean_ge_50`: `True`
- `valid_rate_ge_90`: `True`
- `has_local_gold_zero`: `True`
- `every_sample_has_resolution_evidence`: `True`
- `no_query_contains_resolution_answer`: `True`
- `organic_and_counterfactual_reported_separately`: `True`
- `counterfactual_no_more_than_half`: `True`
- `schema_valid`: `True`

## Verdicts

- `valid_no_gold`: `82`

## Reasons

- `counterfactual_wrong_repo`: `32`
- `external_service`: `7`
- `upstream_dependency`: `41`
- `user_error`: `2`
