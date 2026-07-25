# Abstention Audit Verdict Report

- Audit packet candidates: `82`
- Audit rows: `82`
- Reviewed: `82`
- Pending: `0`
- Valid rate: `1.0`
- Status: `ready_to_export`

## Balanced Clean Preview

- Total: `82`
- Organic: `50`
- Counterfactual: `32`
- Counterfactual share: `0.3902439024390244`

## Gates

- `all_packets_reviewed`: `True`
- `valid_rate_ge_90`: `True`
- `invalid_verdicts_zero`: `True`
- `missing_packet_ids_zero`: `True`
- `balanced_clean_ge_50`: `True`
- `balanced_counterfactual_no_more_than_half`: `True`

## Verdicts

- `valid_no_gold`: `82`

## By Pool

- `counterfactual_wrong_repo`: `{"valid_no_gold": 32}`
- `organic_no_gold`: `{"valid_no_gold": 50}`

## By Reason

- `counterfactual_wrong_repo`: `{"valid_no_gold": 32}`
- `external_service`: `{"valid_no_gold": 7}`
- `upstream_dependency`: `{"valid_no_gold": 41}`
- `user_error`: `{"valid_no_gold": 2}`
