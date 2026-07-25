# Abstention Completion Audit

- Status: `complete`
- Requirement status counts: `{"passed": 20}`

## Current State

- Prefiltered candidates: `82`
- Audit packet candidates: `82`
- Audit reviewed: `82`
- Audit pending: `0`
- Audit valid rate: `1.0`
- Final clean exists: `True`
- Final clean total: `82`
- Final organic/counterfactual: `50` / `32`

## Requirements

- `passed` `two_candidate_pools_built`: prefiltered_by_pool={'counterfactual_wrong_repo': 146, 'organic_no_gold': 64}
- `passed` `prefiltered_candidates_ge_150`: prefiltered_candidates=210
- `passed` `audit_packet_candidates_ge_80`: audit_packet_candidates=82
- `passed` `every_packet_has_resolution_evidence`: crawling status validates resolution evidence for every packet
- `passed` `no_packet_query_contains_resolution_answer`: crawling status validates query/evidence separation
- `passed` `counterfactual_queries_avoid_source_identity`: source_identity_leaks=0
- `passed` `counterfactual_pairing_metadata_present`: missing_pairing_metadata=0
- `passed` `organic_and_counterfactual_reported_separately`: audit_packets_by_pool={'counterfactual_wrong_repo': 32, 'organic_no_gold': 50}
- `passed` `manual_audit_all_packets_reviewed`: reviewed=82, pending=0
- `passed` `valid_no_gold_rate_after_audit_ge_90`: valid_rate=1.0
- `passed` `final_clean_no_gold_samples_ge_50`: final_clean_samples=82
- `passed` `recommended_final_total_60_to_100`: final_clean_samples=82
- `passed` `recommended_organic_30_to_50`: organic_final=50
- `passed` `recommended_counterfactual_30_to_50`: counterfactual_final=32
- `passed` `has_local_gold_zero_in_final_clean`: has_local_gold_final=0
- `passed` `final_every_sample_has_resolution_evidence`: final pilot validates evidence_summary for every clean sample
- `passed` `final_no_query_contains_resolution_answer`: final pilot validates query/evidence separation
- `passed` `final_counterfactual_no_more_than_half`: counterfactual_share=0.3902439024390244
- `passed` `final_schema_valid`: invalid_rows=0
- `passed` `manual_audit_worklist_ready`: core={'by_reason': {'counterfactual_wrong_repo': 36, 'external_service': 10, 'upstream_dependency': 52, 'user_error': 2}, 'by_repo': {'HypothesisWorks/hypothesis': 1, 'astral-sh/ruff': 6, 'caddyserver/caddy': 5, 'clap-rs/clap': 2, 'eslint/eslint': 4, 'gin-gonic/gin': 3, 'huggingface/diffusers': 9, 'huggingface/transformers': 4, 'microsoft/playwright': 4, 'mockito/mockito': 2, 'numpy/numpy': 8, 'pallets/click': 2, 'pre-commit/pre-commit': 2, 'pypa/pip': 6, 'pytest-dev/pytest': 13, 'scrapy/scrapy': 8, 'tokio-rs/tokio': 5, 'tox-dev/tox': 1, 'vitejs/vite': 13, 'vuejs/core': 2}, 'counterfactual': 36, 'counterfactual_share': 0.36, 'organic': 64, 'total': 100}

## Next Required Action

- No required action remains; the plan gates are satisfied.
