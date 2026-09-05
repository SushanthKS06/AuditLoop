# Adversarial Benchmark Report

**Accuracy:** 100.0%

## Test Cases

| Case ID | Description | Expected | Actual | Passed |
|---------|-------------|----------|--------|--------|
| case_01_exact | Clean exact 3-way match | `matched` | `matched` | ✅ |
| case_02_fuzzy_fee | Fee-adjusted fuzzy 3-way match | `matched` | `matched` | ✅ |
| case_03_missing_bank | Missing bank leg cannot fully match | `incomplete_counterparts` | `incomplete_counterparts` | ✅ |
| case_04_missing_ledger | Missing ledger leg cannot fully match | `incomplete_counterparts` | `incomplete_counterparts` | ✅ |
| case_05_missing_both | Both counterparts missing | `explained_no_resolution` | `explained_no_resolution` | ✅ |
| case_06_amount_mismatch | Amount mismatch with all legs present | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_07_currency_mismatch | Currency mismatch cannot match | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_08_formatting | Comma and rupee formatting | `matched` | `matched` | ✅ |
| case_09_accounting_negative | Accounting parentheses are negative | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_10_rounding | Sub-1% rounding | `matched` | `matched` | ✅ |
| case_11_duplicate_ids | Conflicting order IDs (duplicate suspect) | `incomplete_counterparts` | `incomplete_counterparts` | ✅ |
| case_12_date_lag | Two-day settlement lag | `matched` | `matched` | ✅ |
| case_13_whitespace_amount | Whitespace around amount | `matched` | `matched` | ✅ |
| case_14_orphan_bank_only | Orphan bank with no settlement | `explained_no_resolution` | `explained_no_resolution` | ✅ |
| case_15_orphan_ledger_only | Orphan ledger with no settlement | `explained_no_resolution` | `explained_no_resolution` | ✅ |
| case_16_llm_match_rejected | LLM MATCH rejected by verifier | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_17_llm_partial_flag | LLM proposes flag_for_human (partial/uncertain) | `flagged_for_review` | `flagged_for_review` | ✅ |
| case_18_malformed_llm | Malformed LLM JSON | `llm_parse_error` | `llm_parse_error` | ✅ |
| case_19_schema_invalid_llm | Schema-invalid LLM JSON | `llm_parse_error` | `llm_parse_error` | ✅ |
| case_20_llm_unavailable | LLM client returns invalid (unavailable) | `llm_parse_error` | `llm_parse_error` | ✅ |
| case_21_low_confidence_llm | Low-confidence LLM MATCH proposal | `low_confidence` | `low_confidence` | ✅ |
| case_22_prompt_injection | Prompt injection cannot bypass verifier | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_23_pii_still_matches | PII in narration still 3-way matches | `matched` | `matched` | ✅ |
| case_24_timestamp_boundary | Date at 3-day window edge | `matched` | `matched` | ✅ |
| case_25_large_amount | Very large amounts | `matched` | `matched` | ✅ |
| case_26_zero_amount | Zero amounts with agreeing IDs | `matched` | `matched` | ✅ |
| case_27_refund_negative | Refund negative bank vs positive ledger | `llm_deterministic_disagreement` | `llm_deterministic_disagreement` | ✅ |
| case_28_partial_refund_review | Partial-refund string case stays conservative | `flagged_for_review` | `flagged_for_review` | ✅ |
| case_29_prompt_reveal | Reveal-hidden-instructions narration | `matched` | `matched` | ✅ |
| case_30_reproducible_exact | Reproducible duplicate of exact match | `matched` | `matched` | ✅ |
