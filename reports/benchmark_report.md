# Adversarial Benchmark Report

**Generated:** 2026-09-04T10:14:36.467505Z
**Accuracy:** 88.0%

## Test Cases

| Case ID | Description | Expected | Actual | Passed |
|---------|-------------|----------|--------|--------|
| case_1_exact | Clean exact 3-way match | `matched` | `matched` | ✅ |
| case_2_fee_adjust | Bank has net, ledger has gross, fuzzy match catches fee | `matched` | `matched` | ✅ |
| case_3_date_lag | Settlement delayed by 2 days, exact amount | `matched` | `matched` | ✅ |
| case_4_missing_bank | Missing bank leg, should not match completely | `low_confidence` | `matched_llm_verified` | ❌ |
| case_5_formatting | Messy string formatting in amounts | `matched` | `llm_parse_error` | ❌ |
| case_6_rounding | Tiny rounding discrepancy in bank amount | `matched` | `matched` | ✅ |
| case_7_strict_disagree | Amount diff is 10%, should reject | `llm_deterministic_disagreement` | `flagged_for_review` | ❌ |
| case_8_generic | Generic matching case 8 | `matched` | `matched` | ✅ |
| case_9_generic | Generic matching case 9 | `matched` | `matched` | ✅ |
| case_10_generic | Generic matching case 10 | `matched` | `matched` | ✅ |
| case_11_generic | Generic matching case 11 | `matched` | `matched` | ✅ |
| case_12_generic | Generic matching case 12 | `matched` | `matched` | ✅ |
| case_13_generic | Generic matching case 13 | `matched` | `matched` | ✅ |
| case_14_generic | Generic matching case 14 | `matched` | `matched` | ✅ |
| case_15_generic | Generic matching case 15 | `matched` | `matched` | ✅ |
| case_16_generic | Generic matching case 16 | `matched` | `matched` | ✅ |
| case_17_generic | Generic matching case 17 | `matched` | `matched` | ✅ |
| case_18_generic | Generic matching case 18 | `matched` | `matched` | ✅ |
| case_19_generic | Generic matching case 19 | `matched` | `matched` | ✅ |
| case_20_generic | Generic matching case 20 | `matched` | `matched` | ✅ |
| case_21_generic | Generic matching case 21 | `matched` | `matched` | ✅ |
| case_22_generic | Generic matching case 22 | `matched` | `matched` | ✅ |
| case_23_generic | Generic matching case 23 | `matched` | `matched` | ✅ |
| case_24_generic | Generic matching case 24 | `matched` | `matched` | ✅ |
| case_25_generic | Generic matching case 25 | `matched` | `matched` | ✅ |
