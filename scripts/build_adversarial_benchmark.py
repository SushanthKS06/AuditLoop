import json
import os
from datetime import datetime, timedelta

def create_benchmark():
    base_date = datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0)
    cases = []
    
    def add_case(case_id, description, expected_status, sett=None, bank=None, ledger=None):
        cases.append({
            "case_id": case_id,
            "description": description,
            "expected_status": expected_status,
            "settlement": sett,
            "bank": bank,
            "ledger": ledger
        })

    # Case 1: Exact Match
    add_case(
        "case_1_exact", "Clean exact 3-way match", "matched",
        sett={"amount": 1000.0, "settled_amount": 980.0, "fee": 20.0, "settlement_utr": "UTR123", "order_id": "ORD123", "payment_id": "PAY123", "settled_at": base_date.isoformat()},
        bank={"amount": 980.0, "utr": "UTR123", "reference": "PAY123", "value_date": base_date.isoformat()},
        ledger={"expected_amount": 1000.0, "order_id": "ORD123", "payment_id": "PAY123", "order_date": base_date.isoformat()}
    )
    
    # Case 2: Fee Adjusted Match
    add_case(
        "case_2_fee_adjust", "Bank has net, ledger has gross, fuzzy match catches fee", "matched",
        sett={"amount": 1500.0, "settled_amount": 1470.0, "fee": 30.0, "settlement_utr": "", "order_id": "ORD222", "payment_id": "PAY222", "settled_at": base_date.isoformat()},
        bank={"amount": 1470.0, "utr": "", "reference": "PAY222", "value_date": base_date.isoformat()},
        ledger={"expected_amount": 1500.0, "order_id": "ORD222", "payment_id": "PAY222", "order_date": base_date.isoformat()}
    )
    
    # Case 3: Date Lag (2 days)
    add_case(
        "case_3_date_lag", "Settlement delayed by 2 days, exact amount", "matched",
        sett={"amount": 2000.0, "settled_amount": 1960.0, "fee": 40.0, "settlement_utr": "UTR333", "order_id": "ORD333", "payment_id": "PAY333", "settled_at": base_date.isoformat()},
        bank={"amount": 1960.0, "utr": "UTR333", "reference": "PAY333", "value_date": (base_date + timedelta(days=2)).isoformat()},
        ledger={"expected_amount": 2000.0, "order_id": "ORD333", "payment_id": "PAY333", "order_date": base_date.isoformat()}
    )
    
    # Case 4: Missing Bank (Orphan Ledger + Settlement)
    add_case(
        "case_4_missing_bank", "Missing bank leg, should not match completely", "low_confidence",
        sett={"amount": 3000.0, "settled_amount": 2940.0, "fee": 60.0, "settlement_utr": "UTR444", "order_id": "ORD444", "payment_id": "PAY444", "settled_at": base_date.isoformat()},
        bank=None,
        ledger={"expected_amount": 3000.0, "order_id": "ORD444", "payment_id": "PAY444", "order_date": base_date.isoformat()}
    )
    
    # Case 5: Formatting Noise
    add_case(
        "case_5_formatting", "Messy string formatting in amounts", "matched",
        sett={"amount": 4000.0, "settled_amount": 3920.0, "fee": 80.0, "settlement_utr": "UTR555", "order_id": "ORD555", "payment_id": "PAY555", "settled_at": base_date.isoformat()},
        bank={"amount": "(3,920.00)", "utr": "UTR555", "reference": "PAY555", "value_date": base_date.isoformat()},
        ledger={"expected_amount": "₹4,000.00", "order_id": "ORD555", "payment_id": "PAY555", "order_date": base_date.isoformat()}
    )
    
    # Case 6: Minor Rounding Error (<1%)
    add_case(
        "case_6_rounding", "Tiny rounding discrepancy in bank amount", "matched",
        sett={"amount": 5000.0, "settled_amount": 4900.0, "fee": 100.0, "settlement_utr": "UTR666", "order_id": "ORD666", "payment_id": "PAY666", "settled_at": base_date.isoformat()},
        bank={"amount": 4900.25, "utr": "UTR666", "reference": "PAY666", "value_date": base_date.isoformat()},
        ledger={"expected_amount": 5000.0, "order_id": "ORD666", "payment_id": "PAY666", "order_date": base_date.isoformat()}
    )
    
    # Case 7: Strict Disagreement (Amount diff > 5%)
    add_case(
        "case_7_strict_disagree", "Amount diff is 10%, should reject", "llm_deterministic_disagreement",
        sett={"amount": 10000.0, "settled_amount": 9800.0, "fee": 200.0, "settlement_utr": "", "order_id": "ORD777", "payment_id": "PAY777", "settled_at": base_date.isoformat()},
        bank={"amount": 8820.0, "utr": "", "reference": "PAY777", "value_date": base_date.isoformat()},
        ledger={"expected_amount": 10000.0, "order_id": "ORD777", "payment_id": "PAY777", "order_date": base_date.isoformat()}
    )

    # Generate total 25 cases (using generic loops for the remaining 18)
    for i in range(8, 26):
        add_case(
            f"case_{i}_generic", f"Generic matching case {i}", "matched",
            sett={"amount": 100.0 * i, "settled_amount": 98.0 * i, "fee": 2.0 * i, "settlement_utr": f"UTR{i}", "order_id": f"ORD{i}", "payment_id": f"PAY{i}", "settled_at": base_date.isoformat()},
            bank={"amount": 98.0 * i, "utr": f"UTR{i}", "reference": f"PAY{i}", "value_date": base_date.isoformat()},
            ledger={"expected_amount": 100.0 * i, "order_id": f"ORD{i}", "payment_id": f"PAY{i}", "order_date": base_date.isoformat()}
        )
        
    os.makedirs("data", exist_ok=True)
    with open("data/adversarial_benchmark.json", "w") as f:
        json.dump(cases, f, indent=2)

if __name__ == "__main__":
    create_benchmark()
