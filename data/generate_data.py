"""
Synthetic Data Generator for Bank Statements and Internal Ledger

Generates synthetic bank_statement.csv and internal_ledger.csv that link
to real Razorpay settlements (where available) or are fully synthetic
(if no real settlements exist).

Deliberately injects realistic messiness:
- Exact matches (majority)
- Date-format mismatches (DD/MM vs MM/DD, +/-1 day settlement lag)
- Rounding/fee-deduction mismatches (settlement = amount - fee)
- Partial refunds (one ledger row maps to two settlement rows)
- Duplicate entries (same amount, different order — must NOT match)
- Orphaned records with no counterpart (true exceptions)
- Currency/formatting noise (commas, symbols, whitespace)

Generates ground_truth.json answer key for metrics validation.
"""

import os
import json
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from faker import Faker
from dotenv import load_dotenv

load_dotenv()

fake = Faker('en_IN')  # Indian locale for realistic data


class SyntheticDataGenerator:
    """Generate synthetic bank and ledger data linked to real settlements."""
    
    def __init__(self, seed: int = 42, messiness_ratio: float = 0.25):
        """
        Args:
            seed: Random seed for reproducibility
            messiness_ratio: Fraction of records with injected issues (0.0-1.0)
        """
        random.seed(seed)
        fake.seed_instance(seed)
        self.seed = seed
        self.messiness_ratio = messiness_ratio
        self.ground_truth: List[Dict[str, Any]] = []
        
    def generate(
        self,
        num_records: int = 80,
        settlements_df: Optional[pd.DataFrame] = None,
        output_dir: str = "data",
        force_disagreement: bool = False
    ) -> tuple:
        """
        Generate synthetic bank statement and internal ledger data.
        
        Args:
            num_records: Target number of records to generate
            settlements_df: Optional DataFrame from Razorpay API (for linking)
            output_dir: Directory to write CSV files
            force_disagreement: If True, injects an organic disagreement candidate edge case
            
        Returns:
            Tuple of (bank_df, ledger_df, ground_truth_list)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine how many records to base on real settlements vs synthetic
        real_settlements = []
        if settlements_df is not None and len(settlements_df) > 0:
            real_settlements = settlements_df.to_dict('records')
        
        num_real = min(len(real_settlements), int(num_records * 0.6))
        num_synthetic = num_records - num_real
        
        print(f"Generating {num_records} records:")
        print(f"  - {num_real} linked to real Razorpay settlements")
        print(f"  - {num_synthetic} fully synthetic")
        
        settlement_records = []
        bank_records = []
        ledger_records = []
        
        # Include existing real settlements
        for i, settlement in enumerate(real_settlements[:num_real]):
            settlement_records.append(settlement)
            bank_rec, ledger_rec, gt_entry = self._generate_linked_record(
                settlement, index=i
            )
            bank_records.append(bank_rec)
            ledger_records.append(ledger_rec)
            self.ground_truth.append(gt_entry)
        
        # Generate fully synthetic 3-way records
        for i in range(num_synthetic):
            idx = num_real + i
            order_id = f"ORD_SYNTH_{idx:04d}"
            payment_id = f"PAY_SYNTH_{idx:04d}"
            utr = f"UTR{random.randint(100000, 999999)}"
            amount = round(random.uniform(100, 50000), 2)
            fee = round(amount * random.uniform(0.015, 0.025), 2)
            tax = round(fee * 0.18, 2)
            base_date = datetime.now() - timedelta(days=random.randint(0, 30))
            settled_at = base_date.isoformat()
            
            # Create synthetic settlement row
            sett_rec = {
                'entity_id': f"sett_{idx:04d}",
                'type': 'payment',
                'payment_id': payment_id,
                'order_id': order_id,
                'amount': amount,
                'fee': fee,
                'tax': tax,
                'currency': 'INR',
                'settled_amount': round(amount - fee - tax, 2),
                'settlement_id': f"set_{idx:04d}",
                'settlement_utr': utr,
                'created_at': base_date.isoformat(),
                'settled_at': (base_date + timedelta(days=1)).isoformat(),
                'method': random.choice(['UPI', 'Card', 'Netbanking']),
                'card_network': random.choice(['VISA', 'Mastercard', 'RuPay']),
                'source': 'synthetic'
            }
            settlement_records.append(sett_rec)
            
            messiness_type = self._decide_messiness()
            if force_disagreement and i == num_synthetic // 2:
                messiness_type = "date_lag"
            
            bank_rec, ledger_rec, gt_entry = self._create_matched_pair(
                order_id=order_id,
                payment_id=payment_id,
                utr=utr,
                amount=amount,
                fee=fee + tax,
                settled_at=settled_at,
                messiness_type=messiness_type,
                source='synthetic'
            )
            bank_records.append(bank_rec)
            ledger_records.append(ledger_rec)
            self.ground_truth.append(gt_entry)
        
        settlements_out_df = pd.DataFrame(settlement_records)
        bank_df = pd.DataFrame(bank_records)
        ledger_df = pd.DataFrame(ledger_records)
        
        # Save to CSV
        settlements_path = os.path.join(output_dir, "settlements_live.csv")
        bank_path = os.path.join(output_dir, "bank_statement.csv")
        ledger_path = os.path.join(output_dir, "internal_ledger.csv")
        gt_path = os.path.join(output_dir, "ground_truth.json")
        
        settlements_out_df.to_csv(settlements_path, index=False)
        bank_df.to_csv(bank_path, index=False)
        ledger_df.to_csv(ledger_path, index=False)
        
        with open(gt_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2, default=str)
        
        print(f"\nGenerated files:")
        print(f"  - {settlements_path}")
        print(f"  - {bank_path}")
        print(f"  - {ledger_path}")
        print(f"  - {gt_path}")
        
        return bank_df, ledger_df, self.ground_truth
    
    def _generate_linked_record(
        self,
        settlement: Dict[str, Any],
        index: int
    ) -> tuple:
        """Generate bank/ledger records linked to a real settlement."""
        order_id = settlement.get('order_id', f'ORD_SYNTH_{index:04d}')
        payment_id = settlement.get('payment_id', f'PAY_SYNTH_{index:04d}')
        utr = settlement.get('settlement_utr', f'UTR{random.randint(100000, 999999)}')
        amount = settlement.get('amount', fake.random_number(digits=5))
        fee = settlement.get('fee', amount * 0.02)
        settled_at = settlement.get('settled_at', datetime.now().isoformat())
        
        # Decide what type of messiness to inject (if any)
        messiness_type = self._decide_messiness()
        
        bank_rec, ledger_rec, gt_entry = self._create_matched_pair(
            order_id=order_id,
            payment_id=payment_id,
            utr=utr,
            amount=amount,
            fee=fee,
            settled_at=settled_at,
            messiness_type=messiness_type,
            source='razorpay_api'
        )
        
        return bank_rec, ledger_rec, gt_entry
    
    def _generate_fully_synthetic_record(self, index: int) -> tuple:
        """Generate a fully synthetic bank/ledger record pair."""
        order_id = f"ORD_SYNTH_{index:04d}"
        payment_id = f"PAY_SYNTH_{index:04d}"
        utr = f"UTR{random.randint(100000, 999999)}"
        amount = round(random.uniform(100, 50000), 2)
        fee = round(amount * random.uniform(0.01, 0.03), 2)
        base_date = datetime.now() - timedelta(days=random.randint(0, 30))
        settled_at = base_date.isoformat()
        
        messiness_type = self._decide_messiness()
        
        bank_rec, ledger_rec, gt_entry = self._create_matched_pair(
            order_id=order_id,
            payment_id=payment_id,
            utr=utr,
            amount=amount,
            fee=fee,
            settled_at=settled_at,
            messiness_type=messiness_type,
            source='synthetic'
        )
        
        return bank_rec, ledger_rec, gt_entry
    
    def _decide_messiness(self) -> str:
        """Decide what type of messiness to inject."""
        rand = random.random()
        
        if rand > self.messiness_ratio:
            return "exact_match"
        
        messiness_options = [
            "date_lag",           # +/-1 day settlement lag
            "fee_deduction",      # Settlement shows net, ledger shows gross
            "date_format",        # DD/MM vs MM/DD confusion
            "rounding_diff",      # Minor rounding difference (<1%)
            "duplicate_suspect",  # Same amount, different order
            "orphan_bank",        # Bank record with no ledger counterpart
            "orphan_ledger",      # Ledger record with no bank counterpart
            "partial_refund",     # One ledger maps to multiple settlements
            "formatting_noise"    # Commas, symbols, whitespace
        ]
        
        return random.choice(messiness_options)
    
    def _create_matched_pair(
        self,
        order_id: str,
        payment_id: str,
        utr: str,
        amount: float,
        fee: float,
        settled_at: str,
        messiness_type: str,
        source: str
    ) -> tuple:
        """Create a bank/ledger record pair with specified messiness."""
        
        base_date = datetime.fromisoformat(settled_at.replace('Z', '+00:00').split('T')[0])
        
        # Default: exact match
        bank_amount = amount - fee  # Net settlement
        ledger_amount = amount  # Gross amount
        bank_date = base_date
        ledger_date = base_date
        bank_narration = f"NEFT/IMPS - {utr} - {order_id}"
        ledger_ref = order_id
        
        gt_should_match = True
        gt_root_cause = "exact_match"
        gt_notes = "Clean match"
        
        bank_utr = utr
        bank_ref = payment_id
        
        if messiness_type == "date_lag":
            bank_date = base_date + timedelta(days=random.choice([-2, 2]))
            bank_utr = ""  # Strip exact UTR to exercise fuzzy/exception logic
            bank_ref = ""
            gt_root_cause = "timing_lag"
            gt_notes = "Settlement lag of 2 days"
            
        elif messiness_type == "fee_deduction":
            # Bank shows net, ledger shows gross
            bank_utr = ""
            bank_ref = ""
            gt_root_cause = "rounding"
            gt_notes = "Fee deduction mismatch - bank shows net, ledger shows gross"
            
        elif messiness_type == "date_format":
            # Simulate date ambiguity
            bank_date = base_date + timedelta(days=random.randint(3, 10))
            bank_utr = ""
            bank_ref = ""
            gt_root_cause = "timing_lag"
            gt_notes = "Date format ambiguity"
            
        elif messiness_type == "rounding_diff":
            bank_amount = bank_amount + random.uniform(-0.45, 0.45)
            bank_utr = ""
            bank_ref = ""
            gt_root_cause = "rounding"
            gt_notes = "Minor rounding difference"
            
        elif messiness_type == "duplicate_suspect":
            # Create a record that looks similar but shouldn't match
            order_id = f"ORD_DUP_{random.randint(1000, 9999)}"
            bank_utr = f"UTR_DUP_{random.randint(100000, 999999)}"
            bank_ref = f"PAY_DUP_{random.randint(1000, 9999)}"
            gt_should_match = False
            gt_root_cause = "duplicate_suspected"
            gt_notes = "Same amount, different order - should NOT match"
            
        elif messiness_type == "orphan_bank":
            # Bank record with no ledger counterpart
            ledger_amount = None
            ledger_ref = None
            ledger_order_id = f"ORD_ORPHAN_{random.randint(1000, 9999)}"
            order_id = ledger_order_id
            bank_utr = f"UTR_ORPHAN_{random.randint(100000, 999999)}"
            gt_should_match = False
            gt_root_cause = "no_counterpart"
            gt_notes = "Orphaned bank record"
            
        elif messiness_type == "orphan_ledger":
            # Ledger record with no bank counterpart
            bank_amount = None
            bank_narration = None
            bank_utr = ""
            bank_ref = ""
            gt_should_match = False
            gt_root_cause = "no_counterpart"
            gt_notes = "Orphaned ledger record"
            
        elif messiness_type == "partial_refund":
            gt_root_cause = "partial_refund"
            gt_notes = "Partial refund - may need multi-record matching"
            
        elif messiness_type == "formatting_noise":
            # Add formatting noise to narration
            bank_utr = ""
            bank_ref = ""
            bank_narration = f"Rs. {amount:,.2f} / {utr} / {order_id}"
            gt_root_cause = "currency_formatting"
            gt_notes = "Formatting noise in narration"
        
        # Build bank record
        bank_rec = {
            'txn_id': f"TXN_{random.randint(100000, 999999)}",
            'amount': round(bank_amount, 2) if bank_amount is not None else None,
            'value_date': bank_date.strftime('%Y-%m-%d'),
            'narration': bank_narration if bank_narration else '',
            'utr': bank_utr,
            'reference': bank_ref,
            'source': source
        }
        
        # Build ledger record
        ledger_rec = {
            'order_id': order_id,
            'expected_amount': round(ledger_amount, 2) if ledger_amount is not None else None,
            'order_date': ledger_date.strftime('%Y-%m-%d'),
            'customer_ref': f"CUST_{random.randint(1000, 9999)}",
            'status': 'completed' if ledger_amount is not None else 'pending',
            'payment_id': payment_id,
            'source': source
        }
        
        # Build ground truth entry
        gt_entry = {
            'bank_txn_id': bank_rec['txn_id'],
            'ledger_order_id': order_id,
            'payment_id': payment_id,
            'utr': utr,
            'should_match': gt_should_match,
            'root_cause': gt_root_cause,
            'notes': gt_notes,
            'messiness_type': messiness_type,
            'bank_source': source,
            'ledger_source': source
        }
        
        return bank_rec, ledger_rec, gt_entry


def generate_data_cli():
    """CLI entry point for generating synthetic data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic bank statement and internal ledger data"
    )
    parser.add_argument("--records", type=int, default=80,
                        help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--messiness", type=float, default=0.25,
                        help="Fraction of records with injected issues (0.0-1.0)")
    parser.add_argument("--settlements", type=str, default="data/settlements_live.csv",
                        help="Path to settlements CSV for linking")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Load settlements if available
    settlements_df = None
    if os.path.exists(args.settlements):
        settlements_df = pd.read_csv(args.settlements)
        print(f"Loaded {len(settlements_df)} settlements from {args.settlements}")
    else:
        print(f"No settlements file found at {args.settlements}")
        print("Will generate fully synthetic data.")
    
    generator = SyntheticDataGenerator(seed=args.seed, messiness_ratio=args.messiness)
    generator.generate(
        num_records=args.records,
        settlements_df=settlements_df,
        output_dir=args.output_dir
    )
    
    return 0


def generate_all_datasets(
    num_records: int = 50,
    seed: int = 42,
    messiness_factor: float = 0.25,
    link_to_settlements: bool = False,
    settlements_path: str = "data/settlements_live.csv",
    output_dir: str = "data",
    force_disagreement: bool = False
):
    """Programmatic API for generating all datasets.
    
    This function is used by the FastAPI endpoint to generate data.
    
    Args:
        num_records: Number of records to generate
        seed: Random seed for reproducibility
        messiness_factor: Fraction of records with injected issues
        link_to_settlements: Whether to link to real Razorpay settlements
        settlements_path: Path to settlements CSV if linking
        output_dir: Output directory for generated files
        force_disagreement: Force at least one LLM/deterministic disagreement case
    """
    # Load settlements if available and requested
    settlements_df = None
    if link_to_settlements and os.path.exists(settlements_path):
        settlements_df = pd.read_csv(settlements_path)
        print(f"Loaded {len(settlements_df)} settlements from {settlements_path}")
    else:
        print("No settlements file found or linking disabled. Generating fully synthetic data.")
    
    generator = SyntheticDataGenerator(seed=seed, messiness_ratio=messiness_factor)
    generator.generate(
        num_records=num_records,
        settlements_df=settlements_df,
        output_dir=output_dir,
        force_disagreement=force_disagreement
    )
    
    return generator.ground_truth


if __name__ == "__main__":
    exit(generate_data_cli())

