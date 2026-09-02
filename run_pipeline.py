"""
Main Reconciliation Pipeline

Orchestrates the full reconciliation flow:
1. Load data (settlements, bank, ledger)
2. Stage 1: Exact matching
3. Stage 2: Fuzzy matching  
4. Stage 3: LLM exception handling
5. Write audit trail
6. Compute metrics
7. Save results
"""

import os
import json
import argparse
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from data.fetch_settlements import RazorpayReconClient
from data.generate_data import SyntheticDataGenerator
from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher
from llm.client import create_client
from audit.store import AuditStore
from metrics.evaluate import MetricsEvaluator

load_dotenv()


class ReconciliationPipeline:
    """
    Main pipeline orchestrating deterministic + LLM-assisted reconciliation.
    
    Design principle: The LLM is called ONLY for exceptions from Stage 3.
    It proposes but never commits - all proposals are re-verified.
    """
    
    def __init__(self, use_llm: bool = True, force_disagreement_demo: bool = False):
        """
        Args:
            use_llm: Whether to invoke LLM for exceptions (default True)
            force_disagreement_demo: Ensure at least one disagreement case exists
        """
        self.use_llm = use_llm
        self.force_disagreement_demo = force_disagreement_demo
        
        # Initialize components
        self.audit_store = AuditStore()
        self.llm_client = create_client() if use_llm else None
        
        self.matcher = DeterministicMatcher(
            confidence_threshold=0.85,
            amount_threshold_pct=1.0,
            date_window_days=3
        )
        self.matcher.set_audit_callback(self._audit_callback)
        
        self.dispatcher = ExceptionDispatcher(llm_client=self.llm_client)
        self.dispatcher.set_audit_callback(self._audit_callback)
        
        self.evaluator = MetricsEvaluator()  # ground_truth_path reloaded after generation
        
        # Results storage
        self.all_results = []
        self.all_matches = []
    
    def _audit_callback(self, record: dict):
        """Callback for writing audit records."""
        self.audit_store.append(record)
    
    def run(
        self,
        settlements_path: str = "data/settlements_live.csv",
        bank_path: str = "data/bank_statement.csv",
        ledger_path: str = "data/internal_ledger.csv",
        generate_if_missing: bool = True,
        num_records: int = 80,
        seed: int = 42
    ) -> dict:
        """
        Run the full reconciliation pipeline.
        
        Args:
            settlements_path: Path to settlements CSV
            bank_path: Path to bank statement CSV
            ledger_path: Path to internal ledger CSV
            generate_if_missing: Generate synthetic data if files missing
            num_records: Number of synthetic records to generate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        print("="*60)
        print("AUDITLOOP RECONCILIATION PIPELINE")
        print("="*60)
        
        # Step 0: Load or generate data
        print("\n[Step 0] Loading data...")
        settlements_df, bank_df, ledger_df = self._load_or_generate_data(
            settlements_path, bank_path, ledger_path,
            generate_if_missing, num_records, seed
        )
        
        if settlements_df.empty:
            print("ERROR: No settlements data available.")
            return {"error": "No settlements data"}
        
        print(f"  Settlements: {len(settlements_df)} records")
        print(f"  Bank: {len(bank_df)} records")
        print(f"  Ledger: {len(ledger_df)} records")
        
        # Clear previous audit trail
        self.audit_store.clear()
        self.all_results = []
        self.all_matches = []
        
        # Step 1: Exact matching
        print("\n[Step 1] Running exact matching...")
        matched_df, unmatched_sett, unmatched_bank, unmatched_ledger, stage1_audits = \
            self.matcher.stage1_exact_match(settlements_df, bank_df, ledger_df)
        
        if not matched_df.empty:
            for _, row in matched_df.iterrows():
                self.all_matches.append({
                    **row.get('settlement', {}),
                    'match_type': 'exact',
                    'confidence': 1.0,
                    'final_status': 'matched'
                })
        
        print(f"  Exact matches: {len(matched_df)}")
        print(f"  Remaining unmatched: {len(unmatched_sett)} settlements")
        
        # Step 2: Fuzzy matching
        print("\n[Step 2] Running fuzzy matching...")
        fuzzy_matched, low_conf, unmatched_sett2, unmatched_bank2, unmatched_ledger2, stage2_audits = \
            self.matcher.stage2_fuzzy_match(unmatched_sett, unmatched_bank, unmatched_ledger)
        
        if not fuzzy_matched.empty:
            for _, row in fuzzy_matched.iterrows():
                self.all_matches.append({
                    **row.get('settlement', {}),
                    'match_type': row.get('match_type'),
                    'confidence': row.get('confidence'),
                    'final_status': 'matched'
                })
        
        print(f"  Fuzzy matches (high confidence): {len(fuzzy_matched)}")
        print(f"  Low confidence candidates: {len(low_conf)}")
        
        # Step 3: Exception handling via LLM
        print("\n[Step 3] Processing exceptions...")
        exceptions = self.matcher.get_exceptions(
            low_conf, unmatched_sett2, unmatched_bank2, unmatched_ledger2
        )
        print(f"  Exceptions to process: {len(exceptions)}")
        
        if exceptions and self.use_llm:
            processed_exceptions = self.dispatcher.process_exceptions(
                exceptions,
                force_disagreement_case=self.force_disagreement_demo
            )
            
            for exc in processed_exceptions:
                self.all_results.append(exc)
                
                # Count LLM-verified matches
                if exc.get('final_status') == 'matched_llm_verified':
                    self.all_matches.append({
                        **exc.get('settlement', {}),
                        'match_type': 'llm_verified',
                        'confidence': exc.get('llm_confidence'),
                        'final_status': 'matched_llm_verified'
                    })
            
            # Summarize exception outcomes
            status_counts = {}
            for exc in processed_exceptions:
                status = exc.get('final_status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("  Exception outcomes:")
            for status, count in status_counts.items():
                print(f"    - {status}: {count}")
        else:
            # No LLM - mark all exceptions as unresolved
            for exc in exceptions:
                exc['final_status'] = 'unresolved_exception'
                self.all_results.append(exc)
            print("  LLM disabled - all exceptions marked unresolved")
        
        # Step 4: Compile final results
        print("\n[Step 4] Compiling results...")
        
        # Add matched records to results
        for match in self.all_matches:
            self.all_results.append(match)
        
        # Save results
        results_path = "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        print(f"  Results saved to {results_path}")
        
        # Step 5: Compute metrics (always strict — no silent self-grading)
        print("\n[Step 5] Computing metrics...")
        metrics = self.evaluator.evaluate(
            self.all_results,
            coverage_mode="strict"
        )
        self.evaluator.print_summary(metrics)
        
        # Get audit summary
        audit_summary = self.audit_store.get_summary_stats()
        
        return {
            'results': self.all_results,
            'metrics': metrics,
            'audit_summary': audit_summary,
            'matches_count': len(self.all_matches),
            'exceptions_count': len(exceptions)
        }
    
    def _load_or_generate_data(
        self,
        settlements_path: str,
        bank_path: str,
        ledger_path: str,
        generate: bool,
        num_records: int,
        seed: int
    ) -> tuple:
        """Load existing data or generate synthetic data."""
        
        # Try to load settlements
        settlements_df = None
        if os.path.exists(settlements_path):
            settlements_df = pd.read_csv(settlements_path)
            print(f"  Loaded {len(settlements_df)} settlements from {settlements_path}")
        elif generate:
            # Try to fetch from API first
            try:
                client = RazorpayReconClient()
                settlements_df = client.fetch_recon(year=2026, month=9, day=1)
                if len(settlements_df) > 0:
                    settlements_df.to_csv(settlements_path, index=False)
                    print(f"  Fetched {len(settlements_df)} settlements from Razorpay API")
                else:
                    print("  API returned no settlements - will use synthetic data")
            except ValueError as e:
                print(f"  API unavailable: {e}")
                print("  Generating linked synthetic dataset")
        
        # If files are missing or row count does not match requested batch size, generate fresh synchronized datasets
        needs_generation = (
            generate and (
                settlements_df is None
                or not os.path.exists(bank_path)
                or not os.path.exists(ledger_path)
                or len(settlements_df) != num_records
            )
        )
        
        if needs_generation:
            generator = SyntheticDataGenerator(seed=seed, messiness_ratio=0.25)
            bank_df, ledger_df, _ = generator.generate(
                num_records=num_records,
                settlements_df=settlements_df if settlements_df is not None and len(settlements_df) == num_records else None,
                output_dir="data",
                force_disagreement=self.force_disagreement_demo
            )
            if os.path.exists(settlements_path):
                settlements_df = pd.read_csv(settlements_path)
            # Reload evaluator's ground truth from the freshly-written file so
            # that strict-mode coverage reflects the current batch (not a stale
            # 20-record file from a previous run).
            gt_path = os.path.join("data", "ground_truth.json")
            if os.path.exists(gt_path):
                self.evaluator.ground_truth_path = gt_path
                self.evaluator.ground_truth = self.evaluator._load_ground_truth()
                print(f"  Ground truth reloaded: "
                      f"{len(self.evaluator.ground_truth)} entries from {gt_path}")
        else:
            bank_df = pd.read_csv(bank_path) if os.path.exists(bank_path) else pd.DataFrame()
            ledger_df = pd.read_csv(ledger_path) if os.path.exists(ledger_path) else pd.DataFrame()
        
        return settlements_df if settlements_df is not None else pd.DataFrame(), bank_df, ledger_df


def run_pipeline_cli():
    """CLI entry point for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the AuditLoop reconciliation pipeline"
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM calls (all exceptions remain unresolved)")
    parser.add_argument("--force-disagreement", action="store_true",
                        help="Force at least one disagreement case for demo")
    parser.add_argument("--records", type=int, default=80,
                        help="Number of records to generate if needed")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--settlements", type=str, default="data/settlements_live.csv",
                        help="Path to settlements CSV")
    parser.add_argument("--bank", type=str, default="data/bank_statement.csv",
                        help="Path to bank statement CSV")
    parser.add_argument("--ledger", type=str, default="data/internal_ledger.csv",
                        help="Path to internal ledger CSV")
    
    args = parser.parse_args()
    
    pipeline = ReconciliationPipeline(
        use_llm=not args.no_llm,
        force_disagreement_demo=args.force_disagreement
    )
    
    results = pipeline.run(
        settlements_path=args.settlements,
        bank_path=args.bank,
        ledger_path=args.ledger,
        generate_if_missing=True,
        num_records=args.records,
        seed=args.seed
    )
    
    if 'error' in results:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(run_pipeline_cli())
