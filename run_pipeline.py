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
import sys
import json
import argparse
from typing import Optional

# Ensure standard output can handle UTF-8 on Windows cp1252 consoles without crashing
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
from dotenv import load_dotenv

from data.fetch_settlements import RazorpayReconClient
from data.generate_data import SyntheticDataGenerator
from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher
from engine.states import ReconciliationState
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
    
    def __init__(self, use_llm: bool = True, demo_disagreement_demo: bool = False, run_id: Optional[str] = None):
        """
        Args:
            use_llm: Whether to invoke LLM for exceptions (default True)
            demo_disagreement_demo: Ensure at least one disagreement case exists
            run_id: Unique identifier for this pipeline execution
        """
        self.use_llm = use_llm
        self.demo_disagreement_demo = demo_disagreement_demo
        import uuid
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        
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
    
    def _validate_path(self, path: str, allowed_names: set) -> None:
        """Validate input paths to prevent path traversal and arbitrary file reads."""
        if not path:
            return
            
        if ".." in path:
            raise ValueError(f"Path traversal detected: {path}")
            
        basename = os.path.basename(path)
        if basename not in allowed_names:
            raise ValueError(f"Filename '{basename}' is not in the allowed whitelist: {allowed_names}")
            
        real_path = os.path.realpath(path)
        # Check if the resolved path is inside the project's data directory
        data_dir = os.path.realpath("data")
        if not real_path.startswith(data_dir):
            raise ValueError(f"Path must resolve to inside the data/ directory: {path}")

    def run(
        self,
        settlements_path: str = "data/fixtures/settlements_live.csv",
        bank_path: str = "data/fixtures/bank_statement.csv",
        ledger_path: str = "data/fixtures/internal_ledger.csv",
        generate_if_missing: bool = True,
        num_records: int = 80,
        seed: int = 42,
        messiness_ratio: float = 0.25
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
            messiness_ratio: Fraction of synthetic records with injected issues (0.0-1.0).
                             Propagated from the API ``/generate`` endpoint so callers
                             control data quality without touching defaults.

        Returns:
            Dictionary with pipeline results and metrics
        """
        print("="*60)
        print("AUDITLOOP RECONCILIATION PIPELINE")
        print("="*60)
        
        if self.demo_disagreement_demo:
            print("\n[!] WARNING: FORCED DEMO CASE ENABLED [!]")
            print("This run includes at least one fabricated discrepancy to demonstrate")
            print("the LLM-vs-Deterministic conflict resolution UI.")
            print("="*60)
        
        # Validate paths before processing
        allowed_files = {"settlements_live.csv", "bank_statement.csv", "internal_ledger.csv"}
        self._validate_path(settlements_path, allowed_files)
        self._validate_path(bank_path, allowed_files)
        self._validate_path(ledger_path, allowed_files)

        # Step 0: Load or generate data
        print("\n[Step 0] Loading data...")
        settlements_df, bank_df, ledger_df = self._load_or_generate_data(
            settlements_path, bank_path, ledger_path,
            generate_if_missing, num_records, seed,
            messiness_ratio=messiness_ratio
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
                src = row.get('source') or row.get('settlement', {}).get('source') or 'synthetic'
                match = {
                    **row.get('settlement', {}),
                    'match_type': 'exact',
                    'confidence': 1.0,
                    'final_status': ReconciliationState.EXACT_MATCH.value,
                    'source': src,
                    'forced_demo_case': False
                }
                if isinstance(row.get('bank'), dict) and 'txn_id' in row['bank']:
                    match['bank_txn_id'] = row['bank']['txn_id']
                if isinstance(row.get('ledger'), dict) and 'order_id' in row['ledger']:
                    match['ledger_order_id'] = row['ledger']['order_id']
                self.all_matches.append(match)
        
        print(f"  Exact matches: {len(matched_df)}")
        print(f"  Remaining unmatched: {len(unmatched_sett)} settlements")
        
        # Step 2: Fuzzy matching
        print("\n[Step 2] Running fuzzy matching...")
        fuzzy_matched, low_conf, unmatched_sett2, unmatched_bank2, unmatched_ledger2, stage2_audits = \
            self.matcher.stage2_fuzzy_match(unmatched_sett, unmatched_bank, unmatched_ledger)
        
        if not fuzzy_matched.empty:
            for _, row in fuzzy_matched.iterrows():
                src = row.get('source') or row.get('settlement', {}).get('source') or 'synthetic'
                match = {
                    **row.get('settlement', {}),
                    'match_type': row.get('match_type'),
                    'confidence': row.get('confidence'),
                    'final_status': ReconciliationState.FUZZY_MATCH.value,
                    'source': src,
                    'forced_demo_case': False
                }
                if isinstance(row.get('bank'), dict) and 'txn_id' in row['bank']:
                    match['bank_txn_id'] = row['bank']['txn_id']
                if isinstance(row.get('ledger'), dict) and 'order_id' in row['ledger']:
                    match['ledger_order_id'] = row['ledger']['order_id']
                self.all_matches.append(match)
        
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
                demo_disagreement_case=self.demo_disagreement_demo
            )
            
            for exc in processed_exceptions:
                exc['source'] = exc.get('source') or (exc.get('settlement') or {}).get('source') or (exc.get('counterpart') or {}).get('source') or 'synthetic'
                exc['forced_demo_case'] = exc.get('forced_demo_case', False)
                self.all_results.append(exc)
                
                # Count LLM-verified matches
                if exc.get('final_status') == ReconciliationState.MATCHED_LLM_VERIFIED.value:
                    self.all_matches.append({
                        **exc.get('settlement', {}),
                        'match_type': 'llm_verified',
                        'confidence': exc.get('llm_confidence'),
                        'final_status': ReconciliationState.MATCHED_LLM_VERIFIED.value,
                        'source': exc['source'],
                        'forced_demo_case': False
                    })
            
            # Summarize exception outcomes
            status_counts = {}
            for exc in processed_exceptions:
                status = exc.get('final_status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                for status, count in status_counts.items():
                    print(f"    - {status}: {count}")
        else:
            # If no LLM, just add exceptions to results
            for exc in exceptions:
                exc['source'] = exc.get('source') or (exc.get('settlement') or {}).get('source') or (exc.get('counterpart') or {}).get('source') or 'synthetic'
                exc['final_status'] = ReconciliationState.UNRESOLVED_EXCEPTION.value
                self.all_results.append(exc)
            print("  LLM disabled - all exceptions marked unresolved")
        
        # Step 4: Compile final results
        print("\n[Step 4] Compiling results...")
        
        # Add matched records to results
        for match in self.all_matches:
            self.all_results.append(match)
        
        # Step 4.5: Transaction-level deduplication
        # A single real-world transaction must produce EXACTLY ONE transaction evaluation unit.
        # Bank/ledger orphans and duplicate suspects must be separated into their own event streams.
        print("\n[Step 4.5] Grouping and deduplicating canonical transactions...")
        pre_dedup_count = len(self.all_results)
        
        grouped_results = self._deduplicate_results(self.all_results)
        
        transaction_results = grouped_results['transaction_results']
        orphan_events = grouped_results['orphan_events']
        duplicate_events = grouped_results['duplicate_events']
        
        print(f"  Transaction Results: {len(transaction_results)}")
        print(f"  Orphan Events: {len(orphan_events)}")
        print(f"  Duplicate Events: {len(duplicate_events)}")

        # POST-DEDUP SANITY CHECK
        # Ensure no valid input settlement entity_ids were silently dropped
        if settlements_df is not None and not settlements_df.empty:
            output_entities = set()
            for r in transaction_results:
                for k in ['entity_id', 'settlement_id']:
                    if r.get(k): output_entities.add(str(r.get(k)).strip().lower())
                for nest in ['settlement', 'counterpart', 'bank', 'ledger']:
                    if isinstance(r.get(nest), dict):
                        for k in ['entity_id', 'settlement_id']:
                            if r[nest].get(k): output_entities.add(str(r[nest].get(k)).strip().lower())
                            
            import pandas as pd
            input_entities = set(str(e).strip().lower() for e in settlements_df['entity_id'] if pd.notna(e))
            missing = input_entities - output_entities
            if missing:
                raise RuntimeError(
                    f"Silent record loss detected! {len(missing)} settlements were dropped "
                    f"from final results completely. Missing entity_ids: {missing}"
                )
        
        # Save results
        results_dir = os.path.join("runtime", "runs", self.run_id)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.json")
        output_payload = {
            'transaction_results': transaction_results,
            'orphan_events': orphan_events,
            'duplicate_events': duplicate_events,
            'exception_events': []
        }
        with open(results_path, 'w') as f:
            json.dump(output_payload, f, indent=2, default=str)
        print(f"  Results saved to {results_path}")
        
        # Step 5: Computing metrics
        print("\n[Step 5] Computing metrics...")
        metrics = self.evaluator.evaluate(
            results=output_payload,
            input_transaction_ids=[
                str(e) for e in settlements_df['entity_id'].tolist()
            ] if 'entity_id' in settlements_df.columns else [
                str(e) for e in settlements_df['payment_id'].tolist()
            ] if 'payment_id' in settlements_df.columns else None,
            input_transaction_count=len(settlements_df),
        )
        self.evaluator.print_summary(metrics)
        
        # Get audit summary
        audit_summary = self.audit_store.get_summary_stats()
        
        return {
            'results': output_payload,
            'metrics': metrics,
            'audit_summary': audit_summary,
            'matches_count': len(self.all_matches),
            'exceptions_count': len(exceptions)
        }
    
    def _deduplicate_results(self, results: list) -> dict:
        """
        Group and classify results by canonical transaction identity.

        Canonical identity model (deterministic, explainable):
        ------------------------------------------------------
        * STRONG identifiers decide transaction identity: the input
          settlement's ``entity_id`` (else ``settlement_id``). One input
          settlement row = exactly one transaction evaluation unit.
        * ``payment_id`` / ``order_id`` are WEAK identifiers: shared values
          across distinct settlements never merge two transaction units;
          they are evidence, not identity. A weak-ID collision across two
          different ``entity_id`` values yields two separate transaction
          results (plus a duplicate-suspect/orphan event where applicable).
        * Bank/ledger legs (``txn_id``, ``utr``, ledger ``order_id``) are
          EVIDENCE identifiers: they attach to a transaction unit but never
          create one. Evidence without a settlement becomes an orphan event.
        * Exception occurrences (duplicate/orphan/conflict rows) are EVENT
          identifiers: they are routed to ``duplicate_events`` /
          ``orphan_events`` / ``exception_events`` and never emitted as
          additional transaction-level rows.

        Every transaction result is stamped with ``canonical_transaction_id``
        and ``evaluation_unit_id`` (identical values; the former names the
        financial concept, the latter the evaluation concept). Merges record
        ``_merged_from`` evidence (the statuses that were collapsed) so the
        decision stays auditable.
        """
        STATUS_SEVERITY = {
            'llm_deterministic_disagreement': 0,
            'incomplete_counterparts': 0,
            'unresolved_exception': 1,
            'explained_no_resolution': 2,
            'low_confidence': 3,
            'flagged_for_review': 3,
            'llm_error': 4,
            'llm_parse_error': 4,
            'llm_unavailable': 4,
            'llm_provider_failure': 4,
            'matched_llm_verified': 5,
            'fuzzy_match': 6,
            'exact_match': 6,
            'matched': 6,
        }

        def _severity(status: str) -> int:
            return STATUS_SEVERITY.get(status, 3)

        def _get_settlement_id(row: dict) -> Optional[str]:
            # STRONG identity first: entity_id / settlement_id of the input
            # settlement row. payment_id / order_id are only fallbacks for
            # rows that genuinely lack a strong identifier (e.g. legacy
            # exception payloads), never a reason to merge two distinct
            # strong identities.
            sett = row.get('settlement')
            if isinstance(sett, dict):
                strong = sett.get('entity_id') or sett.get('settlement_id')
                if strong and str(strong).strip():
                    return str(strong).strip().lower()
                weak = sett.get('payment_id') or sett.get('order_id')
                return str(weak).strip().lower() or None
            strong = row.get('entity_id') or row.get('settlement_id')
            if strong and str(strong).strip():
                return str(strong).strip().lower()
            # Top-level payment rows (matcher output) carry the settlement
            # fields flattened; payment_id fallback preserves them.
            weak = row.get('payment_id') or row.get('order_id')
            return str(weak).strip().lower() or None

        transaction_results = []
        orphan_events = []
        duplicate_events = []
        
        # Group by canonical settlement ID
        canonical_map = {}
        
        # Sort results deterministically by ID and then stringification to ensure stable processing
        # Since these are dicts, we sort by stringified representation to avoid unorderable dict errors
        sorted_results = sorted(results, key=lambda x: str(x))

        for r in sorted_results:
            final_status = r.get('final_status', '')
            t = r.get('type') or ''
            
            # Explicit duplicates
            if final_status in ('duplicate_suspect', 'rejected_duplicate'):
                duplicate_events.append(r)
                continue
                
            # Explicit orphans
            if ReconciliationState.is_orphan_event(t) or t in ('unmatched_bank', 'unmatched_ledger'):
                orphan_events.append(r)
                continue

            sett_id = _get_settlement_id(r)
            if not sett_id:
                # If there's truly no settlement ID, and it's not marked as an orphan, 
                # we still consider it an orphan/exception event
                orphan_events.append(r)
                continue
            
            if sett_id not in canonical_map:
                canonical_map[sett_id] = []
            canonical_map[sett_id].append(r)
            
        for sett_id, group in canonical_map.items():
            if len(group) == 1:
                row = dict(group[0])
                row['canonical_transaction_id'] = sett_id
                row['evaluation_unit_id'] = sett_id
                transaction_results.append(row)
            else:
                # We have multiple result rows for the exact same input settlement!
                # This could happen if the pipeline spawned multiple exceptions for the same settlement,
                # e.g., one from exact matcher, one from fuzzy matcher, or duplicate rows in input.

                # Check if it's a conflict (disagreements/exceptions win over matched)
                best_idx = min(range(len(group)), key=lambda i: _severity(group[i].get('final_status', '')))
                winner = dict(group[best_idx])
                winner['_merged_from_count'] = len(group)
                # Preserve the evidence that caused the merge: every
                # collapsed row's status/type, so the decision is auditable
                # and a reviewer can see why one leg won over another.
                winner['_merged_from'] = [
                    {
                        'final_status': r.get('final_status'),
                        'type': r.get('type'),
                        'match_type': r.get('match_type'),
                        'confidence': r.get('confidence'),
                    }
                    for r in group
                ]

                # Aggregate info
                for r in group:
                    if r.get('forced_demo_case'):
                        winner['forced_demo_case'] = True

                # If we have multiple different 'unresolved_exception' types for the same settlement,
                # we just keep the highest severity one as the transaction result.
                # However, if one was matched and another was unresolved, the unresolved one wins.
                winner['canonical_transaction_id'] = sett_id
                winner['evaluation_unit_id'] = sett_id
                transaction_results.append(winner)

        return {
            'transaction_results': transaction_results,
            'orphan_events': orphan_events,
            'duplicate_events': duplicate_events
        }

    def _load_or_generate_data(
        self,
        settlements_path: str,
        bank_path: str,
        ledger_path: str,
        generate: bool,
        num_records: int,
        seed: int,
        messiness_ratio: float = 0.25
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
                    # Never overwrite the caller's input path (fixtures are
                    # immutable). Persist fetched settlements to this run's
                    # runtime directory instead.
                    _runtime_dir = os.path.join("runtime", "runs", self.run_id)
                    os.makedirs(_runtime_dir, exist_ok=True)
                    settlements_df.to_csv(
                        os.path.join(_runtime_dir, "settlements_fetched.csv"),
                        index=False,
                    )
                    print(f"  Fetched {len(settlements_df)} settlements from Razorpay API")
                else:
                    print("  API returned no settlements - will use synthetic data")
            except Exception as e:
                print(f"  API unavailable or failed: {e}")
                print("  Generating linked synthetic dataset")
                settlements_df = None
        
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
            generator = SyntheticDataGenerator(seed=seed, messiness_ratio=messiness_ratio)
            
            output_dir = os.path.join("runtime", "runs", self.run_id)
            os.makedirs(output_dir, exist_ok=True)
            
            bank_df, ledger_df, _ = generator.generate(
                num_records=num_records,
                settlements_df=settlements_df if settlements_df is not None and len(settlements_df) == num_records else None,
                output_dir=output_dir,
                demo_disagreement=self.demo_disagreement_demo,
                run_id=self.run_id,
                batch_id=self.batch_id
            )
            generated_settlements_path = os.path.join(output_dir, "settlements_live.csv")
            if os.path.exists(generated_settlements_path):
                settlements_df = pd.read_csv(generated_settlements_path)
                
            # Reload evaluator's ground truth from the freshly-written file so
            # that strict-mode coverage reflects the current batch.
            gt_path = os.path.join(output_dir, "ground_truth.json")
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
    parser.add_argument("--demo-disagreement", action="store_true",
                        help="Force at least one disagreement case for demo")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique identifier for this pipeline run")
    parser.add_argument("--records", type=int, default=20,
                        help="Number of records to generate if needed")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--messiness", type=float, default=0.25,
                        help="Fraction of synthetic records with injected issues (0.0-1.0)")
    parser.add_argument("--settlements", type=str, default="data/fixtures/settlements_live.csv",
                        help="Path to settlements CSV")
    parser.add_argument("--bank", type=str, default="data/fixtures/bank_statement.csv",
                        help="Path to bank statement CSV")
    parser.add_argument("--ledger", type=str, default="data/fixtures/internal_ledger.csv",
                        help="Path to internal ledger CSV")
    
    args = parser.parse_args()
    if args.messiness < 0.0 or args.messiness > 1.0:
        print("Error: --messiness must be between 0.0 and 1.0")
        return 1
    
    pipeline = ReconciliationPipeline(
        use_llm=not args.no_llm,
        demo_disagreement_demo=args.demo_disagreement,
        run_id=args.run_id
    )
    
    results = pipeline.run(
        settlements_path=args.settlements,
        bank_path=args.bank,
        ledger_path=args.ledger,
        generate_if_missing=True,
        num_records=args.records,
        seed=args.seed,
        messiness_ratio=args.messiness
    )
    
    if 'error' in results:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(run_pipeline_cli())
