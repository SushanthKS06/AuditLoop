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
        
        if self.force_disagreement_demo:
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
                src = row.get('source') or row.get('settlement', {}).get('source') or 'synthetic'
                match = {
                    **row.get('settlement', {}),
                    'match_type': 'exact',
                    'confidence': 1.0,
                    'final_status': 'matched',
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
                    'final_status': 'matched',
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
                force_disagreement_case=self.force_disagreement_demo
            )
            
            for exc in processed_exceptions:
                exc['source'] = exc.get('source') or (exc.get('settlement') or {}).get('source') or (exc.get('counterpart') or {}).get('source') or 'synthetic'
                exc['forced_demo_case'] = exc.get('forced_demo_case', False)
                self.all_results.append(exc)
                
                # Count LLM-verified matches
                if exc.get('final_status') == 'matched_llm_verified':
                    self.all_matches.append({
                        **exc.get('settlement', {}),
                        'match_type': 'llm_verified',
                        'confidence': exc.get('llm_confidence'),
                        'final_status': 'matched_llm_verified',
                        'source': exc['source'],
                        'forced_demo_case': False
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
                exc['source'] = exc.get('source') or (exc.get('settlement') or {}).get('source') or (exc.get('counterpart') or {}).get('source') or 'synthetic'
                exc['forced_demo_case'] = False
                self.all_results.append(exc)
            print("  LLM disabled - all exceptions marked unresolved")
        
        # Step 4: Compile final results
        print("\n[Step 4] Compiling results...")
        
        # Add matched records to results
        for match in self.all_matches:
            self.all_results.append(match)
        
        # Step 4.5: Transaction-level deduplication
        # A single real-world transaction can produce multiple rows if different
        # pipeline legs (settlement, bank, ledger) processed it independently.
        # Collapse all rows that share a payment_id / order_id / utr / bank_txn_id
        # into exactly one row per transaction.  Exception status always wins over
        # 'matched' — a transaction is only matched when every leg agrees.
        print("\n[Step 4.5] Deduplicating cross-source results...")
        pre_dedup_count = len(self.all_results)
        
        self.all_results = self._deduplicate_results(self.all_results)
        dedup_removed = pre_dedup_count - len(self.all_results)
        if dedup_removed > 0:
            print(f"  Removed {dedup_removed} duplicate cross-source row(s) — "
                  f"{len(self.all_results)} unique transactions remain.")
        else:
            print(f"  No duplicates found — {len(self.all_results)} unique transactions.")

        # POST-DEDUP SANITY CHECK
        # Ensure no valid input settlement entity_ids were silently dropped
        if settlements_df is not None and not settlements_df.empty:
            output_entities = set()
            for r in self.all_results:
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
        results_path = os.getenv("RESULTS_PATH", "results.json")
        os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
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
    
    def _deduplicate_results(self, results: list) -> list:
        """
        Collapse cross-source result rows so each real-world transaction
        appears exactly once in the output.

        Algorithm
        ---------
        1. Extract every usable identifier from each row into an id_dict.
        2. Build a graph: edges connect rows that share at least one identifier value.
        3. Find connected components.
        4. If a component has ANY conflicting identifiers between any of its members,
           it represents distinct transactions incorrectly tied together (e.g., duplicate UTRs).
           Do NOT collapse them; emit them individually as 'unresolved_exception'.
        5. If a component is conflict-free, collapse it into one representative row.
        """
        STATUS_SEVERITY = {
            'llm_deterministic_disagreement': 0,
            'unresolved_exception': 1,
            'explained_no_resolution': 2,
            'low_confidence': 3,
            'llm_error': 4,
            'llm_unavailable': 4,
            'matched_llm_verified': 5,
            'matched': 6,
        }

        def _severity(status: str) -> int:
            return STATUS_SEVERITY.get(status, 3)

        def _extract_id_dict(row: dict) -> dict:
            ids = {}
            direct_keys = [
                'payment_id', 'order_id',
                'txn_id', 'bank_txn_id', 'entity_id', 'settlement_id'
            ]
            for k in direct_keys:
                v = row.get(k)
                if v and str(v).strip() and str(v).lower() not in ('none', 'nan'):
                    key_group = 'utr' if k in ('utr', 'settlement_utr') else (
                        'txn_id' if k in ('txn_id', 'bank_txn_id') else k
                    )
                    ids.setdefault(key_group, set()).add(str(v).lower().strip())
                    
            for nest_key in ('settlement', 'counterpart', 'bank', 'ledger'):
                nested = row.get(nest_key)
                if isinstance(nested, dict):
                    for k in direct_keys:
                        v = nested.get(k)
                        if v and str(v).strip() and str(v).lower() not in ('none', 'nan'):
                            key_group = 'utr' if k in ('utr', 'settlement_utr') else (
                                'txn_id' if k in ('txn_id', 'bank_txn_id') else k
                            )
                            ids.setdefault(key_group, set()).add(str(v).lower().strip())
            return ids

        def _has_conflict(ids1: dict, ids2: dict) -> bool:
            for k in ids1:
                if k in ids2:
                    if not ids1[k].intersection(ids2[k]):
                        return True
            return False

        def _share_identifier(ids1: dict, ids2: dict) -> bool:
            for k in ids1:
                if k in ids2:
                    if ids1[k].intersection(ids2[k]):
                        return True
            return False

        if not results:
            return results

        n = len(results)
        id_dicts = [_extract_id_dict(r) for r in results]
        
        from collections import defaultdict
        adj = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                if _share_identifier(id_dicts[i], id_dicts[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        groups = []
        for i in range(n):
            if i not in visited:
                comp = []
                q = [i]
                visited.add(i)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                groups.append(comp)

        deduplicated = []
        for comp in groups:
            has_conflict = False
            for i in range(len(comp)):
                for j in range(i+1, len(comp)):
                    if _has_conflict(id_dicts[comp[i]], id_dicts[comp[j]]):
                        has_conflict = True
                        break
                if has_conflict:
                    break
            
            if not has_conflict:
                if len(comp) == 1:
                    deduplicated.append(results[comp[0]])
                else:
                    best_idx = min(
                        comp,
                        key=lambda i: _severity(results[i].get('final_status', ''))
                    )
                    winner = dict(results[best_idx])
                    winner['_merged_from_count'] = len(comp)
                    
                    # Merge missing identifiers and metadata from other rows into the winner
                    # so it doesn't lose its entity_id/payment_id if an orphaned bank row was picked.
                    for idx in comp:
                        r = results[idx]
                        if r.get('forced_demo_case'):
                            winner['forced_demo_case'] = True
                        if not winner.get('source') and r.get('source'):
                            winner['source'] = r.get('source')
                        for key in ['entity_id', 'settlement_id', 'payment_id', 'order_id']:
                            if not winner.get(key) and r.get(key):
                                winner[key] = r.get(key)
                        if isinstance(r.get('settlement'), dict):
                            for key in ['entity_id', 'settlement_id', 'payment_id', 'order_id']:
                                if not winner.get(key) and r['settlement'].get(key):
                                    winner[key] = r['settlement'].get(key)
                    if not winner.get('source'):
                        winner['source'] = 'synthetic'
                    if 'forced_demo_case' not in winner:
                        winner['forced_demo_case'] = False
                                    
                    deduplicated.append(winner)
            else:
                # Component is ambiguous; emit all individually as unresolved
                for idx in comp:
                    row_copy = dict(results[idx])
                    row_copy['final_status'] = 'unresolved_exception'
                    row_copy['type'] = 'ambiguous_shared_identifier'
                    row_copy['_ambiguous_merge'] = True
                    row_copy['source'] = row_copy.get('source', 'synthetic')
                    row_copy['forced_demo_case'] = row_copy.get('forced_demo_case', False)
                    deduplicated.append(row_copy)

        return deduplicated

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
            generator = SyntheticDataGenerator(seed=seed, messiness_ratio=0.40)
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
