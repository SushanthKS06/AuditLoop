"""
Streamlit Dashboard - Thin layer over the reconciliation results

Shows:
- Summary metrics (match rate, precision, recall)
- Exceptions table with drill-down
- LLM-deterministic disagreement cases (Failure Recovery proof)
"""

import os
import sys
import json
from typing import Optional, List, Dict
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add project root to path for clean package imports when run via Streamlit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.store import AuditStore


def load_results(results_path: Optional[str] = None) -> dict:
    """Load reconciliation results, normalising legacy and structured payloads.

    Returns a PipelineResult-style dict with transaction_results /
    orphan_events / duplicate_events / exception_events. Legacy list
    payloads are wrapped as transaction_results so old artifacts still render.
    """
    target_path = results_path or os.getenv("RESULTS_PATH", "results.json")
    candidates = [target_path, "results.json", "runtime/results.json",
                  "demo_artifacts/results.json"]
    # Also pick up the newest per-run artifact if no explicit path exists.
    runs_dir = os.path.join("runtime", "runs")
    if os.path.isdir(runs_dir):
        try:
            run_dirs = sorted(
                (os.path.join(runs_dir, d) for d in os.listdir(runs_dir)),
                key=os.path.getmtime,
            )
            for d in reversed(run_dirs):
                candidates.append(os.path.join(d, "results.json"))
        except OSError:
            # Best-effort run discovery: fall back to the static candidates.
            pass
    payload = None
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    payload = json.load(f)
                break
            except Exception:
                # Unreadable/corrupt candidate: try the next path.
                continue
    if payload is None:
        return {'transaction_results': [], 'orphan_events': [],
                'duplicate_events': [], 'exception_events': []}
    if isinstance(payload, list):
        return {'transaction_results': payload, 'orphan_events': [],
                'duplicate_events': [], 'exception_events': []}
    if isinstance(payload, dict):
        return {
            'transaction_results': payload.get('transaction_results', []),
            'orphan_events': payload.get('orphan_events', []),
            'duplicate_events': payload.get('duplicate_events', []),
            'exception_events': payload.get('exception_events', []),
        }
    return {'transaction_results': [], 'orphan_events': [],
            'duplicate_events': [], 'exception_events': []}


def load_metrics() -> dict:
    """Load metrics report from JSON file checking configured env paths and default locations."""
    paths = []
    if os.getenv("METRICS_PATH"):
        paths.append(os.getenv("METRICS_PATH"))
    paths.extend([
        "metrics/metrics_report.json",
        "metrics_report.json",
        "demo_artifacts/metrics_report.json"
    ])
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def main():
    st.set_page_config(
        page_title="AuditLoop - Reconciliation Dashboard",
        layout="wide"
    )
    
    st.title("AuditLoop Reconciliation Dashboard")
    st.markdown("""
    **Deterministic-first reconciliation with LLM-assisted exception handling.**
    Every decision is logged. Accuracy is measured against ground truth.
    """)
    
    # Load data
    payload = load_results()
    results = payload.get('transaction_results', [])
    orphan_events = payload.get('orphan_events', [])
    duplicate_events = payload.get('duplicate_events', [])
    exception_events = payload.get('exception_events', [])
    metrics = load_metrics()
    audit_store = AuditStore()
    audit_stats = audit_store.get_summary_stats()
    
    # Summary header
    st.header("Summary Metrics")
    
    # Audit Integrity Banner
    integrity = audit_store.verify_integrity()
    if integrity['integrity_verified']:
        st.success(f"**Cryptographic Audit Trail Integrity: VERIFIED** (All {integrity['total_checked']} records chained via SHA-256 block hashing — 0 tampered)")
    else:
        st.error(f"**Cryptographic Audit Trail Integrity Alert:** Tampering detected in record IDs: {integrity['tampered_ids']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Match Rate",
            f"{metrics.get('match_rate', 0)*100:.1f}%",
            help="Percentage of records successfully reconciled"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{metrics.get('precision', 0)*100:.1f}%",
            help="Of all matches, how many were correct (vs ground truth)"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{metrics.get('recall', 0)*100:.1f}%",
            help="Of all true matches, how many did we find"
        )
    
    with col4:
        st.metric(
            "F1 Score",
            f"{metrics.get('f1_score', 0)*100:.1f}%",
            help="Harmonic mean of precision and recall"
        )
    
    # Ground-truth coverage — always visible so it can never be silently assumed
    gt_total = metrics.get('total_input_transactions', metrics.get('total_records', 0))
    gt_unverified = metrics.get('unverified_count', 0)
    gt_scored = gt_total - gt_unverified
    gt_coverage_pct = metrics.get('ground_truth_coverage', 0) * 100
    coverage_mode = metrics.get('coverage_mode', 'strict')
    if gt_total > 0:
        st.caption(
            f"**Ground-truth coverage:** {gt_scored} of {gt_total} records "
            f"({gt_coverage_pct:.1f}%) — "
            f"precision / recall / F1 above are computed **only** over the "
            f"verified subset (mode: `{coverage_mode}`). "
            + (f"{gt_unverified} record(s) had no ground-truth entry and were excluded." if gt_unverified > 0 else "All records verified.")
        )
    
    # Secondary metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Input transactions",
            metrics.get('total_input_transactions', metrics.get('total_records', 0))
        )
    
    with col6:
        st.metric(
            "Exceptions",
            metrics.get('exception_count', 0)
        )
    
    with col7:
        st.metric(
            "Disagreements",
            metrics.get('disagreement_count', 0),
            help="Cases where LLM and deterministic engine disagreed"
        )
    
    with col8:
        st.metric(
            "Unresolved",
            metrics.get('unresolved_count', 0)
        )

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.metric("Orphan bank", metrics.get('orphan_bank_records', 0))
    with o2:
        st.metric("Orphan ledger", metrics.get('orphan_ledger_records', 0))
    with o3:
        st.metric("Duplicates", metrics.get('duplicate_suspects', 0))
    with o4:
        st.metric("Demo-injected", metrics.get('demo_injected_count', 0))
    if metrics.get('demo_mode'):
        st.warning("Demo mode: forced disagreement events are tagged and excluded from organic TP/FP/FN.")
    else:
        st.caption("Benchmark/organic mode: no demo-injected events in this metrics file.")
    
    # Expandable raw ground truth metrics breakdown
    with st.expander("Ground-Truth & Secondary Metrics Breakdown", expanded=False):
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        tp = metrics.get('true_positives', 0)
        fn = metrics.get('false_negatives', 0)
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("True Positives (TP)", tp)
        with m_c2:
            st.metric("False Positives (FP)", fp)
        with m_c3:
            st.metric("True Negatives (TN)", tn)
        with m_c4:
            st.metric("False Negatives (FN)", fn)
            
        neg_count = tn + fp
        st.caption(f"False Positive Rate: {metrics.get('false_positive_rate', 0)*100:.1f}% — Computed on {neg_count} known-negative ground-truth cases (treat as directional, not statistically robust).")
        
        st.divider()
        cov_mode = metrics.get('coverage_mode', 'strict')
        cov_pct = metrics.get('ground_truth_coverage', 0) * 100
        unv = metrics.get('unverified_count', 0)
        tot = metrics.get('total_input_transactions', metrics.get('total_records', 0))
        scored = tot - unv
        st.caption(
            f"Ground-truth coverage: **{scored} of {tot} records ({cov_pct:.1f}%)** — "
            f"metrics above are computed only over the verified subset "
            f"(coverage_mode=`{cov_mode}`). "
            + (f"{unv} record(s) had no ground-truth entry and were excluded from precision/recall/F1/FPR." if unv > 0 else "All records have a ground-truth entry.")
        )
    
    st.divider()
    
    # --- Reconciliation Funnel ---
    st.subheader("Reconciliation Funnel")
    tot_recs = metrics.get('total_input_transactions', metrics.get('total_records', 0))
    
    # Get counts by stage directly from audit store
    stage_counts = {
        'total': tot_recs,
        'exact': 0,
        'fuzzy': 0,
        'llm_resolved': 0,
        'unresolved': metrics.get('unresolved_count', 0),
        'disagreements': metrics.get('disagreement_count', 0)
    }
    
    if results:
        exact_c = sum(1 for r in results if r.get('match_type') == 'exact')
        fuzzy_c = sum(1 for r in results if r.get('match_type') in ('high_confidence', 'fuzzy'))
        llm_c = sum(1 for r in results if r.get('match_type') == 'llm_verified')
        
        stage_counts['exact'] = exact_c
        stage_counts['fuzzy'] = fuzzy_c
        stage_counts['llm_resolved'] = llm_c
        
    fig = go.Figure(go.Funnel(
        y=["Total Input Records", "Exact Match (Stage 1)", "Fuzzy Match (Stage 2)", "LLM Resolved (Stage 3)", "Unresolved Exceptions", "LLM/System Disagreements"],
        x=[
            stage_counts['total'],
            stage_counts['exact'],
            stage_counts['fuzzy'],
            stage_counts['llm_resolved'],
            stage_counts['unresolved'],
            stage_counts['disagreements']
        ],
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(margin={"t": 30, "b": 30})
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Main content tabs
    tab1, tab1b, tab2, tab3, tab4 = st.tabs([
        "Transaction Results", "Orphan & Duplicate Events",
        "Exceptions", "Disagreements", "Audit Hash Chain",
    ])

    with tab1:
        st.subheader("Transaction Results (one row = one input settlement)")
        st.caption(
            f"{len(results)} transaction results · {len(orphan_events)} orphan events · "
            f"{len(duplicate_events)} duplicate events · {len(exception_events)} exception events. "
            "Orphan/duplicate rows are events, not transactions — they never inflate match metrics."
        )
        
        if results:
            # Flatten results for display
            flat_results = []
            for r in results:
                flat = {
                    'canonical_transaction_id': r.get('canonical_transaction_id') or r.get('evaluation_unit_id', ''),
                    'payment_id': r.get('payment_id', '') or r.get('record_ids', ''),
                    'status': r.get('final_status', ''),
                    'source': r.get('source', 'synthetic'),
                    'type': r.get('type', ''),
                    'confidence': r.get('confidence', 0),
                    'demo_flag': "FORCED DEMO" if r.get('forced_demo_case') else "standard",
                    'llm_proposal': r.get('llm_proposed_action', ''),
                    'deterministic_recheck': r.get('deterministic_recheck_passed', ''),
                    'rejection_reason': r.get('rejection_reason', ''),
                }
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download All_Results.csv",
                data=csv_data,
                file_name="All_Results.csv",
                mime="text/csv"
            )
        else:
            st.info("No results loaded. Run the reconciliation pipeline first.")

    with tab1b:
        st.subheader("Orphan & Duplicate Events (not transactions)")
        st.caption(
            "Bank/ledger rows with no settlement counterpart, and duplicate-suspect "
            "candidates. Shown separately so they cannot be mistaken for reconciled transactions."
        )
        event_rows = (
            [{'event_kind': 'orphan', **e} for e in orphan_events]
            + [{'event_kind': 'duplicate', **e} for e in duplicate_events]
            + [{'event_kind': 'exception', **e} for e in exception_events]
        )
        if event_rows:
            flat_events = []
            for e in event_rows:
                flat_events.append({
                    'event_kind': e.get('event_kind', ''),
                    'record_ids': e.get('record_ids', ''),
                    'type': e.get('type', ''),
                    'status': e.get('final_status', ''),
                    'source': e.get('source', 'synthetic'),
                })
            st.dataframe(pd.DataFrame(flat_events), use_container_width=True, hide_index=True)
        else:
            st.info("No orphan or duplicate events in this run.")
    
    with tab2:
        st.subheader("Exceptions Requiring Review")
        
        exceptions = audit_store.get_exceptions()
        
        if exceptions:
            exc_df = pd.DataFrame(exceptions)
            
            # Show key columns
            display_cols = ['record_ids', 'stage', 'source', 'rule_fired', 'confidence', 'decision']
            available_cols = [c for c in display_cols if c in exc_df.columns]
            
            st.dataframe(exc_df[available_cols], use_container_width=True, hide_index=True)
            
            # Drill-down
            st.markdown("### Exception Details")
            selected = st.selectbox(
                "Select exception to view details",
                options=list(range(len(exceptions))),
                format_func=lambda i: f"{exceptions[i]['record_ids']} - {exceptions[i]['decision']}"
            )
            
            if selected is not None:
                exc = exceptions[selected]
                if exc.get('forced_demo_case'):
                    st.warning("🚨 FORCED DEMO CASE: Seeded discrepancy guaranteed for presentation & failure-recovery verification.")
                st.json(exc)
                
                st.markdown("#### Human-in-the-Loop Maker-Checker Action")
                with st.form(key=f"resolve_form_{selected}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        reviewer_id = st.text_input("Financial Reviewer ID (Attribution Label)", value="CONTROLLER_001")
                    with c2:
                        decision_choice = st.selectbox(
                            "Resolution Action",
                            options=["human_approved_match", "human_rejected_duplicate", "human_written_off"],
                            format_func=lambda x: {
                                "human_approved_match": "Approve & Force Match",
                                "human_rejected_duplicate": "Confirm Rejected Duplicate",
                                "human_written_off": "Write-off Anomaly"
                            }[x]
                        )
                    notes = st.text_area("Auditable Reviewer Notes", placeholder="E.g., Confirmed with merchant invoice INV-8821. MDR fee discrepancy authorized.")
                    submitted = st.form_submit_button("Sign & Append to Tamper-Evident Audit Chain")
                    
                    if submitted:
                        if len(notes.strip()) < 5:
                            st.error("Please enter descriptive notes (at least 5 characters).")
                        else:
                            res = audit_store.resolve_exception(
                                record_ids=exc['record_ids'],
                                reviewer_id=reviewer_id,
                                decision=decision_choice,
                                notes=notes
                            )
                            st.success(f"Resolution signed and chained. Entry ID: #{res['audit_entry_id']} | Block Hash: `{res['record_hash'][:16]}...`")
                            st.rerun()
        else:
            st.success("No unresolved exceptions.")
    
    with tab3:
        st.subheader("LLM-Deterministic Disagreement Cases")
        st.markdown("""
        These are cases where the LLM proposed a match but the deterministic 
        re-verification rejected it (or vice versa). This is **intentional** - 
        we never let the LLM have unilateral authority over financial decisions.
        """)
        
        disagreements = audit_store.get_disagreements()
        
        if disagreements:
            for i, d in enumerate(disagreements):
                title = f"Disagreement Case #{i+1}: {d.get('record_ids', 'Unknown')}"
                if d.get('forced_demo_case'):
                    title += " 🚨 FORCED DEMO CASE"
                with st.expander(title):
                    if d.get('forced_demo_case'):
                        st.error("🚨 FORCED DEMO CASE: This is a fabricated discrepancy guaranteed for presentation purposes.")
                    st.write("**Record IDs:**", d.get('record_ids', ''))
                    st.write("**Stage:**", d.get('stage', ''))
                    st.write("**Rule Fired:**", d.get('rule_fired', ''))
                    st.write("**Confidence:**", d.get('confidence', ''))
                    st.write("**Decision:**", d.get('decision', ''))
                    st.write("**LLM proposal:**", d.get('decision', ''), "| reasoning:", d.get('llm_reasoning', ''))
                    st.write("**Deterministic verification:** rejected if this row is a disagreement")
                    st.write("**Final decision:**", d.get('final_status', d.get('decision', '')))
                    
                    # Maker-checker action directly on disagreement
                    with st.form(key=f"disagree_resolve_{i}"):
                        d_reviewer = st.text_input("Reviewer ID", value="CHIEF_AUDITOR", key=f"d_rev_{i}")
                        d_decision = st.selectbox(
                            "Resolution Action",
                            options=["human_approved_match", "human_rejected_duplicate", "human_written_off"],
                            key=f"d_dec_{i}"
                        )
                        d_notes = st.text_input("Compliance Justification", key=f"d_notes_{i}", placeholder="Verified bank cutoff time lag.")
                        if st.form_submit_button("Authorize Resolution"):
                            if len(d_notes.strip()) >= 5:
                                audit_store.resolve_exception(
                                    record_ids=d.get('record_ids', ''),
                                    reviewer_id=d_reviewer,
                                    decision=d_decision,
                                    notes=d_notes
                                )
                                st.success("Disagreement resolved and cryptographically sealed.")
                                st.rerun()
                            else:
                                st.error("Please provide at least 5 characters of notes.")
        else:
            st.info("No disagreement cases found. This may mean:")
            st.write("- The pipeline hasn't run yet")
            st.write("- No exceptions triggered LLM review")
            st.write("- All LLM proposals passed deterministic re-check")
            
    with tab4:
        st.subheader("Tamper-Evident SHA-256 Audit Trail Chain")
        st.markdown("""
        Every stage decision computes `record_hash = SHA256(previous_hash + payload)`, 
        providing cryptographic proof of tamper-evident chaining.
        """)
        
        audit_entries = audit_store.get_all()
        if audit_entries:
            chain_df = pd.DataFrame(audit_entries)
            display_cols = ['id', 'timestamp', 'record_ids', 'stage', 'decision', 'source', 'previous_hash', 'record_hash']
            available_cols = [c for c in display_cols if c in chain_df.columns]
            st.dataframe(chain_df[available_cols], use_container_width=True, hide_index=True)
            
            chain_csv = chain_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Audit_Hash_Chain.csv",
                data=chain_csv,
                file_name="Audit_Hash_Chain.csv",
                mime="text/csv"
            )
        else:
            st.info("Audit log is currently empty.")
    
    st.divider()
    
    # Footer
    st.markdown("""
    ### About AuditLoop
    
    AuditLoop is a reconciliation agent that never lets an LLM have the final say on a financial match.
    
    - **Deterministic matching** runs first (exact + fuzzy)
    - **LLM only explains** and proposes resolutions for exceptions
    - **Every proposal is re-verified** deterministically before counting
    - **Full audit trail** for every decision
    
    Built for Razorpay AI Buildathon 2026 — Track 04: AI Finance Controller
    """)


if __name__ == "__main__":
    main()
