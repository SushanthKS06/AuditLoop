"""
Streamlit Dashboard - Thin layer over the reconciliation results

Shows:
- Summary metrics (match rate, precision, recall)
- Exceptions table with drill-down
- LLM-deterministic disagreement cases (Failure Recovery proof)
"""

import os
import json
import pandas as pd
import streamlit as st

from audit.store import AuditStore


def load_results(results_path: str = "results.json") -> list:
    """Load reconciliation results from JSON file."""
    if not os.path.exists(results_path):
        return []
    with open(results_path, 'r') as f:
        return json.load(f)


def load_metrics(metrics_path: str = "metrics/metrics_report.json") -> dict:
    """Load metrics report from JSON file."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, 'r') as f:
        return json.load(f)


def main():
    st.set_page_config(
        page_title="AuditLoop - Reconciliation Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AuditLoop Reconciliation Dashboard")
    st.markdown("""
    **Deterministic-first reconciliation with LLM-assisted exception handling.**
    Every decision is logged. Accuracy is measured against ground truth.
    """)
    
    # Load data
    results = load_results()
    metrics = load_metrics()
    audit_store = AuditStore()
    audit_stats = audit_store.get_summary_stats()
    
    # Summary header
    st.header("Summary Metrics")
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
    
    # Secondary metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Total Records",
            metrics.get('total_records', 0)
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
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 All Results", "⚠️ Exceptions", "🔴 Disagreements"])
    
    with tab1:
        st.subheader("All Reconciliation Results")
        
        if results:
            # Flatten results for display
            flat_results = []
            for r in results:
                flat = {
                    'payment_id': r.get('payment_id', ''),
                    'status': r.get('final_status', ''),
                    'type': r.get('type', ''),
                    'confidence': r.get('confidence', 0),
                    'llm_root_cause': r.get('llm_root_cause', ''),
                }
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No results loaded. Run the reconciliation pipeline first.")
    
    with tab2:
        st.subheader("Exceptions Requiring Review")
        
        exceptions = audit_store.get_exceptions()
        
        if exceptions:
            exc_df = pd.DataFrame(exceptions)
            
            # Show key columns
            display_cols = ['record_ids', 'stage', 'rule_fired', 'confidence', 'decision']
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
                st.json(exc)
        else:
            st.success("No unresolved exceptions!")
    
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
                with st.expander(f"Disagreement Case #{i+1}: {d.get('record_ids', 'Unknown')}"):
                    st.write("**Record IDs:**", d.get('record_ids', ''))
                    st.write("**Stage:**", d.get('stage', ''))
                    st.write("**Rule Fired:**", d.get('rule_fired', ''))
                    st.write("**Confidence:**", d.get('confidence', ''))
                    st.write("**Decision:**", d.get('decision', ''))
                    st.write("**LLM Reasoning:**", d.get('llm_reasoning', ''))
                    st.warning("This case was flagged for human review due to LLM-deterministic disagreement.")
        else:
            st.info("No disagreement cases found. This may mean:")
            st.write("- The pipeline hasn't run yet")
            st.write("- No exceptions triggered LLM review")
            st.write("- All LLM proposals passed deterministic re-check")
    
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
