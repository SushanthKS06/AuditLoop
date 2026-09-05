```mermaid
flowchart TD
    %% Define Nodes
    DL["Data Layer (reproducible synthetic benchmark)<br/>- settlements.csv (synthetic)<br/>- bank_statement.csv (synthetic)<br/>- internal_ledger.csv (synthetic)<br/>- optional Razorpay test-mode ingestion"]

    S1["Stage 1: Exact Match<br/>- Hash join on UTR/order_id/payment_id<br/>- Confidence: 1.0<br/>- Audit: YES"]

    S2["Stage 2: Fuzzy Match<br/>- Amount delta %, date window, text similarity<br/>- Confidence threshold: 0.85<br/>- Audit: YES"]

    S3["Stage 3: LLM Exceptions<br/>- explain_exception()<br/>- propose_resolution()<br/>- Pydantic validation"]

    DRC["Deterministic Re-Check<br/>- LLM 'match' proposals verified<br/>- Disagreement = flag for human<br/>- Fail CLOSED, never open"]

    AT["Audit Trail (SQLite)<br/>- Append-only log<br/>- Every decision recorded<br/>- Explainability & compliance"]

    MH["Metrics Harness<br/>- Precision/Recall/F1<br/>- vs ground_truth.json<br/>- Reproducible"]

    %% Define Flow / Arrows
    DL --> S1
    S1 --> S2
    S2 --> S3
    S3 --> DRC
    S3 --> AT

    %% Styling based on your original colors
    style DL fill:#d4edda,stroke:#1e8e3e,stroke-width:2px,color:#000
    style S1 fill:#cce5ff,stroke:#0066cc,stroke-width:2px,color:#000
    style S2 fill:#cce5ff,stroke:#0066cc,stroke-width:2px,color:#000
    style S3 fill:#e6ccff,stroke:#9933ff,stroke-width:2px,color:#000
    style DRC fill:#ffe0cc,stroke:#ff6600,stroke-width:2px,color:#000
    style AT fill:#f0f0f0,stroke:#333333,stroke-width:2px,color:#000
    style MH fill:#ffcccc,stroke:#cc0000,stroke-width:2px,color:#000
