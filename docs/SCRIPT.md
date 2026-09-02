# 5-Minute Pitch & Live Demo Script: AuditLoop

**Title:** AuditLoop — *The Deterministic-First Financial Reconciliation Agent*  
**Track:** Razorpay AI Buildathon 2026 — Track 04: AI Finance Controller  
**Target Duration:** 5 Minutes  
**Speaker Persona:** Lead Systems Architect & Founding AI Engineer  

---

## ⏱️ Timeline Summary

| Time | Section | Key Visual / Screen | Core Takeaway |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:45** | **The Industry Problem** | Split screen: Razorpay Recon CSV, Bank statement, ERP ledger | Reconciliation is an unmitigated nightmare of timing lags and MDR fee deductions. |
| **0:45 - 1:30** | **The Architectural Thesis** | Architecture diagram | *The LLM proposes; deterministic verifier disposes.* Never trust an LLM with money. |
| **1:30 - 3:00** | **Live Pipeline Demo** | Streamlit Dashboard & Terminal | Stage 1 ($O(N+M)$ hash match), Stage 2 (fuzzy fee scoring), Stage 3 (Claude CoT + PII sanitization). |
| **3:00 - 3:45** | **Failure Recovery & Disagreements** | Disagreements Tab & HITL Form | Fail-closed demonstration: LLM hallucination caught and stopped cold. Maker-checker human sign-off. |
| **3:45 - 4:30** | **Cryptographic Auditability & Metrics** | `/audit/verify` API & SHA-256 Tab | Mathematical tamper-evidence with SHA-256 block hashing; ground truth metrics ($>90\%$ precision). |
| **4:30 - 5:00** | **Enterprise Scale & Closing** | API Docs & Terminal benchmarks | 10,000 txns/sec, 92% token cost reduction, SOC2/RBI audit-ready. |

---

## 🎬 Word-for-Word Script & Action Cues

### 1. Problem Statement (0:00 - 0:45)
**[Visual: Show three messy CSVs / PDFs: Razorpay settlement with 2.36% MDR deduction, Bank statement with value_date lag, and ERP internal ledger]**

> *"Every enterprise CFO and finance controller faces the same nightmare at month-end: multi-source reconciliation. 
> You have three sources of truth: Razorpay settlements, bank statements, and internal ERP ledgers. But they never match cleanly. 
> Payment gateways deduct MDR fees (2% + 18% GST), banks settle transactions on T+1 or T+2 cycles, and references get truncated. 
> Today, human teams spend hundreds of hours manually cross-checking spreadsheets. 
> Naive AI startups tried to solve this by dumping financial spreadsheets into LLMs. The result? Hallucinated numbers, uncalibrated matches, and zero regulatory auditability. 
> That’s why we built **AuditLoop**."*

---

### 2. The Core Thesis (0:45 - 1:30)
**[Visual: Switch to Architecture Flowchart]**

> *"Our founding architectural principle is uncompromising:  
> **Financial reconciliation's bottleneck isn't generating candidate matches — it's verifying them with enough confidence to trust the numbers.**  
> Therefore, in AuditLoop, **an LLM is never given the final say on a financial match.**  
> 
> We engineered a 3-stage pipeline:
> - **Stage 1**: Vectorized exact hash-join on normalized UTRs, order IDs, and payment IDs in $O(N+M)$ time.
> - **Stage 2**: Deterministic fuzzy matching with MDR fee deduction awareness and date proximity scoring.
> - **Stage 3**: For remaining unresolved exceptions, Claude analyzes the root cause with step-by-step Chain-of-Thought and PII redaction.
> 
> If the LLM proposes an action, our deterministic engine intercepts it and re-verifies the mathematics before committing. If there is any discrepancy, we fail closed."*

---

### 3. Live Pipeline Execution & Dashboard (1:30 - 3:00)
**[Visual: Run `python run_pipeline.py --force-disagreement --records 50` in terminal, then switch to Streamlit Dashboard]**

> *"Let's see it live. We run a batch of 50 multi-source transactions with injected anomalies: settlement delays, gateway fee deductions, and duplicate amounts.
> 
> Notice how fast Stage 1 and 2 execute: 85%+ of records are matched in milliseconds at **zero LLM token cost**.
> 
> Now let's open the AuditLoop Reviewer Dashboard.
> Right at the top, we see our institutional metrics calculated mathematically against a known ground-truth answer key—not cherry-picked demo data.
> We achieve **96% Precision**, **88% Recall**, and **92% F1 score**."*

---

### 4. Failure Recovery & Maker-Checker Workflow (3:00 - 3:45)
**[Visual: Click on 'Disagreements' tab, show red warning box, then resolve via Maker-Checker form]**

> *"Now, let's look at what sets AuditLoop apart: **Failure Recovery**.
> 
> In the Disagreements tab, we see Case #1. (Note: In this live demo run, this specific disagreement is a seeded demo case to guarantee a visible conflict in a small batch, while all other exceptions are organically discovered). 
> The LLM analyzed an anomaly and proposed a match. 
> But our deterministic re-verifier caught an unexplainable amount difference exceeding our threshold. 
> Instead of silently moving money, AuditLoop flagged it as an `llm_deterministic_disagreement`. 
> 
> This triggers our **Human-in-the-Loop Maker-Checker workflow**. 
> As a Financial Controller, I inspect the invoice, enter my credentials `CONTROLLER_001`, authorize the resolution with auditable notes, and sign it directly into the audit trail."*

---

### 5. Cryptographic Audit Chain & REST API (3:45 - 4:30)
**[Visual: Click on 'Audit Hash Chain' tab, then run `curl http://localhost:8000/audit/verify` in terminal]**

> *"Every single decision—Stage 1 exact matches, Stage 2 fuzzy matches, LLM reasoning, and human overrides—is immutably chained using SHA-256 block hashing, exactly like a private blockchain.
> 
> Each record's hash is computed from the previous block's hash plus the full payload:  
> `record_hash = SHA256(previous_hash + timestamp + record_ids + stage + decision)`.
> 
> If an internal DBA attempts to modify a row in SQLite or PostgreSQL, the cryptographic chain instantly breaks. 
> Our automated `/audit/verify` REST endpoint recomputes hashes from genesis to head and guarantees 100% tamper-evident compliance for RBI and SOC-2 audits."*

---

### 6. Production Readiness & Closing (4:30 - 5:00)
**[Visual: Show FastAPI Swagger docs at `http://localhost:8000/docs` and Docker Compose]**

> *"AuditLoop is packaged with a production-grade FastAPI REST API, a 39-test automated test suite passing in under 8 seconds, and one-command Docker deployment.
> 
> By running deterministic matching first, we cut LLM API costs by **92%** while processing up to **10,000 records per second**.
> 
> AuditLoop bridges the gap between state-of-the-art Generative AI and the strict mathematical guarantees required by modern finance. Thank you."*

---

## 🎯 Anticipated Judge & Panel Q&A Playbook

### Q1: "What happens if Anthropic's Claude API goes down during a batch run?"
> **Answer:** *"AuditLoop fails closed gracefully. If the LLM API times out or throws a 500/429 error, exceptions are immediately tagged as `unresolved_exception` or `llm_unavailable` and queued for reviewer inspection. Stage 1 and Stage 2 deterministic matches are completely unaffected and continue executing at wire speed."*

### Q2: "How do you protect sensitive customer PII from leaking into the LLM?"
> **Answer:** *"Our privacy layer (`llm/privacy.py`) intercepts all exception records before prompt serialization. It scrubs emails, Indian phone numbers (`+91[6-9]...`), and customer names with regex and token masking, passing only sanitized structural keys (UTRs, amounts, timestamps, and order IDs) to the external LLM."*

### Q3: "Why not use an off-the-shelf vector database for matching?"
> **Answer:** *"Vector databases compute cosine distance over text embeddings, which is great for semantic search but dangerous for numerical reconciliation. Cosine similarity cannot determine whether 976.40 + 23.60 equals 1000.00. Our deterministic engine uses exact hash tables and arithmetic fee equations, which are mathematically exact, zero-cost, and $1000\times$ faster."*
