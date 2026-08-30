"""
LLM Prompts - Scoped and minimal

The LLM receives only the specific records under consideration,
not the entire dataset. This is a deliberate reliability choice.
"""

EXPLAIN_EXCEPTION_SYSTEM = """You are analyzing financial reconciliation exceptions.
Your job is to identify the ROOT CAUSE of why two records couldn't be matched.

You will receive:
- A settlement record from Razorpay
- Optionally, a counterpart record (bank transaction or ledger entry)

Classify the root cause into ONE of these categories:
- "rounding": Minor numerical differences (<2%) likely from rounding or fee calculations
- "timing_lag": Date mismatches due to settlement delays (T+1, T+2 common in India)
- "duplicate_suspected": Same amount appears multiple times, possible duplicate entry
- "partial_refund": One record may be a partial refund of another
- "no_counterpart": Record has no matching counterpart in other systems
- "currency_formatting": Differences due to currency symbols, commas, formatting
- "unclassified": None of the above clearly apply

Respond with:
1. root_cause: The category that best fits
2. explanation: Brief explanation citing specific field differences
3. confidence: Your confidence 0-1

Be conservative. If unsure, use lower confidence."""


PROPOSE_RESOLUTION_SYSTEM = """You are proposing resolutions for financial reconciliation exceptions.

You will receive:
- A settlement record from Razorpay  
- A counterpart record (bank transaction or ledger entry)

Propose ONE of these actions:
- "match": Records represent the same underlying transaction
- "flag_for_human": Need human review (uncertain, high-value, or complex case)
- "reject_duplicate": Records appear to be duplicates or unrelated

CRITICAL RULES:
1. Only propose "match" if you're confident (>0.8) they represent the same transaction
2. For high-value transactions (>50000 INR), prefer "flag_for_human" when uncertain
3. If amounts differ by >5%, do NOT propose "match" unless there's clear fee explanation
4. Your proposal will be re-verified deterministically before any match is committed

Respond with:
1. action: Your proposed resolution
2. confidence: Your confidence 0-1
3. reasoning: Justification citing specific fields

Your proposal does NOT commit a match - it will be verified before counting."""


def build_explain_prompt(settlement: dict, counterpart: dict = None) -> str:
    """Build the prompt for explain_exception tool call."""
    
    sett_str = _format_record(settlement, "Settlement")
    count_str = _format_record(counterpart, "Counterpart") if counterpart else "No counterpart record available."
    
    return f"""Analyze this exception:

{sett_str}

{count_str}

What is the root cause of the matching failure?"""


def build_propose_prompt(settlement: dict, counterpart: dict) -> str:
    """Build the prompt for propose_resolution tool call."""
    
    sett_str = _format_record(settlement, "Settlement")
    count_str = _format_record(counterpart, "Counterpart")
    
    return f"""Propose a resolution for these records:

{sett_str}

{count_str}

Should these be matched, flagged for review, or rejected as duplicates?"""


def _format_record(record: dict, label: str) -> str:
    """Format a record for the prompt."""
    if not record:
        return f"{label}: None"
    
    lines = [f"{label}:"]
    
    # Key fields to include
    key_fields = [
        ('entity_id', 'Entity ID'),
        ('order_id', 'Order ID'),
        ('payment_id', 'Payment ID'),
        ('settlement_utr', 'UTR'),
        ('amount', 'Amount'),
        ('settled_amount', 'Settled Amount'),
        ('fee', 'Fee'),
        ('created_at', 'Created'),
        ('settled_at', 'Settled'),
        ('txn_id', 'Transaction ID'),
        ('value_date', 'Value Date'),
        ('narration', 'Narration'),
        ('expected_amount', 'Expected Amount'),
        ('order_date', 'Order Date'),
    ]
    
    for key, display in key_fields:
        if key in record and record[key] is not None:
            lines.append(f"  {display}: {record[key]}")
    
    return "\n".join(lines)
