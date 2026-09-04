"""
PII Privacy and Data Sanitization Layer

Sanitizes financial reconciliation records before sending them to external LLM APIs.
Redacts personal identifiable information (PII) including:
  - Email addresses
  - Indian phone numbers
  - PAN card numbers (AAAAA9999A format)
  - Bank account numbers (10-18 contiguous digits)
  - IFSC codes (XXXX0XXXXXX format)
  - UPI VPA handles (handle@provider)
  - Customer names, contact information

Preserves essential matching keys (entity IDs, order IDs, UTRs, amounts, dates)
so downstream deterministic comparisons remain valid.
"""

import re
from typing import Dict, Any, Optional


# ── Regex patterns for sensitive identifiers ───────────────────────────────

# Email addresses
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

# Indian mobile numbers (10-digit starting with 6-9, optional +91 prefix)
PHONE_REGEX = re.compile(r'(?:\+91[\-\s]?)?[6789]\d{9}')

# PAN card: 5 alpha + 4 digit + 1 alpha (e.g. ABCDE1234F)
PAN_REGEX = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')

# Bank account numbers: 10 to 18 contiguous digits (not already part of a longer number)
BANK_ACCOUNT_REGEX = re.compile(r'(?<!\d)\d{10,18}(?!\d)')

# IFSC code: 4 alpha + 0 + 6 alphanumeric (e.g. HDFC0001234)
IFSC_REGEX = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

# UPI VPA: word@word (guards against matching order IDs which use / not @)
UPI_VPA_REGEX = re.compile(r'\b[\w.\-]+@[a-zA-Z]{2,}\b')


# ── Field-name blocklist (exact key matches) ───────────────────────────────

_PII_EXACT_FIELDS = frozenset({
    'customer_name', 'beneficiary_name', 'name',
    'email',
    'phone', 'mobile', 'contact',
    'vpa', 'upi_id', 'upi_handle',
    'pan', 'pan_number',
    'account_number', 'bank_account', 'bank_account_number',
    'ifsc', 'ifsc_code',
})


def sanitize_text(text: Optional[str]) -> str:
    """
    Sanitize raw narration or freeform text by redacting PII patterns.

    Applies regex-based redaction for:
    - Email addresses → [REDACTED_EMAIL]
    - Indian phone numbers → [REDACTED_PHONE]
    - PAN numbers → [REDACTED_PAN]
    - Bank account numbers (10-18 digits) → [REDACTED_ACCOUNT]
    - IFSC codes → [REDACTED_IFSC]
    - UPI VPA handles → [REDACTED_VPA]

    Args:
        text: Input string potentially containing PII

    Returns:
        Redacted string with PII replaced by safe placeholder tokens
    """
    if not text:
        return ""

    s = str(text)
    # Order matters: apply more-specific patterns first to avoid partial matches.
    s = PAN_REGEX.sub("[REDACTED_PAN]", s)
    s = IFSC_REGEX.sub("[REDACTED_IFSC]", s)
    s = EMAIL_REGEX.sub("[REDACTED_EMAIL]", s)
    s = PHONE_REGEX.sub("[REDACTED_PHONE]", s)
    # UPI VPA after email so @-patterns already redacted don't double-match
    s = UPI_VPA_REGEX.sub("[REDACTED_VPA]", s)
    s = BANK_ACCOUNT_REGEX.sub("[REDACTED_ACCOUNT]", s)

    return s


def sanitize_record_for_llm(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Produce a sanitized, privacy-safe copy of a financial record for LLM context.

    Preserves:
    - Entity IDs, Order IDs, Payment IDs, UTRs (needed for correlation)
    - Numerical amounts, fees, taxes, settled amounts
    - ISO timestamps and dates

    Redacts/Masks:
    - Fields in _PII_EXACT_FIELDS → [REDACTED_PII]
    - customer_ref → CUST_[REDACTED]
    - Freeform narration and all other string fields → regex-based scrub

    Args:
        record: Raw transaction dictionary

    Returns:
        Sanitized transaction dictionary safe for external LLM dispatch.
    """
    if not record:
        return None

    clean = {}
    for k, v in record.items():
        if v is None:
            clean[k] = None
            continue

        k_lower = k.lower()

        # Exact field-name blocklist (case-insensitive key match)
        if k_lower in _PII_EXACT_FIELDS:
            clean[k] = "[REDACTED_PII]"
        elif k_lower == 'customer_ref':
            clean[k] = "CUST_[REDACTED]"
        elif k_lower == 'narration':
            # Narration is high-risk freeform text
            clean[k] = sanitize_text(str(v))
        elif isinstance(v, str):
            # Apply regex scrub to all other string values
            clean[k] = sanitize_text(v)
        else:
            # Non-string values (int, float, bool, list, dict) pass through unchanged
            clean[k] = v

    return clean
