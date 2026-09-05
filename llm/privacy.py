"""
PII Privacy and Data Sanitization Layer

Sanitizes financial reconciliation records before sending them to external LLM APIs.
This is field-and-regex redaction, not universal DLP.

Redacts:
  - Email addresses
  - Indian phone numbers
  - PAN card numbers (AAAAA9999A format)
  - Bank account numbers (10-18 contiguous digits)
  - IFSC codes (XXXX0XXXXXX format)
  - UPI VPA handles (handle@provider)
  - Card-like 13–19 digit grouped numbers
  - Beneficiary / customer name fields
  - Nested dicts and lists (unknown future fields included)

Preserves essential matching keys (entity IDs, order IDs, UTRs, amounts, dates).
"""

import re
from typing import Any, Dict, List, Optional, Union


EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'(?:\+91[\-\s]?)?[6789]\d{9}')
PAN_REGEX = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
BANK_ACCOUNT_REGEX = re.compile(r'(?<!\d)\d{10,18}(?!\d)')
IFSC_REGEX = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')
UPI_VPA_REGEX = re.compile(r'\b[\w.\-]+@[a-zA-Z]{2,}\b')
CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{1,7}\b')

_PII_EXACT_FIELDS = frozenset({
    'customer_name', 'beneficiary_name', 'name',
    'email',
    'phone', 'mobile', 'contact',
    'vpa', 'upi_id', 'upi_handle',
    'pan', 'pan_number',
    'account_number', 'bank_account', 'bank_account_number',
    'ifsc', 'ifsc_code',
    'card_number', 'card_pan', 'beneficiary_id', 'customer_id',
})

_PII_KEY_SUBSTRINGS = (
    'email', 'phone', 'mobile', 'pan', 'ifsc', 'vpa', 'upi',
    'account', 'card', 'beneficiary', 'customer_name',
)


def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""

    s = str(text)
    s = PAN_REGEX.sub("[REDACTED_PAN]", s)
    s = IFSC_REGEX.sub("[REDACTED_IFSC]", s)
    s = EMAIL_REGEX.sub("[REDACTED_EMAIL]", s)
    s = PHONE_REGEX.sub("[REDACTED_PHONE]", s)
    s = UPI_VPA_REGEX.sub("[REDACTED_VPA]", s)
    s = CARD_REGEX.sub("[REDACTED_CARD]", s)
    s = BANK_ACCOUNT_REGEX.sub("[REDACTED_ACCOUNT]", s)
    return s


def _key_looks_sensitive(key: str) -> bool:
    k = key.lower()
    if k in _PII_EXACT_FIELDS:
        return True
    return any(token in k for token in _PII_KEY_SUBSTRINGS)


def sanitize_value(value: Any, key: str = "") -> Any:
    """Sanitize an arbitrary JSON-like value (string, dict, list, scalar)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return sanitize_record_for_llm(value)
    if isinstance(value, list):
        return [sanitize_value(item, key) for item in value]
    if isinstance(value, str):
        if _key_looks_sensitive(key) and key.lower() != 'customer_ref':
            return "[REDACTED_PII]"
        return sanitize_text(value)
    return value


def sanitize_record_for_llm(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None

    clean: Dict[str, Any] = {}
    for k, v in record.items():
        if v is None:
            clean[k] = None
            continue

        k_lower = k.lower()

        if k_lower in _PII_EXACT_FIELDS:
            clean[k] = "[REDACTED_PII]"
        elif k_lower == 'customer_ref':
            clean[k] = "CUST_[REDACTED]"
        elif isinstance(v, dict):
            clean[k] = sanitize_record_for_llm(v)
        elif isinstance(v, list):
            clean[k] = [sanitize_value(item, k) for item in v]
        elif isinstance(v, str):
            if _key_looks_sensitive(k) and k_lower not in ('customer_ref',):
                clean[k] = "[REDACTED_PII]"
            else:
                clean[k] = sanitize_text(v)
        else:
            clean[k] = v

    return clean
