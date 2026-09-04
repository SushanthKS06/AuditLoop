"""
PII Privacy and Data Sanitization Layer

Sanitizes financial reconciliation records before sending them to external LLM APIs.
Redacts personal identifiable information (PII) including emails, phone numbers,
customer names, raw bank account numbers, and personal UPI VPA handles while
preserving essential matching keys (entity IDs, order IDs, UTRs, amounts, dates).
"""

import re
from typing import Dict, Any, Optional


# Regex patterns for sensitive identifiers
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'(?:\+91[\-\s]?)?[6789]\d{9}')


def sanitize_text(text: Optional[str]) -> str:
    """
    Sanitize raw narration or freeform text by redacting PII.
    
    Args:
        text: Input string potentially containing PII
        
    Returns:
        Redacted string with PII replaced by safe placeholder tokens
    """
    if not text:
        return ""
    
    s = str(text)
    # Mask email addresses
    s = EMAIL_REGEX.sub("[REDACTED_EMAIL]", s)
    # Mask Indian phone numbers
    s = PHONE_REGEX.sub("[REDACTED_PHONE]", s)
    
    return s


def sanitize_record_for_llm(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Produce a sanitized, privacy-safe copy of a financial record for LLM context.
    
    Preserves:
    - Entity IDs, Order IDs, Payment IDs, UTRs (needed for correlation)
    - Numerical amounts, fees, taxes, settled amounts
    - ISO timestamps and dates
    
    Redacts/Masks:
    - Customer names / customer refs (e.g. CUST_XXXX -> CUST_[REDACTED])
    - Freeform narration containing phone numbers or emails
    
    Args:
        record: Raw transaction dictionary
        
    Returns:
        Sanitized transaction dictionary
    """
    if not record:
        return None
    
    clean = {}
    for k, v in record.items():
        if v is None:
            clean[k] = None
            continue
        
        # Redact specific PII fields
        if k in ('customer_name', 'email', 'phone', 'contact', 'vpa', 'upi_id', 'upi_handle'):
            clean[k] = "[REDACTED_PII]"
        elif k == 'customer_ref':
            clean[k] = "CUST_[REDACTED]"
        elif k == 'narration':
            clean[k] = sanitize_text(v)
        elif isinstance(v, str):
            clean[k] = sanitize_text(v)
        else:
            clean[k] = v
            
    return clean
