"""
Unit tests for PII Privacy Sanitizer layer.
"""

import pytest
from llm.privacy import sanitize_text, sanitize_record_for_llm


class TestPiiSanitizer:
    """Test data sanitization and masking before LLM dispatch."""
    
    def test_sanitize_email_and_phone(self):
        """Test redacting email addresses and Indian phone numbers in freeform text."""
        raw_text = "Payment received from user john.doe@example.com, mobile +91 9876543210."
        clean_text = sanitize_text(raw_text)
        
        assert "john.doe@example.com" not in clean_text
        assert "[REDACTED_EMAIL]" in clean_text
        assert "9876543210" not in clean_text
        assert "[REDACTED_PHONE]" in clean_text
    
    def test_sanitize_record_structure(self):
        """Test redacting sensitive fields while preserving matching identifiers."""
        raw_record = {
            'entity_id': 'sett_001',
            'order_id': 'ORD_1234',
            'payment_id': 'PAY_5678',
            'settlement_utr': 'UTR999888',
            'amount': 1500.0,
            'fee': 35.40,
            'customer_name': 'Rahul Sharma',
            'email': 'rahul.s@example.com',
            'phone': '9876543210',
            'customer_ref': 'CUST_9901',
            'narration': 'UPI / 9876543210@paytm / ORD_1234'
        }
        
        clean_record = sanitize_record_for_llm(raw_record)
        
        # Preserved matching keys
        assert clean_record['entity_id'] == 'sett_001'
        assert clean_record['order_id'] == 'ORD_1234'
        assert clean_record['payment_id'] == 'PAY_5678'
        assert clean_record['settlement_utr'] == 'UTR999888'
        assert clean_record['amount'] == 1500.0
        assert clean_record['fee'] == 35.40
        
        # Redacted PII
        assert clean_record['customer_name'] == '[REDACTED_PII]'
        assert clean_record['email'] == '[REDACTED_PII]'
        assert clean_record['phone'] == '[REDACTED_PII]'
        assert clean_record['customer_ref'] == 'CUST_[REDACTED]'
    def test_nested_dict_and_list(self):
        raw = {
            'order_id': 'ORD_1',
            'meta': {'email': 'a@b.com', 'note': 'PAN ABCDE1234F'},
            'history': [{'phone': '9876543210'}, '4111-1111-1111-1111'],
        }
        clean = sanitize_record_for_llm(raw)
        blob = str(clean)
        assert 'a@b.com' not in blob
        assert 'ABCDE1234F' not in blob
        assert '9876543210' not in blob
        assert '4111-1111-1111-1111' not in blob
        assert clean['order_id'] == 'ORD_1'
