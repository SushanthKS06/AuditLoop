"""
Tests for the deterministic matcher.

Verifies that exact and fuzzy matching work correctly against known inputs.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from engine.matcher import DeterministicMatcher


@pytest.fixture
def matcher():
    """Create a matcher with default thresholds."""
    return DeterministicMatcher()


@pytest.fixture
def sample_settlements():
    """Sample settlements DataFrame."""
    return pd.DataFrame([
        {
            'entity_id': 'sett_001',
            'order_id': 'ORD_001',
            'payment_id': 'PAY_001',
            'settlement_utr': 'UTR123456',
            'amount': 1000.0,
            'settled_amount': 980.0,
            'created_at': '2026-09-01T10:00:00Z',
            'settled_at': '2026-09-02T10:00:00Z'
        },
        {
            'entity_id': 'sett_002',
            'order_id': 'ORD_002',
            'payment_id': 'PAY_002',
            'settlement_utr': 'UTR789012',
            'amount': 5000.0,
            'settled_amount': 4900.0,
            'created_at': '2026-09-01T11:00:00Z',
            'settled_at': '2026-09-02T11:00:00Z'
        }
    ])


@pytest.fixture
def sample_bank():
    """Sample bank statement DataFrame."""
    return pd.DataFrame([
        {
            'txn_id': 'TXN_001',
            'utr': 'UTR123456',
            'amount': 980.0,
            'value_date': '2026-09-02',
            'narration': 'NEFT - UTR123456 - ORD_001',
            'reference': 'PAY_001'
        },
        {
            'txn_id': 'TXN_002',
            'utr': 'UTR789012',
            'amount': 4900.0,
            'value_date': '2026-09-02',
            'narration': 'IMPS - UTR789012 - ORD_002',
            'reference': 'PAY_002'
        }
    ])


@pytest.fixture
def sample_ledger():
    """Sample ledger DataFrame."""
    return pd.DataFrame([
        {
            'order_id': 'ORD_001',
            'expected_amount': 1000.0,
            'order_date': '2026-09-01',
            'payment_id': 'PAY_001',
            'status': 'completed'
        },
        {
            'order_id': 'ORD_002',
            'expected_amount': 5000.0,
            'order_date': '2026-09-01',
            'payment_id': 'PAY_002',
            'status': 'completed'
        }
    ])


class TestStage1ExactMatch:
    """Test Stage 1 exact matching."""
    
    def test_exact_utr_match(self, matcher, sample_settlements, sample_bank):
        """Test that matching UTRs are found."""
        ledger = pd.DataFrame()  # Empty ledger
        
        matched, _, _, _, audits = matcher.stage1_exact_match(
            sample_settlements, sample_bank, ledger
        )
        
        assert len(matched) == 2  # Both should match on UTR
        assert all(m['match_type'] == 'exact_utr' for _, m in matched.iterrows())
        assert all(m['confidence'] == 1.0 for _, m in matched.iterrows())
    
    def test_no_match_on_different_utr(self, matcher):
        """Test that different UTRs don't match."""
        settlements = pd.DataFrame([{
            'entity_id': 'sett_001',
            'settlement_utr': 'UTR111111',
            'order_id': '',
            'payment_id': ''
        }])
        
        bank = pd.DataFrame([{
            'txn_id': 'TXN_001',
            'utr': 'UTR222222',
            'amount': 1000.0,
            'value_date': '2026-09-01',
            'narration': '',
            'reference': ''
        }])
        
        ledger = pd.DataFrame()
        
        matched, unmatched_sett, _, _, _ = matcher.stage1_exact_match(
            settlements, bank, ledger
        )
        
        assert len(matched) == 0
        assert len(unmatched_sett) == 1


class TestStage2FuzzyMatch:
    """Test Stage 2 fuzzy matching."""
    
    def test_fuzzy_amount_match(self, matcher):
        """Test fuzzy matching on similar amounts."""
        settlements = pd.DataFrame([{
            'entity_id': 'sett_001',
            'settlement_utr': '',
            'order_id': '',
            'amount': 1000.0,
            'settled_at': '2026-09-01'
        }])
        
        bank = pd.DataFrame([{
            'txn_id': 'TXN_001',
            'utr': '',
            'amount': 999.5,  # 0.05% difference - should match
            'value_date': '2026-09-01',
            'narration': 'Similar reference',
            'reference': ''
        }])
        
        ledger = pd.DataFrame()
        
        matched, low_conf, _, _, _ = matcher.stage2_fuzzy_match(
            settlements, bank, ledger
        )
        
        # Should be a high-confidence match
        assert len(matched) >= 1 or len(low_conf) >= 1
    
    def test_low_confidence_on_large_amount_diff(self, matcher):
        """Test that large amount differences result in low confidence."""
        settlements = pd.DataFrame([{
            'entity_id': 'sett_001',
            'settlement_utr': '',
            'order_id': '',
            'amount': 1000.0,
            'settled_at': '2026-09-01'
        }])
        
        bank = pd.DataFrame([{
            'txn_id': 'TXN_001',
            'utr': '',
            'amount': 500.0,  # 50% difference - should NOT match well
            'value_date': '2026-09-01',
            'narration': '',
            'reference': ''
        }])
        
        ledger = pd.DataFrame()
        
        matched, low_conf, _, _, _ = matcher.stage2_fuzzy_match(
            settlements, bank, ledger
        )
        
        # Should be low confidence or unmatched
        assert len(matched) == 0


class TestNormalization:
    """Test field normalization functions."""
    
    def test_normalize_amount_with_commas(self, matcher):
        """Test amount normalization with commas."""
        assert matcher._normalize_amount("1,000.50") == 1000.50
        assert matcher._normalize_amount("₹1,500") == 1500.0
    
    def test_normalize_date_formats(self, matcher):
        """Test various date format parsing."""
        assert matcher._normalize_date("2026-09-01") is not None
        assert matcher._normalize_date("01/09/2026") is not None
        assert matcher._normalize_date("2026-09-01T10:00:00Z") is not None
    
    def test_normalize_text(self, matcher):
        """Test text normalization."""
        assert matcher._normalize_text("  HELLO  ") == "hello"
        assert matcher._normalize_text(None) == ""
