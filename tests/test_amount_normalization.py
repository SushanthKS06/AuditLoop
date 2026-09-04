"""
Financial-Grade Amount Normalization Tests

Proves that all amount representations are correctly canonicalized
to Decimal objects while preserving accounting semantics (e.g. negatives).
"""

from decimal import Decimal
from engine.matcher import DeterministicMatcher
import pandas as pd

def test_amount_normalization():
    matcher = DeterministicMatcher()
    
    # Standard numbers
    assert matcher._normalize_amount(1000) == Decimal("1000.00")
    assert matcher._normalize_amount(1000.0) == Decimal("1000.00")
    assert matcher._normalize_amount(Decimal("1000.00")) == Decimal("1000.00")
    
    # Strings with commas
    assert matcher._normalize_amount("1,000") == Decimal("1000.00")
    assert matcher._normalize_amount("1,000.00") == Decimal("1000.00")
    assert matcher._normalize_amount("10,00,000.50") == Decimal("1000000.50")
    
    # Currency symbols
    assert matcher._normalize_amount("₹1,000.00") == Decimal("1000.00")
    assert matcher._normalize_amount("$1000") == Decimal("1000.00")
    assert matcher._normalize_amount("INR 1000.00") == Decimal("1000.00")
    
    # Whitespace
    assert matcher._normalize_amount("  1000.00  ") == Decimal("1000.00")
    
    # Negative values
    assert matcher._normalize_amount("-1000.00") == Decimal("-1000.00")
    assert matcher._normalize_amount("- 1,000.00") == Decimal("-1000.00")
    
    # Accounting parentheses
    assert matcher._normalize_amount("(3,920.00)") == Decimal("-3920.00")
    assert matcher._normalize_amount("(1000)") == Decimal("-1000.00")
    assert matcher._normalize_amount("(₹1,000.00)") == Decimal("-1000.00")
    
    # Zero
    assert matcher._normalize_amount(0) == Decimal("0.00")
    assert matcher._normalize_amount("0.00") == Decimal("0.00")
    assert matcher._normalize_amount("(0.00)") == Decimal("0.00")
    
    # Rounding (Banker's rounding - ROUND_HALF_EVEN)
    assert matcher._normalize_amount("10.005") == Decimal("10.00")
    assert matcher._normalize_amount("10.015") == Decimal("10.02")
    
    # Null/NaN handling
    assert matcher._normalize_amount(None) is None
    assert matcher._normalize_amount(pd.NA) is None
    assert matcher._normalize_amount("") is None
    assert matcher._normalize_amount("invalid") is None
