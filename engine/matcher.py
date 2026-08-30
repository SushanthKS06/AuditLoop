"""
Deterministic Matching Engine

Stage 1: Exact match via hash join on normalized keys
Stage 2: Fuzzy match with scoring on amount delta, date window, reference similarity

No LLM involvement here - pure deterministic logic.
Every decision writes an audit record.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from rapidfuzz import fuzz


class DeterministicMatcher:
    """
    Two-stage deterministic matching engine.
    
    Stage 1: Exact match on normalized keys (UTR, order_id, payment_id)
    Stage 2: Fuzzy match with configurable thresholds
    
    Design decision: LLM is NEVER called here. It only sees exceptions
    from Stage 3 (unmatched or low-confidence records).
    """
    
    def __init__(
        self,
        amount_threshold_pct: float = 1.0,  # Max 1% amount difference for fuzzy match
        date_window_days: int = 3,  # Match within 3-day window
        confidence_threshold: float = 0.85,  # Minimum confidence to auto-match
        text_similarity_threshold: int = 70  # RapidFuzz token ratio threshold
    ):
        """
        Args:
            amount_threshold_pct: Maximum percentage difference in amounts for fuzzy match
            date_window_days: Maximum days between transaction dates for match
            confidence_threshold: Minimum confidence score to auto-commit a match
            text_similarity_threshold: Minimum text similarity score for reference matching
            
        WHY these defaults:
        - 1% amount tolerance handles minor rounding differences without false positives
        - 3-day window accommodates weekend/holiday settlement delays common in India
        - 0.85 confidence requires strong evidence before auto-matching
        - 70% text similarity catches typos and formatting variations
        """
        self.amount_threshold_pct = amount_threshold_pct
        self.date_window_days = date_window_days
        self.confidence_threshold = confidence_threshold
        self.text_similarity_threshold = text_similarity_threshold
        
        # Audit log callback - set by the pipeline
        self.audit_callback = None
    
    def set_audit_callback(self, callback):
        """Set callback for writing audit records."""
        self.audit_callback = callback
    
    def _normalize_amount(self, amount: Any) -> Optional[float]:
        """Normalize amount to float, handling various formats."""
        if amount is None or pd.isna(amount):
            return None
        if isinstance(amount, (int, float)):
            return float(amount)
        try:
            # Handle string amounts with commas, currency symbols
            cleaned = str(amount).replace(',', '').replace('₹', '').replace('$', '').strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    
    def _normalize_date(self, date_str: Any) -> Optional[datetime]:
        """Normalize date string to datetime object."""
        if date_str is None or pd.isna(date_str):
            return None
        if isinstance(date_str, datetime):
            return date_str
        
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%d-%m-%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        return None
    
    def _normalize_text(self, text: Any) -> str:
        """Normalize text for comparison - lowercase, strip whitespace."""
        if text is None or pd.isna(text):
            return ""
        return str(text).lower().strip()
    
    def _hash_key(self, value: str) -> str:
        """Create a hash key for exact matching."""
        return hashlib.md5(value.encode()).hexdigest()
    
    def stage1_exact_match(
        self,
        settlements: pd.DataFrame,
        bank: pd.DataFrame,
        ledger: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Stage 1: Exact match via hash join on normalized keys.
        
        Priority matching keys:
        1. UTR (settlement_utr <-> utr)
        2. order_id
        3. payment_id
        
        Returns:
            Tuple of (matched_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records)
        """
        audit_records = []
        
        # Normalize key columns
        settlements['utr_norm'] = settlements['settlement_utr'].apply(self._normalize_text)
        settlements['order_norm'] = settlements['order_id'].apply(self._normalize_text)
        settlements['payment_norm'] = settlements['payment_id'].apply(self._normalize_text)
        
        bank['utr_norm'] = bank['utr'].apply(self._normalize_text)
        bank['payment_norm'] = bank['reference'].apply(self._normalize_text) if 'reference' in bank.columns else ''
        
        ledger['order_norm'] = ledger['order_id'].apply(self._normalize_text)
        ledger['payment_norm'] = ledger['payment_id'].apply(self._normalize_text) if 'payment_id' in ledger.columns else ''
        
        matched_records = []
        matched_settlement_ids = set()
        matched_bank_ids = set()
        matched_ledger_ids = set()
        
        # Match on UTR first (highest confidence)
        for _, sett in settlements.iterrows():
            if sett['utr_norm'] and sett['utr_norm'] != '':
                bank_match = bank[bank['utr_norm'] == sett['utr_norm']]
                if len(bank_match) > 0:
                    bank_row = bank_match.iloc[0]
                    match_record = {
                        'settlement': sett.to_dict(),
                        'bank': bank_row.to_dict(),
                        'ledger': None,
                        'match_type': 'exact_utr',
                        'confidence': 1.0,
                        'rule_fired': 'utr_exact_match'
                    }
                    matched_records.append(match_record)
                    matched_settlement_ids.add(sett.name if hasattr(sett, 'name') else id(sett))
                    matched_bank_ids.add(bank_row.name if hasattr(bank_row, 'name') else id(bank_row))
                    
                    audit_records.append({
                        'record_ids': f"{sett.get('entity_id', '')}-{bank_row.get('txn_id', '')}",
                        'stage': 'stage1_exact',
                        'rule_fired': 'utr_exact_match',
                        'confidence': 1.0,
                        'decision': 'matched',
                        'match_type': 'exact_utr'
                    })
        
        # Match on order_id
        for _, sett in settlements.iterrows():
            if sett.name in matched_settlement_ids:
                continue
            if sett['order_norm'] and sett['order_norm'] != '':
                ledger_match = ledger[ledger['order_norm'] == sett['order_norm']]
                if len(ledger_match) > 0:
                    ledger_row = ledger_match.iloc[0]
                    match_record = {
                        'settlement': sett.to_dict(),
                        'bank': None,
                        'ledger': ledger_row.to_dict(),
                        'match_type': 'exact_order',
                        'confidence': 1.0,
                        'rule_fired': 'order_exact_match'
                    }
                    matched_records.append(match_record)
                    matched_settlement_ids.add(sett.name if hasattr(sett, 'name') else id(sett))
                    matched_ledger_ids.add(ledger_row.name if hasattr(ledger_row, 'name') else id(ledger_row))
                    
                    audit_records.append({
                        'record_ids': f"{sett.get('entity_id', '')}-{ledger_row.get('order_id', '')}",
                        'stage': 'stage1_exact',
                        'rule_fired': 'order_exact_match',
                        'confidence': 1.0,
                        'decision': 'matched',
                        'match_type': 'exact_order'
                    })
        
        # Build unmatched DataFrames
        unmatched_settlements = settlements[~settlements.index.isin(matched_settlement_ids)]
        unmatched_bank = bank[~bank.index.isin(matched_bank_ids)]
        unmatched_ledger = ledger[~ledger.index.isin(matched_ledger_ids)]
        
        matched_df = pd.DataFrame(matched_records)
        
        if self.audit_callback:
            for record in audit_records:
                self.audit_callback(record)
        
        return matched_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records
    
    def stage2_fuzzy_match(
        self,
        settlements: pd.DataFrame,
        bank: pd.DataFrame,
        ledger: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Stage 2: Fuzzy match with scoring.
        
        Scoring factors:
        - Amount delta percentage (weight: 0.4)
        - Date proximity (weight: 0.3)
        - Text similarity on references (weight: 0.3)
        
        Returns:
            Tuple of (matched_df, low_confidence_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records)
        """
        audit_records = []
        
        matched_records = []
        low_confidence_records = []
        matched_settlement_ids = set()
        matched_bank_ids = set()
        matched_ledger_ids = set()
        
        # For each unmatched settlement, find best candidate match
        for _, sett in settlements.iterrows():
            best_score = 0.0
            best_match = None
            best_match_type = None
            best_rule = None
            
            sett_amount = self._normalize_amount(sett.get('settled_amount') or sett.get('amount'))
            sett_date = self._normalize_date(sett.get('settled_at') or sett.get('created_at'))
            
            # Try matching against bank statements
            for _, bank_row in bank.iterrows():
                bank_amount = self._normalize_amount(bank_row.get('amount'))
                bank_date = self._normalize_date(bank_row.get('value_date'))
                
                score, rule = self._score_pair(
                    sett_amount, sett_date, 
                    bank_amount, bank_date,
                    sett.get('settlement_utr', ''), bank_row.get('narration', '')
                )
                
                if score > best_score:
                    best_score = score
                    best_match = ('bank', bank_row)
                    best_rule = rule
            
            # Try matching against ledger
            for _, ledger_row in ledger.iterrows():
                ledger_amount = self._normalize_amount(ledger_row.get('expected_amount'))
                ledger_date = self._normalize_date(ledger_row.get('order_date'))
                
                score, rule = self._score_pair(
                    sett_amount, sett_date,
                    ledger_amount, ledger_date,
                    sett.get('order_id', ''), ledger_row.get('order_id', '')
                )
                
                if score > best_score:
                    best_score = score
                    best_match = ('ledger', ledger_row)
                    best_rule = rule
            
            if best_match:
                match_record = {
                    'settlement': sett.to_dict(),
                    'bank': best_match[1].to_dict() if best_match[0] == 'bank' else None,
                    'ledger': best_match[1].to_dict() if best_match[0] == 'ledger' else None,
                    'match_type': f"fuzzy_{best_match[0]}",
                    'confidence': best_score,
                    'rule_fired': best_rule
                }
                
                if best_score >= self.confidence_threshold:
                    matched_records.append(match_record)
                    decision = 'matched'
                else:
                    low_confidence_records.append(match_record)
                    decision = 'low_confidence'
                
                if best_match[0] == 'bank':
                    matched_bank_ids.add(best_match[1].name if hasattr(best_match[1], 'name') else id(best_match[1]))
                else:
                    matched_ledger_ids.add(best_match[1].name if hasattr(best_match[1], 'name') else id(best_match[1]))
                
                matched_settlement_ids.add(sett.name if hasattr(sett, 'name') else id(sett))
                
                audit_records.append({
                    'record_ids': f"{sett.get('entity_id', '')}-{best_match[1].get('txn_id', best_match[1].get('order_id', ''))}",
                    'stage': 'stage2_fuzzy',
                    'rule_fired': best_rule,
                    'confidence': best_score,
                    'decision': decision,
                    'match_type': f"fuzzy_{best_match[0]}"
                })
        
        # Build result DataFrames
        matched_df = pd.DataFrame(matched_records) if matched_records else pd.DataFrame()
        low_confidence_df = pd.DataFrame(low_confidence_records) if low_confidence_records else pd.DataFrame()
        unmatched_settlements = settlements[~settlements.index.isin(matched_settlement_ids)]
        unmatched_bank = bank[~bank.index.isin(matched_bank_ids)]
        unmatched_ledger = ledger[~ledger.index.isin(matched_ledger_ids)]
        
        if self.audit_callback:
            for record in audit_records:
                self.audit_callback(record)
        
        return matched_df, low_confidence_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records
    
    def _score_pair(
        self,
        amount1: Optional[float],
        date1: Optional[datetime],
        amount2: Optional[float],
        date2: Optional[datetime],
        text1: str,
        text2: str
    ) -> Tuple[float, str]:
        """
        Score a potential match pair.
        
        Returns:
            Tuple of (confidence_score 0-1, rule_description)
        """
        if amount1 is None or amount2 is None:
            return 0.0, "missing_amount"
        
        # Amount similarity (40% weight)
        amount_diff_pct = abs(amount1 - amount2) / max(amount1, amount2, 1) * 100
        if amount_diff_pct <= self.amount_threshold_pct:
            amount_score = 1.0
        elif amount_diff_pct <= 5:
            amount_score = max(0, 1.0 - (amount_diff_pct - self.amount_threshold_pct) / 5)
        else:
            amount_score = 0.0
        
        # Date proximity (30% weight)
        if date1 and date2:
            date_diff = abs((date1 - date2).days)
            if date_diff <= 1:
                date_score = 1.0
            elif date_diff <= self.date_window_days:
                date_score = max(0, 1.0 - (date_diff - 1) / self.date_window_days)
            else:
                date_score = 0.0
        else:
            date_score = 0.5  # Neutral if dates missing
        
        # Text similarity (30% weight)
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)
        if text1_norm and text2_norm:
            text_score = fuzz.token_ratio(text1_norm, text2_norm) / 100
        else:
            text_score = 0.5
        
        # Weighted total
        total_score = (amount_score * 0.4) + (date_score * 0.3) + (text_score * 0.3)
        
        # Determine which rule fired
        if amount_score >= 0.9 and date_score >= 0.9:
            rule = "high_confidence_amount_date"
        elif amount_score >= 0.8:
            rule = "amount_close_match"
        elif text_score >= (self.text_similarity_threshold / 100):
            rule = "text_similarity_match"
        else:
            rule = "weak_candidate"
        
        return total_score, rule
    
    def get_exceptions(
        self,
        low_confidence: pd.DataFrame,
        unmatched_settlements: pd.DataFrame,
        unmatched_bank: pd.DataFrame,
        unmatched_ledger: pd.DataFrame
    ) -> List[Dict]:
        """
        Collect all exceptions for LLM review (Stage 3).
        
        Exceptions include:
        - Low-confidence matches from Stage 2
        - Unmatched records from all sources
        """
        exceptions = []
        
        # Low-confidence matches
        if not low_confidence.empty:
            for _, row in low_confidence.iterrows():
                exceptions.append({
                    'type': 'low_confidence',
                    'settlement': row.get('settlement', {}),
                    'counterpart': row.get('bank') or row.get('ledger'),
                    'confidence': row.get('confidence', 0),
                    'rule_fired': row.get('rule_fired', '')
                })
        
        # Unmatched settlements
        if not unmatched_settlements.empty:
            for _, row in unmatched_settlements.iterrows():
                exceptions.append({
                    'type': 'unmatched_settlement',
                    'settlement': row.to_dict(),
                    'counterpart': None,
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        # Unmatched bank transactions
        if not unmatched_bank.empty:
            for _, row in unmatched_bank.iterrows():
                exceptions.append({
                    'type': 'unmatched_bank',
                    'settlement': None,
                    'counterpart': row.to_dict(),
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        # Unmatched ledger entries
        if not unmatched_ledger.empty:
            for _, row in unmatched_ledger.iterrows():
                exceptions.append({
                    'type': 'unmatched_ledger',
                    'settlement': None,
                    'counterpart': row.to_dict(),
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        return exceptions
