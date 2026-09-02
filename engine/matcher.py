"""
Deterministic Matching Engine

Stage 1: Exact match via hash join on normalized keys
Stage 2: Fuzzy match with scoring on amount delta, date window, reference similarity, and fee deductions

No LLM involvement here - pure deterministic logic.
Every decision writes an audit record.
"""

import hashlib
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from rapidfuzz import fuzz


class DeterministicMatcher:
    """
    Two-stage deterministic matching engine.
    
    Stage 1: Exact match on normalized keys (UTR, order_id, payment_id)
    Stage 2: Fuzzy match with fee awareness and configurable thresholds
    
    Design decision: LLM is NEVER called here. It only sees exceptions
    from Stage 3 (unmatched or low-confidence records).
    """
    
    def __init__(
        self,
        amount_threshold_pct: float = 1.0,  # Max 1% amount difference for exact fuzzy match
        date_window_days: int = 3,  # Match within 3-day window
        confidence_threshold: float = 0.85,  # Minimum confidence to auto-match
        text_similarity_threshold: int = 70,  # RapidFuzz token ratio threshold
        expected_mdr_fee_pct: float = 2.0  # Standard gateway MDR fee deduction (~2% + GST)
    ):
        """
        Args:
            amount_threshold_pct: Maximum percentage difference in amounts for fuzzy match
            date_window_days: Maximum days between transaction dates for match
            confidence_threshold: Minimum confidence score to auto-commit a match
            text_similarity_threshold: Minimum text similarity score for reference matching
            expected_mdr_fee_pct: Expected payment gateway MDR fee rate
        """
        self.amount_threshold_pct = amount_threshold_pct
        self.date_window_days = date_window_days
        self.confidence_threshold = confidence_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.expected_mdr_fee_pct = expected_mdr_fee_pct
        
        # Audit log callback - set by the pipeline
        self.audit_callback = None
    
    def set_audit_callback(self, callback):
        """Set callback for writing audit records."""
        self.audit_callback = callback
    
    def _normalize_amount(self, amount: Any) -> Optional[float]:
        """
        Normalize amount to high-precision float using Decimal and banker's rounding.
        Handles accounting parentheses (1,250.00) -> -1250.00, currency symbols, and commas.
        """
        if amount is None or pd.isna(amount):
            return None
        if isinstance(amount, (int, float)):
            # Normalize to 2 decimal places using Decimal banker's rounding
            try:
                dec = Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
                return float(dec)
            except (InvalidOperation, ValueError):
                return float(amount)
        try:
            cleaned = str(amount).strip()
            is_negative = False
            if cleaned.startswith('(') and cleaned.endswith(')'):
                is_negative = True
                cleaned = cleaned[1:-1]
            elif cleaned.startswith('-'):
                is_negative = True
                cleaned = cleaned[1:]
            
            # Clean symbols, codes, and formatting
            cleaned = cleaned.replace(',', '').replace('₹', '').replace('$', '').replace('INR', '').replace('EUR', '').replace('GBP', '').strip()
            dec = Decimal(cleaned).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
            val = float(dec)
            return -val if is_negative else val
        except (InvalidOperation, ValueError, TypeError):
            return None
    
    def _normalize_date(self, date_str: Any) -> Optional[datetime]:
        """Normalize date string to UTC datetime object."""
        if date_str is None or pd.isna(date_str):
            return None
        if isinstance(date_str, datetime):
            return date_str if date_str.tzinfo else date_str.replace(tzinfo=timezone.utc)
        
        cleaned = str(date_str).replace('Z', '+00:00').strip()
        
        # Try ISO format first
        try:
            dt = datetime.fromisoformat(cleaned)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(cleaned.split('+')[0].strip(), fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
    
    def _normalize_text(self, text: Any) -> str:
        """Normalize text for comparison - lowercase, strip whitespace, handle null strings."""
        if text is None or pd.isna(text):
            return ""
        s = str(text).lower().strip()
        if s in ('none', 'nan', 'null', 'undefined', '<na>'):
            return ""
        return s
    
    def _hash_key(self, value: str) -> str:
        """Create a hash key for exact matching."""
        return hashlib.md5(value.encode()).hexdigest()
    
    def stage1_exact_match(
        self,
        settlements: pd.DataFrame,
        bank: pd.DataFrame,
        ledger: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Stage 1: Exact match via high-performance hash join on normalized keys.
        
        Complexity: O(N + M) utilizing hash lookups rather than nested DataFrame filters.
        
        Priority matching keys:
        1. UTR (settlement_utr <-> utr)
        2. order_id
        3. payment_id
        
        Returns:
            Tuple of (matched_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records)
        """
        audit_records = []
        
        sett_df = settlements.copy()
        bank_df = bank.copy()
        ledger_df = ledger.copy()
        
        # Normalize key columns
        sett_df['utr_norm'] = sett_df['settlement_utr'].apply(self._normalize_text) if 'settlement_utr' in sett_df.columns else ''
        sett_df['order_norm'] = sett_df['order_id'].apply(self._normalize_text) if 'order_id' in sett_df.columns else ''
        sett_df['payment_norm'] = sett_df['payment_id'].apply(self._normalize_text) if 'payment_id' in sett_df.columns else ''
        
        bank_df['utr_norm'] = bank_df['utr'].apply(self._normalize_text) if 'utr' in bank_df.columns else ''
        bank_df['payment_norm'] = bank_df['reference'].apply(self._normalize_text) if 'reference' in bank_df.columns else ''
        
        ledger_df['order_norm'] = ledger_df['order_id'].apply(self._normalize_text) if 'order_id' in ledger_df.columns else ''
        ledger_df['payment_norm'] = ledger_df['payment_id'].apply(self._normalize_text) if 'payment_id' in ledger_df.columns else ''
        
        # Build high-speed hash map indices for Bank (O(M))
        bank_by_utr: Dict[str, List[Any]] = {}
        bank_by_payment: Dict[str, List[Any]] = {}
        for idx, bank_row in bank_df.iterrows():
            u = bank_row['utr_norm']
            p = bank_row['payment_norm']
            if u:
                bank_by_utr.setdefault(u, []).append(bank_row)
            if p:
                bank_by_payment.setdefault(p, []).append(bank_row)
        
        # Build high-speed hash map indices for Ledger (O(L))
        ledger_by_order: Dict[str, List[Any]] = {}
        ledger_by_payment: Dict[str, List[Any]] = {}
        for idx, ledger_row in ledger_df.iterrows():
            o = ledger_row['order_norm']
            p = ledger_row['payment_norm']
            if o:
                ledger_by_order.setdefault(o, []).append(ledger_row)
            if p:
                ledger_by_payment.setdefault(p, []).append(ledger_row)
        
        matched_records = []
        matched_settlement_ids = set()
        matched_bank_ids = set()
        matched_ledger_ids = set()
        
        # Stage 1: Exact 3-Way Hash Matching (O(N))
        for _, sett in sett_df.iterrows():
            matched_bank_row = None
            matched_ledger_row = None
            rule_fired = []
            
            # 1. Match Bank (UTR first, then payment reference)
            if sett['utr_norm'] and sett['utr_norm'] in bank_by_utr:
                for b in bank_by_utr[sett['utr_norm']]:
                    if b.name not in matched_bank_ids:
                        # Guard: bank row must have a usable amount
                        if self._normalize_amount(b.get('amount')) is None:
                            continue
                        matched_bank_row = b
                        rule_fired.append('utr_exact_match')
                        break
            
            if matched_bank_row is None and sett['payment_norm'] and sett['payment_norm'] in bank_by_payment:
                for b in bank_by_payment[sett['payment_norm']]:
                    if b.name not in matched_bank_ids:
                        # Guard: bank row must have a usable amount
                        if self._normalize_amount(b.get('amount')) is None:
                            continue
                        # Guard: if both settlement and bank have non-empty UTRs and
                        # they DISAGREE, this bank row is an orphan/wrong counterpart.
                        # (The primary UTR path already enforces exact agreement;
                        #  this path only fires when settlement UTR found nothing in bank.)
                        sett_utr = self._normalize_text(sett.get('settlement_utr', ''))
                        bank_utr = self._normalize_text(b.get('utr', ''))
                        if sett_utr and bank_utr and sett_utr != bank_utr:
                            # Cross-identifier UTR conflict: skip this candidate
                            continue
                        matched_bank_row = b
                        rule_fired.append('payment_id_bank_match')
                        break

            
            # 2. Match Ledger (order_id first, then payment_id)
            if sett['order_norm'] and sett['order_norm'] in ledger_by_order:
                for l in ledger_by_order[sett['order_norm']]:
                    if l.name not in matched_ledger_ids:
                        # Guard: ledger row must have a usable amount
                        if self._normalize_amount(l.get('expected_amount')) is None:
                            continue
                        matched_ledger_row = l
                        rule_fired.append('order_exact_match')
                        break
            
            if matched_ledger_row is None and sett['payment_norm'] and sett['payment_norm'] in ledger_by_payment:
                for l in ledger_by_payment[sett['payment_norm']]:
                    if l.name not in matched_ledger_ids:
                        # Guard 1: ledger row must have a usable amount (orphan check)
                        if self._normalize_amount(l.get('expected_amount')) is None:
                            continue
                        # Guard 2: if both settlement and ledger have an order_id and
                        # they DISAGREE, this is a duplicate-suspect / wrong-counterpart
                        # scenario — reject rather than committing on payment_id alone.
                        sett_order = self._normalize_text(sett.get('order_id', ''))
                        ledger_order = self._normalize_text(l.get('order_id', ''))
                        if sett_order and ledger_order and sett_order != ledger_order:
                            # Cross-identifier conflict: skip this candidate
                            continue
                        matched_ledger_row = l
                        rule_fired.append('payment_id_ledger_match')
                        break
            
            # Require bank leg for ANY match
            if matched_bank_row is not None:
                match_type = 'exact_3way' if matched_ledger_row is not None else 'exact_utr'
                
                match_record = {
                    'settlement': sett.to_dict(),
                    'bank': matched_bank_row.to_dict() if matched_bank_row is not None else None,
                    'ledger': matched_ledger_row.to_dict() if matched_ledger_row is not None else None,
                    'match_type': match_type,
                    'confidence': 1.0,
                    'rule_fired': "+".join(rule_fired) if rule_fired else 'exact_match',
                    'final_status': 'matched'
                }
                matched_records.append(match_record)
                matched_settlement_ids.add(sett.name)
                
                if matched_bank_row is not None:
                    matched_bank_ids.add(matched_bank_row.name)
                if matched_ledger_row is not None:
                    matched_ledger_ids.add(matched_ledger_row.name)
                
                record_id_parts = [str(sett.get('entity_id') or sett.get('payment_id') or '')]
                if matched_bank_row is not None:
                    record_id_parts.append(str(matched_bank_row.get('txn_id', '')))
                if matched_ledger_row is not None:
                    record_id_parts.append(str(matched_ledger_row.get('order_id', '')))
                
                audit_records.append({
                    'record_ids': "-".join(filter(None, record_id_parts)),
                    'stage': 'stage1_exact',
                    'rule_fired': "+".join(rule_fired) if rule_fired else 'exact_match',
                    'confidence': 1.0,
                    'decision': 'matched',
                    'match_type': match_type,
                    'final_status': 'matched'
                })

        
        # Build unmatched DataFrames
        unmatched_settlements = sett_df[~sett_df.index.isin(matched_settlement_ids)]
        unmatched_bank = bank_df[~bank_df.index.isin(matched_bank_ids)]
        unmatched_ledger = ledger_df[~ledger_df.index.isin(matched_ledger_ids)]
        
        matched_df = pd.DataFrame(matched_records) if matched_records else pd.DataFrame()
        
        if self.audit_callback:
            for record in audit_records:
                self.audit_callback(record)
        
        return matched_df, unmatched_settlements, unmatched_bank, unmatched_ledger, audit_records
    
    def stage2_fuzzy_match(
        self,
        settlements: pd.DataFrame,
        bank: pd.DataFrame,
        ledger: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Stage 2: Fuzzy match with fee awareness, scoring, and candidate indexing.
        
        Scoring factors:
        - Amount delta percentage or fee-deducted match (weight: 0.4)
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
        
        # Pre-normalize bank and ledger entries for fast candidate iteration
        parsed_bank = []
        for idx, bank_row in bank.iterrows():
            parsed_bank.append({
                'row': bank_row,
                'name': bank_row.name,
                'amount': self._normalize_amount(bank_row.get('amount')),
                'date': self._normalize_date(bank_row.get('value_date')),
                'narration': bank_row.get('narration', '')
            })
            
        parsed_ledger = []
        for idx, ledger_row in ledger.iterrows():
            parsed_ledger.append({
                'row': ledger_row,
                'name': ledger_row.name,
                'amount': self._normalize_amount(ledger_row.get('expected_amount')),
                'date': self._normalize_date(ledger_row.get('order_date')),
                'order_id': ledger_row.get('order_id', '')
            })
        
        # For each unmatched settlement, find best candidate match
        for _, sett in settlements.iterrows():
            best_score = 0.0
            best_match = None
            best_rule = None
            
            sett_amount = self._normalize_amount(sett.get('settled_amount') if pd.notna(sett.get('settled_amount')) else sett.get('amount'))
            sett_date = self._normalize_date(sett.get('settled_at') if pd.notna(sett.get('settled_at')) else sett.get('created_at'))
            sett_fee = self._normalize_amount(sett.get('fee'))
            
            # Try matching against available bank statements
            for b_entry in parsed_bank:
                if b_entry['name'] in matched_bank_ids:
                    continue
                
                score, rule = self._score_pair(
                    sett_amount, sett_date, 
                    b_entry['amount'], b_entry['date'],
                    sett.get('settlement_utr', ''), b_entry['narration'],
                    fee1=sett_fee, is_ledger=False
                )
                
                if score > best_score:
                    best_score = score
                    best_match = ('bank', b_entry['row'])
                    best_rule = rule
            
            # Try matching against available ledger
            for l_entry in parsed_ledger:
                if l_entry['name'] in matched_ledger_ids:
                    continue
                
                score, rule = self._score_pair(
                    sett_amount, sett_date,
                    l_entry['amount'], l_entry['date'],
                    sett.get('order_id', ''), l_entry['order_id'],
                    fee1=sett_fee, is_ledger=True
                )
                
                if score > best_score:
                    best_score = score
                    best_match = ('ledger', l_entry['row'])
                    best_rule = rule
            
            if best_match and best_score > 0.4:
                match_record = {
                    'settlement': sett.to_dict(),
                    'bank': best_match[1].to_dict() if best_match[0] == 'bank' else None,
                    'ledger': best_match[1].to_dict() if best_match[0] == 'ledger' else None,
                    'match_type': f"fuzzy_{best_match[0]}",
                    'confidence': round(best_score, 4),
                    'rule_fired': best_rule
                }
                
                # Ledger matches without a bank leg cannot be fully matched
                if best_score >= self.confidence_threshold and best_match[0] == 'bank':
                    match_record['final_status'] = 'matched'
                    matched_records.append(match_record)
                    decision = 'matched'
                    matched_bank_ids.add(best_match[1].name)
                    matched_settlement_ids.add(sett.name)
                else:
                    match_record['final_status'] = 'low_confidence'
                    low_confidence_records.append(match_record)
                    decision = 'low_confidence'
                
                audit_records.append({
                    'record_ids': f"{sett.get('entity_id', sett.get('payment_id', ''))}-{best_match[1].get('txn_id', best_match[1].get('order_id', ''))}",
                    'stage': 'stage2_fuzzy',
                    'rule_fired': best_rule,
                    'confidence': round(best_score, 4),
                    'decision': decision,
                    'match_type': f"fuzzy_{best_match[0]}",
                    'final_status': decision
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
        text2: str,
        fee1: Optional[float] = None,
        is_ledger: bool = False
    ) -> Tuple[float, str]:
        """
        Score a potential match pair with fee deduction awareness.
        
        Returns:
            Tuple of (confidence_score 0-1, rule_description)
        """
        if amount1 is None or amount2 is None:
            return 0.0, "missing_amount"
        
        # Amount similarity (40% weight)
        amount_diff_pct = abs(amount1 - amount2) / max(amount1, amount2, 1) * 100
        
        amount_score = 0.0
        fee_rule_triggered = False
        
        if amount_diff_pct <= self.amount_threshold_pct:
            amount_score = 1.0
        elif is_ledger and fee1 is not None and abs((amount1 + fee1) - amount2) / max(amount2, 1) * 100 <= 1.0:
            # Net settlement + explicit fee matches gross ledger amount exactly
            amount_score = 0.98
            fee_rule_triggered = True
        elif is_ledger and (1.5 <= amount_diff_pct <= 3.5):
            # Standard MDR fee deduction range (2% + 18% GST = 2.36%)
            amount_score = 0.92
            fee_rule_triggered = True
        elif amount_diff_pct <= 5.0:
            amount_score = max(0.0, 1.0 - (amount_diff_pct - self.amount_threshold_pct) / 5.0)
        else:
            amount_score = 0.0
        
        # Date proximity (30% weight)
        if date1 and date2:
            date_diff = abs((date1 - date2).days)
            if date_diff <= 1:
                date_score = 1.0
            elif date_diff <= self.date_window_days:
                date_score = max(0.0, 1.0 - (date_diff - 1) / self.date_window_days)
            else:
                date_score = 0.0
        else:
            date_score = 0.5  # Neutral if dates missing
        
        # Text similarity (30% weight)
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)
        if text1_norm and text2_norm:
            text_score = fuzz.token_ratio(text1_norm, text2_norm) / 100.0
        else:
            text_score = 0.5
        
        # Weighted total
        total_score = (amount_score * 0.4) + (date_score * 0.3) + (text_score * 0.3)
        
        # Determine which rule fired
        if fee_rule_triggered and date_score >= 0.8:
            rule = "fee_adjusted_settlement_match"
        elif amount_score >= 0.9 and date_score >= 0.9:
            rule = "high_confidence_amount_date"
        elif amount_score >= 0.8:
            rule = "amount_close_match"
        elif text_score >= (self.text_similarity_threshold / 100.0):
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
                sett = row.get('settlement', {}) or {}
                count = row.get('bank') or row.get('ledger') or {}
                sett_id = str(sett.get('entity_id') or sett.get('payment_id') or sett.get('settlement_id') or '')
                count_id = str(count.get('txn_id') or count.get('order_id') or count.get('payment_id') or count.get('customer_ref') or count.get('utr') or '')
                rec_id = f"{sett_id}-{count_id}" if sett_id and count_id else (sett_id or count_id)
                exceptions.append({
                    'type': 'low_confidence',
                    'record_ids': rec_id,
                    'settlement': sett,
                    'counterpart': count,
                    'confidence': row.get('confidence', 0),
                    'rule_fired': row.get('rule_fired', '')
                })
        
        # Unmatched settlements
        if not unmatched_settlements.empty:
            for _, row in unmatched_settlements.iterrows():
                r = row.to_dict()
                exceptions.append({
                    'type': 'unmatched_settlement',
                    'record_ids': str(r.get('entity_id') or r.get('payment_id') or r.get('settlement_id') or ''),
                    'settlement': r,
                    'counterpart': None,
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        # Unmatched bank transactions
        if not unmatched_bank.empty:
            for _, row in unmatched_bank.iterrows():
                r = row.to_dict()
                exceptions.append({
                    'type': 'unmatched_bank',
                    'record_ids': str(r.get('txn_id') or r.get('utr') or ''),
                    'settlement': None,
                    'counterpart': r,
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        # Unmatched ledger entries
        if not unmatched_ledger.empty:
            for _, row in unmatched_ledger.iterrows():
                r = row.to_dict()
                exceptions.append({
                    'type': 'unmatched_ledger',
                    'record_ids': str(r.get('order_id') or r.get('payment_id') or r.get('customer_ref') or ''),
                    'settlement': None,
                    'counterpart': r,
                    'confidence': 0.0,
                    'rule_fired': 'no_candidate_found'
                })
        
        return exceptions

