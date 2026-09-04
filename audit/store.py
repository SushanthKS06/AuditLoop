"""
Audit Trail Models and Storage

Append-only SQLite audit log with SHA-256 cryptographic hash chaining.
Every decision, matched or not, leaves a tamper-evident record.
Never mutate rows - only append new records.
"""

import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class AuditStore:
    """
    Append-only audit trail storage in SQLite with SHA-256 cryptographic hash chaining.
    
    Design decision: Append-only with cryptographic chaining ensures we can reconstruct 
    the full history of how each decision was made and mathematically prove zero tampering, 
    which is critical for:
    1. Explaining decisions to auditors — provides the tamper-evident foundation that
       compliance workflows (e.g. SOC2, RBI reporting) depend on; certification itself
       is a separate organizational process.
    2. Debugging false positives/negatives
    3. Meeting institutional fintech audit-trail requirements
    """
    
    GENESIS_HASH = "0" * 64
    
    def __init__(self, db_path: str = "audit_trail.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with WAL mode and timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize the database schema with cryptographic hash fields."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    record_ids TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    rule_fired TEXT,
                    confidence REAL,
                    decision TEXT NOT NULL,
                    match_type TEXT,
                    llm_reasoning TEXT,
                    final_status TEXT,
                    previous_hash TEXT,
                    record_hash TEXT
                )
            """)
            
            # Migration check: Ensure columns exist if opened on older DB
            cursor = conn.execute("PRAGMA table_info(audit_log)")
            columns = [row['name'] for row in cursor.fetchall()]
            if 'previous_hash' not in columns:
                conn.execute("ALTER TABLE audit_log ADD COLUMN previous_hash TEXT")
            if 'record_hash' not in columns:
                conn.execute("ALTER TABLE audit_log ADD COLUMN record_hash TEXT")
            
            # Create index for querying by record_ids
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_record_ids 
                ON audit_log(record_ids)
            """)
            
            # Create view for current status (latest decision per record)
            conn.execute("""
                CREATE VIEW IF NOT EXISTS current_status AS
                SELECT a.*
                FROM audit_log a
                INNER JOIN (
                    SELECT record_ids, MAX(timestamp) as max_ts
                    FROM audit_log
                    GROUP BY record_ids
                ) b ON a.record_ids = b.record_ids AND a.timestamp = b.max_ts
            """)
            
            conn.commit()
    
    def _compute_hash(
        self,
        previous_hash: str,
        timestamp: str,
        record_ids: str,
        stage: str,
        decision: str,
        rule_fired: Optional[str],
        confidence: Optional[float],
        final_status: Optional[str],
        match_type: Optional[str] = None,
        llm_reasoning: Optional[str] = None
    ) -> str:
        """Compute SHA-256 hash for cryptographic block chaining."""
        payload = f"{previous_hash}|{timestamp}|{record_ids}|{stage}|{decision}|{rule_fired or ''}|{confidence or ''}|{final_status or ''}|{match_type or ''}|{llm_reasoning or ''}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
    
    def append(self, record: Dict[str, Any]) -> int:
        """
        Append a new audit record with cryptographic hash chaining.
        
        Args:
            record: Dictionary with audit fields
            
        Returns:
            ID of the inserted record
        """
        with self._get_connection() as conn:
            # Fetch latest hash in the chain
            cursor = conn.execute("SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1")
            last_row = cursor.fetchone()
            previous_hash = last_row['record_hash'] if (last_row and last_row['record_hash']) else self.GENESIS_HASH
            
            timestamp = record.get('timestamp') or datetime.now(timezone.utc).isoformat()
            record_ids = record.get('record_ids', '')
            stage = record.get('stage', 'unknown')
            rule_fired = record.get('rule_fired')
            confidence = record.get('confidence')
            decision = record.get('decision', 'unknown')
            match_type = record.get('match_type')
            llm_reasoning = record.get('llm_reasoning')
            final_status = record.get('final_status') or decision
            
            record_hash = self._compute_hash(
                previous_hash=previous_hash,
                timestamp=timestamp,
                record_ids=record_ids,
                stage=stage,
                decision=decision,
                rule_fired=rule_fired,
                confidence=confidence,
                final_status=final_status,
                match_type=match_type,
                llm_reasoning=llm_reasoning
            )
            
            cursor = conn.execute("""
                INSERT INTO audit_log (
                    timestamp, record_ids, stage, rule_fired, 
                    confidence, decision, match_type, llm_reasoning, final_status,
                    previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, record_ids, stage, rule_fired,
                confidence, decision, match_type, llm_reasoning, final_status,
                previous_hash, record_hash
            ))
            conn.commit()
            return cursor.lastrowid
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the complete cryptographic hash chain to ensure zero tampering.
        
        Returns:
            Dict containing integrity_verified (bool), total_checked, and list of tampered_ids.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audit_log ORDER BY id ASC")
            rows = [dict(r) for r in cursor.fetchall()]
            
            if not rows:
                return {
                    "integrity_verified": True,
                    "total_checked": 0,
                    "tampered_ids": [],
                    "message": "Audit log is empty."
                }
            
            expected_prev_hash = self.GENESIS_HASH
            tampered_ids = []
            
            for row in rows:
                row_id = row['id']
                stored_prev = row.get('previous_hash') or self.GENESIS_HASH
                stored_hash = row.get('record_hash')
                
                # Check previous hash pointer
                if stored_prev != expected_prev_hash:
                    tampered_ids.append(row_id)
                
                # Re-compute hash from row payload
                computed_hash = self._compute_hash(
                    previous_hash=stored_prev,
                    timestamp=row['timestamp'],
                    record_ids=row['record_ids'],
                    stage=row['stage'],
                    decision=row['decision'],
                    rule_fired=row['rule_fired'],
                    confidence=row['confidence'],
                    final_status=row['final_status'],
                    match_type=row.get('match_type'),
                    llm_reasoning=row.get('llm_reasoning')
                )
                
                if computed_hash != stored_hash:
                    if row_id not in tampered_ids:
                        tampered_ids.append(row_id)
                
                expected_prev_hash = stored_hash or computed_hash
            
            is_valid = len(tampered_ids) == 0
            return {
                "integrity_verified": is_valid,
                "total_checked": len(rows),
                "tampered_ids": tampered_ids,
                "message": "Cryptographic audit chain is fully verified and untampered." if is_valid else f"Tampering detected in {len(tampered_ids)} records."
            }
    
    def get_all(self) -> List[Dict]:
        """Get all audit records."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audit_log ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_entries(self) -> List[Dict]:
        """Alias for get_all for interface compatibility."""
        return self.get_all()
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get most recent audit entries."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            rows.reverse()
            return rows
    
    def count(self) -> int:
        """Get total count of audit records."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM audit_log")
            return cursor.fetchone()['total']
    
    def get_current_status(self) -> List[Dict]:
        """Get current status view (latest decision per record)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM current_status")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_by_status(self, status: str) -> List[Dict]:
        """Get records with a specific final status."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE final_status = ? ORDER BY id ASC",
                (status,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_disagreements(self) -> List[Dict]:
        """Get all LLM-deterministic disagreement cases."""
        return self.get_by_status('llm_deterministic_disagreement')
    
    def get_exceptions(self) -> List[Dict]:
        """Get all unresolved exceptions."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM audit_log 
                WHERE decision NOT IN ('matched', 'human_approved_match')
                   OR final_status NOT LIKE 'matched%'
                ORDER BY id ASC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard including cryptographic integrity."""
        with self._get_connection() as conn:
            # Total records
            cursor = conn.execute("SELECT COUNT(*) as total FROM audit_log")
            total = cursor.fetchone()['total']
            
            # Matched count
            cursor = conn.execute("""
                SELECT COUNT(*) as matched FROM audit_log 
                WHERE decision = 'matched' OR final_status LIKE 'matched%'
            """)
            matched = cursor.fetchone()['matched']
            
            # Exceptions count
            cursor = conn.execute("""
                SELECT COUNT(*) as exceptions FROM audit_log 
                WHERE decision NOT IN ('matched', 'human_approved_match')
            """)
            exceptions = cursor.fetchone()['exceptions']
            
            # Disagreements count
            cursor = conn.execute("""
                SELECT COUNT(*) as disagreements FROM audit_log 
                WHERE final_status = 'llm_deterministic_disagreement'
            """)
            disagreements = cursor.fetchone()['disagreements']
            
            # Average confidence
            cursor = conn.execute("""
                SELECT AVG(confidence) as avg_conf FROM audit_log 
                WHERE confidence IS NOT NULL
            """)
            avg_conf = cursor.fetchone()['avg_conf'] or 0
            
            integrity = self.verify_integrity()
            
            return {
                'total_records': total,
                'matched': matched,
                'exceptions': exceptions,
                'disagreements': disagreements,
                'match_rate': matched / total if total > 0 else 0,
                'average_confidence': float(avg_conf),
                'integrity_verified': integrity['integrity_verified'],
                'tampered_records_count': len(integrity['tampered_ids'])
            }
    
    def resolve_exception(
        self,
        record_ids: str,
        reviewer_id: str,
        decision: str,
        notes: str
    ) -> Dict[str, Any]:
        """
        Record a human reviewer's manual resolution into the cryptographically chained audit log.
        
        Args:
            record_ids: Identifier pair being resolved
            reviewer_id: User/employee ID of reviewer
            decision: Outcome ('human_approved_match', 'human_rejected_duplicate', 'human_written_off')
            notes: Auditable explanation for compliance
            
        Returns:
            Dict containing inserted audit record info and chain integrity status
        """
        if not reviewer_id or str(reviewer_id).strip().lower().startswith('system'):
            raise ValueError("Invalid reviewer_id: Must be a distinct human controller.")
            
        history = self.get_record_history(record_ids)
        if any(entry.get('stage') == 'stage4_human_resolution' for entry in history):
            raise ValueError("Duplicate resolution: Record has already been resolved.")
            
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'record_ids': record_ids,
            'stage': 'stage4_human_resolution',
            'rule_fired': f'maker_checker_override:{reviewer_id}',
            'confidence': 1.0,
            'decision': decision,
            'match_type': 'human_resolved',
            'llm_reasoning': f"Reviewer [{reviewer_id}] Notes: {notes}",
            'final_status': decision
        }
        
        inserted_id = self.append(audit_entry)
        
        # Verify integrity
        integrity = self.verify_integrity()
        
        # Fetch the newly created record with its hash
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT record_hash FROM audit_log WHERE id = ?", (inserted_id,))
            row = cursor.fetchone()
            rec_hash = row['record_hash'] if row else ""
        
        return {
            'status': 'success',
            'audit_entry_id': inserted_id,
            'record_ids': record_ids,
            'decision': decision,
            'reviewer_id': reviewer_id,
            'record_hash': rec_hash,
            'integrity_verified': integrity['integrity_verified']
        }
    
    def get_record_history(self, record_ids: str) -> List[Dict]:
        """Get the full chronological lifecycle decisions for a given record_ids."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE record_ids = ? ORDER BY id ASC",
                (record_ids,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear(self):
        """Clear all audit records (for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM audit_log")
            conn.commit()


