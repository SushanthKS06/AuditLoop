"""
Audit Trail Models and Storage

Append-only SQLite audit log. Every decision, matched or not, leaves a record.
Never mutate rows - only append new records.
"""

import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class AuditStore:
    """
    Append-only audit trail storage in SQLite.
    
    Design decision: Append-only ensures we can reconstruct the full
    history of how each decision was made, which is critical for:
    1. Explaining decisions to auditors
    2. Debugging false positives/negatives
    3. Meeting fintech compliance requirements
    """
    
    def __init__(self, db_path: str = "audit_trail.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize the database schema."""
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
                    final_status TEXT
                )
            """)
            
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
    
    def append(self, record: Dict[str, Any]) -> int:
        """
        Append a new audit record.
        
        Args:
            record: Dictionary with audit fields
            
        Returns:
            ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO audit_log (
                    timestamp, record_ids, stage, rule_fired, 
                    confidence, decision, match_type, llm_reasoning, final_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                record.get('record_ids', ''),
                record.get('stage', 'unknown'),
                record.get('rule_fired'),
                record.get('confidence'),
                record.get('decision', 'unknown'),
                record.get('match_type'),
                record.get('llm_reasoning'),
                record.get('final_status')
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_all(self) -> List[Dict]:
        """Get all audit records."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audit_log ORDER BY timestamp")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_current_status(self) -> List[Dict]:
        """Get current status view (latest decision per record)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM current_status")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_by_status(self, status: str) -> List[Dict]:
        """Get records with a specific final status."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE final_status = ? ORDER BY timestamp",
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
                WHERE decision IN ('unresolved_exception', 'flagged_for_review', 'low_confidence')
                ORDER BY timestamp
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard."""
        with self._get_connection() as conn:
            # Total records
            cursor = conn.execute("SELECT COUNT(*) as total FROM audit_log")
            total = cursor.fetchone()['total']
            
            # Matched count
            cursor = conn.execute("""
                SELECT COUNT(*) as matched FROM audit_log 
                WHERE decision = 'matched'
            """)
            matched = cursor.fetchone()['matched']
            
            # Exceptions count
            cursor = conn.execute("""
                SELECT COUNT(*) as exceptions FROM audit_log 
                WHERE decision IN ('unresolved_exception', 'flagged_for_review', 'low_confidence')
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
            
            return {
                'total_records': total,
                'matched': matched,
                'exceptions': exceptions,
                'disagreements': disagreements,
                'match_rate': matched / total if total > 0 else 0,
                'average_confidence': avg_conf
            }
    
    def clear(self):
        """Clear all audit records (for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM audit_log")
            conn.commit()
