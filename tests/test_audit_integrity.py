"""
Unit tests for Cryptographic Audit Trail and SHA-256 Hash Chaining.
"""

import pytest
import sqlite3
from audit.store import AuditStore


@pytest.fixture
def audit_store(tmp_path):
    """Create a temporary AuditStore instance."""
    db_file = tmp_path / "test_audit.db"
    return AuditStore(db_path=str(db_file))


class TestAuditIntegrity:
    """Test cryptographic chaining and tamper detection."""
    
    def test_genesis_block_creation(self, audit_store):
        """Test first record links to genesis hash."""
        rec_id = audit_store.append({
            'record_ids': 'sett_001-TXN_001',
            'stage': 'stage1_exact',
            'decision': 'matched',
            'confidence': 1.0
        })
        
        entries = audit_store.get_all()
        assert len(entries) == 1
        assert entries[0]['previous_hash'] == "0" * 64
        assert entries[0]['record_hash'] is not None
        assert len(entries[0]['record_hash']) == 64
        
        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is True
        assert integrity['total_checked'] == 1
    
    def test_multi_record_hash_chaining(self, audit_store):
        """Test that multiple records form an unbroken cryptographic chain."""
        for i in range(5):
            audit_store.append({
                'record_ids': f'sett_{i:03d}-TXN_{i:03d}',
                'stage': 'stage1_exact',
                'decision': 'matched',
                'confidence': 1.0
            })
            
        entries = audit_store.get_all()
        assert len(entries) == 5
        
        # Verify chain pointers
        for i in range(1, 5):
            assert entries[i]['previous_hash'] == entries[i - 1]['record_hash']
            
        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is True
        assert len(integrity['tampered_ids']) == 0
    
    def test_tamper_detection_on_modified_record(self, audit_store):
        """Test that tampering directly in SQLite is caught by verify_integrity()."""
        for i in range(3):
            audit_store.append({
                'record_ids': f'sett_{i:03d}-TXN_{i:03d}',
                'stage': 'stage2_fuzzy',
                'decision': 'low_confidence',
                'confidence': 0.75
            })
            
        # Tamper with record #2 directly via SQL update
        with sqlite3.connect(audit_store.db_path) as conn:
            conn.execute("UPDATE audit_log SET decision = 'matched' WHERE id = 2")
            conn.commit()
            
        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is False
        assert 2 in integrity['tampered_ids']
