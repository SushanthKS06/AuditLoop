"""
Human-in-the-Loop (HITL) Maker-Checker Test Suite

Tests:
- Manual exception resolution appending to SQLite audit log
- Cryptographic hash chaining integrity before and after human resolution
- Full chronological record lifecycle history
- REST API /audit/resolve endpoint contract
"""

import pytest
import os
from audit.store import AuditStore
from fastapi.testclient import TestClient
from api.app import app


class TestHumanInTheLoopWorkflow:
    """Test Maker-Checker human resolution and cryptographic integrity."""
    
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        db_file = str(tmp_path / "test_hitl_audit.db")
        self.store = AuditStore(db_path=db_file)
        self.store.clear()
        yield
        self.store.clear()
        
    def test_human_resolution_preserves_hash_chain(self):
        # 1. Append automatic stage 1 and stage 3 records
        id1 = self.store.append({
            'record_ids': 'sett_101-TXN_202',
            'stage': 'stage2_fuzzy',
            'rule_fired': 'weak_candidate',
            'confidence': 0.72,
            'decision': 'low_confidence',
            'final_status': 'low_confidence'
        })
        
        id2 = self.store.append({
            'record_ids': 'sett_101-TXN_202',
            'stage': 'stage3_llm',
            'rule_fired': 'timing_lag',
            'confidence': 0.85,
            'decision': 'llm_deterministic_disagreement',
            'final_status': 'llm_deterministic_disagreement'
        })
        
        # Verify initial chain
        int1 = self.store.verify_integrity()
        assert int1['integrity_verified'] is True
        assert int1['total_checked'] == 2
        
        # 2. Human Controller resolves the exception
        res = self.store.resolve_exception(
            record_ids='sett_101-TXN_202',
            reviewer_id='CFO_JANE_DOE',
            decision='human_approved_match',
            notes='Verified bank value date lag with merchant relationship manager.'
        )
        
        assert res['status'] == 'success'
        assert res['decision'] == 'human_approved_match'
        assert res['integrity_verified'] is True
        assert res['record_hash'] != ""
        
        # 3. Verify complete chain with 3 chained blocks
        int2 = self.store.verify_integrity()
        assert int2['integrity_verified'] is True
        assert int2['total_checked'] == 3
        
        # 4. Check chronological record history
        history = self.store.get_record_history('sett_101-TXN_202')
        assert len(history) == 3
        assert history[0]['stage'] == 'stage2_fuzzy'
        assert history[1]['stage'] == 'stage3_llm'
        assert history[2]['stage'] == 'stage4_human_resolution'
        assert history[2]['decision'] == 'human_approved_match'


class TestHumanResolutionApi:
    """Test REST API endpoint for human resolution."""
    
    def setup_method(self):
        self.client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})
        
    def test_api_resolve_endpoint(self):
        payload = {
            "record_ids": "sett_api_test-TXN_999",
            "decision": "human_approved_match",
            "reviewer_id": "AUDITOR_ALICE",
            "notes": "Authorized after viewing signed physical invoice."
        }
        response = self.client.post("/audit/resolve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["record_ids"] == payload["record_ids"]
        assert data["reviewer_id"] == payload["reviewer_id"]
        assert data["integrity_verified"] is True
        assert len(data["record_hash"]) == 64  # Valid SHA-256 hex string
