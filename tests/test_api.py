"""
Integration tests for AuditLoop FastAPI endpoints.

Tests /health, /reconcile, /metrics, /audit/recent, and /audit/disagreements endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app, headers={"X-API-Key": "dev-secret-key"})


class TestApiEndpoints:
    """Test suite for AuditLoop REST API."""
    
    def test_unauthorized_access(self):
        """Test that missing or invalid API key returns 401 or 403."""
        unauth_client = TestClient(app)
        response = unauth_client.get("/health")
        assert response.status_code == 401
        
        invalid_client = TestClient(app, headers={"X-API-Key": "wrong-key"})
        response = invalid_client.get("/health")
        assert response.status_code == 403

    def test_health_endpoint(self, client):
        """Test GET /health returns 200 with healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_reconcile_endpoint_no_llm(self, client):
        """Test POST /reconcile runs pipeline without LLM and returns metrics."""
        payload = {
            "records": 20,
            "seed": 42,
            "messiness": 0.2,
            "force_disagreement": True,
            "use_llm": False
        }
        response = client.post("/reconcile", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "metrics" in data
        assert "audit_summary" in data
        assert data["audit_log_count"] > 0
        
    def test_reconcile_path_traversal_rejection(self, client):
        """Test that POST /reconcile rejects path traversal attempts."""
        payload = {
            "settlements_path": "../../secrets/passwords.txt"
        }
        response = client.post("/reconcile", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "Path traversal detected" in data["detail"] or "Filename" in data["detail"] or "Path must resolve" in data["detail"]
    
    def test_metrics_endpoint(self, client):
        """Test GET /metrics returns the computed metrics report."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "match_rate" in data
        assert "precision" in data
        assert "recall" in data
    
    def test_audit_recent_endpoint(self, client):
        """Test GET /audit/recent returns audit entries."""
        response = client.get("/audit/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "record_ids" in data[0]
            assert "stage" in data[0]
            assert "decision" in data[0]
    
    def test_audit_disagreements_endpoint(self, client):
        """Test GET /audit/disagreements returns list."""
        response = client.get("/audit/disagreements")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_audit_summary_endpoint(self, client):
        """Test GET /audit/summary returns summary stats."""
        response = client.get("/audit/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_records" in data
        assert "matched" in data
        assert "integrity_verified" in data
    
    def test_audit_verify_endpoint(self, client):
        """Test GET /audit/verify returns cryptographic audit chain verification."""
        response = client.get("/audit/verify")
        assert response.status_code == 200
        data = response.json()
        assert "integrity_verified" in data
        assert "total_checked" in data
        assert "tampered_ids" in data

