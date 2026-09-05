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
        assert data["llm_status"] in ("not_configured", "configured", "degraded", "unavailable", "healthy")
        assert data["llm_status"] != "connected"
    
    def test_reconcile_endpoint_no_llm(self, client):
        """Test POST /reconcile runs pipeline without LLM and returns metrics."""
        payload = {
            "records": 20,
            "seed": 42,
            "messiness": 0.2,
            "demo_disagreement": True,
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
        assert "Path traversal detected" in str(data["detail"]) or "Filename" in str(data["detail"]) or "Path must resolve" in str(data["detail"])
    
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


import os


class TestSecurityHardening:
    """Regression tests for P1 security fixes."""

    def test_near_miss_key_same_length_still_fails(self):
        """
        A key that differs from the real key by exactly one character (same length)
        must fail authentication — validates secrets.compare_digest is used, not ==.
        """
        expected = os.getenv("API_SECRET_KEY", "dev-secret-key")
        last_char = expected[-1]
        replacement = "X" if last_char != "X" else "Y"
        near_miss_key = expected[:-1] + replacement
        assert near_miss_key != expected  # sanity

        near_miss_client = TestClient(app, headers={"X-API-Key": near_miss_key})
        response = near_miss_client.get("/health")
        assert response.status_code == 403, (
            "A near-miss API key of the same length must be rejected (status 403)"
        )

    def test_empty_api_key_is_rejected(self):
        """An empty string API key must be rejected."""
        empty_client = TestClient(app, headers={"X-API-Key": ""})
        response = empty_client.get("/health")
        assert response.status_code in (401, 403)

    def test_production_startup_refuses_default_key(self, monkeypatch):
        """
        ENV=production + API_SECRET_KEY=dev-secret-key must raise RuntimeError at startup.
        """
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("API_SECRET_KEY", "dev-secret-key")
        from api.app import validate_security_configuration
        with pytest.raises(RuntimeError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
            validate_security_configuration()

    def test_production_startup_refuses_unset_key(self, monkeypatch):
        """
        ENV=production + no API_SECRET_KEY must raise RuntimeError at startup.
        """
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("API_SECRET_KEY", raising=False)
        from api.app import validate_security_configuration
        with pytest.raises(RuntimeError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
            validate_security_configuration()

    def test_development_mode_allows_default_key(self, monkeypatch):
        """
        ENV=development + dev-secret-key must NOT raise — only warn.
        """
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("DEMO_MODE", "true")
        monkeypatch.setenv("API_SECRET_KEY", "dev-secret-key")
        from api.app import validate_security_configuration
        validate_security_configuration()  # must not raise

    def test_compare_digest_is_used(self):
        """
        Verify via source inspection that get_api_key uses secrets.compare_digest,
        not a plain == / != comparison (timing-safe requirement).
        """
        import inspect
        from api.app import get_api_key
        source = inspect.getsource(get_api_key)
        assert "secrets.compare_digest" in source, (
            "get_api_key must use secrets.compare_digest, not plain == or !="
        )
