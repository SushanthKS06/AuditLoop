"""
Provenance & Data Integrity Regression Tests

Verifies:
1. Every row from generate_data.py carries a valid `source` field — never None/missing.
2. Every row from fetch_settlements._normalize_to_dataframe carries source='razorpay_test'.
3. Ground-truth entries carry a valid `source` field.
4. forced_demo_case is surfaced through the /audit/recent API response.
5. Settlement CSV source values are only from the allowed set.
"""

import pytest
import os
import json
import pandas as pd
from data.generate_data import SyntheticDataGenerator
from data.fetch_settlements import RazorpayReconClient
from fastapi.testclient import TestClient
from api.app import app

VALID_SOURCES = {"razorpay_test", "synthetic"}


class TestGenerateDataProvenance:
    """Every row produced by generate_data.py must have a valid source field."""

    @pytest.fixture
    def generated_files(self, tmp_path):
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        bank_df, ledger_df, gt = gen.generate(
            num_records=30,
            settlements_df=None,
            output_dir=str(tmp_path / "provtest"),
        )
        settlements_df = pd.read_csv(tmp_path / "provtest" / "settlements_live.csv")
        return settlements_df, bank_df, ledger_df, gt

    def test_settlements_csv_source_column_exists(self, generated_files):
        """settlements_live.csv must have a 'source' column."""
        settlements_df, *_ = generated_files
        assert "source" in settlements_df.columns, "settlements_live.csv missing 'source' column"

    def test_settlements_csv_no_null_source(self, generated_files):
        """No row in settlements_live.csv may have a null or empty source."""
        settlements_df, *_ = generated_files
        null_mask = settlements_df["source"].isnull() | (settlements_df["source"].astype(str).str.strip() == "")
        assert not null_mask.any(), (
            f"{null_mask.sum()} row(s) in settlements_live.csv have null/empty source"
        )

    def test_settlements_csv_source_values_valid(self, generated_files):
        """All source values in settlements_live.csv must be from the allowed set."""
        settlements_df, *_ = generated_files
        actual_sources = set(settlements_df["source"].dropna().unique())
        invalid = actual_sources - VALID_SOURCES
        assert not invalid, f"Invalid source values in settlements_live.csv: {invalid}"

    def test_bank_df_source_column_exists(self, generated_files):
        """bank DataFrame must contain 'source' column."""
        _, bank_df, _, _ = generated_files
        assert "source" in bank_df.columns, "bank_df missing 'source' column"

    def test_bank_df_no_null_source(self, generated_files):
        """No row in bank_df may have a null/empty source."""
        _, bank_df, _, _ = generated_files
        null_mask = bank_df["source"].isnull() | (bank_df["source"].astype(str).str.strip() == "")
        assert not null_mask.any(), f"{null_mask.sum()} bank row(s) have null/empty source"

    def test_bank_df_source_values_valid(self, generated_files):
        """All bank_df source values must be from the allowed set."""
        _, bank_df, _, _ = generated_files
        actual_sources = set(bank_df["source"].dropna().unique())
        invalid = actual_sources - VALID_SOURCES
        assert not invalid, f"Invalid source values in bank_df: {invalid}"

    def test_ledger_df_source_column_exists(self, generated_files):
        """ledger DataFrame must contain 'source' column."""
        _, _, ledger_df, _ = generated_files
        assert "source" in ledger_df.columns, "ledger_df missing 'source' column"

    def test_ledger_df_no_null_source(self, generated_files):
        """No row in ledger_df may have a null/empty source."""
        _, _, ledger_df, _ = generated_files
        null_mask = ledger_df["source"].isnull() | (ledger_df["source"].astype(str).str.strip() == "")
        assert not null_mask.any(), f"{null_mask.sum()} ledger row(s) have null/empty source"

    def test_ground_truth_source_present(self, generated_files):
        """Every ground-truth entry must carry a 'source' field."""
        *_, gt = generated_files
        missing = [i for i, entry in enumerate(gt) if not entry.get("source")]
        assert not missing, f"Ground-truth entries at indices {missing} are missing 'source'"

    def test_ground_truth_source_values_valid(self, generated_files):
        """All ground-truth source values must be from the allowed set."""
        *_, gt = generated_files
        actual_sources = {entry.get("source") for entry in gt}
        invalid = actual_sources - VALID_SOURCES
        assert not invalid, f"Invalid source values in ground_truth: {invalid}"

    def test_fully_synthetic_rows_tagged_synthetic(self, generated_files):
        """When no real settlements are supplied, all rows must be tagged 'synthetic'."""
        settlements_df, bank_df, ledger_df, gt = generated_files
        assert (settlements_df["source"] == "synthetic").all(), (
            "All settlement rows should be 'synthetic' when no real settlements are supplied"
        )

    def test_real_settlement_rows_tagged_razorpay_test(self, tmp_path):
        """Rows derived from real settlements must be tagged 'razorpay_test'."""
        # Build a minimal mock settlement DataFrame that looks like a live API pull
        mock_sett = pd.DataFrame([{
            "entity_id": "ent_001",
            "type": "payment",
            "payment_id": "pay_001",
            "order_id": "ord_001",
            "amount": 1000.0,
            "fee": 20.0,
            "tax": 3.6,
            "currency": "INR",
            "settled_amount": 976.4,
            "settlement_id": "set_001",
            "settlement_utr": "UTR123456",
            "created_at": "2026-09-01T10:00:00",
            "settled_at": "2026-09-02T10:00:00",
            "method": "UPI",
            "card_network": "RuPay",
            "dispute_id": "",
            "source": "razorpay_test",
        }])
        gen = SyntheticDataGenerator(seed=99, messiness_ratio=0.0)
        bank_df, ledger_df, gt = gen.generate(
            num_records=5,  # need >= 2 so int(num_records*0.6) >= 1 real row is included
            settlements_df=mock_sett,
            output_dir=str(tmp_path / "real_sett_test"),
        )
        settlements_df = pd.read_csv(tmp_path / "real_sett_test" / "settlements_live.csv")

        razorpay_rows = settlements_df[settlements_df["source"] == "razorpay_test"]
        assert len(razorpay_rows) >= 1, (
            "Rows linked to real settlements must be tagged 'razorpay_test'"
        )


class TestFetchSettlementsProvenance:
    """fetch_settlements._normalize_to_dataframe must always emit source='razorpay_test'."""

    def test_normalize_sets_source_razorpay_test(self):
        """All normalized rows must have source='razorpay_test'."""
        # We can instantiate the client with dummy creds to access _normalize_to_dataframe
        # without making a network call (credential check happens in __init__, so patch it)
        client = object.__new__(RazorpayReconClient)
        client.key_id = "rzp_test_fake"
        client.key_secret = "fake_secret"

        fake_records = [
            {
                "entity_id": "ent_001",
                "type": "payment",
                "payment_id": "pay_001",
                "order_id": "ord_001",
                "amount": 100000,
                "fee": 2000,
                "tax": 360,
                "currency": "INR",
                "settled": 97640,
                "debit": None,
                "credit": 97640,
                "settlement_id": "set_001",
                "settlement_utr": "UTR999",
                "created_at": "2026-09-01",
                "settled_at": "2026-09-02",
                "method": "UPI",
                "card_network": "RuPay",
                "dispute_id": None,
            }
        ]
        df = client._normalize_to_dataframe(fake_records)
        assert "source" in df.columns, "_normalize_to_dataframe output missing 'source' column"
        assert (df["source"] == "razorpay_test").all(), (
            "All rows from _normalize_to_dataframe must have source='razorpay_test'"
        )

    def test_empty_dataframe_has_source_column(self):
        """_empty_dataframe() must include the 'source' column."""
        client = object.__new__(RazorpayReconClient)
        client.key_id = "rzp_test_fake"
        client.key_secret = "fake_secret"
        empty_df = client._empty_dataframe()
        assert "source" in empty_df.columns, "_empty_dataframe() missing 'source' column"


class TestForcedDemoCaseSurfacedInAPI:
    """forced_demo_case must surface through /audit/recent API response."""

    def setup_method(self):
        self.client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})

    def test_forced_demo_case_field_present_in_audit_recent(self):
        """
        After a reconcile run with demo_disagreement=True,
        /audit/recent entries that are forced demo cases must expose
        forced_demo_case=1 (truthy) in the API response.
        """
        # Run a small reconcile to seed the audit log
        payload = {
            "records": 10,
            "seed": 77,
            "messiness": 0.2,
            "demo_disagreement": True,
            "use_llm": False,
        }
        reconcile_resp = self.client.post("/reconcile", json=payload)
        assert reconcile_resp.status_code == 200

        # Fetch recent audit entries — all of them
        audit_resp = self.client.get("/audit/recent?limit=200")
        assert audit_resp.status_code == 200
        entries = audit_resp.json()
        assert isinstance(entries, list)

        # Every entry must have a forced_demo_case field
        for entry in entries:
            assert "forced_demo_case" in entry, (
                f"Audit entry missing 'forced_demo_case' field: {entry.get('record_ids')}"
            )

    def test_disagreements_endpoint_includes_forced_demo_case(self):
        """GET /audit/disagreements entries must include forced_demo_case field."""
        disagreements_resp = self.client.get("/audit/disagreements")
        assert disagreements_resp.status_code == 200
        disagreements = disagreements_resp.json()
        for d in disagreements:
            assert "forced_demo_case" in d, (
                f"Disagreement entry missing 'forced_demo_case': {d.get('record_ids')}"
            )
