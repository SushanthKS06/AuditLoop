"""
Concurrency tests for the AuditStore hash chain.

Verifies that concurrent writers cannot fork the SHA-256 hash chain
even when multiple threads append records simultaneously.

These tests must pass with zero failures to prove the BEGIN IMMEDIATE +
threading.Lock fix in audit/store.py is effective.
"""

import threading
import pytest
from audit.store import AuditStore


@pytest.fixture
def audit_store(tmp_path):
    """Fresh AuditStore backed by a temporary SQLite file."""
    db_file = tmp_path / "concurrent_test_audit.db"
    return AuditStore(db_path=str(db_file))


def _append_n(store: AuditStore, n: int, prefix: str):
    """Append n records to store using a fixed prefix for record_ids."""
    for i in range(n):
        store.append({
            'record_ids': f'{prefix}_sett_{i:04d}-TXN_{i:04d}',
            'stage': 'stage1_exact',
            'decision': 'matched',
            'confidence': 1.0,
            'source': 'synthetic',
        })


class TestAuditStoreConcurrency:
    """Verify BEGIN IMMEDIATE + Lock prevent hash-chain forks under load."""

    def test_1_writer_chain_unbroken(self, audit_store):
        _append_n(audit_store, 25, 'S')
        assert len(audit_store.get_all()) == 25
        assert audit_store.verify_integrity()['integrity_verified'] is True

    def test_100_concurrent_writers_chain_unbroken(self, audit_store):
        """100 threads writing simultaneously must produce an unbroken chain."""
        n_threads = 100
        records_per_thread = 2
        threads = [
            threading.Thread(target=_append_n, args=(audit_store, records_per_thread, f'T{t}'))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entries = audit_store.get_all()
        assert len(entries) == n_threads * records_per_thread, (
            f"Expected {n_threads * records_per_thread} records, "
            f"got {len(entries)}. Some writes were lost."
        )
        assert audit_store.verify_integrity()['integrity_verified'] is True

        # Chain must be unbroken: each previous_hash == preceding record_hash
        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is True, (
            f"Chain integrity failed after concurrent writes. "
            f"Tampered IDs: {integrity.get('tampered_ids', [])}"
        )
        assert len(integrity.get('tampered_ids', [])) == 0

    def test_50_concurrent_writers_chain_unbroken(self, audit_store):
        """50 concurrent threads — higher contention stress test."""
        n_threads = 50
        records_per_thread = 5
        threads = [
            threading.Thread(target=_append_n, args=(audit_store, records_per_thread, f'T{t}'))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = audit_store.get_all()
        assert len(entries) == n_threads * records_per_thread

        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is True, (
            "Chain integrity failed under 50-thread concurrent load."
        )

    def test_no_duplicate_ids_after_concurrent_writes(self, audit_store):
        """Each inserted record must have a unique database id."""
        n_threads = 20
        records_per_thread = 10
        threads = [
            threading.Thread(target=_append_n, args=(audit_store, records_per_thread, f'T{t}'))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = audit_store.get_all()
        ids = [e['id'] for e in entries]
        assert len(ids) == len(set(ids)), "Duplicate database IDs found — INSERT serialisation broken."

    def test_genesis_hash_appears_exactly_once(self, audit_store):
        """The genesis hash (64 zeros) must only appear as the previous_hash of the FIRST record."""
        _append_n(audit_store, 20, 'GEN')
        entries = audit_store.get_all()

        genesis_hash = '0' * 64
        genesis_links = [e for e in entries if e['previous_hash'] == genesis_hash]
        assert len(genesis_links) == 1, (
            f"Genesis hash appeared {len(genesis_links)} times as previous_hash. "
            "Expected exactly 1 (the first record only)."
        )
        assert genesis_links[0]['id'] == entries[0]['id']

    def test_sequential_appends_form_perfect_chain(self, audit_store):
        """Sequential (single-threaded) appends must form a perfect chain (regression)."""
        for i in range(15):
            audit_store.append({
                'record_ids': f'SEQ_sett_{i:03d}',
                'stage': 'stage2_fuzzy',
                'decision': 'matched',
                'confidence': 0.92,
            })

        entries = audit_store.get_all()
        assert len(entries) == 15

        for i in range(1, 15):
            assert entries[i]['previous_hash'] == entries[i - 1]['record_hash'], (
                f"Chain broken at index {i}: "
                f"previous_hash={entries[i]['previous_hash']} "
                f"!= record_hash[{i-1}]={entries[i-1]['record_hash']}"
            )

        integrity = audit_store.verify_integrity()
        assert integrity['integrity_verified'] is True

    def test_current_status_view_stable_after_concurrent_writes(self, audit_store, tmp_path):
        """
        current_status VIEW uses MAX(id) not MAX(timestamp) — so rapid same-timestamp
        writes must still resolve to the most recently inserted record.
        """
        import sqlite3

        record_ids_key = 'SHARED_REC_001'

        # Write 10 records for the same record_ids in rapid succession
        for i in range(10):
            audit_store.append({
                'record_ids': record_ids_key,
                'stage': f'stage_{i}',
                'decision': 'matched' if i < 9 else 'low_confidence',
                'confidence': float(i) / 10,
            })

        # The view should return exactly 1 row for this record_ids
        with sqlite3.connect(audit_store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM current_status WHERE record_ids = ?",
                (record_ids_key,)
            ).fetchall()

        assert len(rows) == 1, (
            f"current_status VIEW returned {len(rows)} rows for a single record_ids key. "
            "Expected exactly 1 (the latest decision)."
        )
        # The view must return the LAST decision (id = 10), which is 'low_confidence'
        assert rows[0]['decision'] == 'low_confidence', (
            f"current_status returned decision='{rows[0]['decision']}' "
            "instead of the latest 'low_confidence'. MAX(id) ordering is broken."
        )
