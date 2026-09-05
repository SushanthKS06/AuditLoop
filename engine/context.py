"""
Canonical 3-way reconciliation context.

The verifier must never inspect only one financial leg. Callers pass a
ReconciliationContext so missing bank or ledger cannot be overlooked.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _record_present(record: Optional[Dict[str, Any]]) -> bool:
    """True when a counterpart dict exists and is not an empty placeholder."""
    if not record or not isinstance(record, dict):
        return False
    return any(v is not None and v != "" for v in record.values())


@dataclass
class ReconciliationContext:
    """Settlement + bank + ledger + opaque metadata for one evaluation unit."""

    settlement: Optional[Dict[str, Any]]
    bank: Optional[Dict[str, Any]] = None
    ledger: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def settlement_present(self) -> bool:
        return _record_present(self.settlement)

    def bank_present(self) -> bool:
        return _record_present(self.bank)

    def ledger_present(self) -> bool:
        return _record_present(self.ledger)

    def all_required_legs_present(self) -> bool:
        """3-way invariant: settlement, bank, and ledger must all be present."""
        return (
            self.settlement_present()
            and self.bank_present()
            and self.ledger_present()
        )

    def missing_legs(self) -> list:
        missing = []
        if not self.settlement_present():
            missing.append("settlement")
        if not self.bank_present():
            missing.append("bank")
        if not self.ledger_present():
            missing.append("ledger")
        return missing

    @classmethod
    def from_exception(cls, exception: Dict[str, Any]) -> "ReconciliationContext":
        """Build context from a Stage-3 exception record."""
        settlement = exception.get("settlement")
        bank = exception.get("bank")
        ledger = exception.get("ledger")
        counterpart = exception.get("counterpart")

        if not _record_present(bank) and _record_present(counterpart):
            if counterpart.get("txn_id") or counterpart.get("utr") or (
                "amount" in counterpart and "expected_amount" not in counterpart
            ):
                bank = counterpart
            elif counterpart.get("order_id") or "expected_amount" in counterpart:
                ledger = counterpart

        return cls(
            settlement=settlement if isinstance(settlement, dict) else None,
            bank=bank if isinstance(bank, dict) else None,
            ledger=ledger if isinstance(ledger, dict) else None,
            metadata=exception,
        )
