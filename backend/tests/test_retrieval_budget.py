"""Dedicated retrieval accounting with real temporary files; no network."""
import socket

import pytest

from evaluation.retrieval_budget import RetrievalBudget, OPERATIONS, MAX_REQUESTS
from evaluation.trial_ledger import TrialLedger, TrialLedgerError


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("budget tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)


def test_existing_ledger_methods_and_original_limit_are_unchanged(tmp_path):
    for name in ("reserve", "finish", "status", "results", "_replay", "_append"):
        assert getattr(RetrievalBudget, name) is getattr(TrialLedger, name)
    with pytest.raises(TrialLedgerError, match="configuration"):
        TrialLedger(tmp_path / "not-created", "a" * 64, list(OPERATIONS), 48)
    assert not (tmp_path / "not-created").exists()


def test_initialize_is_explicit_and_existing_identity_is_not_reset(tmp_path):
    root = tmp_path / "ledger"
    with pytest.raises(TrialLedgerError):
        RetrievalBudget(root, "a" * 64)
    assert not root.exists()
    ledger = RetrievalBudget(root, "a" * 64, initialize=True)
    assert ledger.max_requests == 48 and ledger.operation_ids == OPERATIONS
    assert ledger.status()["attempts"] == 0 and len(OPERATIONS) == MAX_REQUESTS
    with pytest.raises(TrialLedgerError):
        RetrievalBudget(root, "a" * 64, initialize=True)
    assert RetrievalBudget(root, "a" * 64).status() == ledger.status()
    (root / "journal.jsonl").unlink()
    with pytest.raises(TrialLedgerError):
        RetrievalBudget(root, "a" * 64)
    assert not (root / "journal.jsonl").exists()


def test_fixed_two_stage_order_all_forty_eight_and_duplicate_block(tmp_path):
    ledger = RetrievalBudget(tmp_path / "ledger", "a" * 64, initialize=True)
    with pytest.raises(TrialLedgerError):
        ledger.reserve(OPERATIONS[1], "b" * 64)
    for op in OPERATIONS:
        claim = ledger.reserve(op, "b" * 64)
        assert claim["operation_id"] == op
        ledger.finish(op, "received", "c" * 64)
    assert ledger.status()["attempts"] == 48 and ledger.status()["completed"] == list(OPERATIONS)
    with pytest.raises(TrialLedgerError, match="quota"):
        ledger.reserve(OPERATIONS[0], "b" * 64)


def test_unknown_send_keeps_occupation_after_reopen(tmp_path):
    root = tmp_path / "ledger"
    ledger = RetrievalBudget(root, "a" * 64, initialize=True)
    ledger.reserve(OPERATIONS[0], "b" * 64)
    reopened = RetrievalBudget(root, "a" * 64)
    assert reopened.status()["attempts"] == 1 and reopened.status()["pending"] == [OPERATIONS[0]]
    with pytest.raises(TrialLedgerError, match="pending"):
        reopened.reserve(OPERATIONS[0], "b" * 64)
    with pytest.raises(TrialLedgerError):
        RetrievalBudget(root, "d" * 64)


def test_sent_failure_pauses_and_is_not_refunded(tmp_path):
    ledger = RetrievalBudget(tmp_path / "ledger", "a" * 64, initialize=True)
    ledger.reserve(OPERATIONS[0], "b" * 64)
    ledger.finish(OPERATIONS[0], "execution_error", "c" * 64, "timeout")
    assert ledger.status()["attempts"] == 1 and ledger.status()["paused"]
    with pytest.raises(TrialLedgerError, match="paused"):
        ledger.reserve(OPERATIONS[1], "b" * 64)
