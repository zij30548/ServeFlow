"""Dedicated 24-query / two-stage physical ledger; old limits stay unchanged.

Only the constructor differs from TrialLedger. Its claims, strict operation
order, persistence and unknown-send handling are retained without modification.
This identity is preparation, not permission to send model requests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.trial_ledger import TrialLedger, TrialLedgerError, _canonical, _hash


CASE_IDS = tuple(f"r{number:02d}" for number in range(1, 25))
STAGES = ("rewrite", "rerank")
OPERATIONS = tuple(f"{case_id}:{stage}:1" for case_id in CASE_IDS for stage in STAGES)
MAX_REQUESTS = 48


class RetrievalBudget(TrialLedger):
    """An explicit new 48-operation identity, never an upgrade of old ledgers."""

    def __init__(self, root, protocol_sha256, *, initialize=False):
        if type(initialize) is not bool:
            raise TrialLedgerError("configuration")
        self.root = Path(root).absolute()
        if (self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents)
                or initialize and (self.root.exists() or self.root.is_symlink())):
            raise TrialLedgerError("configuration")
        self.protocol_sha256 = _hash(protocol_sha256)
        self.operation_ids = OPERATIONS
        self.max_requests = MAX_REQUESTS
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
            "operation_ids": list(OPERATIONS), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-trial-ledger:identity:v1\n" + _canonical(self._identity)).hexdigest()
        with self._locked(create=initialize) as (_directory, _identity, journal, claims):
            self._replay(journal, claims)
