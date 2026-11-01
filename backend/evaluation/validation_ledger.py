"""Independent, fail-closed ledger for a fixed 36-operation validation batch.

This is storage preparation, not authorization to send requests. The caller must
bind the canonical batch directory, frozen protocol and a separate authorization
chain before sending. Development ledgers and their allowances are not reusable.
All locking, claims, journal replay and result checks use the unchanged trial
ledger implementation. Nothing here makes network calls.
"""
import hashlib
from pathlib import Path

from evaluation.trial_ledger import (
    VERSION, TrialLedger, TrialLedgerError, _IDENTITY_LIMIT, _canonical, _hash,
)


class ValidationLedger(TrialLedger):
    """The trial ledger's persistence contract for exactly 36 fixed operations.

    The supplied operation order is frozen in an independent identity. Existing
    development identities are refused rather than upgraded or reset. Only the
    constructor differs; reservation and recovery remain fail-closed.
    """

    def __init__(self, root: Path, protocol_sha256: str,
                 operation_ids: list[str], max_requests: int = 36):
        if (type(max_requests) is not int or max_requests != 36
                or not isinstance(operation_ids, list)
                or len(operation_ids) != 36
                or not all(isinstance(op, str) and op for op in operation_ids)
                or len(set(operation_ids)) != len(operation_ids)):
            raise TrialLedgerError("configuration")
        try:
            self.root = Path(root)
        except (TypeError, ValueError):
            raise TrialLedgerError("configuration") from None
        self.protocol_sha256 = _hash(protocol_sha256)
        try:
            for operation_id in operation_ids:
                operation_id.encode("utf-8")
        except UnicodeError:
            raise TrialLedgerError("configuration") from None
        self.operation_ids = tuple(operation_ids)
        self.max_requests = max_requests
        self._identity = {
            "version": VERSION, "protocol_sha256": self.protocol_sha256,
            "operation_ids": list(self.operation_ids), "max_requests": max_requests,
        }
        if len(_canonical(self._identity)) > _IDENTITY_LIMIT:
            raise TrialLedgerError("configuration")
        self._genesis = hashlib.sha256(
            b"serveflow-trial-ledger:identity:v1\n" + _canonical(self._identity)
        ).hexdigest()
        with self._locked(create=True) as (_directory, _identity, journal, claims):
            self._replay(journal, claims)
