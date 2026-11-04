"""Fixed six-turn accounting for a separately authorized diagnostic recheck.

The 138-send bound is a conservative mathematical envelope, not authorization,
an expected usage or a production quota. The runner must separately bind its
canonical root, protocol and any smaller allowance before opening live clients.
Nothing here reads an old trial's remaining allowance or contacts a service.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting


TaskBudgetError = accounting.TaskBudgetError
CASE_IDS = ("D-p07", "D-n01", "D-p01", "D-n02", "D-p09", "D-n03")
PHASE_CAPS = {name: count for name, count in accounting.PHASE_CAPS.items() if name != "judge"}
PER_TURN = sum(PHASE_CAPS.values())
MAX_IN_FLIGHT = accounting.MAX_IN_FLIGHT
MAX_REQUESTS = len(CASE_IDS) * PER_TURN
LIMITS = {"cases": 6, "turns": 6, "physical_requests": MAX_REQUESTS,
          "per_turn": PER_TURN, "in_flight": MAX_IN_FLIGHT,
          "phase_caps": dict(PHASE_CAPS), "judge_allowed": False}


def fixed_plan():
    """Only these ordered jobs may spend; each has exactly one user turn."""
    return [{"case_id": name, "turn_ids": ["turn-1"]} for name in CASE_IDS]


class DiagnosticRepairBudget(accounting.TaskBudget):
    """New six-case/six-turn identity over unchanged durable accounting.

    Inherit the private files, claims, replay, exclusive lease and six-case
    progression. The old constructor remains six cases/seven turns/168 sends;
    this closed constructor and stricter admission never reinterpret that ledger.
    """

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False, send_limit=MAX_REQUESTS):
        if (type(initialize) is not bool or type(case_plan) is not list or case_plan != fixed_plan()
                or any(type(row) is not dict or type(row["case_id"]) is not str
                       or type(row["turn_ids"]) is not list for row in case_plan)
                or type(send_limit) is not int or not 1 <= send_limit <= MAX_REQUESTS):
            raise TaskBudgetError("configuration")
        try:
            self.root = Path(root).absolute()
            if self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents):
                raise TaskBudgetError("configuration")
        except (TypeError, ValueError, OSError):
            raise TaskBudgetError("configuration") from None
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise TaskBudgetError("protocol")
        plan = deepcopy(case_plan)
        self.case_plan = tuple(plan)
        self._case_ids = tuple(row["case_id"] for row in plan)
        self._turns = {row["case_id"]: tuple(row["turn_ids"]) for row in plan}
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        # Authorization is checked by the runner; this bound is also enforced
        # in the atomic reserve/replay path, not merely a pre-send callback.
        # Its value cannot create a new identity or refund an occupied claim.
        self.send_limit = send_limit
        self.operation_ids = tuple(row["case_id"] + ":turn-1" for row in plan)
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-diagnostic-repair-budget:identity:v1\n"
                                       + accounting._canonical(self._identity)).hexdigest()
        self._lease = self._lease_fd = None
        self._initializing = initialize
        try:
            with self._locked(create=initialize) as (directory, _identity, journal, claims):
                self._replay(journal, claims)
                if initialize:
                    fd = os.open("run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
                    try:
                        accounting._private_file(fd)
                        accounting._write(fd, self._run_identity())
                    finally:
                        os.close(fd)
                    os.fsync(directory)
        finally:
            self._initializing = False

    @staticmethod
    def _event_hash(unsigned):
        return hashlib.sha256(b"serveflow-diagnostic-repair-budget:event:v1\n"
                              + accounting._canonical(unsigned)).hexdigest()

    def _metadata(self, case_id, turn_id, phase, agent):
        super()._metadata(case_id, turn_id, phase, agent)
        if phase not in PHASE_CAPS:
            raise TaskBudgetError("protocol")

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        # Parent enforces the unchanged phase/role/turn/lease progression and
        # at most three pending requests. New constraints can only narrow it.
        super()._reserve_check(state, case_id, turn_id, phase, agent)
        turn = (case_id, turn_id)
        if (state["attempts"] >= self.send_limit or state["attempts"] >= MAX_REQUESTS
                or state["turn_attempts"].get(turn, 0) >= PER_TURN
                or state["phase_counts"].get((*turn, phase), 0) >= PHASE_CAPS[phase]):
            raise TaskBudgetError("quota")
