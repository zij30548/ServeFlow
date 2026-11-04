"""One durable allowance for the fixed current-version A/D task comparison.

Twenty-four single turns have a conservative 552-send phase envelope, but this
batch's actual immutable hard limit is 120. Neither limit grants authorization.
Old trial classes, stores and remaining allowances are never opened here. The
established TaskBudget owns private files, locks, claims, replay and run leases;
this closed adapter only narrows admission and binds its distinct fixed plan.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting


TaskBudgetError = accounting.TaskBudgetError
MAX_REQUESTS = 120
PHASE_CAPS = {name: count for name, count in accounting.PHASE_CAPS.items() if name != "judge"}
PER_TURN = sum(PHASE_CAPS.values())
MAX_IN_FLIGHT = accounting.MAX_IN_FLIGHT
CASE_IDS = tuple(arm + f"-p{number:02d}"
                 for number in range(1, 13)
                 for arm in (("A", "D") if number % 2 else ("D", "A")))
MATHEMATICAL_REQUESTS = len(CASE_IDS) * PER_TURN
LIMITS = {"cases": 24, "turns": 24, "physical_requests": MAX_REQUESTS,
          "mathematical_phase_envelope": MATHEMATICAL_REQUESTS,
          "per_turn": PER_TURN, "in_flight": MAX_IN_FLIGHT,
          "phase_caps": dict(PHASE_CAPS), "judge_allowed": False}


def fixed_plan():
    """Only these 24 ordered A/D jobs may spend, each exactly one user turn."""
    return [{"case_id": identity, "turn_ids": ["turn-1"]} for identity in CASE_IDS]


class TaskAblationBudget(accounting.TaskBudget):
    """Root/protocol-bound 120-send ledger with no recovery or refund path."""

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
        # The runner separately authenticates this lower protocol-bound grant.
        # reserve/replay enforce it under the same lock as the global 120 cap.
        self.send_limit = send_limit
        self.operation_ids = tuple(row["case_id"] + ":turn-1" for row in plan)
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-current-task-ablation-budget:identity:v1\n"
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
        return hashlib.sha256(b"serveflow-current-task-ablation-budget:event:v1\n"
                              + accounting._canonical(unsigned)).hexdigest()

    def _metadata(self, case_id, turn_id, phase, agent):
        super()._metadata(case_id, turn_id, phase, agent)
        if phase not in PHASE_CAPS:
            raise TaskBudgetError("protocol")

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        # Keep the existing role/phase/order/three-in-flight checks, then apply
        # the strictly narrower batch grant and no-Judge 23-send turn budget.
        super()._reserve_check(state, case_id, turn_id, phase, agent)
        turn = (case_id, turn_id)
        if (state["attempts"] >= self.send_limit or state["attempts"] >= MAX_REQUESTS
                or state["turn_attempts"].get(turn, 0) >= PER_TURN
                or state["phase_counts"].get((*turn, phase), 0) >= PHASE_CAPS[phase]):
            raise TaskBudgetError("quota")

    def _apply(self, state, event):
        if event["event"] != "case_start":
            return super()._apply(state, event)
        identity = event["case_id"]
        if (type(identity) is not str or identity not in self._turns or type(event["attempt"]) is not int
                or state["active_case"] is not None or state["paused"] or state["pending"]
                or len(state["completed_cases"]) >= len(CASE_IDS)
                or identity != self._case_ids[len(state["completed_cases"])] or identity in state["starts"]
                or event["operation_id"] != "case:" + identity or event["attempt"] != state["attempts"]
                or any(event[key] is not None for key in ("turn_id", "phase", "agent", "ordinal", "payload_sha256",
                    "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
            raise TaskBudgetError("protocol")
        state["active_case"] = identity
        state["starts"].append(identity)
