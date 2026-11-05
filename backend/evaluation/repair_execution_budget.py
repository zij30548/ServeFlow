"""Durable admission for the exact sixteen-position repair verification.

This is a separate ledger identity, not a new allowance. The runner must first
authenticate a protocol-bound grant and pass its limit (at most 96). Preparation
does not authorize sends; old ledgers, unused balances and unknown attempts are
never reset, repaired, refunded or resumed by this module.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting


TaskBudgetError = accounting.TaskBudgetError
ROOT = Path(__file__).resolve().parents[1] / "data/benchmarks/repair-execution-v1"
MAX_REQUESTS = 96
MAX_IN_FLIGHT = accounting.MAX_IN_FLIGHT
TASK_PHASE_CAPS = {name: count for name, count in accounting.PHASE_CAPS.items() if name != "judge"}
RAG_PHASE_CAPS = {"rag_rewrite": 1, "rag_rerank": 1}
TASK_PER_TURN = sum(TASK_PHASE_CAPS.values())
RAG_PER_TURN = sum(RAG_PHASE_CAPS.values())
CASE_IDS = tuple(f"{prefix}{number:02d}" for prefix in ("RT", "RR") for number in range(1, 9))
MATHEMATICAL_REQUESTS = 8 * TASK_PER_TURN + 8 * RAG_PER_TURN
LIMITS = {"cases": 16, "turns": 16, "task_turns": 8, "retrieval_turns": 8,
          "physical_requests": MAX_REQUESTS, "mathematical_phase_envelope": MATHEMATICAL_REQUESTS,
          "task_per_turn": TASK_PER_TURN, "retrieval_per_turn": RAG_PER_TURN,
          "in_flight": MAX_IN_FLIGHT, "task_phase_caps": dict(TASK_PHASE_CAPS),
          "retrieval_phase_caps": dict(RAG_PHASE_CAPS), "judge_allowed": False}


def job_specs():
    """The source/group metadata stays outside model input and quota authority."""
    return [{"case_id": identity, "suite": "tasks" if identity.startswith("RT") else "retrieval",
             "group": "known_regression" if int(identity[2:]) <= 4 else "new_wording"}
            for identity in CASE_IDS]


def fixed_plan():
    return [{"case_id": identity, "turn_ids": ["turn-1"]} for identity in CASE_IDS]


class RepairExecutionBudget(accounting.TaskBudget):
    """Use existing private locks/claims/replay with a new closed identity.

    ``send_limit`` is an authenticated runner input, not a user parameter. The
    durable ledger's upper bound stays 96; no restart lowers the spent counter.
    The runner must use the same protocol-bound grant on every reopening.
    """

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False, send_limit=MAX_REQUESTS):
        if (type(initialize) is not bool or type(case_plan) is not list or case_plan != fixed_plan()
                or any(type(row) is not dict or type(row["case_id"]) is not str
                       or type(row["turn_ids"]) is not list for row in case_plan)
                or type(send_limit) is not int or not 1 <= send_limit <= MAX_REQUESTS):
            raise TaskBudgetError("configuration")
        try:
            self.root = Path(root).absolute()
            if self.root.resolve() != self.root or any(path.is_symlink() for path in self.root.parents):
                raise TaskBudgetError("configuration")
        except (TypeError, ValueError, OSError):
            raise TaskBudgetError("configuration") from None
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise TaskBudgetError("protocol")
        plan = deepcopy(case_plan)
        self.case_plan = tuple(plan)
        self._case_ids = tuple(row["case_id"] for row in plan)
        self._turns = {row["case_id"]: tuple(row["turn_ids"]) for row in plan}
        self._retrieval_ids = frozenset(identity for identity in CASE_IDS if identity.startswith("RR"))
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests, self.send_limit = MAX_REQUESTS, send_limit
        self.operation_ids = tuple(identity + ":turn-1" for identity in self._case_ids)
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-repair-execution-budget:identity:v1\n"
                                       + accounting._canonical(self._identity)).hexdigest()
        self._lease = self._lease_fd = None
        self._initializing = initialize
        try:
            with self._locked(create=initialize) as (directory, _identity, journal, claims):
                self._replay(journal, claims)
                if initialize:
                    descriptor = os.open("run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                         0o600, dir_fd=directory)
                    try:
                        accounting._private_file(descriptor)
                        accounting._write(descriptor, self._run_identity())
                    finally:
                        os.close(descriptor)
                    os.fsync(directory)
        finally:
            self._initializing = False

    @staticmethod
    def _event_hash(unsigned):
        return hashlib.sha256(b"serveflow-repair-execution-budget:event:v1\n"
                              + accounting._canonical(unsigned)).hexdigest()

    def _caps(self, case_id):
        return RAG_PHASE_CAPS if case_id in self._retrieval_ids else TASK_PHASE_CAPS

    def _metadata(self, case_id, turn_id, phase, agent):
        super()._metadata(case_id, turn_id, phase, agent)
        if phase not in self._caps(case_id) or case_id in self._retrieval_ids and agent is not None:
            raise TaskBudgetError("protocol")

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        self._metadata(case_id, turn_id, phase, agent)
        if state["paused"]:
            raise TaskBudgetError("paused")
        if state["active_case"] != case_id:
            raise TaskBudgetError("protocol")
        if len(state["pending"]) >= MAX_IN_FLIGHT:
            raise TaskBudgetError("ledger_busy")
        caps = self._caps(case_id)
        turn = (case_id, turn_id)
        if (state["attempts"] >= self.send_limit or state["attempts"] >= MAX_REQUESTS
                or state["turn_attempts"].get(turn, 0) >= sum(caps.values())
                or state["phase_counts"].get((*turn, phase), 0) >= caps[phase]):
            raise TaskBudgetError("quota")
        if state["active_turn"].get(case_id) not in {None, turn_id}:
            raise TaskBudgetError("protocol")
        if case_id in self._retrieval_ids and phase == "rag_rerank" and (
                state["phase_counts"].get((*turn, "rag_rewrite"), 0) != 1 or state["pending"]):
            raise TaskBudgetError("protocol")

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
