"""One fixed physical-send ledger for the prospective task/RAG comparison.

This is a mathematical ceiling, not authorization or a production quota. The
runner must separately bind any smaller allowance to its frozen protocol. All
task stages share the same counter with standalone retrieval; no Judge stage is
allowed, and local vector-only retrieval creates no physical reservation.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import prospective_validation as design, task_budget as accounting


TASK_PHASE_CAPS = {name: count for name, count in accounting.PHASE_CAPS.items() if name != "judge"}
RAG_PHASE_CAPS = {"rag_rewrite": 1, "rag_rerank": 1}
TASK_PER_TURN = sum(TASK_PHASE_CAPS.values())
RAG_PER_TURN = sum(RAG_PHASE_CAPS.values())
MAX_IN_FLIGHT = accounting.MAX_IN_FLIGHT
MAX_REQUESTS = 24 * TASK_PER_TURN + 12 * RAG_PER_TURN
LIMITS = {"cases": 36, "turns": 36, "task_turns": 24, "retrieval_turns": 12,
          "physical_requests": MAX_REQUESTS, "task_per_turn": TASK_PER_TURN,
          "retrieval_per_turn": RAG_PER_TURN, "in_flight": MAX_IN_FLIGHT,
          "task_phase_caps": dict(TASK_PHASE_CAPS), "retrieval_phase_caps": dict(RAG_PHASE_CAPS),
          "judge_allowed": False}


def fixed_plan():
    """Use the validated design order; arm labels never become model inputs."""
    design.validate()
    rows = design._plan(design.bench.BACKEND_ROOT)
    tasks = [{"case_id": arm + "-" + row["case_id"], "turn_ids": ["turn-1"]}
             for row in rows if row["suite"] == "tasks" for arm in row["arm_order"]]
    retrieval = [{"case_id": "R-" + row["case_id"], "turn_ids": ["turn-1"]}
                 for row in rows if row["suite"] == "retrieval"]
    return tasks + retrieval


class ProspectiveBudget(accounting.TaskBudget):
    """Exact 36-job identity with the existing claims, leases and replay rules."""

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False, send_limit=MAX_REQUESTS):
        if (type(initialize) is not bool or type(case_plan) is not list or case_plan != fixed_plan()
                or type(send_limit) is not int or not 1 <= send_limit <= MAX_REQUESTS):
            raise accounting.TaskBudgetError("configuration")
        try:
            self.root = Path(root).absolute()
            if self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents):
                raise accounting.TaskBudgetError("configuration")
        except (TypeError, ValueError, OSError):
            raise accounting.TaskBudgetError("configuration") from None
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise accounting.TaskBudgetError("protocol")
        plan = deepcopy(case_plan)
        self.case_plan = tuple(plan)
        self._case_ids = tuple(item["case_id"] for item in plan)
        self._turns = {item["case_id"]: tuple(item["turn_ids"]) for item in plan}
        self._retrieval_ids = frozenset(item["case_id"] for item in plan[24:])
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        # The runner supplies the protocol-bound authorization limit. It does
        # not change ledger identity, reset occupied claims or itself grant
        # permission. Default-ceiling instances may inspect the same history.
        self.send_limit = send_limit
        self.operation_ids = tuple(item["case_id"] + ":turn-1" for item in plan)
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-prospective-budget:identity:v1\n"
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
        return hashlib.sha256(b"serveflow-prospective-budget:event:v1\n"
                              + accounting._canonical(unsigned)).hexdigest()

    def _metadata(self, case_id, turn_id, phase, agent):
        super()._metadata(case_id, turn_id, phase, agent)
        caps = RAG_PHASE_CAPS if case_id in self._retrieval_ids else TASK_PHASE_CAPS
        if phase not in caps or case_id in self._retrieval_ids and agent is not None:
            raise accounting.TaskBudgetError("protocol")

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        self._metadata(case_id, turn_id, phase, agent)
        if state["paused"]:
            raise accounting.TaskBudgetError("paused")
        if state["active_case"] != case_id:
            raise accounting.TaskBudgetError("protocol")
        if len(state["pending"]) >= MAX_IN_FLIGHT:
            raise accounting.TaskBudgetError("ledger_busy")
        retrieval = case_id in self._retrieval_ids
        caps, per_turn = (RAG_PHASE_CAPS, RAG_PER_TURN) if retrieval else (TASK_PHASE_CAPS, TASK_PER_TURN)
        turn = (case_id, turn_id)
        if (state["attempts"] >= self.send_limit or state["attempts"] >= MAX_REQUESTS
                or state["turn_attempts"].get(turn, 0) >= per_turn
                or state["phase_counts"].get((*turn, phase), 0) >= caps[phase]):
            raise accounting.TaskBudgetError("quota")
        if state["active_turn"].get(case_id) not in {None, turn_id}:
            raise accounting.TaskBudgetError("protocol")
        if retrieval and phase == "rag_rerank" and (
                state["phase_counts"].get((*turn, "rag_rewrite"), 0) != 1 or state["pending"]):
            raise accounting.TaskBudgetError("protocol")

    def _apply(self, state, event):
        if event["event"] != "case_start":
            return super()._apply(state, event)
        key = event["case_id"]
        if (type(key) is not str or key not in self._turns or type(event["attempt"]) is not int
                or state["active_case"] is not None or state["paused"] or state["pending"]
                or len(state["completed_cases"]) >= 36
                or key != self._case_ids[len(state["completed_cases"])] or key in state["starts"]
                or event["operation_id"] != "case:" + key or event["attempt"] != state["attempts"]
                or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256",
                    "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
            raise accounting.TaskBudgetError("protocol")
        state["active_case"] = key
        state["starts"].append(key)
