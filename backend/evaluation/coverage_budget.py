"""Independent 136-position coverage ledger; 480 physical sends, never a grant.

All storage, claims, chain, leases and uncertain-send rules reuse TaskBudget.
No historical identity, remaining balance, source freeze or journal is changed.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting


TaskBudgetError = accounting.TaskBudgetError
ROOT = Path(__file__).resolve().parents[1] / "data/benchmarks/coverage-execution-v1"
MAX_REQUESTS = 480
MAX_IN_FLIGHT = 3
TASK_PHASE_CAPS = {k: v for k, v in accounting.PHASE_CAPS.items() if k != "judge"}
RAG_PHASE_CAPS = {"rag_rewrite": 1, "rag_rerank": 1}
MATHEMATICAL_REQUESTS = 120 * sum(TASK_PHASE_CAPS.values()) + 48 * 2


def job_specs():
    rows = []
    for prefix, split, tasks, retrieval, conversations in (("D", "development", 12, 16, 2), ("V", "validation", 28, 32, 6)):
        for number in range(1, tasks + 1):
            identity = f"CT-{prefix}{number:02d}"
            for arm in (("A", "D") if number % 2 else ("D", "A")):
                rows.append({"case_id": arm + "-" + identity, "source_case_id": identity,
                    "suite": "tasks", "split": split, "arm": arm})
        for number in range(1, retrieval + 1):
            identity = f"CR-{prefix}{number:02d}"
            rows.append({"case_id": "R-" + identity, "source_case_id": identity,
                "suite": "retrieval", "split": split, "arm": "paired-vector-enhanced"})
        for number in range(1, conversations + 1):
            identity = f"CC-{prefix}{number:02d}"
            rows.append({"case_id": "M-" + identity, "source_case_id": identity,
                "suite": "conversations", "split": split, "arm": "current"})
    return rows


CASE_IDS = tuple(row["case_id"] for row in job_specs())
LIMITS = {"positions": 136, "user_turns": 168, "task_positions": 80, "retrieval_positions": 48,
    "conversation_positions": 8, "conversation_turns": 40, "physical_requests": MAX_REQUESTS,
    "maximum_in_flight": MAX_IN_FLIGHT, "mathematical_phase_envelope": MATHEMATICAL_REQUESTS,
    "task_phase_caps": TASK_PHASE_CAPS, "retrieval_phase_caps": RAG_PHASE_CAPS, "judge_allowed": False}


def fixed_plan():
    return [{"case_id": row["case_id"], "turn_ids": [f"turn-{i}" for i in range(1, 6)]
        if row["suite"] == "conversations" else ["turn-1"]} for row in job_specs()]


class CoverageBudget(accounting.TaskBudget):
    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False, send_limit=MAX_REQUESTS):
        if (type(initialize) is not bool or type(case_plan) is not list or case_plan != fixed_plan()
                or type(send_limit) is not int or not 1 <= send_limit <= MAX_REQUESTS):
            raise TaskBudgetError("configuration")
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents):
            raise TaskBudgetError("configuration")
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise TaskBudgetError("protocol")
        plan = deepcopy(case_plan)
        self.case_plan = tuple(plan)
        self._case_ids = tuple(row["case_id"] for row in plan)
        self._turns = {row["case_id"]: tuple(row["turn_ids"]) for row in plan}
        self._retrieval_ids = frozenset(row["case_id"] for row in job_specs() if row["suite"] == "retrieval")
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests, self.send_limit = MAX_REQUESTS, send_limit
        self.operation_ids = tuple(row["case_id"] + ":" + turn for row in plan for turn in row["turn_ids"])
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
            "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-coverage-budget:identity:v1\n" + accounting._canonical(self._identity)).hexdigest()
        self._lease = self._lease_fd = None
        self._initializing = initialize
        try:
            with self._locked(create=initialize) as (directory, _, journal, claims):
                self._replay(journal, claims)
                if initialize:
                    fd = os.open("run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
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
        return hashlib.sha256(b"serveflow-coverage-budget:event:v1\n" + accounting._canonical(unsigned)).hexdigest()

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
        caps, turn = self._caps(case_id), (case_id, turn_id)
        if (state["attempts"] >= self.send_limit or state["attempts"] >= MAX_REQUESTS
                or state["turn_attempts"].get(turn, 0) >= sum(caps.values())
                or state["phase_counts"].get((*turn, phase), 0) >= caps[phase]):
            raise TaskBudgetError("quota")
        active, planned = state["active_turn"].get(case_id), self._turns[case_id]
        if active is None and turn_id != planned[0] or active not in {None, turn_id} and (
                planned.index(turn_id) != planned.index(active) + 1 or state["pending"]):
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
                or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256", "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
            raise TaskBudgetError("protocol")
        state["active_case"] = identity
        state["starts"].append(identity)
