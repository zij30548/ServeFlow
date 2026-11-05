"""One durable send allowance for the frozen expanded evaluation questions.

The 108 questions yield 132 fixed jobs: one shared intent observation, one
enhanced retrieval job, or two task arms. The mathematical 1212-send phase
envelope is not an allowance. The actual batch ceiling is 480; a separately
authenticated, protocol-bound grant may be lower. This module never authorizes
or sends requests, resets old ledgers, refunds attempts or resumes unknown jobs.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting


TaskBudgetError = accounting.TaskBudgetError
ROOT = Path(__file__).resolve().parents[1] / "data/benchmarks/expanded-evaluation-v1"
MAX_REQUESTS = 480
MAX_IN_FLIGHT = accounting.MAX_IN_FLIGHT
TASK_PHASE_CAPS = {name: count for name, count in accounting.PHASE_CAPS.items() if name != "judge"}
INTENT_PHASE_CAPS = {"intent": 1}
RAG_PHASE_CAPS = {"rag_rewrite": 1, "rag_rerank": 1}
TASK_PER_TURN = sum(TASK_PHASE_CAPS.values())
INTENT_PER_TURN = sum(INTENT_PHASE_CAPS.values())
RAG_PER_TURN = sum(RAG_PHASE_CAPS.values())


def job_specs():
    """Closed authoring order, not a model input or a runtime routing choice."""
    rows = []
    for split in ("development", "validation"):
        for number in range(1, 61):
            development = (number - 1) % 5 < 2
            if development != (split == "development"):
                continue
            key = f"xi{number:03d}"
            rows.append({"case_id": "I-" + key, "source_case_id": key,
                         "suite": "intents", "split": split, "arm": "shared-observation"})
        for number in range(1 if split == "development" else 9,
                            9 if split == "development" else 25):
            key = f"et{number:02d}"
            for arm in (("A", "D") if number % 2 else ("D", "A")):
                rows.append({"case_id": arm + "-" + key, "source_case_id": key,
                             "suite": "tasks", "split": split, "arm": arm})
        for number in range(1 if split == "development" else 9,
                            9 if split == "development" else 25):
            key = f"er{number:02d}"
            rows.append({"case_id": "R-" + key, "source_case_id": key,
                         "suite": "retrieval", "split": split, "arm": "rewrite-rerank"})
    return rows


CASE_IDS = tuple(row["case_id"] for row in job_specs())
MATHEMATICAL_REQUESTS = 60 * INTENT_PER_TURN + 24 * RAG_PER_TURN + 48 * TASK_PER_TURN
LIMITS = {"questions": 108, "cases": 132, "turns": 132, "intent_turns": 60,
          "task_turns": 48, "retrieval_turns": 24, "physical_requests": MAX_REQUESTS,
          "mathematical_phase_envelope": MATHEMATICAL_REQUESTS,
          "intent_per_turn": INTENT_PER_TURN, "task_per_turn": TASK_PER_TURN,
          "retrieval_per_turn": RAG_PER_TURN, "in_flight": MAX_IN_FLIGHT,
          "intent_phase_caps": dict(INTENT_PHASE_CAPS),
          "task_phase_caps": dict(TASK_PHASE_CAPS), "retrieval_phase_caps": dict(RAG_PHASE_CAPS),
          "judge_allowed": False}


def fixed_plan():
    return [{"case_id": identity, "turn_ids": ["turn-1"]} for identity in CASE_IDS]


class ExpandedBudget(accounting.TaskBudget):
    """Exact mixed-suite identity using existing private locks, claims and leases."""

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
        self._intent_ids = frozenset(row["case_id"] for row in job_specs() if row["suite"] == "intents")
        self._retrieval_ids = frozenset(row["case_id"] for row in job_specs() if row["suite"] == "retrieval")
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        # Authenticated by the runner, then enforced under the same write lock.
        self.send_limit = send_limit
        self.operation_ids = tuple(row["case_id"] + ":turn-1" for row in plan)
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-expanded-budget:identity:v1\n"
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
        return hashlib.sha256(b"serveflow-expanded-budget:event:v1\n"
                              + accounting._canonical(unsigned)).hexdigest()

    def _caps(self, case_id):
        return (INTENT_PHASE_CAPS if case_id in self._intent_ids else
                RAG_PHASE_CAPS if case_id in self._retrieval_ids else TASK_PHASE_CAPS)

    def _metadata(self, case_id, turn_id, phase, agent):
        super()._metadata(case_id, turn_id, phase, agent)
        if (phase not in self._caps(case_id)
                or case_id in self._intent_ids | self._retrieval_ids and agent is not None):
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
