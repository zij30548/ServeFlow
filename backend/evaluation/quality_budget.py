"""One durable physical-send ledger for the exact four-arm factorial plan.

The historical ledgers and their limits are unchanged. This subclass reuses
their private files, hash chain, per-attempt claims and exclusive run lease;
only this distinct, fixed 48-case / 52-turn identity admits up to 1248 sends.
No unfinished operation is replayed or refunded.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from evaluation import task_budget as accounting
from evaluation.quality_ablation import load_manifest


MAX_REQUESTS = 1248


def fixed_plan():
    manifest = load_manifest()
    cases = {case["case_id"]: case for case in manifest["cases"]}
    return [{"case_id": arm + "-" + row["case_id"],
             "turn_ids": [f"turn-{n}" for n in range(1, len(cases[row["case_id"]]["turns"]) + 1)]}
            for row in manifest["order"] for arm in row["arm_order"]]


LIMITS = {"cases": 48, "turns": 52, "physical_requests": MAX_REQUESTS,
          "per_turn": accounting.PER_TURN, "in_flight": accounting.MAX_IN_FLIGHT,
          "phase_caps": dict(accounting.PHASE_CAPS)}


class QualityBudget(accounting.TaskBudget):
    """A single global, protocol/root-bound budget, never one budget per arm."""

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False):
        if type(initialize) is not bool or type(case_plan) is not list or case_plan != fixed_plan():
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
        self._case_ids = tuple(p["case_id"] for p in plan)
        self._turns = {p["case_id"]: tuple(p["turn_ids"]) for p in plan}
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        self.operation_ids = tuple(f'{p["case_id"]}:{t}' for p in plan for t in p["turn_ids"])
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-quality-budget:identity:v1\n" + accounting._canonical(self._identity)).hexdigest()
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
        return hashlib.sha256(b"serveflow-quality-budget:event:v1\n" + accounting._canonical(unsigned)).hexdigest()

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        self._metadata(case_id, turn_id, phase, agent)
        if state["paused"]:
            raise accounting.TaskBudgetError("paused")
        if state["active_case"] != case_id:
            raise accounting.TaskBudgetError("protocol")
        if len(state["pending"]) >= accounting.MAX_IN_FLIGHT:
            raise accounting.TaskBudgetError("ledger_busy")
        turn = (case_id, turn_id)
        if (state["attempts"] >= MAX_REQUESTS or state["turn_attempts"].get(turn, 0) >= accounting.PER_TURN
                or state["phase_counts"].get((*turn, phase), 0) >= accounting.PHASE_CAPS[phase]):
            raise accounting.TaskBudgetError("quota")
        active, planned = state["active_turn"].get(case_id), self._turns[case_id]
        if (active is None and turn_id != planned[0] or active is not None and active != turn_id
                and (planned.index(turn_id) != planned.index(active) + 1 or state["pending"])):
            raise accounting.TaskBudgetError("protocol")

    def _apply(self, state, event):
        if event["event"] != "case_start":
            return super()._apply(state, event)
        key = event["case_id"]
        if (type(key) is not str or key not in self._turns or type(event["attempt"]) is not int
                or state["active_case"] is not None or state["paused"] or state["pending"]
                or len(state["completed_cases"]) >= 48
                or key != self._case_ids[len(state["completed_cases"])] or key in state["starts"]
                or event["operation_id"] != "case:" + key or event["attempt"] != state["attempts"]
                or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256",
                    "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
            raise accounting.TaskBudgetError("protocol")
        state["active_case"] = key
        state["starts"].append(key)
