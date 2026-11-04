"""Independent 24-query RAG comparison with phase-bound, audited sends.

The old zero-send preparation and vector baseline are immutable references.
Only this fixed worker opts into audit-management clocks; business retrieval,
prompts, chunking, local embeddings and parsing fallbacks remain unchanged.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import difflib
import hashlib
import json
import subprocess

from core.model_deadline import (EvaluationDeadlinePolicy, eval_deadlines, overhead,
    assert_model_active, current_deadline_failure_reason, ModelDeadlineExceeded)
from evaluation import benchmark as bench, intent_trial as base, retrieval_trial as original
from evaluation.intent_transport import GuardedIntentTransport, TrialTransportError, canonical_payload_hash
from evaluation.phase_deadline_transport import SDK_TIMEOUT_S, _sdk_timeout, _uncancelled
from evaluation.task_recheck import _tree


ROOT = bench.BACKEND_ROOT / "data/benchmarks/retrieval-rewrite-rerank-v2-phase"
BATCH = "serveflow-retrieval-rewrite-rerank-v2-phase"
OLD_ROOT = original.ROOT
OLD = {"commit": "c4b4f1192f39a3992d3ee4433dcc328cb79b77c9",
    "protocol": "c5c56242d748b7b2580a09aa59ecd9a143489cd03639bef4764f23903bd73dfe",
    "tree": "e8b385450efd9ddfbea83e4abd087cd6fe3042ab15f09b4cb15c0c8784fa3670",
    "journal": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "chain": "7deba095fbdec30a32318ef3f838e0a04c5507c132b652274ee3febf4e42544e"}
BASELINE_TREE = "1ee3bdf65f64afe3059d5909dfba2a9c76f2470e775c5c95fea2ce91c20b31b8"
TOOL_MANAGER_PATCH = "a87ea90d91dac5f58106d55cbb1cc2d830a26edc9f2deb2594c4d33028cd447a"
PHASE_SECONDS = {"rewrite": 10, "rerank": 30}
LIMITS = {**deepcopy(original.LIMITS), "request_deadline_s": 30,
          "stage_network_deadlines_s": deepcopy(PHASE_SECONDS)}
POLICY = {**deepcopy(original.POLICY),
    "version": "retrieval-phase-bound-clock-v2", "sdk_timeout_s": 30,
    "active_stage_seconds": deepcopy(PHASE_SECONDS), "network_stage_seconds": deepcopy(PHASE_SECONDS),
    "audit_management_scopes": ["admission", "seal"], "audit_cumulative_seconds_per_call": 30,
    "wall_seconds": "unchanged active phase + 30s management + .25s cleanup grace; never paused",
    "outer_process_deadline_s": 3600,
    "dynamic_payload": "persist full validated SDK JSON and exact UTF-8 wire before reserve, with canonical/wire hashes; no headers",
    "comparison": "new 24-query fixed regression versus sealed local vector baseline; old zero-send preparation is not an execution result",
    "intermediate": "same merged candidate pool before/after rerank is a side observation, not independently executed component ablation",
    "limits_meaning": "configured SDK/network/active ceilings are not measured upstream processing times"}
_SAME_SOURCES = ("tooling/knowledge_base.py", "evaluation/task_data.py", "evaluation/retrieval_budget.py",
    "evaluation/retrieval_baseline.py", "evaluation/benchmark.py", "evaluation/benchmark/retrieval-v1.json", "evaluation/benchmark_metrics.py",
    "knowledge/serveflow-demo-v2.json", "core/llm_utils.py", "core/provider_options.py", "core/safe_diagnostics.py",
    "requirements.txt", "requirements-dev.txt", "demo_business/store.py", "demo_business/fixtures.json")
_EXTRA = ("evaluation/retrieval_phase_trial.py", "tests/test_retrieval_phase_trial.py",
    "core/model_deadline.py", "tests/test_model_deadline.py", "evaluation/phase_deadline_transport.py",
    "tests/test_phase_deadline_transport.py")


def unchanged_retrieval_sources(old):
    hashes = {}
    for name in _SAME_SOURCES:
        hashes[name] = bench.sha256(bench.BACKEND_ROOT / name)
        if hashes[name] != old["artifacts"].get(name):
            base._fail("retrieval_phase_business_source_changed")
    name = "tooling/tool_manager.py"
    try:
        before = subprocess.run(["git", "show", OLD["commit"] + ":backend/" + name],
            cwd=bench.BACKEND_ROOT, capture_output=True, check=True, timeout=10).stdout
        after = (bench.BACKEND_ROOT / name).read_bytes()
        if not before or len(before) > 262144 or len(after) > 262144:
            raise ValueError
        patch = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True),
            after.decode().splitlines(keepends=True), fromfile="before/" + name, tofile="after/" + name, n=3)).encode()
        if (hashlib.sha256(before).hexdigest() != old["artifacts"][name]
                or hashlib.sha256(patch).hexdigest() != TOOL_MANAGER_PATCH):
            raise ValueError
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("retrieval_phase_unapproved_tool_manager_change")
    return {"unchanged_dependency_hashes": hashes, "tool_manager_old_sha256": old["artifacts"][name],
        "tool_manager_current_sha256": hashlib.sha256(after).hexdigest(), "existing_clock_patch_sha256": TOOL_MANAGER_PATCH,
        "scope": "exact existing await_model import and two same-value timeout wrappers only; no prompt/algorithm/200-character change"}


def prior_proof():
    if (_tree(OLD_ROOT) != OLD["tree"] or _tree(original.BASELINE_ROOT) != BASELINE_TREE
            or bench.sha256(base._path(OLD_ROOT, "protocol.json")) != OLD["protocol"]
            or bench.sha256(base._path(OLD_ROOT, "ledger/journal.jsonl")) != OLD["journal"]
            or (OLD_ROOT / "authorization.json").exists()):
        base._fail("retrieval_phase_old_preparation_changed")
    p = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
    original.archive.history._archived_sources(p, "artifacts")
    budget = original.RetrievalBudget(OLD_ROOT / "ledger", OLD["protocol"])
    state, results = budget.status(), budget.results()
    if (p["git_head"] != OLD["commit"] or p["schema_version"] != original.RetrievalTrial.protocol_schema
            or p["batch"] != original.BATCH or p["root_binding_sha256"] != base._root_hash(OLD_ROOT)
            or p["operations"] != list(original.OPERATIONS) or p["authorized_model_requests"] != 0
            or p["proposed_model_requests"] != 48 or p["limits"] != original.LIMITS or p["policy"] != original.POLICY
            or state != {"attempts": 0, "completed": [], "pending": [], "paused": False, "chain_hash": OLD["chain"]}
            or results or p["prior_proof"] != original.prior_proof()
            or p["baseline_binding"] != original.baseline_binding()):
        base._fail("retrieval_phase_old_identity_changed")
    source_proof = unchanged_retrieval_sources(p)
    if _tree(OLD_ROOT) != OLD["tree"] or _tree(original.BASELINE_ROOT) != BASELINE_TREE:
        base._fail("retrieval_phase_old_changed_during_read")
    return {"schema_version": "serveflow-retrieval-preparation-readonly-proof-v2", "old": deepcopy(OLD),
        "old_authorized_requests": 0, "old_actual_requests": 0, "old_allowance_reused": False,
        "old_scores_recomputed": False, "baseline_tree_sha256": BASELINE_TREE,
        "older_proof_sha256": canonical_payload_hash(p["prior_proof"]), "retrieval_sources": source_proof}


def artifacts():
    return {**original.artifacts(), **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in _EXTRA}}


def read_request(root, wire_hash):
    if type(wire_hash) is not str or len(wire_hash) != 64 or any(c not in "0123456789abcdef" for c in wire_hash):
        base._fail("retrieval_phase_request_hash_invalid")
    value = bench.read_json(base._path(root, "request-" + wire_hash + ".json"))
    bench._exact(value, {"schema_version", "payload", "wire_text", "payload_sha256", "request_bytes_sha256"}, "retrieval_phase_request_fields")
    if (value["schema_version"] != "serveflow-retrieval-request-v2" or type(value["wire_text"]) is not str
            or len(value["wire_text"].encode()) > original.LIMITS["request_bytes"]
            or hashlib.sha256(value["wire_text"].encode()).hexdigest() != wire_hash
            or value["request_bytes_sha256"] != wire_hash
            or canonical_payload_hash(value["payload"]) != value["payload_sha256"]
            or json.loads(value["wire_text"], object_pairs_hook=original._unique_pairs,
                          parse_constant=original._reject_constant) != value["payload"]):
        base._fail("retrieval_phase_request_changed")
    return value


def read_audit(root, name):
    value = bench.read_json(base._path(root, name))
    seal = bench.read_json(base._path(root, name.removesuffix(".json") + "-seal.json"))
    if seal != {"artifact_sha256": bench.sha256(base._path(root, name)),
                "protocol_sha256": bench.sha256(base._path(root, "protocol.json"))}:
        base._fail("retrieval_phase_audit_changed")
    return value


def write_audit(root, name, value):
    if (root / name).exists():
        if read_audit(root, name) != value: base._fail("retrieval_phase_audit_changed")
        return
    base._write(root, name, value)
    base._write(root, name.removesuffix(".json") + "-seal.json", {
        "artifact_sha256": bench.sha256(base._path(root, name)),
        "protocol_sha256": bench.sha256(base._path(root, "protocol.json"))})


class _PhaseRetrievalGuard(GuardedIntentTransport):
    """Async bounded admission around the unchanged one-send network guard."""

    def __init__(self, base_url, expected, before_send, *, request, root, transport=None):
        self.frame = original.CURRENT.get()
        if self.frame is None or self.frame[0] not in original.CASE_IDS or self.frame[1] not in PHASE_SECONDS:
            raise TrialTransportError("protocol")
        self.limit_s = PHASE_SECONDS[self.frame[1]]
        self.wire_hash = hashlib.sha256(request.content).hexdigest()
        self.response_received = False
        self.stage_reason = None
        self._admit = before_send
        self._request_root = root
        super().__init__(base_url, expected, lambda: None, transport=transport,
                         timeout_s=self.limit_s, response_limit=original.LIMITS["response_bytes"])

    def _timeout_ceiling(self): return float(SDK_TIMEOUT_S)

    def _prepare_network_request(self, request):
        assert_model_active(); _uncancelled(); _sdk_timeout(request)
        if (original.CURRENT.get() != self.frame
                or hashlib.sha256(request.content).hexdigest() != self.wire_hash):
            raise TrialTransportError("protocol")
        request.extensions["timeout"] = dict.fromkeys(("connect", "read", "write", "pool"), float(self.limit_s))

    async def _response_bytes(self, response):
        raw = await super()._response_bytes(response)
        self.response_received = True
        return raw

    async def handle_async_request(self, request):
        def admit():
            # These are the exact SDK bytes; labels, author expectations and
            # baseline results are never constructed as model inputs here.
            value = {"schema_version": "serveflow-retrieval-request-v2", "wire_text": request.content.decode(),
                "payload": json.loads(request.content, object_pairs_hook=original._unique_pairs,
                                      parse_constant=original._reject_constant),
                "request_bytes_sha256": self.wire_hash, "payload_sha256": self._expected_hash}
            path = self._request_root / ("request-" + self.wire_hash + ".json")
            if path.exists():
                if read_request(self._request_root, self.wire_hash) != value:
                    raise TrialTransportError("protocol")
            else: base._write(self._request_root, path.name, value)
            self._admit()
        try:
            assert_model_active(); _uncancelled(); _sdk_timeout(request)
            self._validate_request(request)
            async with overhead("admission"):
                await original.task_data._owned(asyncio.to_thread(admit))
            # Reservation may have completed immediately before cancellation;
            # leave it occupied, and never start the network afterwards.
            assert_model_active(); _uncancelled()
            return await super().handle_async_request(request)
        except BaseException as error:
            self.stage_reason = current_deadline_failure_reason()
            if self.stage_reason is None and isinstance(error, ModelDeadlineExceeded):
                self.stage_reason = error.reason
            raise
        finally:
            if not self._used:
                await self._cleanup(None)


class RetrievalPhaseTransport(original.RetrievalTransport):
    def make_guard(self, request, payload_hash, admit, delegate):
        return _PhaseRetrievalGuard(self.protocol["base_url"], payload_hash, admit,
            request=request, root=self.trial.root, transport=delegate)

    def failure_code(self, error, admission_code, guard):
        if isinstance(error, asyncio.CancelledError): return "cancelled"
        reason = current_deadline_failure_reason() or (guard.stage_reason if guard else None)
        if reason is not None or isinstance(error, ModelDeadlineExceeded):
            if guard is not None: guard.stage_reason = reason or error.reason
            return "timeout"
        return super().failure_code(error, admission_code, guard)

    def error_observation(self, parsed, response_hash):
        if parsed is not None:
            return deepcopy(parsed), response_hash
        return super().error_observation(parsed, response_hash)

    async def record(self, *args):
        if args[3] == "execution_error":
            # Cleanup-only accounting does not reserve/send or borrow expired
            # active time. Unknown completion still blocks every later send.
            await original.task_data._owned(asyncio.to_thread(self.seal, *args))
        else:
            async with overhead("seal"):
                await original.task_data._owned(asyncio.to_thread(self.seal, *args))

    def physical_diagnostics(self, guard):
        value = {"sdk_limit_s": 30., "network_limit_s": float(guard.limit_s),
            "active_stage_limit_s": float(guard.limit_s), "response_received": guard.response_received}
        if guard.stage_reason is not None: value["stage_deadline_reason"] = guard.stage_reason
        return value


class RetrievalPhaseTrial(original.RetrievalTrial):
    root_path, batch = ROOT, BATCH
    protocol_schema = "serveflow-retrieval-phase-protocol-v2"
    authorization_schema = "serveflow-retrieval-phase-authorization-v2"
    report_schema = "serveflow-retrieval-phase-report-v2"

    def artifact_proof(self): return artifacts()
    def historical_proof(self): return prior_proof()
    def limits(self): return deepcopy(LIMITS)
    def execution_policy(self): return deepcopy(POLICY)
    def sdk_timeout_s(self): return SDK_TIMEOUT_S
    def execution_context(self): return eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=30, cleanup_grace_s=.25))
    def outer_deadline_s(self): return 3600
    def make_transport(self, *args, **kwargs): return RetrievalPhaseTransport(*args, **kwargs)

    def knowledge_ready(self, knowledge, protocol):
        value = {"schema_version": "serveflow-retrieval-actual-data-v2", "proof": deepcopy(knowledge.proof),
            "protocol_sha256": bench.sha256(self.root / "protocol.json")}
        write_audit(self.root, "actual-data-proof.json", value)

    def knowledge_closed(self, success):
        write_audit(self.root, "owned-collection-cleanup.json", {"schema_version": "serveflow-retrieval-cleanup-v2",
            "owned_collection_cleanup_complete": success, "protocol_sha256": bench.sha256(self.root / "protocol.json")})

    def verify(self):
        p = super().verify()
        old = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
        for name in ("model", "base_url", "parameters", "dependencies", "data_identity", "baseline_binding",
                     "cases_sha256", "initial_requests", "operations"):
            if p[name] != old[name]: base._fail("retrieval_phase_fixed_input_changed")
        results = self.budget(p).results()
        for operation, item in results.items():
            number = list(original.OPERATIONS).index(operation) + 1
            if item["output_sha256"] is not None:
                record = bench.read_json(base._path(self.root, f"physical-{number:03d}.json"))
                if read_request(self.root, record["request_bytes_sha256"])["payload_sha256"] != record["payload_sha256"]:
                    base._fail("retrieval_phase_physical_request_changed")
        path = self.root / "actual-data-proof.json"
        if path.exists():
            proof = read_audit(self.root, path.name)
            bench._exact(proof, {"schema_version", "proof", "protocol_sha256"}, "retrieval_phase_actual_data_fields")
            if (proof["schema_version"] != "serveflow-retrieval-actual-data-v2"
                    or proof.get("protocol_sha256") != bench.sha256(self.root / "protocol.json")
                    or {name: proof["proof"][name] for name in original.DATA_FIELDS} != p["data_identity"]):
                base._fail("retrieval_phase_actual_data_changed")
        if (self.root / "owned-collection-cleanup.json").exists():
            cleanup = read_audit(self.root, "owned-collection-cleanup.json")
            bench._exact(cleanup, {"schema_version", "owned_collection_cleanup_complete", "protocol_sha256"}, "retrieval_phase_cleanup_fields")
            if (cleanup["schema_version"] != "serveflow-retrieval-cleanup-v2"
                    or type(cleanup["owned_collection_cleanup_complete"]) is not bool
                    or cleanup["protocol_sha256"] != bench.sha256(self.root / "protocol.json")):
                base._fail("retrieval_phase_cleanup_changed")
        return p

    def report(self):
        report = super().report()
        baseline = bench.read_json(base._path(original.BASELINE_ROOT, "report.json"))
        cleanup = self.root / "owned-collection-cleanup.json"
        report.update({"baseline_splits": deepcopy(baseline["by_split"]), "stage_clock_policy": deepcopy(POLICY),
            "owned_collection_cleanup_complete": cleanup.exists() and read_audit(self.root, cleanup.name)["owned_collection_cleanup_complete"] is True,
            "comparison": "sealed vector baseline versus new complete rewrite/recall/dedup/rerank; no new A execution",
            "current_source_changed_since_zero_call_preparation": "existing active-clock wrapper only in ToolManager; precise source proof retained"})
        results = self.budget(self.verify()).results()
        report["paired_comparison_ready"] = bool(report["paired_comparison_ready"]
            and not report["stopped"] and not report["ledger"]["paused"] and not report["ledger"]["pending"]
            and report["model_requests"] == 48 and set(results) == set(original.OPERATIONS)
            and all(row["output_sha256"] is not None for row in results.values())
            and report["owned_collection_cleanup_complete"])
        return report


def main(argv=None):
    return original._main(RetrievalPhaseTrial, "evaluation.retrieval_phase_trial", argv)


if __name__ == "__main__":
    raise SystemExit(main())
