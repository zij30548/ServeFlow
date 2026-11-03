"""Fixed original retrieval pipeline comparison; prepare grants zero sends.

Calls the unchanged production rewrite/recall/rerank pipeline on the original
24 questions. A new, explicit protocol-bound authorization is mandatory. This
does not run answer generation, a Judge, background memory, or business APIs.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx

from evaluation import benchmark as bench, intent_trial as base, task_data
from evaluation.benchmark_metrics import score_retrieval
from evaluation.intent_transport import (
    GuardedIntentTransport, TrialTransportError, canonical_payload_hash,
    messages_url, _unique_pairs, _reject_constant,
)
from evaluation.retrieval_budget import CASE_IDS, STAGES, OPERATIONS, MAX_REQUESTS, RetrievalBudget
from evaluation.task_repair_archive import paused_predecessor_proof
from evaluation.task_transport import _response
from evaluation.retrieval_baseline import _rows as source_rows
from evaluation import task_recheck as archive
from evaluation.task_budget import TaskBudget


ROOT = bench.BACKEND_ROOT / "data/benchmarks/retrieval-rewrite-rerank-v1"
BATCH = "serveflow-retrieval-rewrite-rerank-v1"
CASES_PATH = bench.DATASET_ROOT / "retrieval-v1.json"
CURRENT = ContextVar("serveflow_retrieval_stage", default=None)
PARAMETERS = {"rewrite": (512, .3), "rerank": (2048, .0)}
LIMITS = {"physical_requests": 48, "per_query": 2, "top_k": 3, "recall_k": 5,
    "rewrites": 3, "request_bytes": 65536, "response_bytes": 262144,
    "request_deadline_s": 8, "sdk_retries": 0, "http_retries": 0, "redirects": False}
DATA_FIELDS = ("chromadb_version", "embedding", "embedding_implementation_sha256",
               "corpus_sha256", "document_count", "chunk_count", "chunk_sha256")
PROTOCOL_FIELDS = {"schema_version", "batch", "created_at", "git_head", "root_binding_sha256",
    "artifacts", "cases_sha256", "model", "base_url", "limits", "parameters",
    "dependencies", "data_identity", "prior_proof", "initial_requests", "operations",
    "authorized_model_requests", "proposed_model_requests", "policy", "baseline_binding"}
POLICY = {"scope": "retrieval-only; no Agent answers or Judge",
    "comparison": "same frozen questions and corpus as vector baseline; one execution per query",
    "dynamic_payload": "freeze rewrite inputs; hash actual rerank SDK bytes before reserve",
    "no_rerank": "seal observed case and pause if fixed two-stage plan is not reached; never add a request to fill quota",
    "failure": "execution/version/data/ledger failure pauses; production parsing fallbacks remain visible",
    "ablation": "vector-only versus unchanged full pipeline; intermediate recall is an observation, not an independently executed arm",
    "meaning": "retrieval hit is not answer correctness; no guaranteed method improvement"}
PAUSED_V3 = bench.BACKEND_ROOT / "data/benchmarks/task-e2e-v3-evidence-fix"
V3_PROTOCOL = "e08044cf73fdbafa47dcd6351ad8fc176af50ffbb5e5a12498e0b36bc60d2c3f"
V3_AUTHORIZATION = "c95d3c0e580d83eb89a57b95f9978f822310524f674246d24d4bfa87a1e986c7"
V3_JOURNAL = "931118a7265840593450a7789294f13760c1dfdda10d0de7c004a389c04d5b61"
V3_CHAIN = "98101c74c470a41292e257adb30a5fe3969ea2d895fd4b3193dac86d87cc17d0"
V3_TREE = "8e93698ec3d43f64e9fc728f466ac6d5b680c7570e66696ade9534966129eaef"
BASELINE_ROOT = bench.BACKEND_ROOT / "data/benchmarks/retrieval-vector-baseline-v1"
BASELINE_PROTOCOL = "a297b441550bf426ae06c7262798fa613dc8d541cb72acd893d68dbb494cc44a"
BASELINE_REPORT = "8242f9acc6e0d3931eba9478a5bb5bccc8539f0a00904a59572c5d898456dfcb"


def baseline_binding():
    """Read the sealed local baseline; never rerun or replace its measurement."""
    if (bench.sha256(base._path(BASELINE_ROOT, "protocol.json")) != BASELINE_PROTOCOL
            or bench.sha256(base._path(BASELINE_ROOT, "report.json")) != BASELINE_REPORT):
        base._fail("retrieval_baseline_changed")
    p = bench.read_json(base._path(BASELINE_ROOT, "protocol.json"))
    report = bench.read_json(base._path(BASELINE_ROOT, "report.json"))
    expected_plan = [{"case_id": c["id"], "split": c["split"],
                      "input_sha256": canonical_payload_hash(bench.model_input(c))} for c in cases()]
    if (p.get("schema_version") != "serveflow-local-retrieval-protocol-v1"
            or p.get("root_binding_sha256") != base._root_hash(BASELINE_ROOT)
            or p.get("plan") != expected_plan or p.get("knowledge") != data_identity()
            or p.get("top_k") != 3 or report.get("protocol_sha256") != BASELINE_PROTOCOL
            or report.get("evidence_kind") != "real-local-vector-retrieval"
            or report.get("execution_state") != "completed" or not report.get("owned_collection_cleanup_complete")
            or report.get("model_api_requests") != 0 or report.get("external_business_requests") != 0
            or type(report.get("observations")) is not list or len(report["observations"]) != 24):
        base._fail("retrieval_baseline_identity_changed")
    outputs = {}
    for plan, observed in zip(expected_plan, report["observations"]):
        case_id = plan["case_id"]
        output = base._path(BASELINE_ROOT, f"case-{case_id}.json")
        seal = bench.read_json(base._path(BASELINE_ROOT, f"case-{case_id}-seal.json"))
        digest = bench.sha256(output)
        if (observed != bench.read_json(output) or observed.get("case_id") != case_id
                or observed.get("split") != plan["split"] or observed.get("input_sha256") != plan["input_sha256"]
                or seal != {"protocol_sha256": BASELINE_PROTOCOL, "output_sha256": digest,
                            "claim_sha256": bench.sha256(base._path(BASELINE_ROOT, f"case-{case_id}-start.json"))}):
            base._fail("retrieval_baseline_output_changed")
        outputs[case_id] = digest
    return {"protocol_sha256": BASELINE_PROTOCOL, "report_sha256": BASELINE_REPORT,
            "cases_sha256": bench.sha256(CASES_PATH), "data_sha256": canonical_payload_hash(p["knowledge"]),
            "output_sha256": outputs, "measured_queries": 24, "model_requests": 0}


def prior_proof():
    """Bind both preserved paused batches, never continue or rescore either."""
    old = paused_predecessor_proof()
    before = archive._tree(PAUSED_V3)
    if (before != V3_TREE or bench.sha256(base._path(PAUSED_V3, "protocol.json")) != V3_PROTOCOL
            or bench.sha256(base._path(PAUSED_V3, "authorization.json")) != V3_AUTHORIZATION
            or bench.sha256(base._path(PAUSED_V3, "ledger/journal.jsonl")) != V3_JOURNAL):
        base._fail("retrieval_paused_v3_changed")
    p = bench.read_json(base._path(PAUSED_V3, "protocol.json"))
    if (p.get("schema_version") != "serveflow-task-protocol-v3"
            or p.get("git_head") != "2feaa7c259a431a4ec89499a56eac4614c947aa9"
            or p.get("root_binding_sha256") != base._root_hash(PAUSED_V3)
            or p.get("prior_proof") != old):
        base._fail("retrieval_paused_v3_changed")
    archive.history._archived_sources(p, "artifacts")
    ledger = TaskBudget(PAUSED_V3 / "ledger", V3_PROTOCOL, p["plan"])
    state, results = ledger.status(), ledger.results()
    if (state["attempts"] != 31 or state["pending"] or state["active_case"] != "e05"
            or state["paused"] is not True or state["chain_hash"] != V3_CHAIN
            or state["completed_cases"] != ["e01", "e02", "e03", "e04"]
            or len(results["operations"]) != 31 or set(results["cases"]) != {"e01", "e02", "e03", "e04"}):
        base._fail("retrieval_paused_v3_ledger_changed")
    timeouts = []
    for operation_id, row in results["operations"].items():
        if row["output_sha256"] != bench.sha256(base._path(PAUSED_V3, f"physical-{row['attempt']:03d}.json")):
            base._fail("retrieval_paused_v3_output_changed")
        if row["outcome"] != "received":
            timeouts.append((operation_id, row["attempt"], row["outcome"], row["error_code"]))
    if timeouts != [("e05:turn-1:agent:billing:2", 28, "execution_error", "timeout")]:
        base._fail("retrieval_paused_v3_timeout_changed")
    for case_id, row in results["cases"].items():
        if row["artifact_sha256"] != bench.sha256(base._path(PAUSED_V3, f"case-{case_id}.json")):
            base._fail("retrieval_paused_v3_output_changed")
    if any((PAUSED_V3 / name).exists() for name in ("case-e05.json", "case-e06.json")):
        base._fail("retrieval_paused_v3_unexecuted_rewritten")
    if archive._tree(PAUSED_V3) != before:
        base._fail("retrieval_archive_changed_during_read")
    return {"schema_version": "serveflow-retrieval-prior-v1", "paused_v2": old,
        "paused_v3": {"protocol_sha256": V3_PROTOCOL, "authorization_sha256": V3_AUTHORIZATION,
            "journal_sha256": V3_JOURNAL, "tree_sha256": before, "chain_hash": V3_CHAIN,
            "reserved_requests": 31, "received": 30, "timeouts": 1,
            "active_case": "e05", "completed_cases": list(state["completed_cases"]),
            "old_allowance_reused": False, "old_scores_recomputed": False},
        "new_authorized_requests": 0}


@contextmanager
def stage(case_id, name):
    if case_id not in CASE_IDS or name not in STAGES:
        base._fail("retrieval_stage_invalid")
    token = CURRENT.set((case_id, name))
    try:
        yield
    finally:
        CURRENT.reset(token)


def cases():
    value = bench.read_json(CASES_PATH)
    bench.validate_suite(value)
    rows = value["cases"]
    if tuple(c["id"] for c in rows) != CASE_IDS:
        base._fail("retrieval_case_order_changed")
    return rows


def data_identity():
    proof = task_data.offline_data_proof()["knowledge"]
    return {key: proof[key] for key in DATA_FIELDS}


def artifacts():
    extra = ["evaluation/retrieval_trial.py", "evaluation/retrieval_budget.py",
        "tests/test_retrieval_trial.py", "tests/test_retrieval_budget.py",
        "evaluation/task_data.py", "evaluation/task_repair_archive.py",
        "evaluation/task_recheck.py", "evaluation/task_trial.py", "evaluation/task_budget.py",
        "evaluation/retrieval_baseline.py", "tests/test_retrieval_baseline.py",
        "evaluation/task_supervisor.py", "tests/test_task_supervisor.py",
        "evaluation/intent_trial.py", "evaluation/intent_transport.py", "evaluation/intent_recheck.py",
        "evaluation/intent_validation.py", "evaluation/validation_ledger.py", "evaluation/trial_ledger.py"]
    return {name: bench.sha256(bench.BACKEND_ROOT / name)
            for name in sorted(set(bench.artifact_paths() + extra))}


def dependencies():
    from evaluation.task_trial import _dependencies
    return _dependencies()


async def build_owned_knowledge():
    """Do not release the run lease while a cancelled native builder is alive."""
    worker = asyncio.create_task(asyncio.to_thread(task_data._build_knowledge))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled() and worker.exception() is None:
            await task_data._owned(asyncio.to_thread(worker.result().close))
        raise


async def capture_rewrite(case, endpoint):
    from anthropic import AsyncAnthropic
    from tooling.tool_manager import MCPToolManager
    captured = []
    async def handle(request):
        captured.append({"payload": json.loads(request.content),
                         "wire_sha256": hashlib.sha256(request.content).hexdigest()})
        return httpx.Response(200, json={"id": "offline-capture", "type": "message", "role": "assistant",
            "model": base.MODEL, "content": [{"type": "text", "text": "[]"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=False, trust_env=False)
    sdk = AsyncAnthropic(api_key="offline-capture-only", base_url=endpoint, max_retries=0, http_client=http)
    manager = object.__new__(MCPToolManager)
    manager._client, manager._model = sdk, base.MODEL
    try:
        with base._quiet_clients():
            await manager.rewrite_query(case["input"]["message"], n=3)
        if len(captured) != 1:
            base._fail("retrieval_capture_failed")
        value = captured[0]
        return {"case_id": case["id"], "input_sha256": canonical_payload_hash(bench.model_input(case)),
                "payload": value["payload"], "payload_sha256": canonical_payload_hash(value["payload"]),
                "request_bytes_sha256": value["wire_sha256"]}
    finally:
        await sdk.close()


class RetrievalTransport(httpx.AsyncBaseTransport):
    """All original pipeline model requests cross one sticky send boundary."""

    def __init__(self, trial, protocol, *, delegate_factory=None):
        self.trial, self.protocol, self.factory = trial, protocol, delegate_factory
        self.fault = None
        self.kind = "controlled-transport" if delegate_factory is not None else "live-model"

    def healthy(self):
        if self.fault is not None:
            raise TrialTransportError(self.fault)

    def stop(self, code):
        code = code if code in {"auth", "quota", "rate_limit", "network", "timeout", "cancelled", "response_limit", "protocol", "storage_failure"} else "protocol"
        self.fault = self.fault or code
        if not (self.trial.root / "stopped.json").exists():
            base._write(self.trial.root, "stopped.json", {"schema_version": "serveflow-retrieval-stop-v1",
                "error_code": self.fault, "recorded_at": base._now()})

    async def handle_async_request(self, request):
        frame, guard, operation = CURRENT.get(), None, None
        admission_code = None
        started = time.perf_counter()
        try:
            self.healthy()
            if frame is None:
                raise TrialTransportError("protocol")
            case_id, name = frame
            if case_id not in CASE_IDS or name not in STAGES or len(request.content) > LIMITS["request_bytes"]:
                raise TrialTransportError("protocol")
            payload = json.loads(request.content.decode(), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            expected = PARAMETERS.get(name)
            if (request.method != "POST" or str(request.url) != messages_url(self.protocol["base_url"])
                    or len(request.content) > LIMITS["request_bytes"] or expected is None
                    or set(payload) != {"model", "messages", "max_tokens", "temperature", "thinking"}
                    or payload["model"] != base.MODEL or type(payload["max_tokens"]) is not int
                    or type(payload["temperature"]) not in {int, float}
                    or not math.isfinite(payload["temperature"])
                    or (payload["max_tokens"], payload["temperature"]) != expected
                    or payload["thinking"] != {"type": "disabled"} or type(payload["messages"]) is not list):
                raise TrialTransportError("protocol")
            payload_hash = canonical_payload_hash(payload)
            wire_hash = hashlib.sha256(request.content).hexdigest()
            operation_id = f"{case_id}:{name}:1"
            def admit():
                nonlocal operation, admission_code
                try:
                    self.healthy()
                    self.trial.verify()
                    self.trial.approval(self.protocol)
                    if name == "rewrite":
                        frozen = next(row for row in self.protocol["initial_requests"] if row["case_id"] == case_id)
                        if frozen["payload_sha256"] != payload_hash or frozen["request_bytes_sha256"] != wire_hash:
                            raise TrialTransportError("protocol")
                    base._write(self.trial.root, f"admission-{case_id}-{name}.json", {
                        "operation_id": operation_id, "payload_sha256": payload_hash,
                        "request_bytes_sha256": wire_hash, "request_model": base.MODEL})
                    operation = self.trial.budget(self.protocol).reserve(operation_id, payload_hash)
                except Exception as error:
                    admission_code = getattr(error, "code", "protocol")
                    raise
            delegate = self.factory(frame) if self.factory is not None else None
            if self.factory is not None and not isinstance(delegate, httpx.MockTransport):
                raise TrialTransportError("protocol")
            guard = GuardedIntentTransport(self.protocol["base_url"], payload_hash, admit, transport=delegate)
            response = await guard.handle_async_request(request)
            try:
                parsed, outcome = _response(response.content), "received"
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError, TrialTransportError):
                parsed = {"content": [], "upstream_model": None, "stop_reason": None,
                          "usage": {"input_tokens": None, "output_tokens": None}}
                outcome = "structure_failed"
            self.seal(operation, guard, parsed, outcome, None, (time.perf_counter()-started)*1000,
                      hashlib.sha256(response.content).hexdigest(), wire_hash)
            return response
        except BaseException as error:
            code = admission_code or ("cancelled" if isinstance(error, asyncio.CancelledError) else
                    base._error(guard.failure_code) if guard and guard.failure_code else getattr(error, "code", "protocol"))
            self.stop(code)
            if operation is not None:
                try:
                    self.seal(operation, guard, {"content": [], "upstream_model": None,
                        "stop_reason": None, "usage": {"input_tokens": None, "output_tokens": None}},
                        "execution_error", self.fault, (time.perf_counter()-started)*1000, None, wire_hash)
                except Exception:
                    pass  # Occupied unknown claims are retained, never refunded.
            raise TrialTransportError(self.fault) from None

    def seal(self, operation, guard, parsed, outcome, code, latency_ms, response_hash, wire_hash):
        record = {"schema_version": "serveflow-retrieval-physical-v1", "operation_id": operation["operation_id"],
            "attempt": operation["attempt"], "evidence_kind": self.kind, "outcome": outcome,
            "error_code": code, "payload_sha256": operation["payload_sha256"],
            "request_bytes_sha256": wire_hash, "request_model": base.MODEL,
            "request_id": guard.request_id, "http_status": guard.http_status,
            "delegate_entered": guard.sent, "latency_ms": latency_ms,
            "response_sha256": response_hash, "response": parsed}
        filename = f"physical-{operation['attempt']:03d}.json"
        base._write(self.trial.root, filename, record)
        self.trial.budget(self.protocol).finish(operation["operation_id"], outcome,
                                               bench.sha256(self.trial.root / filename), code)


class RetrievalTrial:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            base._fail("fixed_retrieval_root_required")

    def budget(self, protocol):
        return RetrievalBudget(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")))

    async def prepare(self):
        if self.root.exists():
            base._fail("retrieval_root_already_exists")
        values, config = cases(), base._configuration()
        requests = [await capture_rewrite(case, config["base_url"]) for case in values]
        value = {"schema_version": "serveflow-retrieval-protocol-v1", "batch": BATCH,
            "created_at": base._now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                cwd=bench.BACKEND_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": artifacts(),
            "cases_sha256": bench.sha256(CASES_PATH), "model": base.MODEL, "base_url": config["base_url"],
            "limits": deepcopy(LIMITS), "parameters": deepcopy(PARAMETERS), "policy": deepcopy(POLICY),
            "dependencies": dependencies(), "data_identity": data_identity(), "prior_proof": prior_proof(),
            "baseline_binding": baseline_binding(),
            "initial_requests": requests, "operations": list(OPERATIONS),
            "authorized_model_requests": 0, "proposed_model_requests": 48}
        value = json.loads(json.dumps(value))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", value)
        RetrievalBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), initialize=True)
        fd = os.open(self.root / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        return value

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        bench._exact(p, PROTOCOL_FIELDS, "retrieval_protocol_fields")
        config = base._configuration()
        if (p["schema_version"] != "serveflow-retrieval-protocol-v1" or p["batch"] != BATCH
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["limits"] != LIMITS
                or p["parameters"] != json.loads(json.dumps(PARAMETERS)) or p["policy"] != POLICY
                or p["operations"] != list(OPERATIONS) or p["dependencies"] != dependencies()
                or p["data_identity"] != data_identity() or p["prior_proof"] != prior_proof()
                or p["baseline_binding"] != baseline_binding()
                or type(p["authorized_model_requests"]) is not int or p["authorized_model_requests"] != 0
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != 48):
            base._fail("retrieval_protocol_changed")
        if p["artifacts"] != artifacts() or p["cases_sha256"] != bench.sha256(CASES_PATH):
            base._fail("retrieval_sources_changed")
        if type(p["initial_requests"]) is not list or len(p["initial_requests"]) != 24:
            base._fail("retrieval_initial_requests_changed")
        for case, row in zip(cases(), p["initial_requests"]):
            if (set(row) != {"case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or row["case_id"] != case["id"]
                    or row["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                    or row["payload_sha256"] != canonical_payload_hash(row["payload"])):
                base._fail("retrieval_initial_requests_changed")
        state, results = self.budget(p).status(), self.budget(p).results()
        for operation_id, result in results.items():
            number = list(OPERATIONS).index(operation_id) + 1
            if result["output_sha256"] is not None and bench.sha256(base._path(self.root, f"physical-{number:03d}.json")) != result["output_sha256"]:
                base._fail("retrieval_physical_changed")
        for case in cases():
            seal = base._path(self.root, f"sample-seal-{case['id']}.json")
            sample = base._path(self.root, f"sample-{case['id']}.json")
            if seal.exists():
                record = bench.read_json(seal)
                if record != {"sample_sha256": bench.sha256(sample), "protocol_sha256": bench.sha256(self.root / "protocol.json")}:
                    base._fail("retrieval_sample_changed")
            elif sample.exists():
                base._fail("retrieval_sample_unsealed")
        return p

    def authorize(self, reference, *, confirmed=False):
        p = self.verify()
        if confirmed is not True or type(reference) is not str or not reference.strip() or len(reference) > 500:
            base._fail("retrieval_new_authorization_required")
        ledger = self.budget(p)
        if ledger.status()["attempts"]:
            base._fail("retrieval_already_started")
        value = {"schema_version": "serveflow-retrieval-authorization-v1", "reference": reference,
            "recorded_at": base._now(), "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "root_binding_sha256": base._root_hash(self.root), "initial_chain_hash": ledger.initial_chain_hash,
            "max_requests": 48, "operations": list(OPERATIONS)}
        base._write(self.root, "authorization.json", value)
        return value

    def approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256",
            "initial_chain_hash", "max_requests", "operations"}, "retrieval_authorization_fields")
        if (value["schema_version"] != "serveflow-retrieval-authorization-v1"
                or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.budget(p).initial_chain_hash
                or type(value["max_requests"]) is not int or value["max_requests"] != 48
                or value["operations"] != list(OPERATIONS)):
            base._fail("retrieval_authorization_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_knowledge=None):
        from anthropic import AsyncAnthropic
        from tooling.tool_manager import MCPToolManager, Tool
        p = self.verify()
        self.approval(p)  # Before secrets, embedding inference or SDK clients.
        if (delegate_factory is None) != (controlled_knowledge is None):
            base._fail("retrieval_controlled_boundary_incomplete")
        if (self.root / "stopped.json").exists():
            base._fail("retrieval_stopped")
        lock = os.open(base._path(self.root, "run.lock"), os.O_RDWR | os.O_NOFOLLOW)
        knowledge, sdk = None, None
        owns_run = False
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owns_run = True
            except BlockingIOError:
                base._fail("retrieval_run_busy")
            self.verify()
            state = self.budget(p).status()
            if state["paused"] or state["pending"]:
                base._fail("retrieval_pending_or_paused")
            for case in cases():
                started = any(op in state["completed"] for op in (f"{case['id']}:rewrite:1", f"{case['id']}:rerank:1"))
                if started and not (self.root / f"sample-seal-{case['id']}.json").exists():
                    base._fail("retrieval_unfinished_sample")
            if all((self.root / f"sample-seal-{c['id']}.json").exists() for c in cases()):
                return self.report()
            config = base._configuration(secret=delegate_factory is None)
            knowledge = controlled_knowledge or await build_owned_knowledge()
            if {k: knowledge.proof[k] for k in DATA_FIELDS} != p["data_identity"]:
                base._fail("retrieval_actual_data_changed")
            transport = RetrievalTransport(self, p, delegate_factory=delegate_factory)
            http = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False,
                timeout=8, headers={"Accept-Encoding": "identity"})
            sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                max_retries=0, timeout=8, http_client=http)
            for case in cases():
                if (self.root / f"sample-seal-{case['id']}.json").exists():
                    continue
                began = time.perf_counter()
                manager = MCPToolManager("unused-replaced-before-use", base_url=config["base_url"], model=base.MODEL)
                old = manager._client
                manager._client = sdk
                await old.close()
                manager.register(Tool("knowledge_search", "Frozen local policy collection", knowledge.kb.search_handler,
                    schema={"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                            "required": ["query"], "additionalProperties": False},
                    cache_ttl=300, supports_rerank=True, timeout_s=8))
                original_rewrite, original_rerank = manager.rewrite_query, manager._rerank_with_status
                method = {"rewrite_queries": [], "rerank_applied": False, "rerank_input_count": None,
                          "merged_sources": [], "rewrite_recall_top3": []}
                async def rewrite(*args, **kwargs):
                    transport.healthy()
                    with stage(case["id"], "rewrite"):
                        value = await original_rewrite(*args, **kwargs)
                    transport.healthy()
                    method["rewrite_queries"] = value
                    return value
                async def rerank(query, items, top_k):
                    transport.healthy()
                    method["rerank_input_count"] = len(items)
                    if len(items) > 20:
                        base._fail("retrieval_merged_limit")
                    method["merged_sources"] = [source_rows([item])[0] for item in items]
                    method["rewrite_recall_top3"] = [row["title"] for row in method["merged_sources"][:3]]
                    with stage(case["id"], "rerank"):
                        value, applied = await original_rerank(query, items, top_k)
                    transport.healthy()
                    method["rerank_applied"] = applied
                    return value, applied
                manager.rewrite_query, manager._rerank_with_status = rewrite, rerank
                with base._quiet_clients():
                    result = await manager.search_with_rewrite("knowledge_search", case["input"]["message"], top_k=3)
                transport.healthy()
                data = result.data if isinstance(result.data, list) else []
                sources = source_rows(data)
                titles = []
                for row in data:
                    if row["title"] not in titles:
                        titles.append(row["title"])
                physical = []
                for name in STAGES:
                    number = list(OPERATIONS).index(f"{case['id']}:{name}:1") + 1
                    path = base._path(self.root, f"physical-{number:03d}.json")
                    if path.exists():
                        physical.append(bench.read_json(path))
                usage = {key: sum(item["response"]["usage"][key] for item in physical)
                         if all(item["response"]["usage"][key] is not None for item in physical) else None
                         for key in ("input_tokens", "output_tokens")}
                usage["physical_requests"] = len(physical)
                record = {"case_id": case["id"], "status": "success" if result.success else "failed",
                    "latency_ms": (time.perf_counter()-began)*1000, "usage": usage,
                    "ranked_titles": titles}
                sample = {"schema_version": "serveflow-retrieval-sample-v1", "case_id": case["id"],
                    "split": case["split"], "evidence_kind": transport.kind, "record": record,
                    "method_observation": method, "sources": sources,
                    "stage_outputs": {item["operation_id"]: bench.sha256(self.root / f"physical-{item['attempt']:03d}.json") for item in physical}}
                filename = f"sample-{case['id']}.json"
                base._write(self.root, filename, sample)
                base._write(self.root, f"sample-seal-{case['id']}.json", {"sample_sha256": bench.sha256(self.root / filename),
                    "protocol_sha256": bench.sha256(self.root / "protocol.json")})
                if len(physical) != 2:
                    transport.stop("protocol")
                    base._fail("retrieval_two_stage_plan_not_reached")
        except BaseException as error:
            if owns_run and not (self.root / "stopped.json").exists():
                base._write(self.root, "stopped.json", {"schema_version": "serveflow-retrieval-stop-v1",
                    "error_code": "cancelled" if isinstance(error, asyncio.CancelledError) else "protocol",
                    "recorded_at": base._now()})
            raise
        finally:
            try:
                if sdk is not None:
                    await task_data._owned(sdk.close())
            finally:
                try:
                    if knowledge is not None:
                        await task_data._owned(asyncio.to_thread(knowledge.close))
                finally:
                    os.close(lock)
        return self.report()

    def report(self):
        p = self.verify()
        records = []
        intermediate = []
        kind = []
        ledger = self.budget(p)
        state, results = ledger.status(), ledger.results()
        attempted = set(state["completed"]) | set(state["pending"])
        physical = {}
        for operation_id, item in results.items():
            number = list(OPERATIONS).index(operation_id) + 1
            if item["output_sha256"] is not None:
                physical[operation_id] = bench.read_json(base._path(self.root, f"physical-{number:03d}.json"))
        for case in cases():
            path = base._path(self.root, f"sample-{case['id']}.json")
            if path.exists():
                sample = bench.read_json(path)
                records.append(sample["record"])
                intermediate.append({**sample["record"], "latency_ms": None,
                    "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None},
                    "ranked_titles": sample["method_observation"]["rewrite_recall_top3"]})
                kind.append(sample["evidence_kind"])
            else:
                operations = [op for op in OPERATIONS if op.startswith(case['id'] + ':') and op in attempted]
                if not operations:
                    continue  # Only cases with no reservation are unexecuted.
                errors = [results[op] for op in operations if op in results and results[op]["outcome"] in {"execution_error", "cancelled", "unknown"}]
                status = ("cancelled" if any(row["outcome"] == "cancelled" or row["error_code"] == "cancelled" for row in errors)
                          else "failed" if any(row["outcome"] == "execution_error" for row in errors) else "unknown")
                outputs = [physical[op] for op in operations if op in physical]
                usage = {key: sum(row["response"]["usage"][key] for row in outputs)
                         if len(outputs) == len(operations) and all(row["response"]["usage"][key] is not None for row in outputs) else None
                         for key in ("input_tokens", "output_tokens")}
                placeholder = {"case_id": case["id"], "status": status, "latency_ms": None,
                    "usage": {**usage, "physical_requests": len(operations)}, "ranked_titles": []}
                records.append(placeholder)
                intermediate.append({**placeholder, "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None}})
                kind.extend(row["evidence_kind"] for row in outputs)
        token_totals = {}
        for key in ("input_tokens", "output_tokens"):
            known = [row["response"]["usage"][key] for row in physical.values() if row["response"]["usage"][key] is not None]
            token_totals[key] = {"known_sum": sum(known) if known else None,
                "known_requests": len(known), "unknown_requests": state["attempts"]-len(known)}
        return {"schema_version": "serveflow-retrieval-report-v1", "batch": BATCH,
            "model_requests": state["attempts"], "usage": token_totals,
            "authorized_requests": 48 if (self.root / "authorization.json").exists() and self.approval(p) else 0,
            "ledger": state, "evidence_kinds": sorted(set(kind)), "stopped": (self.root / "stopped.json").exists(),
            "baseline_binding": p["baseline_binding"],
            "paired_comparison_ready": all((self.root / f"sample-seal-{c['id']}.json").exists() for c in cases()),
            "splits": {split: score_retrieval([c for c in cases() if c["split"] == split],
                [r for r in records if next(c for c in cases() if c["id"] == r["case_id"])["split"] == split])
                for split in ("development", "validation")},
            "intermediate_rewrite_recall_splits": {split: score_retrieval([c for c in cases() if c["split"] == split],
                [r for r in intermediate if next(c for c in cases() if c["id"] == r["case_id"])["split"] == split])
                for split in ("development", "validation")},
            "limitations": ["known authored questions and six synthetic policy documents", "no generated answers or semantic answer score",
                "one measurement, no guaranteed improvement",
                "original query is retained first with recall_k=5; intermediate top3 may match the vector baseline",
                "production reranker sees the first 200 characters of each serialized candidate, unchanged",
                "no relevance threshold: nonempty retrieval is not evidence that an out-of-scope question can be answered"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "report", "authorize", "run", "_worker"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-new-authorization", action="store_true")
    args = parser.parse_args(argv)
    trial = RetrievalTrial()
    try:
        if args.operation == "prepare":
            value = asyncio.run(trial.prepare())
        elif args.operation == "verify":
            trial.verify(); value = {"verified": True, "model_requests": trial.report()["model_requests"]}
        elif args.operation == "authorize":
            value = trial.authorize(args.reference, confirmed=args.confirm_new_authorization)
        elif args.operation == "_worker":
            value = asyncio.run(trial.execute())
        elif args.operation == "run":
            trial.approval(trial.verify())
            from evaluation.task_supervisor import supervise
            os.chdir(bench.BACKEND_ROOT)
            value = supervise([sys.executable, "-B", "-u", "-m", "evaluation.retrieval_trial", "_worker"], root=ROOT)
        else:
            value = trial.report()
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        if args.operation == "run" and (value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled")):
            return 2
        return 0
    except Exception:
        print("STOP: retrieval_trial_failed; inspect private evidence, never reinitialize or auto-retry", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
