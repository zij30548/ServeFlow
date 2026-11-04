"""New-question execution preparation; ordinary commands never grant sends.

prepare/verify/status/report are offline. authorize records a separately obtained
allowance; this module does not obtain one. run requires it before credentials,
data stores or a real HTTP transport can be opened. Old evidence is read-only.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

from evaluation import benchmark as bench, intent_trial as base, prospective_validation as design
from evaluation import task_data, task_trial as engine, task_recheck as history
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.prospective_budget import ProspectiveBudget, fixed_plan, LIMITS, MAX_REQUESTS
from evaluation.task_transport import BoundClient, Frame, context, PHASE_PARAMETERS, AGENT_PARAMETERS
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines


ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-execution-v1"
DESIGN_COMMIT = "bcd7b8aa0d915487d5595037b82104add395c271"
DESIGN_HASH = "64eb04c27fa6585b370a4c4ced0468256d0d5b86e937ddf4bc7b7afcd16a7556"
EXECUTION_FILES = tuple(f"{folder}/{name}.py" for folder, names in (
    ("evaluation", ("prospective_budget", "prospective_task_runtime", "prospective_retrieval_runtime", "prospective_trial")),
    ("tests", ("test_prospective_budget", "test_prospective_task_runtime", "test_prospective_retrieval_runtime", "test_prospective_trial"))) for name in names)
STOP_FILES = ("stopped.json", "worker-stop.json", "supervisor-stop.json")
KNOWLEDGE_FIELDS = ("chromadb_version", "embedding", "embedding_implementation_sha256", "corpus_sha256",
                    "document_count", "chunk_count", "chunk_sha256")
POLICY = {"scope": "new authored questions, same shared synthetic sources, not independent blind/population evaluation",
    "tasks": "24 single-turn A/D jobs; no Judge, no gold sent; actual native/handler/source evidence and seven-dimensional executor review",
    "retrieval": "12 paired local-A/existing-B jobs; original per-case arm order; no new algorithm or threshold",
    "clock": "SDK ceiling30; original active/network intent8 agent15 rewrite10 rerank30 composer8; overhead30 cleanup.25; outer3600 seconds",
    "payload": "initial intent and rewrite requests frozen offline; unknown follow-ups captured and checked as exact SDK bytes before atomic reservation",
    "errors": "ordinary semantic/structure errors retained; accounting, source, authorization, version, transport or unknown send stops the batch",
    "authorization": "independent receipt required; optional lower limit is checked inside the same durable reservation lock",
    "limits": "576 is a mathematical ceiling, not an allowance, expectation, monetary cost or a requirement to use it up",
    "score": "source/structure checks never auto-pass semantics; no existing output is repaired or inserted into new scores"}


def _safe_file(path):
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        base._fail("prospective_unsafe_file")
    return path


def design_proof():
    """Explicit addition proof, not an ignore-version switch on the old verify.

The old dynamic manifest must reject added code. Preserve that behavior and
independently check every original Git blob/current byte, plus a closed list of
new execution files. A later unknown helper or modified business file fails.
"""
    path = _safe_file(design.ROOT / "protocol.json")
    if bench.sha256(path) != DESIGN_HASH:
        base._fail("prospective_original_design_changed")
    old = bench.read_json(path)
    if (old["source_commit"] != DESIGN_COMMIT or old["root_binding_sha256"] != design._hash(str(design.ROOT))
            or old["schema_version"] != "serveflow-prospective-design-freeze-v1"
            or old["design"] != design.design() or old["plan"] != design._plan(bench.BACKEND_ROOT)
            or old["suites"] != design.validate() or old["dependencies"] != design._dependencies()
            or any(type(old[name]) is not int or old[name] != 0
                   for name in ("authorized_model_requests", "executed_model_requests", "external_business_requests"))
            or old["live_trial_ready"] is not False or old["live_payloads_frozen"] is not False):
        base._fail("prospective_original_design_identity_changed")
    sources = old["sources"]
    request = "".join(f"{DESIGN_COMMIT}:backend/{name}\n" for name in sources).encode()
    result = subprocess.run(["git", "cat-file", "--batch"], input=request, cwd=bench.BACKEND_ROOT,
                            check=True, capture_output=True, timeout=15)
    raw, position = result.stdout, 0
    if len(raw) > 16 * 1024 * 1024:
        base._fail("prospective_archived_source_invalid")
    for name, expected in sources.items():
        end = raw.find(b"\n", position)
        header = raw[position:end].split()
        if end < 0 or len(header) != 3 or header[1] != b"blob":
            base._fail("prospective_archived_source_invalid")
        size = int(header[2]); position = end+1
        blob = raw[position:position+size]; position += size
        if (not 0 <= size <= 2*1024*1024 or raw[position:position+1] != b"\n"
                or hashlib.sha256(blob).hexdigest() != expected
                or bench.sha256(_safe_file(bench.BACKEND_ROOT / name)) != expected):
            base._fail("prospective_original_source_changed")
        position += 1
    if position != len(raw):
        base._fail("prospective_archived_source_invalid")
    current = design._sources(bench.BACKEND_ROOT)
    added = {name for name in EXECUTION_FILES if name.startswith("evaluation/")}
    if set(current) != set(sources) | added or any(current[name] != digest for name, digest in sources.items()):
        base._fail("prospective_unapproved_source_set")
    return {"design_protocol_sha256": DESIGN_HASH, "design_commit": DESIGN_COMMIT,
            "original_files": len(sources), "original_file_set_sha256": canonical_payload_hash(sources),
            "original_authorization_and_calls": 0, "explicit_additions": list(EXECUTION_FILES)}


def artifacts():
    return {**design._sources(bench.BACKEND_ROOT), **{
        name: bench.sha256(_safe_file(bench.BACKEND_ROOT / name)) for name in EXECUTION_FILES}}


def prior_ledgers():
    """Bind old ledger bytes without initializing, replaying or spending them."""
    output = {}
    for journal in sorted((bench.BACKEND_ROOT / "data/benchmarks").glob("*/ledger/journal.jsonl")):
        if journal.parent.parent == ROOT:
            continue
        folder = journal.parent.parent
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                base._fail("prospective_prior_evidence_symlink")
            if path.is_file():
                output[str(path.relative_to(bench.BACKEND_ROOT))] = bench.sha256(path)
    return {"old_files": len(output), "files_sha256": canonical_payload_hash(output),
            "reuse_allowance": False, "rescore_old_outputs": False}


def _phases():
    return {name: value for name, value in PHASE_PARAMETERS.items() if name != "judge"}


def _binding(data_proof, expected):
    """Expected local bytes vs actual isolated DB/index, never an application DB."""
    if (data_proof.get("schema_version") != "serveflow-task-live-data-v1"
            or data_proof.get("mysql", {}).get("literal_schema_grants") is not True
            or data_proof["mysql"].get("fixture_file_sha256") != expected["fixture_file_sha256"]
            or data_proof["mysql"].get("fixture_canonical_sha256") != expected["fixture_canonical_sha256"]
            or any(data_proof.get("knowledge", {}).get(key) != expected["knowledge"][key] for key in KNOWLEDGE_FIELDS)):
        base._fail("prospective_live_data_identity_changed")


class ProspectiveTransport(PhaseDeadlineTaskHTTPTransport):
    """Reuse exact-payload audit and guard, adding the fixed RAG first request."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = bench.read_json(base._path(self.root, "protocol.json"))
        self._rewrites = {row["case_id"]: row for row in p["initial_rewrite_requests"]}

    def _validate(self, request, frame):
        canonical, wire = super()._validate(request, frame)
        if frame.case_id.startswith("R-"):
            if frame.phase not in {"rag_rewrite", "rag_rerank"} or frame.agent is not None:
                raise TrialTransportError("protocol")
            if frame.phase == "rag_rewrite":
                expected = self._rewrites.get(frame.case_id)
                if expected is None or (canonical, wire) != (expected["payload_sha256"], expected["request_bytes_sha256"]):
                    raise TrialTransportError("protocol")
        if frame.phase == "judge":
            raise TrialTransportError("protocol")
        return canonical, wire


class ProspectiveTrial:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            base._fail("prospective_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return ProspectiveBudget(self.root/"ledger", bench.sha256(self.root/"protocol.json"), p["plan"],
                                 initialize=initialize, send_limit=send_limit)

    async def prepare(self):
        if self.root.exists(): base._fail("prospective_execution_root_exists")
        proof, config = design_proof(), base._configuration()
        from evaluation.retrieval_trial import capture_rewrite
        initial, rewrites = [], []
        task_cases = {c["id"]: c for c in design._cases(bench.BACKEND_ROOT, "tasks")["cases"]}
        for row in design._plan(bench.BACKEND_ROOT):
            case = task_cases[row["case_id"]] if row["suite"] == "tasks" else next(
                c for c in design._cases(bench.BACKEND_ROOT, "retrieval")["cases"] if c["id"] == row["case_id"])
            if row["suite"] == "tasks":
                capture = await base._capture(case, config["base_url"])
                capture.pop("operation_id")
                for arm in row["arm_order"]:
                    initial.append({**deepcopy(capture), "source_case_id": case["id"], "case_id": arm+"-"+case["id"]})
            else:
                capture = await capture_rewrite(case, config["base_url"])
                rewrites.append({**capture, "source_case_id": case["id"], "case_id": "R-"+case["id"]})
        p = {"schema_version": "serveflow-prospective-execution-protocol-v1", "created_at": base._now(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT, check=True,
                                       capture_output=True, text=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "design_proof": proof,
            "artifacts": artifacts(), "dependencies": engine._dependencies(), "model": base.MODEL,
            "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "plan": fixed_plan(), "cases": design._plan(bench.BACKEND_ROOT), "limits": deepcopy(LIMITS),
            "http_limits": deepcopy(PHASE_HTTP_LIMITS), "policy": deepcopy(POLICY),
            "phase_parameters": _phases(), "agent_parameters": deepcopy(AGENT_PARAMETERS),
            "initial_intent_requests": initial, "initial_rewrite_requests": rewrites,
            "offline_data_proof": task_data.offline_data_proof(), "effective_environment": history.effective_environment(),
            "prior_ledgers": prior_ledgers(), "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "judge_requests": 0, "external_business_requests": 0}
        p = json.loads(json.dumps(p))
        self.root.mkdir(parents=True, mode=0o700, exist_ok=False)
        base._write(self.root, "protocol.json", p)
        self.ledger(p, initialize=True)
        return p

    def _sealed(self, p):
        from evaluation.quality_trial import _read_request
        sealed = self.ledger(p).results()
        for operation_id, op in sealed["operations"].items():
            if op["output_sha256"] is None:
                # A closed but unsealed error is still spent/unknown, never
                # recoverable as a success or a reason to resend it.
                continue
            name = f"physical-{op['attempt']:03d}.json"
            if bench.sha256(base._path(self.root, name)) != op["output_sha256"]:
                base._fail("prospective_physical_changed")
            row = bench.read_json(self.root/name)
            if (row["operation_id"] != operation_id or row["case_id"] != op["case_id"]
                    or row["outcome"] != op["outcome"] or row["payload_sha256"] != op["payload_sha256"]
                    or _read_request(self.root, op["request_bytes_sha256"])["payload_sha256"] != op["payload_sha256"]):
                base._fail("prospective_physical_binding_changed")
        proof_path = self.root/"data-proof.json"
        proof_hash = None
        if proof_path.exists():
            proof_hash = bench.sha256(base._path(self.root, proof_path.name))
            proof = bench.read_json(proof_path)
            seal = bench.read_json(base._path(self.root, "data-proof-seal.json"))
            if seal != {"protocol_sha256": bench.sha256(self.root/"protocol.json"), "data_proof_sha256": proof_hash}:
                base._fail("prospective_data_proof_seal_changed")
            if proof.get("kind") != "controlled-fixture": _binding(proof, p["offline_data_proof"])
        if (self.root/"cleanup-complete.json").exists():
            if bench.read_json(base._path(self.root, "cleanup-complete.json")) != {
                    "owned_resources_closed": True, "protocol_sha256": bench.sha256(self.root/"protocol.json")}:
                base._fail("prospective_cleanup_record_changed")
        for key, row in sealed["cases"].items():
            if bench.sha256(base._path(self.root, f"case-{key}.json")) != row["artifact_sha256"]:
                base._fail("prospective_case_changed")
            artifact = bench.read_json(self.root/f"case-{key}.json")
            if (artifact["logical_case_id"] != key or artifact["protocol_sha256"] != bench.sha256(self.root/"protocol.json")
                    or proof_hash is None or artifact.get("data_proof_sha256") != proof_hash):
                base._fail("prospective_case_binding_changed")
        return sealed

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        config = base._configuration()
        expected = {"schema_version": "serveflow-prospective-execution-protocol-v1",
            "root_binding_sha256": base._root_hash(self.root), "design_proof": design_proof(),
            "artifacts": artifacts(), "dependencies": engine._dependencies(), "model": base.MODEL,
            "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "plan": fixed_plan(), "cases": design._plan(bench.BACKEND_ROOT), "limits": LIMITS,
            "http_limits": PHASE_HTTP_LIMITS, "policy": POLICY, "phase_parameters": _phases(),
            "agent_parameters": AGENT_PARAMETERS, "offline_data_proof": task_data.offline_data_proof(),
            "effective_environment": history.effective_environment(), "prior_ledgers": prior_ledgers()}
        expected = json.loads(json.dumps(expected))
        keys = set(expected) | {"created_at", "git_head", "initial_intent_requests", "initial_rewrite_requests",
                               "authorized_model_requests", "proposed_model_requests", "judge_requests", "external_business_requests"}
        bench._exact(p, keys, "prospective_protocol_fields")
        if (any(p[key] != value for key, value in expected.items())
                or any(type(p[name]) is not int or p[name] != 0
                       for name in ("authorized_model_requests", "judge_requests", "external_business_requests"))
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != MAX_REQUESTS):
            base._fail("prospective_execution_freeze_changed")
        for key, ids in (("initial_intent_requests", [r["case_id"] for r in p["plan"][:24]]),
                         ("initial_rewrite_requests", [r["case_id"] for r in p["plan"][24:]])):
            values = p[key]
            if type(values) is not list or [v.get("case_id") for v in values] != ids:
                base._fail("prospective_initial_requests_changed")
            for value in values:
                bench._exact(value, {"case_id", "source_case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}, "prospective_initial_fields")
                case = next(c for c in p["cases"] if c["case_id"] == value["source_case_id"])
                if (value["source_case_id"] != value["case_id"][2:]
                        or value["input_sha256"] != canonical_payload_hash(case["conversation_input"])
                        or value["payload_sha256"] != canonical_payload_hash(value["payload"])
                        or type(value["request_bytes_sha256"]) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", value["request_bytes_sha256"]) is None):
                    base._fail("prospective_initial_requests_changed")
        self.ledger(p).status(); self._sealed(p)
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p = self.verify()
        if (confirmed is not True or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference) > 500):
            base._fail("prospective_explicit_authorization_required")
        state = self.ledger(p).status()
        if state["attempts"] or state["active_case"] or state["completed_cases"]:
            base._fail("prospective_already_started")
        value = {"schema_version": "serveflow-prospective-authorization-v1", "reference": reference,
            "recorded_at": base._now(), "protocol_sha256": bench.sha256(self.root/"protocol.json"),
            "root_binding_sha256": base._root_hash(self.root), "initial_chain_hash": self.ledger(p).initial_chain_hash,
            "max_requests": max_requests, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256",
                             "initial_chain_hash", "max_requests", "plan"}, "prospective_approval_fields")
        if (value["schema_version"] != "serveflow-prospective-authorization-v1"
                or value["protocol_sha256"] != bench.sha256(self.root/"protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash or value["plan"] != p["plan"]
                or type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= MAX_REQUESTS):
            base._fail("prospective_approval_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        # No credential, database, SDK or delegate may be opened before approval.
        p = self.verify(); approval = self._approval(p)
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("prospective_controlled_boundary_required")
        if delegate_factory is None and self.root != ROOT:
            base._fail("prospective_fixed_live_root_required")
        if any((self.root/name).exists() for name in STOP_FILES):
            base._fail("prospective_paused")
        if self.ledger(p).status()["completed_cases"] == [row["case_id"] for row in p["plan"]]:
            return self.report()  # No credentials, resource reopen, or second cleanup.
        config = base._configuration(secret=delegate_factory is None)
        ledger, data, sdk, transport = self.ledger(p, send_limit=approval["max_requests"]), controlled_data, None, None
        approval_hash = bench.sha256(self.root/"authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                if data is None:
                    data = await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("prospective_controlled_data_required")
                if controlled_data is None: _binding(data.proof, p["offline_data_proof"])
                data.assert_healthy()
                proof_path = self.root/"data-proof.json"
                if proof_path.exists():
                    if bench.read_json(base._path(self.root, proof_path.name)) != data.proof:
                        base._fail("prospective_data_changed")
                else:
                    base._write(self.root, proof_path.name, data.proof)
                    base._write(self.root, "data-proof-seal.json", {"protocol_sha256": bench.sha256(self.root/"protocol.json"),
                                                                  "data_proof_sha256": bench.sha256(proof_path)})
                proof_hash = bench.sha256(proof_path)
                def admission():
                    self.verify(); self._approval(p)
                    if (bench.sha256(self.root/"authorization.json") != approval_hash
                            or bench.sha256(base._path(self.root, "data-proof.json")) != proof_hash):
                        base._fail("prospective_run_identity_changed")
                    data.assert_healthy()
                transport = ProspectiveTransport(config["base_url"], self.root, ledger, lease, admission,
                                                 delegate_factory=delegate_factory)
                from anthropic import AsyncAnthropic
                import httpx
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False,
                                        trust_env=False, headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                                     max_retries=0, timeout=SDK_TIMEOUT_S, http_client=http)
                def healthy():
                    if transport.fault: base._fail("prospective_transport_stopped")
                    admission()
                from evaluation.prospective_task_runtime import execute_arm_case
                from evaluation.prospective_retrieval_runtime import execute_retrieval_case
                for row in p["plan"]:
                    logical = row["case_id"]
                    if logical in ledger.status()["completed_cases"]: continue
                    healthy(); ledger.start_case(logical, lease=lease)
                    with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                        if not logical.startswith("R-"):
                            artifact = await execute_arm_case(logical[0], logical[2:], sdk, data,
                                ensure_healthy=healthy, frame_case_id=logical)
                        else:
                            case = next(c for c in design._cases(bench.BACKEND_ROOT, "retrieval")["cases"] if c["id"] == logical[2:])
                            item = next(c for c in p["cases"] if c["case_id"] == case["id"])
                            event_index = 0
                            def observe(value):
                                nonlocal event_index
                                event_index += 1
                                base._write(self.root, f"observation-{logical}-{event_index:03d}.json", value)
                            def client(case_id, stage):
                                if case_id != logical[2:] or stage not in {"rewrite", "rerank"}:
                                    base._fail("prospective_rag_frame_changed")
                                return BoundClient(sdk, phase="rag_"+stage)
                            with context(Frame(logical, "turn-1")):
                                artifact = await execute_retrieval_case(case, client_factory=client,
                                    knowledge=SimpleNamespace(kb=SimpleNamespace(search_handler=data.knowledge_handler)),
                                    ensure_healthy=healthy, arm_order=item["arm_order"], on_observation=observe)
                    healthy()
                    if logical.startswith("R-"):
                        operations = [v for v in ledger.results()["operations"].values() if v["case_id"] == logical]
                        physical = [bench.read_json(base._path(self.root, f"physical-{v['attempt']:03d}.json"))
                                    for v in operations if v["output_sha256"] is not None]
                        usage = {"physical_requests": len(operations)}
                        for field in ("input_tokens", "output_tokens"):
                            values = [v["response"]["usage"][field] for v in physical]
                            usage[field] = (sum(values) if len(values) == len(operations)
                                            and all(type(v) is int for v in values) else None)
                        artifact["arms"]["B"]["record"]["usage"] = usage
                    artifact.update({"logical_case_id": logical, "protocol_sha256": bench.sha256(self.root/"protocol.json"),
                                     "evidence_kind": transport.kind, "data_proof_sha256": proof_hash})
                    name = f"case-{logical}.json"; base._write(self.root, name, artifact)
                    outcome = (artifact["execution_status"] if not logical.startswith("R-") else "success"
                        if all(v["record"]["status"] == "success" for v in artifact["arms"].values()) else "failed")
                    if outcome not in {"success", "partial", "failed"}: base._fail("prospective_case_status_invalid")
                    ledger.finish_case(logical, bench.sha256(self.root/name), outcome=outcome, lease=lease)
            except BaseException:
                if not (self.root/"worker-stop.json").exists():
                    base._write(self.root, "worker-stop.json", {"schema_version": "serveflow-prospective-stop-v1",
                        "reason": "execution_stopped_no_auto_resume", "recorded_at": base._now()})
                raise
            finally:
                try:
                    try:
                        if sdk is not None: await task_data._owned(sdk.close())
                    finally:
                        if data is not None: await task_data._owned(data.close())
                    if data is not None:
                        base._write(self.root, "cleanup-complete.json", {"owned_resources_closed": True,
                            "protocol_sha256": bench.sha256(self.root/"protocol.json")})
                except BaseException:
                    if not (self.root/"worker-stop.json").exists():
                        base._write(self.root, "worker-stop.json", {"schema_version": "serveflow-prospective-stop-v1",
                            "reason": "cleanup_failed_or_cancelled_no_auto_resume", "recorded_at": base._now()})
                    raise
        return self.report()

    def _review(self, value, p):
        bench._exact(value, {"logical_case_id", "artifact_sha256", "reviewer", "method", "dimensions",
                             "blocking_errors", "reading_burden", "reason"}, "prospective_review_fields")
        key = value["logical_case_id"]
        if type(key) is not str or key not in {r["case_id"] for r in p["plan"][:24]}:
            base._fail("prospective_review_identity")
        if value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{key}.json")):
            base._fail("prospective_review_binding")
        for field in ("reviewer", "reason"): bench._text(value[field], 4000)
        bench._strings(value["blocking_errors"])
        bench._exact(value["dimensions"], set(design.DIMENSIONS), "prospective_review_dimensions")
        if (any(type(v) is not bool for v in value["dimensions"].values())
                or value["method"] not in {"executor-rubric-review", "controlled-fixture", "independent-human-review"}
                or value["reading_burden"] not in {"none", "some", "heavy"}):
            base._fail("prospective_review_invalid")
        return value

    def review(self, value):
        p = self.verify(); self._review(value, p)
        if value["logical_case_id"] not in self._sealed(p)["cases"]:
            base._fail("prospective_review_requires_sealed_case")
        base._write(self.root, f"review-{value['logical_case_id']}.json", value)
        return value

    def report(self):
        p = self.verify(); state = self.ledger(p).status(); sealed = self._sealed(p)
        from evaluation.prospective_task_runtime import checks
        rows, samples, partial = [], [], []
        for planned in p["plan"]:
            key = planned["case_id"]
            row = {"logical_case_id": key, "status": "unknown" if state["active_case"] == key else "not_executed"}
            if key in sealed["cases"]:
                artifact = bench.read_json(base._path(self.root, f"case-{key}.json"))
                row.update(evidence_kind=artifact["evidence_kind"], artifact_sha256=sealed["cases"][key]["artifact_sha256"])
                if key.startswith("R-"):
                    samples.append(artifact); row["status"] = "retrieval_observed_not_answer_quality"
                else:
                    observed = checks(key[2:], artifact)
                    if observed != artifact["shared_checks"]: base._fail("prospective_checks_changed")
                    row["checks"] = observed
                    row["status"] = "pending_review" if all(observed["checks"].values()) and not observed["blockers"] else "failed"
                    row["tool_status"] = "pending_review" if observed["checks"].get("tool_task_correct") else "failed"
                    review_path = self.root/f"review-{key}.json"
                    if review_path.exists():
                        review = bench.read_json(base._path(self.root, review_path.name))
                        # Recheck imported evidence without ever overwriting it.
                        self._review(review, p)
                        row["review"] = review
                        row["status"] = "passed" if (row["status"] == "pending_review"
                            and all(review["dimensions"].values()) and not review["blocking_errors"]) else "failed"
                        row["tool_status"] = "passed" if (row["tool_status"] == "pending_review"
                            and review["dimensions"]["tool_permissions"]) else "failed"
            elif key.startswith("R-"):
                events = sorted(self.root.glob(f"observation-{key}-*.json"))
                if events:
                    partial.append({"logical_case_id": key, "completed_pair": False,
                        "events": [{"name": path.name, "sha256": bench.sha256(_safe_file(path)),
                                    "event": bench.read_json(path)} for path in events]})
            rows.append(row)
        from evaluation.prospective_retrieval_runtime import summarize_retrieval
        from evaluation.benchmark_metrics import aggregate_resources
        resources, delegates = [], []
        for operation_id, op in sealed["operations"].items():
            physical = (bench.read_json(base._path(self.root, f"physical-{op['attempt']:03d}.json"))
                        if op["output_sha256"] is not None else None)
            resources.append({"case_id": operation_id, "status": "success" if op["outcome"] == "received" else "failed",
                "latency_ms": physical["latency_ms"] if physical else None,
                "usage": {**(physical["response"]["usage"] if physical else {"input_tokens": None,"output_tokens": None}),
                          "physical_requests": 1}})
            delegates.append(bool(physical and physical.get("delegate_entered")))
        def metric(arm, predicate, status_key="status"):
            group = [r for r in rows[:24] if r["logical_case_id"].startswith(arm+"-")
                     and predicate(next(c for c in design._cases(bench.BACKEND_ROOT, "tasks")["cases"] if c["id"] == r["logical_case_id"][2:]))]
            return {"numerator": sum(r.get(status_key, r["status"]) == "passed" for r in group), "denominator": len(group),
                    "counts": {s: sum(r.get(status_key, r["status"]) == s for r in group) for s in ("passed", "failed", "pending_review", "unknown", "not_executed")}}
        task_metrics = {arm: {"correct_handling": metric(arm, lambda _: True),
            "supported_completion": metric(arm, lambda c: not c["expected"]["boundary"]),
            "tool_tasks": metric(arm, lambda c: c["expected"]["tool_task"], "tool_status"),
            "boundary_handling": metric(arm, lambda c: c["expected"]["boundary"])} for arm in ("A", "D")}
        paired = []
        for index in range(1, 13):
            left = next(r for r in rows if r["logical_case_id"] == f"A-p{index:02d}")
            right = next(r for r in rows if r["logical_case_id"] == f"D-p{index:02d}")
            paired.append({"case_id": f"p{index:02d}", "A": left["status"], "D": right["status"]})
        pair_counts = {name: sum(predicate(row) for row in paired) for name, predicate in {
            "both_passed": lambda r: r["A"] == r["D"] == "passed",
            "only_A_passed": lambda r: r["A"] == "passed" and r["D"] == "failed",
            "only_D_passed": lambda r: r["D"] == "passed" and r["A"] == "failed",
            "both_failed": lambda r: r["A"] == r["D"] == "failed",
            "unpaired_pending_unknown_or_unexecuted": lambda r: any(r[a] not in {"passed", "failed"} for a in ("A", "D"))}.items()}
        complete = (len(sealed["cases"]) == len(p["plan"]) and not state["pending"] and not state["paused"]
                    and not any((self.root/n).exists() for n in STOP_FILES) and (self.root/"cleanup-complete.json").exists())
        return {"schema_version": "serveflow-prospective-progress-v1", "protocol_sha256": bench.sha256(self.root/"protocol.json"),
            "authorization_present": (self.root/"authorization.json").exists(), "mathematical_ceiling": MAX_REQUESTS,
            "reserved_requests": state["attempts"], "closed_operations": len(sealed["operations"]),
            "sealed_physical": sum(v["output_sha256"] is not None for v in sealed["operations"].values()),
            "pending_physical": len(state["pending"]), "paused": state["paused"] or any((self.root/n).exists() for n in STOP_FILES),
            "known_delegate_entries": sum(delegates), "physical_resources": aggregate_resources(resources),
            "planned_jobs": len(p["plan"]), "completed_jobs": len(sealed["cases"]), "cases": rows,
            "retrieval": summarize_retrieval(design._cases(bench.BACKEND_ROOT, "retrieval")["cases"], samples),
            "retrieval_metrics_scope": "fully sealed pairs only; partial events are shown separately, never mistaken for unattempted or a complete comparison",
            "partial_retrieval_observations": partial, "task_metrics": task_metrics, "task_pairs": paired,
            "task_pair_counts": pair_counts,
            "task_delta_percentage_points": ((pair_counts["only_D_passed"]-pair_counts["only_A_passed"])*100/12
                if not pair_counts["unpaired_pending_unknown_or_unexecuted"] else None),
            "comparison_ready": complete and all(r["status"] in {"passed", "failed"} for r in rows[:24]),
            "quality_pass_claim": False, "semantic_reviews_imported": sum("review" in r for r in rows),
            "note": "progress only; model/native/handler success does not auto-pass the frozen semantic rubric"}


def supervise(trial):
    """One closed worker; reuse bounded process-group cleanup, no shell/retry."""
    from evaluation import task_supervisor as guard
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise guard.SupervisorError("supervisor_main_thread_posix_required")
    trial.verify(); trial._approval(bench.read_json(trial.root/"protocol.json"))
    root_fd = guard._open_root(trial.root)
    process, previous = None, {}
    state = {"spawning": True, "cancelled": False, "stopping": False}
    def cancelled(_number, _frame):
        state["cancelled"] = True
        if not state["spawning"] and not state["stopping"]:
            raise guard._ParentCancelled()
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.getsignal(number); signal.signal(number, cancelled)
        try:
            try:
                if not state["cancelled"]:
                    process = subprocess.Popen([sys.executable, "-B", "-u", "-m", "evaluation.prospective_trial", "_worker"],
                        cwd=bench.BACKEND_ROOT, shell=False, start_new_session=True, close_fds=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally: state["spawning"] = False
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, guard._ParentCancelled):
            state["cancelled"] = True
        if process is None:
            guard._write_stop(root_fd, reason="parent_cancelled", deadline_s=3600,
                result={"exit_code": None, "timed_out": False, "cancelled": True}, reaped=True, group_gone=True, pid=None)
            return 2
        timed_out, unexpected = False, False
        if not state["cancelled"]:
            try: return process.wait(timeout=3600)
            except subprocess.TimeoutExpired: timed_out = True
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, guard._ParentCancelled): state["cancelled"] = True
            except Exception: unexpected = True
        state["stopping"] = True
        code, reaped, gone = guard._terminate_group(process)
        guard._write_stop(root_fd, reason="supervisor_error" if unexpected else "deadline_exceeded" if timed_out else "parent_cancelled",
            deadline_s=3600, result={"exit_code": code, "timed_out": timed_out, "cancelled": state["cancelled"]},
            reaped=reaped, group_gone=gone, pid=process.pid)
        return 2
    finally:
        try:
            for number, handler in previous.items(): signal.signal(number, handler)
        finally: os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "status", "report", "review", "authorize", "run", "_worker"))
    parser.add_argument("--reference"); parser.add_argument("--max-requests", type=int)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--confirm-explicit-authorization", action="store_true")
    args = parser.parse_args(); trial = ProspectiveTrial()
    try:
        if args.command == "prepare": value = asyncio.run(trial.prepare())
        elif args.command == "authorize":
            value = trial.authorize(args.reference, max_requests=args.max_requests, confirmed=args.confirm_explicit_authorization)
        elif args.command == "run": raise SystemExit(supervise(trial))
        elif args.command == "_worker": value = asyncio.run(trial.execute())
        elif args.command in {"report", "status"}: value = trial.report()
        elif args.command == "review": value = trial.review(bench.read_json(args.review_file))
        else: value = trial.verify()
        print(json.dumps({"result": args.command, "reserved_requests": value.get("reserved_requests", 0),
            "proposed_upper_bound": MAX_REQUESTS, "authorized_by_prepare": 0,
            "completed_jobs": value.get("completed_jobs", 0)}, ensure_ascii=False))
    except Exception:
        parser.exit(2, "Prospective execution stopped; inspect private records, do not overwrite, retry or grant a new allowance automatically.\n")


if __name__ == "__main__": main()
