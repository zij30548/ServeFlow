"""One separately authorized coverage execution; ordinary preparation sends zero.

136 fixed positions share one new <=480 physical-request ledger. No old grant,
result, source guard, fixture or score is rewritten, resumed or supplemented.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, benchmark_metrics as metrics
from evaluation import coverage_expansion as preparation, intent_trial as base
from evaluation import task_data, task_recheck as history, task_trial as engine
from evaluation.coverage_budget import CoverageBudget, ROOT, LIMITS, MAX_REQUESTS, fixed_plan, job_specs
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError, validated_endpoint
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.quality_trial import _read_request
from evaluation.expanded_validation import DIMENSIONS


KIND = "coverage-execution-v1"
STOP_FILES = ("stopped.json", "worker-stop.json", "supervisor-stop.json")
PREFLIGHT = bench.BACKEND_ROOT / "data/benchmarks/coverage-expansion-preflight-v1"
CACHE_SNAPSHOT = PREFLIGHT / "before-preservation.json"
REDIS_OWNER_KEY = "serveflow:coverage-execution-v1:owner"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/apps/anthropic"
POLICY = {"authorization": "new protocol/root/genesis-bound <=480 grant; no old 480/96 reuse",
    "plan": "development then validation; alternating A/D tasks, paired local/enhanced RAG, five real HTTP/memory turns per conversation",
    "stop": "unknown send, timeout, cancel, source/data/auth/accounting failure stops without restart; ordinary answer/structure failure is retained",
    "payload": "offline first SDK capture; every validated full canonical and exact wire payload sealed before durable reserve/send",
    "review": "seven manual dimensions plus actual native/source checks; no Judge, automatic semantic pass or best-output selection",
    "conversation_review": "review all five turns separately; a group dimension is true only if every turn satisfies it; any material omission, fabricated claim or permission error fails the whole group; 136 positions and 168 user turns stay fixed denominators",
    "data": "explicit controlled order fixture and owned cached-only ephemeral Chroma; isolated Redis memory; no MySQL or application stores",
    "limits": "authored visible synthetic facts; one observation each; 2856 phase envelope is not an allowance; unknown usage stays unknown"}


def configuration(*, secret=False):
    """Only explicit process settings, never the user's .env or application DB."""
    endpoint = validated_endpoint(os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_ENDPOINT))
    if os.environ.get("ANTHROPIC_MODEL", base.MODEL) != base.MODEL:
        base._fail("coverage_model_changed")
    result = {"base_url": endpoint, "model": base.MODEL}
    if secret:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if type(key) is not str or not key or key.strip() != key or any(not 32 < ord(c) < 127 for c in key):
            base._fail("coverage_process_key_required")
        result["api_key"] = key
    return result


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def preserved_evidence():
    files = {}
    for path in sorted((bench.BACKEND_ROOT / "data/benchmarks").rglob("*")):
        if any(path.is_relative_to(folder) for folder in (ROOT, preparation.ROOT, PREFLIGHT)):
            continue
        if path.is_symlink():
            base._fail("coverage_prior_symlink")
        if path.is_file():
            files[str(path.absolute())] = bench.sha256(path)
    snapshot = bench.read_json(CACHE_SNAPSHOT)
    bench._exact(snapshot, {"original_caches", "old_evidence_files", "environment_file_sha256", "baseline_head"}, "coverage_preservation_fields")
    expected = {str(Path(name) if Path(name).is_absolute() else bench.BACKEND_ROOT / name): digest
        for name, digest in snapshot["old_evidence_files"].items()}
    if files != expected:
        base._fail("coverage_prior_evidence_changed")
    caches = snapshot.get("original_caches")
    if type(caches) is not dict or len(caches) != 5:
        base._fail("coverage_original_cache_snapshot")
    for name, digest in caches.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or bench.sha256(path) != digest:
            base._fail("coverage_original_cache_changed")
    if bench.sha256(bench.BACKEND_ROOT / ".env") != snapshot["environment_file_sha256"]:
        base._fail("coverage_application_environment_changed")
    return {"old_files": len(files), "files_manifest_sha256": canonical_payload_hash(files),
        "original_caches": caches, "environment_file_sha256": snapshot["environment_file_sha256"],
        "baseline_head": snapshot["baseline_head"], "snapshot_sha256": bench.sha256(CACHE_SNAPSHOT),
        "old_allowance_reused": False, "old_scores_changed": False}


def redis_binding():
    """Public identity of the explicitly owned instance; no URL credential log."""
    from urllib.parse import urlsplit
    parsed = urlsplit(os.environ.get("SERVEFLOW_EVAL_REDIS_URL", ""))
    run_id, owner = os.environ.get("SERVEFLOW_EVAL_REDIS_RUN_ID", ""), os.environ.get("SERVEFLOW_EVAL_REDIS_OWNER_MARKER", "")
    if (parsed.scheme != "redis" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port in {None, 6379} or parsed.query or parsed.fragment
            or not parsed.path.removeprefix("/").isdigit()
            or os.environ.get("SERVEFLOW_EVAL_REDIS_ISOLATED") != "true"
            or not 16 <= len(run_id) <= 128 or not run_id.isalnum()
            or not 16 <= len(owner) <= 128 or not owner.isalnum()):
        base._fail("coverage_isolated_redis_binding_required")
    return {"host": parsed.hostname, "port": parsed.port, "db": int(parsed.path[1:]),
        "run_id": run_id, "owner_key": REDIS_OWNER_KEY, "owner_marker": owner}


async def verify_redis(redis, expected):
    server = await asyncio.wait_for(redis.info("server"), 3)
    marker = await asyncio.wait_for(redis.get(expected["owner_key"]), 3)
    actual = server.get("run_id", server.get(b"run_id"))
    actual = actual.decode() if type(actual) is bytes else actual
    marker = marker.decode() if type(marker) is bytes else marker
    if actual != expected["run_id"] or marker != expected["owner_marker"]:
        base._fail("coverage_redis_owner_or_instance_changed")


def frozen_preparation():
    path = preparation.ROOT / "protocol.json"
    value = bench.read_json(base._path(preparation.ROOT, path.name))
    if (value.get("schema_version") != preparation.SCHEMA or value.get("authorized_model_requests") != 0
            or value.get("executed_model_requests") != 0 or value.get("live_sender_ready") is not False
            or value.get("data_sha256") != {p: bench.sha256(bench.BACKEND_ROOT / p) for p in preparation.FILES}
            or value.get("summary") != preparation.validate()):
        base._fail("coverage_preparation_changed")
    return {"protocol_sha256": bench.sha256(path), "data_sha256": value["data_sha256"], "summary": value["summary"]}


def source_cases():
    _, _, cases, conversations = preparation.load_validated()
    return {**{row["id"]: row for suite in ("tasks", "retrieval") for row in cases[suite]},
        **{row["id"]: row for row in conversations["conversations"]}}


async def capture_first_requests(config, cases):
    """Actual SDK bytes through MockTransport only, without any real key."""
    from evaluation.retrieval_trial import capture_rewrite
    intent, rewrites, shared = [], [], {}
    for spec in job_specs():
        case = cases[spec["source_case_id"]]
        if spec["suite"] == "conversations":
            projected = {"id": case["id"], "input": {"message": case["turns"][0]["message"], "history": []}}
        else:
            projected = {"id": case["id"], "input": preparation.model_input(case)}
        key = case["id"]
        if key not in shared:
            row = await (capture_rewrite(projected, config["base_url"]) if spec["suite"] == "retrieval"
                else base._capture(projected, config["base_url"]))
            row.pop("operation_id", None)
            shared[key] = row
        row = {**deepcopy(shared[key]), "source_case_id": key, "case_id": spec["case_id"]}
        (rewrites if spec["suite"] == "retrieval" else intent).append(row)
    return {"initial_intent_requests": intent, "initial_rewrite_requests": rewrites}


class CoverageTransport(PhaseDeadlineTaskHTTPTransport):
    """Reuse QualityHTTPTransport sealing and exact phase deadline protections."""
    def __init__(self, *args, **kwargs):
        self._redis_guard = kwargs.pop("redis_guard", None)
        super().__init__(*args, **kwargs)
        p = bench.read_json(base._path(self.root, "protocol.json"))
        self._jobs = {row["case_id"]: row for row in p["jobs"]}
        self._turns = {row["case_id"]: row["turn_ids"] for row in p["plan"]}
        self._rewrites = {row["case_id"]: row for row in p["initial_rewrite_requests"]}

    async def _prepare_admission(self):
        if self._redis_guard is not None:
            await self._redis_guard()
        # Keep the parent's complete source check last; no new await follows
        # that check before exact-payload recheck and durable reservation.
        return await super()._prepare_admission()

    def _validate(self, request, frame):
        spec = self._jobs.get(frame.case_id) if frame is not None else None
        if (spec is None or frame.turn_id not in self._turns[frame.case_id] or frame.phase == "judge"
                or spec["suite"] == "retrieval" and (frame.phase not in {"rag_rewrite", "rag_rerank"} or frame.agent is not None)):
            raise TrialTransportError("protocol")
        canonical, wire = super()._validate(request, frame)
        if spec["suite"] == "retrieval" and frame.phase == "rag_rewrite":
            expected = self._rewrites.get(frame.case_id)
            if expected is None or (canonical, wire) != (expected["payload_sha256"], expected["request_bytes_sha256"]):
                raise TrialTransportError("protocol")
        return canonical, wire


class CoverageExecutionTrial:
    protocol_schema = "serveflow-coverage-execution-protocol-v1"
    authorization_schema = "serveflow-coverage-execution-authorization-v1"

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            base._fail("coverage_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return CoverageBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), p["plan"],
            initialize=initialize, send_limit=send_limit)

    def expected(self):
        return {"schema_version": self.protocol_schema, "kind": KIND,
            "root_binding_sha256": base._root_hash(self.root), "git_head": git_head(),
            "preparation": frozen_preparation(), "source_inventory": preparation.inventory(),
            "prior_evidence": preserved_evidence(), "cases": source_cases(), "jobs": job_specs(), "plan": fixed_plan(),
            "configuration": configuration(), "public_parameters": preparation.public_parameters(),
            "redis_binding": redis_binding(),
            "effective_environment": history.effective_environment(), "dependencies": engine._dependencies(),
            "limits": LIMITS, "http_limits": PHASE_HTTP_LIMITS, "policy": POLICY,
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "judge_requests": 0, "external_business_requests": 0, "quality_pass_claim": False}

    async def prepare(self):
        if self.root.exists():
            base._fail("coverage_execution_root_exists")
        expected = self.expected()
        captured = await capture_first_requests(expected["configuration"], expected["cases"])
        if captured != await capture_first_requests(expected["configuration"], expected["cases"]) or expected != self.expected():
            base._fail("coverage_capture_or_context_changed")
        value = json.loads(json.dumps({**expected, **captured, "created_at": base._now()}))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        base._write(self.root, "protocol.json", value)
        self.ledger(value, initialize=True)
        return value

    def _sealed(self, p):
        sealed = self.ledger(p).results()
        protocol_hash = bench.sha256(self.root / "protocol.json")
        proof_path = self.root / "data-proof.json"
        proof_hash = bench.sha256(base._path(self.root, proof_path.name)) if proof_path.exists() else None
        if proof_hash is not None:
            proof = bench.read_json(proof_path)
            if (proof.get("kind") != "controlled-orders-and-isolated-chroma"
                    or proof.get("source_files") != p["preparation"]["data_sha256"]
                    or proof.get("database_connections") != 0
                    or bench.read_json(base._path(self.root, "data-proof-seal.json")) != {
                        "protocol_sha256": protocol_hash, "data_proof_sha256": proof_hash}):
                base._fail("coverage_data_proof_changed")
        for operation_id, op in sealed["operations"].items():
            request = _read_request(self.root, op["request_bytes_sha256"])
            if request["payload_sha256"] != op["payload_sha256"]:
                base._fail("coverage_payload_seal_changed")
            if op["output_sha256"] is None:
                continue
            name = f"physical-{op['attempt']:03d}.json"
            if bench.sha256(base._path(self.root, name)) != op["output_sha256"]:
                base._fail("coverage_physical_changed")
            row = bench.read_json(self.root / name)
            if any(row.get(k) != op[k] for k in ("case_id", "turn_id", "phase", "agent", "outcome", "payload_sha256", "request_bytes_sha256")) or row.get("operation_id") != operation_id:
                base._fail("coverage_physical_binding")
        for identity, item in sealed["cases"].items():
            name = f"case-{identity}.json"
            if bench.sha256(base._path(self.root, name)) != item["artifact_sha256"]:
                base._fail("coverage_case_changed")
            artifact = bench.read_json(self.root / name)
            if (artifact.get("frame_case_id") != identity or artifact.get("protocol_sha256") != protocol_hash
                    or artifact.get("data_proof_sha256") != proof_hash or proof_hash is None
                    or artifact.get("trial_kind") != KIND or artifact.get("evidence_kind") not in {"live-model", "controlled-transport"}):
                base._fail("coverage_case_binding")
        cleanup = self.root / "cleanup-complete.json"
        if cleanup.exists() and bench.read_json(cleanup) != {"owned_resources_closed": True, "protocol_sha256": protocol_hash}:
            base._fail("coverage_cleanup_changed")
        return sealed

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        expected = json.loads(json.dumps(self.expected()))
        bench._exact(p, set(expected) | {"initial_intent_requests", "initial_rewrite_requests", "created_at"}, "coverage_protocol_fields")
        if any(p[k] != v for k, v in expected.items()):
            base._fail("coverage_freeze_changed")
        for field, suite in (("initial_rewrite_requests", "retrieval"), ("initial_intent_requests", "intent")):
            planned = [s for s in p["jobs"] if (s["suite"] == "retrieval") == (suite == "retrieval")]
            values = p[field]
            if type(values) is not list or len(values) != len(planned):
                base._fail("coverage_initial_count")
            shared = {}
            for value, spec in zip(values, planned):
                bench._exact(value, {"case_id", "source_case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}, "coverage_initial_fields")
                case = p["cases"][spec["source_case_id"]]
                inp = {"message": case["turns"][0]["message"], "history": []} if spec["suite"] == "conversations" else preparation.model_input(case)
                wire = json.dumps(value["payload"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                if (value["case_id"] != spec["case_id"] or value["source_case_id"] != spec["source_case_id"]
                        or value["input_sha256"] != canonical_payload_hash(inp)
                        or value["payload_sha256"] != canonical_payload_hash(value["payload"])
                        or value["request_bytes_sha256"] != hashlib.sha256(wire).hexdigest()):
                    base._fail("coverage_initial_binding")
                same = {k: v for k, v in value.items() if k != "case_id"}
                if spec["source_case_id"] in shared and shared[spec["source_case_id"]] != same:
                    base._fail("coverage_arm_first_payload_mismatch")
                shared[spec["source_case_id"]] = same
        self.ledger(p).status()
        self._sealed(p)
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p = self.verify()
        if (confirmed is not True or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference) > 500):
            base._fail("coverage_explicit_authorization_required")
        self._unstarted(p)
        value = {"schema_version": self.authorization_schema, "reference": reference, "recorded_at": base._now(),
            "protocol_sha256": bench.sha256(self.root / "protocol.json"), "root_binding_sha256": base._root_hash(self.root),
            "initial_chain_hash": self.ledger(p).initial_chain_hash, "max_requests": max_requests, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256", "initial_chain_hash", "max_requests", "plan"}, "coverage_authorization_fields")
        if (value["schema_version"] != self.authorization_schema or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root) or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash
                or value["plan"] != p["plan"] or type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= MAX_REQUESTS):
            base._fail("coverage_authorization_changed")
        return value

    def _unstarted(self, p):
        state = self.ledger(p).status()
        if (any((self.root / name).exists() for name in (*STOP_FILES, "worker-started.json", "data-proof.json"))
                or state["attempts"] or state["active_case"] or state["completed_cases"] or state["pending"]):
            base._fail("coverage_no_restart_or_resume")

    async def _job(self, spec, case, sdk, data, healthy):
        from evaluation.coverage_runtime import execute_task, run_retrieval
        if spec["suite"] == "tasks":
            return await execute_task(case, spec["arm"], sdk, data, ensure_healthy=healthy, frame_case_id=spec["case_id"])
        if spec["suite"] == "retrieval":
            return await run_retrieval(case, sdk, data, ensure_healthy=healthy, frame_case_id=spec["case_id"])
        from evaluation.coverage_conversation_runtime import execute_conversation
        return await execute_conversation(case, sdk, data, ensure_healthy=healthy, frame_case_id=spec["case_id"])

    async def execute(self, *, delegate_factory=None, controlled_data=None, redis=None):
        p = self.verify()
        approval = self._approval(p)  # Before credentials, resources, SDK, transport.
        if (delegate_factory is None) != (controlled_data is None) or (delegate_factory is None) != (self.root == ROOT):
            base._fail("coverage_live_controlled_root_mismatch")
        self._unstarted(p)
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        data, sdk, http, transport, owned_redis = controlled_data, None, None, None, None
        with ledger.exclusive_run() as lease:
            try:
                base._write(self.root, "worker-started.json", {"protocol_sha256": bench.sha256(self.root / "protocol.json"), "recorded_at": base._now()})
                config = configuration(secret=delegate_factory is None)
                if data is None:
                    from evaluation.coverage_runtime import build_data
                    data = await build_data()
                if redis is None and delegate_factory is None:
                    value = os.environ.get("SERVEFLOW_EVAL_REDIS_URL", "")
                    from redis.asyncio import Redis
                    owned_redis = Redis.from_url(value, socket_connect_timeout=3, socket_timeout=3, decode_responses=True)
                    await asyncio.wait_for(owned_redis.ping(), 3)
                    redis = owned_redis
                if delegate_factory is None:
                    await verify_redis(redis, p["redis_binding"])
                if redis is not None:
                    data.redis = redis
                data.assert_healthy()
                base._write(self.root, "data-proof.json", data.proof)
                protocol_hash, proof_hash = bench.sha256(self.root / "protocol.json"), bench.sha256(self.root / "data-proof.json")
                approval_hash = bench.sha256(self.root / "authorization.json")
                base._write(self.root, "data-proof-seal.json", {"protocol_sha256": protocol_hash, "data_proof_sha256": proof_hash})
                faults = []
                def healthy():
                    try:
                        if faults or transport is not None and transport.fault or any((self.root / n).exists() for n in STOP_FILES):
                            base._fail("coverage_context_faulted")
                        self.verify()
                        self._approval(p)
                        if (bench.sha256(self.root / "authorization.json") != approval_hash
                                or bench.sha256(self.root / "data-proof.json") != proof_hash
                                or canonical_payload_hash(data.proof) != canonical_payload_hash(bench.read_json(self.root / "data-proof.json"))):
                            base._fail("coverage_run_identity_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise
                async def redis_guard():
                    await verify_redis(redis, p["redis_binding"])
                transport = CoverageTransport(config["base_url"], self.root, ledger, lease, healthy,
                    delegate_factory=delegate_factory, redis_guard=redis_guard if delegate_factory is None else None)
                import httpx
                from anthropic import AsyncAnthropic
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False, trust_env=False, headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"], max_retries=0, timeout=SDK_TIMEOUT_S, http_client=http)
                for spec in p["jobs"]:
                    healthy()
                    if delegate_factory is None:
                        await verify_redis(redis, p["redis_binding"])
                    ledger.start_case(spec["case_id"], lease=lease)
                    with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                        artifact = await self._job(spec, p["cases"][spec["source_case_id"]], sdk, data, healthy)
                    healthy()
                    artifact.update(frame_case_id=spec["case_id"], trial_kind=KIND, protocol_sha256=protocol_hash,
                        data_proof_sha256=proof_hash, evidence_kind=transport.kind, quality_pass_claim=False)
                    outcome = artifact.get("execution_status", artifact.get("status"))
                    if outcome not in {"success", "partial", "failed"}:
                        base._fail("coverage_artifact_status")
                    base._write(self.root, f"case-{spec['case_id']}.json", artifact)
                    ledger.finish_case(spec["case_id"], bench.sha256(self.root / f"case-{spec['case_id']}.json"), outcome=outcome, lease=lease)
            except BaseException:
                if not (self.root / "worker-stop.json").exists():
                    base._write(self.root, "worker-stop.json", {"reason": "execution_stopped_no_auto_resume", "recorded_at": base._now()})
                raise
            finally:
                cleanup_errors = []
                closers = [sdk.close] if sdk is not None else [http.aclose] if http is not None else []
                closers += ([owned_redis.aclose] if owned_redis is not None else []) + ([data.close] if data is not None else [])
                for close in closers:
                    try:
                        await task_data._owned(close())
                    except BaseException as error:
                        cleanup_errors.append(error)
                if cleanup_errors:
                    if not (self.root / "worker-stop.json").exists():
                        base._write(self.root, "worker-stop.json", {"reason": "cleanup_failed_no_auto_resume", "recorded_at": base._now()})
                    raise cleanup_errors[0]
                if data is not None:
                    base._write(self.root, "cleanup-complete.json", {"owned_resources_closed": True, "protocol_sha256": bench.sha256(self.root / "protocol.json")})
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("coverage_fixed_worker_required")
        p = self.verify(); self._approval(p); self._unstarted(p)
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.coverage_execution", "_worker"], root=ROOT, deadline_s=3600)

    def review(self, value):
        p = self.verify()
        self._validate_review(value, p)
        if value["frame_case_id"] not in self._sealed(p)["cases"]:
            base._fail("coverage_review_unsealed")
        base._write(self.root, f"review-{value['frame_case_id']}.json", value)
        return value

    def _validate_review(self, value, p):
        bench._exact(value, {"frame_case_id", "artifact_sha256", "reviewer", "method", "dimensions", "blocking_errors", "reading_burden", "reason"}, "coverage_review_fields")
        spec = next((s for s in p["jobs"] if s["case_id"] == value["frame_case_id"] and s["suite"] != "retrieval"), None)
        if spec is None or value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{value['frame_case_id']}.json")):
            base._fail("coverage_review_binding")
        bench._exact(value["dimensions"], set(DIMENSIONS), "coverage_review_dimensions")
        if (any(type(v) is not bool for v in value["dimensions"].values()) or value["reading_burden"] not in {"none", "some", "heavy"}
                or value["method"] not in {"executor-rubric-review", "controlled-fixture", "independent-human-review"}):
            base._fail("coverage_review_value")
        artifact = bench.read_json(self.root / f"case-{value['frame_case_id']}.json")
        if (artifact["evidence_kind"] == "controlled-transport") != (value["method"] == "controlled-fixture"):
            base._fail("coverage_review_evidence_kind")
        for field in ("reviewer", "reason"):
            bench._text(value[field], 4000)
        bench._strings(value["blocking_errors"])

    def report(self):
        p = self.verify(); state = self.ledger(p).status(); sealed = self._sealed(p)
        rows, resources, retrieval = [], [], {"baseline": [], "enhanced": []}
        for spec in p["jobs"]:
            identity, artifact = spec["case_id"], None
            case = p["cases"][spec["source_case_id"]]
            item = {**spec, "tags": case["tags"], "status": "unknown" if state["active_case"] == identity else "not_executed",
                "tool_task": case["expected"]["tool_task"] if spec["suite"] == "tasks" else None,
                "boundary": case["expected"]["boundary"] if spec["suite"] == "tasks" else None}
            if identity in sealed["cases"]:
                artifact = bench.read_json(self.root / f"case-{identity}.json")
                ops = [op for op in sealed["operations"].values() if op["case_id"] == identity]
                physical_ok = all(op["outcome"] == "received" and op["output_sha256"] for op in ops)
                if spec["suite"] == "retrieval":
                    physical_ok = physical_ok and Counter(op["phase"] for op in ops) == {"rag_rewrite": 1, "rag_rerank": 1}
                item.update(execution_status=artifact.get("execution_status", artifact.get("status")), physical_structure_valid=bool(physical_ok))
                if spec["suite"] == "tasks":
                    from evaluation.coverage_runtime import task_checks
                    observed = task_checks(p["cases"][spec["source_case_id"]], spec["arm"], artifact)
                    item["checks"] = observed
                    eligible = all(observed["checks"].values()) and not observed["blockers"] and physical_ok
                elif spec["suite"] == "conversations":
                    from evaluation.coverage_conversation_runtime import conversation_checks
                    observed = conversation_checks(p["cases"][spec["source_case_id"]], artifact)
                    item["checks"] = observed
                    flags = observed.get("checks", observed) if type(observed) is dict else {}
                    eligible = bool(flags) and all(type(v) is bool and v for v in flags.values()) and not observed.get("blockers") and physical_ok
                else:
                    eligible = artifact.get("status", artifact.get("execution_status")) == "success" and physical_ok
                    case = p["cases"][spec["source_case_id"]]
                    for arm in retrieval:
                        recorded = artifact[arm]
                        valid = arm == "baseline" or physical_ok
                        retrieval[arm].append(preparation.retrieval_metrics(case,
                            recorded["rows"] if valid else [], recorded["status"] if valid else "failed"))
                item["status"] = "recorded" if spec["suite"] == "retrieval" and eligible else "pending_review" if eligible else "failed"
                item["execution_eligible_for_review"] = bool(eligible)
                review_path = self.root / f"review-{identity}.json"
                if spec["suite"] != "retrieval" and review_path.exists():
                    review = bench.read_json(base._path(self.root, review_path.name)); self._validate_review(review, p)
                    item.update(review=review, status="passed" if eligible and all(review["dimensions"].values()) and not review["blocking_errors"] else "failed")
            elif spec["suite"] == "retrieval":
                case = p["cases"][spec["source_case_id"]]
                for arm in retrieval:
                    retrieval[arm].append(preparation.retrieval_metrics(case, [], "unknown" if state["active_case"] == identity else "not_run"))
            rows.append(item)
        resource_operations = set()
        for operation_id, op in sealed["operations"].items():
            resource_operations.add(operation_id)
            physical = bench.read_json(self.root / f"physical-{op['attempt']:03d}.json") if op["output_sha256"] else None
            resources.append({"case_id": str(op["attempt"]), "status": "unknown" if operation_id in state["pending"] else "success" if op["outcome"] == "received" else "failed",
                "latency_ms": physical["latency_ms"] if physical else None, "usage": {**(physical["response"]["usage"] if physical else {"input_tokens": None, "output_tokens": None}), "physical_requests": 1}})
        resources.extend({"case_id": "pending:" + key, "status": "unknown", "latency_ms": None,
            "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 1}} for key in state["pending"] if key not in resource_operations)
        if len(resources) != state["attempts"]:
            base._fail("coverage_physical_resource_count")
        groups = {f"{split}/{suite}/{arm}": {"planned": len(selected), "statuses": dict(Counter(r["status"] for r in selected)),
                "correct_handling": {"numerator": sum(r["status"] == "passed" for r in selected), "denominator": len(selected)} if suite != "retrieval" else None,
                "semantic_reviews": sum("review" in r for r in selected),
                "dimensions": {dimension: {"numerator": sum(bool(r.get("execution_eligible_for_review") and r.get("review", {}).get("dimensions", {}).get(dimension)) for r in selected),
                    "denominator": len(selected)} for dimension in DIMENSIONS} if suite != "retrieval" else None}
            for split in ("development", "validation") for suite in ("tasks", "retrieval", "conversations") for arm in ("A", "D", "paired-vector-enhanced", "current")
            if (selected := [r for r in rows if (r["split"], r["suite"], r["arm"]) == (split, suite, arm)])}
        retrieval_groups = {arm: {f"{split}/{kind}": {"planned": len(selected),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "document_hits_at_3": sum(r["document_hit_at_3"] for r in selected),
                "all_required_facts_at_3": sum(r["all_required_facts_at_3"] for r in selected),
                "macro_required_fact_recall_at_3": sum(r["required_fact_recall_at_3"] or 0 for r in selected) / len(selected) if kind != "out_of_scope" else None,
                "healthy_empty_rejections": sum(r["healthy_empty_rejection"] for r in selected),
                "out_of_scope_false_hits": sum(r["out_of_scope_false_hit"] for r in selected)}
            for split in ("development", "validation") for kind in ("answerable", "insufficient", "out_of_scope")
            if (selected := [r for r in results if (r["split"], r["kind"]) == (split, kind)])}
            for arm, results in retrieval.items()}
        paired = [{"case_id": case["id"], "split": case["split"], "A": next(r["status"] for r in rows if r["case_id"] == "A-" + case["id"]),
            "D": next(r["status"] for r in rows if r["case_id"] == "D-" + case["id"])} for case in p["cases"].values() if case["id"].startswith("CT-")]
        authorized = self._approval(p)["max_requests"] if (self.root / "authorization.json").exists() else 0
        return {"schema_version": "serveflow-coverage-execution-progress-v1", "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "authorized_requests": authorized, "reserved_requests": state["attempts"], "maximum_requests": MAX_REQUESTS,
            "planned_positions": 136, "planned_user_turns": 168, "completed_positions": len(sealed["cases"]),
            "pending_physical": len(state["pending"]), "paused": state["paused"] or any((self.root / n).exists() for n in STOP_FILES),
            "rows": rows, "groups": groups, "task_pairs": paired, "retrieval": retrieval, "retrieval_groups": retrieval_groups,
            "physical_resources": metrics.aggregate_resources(resources), "quality_pass_claim": False,
            "semantic_reviews": sum("review" in r for r in rows),
            "execution_complete": len(sealed["cases"]) == 136 and not state["pending"] and not state["paused"]
                and not any((self.root / n).exists() for n in STOP_FILES) and (self.root / "cleanup-complete.json").exists()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "authorize", "run", "_worker", "report", "review"))
    parser.add_argument("--reference"); parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-explicit-authorization", action="store_true"); parser.add_argument("--review-file", type=Path)
    args, trial = parser.parse_args(argv), CoverageExecutionTrial()
    try:
        if args.command == "prepare": value = asyncio.run(trial.prepare())
        elif args.command == "authorize": value = trial.authorize(args.reference, max_requests=args.max_requests, confirmed=args.confirm_explicit_authorization)
        elif args.command == "_worker": value = asyncio.run(trial.execute())
        elif args.command == "run":
            value = trial.run()
            if value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled"):
                base._fail("coverage_supervised_worker_failed")
        elif args.command == "review": value = trial.review(bench.read_json(args.review_file))
        else: value = trial.report() if args.command == "report" else trial.verify()
        print(json.dumps(value if args.command == "report" else {"result": args.command, "reserved_requests": value.get("reserved_requests", 0)}, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"error": "coverage_execution_stopped_inspect_private_evidence", "no_auto_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
