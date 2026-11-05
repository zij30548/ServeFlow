"""Five real HTTP turns with explicit, isolated memory resources.

The caller owns the counted SDK, source data, Redis server and execution clock.
Orders remain controlled fixture projections; this adapter does not send models,
open a database, authorize execution or claim production account isolation.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import hashlib
import re
import time
import threading
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import httpx

from core.concurrency import KeyedLocks
from core.order_evidence import extract_order_evidence, needs_order_evidence, resolve_order_id, source_bound_order_history
from demo_business.session import COOKIE_NAME, DemoSessionService
from evaluation import benchmark as bench, coverage_expansion as preparation, coverage_runtime as shared
from evaluation import task_data
from evaluation.expanded_observation_runtime import _bounded, _usage
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_evidence import CaptureTools, deterministic_checks, fixture_order_result
from evaluation.task_runtime import build_runtime, _result
from evaluation.task_transport import BoundClient, CURRENT, Frame, context
from memory.conversation_memory import MemoryManager


SCHEMA = "serveflow-coverage-conversation-artifact-v1"
_APP_LOCK = threading.Lock()
_GLOBALS = ("_orchestrator", "_memory", "_profile_updater", "_monitor", "_demo_store", "_demo_sessions")
_ACTUAL_CHECKS = ("trace_integrity", "tool_permissions", "trusted_identity", "tool_arguments",
                  "business_evidence", "auxiliary_disclosure_check", "required_tools")


def _fail(code):
    raise ValueError("coverage_conversation_" + code)


@asynccontextmanager
async def _exclusive_api():
    if not _APP_LOCK.acquire(blocking=False):
        _fail("api_busy")
    try:
        yield
    finally:
        _APP_LOCK.release()


class _Client(shared._AuditClient):
    async def create(self, **kwargs):
        from core.model_deadline import assert_model_active, overhead
        row = None
        try:
            async with overhead("admission"):
                self._healthy()
            assert_model_active()
            frame = CURRENT.get()
            if frame is None or frame.phase not in {"intent", "agent", "composer", "rag_rewrite", "rag_rerank"}:
                _fail("phase")
            row = {"case_id": frame.case_id, "turn_id": frame.turn_id, "phase": frame.phase, "agent": frame.agent,
                "sdk_arguments_sha256": canonical_payload_hash(shared._argument_json(kwargs)),
                "request_model": kwargs.get("model"), "response": None}
            self._observations.append(row)
            response = await self._request(**kwargs)
            async with overhead("admission"):
                self._healthy()
            row["response"] = shared.base._response_metadata(response)
            return response
        except BaseException as error:
            if row is not None and shared._ordinary_structure_error(error):
                row["structure_failure"] = {"error_type": type(error).__name__, "http_status": 200}
            elif not self._faults:
                self._faults.append(error)
            raise


def _source(case, data, frame_case_id):
    snapshot = shared.source_snapshot(source_root=getattr(data, "source_root", bench.BACKEND_ROOT))
    shared._data_contract(data, snapshot)
    groups = bench.read_json(snapshot["source_root"] / preparation.FILES[-1])["conversations"]
    original = next((row for row in groups if type(case) is dict and row["id"] == case.get("id")), None)
    if original is None or original != case or frame_case_id != "M-" + original["id"]:
        _fail("case_binding")
    return deepcopy(original), snapshot


@asynccontextmanager
async def memory_resources(sdk, data, *, namespace, ensure_healthy):
    """Default factory: injected Redis and two owned ephemeral Chroma stores.

    A custom async-context-manager factory has the same signature and yields a
    MemoryManager. Ordinary tests inject fakeredis and in-memory collections.
    Neither factory nor runtime closes the caller's SDK, Redis or CoverageData.
    """
    ensure_healthy()
    redis = getattr(data, "redis", None)
    knowledge = getattr(data, "_knowledge", None)
    client = getattr(knowledge, "client", None)
    collection = getattr(getattr(knowledge, "kb", None), "_collection", None)
    embedding = getattr(collection, "_embedding_function", None)
    if redis is None or client is None or not callable(embedding):
        _fail("memory_resources_required")
    names, collections = [], []
    try:
        for suffix in ("episodic", "profile"):
            name = namespace + "_" + suffix
            def create(name=name):
                value = client.create_collection(name=name, embedding_function=embedding,
                    metadata={"description": "Owned empty coverage conversation memory"})
                names.append(name)
                collections.append(value)
            await task_data._owned(asyncio.to_thread(create))
        memory = MemoryManager.__new__(MemoryManager)
        memory._redis, memory._request_locks = redis, KeyedLocks()
        memory._episodic, memory._profile = collections
        memory._client, memory._model = BoundClient(sdk, "memory"), shared.base.MODEL
        memory._coverage_resource_proof = {
            "redis": "explicit caller-injected isolated client; server owned by caller",
            "episodic_profile": "two owned ephemeral Chroma collections",
            "collections": list(names), "background_profile_updater": "disabled",
            "summary_threshold": memory.COMPRESS_AT, "sdk_owner": "caller-counted-shared-sdk"}
        yield memory
    finally:
        for name in reversed(names):
            await task_data._owned(asyncio.to_thread(client.delete_collection, name=name))


def _context_record(value):
    return {"recent_messages": [{"role": message.role.value, "content": message.content,
                "metadata": deepcopy(message.metadata), "timestamp": message.timestamp.isoformat()}
                for message in value.recent_messages],
            "reference_history": value.reference_history(), "state": deepcopy(value.state),
            "summary": value.summary, "user_profile": deepcopy(value.user_profile),
            "relevant_history": list(value.relevant_history)}


def _normal_history(value):
    return [{"role": item["role"], "content": item["content"]} for item in value["recent_messages"][-5:]] or None


def _keys(memory, user_id, conv_id):
    return {"working": memory._wm_key(user_id, conv_id), "state": memory._state_key(user_id, conv_id),
            "source_history": memory._source_history_key(user_id, conv_id),
            "summary": memory._summary_key(user_id, conv_id),
            "conversation_lease": "lock:conversation:" + memory._conversation_id(user_id, conv_id)}


async def _redis_record(memory, user_id, conv_id):
    keys = _keys(memory, user_id, conv_id)
    return {"keys": keys,
            "working_messages": await memory._redis.llen(keys["working"]),
            "source_events": await memory._redis.llen(keys["source_history"]),
            "working_ttl": await memory._redis.ttl(keys["working"]),
            "source_ttl": await memory._redis.ttl(keys["source_history"]),
            "summary_exists": bool(await memory._redis.exists(keys["summary"]))}


async def _cleanup(memory, user_id, conv_id, clients, requests):
    keys = [*_keys(memory, user_id, conv_id).values(),
            *(memory.receipt_key(user_id, conv_id, identity) for identity in clients),
            *("trace:" + identity for identity in requests)]
    if keys:
        await memory._redis.delete(*keys)
    for identity in requests:
        await memory._redis.lrem("trace:recent", 0, identity)
    return {"owned_keys": keys, "remaining_owned_keys": await memory._redis.exists(*keys),
            "shared_resources_closed": False}


@asynccontextmanager
async def _api_scope(api, orchestrator, memory, sessions, reader):
    from core import access
    saved = {name: getattr(api, name) for name in _GLOBALS}
    identity = access.identity
    try:
        # This dedicated worker's API transport tests synthetic demo sessions.
        # Host Bearer settings are explicitly outside this controlled identity
        # adapter; no model/user text can select the signed demo customer.
        access.identity = lambda _authorization: None
        api._orchestrator, api._memory = orchestrator, memory
        api._profile_updater, api._monitor = None, None
        api._demo_store, api._demo_sessions = SimpleNamespace(read_order=reader), sessions
        yield
    finally:
        for name, value in saved.items():
            setattr(api, name, value)
        access.identity = identity


def conversation_checks(case, artifact, *, source_root=bench.BACKEND_ROOT):
    """Read-only recomputation from frozen sources and original observations.

    No recorded check/score is trusted. Redis no longer needs to be running;
    the recorded read/commit/receipt chain is reconciled with native handlers.
    File/ledger integrity remains the sender's separate responsibility.
    """
    names = ("artifact_binding", "five_actual_http_turns", "trace_integrity", "tool_permissions",
             "trusted_identity", "tool_arguments", "business_evidence", "auxiliary_disclosure_check",
             "required_tools", "expected_status", "execution_eligible", "memory_chain", "source_provenance",
             "receipt_binding", "namespace_binding", "order_results_valid", "fresh_read",
             "policy_results_valid", "phase_binding", "model_response_structure_valid",
             "no_background_models", "cleanup_binding")
    invalid = {"checks": dict.fromkeys(names, False), "blockers": ["invalid_conversation_artifact"],
               "semantic_review_required": True, "quality_pass_claim": False}
    try:
        snapshot = shared.source_snapshot(source_root=source_root)
        source = bench.read_json(snapshot["source_root"] / preparation.FILES[-1])["conversations"]
        original = next((row for row in source if row["id"] == case["id"]), None)
        if original is None or original != case:
            return invalid
        identity, logical = "M-" + case["id"], case["id"]
        turns = artifact["turns"]
        if type(turns) is not list or len(turns) != 5:
            return invalid
        namespace, observations = artifact["namespace"], artifact["phase_observations"]
        user, conv = namespace["workspace_user_id"], namespace["conversation_id"]
        keys = _keys(MemoryManager.__new__(MemoryManager), user, conv)
        checks = dict.fromkeys(names, True)
        checks["artifact_binding"] = bool(artifact["schema_version"] == SCHEMA and artifact["suite"] == "conversations"
            and artifact["case_id"] == identity and artifact["logical_case_id"] == artifact["source_case_id"] == logical
            and artifact["case_sha256"] == canonical_payload_hash(case) and artifact["data_kind"] == shared.DATA_KIND
            and artifact["data_source_files"] == snapshot["source_files"] and artifact["quality_pass_claim"] is False
            and artifact["judge"] is None and artifact["semantic_review"] is None)
        checks["namespace_binding"] = bool(re.fullmatch(r"sf_demo_[0-9a-f]{32}", user)
            and re.fullmatch(r"sf_cov_mem_[0-9a-f]{32}", namespace["resource_id"]) and conv == "observed-conversation")
        previous, previous_sources, previous_after, requests, receipt_keys = [], [], None, [], []
        for number, (expected, turn) in enumerate(zip(case["turns"], turns), 1):
            output, body, captured = turn["result"], turn["http_response"], turn["tool_evidence"]
            before, after, redis = turn["memory_before"], turn["memory_after"], turn["redis"]
            client_id, request_id = f"observed-client-{number}", turn["request_id"]
            requests.append(request_id)
            if (type(request_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)
                    or request_id in requests[:-1]):
                return invalid
            accepted = turn["http_status"] == 200 and output["status"] in {"success", "partial"}
            checks["artifact_binding"] &= bool(turn["turn_id"] == f"turn-{number}" and turn["source_turn_id"] == expected["id"]
                and turn["question"] == expected["message"] and turn["client_request_id"] == client_id
                and request_id == output["request_id"] == body["request_id"]
                and body["response"] == output["response"] and body["status"] == output["status"])
            checks["execution_eligible"] &= accepted
            checks["expected_status"] &= accepted and (expected["expected"]["expected_status"] == "not-prescribed"
                or output["status"] == expected["expected"]["expected_status"])
            if not accepted and not (turn["http_status"] == 503 and output["status"] == "failed"):
                checks["artifact_binding"] = False
            normal = _normal_history(before)
            order = source_bound_order_history(before["reference_history"], normal)
            committed = previous + ([{"role": "user", "content": expected["message"]},
                {"role": "assistant", "content": body["response"]}] if accepted else [])
            event = {"version": 1, "source": "user_message", "role": "user", "content": expected["message"],
                "client_request_id": client_id, "request_id": request_id, "status": output["status"]}
            wanted_sources = previous_sources + ([event] if accepted else [])
            checks["memory_chain"] &= bool(
                [{"role": m["role"], "content": m["content"]} for m in before["recent_messages"]] == previous
                and [{"role": m["role"], "content": m["content"]} for m in after["recent_messages"]] == committed
                and turn["history"] == normal and turn["order_history"] == order
                and (previous_after is None or before == previous_after)
                and (accepted or after["state"] == before["state"]))
            checks["source_provenance"] &= bool(before["reference_history"] == previous_sources
                and after["reference_history"] == wanted_sources
                and (not accepted or all(m["metadata"].get("request_id") == client_id for m in after["recent_messages"][-2:])))
            receipt_key = "turn_receipt:" + MemoryManager._conversation_id(user, conv) + ":" + hashlib.sha256(client_id.encode()).hexdigest()
            receipt_keys.append(receipt_key)
            checks["receipt_binding"] &= bool(turn["receipt_key"] == receipt_key
                and turn["receipt"] == {"message": expected["message"], "response": body})
            checks["namespace_binding"] &= bool(redis["keys"] == keys and body["conv_id"] == conv
                and redis["working_messages"] == len(committed) and redis["source_events"] == len(wanted_sources))
            checks["no_background_models"] &= not after["summary"] and not after["user_profile"] and not redis["summary_exists"]
            replay = CaptureTools(logical, number)
            replay.calls = deepcopy(captured["calls"])
            reconciled = replay.reconcile(SimpleNamespace(**output))
            checks["trace_integrity"] &= not reconciled["blockers"] and reconciled["calls"] == captured["calls"]
            synthetic = {"id": logical, "input": {"message": expected["message"], "history": order,
                "customer_id": case["customer_id"]}, "expected": {"required_tools": expected["expected"]["required_tools"]}}
            observed = deterministic_checks(synthetic, SimpleNamespace(**output), captured, snapshot["fixture"])["checks"]
            for name in _ACTUAL_CHECKS:
                checks[name] &= bool(observed[name])
            selected, valid_reads = resolve_order_id(expected["message"], order), 0
            for call in captured["calls"]:
                checks["artifact_binding"] &= bool(call["case_id"] == logical and call["turn_id"] == number
                    and call["request_id"] == request_id)
                if call["name"] == "search_knowledge_base":
                    value = call["result"]
                    if (type(value) is not dict or value.get("success") is not True
                            or value.get("query") != expected["message"].strip()):
                        checks["policy_results_valid"] = False
                    else:
                        try:
                            shared._rows(value.get("results"), snapshot)
                        except (KeyError, ValueError, TypeError):
                            checks["policy_results_valid"] = False
                if call["name"] != "read_demo_order":
                    continue
                valid = bool(needs_order_evidence(expected["message"], order) and selected.status == "resolved"
                    and call["input"] == {"order_id": selected.order_id}
                    and canonical_payload_hash(call["result"]) == canonical_payload_hash(
                        fixture_order_result(snapshot["fixture"], case["customer_id"], selected.order_id)))
                if valid:
                    valid = extract_order_evidence(call["result"], expected_order_id=selected.order_id,
                        trusted_customer_id=case["customer_id"]).status in {"found", "not_found"}
                checks["order_results_valid"] &= valid
                valid_reads += bool(valid and call["tool_use_id"] and call["success"])
            if "read_demo_order" in expected["expected"]["required_tools"]:
                checks["fresh_read"] &= valid_reads > 0
            previous, previous_sources, previous_after = committed, wanted_sources, after
        expected_execution = ("failed" if any(t["result"]["status"] == "failed" for t in turns)
            else "success" if all(t["result"]["status"] == "success" for t in turns) else "partial")
        checks["artifact_binding"] &= artifact["answer"] == turns[-1]["result"]["response"] and artifact["execution_status"] == expected_execution
        checks["phase_binding"] = bool(type(observations) is list and len(observations) == artifact["physical_requests_observed"]
            and all(row["case_id"] == identity and row["turn_id"] in {f"turn-{n}" for n in range(1, 6)}
                and row["phase"] in {"intent", "agent", "composer", "rag_rewrite", "rag_rerank"}
                and row["request_model"] == shared.base.MODEL
                and re.fullmatch(r"[0-9a-f]{64}", row["sdk_arguments_sha256"])
                and (row["response"] is not None or row.get("structure_failure") in (
                    {"error_type": "JSONDecodeError", "http_status": 200},
                    {"error_type": "APIResponseValidationError", "http_status": 200}))
                for row in observations)
            and all(sum(row["phase"] == "intent" and row["turn_id"] == f"turn-{n}" for row in observations) == 1
                for n in range(1, 6)) and artifact["usage_observation"] == _usage(observations))
        checks["model_response_structure_valid"] = bool(all(row["response"] is not None
            and row.get("structure_failure") is None for row in observations))
        checks["no_background_models"] &= artifact["background_memory_model_requests"] == 0
        cleanup = artifact["memory_cleanup"]
        checks["cleanup_binding"] = bool(cleanup["remaining_owned_keys"] == 0 and cleanup["shared_resources_closed"] is False
            and cleanup["owned_keys"] == [*keys.values(), *receipt_keys, *("trace:" + identity for identity in requests)])
        checks = {key: bool(value) for key, value in checks.items()}
        return {"checks": checks, "blockers": [key for key, passed in checks.items() if not passed],
                "semantic_review_required": True, "quality_pass_claim": False}
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
        return invalid


async def execute_conversation(case, sdk, data, *, ensure_healthy, frame_case_id, memory_factory=None):
    """Execute original messages through /chat, reading each committed turn.

    The runner must enter eval_deadlines and supply its version/ledger health
    callback. Source labels, expected answers and references never enter model
    prompts. Any infrastructure/cancellation fault sticks across fallbacks.
    """
    case, snapshot = _source(case, data, frame_case_id)
    healthy, faults, close, reject = shared._scope(data, ensure_healthy)
    healthy()
    started, observations, turns = time.perf_counter(), [], []
    namespace = "sf_cov_mem_" + uuid.uuid4().hex
    conv_id, client_ids, request_ids = "observed-conversation", [], []
    sessions = DemoSessionService(uuid.uuid4().bytes + uuid.uuid4().bytes, ("http://127.0.0.1:5173",))
    token, claims = sessions.issue(case["customer_id"])
    user_id = sessions.workspace_id(claims)
    audited = _Client(sdk, healthy, observations, faults)

    async def reader(customer, order_id):
        healthy()
        value = await data.order_reader(customer, order_id)
        healthy()
        if canonical_payload_hash(value) != canonical_payload_hash(fixture_order_result(snapshot["fixture"], customer, order_id)):
            reject("order_source_changed")
        return value

    async def search(params, tool_context):
        healthy()
        value = await data.knowledge_handler(deepcopy(params), tool_context)
        healthy()
        try:
            return shared._rows(value, snapshot)
        except (KeyError, ValueError, TypeError):
            reject("policy_source_changed")

    resource_proof, cleanup = None, None
    try:
        async with _exclusive_api():
            healthy()
            factory = memory_factory or memory_resources
            async with factory(audited, data, namespace=namespace, ensure_healthy=healthy) as memory:
                if not isinstance(memory, MemoryManager):
                    reject("memory_manager_required")
                resource_proof = deepcopy(getattr(memory, "_coverage_resource_proof", {
                    "redis": "controlled factory injection", "episodic_profile": "controlled factory injection"}))
                initial = _context_record(await memory.get_context(user_id, conv_id))
                if any(initial[key] for key in initial):
                    reject("memory_not_empty")
                orchestrator, _manager = await build_runtime(audited,
                    SimpleNamespace(order_reader=reader, knowledge_handler=search))
                original_run, original_get = orchestrator.run, memory.get_context
                # The evaluation worker receives explicit, authorized config;
                # importing the HTTP entrypoint must not read application .env.
                with patch("dotenv.load_dotenv", return_value=False):
                    import api.main as api
                try:
                    async with _api_scope(api, orchestrator, memory, sessions, reader):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                                base_url="http://127.0.0.1:8816", trust_env=False) as http:
                            http.cookies.set(COOKIE_NAME, token)
                            previous, previous_sources = [], []
                            for number, source_turn in enumerate(case["turns"], 1):
                                healthy()
                                current = {"open": True, "contexts": [], "run": None}
                                capture = CaptureTools(case["id"], number, before_call=lambda _meta: healthy(),
                                    call_context=lambda role: context(agent=role))
                                capture.install(orchestrator)
                                client_id = f"observed-client-{number}"
                                client_ids.append(client_id)

                                async def get_context(user, conv, query="", *, current=current, source_turn=source_turn):
                                    healthy()
                                    if not current["open"]:
                                        reject("closed_turn")
                                    if (user, conv, query) != (user_id, conv_id, source_turn["message"]):
                                        reject("memory_request_binding")
                                    value = await original_get(user, conv, query)
                                    healthy()
                                    current["contexts"].append(_context_record(value))
                                    return value

                                async def run(request, *, current=current, source_turn=source_turn, capture=capture):
                                    healthy()
                                    if not current["open"]:
                                        reject("closed_turn")
                                    if len(current["contexts"]) != 1 or current["run"] is not None:
                                        reject("http_runtime_binding")
                                    before = current["contexts"][0]
                                    normal = _normal_history(before)
                                    order = source_bound_order_history(before["reference_history"], normal)
                                    if (request.message != source_turn["message"] or request.user_id != user_id
                                            or request.conv_id != conv_id or request.demo_customer_id != case["customer_id"]
                                            or request.history != normal or request.order_history != order):
                                        reject("http_request_binding")
                                    selected = resolve_order_id(request.message, request.business_history)
                                    capture.expected_order_id = selected.order_id if selected.status == "resolved" else None
                                    actual = await original_run(request)
                                    healthy()
                                    observed = capture.reconcile(actual)
                                    capture.assert_healthy()
                                    healthy()
                                    current["run"] = {"request_id": request.request_id, "result": _result(actual),
                                        "history": deepcopy(request.history), "order_history": deepcopy(request.order_history),
                                        "tool_evidence": observed, "selection": {"status": selected.status,
                                            "order_id": selected.order_id, "source": selected.source}}
                                    return actual

                                def admission(current=current, capture=capture):
                                    healthy()
                                    if not current["open"]:
                                        reject("closed_turn")
                                    capture.assert_send_allowed()

                                memory.get_context, orchestrator.run = get_context, run
                                try:
                                    with context(Frame(frame_case_id, f"turn-{number}", evidence_guard=admission)):
                                        response = await http.post("/chat", json={"message": source_turn["message"],
                                            "user_id": "controlled-client-untrusted", "conv_id": conv_id,
                                            "client_request_id": client_id}, headers={"Origin": sessions.origins[0],
                                            "X-ServeFlow-Demo-CSRF": sessions.csrf(claims)})
                                        if current["run"] is not None:
                                            request_ids.append(current["run"]["request_id"])
                                        healthy()
                                        if response.status_code not in {200, 503} or current["run"] is None:
                                            reject("http_failed")
                                        body, observed = response.json(), current["run"]
                                        accepted = response.status_code == 200 and body.get("status") in {"success", "partial"}
                                        if not accepted and not (response.status_code == 503 and body.get("status") == "failed"):
                                            reject("http_response_binding")
                                        receipt = await memory.get_turn_receipt(user_id, conv_id, client_id)
                                        after = _context_record(await original_get(user_id, conv_id, source_turn["message"]))
                                        redis = await _redis_record(memory, user_id, conv_id)
                                        healthy()
                                        before = current["contexts"][0]
                                        committed = previous + ([{"role": "user", "content": source_turn["message"]},
                                            {"role": "assistant", "content": body["response"]}] if accepted else [])
                                        source = {"version": 1, "source": "user_message", "role": "user",
                                            "content": source_turn["message"], "client_request_id": client_id,
                                            "request_id": observed["request_id"], "status": body["status"]}
                                        memory_checks = {
                                            "read_previous_committed_turns": [{"role": m["role"], "content": m["content"]}
                                                for m in before["recent_messages"]] == previous,
                                            "read_previous_original_user_events": before["reference_history"] == previous_sources,
                                            "committed_complete_turn": [{"role": m["role"], "content": m["content"]}
                                                for m in after["recent_messages"]] == committed,
                                            "original_user_provenance": after["reference_history"] == previous_sources + ([source] if accepted else []),
                                            "receipt_binding": receipt == {"message": source_turn["message"], "response": body},
                                            "request_binding": body["request_id"] == observed["request_id"],
                                            "redis_counts": redis["working_messages"] == len(committed) and redis["source_events"] == len(committed) // 2,
                                            "no_summary_or_profile": not after["summary"] and not after["user_profile"],
                                            "failure_did_not_change_state": accepted or after["state"] == before["state"]}
                                        if not all(memory_checks.values()):
                                            reject("memory_commit_binding")
                                        synthetic_case = {"id": case["id"], "input": {"message": source_turn["message"],
                                            "history": observed["order_history"], "customer_id": case["customer_id"]},
                                            "expected": {"required_tools": source_turn["expected"]["required_tools"]}}
                                        d = deterministic_checks(synthetic_case, SimpleNamespace(**observed["result"]),
                                            observed["tool_evidence"], snapshot["fixture"])
                                        gates = {key: bool(d["checks"][key]) for key in _ACTUAL_CHECKS}
                                        gates.update(memory_checks)
                                        gates["expected_status"] = accepted and (source_turn["expected"]["expected_status"] == "not-prescribed"
                                            or body["status"] == source_turn["expected"]["expected_status"])
                                        gates["execution_eligible"] = accepted
                                        turns.append({"turn_id": f"turn-{number}", "source_turn_id": source_turn["id"],
                                            "question": source_turn["message"], "client_request_id": client_id,
                                            **observed, "http_status": response.status_code,
                                            "http_response": body, "memory_before": before, "memory_after": after,
                                            "redis": redis, "receipt_key": memory.receipt_key(user_id, conv_id, client_id),
                                            "receipt": receipt, "checks": gates, "semantic_review": None})
                                        previous, previous_sources = committed, previous_sources + ([source] if accepted else [])
                                finally:
                                    current["open"] = False
                                    capture.restore()
                                    memory.get_context, orchestrator.run = original_get, original_run
                finally:
                    memory.get_context, orchestrator.run = original_get, original_run
                    cleanup = await task_data._owned(asyncio.create_task(_cleanup(memory, user_id, conv_id,
                        client_ids, request_ids)))
        healthy()
        artifact = {"schema_version": SCHEMA, "suite": "conversations", "case_id": frame_case_id,
            "logical_case_id": case["id"], "source_case_id": case["id"], "case_sha256": canonical_payload_hash(case),
            "data_kind": shared.DATA_KIND, "data_source_files": snapshot["source_files"],
            "order_data_mode": "controlled 32-order fixture projections; no MySQL connection",
            "identity_mode": "signed synthetic DemoSession; dedicated API Bearer bypass; not production account isolation",
            "namespace": {"resource_id": namespace, "workspace_user_id": user_id, "conversation_id": conv_id},
            "memory_resources": resource_proof, "memory_cleanup": cleanup, "turns": turns,
            "phase_observations": observations, "physical_requests_observed": len(observations),
            "usage_observation": _usage(observations), "background_memory_model_requests": 0,
            "execution_status": ("failed" if any(turn["result"]["status"] == "failed" for turn in turns)
                else "success" if all(turn["result"]["status"] == "success" for turn in turns) else "partial"),
            "answer": turns[-1]["result"]["response"], "wall_latency_ms": (time.perf_counter() - started) * 1000,
            "actor_path_latency_ms": sum(turn["result"]["latency_ms"] for turn in turns),
            "judge": None, "semantic_review": None, "quality_pass_claim": False,
            "tool_checks": None}
        artifact["tool_checks"] = conversation_checks(case, artifact, source_root=snapshot["source_root"])
        artifact["review_status"] = "pending_review" if not artifact["tool_checks"]["blockers"] else "failed"
        return _bounded(artifact)
    except BaseException:
        if faults:
            raise faults[0] from None
        raise
    finally:
        close()
