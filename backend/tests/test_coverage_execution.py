"""Counted SDK MockTransport checks; no real key, Redis, DB, index or sender."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from core.model_deadline import await_model

from evaluation import coverage_execution as module, coverage_expansion as prep
from evaluation.coverage_budget import fixed_plan, job_specs
from evaluation.task_transport import Frame, context


H = "a" * 64
_captures = {}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("coverage execution tests are controlled only")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == ".env" or path.suffix in {".key", ".pem"}:
            denied()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "never-use-real-key")


@pytest_asyncio.fixture
async def trial(tmp_path, monkeypatch):
    cases = module.source_cases()
    files = {path: prep.bench.sha256(prep.bench.BACKEND_ROOT / path) for path in prep.FILES}
    state = {"sources": {"controlled-runtime.py": H}, "head": "b" * 40,
        "prior": {"snapshot_sha256": H, "old_files": 4277, "old_allowance_reused": False},
        "config": {"base_url": module.validated_endpoint(module.DEFAULT_ENDPOINT), "model": module.base.MODEL}}
    monkeypatch.setattr(module, "source_cases", lambda: deepcopy(cases))
    monkeypatch.setattr(module, "git_head", lambda: state["head"])
    monkeypatch.setattr(module, "frozen_preparation", lambda: {"protocol_sha256": H, "data_sha256": files, "summary": {"controlled": True}})
    monkeypatch.setattr(module, "preserved_evidence", lambda: deepcopy(state["prior"]))
    monkeypatch.setattr(prep, "inventory", lambda: deepcopy(state["sources"]))
    monkeypatch.setattr(module.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(module.engine, "_dependencies", lambda: {"controlled": "offline"})
    monkeypatch.setattr(module, "redis_binding", lambda: {"host": "127.0.0.1", "port": 16389, "db": 0,
        "run_id": "r" * 40, "owner_key": module.REDIS_OWNER_KEY, "owner_marker": "m" * 32})
    original_capture = module.capture_first_requests
    async def cached(config, values):
        key = module.canonical_payload_hash({"configuration": config, "cases": values})
        if key not in _captures:
            _captures[key] = await original_capture(config, values)
        return deepcopy(_captures[key])
    monkeypatch.setattr(module, "capture_first_requests", cached)
    value = module.CoverageExecutionTrial(tmp_path / "trial")
    await value.prepare()
    value.controlled_state, value.controlled_cases = state, cases
    return value


def data_for(trial):
    p = trial.verify()
    return SimpleNamespace(proof={"kind": "controlled-orders-and-isolated-chroma",
        "source_files": p["preparation"]["data_sha256"], "database_connections": 0},
        assert_healthy=lambda: None, close=AsyncMock())


def delegate(*, malformed=False, fail_status=None, captured=None):
    def factory(frame):
        async def send(request):
            if captured is not None:
                captured.append((frame, json.loads(request.content)))
            if fail_status:
                return httpx.Response(fail_status, json={"error": {"message": "private-do-not-retain"}})
            if malformed:
                return httpx.Response(200, json={"id": "broken-but-received", "type": "message", "role": "assistant", "model": module.base.MODEL, "content": [{"type": "text", "text": 7}], "stop_reason": "end_turn", "usage": {}})
            return httpx.Response(200, json={"id": "controlled-once", "type": "message", "role": "assistant", "model": module.base.MODEL,
                "content": [{"type": "text", "text": "controlled response"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1}})
        return httpx.MockTransport(send)
    return factory


async def synthetic_job(trial, spec, case, sdk, data, healthy):
    p = module.bench.read_json(trial.root / "protocol.json")
    for number in range(1, 6 if spec["suite"] == "conversations" else 2):
        turn = f"turn-{number}"
        if spec["suite"] == "retrieval":
            record = next(r for r in p["initial_rewrite_requests"] if r["case_id"] == spec["case_id"])
            phases = [("rag_rewrite", record["payload"]), ("rag_rerank", {"model": module.base.MODEL, "max_tokens": 2048, "temperature": 0., "messages": [{"role": "user", "content": "controlled rerank"}], "thinking": {"type": "disabled"}})]
        else:
            record = next(r for r in p["initial_intent_requests"] if r["case_id"] == spec["case_id"])
            phases = [("intent", deepcopy(record["payload"]))]
            if number > 1:
                phases[0][1]["messages"] = [{"role": "user", "content": f"controlled dynamic history turn {number}"}]
        for phase, payload in phases:
            payload = deepcopy(payload)
            thinking = payload.pop("thinking")
            with context(Frame(spec["case_id"], turn, phase, None)):
                try:
                    await await_model(sdk.messages.create(**payload, extra_body={"thinking": thinking}), timeout_s=30 if phase == "rag_rerank" else 10 if phase == "rag_rewrite" else 8)
                except module.TrialTransportError:
                    raise
                except Exception:
                    # An ordinary HTTP200 SDK/response structure failure is
                    # recorded as a failed position, without another send.
                    healthy()
    result = {"execution_status": "failed", "status": "failed", "tool_checks": {"checks": {"controlled_no_quality": False}, "blockers": ["controlled"]}}
    if spec["suite"] == "retrieval":
        result.update(baseline={"rows": [], "status": "failed"}, enhanced={"rows": [], "status": "failed"})
    return result


@pytest.fixture
def simplified_report(monkeypatch):
    from evaluation import coverage_runtime, coverage_conversation_runtime
    monkeypatch.setattr(coverage_runtime, "task_checks", lambda *a, **k: {"checks": {"controlled_no_quality": False}, "blockers": ["controlled"]})
    monkeypatch.setattr(coverage_conversation_runtime, "conversation_checks", lambda *a, **k: {"checks": {"controlled_no_quality": False}, "blockers": ["controlled"]}, raising=False)
    monkeypatch.setattr(prep, "retrieval_metrics", lambda case, rows, status, **k: {"case_id": case["id"], "split": case["split"], "kind": case["expected"]["kind"], "status": status,
        "document_hit_at_3": False, "all_required_facts_at_3": False, "required_fact_recall_at_3": 0., "healthy_empty_rejection": False, "out_of_scope_false_hit": False})


def test_redis_config_requires_frozen_instance_marker_and_explicit_isolation(monkeypatch):
    for name in ("SERVEFLOW_EVAL_REDIS_URL", "SERVEFLOW_EVAL_REDIS_RUN_ID", "SERVEFLOW_EVAL_REDIS_OWNER_MARKER", "SERVEFLOW_EVAL_REDIS_ISOLATED"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(module.base.TrialError): module.redis_binding()
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_URL", "redis://127.0.0.1:16389/0")
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_RUN_ID", "r" * 40)
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_OWNER_MARKER", "m" * 32)
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_ISOLATED", "true")
    value = module.redis_binding()
    assert value["port"] == 16389 and value["db"] == 0
    assert "api_key" not in value and "url" not in value
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_URL", "redis://127.0.0.1:6379/15")
    with pytest.raises(module.base.TrialError): module.redis_binding()


@pytest.mark.asyncio
async def test_redis_run_id_and_owner_cannot_be_replaced_by_same_db_number():
    expected = {"run_id": "owned-run", "owner_key": module.REDIS_OWNER_KEY, "owner_marker": "owned-marker"}
    redis = SimpleNamespace(info=AsyncMock(return_value={"run_id": "owned-run"}), get=AsyncMock(return_value="owned-marker"))
    await module.verify_redis(redis, expected)
    redis.info.return_value = {"run_id": "different-run"}
    with pytest.raises(module.base.TrialError): await module.verify_redis(redis, expected)
    redis.info.return_value = {"run_id": "owned-run"}; redis.get.return_value = "wrong-owner"
    with pytest.raises(module.base.TrialError): await module.verify_redis(redis, expected)


@pytest.mark.asyncio
async def test_zero_call_freeze_binds_all136_first_sdk_payloads_and_arm_equality(trial):
    p = trial.verify()
    assert len(p["initial_intent_requests"]) == 88 and len(p["initial_rewrite_requests"]) == 48
    assert p["authorized_model_requests"] == 0 and trial.ledger(p).status()["attempts"] == 0
    by_source = {}
    for row in p["initial_intent_requests"]:
        comparable = {k: v for k, v in row.items() if k != "case_id"}
        assert row["source_case_id"] not in by_source or by_source[row["source_case_id"]] == comparable
        by_source[row["source_case_id"]] = comparable
        assert row["payload_sha256"] == module.canonical_payload_hash(row["payload"])
        text = json.dumps(row["payload"])
        assert "expected_status" not in text and "required_fact_ids" not in text and "demo-customer-a" not in text
    with pytest.raises(module.base.TrialError): await trial.prepare()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 481, True, "480", 480.0])
async def test_bad_or_unconfirmed_grants_never_create_authorization(trial, limit):
    with pytest.raises(module.base.TrialError): trial.authorize("user-new-batch", max_requests=limit, confirmed=True)
    assert not (trial.root / "authorization.json").exists()


@pytest.mark.asyncio
async def test_approval_precedes_secret_resources_and_transport(trial, monkeypatch):
    touched = []
    monkeypatch.setattr(module, "configuration", lambda **kwargs: touched.append(kwargs) or trial.controlled_state["config"])
    with pytest.raises((FileNotFoundError, module.base.TrialError)):
        await trial.execute(delegate_factory=delegate(), controlled_data=SimpleNamespace())
    assert touched == [{}]  # Public verify only, never secret=True.
    assert not (trial.root / "worker-started.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["sources", "head", "prior"])
async def test_changed_source_head_or_old_evidence_rejects_before_any_send(trial, field):
    trial.authorize("new480", max_requests=480, confirmed=True)
    trial.controlled_state[field] = {} if field != "head" else "c" * 40
    with pytest.raises(module.base.TrialError): trial.verify()


@pytest.mark.asyncio
async def test_full136_position_mock_sdk_flow_shares_one216_send_ledger(trial, monkeypatch, simplified_report):
    trial.authorize("controlled grant, not real permission", max_requests=480, confirmed=True)
    monkeypatch.setattr(trial, "_job", lambda *args: synthetic_job(trial, *args))
    captured, data = [], data_for(trial)
    report = await trial.execute(delegate_factory=delegate(captured=captured), controlled_data=data)
    assert report["reserved_requests"] == len(captured) == 216
    assert report["completed_positions"] == 136 and report["planned_user_turns"] == 168
    assert report["execution_complete"] and not report["paused"]
    assert report["semantic_reviews"] == 0 and report["quality_pass_claim"] is False
    assert {row["status"] for row in report["rows"]} == {"failed"}
    assert len(report["task_pairs"]) == 40 and len(report["retrieval"]["enhanced"]) == 48
    assert len(list(trial.root.glob("physical-*.json"))) == 216
    sealed = trial._sealed(trial.verify())
    for op in sealed["operations"].values():
        assert module._read_request(trial.root, op["request_bytes_sha256"])["payload_sha256"] == op["payload_sha256"]
    data.close.assert_awaited_once()
    with pytest.raises(module.base.TrialError, match="no_restart"):
        await trial.execute(delegate_factory=delegate(), controlled_data=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 503])
async def test_transport_fault_spends_once_and_stops_without_retry(trial, monkeypatch, simplified_report, status):
    trial.authorize("controlled grant", max_requests=480, confirmed=True)
    monkeypatch.setattr(trial, "_job", lambda *args: synthetic_job(trial, *args))
    captured, data = [], data_for(trial)
    with pytest.raises((module.TrialTransportError, module.base.TrialError)):
        await trial.execute(delegate_factory=delegate(fail_status=status, captured=captured), controlled_data=data)
    assert len(captured) == trial.ledger(trial.verify()).status()["attempts"] == 1
    assert (trial.root / "worker-stop.json").exists() and (trial.root / "stopped.json").exists()
    assert not list(trial.root.glob("case-*.json"))
    assert "private-do-not-retain" not in "".join(p.read_text() for p in trial.root.glob("physical-*.json"))
    with pytest.raises(module.base.TrialError):
        await trial.execute(delegate_factory=delegate(), controlled_data=data)


@pytest.mark.asyncio
async def test_structure_failure_is_retained_and_next_position_still_executes(trial, monkeypatch, simplified_report):
    trial.authorize("controlled lower cap", max_requests=2, confirmed=True)
    monkeypatch.setattr(trial, "_job", lambda *args: synthetic_job(trial, *args))
    data = data_for(trial)
    with pytest.raises((module.TrialTransportError, module.base.TrialError)):
        await trial.execute(delegate_factory=delegate(malformed=True), controlled_data=data)
    state = trial.ledger(trial.verify()).status()
    assert state["attempts"] == 2 and len(state["completed_cases"]) == 2
    assert [module.bench.read_json(path)["outcome"] for path in sorted(trial.root.glob("physical-*.json"))] == ["structure_failed", "structure_failed"]
    assert len(list(trial.root.glob("case-*.json"))) == 2


@pytest.mark.asyncio
async def test_zero_send_startup_failure_is_one_shot_and_preserves_stop(trial, monkeypatch):
    trial.authorize("controlled grant", max_requests=480, confirmed=True)
    data = data_for(trial)
    def broken(): raise ValueError("controlled startup failure")
    data.assert_healthy = broken
    with pytest.raises(ValueError): await trial.execute(delegate_factory=delegate(), controlled_data=data)
    assert trial.ledger(trial.verify()).status()["attempts"] == 0
    before = (trial.root / "worker-stop.json").read_bytes()
    with pytest.raises(module.base.TrialError): await trial.execute(delegate_factory=delegate(), controlled_data=data)
    assert (trial.root / "worker-stop.json").read_bytes() == before
    data.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_physical_is_counted_once_with_unknown_tokens(trial, simplified_report):
    p = trial.verify()
    trial.authorize("controlled grant", max_requests=480, confirmed=True)
    first = p["initial_intent_requests"][0]
    wire = json.dumps(first["payload"], ensure_ascii=False, separators=(",", ":"))
    module.base._write(trial.root, "request-" + first["request_bytes_sha256"] + ".json", {
        "schema_version": "serveflow-quality-request-v1", "payload_sha256": first["payload_sha256"],
        "request_bytes_sha256": first["request_bytes_sha256"], "payload": first["payload"], "wire_text": wire})
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case(first["case_id"], lease=lease)
        ledger.reserve(first["case_id"], "turn-1", "intent", None, first["payload_sha256"], first["request_bytes_sha256"], lease=lease)
    report = trial.report()
    assert report["reserved_requests"] == report["pending_physical"] == 1
    resources = report["physical_resources"]
    assert resources["recorded_total"] == 1
    assert resources["usage"]["physical_requests"]["known_sum"] == 1
    assert resources["usage"]["input_tokens"]["unknown_count"] == 1
    assert resources["usage"]["output_tokens"]["unknown_count"] == 1


@pytest.mark.asyncio
async def test_report_recomputes_real_http_conversation_artifact_without_semantic_auto_pass(trial, monkeypatch):
    p = trial.verify()
    from test_coverage_conversation_runtime import execute
    from evaluation import coverage_conversation_runtime
    assert callable(coverage_conversation_runtime.conversation_checks)
    artifact, _, _, factory = await execute(0)
    try:
        expected = json.loads(json.dumps(trial.expected()))
        assert not [key for key, value in expected.items() if p[key] != value], [key for key, value in expected.items() if p[key] != value]
        identity = "M-CC-D01"
        module.base._write(trial.root, f"case-{identity}.json", artifact)
        monkeypatch.setattr(trial, "verify", lambda: p)
        monkeypatch.setattr(trial, "_sealed", lambda p: {"operations": {}, "cases": {identity: {"artifact_sha256": module.bench.sha256(trial.root / f"case-{identity}.json")}}})
        report = trial.report()
        row = next(r for r in report["rows"] if r["case_id"] == identity)
        assert row["status"] == "pending_review" and report["quality_pass_claim"] is False
        artifact["turns"][0]["receipt"]["response_sha256"] = "0" * 64
        artifact["tool_checks"] = {"checks": {"all_good": True}, "blockers": []}
        (trial.root / f"case-{identity}.json").write_text(json.dumps(artifact))
        row = next(r for r in trial.report()["rows"] if r["case_id"] == identity)
        assert row["status"] == "failed"
    finally:
        await factory.redis.aclose()


@pytest.mark.asyncio
async def test_counted_sender_runs_real_ad_native_handlers_before_controlled_boundary_stop(trial, monkeypatch):
    from test_coverage_runtime import Data, Replies
    data = Data()
    data.close = AsyncMock()
    trial.authorize("controlled native integration", max_requests=480, confirmed=True)
    actual, handlers = trial._job, {}
    async def stop_after_two(spec, *args):
        if spec["case_id"] not in {"A-CT-D01", "D-CT-D01"}:
            raise ValueError("controlled boundary after the first pair")
        return await actual(spec, *args)
    def factory(frame):
        if frame.case_id not in handlers:
            handlers[frame.case_id] = Replies(trial.controlled_cases[frame.case_id[2:]])
        return httpx.MockTransport(handlers[frame.case_id])
    monkeypatch.setattr(trial, "_job", stop_after_two)
    with pytest.raises(ValueError, match="controlled boundary"):
        await trial.execute(delegate_factory=factory, controlled_data=data)
    p = trial.verify()
    assert trial.ledger(p).status()["completed_cases"] == ["A-CT-D01", "D-CT-D01"]
    assert trial.ledger(p).status()["attempts"] == sum(len(handler.calls) for handler in handlers.values())
    assert data.orders == [("demo-customer-a", "DEMO-2001")] * 2
    for identity in ("A-CT-D01", "D-CT-D01"):
        artifact = module.bench.read_json(trial.root / f"case-{identity}.json")
        assert all(artifact["tool_checks"]["checks"].values())
        assert artifact["semantic_review"] is None and artifact["quality_pass_claim"] is False
        assert artifact["evidence_kind"] == "controlled-transport"
    data.close.assert_awaited_once()
