"""Targeted freeze, accounting and real SDK/handler wiring; entirely controlled.

The examples below never authorize a real model, database or business request.
Socket access is denied; Anthropic SDK requests use an in-memory MockTransport.
"""
import asyncio
from copy import deepcopy
import json
import signal
import socket
import stat
from types import SimpleNamespace

import httpx
import pytest

from agents.agent_orchestrator import AgentType
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, diagnostic_repair_trial as module
from evaluation import prospective_trial as previous, task_supervisor
from evaluation.intent_transport import TrialTransportError
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import CURRENT
from test_prospective_trial import BASE, Data, response, send


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("targeted repair tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(module.os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module.proof, "source_proof", lambda: {"controlled_fixed_source": "a" * 64})
    monkeypatch.setattr(module.proof, "artifacts", lambda: {"controlled_new_files": "b" * 64})
    monkeypatch.setattr(module.proof, "prior_proof", lambda: {"old_paused_ledger_unchanged": "c" * 64})
    monkeypatch.setattr(module.engine, "_dependencies", lambda: {"controlled_only": "1"})
    monkeypatch.setattr(module.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(module.task_data, "offline_data_proof", lambda: {"controlled_fixture": True})
    def configuration(*, secret=False):
        return {"base_url": BASE, "model": module.base.MODEL,
                **({"api_key": "controlled-only-not-a-secret"} if secret else {})}
    monkeypatch.setattr(module.base, "_configuration", configuration)
    return tmp_path


@pytest.fixture
def prepared(configured):
    value = module.FocusedTrial(configured / "targeted-execution")
    asyncio.run(value.prepare())
    return value


def approve(trial, limit=module.MAX_REQUESTS):
    return trial.authorize("controlled test receipt only; no live model allowance",
                           max_requests=limit, confirmed=True)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def review(trial, key, *, all_true=True):
    dimensions = dict.fromkeys(module.design.DIMENSIONS, True)
    if not all_true:
        dimensions["coverage"] = False
    return {"logical_case_id": key,
            "artifact_sha256": bench.sha256(trial.root / f"case-{key}.json"),
            "reviewer": "controlled executor with source checks",
            "method": "controlled-fixture", "dimensions": dimensions,
            "blocking_errors": [] if all_true else ["Controlled missing requested content."],
            "reading_burden": "none",
            "reason": "Controlled rubric plumbing, never a live quality result."}


def test_prepare_captures_six_real_sdk_intent_payloads_without_gold_or_authorization(prepared):
    p = prepared.verify()
    assert p["schema_version"] == module.FocusedTrial.protocol_schema
    assert p["authorized_model_requests"] == p["judge_requests"] == p["external_business_requests"] == 0
    assert p["proposed_model_requests"] == 138 and p["plan"] == module.fixed_plan()
    assert [row["case_id"] for row in p["initial_intent_requests"]] == [
        "D-p07", "D-n01", "D-p01", "D-n02", "D-p09", "D-n03"]
    assert len(p["initial_intent_requests"]) == 6 and "judge" not in p["phase_parameters"]
    for case, captured in zip(p["cases"], p["initial_intent_requests"]):
        wire = json.dumps(captured["payload"], ensure_ascii=False)
        assert case["input"]["message"] in wire
        assert captured["payload"]["model"] == module.base.MODEL
        for token in ("required_facts", "forbidden_claims", "base_case_id", "expected_status",
                      "seen_regression", "new_authored_rephrase", "diagnostic-repair-" + case["id"]):
            assert token not in wire
        for fact in case["expected"]["required_facts"]:
            assert fact not in wire
    assert prepared.report()["reserved_requests"] == 0
    assert not (prepared.root / "authorization.json").exists()
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((prepared.root / "protocol.json").stat().st_mode) == 0o600
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
               for path in (prepared.root / "ledger").rglob("*") if path.is_file())


def test_original_questions_and_gold_are_immutable_and_new_groups_are_not_new_facts():
    original = {c["id"]: c for c in module.design._cases(bench.BACKEND_ROOT, "tasks")["cases"]}
    selected = module.cases()
    assert [c["id"] for c in selected] == list(module.IDS)
    assert [c["group"] for c in selected] == list(module.GROUPS) * 3
    for c in selected:
        old = original[c["base_case_id"]]
        assert c["expected"] == old["expected"] and c["references"] == old["references"]
        assert c["domain"] == old["domain"] and c["related_prior_cases"] == old["related_prior_cases"]
        assert c["input"]["history"] == [] and c["input"]["customer_id"] == "demo-customer-a"
        if c["group"] == "seen_regression":
            assert c["input"] == old["input"]
        else:
            assert bench._normal(c["input"]["message"]) not in {
                bench._normal(row["input"]["message"]) for row in original.values()}
    selected[0]["expected"]["required_tools"].clear()
    assert module.cases()[0]["expected"]["required_tools"] == ["lookup_error_code"]


@pytest.mark.parametrize("damage", ["order", "arm", "gold", "history", "identity", "old-question", "group"])
def test_dataset_drift_rejects_before_freezing_anything(configured, monkeypatch, damage):
    dataset = bench.read_json(module.MANIFEST)
    if damage == "order": dataset["cases"].reverse()
    elif damage == "arm": dataset["cases"][0]["id"] = "A-p07"
    elif damage == "gold": dataset["cases"][1]["expected"]["required_tools"] = []
    elif damage == "history": dataset["cases"][1]["input"]["history"] = [{"role": "user", "content": "gold"}]
    elif damage == "identity": dataset["cases"][1]["input"]["customer_id"] = "demo-customer-b"
    elif damage == "old-question": dataset["cases"][0]["input"]["message"] += "改动"
    else: dataset["cases"][0]["group"] = "new_authored_rephrase"
    path = configured / "controlled-dataset.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False))
    monkeypatch.setattr(module, "MANIFEST", path)
    root = configured / "must-not-exist"
    with pytest.raises((module.base.TrialError, bench.BenchmarkError)):
        asyncio.run(module.FocusedTrial(root).prepare())
    assert not root.exists()


def test_no_authorization_opens_no_credentials_data_sdk_or_worker(prepared, monkeypatch):
    import anthropic
    config = module.base._configuration
    def no_secret(*, secret=False):
        assert secret is False, "must not read credentials before approval"
        return config(secret=False)
    def denied(*args, **kwargs):
        raise AssertionError("must not open resources or spawn without approval")
    monkeypatch.setattr(module.base, "_configuration", no_secret)
    monkeypatch.setattr(module.task_data, "build_live_data", denied)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", denied)
    monkeypatch.setattr(module.subprocess, "Popen", denied)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    with pytest.raises(FileNotFoundError): asyncio.run(prepared.execute())
    with pytest.raises(FileNotFoundError): prepared.run()
    assert prepared.report()["reserved_requests"] == 0


def test_prepare_authorization_and_existing_root_never_overwrite(prepared):
    before = snapshot(prepared.root)
    with pytest.raises(Exception, match="root_exists"): asyncio.run(prepared.prepare())
    assert snapshot(prepared.root) == before
    approve(prepared, 1)
    receipt = (prepared.root / "authorization.json").read_bytes()
    with pytest.raises(FileExistsError): approve(prepared, 2)
    assert (prepared.root / "authorization.json").read_bytes() == receipt
    with pytest.raises(Exception, match="fixed_root_required"):
        module.FocusedTrial(previous.ROOT)


@pytest.mark.parametrize("limit", [0, -1, 139, True, 1.5, None])
def test_invalid_authorization_does_not_create_an_allowance(prepared, limit):
    with pytest.raises(Exception): approve(prepared, limit)
    assert not (prepared.root / "authorization.json").exists()


def test_prior_batch_authorization_cannot_be_transplanted(prepared):
    p = prepared.verify()
    forged = {"schema_version": "serveflow-prospective-authorization-v2-readonly-scope",
              "reference": "controlled old receipt", "recorded_at": module.base._now(),
              "protocol_sha256": "d" * 64, "root_binding_sha256": "e" * 64,
              "initial_chain_hash": "f" * 64, "max_requests": 138, "plan": p["plan"]}
    for new_schema in (False, True):
        if new_schema: forged["schema_version"] = prepared.authorization_schema
        module.base._write(prepared.root, "authorization.json", forged)
        with pytest.raises(Exception, match="approval_changed"): prepared._approval(p)
        (prepared.root / "authorization.json").unlink()  # Temporary forged test receipt only.
    assert prepared.report()["reserved_requests"] == 0


@pytest.mark.parametrize("damage", ["source", "artifacts", "prior", "environment", "dependencies", "data"])
def test_frozen_inputs_remain_bound_before_any_send(prepared, monkeypatch, damage):
    changed = lambda: {"controlled_changed": True}
    if damage == "source": monkeypatch.setattr(module.proof, "source_proof", changed)
    elif damage == "artifacts": monkeypatch.setattr(module.proof, "artifacts", changed)
    elif damage == "prior": monkeypatch.setattr(module.proof, "prior_proof", changed)
    elif damage == "environment": monkeypatch.setattr(module.history, "effective_environment", changed)
    elif damage == "dependencies": monkeypatch.setattr(module.engine, "_dependencies", changed)
    else: monkeypatch.setattr(module.task_data, "offline_data_proof", changed)
    with pytest.raises(Exception, match="freeze_changed"): prepared.verify()


@pytest.mark.parametrize("damage", ["payload", "plan", "group", "authorization", "source-id"])
def test_protocol_edit_cannot_change_plan_or_add_model_gold(prepared, damage):
    value = prepared.verify()
    if damage == "payload": value["initial_intent_requests"][0]["payload"]["gold"] = "answer"
    elif damage == "plan": value["plan"].pop()
    elif damage == "group": value["cases"][0]["group"] = "new_authored_rephrase"
    elif damage == "authorization": value["authorized_model_requests"] = 138
    else: value["initial_intent_requests"][0]["source_case_id"] = "n01"
    (prepared.root / "protocol.json").write_text(json.dumps(value, ensure_ascii=False))
    with pytest.raises(Exception): prepared.verify()


def test_lower_send_allowance_stays_spent_and_cannot_be_refilled_after_reopen(prepared):
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    captured = []
    async def handle(request): captured.append(request); return response()
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        transport = module.FocusedTransport(BASE, prepared.root, budget, lease, prepared.verify,
            delegate_factory=lambda frame: httpx.MockTransport(handle))
        async def perform():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await send(transport, job="D-p07")
                with pytest.raises(TrialTransportError): await send(transport, job="D-p07")
        asyncio.run(perform())
    reopened = module.FocusedTrial(prepared.root)
    assert reopened.report()["reserved_requests"] == len(captured) == 1
    assert reopened.report()["paused"] is True
    with pytest.raises(Exception, match="paused"):
        asyncio.run(reopened.execute(delegate_factory=lambda frame: None, controlled_data=Data()))
    with pytest.raises(Exception, match="already_started"): approve(reopened, 138)
    assert reopened.report()["reserved_requests"] == 1


def test_pending_uncertain_attempt_remains_unknown_and_cannot_be_resent(prepared):
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        budget.reserve("D-p07", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    reopened = module.FocusedTrial(prepared.root)
    report = reopened.report()
    assert report["reserved_requests"] == report["pending_physical"] == 1
    assert report["cases"][0]["status"] == "unknown"
    assert report["counts"] == {"planned": 6, "passed": 0, "failed": 0, "pending_review": 0,
                                "unknown": 1, "not_executed": 5}
    with pytest.raises(TaskBudgetError):
        with reopened.ledger(p).exclusive_run():
            pytest.fail("unknown reservation cannot open another run")
    with pytest.raises(Exception, match="already_started"): approve(reopened, 2)
    assert reopened.report()["reserved_requests"] == 1


def test_focused_transport_rejects_old_arm_and_judge_before_a_send(prepared):
    from evaluation.task_transport import Frame, context
    from test_prospective_trial import wire
    p = prepared.verify(); budget = prepared.ledger(p); sent = []
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        transport = module.FocusedTransport(BASE, prepared.root, budget, lease, prepared.verify,
            delegate_factory=lambda frame: httpx.MockTransport(lambda request: sent.append(request)))
        for frame in (Frame("A-p07", "turn-1", "agent", "general"),
                      Frame("D-p07", "turn-1", "judge", None)):
            with context(frame), pytest.raises(TrialTransportError):
                transport._validate(wire(), frame)
    assert sent == [] and prepared.report()["reserved_requests"] == 0


def test_real_sdk_native_capture_and_fixture_handlers_for_all_three_responsibilities(prepared):
    approve(prepared)
    data = Data(); captured = []
    async def handle(request):
        frame = CURRENT.get(); payload = json.loads(request.content)
        captured.append((frame, deepcopy(payload)))
        if frame.phase == "intent":
            # A controlled classification exercises native/handler wiring,
            # not the model's unmeasured intent accuracy. A compound request
            # must retain diagnosis even with an order-only classifier label.
            intent = "technical" if frame.case_id in {"D-p07", "D-n01"} else "order_status"
            return response(json.dumps({"intent": intent, "confidence": .99, "reasoning": "controlled"}))
        assert frame.phase == "agent", "no Judge, summary or background calls are expected here"
        choice = payload.get("tool_choice", {}).get("name")
        assert choice in {"read_demo_order", "lookup_error_code"}
        args = ({"error_code": "403"} if choice == "lookup_error_code" else
                {"order_id": "DEMO-1007" if frame.case_id in {"D-p01", "D-n02"} else "DEMO-1002"})
        return response(content=[{"type": "tool_use", "id": "controlled_" + str(len(captured)),
                                  "name": choice, "input": args}])
    report = asyncio.run(prepared.execute(delegate_factory=lambda frame: httpx.MockTransport(handle),
                                           controlled_data=data))
    assert data.closed and data.orders and all(customer == "demo-customer-a" for customer, _ in data.orders)
    assert report["planned_jobs"] == report["completed_jobs"] == 6
    assert report["reserved_requests"] == len(captured) and report["pending_physical"] == 0
    assert report["paused"] is False and report["review_ready"] is True
    assert report["counts"] == {"planned": 6, "passed": 0, "failed": 0,
                                "pending_review": 6, "unknown": 0, "not_executed": 0}
    assert report["semantic_reviews_imported"] == 0 and report["quality_pass_claim"] is False
    for group in module.GROUPS:
        assert report["groups"][group]["planned"] == report["groups"][group]["pending_review"] == 3
    for c in module.cases():
        artifact = bench.read_json(prepared.root / f"case-D-{c['id']}.json")
        assert artifact["semantic_review"] is artifact["judge"] is None
        assert artifact["arm"] == "D" and artifact["evidence_kind"] == "controlled-transport"
        assert artifact["shared_checks"]["blockers"] == []
        calls = artifact["turns"][0]["tool_evidence"]["calls"]
        assert set(c["expected"]["required_tools"]) <= {call["name"] for call in calls}
        assert all(call["tool_use_id"] and call["success"] for call in calls)
        for call in calls:
            if call["name"] == "lookup_error_code":
                assert call["input"] == {"error_code": "403"}
                assert call["result"]["server_log_checked"] is False
            else:
                assert call["trusted_customer_id"] == "demo-customer-a"
        if c["base_case_id"] == "p01":
            assert "订单取消状态：未知" in artifact["answer"]
            assert "159.00 元" in artifact["answer"]
            assert "已取消或未取消" in artifact["answer"]
        if c["base_case_id"] == "p09":
            assert "899.00 元" in artifact["answer"] and "preorder" in artifact["answer"]
        partial = deepcopy(artifact)
        partial["execution_status"] = partial["turns"][0]["result"]["status"] = "partial"
        observed = module.checks(c, partial)
        assert observed["checks"]["execution_eligible"] is True
        assert observed["checks"]["expected_status"] is (c["expected"]["expected_status"] == "not-prescribed")
    assert not any(frame.phase == "judge" for frame, _ in captured)
    for _, payload in captured:
        wire = json.dumps(payload, ensure_ascii=False)
        for banned in ("required_facts", "forbidden_claims", "base_case_id", "new_authored_rephrase", "seen_regression"):
            assert banned not in wire
    # A truthful source/tool observation still needs a semantic reviewer. A
    # negative coverage finding cannot be concealed by the favorable new group.
    prepared.review(review(prepared, "D-p01", all_true=False))
    after = prepared.report()
    assert after["counts"]["failed"] == 1 and after["counts"]["planned"] == 6
    assert after["groups"]["seen_regression"]["failed"] == 1
    assert after["groups"]["new_authored_rephrase"]["pending_review"] == 3
    assert after["quality_pass_claim"] is False
    with pytest.raises(FileExistsError): prepared.review(review(prepared, "D-p01"))


@pytest.mark.parametrize("missing_technical", [False, True])
def test_order_only_controlled_intent_has_real_compound_sources_or_safe_partial(monkeypatch, missing_technical):
    class SDK:
        base_url = BASE
        def __init__(self):
            self.messages = self
            self.calls = []
        async def create(self, **kwargs):
            frame = CURRENT.get()
            self.calls.append((frame, deepcopy(kwargs)))
            if frame.phase == "intent":
                return SimpleNamespace(content=[{"type": "text", "text": json.dumps({
                    "intent": "order_status", "confidence": .99, "reasoning": "controlled"})}],
                    stop_reason="end_turn")
            assert frame.phase == "agent"
            name = kwargs["tool_choice"]["name"]
            assert name in {"read_demo_order", "lookup_error_code"}
            args = {"order_id": "DEMO-1002"} if name == "read_demo_order" else {"error_code": "403"}
            return SimpleNamespace(content=[{"type": "tool_use", "id": "controlled_native_read",
                "name": name, "input": args}], stop_reason="tool_use")
    sdk, data = SDK(), Data()
    if missing_technical:
        build = module.build_arm_runtime
        async def missing_role(*args, **kwargs):
            runtime, manager = await build(*args, **kwargs)
            runtime._pool.pop(AgentType.TECHNICAL)
            return runtime, manager
        monkeypatch.setattr(module, "build_arm_runtime", missing_role)
    case = next(c for c in module.cases() if c["id"] == "p09")
    artifact = asyncio.run(module.execute_case(case, sdk, data, data.assert_healthy))
    assert "899.00 元" in artifact["answer"]
    if missing_technical:
        assert artifact["execution_status"] == "partial"
        assert "技术排查部分本次未完成" in artifact["answer"]
        assert artifact["shared_checks"]["checks"]["required_tools"] is False
        assert "required_tools" in artifact["shared_checks"]["blockers"]
    else:
        assert artifact["execution_status"] == "success"
        assert artifact["shared_checks"]["blockers"] == []
        assert all(artifact["shared_checks"]["checks"].values())
        observed = artifact["turns"][0]["tool_evidence"]["calls"]
        assert {call["name"] for call in observed} >= {"read_demo_order", "lookup_error_code"}
        assert any(call["result"].get("server_log_checked") is False for call in observed)
    assert artifact["semantic_review"] is None
    assert data.orders and all(c == "demo-customer-a" for c, _ in data.orders)
    assert not any(frame.phase == "judge" for frame, _ in sdk.calls)


def test_ordinary_missing_native_output_fails_case_but_continues_fixed_remaining_plan(prepared):
    approve(prepared)
    data, captured = Data(), []
    async def handle(request):
        frame = CURRENT.get(); payload = json.loads(request.content)
        captured.append(frame)
        if frame.phase == "intent":
            intent = "order_status" if frame.case_id in {"D-p01", "D-n02"} else "technical"
            return response(json.dumps({"intent": intent, "confidence": .99, "reasoning": "controlled"}))
        assert frame.phase == "agent"
        if frame.case_id == "D-p07":
            return response("受控普通失败：本回复没有执行要求的工具。")
        tool = payload["tool_choice"]["name"]
        args = ({"error_code": "403"} if tool == "lookup_error_code" else
                {"order_id": "DEMO-1007" if frame.case_id in {"D-p01", "D-n02"} else "DEMO-1002"})
        return response(content=[{"type": "tool_use", "id": "controlled_" + str(len(captured)),
                                  "name": tool, "input": args}])
    report = asyncio.run(prepared.execute(delegate_factory=lambda frame: httpx.MockTransport(handle),
                                           controlled_data=data))
    assert report["paused"] is False and data.closed
    assert report["completed_jobs"] == 6 and report["counts"]["planned"] == 6
    assert report["counts"]["failed"] == 1 and report["counts"]["pending_review"] == 5
    assert [row["logical_case_id"] for row in report["cases"]] == [r["case_id"] for r in module.fixed_plan()]
    assert report["cases"][0]["status"] == "failed"
    assert "required_tools" in report["cases"][0]["checks"]["blockers"]
    assert report["groups"]["seen_regression"]["failed"] == 1
    assert report["groups"]["new_authored_rephrase"]["pending_review"] == 3
    assert report["reserved_requests"] == len(captured)
    # A reviewer cannot turn the missing actual native execution into success.
    prepared.review(review(prepared, "D-p07"))
    assert prepared.report()["cases"][0]["status"] == "failed"
    assert prepared.report()["quality_pass_claim"] is False


def test_all_failed_cases_keep_full_denominators_and_cannot_be_raised_by_manual_flags(prepared):
    p = prepared.verify(); data = Data()
    module.base._write(prepared.root, "data-proof.json", data.proof)
    proof_hash = bench.sha256(prepared.root / "data-proof.json")
    module.base._write(prepared.root, "data-proof-seal.json", {
        "protocol_sha256": bench.sha256(prepared.root / "protocol.json"), "data_proof_sha256": proof_hash})
    budget = prepared.ledger(p)
    with budget.exclusive_run() as lease:
        for case in p["cases"]:
            key = "D-" + case["id"]
            artifact = {"schema_version": "serveflow-diagnostic-repair-artifact-v1", "logical_case_id": key,
                "case_id": key, "source_case_id": case["id"], "arm": "D", "data_kind": "controlled-fixture",
                "evidence_kind": "controlled-fixture", "execution_status": "failed", "turns": [],
                "protocol_sha256": bench.sha256(prepared.root / "protocol.json"), "data_proof_sha256": proof_hash}
            artifact["shared_checks"] = module.checks(case, artifact)
            module.base._write(prepared.root, f"case-{key}.json", artifact)
            budget.start_case(key, lease=lease)
            budget.finish_case(key, bench.sha256(prepared.root / f"case-{key}.json"), outcome="failed", lease=lease)
    for case in p["cases"]:
        prepared.review(review(prepared, "D-" + case["id"]))
    report = prepared.report()
    assert report["counts"]["failed"] == report["counts"]["planned"] == 6
    assert all(report["groups"][g]["failed"] == report["groups"][g]["planned"] == 3 for g in module.GROUPS)
    assert report["semantic_reviews_imported"] == 6 and report["quality_pass_claim"] is False
    assert report["reserved_requests"] == 0
    assert report["review_ready"] is False  # No runtime cleanup record is invented.


def test_unexecuted_cases_and_empty_resource_population_are_not_success(prepared):
    report = prepared.report()
    assert report["counts"] == {"planned": 6, "passed": 0, "failed": 0,
                                "pending_review": 0, "unknown": 0, "not_executed": 6}
    assert all(report["groups"][g]["not_executed"] == report["groups"][g]["planned"] == 3 for g in module.GROUPS)
    assert report["quality_pass_claim"] is report["review_ready"] is False
    assert report["resources"]["usage"]["input_tokens"]["known_sum"] is None


def test_temporary_or_wrong_worker_context_cannot_spawn(prepared, monkeypatch):
    approve(prepared)
    def denied(*args, **kwargs):
        raise AssertionError("temporary/wrong context must not spawn any worker")
    monkeypatch.setattr(module.subprocess, "Popen", denied)
    with pytest.raises(Exception, match="fixed_worker_required"): prepared.run()
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(prepared.root.parent)
    with pytest.raises(Exception, match="fixed_worker_required"): prepared.run()


def test_fixed_worker_mapping_runs_only_targeted_module_without_resources(prepared, monkeypatch):
    approve(prepared)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    seen = []
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    class Process:
        def wait(self, timeout):
            assert timeout == 900
            return 0
    def spawn(command, **kwargs):
        seen.append(command)
        assert command == [module.sys.executable, "-B", "-u", "-m", "evaluation.diagnostic_repair_trial", "_worker"]
        assert kwargs["shell"] is False and kwargs["start_new_session"] is kwargs["close_fds"] is True
        return Process()
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    assert prepared.run() == {"exit_code": 0, "timed_out": False, "cancelled": False}
    assert len(seen) == 1
    assert {s: signal.getsignal(s) for s in before} == before
    assert prepared.report()["reserved_requests"] == 0
    assert not (prepared.root / "data-proof.json").exists()
