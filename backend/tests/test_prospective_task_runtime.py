"""Real production role/handler wiring under controlled native responses only."""
from collections import Counter
from copy import deepcopy
import asyncio
import json
import os
import socket
from types import SimpleNamespace

import pytest

from demo_business.store import load_fixture
from evaluation import prospective_task_runtime as runtime
from evaluation.task_evidence import EvidenceError, fixture_order_result
from evaluation.task_transport import CURRENT


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("prospective controlled tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for name in tuple(os.environ):
        if name.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(name)


class Data:
    def __init__(self):
        self.proof = {"kind": "controlled-fixture"}
        self.fixture = load_fixture()
        self.orders = []
        self.fault = None

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def order_reader(self, customer, order):
        self.orders.append((customer, order))
        return fixture_order_result(self.fixture, customer, order)

    async def knowledge_handler(self, params, context):
        raise AssertionError("these controlled scenarios do not request RAG")


class SDK:
    base_url = "https://dashscope.aliyuncs.com/apps/anthropic"

    def __init__(self, *, intent="order_status", tool="read_demo_order", args=None,
                 answer="受控自由摘要，语义仍须评阅。", refuse=False):
        self.messages = self
        self.calls = []
        self.seen = Counter()
        self.intent, self.tool, self.args = intent, tool, args or {"order_id": "DEMO-1007"}
        self.answer, self.refuse = answer, refuse

    @staticmethod
    def text(value):
        return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")

    async def create(self, **kwargs):
        frame = CURRENT.get()
        assert frame is not None
        self.calls.append((frame, deepcopy(kwargs)))
        key = (frame.phase, frame.agent)
        self.seen[key] += 1
        if frame.phase == "intent":
            return self.text(json.dumps({"intent": self.intent, "confidence": 0.99,
                                         "reasoning": "controlled classification"}))
        if frame.phase == "composer":
            return self.text(self.answer)
        assert frame.phase == "agent", "Judge and background calls are forbidden"
        if self.refuse:
            return self.text(self.answer)
        if self.seen[key] == 1:
            return SimpleNamespace(content=[{"type": "tool_use", "id": "controlled_" + frame.agent,
                "name": self.tool, "input": deepcopy(self.args)}], stop_reason="tool_use")
        return self.text(self.answer)


async def execute(arm="D", case_id="p01", *, sdk=None, data=None):
    sdk, data = sdk or SDK(), data or Data()
    artifact = await runtime.execute_arm_case(arm, case_id, sdk, data,
        ensure_healthy=data.assert_healthy, frame_case_id=f"{arm}-{case_id}")
    return artifact, sdk, data


def test_closed_fixed_new_cases_and_physical_envelope_have_no_judge():
    cases = runtime.cases()
    assert [c["id"] for c in cases] == [f"p{i:02d}" for i in range(1, 13)]
    assert len(cases) == 12 and sum(c["expected"]["tool_task"] for c in cases) == 11
    assert runtime.ARMS == ("A", "D")
    assert runtime.PHASE_CAPS == {"intent": 1, "agent": 15, "composer": 1,
                                 "rag_rewrite": 3, "rag_rerank": 3}
    assert runtime.PER_CASE_ARM_MAX == 23
    assert runtime.MAX_TASK_REQUESTS == 552
    cases[0]["input"]["message"] = "mutation"
    assert runtime.cases()[0]["input"]["message"] != "mutation"


@pytest.mark.parametrize("arm,case_id,frame", [
    ("B", "p01", "B-p01"), ("a", "p01", "a-p01"),
    ("A", "e01", "A-e01"), ("D", "p13", "D-p13"),
    ("A", "p01", "D-p01"), ("D", "p01", "p01"),
])
def test_old_or_wrong_identity_never_builds_runtime_or_spends(arm, case_id, frame):
    sdk, data = SDK(), Data()
    with pytest.raises(ValueError):
        asyncio.run(runtime.execute_arm_case(arm, case_id, sdk, data,
            ensure_healthy=data.assert_healthy, frame_case_id=frame))
    assert sdk.calls == [] and data.orders == []


@pytest.mark.parametrize("arm", runtime.ARMS)
def test_real_native_tool_handler_capture_and_source_binding_without_judge(arm):
    artifact, sdk, data = asyncio.run(execute(arm))
    assert data.orders == [("demo-customer-a", "DEMO-1007")] * 2
    assert artifact["data_kind"] == "controlled-fixture"
    assert artifact["case_id"] == f"{arm}-p01" and artifact["source_case_id"] == "p01"
    assert artifact["arm"] == arm and artifact["semantic_review"] is None
    assert artifact["judge"] is None and artifact["review_status"] == "pending_review"
    assert artifact["shared_checks"]["blockers"] == []
    assert all(artifact["shared_checks"]["checks"].values())
    evidence = artifact["turns"][0]["tool_evidence"]["calls"]
    assert len(evidence) == 2 and all(call["tool_use_id"] for call in evidence)
    assert evidence[0]["trusted_customer_id"] == "demo-customer-a"
    assert evidence[0]["input"] == {"order_id": "DEMO-1007"}
    assert evidence[0]["result"] == fixture_order_result(data.fixture, "demo-customer-a", "DEMO-1007")
    assert all(frame.case_id == f"{arm}-p01" for frame, _ in sdk.calls)
    actor_calls = [kwargs for frame, kwargs in sdk.calls if frame.phase == "agent"]
    if arm == "D":
        assert len(actor_calls) == 2
        assert all(call["tool_choice"] == {"type": "tool", "name": "read_demo_order"} for call in actor_calls)
        assert all([tool["name"] for tool in call["tools"]] == ["read_demo_order"] for call in actor_calls)
    else:
        assert len(actor_calls) == 4
        for role in ("general", "billing"):
            role_calls = [kwargs for frame, kwargs in sdk.calls if frame.phase == "agent" and frame.agent == role]
            assert len(role_calls) == 2 and "tool_choice" not in role_calls[0]
            results = [block for message in role_calls[1]["messages"] if isinstance(message["content"], list)
                       for block in message["content"] if block.get("type") == "tool_result"]
            captured = next(call for call in evidence if call["agent"] == role)
            assert results[0]["tool_use_id"] == captured["tool_use_id"]
            assert json.loads(results[0]["content"]) == captured["result"]
    assert not any(frame.phase == "judge" for frame, _ in sdk.calls)
    assert CURRENT.get() is None


def test_gold_labels_private_case_and_arm_identity_never_enter_model_payloads():
    artifact, sdk, _ = asyncio.run(execute())
    wire = json.dumps([kwargs for _, kwargs in sdk.calls], ensure_ascii=False)
    source = runtime.cases()[0]
    assert source["input"]["message"] in wire
    for token in ["p01", "D-p01", "prospective-task-p01", "required_facts", "forbidden_claims",
                  "expected_status", "quality-common", "scope_proposal"]:
        assert token not in wire
    for fact in source["expected"]["required_facts"]:
        assert fact not in wire
    assert artifact["semantic_review"] is None


@pytest.mark.parametrize("change", [
    lambda a: a.update(source_case_id="p02"),
    lambda a: a.update(case_id="p01"),
    lambda a: a.update(arm="B"),
    lambda a: a.update(case_id="D-p02"),
    lambda a: a["turns"][0].update(question="different question"),
    lambda a: a["turns"][0]["tool_evidence"]["calls"][0].update(trusted_customer_id="demo-customer-b"),
    lambda a: a["turns"][0]["tool_evidence"]["calls"][0].update(input={"order_id": "DEMO-1001"}),
    lambda a: a["turns"][0]["tool_evidence"]["calls"][0].update(tool_use_id="fabricated_native_id"),
    lambda a: a["turns"][0]["tool_evidence"]["calls"][0]["result"]["order"].update(total_amount_minor=0),
])
def test_bound_artifact_identity_source_native_id_and_parameters_are_recomputed(change):
    artifact, _, _ = asyncio.run(execute())
    change(artifact)
    value = runtime.checks("p01", artifact)
    assert not all(value["checks"].values()) and value["blockers"]


def test_partial_unknown_is_eligible_for_review_without_becoming_an_automatic_pass():
    artifact, _, _ = asyncio.run(execute())
    artifact["turns"][0]["result"]["status"] = "partial"
    artifact["execution_status"] = "partial"
    value = runtime.checks("p01", artifact)
    assert value["execution_observation"]["original_mutation_only_gate"] is False
    assert value["checks"]["execution_eligible"] is value["checks"]["expected_status"] is True
    assert value["semantic_review_required"] is True
    artifact["turns"][0]["result"]["status"] = "failed"
    artifact["execution_status"] = "failed"
    value = runtime.checks("p01", artifact)
    assert not value["checks"]["execution_eligible"]
    assert "execution_eligible" in value["blockers"]


def test_p11_allows_tool_free_safe_refusal_but_does_not_autograde_its_meaning():
    sdk = SDK(refuse=True, answer="当前身份不能读取他人的订单，我没有按你自报的身份切换查询。")
    artifact, _, data = asyncio.run(execute("A", "p11", sdk=sdk))
    assert data.orders == []
    assert artifact["shared_checks"]["tool_task_applicable"] is False
    assert artifact["shared_checks"]["checks"]["required_tools"] is True
    assert artifact["semantic_review"] is None
    # A text without a valid read is legitimately partial; status does not
    # establish that its refusal is relevant or complete.
    assert artifact["review_status"] == "pending_review"


def test_p11_actual_read_uses_b_identity_and_returns_not_found_without_a_fields():
    sdk = SDK(args={"order_id": "DEMO-1007"})
    artifact, _, data = asyncio.run(execute("D", "p11", sdk=sdk))
    assert data.orders == [("demo-customer-b", "DEMO-1007")]
    assert artifact["turns"][0]["tool_evidence"]["calls"][0]["result"]["status"] == "not_found"
    assert "演示智能插座套装" not in artifact["answer"]
    assert artifact["shared_checks"]["blockers"] == []


def test_bad_native_target_is_not_repaired_or_silently_executed():
    artifact, sdk, data = asyncio.run(execute(sdk=SDK(args={"order_id": "DEMO-1001"})))
    assert data.orders == []
    assert artifact["review_status"] == "failed"
    assert "tool_input_invalid" in artifact["shared_checks"]["blockers"]
    assert Counter(frame.agent for frame, _ in sdk.calls if frame.phase == "agent") == {"general": 1, "billing": 1}


def test_data_failure_hidden_by_production_fallback_still_propagates_and_closes_capture():
    class FailingData(Data):
        async def order_reader(self, customer, order):
            self.fault = "controlled_data_failure"
            raise ValueError(self.fault)
    sdk, data = SDK(), FailingData()
    with pytest.raises(ValueError, match="controlled_data_failure"):
        asyncio.run(execute(sdk=sdk, data=data))
    assert CURRENT.get() is None
    frames = [frame for frame, _ in sdk.calls]
    with pytest.raises(EvidenceError, match="capture_not_active"):
        frames[0].evidence_guard()


def test_cancelled_case_closes_capture_and_old_frame_cannot_send_later():
    class BlockingSDK(SDK):
        async def create(self, **kwargs):
            frame = CURRENT.get()
            if frame.phase == "agent":
                self.calls.append((frame, deepcopy(kwargs)))
                await asyncio.Event().wait()
            return await super().create(**kwargs)
    sdk = BlockingSDK()
    async def run():
        task = asyncio.create_task(execute(sdk=sdk))
        for _ in range(100):
            if any(frame.phase == "agent" for frame, _ in sdk.calls):
                break
            await asyncio.sleep(0)
        assert any(frame.phase == "agent" for frame, _ in sdk.calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert CURRENT.get() is None
    frames = [frame for frame, _ in sdk.calls]
    with pytest.raises(EvidenceError, match="capture_not_active"):
        frames[-1].evidence_guard()


@pytest.mark.parametrize("case_id,code,intent", [
    ("p07", "403", "technical_login"), ("p08", "500", "technical_crash"),
])
def test_native_diagnostic_tool_produces_separate_local_suggestions_not_order_facts(case_id, code, intent):
    sdk = SDK(intent=intent, tool="lookup_error_code", args={"error_code": code})
    artifact, sdk, data = asyncio.run(execute("D", case_id, sdk=sdk))
    assert data.orders == []
    captured = artifact["turns"][0]["tool_evidence"]["calls"]
    assert len(captured) == 1 and captured[0]["name"] == "lookup_error_code"
    assert captured[0]["input"] == {"error_code": code}
    assert captured[0]["result"]["server_log_checked"] is False
    assert captured[0]["tool_use_id"]
    assert artifact["shared_checks"]["blockers"] == []
    assert artifact["semantic_review"] is None
    assert all(frame.phase in {"intent", "agent", "composer"} for frame, _ in sdk.calls)


def test_seen_scope_regression_is_partial_and_the_old_success_mismatch_is_still_rejected():
    sdk = SDK(intent="refund", args={"order_id": "DEMO-1009"})
    artifact, _, data = asyncio.run(execute("D", "p12", sdk=sdk))
    assert data.orders == [("demo-customer-a", "DEMO-1009")]
    # The original 4acc516 controlled success observation and frozen partial
    # expectation remain archived. This is a new-code development regression,
    # not a new live result or an unseen validation example.
    assert artifact["execution_status"] == "partial"
    assert artifact["shared_checks"]["execution_observation"]["expected_status"] == "partial"
    assert artifact["shared_checks"]["checks"]["expected_status"]
    assert artifact["review_status"] == "pending_review"
    assert artifact["semantic_review"] is None
    assert "自动续费：开启" in artifact["answer"] and "这些操作尚未完成" in artifact["answer"]
    wrong = deepcopy(artifact)
    wrong["execution_status"] = wrong["turns"][0]["result"]["status"] = "success"
    assert not runtime.checks("p12", wrong)["checks"]["expected_status"]


def test_case_and_arm_factories_are_fresh_and_never_share_retained_order_evidence(monkeypatch):
    original = runtime.build_arm_runtime
    instances = []
    async def build(*args, **kwargs):
        orc, manager = await original(*args, **kwargs)
        instances.append((orc, manager))
        return orc, manager
    monkeypatch.setattr(runtime, "build_arm_runtime", build)
    async def run():
        await execute("A")
        await execute("D")
        await execute("A")
    asyncio.run(run())
    assert len({id(orc) for orc, _ in instances}) == 3
    assert len({id(manager) for _, manager in instances}) == 3
    assert len({id(orc._arm_state) for orc, _ in instances}) == 3
    assert len({id(orc._pool) for orc, _ in instances}) == 3


def test_preexisting_execution_failure_or_invalid_data_never_enters_model():
    sdk, data = SDK(), Data()
    data.fault = "preexisting_count_or_data_error"
    with pytest.raises(ValueError, match="preexisting_count_or_data_error"):
        asyncio.run(execute(sdk=sdk, data=data))
    assert sdk.calls == []
    data.fault = None
    data.proof = {"kind": "pretend-real-source"}
    with pytest.raises(ValueError, match="invalid_evaluation_data_kind"):
        asyncio.run(execute(sdk=sdk, data=data))
    assert sdk.calls == []
