"""Real role/native-handler wiring under controlled SDK responses only."""
from collections import Counter
from copy import deepcopy
import asyncio
import hashlib
import json
import os
import socket
from types import SimpleNamespace

import pytest

from demo_business.store import load_fixture
from evaluation import expanded_task_runtime as runtime
from evaluation.task_evidence import EvidenceError, fixture_order_result
from evaluation.task_transport import CURRENT


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("expanded task tests cannot contact services")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for name in tuple(os.environ):
        if name.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(name)


class Data:
    def __init__(self):
        self.proof = {"kind": "controlled-fixture"}
        self.fixture, self.orders, self.fault = load_fixture(), [], None

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def order_reader(self, customer, order):
        self.orders.append((customer, order))
        return fixture_order_result(self.fixture, customer, order)

    async def knowledge_handler(self, params, context):
        raise AssertionError("these task-wiring scenarios do not request retrieval")


class SDK:
    base_url = "https://dashscope.aliyuncs.com/apps/anthropic"

    def __init__(self, *, order="DEMO-1008", intent="order_status", answer="受控自由摘要，语义仍须另评。", refuse=False):
        self.messages = self
        self.calls, self.seen = [], Counter()
        self.order, self.intent, self.answer, self.refuse = order, intent, answer, refuse
        self.cancel = False

    @staticmethod
    def text(value):
        return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")

    async def create(self, **kwargs):
        frame = CURRENT.get()
        assert frame is not None
        self.calls.append((frame, deepcopy(kwargs)))
        if self.cancel:
            raise asyncio.CancelledError()
        key = (frame.phase, frame.agent)
        self.seen[key] += 1
        if frame.phase == "intent":
            return self.text(json.dumps({"intent": self.intent, "confidence": 0.99, "reasoning": "controlled"}))
        if frame.phase == "composer":
            return self.text(self.answer)
        assert frame.phase == "agent", "Judge/background model requests are forbidden"
        if self.refuse:
            return self.text(self.answer)
        if self.seen[key] == 1:
            return SimpleNamespace(content=[{"type": "tool_use", "id": "controlled_" + frame.agent,
                "name": "read_demo_order", "input": {"order_id": self.order}}], stop_reason="tool_use")
        return self.text(self.answer)


def source(case_id):
    return next(case for case in runtime.cases() if case["id"] == case_id)


async def execute(case_id="et01", *, arm="D", sdk=None, data=None):
    sdk, data = sdk or SDK(), data or Data()
    artifact = await runtime.execute_task(arm, source(case_id), sdk, data,
        ensure_healthy=data.assert_healthy, frame_case_id=f"{arm}-{case_id}")
    return artifact, sdk, data


def test_fixed_source_and_no_semantic_auto_pass():
    values = runtime.cases()
    assert [c["id"] for c in values] == list(runtime.CASE_IDS)
    assert runtime.ARMS == ("A", "D")
    values[0]["input"]["message"] = "changed"
    assert runtime.cases()[0]["input"]["message"] != "changed"
    artifact, _, _ = asyncio.run(execute())
    assert artifact["semantic_review"] is None and artifact["judge"] is None
    assert artifact["shared_checks"]["semantic_review_required"]


@pytest.mark.parametrize("arm,frame", [("B", "B-et01"), ("a", "a-et01"), ("A", "D-et01"), ("D", "et01")])
def test_wrong_arm_or_frame_never_builds_or_spends(arm, frame):
    sdk, data = SDK(), Data()
    with pytest.raises(ValueError):
        asyncio.run(runtime.execute_task(arm, source("et01"), sdk, data,
            ensure_healthy=data.assert_healthy, frame_case_id=frame))
    assert sdk.calls == data.orders == []


def test_changed_question_or_gold_cannot_be_substituted_before_send():
    sdk, data = SDK(), Data()
    case = source("et01")
    case["expected"]["required_facts"] = []
    with pytest.raises(ValueError, match="expanded_task_case_changed"):
        asyncio.run(runtime.execute_task("D", case, sdk, data,
            ensure_healthy=data.assert_healthy, frame_case_id="D-et01"))
    assert sdk.calls == data.orders == []


@pytest.mark.parametrize("arm", runtime.ARMS)
def test_real_native_calls_and_source_capture_are_bound_for_both_arms(arm):
    artifact, sdk, data = asyncio.run(execute(arm=arm))
    assert data.orders == [("demo-customer-a", "DEMO-1008")]
    assert all(artifact["shared_checks"]["checks"].values())
    assert artifact["review_status"] == "pending_review"
    calls = artifact["turns"][0]["tool_evidence"]["calls"]
    assert all(c["tool_use_id"] and c["request_id"] == "expanded-task-et01-1" for c in calls)
    assert {c["trusted_customer_id"] for c in calls} == {"demo-customer-a"}
    assert {f.case_id for f, _ in sdk.calls} == {f"{arm}-et01"}
    assert {f.turn_id for f, _ in sdk.calls} == {"turn-1"}
    assert not any(f.phase == "judge" for f, _ in sdk.calls)
    assert artifact["shared_checks"]["mechanisms"]["exact_projection"] is (arm == "D")


def test_free_summary_is_structurally_eligible_without_matching_d_template():
    artifact, _, _ = asyncio.run(execute(arm="A", sdk=SDK(answer="这是待人工评阅的自由概括，不证明完整正确。")))
    assert artifact["shared_checks"]["mechanisms"]["exact_projection"] is False
    assert all(artifact["shared_checks"]["checks"].values())
    assert artifact["semantic_review"] is None


def test_user_history_target_is_preserved_and_assistant_claim_is_not_evidence():
    case = source("et14")
    artifact, sdk, data = asyncio.run(execute("et14", sdk=SDK(order="DEMO-1004")))
    assert data.orders == [("demo-customer-a", "DEMO-1004")] * 2
    assert artifact["turns"][0]["history"] == case["input"]["history"]
    assert artifact["shared_checks"]["execution_observation"]["provided_history_preserved"]
    assert all(artifact["shared_checks"]["checks"].values())
    assert "记录金额：未知" in artifact["answer"]
    wire = json.dumps([v for _, v in sdk.calls], ensure_ascii=False)
    assert "上一条未经核验的说法" in wire
    for forbidden in ("required_facts", "forbidden_claims", "review_status", "related_prior_cases",
                      "D-et14", "expanded-task-et14", "et14"):
        assert forbidden not in wire


def test_current_explicit_order_replaces_old_history_order():
    artifact, _, data = asyncio.run(execute("et13", sdk=SDK(order="DEMO-1008")))
    assert data.orders == [("demo-customer-a", "DEMO-1008")]
    assert all(artifact["shared_checks"]["checks"].values())


def test_self_reported_assistant_identity_does_not_replace_trusted_b():
    artifact, _, data = asyncio.run(execute("et18", sdk=SDK(order="DEMO-1009")))
    assert data.orders and all(c == ("demo-customer-b", "DEMO-1009") for c in data.orders)
    assert "129.00" not in artifact["answer"] and "当前演示身份的可见记录中未找到" in artifact["answer"]
    assert all(artifact["shared_checks"]["checks"].values())


def test_ambiguous_target_refusal_can_be_eligible_without_order_read():
    artifact, _, data = asyncio.run(execute("et15", sdk=SDK(refuse=True, answer="请先确认你要核对哪一笔；尚未替你查询。")))
    assert data.orders == []
    assert artifact["shared_checks"]["checks"]["required_tools"]
    assert artifact["shared_checks"]["checks"]["order_results_valid"]
    assert artifact["semantic_review"] is None  # Whether the prose clarified correctly remains manual.


def test_quoted_extra_order_is_not_hidden_by_an_empty_history_checker():
    # Current business resolver reports ambiguous for this frozen challenge.
    # Keep that existing limitation; do not change gold to turn it into a pass.
    artifact, _, data = asyncio.run(execute("et22", sdk=SDK(refuse=True, answer="需要先澄清订单。")))
    assert data.orders == []
    assert artifact["shared_checks"]["checks"]["required_tools"] is False
    assert artifact["shared_checks"]["checks"]["tool_task_correct"] is False
    assert artifact["review_status"] == "failed"


@pytest.mark.parametrize("mutation", ["history", "request", "handler_hash", "native_id", "identity"])
def test_replay_rejects_tampered_association_and_source(mutation):
    artifact, _, _ = asyncio.run(execute("et14", sdk=SDK(order="DEMO-1004")))
    changed = deepcopy(artifact)
    call = changed["turns"][0]["tool_evidence"]["calls"][0]
    if mutation == "history":
        changed["turns"][0]["history"] = []
    elif mutation == "request":
        call["request_id"] = "old-request"
    elif mutation == "handler_hash":
        call["result"]["order"]["total_amount_minor"] += 1
        raw = json.dumps(call["result"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        call["result_sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "native_id":
        call["tool_use_id"] = "not_the_native_id"
    else:
        call["trusted_customer_id"] = "demo-customer-b"
    checked = runtime.checks(source("et14"), changed)
    assert not all(checked["checks"].values()) and checked["blockers"]


def test_closed_capture_blocks_retained_frame_and_cancel_cleans_it():
    _, sdk, _ = asyncio.run(execute())
    with pytest.raises(EvidenceError):
        sdk.calls[0][0].evidence_guard()
    sdk = SDK(); sdk.cancel = True
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(sdk=sdk))
    with pytest.raises(EvidenceError):
        sdk.calls[0][0].evidence_guard()
    assert CURRENT.get() is None


def test_data_fault_after_production_fallback_cannot_publish_success():
    data, sdk = Data(), SDK()
    original = data.order_reader
    async def broken(*args):
        value = await original(*args)
        data.fault = "controlled_data_failure"
        return value
    data.order_reader = broken
    with pytest.raises(ValueError, match="controlled_data_failure"):
        asyncio.run(execute(sdk=sdk, data=data))
    with pytest.raises(EvidenceError):
        sdk.calls[0][0].evidence_guard()
