"""Controlled E2E wiring checks: real Orchestrator, fake model/data, no services.

These tests establish execution/evidence plumbing, not model answer quality.
The four-score Judge is deliberately optimistic; semantic review stays pending.
"""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result
from evaluation.task_runtime import build_runtime, execute_case
from evaluation.task_transport import CURRENT, BoundClient


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("controlled runtime tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    # Test defaults rather than machine-local tuning. No dotenv is loaded.
    for key in tuple(os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


def case(case_id):
    cases = json.loads((ROOT / "evaluation/benchmark/task-e2e-v1.json").read_text())
    return next(value for value in cases["cases"] if value["id"] == case_id)


class ControlledData:
    """Pure fixture adapter; never mistaken for actual MySQL or Chroma."""

    def __init__(self, *, fail_order=False):
        self.proof = {"kind": "controlled-fixture"}
        self.fixture = load_fixture()
        self.documents = json.loads((ROOT / "knowledge/serveflow-demo-v2.json").read_text())
        self.orders, self.searches = [], []
        self.fault = None
        self.fail_order = fail_order

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def order_reader(self, customer, order):
        self.orders.append((customer, order))
        if self.fail_order:
            self.fault = "controlled_eval_data_failure"
            raise ValueError(self.fault)
        return fixture_order_result(self.fixture, customer, order)

    async def knowledge_handler(self, params, context):
        self.searches.append(deepcopy(params))
        # Six synthetic docs exist. Recall respects the requested limit and puts
        # the relevant delivery document first for this controlled wiring check.
        docs = sorted(self.documents, key=lambda value: value["title"] != "配送说明")
        return [dict(document_id=f"controlled-{index}", chunk=0, score=0.8,
                     version="serveflow-demo-v2", **value)
                for index, value in enumerate(docs[:params["top_k"]])]


class ControlledSDK:
    """Native text/tool_use blocks only; all real execution stays in production."""

    base_url = "https://dashscope.aliyuncs.com/apps/anthropic"

    def __init__(self, *, e05_intent="payment_issue"):
        self.messages = SimpleNamespace(create=self.create)
        self.calls = []
        self.seen = Counter()
        self.e05_intent = e05_intent

    @staticmethod
    def text(value):
        return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")

    async def create(self, **kwargs):
        frame = CURRENT.get()
        assert frame is not None and frame.phase is not None
        self.calls.append((frame, deepcopy(kwargs)))
        key = (frame.case_id, frame.turn_id, frame.phase, frame.agent)
        self.seen[key] += 1
        if frame.phase == "intent":
            intent = {"e02": "refund", "e05": self.e05_intent}.get(frame.case_id, "logistics")
            return self.text(json.dumps({"intent": intent, "confidence": 0.99,
                                         "reasoning": "controlled classification"}))
        if frame.phase == "rag_rewrite":
            return self.text(json.dumps(["设备配件拆单配送", "数字订阅快递单号", "统一到货时间与免邮门槛"]))
        if frame.phase == "rag_rerank":
            return self.text("[0,1,2,3,4]")
        if frame.phase == "judge":
            return self.text(json.dumps(dict.fromkeys(
                ("relevance", "accuracy", "completeness", "helpfulness"), 1.0)))
        if frame.phase == "composer":
            return self.text(f"controlled composed reply {frame.case_id}/{frame.turn_id}")
        assert frame.phase == "agent"
        if self.seen[key] == 1:
            if frame.case_id == "e03":
                name, args = "search_knowledge_base", {"query": "配送与数字订阅的演示政策", "top_k": 1}
            elif frame.case_id == "e05" and frame.agent == "technical":
                name, args = "build_diagnostic_plan", {"environment": "controlled-demo", "reproduced": True}
            else:
                order = {"e01": "DEMO-1001", "e02": "DEMO-1004", "e04": "DEMO-1002", "e05": "DEMO-1003", "e06": "DEMO-1005"}[frame.case_id]
                name, args = "read_demo_order", {"order_id": order}
            # A native model tool request is executed by the actual role loop.
            return SimpleNamespace(content=[{"type": "tool_use", "id": f"tool_{frame.turn_id}_{frame.agent}",
                                              "name": name, "input": args}], stop_reason="tool_use")
        return self.text(f"controlled actual assistant {frame.case_id}/{frame.turn_id}/{frame.agent}")


def wire_text(kwargs):
    return json.dumps(kwargs, ensure_ascii=False)


def native_tool_results(kwargs):
    return [json.loads(block["content"]) for message in kwargs["messages"]
            if isinstance(message.get("content"), list) for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"]


def test_order_tool_executes_real_handler_and_full_judge_is_not_semantic_pass():
    async def run():
        sdk, data = ControlledSDK(), ControlledData()
        first, manager = await build_runtime(sdk, data)
        second, _ = await build_runtime(sdk, data)
        assert first is not second and first._pool is not second._pool
        assert not hasattr(first, "_memory_manager")
        clients = [first._intent_recognizer.client, first._composer._client, manager._client]
        clients += [a._client for pool in first._pool.values() for a in pool]
        assert all(isinstance(c, BoundClient) and c._sdk is sdk for c in clients)
        assert sdk.calls == []  # Building has not recognized or invoked anything.
        output = await execute_case(case("e01"), sdk, data, ensure_healthy=data.assert_healthy)
        assert data.orders == [("demo-customer-a", "DEMO-1001")]
        assert output["data_kind"] == "controlled-fixture"
        assert output["execution_status"] == "success"
        assert output["deterministic"]["blockers"] == []
        tool = output["deterministic"]["evidence"][0]
        assert tool["name"] == "read_demo_order" and tool["tool_use_id"]
        assert tool["result"] == fixture_order_result(data.fixture, "demo-customer-a", "DEMO-1001")
        assert tool["result"]["order"]["shipments"][1]["tracking_id"] is None
        assert output["turns"][0]["judge_overall"] == 1.0 and output["judge_ready"]
        assert output["semantic_review"] is None
        assert output["deterministic"]["semantic_review_required"] is True
        second_actor = [kw for frame, kw in sdk.calls if frame.phase == "agent"][1]
        assert '"type": "tool_result"' in wire_text(second_actor)
        assert "DEMO-TRACK-1001-A" in wire_text(second_actor)
    asyncio.run(run())


def test_trusted_identity_survives_user_spoof_and_cross_owner_not_found_is_usable():
    async def run():
        sdk, data = ControlledSDK(), ControlledData()
        output = await execute_case(case("e06"), sdk, data, ensure_healthy=data.assert_healthy)
        assert data.orders == [("demo-customer-a", "DEMO-1005")]
        assert output["deterministic"]["blockers"] == []
        record = output["deterministic"]["evidence"][0]
        assert record["trusted_customer_id"] == "demo-customer-a"
        assert record["result"]["status"] == "not_found"
        assert record["result"]["order"] is None and record["result_success"] is None
        actor_after_tool = [kw for frame, kw in sdk.calls if frame.phase == "agent"][1]
        native_result = native_tool_results(actor_after_tool)[0]
        assert native_result == fixture_order_result(data.fixture, "demo-customer-a", "DEMO-1005")
        assert native_result["status"] == "not_found" and native_result["order"] is None
        private_names = [row["name"] for row in data.fixture["items"] if row["order_id"] == "DEMO-1005"]
        private_tracks = [row["tracking_id"] for row in data.fixture["shipments"]
                          if row["order_id"] == "DEMO-1005" and row["tracking_id"]]
        assert all(secret not in wire_text(actor_after_tool) for secret in private_names + private_tracks)
    asyncio.run(run())


def test_policy_executes_original_rewrite_recall_rerank_without_gold_in_actor_input():
    async def run():
        sdk, data = ControlledSDK(), ControlledData()
        current = case("e03")
        output = await execute_case(current, sdk, data, ensure_healthy=data.assert_healthy)
        assert not data.orders and len(data.documents) == 6
        phases = [frame.phase for frame, _ in sdk.calls]
        # The real router can select General and Billing for this mixed policy
        # question. Each selected role executes its own RAG call; shared recall
        # caching may collapse the identical queries, never their evidence.
        rag_calls = output["deterministic"]["evidence"]
        assert len(rag_calls) >= 1
        assert phases.count("rag_rewrite") == len(rag_calls)
        assert phases.count("rag_rerank") == len(rag_calls)
        assert 4 <= len(data.searches) <= 4 * len(rag_calls)
        assert all(params["top_k"] == 5 for params in data.searches)
        assert len({params["query"] for params in data.searches}) == 4
        assert output["deterministic"]["blockers"] == []
        for evidence in rag_calls:
            assert evidence["name"] == "search_knowledge_base"
            assert evidence["result"]["success"] and evidence["result"]["reranked"]
            assert len(evidence["result"]["results"]) == 1
            assert evidence["result"]["results"][0]["title"] == "配送说明"
        actor_inputs = "\n".join(wire_text(kw) for frame, kw in sdk.calls if frame.phase in {"intent", "agent"})
        assert "frozen_required_facts" not in actor_inputs
        assert "frozen_forbidden_claims" not in actor_inputs
        assert current["expected"]["required_facts"][0] not in actor_inputs
        judge_inputs = wire_text(next(kw for frame, kw in sdk.calls if frame.phase == "judge"))
        assert "frozen_required_facts" in judge_inputs and current["expected"]["required_facts"][0] in judge_inputs
    asyncio.run(run())


def test_two_actual_turns_carry_produced_assistant_and_structured_order_state():
    async def run():
        sdk, data = ControlledSDK(), ControlledData()
        output = await execute_case(case("e04"), sdk, data, ensure_healthy=data.assert_healthy)
        assert len(output["turns"]) == 2 and output["execution_status"] == "success"
        assert data.orders == [("demo-customer-a", "DEMO-1002")] * 2
        actual_first = output["turns"][0]["result"]["response"]
        assert actual_first.startswith("controlled actual assistant e04/turn-1/")
        second_intent = next(kw for frame, kw in sdk.calls if frame.phase == "intent" and frame.turn_id == "turn-2")
        second_actor = next(kw for frame, kw in sdk.calls if frame.phase == "agent" and frame.turn_id == "turn-2")
        assert actual_first in wire_text(second_intent) and actual_first in wire_text(second_actor)
        assert "DEMO-1002" in wire_text(second_actor)
        assert '"order_id"' in wire_text(second_actor)
        assert output["deterministic"]["blockers"] == []
        assert all(t["judge"]["attempts"] == 1 for t in output["turns"])
        first_judge = next(kw for frame, kw in sdk.calls if frame.phase == "judge" and frame.turn_id == "turn-1")
        assert '"frozen_required_facts":[]' in first_judge["messages"][0]["content"].replace(" ", "")
    asyncio.run(run())


def test_data_fault_cannot_be_hidden_by_actor_fallback_or_produce_judge_score():
    async def run():
        sdk, data = ControlledSDK(), ControlledData(fail_order=True)
        with pytest.raises(ValueError, match="controlled_eval_data_failure"):
            await execute_case(case("e01"), sdk, data, ensure_healthy=data.assert_healthy)
        assert data.orders and data.fault == "controlled_eval_data_failure"
        assert not any(frame.phase == "judge" for frame, _ in sdk.calls)
    asyncio.run(run())


@pytest.mark.parametrize("intent,roles", [("payment_issue", ["billing", "technical"]),
                                         ("technical", ["technical", "billing"])])
def test_paid_activation_uses_billing_records_and_technical_diagnostics(intent, roles):
    """Real production loops with controlled model choices, not quality scores.

    Do not change the question/gold to make missing technical work disappear.
    The actual LLM's classification and response remain unmeasured here.
    """
    async def run():
        sdk, data = ControlledSDK(e05_intent=intent), ControlledData()
        output = await execute_case(case("e05"), sdk, data, ensure_healthy=data.assert_healthy)
        result = output["turns"][0]["result"]
        assert result["intent"] == intent
        assert result["agent_types"] == roles
        assert data.orders == [("demo-customer-a", "DEMO-1003")]
        assert output["deterministic"]["checks"]["business_evidence"]
        assert all(output["deterministic"]["checks"].values())
        assert output["deterministic"]["blockers"] == []
        observed = output["turns"][0]["tool_evidence"]["calls"]
        assert {(call["agent"], call["name"]) for call in observed} == {
            ("billing", "read_demo_order"), ("technical", "build_diagnostic_plan")}
        native = {frame.agent: native_tool_results(arguments)
                  for frame, arguments in sdk.calls if frame.phase == "agent" and native_tool_results(arguments)}
        assert native["billing"][0]["order"]["payments"][0]["amount_minor"] == 3900
        assert "diagnostic_steps" in native["technical"][0]
        assert "order" not in native["technical"][0]
        assert any(frame.phase == "composer" for frame, _ in sdk.calls)
        assert output["turns"][0]["judge_overall"] == 1.0
        assert output["semantic_review"] is None  # Controlled Judge 1.0 is not semantic acceptance.
    asyncio.run(run())


@pytest.mark.parametrize('case_id,failed_role', [('e01','general'),('e02','billing')])
def test_bad_capture_result_stops_native_loop_fallback_and_judge(case_id, failed_role):
    from evaluation.task_evidence import EvidenceError
    class NonJSONData(ControlledData):
        async def order_reader(self, customer, order):
            value = await super().order_reader(customer, order)
            value['order']['entitlements'] = ()
            return value
    async def run():
        sdk, data = ControlledSDK(), NonJSONData()
        with pytest.raises(EvidenceError,match='evidence_invalid_json'):
            await execute_case(case(case_id),sdk,data,ensure_healthy=data.assert_healthy)
        assert [(frame.phase,frame.agent) for frame,_ in sdk.calls] == [('intent',None),('agent',failed_role)]
        assert len(data.orders)==1
        assert CURRENT.get() is None
    asyncio.run(run())


def test_rag_fallback_cannot_swallow_shared_evidence_fault_and_call_reranker():
    from copy import deepcopy
    from evaluation.task_evidence import EvidenceError
    class FaultingRAGSDK(ControlledSDK):
        async def create(self,**kwargs):
            frame=CURRENT.get()
            if frame.phase=='rag_rewrite':
                self.calls.append((frame,deepcopy(kwargs)))
                owner=frame.evidence_guard.__self__
                owner._block('evidence_invalid_json')
                raise RuntimeError('controlled rewriting failure')
            return await super().create(**kwargs)
    async def run():
        sdk,data=FaultingRAGSDK(),ControlledData()
        with pytest.raises(EvidenceError,match='evidence_invalid_json'):
            await execute_case(case('e03'),sdk,data,ensure_healthy=data.assert_healthy)
        phases=[frame.phase for frame,_ in sdk.calls]
        assert phases.count('intent')==1 and phases.count('agent')==1 and phases.count('rag_rewrite')==1
        assert 'rag_rerank' not in phases and 'composer' not in phases and 'judge' not in phases
        assert CURRENT.get() is None
    asyncio.run(run())


def test_real_native_bad_parameter_correction_is_not_a_capture_execution_fault():
    import hashlib
    from evaluation.task_evidence import _clone
    from types import SimpleNamespace
    class RecoveringSDK(ControlledSDK):
        def __init__(self):super().__init__();self.actor_count=0
        async def create(self,**kwargs):
            frame=CURRENT.get()
            if frame.phase!='agent':return await super().create(**kwargs)
            self.calls.append((frame,kwargs));self.actor_count+=1
            if self.actor_count<=2:
                return SimpleNamespace(content=[{'type':'tool_use','id':'controlled_'+str(self.actor_count),
                    'name':'read_demo_order','input':{'order_id':123 if self.actor_count==1 else 'DEMO-1001'}}],stop_reason='tool_use')
            return self.text('Controlled recovered reply; semantic quality remains unmeasured.')
    async def run():
        sdk,data=RecoveringSDK(),ControlledData()
        result=await execute_case(case('e01'),sdk,data,ensure_healthy=data.assert_healthy)
        assert sdk.actor_count==3 and len(data.orders)==1
        assert result['judge_ready'] and result['semantic_review'] is None
        observed=result['turns'][0]['tool_evidence']
        assert len(observed['calls'])==1 and observed['calls'][0]['tool_use_id']=='controlled_2'
        assert observed['events'][0]['event']=='model_tool_input_rejected'
        assert observed['events'][0]['input_sha256']==hashlib.sha256(_clone({'order_id':123})[1]).hexdigest()
        assert 'input' not in observed['events'][0]
        assert observed['blockers']==['tool_input_invalid']
        assert not result['deterministic']['checks']['trace_integrity']  # Ordinary sample failure is retained.
        assert any(frame.phase=='judge' for frame,_ in sdk.calls)
    asyncio.run(run())
