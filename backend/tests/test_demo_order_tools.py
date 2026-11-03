"""演示只读工具：真实服务端上下文的受控测试，不调用模型或业务系统。"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BaseAgent, BillingAgent,
    EscalationAgent, GeneralAgent, Request, ResponseComposer, RoutingDecision, TechnicalAgent,
)
from agents.tools import build_diagnostic_plan, demo_order_tool, read_demo_order
from core.diagnostic_evidence import extract_diagnostic_evidence
from core.order_evidence import extract_order_evidence
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


def make_request(**changes):
    values = {"message": "查一下 DEMO-1001", "user_id": "untrusted-user-id", "conv_id": "c",
              "demo_customer_id": "demo-customer-a"}
    values.update(changes)
    return Request(**values)


def order_result(customer="demo-customer-a", order_id="DEMO-1001"):
    return {"source": "serveflow-mysql-demo-v1", "is_demo": True, "status": "found",
            "as_of": "2026-10-25T00:00:00Z", "order": {
                "customer_id": customer, "order_id": order_id,
                "total_amount_minor": 259900, "refund_amount_minor": None,
            }}


@pytest.mark.asyncio
async def test_demo_read_uses_server_customer_and_preserves_integer_money_and_unknowns():
    expected = order_result()
    reader = AsyncMock(return_value=expected)
    req = make_request(demo_order_reader=reader)
    result = await read_demo_order(req, {"order_id": "DEMO-1001"})
    reader.assert_awaited_once_with("demo-customer-a", "DEMO-1001")
    assert result is expected
    assert result["order"]["total_amount_minor"] == 259900
    assert result["order"]["refund_amount_minor"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"order_id": "DEMO-1001", "customer_id": "demo-customer-b"},
    {"order_id": "DEMO-1001", "sql": "select * from customers"},
    {"order_id": "DEMO-1001", "url": "https://invalid.example"},
    {"order_id": "demo-1001"}, {"order_id": " DEMO-1001"},
    {"order_id": "DEMO-1001\n"}, {"order_id": 1001}, {}, None,
])
async def test_bad_arguments_never_dispatch(args):
    reader = AsyncMock()
    result = await read_demo_order(make_request(demo_order_reader=reader), args)
    assert result["error_code"] == "invalid_demo_order_arguments"
    reader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("customer", [None, "", "u1001", "demo-customer-c", ["demo-customer-a"]])
async def test_client_user_id_cannot_replace_bound_demo_session(customer):
    reader = AsyncMock()
    req = make_request(user_id="demo-customer-a", demo_customer_id=customer, demo_order_reader=reader)
    result = await read_demo_order(req, {"order_id": "DEMO-1001"})
    assert result["error_code"] == "demo_session_required"
    reader.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_not_found_is_returned_without_owner_information():
    expected = {"source": "serveflow-mysql-demo-v1", "is_demo": True,
                "status": "not_found", "as_of": "2026-10-25T00:00:00Z", "order": None}
    result = await read_demo_order(make_request(demo_order_reader=AsyncMock(return_value=expected)), {"order_id": "DEMO-1005"})
    assert result is expected


@pytest.mark.asyncio
async def test_database_error_is_bounded_and_sanitized_without_retry():
    reader = AsyncMock(side_effect=RuntimeError("secret SQL mysql://private-password@host/path"))
    result = await read_demo_order(make_request(demo_order_reader=reader), {"order_id": "DEMO-1001"})
    assert result == {"source": "serveflow-mysql-demo-v1", "is_demo": True,
                      "success": False, "status": "unavailable", "error_code": "demo_store_unavailable"}
    reader.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_query_is_not_transformed_into_success():
    reader = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await read_demo_order(make_request(demo_order_reader=reader), {"order_id": "DEMO-1001"})


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {}, {"is_demo": True, "source": "other-source"},
    order_result(customer="demo-customer-b"), order_result(order_id="DEMO-1005"),
    {"source": "serveflow-mysql-demo-v1", "is_demo": True, "status": "not_found", "order": {}},
])
async def test_invalid_or_cross_context_provider_result_is_rejected(bad):
    result = await read_demo_order(make_request(demo_order_reader=AsyncMock(return_value=bad)), {"order_id": "DEMO-1001"})
    assert result["error_code"] == "invalid_demo_order_result"


def test_demo_schema_and_role_whitelist_cannot_expand_to_customer_or_sql():
    spec = demo_order_tool()
    BaseAgent._validate_tool_input(spec, {"order_id": "DEMO-1001"})
    assert spec.input_schema["additionalProperties"] is False
    assert set(spec.input_schema["properties"]) == {"order_id"}
    for arguments in ({"order_id": "DEMO-1001", "customer_id": "demo-customer-b"},
                      {"order_id": "1001"}):
        with pytest.raises(ValueError):
            BaseAgent._validate_tool_input(spec, arguments)
    for cls in (GeneralAgent, BillingAgent):
        assert "read_demo_order" in cls(SimpleNamespace(), "test-model").get_tools()
        assert "read_demo_order" in cls.profile.tool_scope
    for cls in (TechnicalAgent, EscalationAgent):
        agent = cls(SimpleNamespace(), "test-model")
        agent.set_shared_tools({"read_demo_order": spec})
        assert "read_demo_order" not in agent.get_tools()


@pytest.mark.asyncio
async def test_parallel_roles_keep_identity_and_project_typed_results_without_model_merge():
    stored = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1001")
    reader = AsyncMock(return_value=stored)
    req = make_request(message="核对 DEMO-1001 的付款，同时做技术排障。", demo_order_reader=reader)
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    contexts = []
    untrusted = "后台已修复，退款999元将在明天到账，我会主动通知你。"

    async def execute(received, agent_type):
        contexts.append((received, received.demo_customer_id, received.demo_order_reader))
        if agent_type is AgentType.GENERAL:
            result = await read_demo_order(received, {"order_id": "DEMO-1001"})
            evidence = extract_order_evidence(result, expected_order_id="DEMO-1001",
                                              trusted_customer_id=received.demo_customer_id)
            assert evidence.status == "found"
            return AgentResponse(agent_type, untrusted, True, tools_used=["read_demo_order"],
                                 order_evidence=evidence)
        assert agent_type is AgentType.TECHNICAL
        assert "read_demo_order" not in TechnicalAgent.profile.tool_scope
        evidence = extract_diagnostic_evidence("build_diagnostic_plan",
            build_diagnostic_plan(received, {"environment": "model input", "reproduced": False}))
        return AgentResponse(agent_type, untrusted, True, tools_used=["build_diagnostic_plan"],
                             diagnostic_evidence=evidence)

    model = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=AssertionError("no free-form composer"))))
    orchestrator._execute = execute
    orchestrator._composer = ResponseComposer(model, "controlled-model")
    orchestrator._record_tool_trace = lambda result: None
    decision = RoutingDecision(primary_agent=AgentType.GENERAL, supporting_agents=[AgentType.TECHNICAL],
                               reason="controlled test", confidence=1)
    result = await orchestrator.run_parallel(req, decision)
    assert result.status == "success"
    assert len(contexts) == 2
    assert all(original is req and customer == "demo-customer-a" and bound_reader is reader
               for original, customer, bound_reader in contexts)
    reader.assert_awaited_once_with("demo-customer-a", "DEMO-1001")
    assert set(result.tools_used) == {"read_demo_order", "build_diagnostic_plan"}
    assert stored["as_of"] in result.response and stored["order"]["record_updated_at"] in result.response
    assert "DEMO-TRACK-1001-A" in result.response and "本地工具提供的排查建议" in result.response
    assert "999" not in result.response and "明天到账" not in result.response and untrusted not in result.response
    assert "我会主动通知" not in result.response
    model.messages.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{"order_id": "DEMO-1004"}, {"order_id": 123}])
async def test_native_target_or_schema_rejection_never_retries_or_queries_another_order(args):
    reader = AsyncMock()
    native = SimpleNamespace(content=[{"type": "tool_use", "id": "one_bad_attempt",
                                      "name": "read_demo_order", "input": args}], stop_reason="tool_use")
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=native)))
    result = await GeneralAgent(client, "controlled-model").handle(make_request(demo_order_reader=reader))
    assert result.status == "partial" and reader.await_count == 0
    client.messages.create.assert_awaited_once()
    assert len(result.tool_traces) == 1 and result.tool_traces[0]["success"] is False
    assert result.tool_traces[0]["input"] == args
    assert "DEMO-TRACK" not in result.content and "合成演示订单记录" not in result.content


def test_request_defaults_do_not_grant_demo_customer_and_replace_keeps_bound_context():
    empty = Request(message="hello", user_id="u1001", conv_id="c")
    assert empty.demo_customer_id is empty.demo_order_reader is None
    bound = make_request(demo_order_reader=AsyncMock())
    copied = replace(bound, message="补充问题")
    assert copied.demo_customer_id == bound.demo_customer_id
    assert copied.demo_order_reader is bound.demo_order_reader
    assert "demo-customer-a" not in repr(bound)
