"""演示只读工具：真实服务端上下文的受控测试，不调用模型或业务系统。"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BaseAgent, BillingAgent,
    EscalationAgent, GeneralAgent, Request, RoutingDecision, TechnicalAgent,
)
from agents.tools import demo_order_tool, read_demo_order


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
async def test_parallel_roles_receive_same_trusted_context_and_composer_keeps_results():
    reader = AsyncMock(return_value=order_result())
    req = make_request(demo_order_reader=reader)
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    contexts = []

    async def execute(received, agent_type):
        contexts.append((received, received.demo_customer_id, received.demo_order_reader))
        result = await read_demo_order(received, {"order_id": "DEMO-1001"})
        assert result["status"] == "found"
        return AgentResponse(agent_type, "演示订单已核对", True, tools_used=["read_demo_order"])

    orchestrator._execute = execute
    orchestrator._composer = SimpleNamespace(compose_result=AsyncMock(return_value=SimpleNamespace(content="演示结果", status="success")))
    orchestrator._record_tool_trace = lambda result: None
    decision = RoutingDecision(primary_agent=AgentType.GENERAL, supporting_agents=[AgentType.BILLING], reason="controlled test", confidence=1)
    result = await orchestrator.run_parallel(req, decision)
    assert result.status == "success"
    assert len(contexts) == 2
    assert all(original is req and customer == "demo-customer-a" and bound_reader is reader
               for original, customer, bound_reader in contexts)
    assert "read_demo_order" in result.tools_used


def test_request_defaults_do_not_grant_demo_customer_and_replace_keeps_bound_context():
    empty = Request(message="hello", user_id="u1001", conv_id="c")
    assert empty.demo_customer_id is empty.demo_order_reader is None
    bound = make_request(demo_order_reader=AsyncMock())
    copied = replace(bound, message="补充问题")
    assert copied.demo_customer_id == bound.demo_customer_id
    assert copied.demo_order_reader is bound.demo_order_reader
    assert "demo-customer-a" not in repr(bound)
