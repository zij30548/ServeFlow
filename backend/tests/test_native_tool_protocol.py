"""Native response-boundary checks with mocked providers; no model or DB calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx
from anthropic import APITimeoutError

from agents.agent_orchestrator import GeneralAgent, Request
from agents.tools import make_tool


def tool_block(*, tool_id="toolu_valid", arguments=None):
    return {"type": "tool_use", "id": tool_id, "name": "probe",
            "input": {} if arguments is None else arguments}


def response(blocks, stop="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


def setup_agent(responses):
    provider = AsyncMock(side_effect=responses)
    handler = AsyncMock(return_value={"success": True})
    agent = GeneralAgent(SimpleNamespace(messages=SimpleNamespace(create=provider)), "controlled-model")
    agent.set_shared_tools({"probe": make_tool("probe", "Controlled read", {}, handler)})
    req = Request(message="controlled query", user_id="isolated", conv_id="controlled")
    return agent, provider, handler, req


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [None, [], 0, "", False, ["value"], "{}"])
async def test_non_object_input_rejected_without_dispatch_or_repair(arguments):
    block = tool_block()
    block["input"] = arguments
    agent, provider, handler, req = setup_agent([response([block])])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_protocol_error"
    assert result.tools_used == result.tool_traces == req.tool_attempts == []
    handler.assert_not_called()
    assert provider.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_id", [None, "", " ", "tool with space", "bad\n", "x" * 201, 7, "工具"])
async def test_invalid_id_rejected_before_dispatch(tool_id):
    agent, provider, handler, req = setup_agent([response([tool_block(tool_id=tool_id)])])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_protocol_error"
    assert result.tools_used == req.tool_attempts == []
    handler.assert_not_called()
    assert provider.await_count == 1


@pytest.mark.asyncio
async def test_missing_id_rejected_before_dispatch():
    block = tool_block()
    del block["id"]
    agent, provider, handler, req = setup_agent([response([block])])
    result = await agent.handle(req)
    assert result.error_code == "agent_protocol_error"
    handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [
    tool_block(tool_id="toolu_valid"), tool_block(tool_id=None),
    tool_block(tool_id="toolu_second", arguments=[]),
    {**tool_block(tool_id="toolu_second"), "input": None},
])
async def test_entire_batch_checked_before_first_handler(second):
    agent, provider, handler, req = setup_agent([response([tool_block(), second])])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_protocol_error"
    assert result.tools_used == result.tool_traces == req.tool_attempts == []
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_id_in_later_round_preserves_only_prior_execution():
    agent, provider, handler, req = setup_agent([response([tool_block()]), response([tool_block()])])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_protocol_error"
    assert provider.await_count == 2 and handler.await_count == 1
    assert result.tools_used == ["probe"]
    assert [trace["tool_use_id"] for trace in result.tool_traces] == ["toolu_valid"]
    assert len(req.tool_attempts) == 1


@pytest.mark.asyncio
async def test_truncated_tool_response_never_dispatches():
    agent, provider, handler, req = setup_agent([response([tool_block()], stop="max_tokens")])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_protocol_error"
    assert result.tools_used == result.tool_traces == req.tool_attempts == []
    handler.assert_not_called()
    assert provider.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_object", [False, True])
@pytest.mark.parametrize("tool_id", ["toolu_valid", "a" * 200])
async def test_valid_dictionary_and_sdk_blocks_keep_native_feedback(sdk_object, tool_id):
    block = tool_block(tool_id=tool_id)
    if sdk_object:
        block = SimpleNamespace(**block)
    agent, provider, handler, req = setup_agent([
        response([block]), response([{"type": "text", "text": "Controlled answer"}], stop="end_turn")])
    result = await agent.handle(req)
    assert result.success and result.status == "success" and result.tools_used == ["probe"]
    assert provider.await_count == 2
    handler.assert_awaited_once_with(req, {})
    feedback = provider.await_args_list[1].kwargs["messages"][-1]["content"][0]
    assert feedback["type"] == "tool_result" and feedback["tool_use_id"] == tool_id
    assert feedback["is_error"] is False


@pytest.mark.asyncio
async def test_text_only_truncation_keeps_partial_semantics():
    agent, provider, handler, req = setup_agent([
        response([{"type": "text", "text": "Incomplete text"}], stop="max_tokens")])
    result = await agent.handle(req)
    assert result.success and result.status == "partial" and result.error_code is None
    assert agent.stats.success == 0
    handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    TimeoutError("controlled timeout"),
    APITimeoutError(request=httpx.Request("POST", "https://controlled.invalid/")),
])
async def test_provider_timeout_is_classified_without_tool_dispatch(error):
    agent, provider, handler, req = setup_agent([error])
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_timeout"
    assert result.tools_used == result.tool_traces == req.tool_attempts == []
    handler.assert_not_called()
    assert provider.await_count == 1


@pytest.mark.asyncio
async def test_task_cancellation_is_not_changed_into_a_model_error():
    agent, provider, handler, req = setup_agent([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await agent.handle(req)
    handler.assert_not_called()
