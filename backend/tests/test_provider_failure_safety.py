"""Private provider errors cannot become user answers or LLM failure logs."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import APIStatusError
import fakeredis.aioredis
import httpx
import pytest

from agents.agent_orchestrator import AgentCallError, GeneralAgent, Request
from memory.conversation_memory import MemoryManager, Message, MsgRole
from tooling.tool_manager import MCPToolManager


PRIVATE = "private-provider-body-prompt-and-cookie"


def private_error():
    request = httpx.Request("POST", f"https://private.invalid/{PRIVATE}", headers={"Authorization": PRIVATE})
    response = httpx.Response(503, request=request, text=PRIVATE)
    return APIStatusError(PRIVATE, response=response, body={"private": PRIVATE})


def client(error):
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=error)))


def assert_safe(caplog):
    assert PRIVATE not in caplog.text and "private.invalid" not in caplog.text
    assert all(record.exc_info is None and record.exc_text is None for record in caplog.records)


@pytest.mark.asyncio
async def test_agent_generic_provider_error_is_fixed_and_logs_only_safe_code(caplog):
    agent = GeneralAgent(client(private_error()), "controlled-model")
    req = Request(message=PRIVATE, user_id="isolated", conv_id="controlled")
    with pytest.raises(AgentCallError) as caught:
        await agent._call_llm(req)
    assert str(caught.value) == "模型调用未完成"
    assert PRIVATE not in str(caught.value)
    result = await agent.handle(req)
    assert not result.success and result.error_code == "agent_failed"
    assert PRIVATE not in result.content and "error_code=agent_failed" in caplog.text
    assert_safe(caplog)


@pytest.mark.asyncio
async def test_untrusted_error_class_and_code_are_not_logged(caplog):
    class UnknownFailure(Exception):
        def __str__(self):
            raise AssertionError("Never stringify unknown provider failures")
    UnknownFailure.__name__ = PRIVATE
    agent = GeneralAgent(client(UnknownFailure()), "controlled-model")
    result = await agent.handle(Request(message="controlled", user_id="isolated", conv_id="controlled"))
    assert not result.success and result.error_code == "agent_failed"
    assert_safe(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [PRIVATE, {}, [], None])
async def test_unknown_agent_error_codes_are_collapsed_safely(caplog, code):
    agent = GeneralAgent(client(AgentCallError(PRIVATE, [], [], error_code=code)), "controlled-model")
    result = await agent.handle(Request(message="controlled", user_id="isolated", conv_id="controlled"))
    assert not result.success and result.error_code == "agent_failed"
    assert_safe(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["rewrite", "rerank"])
async def test_rag_provider_error_keeps_fallback_without_private_body(caplog, target):
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._client, manager._model = client(private_error()), "controlled-model"
    if target == "rewrite":
        assert await manager.rewrite_query(PRIVATE) == [PRIVATE]
    else:
        assert await manager._rerank_with_status(PRIVATE, ["one", "two"], 1) == (["one"], False)
    assert manager._client.messages.create.await_count == 1
    assert f"phase=retrieval.{target}" in caplog.text and "error_code=provider_status_error" in caplog.text
    assert_safe(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["profile", "summary", "merge"])
async def test_memory_provider_error_keeps_state_and_safe_fallback(caplog, target):
    memory = MemoryManager.__new__(MemoryManager)
    memory._client, memory._model = client(private_error()), "controlled-model"
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._redis = redis
    try:
        if target == "profile":
            memory._get_working_memory = AsyncMock(return_value=[Message(MsgRole.USER, PRIVATE)])
            memory._get_profile = AsyncMock(return_value={})
            memory._profile = SimpleNamespace(upsert=lambda **kwargs: pytest.fail("No profile write after model failure"))
            await memory._update_profile("isolated", "controlled")
        elif target == "summary":
            key = memory._wm_key("isolated", "controlled")
            raw = json.dumps({"role": "user", "content": PRIVATE, "ts": "2026-10-25T00:00:00"})
            await redis.lpush(key, *([raw] * memory.COMPRESS_AT))
            await memory._compress("isolated", "controlled")
            assert await redis.llen(key) == memory.COMPRESS_AT
        else:
            assert await memory._merge_summary("old", "new") == "old\nnew"
        assert memory._client.messages.create.await_count == 1
        assert f"phase=memory.{target}" in caplog.text and "error_code=provider_status_error" in caplog.text
        assert_safe(caplog)
    finally:
        await redis.aclose()
