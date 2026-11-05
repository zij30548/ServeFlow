"""Controlled persistence checks; no model, database or network service is used."""
import asyncio
from collections import deque
from copy import deepcopy
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
from anthropic import AsyncAnthropic
import httpx
import pytest
import pytest_asyncio

from core.concurrency import KeyedLocks
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from memory.conversation_memory import MemoryContext, MemoryManager


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("conversation persistence tests forbid network services")
    for method in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def response(text, *, stop="end_turn"):
    return SimpleNamespace(content=[{"type": "text", "text": text}], stop_reason=stop)


class MemoryCollection:
    """Small Chroma-shaped store without embeddings, disk or network access."""

    def __init__(self):
        self.rows = {}

    def upsert(self, *, ids, documents, metadatas):
        self.rows.update({identity: (document, metadata)
                          for identity, document, metadata in zip(ids, documents, metadatas)})

    add = upsert

    def get(self, *, ids=None, where=None):
        rows = [value for identity, value in self.rows.items()
                if (ids is None or identity in ids) and self.matches(value[1], where)]
        return {"documents": [row[0] for row in rows], "metadatas": [row[1] for row in rows]}

    def query(self, *, query_texts, n_results, where):
        rows = self.get(where=where)
        return {key: [values[:n_results]] for key, values in rows.items()}

    @classmethod
    def matches(cls, metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(cls.matches(metadata, child) for child in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())


@pytest_asyncio.fixture
async def memory():
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    manager._request_locks = KeyedLocks()
    manager._episodic, manager._profile = MemoryCollection(), MemoryCollection()
    manager._model = "controlled-model"
    manager._client = SimpleNamespace(messages=SimpleNamespace(
        create=AsyncMock(return_value=response("保留未解决的问题，记录不是本次查询结果。"))))
    yield manager
    await manager._redis.aclose()


async def commit(memory, number, *, user="u", conv="c", status="success", message=None):
    request_id = f"runtime-{number}"
    result = {"status": status, "response": f"reply-{number}", "request_id": request_id}
    await memory.commit_turn(user, conv, f"client-{number}", message or f"message-{number}", result,
                             state={"sources": {"order_id": request_id}})


@pytest.mark.asyncio
async def test_compression_preserves_both_messages_of_the_oldest_retained_turn(memory):
    for number in range(8):
        await commit(memory, number)
    await memory._compress("u", "c")

    messages = await memory._get_working_memory("u", "c")
    assert [message.content for message in messages] == [
        "message-5", "reply-5", "message-6", "reply-6", "message-7", "reply-7"]
    assert messages[0].metadata["request_id"] == messages[1].metadata["request_id"]
    stored = memory._episodic.get(where={"user_id": "u", "conv_id": "c"})
    assert len(stored["documents"]) == 1
    assert "message-5" not in stored["metadatas"][0]["full_text"]
    original = (await memory.get_context("u", "c", query="history")).reference_history()
    assert [event["content"] for event in original] == [f"message-{number}" for number in range(8)]
    assert original[0] == {"version": 1, "source": "user_message", "role": "user", "content": "message-0",
                           "client_request_id": "client-0", "request_id": "runtime-0", "status": "success"}


@pytest.mark.asyncio
async def test_compression_keeps_a_complete_turn_committed_during_summarization(memory):
    for number in range(8):
        await commit(memory, number)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def summarize(**kwargs):
        entered.set()
        await finish.wait()
        return response("旧轮次摘要")

    memory._client.messages.create.side_effect = summarize
    pending = asyncio.create_task(memory._compress("u", "c"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await commit(memory, 8)
        finish.set()
        await asyncio.wait_for(pending, 1)
        messages = await memory._get_working_memory("u", "c")
        assert [message.content for message in messages] == [
            "message-5", "reply-5", "message-6", "reply-6", "message-7", "reply-7", "message-8", "reply-8"]
        assert len(memory._episodic.rows) == 1
    finally:
        finish.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_truncated_merged_summary_keeps_both_existing_and_new_summary(memory):
    memory._client.messages.create.return_value = response("旧摘要的残缺改写", stop="max_tokens")
    assert await memory._merge_summary("旧摘要：尚未核实", "新摘要：用户纠正了信息") == (
        "旧摘要：尚未核实\n新摘要：用户纠正了信息")
    memory._client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_truncated_profile_does_not_replace_existing_profile(memory):
    memory._profile.upsert(ids=[memory._profile_doc_id("u")],
        documents=[json.dumps({"preferences": ["原偏好"]})], metadatas=[{"user_id": "u"}])
    before = dict(memory._profile.rows)
    await commit(memory, 1)
    memory._client.messages.create.return_value = response('{"preferences":["不完整的新偏好"]}', stop="max_tokens")
    await memory.update_profile("u", "c")
    assert memory._profile.rows == before
    memory._client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_turn_has_receipt_but_cannot_change_existing_history_or_state(memory):
    await commit(memory, 1)
    before = await memory.get_context("u", "c", query="follow-up")
    await commit(memory, 2, status="failed", message="错误的新订单 DEMO-1005")
    after = await memory.get_context("u", "c", query="follow-up")
    assert after.recent_messages == before.recent_messages
    assert after.state == before.state
    assert (await memory.get_turn_receipt("u", "c", "client-2"))["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_actual_memory_roundtrip_is_scoped_by_user_and_conversation(memory):
    await commit(memory, 1, user="customer-a", conv="same-conversation", message="我的原始订单 DEMO-1001")
    await commit(memory, 2, user="customer-b", conv="same-conversation", message="另一客户 DEMO-1005")
    await commit(memory, 3, user="customer-a", conv="other-conversation", message="另一个会话 DEMO-1002")
    context = await memory.get_context("customer-a", "same-conversation", query="订单")
    assert [item.content for item in context.recent_messages] == ["我的原始订单 DEMO-1001", "reply-1"]
    assert context.state["sources"]["order_id"] == "runtime-1"
    assert "DEMO-1005" not in context.to_prompt_text()
    assert "DEMO-1002" not in context.to_prompt_text()
    assert [row["content"] for row in context.reference_history()] == ["我的原始订单 DEMO-1001"]


@pytest.mark.asyncio
async def test_source_history_survives_long_chat_and_has_the_working_memory_ttl(memory):
    for number in range(24):
        await commit(memory, number, status="partial" if number == 4 else "success")
    await memory._compress("u", "c")
    context = await memory.get_context("u", "c", query="history")
    assert len(context.recent_messages) == 6
    events = context.reference_history()
    assert [row["content"] for row in events] == [f"message-{number}" for number in range(4, 24)]
    assert events[0]["status"] == "partial"
    working_ttl = await memory._redis.ttl(memory._wm_key("u", "c"))
    source_ttl = await memory._redis.ttl(memory._source_history_key("u", "c"))
    assert 86395 <= source_ttl <= working_ttl <= 86400


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(source="model_summary"),
    lambda rows: rows[0].update(version=True),
    lambda rows: rows[0].update(role="assistant"),
    lambda rows: rows[0].update(status="failed"),
    lambda rows: rows[0].update(status={}),
    lambda rows: rows[0].update(content=""),
    lambda rows: rows[0].update(client_request_id=""),
    lambda rows: rows[0].update(request_id=3),
    lambda rows: rows[0].update(trusted_customer_id="demo-customer-a"),
    lambda rows: rows.extend(deepcopy(rows) * 20),
])
def test_source_history_rejects_corrupt_events_without_reviving_older_input(mutation):
    events = [{"version": 1, "source": "user_message", "role": "user", "content": "原始输入",
               "client_request_id": "client-1", "request_id": "runtime-1", "status": "success"}]
    mutation(events)
    context = MemoryContext([], ["历史DEMO-1005"], {"order_id": "DEMO-1005"}, "摘要DEMO-1005",
                            state={"entities": {"order_id": ["DEMO-1005"]}}, source_history=events)
    assert context.reference_history() == []


@pytest.mark.asyncio
async def test_corrupt_source_storage_does_not_recover_events_from_other_memory_tiers(memory):
    await commit(memory, 1, message="查询DEMO-1001")
    key = memory._source_history_key("u", "c")
    await memory._redis.lpush(key, "broken-json")
    await memory._redis.set(memory._summary_key("u", "c"), "查询过DEMO-1001")
    context = await memory.get_context("u", "c", query="订单")
    assert context.recent_messages and context.summary and context.state
    assert context.reference_history() == []


@pytest_asyncio.fixture
async def http_memory(memory, monkeypatch):
    """HTTP -> actual MemoryManager -> actual agent -> SDK memory transport."""
    import api.main as api
    from agents.agent_orchestrator import AgentOrchestrator, AgentType, GeneralAgent, ResponseComposer

    monkeypatch.setenv("ECHOMIND_ENV", "development")
    monkeypatch.delenv("ECHOMIND_API_TOKENS", raising=False)
    payloads, control = [], {"mode": "reply"}
    entered, finish = asyncio.Event(), asyncio.Event()

    async def send(request):
        payloads.append(json.loads(request.content))
        if control["mode"] == "cancel":
            entered.set()
            await finish.wait()
        if control["mode"] == "fail":
            return httpx.Response(503, json={"error": {"type": "overloaded_error", "message": "controlled"}})
        return httpx.Response(200, json={"id": "controlled-sdk-response", "type": "message",
            "role": "assistant", "model": "controlled-model",
            "content": [{"type": "text", "text": "受控回复：已了解您的表达偏好。"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as provider:
        sdk = AsyncAnthropic(api_key="controlled-not-a-real-key", base_url="https://controlled.invalid",
                            http_client=provider, max_retries=0)
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._pool = {AgentType.GENERAL: [GeneralAgent(sdk, "controlled-model")]}
        orchestrator._composer = ResponseComposer(sdk, "controlled-model")
        orchestrator._recent_tool_traces = deque(maxlen=50)
        orchestrator.recognize_intent = AsyncMock(return_value=IntentResult(
            IntentCategory.REQUEST, .99, UrgencyLevel.LOW, "request", {}, "controlled routing", 0))
        monkeypatch.setattr(api, "_memory", memory)
        monkeypatch.setattr(api, "_orchestrator", orchestrator)
        monkeypatch.setattr(api, "_profile_updater", None)
        monkeypatch.setattr(api, "_monitor", None)
        monkeypatch.setattr(api, "_demo_store", None)
        monkeypatch.setattr(api, "_demo_sessions", None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            yield SimpleNamespace(http=http, memory=memory, payloads=payloads, control=control,
                                  entered=entered, finish=finish, orchestrator=orchestrator)
        await sdk.close()


def turn(message, number, *, user="u", conv="c"):
    return {"message": message, "client_request_id": f"http-{number}", "user_id": user, "conv_id": conv}


@pytest.mark.asyncio
async def test_http_next_turn_reads_committed_memory_without_injected_history(http_memory):
    service = http_memory
    first = await service.http.post("/chat", json=turn("请用中文和我沟通", 1))
    assert first.status_code == 200
    persisted = await service.memory._get_working_memory("u", "c")
    assert [message.content for message in persisted] == ["请用中文和我沟通", first.json()["response"]]
    second = await service.http.post("/chat", json=turn("请继续沿用我的表达偏好", 2))
    assert second.status_code == 200
    history = service.orchestrator.recognize_intent.call_args.kwargs["history"]
    assert [message["content"] for message in history] == [item.content for item in persisted]
    background = service.payloads[1]["messages"][0]["content"]
    assert "请用中文和我沟通" in background and first.json()["response"] in background
    assert len(service.payloads) == 2
    context = await service.memory.get_context("u", "c", query="偏好")
    assert [row["content"] for row in context.reference_history()] == [
        "请用中文和我沟通", "请继续沿用我的表达偏好"]
    assert context.reference_history()[0]["request_id"] == first.json()["request_id"]
    assert context.reference_history()[1]["request_id"] == second.json()["request_id"]


@pytest.mark.asyncio
async def test_http_idempotent_retry_has_one_original_source_event(http_memory):
    service = http_memory
    submitted = turn("请用中文和我沟通", 1)
    first = await service.http.post("/chat", json=submitted)
    repeated = await service.http.post("/chat", json=submitted)
    assert repeated.status_code == 200 and repeated.json() == first.json()
    assert len(service.payloads) == 1
    events = (await service.memory.get_context("u", "c", query="偏好")).reference_history()
    assert len(events) == 1 and events[0]["client_request_id"] == "http-1"
    conflict = await service.http.post("/chat", json=turn("换成不同的原始消息", 1))
    assert conflict.status_code == 409
    assert (await service.memory.get_context("u", "c", query="偏好")).reference_history() == events


@pytest.mark.asyncio
async def test_http_source_events_follow_authenticated_identity_and_conversation(http_memory, monkeypatch):
    service = http_memory
    token_a, token_b = "controlled-token-a-at-least-24-chars", "controlled-token-b-at-least-24-chars"
    monkeypatch.setenv("ECHOMIND_API_TOKENS", json.dumps({
        token_a: {"user_id": "bound-a", "role": "user"},
        token_b: {"user_id": "bound-b", "role": "user"},
    }))
    headers_a, headers_b = {"Authorization": "Bearer " + token_a}, {"Authorization": "Bearer " + token_b}
    first = await service.http.post("/chat", json=turn("客户A的原始偏好", 1, user="forged"), headers=headers_a)
    assert first.status_code == 200
    other_conversation = await service.http.post("/chat", json=turn("A的新会话", 2, user="forged", conv="other"),
                                                 headers=headers_a)
    other_identity = await service.http.post("/chat", json=turn("客户B的原始偏好", 3, user="bound-a"), headers=headers_b)
    assert other_conversation.status_code == other_identity.status_code == 200
    assert "客户A的原始偏好" not in json.dumps(service.payloads[1:], ensure_ascii=False)
    for user, conv, expected in (("bound-a", "c", "客户A的原始偏好"),
                                 ("bound-a", "other", "A的新会话"),
                                 ("bound-b", "c", "客户B的原始偏好")):
        context = await service.memory.get_context(user, conv, query="偏好")
        assert [row["content"] for row in context.reference_history()] == [expected]
        returned = context.reference_history()
        returned[0]["content"] = "changed-return-value"
        assert context.reference_history()[0]["content"] == expected
    assert not await service.memory._redis.exists(service.memory._source_history_key("forged", "c"))


@pytest.mark.asyncio
async def test_http_provider_failure_does_not_enter_the_next_turn_history(http_memory):
    service = http_memory
    first = await service.http.post("/chat", json=turn("请用中文和我沟通", 1))
    before = deepcopy(await service.memory.get_context("u", "c", query="偏好"))
    service.control["mode"] = "fail"
    failed = await service.http.post("/chat", json=turn("本轮失败的表达偏好是英文", 2))
    assert failed.status_code == 503 and failed.json()["status"] == "failed"
    assert (await service.memory.get_context("u", "c", query="偏好")) == before
    service.control["mode"] = "reply"
    resumed = await service.http.post("/chat", json=turn("继续刚才成功的对话", 3))
    assert resumed.status_code == 200
    history = service.orchestrator.recognize_intent.call_args.kwargs["history"]
    assert [item["content"] for item in history] == ["请用中文和我沟通", first.json()["response"]]


@pytest.mark.asyncio
async def test_http_cancelled_provider_wait_does_not_commit_history_state_or_receipt(http_memory):
    service = http_memory
    await service.http.post("/chat", json=turn("请用中文和我沟通", 1))
    before = await service.memory.get_context("u", "c", query="偏好")
    service.control["mode"] = "cancel"
    pending = asyncio.create_task(service.http.post("/chat", json=turn("本轮将被取消", 2)))
    await asyncio.wait_for(service.entered.wait(), 1)
    pending.cancel()
    await asyncio.sleep(0)
    service.finish.set()
    result = await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    assert await service.memory.get_turn_receipt("u", "c", "http-2") is None
    assert (await service.memory.get_context("u", "c", query="偏好")) == before
    assert not await service.memory._redis.exists("lock:conversation:" + service.memory._conversation_id("u", "c"))
    service.control["mode"] = "reply"
    resumed = await service.http.post("/chat", json=turn("继续成功的对话", 3))
    assert resumed.status_code == 200
    history = service.orchestrator.recognize_intent.call_args.kwargs["history"]
    assert "本轮将被取消" not in str(history)
