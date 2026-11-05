"""HTTP persistence and native SDK evidence, with no external service calls."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import re
from types import SimpleNamespace
from unittest.mock import patch

from anthropic import AsyncAnthropic
import fakeredis.aioredis
import httpx
import pytest

with patch("dotenv.load_dotenv", return_value=False):
    import api.main as api
from core.concurrency import KeyedLocks
from core.diagnostic_evidence import select_diagnostic_request
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from core.order_evidence import resolve_order_id, source_bound_order_history
from evaluation import benchmark as bench, coverage_expansion as preparation
from evaluation import coverage_conversation_runtime as runtime
from evaluation.task_transport import CURRENT
from memory.conversation_memory import MemoryManager
from test_conversation_memory import MemoryCollection
from test_coverage_runtime import Data, offline


def cases():
    return bench.read_json(bench.BACKEND_ROOT / preparation.FILES[-1])["conversations"]


class MemoryFactory:
    def __init__(self):
        self.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        self.memory, self.closed = None, False

    @asynccontextmanager
    async def __call__(self, sdk, data, *, namespace, ensure_healthy):
        ensure_healthy()
        memory = MemoryManager.__new__(MemoryManager)
        memory._redis, memory._request_locks = self.redis, KeyedLocks()
        memory._episodic, memory._profile = MemoryCollection(), MemoryCollection()
        memory._client, memory._model = sdk, "qwen3.7-flash-2026-07-15"
        memory._coverage_resource_proof = {"redis": "fakeredis", "episodic_profile": "in-memory Chroma-shaped stores"}
        self.memory = memory
        try:
            yield memory
        finally:
            self.closed = True


class Replies:
    def __init__(self, case, factory, *, fault=None):
        self.case, self.factory, self.fault = case, factory, fault
        self.calls = []

    async def __call__(self, request):
        frame, payload = CURRENT.get(), json.loads(request.content)
        assert frame is not None
        self.calls.append((frame, deepcopy(payload)))
        if self.fault == "bad-json" and len(self.calls) == 1:
            return httpx.Response(200, text="{", headers={"Content-Type": "application/json"})
        if self.fault == "cancel":
            raise asyncio.CancelledError()
        if self.fault == "timeout":
            raise httpx.ReadTimeout("controlled provider wait", request=request)
        number = int(frame.turn_id.split("-")[1])
        message = self.case["turns"][number - 1]["message"]
        if self.fault == "stale-frame" and number == 2 and frame.phase == "intent":
            self.calls[0][0].evidence_guard()
        if frame.phase == "intent":
            raw = '{"intent":"query","confidence":0.99,"reasoning":"controlled routing"}'
        elif frame.phase == "rag_rewrite":
            question = json.loads(payload["messages"][0]["content"].rsplit("\n", 1)[-1])
            raw = json.dumps(question["allowed_units"][:3], ensure_ascii=False)
        elif frame.phase == "rag_rerank":
            raw = json.dumps([int(i) for i in re.findall(r"(?m)^([0-9]+)\. ", payload["messages"][0]["content"])][:3])
        elif frame.phase == "agent":
            if self.fault == "empty-actor":
                return self.response([])
            name = payload.get("tool_choice", {}).get("name")
            if name:
                if name == "read_demo_order":
                    # The controlled model's target oracle reads actual stored
                    # user events. No history is passed into the HTTP request.
                    redis = self.factory.redis
                    keys = [key async for key in redis.scan_iter(match="source_history_v1:*")]
                    events = [] if not keys else [json.loads(row) for row in reversed(await redis.lrange(keys[0], 0, -1))]
                    working = [key async for key in redis.scan_iter(match="wm_v2:*")]
                    recent = [] if not working else [json.loads(row) for row in reversed(await redis.lrange(working[0], 0, 4))]
                    history = source_bound_order_history(events,
                        [{"role": row["role"], "content": row["content"]} for row in recent])
                    selected = resolve_order_id(message, history)
                    assert selected.status == "resolved"
                    args = {"order_id": selected.order_id}
                elif name == "search_knowledge_base":
                    args = {"query": message, "top_k": 3}
                elif name == "lookup_error_code":
                    args = {"error_code": select_diagnostic_request(message).error_code}
                elif name == "build_diagnostic_plan":
                    args = {"environment": "App", "reproduced": False}
                else:
                    raise AssertionError("unexpected native tool")
                return self.response([{"type": "tool_use", "name": name, "input": args,
                    "id": "native_http_" + str(len(self.calls))}], stop="tool_use")
            raw = "受控接待文字；语义质量仍待独立逐项审阅。"
        elif frame.phase == "composer":
            raw = "受控汇总文字。"
        else:
            raise AssertionError("no Judge, remote embedding or background memory model")
        return self.response([{"type": "text", "text": raw}])

    @staticmethod
    def response(content, *, stop="end_turn"):
        return httpx.Response(200, json={"id": "controlled-http-memory", "type": "message", "role": "assistant",
            "model": "qwen3.7-flash-2026-07-15", "content": content, "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 1}})


async def execute(position=0, *, case=None, data=None, factory=None, fault=None, healthy=None):
    row, data, factory = case or cases()[position], data or Data(), factory or MemoryFactory()
    replies = Replies(row, factory, fault=fault)
    async with httpx.AsyncClient(transport=httpx.MockTransport(replies), trust_env=False) as http:
        sdk = AsyncAnthropic(api_key="controlled-not-real", base_url="https://coverage.invalid", max_retries=0, http_client=http)
        try:
            with eval_deadlines(EvaluationDeadlinePolicy()):
                artifact = await runtime.execute_conversation(row, sdk, data,
                    ensure_healthy=healthy or data.assert_healthy, frame_case_id="M-" + row["id"], memory_factory=factory)
            return artifact, replies, data, factory
        finally:
            await sdk.close()


@pytest.mark.parametrize("position", range(8))
def test_all_frozen_conversations_use_five_actual_http_memory_and_sdk_turns(position):
    async def check():
        saved = {name: getattr(api, name) for name in runtime._GLOBALS}
        artifact, replies, data, factory = await execute(position)
        try:
            assert {name: getattr(api, name) for name in runtime._GLOBALS} == saved
            assert len(artifact["turns"]) == 5 and artifact["quality_pass_claim"] is False
            assert artifact["judge"] is artifact["semantic_review"] is None
            assert artifact["background_memory_model_requests"] == 0
            assert artifact["data_kind"] == "controlled-orders-and-isolated-chroma"
            assert artifact["physical_requests_observed"] == len(replies.calls)
            assert Counter(frame.turn_id for frame, _ in replies.calls if frame.phase == "intent") == {
                f"turn-{number}": 1 for number in range(1, 6)}
            assert CURRENT.get() is None and factory.closed
            assert artifact["memory_cleanup"]["remaining_owned_keys"] == 0
            assert await factory.redis.dbsize() == 0
            for number, turn in enumerate(artifact["turns"], 1):
                assert turn["http_status"] == 200
                assert turn["redis"]["working_messages"] == number * 2
                assert turn["redis"]["source_events"] == number
                assert len(turn["memory_before"]["reference_history"]) == number - 1
                assert turn["memory_after"]["reference_history"][-1]["request_id"] == turn["request_id"]
                for call in turn["tool_evidence"]["calls"]:
                    assert call["tool_use_id"] and call["request_id"] == turn["request_id"]
                    assert call["trusted_customer_id"] == cases()[position]["customer_id"]
            for frame, payload in replies.calls:
                text = json.dumps(payload, ensure_ascii=False)
                assert frame.case_id == "M-" + cases()[position]["id"]
                assert "required_facts" not in text and "review_status" not in text and "provenance" not in text
                assert cases()[position]["id"] not in text
                assert frame.phase in {"intent", "agent", "composer", "rag_rewrite", "rag_rerank"}
            assert all(artifact["tool_checks"]["checks"].values()), artifact["tool_checks"]
        finally:
            await factory.redis.aclose()
    asyncio.run(check())


def test_first_order_target_is_recovered_from_persisted_original_input_after_three_neutral_turns():
    async def check():
        artifact, _, data, factory = await execute(0)
        try:
            turn = artifact["turns"][-1]
            assert "DEMO-2001" not in str(turn["history"])
            assert turn["order_history"][0]["content"] == cases()[0]["turns"][0]["message"]
            assert all(call == ("demo-customer-a", "DEMO-2001") for call in data.orders)
            read_turns = {number for number, value in enumerate(artifact["turns"], 1)
                          if any(call["name"] == "read_demo_order" for call in value["tool_evidence"]["calls"])}
            assert read_turns == {1, 5}
        finally:
            await factory.redis.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("fault", ["cancel", "timeout"])
def test_sdk_fault_latches_without_retry_and_restores_app_memory_and_old_frames(fault):
    async def check():
        factory, data = MemoryFactory(), Data()
        row = cases()[0]
        replies = Replies(row, factory, fault=fault)
        saved = {name: getattr(api, name) for name in runtime._GLOBALS}
        async with httpx.AsyncClient(transport=httpx.MockTransport(replies), trust_env=False) as http:
            sdk = AsyncAnthropic(api_key="controlled-not-real", base_url="https://coverage.invalid", max_retries=0, http_client=http)
            try:
                with eval_deadlines(EvaluationDeadlinePolicy()):
                    with pytest.raises(BaseException) as failed:
                        await runtime.execute_conversation(row, sdk, data, ensure_healthy=data.assert_healthy,
                            frame_case_id="M-" + row["id"], memory_factory=factory)
                assert len(replies.calls) == 1 and data.orders == []
                assert {name: getattr(api, name) for name in runtime._GLOBALS} == saved
                assert CURRENT.get() is None and factory.closed
                assert await factory.redis.dbsize() == 0
                with pytest.raises(type(failed.value)):
                    replies.calls[0][0].evidence_guard()
            finally:
                await sdk.close()
                await factory.redis.aclose()
    asyncio.run(check())


def test_case_change_and_missing_explicit_memory_resources_never_send():
    async def check():
        row, data, sent = cases()[0], Data(), []
        async def send(request):
            sent.append(request)
            raise AssertionError("no provider request expected")
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as http:
            sdk = AsyncAnthropic(api_key="controlled-not-real", base_url="https://coverage.invalid", max_retries=0, http_client=http)
            try:
                changed = deepcopy(row)
                changed["turns"][0]["message"] += " altered"
                with pytest.raises(ValueError, match="case_binding"):
                    await runtime.execute_conversation(changed, sdk, data, ensure_healthy=data.assert_healthy,
                        frame_case_id="M-" + row["id"])
                with pytest.raises(ValueError, match="memory_resources_required"):
                    await runtime.execute_conversation(row, sdk, data, ensure_healthy=data.assert_healthy,
                        frame_case_id="M-" + row["id"])
                assert sent == data.orders == []
            finally:
                await sdk.close()
    asyncio.run(check())


def test_completed_or_stale_turn_frames_cannot_authorize_future_model_calls():
    async def check():
        artifact, replies, _, factory = await execute(0)
        try:
            assert artifact["review_status"] == "pending_review"
            for frame, _ in replies.calls:
                with pytest.raises(ValueError, match="closed_context"):
                    frame.evidence_guard()
        finally:
            await factory.redis.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("change", ["message", "native_id", "customer", "order_result", "receipt", "before", "source", "phase", "request"])
def test_recomputed_checks_reject_tampered_observations_even_if_saved_checks_claim_success(change):
    async def check():
        artifact, _, _, factory = await execute(0)
        try:
            altered = deepcopy(artifact)
            turn = altered["turns"][-1]
            call = turn["tool_evidence"]["calls"][0]
            if change == "message":
                turn["question"] += " different"
            elif change == "native_id":
                call["tool_use_id"] = "unassociated-native"
            elif change == "customer":
                call["trusted_customer_id"] = "demo-customer-b"
            elif change == "order_result":
                call["result"]["order"]["total_amount_minor"] += 1
            elif change == "receipt":
                turn["receipt"]["message"] = "replayed old message"
            elif change == "before":
                turn["memory_before"]["recent_messages"].pop()
            elif change == "source":
                turn["memory_after"]["reference_history"][-1]["source"] = "model-summary"
            elif change == "phase":
                altered["phase_observations"][-1]["case_id"] = "M-old-case"
            else:
                turn["request_id"] = "old-request"
            altered["tool_checks"] = {"checks": {"all": True}, "blockers": []}
            observed = runtime.conversation_checks(cases()[0], altered)
            assert observed["blockers"] and not all(observed["checks"].values())
            artifact["tool_checks"] = {"checks": {"all": False}, "blockers": ["pretend-failure"]}
            assert not runtime.conversation_checks(cases()[0], artifact)["blockers"]
        finally:
            await factory.redis.aclose()
    asyncio.run(check())


def test_received_empty_actor_output_is_preserved_as_failure_and_later_planned_turns_continue():
    async def check():
        artifact, replies, _, factory = await execute(0, fault="empty-actor")
        try:
            assert len(artifact["turns"]) == 5 and artifact["execution_status"] == "failed"
            assert artifact["review_status"] == "failed" and artifact["quality_pass_claim"] is False
            failed = [turn for turn in artifact["turns"] if turn["result"]["status"] == "failed"]
            assert failed and all(turn["http_status"] == 503 for turn in failed)
            for turn in failed:
                assert turn["memory_before"] == turn["memory_after"]
                assert turn["receipt"]["response"]["status"] == "failed"
            assert {frame.turn_id for frame, _ in replies.calls if frame.phase == "intent"} == {f"turn-{n}" for n in range(1, 6)}
            assert artifact["memory_cleanup"]["remaining_owned_keys"] == 0
        finally:
            await factory.redis.aclose()
    asyncio.run(check())


def test_default_resource_factory_uses_explicit_redis_and_owns_only_two_memory_collections():
    class Client:
        def __init__(self):
            self.created, self.deleted, self.collections = [], [], {}
        def create_collection(self, *, name, embedding_function, metadata):
            assert callable(embedding_function)
            self.created.append(name)
            self.collections[name] = MemoryCollection()
            return self.collections[name]
        def delete_collection(self, *, name):
            self.deleted.append(name)
            del self.collections[name]

    async def check():
        row, data = cases()[0], Data()
        data.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        client = Client()
        data._knowledge = SimpleNamespace(client=client, kb=SimpleNamespace(
            _collection=SimpleNamespace(_embedding_function=lambda value: value)))
        replies = Replies(row, SimpleNamespace(redis=data.redis))
        async with httpx.AsyncClient(transport=httpx.MockTransport(replies), trust_env=False) as http:
            sdk = AsyncAnthropic(api_key="controlled-not-real", base_url="https://coverage.invalid", max_retries=0, http_client=http)
            try:
                with eval_deadlines(EvaluationDeadlinePolicy()):
                    artifact = await runtime.execute_conversation(row, sdk, data, ensure_healthy=data.assert_healthy,
                        frame_case_id="M-" + row["id"])
                assert len(client.created) == 2 and set(client.created) == set(client.deleted)
                assert client.collections == {} and await data.redis.dbsize() == 0
                assert all(name.startswith(artifact["namespace"]["resource_id"]) for name in client.created)
                assert not artifact["tool_checks"]["blockers"]
            finally:
                await sdk.close()
                await data.redis.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("fault", ["stale-frame", "order-source"])
def test_old_turn_callback_or_changed_order_source_stops_before_another_planned_turn(fault):
    async def check():
        data, factory = Data(corrupt_order=fault == "order-source"), MemoryFactory()
        row = cases()[0]
        replies = Replies(row, factory, fault=fault)
        async with httpx.AsyncClient(transport=httpx.MockTransport(replies), trust_env=False) as http:
            sdk = AsyncAnthropic(api_key="controlled-not-real", base_url="https://coverage.invalid", max_retries=0, http_client=http)
            try:
                with eval_deadlines(EvaluationDeadlinePolicy()):
                    with pytest.raises(ValueError, match="closed_turn" if fault == "stale-frame" else "order_source_changed"):
                        await runtime.execute_conversation(row, sdk, data, ensure_healthy=data.assert_healthy,
                            frame_case_id="M-" + row["id"], memory_factory=factory)
                assert not any(frame.turn_id == "turn-3" for frame, _ in replies.calls)
                assert len(data.orders) == 1 and await factory.redis.dbsize() == 0
                assert factory.closed and CURRENT.get() is None
            finally:
                await sdk.close()
                await factory.redis.aclose()
    asyncio.run(check())


def test_known_received_http200_json_structure_failure_is_recorded_without_retry_or_batch_fault():
    async def check():
        artifact, replies, _, factory = await execute(0, fault="bad-json")
        try:
            assert len(artifact["turns"]) == 5
            assert artifact["phase_observations"][0]["structure_failure"] == {
                "error_type": "JSONDecodeError", "http_status": 200}
            assert artifact["review_status"] == "failed"
            assert artifact["tool_checks"]["checks"]["phase_binding"]
            assert not artifact["tool_checks"]["checks"]["model_response_structure_valid"]
            assert Counter(frame.turn_id for frame, _ in replies.calls if frame.phase == "intent") == {
                f"turn-{n}": 1 for n in range(1, 6)}
            assert factory.closed and await factory.redis.dbsize() == 0
        finally:
            await factory.redis.aclose()
    asyncio.run(check())
