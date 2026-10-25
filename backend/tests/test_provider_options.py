"""Provider payload capture with in-memory HTTP only; no live credentials."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import AsyncAnthropic
import httpx
import pytest

from agents.agent_orchestrator import GeneralAgent, Request
from agents.tools import make_tool
from core.provider_options import FLASH_SNAPSHOT, nonthinking_options


@pytest.mark.parametrize("origin", [
    "https://dashscope.aliyuncs.com/apps/anthropic",
    "https://dashscope.aliyuncs.com/apps/anthropic/",
    "https://workspace-example.cn-beijing.maas.aliyuncs.com/apps/anthropic",
    "https://workspace-example.cn-beijing.maas.aliyuncs.com/apps/anthropic/",
])
def test_fixed_official_beijing_flash_disables_thinking(origin):
    assert nonthinking_options(SimpleNamespace(base_url=origin), FLASH_SNAPSHOT) == {
        "extra_body": {"thinking": {"type": "disabled"}}}


@pytest.mark.parametrize("origin", [
    "https://api.anthropic.com", "https://private.invalid/apps/anthropic",
    "http://dashscope.aliyuncs.com/apps/anthropic",
    "https://dashscope-intl.aliyuncs.com/apps/anthropic",
    "https://w.cn-shanghai.maas.aliyuncs.com/apps/anthropic",
    "https://dashscope.aliyuncs.com.evil.invalid/apps/anthropic",
    "https://dashscope.aliyuncs.com:444/apps/anthropic",
    "https://user:password@dashscope.aliyuncs.com/apps/anthropic",
    "https://dashscope.aliyuncs.com/apps/anthropic?x=1",
    "https://dashscope.aliyuncs.com/apps/anthropic#fragment",
    "https://dashscope.aliyuncs.com/apps/anthropic/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com:invalid/apps/anthropic", "", None,
])
def test_other_origins_are_unchanged(origin):
    assert nonthinking_options(SimpleNamespace(base_url=origin), FLASH_SNAPSHOT) == {}


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "qwen3.7-flash", "qwen3-vl-plus", "", None])
def test_other_models_are_unchanged(model):
    assert nonthinking_options(SimpleNamespace(base_url="https://dashscope.aliyuncs.com/apps/anthropic"), model) == {}


def test_request_options_do_not_share_mutable_state():
    client = SimpleNamespace(base_url="https://dashscope.aliyuncs.com/apps/anthropic")
    first = nonthinking_options(client, FLASH_SNAPSHOT)
    first["extra_body"]["thinking"]["type"] = "changed"
    assert nonthinking_options(client, FLASH_SNAPSHOT) == {"extra_body": {"thinking": {"type": "disabled"}}}


def native_response(blocks, *, stop="end_turn", model=FLASH_SNAPSHOT):
    return {"id": "msg_controlled", "type": "message", "role": "assistant", "model": model,
            "content": blocks, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.mark.asyncio
async def test_real_sdk_payload_keeps_tools_and_native_feedback_with_thinking_disabled():
    captured = []
    replies = [native_response([{"type": "tool_use", "id": "toolu_probe", "name": "probe", "input": {}}], stop="tool_use"),
               native_response([{"type": "text", "text": "Controlled answer"}])]

    def mock_send(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_send)) as http:
        client = AsyncAnthropic(api_key="controlled-not-real", base_url="https://dashscope.aliyuncs.com/apps/anthropic/",
                                http_client=http, max_retries=0)
        agent = GeneralAgent(client, FLASH_SNAPSHOT)
        handler = AsyncMock(return_value={"success": True})
        agent.set_shared_tools({"probe": make_tool("probe", "Controlled read", {}, handler)})
        result = await agent.handle(Request(message="Controlled query", user_id="isolated", conv_id="controlled"))
    assert result.success and len(captured) == 2
    assert handler.await_count == 1
    for payload in captured:
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["model"] == FLASH_SNAPSHOT and payload["max_tokens"] == agent.profile.max_tokens
        assert payload["temperature"] == agent.profile.temperature
        assert {tool["name"] for tool in payload["tools"]} == set(agent.get_tools())
    assert captured[1]["messages"][-1]["content"][0]["tool_use_id"] == "toolu_probe"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin,model", [
    ("https://api.anthropic.com", FLASH_SNAPSHOT),
    ("https://dashscope.aliyuncs.com/apps/anthropic", "claude-sonnet-4-5"),
])
async def test_other_real_sdk_request_payloads_do_not_gain_thinking(origin, model):
    captured = []

    def mock_send(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=native_response([{"type": "text", "text": "Controlled"}], model=model))

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_send)) as http:
        client = AsyncAnthropic(api_key="controlled-not-real", base_url=origin, http_client=http, max_retries=0)
        await client.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": "Controlled"}],
                                     **nonthinking_options(client, model))
    assert len(captured) == 1 and "thinking" not in captured[0]


def test_all_native_call_sites_opt_in_explicitly_and_clients_disable_sdk_retries():
    """Prevent future native call sites accidentally retaining provider defaults."""
    root = Path(__file__).resolve().parents[1]
    files = ("core/intent_recognizer.py", "agents/agent_orchestrator.py", "tooling/tool_manager.py",
             "memory/conversation_memory.py", "evaluation/evaluator.py")
    for relative in files:
        tree = ast.parse((root / relative).read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "AsyncAnthropic":
                assert any(keyword.arg == "max_retries" and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value == 0 for keyword in node.keywords), relative
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "create"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "messages"):
                continue
            explicit = any(keyword.arg is None and isinstance(keyword.value, ast.Call)
                           and isinstance(keyword.value.func, ast.Name)
                           and keyword.value.func.id == "nonthinking_options" for keyword in node.keywords)
            if not explicit:
                # The Agent assembles the existing tools payload as a dictionary.
                parent = node
                while not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent = parents[parent]
                explicit = any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and item.func.attr == "update" and isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "request_kwargs" and item.args
                    and isinstance(item.args[0], ast.Call) and isinstance(item.args[0].func, ast.Name)
                    and item.args[0].func.id == "nonthinking_options" for item in ast.walk(parent))
            assert explicit, f"{relative}:{node.lineno}"
