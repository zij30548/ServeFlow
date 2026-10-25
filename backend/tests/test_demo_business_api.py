from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from api import main
from core.intent_recognizer import IntentCategory, UrgencyLevel
from demo_business.session import COOKIE_NAME, DemoSessionService
from demo_business.store import DemoStoreError


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv('ECHOMIND_ENV', 'development')
    monkeypatch.setenv('ECHOMIND_API_TOKENS', '')
    store = SimpleNamespace(
        list_sample_orders=lambda customer: [{'order_id': 'DEMO-1001', 'scenario': '合成拆单订单'}],
        read_order=AsyncMock(return_value={'source': 'serveflow-mysql-demo-v1', 'is_demo': True,
            'status': 'not_found', 'as_of': '2026-10-25T00:00:00Z', 'order': None}),
    )
    sessions = DemoSessionService(b'x' * 32, ('http://127.0.0.1:5173',))
    monkeypatch.setattr(main, '_demo_store', store)
    monkeypatch.setattr(main, '_demo_sessions', sessions)
    return store, sessions


@pytest.mark.asyncio
async def test_bootstrap_and_read_bind_private_cookie_without_client_customer(demo):
    store, sessions = demo
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://127.0.0.1:8816') as client:
        initial = await client.get('/demo/orders/DEMO-1001')
        assert initial.status_code == 401
        store.read_order.assert_not_called()
        bootstrap = await client.get('/demo/session')
        assert bootstrap.status_code == 200
        assert bootstrap.headers['cache-control'] == 'no-store'
        cookie = bootstrap.headers['set-cookie']
        assert 'HttpOnly' in cookie and 'SameSite=strict' in cookie
        claims = sessions.read(client.cookies.get(COOKIE_NAME))
        assert bootstrap.json()['workspace_user_id'] == sessions.workspace_id(claims)
        await client.get('/demo/orders/DEMO-1001')
        store.read_order.assert_awaited_once_with('demo-customer-a', 'DEMO-1001')
        again = await client.get('/demo/session')
        assert again.json()['workspace_user_id'] == bootstrap.json()['workspace_user_id']
        before = store.read_order.await_count
        assert (await client.get('/demo/orders/DEMO-1001?customer_id=demo-customer-b')).status_code == 400
        assert store.read_order.await_count == before


@pytest.mark.asyncio
async def test_wrong_origin_and_store_failure_are_safe(demo):
    store, _ = demo
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://127.0.0.1:8816') as client:
        assert (await client.get('/demo/session', headers={'Origin': 'https://evil.example'})).status_code == 403
        await client.get('/demo/session')
        store.read_order.side_effect = DemoStoreError('demo_store_unavailable')
        response = await client.get('/demo/orders/DEMO-1001')
        assert response.status_code == 503
        assert response.json() == {'detail': 'demo_store_unavailable'}


@pytest.mark.asyncio
async def test_demo_disabled_keeps_original_routes_compatible(monkeypatch):
    monkeypatch.setenv('ECHOMIND_API_TOKENS', '')
    monkeypatch.setattr(main, '_demo_store', None)
    monkeypatch.setattr(main, '_demo_sessions', None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://127.0.0.1:8816') as client:
        assert (await client.get('/demo/session')).status_code == 404
        assert (await client.get('/demo/orders/DEMO-1001')).status_code == 404


@pytest.mark.asyncio
async def test_chat_authority_is_server_bound_and_checked_before_model(demo, monkeypatch):
    store, sessions = demo
    captured = []

    @asynccontextmanager
    async def conversation(*args):
        yield

    intent = SimpleNamespace(intent=IntentCategory.ORDER_STATUS, intent_group='query',
                             urgency=UrgencyLevel.LOW, confidence=.9, entities={}, source_scores={})

    async def run(req):
        captured.append(req)
        return SimpleNamespace(request_id=req.request_id, response='受控工具接线结果，不是实时模型回答',
            intent=req.intent, agent_type=SimpleNamespace(value='general'), agent_types=[],
            primary_agent=None, supporting_agents=[], tools_used=['read_demo_order'], tool_traces=[],
            routing_reason='controlled', routing_confidence=.9, escalated=False,
            status='success', retryable=False, error_code=None)

    orchestrator = SimpleNamespace(recognize_intent=AsyncMock(return_value=intent), run=run,
                                   get_tool_trace=lambda request_id: {})
    memory = SimpleNamespace(conversation=conversation, get_turn_receipt=AsyncMock(return_value=None),
        get_context=AsyncMock(return_value=SimpleNamespace(recent_messages=[], state={}, to_prompt_text=lambda: '')),
        commit_turn=AsyncMock())
    monkeypatch.setattr(main, '_orchestrator', orchestrator)
    monkeypatch.setattr(main, '_memory', memory)
    monkeypatch.setattr(main, '_profile_updater', None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://127.0.0.1:8816') as client:
        bootstrap = (await client.get('/demo/session')).json()
        body = {'message': '查DEMO-1001', 'user_id': 'client-forged-owner'}
        assert (await client.post('/chat', json=body)).status_code == 403
        orchestrator.recognize_intent.assert_not_called()
        response = await client.post('/chat', json=body, headers={
            'Origin': 'http://127.0.0.1:5173', 'X-ServeFlow-Demo-CSRF': bootstrap['csrf_token'],
        })
        assert response.status_code == 200
        assert captured[0].user_id == bootstrap['workspace_user_id']
        assert captured[0].demo_customer_id == 'demo-customer-a'
        assert captured[0].demo_order_reader == store.read_order
