"""Run the native test window against the real FastAPI routes on ephemeral loopback.

The order store and classifier are controlled. No application lifespan, dotenv,
model client, external database or deployed gateway is used by this check.
"""
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import json
import hmac
import os
from pathlib import Path
import secrets
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import uvicorn
from starlette.responses import Response

backend = Path(__file__).resolve().parents[2] / 'backend'
sys.path.insert(0, str(backend))
os.environ.update(SERVEFLOW_ENV='development', SERVEFLOW_API_TOKENS='')
with patch('dotenv.load_dotenv', return_value=False):
    from api import main
from agents.agent_orchestrator import AgentOrchestrator
from core.intent_recognizer import IntentCategory, UrgencyLevel
from demo_business.session import DemoSessionService
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


async def run(binary, release_order):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    origin = f'http://127.0.0.1:{port}'
    password = secrets.token_urlsafe(20)
    basic = b'Basic ' + base64.b64encode(f'seat:{password}'.encode())
    fixture = load_fixture()

    @asynccontextmanager
    async def conversation(*_):
        yield

    async def read_order(customer, order_id):
        # Hold the reply until the native window has actually edited conversation B.
        async with asyncio.timeout(10):
            while not release_order.exists():
                await asyncio.sleep(.01)
        return fixture_order_result(fixture, customer, order_id)

    main._demo_store = SimpleNamespace(list_sample_orders=lambda _: [], read_order=AsyncMock(side_effect=read_order))
    main._demo_sessions = DemoSessionService(secrets.token_bytes(32), (origin,))
    # Use the Agent and workflow code paths; replace only inference, retrieval and storage.
    policy = next(doc for doc in json.loads((backend / 'knowledge/serveflow-demo-v2.json').read_text())
                  if doc['title'] == '配送说明')
    row = {**policy, 'chunk': 0, 'score': 1.0, 'document_id': 'controlled-shipping-policy',
           'version': hashlib.sha256(policy['content'].encode()).hexdigest()}
    retrieve = AsyncMock(return_value=SimpleNamespace(success=True, reranked=True, data=[row]))
    use_tool = SimpleNamespace(stop_reason='tool_use', content=[{'type': 'tool_use', 'id': 'ui-fixture',
        'name': 'search_knowledge_base', 'input': {'query': '配送规则'}}])
    model = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[use_tool])))
    def intent(category):
        return SimpleNamespace(intent=category, intent_group='query', confidence=1.0,
                               urgency=UrgencyLevel.LOW, entities={}, source_scores={})
    classifier = SimpleNamespace(recognize=AsyncMock(side_effect=[intent(IntentCategory.ORDER_STATUS),
                                                                  intent(IntentCategory.LOGISTICS)]))
    with patch('agents.agent_orchestrator.AsyncAnthropic', return_value=model), \
         patch('agents.agent_orchestrator.IntentRecognizer', return_value=classifier):
        orc = AgentOrchestrator(api_key='controlled', model='controlled',
                                rag_tool_manager=SimpleNamespace(search_with_rewrite=retrieve))
    main._orchestrator = orc
    main._memory = SimpleNamespace(conversation=conversation, commit_turn=AsyncMock(),
        get_context=AsyncMock(return_value=SimpleNamespace(recent_messages=[], state={}, to_prompt_text=lambda: '')),
        get_turn_receipt=AsyncMock(return_value=None))

    async def gateway(scope, receive, send):
        auth = dict(scope['headers']).get(b'authorization', b'')
        path = scope['path']
        if not hmac.compare_digest(auth, basic) or path not in ('/api/python/demo/session', '/api/python/chat'):
            return await Response(status_code=403)(scope, receive, send)
        path = path.removeprefix('/api/python')
        scope = {**scope, 'path': path, 'raw_path': path.encode(),
                 'headers': [(k, v) for k, v in scope['headers'] if k != b'authorization']}
        await main.app(scope, receive, send)

    server = uvicorn.Server(uvicorn.Config(gateway, lifespan='off', access_log=False, log_level='error'))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError('test gateway did not start')
            await asyncio.sleep(.01)
        env = {**os.environ, 'SERVEFLOW_FIXTURE_PORT': str(port), 'SERVEFLOW_FIXTURE_PASSWORD': password,
               'SERVEFLOW_FIXTURE_RELEASE_ORDER': str(release_order)}
        process = await asyncio.create_subprocess_exec(binary, env=env)
        try:
            result = await asyncio.wait_for(process.wait(), 30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if result: return result
        main._demo_store.read_order.assert_awaited_once_with('demo-customer-a', 'DEMO-1001')
        assert main._memory.commit_turn.await_count == 2
        retrieve.assert_awaited_once()
        model.messages.create.assert_awaited_once()
        return 0
    finally:
        server.should_exit = True
        await serving
        listener.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='serveflow-native-flow-') as directory:
        sys.exit(asyncio.run(run(sys.argv[1], Path(directory) / 'release-order')))
