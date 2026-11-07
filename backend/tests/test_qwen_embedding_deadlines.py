"""Controlled timing diagnoses; no new live model-quality measurements.

Keep both shipped deadlines. HTTP 200 headers are not a completed embedding;
four separately valid round trips can exceed the cold match's shared deadline.
"""
import asyncio
import json
import socket
import time

import httpx
import pytest

from core import protected_demo_quota as quota
from core import qwen_embedding as embedding
from core.intent_embedding import IntentEmbedding
from core.intent_recognizer import _TEMPLATES

BASE = 'https://controlled-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
KEY = 'controlled-only-offline-key'


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def deny(*a, **k): raise AssertionError('diagnostic tests forbid real network')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setenv('ECHOMIND_ENV', 'protected-demo')
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER', 'qwen')
    monkeypatch.setenv('QWEN_EMBEDDING_BASE_URL', BASE)
    monkeypatch.setenv('QWEN_EMBEDDING_API_KEY', KEY)
    path = tmp_path.resolve() / 'usage'
    monkeypatch.setenv('SERVEFLOW_DEMO_QUOTA_DIR', str(path))
    monkeypatch.setenv('SERVEFLOW_DEMO_MODEL_LIMIT', '5')
    quota.initialize()
    return path


def events(path):
    return [json.loads(line) for line in (path / 'journal.jsonl').read_text().splitlines()]


def body(request):
    texts = json.loads(request.content)['input']
    return {'model': embedding.MODEL, 'data': [
        {'index': i, 'embedding': [1.] + [0.] * 1023} for i in range(len(texts))]}


@pytest.mark.parametrize('buffering_observer', [False, True])
def test_headers_200_with_stalled_body_times_out_before_vector_validation(
        isolated, monkeypatch, buffering_observer):
    async def run():
        observed = {'headers': False, 'body_complete': False, 'closed': False, 'parsed': False}
        class Stalled(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                observed['body_complete'] = True
                yield b'{}'
            async def aclose(self): observed['closed'] = True
        async def provider(request):
            observed['headers'] = True
            return httpx.Response(200, stream=Stalled())
        class Observer(httpx.AsyncBaseTransport):
            def __init__(self): self.inner = httpx.MockTransport(provider)
            async def handle_async_request(self, request):
                response = await self.inner.handle_async_request(request)
                assert response.status_code == 200
                try:
                    pieces = [part async for part in response.aiter_raw()]
                    return httpx.Response(200, content=b''.join(pieces), request=request)
                finally: await response.aclose()
            async def aclose(self): await self.inner.aclose()
        monkeypatch.setattr(embedding, 'TIMEOUT', .04)
        monkeypatch.setattr(embedding, '_delegate',
            (lambda: Observer()) if buffering_observer else (lambda: httpx.MockTransport(provider)))
        original = embedding.parse_vectors
        def parse(*a, **k):
            observed['parsed'] = True
            return original(*a, **k)
        monkeypatch.setattr(embedding, 'parse_vectors', parse)
        with pytest.raises(embedding.EmbeddingError, match='^embedding_timeout$'):
            await embedding.request_embeddings(['controlled'], base_url=BASE, api_key=KEY)
        assert observed == {'headers': True, 'body_complete': False, 'closed': True, 'parsed': False}
        final = events(isolated)[-1]
        assert final['outcome'] == 'timeout' and final['delegate_entered'] is True
        # A buffering observer hides headers from the outer client until EOF.
        # The production reader receives the status directly without it.
        assert final['http_status'] == (None if buffering_observer else 200)
        assert quota.status()['used'] == 1 and quota.status()['unresolved'] == 0
    asyncio.run(run())


def test_cold_start_exceeds_shared_eight_seconds_without_any_six_second_call_timeout(isolated, monkeypatch):
    async def run():
        calls = []
        async def provider(request):
            texts = json.loads(request.content)['input']
            calls.append(len(texts))
            await asyncio.sleep(2.05)  # below each real six-second request bound
            return httpx.Response(200, json=body(request))
        monkeypatch.setattr(embedding, '_delegate', lambda: httpx.MockTransport(provider))
        client = IntentEmbedding()
        start = time.monotonic()
        groups, vector, source = await client.vectors('controlled query', _TEMPLATES, lambda t: [1., 0.])
        assert time.monotonic() - start >= 7.9
        assert calls == [20, 20, 14, 1]
        assert source == 'local-ngram' and len(vector) == len(next(iter(groups.values()))[0]) == 2
        assert client.status()['last_error'] == 'TimeoutError'
        completed = [e for e in events(isolated) if e['event'] == 'finish']
        assert [e['outcome'] for e in completed] == ['received', 'received', 'received', 'cancelled']
        assert quota.status()['used'] == 4 and quota.status()['unresolved'] == 0
    asyncio.run(run())


def test_cached_templates_leave_only_one_remote_query_on_the_warm_path(isolated, monkeypatch):
    async def run():
        calls = []
        async def provider(request):
            calls.append(len(json.loads(request.content)['input']))
            return httpx.Response(200, json=body(request))
        monkeypatch.setattr(embedding, '_delegate', lambda: httpx.MockTransport(provider))
        client = IntentEmbedding()
        await client.vectors('controlled cold query', _TEMPLATES, lambda t: [1., 0.])
        assert calls == [20, 20, 14, 1]
        _, vector, source = await client.vectors('controlled warm query', _TEMPLATES, lambda t: [1., 0.])
        assert calls == [20, 20, 14, 1, 1] and len(vector) == 1024
        assert source == 'qwen/' + embedding.MODEL and quota.status()['used'] == 5
    asyncio.run(run())
