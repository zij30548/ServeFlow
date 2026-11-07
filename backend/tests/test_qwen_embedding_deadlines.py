"""Controlled timing diagnoses; no new live model-quality measurements.

HTTP 200 headers are not a completed embedding. Explicit preparation now has
its own deadline, so it cannot spend the live query's eight-second deadline.
"""
import asyncio
import json
import socket
import time

import httpx
import pytest

from core import protected_demo_quota as quota
from core import qwen_embedding as embedding
from core import intent_embedding as matching
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
    monkeypatch.setenv('QWEN_INTENT_CACHE_DIRECTORY', str(tmp_path.resolve() / 'templates'))
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
        diagnostic = {}
        with pytest.raises(embedding.EmbeddingError, match='^embedding_timeout$'):
            await embedding.request_embeddings(['controlled'], base_url=BASE, api_key=KEY, diagnostic=diagnostic)
        assert observed == {'headers': True, 'body_complete': False, 'closed': True, 'parsed': False}
        final = events(isolated)[-1]
        assert final['outcome'] == 'timeout' and final['delegate_entered'] is True
        # A buffering observer hides headers from the outer client until EOF.
        # The production reader receives the status directly without it.
        assert final['http_status'] == (None if buffering_observer else 200)
        assert diagnostic['phase'] == ('waiting_headers' if buffering_observer else 'reading_body')
        assert diagnostic['body_complete_ms'] is None and diagnostic['response_bytes'] == 0
        assert quota.status()['used'] == 1 and quota.status()['unresolved'] == 0
    asyncio.run(run())


def test_explicit_preparation_can_exceed_eight_seconds_without_spending_query_budget(isolated, monkeypatch):
    async def run():
        calls = []
        async def provider(request):
            texts = json.loads(request.content)['input']
            calls.append(len(texts))
            await asyncio.sleep(2.75 if len(texts) > 1 else .01)
            return httpx.Response(200, json=body(request))
        monkeypatch.setattr(embedding, '_delegate', lambda: httpx.MockTransport(provider))
        client = IntentEmbedding()
        start = time.monotonic()
        async def prepare(texts):
            return await embedding.request_embeddings(texts, base_url=BASE, api_key=KEY, purpose='prepare')
        await client.template_cache.prepare(_TEMPLATES, prepare, max_requests=3)
        assert time.monotonic() - start >= 8.2
        groups, vector, source = await client.vectors('controlled query', _TEMPLATES, lambda t: [1., 0.])
        assert calls == [20, 20, 14, 1]
        assert source == 'qwen/' + embedding.MODEL and len(vector) == len(next(iter(groups.values()))[0]) == 1024
        assert client.status()['last_error'] is None
        completed = [e for e in events(isolated) if e['event'] == 'finish']
        assert [e['outcome'] for e in completed] == ['received'] * 4
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
        async def prepare(texts):
            return await embedding.request_embeddings(texts, base_url=BASE, api_key=KEY, purpose='prepare')
        await client.template_cache.prepare(_TEMPLATES, prepare, max_requests=3)
        await client.vectors('controlled cold query', _TEMPLATES, lambda t: [1., 0.])
        assert calls == [20, 20, 14, 1]
        _, vector, source = await client.vectors('controlled warm query', _TEMPLATES, lambda t: [1., 0.])
        assert calls == [20, 20, 14, 1, 1] and len(vector) == 1024
        assert source == 'qwen/' + embedding.MODEL and quota.status()['used'] == 5
    asyncio.run(run())


def test_partial_body_diagnostic_records_bytes_and_first_byte_without_content(isolated, monkeypatch):
    async def run():
        class Partial(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'private partial bytes'
                await asyncio.Event().wait()
        monkeypatch.setattr(embedding, 'TIMEOUT', .03)
        monkeypatch.setattr(embedding, '_delegate', lambda: httpx.MockTransport(
            lambda r: httpx.Response(200, stream=Partial())))
        diagnostic = {}
        with pytest.raises(embedding.EmbeddingError, match='embedding_timeout'):
            await embedding.request_embeddings(['q'], base_url=BASE, api_key=KEY, diagnostic=diagnostic)
        assert diagnostic['phase'] == 'reading_body'
        assert diagnostic['response_bytes'] == len(b'private partial bytes')
        assert diagnostic['headers_ms'] is not None and diagnostic['first_byte_ms'] is not None
        assert diagnostic['body_complete_ms'] is None and diagnostic['delegate_entered'] is True
        assert 'private' not in json.dumps(diagnostic) and KEY not in json.dumps(diagnostic)
    asyncio.run(run())


def test_preparation_and_query_use_different_bounded_request_deadlines(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(embedding, 'TIMEOUT', .02)
        monkeypatch.setattr(embedding, 'PREPARE_TIMEOUT', .2)
        async def provider(request):
            await asyncio.sleep(.04)
            return httpx.Response(200, json=body(request))
        monkeypatch.setattr(embedding, '_delegate', lambda: httpx.MockTransport(provider))
        with pytest.raises(embedding.EmbeddingError, match='embedding_timeout'):
            await embedding.request_embeddings(['q'], base_url=BASE, api_key=KEY)
        diagnostic = {}
        values = await embedding.request_embeddings(['template'], base_url=BASE, api_key=KEY,
                                                    purpose='prepare', diagnostic=diagnostic)
        assert len(values[0]) == 1024 and diagnostic['purpose'] == 'prepare'
        assert diagnostic['deadline_s'] == .2 and diagnostic['phase'] == 'complete'
        assert diagnostic['body_complete_ms'] is not None and diagnostic['error_code'] is None
    asyncio.run(run())


def test_qwen_live_match_does_not_keep_the_old_eight_second_outer_limit(isolated,monkeypatch):
    async def run():
        # Scale the policy to keep this deterministic offline check quick.
        monkeypatch.setattr(embedding,'TIMEOUT',.12)
        monkeypatch.setattr(matching,'QWEN_MATCH_TIMEOUT',.15)
        async def provider(request):
            await asyncio.sleep(.08)
            return httpx.Response(200,json=body(request))
        monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(provider))
        client=IntentEmbedding()
        async def local_preparation(texts):return [[1.]+[0.]*1023 for _ in texts]
        await client.template_cache.prepare(_TEMPLATES,local_preparation,max_requests=3)
        _,vector,source=await client.vectors('controlled',_TEMPLATES,lambda t:[1.,0.])
        assert source=='qwen/'+embedding.MODEL and len(vector)==1024
        assert client.last_diagnostic['deadline_s']==.12 and client.last_error is None
        assert quota.status()['used']==1
    asyncio.run(run())


def test_qwen_match_deadline_cancels_and_charges_once_without_local_retry(isolated,monkeypatch):
    async def run():
        monkeypatch.setattr(embedding,'TIMEOUT',.15)
        monkeypatch.setattr(matching,'QWEN_MATCH_TIMEOUT',.04)
        calls=[]
        async def provider(request):calls.append(request);await asyncio.Event().wait()
        monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(provider))
        client=IntentEmbedding()
        async def local_preparation(texts):return [[1.]+[0.]*1023 for _ in texts]
        await client.template_cache.prepare(_TEMPLATES,local_preparation,max_requests=3)
        result=await client.vectors('controlled',_TEMPLATES,lambda t:[1.,0.])
        assert result[2]=='local-ngram' and client.last_error=='embedding_match_timeout'
        assert len(calls)==quota.status()['used']==1 and quota.status()['unresolved']==0
    asyncio.run(run())
