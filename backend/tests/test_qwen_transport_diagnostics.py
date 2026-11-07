"""Controlled failure categories. No inference about the old live interruption."""
import asyncio
import json
import socket
import ssl

import httpcore
import httpx
import pytest

from core import protected_demo_quota as quota, qwen_embedding as embedding
from core import intent_templates as commands
from core.intent_template_cache import TemplateCache, TemplateCacheError

BASE = 'https://controlled.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
KEY = 'controlled-only-not-a-real-key'
PRIVATE = 'private-model-text credential /private/file'


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def denied(*a, **k): raise AssertionError('real network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setenv('ECHOMIND_ENV', 'protected-demo')
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER', 'qwen')
    monkeypatch.setenv('QWEN_EMBEDDING_BASE_URL', BASE)
    monkeypatch.setenv('QWEN_EMBEDDING_API_KEY', KEY)
    monkeypatch.setenv('QWEN_INTENT_CACHE_DIRECTORY', str(tmp_path.resolve()/'templates'))
    monkeypatch.setenv('SERVEFLOW_DEMO_QUOTA_DIR', str(tmp_path.resolve()/'usage'))
    monkeypatch.setenv('SERVEFLOW_DEMO_MODEL_LIMIT', '5')
    quota.initialize()
    return tmp_path.resolve()/'usage'


@pytest.mark.parametrize('outer,inner,expected,error_code,outcome', [
    (httpx.RemoteProtocolError, httpcore.RemoteProtocolError,
     ['remote_protocol_error','core_remote_protocol_error'], 'embedding_network_error','network_error'),
    (httpx.ReadError, ConnectionResetError,
     ['read_error','connection_reset'], 'embedding_network_error','network_error'),
    (httpx.ReadError, ssl.SSLEOFError,
     ['read_error','tls_unexpected_eof'], 'embedding_network_error','network_error'),
    (httpx.ReadTimeout, httpcore.ReadTimeout,
     ['read_timeout','core_read_timeout'], 'embedding_timeout','timeout'),
])
def test_partial_response_preserves_protocol_failure_without_claiming_full_body(
        offline, monkeypatch, caplog, outer, inner, expected, error_code, outcome):
    async def run():
        calls=[];closed=[];body=b'x'*49255
        class Interrupted(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield body
                raise outer(PRIVATE+' '+KEY) from inner(PRIVATE)
            async def aclose(self): closed.append(True)
        def provider(request):
            calls.append(request)
            return httpx.Response(200, stream=Interrupted(), headers={
                'content-length':'90000','x-request-id':'controlled-request-01',
                'set-cookie':'private-session=private', 'untrusted':PRIVATE},
                extensions={'http_version':b'HTTP/1.1'})
        monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(provider))
        monkeypatch.setattr(embedding,'parse_vectors',lambda *a:pytest.fail('incomplete response parsed'))
        diagnostic={}
        with pytest.raises(embedding.EmbeddingError,match='^'+error_code+'$'):
            await embedding.request_embeddings(['controlled'],base_url=BASE,api_key=KEY,
                                               purpose='prepare',diagnostic=diagnostic)
        assert len(calls)==1 and closed
        assert diagnostic['transport_failure_codes']==expected
        assert diagnostic['phase']=='reading_body' and diagnostic['response_bytes']==49255
        assert diagnostic['expected_response_bytes']==90000
        assert diagnostic['response_framing']=='content_length' and diagnostic['http_version']=='HTTP/1.1'
        assert diagnostic['request_id']=='controlled-request-01' and diagnostic['http_status']==200
        assert diagnostic['first_byte_ms'] is not None and diagnostic['body_complete_ms'] is None
        assert diagnostic['deadline_s']==15 and diagnostic['elapsed_ms']<15000
        events=[json.loads(s) for s in (offline/'journal.jsonl').read_text().splitlines()]
        assert events[-1]['outcome']==outcome and events[-1]['http_status']==200
        assert quota.status()['used']==1 and quota.status()['unresolved']==0
        text=json.dumps(diagnostic)+caplog.text
        assert PRIVATE not in text and KEY not in text and BASE not in text
        assert 'private-session' not in text and 'untrusted' not in text
    asyncio.run(run())


@pytest.mark.parametrize('error,code', [
    (httpx.ConnectTimeout,'connect_timeout'), (httpx.ConnectError,'connect_error'),
    (httpx.WriteError,'write_error'), (httpx.WriteTimeout,'write_timeout'),
    (httpx.PoolTimeout,'pool_timeout'), (httpx.LocalProtocolError,'local_protocol_error'),
])
def test_failure_before_headers_has_safe_category_and_no_response_claim(monkeypatch,error,code):
    async def run():
        def provider(request): raise error(PRIVATE+' '+KEY,request=request)
        monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(provider))
        diagnostic={}
        with pytest.raises(embedding.EmbeddingError):
            await embedding.request_embeddings(['q'],base_url=BASE,api_key=KEY,diagnostic=diagnostic)
        assert diagnostic['transport_failure_codes']==[code]
        assert diagnostic['phase']=='waiting_headers' and diagnostic['http_status'] is None
        assert diagnostic['response_bytes']==0 and diagnostic['expected_response_bytes'] is None
        assert diagnostic['request_id']=='unknown' and diagnostic['http_version']=='unknown'
        assert PRIVATE not in json.dumps(diagnostic) and KEY not in json.dumps(diagnostic)
    asyncio.run(run())


def test_error_chain_is_bounded_and_does_not_copy_unknown_type_names_or_arguments():
    class PrivateCustomerError(Exception): pass
    secret=PrivateCustomerError(PRIVATE, {'input':KEY})
    root=httpx.ReadError(PRIVATE)
    root.__cause__=secret
    secret.__cause__=root
    assert embedding._failure_codes(root)==['read_error']
    assert embedding._failure_codes(secret)==['read_error']
    assert embedding._failure_codes(PrivateCustomerError(PRIVATE))==[]
    assert embedding._failure_codes(asyncio.CancelledError())==['cancelled']
    assert embedding._failure_codes(TimeoutError(PRIVATE))==['timeout']


@pytest.mark.parametrize('header', ['private value /path',KEY+'?query=private','x'*151,'中文'])
def test_response_metadata_rejects_untrusted_ids_and_keeps_only_allowed_fields(header):
    response=httpx.Response(200,headers={'x-request-id':header.encode('utf-8'),
        'transfer-encoding':'chunked','set-cookie':PRIVATE})
    value=embedding._response_details(response)
    assert value=={'expected_response_bytes':None,'response_framing':'chunked',
                  'http_version':'unknown','request_id':'unknown'}
    assert PRIVATE not in json.dumps(value) and header not in json.dumps(value)


def test_even_validly_shaped_request_id_cannot_echo_the_api_key():
    value=embedding._response_details(httpx.Response(200,headers={'x-request-id':KEY}),api_key=KEY)
    assert value['request_id']=='unknown' and KEY not in json.dumps(value)


def test_full_large_response_is_not_cut_off_at_the_old_observed_byte_count(monkeypatch):
    async def run():
        raw=json.dumps({'model':embedding.MODEL,'data':[
            {'index':i,'embedding':[.000123456789]*(embedding.DIMENSIONS)} for i in range(20)]}).encode()
        assert 49255<len(raw)<embedding.RESPONSE_LIMIT
        class Complete(httpx.AsyncByteStream):
            async def __aiter__(self):
                for start in range(0,len(raw),17000): yield raw[start:start+17000]
        monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(lambda r:
            httpx.Response(200,stream=Complete(),headers={'content-length':str(len(raw))})))
        diagnostic={}
        result=await embedding.request_embeddings(['t']*20,base_url=BASE,api_key=KEY,
                                                 purpose='prepare',diagnostic=diagnostic)
        assert len(result)==20 and all(len(v)==1024 for v in result)
        assert diagnostic['phase']=='complete' and diagnostic['transport_failure_codes']==[]
        assert diagnostic['body_complete_ms'] is not None
        assert diagnostic['response_bytes']==diagnostic['expected_response_bytes']==len(raw)
    asyncio.run(run())


def test_provider_error_crosses_cache_boundary_and_failed_claim_still_blocks_resend():
    async def run():
        cache=TemplateCache(BASE);calls=[]
        async def failure(texts):calls.append(texts);raise embedding.EmbeddingError('embedding_network_error')
        with pytest.raises(embedding.EmbeddingError,match='^embedding_network_error$'):
            await cache.prepare({'a':['template']},failure,max_requests=1)
        with pytest.raises(TemplateCacheError,match='preparation_incomplete'):
            await cache.prepare({'a':['template']},failure,max_requests=1)
        assert len(calls)==1 and quota.status()['used']==0
    asyncio.run(run())


def test_explicit_cli_reports_network_origin_not_cache_corruption(monkeypatch,capsys):
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:False)
    calls=[]
    def provider(request):calls.append(request);raise httpx.ConnectError(PRIVATE,request=request)
    monkeypatch.setattr(embedding,'_delegate',lambda:httpx.MockTransport(provider))
    assert commands.main(['prepare','--max-requests','3'])==2
    first=json.loads(capsys.readouterr().out)
    assert first['error_code']=='embedding_network_error'
    assert first['request_diagnostics'][0]['transport_failure_codes']==['connect_error']
    assert commands.main(['prepare','--max-requests','3'])==2
    second=json.loads(capsys.readouterr().out)
    assert second['error_code']=='embedding_template_preparation_incomplete'
    assert len(calls)==quota.status()['used']==1
