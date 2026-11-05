"""SDK MockTransport and real temporary quota; never real provider calls."""
import asyncio
from copy import deepcopy
import json
import socket

from anthropic import AsyncAnthropic as SDK
import httpx
import pytest

from core import model_client as m, protected_demo_quota as q


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
ARGS = {"model": q.MODEL, "messages": [{"role": "user", "content": "controlled private prompt"}],
        "max_tokens": 16, "temperature": .1}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs): raise AssertionError("ordinary tests cannot contact provider")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    path = tmp_path.resolve() / "permits"
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    monkeypatch.setenv("ANTHROPIC_MODEL", q.MODEL)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", BASE)
    monkeypatch.setenv("SERVEFLOW_DEMO_QUOTA_DIR", str(path))
    monkeypatch.setenv("SERVEFLOW_DEMO_MODEL_LIMIT", "3")
    q.initialize(path, 3)
    return path


def response(request, content=None, **kwargs):
    return httpx.Response(200, request=request, json={"id": "mock", "type": "message", "role": "assistant",
        "content": content or [{"type": "text", "text": "controlled provider output"}], "model": q.MODEL,
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}, **kwargs)


def test_nonprotected_factory_transparently_preserves_original_sdk(monkeypatch):
    calls = []
    def original(*args, **kwargs): calls.append((args, kwargs));return "original SDK"
    monkeypatch.setattr(m.anthropic, "AsyncAnthropic", original)
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    assert m.AsyncAnthropic("a", api_key="not-real", base_url="http://controlled.invalid", max_retries=2) == "original SDK"
    assert calls == [(("a",), {"api_key": "not-real", "base_url": "http://controlled.invalid", "max_retries": 2})]


def test_unsupported_platform_blocks_only_the_opt_in_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(q,"fcntl",None)
    with pytest.raises(q.ProtectedDemoQuotaError):q.QuotaStore(tmp_path.resolve()/"new",3)
    monkeypatch.setenv("ECHOMIND_ENV","development")
    monkeypatch.setattr(m.anthropic,"AsyncAnthropic",lambda **kwargs:kwargs)
    assert m.AsyncAnthropic(api_key="ordinary") == {"api_key":"ordinary"}


@pytest.mark.parametrize("value", [BASE, BASE+"/", "https://workspace-123.cn-beijing.maas.aliyuncs.com/apps/anthropic"])
def test_exact_beijing_official_endpoints(value):
    assert m.canonical_endpoint(value) == value.rstrip("/")


@pytest.mark.parametrize("value", ["http://dashscope.aliyuncs.com/apps/anthropic", BASE+"?x=1", BASE+"#x", BASE+"/other",
    "https://dashscope.aliyuncs.com:443/apps/anthropic", "https://u:p@dashscope.aliyuncs.com/apps/anthropic",
    "https://example.org/apps/anthropic", "https://a.cn-shanghai.maas.aliyuncs.com/apps/anthropic", BASE+" "])
def test_other_origins_ports_credentials_and_paths_reject(value):
    with pytest.raises(q.ProtectedDemoQuotaError): m.canonical_endpoint(value)


def test_real_sdk_request_is_counted_before_delegate_and_no_body_is_recorded(configured, monkeypatch):
    async def run():
        calls=[]
        async def delegate(req):
            assert q.status()["used"] == 1
            assert req.extensions["timeout"] == {k:30. for k in ("connect", "read", "write", "pool")}
            assert req.headers["accept-encoding"] == "identity"
            calls.append(json.loads(req.content))
            return response(req)
        monkeypatch.setattr(m, "_delegate", lambda: httpx.MockTransport(delegate))
        sdk=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE,max_retries=0)
        try:
            with q.request_scope():
                out=await sdk.messages.create(**ARGS)
                q.raise_if_request_blocked()
            assert out.content[0].text == "controlled provider output"
            assert calls == [ARGS]
            assert q.status()["used"] == 1 and not q.status()["unresolved"]
            raw=(configured/'journal.jsonl').read_text()
            assert "controlled private prompt" not in raw and "controlled provider output" not in raw
            assert "controlled-not-real" not in raw
        finally:await sdk.close()
    asyncio.run(run())


@pytest.mark.parametrize("options", [{"http_client":object()}, {"max_retries":1}, {"base_url":"https://example.com/apps/anthropic"},
    {"default_query":{"a":"b"}}, {"default_headers":{"Host":"example.com"}}])
def test_factory_cannot_bypass_counting_or_destination(configured, options):
    args={"api_key":"controlled-not-real","base_url":BASE,**options}
    with pytest.raises(q.ProtectedDemoQuotaError):m.AsyncAnthropic(**args)
    assert q.status()["used"] == 0


@pytest.mark.parametrize("fault", ["network", "http401", "http429", "oversize", "compressed", "cancel"])
def test_sent_failures_count_once_and_block_fallback_in_same_request(configured, monkeypatch, fault):
    async def run():
        calls=[]
        async def delegate(req):
            calls.append(req)
            if fault=="network":raise httpx.ReadTimeout("private upstream error",request=req)
            if fault=="cancel":raise asyncio.CancelledError()
            if fault.startswith('http'):return httpx.Response(int(fault[4:]),text="private error body",request=req)
            if fault=="compressed":return httpx.Response(200,stream=httpx.ByteStream(b"x"),headers={"content-encoding":"gzip"},request=req)
            return httpx.Response(200,content=b"x"*(m.RESPONSE_LIMIT+1),request=req)
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(delegate))
        sdk=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE,max_retries=0)
        try:
            with q.request_scope():
                with pytest.raises(BaseException):await sdk.messages.create(**ARGS)
                with pytest.raises(q.ProtectedDemoQuotaError):q.raise_if_request_blocked()
                with pytest.raises(BaseException):await sdk.messages.create(**ARGS)
            assert len(calls)==1 and q.status()["used"]==1 and q.status()["unresolved"]==0
            assert "private upstream error" not in (configured/'journal.jsonl').read_text()
        finally:await sdk.close()
    asyncio.run(run())


def test_quota_exhaustion_rejects_without_delegate_and_constructor_allows_readonly_start(configured, monkeypatch):
    async def run():
        sent=[]
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(lambda r:(sent.append(r),response(r))[1]))
        sdk=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            for _ in range(3):
                with q.request_scope():await sdk.messages.create(**ARGS)
            with q.request_scope():
                with pytest.raises(Exception):await sdk.messages.create(**ARGS)
                with pytest.raises(q.ProtectedDemoQuotaError,match="exhausted"):q.raise_if_request_blocked()
            assert len(sent)==3 and q.status()["used"]==3
            other=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
            await other.close()
        finally:await sdk.close()
    asyncio.run(run())


@pytest.mark.parametrize("change", [{"model":"arbitrary-model"},{"stream":True},{"max_tokens":4097},
    {"messages":[{"role":"user","content":"x"*65536}]},{"temperature":float('inf')}])
def test_invalid_or_oversize_payload_never_consumes_or_sends(configured,monkeypatch,change):
    async def run():
        sent=[]
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(lambda r:(sent.append(r),response(r))[1]))
        sdk=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            with q.request_scope():
                with pytest.raises(Exception):await sdk.messages.create(**{**ARGS,**change})
            assert not sent and q.status()["used"]==0
        finally:await sdk.close()
    asyncio.run(run())


def test_storage_loss_prevents_all_new_sends(configured,monkeypatch):
    async def run():
        sent=[]
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(lambda r:(sent.append(r),response(r))[1]))
        sdk=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            (configured/'journal.jsonl').unlink()
            with q.request_scope():
                with pytest.raises(Exception):await sdk.messages.create(**ARGS)
                with pytest.raises(q.ProtectedDemoQuotaError,match="unavailable"):q.raise_if_request_blocked()
            assert not sent and not (configured/'journal.jsonl').exists()
        finally:await sdk.close()
    asyncio.run(run())


def test_multiple_live_clients_share_one_cap_while_requests_overlap(configured, monkeypatch):
    async def run():
        sent=[]
        release=asyncio.Event()
        async def delegate(req):
            sent.append(req)
            await release.wait()
            return response(req)
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(delegate))
        clients=[m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE) for _ in range(4)]
        async def call(client):
            with q.request_scope():return await client.messages.create(**ARGS)
        tasks=[]
        try:
            for client in clients[:3]:
                tasks.append(asyncio.create_task(call(client)))
                for _ in range(100):
                    if len(sent)>=len(tasks):break
                    await asyncio.sleep(.001)
            assert len(sent)==q.status()["used"]==q.status()["unresolved"]==3
            with pytest.raises(Exception):await call(clients[3])
            assert len(sent)==3
            release.set()
            await asyncio.gather(*tasks)
            assert q.status()["used"]==3 and q.status()["unresolved"]==0
        finally:
            release.set()
            await asyncio.gather(*tasks,return_exceptions=True)
            await asyncio.gather(*(c.close() for c in clients))
    asyncio.run(run())


def test_actual_task_cancel_after_delegate_preserves_charge_and_closes_pending(configured, monkeypatch):
    async def run():
        entered=asyncio.Event()
        async def delegate(req):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(delegate))
        client=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            async def call():
                with q.request_scope():await client.messages.create(**ARGS)
            task=asyncio.create_task(call())
            await asyncio.wait_for(entered.wait(),1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
            assert q.status()["used"]==1 and q.status()["unresolved"]==0
            assert '"outcome":"cancelled"' in (configured/'journal.jsonl').read_text()
        finally:await client.close()
    asyncio.run(run())


def test_cancel_observed_after_permit_does_not_enter_delegate(configured, monkeypatch):
    async def run():
        sent=[]
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(lambda r:(sent.append(r),response(r))[1]))
        original=q.QuotaStore.reserve
        def cancel_after_reserve(store,**kwargs):
            operation=original(store,**kwargs)
            asyncio.current_task().cancel()
            return operation
        monkeypatch.setattr(q.QuotaStore,"reserve",cancel_after_reserve)
        client=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            async def call():
                with q.request_scope():await client.messages.create(**ARGS)
            task=asyncio.create_task(call())
            with pytest.raises(asyncio.CancelledError):await task
            assert not sent and q.status()["used"]==1 and q.status()["unresolved"]==0
            events=[json.loads(line) for line in (configured/'journal.jsonl').read_text().splitlines()]
            assert events[-1]['outcome']=='not_sent' and events[-1]['delegate_entered'] is False
        finally:await client.close()
    asyncio.run(run())


def test_failed_finish_is_not_retried_or_refunded(configured, monkeypatch):
    async def run():
        writes=[]
        original=q._write
        def uncertain(fd,raw):
            writes.append(raw)
            if len(writes)==2:raise OSError("private path and disk exception must not leak")
            original(fd,raw)
        monkeypatch.setattr(q,"_write",uncertain)
        monkeypatch.setattr(m,"_delegate",lambda:httpx.MockTransport(response))
        client=m.AsyncAnthropic(api_key="controlled-not-real",base_url=BASE)
        try:
            with q.request_scope():
                with pytest.raises(Exception):await client.messages.create(**ARGS)
                with pytest.raises(q.ProtectedDemoQuotaError,match='unavailable'):q.raise_if_request_blocked()
            assert len(writes)==2  # reserve + failed finish, no uncertain write retry
            assert q.status()['used']==q.status()['unresolved']==1
            with pytest.raises(q.ProtectedDemoQuotaError):q.ensure_available()
        finally:await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('slow_body',[True,False])
def test_entire_send_and_response_have_one_bounded_deadline(configured, monkeypatch, slow_body):
    async def run():
        sent=[]
        monkeypatch.setattr(m,'TIMEOUT',.02)
        async def delegate(req):
            sent.append(req)
            await asyncio.Event().wait()
        monkeypatch.setattr(m,'_delegate',lambda:httpx.MockTransport(delegate))
        transport=m.ProtectedDemoTransport(BASE,'provider')
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield b''
        request=(httpx.Request('POST',BASE+'/v1/messages',stream=Body()) if slow_body else
                 httpx.Request('POST',BASE+'/v1/messages',json=ARGS))
        try:
            with q.request_scope():
                with pytest.raises(q.ProtectedDemoQuotaError):
                    await asyncio.wait_for(transport.handle_async_request(request),.5)
            assert len(sent)==(0 if slow_body else 1)
            assert q.status()['used']==(0 if slow_body else 1)
            assert q.status()['unresolved']==0
        finally:await transport.aclose()
    asyncio.run(run())
