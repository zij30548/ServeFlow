"""The intent trial transport is exercised entirely with controlled transports."""
import asyncio
import hashlib
import json
import socket

import httpx
import pytest

from evaluation.intent_transport import (
    GuardedIntentTransport,
    TrialTransportError,
    canonical_payload_hash,
    messages_url,
    validated_endpoint,
)


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PAYLOAD = {"model": "qwen3.7-flash-2026-07-15", "messages": [{"role": "user", "content": "合成题"}]}
PRIVATE = "private-credential-and-provider-response"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Network is forbidden in intent transport tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def request(*, method="POST", url=None, body=None):
    return httpx.Request(method, url or messages_url(BASE),
                         content=json.dumps(PAYLOAD).encode() if body is None else body,
                         headers={"x-api-key": PRIVATE})


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0, close_error=False):
        self.chunks = chunks
        self.delay = delay
        self.close_error = close_error
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True
        if self.close_error:
            raise ValueError(PRIVATE)


class ControlledTransport(httpx.AsyncBaseTransport):
    def __init__(self, response=None, *, error=None, delay=0, close_error=False):
        self.response = response
        self.error = error
        self.delay = delay
        self.close_error = close_error
        self.calls = 0
        self.closed = False

    async def handle_async_request(self, req):
        self.calls += 1
        assert req.headers["accept-encoding"] == "identity"
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self):
        self.closed = True
        if self.close_error:
            raise ValueError(PRIVATE)


@pytest.mark.parametrize("base", [BASE, BASE + "/", "https://space-1.cn-beijing.maas.aliyuncs.com/apps/anthropic/"])
def test_endpoint_canonicalization(base):
    assert validated_endpoint(base) == base.rstrip("/")
    assert messages_url(base) == base.rstrip("/") + "/v1/messages"


@pytest.mark.parametrize("base", [
    "http://dashscope.aliyuncs.com/apps/anthropic", BASE.replace(".com/", ".com:443/"),
    BASE.replace(".com/", ".com:/"), BASE + "?", BASE + "#", BASE + "?key=secret",
    BASE.replace("https://", "https://user:secret@"), BASE + "/v1/messages",
    BASE.replace("dashscope.aliyuncs.com", "dashscope.aliyuncs.com.evil.invalid"),
    BASE.replace("dashscope.aliyuncs.com", "x.cn-shanghai.maas.aliyuncs.com"),
    BASE.replace("https://", "https://\n"), BASE + " ", BASE + "\\", None,
])
def test_endpoint_rejects_ambiguous_or_unplanned_origins(base):
    with pytest.raises(TrialTransportError, match="^invalid_endpoint$"):
        validated_endpoint(base)


def test_hash_is_canonical_and_uses_strict_json():
    assert canonical_payload_hash({"b": [True, None], "a": "合成"}) == canonical_payload_hash({"a": "合成", "b": [True, None]})
    for invalid in ({"x": float("nan")}, {"x": float("inf")}, {1: "x"}, {"x": (1, 2)}, [], {"x": "\ud800"}):
        with pytest.raises(TrialTransportError, match="^invalid_payload$"):
            canonical_payload_hash(invalid)


@pytest.mark.parametrize(("req", "code"), [
    (request(method="GET"), "invalid_request"),
    (request(url=messages_url(BASE) + "?"), "invalid_request"),
    (request(url=messages_url(BASE).replace("https:", "http:")), "invalid_request"),
    (request(url="https://example.invalid/v1/messages"), "invalid_request"),
    (request(body=b"{"), "invalid_payload"),
    (request(body=b'{"x":1,"x":2}'), "invalid_payload"),
    (request(body=b'{"x":{"a":1,"a":2}}'), "invalid_payload"),
    (request(body=b'{"x":NaN}'), "invalid_payload"),
    (request(body=b'{"x":Infinity}'), "invalid_payload"),
    (request(body=b'{"x":1e1000}'), "invalid_payload"),
    (request(body=b"[]"), "invalid_payload"),
    (request(body=b" " * 65_537), "invalid_payload"),
    (request(body=b'{"x":1}'), "payload_mismatch"),
])
def test_invalid_request_is_rejected_before_admission(req, code):
    async def run():
        admitted = []
        lower = ControlledTransport(httpx.Response(200, content=b"{}"))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: admitted.append(True), lower)
        with pytest.raises(TrialTransportError) as caught:
            await guard.handle_async_request(req)
        assert caught.value.code == guard.failure_code == code
        assert not guard.sent and lower.calls == 0 and admitted == [] and lower.closed
        assert PRIVATE not in str(caught.value)
    asyncio.run(run())


def test_admission_failure_never_enters_network_delegate():
    async def run():
        def fail():
            raise OSError(PRIVATE)
        lower = ControlledTransport(httpx.Response(200, content=b"{}"))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), fail, lower)
        with pytest.raises(TrialTransportError, match="^before_send_failed$"):
            await guard.handle_async_request(request())
        assert not guard.sent and lower.calls == 0 and lower.closed
    asyncio.run(run())


def test_raw_serialized_hash_is_available_to_admission_callback():
    async def run():
        req = request()
        observed = []
        guard = None
        def admit():
            observed.append(guard.request_bytes_sha256)
            assert not guard.sent
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), admit,
                                        httpx.MockTransport(lambda req: httpx.Response(200, content=b"{}")))
        await guard.handle_async_request(req)
        assert observed == [hashlib.sha256(req.content).hexdigest()]
    asyncio.run(run())


def test_success_preserves_bytes_and_only_necessary_response_headers():
    async def run():
        events = []
        stream = Chunks([b"{", b'"content":[]', b"}"])
        response = httpx.Response(200, stream=stream, headers={
            "content-type": "application/json", "x-request-id": "request-123",
            "set-cookie": PRIVATE, "authorization": PRIVATE,
        })
        class OrderedTransport(ControlledTransport):
            async def handle_async_request(self, req):
                events.append("send")
                return await super().handle_async_request(req)
        lower = OrderedTransport(response)
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: events.append("admit"), lower)
        result = await guard.handle_async_request(request())
        assert events == ["admit", "send"]
        assert result.content == b'{"content":[]}'
        assert result.headers["request-id"] == guard.request_id == "request-123"
        assert result.headers["content-type"] == "application/json"
        assert "set-cookie" not in result.headers and "authorization" not in result.headers
        assert guard.sent and guard.failure_code is None and guard.http_status == 200
        assert stream.closed and lower.closed and response.is_closed
        await guard.aclose()
        with pytest.raises(TrialTransportError, match="^already_sent$"):
            await guard.handle_async_request(request())
        assert lower.calls == 1
    asyncio.run(run())


@pytest.mark.parametrize(("status", "code"), [
    (301, "redirect"), (302, "redirect"), (307, "redirect"), (308, "redirect"),
    (400, "http_error"), (401, "authentication_error"), (402, "payment_required"),
    (403, "authentication_error"), (429, "rate_limited"), (500, "upstream_error"), (503, "upstream_error"),
])
def test_error_status_has_no_body_read_or_retry_even_if_client_follows_redirects(status, code):
    async def run():
        stream = Chunks([PRIVATE.encode()])
        response = httpx.Response(status, stream=stream, headers={"location": "https://example.invalid/"})
        sends = []
        def send(req):
            sends.append(req)
            return response
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, httpx.MockTransport(send))
        async with httpx.AsyncClient(transport=guard, follow_redirects=True) as client:
            with pytest.raises(TrialTransportError) as caught:
                await client.send(request())
        assert caught.value.code == guard.failure_code == code
        assert caught.value.http_status == guard.http_status == status
        assert len(sends) == 1 and stream.reads == 0 and stream.closed
        assert PRIVATE not in str(caught.value)
    asyncio.run(run())


def test_stream_limit_stops_at_first_excess_chunk_and_closes():
    async def run():
        stream = Chunks([b"123", b"456", b"must-not-be-read"])
        lower = ControlledTransport(httpx.Response(200, stream=stream))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower, response_limit=5)
        with pytest.raises(TrialTransportError, match="^response_limit$"):
            await guard.handle_async_request(request())
        assert stream.reads == 2 and stream.closed and lower.closed and lower.calls == 1
    asyncio.run(run())


@pytest.mark.parametrize("during_body", [False, True])
def test_total_timeout_covers_send_and_stream_read(during_body):
    async def run():
        stream = Chunks([b"a", b"b"], delay=.03 if during_body else 0)
        lower = ControlledTransport(httpx.Response(200, stream=stream), delay=0 if during_body else .1)
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower, timeout_s=.04)
        with pytest.raises(TrialTransportError, match="^timeout$"):
            await guard.handle_async_request(request())
        assert guard.failure_code == "timeout" and guard.sent and lower.calls == 1 and lower.closed
        if during_body:
            assert stream.reads == 1 and stream.closed
    asyncio.run(run())


def test_cancellation_is_fixed_code_and_never_retried():
    async def run():
        lower = ControlledTransport(error=asyncio.CancelledError(PRIVATE))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower)
        with pytest.raises(TrialTransportError, match="^cancelled$"):
            await guard.handle_async_request(request())
        assert guard.sent and guard.failure_code == "cancelled" and lower.calls == 1 and lower.closed
    asyncio.run(run())


def test_task_cancelled_while_reading_closes_response_and_delegate():
    async def run():
        entered = asyncio.Event()
        class WaitingStream(Chunks):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"unreachable"
        stream = WaitingStream([])
        lower = ControlledTransport(httpx.Response(200, stream=stream))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower)
        task = asyncio.create_task(guard.handle_async_request(request()))
        await entered.wait()
        task.cancel()
        with pytest.raises(TrialTransportError, match="^cancelled$"):
            await task
        assert guard.failure_code == "cancelled" and stream.closed and lower.closed
        assert lower.calls == 1
    asyncio.run(run())


def test_network_failure_and_cleanup_errors_do_not_expose_or_replace_primary_error():
    async def run():
        lower = ControlledTransport(error=httpx.ConnectError(PRIVATE), close_error=True)
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower)
        with pytest.raises(TrialTransportError, match="^network_error$"):
            await guard.handle_async_request(request())
        assert lower.calls == 1 and lower.closed
        with pytest.raises(TrialTransportError, match="^already_sent$"):
            await guard.handle_async_request(request())
        assert lower.calls == 1
        stream = Chunks([PRIVATE.encode()], close_error=True)
        lower2 = ControlledTransport(httpx.Response(429, stream=stream), close_error=True)
        guard2 = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower2)
        with pytest.raises(TrialTransportError, match="^rate_limited$"):
            await guard2.handle_async_request(request())
        assert guard2.failure_code == "rate_limited" and stream.closed and lower2.closed
    asyncio.run(run())


def test_close_failure_after_success_is_reported_with_a_safe_code():
    async def run():
        lower = ControlledTransport(httpx.Response(200, content=b"{}"), close_error=True)
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower)
        with pytest.raises(TrialTransportError, match="^network_error$"):
            await guard.handle_async_request(request())
        assert guard.failure_code == "network_error" and guard.http_status == 200
    asyncio.run(run())


def test_concurrent_calls_cannot_enter_delegate_twice():
    async def run():
        lower = ControlledTransport(httpx.Response(200, content=b"{}"), delay=.01)
        reservations = []
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD),
                                        lambda: reservations.append(True), lower)
        results = await asyncio.gather(guard.handle_async_request(request()),
                                       guard.handle_async_request(request()), return_exceptions=True)
        assert isinstance(results[0], httpx.Response)
        assert isinstance(results[1], TrialTransportError) and results[1].code == "already_sent"
        assert lower.calls == 1 and reservations == [True]
        assert guard.failure_code == "already_sent"
    asyncio.run(run())


def test_compressed_response_is_rejected_without_reading_and_bad_json_is_unchanged():
    async def run():
        stream = Chunks([b"not-read"])
        lower = ControlledTransport(httpx.Response(200, stream=stream, headers={"content-encoding": "gzip"}))
        guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None, lower)
        with pytest.raises(TrialTransportError, match="^response_encoding$"):
            await guard.handle_async_request(request())
        assert stream.reads == 0 and stream.closed
        guard2 = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None,
                                        httpx.MockTransport(lambda req: httpx.Response(200, content=b"{bad-json")))
        result = await guard2.handle_async_request(request())
        assert result.content == b"{bad-json" and guard2.failure_code is None
    asyncio.run(run())


def test_default_delegate_has_no_retries_or_environment_configuration(monkeypatch):
    options = []
    def build(**kwargs):
        options.append(kwargs)
        return ControlledTransport()
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", build)
    guard = GuardedIntentTransport(BASE, canonical_payload_hash(PAYLOAD), lambda: None)
    assert options == [{"retries": 0, "trust_env": False}]
    asyncio.run(guard.aclose())
