"""Offline only: real HTTPX/temporary journal, controlled provider vectors."""
import asyncio
import hashlib
import json
import socket

import httpx
import pytest

from core import qwen_embedding as e
from core import protected_demo_quota as q
from core import model_client as m
from core.intent_embedding import IntentEmbedding

BASE = "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
KEY = "controlled-only-not-a-real-key"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*a, **k):
        raise AssertionError("real services forbidden")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_EMBEDDING_API_KEY", KEY)
    monkeypatch.setenv("QWEN_EMBEDDING_BASE_URL", BASE)
    monkeypatch.setenv("QWEN_INTENT_CACHE_DIRECTORY", str(tmp_path.resolve() / "templates"))


@pytest.fixture
def counted(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    path = tmp_path.resolve() / "usage"
    monkeypatch.setenv("SERVEFLOW_DEMO_QUOTA_DIR", str(path))
    monkeypatch.setenv("SERVEFLOW_DEMO_MODEL_LIMIT", "5")
    q.initialize(path, 5)
    return path


def output(request, *, changes=None):
    texts = json.loads(request.content)["input"]
    data = {"model": e.MODEL, "object": "list", "usage": {"total_tokens": len(texts)},
            "data": [{"index": i, "embedding": [3., 4.] + [0.] * (e.DIMENSIONS - 2)}
                     for i in reversed(range(len(texts)))]}
    if changes:
        changes(data)
    return httpx.Response(200, json=data, request=request)


async def send(texts=None, **kwargs):
    return await e.request_embeddings(texts or ["controlled question"], base_url=BASE, api_key=KEY, **kwargs)


async def prepare_cache(embedding, templates, *, real_adapter=True):
    async def fetch(texts):
        if real_adapter: return await e.request_embeddings(texts, base_url=BASE, api_key=KEY, purpose='prepare')
        return [[1.] + [0.] * 1023 for _ in texts]
    count = (sum(len(v) for v in templates.values()) + 19) // 20
    return await embedding.template_cache.prepare(templates, fetch, max_requests=count)


@pytest.mark.parametrize("base", [BASE, "https://dashscope.aliyuncs.com/compatible-mode/v1"])
def test_exact_beijing_endpoint(base):
    assert e.canonical_base_url(base) == base


@pytest.mark.parametrize("base", [
    BASE + "/", BASE + "?", BASE + "#", BASE + "?token=private", BASE + "#private",
    BASE.replace("https:", "http:"), BASE.replace("cn-beijing", "cn-shanghai"),
    BASE.replace("https://", "https://user:password@"), BASE.replace(".com/", ".com:443/"),
    BASE.replace(".com/", ".com.evil.test/"), BASE.replace("/compatible-mode/v1", "/apps/anthropic"),
    "https://evil.test/compatible-mode/v1", " " + BASE, BASE + "\n", BASE.upper(),
])
def test_reject_other_regions_origins_paths_ports_and_url_components(base):
    with pytest.raises(e.EmbeddingError, match="beijing_endpoint"):
        e.canonical_base_url(base)


def test_httpx_request_fixed_model_parameters_no_metadata_and_order_reconstruction(monkeypatch):
    async def run():
        calls = []
        async def provider(request):
            calls.append(request)
            assert str(request.url) == BASE + "/embeddings"
            assert request.headers["authorization"] == "Bearer " + KEY
            assert request.headers["accept-encoding"] == "identity"
            assert json.loads(request.content) == {"model": e.MODEL, "dimensions": 1024,
                "encoding_format": "float", "input": ["first", "second"]}
            return output(request)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        vectors = await send(["first", "second"])
        assert len(calls) == 1 and len(vectors) == 2 and len(vectors[0]) == 1024
        assert vectors[0][:2] == [.6, .8]
    asyncio.run(run())


@pytest.mark.parametrize("texts", [[], [""], [" "], ["a"] * 21, [None], ["a" * 4001],
                                     ["中" * 4000] * 20])
def test_invalid_input_never_creates_transport_or_reserves(counted, monkeypatch, texts):
    monkeypatch.setattr(e, "_delegate", lambda: pytest.fail("invalid input sent"))
    with pytest.raises(e.EmbeddingError):
        asyncio.run(e.request_embeddings(texts, base_url=BASE, api_key=KEY))
    assert q.status()["used"] == 0


@pytest.mark.parametrize("damage", ["count", "dimension", "duplicate", "boolean_index", "nan", "zero",
                                    "bool_value", "wrong_model", "not_list", "bad_json", "duplicate_key"])
def test_schema_rejection_is_charged_and_raw_data_never_leaks(counted, monkeypatch, damage):
    async def run():
        def provider(request):
            def mutate(data):
                row = data["data"][0]
                if damage == "count": data["data"] = []
                elif damage == "dimension": row["embedding"].pop()
                elif damage == "duplicate": data["data"] = [row, row]
                elif damage == "boolean_index": row["index"] = True
                elif damage == "nan": row["embedding"][0] = "not-a-number"
                elif damage == "zero": row["embedding"] = [0.] * 1024
                elif damage == "bool_value": row["embedding"][0] = True
                elif damage == "wrong_model": data["model"] = "another-model"
                elif damage == "not_list": row["embedding"] = {"private": "body"}
            if damage == "bad_json": return httpx.Response(200, text="private invalid output")
            if damage == "duplicate_key": return httpx.Response(200, content=b'{"data":[],"data":[]}')
            return output(request, changes=mutate)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        with pytest.raises(e.EmbeddingError, match="^embedding_response_invalid$"):
            await send()
        assert q.status()["used"] == 1 and q.status()["unresolved"] == 0
        journal = (counted / "journal.jsonl").read_text()
        assert "private" not in journal and KEY not in journal and "controlled question" not in journal
    asyncio.run(run())


def test_template_batch_size_cache_invalidation_and_cold_single_flight(monkeypatch):
    async def run():
        calls = []
        def provider(request):
            calls.append(json.loads(request.content)["input"])
            return output(request)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        embedding = IntentEmbedding()
        templates = {"a": [str(i) for i in range(41)]}
        local = lambda t: [1., 0.]
        # Preparation is explicit; concurrent live questions never initialize.
        await prepare_cache(embedding, templates)
        results = await asyncio.gather(*(embedding.vectors("q" + str(i), templates, local) for i in range(3)))
        assert [len(c) for c in calls[:3]] == [20, 20, 1]
        assert len(calls) == 6  # three template batches + three questions
        assert all(r[2] == "qwen/" + e.MODEL and len(r[1]) == 1024 for r in results)
        assert embedding.status()["model"] == e.MODEL
        await embedding.vectors("later", templates, local)
        assert len(calls) == 7 and calls[-1] == ["later"]
        before = len(calls)
        assert (await embedding.vectors("after learn", {"a": ["new template"]}, local))[2] == 'local-ngram'
        assert len(calls) == before
        await prepare_cache(embedding, {"a": ["new template"]})
        await embedding.vectors("after learn", {"a": ["new template"]}, local)
        assert calls[-2:] == [["new template"], ["after learn"]]
    asyncio.run(run())


def test_query_failure_falls_back_to_local_space_with_visible_source_and_cooldown(monkeypatch):
    async def run():
        calls = []
        def provider(request):
            texts = json.loads(request.content)["input"]
            calls.append(texts)
            return httpx.Response(503, text="private body") if texts == ["question"] else output(request)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        embedding = IntentEmbedding()
        await prepare_cache(embedding, {"a": ["template"]})
        for question in ("question", "during cooldown"):
            groups, vector, source = await embedding.vectors(question, {"a": ["template"]}, lambda t: [1., 0.])
            assert source == "local-ngram" and len(vector) == len(groups["a"][0]) == 2
        assert len(calls) == 2 and embedding.status()["last_error"] == "embedding_http_error"
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["timeout", "network", "http401", "http429", "redirect", "compressed", "oversize", "cancel"])
def test_failures_count_once_without_retry_and_close_response(counted, monkeypatch, fault):
    async def run():
        calls, streams = [], []
        class Body(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                yield b"x" * (e.RESPONSE_LIMIT + 1)
            async def aclose(self): self.closed = True
        async def provider(request):
            calls.append(request)
            if fault == "timeout": raise httpx.ReadTimeout("private timeout", request=request)
            if fault == "network": raise httpx.ConnectError("private network", request=request)
            if fault == "cancel": raise asyncio.CancelledError()
            if fault.startswith("http"): return httpx.Response(int(fault[4:]), text="private body")
            if fault == "redirect": return httpx.Response(307, headers={"location": "https://evil.test/"})
            if fault == "compressed": return httpx.Response(200, content=b"private", headers={"content-encoding": "gzip"})
            stream = Body(); streams.append(stream)
            return httpx.Response(200, stream=stream)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else e.EmbeddingError): await send()
        assert len(calls) == q.status()["used"] == 1 and q.status()["unresolved"] == 0
        assert all(s.closed for s in streams)
        assert "private" not in (counted / "journal.jsonl").read_text()
    asyncio.run(run())


def test_total_response_deadline_and_cancellation_keep_occupied_permit(counted, monkeypatch):
    async def run():
        entered = asyncio.Event()
        async def provider(request):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        task = asyncio.create_task(send())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert q.status()["used"] == 1 and q.status()["unresolved"] == 0
        monkeypatch.setattr(e, "TIMEOUT", .02)
        with pytest.raises(e.EmbeddingError, match="timeout"): await asyncio.wait_for(send(), .5)
        assert q.status()["used"] == 2
    asyncio.run(run())


def test_old_prefix_is_preserved_and_messages_and_embedding_share_same_cap(counted, monkeypatch):
    async def run():
        store = q.QuotaStore.from_env()
        old = store.reserve(model=q.MODEL, phase="agent", payload_sha256="a" * 64)
        store.finish(old, "received", 200)
        prefix = (counted / "journal.jsonl").read_bytes()
        identity = (counted / "identity.json").read_bytes()
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(output))
        for _ in range(4): await send()
        with pytest.raises(q.ProtectedDemoQuotaError, match="exhausted"): await send()
        assert q.QuotaStore(counted, 5).status()["used"] == 5
        assert (counted / "journal.jsonl").read_bytes().startswith(prefix)
        assert (counted / "identity.json").read_bytes() == identity
        records = [json.loads(line) for line in (counted / "journal.jsonl").read_text().splitlines()]
        assert records[2]["phase"] == "intent_embedding" and records[2]["model"] == e.MODEL
        assert records[2]["payload_sha256"] == hashlib.sha256(e._payload(["controlled question"])).hexdigest()
    asyncio.run(run())


def test_simultaneous_requests_cannot_exceed_global_cap(counted, monkeypatch):
    async def run():
        release, sent = asyncio.Event(), []
        async def provider(request):
            sent.append(request)
            await release.wait()
            return output(request)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        tasks = [asyncio.create_task(send()) for _ in range(8)]
        for _ in range(100):
            if len(sent) == 5: break
            await asyncio.sleep(.001)
        assert len(sent) == q.status()["used"] == 5
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(isinstance(x, q.ProtectedDemoQuotaError) for x in results) == 3
        assert q.status()["unresolved"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["missing", "unknown", "failed_finish"])
def test_uncertain_or_broken_accounting_blocks_fallback_and_new_sends(counted, monkeypatch, fault):
    async def run():
        calls = []
        embedding = IntentEmbedding()
        await prepare_cache(embedding, {"a": ["t"]}, real_adapter=False)
        def provider(request): calls.append(request); return output(request)
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(provider))
        if fault == "missing": (counted / "journal.jsonl").unlink()
        elif fault == "unknown":
            operation = q.QuotaStore.from_env().reserve(model=q.MODEL, phase="agent", payload_sha256="a" * 64)
            q._ACTIVE.discard((str(counted), operation))
        else:
            original = q._write
            writes = []
            def fail_finish(fd, raw):
                writes.append(raw)
                if len(writes) == 2: raise OSError("private disk failure")
                original(fd, raw)
            monkeypatch.setattr(q, "_write", fail_finish)
        with q.request_scope():
            with pytest.raises(q.ProtectedDemoQuotaError):
                await embedding.vectors("q", {"a": ["t"]}, lambda t: [1., 0.])
            with pytest.raises(q.ProtectedDemoQuotaError): q.raise_if_request_blocked()
            with pytest.raises(q.ProtectedDemoQuotaError): await send()
        assert len(calls) == (1 if fault == "failed_finish" else 0)
    asyncio.run(run())


@pytest.mark.parametrize("phase,model", [("intent", e.MODEL), ("intent_embedding", q.MODEL),
                                         ("intent_embedding", "other-model")])
def test_model_phase_cross_wiring_rejected(counted, phase, model):
    with pytest.raises(q.ProtectedDemoQuotaError):
        q.QuotaStore.from_env().reserve(model=model, phase=phase, payload_sha256="a" * 64)
    assert q.status()["used"] == 0


def test_protected_direct_call_cannot_override_configured_destination(counted, monkeypatch):
    monkeypatch.setattr(e, "_delegate", lambda: pytest.fail("overridden configuration sent"))
    with pytest.raises(q.ProtectedDemoQuotaError):
        asyncio.run(e.request_embeddings(["q"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=KEY))
    assert q.status()["used"] == 0


def test_real_sdk_and_recognizer_use_qwen_templates_and_messages_in_same_request(counted, monkeypatch):
    async def run():
        from core.intent_recognizer import IntentRecognizer
        calls = []
        messages_base = BASE.removesuffix("/compatible-mode/v1") + "/apps/anthropic"
        monkeypatch.setenv("ANTHROPIC_MODEL", q.MODEL)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", messages_base)
        def embedding_provider(request):
            calls.append(json.loads(request.content)["model"])
            return output(request)
        def chat_provider(request):
            calls.append(json.loads(request.content)["model"])
            return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
                "model": q.MODEL, "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [{"type": "text", "text": json.dumps({"intent": "refund", "confidence": .95,
                    "reasoning": "controlled routing observation"})}]})
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(embedding_provider))
        monkeypatch.setattr(m, "_delegate", lambda: httpx.MockTransport(chat_provider))
        recognizer = IntentRecognizer(KEY, base_url=messages_base, model=q.MODEL)
        try:
            from core.intent_recognizer import _TEMPLATES
            await prepare_cache(recognizer._embedding, _TEMPLATES)
            with q.request_scope():
                result = await recognizer.recognize("退款进度怎么查")
                q.raise_if_request_blocked()
            assert result.embedding_source == "qwen/" + e.MODEL
            assert set(result.source_scores) >= {"llm", "embedding", "pattern"}
            assert calls.count(q.MODEL) == 1 and calls.count(e.MODEL) == 4
            assert q.status()["used"] == 5
            # Request cache returns the same observation without another send.
            await recognizer.recognize("退款进度怎么查")
            assert len(calls) == 5
        finally:
            await recognizer.client.close()
    asyncio.run(run())


def test_cancellation_after_reservation_before_send_is_not_refunded(counted, monkeypatch):
    async def run():
        monkeypatch.setattr(e, "_delegate", lambda: httpx.MockTransport(lambda r: pytest.fail("cancelled operation sent")))
        original = q.QuotaStore.reserve
        def cancel(store, **kwargs):
            operation = original(store, **kwargs)
            asyncio.current_task().cancel()
            return operation
        monkeypatch.setattr(q.QuotaStore, "reserve", cancel)
        with pytest.raises(asyncio.CancelledError): await asyncio.create_task(send())
        assert q.status()["used"] == 1 and q.status()["unresolved"] == 0
        last = json.loads((counted / "journal.jsonl").read_text().splitlines()[-1])
        assert last["outcome"] == "not_sent" and last["delegate_entered"] is False
    asyncio.run(run())


@pytest.mark.parametrize("raw", [b'{"data":[{"index":0,"embedding":[NaN]}]}',
    b'{"data":[{"index":0,"embedding":[Infinity]}]}', b'\xff'])
def test_invalid_numeric_json_and_encoding_are_safe(raw):
    with pytest.raises(e.EmbeddingError, match="response_invalid"): e.parse_vectors(raw, 1)


def test_duplicate_indices_are_not_repaired_or_truncated():
    row = {"index": 0, "embedding": [1.] * 1024}
    with pytest.raises(e.EmbeddingError): e.parse_vectors(json.dumps({"data": [row, row]}), 2)


def test_provider_error_diagnostic_is_always_a_fixed_code():
    assert str(e.EmbeddingError("private upstream text and credentials")) == "embedding_unavailable"


@pytest.mark.parametrize("key", ["", "short", "x" * 513, "x" * 16 + "\n"])
def test_explicit_qwen_requires_valid_key_without_silent_local_mode(monkeypatch, key):
    monkeypatch.setenv("QWEN_EMBEDDING_API_KEY", key)
    with pytest.raises(e.EmbeddingError, match="api_key"): IntentEmbedding()
