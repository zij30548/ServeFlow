"""Only fake database, Chroma and embeddings; no live stores or downloads."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import pytest

from demo_business import store as demo
from evaluation import task_data as data


URL = "mysql://eval-reader:fake-private-password@127.0.0.1:13306/serveflow_demo_eval_tasks"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("task data tests cannot contact a service")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


class Cursor:
    def __init__(self, grants):
        self.grants, self.calls = grants, []
    def __enter__(self): return self
    def __exit__(self, *_args): pass
    def execute(self, sql): self.calls.append(sql)
    def fetchall(self): return [{"grant": line} for line in self.grants]


class Connection:
    def __init__(self, grants): self.cur, self.closed = Cursor(grants), 0
    def cursor(self): return self.cur
    def close(self): self.closed += 1


def grants(database="serveflow_demo_eval_tasks"):
    return ["GRANT USAGE ON *.* TO `eval-reader`@`localhost`",
            f"GRANT SELECT ON `{database}`.* TO `eval-reader`@`localhost`"]


@pytest.mark.parametrize("url", ["", None, "mysql://u:p@127.0.0.1/serveflow_demo_original",
    "mysql://u:p@remote/serveflow_demo_eval_tasks", "mysql://u:p@127.0.0.1/serveflow_demo_eval_",
    "mysql://u:p@127.0.0.1/serveflow_demo_eval_tasks?read_only=true"])
def test_explicit_eval_url_required_no_environment_fallback(monkeypatch, url):
    monkeypatch.setenv("SERVEFLOW_DEMO_MYSQL_URL", URL)
    with pytest.raises(data.TaskDataError): data.parse_mysql_url(url)


def test_grants_require_readonly_eval_scope_and_close_connection():
    connection = Connection(grants())
    result = data._verify_grants(data.parse_mysql_url(URL), lambda _: connection)
    assert len(result) == 64
    assert connection.cur.calls == ["START TRANSACTION READ ONLY", "SHOW GRANTS FOR CURRENT_USER()"]
    assert connection.closed == 1


@pytest.mark.parametrize("lines", [
    ["GRANT ALL PRIVILEGES ON `serveflow_demo_eval_tasks`.* TO `u`@`localhost`"],
    ["GRANT SELECT, INSERT ON `serveflow_demo_eval_tasks`.* TO `u`@`localhost`"],
    ["GRANT SELECT ON *.* TO `u`@`localhost`"],
    ["GRANT SELECT ON `serveflow_demo_original`.* TO `u`@`localhost`"],
    ["GRANT SELECT ON `serveflow_demo_eval_tasks`.* TO `u`@`localhost` WITH GRANT OPTION"],
    ["GRANT `readonly_role`@`%` TO `u`@`localhost`"],
    ["GRANT USAGE ON *.* TO `u`@`localhost`"],
])
def test_global_other_database_write_roles_and_grant_option_refused(lines):
    connection = Connection(lines)
    with pytest.raises(data.TaskDataError, match="evaluation_select_only_account_required"):
        data._verify_grants(data.parse_mysql_url(URL), lambda _: connection)
    assert connection.closed == 1


def test_database_fixture_verification_uses_existing_verifier_and_redacts_config(monkeypatch):
    config = data.parse_mysql_url(URL)
    connection = Connection(grants())
    monkeypatch.setattr(demo, "connect", lambda _: connection)
    calls = []
    fixture = data._fixture()
    expected = {"schema_version": demo.SCHEMA_VERSION, "seed_sha256": demo.fixture_hash(),
                "is_demo": True, "as_of": fixture["as_of"],
                "counts": {name: len(fixture[name]) for name in demo.TABLE_COLUMNS}}
    def verify(actual):
        calls.append(actual)
        return deepcopy(expected)
    monkeypatch.setattr(data, "verify_database", verify)
    store, proof = data._verify_mysql(config)
    assert isinstance(store, demo.DemoBusinessStore) and calls == [config]
    assert proof["fixture_file_sha256"] == data.FIXTURE_SHA256
    assert len(proof["configuration_identity_sha256"]) == 64
    serialized = json.dumps(proof)
    assert "fake-private-password" not in serialized and "eval-reader" not in serialized and URL not in serialized
    monkeypatch.setattr(data, "verify_database", lambda _: {**expected, "seed_sha256": "a" * 64})
    with pytest.raises(data.TaskDataError, match="evaluation_database_fixture_mismatch"):
        data._verify_mysql(config)


def test_mysql_errors_never_echo_driver_message():
    def broken(_): raise RuntimeError(URL)
    with pytest.raises(data.TaskDataError) as caught:
        data._verify_grants(data.parse_mysql_url(URL), broken)
    assert str(caught.value) == "evaluation_mysql_preflight_failed"
    assert "password" not in str(caught.value)


def fake_cache(tmp_path, monkeypatch):
    hashes = {}
    for name in data.MODEL_FILES:
        raw = ("controlled model file " + name).encode()
        (tmp_path / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(data, "MODEL_FILES", hashes)
    return hashes


def test_model_cache_hashes_missing_corruption_and_symlinks(tmp_path, monkeypatch):
    hashes = fake_cache(tmp_path, monkeypatch)
    proof = data.model_cache_proof(tmp_path)
    assert proof["downloads"] == "forbidden" and set(proof["files"]) == set(hashes)
    (tmp_path / "model.onnx").write_bytes(b"changed")
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_changed"):
        data.model_cache_proof(tmp_path)
    (tmp_path / "model.onnx").unlink()
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_unavailable"):
        data.model_cache_proof(tmp_path)
    (tmp_path / "model.onnx").symlink_to(tmp_path / "config.json")
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_unavailable"):
        data.model_cache_proof(tmp_path)


def test_cached_embedding_never_calls_download_or_substitutes_model(monkeypatch):
    proof = {"model": data.MODEL_NAME, "files": {"controlled": True}}
    monkeypatch.setattr(data, "model_cache_proof", lambda: deepcopy(proof))
    class DefaultEmbedding:
        def __call__(self, input):
            self._download_model_if_not_exists()
            return [[1.0, 0.0] for _ in input]
        def _download_model_if_not_exists(self):
            raise AssertionError("default downloader must not run")
    embedding = data._cached_only_embedding(DefaultEmbedding, proof)
    assert embedding(["text"]) == [[1.0, 0.0]]
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_download_forbidden"):
        embedding._download("https://invalid.example", "unused")
    monkeypatch.setattr(data, "model_cache_proof", lambda: {"changed": True})
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_changed"):
        embedding(["text"])


class Collection:
    def __init__(self): self.rows = {}; self.query_args = None
    def get(self, where, include):
        found = [(key, doc, meta) for key, (doc, meta) in self.rows.items()
                 if all(meta.get(k) == v for k, v in where.items())]
        return {"ids": [row[0] for row in found], "metadatas": [row[2] for row in found]}
    def upsert(self, ids, documents, metadatas):
        for key, doc, meta in zip(ids, documents, metadatas): self.rows[key] = (doc, meta)
    def delete(self, ids):
        for key in ids: self.rows.pop(key, None)
    def count(self): return len(self.rows)
    def query(self, **kwargs):
        self.query_args = kwargs
        rows = list(self.rows.values())[:kwargs["n_results"]]
        return {"documents": [[row[0] for row in rows]], "metadatas": [[row[1] for row in rows]],
                "distances": [[.1] * len(rows)]}


class ChromaClient:
    def __init__(self): self.collection = Collection(); self.created = []; self.deleted = []
    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        return self.collection
    def delete_collection(self, name): self.deleted.append(name)


def test_knowledge_reuses_real_kb_methods_but_only_fake_ephemeral_storage(monkeypatch):
    import chromadb
    client = ChromaClient()
    monkeypatch.setattr(chromadb, "EphemeralClient", lambda **kwargs: client)
    def forbidden(*_args, **_kwargs): raise AssertionError("persistent or HTTP client forbidden")
    monkeypatch.setattr(chromadb, "HttpClient", forbidden)
    monkeypatch.setattr(chromadb, "PersistentClient", forbidden)
    monkeypatch.setattr(data.metadata, "version", lambda _: data.CHROMA_VERSION)
    monkeypatch.setattr(data, "model_cache_proof", lambda: {"model": data.MODEL_NAME, "files": {}})
    monkeypatch.setattr(data, "_cached_only_embedding", lambda *_: object())
    runtime = data._build_knowledge()
    assert runtime.proof["document_count"] == 6
    assert runtime.proof["chunk_count"] == runtime.kb.doc_count == 6
    assert client.created[0]["name"].startswith("serveflow_task_eval_")
    assert client.created[0]["name"] != "knowledge_base"
    result = asyncio.run(runtime.kb.search_handler({"query": "配送", "top_k": 1}, None))
    assert len(result) == 1 and result[0]["content"]
    assert client.collection.query_args == {"query_texts": ["配送"], "n_results": 1}
    runtime.close()
    assert client.deleted == [client.created[0]["name"]]


def test_knowledge_missing_cache_stops_before_any_client(monkeypatch):
    import chromadb
    monkeypatch.setattr(data.metadata, "version", lambda _: data.CHROMA_VERSION)
    monkeypatch.setattr(data, "model_cache_proof", lambda: (_ for _ in ()).throw(data.TaskDataError("evaluation_embedding_cache_unavailable")))
    monkeypatch.setattr(chromadb, "EphemeralClient", lambda **_: (_ for _ in ()).throw(AssertionError("must not construct")))
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_unavailable"):
        data._build_knowledge()


def test_controlled_fixture_reader_preserves_ownership_null_and_raw_money():
    async def run():
        found = await data.fixture_order_reader("demo-customer-a", "DEMO-1004")
        assert found["source"] == demo.SOURCE and found["is_demo"] is True
        assert found["evaluation_data_kind"] == "controlled-fixture-projection"
        assert found["order"]["refunds"][0]["amount_minor"] is None
        assert found["order"]["total_amount_minor"] == 29900
        found["order"]["total_amount_minor"] = 1
        again = await data.fixture_order_reader("demo-customer-a", "DEMO-1004")
        assert again["order"]["total_amount_minor"] == 29900
        denied = await data.fixture_order_reader("demo-customer-a", "DEMO-1005")
        assert denied["status"] == "not_found" and denied["order"] is None
        allowed = await data.fixture_order_reader("demo-customer-b", "DEMO-1005")
        assert allowed["status"] == "found" and allowed["order"]["customer_id"] == "demo-customer-b"
    asyncio.run(run())


def test_live_data_cancel_and_close_wait_for_inflight_work_once():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        closed = []
        async def read(customer, order):
            started.set(); await release.wait(); return await data.fixture_order_reader(customer, order)
        knowledge = SimpleNamespace(close=lambda: closed.append("closed"))
        live = data.LiveData(SimpleNamespace(read_order=read), knowledge, {"kind": "controlled"})
        call = asyncio.create_task(live.order_reader("demo-customer-a", "DEMO-1001"))
        await started.wait(); call.cancel()
        cleanup = asyncio.create_task(live.close())
        await asyncio.sleep(0)
        assert not cleanup.done() and not call.done() and not closed
        release.set()
        with pytest.raises(asyncio.CancelledError): await call
        await cleanup; await live.close()
        assert closed == ["closed"]
        with pytest.raises(data.TaskDataError, match="evaluation_data_closed"):
            await live.order_reader("demo-customer-a", "DEMO-1001")
    asyncio.run(run())


def test_build_live_data_is_explicit_and_cancellation_cleans_created_knowledge(monkeypatch):
    async def run():
        started, release = threading.Event(), threading.Event()
        closed = []
        knowledge = SimpleNamespace(close=lambda: closed.append("closed"), proof={"kind": "controlled"})
        monkeypatch.setattr(data, "_verify_mysql", lambda config: (object(), {"kind": "controlled"}))
        def build():
            started.set(); assert release.wait(2); return knowledge
        monkeypatch.setattr(data, "_build_knowledge", build)
        task = asyncio.create_task(data.build_live_data(URL))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel(); await asyncio.sleep(0)
        assert not task.done() and not closed
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert closed == ["closed"]
    asyncio.run(run())


def test_offline_proof_never_constructs_client_database_or_embedding(tmp_path, monkeypatch):
    source = tmp_path / "embedding.py"; source.write_text("controlled implementation")
    monkeypatch.setattr(data.metadata, "version", lambda _: data.CHROMA_VERSION)
    monkeypatch.setattr(data.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: source))
    monkeypatch.setattr(data, "model_cache_proof", lambda: {"model": data.MODEL_NAME, "files": {"fake": True}})
    monkeypatch.setattr(demo, "connect", lambda _: (_ for _ in ()).throw(AssertionError("no DB")))
    monkeypatch.setattr(data, "_build_knowledge", lambda: (_ for _ in ()).throw(AssertionError("no Chroma")))
    monkeypatch.setenv("SERVEFLOW_DEMO_MYSQL_URL", URL)
    proof = data.offline_data_proof()
    assert proof["mysql_preflight"] == "required-not-performed"
    assert proof["embedding_inference"] == "not-performed"
    assert proof["knowledge"]["document_count"] == proof["knowledge"]["chunk_count"] == 6
    assert proof["knowledge"]["embedding_implementation_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "fake-private-password" not in json.dumps(proof)


def healthy_live(store=None, kb=None):
    if store is None: store = SimpleNamespace(read_order=data.fixture_order_reader)
    if kb is None:
        async def query(params, context):
            doc = data._policies()[0]
            return [{"title": doc["title"], "content": doc["content"],
                     "version": hashlib.sha256(doc["content"].encode()).hexdigest()}]
        kb = SimpleNamespace(search_handler=query)
    return data.LiveData(store, SimpleNamespace(kb=kb, close=lambda: None), {"kind": "controlled"})


def test_live_data_correct_other_legal_order_is_not_marked_as_drift():
    async def run():
        live = healthy_live()
        first = await live.order_reader("demo-customer-a", "DEMO-1004")
        second = await live.order_reader("demo-customer-a", "DEMO-1001")
        denied = await live.order_reader("demo-customer-a", "DEMO-1005")
        unknown = await live.order_reader("demo-customer-a", "DEMO-9999")
        assert first["order"]["order_id"] == "DEMO-1004"
        assert second["order"]["order_id"] == "DEMO-1001"
        assert denied["status"] == unknown["status"] == "not_found"
        live.assert_healthy(); assert live.fault is None
        await live.close()
    asyncio.run(run())


def test_live_data_mysql_mutation_sets_sticky_memory_fault_without_extra_queries():
    async def run():
        calls = []
        async def mutate(customer, order):
            calls.append((customer, order))
            packet = await data.fixture_order_reader(customer, order)
            packet["order"]["total_amount_minor"] += 1
            return packet
        live = healthy_live(SimpleNamespace(read_order=mutate))
        with pytest.raises(data.TaskDataError, match="eval_mysql_changed"):
            await live.order_reader("demo-customer-a", "DEMO-1004")
        assert live.fault == "eval_mysql_changed"
        for _ in range(3):
            with pytest.raises(data.TaskDataError, match="eval_mysql_changed"): live.assert_healthy()
        with pytest.raises(data.TaskDataError, match="eval_mysql_changed"):
            await live.order_reader("demo-customer-a", "DEMO-1001")
        assert calls == [("demo-customer-a", "DEMO-1004")]
        await live.close()
    asyncio.run(run())


def test_live_data_mysql_failure_is_safe_and_stops_later_reads():
    async def run():
        async def failure(*_): raise RuntimeError(URL)
        live = healthy_live(SimpleNamespace(read_order=failure))
        with pytest.raises(data.TaskDataError) as caught:
            await live.order_reader("demo-customer-a", "DEMO-1004")
        assert str(caught.value) == live.fault == "eval_mysql_unavailable"
        assert caught.value.__suppress_context__ is True
        await live.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure", [False, True])
def test_knowledge_failure_or_changed_content_sets_memory_fault(failure):
    async def run():
        async def search(*_):
            if failure: raise RuntimeError(URL)
            return [{"title": "退款政策", "content": "unrecorded material", "version": "fake"}]
        live = healthy_live(kb=SimpleNamespace(search_handler=search))
        expected = "eval_knowledge_unavailable" if failure else "eval_knowledge_changed"
        with pytest.raises(data.TaskDataError, match=expected): await live.knowledge_handler({"query": "政策"}, None)
        with pytest.raises(data.TaskDataError, match=expected): live.assert_healthy()
        await live.close()
    asyncio.run(run())


def test_knowledge_success_reads_pinned_corpus_and_does_not_mutate_snapshot():
    async def run():
        live = healthy_live()
        rows = await live.knowledge_handler({"query": "政策"}, None)
        rows[0]["content"] = "client changed"
        again = await live.knowledge_handler({"query": "政策"}, None)
        assert again[0]["content"] != "client changed"
        assert live.fault is None
        live.assert_healthy(); await live.close()
    asyncio.run(run())


def test_knowledge_index_failure_deletes_only_new_collection(monkeypatch):
    import chromadb
    client = ChromaClient()
    def broken(**kwargs): raise RuntimeError(URL)
    client.collection.upsert = broken
    monkeypatch.setattr(chromadb, "EphemeralClient", lambda **kwargs: client)
    monkeypatch.setattr(data.metadata, "version", lambda _: data.CHROMA_VERSION)
    monkeypatch.setattr(data, "model_cache_proof", lambda: {"model": data.MODEL_NAME, "files": {}})
    monkeypatch.setattr(data, "_cached_only_embedding", lambda *_: object())
    with pytest.raises(data.TaskDataError, match="evaluation_knowledge_preflight_failed"):
        data._build_knowledge()
    assert client.deleted == [client.created[0]["name"]]
    assert client.deleted[0] != "knowledge_base"


def test_embedding_cache_change_during_forward_is_rejected(monkeypatch):
    initial = {"files": {"version": 1}}
    state = deepcopy(initial)
    monkeypatch.setattr(data, "model_cache_proof", lambda: deepcopy(state))
    class DefaultEmbedding:
        def __call__(self, input):
            self._download_model_if_not_exists()
            state["files"]["version"] = 2
            return [[1.0]]
    embedding = data._cached_only_embedding(DefaultEmbedding, initial)
    with pytest.raises(data.TaskDataError, match="evaluation_embedding_cache_changed"):
        embedding(["text"])
