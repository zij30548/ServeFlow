"""Isolated data adapters for task evaluation; importing performs no I/O.

Live mode needs an explicitly supplied localhost evaluation MySQL URL. It never
initializes a database, reads application environment files or opens an existing
Chroma collection. The policy corpus and already-cached default ONNX model are
byte-bound; missing dependencies/cache cause a safe failure, never a download.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import stat
import threading
import uuid
from typing import Any

from demo_business import store as demo
from demo_business.initialize import verify_database


BACKEND_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = BACKEND_ROOT / "knowledge/serveflow-demo-v2.json"
CORPUS_SHA256 = "0bdf8fea5d9cf758ceff80ba215382461a3589f11579e472b2ef29bde6ef8cb4"
FIXTURE_SHA256 = "5b7132202bc66f0141ab70e251ca293f89a743d8c15e589e6435c0cb8ccb4ddd"
EVAL_DATABASE = re.compile(r"serveflow_demo_eval_[a-z0-9_]{1,39}\Z")
CHROMA_VERSION = "0.5.23"
MODEL_NAME = "all-MiniLM-L6-v2"
# Measured existing cache bytes, not an independent provenance attestation.
MODEL_FILES = {
    "config.json": "b567c7d5a55b636c95186aaf993f9a8920842b7e05a9e703e68b23cab2c3a670",
    "model.onnx": "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "7702051bbc4953b94d47fa1d61b42ed4cbb3c71b501a8dd7183a823f8bea1f20",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


class TaskDataError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _fixture() -> dict:
    try:
        raw = demo.FIXTURE_PATH.read_bytes()
        if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise TaskDataError("evaluation_fixture_changed")
        return deepcopy(demo.validate_fixture(json.loads(raw)))
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_fixture_invalid") from None


def _policies() -> list[dict]:
    try:
        raw = CORPUS_PATH.read_bytes()
        if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
            raise TaskDataError("evaluation_corpus_changed")
        docs = json.loads(raw)
        if (type(docs) is not list or len(docs) != 6
                or any(type(d) is not dict or set(d) != {"title", "content"}
                       or any(type(d[k]) is not str or not d[k].strip() for k in ("title", "content")) for d in docs)
                or len({d["title"] for d in docs}) != 6):
            raise TaskDataError("evaluation_corpus_invalid")
        return docs
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_corpus_invalid") from None


def parse_mysql_url(mysql_url: str) -> demo.DemoDatabaseConfig:
    try:
        config = demo.DemoDatabaseConfig.parse(mysql_url)
    except Exception:
        raise TaskDataError("evaluation_mysql_url_required") from None
    if not EVAL_DATABASE.fullmatch(config.database):
        raise TaskDataError("evaluation_database_name_required")
    return config


def _verify_grants(config: demo.DemoDatabaseConfig, connector) -> str:
    connection = None
    try:
        connection = connector(config)
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            # Database names contain underscores. With ordinary MySQL grants
            # these may be wildcards even inside backticks; require literal
            # schema grants on this dedicated evaluation server.
            cursor.execute("SELECT @@GLOBAL.partial_revokes AS partial_revokes")
            literal_scope = cursor.fetchone()
            if (type(literal_scope) is not dict or set(literal_scope) != {"partial_revokes"}
                    or type(literal_scope["partial_revokes"]) is not int or literal_scope["partial_revokes"] != 1):
                raise TaskDataError("evaluation_literal_schema_grants_required")
            cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
            rows = cursor.fetchall()
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise TaskDataError("evaluation_select_only_account_required")
        statements = []
        has_select = False
        for row in rows:
            if type(row) is not dict or len(row) != 1:
                raise TaskDataError("evaluation_select_only_account_required")
            line = next(iter(row.values()))
            if type(line) is not str or len(line) > 4000 or ";" in line or any(ord(char) < 32 for char in line):
                raise TaskDataError("evaluation_select_only_account_required")
            match = re.fullmatch(r"GRANT\s+(USAGE|SELECT)\s+ON\s+(\S+)\s+TO\s+.+", line, re.I)
            if not match or re.search(r"\b(?:WITH|IDENTIFIED)\b", line, re.I):
                raise TaskDataError("evaluation_select_only_account_required")
            privilege, target = match.groups()
            target = target.replace("`", "")
            if privilege.upper() == "USAGE":
                if target != "*.*":
                    raise TaskDataError("evaluation_select_only_account_required")
            elif target != config.database + ".*":
                raise TaskDataError("evaluation_select_only_account_required")
            else:
                has_select = True
            statements.append(line)
        if not has_select:
            raise TaskDataError("evaluation_select_only_account_required")
        return _hash({"statements": sorted(statements), "partial_revokes": True})
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_mysql_preflight_failed") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _verify_mysql(config: demo.DemoDatabaseConfig) -> tuple[demo.DemoBusinessStore, dict]:
    fixture = _fixture()
    grants = _verify_grants(config, demo.connect)
    try:
        result = verify_database(config)
        expected_counts = {key: len(fixture[key]) for key in demo.TABLE_COLUMNS}
        expected_seed = hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if (result != {"schema_version": demo.SCHEMA_VERSION, "seed_sha256": expected_seed,
                       "is_demo": True, "as_of": fixture["as_of"], "counts": expected_counts}):
            raise TaskDataError("evaluation_database_fixture_mismatch")
        identity = {"host": config.host, "port": config.port,
                    "database": config.database, "username": config.username}
        return demo.DemoBusinessStore(config), {
            "kind": "isolated-local-mysql-synthetic-fixture", "configuration_identity_sha256": _hash(identity),
            "grants_sha256": grants, "fixture_file_sha256": FIXTURE_SHA256,
            "fixture_canonical_sha256": result["seed_sha256"], "as_of": fixture["as_of"],
            "counts": expected_counts, "database_access": "SELECT-only; no initialization",
            "literal_schema_grants": True}
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_mysql_preflight_failed") from None


def model_cache_proof(folder: Path | None = None) -> dict:
    folder = folder or Path.home() / ".cache/chroma/onnx_models" / MODEL_NAME / "onnx"
    try:
        if folder.resolve() != folder or not folder.is_dir():
            raise TaskDataError("evaluation_embedding_cache_unavailable")
        files = {}
        for name, expected in MODEL_FILES.items():
            path = folder / name
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or not 0 < path.stat().st_size <= 128 * 1024 * 1024:
                raise TaskDataError("evaluation_embedding_cache_unavailable")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise TaskDataError("evaluation_embedding_cache_changed")
            files[name] = {"sha256": expected, "bytes": path.stat().st_size}
        return {"model": MODEL_NAME, "files": files, "downloads": "forbidden",
                "provenance": "measured existing local cache; not independently certified"}
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_embedding_cache_unavailable") from None



def offline_data_proof() -> dict:
    """Read fixed files/cache metadata only; no DB, client or vector inference.

    An absent MySQL URL is neither read nor inferred here. Its separate live
    SELECT-only preflight remains mandatory before any model operation.
    """
    fixture, docs = _fixture(), _policies()
    try:
        if metadata.version("chromadb") != CHROMA_VERSION:
            raise TaskDataError("evaluation_chroma_version_unsupported")
        implementation = metadata.distribution("chromadb").locate_file(
            "chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py")
        source_hash = hashlib.sha256(Path(implementation).read_bytes()).hexdigest()
    except TaskDataError:
        raise
    except Exception:
        raise TaskDataError("evaluation_embedding_dependency_unavailable") from None
    # The exact pinned corpus consists of six short documents. For these inputs
    # the production _chunk_text returns the original content as one chunk.
    if any(len(doc["content"]) > 500 for doc in docs):
        raise TaskDataError("evaluation_corpus_contract_changed")
    return {"schema_version": "serveflow-task-offline-data-proof-v1",
        "mysql_preflight": "required-not-performed",
        "mysql_environment_variable": "SERVEFLOW_EVAL_MYSQL_URL",
        "mysql_database_requirement": "serveflow_demo_eval_*; SELECT-only",
        "fixture_file_sha256": FIXTURE_SHA256,
        "fixture_canonical_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "as_of": fixture["as_of"],
        "fixture_counts": {name: len(fixture[name]) for name in demo.TABLE_COLUMNS},
        "knowledge": {"kind": "expected-isolated-ephemeral-chroma",
            "chromadb_version": CHROMA_VERSION, "embedding": model_cache_proof(),
            "embedding_implementation_sha256": source_hash,
            "corpus_sha256": CORPUS_SHA256, "document_count": 6, "chunk_count": 6,
            "chunk_sha256": [hashlib.sha256(doc["content"].encode()).hexdigest() for doc in docs]},
        "embedding_inference": "not-performed"}


def _cached_only_embedding(base_type, initial_proof):
    class CachedOnlyMiniLM(base_type):
        def __call__(self, input):
            result = super().__call__(input)
            self._download_model_if_not_exists()
            return result

        def _download_model_if_not_exists(self):
            if model_cache_proof() != initial_proof:
                raise TaskDataError("evaluation_embedding_cache_changed")

        def _download(self, *args, **kwargs):
            raise TaskDataError("evaluation_embedding_download_forbidden")
    return CachedOnlyMiniLM()


def _build_knowledge():
    client = None
    name = "serveflow_task_eval_" + uuid.uuid4().hex
    try:
        if metadata.version("chromadb") != CHROMA_VERSION:
            raise TaskDataError("evaluation_chroma_version_unsupported")
        cache = model_cache_proof()
        import chromadb
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        from tooling.knowledge_base import KnowledgeBase
        embedding = _cached_only_embedding(ONNXMiniLM_L6_V2, cache)
        client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
        collection = client.create_collection(name=name, embedding_function=embedding,
            metadata={"description": "ServeFlow isolated synthetic evaluation corpus"})
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._write_lock = threading.RLock()
        kb._client, kb._collection, kb._use_server = client, collection, False
        docs = _policies()
        chunks = [chunk for doc in docs for chunk in kb._chunk_text(doc["content"], chunk_size=500)]
        inserted = kb.add_documents(docs)
        if inserted != len(chunks) or kb.doc_count != len(chunks):
            raise TaskDataError("evaluation_corpus_index_mismatch")
        import inspect
        source = Path(inspect.getfile(ONNXMiniLM_L6_V2))
        proof = {"kind": "isolated-ephemeral-chroma", "chromadb_version": CHROMA_VERSION,
                 "embedding": cache, "embedding_implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "corpus_sha256": CORPUS_SHA256, "document_count": 6, "chunk_count": len(chunks),
                 "chunk_sha256": [hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks],
                 "storage": "ephemeral-memory-only; only owned collection deleted on close"}
        return _KnowledgeRuntime(kb, client, name, proof)
    except Exception as error:
        if client is not None:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        if isinstance(error, TaskDataError):
            raise
        raise TaskDataError("evaluation_knowledge_preflight_failed") from None


class _KnowledgeRuntime:
    def __init__(self, kb, client, name, proof):
        self.kb, self.client, self.name, self.proof = kb, client, name, proof

    def close(self):
        self.client.delete_collection(self.name)


async def _owned(awaitable):
    worker = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done():
            try:
                worker.result()
            except BaseException:
                pass
        raise


class LiveData:
    def __init__(self, store, knowledge, proof):
        self._store, self._knowledge = store, knowledge
        self.proof = deepcopy(proof)
        self._fixture_snapshot = _fixture()
        self._policy_snapshot = _policies()
        self.fault = None
        self._closed, self._closing, self._pending = False, None, set()

    def assert_healthy(self):
        """Only inspect in-memory state; admission never performs extra queries."""
        if self.fault is not None:
            raise TaskDataError(self.fault) from None
        if self._closed:
            raise TaskDataError("evaluation_data_closed")

    def _fail(self, code):
        if self.fault is None:
            self.fault = code
        raise TaskDataError(self.fault) from None

    async def _run(self, operation):
        self.assert_healthy()
        worker = asyncio.create_task(operation())
        self._pending.add(worker)
        try:
            return await _owned(worker)
        except asyncio.CancelledError:
            raise
        except TaskDataError:
            raise
        except Exception:
            raise TaskDataError("evaluation_data_read_failed") from None
        finally:
            self._pending.discard(worker)

    async def order_reader(self, customer_id: str, order_id: str) -> dict:
        async def read():
            try:
                actual = await self._store.read_order(customer_id, order_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._fail("eval_mysql_unavailable")
            try:
                expected = _fixture_projection(self._fixture_snapshot, customer_id, order_id)
                projected = {key: value for key, value in actual.items() if key != "evaluation_data_kind"}
                if _hash(projected) != _hash(expected):
                    self._fail("eval_mysql_changed")
            except TaskDataError:
                raise
            except Exception:
                self._fail("eval_mysql_changed")
            return deepcopy(actual)
        return await self._run(read)

    async def knowledge_handler(self, params: dict, context: Any):
        async def search():
            try:
                actual = await self._knowledge.kb.search_handler(params, context)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._fail("eval_knowledge_unavailable")
            expected = {doc["title"]: doc["content"] for doc in self._policy_snapshot}
            try:
                if type(actual) is not list or len(actual) > 20:
                    self._fail("eval_knowledge_changed")
                for row in actual:
                    content = expected.get(row.get("title"))
                    if (content is None or row.get("content") != content
                            or row.get("version") != hashlib.sha256(content.encode()).hexdigest()):
                        self._fail("eval_knowledge_changed")
            except TaskDataError:
                raise
            except Exception:
                self._fail("eval_knowledge_changed")
            return deepcopy(actual)
        return await self._run(search)

    async def close(self):
        if self._closing is None:
            self._closed = True
            async def cleanup():
                if self._pending:
                    await asyncio.gather(*tuple(self._pending), return_exceptions=True)
                await asyncio.to_thread(self._knowledge.close)
            self._closing = asyncio.create_task(cleanup())
        try:
            await _owned(self._closing)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise TaskDataError("evaluation_data_cleanup_failed") from None


async def build_live_data(mysql_url: str) -> LiveData:
    """Explicit preflight only; caller supplies SERVEFLOW_EVAL_MYSQL_URL value.

    This function does not read that variable itself, and never falls back to the
    application/demo database, persistent Chroma or a different embedding model.
    """
    config = parse_mysql_url(mysql_url)
    store, mysql_proof = await _owned(asyncio.to_thread(_verify_mysql, config))
    worker = asyncio.create_task(asyncio.to_thread(_build_knowledge))
    try:
        knowledge = await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled() and worker.exception() is None:
            await _owned(asyncio.to_thread(worker.result().close))
        raise
    return LiveData(store, knowledge, {"schema_version": "serveflow-task-live-data-v1",
                                     "mysql": mysql_proof, "knowledge": knowledge.proof})


def _fixture_projection(fixture: dict, customer_id: str, order_id: str) -> dict:
    if not isinstance(customer_id, str) or customer_id not in demo.CUSTOMER_IDS or not isinstance(order_id, str) or not demo.ORDER_PATTERN.fullmatch(order_id):
        raise TaskDataError("evaluation_fixture_identity_invalid")
    result = {"source": demo.SOURCE, "is_demo": True, "as_of": fixture["as_of"],
              "status": "not_found", "order": None}
    order = next((row for row in fixture["orders"] if row["order_id"] == order_id and row["customer_id"] == customer_id), None)
    if order is None:
        return result
    packet = deepcopy(order)
    packet.update({table: [deepcopy(row) for row in fixture[table] if row["order_id"] == order_id]
                   for table in ("items", "payments", "shipments", "entitlements", "refunds")})
    return {**result, "status": "found", "order": packet}


async def fixture_order_reader(customer_id: str, order_id: str) -> dict:
    """Controlled fixture projection, never a real MySQL integration result."""
    return {**_fixture_projection(_fixture(), customer_id, order_id),
            "evaluation_data_kind": "controlled-fixture-projection"}
