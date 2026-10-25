import sys
import types
import threading
import hashlib
import json
from pathlib import Path

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules['chromadb'] = types.ModuleType('chromadb')

from tooling.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)
        for i, doc, meta in zip(kwargs['ids'], kwargs['documents'], kwargs['metadatas']):
            self.rows[i] = (doc, meta)

    def get(self, where, include):
        rows = [(i, d, m) for i, (d, m) in self.rows.items() if all(m.get(k) == v for k, v in where.items())]
        return {'ids': [r[0] for r in rows], 'metadatas': [r[2] for r in rows]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._write_lock = threading.RLock()
    kb._collection = FakeCollection()
    return kb


def test_repeated_document_import_is_idempotent():
    kb = make_kb()
    doc = {'title': '退款政策', 'content': '7 天内可以申请退款。'}
    assert kb.add_documents([doc, doc]) == 1
    assert kb.add_documents([doc]) == 1
    assert len(kb._collection.rows) == 1


def test_shorter_document_replaces_chunks_and_legacy_data():
    kb = make_kb()
    kb._collection.rows['legacy'] = ('旧内容', {'title': '政策'})
    kb.add_documents([{'title': '政策', 'content': ('旧规则' * 100 + '。') * 5}])
    assert len(kb._collection.rows) > 1
    assert 'legacy' not in kb._collection.rows
    kb.add_documents([{'title': '政策', 'content': '新的政策'}])
    assert len(kb._collection.rows) == 1
    assert next(iter(kb._collection.rows.values()))[0] == '新的政策'


def test_explicit_ids_allow_documents_with_same_title():
    kb = make_kb()
    kb.add_documents([{'document_id': 'a', 'title': '政策', 'content': '甲'},
                      {'document_id': 'b', 'title': '政策', 'content': '乙'}])
    assert len(kb._collection.rows) == 2
    kb.add_documents([{'document_id': 'a', 'title': '改名', 'content': '新甲'}])
    assert len(kb._collection.rows) == 2
    assert {d for d, m in kb._collection.rows.values()} == {'新甲', '乙'}


def test_long_unpunctuated_text_respects_chunk_size():
    kb = make_kb()
    chunks = kb._chunk_text('长' * 1201, chunk_size=500)
    assert [len(c) for c in chunks] == [500, 500, 201]
    assert ''.join(chunks) == '长' * 1201


def test_serveflow_demo_policies_keep_complete_conditions_and_explicit_scope(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kb = make_kb()
    kb._load_default_docs()
    rows = list(kb._collection.rows.values())
    assert len(rows) == 6
    assert {meta['title'] for _, meta in rows} == {
        '退款政策', '订单查询', '账户安全', '技术故障排查', '会员与积分', '配送说明',
    }
    for body, meta in rows:
        assert 0 < len(body) <= 500
        assert body.startswith('【ServeFlow演示规则｜')
        assert '不代表真实企业承诺' in body
        assert meta['total_chunks'] == 1
        assert meta['version'] == hashlib.sha256(body.encode()).hexdigest()


def test_serveflow_policy_update_reuses_document_ids_and_preserves_unrelated_docs():
    kb = make_kb()
    corpus = Path(__file__).resolve().parents[1] / 'knowledge' / 'serveflow-demo-v1.json'
    docs = json.loads(corpus.read_text(encoding='utf-8'))
    kb.add_documents([{'title': d['title'], 'content': '旧演示内容。'} for d in docs])
    old_ids = set(kb._collection.rows)
    kb.add_documents([{'title': '自定义资料', 'content': '用户维护的资料。'},
                      {'document_id': 'custom-refund', 'title': '退款政策', 'content': '单独维护的合同规则。'}])
    unrelated = {key: value for key, value in kb._collection.rows.items() if key not in old_ids}
    kb._load_default_docs()
    assert len(kb._collection.rows) == 8
    assert old_ids <= set(kb._collection.rows)
    assert {key: kb._collection.rows[key] for key in unrelated} == unrelated
    assert all(kb._collection.rows[key][0] != '旧演示内容。' for key in old_ids)
    after = dict(kb._collection.rows)
    kb._load_default_docs()
    assert kb._collection.rows == after


def test_current_demo_loader_uses_v2_and_preserves_original_v1_package():
    folder = Path(__file__).resolve().parents[1] / 'knowledge'
    assert hashlib.sha256((folder / 'serveflow-demo-v1.json').read_bytes()).hexdigest() == (
        '99bf1b31726c214cb0f5b075896a4342511fc483c21cf6194c58bb767b5f25c3'
    )
    expected = {doc['title']: doc['content'] for doc in json.loads((folder / 'serveflow-demo-v2.json').read_text())}
    kb = make_kb()
    kb._load_default_docs()
    assert {meta['title']: content for content, meta in kb._collection.rows.values()} == expected
