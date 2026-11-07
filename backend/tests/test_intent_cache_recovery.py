"""Explicit operator publication never retries, resets, or erases a failed send."""
import asyncio
import json
import socket
from unittest.mock import AsyncMock

import pytest

from core import intent_template_cache as c, qwen_embedding as q
from core.intent_recognizer import _TEMPLATES

BASE='https://controlled.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a,**k):raise AssertionError('no real network')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket.socket,'connect_ex',denied)
    monkeypatch.setattr(socket,'create_connection',denied)
    monkeypatch.setattr(q,'_delegate',lambda:pytest.fail('import sent a model request'))


async def fixture(tmp_path,snapshot=None):
    snapshot=snapshot or {'a':['template']}
    target=c.TemplateCache(BASE,tmp_path.resolve()/'target')
    source=c.TemplateCache(BASE,tmp_path.resolve()/'source')
    async def failed(texts):raise q.EmbeddingError('embedding_network_error')
    count=(sum(len(v) for v in snapshot.values())+19)//20
    with pytest.raises(q.EmbeddingError):await target.prepare(snapshot,failed,max_requests=count)
    fetch=AsyncMock(side_effect=lambda texts:[[1.]+[0.]*1023 for _ in texts])
    await source.prepare(snapshot,fetch,max_requests=count)
    claim=target.path/c.digest(target.plan(snapshot)[0])/'batch-000.claim.json'
    args={'source_token':source.read(snapshot)[1],'claim_digest':c.digest(json.loads(claim.read_text())),
          'authorization_sha256':'a'*64}
    return target,source,snapshot,args


def test_import_preserves_original_files_and_is_idempotent_without_sends(tmp_path):
    async def run():
        target,source,snapshot,args=await fixture(tmp_path)
        before={p:p.read_bytes() for p in target.path.rglob('*') if p.is_file()}
        never=AsyncMock()
        with pytest.raises(c.TemplateCacheError,match='preparation_incomplete'):
            await target.prepare(snapshot,never,max_requests=1)
        first=target.import_prepared(snapshot,source,**args)
        assert first['imported'] and first['model_requests']==0
        assert target.read(snapshot)==source.read(snapshot)
        assert all(p.read_bytes()==raw for p,raw in before.items())
        receipt=target.path/('recovery-'+'a'*64+'.json')
        assert json.loads(receipt.read_text())['incomplete_claim_digest']==args['claim_digest']
        assert receipt.stat().st_mode&0o777==0o600
        files={p:p.read_bytes() for p in target.path.rglob('*') if p.is_file()}
        assert not target.import_prepared(snapshot,source,**args)['imported']
        assert files=={p:p.read_bytes() for p in target.path.rglob('*') if p.is_file()}
        assert (await target.prepare(snapshot,never,max_requests=1))['new_requests']==0
        never.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('field,value',[('source_token','b'*64),('claim_digest','c'*64),
    ('authorization_sha256',''),('authorization_sha256','../unsafe'),('source_token',True)])
def test_wrong_identity_token_claim_or_authorization_does_not_modify_target(tmp_path,field,value):
    async def run():
        target,source,snapshot,args=await fixture(tmp_path)
        before={p:p.read_bytes() for p in target.path.rglob('*') if p.is_file()}
        args[field]=value
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,source,**args)
        assert before=={p:p.read_bytes() for p in target.path.rglob('*') if p.is_file()}
    asyncio.run(run())


def test_source_corruption_or_cross_endpoint_never_published(tmp_path):
    async def run():
        target,source,snapshot,args=await fixture(tmp_path)
        other=c.TemplateCache('https://other.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',source.path)
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,other,**args)
        path=source.path/c.digest(source.plan(snapshot)[0])/'templates.json'
        path.write_text('{}')
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,source,**args)
        assert target.status(snapshot)['state']=='unprepared'
        assert not list(target.path.glob('recovery-*'))
    asyncio.run(run())


def test_existing_completed_batch_is_never_replaced(tmp_path):
    async def run():
        target,source,snapshot,args=await fixture(tmp_path)
        version=target.path/c.digest(target.plan(snapshot)[0])
        (version/'batch-000.result.json').write_text('{}')
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,source,**args)
        assert (version/'batch-000.result.json').read_text()=='{}'
        assert not list(target.path.glob('recovery-*'))
    asyncio.run(run())


def test_explicit_import_can_preserve_an_exact_legacy_float_payload_claim(tmp_path):
    async def run():
        import hashlib
        target,source,snapshot,args=await fixture(tmp_path)
        claim=target.path/c.digest(target.plan(snapshot)[0])/'batch-000.claim.json'
        data=json.loads(claim.read_text())
        data['payload_sha256']=hashlib.sha256(q._payload(['template'],encoding_format='float')).hexdigest()
        claim.write_bytes(c.canonical(data)+b'\n')
        original=claim.read_bytes();args['claim_digest']=c.digest(data)
        assert target.import_prepared(snapshot,source,**args)['imported']
        assert claim.read_bytes()==original
    asyncio.run(run())


def test_receipt_collision_or_busy_preparer_stops_without_publishing(tmp_path):
    async def run():
        import fcntl,os
        target,source,snapshot,args=await fixture(tmp_path)
        version=target.path/c.digest(target.plan(snapshot)[0])
        fd=os.open(version/'prepare.lock',os.O_RDWR)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with pytest.raises(c.TemplateCacheError,match='preparation_busy'):
                target.import_prepared(snapshot,source,**args)
        finally:os.close(fd)
        receipt=target.path/('recovery-'+'a'*64+'.json')
        receipt.write_text('{}');receipt.chmod(0o600)
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,source,**args)
        assert receipt.read_text()=='{}' and target.status(snapshot)['state']=='unprepared'
    asyncio.run(run())


def test_import_publication_failure_keeps_receipt_and_claim_and_recovery_sends_nothing(tmp_path,monkeypatch):
    async def run():
        target,source,snapshot,args=await fixture(tmp_path)
        publish=c._publish
        def failure(*a,**k):raise OSError('controlled disk error')
        monkeypatch.setattr(c,'_publish',failure)
        with pytest.raises(c.TemplateCacheError):target.import_prepared(snapshot,source,**args)
        assert list(target.path.glob('recovery-*'))
        monkeypatch.setattr(c,'_publish',publish)
        assert target.import_prepared(snapshot,source,**args)['imported']
    asyncio.run(run())


def test_cli_requires_all_explicit_recovery_inputs_and_does_not_load_keys_for_publication(tmp_path,monkeypatch,capsys):
    async def run():return await fixture(tmp_path,_TEMPLATES)
    target,source,snapshot,args=asyncio.run(run())
    from core import intent_templates as cli
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:False)
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER','qwen')
    monkeypatch.setenv('QWEN_EMBEDDING_BASE_URL',BASE)
    monkeypatch.setenv('QWEN_INTENT_CACHE_DIRECTORY',str(target.path))
    monkeypatch.delenv('QWEN_EMBEDDING_API_KEY',raising=False)
    with pytest.raises(SystemExit):cli.main(['import-prepared'])
    with pytest.raises(SystemExit):cli.main(['status','--source-cache',str(source.path)])
    assert cli.main(['import-prepared','--source-cache',str(source.path),'--source-token',args['source_token'],
        '--claim-digest',args['claim_digest'],'--authorization-sha256',args['authorization_sha256']])==0
    result=json.loads(capsys.readouterr().out)
    assert result['result']['model_requests']==0 and target.status(snapshot)['state']=='ready'
