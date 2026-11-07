"""Private-version integrity, explicit preparation and no implicit model calls."""
import asyncio
import json
import os
from pathlib import Path
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from core import intent_template_cache as c
from core import qwen_embedding as e
from core.intent_embedding import IntentEmbedding
from core.intent_recognizer import IntentRecognizer, IntentCategory, _TEMPLATES

BASE = 'https://controlled.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
KEY = 'controlled-only-not-real'


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*a, **k): raise AssertionError('ordinary tests forbid real services')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setenv('ECHOMIND_ENV', 'development')
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER', 'qwen')
    monkeypatch.setenv('QWEN_EMBEDDING_BASE_URL', BASE)
    monkeypatch.setenv('QWEN_EMBEDDING_API_KEY', KEY)
    monkeypatch.setenv('QWEN_INTENT_CACHE_DIRECTORY', str(tmp_path.resolve() / 'templates'))


async def vectors(texts): return [[1.] + [0.] * 1023 for _ in texts]


def version(cache, snapshot): return cache.path / c.digest(cache.plan(snapshot)[0])


def test_missing_cache_never_sends_or_creates_directories(monkeypatch):
    async def run():
        monkeypatch.setattr(e, '_delegate', lambda: pytest.fail('implicit preparation/query'))
        embedding = IntentEmbedding()
        groups, value, source = await embedding.vectors('question', {'a':['template']}, lambda t:[1.,0.])
        assert source == 'local-ngram' and len(value) == len(groups['a'][0]) == 2
        assert not embedding.template_cache.path.exists()
        assert embedding.last_error == 'embedding_templates_not_prepared'
        assert embedding.status()['template_cache']['state'] == 'unprepared'
    asyncio.run(run())


def test_preparation_persists_across_instances_and_reuses_exact_version_without_sending():
    async def run():
        snapshot = {'a':[str(i) for i in range(41)]}
        cache = c.TemplateCache(BASE)
        fetch = AsyncMock(side_effect=vectors)
        first = await cache.prepare(snapshot, fetch, max_requests=3)
        groups, token = c.TemplateCache(BASE).read(snapshot)
        assert len(groups['a']) == 41 and len(groups['a'][0]) == 1024
        assert first['new_requests'] == fetch.await_count == 3
        original = (version(cache,snapshot)/'templates.json').read_bytes()
        second = await c.TemplateCache(BASE).prepare(snapshot, fetch, max_requests=3)
        assert second['new_requests'] == 0 and fetch.await_count == 3
        assert c.TemplateCache(BASE).read(snapshot)[1] == token
        assert (version(cache,snapshot)/'templates.json').read_bytes() == original
    asyncio.run(run())


def test_template_and_endpoint_versions_never_mix():
    async def run():
        cache = c.TemplateCache(BASE)
        await cache.prepare({'a':['one']}, vectors, max_requests=1)
        for other, snapshot in [(cache,{'a':['different']}), (cache,{'b':['one']}),
            (c.TemplateCache('https://other.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'),{'a':['one']})]:
            with pytest.raises(c.TemplateCacheError, match='not_prepared'): other.read(snapshot)
    asyncio.run(run())


def test_cancelled_or_uncertain_batch_cannot_be_resent_after_restart():
    async def run():
        snapshot = {'a':['one']};cache = c.TemplateCache(BASE)
        async def cancel(texts): raise asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError): await cache.prepare(snapshot, cancel, max_requests=1)
        fetch = AsyncMock(side_effect=vectors)
        with pytest.raises(c.TemplateCacheError, match='preparation_incomplete'):
            await c.TemplateCache(BASE).prepare(snapshot, fetch, max_requests=1)
        assert fetch.await_count == 0 and cache.status(snapshot)['state'] == 'unprepared'
    asyncio.run(run())


def test_known_completed_batches_resume_without_duplicate_sends(monkeypatch):
    async def run():
        snapshot={'a':[str(i) for i in range(21)]};cache=c.TemplateCache(BASE)
        publish=c._publish
        def interrupted(directory,name,value):
            publish(directory,name,value)
            if name=='batch-000.result.json': raise RuntimeError('controlled process interruption')
        monkeypatch.setattr(c,'_publish',interrupted)
        fetch=AsyncMock(side_effect=vectors)
        with pytest.raises(RuntimeError):await cache.prepare(snapshot,fetch,max_requests=2)
        assert fetch.await_count==1
        monkeypatch.setattr(c,'_publish',publish)
        final=await c.TemplateCache(BASE).prepare(snapshot,fetch,max_requests=2)
        assert final['new_requests']==1 and fetch.await_count==2
    asyncio.run(run())


def test_concurrent_preparers_have_one_writer_and_no_duplicate_calls():
    async def run():
        entered,release=asyncio.Event(),asyncio.Event();calls=[];snapshot={'a':['one']}
        async def fetch(texts):calls.append(texts);entered.set();await release.wait();return await vectors(texts)
        first=asyncio.create_task(c.TemplateCache(BASE).prepare(snapshot,fetch,max_requests=1))
        await entered.wait()
        try:
            with pytest.raises(c.TemplateCacheError,match='preparation_busy'):
                await c.TemplateCache(BASE).prepare(snapshot,fetch,max_requests=1)
            assert len(calls)==1
        finally:release.set();await first
    asyncio.run(run())


@pytest.mark.parametrize('damage',['hash','model','dimension','template','short','nan','not_normalized','unknown_field'])
def test_invalid_cache_is_rejected_without_repair_or_network(damage,monkeypatch):
    async def run():
        cache=c.TemplateCache(BASE);snapshot={'a':['one']}
        await cache.prepare(snapshot,vectors,max_requests=1)
        path=version(cache,snapshot)/'templates.json';value=json.loads(path.read_bytes())
        if damage=='hash':value['sha256']='a'*64
        elif damage=='model':value['identity']['model']='different'
        elif damage=='dimension':value['identity']['dimensions']=512
        elif damage=='template':value['identity']['templates_sha256']='b'*64
        elif damage=='short':value['vectors'][0].pop()
        elif damage=='nan':value['vectors'][0][0]='nan'
        elif damage=='not_normalized':value['vectors'][0][0]=2.
        else:value['unknown']='field'
        if damage not in {'hash','unknown_field'}:value['sha256']=c.digest({k:v for k,v in value.items() if k!='sha256'})
        path.write_bytes(c.canonical(value));before=path.read_bytes()
        with pytest.raises(c.TemplateCacheError):cache.read(snapshot)
        monkeypatch.setattr(e,'_delegate',lambda:pytest.fail('invalid cache queried provider'))
        result=await IntentEmbedding().vectors('q',snapshot,lambda t:[1.,0.])
        assert result[2]=='local-ngram' and path.read_bytes()==before
    asyncio.run(run())


@pytest.mark.parametrize('damage',['symlink','hardlink','permissions','fifo','oversize'])
def test_unsafe_files_never_read_or_overwrite_unrelated_data(tmp_path,damage):
    async def run():
        cache=c.TemplateCache(BASE);snapshot={'a':['one']}
        await cache.prepare(snapshot,vectors,max_requests=1)
        path=version(cache,snapshot)/'templates.json';other=tmp_path/'unrelated'
        other.write_bytes(b'keep-private');other.chmod(0o600)
        if damage=='permissions':path.chmod(0o640)
        elif damage=='oversize':path.write_bytes(b'x'*(c.FILE_LIMIT+1))
        else:
            path.unlink()
            if damage=='symlink':path.symlink_to(other)
            elif damage=='hardlink':os.link(other,path)
            else:os.mkfifo(path,0o600)
        with pytest.raises(c.TemplateCacheError):cache.read(snapshot)
        assert other.read_bytes()==b'keep-private'
    asyncio.run(run())


def test_existing_unmarked_directory_is_not_adopted_or_modified(tmp_path):
    async def run():
        root=tmp_path/'existing';root.mkdir(mode=0o700);(root/'existing-data').write_bytes(b'keep')
        cache=c.TemplateCache(BASE,root);fetch=AsyncMock(side_effect=vectors)
        with pytest.raises(c.TemplateCacheError):await cache.prepare({'a':['one']},fetch,max_requests=1)
        assert fetch.await_count==0 and set(p.name for p in root.iterdir())=={'existing-data'}
    asyncio.run(run())


def test_preparation_deadline_keeps_claim_and_cleans_lock(monkeypatch):
    async def run():
        monkeypatch.setattr(c,'PREPARE_DEADLINE',.03)
        async def stuck(texts):await asyncio.Event().wait()
        cache=c.TemplateCache(BASE);snapshot={'a':['one']}
        with pytest.raises(c.TemplateCacheError,match='prepare_timeout'):await cache.prepare(snapshot,stuck,max_requests=1)
        with pytest.raises(c.TemplateCacheError,match='preparation_incomplete'):
            await cache.prepare(snapshot,vectors,max_requests=1)
    asyncio.run(run())


def test_snapshot_limit_and_explicit_request_cap_are_checked_before_send():
    async def run():
        cache=c.TemplateCache(BASE);fetch=AsyncMock(side_effect=vectors)
        with pytest.raises(c.TemplateCacheError,match='request_limit'):
            await cache.prepare({'a':['one']},fetch,max_requests=2)
        with pytest.raises(c.TemplateCacheError,match='template_limit'):
            await cache.prepare({'a':['one']*257},fetch,max_requests=13)
        assert not cache.path.exists() and fetch.await_count==0
    asyncio.run(run())


def test_classification_cache_invalidates_when_templates_become_ready(monkeypatch):
    async def run():
        recognizer=IntentRecognizer('controlled')
        recognizer._llm_recognize=AsyncMock(return_value={'intent':IntentCategory.REFUND,'confidence':.95})
        calls=[]
        def provider(request):
            calls.append(json.loads(request.content)['input'])
            return httpx.Response(200,json={'data':[{'index':0,'embedding':[1.]+[0.]*1023}]})
        monkeypatch.setattr(e,'_delegate',lambda:httpx.MockTransport(provider))
        try:
            first=await recognizer.recognize('退款进度')
            assert first.embedding_source=='local-ngram' and calls==[]
            await recognizer._embedding.template_cache.prepare(_TEMPLATES,vectors,max_requests=3)
            second=await recognizer.recognize('退款进度')
            assert second.embedding_source=='qwen/'+e.MODEL and len(calls)==1
            assert recognizer._llm_recognize.await_count==2
        finally:await recognizer.client.close()
    asyncio.run(run())


def test_cache_schema_rejects_bool_version_even_with_recomputed_hash():
    async def run():
        cache=c.TemplateCache(BASE);snapshot={'a':['one']}
        await cache.prepare(snapshot,vectors,max_requests=1)
        path=version(cache,snapshot)/'templates.json';value=json.loads(path.read_text())
        value['identity']['version']=True
        value['sha256']=c.digest({k:v for k,v in value.items() if k!='sha256'})
        path.write_bytes(c.canonical(value))
        with pytest.raises(c.TemplateCacheError):cache.read(snapshot)
    asyncio.run(run())


def test_cache_cli_status_is_readonly_and_prepare_requires_explicit_cap(monkeypatch,capsys):
    from core import intent_templates as commands
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:False)
    monkeypatch.setattr(e,'_delegate',lambda:pytest.fail('status sent a model request'))
    assert commands.main(['status'])==0
    assert json.loads(capsys.readouterr().out)['result']['state']=='unprepared'
    with pytest.raises(SystemExit):commands.main(['prepare'])
    assert not c.TemplateCache(BASE).path.exists()


def test_cache_cli_prepares_only_requested_batches_and_second_run_reuses(monkeypatch,capsys):
    from core import intent_templates as commands
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:False)
    calls=[]
    def provider(request):
        texts=json.loads(request.content)['input'];calls.append(texts)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.]+[0.]*1023} for i in range(len(texts))]})
    monkeypatch.setattr(e,'_delegate',lambda:httpx.MockTransport(provider))
    assert commands.main(['prepare','--max-requests','3'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['result']['new_requests']==3 and len(result['request_diagnostics'])==3
    assert commands.main(['prepare','--max-requests','3'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['result']['new_requests']==0 and len(calls)==3


def test_successful_batch_with_failed_publication_is_not_sent_again(monkeypatch):
    async def run():
        cache=c.TemplateCache(BASE);snapshot={'a':['one']};fetch=AsyncMock(side_effect=vectors)
        def fail(*a,**k):raise OSError('controlled private write failure')
        monkeypatch.setattr(c,'_publish',fail)
        with pytest.raises(c.TemplateCacheError):await cache.prepare(snapshot,fetch,max_requests=1)
        with pytest.raises(c.TemplateCacheError,match='preparation_incomplete'):
            await cache.prepare(snapshot,fetch,max_requests=1)
        assert fetch.await_count==1
    asyncio.run(run())


def test_excess_template_count_reports_unready_without_network(monkeypatch):
    async def run():
        monkeypatch.setattr(e,'_delegate',lambda:pytest.fail('oversize template plan sent'))
        client=IntentEmbedding();snapshot={'a':['one']*257}
        assert await client.cache_token(snapshot)=='embedding_template_limit'
        assert (await client.vectors('q',snapshot,lambda t:[1.,0.]))[2]=='local-ngram'
        assert not client.template_cache.path.exists()
    asyncio.run(run())
