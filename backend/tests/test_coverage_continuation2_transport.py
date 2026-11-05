"""Typed provenance uses real SDK/MockTransport, never private live evidence."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from anthropic import APIConnectionError
import httpx
import pytest
import pytest_asyncio

from core.model_deadline import EvaluationDeadlinePolicy, ModelDeadlineExceeded, await_model, eval_deadlines
from agents.tools import AgentToolSpec
from evaluation import coverage_continuation2_transport as module
from evaluation import coverage_runtime as shared, coverage_conversation_runtime as memory
from evaluation.intent_transport import TrialTransportError
from evaluation.task_evidence import CaptureTools, EvidenceError
from evaluation.task_transport import Frame, context
from test_coverage_continuation_transport import (
    fixture as original_fixture, offline, Budget, response, sdk, sdk_send, CASES, ENDPOINT,
)


@pytest_asyncio.fixture
async def fixture(original_fixture):
    root, protocol, _, budget = original_fixture
    old_manifest=(root/'continuation-v1.json').read_bytes()
    old_stop=b'{"prior-v1-stop":"keep bytes"}\n'
    (root/'continuation-v1-stop.json').write_bytes(old_stop)
    module.base._write(root,module.MANIFEST,{'schema_version':module.SCHEMA,'controlled':True})
    digest=hashlib.sha256((root/module.MANIFEST).read_bytes()).hexdigest()
    budget.results=lambda:{'operations':{r['operation_id']:deepcopy(r) for r in budget.finished}}
    return root,protocol,digest,budget,old_manifest,old_stop


def transport(fixture,factory):
    root,_,digest,budget,*_=fixture
    return module.CoverageContinuation2Transport(ENDPOINT,root,budget,budget.lease,lambda:None,
        continuation_sha256=digest,delegate_factory=lambda frame:httpx.MockTransport(factory(frame)))


def timeout_factory(frame):
    async def send(request):raise httpx.ReadTimeout('controlled')
    return send


async def rag_send(client,*,active_timeout=10):
    return await await_model(client.messages.create(model=module.base.MODEL,max_tokens=512,temperature=.3,
        messages=[{'role':'user','content':'controlled rewrite'}],extra_body={'thinking':{'type':'disabled'}}),timeout_s=active_timeout)


def spec(handler,*,allowed=('billing',)):
    return AgentToolSpec(name='search_knowledge_base',description='unchanged controlled tool',
        input_schema={'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False},
        handler=handler,allowed_agents=allowed,retry_safe=True)


def request():
    return SimpleNamespace(request_id='controlled-request-1',demo_customer_id='demo-customer-a')


async def wrapped_timeout(fixture,*,factory=timeout_factory,before=None):
    _,_,_,budget,*_=fixture
    t=transport(fixture,factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    async def handler(req,args):
        with context(phase='rag_rewrite'):
            return await rag_send(client)
    error=None
    with module.observe_case_errors(t) as observer:
        capture=shared.CaptureTools('CT-D01',1,before_call=before)
        native=capture._handler('billing',('search_knowledge_base',),spec(handler))
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with context(Frame(CASES[0],'turn-1','agent','billing')):
                try:
                    await native(request(),{'query':'controlled'})
                except EvidenceError:
                    # Actual production runtime rethrows a fresh EvidenceError
                    # at capture.assert_healthy, losing the original traceback.
                    try:capture.assert_healthy()
                    except EvidenceError as ex:error=ex
    await t.finish_case()
    return t,client,observer,capture,error


@pytest.mark.asyncio
async def test_v2_durable_same_case_timeout_survives_secondary_evidence_wrapper(fixture):
    root,_,_,budget,old_manifest,old_stop=fixture
    t,client,observer,capture,error=await wrapped_timeout(fixture)
    try:
        assert type(error) is EvidenceError and error.reason=='tool_handler_failed'
        assert capture.fault=='tool_handler_failed'
        assert len(observer.events)==1 and observer.events[0]['error_type']=='APIConnectionError'
        assert observer.events[0]['typed_sealed_origin'] is True
        assert observer.events[0]['handler_phase']=='agent' and observer.events[0]['proofs'][0]['phase']=='rag_rewrite'
        assert module.is_recoverable_case_error(error,t,observer) is True
        assert budget.attempts==21 and not budget.pending
        assert (root/'continuation-v1.json').read_bytes()==old_manifest
        assert (root/'continuation-v1-stop.json').read_bytes()==old_stop
        assert shared.CaptureTools is CaptureTools and memory.CaptureTools is CaptureTools
        record=json.loads((root/'physical-021.json').read_text())
        assert record['continuation_sha256']==fixture[2] and record['network_limit_s']==10
        serialized=json.dumps(observer.summary())
        assert 'controlled rewrite' not in serialized and 'controlled-not-a-real-key' not in serialized
    finally:await client.close()


@pytest.mark.asyncio
async def test_plain_forged_handler_error_has_no_permission_to_recover(fixture):
    _,protocol,_,budget,*_=fixture
    t=transport(fixture,timeout_factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    try:
        with module.observe_case_errors(t) as observer:
            with eval_deadlines(EvaluationDeadlinePolicy()):
                with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
        await t.finish_case()
        assert not module.is_recoverable_case_error(EvidenceError('tool_handler_failed'),t,observer)
        assert not module.is_recoverable_case_error(ModelDeadlineExceeded('active'),t,observer)
        assert module.is_recoverable_case_error(None,t,observer)
    finally:await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['permission','source','wrong-url','wrong-case','cancel'])
async def test_real_source_permission_or_unbound_handler_faults_stay_fatal(fixture,fault):
    _,protocol,_,budget,*_=fixture
    t=transport(fixture,timeout_factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    error=None
    async def handler(req,args):
        if fault=='source':raise ValueError('private-source-details-never-log')
        if fault=='wrong-url':raise APIConnectionError(request=httpx.Request('POST','https://untrusted.invalid/'))
        if fault=='wrong-case':raise module.CaseExecutionFailure(CASES[1],'timeout')
        if fault=='cancel':raise asyncio.CancelledError
        return {'success':True}
    try:
        with module.observe_case_errors(t) as observer:
            capture=shared.CaptureTools('CT-D01',1)
            # Establish a separately sealed transport failure first. It must
            # not forgive a coincident, unrelated native-handler failure.
            with eval_deadlines(EvaluationDeadlinePolicy()):
                with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
            native=capture._handler('billing',('search_knowledge_base',),spec(handler,allowed=('general',) if fault=='permission' else ('billing',)))
            with context(Frame(CASES[0],'turn-1','agent','billing')):
                try:await native(request(),{'query':'controlled'})
                except BaseException as ex:error=ex
        await t.finish_case()
        assert not module.is_recoverable_case_error(error,t,observer)
        if fault=='permission':assert capture.fault=='tool_permission_denied' and not observer.events
        else:assert observer.events and not observer.events[0]['typed_sealed_origin']
        assert 'private-source-details' not in json.dumps(observer.summary())
    finally:await client.close()


@pytest.mark.asyncio
async def test_before_call_stale_case_guard_has_sealed_typed_origin(fixture):
    _,protocol,_,budget,*_=fixture
    t=transport(fixture,timeout_factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    entered=[]
    async def handler(req,args):entered.append(1);return {'success':True}
    error=None
    try:
        with module.observe_case_errors(t) as observer:
            capture=shared.CaptureTools('CT-D01',1,before_call=lambda meta:t.assert_case_healthy())
            with eval_deadlines(EvaluationDeadlinePolicy()):
                with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
            native=capture._handler('billing',('search_knowledge_base',),spec(handler))
            with context(Frame(CASES[0],'turn-1','agent','billing')):
                try:await native(request(),{'query':'controlled'})
                except EvidenceError as ex:error=ex
        await t.finish_case()
        assert not entered and module.is_recoverable_case_error(error,t,observer)
        assert observer.events[0]['site']=='before_call' and observer.events[0]['error_type']=='CaseExecutionFailure'
    finally:await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('changed',['pending','physical','capture','observer','unclosed'])
async def test_changed_or_unclosed_evidence_cannot_be_recovered(fixture,changed):
    root,_,_,budget,*_=fixture
    t,client,observer,capture,error=await wrapped_timeout(fixture)
    try:
        assert module.is_recoverable_case_error(error,t,observer)
        if changed=='pending':budget.pending['unsealed']={'attempt':22}
        if changed=='physical':(root/'physical-021.json').write_text('{}')
        if changed=='capture':capture.fault='tool_trace_mismatch'
        if changed=='observer':observer.invalid=True
        if changed=='unclosed':observer.closed=False
        assert not module.is_recoverable_case_error(error,t,observer)
    finally:await client.close()


@pytest.mark.asyncio
async def test_observer_keeps_normal_handler_inputs_results_schema_and_scope(fixture):
    _,_,_,budget,*_=fixture
    t=transport(fixture,lambda frame:lambda request:response())
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    args={'query':'normal'};result={'success':True,'value':'unchanged'};calls=[]
    def handler(req,value):calls.append((req,value));return result
    with module.observe_case_errors(t) as observer:
        cap=shared.CaptureTools('CT-D01',1)
        original=spec(handler)
        native=cap._handler('billing',('search_knowledge_base',),original)
        with context(Frame(CASES[0],'turn-1','agent','billing')):
            assert await native(request(),args) is result
        assert original.handler is handler and original.input_schema['additionalProperties'] is False
        assert cap.calls[0]['input']==args and cap.calls[0]['result']==result and cap.calls[0]['success']
    await t.finish_case()
    assert not observer.events and not module.is_recoverable_case_error(None,t,observer)
    assert len(calls)==1


def test_factory_scope_restores_on_exception_and_rejects_overlap(fixture):
    _,_,_,budget,*_=fixture
    t=transport(fixture,lambda frame:lambda request:response())
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    with pytest.raises(ValueError):
        with module.observe_case_errors(t) as observer:
            assert shared.CaptureTools is not CaptureTools
            with pytest.raises(TrialTransportError):
                with module.observe_case_errors(t):pass
            raise ValueError('controlled')
    assert observer.closed and shared.CaptureTools is CaptureTools and memory.CaptureTools is CaptureTools


@pytest.mark.asyncio
async def test_direct_owned_active_deadline_requires_its_sealed_stage_proof(fixture):
    _,protocol,_,budget,*_=fixture
    def factory(frame):
        async def send(request):await asyncio.sleep(30);return response()
        return send
    t=transport(fixture,factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    payload=deepcopy(protocol['initial_intent_requests'][0]['payload']);thinking=payload.pop('thinking')
    error=None
    try:
        with module.observe_case_errors(t) as observer:
            with eval_deadlines(EvaluationDeadlinePolicy()):
                with context(Frame(CASES[0],'turn-1','intent',None)):
                    try:await await_model(client.messages.create(**payload,extra_body={'thinking':thinking}),timeout_s=.03)
                    except BaseException as ex:error=ex
            await t.finish_case()
        assert isinstance(error,ModelDeadlineExceeded) and error.reason=='active'
        assert t.sealed_case_error_proofs()[0]['stage_deadline_reason']=='active'
        assert module.is_recoverable_case_error(error,t,observer)
        assert not module.is_recoverable_case_error(ModelDeadlineExceeded('overhead'),t,observer)
    finally:await client.close()


def test_invalid_transport_cannot_leak_the_observer_factory_lock(fixture):
    with pytest.raises(TrialTransportError):
        with module.observe_case_errors(SimpleNamespace()):pass
    _,_,_,budget,*_=fixture
    t=transport(fixture,lambda frame:lambda request:response())
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    with module.observe_case_errors(t) as observer:
        assert shared.CaptureTools is not CaptureTools
    assert observer.closed and shared.CaptureTools is CaptureTools


@pytest.mark.asyncio
async def test_v2_auth_stop_is_separate_and_preserves_v1_records(fixture):
    root,protocol,_,budget,old_manifest,old_stop=fixture
    def factory(frame):
        async def send(request):return httpx.Response(401,json={'error':{'message':'private'}})
        return send
    t=transport(fixture,factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    try:
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
        await t.finish_case()
        assert t.fault=='auth' and (root/module.STOP_FILE).exists()
        assert (root/'continuation-v1.json').read_bytes()==old_manifest
        assert (root/'continuation-v1-stop.json').read_bytes()==old_stop
    finally:await client.close()
