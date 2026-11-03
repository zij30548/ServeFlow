"""Native policy-tool wiring and deterministic projection, controlled providers.

These are not live Flash semantic or provider compatibility measurements.
"""
import asyncio
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BillingAgent, GeneralAgent,
    Request, ResponseComposer, TechnicalAgent,
)
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.policy_grounding import extract_policy_evidence


POLICY_QUESTION = '请说明配送规则、送达时效和免邮门槛。'
BODY = '演示资料：设备与配件可能分别出库；数字订阅没有快递单号。没有全国统一到货天数或免邮门槛。'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs): raise AssertionError('policy unit tests must not contact services')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket.socket,'connect_ex',denied)
    monkeypatch.setattr(socket,'create_connection',denied)


def row():
    return {'document_id':'d'*64,'title':'受控配送规则','content':BODY,'version':'a'*64,'chunk':0,'score':.5}


def text(value): return SimpleNamespace(content=[{'type':'text','text':value}],stop_reason='end_turn')


def tool(name='search_knowledge_base',args=None):
    return SimpleNamespace(content=[{'type':'tool_use','id':'call_policy_1','name':name,
                                     'input':args or {'query':'配送规则','top_k':3}}],stop_reason='tool_use')


class Client:
    def __init__(self,*responses): self.responses=list(responses);self.calls=[];self.messages=self
    async def create(self,**kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, 'unexpected extra model call'
        result=self.responses.pop(0)
        if isinstance(result,Exception):raise result
        return result


class Retrieval:
    def __init__(self,*,rows=None,success=True):self.rows=rows if rows is not None else [row()];self.success=success;self.calls=[]
    async def search_with_rewrite(self,name,query,top_k=5):
        self.calls.append((name,query,top_k))
        return SimpleNamespace(success=self.success,data=deepcopy(self.rows),reranked=True,error=None if self.success else 'controlled failure')


def req(message=POLICY_QUESTION):
    return Request(message=message,user_id='controlled',conv_id='policy-test',intent=IntentCategory.LOGISTICS,
                   urgency=UrgencyLevel.LOW,intent_confidence=.99)


@pytest.mark.parametrize('role',[GeneralAgent,BillingAgent,TechnicalAgent])
def test_policy_requires_real_native_search_and_projects_only_actual_source(role):
    async def run():
        client=Client(tool());manager=Retrieval();agent=role(client,'controlled-model')
        agent.set_shared_tools(build_shared_rag_tools(manager))
        result=await agent.handle(req())
        assert result.status=='success' and result.success
        assert result.policy_evidence is not None and result.policy_evidence.status=='sources'
        assert len(client.calls)==1 and len(manager.calls)==1
        assert client.calls[0]['tool_choice']=={'type':'tool','name':'search_knowledge_base'}
        assert result.tools_used==['search_knowledge_base']
        assert result.tool_traces[0]['success'] and result.tool_traces[0]['tool_use_id']=='call_policy_1'
        assert BODY in result.content and '受控配送规则' in result.content and 'aaaaaaaaaaaa' in result.content
        assert not result.escalate
    asyncio.run(run())


def test_missing_registered_tool_does_not_generate_or_invent_policy():
    async def run():
        client=Client();result=await GeneralAgent(client,'controlled-model').handle(req())
        assert result.status=='partial' and client.calls==[]
        assert result.policy_evidence.status=='unavailable'
    asyncio.run(run())


@pytest.mark.parametrize('response',[
    text('无依据自由正文：主要城市2–5天，Prime会员免邮。'),
    tool('read_demo_order',{'order_id':'DEMO-1001'}),
])
def test_ignored_tool_choice_or_wrong_tool_cannot_become_policy_answer(response):
    async def run():
        client=Client(response);manager=Retrieval();agent=GeneralAgent(client,'controlled-model')
        agent.set_shared_tools(build_shared_rag_tools(manager))
        result=await agent.handle(req())
        assert result.status=='partial' and result.policy_evidence.status=='unverified'
        assert manager.calls==[] and result.tools_used==[]
        assert '2–5' not in result.content and 'Prime' not in result.content
    asyncio.run(run())


@pytest.mark.parametrize('rows,success,status',[
    ([],True,'no_materials'),([],False,'unavailable'),([{'title':'missing content'}],True,'invalid'),
])
def test_no_usable_material_has_fixed_partial_result_and_no_summary_retry(rows,success,status):
    async def run():
        client=Client(tool());manager=Retrieval(rows=rows,success=success);agent=GeneralAgent(client,'controlled-model')
        agent.set_shared_tools(build_shared_rag_tools(manager));result=await agent.handle(req())
        assert result.status=='partial' and result.policy_evidence.status==status
        assert len(client.calls)==1 and len(manager.calls)==1
    asyncio.run(run())


def test_composer_projects_evidence_and_never_rewrites_actor_claims():
    async def run():
        evidence=extract_policy_evidence({'success':True,'results':[row()]})
        responses=[AgentResponse(AgentType.GENERAL,'unverified made-up days',True,policy_evidence=evidence),
                   AgentResponse(AgentType.BILLING,'unverified made-up membership',True,policy_evidence=evidence)]
        client=Client();result=await ResponseComposer(client,'controlled-model').compose_result(req(),responses)
        assert client.calls==[] and result.status=='success'
        assert result.content.count(BODY)==1
        assert 'made-up' not in result.content
    asyncio.run(run())


@pytest.mark.parametrize('message', [
    '核对演示订单DEMO-1001的状态，并说明配送规则。',
    '请说明配送规则，并查一下订单状态。',
    '我的订单可以退款吗？你们的退款条件是什么？',
    'Please explain shipping rules and track the order.',
    '配送规则是什么？帮我取消这笔订单。',
    '配送规则是什么？请修改我的收货地址。',
    '退款条件是什么？我要申请退款。',
])
def test_mixed_record_and_policy_request_remains_explicitly_partial(message):
    async def run():
        client=Client(tool());manager=Retrieval();agent=GeneralAgent(client,'controlled-model')
        agent.set_shared_tools(build_shared_rag_tools(manager))
        question=req(message)
        result=await agent.handle(question)
        assert result.status=='partial' and '订单或交易部分尚未核实' in result.content
        assert result.tools_used==['search_knowledge_base']
        composer=Client();merged=await ResponseComposer(composer,'controlled-model').compose_result(question,[result])
        assert merged.status=='partial' and composer.calls==[]
    asyncio.run(run())


def test_untyped_policy_actor_prose_is_not_accepted_as_retrieved_evidence():
    async def run():
        client=Client()
        result=await ResponseComposer(client,'controlled-model').compose_result(req(),[
            AgentResponse(AgentType.GENERAL,'未经检索的免邮承诺',True)])
        assert result.status=='partial' and client.calls==[]
        assert '免邮承诺' not in result.content and '尚未取得' in result.content
    asyncio.run(run())


def test_quoted_handoff_words_in_source_are_not_executed_as_escalation():
    async def run():
        source=row();source['content']='资料说明：如有争议，可以联系人工客服。'
        client=Client(tool());agent=GeneralAgent(client,'controlled-model')
        agent.set_shared_tools(build_shared_rag_tools(Retrieval(rows=[source])))
        result=await agent.handle(req())
        assert result.success and not result.escalate
        assert source['content'] in result.content
    asyncio.run(run())


def test_policy_fallback_keeps_native_retrieval_gate():
    async def run():
        client=Client(RuntimeError('controlled provider error'),tool());manager=Retrieval()
        orc=AgentOrchestrator.__new__(AgentOrchestrator)
        orc._pool={AgentType.GENERAL:[GeneralAgent(client,'controlled-model')],AgentType.BILLING:[BillingAgent(client,'controlled-model')]}
        for agents in orc._pool.values():
            for agent in agents:agent.set_shared_tools(build_shared_rag_tools(manager))
        result=await orc._execute(req(),AgentType.BILLING)
        assert result.success and result.policy_evidence.status=='sources'
        assert len(client.calls)==2 and len(manager.calls)==1
        assert all(c['tool_choice']=={'type':'tool','name':'search_knowledge_base'} for c in client.calls)
        assert any(t.get('event')=='agent_fallback' for t in result.tool_traces)
    asyncio.run(run())
