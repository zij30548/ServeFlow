"""Dispatch a whole chat turn to the order workflow, agents, or both."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import re
import time

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, OrchestratorResult, Request,
    _project_business_result, update_conversation_state,
)
from core.diagnostic_evidence import select_diagnostic_request
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.order_evidence import requests_business_mutation
from core.policy_grounding import requires_policy_evidence
from core.safety_guidance import select_safety_guidance
from workflows.order_lookup import TOOL_NAME, WORKFLOW_ID, run_order_lookup


_POLITE = r"(?:(?:麻烦(?:你)?|请问|请|帮我|给我)\s*){0,3}"
_READ = r"(?:查询|查一下|查一查|查查|查下|查看|核对一下|核对|核实|看看|查)"
_ID = r"DEMO-[0-9]{4}"
_OWNER = r"(?:我的|我这笔|这笔|这单|这个|该|刚才那笔)"
_TARGET = rf"(?:(?:{_OWNER}?\s*订单(?:号)?\s*[:：]?\s*)?{_ID}|{_OWNER}?\s*订单|这笔|这单|刚才那笔)"
_FIELD = r"(?:(?:订单)?(?:付款|支付|退款|物流|配送|发货)(?:状态|进度|详情|记录)|(?:订单|实付|支付|付款|退款)金额)"
_GENERIC_FIELD = r"(?:订单)?(?:状态|进度|详情|记录)"
_STATE = (
    r"(?:退款(?:是否到账|(?:还没|尚未)到账|到账了?(?:吗|没有))|"
    r"(?:还|又)?(?:好像|似乎|是否|是不是|有没有)?(?:被)?(?:重复扣款|扣了两次(?:款|钱)?)|"
    r"(?:是否|有没有)(?:发货|付款|支付成功)|"
    r"(?:已经|已)?(?:发货|付款|支付成功)(?:了?吗|了没有))"
)
_ORDER_QUERY = re.compile(
    rf"{_POLITE}(?:{_READ}\s*{_TARGET}|"
    rf"(?:{_READ}\s*)?(?:(?:{_TARGET}|我的)\s*(?:的\s*)?)?(?:{_FIELD}|{_STATE})|"
    rf"(?:{_READ}\s*)?{_TARGET}\s*(?:的\s*)?{_GENERIC_FIELD})\s*(?:呢|吗|了|是什么|怎么样)?",
    re.I,
)
_CHECK_SUFFIX = re.compile(rf"[,，]\s*{_POLITE}(?:查一下|查一查|核对一下|看看)\s*$")
_TECHNICAL = re.compile(
    rf"{_POLITE}(?:(?:排查|诊断)(?:一下)?\s*)?"
    r"(?:登录\s*(?:报错?|失败|过期|失效)\s*(?:(?:HTTP\s*)?(?:错误码\s*[:：]?\s*)?[45]\d{2})?"
    r"|无法登录|登不上|(?:页面|应用|系统)\s*(?:报错?|崩溃|白屏)\s*(?:(?:HTTP\s*)?[45]\d{2})?)"
    r"(?:了|怎么办|怎么解决)?",
    re.I,
)


def select_workflow(req: Request) -> tuple[str | None, str | None]:
    """Return the workflow and optional independent technical clause.

    Match complete supported clauses to avoid dropping an additional request
    just because a sentence also contains an order id or payment keyword.
    """
    if (len(req.message) > 2000 or req.urgency is UrgencyLevel.CRITICAL
            or req.intent in {IntentCategory.HUMAN_HANDOFF, IntentCategory.ESCALATION}
            or requests_business_mutation(req.message)
            or requires_policy_evidence(req.message, req.history)
            or select_safety_guidance(req.message).required):
        return None, None
    message = re.sub(r"[^\S\n]+", " ", req.message.strip().rstrip("。！？.!?"))
    message = _CHECK_SUFFIX.sub("", message)
    clauses = [part.strip().rstrip("！？!?") for part in
               re.split(r"[，,；;。\n]|同时|而且|另外", message) if part.strip()]
    if len(clauses) == 1 and _ORDER_QUERY.fullmatch(clauses[0]):
        return WORKFLOW_ID, None
    if len(clauses) == 2:
        for order, technical in (clauses, clauses[::-1]):
            if (_ORDER_QUERY.fullmatch(order) and _TECHNICAL.fullmatch(technical)
                    and select_diagnostic_request(technical).required):
                return WORKFLOW_ID, technical
    return None, None


async def dispatch_chat(orchestrator: AgentOrchestrator, req: Request) -> OrchestratorResult:
    workflow_id, technical_message = select_workflow(req)
    if workflow_id is None:
        return await orchestrator.run(req)

    started = time.monotonic()
    update_conversation_state(req)
    technical: AgentResponse | None = None
    if technical_message is None:
        order = await run_order_lookup(req)
    else:
        diagnostic = select_diagnostic_request(technical_message)
        # This branch gets only its own question and never receives a reader,
        # trusted demo identity, or the order workflow's result.
        technical_req = replace(
            req, message=technical_message, context="", history=None, order_history=None,
            entities={"error_code": [diagnostic.error_code]} if diagnostic.error_code else {},
            intent=IntentCategory.TECHNICAL, intent_group="technical", conversation_state=None,
            demo_customer_id=None, demo_order_reader=None,
        )
        async with asyncio.TaskGroup() as branches:
            order_task = branches.create_task(run_order_lookup(req))
            technical_task = branches.create_task(orchestrator.run_technical_branch(technical_req))
        order, technical = order_task.result(), technical_task.result()

    projected = _project_business_result(
        req, order=order.evidence,
        diagnostic=technical.diagnostic_evidence if technical and technical.success else None,
        require_order=True, require_diagnostic=technical is not None,
        incomplete=bool(technical and (not technical.success or technical.status != "success")),
        answer_order_questions=True,
    )
    traces = ([order.trace] if order.trace else []) + (technical.tool_traces if technical else [])
    tools = ([TOOL_NAME] if order.trace else []) + (technical.tools_used if technical else [])
    agent_type = technical.agent_type if technical else None
    result = OrchestratorResult(
        request_id=req.request_id,
        response="\n\n".join(filter(None, (order.clarification, projected.content))),
        agent_type=agent_type,
        intent=req.intent,
        escalated=bool(technical and technical.escalate),
        latency_ms=(time.monotonic() - started) * 1000,
        agent_types=[agent_type] if agent_type else [],
        primary_agent=agent_type,
        tools_used=list(dict.fromkeys(tools)),
        tool_traces=traces,
        routing_reason="订单查询流程与技术支持并行执行" if technical else "明确的订单查询走只读流程",
        status=projected.status,
        route_kind="hybrid" if technical else "workflow",
        workflow_id=workflow_id,
    )
    orchestrator._record_tool_trace(result)
    return result
