"""Deterministic read-only workflow for a selected order inquiry."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

from agents.tools import read_demo_order
from core.order_evidence import OrderEvidence, extract_order_evidence, resolve_order_id
from demo_business.store import CUSTOMER_IDS

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request

WORKFLOW_ID = "order_lookup"
TOOL_NAME = "read_demo_order"


@dataclass
class OrderLookupResult:
    evidence: OrderEvidence | None = None
    trace: dict | None = None
    clarification: str = ""


async def run_order_lookup(req: Request) -> OrderLookupResult:
    selection = resolve_order_id(req.message, req.business_history)
    if selection.status != "resolved":
        return OrderLookupResult(clarification="请提供本次要查询的一个有效演示订单号。")
    if (type(req.demo_customer_id) is not str or req.demo_customer_id not in CUSTOMER_IDS
            or not callable(req.demo_order_reader)):
        return OrderLookupResult(clarification="当前会话没有可用的演示订单查询身份，订单状态尚未核实。")

    args = {"order_id": selection.order_id}
    started = time.monotonic()
    req.tool_attempts.append({"tool_name": TOOL_NAME, "retry_safe": True})
    result = await read_demo_order(req, args)
    evidence = extract_order_evidence(
        result, expected_order_id=selection.order_id, trusted_customer_id=req.demo_customer_id,
    )
    return OrderLookupResult(evidence=evidence, trace={
        "execution_kind": "workflow",
        "workflow_id": WORKFLOW_ID,
        "tool_name": TOOL_NAME,
        "input": args,
        "success": True,
        "result_success": evidence.status in {"found", "not_found"},
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "source": "serveflow-mysql-demo-v1",
        "retry_safe": True,
    })
