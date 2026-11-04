"""Finite answers to explicit order-evidence questions, without new facts.

The caller supplies server-bound OrderEvidence from an actual read. This pure
module revalidates it and answers only a few explicit current-message questions.
It neither reads a store nor treats model text, assistant history, local error
advice or payment as proof of fulfilment or an actual permission inspection.
"""
from __future__ import annotations

import re

from core.diagnostic_evidence import select_diagnostic_request
from core.order_evidence import _checked_evidence, needs_order_evidence, resolve_order_id
from demo_business.store import SOURCE


_QUOTED = re.compile(r'“[^”\n]*”|「[^」\n]*」|‘[^’\n]*’|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
_META = re.compile(
    r"(?:翻译|怎么读|读音|这个词|这几个字|字面意思|是什么意思|什么含义|术语定义|"
    r"(?:写|编|制作|给).{0,8}(?:教程|示例)|仅(?:作|是)?引用|只是引用)|"
    r"\b(?:translat\w*|pronounc\w*|tutorial|example|quotation)\b", re.I)
_HOW_TO = re.compile(
    r"(?:如何|怎么|怎样|哪里|在哪).{0,6}(?:查询|查看|查找|找到|看|设置|修改|填写|显示)|"
    r"\bhow\s+(?:to|do\s+I)\s+(?:look\s+up|find|set|edit)\b", re.I)
_BARE_BIE = r"(?:(?<!\w)|(?<=请)|(?<=你)|(?<=您)|(?<=现在)|(?<=暂时))别"
_DECLINED = re.compile(
    r"(?:不需要|不想|不用|不要|无需|无须|先别|暂不|不问|不查|不核对|不解释|不说明|不讨论|不是问|" + _BARE_BIE + r")\s*"
    r"(?:(?:再|先|现在|帮我|给我|查|查询|查看|查阅|读取|看|核对|核实|检查|解释|说明|讨论|说|知道|了解)\s*){0,5}"
    r"(?:这笔订单的|该订单的|这个订单的|当前订单的|这单的|订单的)?"
    r"(?:付款|支付|扣款|出库|发货|接口|权限|何时|哪天|什么时候|"
    r"(?:这(?:个|些|两(?:个|项))?|该|此|上述)(?:说法|推断|结论|判断))|"
    r"\b(?:do\s+not|don't|no\s+need\s+to)\s+(?:explain|check|discuss|tell)\b", re.I)
_REPORTED = re.compile(
    r"(?:有人|客服|文档|同事|对方|示例|教程|他|她).{0,8}(?:说|写|提到)|"
    r"\b(?:someone|agent|document|guide|example)\s+(?:says?|said|writes?|wrote|mentions?)\b", re.I)
_ASK = re.compile(
    r"(?:查询|查看|查阅|读取|查|核对|核实|判断|确认|说明|解释|能否|能不能|是不是|是否|有没有|什么|哪|何时|吗|呢)|"
    r"\b(?:when|whether|does?|is|can|could|check|verify|explain|mean|prove)\b", re.I)
_REPORTED_ASK = re.compile(
    r"(?:请|麻烦|帮我|我想|我要|我需要).{0,8}(?:核实|核对|判断|确认|解释|说明)|"
    r"(?:核实|核对|判断|确认).{0,24}(?:说法|推断|结论|正确|成立|有依据)|"
    r"(?:是否|是不是|能否|能不能).{0,12}(?:正确|成立|有依据|支持)|"
    r"\b(?:verify|check|assess|confirm)\b.{0,35}\b(?:claim|statement|whether|document|guide)\b|"
    r"\b(?:is|was|does)\b.{0,25}\b(?:true|correct)\b", re.I)
_REFERENCE_ASK = re.compile(
    r"(?:核对|核实|判断|说明|解释).{0,28}(?:推断|说法|结论|判断)|"
    r"(?:推断|说法|结论|判断).{0,16}(?:正确|成立|可靠|有依据|支持|站得住)|"
    r"这(?:种说法|个结论|个推断|句话|样)?(?:是否|能否|是不是|能不能)?.{0,8}(?:对吗|成立|有依据|站得住)|"
    r"\b(?:is\s+that|does\s+that|can\s+that|verify\s+(?:this|that)).{0,24}(?:true|correct|prove|mean|claim)\b", re.I)
_PAYMENT = re.compile(
    r"(?:付款(?:成功|记录|凭证)?|付(?:了|过)款|支付(?:成功|记录|凭证)|扣款(?:成功|记录)?)|"
    r"\b(?:paid|payment|charged)\b", re.I)
_FULFILMENT = re.compile(r"(?:已经|已|是否|就)?(?:发货|出库)|\b(?:shipped|dispatched|shipment)\b", re.I)
_PERMISSION = re.compile(
    r"(?:接口|API|资源|服务|套餐|账号|账户).{0,8}(?:权限|授权)|"
    r"权限.{0,8}(?:已开通|已经开通|正常|有效|开通)|"
    r"\b(?:API|resource|service|account)\s+(?:access|permission|authori[sz]ation)\b", re.I)
_RELATION = re.compile(
    r"(?:代表|等于|说明|证明|当作|当成|就算|就是|意味着|推断|支持|保证|一定|肯定)|"
    r"\b(?:mean|means|prove|proves|imply|implies|confirm|confirms|guarantee)\b|=>|⇒", re.I)
_BOUNDARY = re.compile(r"(?:不要|不能|不可|" + _BARE_BIE + r").{0,3}把.{0,22}(?:当作|当成|当)", re.I)
_CONTINUATION = re.compile(r"^(?:也就|这也|因此|所以|进而|then\b|thus\b)", re.I)
_CURRENT_ID = re.compile(r"(?<![A-Za-z0-9_])DEMO-[0-9]{4}(?![A-Za-z0-9_-])(?:的)?", re.I)
_DISPATCH_DATE = re.compile(
    r"(?:出库|发货)(?:的)?(?:日期|时间|日|时刻)|"
    r"(?:哪天|何时|什么时候|几点).{0,20}(?:出库|发货)|"
    r"(?:出库|发货)(?:的|是|在|为|会|能|大概|预计|实际|计划|到底|应该|可以|\s){0,5}(?:哪天|何时|什么时候|几点)|"
    r"\b(?:dispatch|shipping|ship)[ _-]?(?:date|time)\b|\bwhen\b.{0,28}\b(?:ship\w*|dispatch\w*)\b", re.I)


def _segments(text):
    for match in re.finditer(r"([^，,。！？!?；;\n]+)([，,。！？!?；;\n]*)", text):
        parts = re.split(r"但是|不过|而是|但", match[1])
        for index, part in enumerate(parts):
            value = part.strip()
            if value:
                yield value, index == len(parts) - 1 and any(mark in match[2] for mark in "?？")


def _question_scopes(message):
    """Bounded language guards, not a general intent/entailment classifier."""
    outside = _QUOTED.sub(" ", message)
    active, claims = [], []
    for clause, question in _segments(outside):
        # This normalization is exclusively a negative-scope guard. Order
        # selection and evidence binding still use the untouched message.
        negative_scope = _CURRENT_ID.sub("", clause)
        if _META.search(clause) or _HOW_TO.search(clause) or _DECLINED.search(negative_scope):
            continue
        # A reported relation may itself contain verbs such as "说明" or
        # "prove". Those are not a user's request to inspect that relation.
        # Keep the claim inert unless a direct verification question exists,
        # here or via the separately bounded reference question below.
        if _REPORTED.search(clause) and not (_REPORTED_ASK.search(clause) or question):
            claims.append(clause)
        else:
            active.append((clause, question))
    reference = any(_REFERENCE_ASK.search(clause) for clause, _ in active)
    if reference:
        claims.extend(match[0][1:-1] for match in _QUOTED.finditer(message))
    date = any(_DISPATCH_DATE.search(clause) and (_ASK.search(clause) or question)
               for clause, question in active)
    fulfilment = permission = False
    previous_claim = None
    for clause, question in active:
        if _BOUNDARY.search(clause):
            # A declined explanation is not a question. An explicit request
            # to keep payment separate from a currently reported API problem
            # is, however, an applicable evidence-scope instruction.
            if select_diagnostic_request(message).required and _PAYMENT.search(clause) and _PERMISSION.search(clause):
                permission = True
            previous_claim = None
            continue
        if (_PAYMENT.search(clause) and _RELATION.search(clause)
                and (_ASK.search(clause) or question)):
            fulfilment |= bool(_FULFILMENT.search(clause))
            permission |= bool(_PERMISSION.search(clause))
        elif (previous_claim and _CONTINUATION.search(clause) and _RELATION.search(clause)
              and (_ASK.search(clause) or question)):
            # Only an immediately preceding payment claim may supply the
            # subject of an explicit "therefore/also prove ...?" question.
            fulfilment |= bool(_FULFILMENT.search(previous_claim) or _FULFILMENT.search(clause))
            permission |= bool(_PERMISSION.search(previous_claim) or _PERMISSION.search(clause))
        previous_claim = clause if _PAYMENT.search(clause) and _RELATION.search(clause) else None
    if reference:
        for claim in claims:
            if _PAYMENT.search(claim) and _RELATION.search(claim):
                fulfilment |= bool(_FULFILMENT.search(claim))
                permission |= bool(_PERMISSION.search(claim))
    return bool(date), fulfilment, permission


def render_order_answers(evidence, *, message, history=None, diagnostic=None) -> str:
    """Answer finite explicit subquestions for one validated visible snapshot.

    Identity must already have been bound by the caller; this API does not
    authenticate a user. It rechecks the retained canonical snapshot/customer,
    single target and current/user-history selection. Assistant messages cannot
    supply an order ID. ``diagnostic`` never grants proof of live permissions;
    this layer does not inspect its free text or claim an additional tool read.
    Unresolved, unavailable, conflicting or irrelevant input returns no answer.
    """
    if (type(message) is not str or not 1 <= len(message) <= 2000
            or not _checked_evidence(evidence) or evidence.status != "found"
            or len(evidence.records) != 1 or len(evidence.requested_order_ids) != 1
            or not needs_order_evidence(message, history)):
        return ""
    selection = resolve_order_id(message, history)
    record = evidence.records[0]
    if (selection.status != "resolved" or selection.order_id != record.order_id
            or evidence.requested_order_ids != (selection.order_id,)):
        return ""
    date, fulfilment, permission = _question_scopes(message)
    if not any((date, fulfilment, permission)):
        return ""
    order = record.to_dict()["order"]
    lines = [f"针对本次问题的只读说明（订单 {record.order_id}；来源 {SOURCE}；数据截至 {record.as_of}）："]
    if date:
        lines.append("出库/发货日期：当前快照未提供独立的实际出库或发货时间，因此无法确定这笔订单哪天出库或发货；记录更新时间和预计送达日期均不能代替实际出库/发货日期。")
    if fulfilment:
        lines.append("付款记录不能单独证明订单或全部包裹已经发货；是否发货须逐个包裹核对，不能把某个包裹的状态推广为整单状态。")
    if order["order_type"] == "digital" and (date or fulfilment):
        lines.append("当前为数字订单，没有实物包裹运输记录；实物出库/发货不适用，应另核对数字权益记录，不能将付款状态当作开通结果。")
    if permission:
        lines.append("付款记录不能单独证明接口权限已经开通；付款、履约与接口权限属于不同的证据范围。这里没有查询或核验实际接口权限，不能据此确认权限状态或本次故障根因。")
    return "\n".join(lines)
