"""Finite answers to explicit order-evidence questions, without new facts.

The caller supplies server-bound OrderEvidence from an actual read. This pure
module revalidates it and answers only a few explicit current-message questions.
It neither reads a store nor treats model text, assistant history, local error
advice or payment as proof of fulfilment or an actual permission inspection.
"""
from __future__ import annotations

import re

from core.diagnostic_evidence import select_diagnostic_request
from core.order_evidence import _amount, _checked_evidence, _text, needs_order_evidence, resolve_order_id
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
    r"(?:(?:再|先|现在|帮我|给我|查|查询|查看|查阅|读取|看|核对|核实|检查|解释|说明|讨论|说|知道|了解|计算|算|合计|比较)\s*){0,5}"
    r"(?:这笔订单的|该订单的|这个订单的|当前订单的|这单的|订单的)?"
    r"(?:付款|支付|扣款|出库|发货|接口|权限|退款|签收|收货|丢件|物流|续费|到期|权益|材料|原因|何时|哪天|什么时候|"
    r"金额|价格|商品|明细|合计|总计|运费|邮费|税费|优惠|折扣|差额|分摊|"
    r"(?:这(?:个|些|两(?:个|项))?|该|此|上述)(?:说法|推断|结论|判断))|"
    r"\b(?:do\s+not|don't|no\s+need\s+to)\s+(?:explain|check|discuss|tell|calculate|sum|compare)\b", re.I)
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
_ITEMS = re.compile(r"商品|明细|货品|单价|标价|\b(?:items?|products?|unit\s+prices?|line\s+items?)\b", re.I)
_ITEM_SUM = re.compile(r"合计|总计|合起来|加起来|加总|一共|总共|小计|总价|\b(?:sum|subtotal|total|add\s+up)\b", re.I)
_AMOUNT_COMPARE = re.compile(r"比较|对比|等于|一致|相同|相符|一样|对得上|吻合|差额|差多少|\b(?:compare|equal|match|difference|same)\b", re.I)
_PAYMENT_AMOUNT = re.compile(r"实付|付款|支付|扣款|\b(?:paid|payments?|charges?)\b", re.I)
_ORDER_AMOUNT = re.compile(r"订单.{0,3}(?:金额|总额|总价)|\border\s+(?:amount|total)\b", re.I)
_PRICE_REQUEST = re.compile(r"计算|算一下|算出|算算|合计|比较|对比|多少(?:元|钱)|\b(?:calculate|sum|compare|add|how\s+much)\b", re.I)
_UNKNOWN_AMOUNT_BOUNDARY = re.compile(r"(?:未知|缺失|未提供).{0,10}(?:不要|不能|不可|别).{0,6}(?:补|填|按|算)", re.I)
_ITEM_FILTER = re.compile(
    r"(?:只|仅)(?:把|将)?(?:算|计算|合计|加总|核算|看|核对)(?!\s*(?:的)?\s*(?:所有|全部|整单|商品(?:明细|合计|总)))|"
    r"其中|部分商品|(?:第|前|后)[一二三四五六七八九十0-9]+(?:件|项)|(?:单个|某个|某件)商品", re.I)
_ALL_ITEMS = re.compile(r"所有商品|全部商品|整单商品|全部明细|所有明细|\ball\s+(?:items|products|line\s+items)\b", re.I)


def _amount_answers(order, message):
    """Compute only a requested full-item total from checked integer fields.

    This source has no fee, discount or per-item transaction allocation fields.
    A record-consistency check never creates those facts or a net payment.
    Filtered item questions are left outside this finite full-order helper.
    """
    active, claims, reference = _active_scopes(message)
    questions = [clause for clause, punctuated in active
                 if punctuated or _ASK.search(clause) or _PRICE_REQUEST.search(clause)
                 or _UNKNOWN_AMOUNT_BOUNDARY.search(clause)]
    if reference:
        questions.extend(claims)
    text = "；".join(questions)
    product_context = (any(_ITEMS.search(clause) for clause, _ in active)
                       or (reference and any(_ITEMS.search(clause) for clause in claims)))
    item_sum = any(_ITEM_SUM.search(clause)
                   and (_ITEMS.search(clause)
                        or re.search(r"合起来|加起来|加总|^(?:这(?:些|几项|几件))?合计", clause))
                   for clause in questions)
    total = product_context and (item_sum or any(
        _AMOUNT_COMPARE.search(clause) and _ITEMS.search(clause)
        and (_PAYMENT_AMOUNT.search(clause) or _ORDER_AMOUNT.search(clause))
        and not re.search(r"单价|每件|每项|每个商品|付款状态|支付状态|\b(?:unit\s+price|per.item|payment\s+status)\b", clause, re.I)
        for clause in questions))
    # Never silently interpret a request for one named/filtered item as the
    # full-order total. Names remain source data, not an instruction parser.
    named = [row for row in order["items"]
             if row["name"] in text or row["item_id"] in text]
    if (_ITEM_FILTER.search(_CURRENT_ID.sub("", text))
            or (named and len(named) != len(order["items"]) and not _ALL_ITEMS.search(text))
            or (re.search(r"数量|件数|几件|多少件|\bquantity\b", text, re.I)
                and not re.search(r"金额|价格|单价|总价|实付|付款|支付|元|\b(?:price|amount|paid|payment|cost)\b", text, re.I))):
        total = False
    lines = []
    if total:
        subtotal = sum(row["quantity"] * row["unit_amount_minor"] for row in order["items"])
        lines.append(f"商品金额核算（币种 {order['currency']}，人民币；按记录单价×数量）：")
        for row in order["items"]:
            lines.append(f"- {_text(row['name'])}：{_amount(row['unit_amount_minor'])} × {row['quantity']} = {_amount(row['unit_amount_minor'] * row['quantity'])}。")
        lines.append(f"商品明细合计：{_amount(subtotal)}。")
        if _ORDER_AMOUNT.search(text) or _PAYMENT_AMOUNT.search(text):
            lines.append(f"与订单金额记录比较：{_amount(order['total_amount_minor'])}，与商品明细合计数值一致。")
        if _PAYMENT_AMOUNT.search(text):
            if order["payments"]:
                payment_total = sum(row["amount_minor"] for row in order["payments"])
                state = order["payment_status"]
                lines.append(f"与付款记录比较：{len(order['payments'])}条付款记录的原始金额合计 {_amount(payment_total)}（{state}），与商品明细合计数值一致。")
                if state == "refunded":
                    lines.append("付款记录状态为refunded；上述原始金额不能视为退款后的当前净实付，真实退款到账仍未核验。")
            else:
                lines.append("当前记录为未付款且未提供付款记录，无法取得可比较的实付金额；不能将空付款记录按0元实付处理。")
        lines.append("这是本次合成演示记录内的金额核算，不能证明真实支付渠道入账或履约完成；商品记录小计也不能直接作为独立成交价、优惠分摊或可退金额。")
    missing = []
    for label, pattern in (("运费/邮费", r"运费|邮费|\b(?:shipping\s+(?:fee|cost)|freight|postage)\b"),
                           ("优惠/折扣", r"优惠|折扣|优惠券|\b(?:discount|coupon)\w*\b"),
                           ("税费", r"税费|税额|\btax\w*\b")):
        if re.search(pattern, text, re.I):
            missing.append(label)
    if missing:
        lines.append(f"当前快照未提供独立的{'、'.join(missing)}金额字段，具体金额未知，不能按0元补齐；商品合计与付款记录相等也不能证明这些费用不存在或已如何分摊。")
    if re.search(r"分摊|每(?:件|项|个商品).{0,10}(?:实付|实际支付|实际付款|成交)|\b(?:allocat\w*|per.item\s+(?:paid|transaction))\b", text, re.I):
        lines.append("当前记录没有每件商品的成交价、优惠分摊或净实付字段；不能把记录单价或商品小计直接称为每件实际成交/实付金额，具体分摊仍未知。")
    return lines


def _segments(text):
    for match in re.finditer(r"([^，,。！？!?；;\n]+)([，,。！？!?；;\n]*)", text):
        parts = re.split(r"但是|不过|而是|但", match[1])
        for index, part in enumerate(parts):
            value = part.strip()
            if value:
                yield value, index == len(parts) - 1 and any(mark in match[2] for mark in "?？")


def _active_scopes(message):
    """Separate direct questions from inert examples and declined topics."""
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
    return active, claims, reference


def _question_scopes(message):
    """Bounded language guards, not a general intent/entailment classifier."""
    active, claims, reference = _active_scopes(message)
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


def _record_relationship_scopes(message):
    """Finite evidence-boundary topics, not a new free-form answer writer.

    Match the user's question, never model entities or assistant prose. The
    renderer additionally gates each explanation on the validated record's
    actual shape/status. Unsupported phrasings remain outside this helper;
    these flags cannot authorize another lookup or a business mutation.
    """
    active, claims, reference = _active_scopes(message)
    questions = [clause for clause, punctuated in active
                 if punctuated or _ASK.search(clause)
                 or re.search(r"整理|应该保留|需要保留|为什么|为何|\bwhy\b", clause, re.I)]
    if reference:
        questions.extend(claims)
    topics = set()
    for clause in questions:
        if re.search(r"(?:签收|配送|物流).{0,6}争议|本人.{0,5}(?:收到|收货)|丢件|丢失|承运方.{0,8}责任|\b(?:delivery.disput\w*|personally.receiv\w*|lost.parcel)\b", clause, re.I):
            topics.add("delivery_dispute")
        if (re.search(r"退款|\brefund\w*\b", clause, re.I)
                and re.search(r"没有|没(?:有|看到|列出|记录)|无(?:记录|退款)|空|未(?:提供|列出|记载)|查不到|零条|缺失|从未|从来|自动|申请|\b(?:empty|missing|never|automatic)\b", clause, re.I)):
            topics.add("refund_absence")
        if (re.search(r"权益|开通|激活|\b(?:entitlement|activ\w*)\b", clause, re.I)
                and re.search(r"为什么|为何|原因|根因|故障|怎么回事|导致|缘故|\b(?:why|cause|reason)\b", clause, re.I)):
            topics.add("activation_reason")
        if (re.search(r"有效期|到期|期限|永不过期|不会过期|\b(?:expir\w*|permanent|forever)\b", clause, re.I)
                and re.search(r"空|没有|没写|未(?:提供|记载|列出)|缺失|不显示|未知|永久|终身|一直|永不|不会过期|\b(?:null|missing|permanent|forever)\b", clause, re.I)):
            topics.add("expiry_unknown")
        if (re.search(r"续费|扣款|\b(?:renew\w*|charg\w*)\b", clause, re.I)
                and re.search(r"未来|以后|后续|今后|往后|后面|到时候|再扣|还会|一直|到期|保证|证明|一定|\b(?:future|guarantee|prove|expir\w*)\b", clause, re.I)):
            topics.add("future_renewal")
        if (re.search(r"材料|凭证|证据|包装|\b(?:materials|evidence|packaging)\b", clause, re.I)
                and re.search(r"整理|保留|留存|准备|提供|哪些|什么|\b(?:keep|retain|prepare|which|what)\b", clause, re.I)):
            topics.add("dispute_materials")
        if (re.search(r"删除|退款|\b(?:delet\w*|refund\w*)\b", clause, re.I)
                and _RELATION.search(clause)):
            topics.add("diagnostic_inference")
    return topics


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
    topics = _record_relationship_scopes(message)
    order = record.to_dict()["order"]
    lines = _amount_answers(order, message)
    if date:
        lines.append("出库/发货日期：当前快照未提供独立的实际出库或发货时间，因此无法确定这笔订单哪天出库或发货；记录更新时间和预计送达日期均不能代替实际出库/发货日期。")
    if fulfilment:
        lines.append("付款记录不能单独证明订单或全部包裹已经发货；是否发货须逐个包裹核对，不能把某个包裹的状态推广为整单状态。")
    if order["order_type"] == "digital" and (date or fulfilment):
        lines.append("当前为数字订单，没有实物包裹运输记录；实物出库/发货不适用，应另核对数字权益记录，不能将付款状态当作开通结果。")
    if permission:
        lines.append("付款记录不能单独证明接口权限已经开通；付款、履约与接口权限属于不同的证据范围。这里没有查询或核验实际接口权限，不能据此确认权限状态或本次故障根因。")
    disputed = any(row["status"] == "delivery_disputed" for row in order["shipments"])
    if disputed and "delivery_dispute" in topics:
        lines.append("包裹的签收/配送争议记录不等于本人已收到，也不能单独证明承运方丢件或责任；真实签收经过、是否丢件及责任结论仍未知，不能将delivery_disputed改写成签收成功。")
    if not order["refunds"] and "refund_absence" in topics:
        lines.append("当前快照没有退款行，只表示本次记录未提供退款证据；不能证明已自动退款或实际未退款，也不能证明从未申请。实际申请、审核及到账情况仍需业务系统或官方渠道核验。")
    if ("activation_reason" in topics
            and any(row["status"] == "activation_failed" for row in order["entitlements"])):
        lines.append("权益记录为开通失败（activation_failed），但没有提供本例失败原因；付款记录不能证明已开通，具体根因和真实当前可用性仍未知，不能以猜测的同步或系统故障作结论，也不应据此重复付款。")
    if ("expiry_unknown" in topics
            and any(row["expires_at"] is None for row in order["entitlements"])):
        lines.append("权益到期日为空表示有效期未知，不能理解为永久有效；它也没有说明为何缺失，不能自行补造到期日或原因。")
    if "future_renewal" in topics and order["entitlements"]:
        lines.append("记录更新时间、权益到期日与当前自动续费开关含义不同；到期记录或当前开关不能保证未来续费、扣款或可用状态，未来实际情况尚未核验。")
    if disputed and "dispute_materials" in topics:
        lines.append("材料准备建议：按需保留外包装、商品清单、包裹/运单及签收信息、脱敏照片和沟通凭证；不要在聊天提交完整地址、电话、证件、密码或验证码。")
        lines.append("本次只读查询没有联系承运方、发起调查或创建真实工单；签收实情、责任及退款/赔偿结论仍需官方渠道核验，不能保证退款或赔偿。")
    diagnostic_request = select_diagnostic_request(message)
    if "diagnostic_inference" in topics and diagnostic_request.error_code == "404":
        lines.append("接口404、包裹状态与退款状态属于不同证据范围；404不能单独证明数据库订单已删除或退款成功，真实接口根因及商家当前状态仍未知。这份合成记录不是实际商家后台核验。")
    if not lines:
        return ""
    return "\n".join([f"针对本次问题的只读说明（订单 {record.order_id}；来源 {SOURCE}；数据截至 {record.as_of}）：", *lines])
