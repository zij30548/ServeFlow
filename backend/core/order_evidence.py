"""Pure, fail-closed projection of read-only demo order records.

This layer never queries a store, executes an operation or asks a model to
rewrite business facts. Store row and relation checks remain authoritative;
this boundary additionally checks request identity, JSON shape and references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re

from demo_business.store import (
    CUSTOMER_IDS, MAX_RELATED_ROWS, ORDER_PATTERN, SOURCE, TABLE_COLUMNS,
    DemoStoreError, _validate_order_relations, _validate_row,
)
from core.policy_grounding import has_record_question


MAX_EVIDENCE_BYTES = 65_536
MAX_ORDERS = 6
MAX_MERGE_INPUTS = 12
_RELATED = ("items", "payments", "shipments", "entitlements", "refunds")
_ERROR_CODES = frozenset({"invalid_demo_order_arguments", "demo_session_required",
                         "demo_store_unavailable", "invalid_demo_order_result"})
_REASONS = frozenset({"", "invalid_context", "invalid_result", "conflicting_records", "evidence_limit"})
_STATUSES = frozenset({"found", "not_found", "unavailable", "invalid", "partial"})


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    as_of: str
    canonical_json: str = field(repr=False)

    def to_dict(self):
        """A fresh copy; callers cannot mutate the retained evidence."""
        return json.loads(self.canonical_json)


@dataclass(frozen=True)
class OrderEvidence:
    status: str
    records: tuple[OrderRecord, ...] = ()
    requested_order_ids: tuple[str, ...] = ()
    not_found_order_ids: tuple[str, ...] = ()
    unavailable_order_ids: tuple[str, ...] = ()
    reason: str = ""
    customer_id: str | None = field(default=None, repr=False)
    not_found_as_of: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OrderSelection:
    status: str
    order_id: str | None = None
    source: str = "none"


def _invalid(reason="invalid_result", customer_id=None):
    return OrderEvidence("invalid", reason=reason if reason in _REASONS else "invalid_result",
                         customer_id=customer_id if customer_id in CUSTOMER_IDS else None)


def _order_id(value):
    return type(value) is str and ORDER_PATTERN.fullmatch(value) is not None


def _timestamp(value):
    if type(value) is not str or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def _json_copy(value):
    count = 0

    def check(item, depth=0):
        nonlocal count
        count += 1
        if count > 4096 or depth > 12:
            raise ValueError
        if type(item) is dict:
            if len(item) > 32 or any(type(key) is not str or len(key) > 160 for key in item):
                raise ValueError
            for child in item.values():
                check(child, depth + 1)
        elif type(item) is list:
            if len(item) > MAX_RELATED_ROWS:
                raise ValueError
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            if len(item) > MAX_EVIDENCE_BYTES:
                raise ValueError
        elif type(item) not in (int, float, bool, type(None)):
            raise ValueError

    def unique(pairs):
        result = {}
        for key, child in pairs:
            if key in result:
                raise ValueError
            result[key] = child
        return result

    if type(value) is str:
        if len(value) > MAX_EVIDENCE_BYTES or len(value.encode("utf-8")) > MAX_EVIDENCE_BYTES:
            raise ValueError
        value = json.loads(value, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    check(value)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise ValueError
    return json.loads(raw), raw


def extract_order_evidence(result, *, expected_order_id, trusted_customer_id):
    """Validate one tool result against the server-bound request and identity."""
    if not _order_id(expected_order_id) or type(trusted_customer_id) is not str or trusted_customer_id not in CUSTOMER_IDS:
        return _invalid("invalid_context")
    try:
        value, raw = _json_copy(result)
        if type(value) is not dict or value.get("source") != SOURCE or value.get("is_demo") is not True:
            raise ValueError
        ids = (expected_order_id,)
        if value.get("status") == "unavailable":
            if (set(value) != {"source", "is_demo", "success", "status", "error_code"}
                    or value["success"] is not False or type(value["error_code"]) is not str
                    or value["error_code"] not in _ERROR_CODES):
                raise ValueError
            return OrderEvidence("unavailable", requested_order_ids=ids, unavailable_order_ids=ids,
                                 customer_id=trusted_customer_id)
        if set(value) != {"source", "is_demo", "as_of", "status", "order"} or not _timestamp(value["as_of"]):
            raise ValueError
        if value["status"] == "not_found":
            if value["order"] is not None:
                raise ValueError
            return OrderEvidence("not_found", requested_order_ids=ids, not_found_order_ids=ids,
                                 customer_id=trusted_customer_id,
                                 not_found_as_of=((expected_order_id, value["as_of"]),))
        if value["status"] != "found":
            raise ValueError
        order = value["order"]
        if type(order) is not dict or set(order) != set(TABLE_COLUMNS["orders"]) | set(_RELATED):
            raise ValueError
        _validate_row("orders", {key: order[key] for key in TABLE_COLUMNS["orders"]})
        if order["order_id"] != expected_order_id or order["customer_id"] != trusted_customer_id:
            raise ValueError
        for table in _RELATED:
            rows = order[table]
            if type(rows) is not list or len(rows) > MAX_RELATED_ROWS:
                raise ValueError
            seen = set()
            for row in rows:
                _validate_row(table, row)
                primary = row[TABLE_COLUMNS[table][0]]
                if row["order_id"] != expected_order_id or primary in seen:
                    raise ValueError
                seen.add(primary)
        _validate_order_relations(order)
        return OrderEvidence("found", (OrderRecord(expected_order_id, value["as_of"], raw),), ids,
                             customer_id=trusted_customer_id)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError, DemoStoreError):
        return _invalid(customer_id=trusted_customer_id)


def _checked_evidence(evidence):
    if (type(evidence) is not OrderEvidence or type(evidence.status) is not str
            or evidence.status not in _STATUSES or type(evidence.reason) is not str
            or evidence.reason not in _REASONS or type(evidence.customer_id) is not str
            or evidence.customer_id not in CUSTOMER_IDS):
        return False
    groups = (evidence.records, evidence.requested_order_ids, evidence.not_found_order_ids, evidence.unavailable_order_ids)
    if any(type(group) is not tuple or len(group) > MAX_ORDERS for group in groups):
        return False
    if evidence.status == "invalid":
        return not any(groups) and evidence.not_found_as_of == ()
    for group in groups[1:]:
        if any(not _order_id(identity) for identity in group) or len(set(group)) != len(group):
            return False
    if not evidence.requested_order_ids:
        return False
    seen = set()
    for record in evidence.records:
        if type(record) is not OrderRecord or not _order_id(record.order_id) or record.order_id in seen:
            return False
        checked = extract_order_evidence(record.canonical_json, expected_order_id=record.order_id,
                                         trusted_customer_id=evidence.customer_id)
        if checked.status != "found" or checked.records[0] != record:
            return False
        seen.add(record.order_id)
    missing = set(evidence.not_found_order_ids)
    unavailable = set(evidence.unavailable_order_ids)
    if (type(evidence.not_found_as_of) is not tuple or len(evidence.not_found_as_of) != len(missing)
            or any(type(pair) is not tuple or len(pair) != 2 or not _order_id(pair[0])
                   or not _timestamp(pair[1]) for pair in evidence.not_found_as_of)
            or {pair[0] for pair in evidence.not_found_as_of} != missing):
        return False
    if seen & missing or seen & unavailable or missing & unavailable:
        return False
    if set(evidence.requested_order_ids) != seen | missing | unavailable:
        return False
    expected = ("partial" if sum(bool(group) for group in (seen, missing, unavailable)) > 1 else
                "found" if seen else "not_found" if missing else "unavailable")
    return evidence.status == expected and evidence.reason == ""


def merge_order_evidence(evidences):
    """Merge bounded same-identity results; never silently choose between facts."""
    if type(evidences) not in (list, tuple) or not 1 <= len(evidences) <= MAX_MERGE_INPUTS:
        return _invalid("evidence_limit")
    customer = None
    records = {}
    outcomes = {}
    missing_times = {}
    requested = []
    for evidence in evidences:
        if not _checked_evidence(evidence) or evidence.status == "invalid":
            return _invalid()
        if customer is not None and customer != evidence.customer_id:
            return _invalid("invalid_context")
        customer = evidence.customer_id
        for identity in evidence.requested_order_ids:
            if identity not in requested:
                requested.append(identity)
        for status, identities in (("found", [record.order_id for record in evidence.records]),
                                   ("not_found", evidence.not_found_order_ids),
                                   ("unavailable", evidence.unavailable_order_ids)):
            for identity in identities:
                # Distinct outcomes for one request identity are not silently
                # repaired, even if one of them contains no usable data.
                if identity in outcomes and outcomes[identity] != status:
                    return _invalid("conflicting_records", customer)
                outcomes[identity] = status
        for identity, as_of in evidence.not_found_as_of:
            if identity in missing_times and missing_times[identity] != as_of:
                return _invalid("conflicting_records", customer)
            missing_times[identity] = as_of
        for record in evidence.records:
            if record.order_id in records and records[record.order_id] != record:
                return _invalid("conflicting_records", customer)
            records[record.order_id] = record
        if len(requested) > MAX_ORDERS or sum(len(row.canonical_json.encode("utf-8")) for row in records.values()) > MAX_EVIDENCE_BYTES:
            return _invalid("evidence_limit", customer)
    statuses = set(outcomes.values())
    status = next(iter(statuses)) if len(statuses) == 1 else "partial"
    return OrderEvidence(status, tuple(records.values()), tuple(requested),
                         tuple(identity for identity in requested if outcomes[identity] == "not_found"),
                         tuple(identity for identity in requested if outcomes[identity] == "unavailable"),
                         customer_id=customer, not_found_as_of=tuple(missing_times.items()))


def _text(value):
    # Free-text business names are data, not Markdown or instructions.
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", text)


def _amount(value):
    return "未知" if value is None else f"{value // 100}.{value % 100:02d} 元"


def render_order_evidence(evidence):
    """Deterministic record projection, not a policy or merchant promise."""
    if not _checked_evidence(evidence) or evidence.status == "invalid":
        return "订单记录校验未通过，无法可靠展示；本次未执行任何订单或资金操作。"
    lines = ["以下为合成演示订单记录，不是真实商家交易；仅展示读取结果，未执行订单或资金操作。",
             "记录中的付款或退款状态不能代替真实支付渠道的入账核验，也不能单凭一笔记录认定重复扣款。"]
    for record in evidence.records:
        order = record.to_dict()["order"]
        lines.extend([f"\n订单 {record.order_id}（演示数据）", f"数据截至：{record.as_of}",
                      f"订单记录更新时间：{order['record_updated_at']}",
                      f"订单类型：{dict(physical='实物', digital='数字', bundle='组合')[order['order_type']]}；币种：CNY",
                      f"订单金额：{_amount(order['total_amount_minor'])}；付款状态：{dict(paid='已付款', unpaid='未付款', refunded='已退款')[order['payment_status']]}（{order['payment_status']}）",
                      # The validated current source has no lifecycle/cancellation
                      # field. This explains that limit without adding a fact to
                      # the snapshot or treating payment/fulfilment as cancellation.
                      "订单取消状态：未知（当前快照未提供取消状态）；付款状态及付款/发货记录是否存在，均不能证明订单已取消或未取消。",
                      "商品明细："])
        for row in order["items"]:
            lines.append(f"- {_text(row['name'])}（商品ID：{_text(row['item_id'])}；类型：{row['item_type']}），数量 {row['quantity']}，单价 {_amount(row['unit_amount_minor'])}")
        lines.append("付款记录：" if order["payments"] else "当前快照未提供付款记录。")
        for row in order["payments"]:
            lines.append(f"- 付款 {_text(row['payment_id'])}：金额 {_amount(row['amount_minor'])}，状态 {dict(paid='已付款', refunded='已退款')[row['status']]}（{row['status']}），记录渠道 {row['channel']}；记录更新时间：{row['record_updated_at']}")
        lines.append("逐包裹运输记录：" if order["shipments"] else "当前快照未提供包裹运输记录。")
        names = {row["item_id"]: row["name"] for row in order["items"]}
        for row in order["shipments"]:
            label = dict(in_transit="运输中", awaiting_dispatch="待发货", preorder="预售", delivered="已签收", delivery_disputed="签收状态有争议")[row["status"]]
            estimate = "未知" if row["estimated_delivery"] is None else f"{row['estimated_delivery']}（记录中的预计日期，仅供参考，不保证送达）"
            tracking = "未知" if row["tracking_id"] is None else _text(row["tracking_id"])
            lines.append(f"- 包裹 {_text(row['shipment_id'])}，对应商品 {_text(names[row['item_id']])}（{_text(row['item_id'])}）：{label}（{row['status']}）；物流单号：{tracking}；预计送达：{estimate}；记录更新时间：{row['record_updated_at']}")
        if len({row["status"] for row in order["shipments"]}) > 1:
            lines.append("各包裹状态不同，应逐包核对；不能仅凭某个包裹的状态概括为整单均已发货。")
        lines.append("数字权益记录：" if order["entitlements"] else "当前快照未提供数字权益记录。")
        for row in order["entitlements"]:
            label = dict(activation_failed="开通失败", revoked_after_refund="退款后已撤销", active="有效")[row["status"]]
            expiry = row["expires_at"] if row["expires_at"] is not None else "未知"
            renewal = "开启" if row["auto_renew"] else "关闭"
            lines.append(f"- 权益 {_text(row['entitlement_id'])}，{_text(row['package_name'])}：{label}（{row['status']}）；自动续费：{renewal}；有效期截至：{expiry}；记录更新时间：{row['record_updated_at']}")
        if order["payment_status"] == "paid" and any(row["status"] == "activation_failed" for row in order["entitlements"]):
            lines.append("付款已成功，但存在开通失败的权益记录；支付成功不等于权益已开通，不能据此要求重复付款。")
        if any(row["status"] == "active" and row["auto_renew"] is False for row in order["entitlements"]):
            lines.append("当前记录中存在有效且已关闭自动续费的权益；关闭续费不等于当前权益撤销，也不等于退款，是否退款以退款记录为准。")
        lines.append("退款记录：" if order["refunds"] else "当前快照未提供退款记录。")
        for row in order["refunds"]:
            label = dict(pending_review="待审核", completed="已完成")[row["status"]]
            lines.append(f"- 退款 {_text(row['refund_id'])}：{label}（{row['status']}）；记录金额：{_amount(row['amount_minor'])}；记录更新时间：{row['record_updated_at']}")
        if order["refunds"]:
            lines.append("当前退款记录未提供到账日期，到账时间未知；订单金额不能代替退款金额。")
    for identity in evidence.not_found_order_ids:
        lines.append(f"订单 {identity}：在当前演示身份的可见记录中未找到（数据截至：{dict(evidence.not_found_as_of)[identity]}）；无法据此判断其他身份是否存在该订单。")
    for identity in evidence.unavailable_order_ids:
        lines.append(f"订单 {identity}：本次读取未完成，记录未知；不能当作订单不存在或业务已完成。")
    if evidence.status == "partial":
        lines.append("本次仅取得部分查询结果，未取得的记录仍未知。")
    return "\n".join(lines)


_TOKEN = re.compile(r"(?<![A-Za-z0-9_])DEMO-[A-Za-z0-9_-]*", re.I)
_RESET = re.compile(r"(?:换(?:一)?(?:个|笔)|另(?:外)?(?:一)?(?:个|笔)|新(?:的)?)(?:问题|订单)|\b(?:another|different|new)\s+order\b", re.I)
_QUOTED = re.compile(r'“[^”\n]*”|「[^」\n]*」|‘[^’\n]*’|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
_READ = r"(?:查询|查一下|查查|查看|查|核对|核实|检查|看看|跟踪|追踪|读取|look\s+up|check|track|query|verify)"
_DECLINED = re.compile(
    rf"(?:不需要|不用|不要|不想|无需|无须|别|先别|暂不|不再|不查询|不查看|不查|不核对|不跟踪|不是|并非)"
    rf"\s*(?:(?:再|继续|现在|帮我|给我)\s*){{0,3}}(?:{_READ}\s*)?"
    r"(?:(?:我的|这个|这笔|该|那个|当前)\s*)?(?:订单|包裹|退款|付款|权益|订阅|会员|DEMO-)", re.I)
_DECLINED_EN = re.compile(r"\b(?:do not|don't|dont|no need to)\s+(?:check|track|query|verify|look up|read)\b", re.I)
_META = re.compile(r"(?:这个词|这几个字|字样|怎么翻译|翻译|是什么意思|怎么读|仅(?:为|是)?示例|只是示例|举个例子|用作例子|引用|\b(?:translate|translation|example|quoted?)\b)", re.I)
_REPORTED = re.compile(r"(?:客服|文档|示例|对方|有人|以前).{0,8}(?:说|写|提到)|\b(?:agent|document|someone|they)\s+(?:said|wrote|mentioned)\b", re.I)
_COURTESY = re.compile(r"(?:谢谢(?:你|您)?|多谢|你好|您好|好的|明白了|知道了|hello|hi|thanks|thank you|ok)[！!。.?？\s]*", re.I)


def _order_tokens(text):
    # These two literal labels name demo customers, not orders. Ignoring their
    # spelling here grants no identity: the server-bound customer remains the
    # sole authorization input to extract_order_evidence/read_demo_order.
    return [token for token in _TOKEN.findall(text) if token.lower() not in CUSTOMER_IDS]


def _active_order_text(message):
    """Remove declined/reported IDs without reassigning them to old orders.

    These are bounded request guards, not a general semantic classifier. An
    unresolved reference remains unresolved instead of granting a tool read.
    """
    active = []
    blocked_reference = False
    for clause in re.split(r"[，,。！？!?；;\n]", message):
        if not clause.strip():
            continue
        ids = bool(_order_tokens(clause))
        if _DECLINED.search(clause) or _DECLINED_EN.search(clause):
            blocked_reference = True
            continue
        if ids and (_META.search(clause) or _COURTESY.fullmatch(_TOKEN.sub("", clause).strip())):
            blocked_reference = True
            continue
        outside = _QUOTED.sub(" ", clause)
        if ids and _REPORTED.search(clause) and not re.search(_READ, outside, re.I):
            blocked_reference = True
            continue
        if ids and not _order_tokens(outside) and not re.search(_READ, outside, re.I):
            blocked_reference = True
            continue
        active.append(clause)
    return "，".join(active), blocked_reference


def _selection_from_text(text, source):
    tokens = _order_tokens(text)
    if not tokens:
        return None
    ids = list(dict.fromkeys(token.upper() for token in tokens))
    if any(not _order_id(identity) for identity in ids):
        return OrderSelection("invalid", source=source)
    if len(ids) != 1:
        return OrderSelection("ambiguous", source=source)
    return OrderSelection("resolved", ids[0], source)


def resolve_order_id(message, history=None, entities=None):
    """Resolve explicit/context IDs, not intent, ownership or read permission.

    Current explicit IDs win. A task reset or malformed/ambiguous current ID
    cannot fall back to an older order. Only actual user history is considered.
    The legacy entities argument is deliberately ignored: model-produced
    entities may neither grant a read nor override an actual user identity.
    """
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return OrderSelection("invalid")
    active, blocked_reference = _active_order_text(message)
    selected = _selection_from_text(active, "message")
    if selected is not None:
        return selected
    if blocked_reference:
        return OrderSelection("missing")
    if _RESET.search(message):
        return OrderSelection("missing")
    if history is not None and type(history) not in (list, tuple):
        return OrderSelection("invalid", source="history")
    for item in reversed((history or [])[-6:]):
        if type(item) is not dict or item.get("role") != "user":
            continue
        prior = item.get("content")
        if type(prior) is not str or not 1 <= len(prior) <= 2000:
            return OrderSelection("invalid", source="history")
        active_prior, blocked_prior = _active_order_text(prior)
        selected = _selection_from_text(active_prior, "history")
        if selected is not None:
            return selected
        if blocked_prior or _RESET.search(prior):
            return OrderSelection("missing")
    return OrderSelection("missing")


def needs_order_evidence(message, history=None):
    """Whether the user actually asks about a concrete, fresh order record.

    A true result still requires resolve_order_id and server-side ownership
    checks. Multiple IDs request clarification, never selection of the first.
    Model entities are deliberately not accepted as a lookup source here.
    """
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return False
    active, _ = _active_order_text(message)
    if not active.strip() or _COURTESY.fullmatch(active.strip()):
        return False
    if has_record_question(active):
        return True
    selected = resolve_order_id(message, history)
    if selected.status != "resolved":
        return False
    subject = r"(?:会员|权益|订阅|包裹|物流|付款|退款|订单|配件|设备|自动续费|\b(?:membership|entitlement|subscription|package|shipment|payment|refund|order|renewal)\b)"
    detail = r"(?:有效|到期|截止|续费|开通|激活|状态|进度|单号|金额|到账|发货|送达|关闭|开启|\b(?:expir\w*|valid|renew\w*|activ\w*|status|amount|tracking|deliver\w*)\b)"
    query = rf"(?:{_READ}|什么时候|到什么时候|何时|哪天|多少|是否|是不是|有没有|[？?]|呢|吗|\b(?:when|what|where|how|is|does|has)\b)"
    return bool(re.search(subject, active, re.I) and re.search(detail, active, re.I)
                and (re.search(query, active, re.I) or re.search(r"[？?]", message)))


_MUTATION_OWNER = r"(?:(?:我的|这笔订单的|这个订单的|该订单的|这单的|这笔|这个|该|当前|本次)\s*)?"
_NEXT_CHARGE = r"(?:(?:下一笔|下笔|这笔|该笔|后续|下次的|下次)\s*)?(?:扣款|付款|支付|交易)"
_MUTATION_OBJECT = (r"(?:自动续费(?:开关)?|续费(?:开关)?|订阅|订单(?:\s*DEMO-[0-9]{4})?|"
                    r"DEMO-[0-9]{4}|交易|设备|账户|账号|银行卡|手机号|收货地址|订单地址|" + _NEXT_CHARGE + r")")
_BA_MUTATION = (rf"(?:把|将)\s*{_MUTATION_OWNER}{_MUTATION_OBJECT}\s*"
                r"(?:给)?(?:关闭|关掉|停止|取消|冻结|解绑|修改|更改)")
_PRONOUN_MUTATION = r"(?:把|将)\s*它\s*(?:给)?(?:关闭|关掉|停止|取消|冻结|解绑)"
_MUTATION = (
    r"(?:申请|发起|办理|执行|提交)?(?:退款|退货)|"
    rf"取消{_MUTATION_OWNER}(?:订单|自动续费|续费|订阅|DEMO-[0-9]{{4}})|"
    rf"(?:关闭|关掉|停止|开启|打开){_MUTATION_OWNER}(?:自动续费|续费|订阅)|"
    rf"冻结{_MUTATION_OWNER}(?:{_NEXT_CHARGE}|账户|账号)|"
    rf"解绑{_MUTATION_OWNER}(?:设备|账户|账号|银行卡|手机号)|"
    rf"(?:修改|更改){_MUTATION_OWNER}(?:收货地址|订单地址)|{_BA_MUTATION}"
)
_MUTATION_DIRECTIVE = r"(?:(?:请|麻烦|现在|立即|直接|再|替我|帮我|给我|我要|我想|我需要|我希望)\s*)+"
_MUTATION_DECLINE = re.compile(
    r"(?:不需要|不用|不要|不想|不必|无需|无须|先别|暂不|不再|别|并非|不是)"
    r"\s*(?:(?:再|现在|立即|直接|替我|帮我|给我|执行|办理|申请|发起|提交)\s*){0,5}"
    rf"(?:{_MUTATION}|{_PRONOUN_MUTATION})", re.I)
_MUTATION_EXPLAIN = re.compile(
    r"(?:怎么|如何|怎样|能不能|可不可以|是否可以|能否|是什么意思|规则|政策|流程|条件|步骤|材料|"
    r"手续费|时效|多久|什么时候|哪天|什么状态|什么进度|现在的状态|当前状态)")
_MUTATION_PAST = re.compile(
    rf"(?:我|客服|系统|平台|商家|对方|他|她)?\s*(?:已经|已|昨天|此前|之前|早就)"
    rf"\s*(?:(?:替我|帮我|给我|办理|申请|发起|完成|成功|曾经)\s*){{0,3}}(?:{_MUTATION}|{_PRONOUN_MUTATION})")
_MUTATION_CONTEXT = re.compile(
    r"自动续费|续费|订阅|扣款|订单|设备|账户|账号|交易|\b(?:auto[- ]?renewal|subscription|order|charge|account|transaction|device)\b", re.I)
_MUTATION_PARTICLE = r"(?:了|掉|吧|一下|下)?"
_EN_MUTATION = (r"(?:(?:refund|cancel|freeze|unlink)\s+(?:(?:my|this|the|next)\s+){0,2}"
                r"(?:order|payment|transaction|account|device|subscription|charge)|"
                r"(?:turn|switch)\s+off\s+(?:(?:my|this|the)\s+)?(?:auto[- ]?renewal|subscription))\b")


def _mutation_text(message):
    # Quoting a known object is not quoting an instruction. All complete quoted
    # commands remain untrusted data and are removed, including their objects.
    def quoted(match):
        inner = match.group()[1:-1].strip()
        return inner if re.fullmatch(_MUTATION_OBJECT, inner, re.I) else " "
    text = _QUOTED.sub(quoted, message)
    # A new explicit request after contrast/sequence has its own negation and
    # explanation scope. Bare coordinated actions retain the preceding scope:
    # "不要取消并冻结" is not turned into a positive freeze request.
    return re.sub(
        r"但是|不过|而是|(?<!不)但|然后|随后|接着|"
        r"(?:并且|而且|并)(?=\s*(?:请|替我|帮我|解释|说明|介绍|告诉))|"
        r"\b(?:but|then)\b|\band\b(?=\s*(?:please|now|explain|tell|describe)\b)",
        "；", text, flags=re.I)


def requests_business_mutation(message):
    """Signal an explicit execution request so a read cannot pretend to fulfil it.

    This finite guard never authorizes an operation. Unknown phrasing remains
    outside its coverage; the runtime still has no writing/refund tool and may
    never claim an operation completed on the strength of this boolean.
    """
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return False
    text = _mutation_text(message)
    business_context = False
    for fragment in re.findall(r"[^，,。！？!?；;\n]+[！？!?]?", text):
        questioned = fragment.rstrip().endswith(("?", "？"))
        clause = fragment.strip().rstrip("！？!?").strip()
        past = _MUTATION_PAST.search(clause)
        if past:
            # A completed action does not swallow a separately explicit new
            # request in the same message, even when punctuation is omitted.
            tail = clause[past.end():]
            follow = re.search(r"(?:现在|接下来|再)\s*(?=请|帮我|替我)", tail)
            if follow is None:
                continue
            business_context |= bool(_MUTATION_CONTEXT.search(clause[:past.end()]))
            clause = tail[follow.start():]
        if not clause or _MUTATION_DECLINE.search(clause) or _REPORTED.search(clause):
            continue
        earlier_context = business_context
        business_context |= bool(_MUTATION_CONTEXT.search(clause))
        # A question about performing an action is not an order to perform it.
        if _MUTATION_EXPLAIN.search(clause):
            continue
        pronoun = re.search(_PRONOUN_MUTATION, clause)
        antecedent = earlier_context or bool(pronoun and _MUTATION_CONTEXT.search(clause[:pronoun.start()]))
        action = rf"(?:{_MUTATION}|{_PRONOUN_MUTATION})" if antecedent else rf"(?:{_MUTATION})"
        match = re.search(rf"{_MUTATION_DIRECTIVE}{action}", clause)
        if match:
            remainder = clause[match.end():].strip()
            if not re.match(r"(?:的)?(?:状态|进度|记录|金额|到账|查询|核对|说明|分析|规则|政策)", remainder):
                return True
        # An unqualified imperative is accepted only as the complete clause.
        if not questioned and re.fullmatch(rf"{action}{_MUTATION_PARTICLE}", clause):
            return True
        if not re.search(r"\b(?:how|when|what|whether|status|policy|rules|explain|translate|don't|do not)\b", clause, re.I):
            if re.search(rf"\b(?:please|now|immediately)\s+{_EN_MUTATION}", clause, re.I):
                return True
            english_pronoun = re.search(r"\b(?:please|now|immediately)\s+(?:turn|switch)\s+it\s+off\b", clause, re.I)
            if english_pronoun and (earlier_context or _MUTATION_CONTEXT.search(clause[:english_pronoun.start()])):
                return True
    return False
