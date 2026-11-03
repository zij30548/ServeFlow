"""Pure policy requirements and bounded source projection, with no I/O.

These finite request rules do not prove semantic relevance, coverage or current
applicability. Source text is displayed as data, never executed or rewritten.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import re

MAX_SOURCES = 3
MAX_RESULT_ROWS = 20
MAX_MERGE_INPUTS = 3
MAX_INTERNAL_UNITS = 60
MAX_BATCH_BYTES = 65536
MAX_CONTENT_CHARS = 2000
_STATUSES = frozenset({'sources', 'no_materials', 'unavailable', 'unverified', 'invalid'})
_ROW_FIELDS = frozenset({'document_id', 'title', 'content', 'version', 'chunk', 'score'})
_TOP_FIELDS = frozenset({'success', 'query', 'top_k', 'results', 'reranked', 'error'})
_ID = re.compile(r'(?<![A-Za-z0-9_])DEMO-[0-9]{4}(?![A-Za-z0-9_])', re.I)
_QUOTED = re.compile(r'“[^”\n]*”|「[^」\n]*」|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
_BUSINESS = r'(?:配送|物流|快递|包裹|发货|送达|运费|邮费|免邮|退款|退货|退换|售后|保修|会员|订阅|数字权益|数字服务|数字订阅|优惠|发票|账户安全)'
_POLICY = r'(?:政策|规则|规定|条款|条件|限制|范围|标准|时效|门槛|统一.{0,6}(?:天数|时间|费用)|例外)'
_BUSINESS_EN = r'\b(?:shipping|delivery|refunds?|returns?|warranty|membership|subscriptions?|invoices?|freight)\b'
_POLICY_EN = r'\b(?:polic(?:y|ies)|rules?|terms|conditions?|coverage|fees?|thresholds?|exceptions?|timeframes?)\b'
_POLICY_DECLINE_EN = re.compile(
    r"\b(?:not (?:asking|inquiring)(?: about| for)?|not about|(?:do not|don't|dont) (?:want|need|ask)(?: about| for)?|no need (?:for|to))\s+"
    r'(?:(?:a|an|the|your|to|know|about|ask)\s+){0,4}'
    r'(?:(?:shipping|delivery|refunds?|returns?|warranty|membership|subscriptions?|invoices?|freight)\s+)?'
    r'(?:policy|policies|rules?|terms|conditions?|coverage|thresholds?|exceptions?|timeframes?)\b')
_POLICY_DECLINE = re.compile(
    rf'(?<!不是)(?<!并非)(?<!不)(?:并不是|不是|并非|不需要|不想|不用|不要|无需|无须|先别|暂不|别|不问|不查|不咨询|不讨论|不讲)\s*'
    rf'(?:(?:再|先|现在|给我|帮我|问|查|查询|咨询|讨论|了解|讲|说|核对|解释)\s*){{0,4}}'
    rf'(?:(?:{_BUSINESS})\s*(?:的)?\s*)?{_POLICY}')
_GENERIC_DECLINE = re.compile(r'(?:不用|不要|不必|无需|别|不需要)(?:再|继续)?(?:讲|说|解释|查|查询|讨论)(?:政策|规则|条件|时效|门槛)')


def _label(value, maximum):
    return (type(value) is str and 1 <= len(value) <= maximum and bool(value.strip())
            and not any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _valid_unit(unit):
    return (type(unit) is PolicyUnit and _label(unit.document_id, 128)
            and _label(unit.title, 160) and _label(unit.version, 128)
            and type(unit.content) is str and 1 <= len(unit.content) <= MAX_CONTENT_CHARS
            and bool(unit.content.strip()) and type(unit.chunk) is int and unit.chunk >= 0
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in unit.content))


def _unit_dict(unit):
    return {k: getattr(unit, k) for k in ('document_id', 'title', 'content', 'version', 'chunk')}


def _within_bytes(value):
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')) <= MAX_BATCH_BYTES
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        return False


@dataclass(frozen=True, slots=True)
class PolicyUnit:
    document_id: str
    title: str
    content: str
    version: str
    chunk: int

    def __post_init__(self):
        if not _valid_unit(self):
            raise ValueError('invalid_policy_unit')


@dataclass(frozen=True, slots=True)
class PolicyEvidence:
    status: str
    units: tuple[PolicyUnit, ...] = ()
    mixed: bool = False
    omitted: int = 0

    def __post_init__(self):
        if (type(self.status) is not str or self.status not in _STATUSES or type(self.units) is not tuple
                or type(self.mixed) is not bool or type(self.omitted) is not int or self.omitted < 0
                or len(self.units) > MAX_INTERNAL_UNITS or not all(_valid_unit(u) for u in self.units)
                or (self.status == 'sources') != bool(self.units)
                or self.status != 'sources' and self.omitted != 0):
            raise ValueError('invalid_policy_evidence')
        expected = max(0, len(self.units) - MAX_SOURCES)
        if self.omitted not in (0, expected) or not _within_bytes([_unit_dict(u) for u in self.units]):
            raise ValueError('invalid_policy_evidence')
        object.__setattr__(self, 'omitted', expected)


def _invalid(mixed=False):
    return PolicyEvidence('invalid', mixed=mixed if type(mixed) is bool else False)


def _valid_evidence(evidence):
    try:
        return (type(evidence) is PolicyEvidence and type(evidence.status) is str and evidence.status in _STATUSES
                and type(evidence.mixed) is bool and type(evidence.units) is tuple
                and len(evidence.units) <= MAX_INTERNAL_UNITS
                and all(_valid_unit(u) for u in evidence.units)
                and (evidence.status == 'sources') == bool(evidence.units)
                and type(evidence.omitted) is int and evidence.omitted == max(0, len(evidence.units) - MAX_SOURCES)
                and _within_bytes([_unit_dict(u) for u in evidence.units]))
    except (AttributeError, TypeError, ValueError):
        return False


def has_record_question(message):
    """A concrete-record signal is not permission to access that record."""
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return False
    text = _QUOTED.sub(' ', message) if re.search(r'这个词|术语|字样|怎么翻译|是什么意思', message) else message
    record = r'(?:订单|包裹|退款|付款|扣款|实付|权益|订阅|会员)'
    action = r'(?:查一下|查询|查查|查|查看|核对|核实|检查|看看|跟踪|追踪)'
    # Declining one lookup does not erase a positive request later. Negative
    # payment/activation symptoms are not lookup exclusions.
    text = re.sub(
        rf'(?<!不是)(?<!并非)(?<!不)(?:不需要|不用|不要|不想|无需|别|不查询|不查看|不查|不核对|不跟踪)\s*'
        rf'(?:{action}\s*)?(?:我的|这个|这笔|该)?{record}(?:\s*DEMO-[0-9]{{4}})?'
        r'(?:的)?(?:状态|进度|详情|记录|金额|单号)?', ' ', text, flags=re.I)
    text = re.sub(
        r"\b(?:do not|don't|dont)\s+(?:track|check|look up|query|verify|find)\s+"
        r'(?:(?:my|this|that|the|an?)\s+){0,2}(?:order|refund|shipment|payment|subscription|entitlement|membership)\b'
        r'(?:\s+(?:status|details|record))?', ' ', text, flags=re.I)
    # These are only signals for an unverified business operation. They grant
    # no cancellation/address/refund tool or execution permission.
    operation = (
        r'(?:取消(?:我的|这笔|这个|当前|该)?订单|'
        r'(?:修改|更改)(?:我的|这笔订单的|该订单的)?收货地址|'
        r'申请(?:我的|(?:这笔|这个|该)?订单(?:的)?)?退款)'
    )
    operation_text = _QUOTED.sub(' ', text)
    operation_text = re.sub(
        rf'(?<!不是)(?<!并非)(?<!不)(?:不需要|不想|不要|不用|不必|无需|别|先别|暂不|不)'
        rf'(?:(?:再|现在|帮我|给我|先)\s*){{0,3}}{operation}', ' ', operation_text)
    directive = r'(?:请(?!问|解释|说明|告诉|教|说)|帮我|给我|麻烦|我(?:要|想|需要|希望))'
    link = r'(?:(?!如何|怎么|怎样|方法|流程|规则|政策|了解|说明|解释|告诉|知道|教)[^，。！？；\n]){0,8}'
    if re.search(rf'{directive}{link}{operation}(?!(?:的)?(?:规则|政策|方法|流程|条件|步骤|意思))', operation_text):
        return True
    if _ID.search(text):
        return True
    owns = r'(?:我的|我这|这笔|这个|这单|该|刚才那(?:个|笔)|刚才的)'
    query = r'(?:状态|进度|到哪|多少|哪天|到账|单号|发货|开通|激活|查|核对|核实|查询)'
    if re.search(rf'{owns}[^，。！？；\n]{{0,8}}{record}[^，。！？；\n]{{0,16}}{query}', text):
        return True
    if re.search(r'\b(?:my|this|that)\s+(?:order|refund|shipment|payment|subscription|entitlement)\b[^.;!?\n]{0,24}\b(?:status|where|when|tracking|charged|paid|delivery)\b', text, re.I):
        return True
    if re.search(rf'{action}[^，。！？；\n]{{0,8}}{record}[^，。！？；\n]{{0,12}}(?:状态|进度|详情|记录|金额|单号|到账|到货)', text):
        return True
    if re.search(rf'{action}(?:一下|下|我的|这个|这笔|该)?{record}\s*(?:[，。！？；\n]|$)', text):
        return True
    if re.search(rf'{owns}[^，。！？；\n]{{0,8}}{record}[^，。！？；\n]{{0,10}}(?:是否|能否|可以|可不可以|能不能|符合|满足|适用)[^，。！？；\n]{{0,6}}(?:退款|退货|退换|免邮|保修|取消|开通)', text):
        return True
    if re.search(r'\b(?:track|check|look up|query|verify|find)\s+(?:(?:my|this|that|the|an?|current|previous)\s+){0,2}(?:order|refund|shipment|payment|subscription|entitlement|membership)\b', text, re.I):
        return True
    return bool(re.search(rf'(?:核对|核实|查询|检查)[^，。！？；\n]{{0,8}}(?:这笔|我的|该){record}', text))


def _active_text(message):
    text = message.strip().lower()
    # A quoted label alone is not a request. Actual questions about a quoted
    # business field (你们的“免邮门槛”是多少) still require evidence.
    metalinguistic = r'这个词|术语|字样|怎么翻译|是什么意思|\b(?:term|word|translate|meaning|means?)\b'
    reported = r'(?:客服|对方|以前|示例).{0,8}(?:说|写|提到)|\b(?:support|agent|example|someone|they)\s+(?:said|wrote|mentioned)\b'
    if re.search(metalinguistic, text):
        text = _QUOTED.sub(' ', text)
    elif re.search(reported, text):
        outside = _QUOTED.sub(' ', text)
        # A report plus an actual request to verify that policy still needs
        # sources; a quoted label alone or a switch to order tracking does not.
        if not (re.search(rf'{_POLICY}|{_POLICY_EN}', outside) and re.search(r'[？?]|什么|哪些|是否|\b(?:is|what|which|how)\b', outside)):
            text = outside
    return _POLICY_DECLINE_EN.sub(' ', _POLICY_DECLINE.sub(' ', text))


def _direct_policy_request(message):
    text = _active_text(message)
    if re.search(rf'{_BUSINESS}|{_BUSINESS_EN}', text) and re.search(rf'{_POLICY}|{_POLICY_EN}', text):
        return True
    if has_record_question(message):
        return False
    combinations = (
        r'(?:设备|配件|组合订单|商品)[^，。！？；\n]{0,14}(?:一起|分开|分别|合并)[^，。！？；\n]{0,8}(?:发货|送达|到|寄)',
        r'(?:数字订阅|数字服务|数字权益)[^，。！？；\n]{0,14}(?:快递|物流|单号)',
        r'(?:配送|物流|快递|退款|退货)[^，。！？；\n]{0,12}(?:多久|几天|多长时间|多长|什么时候)',
        r'(?:一般|通常|你们|本店)[^，。！？；\n]{0,8}(?:怎么|如何)[^，。！？；\n]{0,8}(?:退款|退货|配送|保修)',
        r'(?:是否|能否|可以|怎样|怎么)[^，。！？；\n]{0,8}(?:免邮|退换|保修)',
        r'(?:运费|邮费)[^，。！？；\n]{0,12}(?:多少|怎么(?:算|收)|如何(?:计算|收取)|价格|费用)',
        r'(?:会员|订阅)[^，。！？；\n]{0,12}(?:多少钱|价格|费用|收费|价位)',
        r'(?:退货|退款|退换|售后|保修)[^，。！？；\n]{0,16}(?:准备|提供|出示|提交)[^，。！？；\n]{0,8}(?:材料|凭证|证明|单据|资料)',
        r'(?:怎么|如何|怎样)[^，。！？；\n]{0,8}(?:取消订单|修改收货地址|申请退款)',
    )
    return any(re.search(pattern, text) for pattern in combinations)


def requires_policy_evidence(message, history=None):
    """Finite request checks; history can extend only the nearest user topic."""
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return False
    if _direct_policy_request(message):
        return True
    if _POLICY_DECLINE.search(message) or _POLICY_DECLINE_EN.search(message.lower()) or _GENERIC_DECLINE.search(message) or has_record_question(message):
        return False
    follow = r'条件|限制|范围|时效|门槛|例外|期限|多久|几天|\b(?:conditions?|limits?|coverage|thresholds?|exceptions?|timeframes?)\b'
    question = r'呢|吗|[？?]|什么|哪些|如何|怎么|多久|几天|说明|说说|介绍|\b(?:what|which|how|when|explain)\b'
    if not re.search(follow, _active_text(message)) or not re.search(question, message, re.I):
        return False
    if type(history) not in (list, tuple):
        return False
    for item in reversed(history[-6:]):
        if type(item) is dict and item.get('role') == 'user':
            prior = item.get('content')
            return type(prior) is str and 1 <= len(prior) <= 2000 and _direct_policy_request(prior)
    return False


def extract_policy_evidence(result, mixed=False):
    """Validate the full result, including rows that will not fit on screen."""
    if type(mixed) is not bool or type(result) is not dict or set(result) - _TOP_FIELDS or not {'success', 'results'} <= set(result):
        return _invalid(mixed)
    if type(result['success']) is not bool or type(result['results']) is not list or len(result['results']) > MAX_RESULT_ROWS:
        return _invalid(mixed)
    if ('query' in result and (type(result['query']) is not str or len(result['query']) > 2000)
            or 'top_k' in result and (type(result['top_k']) is not int or not 1 <= result['top_k'] <= 20)
            or 'reranked' in result and type(result['reranked']) is not bool
            or 'error' in result and result['error'] is not None and (type(result['error']) is not str or len(result['error']) > 2000)):
        return _invalid(mixed)
    if not result['success']:
        return _invalid(mixed) if result['results'] or not _within_bytes(result) else PolicyEvidence('unavailable', mixed=mixed)
    if result.get('error'):
        return _invalid(mixed)
    units = []
    try:
        for row in result['results']:
            if (type(row) is not dict or set(row) != _ROW_FIELDS or type(row['score']) not in (int, float)
                    or not math.isfinite(row['score'])):
                return _invalid(mixed)
            units.append(PolicyUnit(**{k: row[k] for k in _ROW_FIELDS if k != 'score'}))
        # Bound individual values before allocating a canonical copy of the
        # entire batch. A malformed huge body is rejected by PolicyUnit first.
        if not _within_bytes(result):
            return _invalid(mixed)
        if not units:
            return PolicyEvidence('no_materials', mixed=mixed)
        return merge_policy_evidence([PolicyEvidence('sources', tuple(units), mixed=mixed)], mixed=mixed)
    except (TypeError, ValueError, OverflowError):
        return _invalid(mixed)


def merge_policy_evidence(evidences, mixed=False):
    """Keep all bounded full units internally, so cross-role dedup is exact."""
    if type(mixed) is not bool or type(evidences) not in (list, tuple) or len(evidences) > MAX_MERGE_INPUTS:
        return _invalid(mixed)
    if not all(_valid_evidence(item) for item in evidences):
        return _invalid(mixed)
    actual_mixed = mixed or any(item.mixed for item in evidences)
    if any(item.status == 'invalid' for item in evidences):
        return _invalid(actual_mixed)
    unique = {}
    for evidence in evidences:
        for unit in evidence.units:
            key = (unit.document_id, unit.version, unit.chunk)
            previous = unique.get(key)
            if previous is not None and previous != unit:
                return _invalid(actual_mixed)
            unique[key] = unit
    if unique:
        try:
            return PolicyEvidence('sources', tuple(unique.values()), mixed=actual_mixed)
        except ValueError:
            return _invalid(actual_mixed)
    status = next((name for name in ('unavailable', 'unverified', 'no_materials') if any(item.status == name for item in evidences)), 'unverified')
    return PolicyEvidence(status, mixed=actual_mixed)


def render_policy_evidence(evidence):
    """Plain text for Vue interpolation. Source bodies are retained verbatim."""
    if not _valid_evidence(evidence):
        evidence = _invalid()
    elif evidence.status == 'sources':
        # A caller can construct typed objects directly. Recheck full-identity
        # conflicts and deduplicate before deriving the displayed omission count.
        evidence = merge_policy_evidence([evidence], mixed=evidence.mixed)
    messages = {
        'no_materials': '未找到相关资料。当前知识库没有提供可用的政策依据，本次不生成政策结论。',
        'unavailable': '资料检索暂不可用，本次无法核实政策。请稍后主动重试或通过已核实的服务渠道查询。',
        'unverified': '本次尚未取得可核验的知识库资料，不生成政策结论。',
        'invalid': '资料格式或来源信息校验未通过，本次不展示政策结论。',
    }
    if evidence.status == 'sources':
        lines = ['知识库资料片段', '相关资料不等于问题已完整回答，也不保证当前适用或构成现行承诺。']
        for index, unit in enumerate(evidence.units[:MAX_SOURCES], 1):
            lines.append(f'资料 {index}：{unit.title}\n版本：{unit.version[:12]} · 片段索引：{unit.chunk}\n{unit.content}')
        if evidence.omitted:
            lines.append(f'另有{evidence.omitted}个不同资料片段未展开。')
        output = '\n\n'.join(lines)
    else:
        output = messages[evidence.status]
    if evidence.mixed:
        output += '\n\n本次仅展示政策资料；订单或交易部分尚未核实，不能据此判断具体订单状态、金额或到账时间。'
    return output
