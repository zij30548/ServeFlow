"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import math
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.model_client import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.provider_options import nonthinking_options
from core.safe_diagnostics import record_llm_failure
from core.model_deadline import await_model

logger = logging.getLogger(__name__)

# Explicit policy identity for future freezes; old trial protocols stay intact.
INTENT_POLICY_VERSION = "three-way-intent-v2-scoped-refinement"


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款/退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付/扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录认证故障
    TECHNICAL_CRASH = "technical_crash"  # 崩溃/错误码
    HUMAN_HANDOFF = "human_handoff"      # 转人工
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    embedding_source: str = "local-ngram"


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单有没有发货？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "退货退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    LLM 使用现有 Messages 接口；模板向量独立选择本地算法或远程服务。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs, max_retries=0)
        self.model     = model
        self.threshold = confidence_threshold
        from core.intent_embedding import IntentEmbedding
        self._embedding = IntentEmbedding()
        self._embedding_enabled = True
        self._cache: Dict[str, IntentResult] = {}
        self._cache_expiries: Dict[str, float] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache and time.monotonic() < self._cache_expiries.get(key, 0):
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            embedding_source=emb.get("source", "local-ngram"),
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
                self._cache_expiries.pop(k, None)
        self._cache[key] = result
        self._cache_expiries[key] = time.monotonic() + (60 if result.embedding_source == "local-ngram" and self._embedding.enabled else 300)
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._embedding.remote_templates.clear()
            self._embedding.local_templates.clear()
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            self._cache_expiries.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

粗细类别规则：
- query 下：订单受理、制作、关闭或履约状态为 order_status；包裹运输、配送和到货为 logistics。
- billing 下：退货、退款申请和退款进度为 refund；票据开具、获取和查验为 invoice；实际支付或扣款差额、重复交易、付款失败为 payment_issue。
- technical 下：登录、认证、会话失效为 technical_login；其他页面报错、白屏、闪退为 technical_crash。
- account 下：盗号、未授权变更、遗失设备会话撤销为 account_security。
- escalation 下：明确要求人工、主管或负责人本人接管并继续沟通处理为 human_handoff；仅提及这些角色不代表要求转接。
能确定子类时输出子类；只有现有信息无法区分子类时保留粗类，不凭空补全。
区分排除话题与故障现象：用户明确否认咨询某事时，不以被否认的词分类；没有收到退款、不能登录等仍是对应事项的故障。
结合历史解析指代，以当前实际诉求为准；不要把旧话题或引用文字当成当前请求。示例中的粗类只说明业务域，不覆盖上述细类规则。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        started = time.monotonic()
        phase = "intent.transport"
        try:
            resp = await await_model(self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                **nonthinking_options(self.client, self.model),
            ), timeout_s=max(0.1, float(os.getenv("ECHOMIND_INTENT_TIMEOUT_S", "8"))))
            phase = "intent.parse"
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            phase = "intent.validate"
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("意图 confidence 必须是 0 到 1 的有限数字")
            return data
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            record_llm_failure(logger, phase, ex, (time.monotonic() - started) * 1000)
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            active = self._active_intent_text(message)
            if not active.strip():
                return {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "local-ngram"}
            templates, msg_vec, source = await self._embedding.vectors(
                active, _TEMPLATES, self._local_embedding)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in templates.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": min(1.0, max(0.0, best_score)), "source": source}
        except Exception:
            # Never format provider bodies, credentials or user text here.
            logger.warning("Embedding 识别失败，未取得可用模板匹配")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = self._active_intent_text(message)
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工"],
            IntentCategory.ORDER_STATUS: ["订单状态", "发货了吗", "处理到哪", "order status"],
            IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单", "delivery", "shipping"],
            IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["重复扣款", "多扣", "支付失败", "扣费", "payment failed"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "401", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
        }
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        # Strong, scoped combinations may justify a subtype. Bare nouns still
        # contribute their original weak vote, but have no refinement authority.
        strong = self._specific_evidence(msg)
        if len(strong) == 1:
            best_cat = next(iter(strong))
            _, score = self._best_pattern_match(msg, {best_cat: specific_patterns[best_cat]})
            return {"intent": best_cat, "confidence": max(0.5, score),
                    "refinement_allowed": True}

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score,
                    "refinement_allowed": False}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score,
                "refinement_allowed": False}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        cleaned = []
        for source in (llm, emb, pat):
            value = source.get("confidence")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                source = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
            cleaned.append(source)
        llm, emb, pat = cleaned
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        group_scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            weighted_score = w * conf
            scores[cat] = scores.get(cat, 0.0) + weighted_score
            group = _INTENT_GROUPS.get(cat, cat)
            group_scores[group] = group_scores.get(group, 0.0) + weighted_score

        best_group = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_group]
        best = max(
            (cat for cat in scores if _INTENT_GROUPS.get(cat, cat) == best_group),
            key=scores.get,
        )
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        # Local character similarity alone cannot silently replace a winning
        # coarse LLM label with a subtype. Keep it unless scoped evidence agrees.
        llm_intent = llm.get("intent", IntentCategory.OTHER)
        if (llm_intent in _GENERIC_INTENTS and llm_intent == best_group
                and best in _SPECIFIC_INTENTS):
            best = llm_intent
        if (best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS
                and _INTENT_GROUPS.get(pat_intent, pat_intent) == best_group
                and pat.get("refinement_allowed") is True and pat_conf >= 0.5):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        msg = re.sub(r"(?:不(?:是|用|需要)?|无需|无须|非|并不|没那么|暂不)\s*(?:紧急|立刻|马上|尽快|快点)", "", msg)
        msg = re.sub(r"\b(?:not|isn't|is not|no)\s+(?:an?\s+)?(?:emergency|urgent|asap|now)\b", "", msg)
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", msg) if kw.isascii() else kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message), "intent_policy_version": INTENT_POLICY_VERSION}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _active_intent_text(message: str) -> str:
        """Mask explicit topic/request exclusions, preserving nearby assertions.

        This intentionally covers a finite grammar, not general negation. In
        particular, negative symptoms such as 没收到退款 / 无法登录 stay intact.
        The original full message still goes to the LLM for context reasoning.
        """
        text = IntentRecognizer._clean_text(message).lower()
        topics = (r"退货退款|退款|退货|发票|开票|抬头|税号|物流|快递|配送|运单|"
                  r"订单状态|订单进度|转人工|人工客服|人工|主管|负责人|经理|"
                  r"登录故障|登录失败|不能登录|无法登录|登录|报错|崩溃|闪退|"
                  r"扣款|支付失败|支付问题|支付|账户安全|账号安全")
        topic_group = rf"(?:{topics})(?:\s*(?:或|和|及|、|与|/)\s*(?:{topics}))*"
        verbs = r"(?:(?:在|再|先|现在|马上|立即|给我|帮我|要|想|需要|让|找|转接|转给|联系|安排|问|咨询|查询|查|核对|讨论|说|办理|申请|开|处理|要求|关于)\s*){0,4}"
        exclusion = (r"(?<!不是)(?<!并非)(?<!不)"
                     r"(?:并不是|不是|并非|不需要|不想|不用|不要|不必|无需|无须|先别|暂不|别)\s*")
        text = re.sub(exclusion + verbs + topic_group, " ", text)
        # Directly declining handoff is a request exclusion, unlike 不能登录.
        text = re.sub(r"(?<!不是)(?<!并非)(?<!不)不(?:转人工|找人工|转接人工|找主管)", " ", text)
        english_topics = r"(?:invoices?|refunds?|shipping|delivery|order status|human (?:agents?|support)|supervisors?)"
        english_exclusion = (
            r"\b(?:not (?:asking|inquiring)(?: about| for)?|not interested in|"
            r"(?:do not|don't|dont) (?:want|need|request)|no need (?:for|to))\s+"
            r"(?:(?:to|a|an|the|any|ask|about|for|request|check|process|issue)\s+){0,4}"
        )
        text = re.sub(english_exclusion + english_topics + r"\b", " ", text)
        text = re.sub(
            r"\b(?:do not|don't|dont) (?:transfer|connect|handoff)\s+(?:me\s+)?(?:to\s+)?(?:a\s+|an\s+|the\s+)?"
            r"(?:human(?: agent| support)?|supervisor)\b", " ", text)
        return text

    @staticmethod
    def _specific_evidence(text: str) -> set[IntentCategory]:
        """Unambiguous lexical combinations eligible for same-group refinement.

        These are bounded heuristics, not calibrated probabilities. Competing
        strong subtopics deliberately withhold refinement authority.
        """
        clause = r"[^，。！？；\n]"
        directive = r"(?:请(?!问|解释|说明|告诉)|麻烦|帮我|给我|我(?:要|想|需要|希望)|希望|要求)"
        linking = r"(?:(?:帮我|给我|把(?:这段|本次|当前)?(?:会话|对话)|把我|由|让|直接|马上|立刻)\s*){0,3}"
        patterns = {
            IntentCategory.HUMAN_HANDOFF: [
                rf"{directive}\s*{linking}(?:转接|转给|交给|接通|找|联系|安排){clause}{{0,8}}(?:人工(?:客服|同事)?|真人|主管|负责人)",
                rf"{directive}\s*{linking}(?:人工(?:客服|同事)?|真人|主管|负责人){clause}{{0,8}}(?:接管|接着(?:沟通|处理)|继续(?:沟通|处理)|回访)",
                rf"(?:^|[，。！？；\n])\s*(?:{directive}\s*)?(?:转人工|找人工)(?:\s|[，。！？；\n]|$)",
                r"(?:^|[.;!?\n])\s*(?:please\s+|i (?:want|need) (?:you )?to\s+)?(?:transfer|connect|handoff)\b.{0,24}\b(?:human|supervisor|agent)\b",
            ],
            IntentCategory.PAYMENT_ISSUE: [
                rf"(?:支付|付款|扣款|交易)(?:通道)?{clause}{{0,12}}(?:失败|不可用|不成功|被拒|无法完成)",
                rf"(?:重复|多收|多扣){clause}{{0,8}}(?:款|钱|费|交易)",
                r"(?:实付|实扣|实际支付|实际扣款|支付凭证|支付金额|扣款金额|付款金额)[^。！？\n]{0,96}(?:差额|不一致|不符|对不上|金额不对|多扣|多收|比[^。！？\n]{0,16}[高低多少])",
                r"\b(?:payment failed|charged twice|duplicate charge|payment amount mismatch)\b",
            ],
            IntentCategory.REFUND: [
                rf"(?:申请|办理|查询|核对|检查|想|要|准备){clause}{{0,8}}(?:退款|退货)",
                rf"(?:退款|退货){clause}{{0,16}}(?:申请|进度|审核|到账|没收到|未收到|没到账|未到账|流程|规则|多久|什么时候)",
                r"\b(?:request|apply for|not received)\b.{0,12}\brefund\b|\brefund\b.{0,16}\b(?:pending|not received|not arrived)\b",
            ],
            IntentCategory.INVOICE: [
                rf"(?:开具|补开|重开|开|下载|获取|领取|查验|核验|重发|查询|修改|更改){clause}{{0,12}}(?:发票|抬头|税号)",
                rf"(?:发票|抬头|税号){clause}{{0,14}}(?:开具|下载|在哪|哪里|怎么|如何|没收到|收不到|真伪|查验|修改)",
                r"\b(?:issue|download|check|request)\b.{0,12}\binvoice\b",
            ],
            IntentCategory.ORDER_STATUS: [rf"(?:订单|预订|预购){clause}{{0,18}}(?:状态|进展|受理|制作|关闭|发货了吗|处理到哪)"],
            IntentCategory.LOGISTICS: [rf"(?:物流|快递|配送|运单|包裹){clause}{{0,16}}(?:在哪|哪里|到哪|多久|什么时候|进度|状态|不更新|延误|路线|派送)"],
            IntentCategory.ACCOUNT_SECURITY: [
                rf"(?:账号|账户){clause}{{0,12}}(?:被盗|盗用|冒用|异常登录|未授权)",
                rf"(?:丢失|丢了|遗失){clause}{{0,20}}(?:登录|会话)",
            ],
            IntentCategory.TECHNICAL_LOGIN: [
                r"不能登录|无法登录|登不上|\bcannot log in\b",
                rf"登录{clause}{{0,12}}(?:失败|报错|401|过期|失效|不可用)",
                rf"(?:验证码|登录链接|免密链接){clause}{{0,12}}(?:收不到|没收到|不可用|失败|失效)",
            ],
            IntentCategory.TECHNICAL_CRASH: [rf"(?:页面|应用|客户端|窗口|系统|界面){clause}{{0,12}}(?:崩溃|闪退|白屏|无响应|报错|500)"],
        }
        # Quoted handoff terminology and definitions are not a current request.
        # Keep quoted error messages intact for the other symptom classifiers.
        handoff_text = re.sub(r'“[^”\n]*”|「[^」\n]*」|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', " ", text)
        return {intent for intent, expressions in patterns.items()
                if any(re.search(expression, handoff_text if intent is IntentCategory.HUMAN_HANDOFF else text)
                       for expression in expressions)}

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if (
                bool(re.search(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", message))
                if kw.isascii() and any(char.isalpha() for char in kw) else kw in message))
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
