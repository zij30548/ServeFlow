"""Finite read-only scope grammar: pure function, no model or service calls.

These development examples test explicit unsupported action requests, not a
general-purpose semantic classifier. A positive signal grants no permission and
does not claim the requested action was performed. Unknown references stay
unresolved instead of being assigned an object from an earlier conversation.
"""
import pytest

from core.order_evidence import requests_business_mutation


@pytest.mark.parametrize("message", [
    "请关闭自动续费。",
    "请帮我关掉自动续费。",
    "请关闭这笔订单的自动续费。",
    "帮我把自动续费关掉。",
    "请把这个订阅取消。",
    "请把“自动续费”关掉。",
    "请冻结下一笔扣款。",
    "请冻结下一笔付款。",
    "先核对当前订阅的自动续费，再替我把它关掉。",
    "核对自动续费状态；然后请把它关掉。",
    "请取消订阅，并冻结下一笔扣款。",
    "不要申请退款，但请帮我关闭自动续费。",
    "不要退款而是请帮我关闭自动续费。",
    "先说明怎么取消订阅，然后请替我取消订阅。",
    "客服说不必关闭但现在请替我关闭自动续费。",
    "我已申请退款，但请帮我取消订阅。",
    "Please cancel my subscription.",
    "Please turn off auto-renewal.",
    "Please freeze the next charge.",
    "Please cancel my subscription and explain how refunds work.",
    "Do not refund this order, but please cancel my subscription.",
    "How can I cancel? Please cancel my subscription now.",
    "Check the subscription auto-renewal, then please turn it off.",
    "I do not want a refund, but please turn off auto-renewal.",
    "先查当前续费状态，然后替我把它关掉。",
    "我已关闭自动续费现在请冻结下一笔扣款。",
    "客服已经退款现在请取消订阅。",
    "我已经核对订单现在请关闭自动续费。",
])
def test_explicit_action_with_local_scope_is_an_unfinished_readonly_request(message):
    assert requests_business_mutation(message) is True


@pytest.mark.parametrize("message", [
    "怎么关闭自动续费？",
    "如何申请退款？",
    "请告诉我怎么冻结交易。",
    "我想知道怎么取消订阅。",
    "能不能关闭自动续费？",
    "关闭自动续费需要什么条件？",
    "请查询关闭自动续费后的权益状态。",
    "退款什么时候到账？",
    "我已经关闭自动续费，请核对权益。",
    "自动续费已经关了，请核对当前状态。",
    "我昨天取消了订阅，现在只查询退款进度。",
    "客服已经帮我申请退款了，帮我核对记录。",
    "请确认是否已经取消订阅。",
    "不要帮我关闭自动续费。",
    "请不要冻结下一笔扣款。",
    "先别把自动续费关掉。",
    "不用取消订阅，只查询状态。",
    "我不想关闭自动续费。",
    "客服说“请帮我关闭自动续费”，这句话是什么意思？",
    "请解释“冻结下一笔扣款”的意思。",
    "文档中写道“立即取消订阅”。",
    "有人说请帮我关闭自动续费，但我只想查询。",
    "“请关闭自动续费”",
    "请把“帮我关闭自动续费”翻译成英文。",
    "帮我把它关掉。",
    "请把它冻结。",
    "我有个台灯，帮我把它关掉。",
    "How do I cancel my subscription?",
    "Please tell me how to cancel my subscription.",
    "Please check my refund status.",
    "I already cancelled my subscription; please check the record.",
    "Please do not cancel my subscription.",
    "Do not turn off auto-renewal.",
    "The document says 'please freeze the payment'; what does it mean?",
    "Please translate 'cancel my subscription'.",
    "Please turn it off.",
    "Could you tell me whether auto-renewal is already off?",
    "Please do not cancel my subscription but explain the refund policy.",
    "我已关闭自动续费现在请核对记录。",
    "我已关闭自动续费现在请不要冻结下一笔扣款。",
    "续费状态是什么？",
])
def test_explanation_past_negation_quotation_and_unresolved_reference_are_not_actions(message):
    assert requests_business_mutation(message) is False


@pytest.mark.parametrize("message", [None, "", " ", True, 1, b"cancel subscription", [], {}, "x" * 2001])
def test_invalid_or_empty_message_does_not_create_a_mutation_request(message):
    assert requests_business_mutation(message) is False
