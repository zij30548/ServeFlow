import pytest

from core.order_evidence import resolve_order_id, source_bound_order_history


@pytest.mark.parametrize("reset", [
    "请清空查单目标。", "把刚才的订单上下文清掉，清空当前订单目标。",
    "接下来别用这笔订单了，先清空这次查单目标。",
    "清掉刚才的订单目标。", "Reset the order context, please.",
])
def test_explicit_target_reset_is_a_source_history_barrier(reset):
    rows = []
    for i, text in enumerate(["查DEMO-2001的付款", reset, "等一下", "请用短句"]):
        rows.append({"version": 1, "source": "user_message", "role": "user",
                     "content": text, "client_request_id": f"op-{i}",
                     "request_id": f"request-{i}", "status": "success"})
    history = source_bound_order_history(rows, [{"role": "user", "content": "请用短句"}])
    assert resolve_order_id("这笔订单付了多少？", history).status == "missing"


@pytest.mark.parametrize("pause", [
    "先不用继续查，我先忙一下。", "不要继续查询，等我回来。",
    "不要清空订单目标。", "Don't reset the order context.",
])
def test_temporary_pause_or_negated_clear_preserves_explicit_order(pause):
    history = [{"role": "user", "content": "查DEMO-2001的付款"},
               {"role": "user", "content": pause}]
    assert resolve_order_id("这笔订单付了多少？", history).order_id == "DEMO-2001"
