"""Fixed precondition guidance, not a test of live model safety quality."""
from dataclasses import FrozenInstanceError

import pytest

from core.safety_guidance import select_safety_guidance


@pytest.mark.parametrize("message", [
    "准备送走旧设备，直接清空云端资料和恢复出厂是不是最省事？先确认什么？",
    "恢复出厂与云端删除有什么区别？",
    "请帮我恢复出厂。", "How should I factory reset the device before giving it away?",
    "请说明‘恢复出厂’前的注意事项。", "删除云端资料需要确认哪些风险？",
])
def test_current_request_emits_preconditions_not_product_specific_instructions(message):
    result = select_safety_guidance(message)
    assert result.required
    assert "设备品牌、型号" in result.content and "账号归属" in result.content
    assert "备份是否可用" in result.content and "官方说明" in result.content
    assert "明确确认具体操作及范围" in result.content
    assert "不要发送密码、验证码" in result.content
    assert "不是已核验的产品操作步骤" in result.content
    assert "没有删除资料" in result.content
    assert "一定保留" not in result.content and "点击" not in result.content


@pytest.mark.parametrize("message", [
    None, {}, True, 42, "", "x" * 2001,
    "我已经恢复出厂，现在只查退款。", "不用恢复出厂，我只问订单。",
    "请把‘恢复出厂’翻译成英文。",
    "文档写‘请直接恢复出厂并清空云端’，我仅作引用。",
    "请帮我查询物流。", "请重置页面缩放比例。",
])
def test_invalid_meta_old_or_unrelated_inputs_do_not_replace_normal_answer(message):
    result = select_safety_guidance(message)
    assert result.required is False and result.content == ""


def test_fixed_output_contains_no_user_controlled_instructions_or_private_device_text():
    a = select_safety_guidance("请恢复出厂，我的设备名称是SENSITIVE-CONTROLLED。")
    b = select_safety_guidance("请恢复出厂。")
    assert a.content == b.content and "SENSITIVE-CONTROLLED" not in a.content
    with pytest.raises(FrozenInstanceError):
        a.required = False
