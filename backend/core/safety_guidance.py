"""Finite preconditions for destructive advice, not product-specific instructions.

The current user message can ask about a reset without granting permission to
delete data. No account, device, cloud state or official procedure is verified
by this pure helper; it emits a fixed clarification checklist, not action steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_TOPIC = re.compile(
    r"恢复出厂|出厂(?:设置|重置)|重置设备|设备重置|清空(?:全部)?云端|删除(?:全部)?云端|"
    r"清空(?:全部)?(?:账号|账户)资料|删除(?:全部)?(?:账号|账户)数据|"
    r"\bfactory\s+reset\b|\b(?:erase|delete|clear)\s+(?:all\s+)?cloud\s+(?:data|files)\b", re.I)
_QUOTED = re.compile(r'“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|"[^"\n]*"|`[^`\n]*`|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
_META = re.compile(r"翻译|读音|怎么读|仅作引用|只是引用|\btranslate|\bpronounce", re.I)
_ASK = re.compile(r"确认|核对|怎么|如何|是否|是不是|该先|先做|风险|影响|注意|需要|区别|不要混|请|帮我|替我|直接|想|打算|考虑|"
                  r"\b(?:please|how|should|can|check|confirm|risk|difference)\b", re.I)
_DECLINED = re.compile(r"(?:不用|不想|不需要|无需|不讨论|别讲|不要讲).{0,8}(?:恢复出厂|重置|删除|清空)|"
                       r"\b(?:do not|don't|no need to)\b.{0,16}(?:reset|erase|delete|clear)", re.I)
_DONE = re.compile(r"(?:已经|已|早就).{0,8}(?:恢复出厂|重置|删除|清空)|"
                  r"\b(?:already|previously)\b.{0,16}(?:reset|erased|deleted|cleared)", re.I)
_RESET_CHECKLIST = (
    "设备重置与云端资料删除是不同范围的操作，不能据此假定会一起发生，"
    "也不能保证其中一种不会影响另一种；当前设备和官方指引尚未核实。\n"
    "在采取任何操作前，请先确认：\n"
    "- 设备品牌、型号、系统版本及账号归属；只提供脱敏信息，不要发送密码、验证码或远程控制码。\n"
    "- 你要保留、转移或删除哪些本地资料、云端资料与共享权限，以及备份是否可用。\n"
    "- 对应产品官方说明对本地重置、云端删除、账号解绑和共享权限的分别影响；不能套用其他设备的步骤。\n"
    "- 核对影响后，由账号所有者明确确认具体操作及范围，再自行通过已核实的官方入口处理。\n"
    "以上只是操作前的安全核对清单，不是已核验的产品操作步骤；"
    "本助手没有删除资料、解绑账号或执行恢复出厂设置。"
)


@dataclass(frozen=True)
class SafetyGuidance:
    required: bool = False

    @property
    def content(self) -> str:
        return _RESET_CHECKLIST if self.required else ""


def select_safety_guidance(message) -> SafetyGuidance:
    """Detect current explicit reset/data-erasure advice, never old model text.

    Quoted instructions, translations and merely completed historical actions
    do not establish a new request. A standalone quoted topic name is allowed
    when the user asks about it; it still grants no execution authority.
    """
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return SafetyGuidance()

    def quoted(match):
        inner = match.group()[1:-1].strip()
        return inner if _TOPIC.fullmatch(inner) else " "

    text = _QUOTED.sub(quoted, message)
    for clause in re.split(r"[，,。！？!?；;\n]", text):
        if not _TOPIC.search(clause) or _META.search(clause):
            continue
        if _DECLINED.search(clause):
            continue
        if _DONE.search(clause) and not _ASK.search(clause):
            continue
        if _ASK.search(clause):
            return SafetyGuidance(True)
    return SafetyGuidance()
