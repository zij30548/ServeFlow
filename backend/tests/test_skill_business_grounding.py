"""实际 Skill 加载、角色隔离及引用的离线结构回归，不评估模型语义。

旧 e02 的真实回复在待审核、退款金额未知的记录后，扩写了手续费、
原路退款和未来通知，并遗漏记录时间。旧输出和评分保持原样。
本文件只确认修正规则实际被完整加载、注入且不扩大工具权限。
真正的旧输出回归需由订单证据集成测试验证：将旧扩写作为不可信
Actor/Composer 正文，保留实际演示记录、查询时间、更新时间和 null，
确定性渲染不能采纳该正文。这里不把“包含某句提示”当成语义改善，
也不调用模型重新评分或把受控结果记为真实模型质量提升。
"""
from pathlib import Path
import re
import socket

import pytest

from agents.agent_orchestrator import BillingAgent, GeneralAgent, Request, TechnicalAgent
from core.skill_loader import SkillManager


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
ROLES = {
    "billing": (BillingAgent, "账单退款处理规范", "billing_support/SKILL.md", "退款"),
    "general": (GeneralAgent, "通用客服接待规范", "general_customer_service/SKILL.md", "订单"),
    "technical": (TechnicalAgent, "技术支持处理规范", "technical_support/SKILL.md", "接口报错"),
}


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Skill structure tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def manager():
    value = SkillManager(str(SKILLS_ROOT), max_prompt_chars=5000)
    value.load()
    assert value.errors == []
    return value


class NoModelClient:
    def __init__(self):
        self.messages = self
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("Loading Skills must not send a model request")


def test_real_skill_inventory_remains_three_enabled_scoped_roles(manager):
    assert len(manager.skills) == len(ROLES)
    actual = {}
    for skill in manager.skills:
        assert skill.enabled is True
        assert len(skill.agents) == 1
        role = skill.agents[0]
        assert role in ROLES and role not in actual
        _, title, relative_path, _ = ROLES[role]
        assert skill.name == title
        assert Path(skill.path) == SKILLS_ROOT / relative_path
        assert skill.description and skill.keywords
        assert skill.content and skill.content.startswith("## 角色定位")
        actual[role] = skill
    assert set(actual) == set(ROLES)


@pytest.mark.parametrize("role", ROLES)
def test_actual_role_prompt_includes_complete_skill_without_truncation(manager, role):
    cls, title, _, _ = ROLES[role]
    skill = next(skill for skill in manager.skills if skill.agents == [role])
    # SkillManager has a separate 3200-character per-Skill bound. No newly
    # added boundary, existing role duty or final example may silently vanish.
    assert len(skill.content) <= 3200
    block = skill.to_prompt_block()
    assert block.endswith(skill.content)
    message = "订单退款与接口报错，请分别核对并说明。"
    prompt = manager.prompt_for(message, role)
    assert block in prompt
    for other_role, (_, other_title, _, _) in ROLES.items():
        if other_role != role:
            assert other_title not in prompt
    client = NoModelClient()
    agent = cls(client, "controlled-model", skill_manager=manager)
    request = Request(message=message, user_id="controlled", conv_id="skill-test")
    assert block in agent._build_system_prompt(request)
    assert client.calls == []


@pytest.mark.parametrize("role", ROLES)
def test_named_tool_references_resolve_within_existing_role_scope(manager, role):
    cls, _, _, _ = ROLES[role]
    skill = next(skill for skill in manager.skills if skill.agents == [role])
    # Check code-form tool references, not prose claims about model compliance.
    references = set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", skill.content))
    assert references <= set(cls.profile.tool_scope)
    if role in {"general", "billing"}:
        assert references == {"read_demo_order", "search_knowledge_base"}
    client = NoModelClient()
    tools = cls(client, "controlled-model", skill_manager=manager).get_tools()
    assert ("read_demo_order" in tools) is (role != "technical")
    assert client.calls == []


@pytest.mark.parametrize("role", ROLES)
def test_role_skill_is_still_selected_by_existing_business_keywords(manager, role):
    _, _, _, message = ROLES[role]
    matched = [skill for skill in manager.skills if skill.matches(message, role)]
    assert len(matched) == 1 and matched[0].agents == [role]
    assert manager.prompt_for("zz-no-business-keyword-zz", role) == ""


def test_reload_is_repeatable_without_duplicate_injection(manager):
    before = manager.summary()
    before_prompts = {role: manager.prompt_for(message, role)
                      for role, (_, _, _, message) in ROLES.items()}
    manager.reload()
    assert manager.summary() == before
    assert {role: manager.prompt_for(message, role)
            for role, (_, _, _, message) in ROLES.items()} == before_prompts
