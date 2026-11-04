"""Question/fact/freeze integrity only. No semantic model quality claims."""
from collections import Counter
from copy import deepcopy
import json
import shutil
import socket
import stat
from types import SimpleNamespace

import pytest

from evaluation import benchmark as bench
from evaluation import prospective_validation as module


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline preparation cannot contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    def git(args, **kwargs):
        assert args == ["git", "rev-parse", "HEAD"]
        return SimpleNamespace(stdout="a"*40+"\n")
    monkeypatch.setattr(module.subprocess, "run", git)


def dataset(suite):
    return module._cases(bench.BACKEND_ROOT, suite)


def test_real_questions_have_source_validity_balance_and_no_exact_old_overlap():
    summaries = module.validate()
    assert summaries[0] == {"suite": "tasks", "total": 12, "supported_completion": 10,
        "tool_tasks": 11, "boundary_handling": 2, "domains": dict.fromkeys(module.DOMAINS, 2)}
    assert summaries[1] == {"suite": "retrieval", "total": 12,
        "kinds": {"answerable": 6, "insufficient": 3, "out_of_scope": 3},
        "evidence_denominator": 9, "out_of_scope_denominator": 3}


def test_gold_notes_and_case_identity_are_not_model_inputs_and_are_deep_copied():
    for suite in module.SUITES:
        for case in dataset(suite)["cases"]:
            before = deepcopy(case)
            model = bench.model_input(case)
            assert set(model) == {"message", "history"}
            assert model == {"message": case["input"]["message"], "history": []}
            assert bench.fixture_context(case) == ({"customer_id": case["input"]["customer_id"]} if suite == "tasks" else {})
            model["history"].append({"role": "user", "content": "local mutation"})
            assert case == before


@pytest.mark.parametrize("change", [
    lambda d: d.update(provenance="independent-human-blind-test"),
    lambda d: d["cases"][1].update(id="p01"),
    lambda d: d["cases"][0]["input"].update(system="gold"),
    lambda d: d["cases"][0]["input"].update(customer_id="client-forged-customer"),
    lambda d: d["cases"][0]["input"]["history"].append({"role": "assistant", "content": "fabricated answer"}),
    lambda d: d["cases"][0]["related_prior_cases"].append("nonexistent-old-question"),
    lambda d: d["cases"][0]["expected"].update(tool_task=False),
    lambda d: d["cases"][0]["expected"].update(boundary=1),
    lambda d: d["cases"][0]["expected"].update(expected_status="must-use-D"),
    lambda d: d["cases"][0]["expected"]["required_tools"].append("refund_money"),
])
def test_invalid_or_contradictory_cases_cannot_be_frozen(change):
    data = dataset("tasks")
    change(data)
    with pytest.raises(bench.BenchmarkError):
        module.validate_suite(data)


@pytest.mark.parametrize("suite", module.SUITES)
def test_old_question_and_new_duplicate_are_rejected(suite):
    data = dataset(suite)
    old = bench.read_json(bench.DATASET_ROOT / f"{suite}-v1.json")["cases"][0]
    data["cases"][0]["input"]["message"] = old["input"]["message"]+"！ "
    with pytest.raises(bench.BenchmarkError, match="prospective_question_overlap"):
        module.validate_suite(data)
    data = dataset(suite)
    data["cases"][1]["input"]["message"] = data["cases"][0]["input"]["message"]
    with pytest.raises(bench.BenchmarkError, match="prospective_question_overlap"):
        module.validate_suite(data)


def test_claimed_quote_must_belong_to_its_own_document():
    data = dataset("retrieval")
    data["cases"][0]["references"][0]["quote"] = "所有订单保证次日送达。"
    with pytest.raises(bench.BenchmarkError, match="unsupported_knowledge_quote"):
        module.validate_suite(data)


def test_unknown_money_and_boolean_identity_are_not_zero_integer_aliases():
    for mutate in (lambda ref: ref["fields"].update(amount_minor=0),
                   lambda ref: ref["fields"].update(amount_minor=False)):
        data = dataset("tasks")
        ref = next(r for c in data["cases"] for r in c["references"]
                   if r.get("table") == "refunds" and r["fields"].get("amount_minor", "absent") is None)
        mutate(ref)
        with pytest.raises(bench.BenchmarkError, match="unsupported_fixture_value"):
            module.validate_suite(data)


def test_missing_material_is_not_relabelled_as_supported_and_gold_requires_own_reference():
    data = dataset("retrieval")
    case = data["cases"][-1]
    case["expected"]["relevant_titles"] = ["配送说明"]
    with pytest.raises(bench.BenchmarkError, match="contradictory_prospective_retrieval"):
        module.validate_suite(data)
    data = dataset("retrieval")
    data["cases"][0]["references"].pop()
    with pytest.raises(bench.BenchmarkError, match="contradictory_prospective_retrieval"):
        module.validate_suite(data)


def test_prepare_freezes_every_case_both_orders_and_private_zero_send_scope(tmp_path):
    directory = tmp_path / "preparation"
    original = module.prepare(directory)
    assert module.verify(directory) == original
    assert original["authorized_model_requests"] == original["executed_model_requests"] == original["external_business_requests"] == 0
    assert original["live_trial_ready"] is original["live_payloads_frozen"] is False
    assert original["future_budget"]["task_physical_upper_bound"] is None
    assert len(original["plan"]) == 24
    for suite, arms in (("tasks", ["A", "D"]), ("retrieval", ["A", "B"])):
        plan = [row for row in original["plan"] if row["suite"] == suite]
        assert [row["case_id"] for row in plan] == [c["id"] for c in dataset(suite)["cases"]]
        assert dict(Counter(tuple(row["arm_order"]) for row in plan)) == {tuple(arms): 6, tuple(arms[::-1]): 6}
        assert all(set(row["conversation_input"]) == {"message", "history"} for row in plan)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "protocol.json").stat().st_mode) == 0o600
    assert list(directory.iterdir()) == [directory / "protocol.json"]
    old_bytes = (directory / "protocol.json").read_bytes()
    with pytest.raises(bench.BenchmarkError, match="already_exists"):
        module.prepare(directory)
    assert (directory / "protocol.json").read_bytes() == old_bytes


@pytest.mark.parametrize("change", [
    lambda d: d.update(authorized_model_requests=1),
    lambda d: d.update(executed_model_requests=False),
    lambda d: d.update(live_trial_ready=True),
    lambda d: d.update(live_payloads_frozen=True),
    lambda d: d["plan"][0]["conversation_input"].update(expected="secret gold"),
    lambda d: d["plan"][0]["arm_order"].reverse(),
    lambda d: d["design"]["tasks"]["not_primary"].clear(),
    lambda d: d["sources"].update({"agents/untracked-new-role.py": "0"*64}),
    lambda d: d["future_budget"].update(task_physical_upper_bound=1),
])
def test_manifest_cannot_be_converted_into_authorization_or_different_test(tmp_path, change):
    directory = tmp_path / "preparation"
    value = module.prepare(directory)
    change(value)
    (directory / "protocol.json").write_text(json.dumps(value))
    with pytest.raises(bench.BenchmarkError, match="prospective_frozen_design_changed"):
        module.verify(directory)


def test_real_code_source_and_new_file_changes_are_detected(tmp_path):
    root = tmp_path / "backend"
    for relative in module._sources(bench.BACKEND_ROOT):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bench.BACKEND_ROOT / relative, target)
    directory = tmp_path / "preparation"
    module.prepare(directory, root)
    source = root / "agents/agent_orchestrator.py"
    original = source.read_bytes()
    source.write_bytes(original + b"\n# changed after freeze\n")
    with pytest.raises(bench.BenchmarkError, match="prospective_frozen_design_changed"):
        module.verify(directory, root)
    source.write_bytes(original)
    module.verify(directory, root)
    (root / "agents/new_role.py").write_text("# new runtime source\n")
    with pytest.raises(bench.BenchmarkError, match="prospective_frozen_design_changed"):
        module.verify(directory, root)
    (root / "agents/new_role.py").unlink()
    external = tmp_path / "outside-source"
    external.mkdir()
    (external / "role.py").write_text("# source outside the frozen root\n")
    (root / "agents/extra").symlink_to(external, target_is_directory=True)
    with pytest.raises(bench.BenchmarkError, match="prospective_source_symlink"):
        module.verify(directory, root)


def test_bad_preflight_does_not_create_directory_and_links_are_not_followed(tmp_path, monkeypatch):
    directory = tmp_path / "preparation"
    def invalid(*args):
        raise bench.BenchmarkError("bad_source_before_writes")
    monkeypatch.setattr(module, "validate", invalid)
    with pytest.raises(bench.BenchmarkError, match="bad_source_before_writes"):
        module.prepare(directory)
    assert not directory.exists()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(bench.BenchmarkError, match="already_exists_or_symlink"):
        module.prepare(link)
    assert list(target.iterdir()) == []
