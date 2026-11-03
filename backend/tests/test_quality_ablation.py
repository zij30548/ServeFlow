"""Design/scoring checks with supplied controlled records, never model results."""
from copy import deepcopy
import json
import socket
import stat
from types import SimpleNamespace

import pytest

from evaluation import quality_ablation as module


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("quality design tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    def git(args, **kwargs):
        assert args == ["git", "rev-parse", "HEAD"]
        return SimpleNamespace(stdout="a" * 40 + "\n")
    monkeypatch.setattr(module.subprocess, "run", git)


def prepared(tmp_path):
    tool = module.QualityAblation(tmp_path / "new-design")
    tool.prepare()
    return tool


def results(records=()):
    return {"schema_version": "serveflow-controlled-ablation-results-v1", "evidence_kind": "controlled-fixture",
            "design_sha256": module._hash(module.load_manifest()), "records": list(records)}


def completed(arm, identity):
    case = next(c for c in module.load_manifest()["cases"] if c["case_id"] == identity)
    value = module.blank_record(arm, identity)
    value.update(outcome="completed", execution_complete=True, source_supported=True,
                 dimensions=dict.fromkeys(module.DIMENSIONS, True))
    for flag, key in (("supported_completion", "supported_task_complete"),
                      ("tool_task", "tool_task_correct"), ("boundary_handling", "boundary_handled_correctly")):
        value[key] = True if case[flag] else None
    return value


def test_exact_four_arms_original_questions_gold_hashes_and_scope_denominators():
    raw_before = {name: (module.BACKEND / name).read_bytes() for name in module.SOURCE_FILES}
    design = module.load_manifest()
    assert tuple(c["case_id"] for c in design["cases"]) == module.CASE_IDS
    assert sum(len(c["turns"]) for c in design["cases"]) == 13
    assert design["denominators"] == dict(correct_handling=12, supported_completion=8, tool_tasks=10, boundary_handling=4)
    assert design["arms"] == {
        "A": dict(force_fresh_read=False, typed_order_projection=False),
        "B": dict(force_fresh_read=True, typed_order_projection=False),
        "C": dict(force_fresh_read=False, typed_order_projection=True),
        "D": dict(force_fresh_read=True, typed_order_projection=True)}
    for case in design["cases"]:
        source = json.loads(raw_before[case["source_file"]])
        original = next(c for c in source["cases"] if c["id"] == case["source_case_id"])
        assert case["expected_sha256"] == module._hash(original["expected"])
        assert case["references_sha256"] == module._hash({"references": original["references"]})
        assert case["turns"] == (original["turns"] if case["case_id"].startswith("e") else [original["input"]["message"]])
        assert case["trusted_customer_id"] == (original["customer_id"] if case["case_id"].startswith("e") else original["input"]["customer_id"])
    assert raw_before == {name: (module.BACKEND / name).read_bytes() for name in module.SOURCE_FILES}


def test_counterbalanced_order_is_preplanned_not_chosen_after_results():
    order = module.load_manifest()["order"]
    assert [row["case_id"] for row in order] == list(module.CASE_IDS)
    for position in range(4):
        assert {arm: sum(row["arm_order"][position] == arm for row in order) for arm in module.ARMS} == dict.fromkeys(module.ARMS, 3)


def test_prepare_is_private_zero_send_and_explicitly_not_a_live_payload_freeze(tmp_path):
    tool = prepared(tmp_path)
    value = tool.verify()
    assert value["authorized_model_requests"] == value["actual_model_requests"] == 0
    assert value["runtime_arms_ready"] is value["live_payloads_frozen"] is False
    assert value["git_head"] == "a" * 40
    assert set(value["artifacts"]) == {*module.SOURCE_FILES, "evaluation/quality_ablation.py",
        "evaluation/benchmark/quality-ablation-v1.json", "tests/test_quality_ablation.py"}
    assert stat.S_IMODE(tool.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((tool.root / "design-freeze.json").stat().st_mode) == 0o600
    assert list(tool.root.iterdir()) == [tool.root / "design-freeze.json"]


def test_reprepare_never_overwrites_existing_design(tmp_path):
    tool = prepared(tmp_path)
    before = (tool.root / "design-freeze.json").read_bytes()
    with pytest.raises(module.QualityAblationError, match="design_root_already_exists"):
        tool.prepare()
    assert (tool.root / "design-freeze.json").read_bytes() == before


@pytest.mark.parametrize("name", ["task-business-v4", "task-business-v5-observation", "task-business-v4/new-plan"])
def test_historical_directories_cannot_become_output_roots(name):
    with pytest.raises(module.QualityAblationError, match="fixed_design_root_required"):
        module.QualityAblation(module.BACKEND / "data/benchmarks" / name)


def test_symlink_output_roots_are_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual)
    with pytest.raises(module.QualityAblationError, match="unsafe_design_root"):
        module.QualityAblation(link)


@pytest.mark.parametrize("change", ["unknown", "runtime", "authorization", "question", "boolean_as_integer"])
def test_modified_design_is_rejected_instead_of_accepting_new_labels(tmp_path, monkeypatch, change):
    value = module.load_manifest()
    if change == "unknown": value["unapproved"] = True
    elif change == "runtime": value["runtime_arms_ready"] = True
    elif change == "authorization": value["authorized_model_requests"] = 1248
    elif change == "question": value["cases"][0]["turns"][0] = "changed after review"
    else: value["runtime_arms_ready"] = 0
    path = tmp_path / "design.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(module, "MANIFEST", path)
    with pytest.raises(module.QualityAblationError, match="design_or_original_cases_changed"):
        module.load_manifest()


def test_frozen_source_change_stops_verify_without_rewriting_freeze(tmp_path, monkeypatch):
    tool = prepared(tmp_path)
    before = (tool.root / "design-freeze.json").read_bytes()
    monkeypatch.setattr(tool, "_artifacts", lambda: {"changed.py": "f" * 64})
    with pytest.raises(module.QualityAblationError, match="frozen_design_changed"):
        tool.verify()
    assert before == (tool.root / "design-freeze.json").read_bytes()


def test_empty_template_retains_all_denominators_and_cannot_claim_an_uplift(tmp_path):
    tool = prepared(tmp_path)
    template = tool.template()
    assert len(template["records"]) == 48
    report = tool.summary(template)
    assert report["planned_arm_cases"] == 48 and report["planned_arm_turns"] == 52
    assert report["runtime_arms_ready"] is report["model_quality_evidence"] is False
    assert report["actual_model_requests"] == report["authorized_model_requests"] == 0
    for arm in report["arms"].values():
        for field, denominator in report["denominators_per_arm"].items():
            assert arm[field]["denominator"] == denominator
            assert arm[field]["counts"]["not_executed"] == denominator
    assert all(pair["full_plan_delta_percentage_points"] is None for pair in report["pairs"].values())


def test_missing_records_stay_unexecuted_not_dropped_from_plan():
    report = module.summarize(results([completed("D", "e02")]))
    d = report["arms"]["D"]
    assert d["correct_handling"]["numerator"] == 1 and d["correct_handling"]["denominator"] == 12
    assert d["correct_handling"]["counts"]["not_executed"] == 11
    assert report["pairs"]["A_vs_D"]["unpaired_unknown_or_unexecuted"] == 12
    assert report["pairs"]["A_vs_D"]["full_plan_delta_percentage_points"] is None


def test_same_semantic_gate_allows_natural_summary_and_does_not_reward_mechanism_flags():
    natural, typed = completed("A", "e02"), completed("D", "e02")
    natural["mechanism_checks"] = dict.fromkeys(module.MECHANISMS, False)
    typed["mechanism_checks"] = dict.fromkeys(module.MECHANISMS, True)
    report = module.summarize(results([natural, typed]))
    assert report["arms"]["A"]["correct_handling"]["numerator"] == 1
    assert report["arms"]["D"]["correct_handling"]["numerator"] == 1
    assert report["pairs"]["A_vs_D"]["both_passed"] == 1
    typed["dimensions"]["coverage"] = False
    bad = module.summarize(results([natural, typed]))
    assert bad["arms"]["D"]["correct_handling"]["numerator"] == 0
    assert bad["pairs"]["A_vs_D"]["only_left_passed"] == 1


def test_boundary_refusal_and_handoff_preparation_are_not_supported_task_completion():
    report = module.summarize(results([completed("D", "t09"), completed("D", "t10"), completed("D", "t12")]))
    arm = report["arms"]["D"]
    assert arm["boundary_handling"]["numerator"] == 3 and arm["boundary_handling"]["denominator"] == 4
    assert arm["supported_completion"]["numerator"] == 0 and arm["supported_completion"]["denominator"] == 8
    assert arm["tool_tasks"]["numerator"] == 1 and arm["tool_tasks"]["denominator"] == 10


def test_mixed_case_cannot_pass_when_refusal_component_is_wrong():
    value = completed("D", "t09")
    value["boundary_handled_correctly"] = False
    report = module.summarize(results([value]))
    assert report["arms"]["D"]["correct_handling"]["numerator"] == 0
    assert report["arms"]["D"]["boundary_handling"]["numerator"] == 0


def test_known_bad_tool_trajectory_is_not_replaced_with_handler_success_count():
    value = completed("D", "e04")
    value["tool_task_correct"] = False
    report = module.summarize(results([value]))
    assert report["arms"]["D"]["tool_tasks"]["numerator"] == 0
    value["tool_task_correct"] = True
    value["dimensions"]["tool_permissions"] = False
    report = module.summarize(results([value]))
    assert report["arms"]["D"]["tool_tasks"]["numerator"] == 0


def test_structurally_valid_mechanism_and_full_review_cannot_replace_missing_actual_support():
    value = completed("D", "e02")
    value["mechanism_checks"] = dict.fromkeys(module.MECHANISMS, True)
    value["source_supported"] = False
    report = module.summarize(results([value]))
    assert report["arms"]["D"]["correct_handling"]["counts"]["failed"] == 1
    assert report["arms"]["D"]["supported_completion"]["numerator"] == 0
    assert report["arms"]["D"]["tool_tasks"]["numerator"] == 1


def test_all_planned_pairs_have_four_quadrants_and_actual_percentage_point_difference():
    records = [completed(arm, identity) for arm in ("A", "D") for identity in module.CASE_IDS]
    by_id = {(r["arm"], r["case_id"]): r for r in records}
    by_id["A", "e01"]["dimensions"]["coverage"] = False
    by_id["D", "e02"]["dimensions"]["coverage"] = False
    for arm in ("A", "D"):
        by_id[arm, "e03"]["dimensions"]["coverage"] = False
    report = module.summarize(results(records))
    pair = report["pairs"]["A_vs_D"]
    assert pair == {"planned_denominator": 12, "both_passed": 9, "only_left_passed": 1,
        "only_right_passed": 1, "neither_passed": 1, "unpaired_unknown_or_unexecuted": 0,
        "paired_terminal_count": 12, "full_plan_delta_percentage_points": 0.0}
    assert report["model_quality_evidence"] is False


@pytest.mark.parametrize("change", ["duplicate", "extra", "bad_arm", "bad_case", "live", "not_executed_score", "wrong_scope", "unknown_score", "integer_bool", "missing_review", "wrong_design"])
def test_invalid_or_unproven_results_are_rejected(change):
    value = results([completed("A", "e02")])
    record = value["records"][0]
    if change == "duplicate": value["records"].append(deepcopy(record))
    elif change == "extra": record["raw_model_response"] = "do not import"
    elif change == "bad_arm": record["arm"] = "any_runtime_switch"
    elif change == "bad_case": record["case_id"] = "new_case_after_results"
    elif change == "live": value["evidence_kind"] = "live-model"
    elif change == "not_executed_score": record["outcome"] = "not_executed"
    elif change == "wrong_scope": record["boundary_handled_correctly"] = True
    elif change == "unknown_score": record["outcome"] = "unknown"
    elif change == "integer_bool": record["dimensions"]["coverage"] = 1
    elif change == "missing_review": record["dimensions"]["coverage"] = None
    else: value["design_sha256"] = "f" * 64
    with pytest.raises(module.QualityAblationError):
        module.summarize(value)


@pytest.mark.parametrize("raw,code", [(b'{"x":1,"x":2}', "duplicate_json_key"),
    (b'{"x":NaN}', "nonfinite_json"), (b'{"x":Infinity}', "nonfinite_json"),
    (b'{', "invalid_json"), (b'x' * (module.MAX_JSON + 1), "json_size_limit")])
def test_bounded_strict_json_reader(tmp_path, raw, code):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(module.QualityAblationError, match=code):
        module._read(path)


def test_cli_has_no_paid_run_or_authorization_operation():
    with pytest.raises(SystemExit) as exc:
        module.main(["run"])
    assert exc.value.code == 2
