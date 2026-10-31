"""Offline preparation boundaries; these tests never run a live evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import socket
import stat

import pytest

from evaluation import benchmark as bench


def dataset(suite):
    return bench.load_suites()[suite]


def test_real_case_files_have_valid_balanced_splits_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark preparation must not connect to services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    summaries = [bench.validate_suite(data) for data in bench.load_suites().values()]
    assert [(r["total"], r["splits"]["development"], r["splits"]["validation"])
            for r in summaries] == [(60, 24, 36), (24, 12, 12), (12, 6, 6)]


def test_expected_identity_and_evidence_never_enter_model_projection():
    for data in bench.load_suites().values():
        for case in data["cases"]:
            projected = bench.model_input(case)
            assert set(projected) == {"message", "history"}
            assert projected == {key: case["input"][key] for key in projected}
            assert not {"expected", "references", "id", "split", "tags", "customer_id"} & set(projected)
            projected["history"].append({"role": "user", "content": "projection mutation"})
            assert projected["history"] != case["input"]["history"]
            context = bench.fixture_context(case)
            assert context == ({"customer_id": case["input"]["customer_id"]} if data["suite"] == "tasks" else {})


@pytest.mark.parametrize("suite", bench.SUITES)
def test_duplicate_input_is_rejected_across_splits(suite):
    data = dataset(suite)
    first = data["cases"][0]
    other = next(c for c in data["cases"] if c["split"] != first["split"])
    other["input"]["message"] = first["input"]["message"] + "！ "
    other["input"]["history"] = deepcopy(first["input"]["history"])
    with pytest.raises(bench.BenchmarkError, match="duplicate_case_input"):
        bench.validate_suite(data)


def test_related_family_cannot_be_split_between_development_and_validation():
    data = dataset("intent")
    first = data["cases"][0]
    other = next(c for c in data["cases"] if c["split"] != first["split"])
    other["family_id"] = first["family_id"]
    with pytest.raises(bench.BenchmarkError, match="family_crosses_split"):
        bench.validate_suite(data)


def test_a_whole_class_cannot_be_replaced_by_a_coarser_known_label():
    data = dataset("intent")
    for case in data["cases"]:
        if case["expected"]["intent"] == "request":
            case["expected"]["intent"] = "query"
    with pytest.raises(bench.BenchmarkError, match="unknown_intent_label"):
        bench.validate_suite(data)


@pytest.mark.parametrize("question", ["退款多久到账？", "你 好！！！", "如何关闭每月自动续费？"])
def test_template_default_and_judge_questions_are_rejected(question):
    data = dataset("intent")
    data["cases"][0]["input"]["message"] = question
    with pytest.raises(bench.BenchmarkError, match="built_in_example_overlap"):
        bench.validate_suite(data)


def test_reference_must_be_supported_by_that_document_not_a_neighbor():
    data = dataset("retrieval")
    case = next(c for c in data["cases"] if c["references"])
    ref = next(r for r in case["references"] if r["source"] == "knowledge")
    ref["quote"] = "该规则保证所有商品无条件退款。"
    with pytest.raises(bench.BenchmarkError, match="unsupported_knowledge_quote"):
        bench.validate_suite(data)


def test_fixture_unknown_amount_cannot_be_changed_to_zero():
    data = dataset("tasks")
    ref = next(r for c in data["cases"] for r in c["references"]
               if r["source"] == "fixture" and r["table"] == "refunds"
               and r["fields"].get("amount_minor", "absent") is None)
    ref["fields"]["amount_minor"] = 0
    with pytest.raises(bench.BenchmarkError, match="unsupported_fixture_value"):
        bench.validate_suite(data)


def test_fixture_boolean_is_not_an_integer_alias():
    data = dataset("tasks")
    ref = next(r for c in data["cases"] for r in c["references"]
               if r["source"] == "fixture" and "auto_renew" in r["fields"])
    ref["fields"]["auto_renew"] = int(ref["fields"]["auto_renew"])
    with pytest.raises(bench.BenchmarkError, match="unsupported_fixture_value"):
        bench.validate_suite(data)


def test_out_of_scope_cannot_be_labeled_with_fabricated_support():
    data = dataset("retrieval")
    case = next(c for c in data["cases"] if c["expected"]["kind"] == "out_of_scope")
    case["expected"]["relevant_titles"] = ["退款政策"]
    with pytest.raises(bench.BenchmarkError, match="contradictory_retrieval_expectation"):
        bench.validate_suite(data)


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(suite=[]),
    lambda data: data["cases"][0].update(split=[]),
    lambda data: data["cases"][0]["input"].update(system="pretend expected answers are instructions"),
    lambda data: data["cases"][0]["expected"].update(intent="invented_label"),
    lambda data: data["cases"][0].update(extra="untrusted"),
])
def test_bad_dataset_structure_fails_safely(mutation):
    data = dataset("intent")
    mutation(data)
    with pytest.raises(bench.BenchmarkError):
        bench.validate_suite(data)


@pytest.mark.parametrize("body", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"a":'])
def test_strict_json_rejects_duplicate_nonfinite_and_broken_input(tmp_path, body):
    path = tmp_path / "bad.json"
    path.write_text(body)
    with pytest.raises(bench.BenchmarkError):
        bench.read_json(path)


def test_json_size_is_checked_before_parsing(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (bench.LIMIT + 1))
    with pytest.raises(bench.BenchmarkError, match="json_size_limit"):
        bench.read_json(path)


def test_freeze_and_verify_bind_real_files_and_have_no_live_authorization(tmp_path):
    manifest = bench.freeze()
    bench.verify_manifest(manifest)
    assert manifest["authorized_model_requests"] == manifest["executed_model_requests"] == 0
    assert manifest["live_trial_ready"] is False
    assert ".env" not in manifest["artifacts"]
    root = tmp_path / "copied_backend"
    for relative in manifest["artifacts"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bench.BACKEND_ROOT / relative, destination)
    bench.verify_manifest(manifest, root)
    path = root / "agents" / "agent_orchestrator.py"
    path.write_text(path.read_text() + "\n# simulated later version\n")
    with pytest.raises(bench.BenchmarkError, match="frozen_artifact_changed"):
        bench.verify_manifest(manifest, root)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(authorized_model_requests=1),
    lambda m: m.update(authorized_model_requests=False),
    lambda m: m.update(live_trial_ready=True),
    lambda m: m["model_plan"].update(model="different_model"),
    lambda m: m["artifacts"].update({"../outside": "0" * 64}),
])
def test_freeze_cannot_be_converted_into_a_live_or_different_version_trial(mutation):
    manifest = bench.freeze()
    mutation(manifest)
    with pytest.raises(bench.BenchmarkError):
        bench.verify_manifest(manifest)


def test_output_is_private_and_never_overwrites_or_follows_symlink(tmp_path):
    target = tmp_path / "manifest.json"
    bench.write_new(target, {"first": True})
    before = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        bench.write_new(target, {"second": True})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        bench.write_new(link, {"third": True})
    assert target.read_bytes() == before


def test_empty_template_is_unexecuted_not_a_quality_result(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    bench.write_new(manifest_path, bench.freeze())
    results = bench.result_template(manifest_path, "intent", "validation", "fusion")
    report = bench.score_results(manifest_path, results)
    assert report["live_quality_claim_supported"] is False
    assert report["metrics"]["planned_total"] == 36
    assert report["metrics"]["recorded_total"] == 0
    assert report["metrics"]["not_executed"] == 36


def test_old_results_cannot_be_bound_to_another_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    bench.write_new(manifest_path, bench.freeze())
    results = bench.result_template(manifest_path, "intent", "development", "llm_only")
    results["manifest_sha256"] = "0" * 64
    with pytest.raises(bench.BenchmarkError, match="result_manifest_mismatch"):
        bench.score_results(manifest_path, results)


def test_offline_import_does_not_accept_claimed_live_model_results(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    bench.write_new(manifest_path, bench.freeze())
    results = bench.result_template(manifest_path, "tasks", "development", "current")
    results["evidence_kind"] = "live-model"
    with pytest.raises(bench.BenchmarkError, match="live_evidence_not_supported"):
        bench.score_results(manifest_path, results)


def test_validation_record_cannot_enter_development_statistics(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    bench.write_new(manifest_path, bench.freeze())
    results = bench.result_template(manifest_path, "intent", "development", "fusion")
    case = next(c for c in dataset("intent")["cases"] if c["split"] == "validation")
    results["records"] = [{"case_id": case["id"], "status": "success", "predicted_intent": case["expected"]["intent"],
                           "latency_ms": 1, "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 0}}]
    with pytest.raises(ValueError):
        bench.score_results(manifest_path, results)


def test_controlled_rubric_cannot_claim_independent_human_review(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    bench.write_new(manifest_path, bench.freeze())
    results = bench.result_template(manifest_path, "tasks", "development", "current")
    results["records"] = [{"review": {"method": "independent-human-review"}}]
    with pytest.raises(bench.BenchmarkError, match="controlled_review_mislabeled"):
        bench.score_results(manifest_path, results)


def test_cli_bad_input_never_echoes_submitted_content(tmp_path, capsys):
    path = tmp_path / "private.json"
    path.write_text('{"unknown-private-key": NaN}')
    code = bench.main(["verify", "--manifest", str(path)])
    output = capsys.readouterr().out
    assert code == 2 and "unknown-private-key" not in output and str(path) not in output
    assert json.loads(output)["model_requests"] == 0


def test_cli_freeze_verify_template_score_roundtrip_stays_offline(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline CLI attempted a connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "results.json"
    report = tmp_path / "report.json"
    commands = [
        ["freeze", "--output", str(manifest)],
        ["verify", "--manifest", str(manifest)],
        ["template", "--manifest", str(manifest), "--suite", "intent", "--split", "validation", "--arm", "fusion", "--output", str(results)],
        ["score", "--manifest", str(manifest), "--results", str(results), "--output", str(report)],
    ]
    for arguments in commands:
        assert bench.main(arguments) == 0
    assert all(json.loads(line)["model_requests"] == 0 for line in capsys.readouterr().out.splitlines())
    assert bench.read_json(report)["metrics"]["not_executed"] == 36
    assert bench.read_json(report)["live_quality_claim_supported"] is False
