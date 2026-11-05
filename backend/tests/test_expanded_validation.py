"""Source-bound question preparation, with no live model or database clients."""
from copy import deepcopy
import json
import shutil
import socket

import pytest

from evaluation import benchmark as bench, expanded_validation as expanded


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline test attempted network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def copied_root(tmp_path):
    source = bench.BACKEND_ROOT
    names = {"knowledge/serveflow-demo-v2.json", "demo_business/fixtures.json"}
    names.update(str(p.relative_to(source)) for p in (source / "evaluation/benchmark").glob("*.json"))
    for suite in expanded.SUITES:
        data = bench.read_json(expanded._path(source, suite))
        names.update(expanded._sources(data, source))
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return tmp_path


def change(root, suite, edit):
    path = expanded._path(root, suite)
    data = bench.read_json(path)
    edit(data)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


def test_actual_questions_are_balanced_source_checked_and_offline():
    result = expanded.validate()
    assert [(r["suite"], r["total"], r["splits"]) for r in result] == [
        ("intents", 60, {"development": 24, "validation": 36}),
        ("tasks", 24, {"development": 8, "validation": 16}),
        ("retrieval", 24, {"development": 8, "validation": 16}),
    ]


def test_duplicate_old_question_rejected(copied_root):
    old = bench.read_json(copied_root / "evaluation/benchmark/intent-v1.json")["cases"][0]["input"]
    change(copied_root, "intents", lambda d: d["cases"][0].update(input=deepcopy(old)))
    with pytest.raises(bench.BenchmarkError, match="expanded_duplicate_question"):
        expanded.validate(copied_root)


def test_case_id_and_family_leakage_are_not_allowed(copied_root):
    path = expanded._path(copied_root, "tasks")
    original = bench.read_json(path)
    change(copied_root, "tasks", lambda d: d["cases"][1].update(id=d["cases"][0]["id"]))
    with pytest.raises(bench.BenchmarkError, match="expanded_duplicate_id"):
        expanded.validate(copied_root)
    path.write_text(json.dumps(original))
    change(copied_root, "tasks", lambda d: next(c for c in d["cases"] if c["split"] == "validation").update(family_id=d["cases"][0]["family_id"]))
    with pytest.raises(bench.BenchmarkError, match="expanded_family_leakage"):
        expanded.validate(copied_root)


def test_untrusted_identity_is_not_a_question_fixture(copied_root):
    change(copied_root, "tasks", lambda d: d["cases"][0]["input"].update(customer_id="model-supplied-customer"))
    with pytest.raises(bench.BenchmarkError, match="expanded_trusted_identity"):
        expanded.validate(copied_root)


@pytest.mark.parametrize("key", ["expected", "label", "references", "case_id"])
def test_evaluation_labels_cannot_be_added_to_model_input(copied_root, key):
    change(copied_root, "tasks", lambda d: d["cases"][0]["input"].update({key: "do not send"}))
    with pytest.raises(bench.BenchmarkError, match="expanded_input_fields"):
        expanded.validate(copied_root)


def test_model_projection_excludes_gold_identity_and_review_metadata():
    case = bench.read_json(expanded._path(bench.BACKEND_ROOT, "tasks"))["cases"][0]
    projected = bench.model_input(case)
    assert set(projected) == {"message", "history"}
    assert bench.fixture_context(case) == {"customer_id": case["input"]["customer_id"]}
    projected["message"] = "changed"
    assert case["input"]["message"] != "changed"


def test_unknown_label_rejected_without_running_classifier(copied_root):
    change(copied_root, "intents", lambda d: d["cases"][0]["expected"].update(intent="not-a-label"))
    with pytest.raises(bench.BenchmarkError, match="expanded_intent_label"):
        expanded.validate(copied_root)


def test_role_name_cannot_become_a_task_pass_condition(copied_root):
    change(copied_root, "tasks", lambda d: d["cases"][0]["execution_plan"].update(role_names_are_scoring_gate=True))
    with pytest.raises(bench.BenchmarkError, match="expanded_role_not_scoring_gate"):
        expanded.validate(copied_root)


def test_unsupported_amount_and_empty_collection_claim_rejected(copied_root):
    path = expanded._path(copied_root, "tasks")
    original = bench.read_json(path)
    def wrong_amount(data):
        ref = next(r for c in data["cases"] for r in c["references"]
                   if r["source"] == "fixture" and "total_amount_minor" in r["fields"])
        ref["fields"]["total_amount_minor"] += 1
    change(copied_root, "tasks", wrong_amount)
    with pytest.raises(bench.BenchmarkError, match="unsupported_fixture_value"):
        expanded.validate(copied_root)
    path.write_text(json.dumps(original))
    def wrong_empty(data):
        next(c for c in data["cases"] if c["collection_assertions"])["collection_assertions"][0]["expected_count"] = 1
    change(copied_root, "tasks", wrong_empty)
    with pytest.raises(bench.BenchmarkError, match="expanded_collection_count"):
        expanded.validate(copied_root)


def test_fake_quote_and_out_of_scope_gold_are_not_accepted(copied_root):
    path = expanded._path(copied_root, "retrieval")
    original = bench.read_json(path)
    change(copied_root, "retrieval", lambda d: d["cases"][0]["references"][0].update(quote="fabricated source assertion"))
    with pytest.raises(bench.BenchmarkError, match="unsupported_knowledge_quote"):
        expanded.validate(copied_root)
    path.write_text(json.dumps(original))
    def wrong_gold(data):
        negative = next(c for c in data["cases"] if c["expected"]["kind"] == "out_of_scope")
        positive = next(c for c in data["cases"] if c["expected"]["kind"] == "answerable")
        negative["references"] = deepcopy(positive["references"])
        negative["expected"]["relevant_titles"] = deepcopy(positive["expected"]["relevant_titles"])
    change(copied_root, "retrieval", wrong_gold)
    with pytest.raises(bench.BenchmarkError, match="expanded_retrieval_empty_gold"):
        expanded.validate(copied_root)


@pytest.mark.parametrize("value", ["../../.env", "/tmp/outside.json", "backend/data/private.json"])
def test_source_paths_are_not_arbitrary(copied_root, value):
    change(copied_root, "tasks", lambda d: d["sources"][0].update(path=value))
    with pytest.raises(bench.BenchmarkError, match="expanded_source_path"):
        expanded.validate(copied_root)


def test_source_hash_change_stops_preparation(copied_root):
    source = copied_root / "demo_business/fixtures.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(bench.BenchmarkError, match="expanded_source_changed"):
        expanded.validate(copied_root)


def test_unresolved_labels_cannot_be_frozen(copied_root):
    change(copied_root, "tasks", lambda d: d["cases"][0].update(review_status="needs_adjudication"))
    assert expanded.validate(copied_root)[1]["review_status"]["needs_adjudication"] == 1
    with pytest.raises(bench.BenchmarkError, match="expanded_unresolved_labels"):
        expanded.freeze(copied_root)


def test_freeze_is_question_only_and_cannot_overwrite(tmp_path):
    manifest = expanded.freeze()
    assert manifest["authorized_model_requests"] == manifest["executed_model_requests"] == 0
    assert manifest["sdk_payloads_frozen"] is manifest["live_trial_ready"] is False
    assert expanded.verify(manifest)["question_cases"] == 108
    altered = deepcopy(manifest)
    altered["authorized_model_requests"] = 1
    with pytest.raises(bench.BenchmarkError, match="expanded_freeze_changed"):
        expanded.verify(altered)
    target = tmp_path / "questions.json"
    bench.write_new(target, manifest)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        bench.write_new(target, manifest)
    assert target.read_bytes() == before


def test_existing_safe_json_reader_rejects_duplicates_and_nonfinite(tmp_path):
    path = tmp_path / "untrusted.json"
    for raw in ('{"cases": [], "cases": []}', '{"value": NaN}'):
        path.write_text(raw)
        with pytest.raises(bench.BenchmarkError):
            bench.read_json(path)
