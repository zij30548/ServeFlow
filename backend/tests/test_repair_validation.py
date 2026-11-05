"""Offline acceptance preparation only: no real model, index or database."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket

import pytest

from evaluation import benchmark as bench, repair_validation as prep
from evaluation.intent_transport import canonical_payload_hash


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repair preparation cannot contact any service")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def copied_root(tmp_path):
    root = tmp_path / "backend"
    paths = ["demo_business/fixtures.json", "knowledge/serveflow-demo-v2.json", prep.DATASET]
    paths += [str(p.relative_to(bench.BACKEND_ROOT))
              for p in bench.DATASET_ROOT.glob("*.json")]
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bench.BACKEND_ROOT / name, target)
    return root


def load(root):
    return bench.read_json(root / prep.DATASET)


def write(root, data):
    (root / prep.DATASET).write_text(json.dumps(data, ensure_ascii=False))


def test_actual_source_cases_have_separate_known_and_new_denominators():
    result = prep.validate()
    assert result["tasks"] == {"total": 8, "known_regression": 4, "new_wording": 4, "kinds": {}}
    assert result["retrieval"]["kinds"] == {"answerable": 3, "insufficient": 2, "out_of_scope": 3}
    assert prep.MATHEMATICAL_ENVELOPE == 200
    assert prep.PROPOSED_CEILING == 96 < prep.MATHEMATICAL_ENVELOPE


@pytest.mark.parametrize("suite", ["tasks", "retrieval"])
def test_legacy_questions_and_every_expected_field_must_stay_exact(copied_root, suite):
    data = load(copied_root)
    original = data[suite][0]
    path, digest, ids = prep.OLD_DATASETS[suite]
    prior = bench.read_json(copied_root / path)
    assert original["case"] == next(c for c in prior["cases"] if c["id"] == ids[0])
    assert bench.sha256(copied_root / path) == digest
    original["case"]["expected"]["required_facts"].append("new requirement must not change old gold")
    write(copied_root, data)
    with pytest.raises(bench.BenchmarkError, match="repair_old_case_rewritten"):
        prep.validate(copied_root)


@pytest.mark.parametrize("edit,code", [
    (lambda d: d["tasks"].pop(), "repair_case_count"),
    (lambda d: d["tasks"][1].update(id="RT01"), "repair_identity_or_order"),
    (lambda d: d["tasks"][4].update(group="known_regression"), "repair_group_or_review"),
    (lambda d: d["tasks"][4]["case"].update(review_status="needs_adjudication"), "repair_group_or_review"),
    (lambda d: d["tasks"][4]["case"]["input"].update(customer_id="model-claimed-owner"), "expanded_trusted_identity"),
    (lambda d: d["tasks"][4]["case"]["input"].update(expected="must not be sent"), "expanded_input_fields"),
    (lambda d: d["tasks"][4]["case"]["expected"].update(required_agents=["billing"]), "repair_expected_fields"),
    (lambda d: d["tasks"][4]["case"]["expected"].update(tool_task=False), "repair_task_expectations"),
    (lambda d: d["tasks"][4]["case"]["execution_plan"].update(role_names_are_scoring_gate=True), "repair_task_expectations"),
    (lambda d: d["tasks"][4]["case"]["collection_assertions"][0].update(expected_count=1), "expanded_collection_count"),
    (lambda d: d["tasks"][4]["case"]["references"][0]["fields"].update(total_amount_minor=1), "unsupported_fixture_value"),
    (lambda d: d["retrieval"][4]["case"]["references"][0].update(quote="fabricated official rule"), "unsupported_knowledge_quote"),
    (lambda d: d["retrieval"][4]["case"]["expected"].update(relevant_titles=["退款政策"]), "repair_retrieval_gold"),
    (lambda d: d["retrieval"][7]["case"]["expected"]["required_facts"].append("invent a price"), "repair_retrieval_applicability"),
    (lambda d: d["retrieval"][5]["case"]["expected"]["missing_facts"].append("unasked diagnosis"), "repair_retrieval_applicability"),
])
def test_invalid_labels_sources_scopes_and_identity_fail_before_capture(copied_root, edit, code):
    data = load(copied_root)
    edit(data)
    write(copied_root, data)
    with pytest.raises(bench.BenchmarkError, match=code):
        prep.validate(copied_root)


def test_new_wording_is_exact_deduplicated_but_not_claimed_blind(copied_root):
    data = load(copied_root)
    old = bench.read_json(copied_root / prep.OLD_DATASETS["tasks"][0])["cases"][0]
    data["tasks"][4]["case"]["input"] = deepcopy(old["input"])
    write(copied_root, data)
    with pytest.raises(bench.BenchmarkError, match="repair_duplicate_question"):
        prep.validate(copied_root)
    assert any("not independent" in line for line in prep.LIMITATIONS)


def test_model_input_never_projects_expected_sources_or_trusted_context():
    data = load(bench.BACKEND_ROOT)
    row = deepcopy(data["tasks"][4])
    row["case"]["expected"]["rationale"] = "GOLD_CANARY_NO_MODEL"
    inp = bench.model_input(row["case"])
    assert set(inp) == {"message", "history"}
    assert "GOLD_CANARY_NO_MODEL" not in json.dumps(inp)
    assert bench.fixture_context(row["case"]) == {"customer_id": "demo-customer-a"}
    inp["message"] = "changed"
    assert row["case"]["input"]["message"] != "changed"


def test_closed_preparation_rejects_new_execution_sources_without_rebinding():
    # The next independently frozen execution adds files. Never widen this
    # completed preparation's source allowlist or replace its protocol hash.
    with pytest.raises(bench.BenchmarkError, match="repair_unapproved_source_set"):
        prep.source_proof()
    from evaluation import expanded_trial
    with pytest.raises(Exception, match="expanded_unapproved_source_set"):
        expanded_trial.source_proof()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # All private storage/cache/environment proofs are controlled in these
    # unit tests. SDK requests below still use the real in-memory transport.
    monkeypatch.setattr(prep, "source_proof", lambda: {"business_modified": False, "test_source": "fixed"})
    monkeypatch.setattr(prep, "prior_proof", lambda: {"files": 0, "caches": 0, "test_source": "fixed"})
    monkeypatch.setattr(prep.base, "_configuration", lambda: {
        "base_url": "https://dashscope.aliyuncs.com/apps/anthropic", "model": prep.base.MODEL})
    monkeypatch.setattr(prep.engine, "_dependencies", lambda: {"test_dependencies": "fixed"})
    monkeypatch.setattr(prep.history, "effective_environment", lambda: {"test_environment": "fixed"})
    monkeypatch.setattr(prep.task_data, "offline_data_proof", lambda: {"database": "not-connected", "inference": "not-performed"})
    return tmp_path / "preparation"


def test_real_sdk_mock_capture_freezes_sixteen_first_requests_and_zero_authorization(isolated):
    value = asyncio.run(prep.prepare(isolated))
    assert value["authorized_model_requests"] == value["executed_model_requests"] == 0
    assert value["live_sender_ready"] is False and len(value["sdk_first_captures"]) == 16
    for capture in value["sdk_first_captures"]:
        body = capture["payload"]
        assert body["model"] == prep.base.MODEL and body["thinking"] == {"type": "disabled"}
        assert body.get("stream", False) is False and "api_key" not in json.dumps(body)
        assert capture["evidence_kind"] == "offline-sdk-capture-not-live"
        assert body["max_tokens"] == (256 if capture["suite"] == "tasks" else 512)
        assert body["temperature"] == (.1 if capture["suite"] == "tasks" else .3)
    assert not (isolated / "authorization.json").exists() and not (isolated / "ledger").exists()
    result = asyncio.run(prep.verify(isolated))
    assert result["verified"] and result["quality_scores"] is None and not result["live_sender_ready"]


def test_prepare_is_exclusive_and_never_replaces_existing_preparation(isolated):
    asyncio.run(prep.prepare(isolated))
    path = isolated / "protocol.json"
    digest = bench.sha256(path)
    with pytest.raises(bench.BenchmarkError, match="repair_preparation_exists"):
        asyncio.run(prep.prepare(isolated))
    assert bench.sha256(path) == digest


@pytest.mark.parametrize("field", ["source_proof", "prior_proof", "offline_data_proof"])
def test_version_old_evidence_and_data_proofs_cannot_drift(isolated, monkeypatch, field):
    asyncio.run(prep.prepare(isolated))
    owner = prep.task_data if field == "offline_data_proof" else prep
    monkeypatch.setattr(owner, field, lambda: {"changed_after_preparation": True})
    with pytest.raises(bench.BenchmarkError, match="repair_protocol_changed"):
        asyncio.run(prep.verify(isolated))


def test_updated_payload_and_both_hashes_still_cannot_change_frozen_input(isolated):
    asyncio.run(prep.prepare(isolated))
    path = isolated / "protocol.json"
    value = bench.read_json(path)
    capture = value["sdk_first_captures"][0]
    capture["payload"]["model"] = "unapproved-model"
    capture["payload_sha256"] = canonical_payload_hash(capture["payload"])
    raw = json.dumps(capture["payload"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    capture["request_bytes_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(value, ensure_ascii=False))
    with pytest.raises(bench.BenchmarkError, match="repair_capture_changed"):
        asyncio.run(prep.verify(isolated))


@pytest.mark.parametrize("command", ["run", "authorize", "_worker"])
def test_no_send_or_authorization_entrypoint_is_exposed(command):
    with pytest.raises(SystemExit) as exc:
        prep.main([command])
    assert exc.value.code == 2
