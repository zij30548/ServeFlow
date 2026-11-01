"""Offline-only runner checks; fixtures never load .env or open a socket."""
import asyncio
from copy import deepcopy
import json
import shutil
import socket

import httpx
import pytest

from evaluation import benchmark as bench
from evaluation import intent_validation as validation
from evaluation import intent_trial as base

BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PRIVATE = "fake-key-never-log-this"
PROOF = {"root_binding_sha256": "1" * 64, "protocol_sha256": "2" * 64,
         "ledger_sha256": "3" * 64, "ledger_chain_hash": "4" * 64,
         "observation_sha256": ["5" * 64] * 24}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all sockets are forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(validation, "_configuration", lambda secret=False: {
        "base_url": BASE, "model": base.MODEL, **({"api_key": PRIVATE} if secret else {})})
    monkeypatch.setattr(validation, "development_proof", lambda root: deepcopy(PROOF))


def prepare(tmp_path):
    manifest = tmp_path / "offline.json"
    bench.write_new(manifest, bench.freeze())
    trial = validation.ValidationTrial(tmp_path / "validation", tmp_path / "sealed-development")
    protocol = asyncio.run(trial.prepare(manifest))
    return trial, protocol


def grant(trial):
    return trial.authorize("Controlled test only: 36 mock sends, zero live authorization.", confirmed=True)


def response(label="other", *, text=None):
    return httpx.Response(200, json={
        "id": "controlled", "type": "message", "role": "assistant", "model": base.MODEL,
        "content": [{"type": "text", "text": text if text is not None else json.dumps({
            "intent": label, "confidence": 0.95, "reasoning": "controlled"})}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5}})


def factory(calls, *, first=None):
    labels = {case["id"]: case["expected"]["intent"] for case in validation.validation_cases()}
    def create(operation):
        async def handler(request):
            calls.append(operation["operation_id"])
            assert PRIVATE not in request.content.decode()
            if len(calls) == 1 and first is not None:
                if isinstance(first, BaseException):
                    raise first
                return first
            return response(labels[operation["case_id"]])
        return httpx.MockTransport(handler)
    return create


def finish_first(trial, protocol):
    operation = protocol["operations"][0]
    return asyncio.run(trial._execute_operation(protocol, operation, validation.validation_cases()[0],
        config={"api_key": PRIVATE}, delegate=factory([])(operation)))


def test_preparation_freezes_only_ordered_validation_with_zero_allowance(tmp_path):
    trial, protocol = prepare(tmp_path)
    before = (tmp_path / "offline.json").read_bytes()
    ids = [case["id"] for case in validation.validation_cases()]
    assert len(ids) == len(protocol["operations"]) == 36
    assert [op["case_id"] for op in protocol["operations"]] == ids
    assert not set(ids) & {case["id"] for case in base._development_cases()}
    assert protocol["split"] == "validation" and protocol["authorized_model_requests"] == 0
    assert protocol["proposed_model_requests"] == protocol["limits"]["physical_requests"] == 36
    assert protocol["messages_url"] == BASE + "/v1/messages"
    assert protocol["development_proof"] == PROOF and trial.verify() == protocol
    assert not (trial.root / "authorization.json").exists() and trial.status()["attempts"] == 0
    assert trial.report()["paired_counts"]["not_executed"] == 36
    assert (tmp_path / "offline.json").read_bytes() == before
    with pytest.raises(validation.ValidationError, match="trial_already_prepared"):
        asyncio.run(trial.prepare(tmp_path / "offline.json"))


def test_capture_reuses_exact_production_payload_and_serialization_without_gold():
    case = deepcopy(validation.validation_cases()[0])
    case["expected"]["rationale"] = "HIDDEN_GOLD"
    captured = asyncio.run(validation._capture(case, BASE))
    original = asyncio.run(base._capture(case, BASE))
    assert captured["operation_id"].startswith(validation.BATCH + ":")
    assert {key: value for key, value in captured.items() if key != "operation_id"} == {
        key: value for key, value in original.items() if key != "operation_id"}
    assert "HIDDEN_GOLD" not in json.dumps(captured["payload"])
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert captured["payload"]["temperature"] == 0.1 and captured["payload"]["max_tokens"] == 256


def test_no_authorization_loads_no_secret_and_sends_nothing(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path)
    calls = []
    def config(secret=False):
        assert not secret
        return {"base_url": BASE, "model": base.MODEL}
    monkeypatch.setattr(validation, "_configuration", config)
    with pytest.raises(validation.ValidationError, match="explicit_authorization_required"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert not calls and trial.status()["attempts"] == 0
    with pytest.raises(validation.ValidationError, match="explicit_authorization_required"):
        trial.authorize("no confirmation")
    grant(trial)
    before = (trial.root / "authorization.json").read_bytes()
    with pytest.raises(FileExistsError):
        grant(trial)
    assert (trial.root / "authorization.json").read_bytes() == before


def test_shared_observation_counts_36_once_and_reentry_never_resends(tmp_path):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    calls = []
    report = asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == len(set(calls)) == 36
    assert report["evidence_kind"] == "controlled-shared-observation"
    assert report["shared_resources"]["usage"]["physical_requests"]["known_sum"] == 36
    assert report["shared_resources"]["usage"]["input_tokens"]["known_sum"] == 360
    for arm in ("llm_only", "fusion"):
        assert report["scores"][arm]["planned_total"] == 36
        assert report["scores"][arm]["resources"]["usage"]["physical_requests"]["known_sum"] is None
    assert sum(report["paired_counts"].values()) == 36
    value = bench.read_json(trial.root / validation._observation_name(protocol["operations"][0]))
    assert value["fusion"]["embedding_source"] == "local-ngram"
    again = asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == 36 and report["ledger"]["chain_hash"] == again["ledger"]["chain_hash"]


@pytest.mark.parametrize("answer,failed", [("not JSON", True),
    ('{"intent":"other","confidence":0.95,"reasoning":"wrong label"}', False)])
def test_ordinary_structure_or_label_errors_continue_fixed_plan(tmp_path, answer, failed):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    calls = []
    result = asyncio.run(trial.execute(delegate_factory=factory(calls, first=response(text=answer))))
    value = bench.read_json(trial.root / validation._observation_name(protocol["operations"][0]))
    assert len(calls) == 36 and not result["ledger"]["paused"]
    assert value["llm"]["parser_failed"] is failed
    assert result["scores"]["llm_only"]["accuracy"]["numerator"] == 35


@pytest.mark.parametrize("failure,code", [
    (httpx.Response(401, text=PRIVATE), "auth"), (httpx.Response(429, text=PRIVATE), "rate_limit"),
    (httpx.ConnectError(PRIVATE), "network"), (httpx.ReadTimeout(PRIVATE), "timeout"),
    (asyncio.CancelledError(), "cancelled"),
])
def test_transport_failure_occupies_one_pauses_and_never_leaks_or_retries(tmp_path, failure, code, capsys):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    calls = []
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(trial.execute(delegate_factory=factory(calls, first=failure)))
    value = bench.read_json(trial.root / validation._observation_name(protocol["operations"][0]))
    assert value["transport"]["error_code"] == code
    assert value["response"]["text"] == "" and value["fusion"] is None
    assert len(calls) == trial.status()["attempts"] == 1 and trial.status()["paused"]
    assert PRIVATE not in json.dumps(value) + capsys.readouterr().out + capsys.readouterr().err
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == 1
    report = trial.report()
    assert report["scores"]["llm_only"]["accuracy"]["denominator"] == 36
    assert report["scores"]["llm_only"]["not_executed"] == 35


def test_pending_crash_is_unknown_on_restart_and_cannot_be_replayed(tmp_path):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    operation = protocol["operations"][0]
    trial._ledger(protocol).reserve(operation["operation_id"], operation["payload_sha256"])
    restarted = validation.ValidationTrial(trial.root, trial.development_root)
    calls = []
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(restarted.execute(delegate_factory=factory(calls)))
    report = restarted.report()
    assert not calls and report["paired_counts"]["execution_unknown"] == 1
    assert report["paired_counts"]["not_executed"] == 35
    assert report["shared_resources"]["usage"]["physical_requests"]["unknown_count"] == 1


def test_written_but_unsealed_pending_output_is_unknown_not_scored_or_resent(tmp_path):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    operation = protocol["operations"][0]
    trial._ledger(protocol).reserve(operation["operation_id"], operation["payload_sha256"])
    # Simulate a crash between exclusive observation write and ledger.finish.
    # No unsealed output, even a valid-looking one, can earn a result or usage.
    base._write(trial.root, validation._observation_name(operation), {"intent": "order_status"})
    before = (trial.root / validation._observation_name(operation)).read_bytes()
    restarted = validation.ValidationTrial(trial.root, trial.development_root)
    report = restarted.report()
    assert report["paired_counts"]["execution_unknown"] == 1
    assert report["paired_counts"]["not_executed"] == 35
    assert report["observation_sha256"] == []
    assert report["scores"]["llm_only"]["accuracy"]["numerator"] == 0
    assert report["shared_resources"]["usage"]["physical_requests"]["unknown_count"] == 1
    calls = []
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(restarted.execute(delegate_factory=factory(calls)))
    assert not calls and restarted.status()["attempts"] == 1
    assert (trial.root / validation._observation_name(operation)).read_bytes() == before


def test_finish_write_failure_keeps_real_execution_window_unknown(tmp_path, monkeypatch):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    calls = []
    def failed_finish(*args, **kwargs):
        raise OSError("controlled persistence failure")
    monkeypatch.setattr(validation.ValidationLedger, "finish", failed_finish)
    with pytest.raises(validation.ValidationError, match="execution_stopped"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == 1 and trial.status()["attempts"] == 1
    output = trial.root / validation._observation_name(protocol["operations"][0])
    before = {path: path.read_bytes() for path in [output, trial.root / "ledger" / "journal.jsonl"]}
    restarted = validation.ValidationTrial(trial.root, trial.development_root)
    report = restarted.report()
    assert report["paired_counts"]["execution_unknown"] == 1
    assert report["paired_counts"]["not_executed"] == 35
    assert report["observation_sha256"] == []
    assert report["shared_resources"]["usage"]["physical_requests"]["unknown_count"] == 1
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(restarted.execute(delegate_factory=factory(calls)))
    assert len(calls) == 1
    assert all(path.read_bytes() == value for path, value in before.items())


def test_nonpending_unsealed_output_is_rejected_without_scoring(tmp_path):
    trial, protocol = prepare(tmp_path)
    base._write(trial.root, validation._observation_name(protocol["operations"][0]), {})
    with pytest.raises(validation.ValidationError, match="unsealed_observation"):
        trial.report()
    assert trial.status()["attempts"] == 0


@pytest.mark.parametrize("relationship", ["equal", "inside", "parent"])
def test_validation_never_writes_inside_or_around_development_tree(tmp_path, relationship):
    previous = tmp_path / "development"
    previous.mkdir()
    marker = previous / "keep"
    marker.write_bytes(b"old sealed evidence")
    current = {"equal": previous, "inside": previous / "new", "parent": tmp_path}[relationship]
    with pytest.raises(validation.ValidationError, match="development_root_forbidden"):
        validation.ValidationTrial(current, previous)
    assert list(previous.iterdir()) == [marker]
    assert marker.read_bytes() == b"old sealed evidence"


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_damaged_sealed_observation_blocks_before_next_delegate(tmp_path, damage):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    finish_first(trial, protocol)
    path = trial.root / validation._observation_name(protocol["operations"][0])
    if damage == "missing":
        path.unlink()
    else:
        with path.open("a") as stream:
            stream.write(" ")
    calls = []
    with pytest.raises(validation.ValidationError, match="completed_observation_missing|observation_hash_changed"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert not calls and trial.status()["attempts"] == 1


def test_old_authorization_cannot_apply_and_copied_root_cannot_verify(tmp_path):
    trial, protocol = prepare(tmp_path)
    value = {"schema_version": "serveflow-intent-authorization-v1", "batch": base.BATCH,
             "protocol_sha256": bench.sha256(trial.root / "protocol.json"),
             "root_binding_sha256": base._root_hash(trial.root), "max_requests": 24,
             "user_authorization_reference": "already consumed old 24", "created_at": base._now(),
             "initial_ledger_chain_hash": trial._ledger(protocol).initial_chain_hash}
    base._write(trial.root, "authorization.json", value)
    calls = []
    with pytest.raises(validation.ValidationError, match="authorization_binding"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert not calls
    copied = tmp_path / "copied"
    shutil.copytree(trial.root, copied)
    with pytest.raises(validation.ValidationError, match="protocol_version"):
        validation.ValidationTrial(copied, trial.development_root).verify()
    with pytest.raises(validation.ValidationError, match="development_root_forbidden"):
        validation.ValidationTrial(trial.development_root, trial.development_root)


def test_changed_frozen_sources_or_development_evidence_fail_closed(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path)
    grant(trial)
    monkeypatch.setattr(validation, "development_proof", lambda root: {**PROOF, "ledger_chain_hash": "a" * 64})
    with pytest.raises(validation.ValidationError, match="development_evidence_changed"):
        trial.verify()
    monkeypatch.setattr(validation, "development_proof", lambda root: deepcopy(PROOF))
    original = bench.sha256
    monkeypatch.setattr(bench, "sha256", lambda path: "a" * 64 if str(path).endswith("evaluation/intent_validation.py") else original(path))
    with pytest.raises(validation.ValidationError, match="extension_artifacts_changed"):
        trial.verify()


def test_stop_marker_created_during_dispatch_blocks_next_send(tmp_path):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    calls = []
    def create(operation):
        async def handler(request):
            calls.append(operation["operation_id"])
            trial._stop()
            return response()
        return httpx.MockTransport(handler)
    with pytest.raises(validation.ValidationError, match="trial_paused"):
        asyncio.run(trial.execute(delegate_factory=create))
    assert len(calls) == 1


def test_development_evidence_drift_during_dispatch_stops_before_second_send(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path)
    grant(trial)
    calls = []
    def create(operation):
        async def handler(request):
            calls.append(operation["operation_id"])
            monkeypatch.setattr(validation, "development_proof", lambda root: {
                **PROOF, "protocol_sha256": "a" * 64})
            return response()
        return httpx.MockTransport(handler)
    with pytest.raises(validation.ValidationError, match="execution_stopped"):
        asyncio.run(trial.execute(delegate_factory=create))
    assert len(calls) == 1 and (trial.root / "stopped.json").is_file()
    # Inspect the ledger independently of the deliberately invalidated proof.
    assert trial._ledger(bench.read_json(trial.root / "protocol.json")).status()["attempts"] == 1


def test_invalid_controlled_factory_cannot_fall_through_to_live_transport(tmp_path):
    trial, _ = prepare(tmp_path)
    grant(trial)
    with pytest.raises(validation.ValidationError, match="execution_stopped"):
        asyncio.run(trial.execute(delegate_factory=lambda operation: None))
    assert trial.status()["attempts"] == 0


def test_controlled_and_live_evidence_cannot_mix(tmp_path):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    finish_first(trial, protocol)
    with pytest.raises(validation.ValidationError, match="mixed_or_unknown_evidence"):
        asyncio.run(trial.execute())
    assert trial.status()["attempts"] == 1


def test_report_detects_concurrent_ledger_change_without_writing_chain(tmp_path, monkeypatch):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    ledger = trial._ledger(protocol)
    original_status = ledger.status
    calls = 0
    def status():
        nonlocal calls
        calls += 1
        value = original_status()
        if calls >= 3:
            value["chain_hash"] = "a" * 64
        return value
    monkeypatch.setattr(ledger, "status", status)
    monkeypatch.setattr(trial, "_ledger", lambda protocol: ledger)
    before = (trial.root / "ledger" / "journal.jsonl").read_bytes()
    with pytest.raises(validation.ValidationError, match="ledger_changed_during_report"):
        trial.report()
    assert (trial.root / "ledger" / "journal.jsonl").read_bytes() == before


def test_cli_has_no_arbitrary_root_endpoint_or_model_options():
    for option in ("--root", "--base-url", "--model"):
        with pytest.raises(SystemExit):
            validation.main(["status", option, "untrusted"])
