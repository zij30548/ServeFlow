"""The intent trial is prepared and exercised with controlled transports only."""
import asyncio
from copy import deepcopy
import json
import logging
from pathlib import Path
import shutil
import socket

import httpx
import pytest

from evaluation import benchmark as bench
from evaluation import intent_trial as trial


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PRIVATE = "synthetic-private-key-must-not-be-logged"


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live network is forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(trial, "_configuration", lambda secret=False: {
        "base_url": BASE, "model": trial.MODEL, **({"api_key": PRIVATE} if secret else {})})


def prepare(tmp_path):
    manifest = tmp_path / "offline.json"
    bench.write_new(manifest, bench.freeze())
    root = (tmp_path / "trial").resolve()
    protocol = asyncio.run(trial.prepare(manifest, root=root))
    return root, protocol


def grant(root):
    return trial.authorize("Controlled test only: 24 mock delegate entries, no live authorization.", root=root, confirmed=True)


def answer_response(label="other", *, text=None, usage=True):
    body = {"id": "controlled_response", "type": "message", "role": "assistant", "model": trial.MODEL,
            "content": [{"type": "text", "text": text if text is not None else json.dumps({"intent": label, "confidence": 0.95, "reasoning": "controlled fixture"})}],
            "stop_reason": "end_turn", "stop_sequence": None}
    if usage:
        body["usage"] = {"input_tokens": 10, "output_tokens": 5}
    return httpx.Response(200, json=body, headers={"request-id": "controlled_request"})


def factory(calls, *, first=None):
    expected = {c["id"]: c["expected"]["intent"] for c in trial._development_cases()}

    def create(op):
        async def handler(request):
            calls.append(op["operation_id"])
            assert PRIVATE not in request.content.decode()
            if first is not None and len(calls) == 1:
                if isinstance(first, BaseException):
                    raise first
                return first
            return answer_response(expected[op["case_id"]])
        return httpx.MockTransport(handler)
    return create


def test_capture_uses_existing_request_without_gold_or_extra_calls():
    case = deepcopy(trial._development_cases()[0])
    case["expected"]["rationale"] = "HIDDEN_GOLD_NOT_FOR_MODEL"
    op = asyncio.run(trial._capture(case, BASE))
    text = json.dumps(op["payload"], ensure_ascii=False)
    assert "HIDDEN_GOLD_NOT_FOR_MODEL" not in text
    assert case["input"]["message"] in op["payload"]["messages"][0]["content"]
    assert set(op["payload"]) == {"model", "max_tokens", "temperature", "messages", "thinking"}
    assert op["payload"]["max_tokens"] == 256 and op["payload"]["temperature"] == 0.1
    assert op["payload"]["thinking"] == {"type": "disabled"}


def test_prepare_is_unapproved_ordered_and_does_not_change_old_freeze(tmp_path):
    root, protocol = prepare(tmp_path)
    before = (tmp_path / "offline.json").read_bytes()
    assert protocol["authorized_model_requests"] == 0
    assert protocol["proposed_model_requests"] == 24
    assert [op["case_id"] for op in protocol["operations"]] == [c["id"] for c in trial._development_cases()]
    assert not (root / "authorization.json").exists()
    assert trial._ledger(protocol, root).status()["attempts"] == 0
    assert trial.verify(root) == protocol
    assert (tmp_path / "offline.json").read_bytes() == before
    with pytest.raises(trial.TrialError, match="trial_already_prepared"):
        asyncio.run(trial.prepare(tmp_path / "offline.json", root=root))


def test_no_approval_means_no_delegate_or_secret_loading(tmp_path, monkeypatch):
    root, protocol = prepare(tmp_path)
    calls = []
    def config(secret=False):
        assert not secret
        return {"base_url": BASE, "model": trial.MODEL}
    monkeypatch.setattr(trial, "_configuration", config)
    with pytest.raises(trial.TrialError, match="explicit_authorization_required"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert calls == [] and trial._ledger(protocol, root).status()["attempts"] == 0


def test_authorization_needs_explicit_confirmation_and_is_not_overwritten(tmp_path):
    root, _ = prepare(tmp_path)
    with pytest.raises(trial.TrialError, match="explicit_authorization_required"):
        trial.authorize("not confirmed", root=root)
    receipt = grant(root)
    assert receipt["max_requests"] == 24 and trial._approval(root) == receipt
    before = (root / "authorization.json").read_bytes()
    with pytest.raises(FileExistsError):
        grant(root)
    assert (root / "authorization.json").read_bytes() == before


def test_one_observation_produces_two_decisions_without_double_counting(tmp_path, capsys):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    result = asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert len(calls) == 24 and len(set(calls)) == 24
    assert result["evidence_kind"] == "controlled-shared-observation"
    assert result["ledger"]["attempts"] == 24
    assert result["shared_resources"]["usage"]["physical_requests"]["known_sum"] == 24
    assert result["shared_resources"]["usage"]["input_tokens"]["known_sum"] == 240
    assert result["scores"]["llm_only"]["resources"]["usage"]["physical_requests"]["known_sum"] is None
    assert result["scores"]["fusion"]["planned_total"] == 24
    assert sum(result["paired_counts"].values()) == 24
    for op in protocol["operations"]:
        value = bench.read_json(root / trial._observation_name(op))
        assert value["fusion"]["embedding_source"] == "local-ngram"
        assert value["evidence_kind"] == "controlled-transport"
    again = asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert len(calls) == 24 and again["ledger"]["chain_hash"] == result["ledger"]["chain_hash"]


def test_ordinary_parse_failure_is_retained_and_remaining_planned_cases_continue(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    result = asyncio.run(trial.execute(root=root, delegate_factory=factory(calls, first=answer_response(text="not JSON"))))
    first = bench.read_json(root / trial._observation_name(protocol["operations"][0]))
    assert len(calls) == 24 and result["ledger"]["attempts"] == 24 and not result["ledger"]["paused"]
    assert first["llm"]["parser_failed"] is True
    assert first["fusion"]["embedding_source"] == "local-ngram"
    assert result["scores"]["llm_only"]["status_counts"]["failed"] == 1


@pytest.mark.parametrize("field,value", [("content", 1), ("content", True), ("stop_reason", [])])
def test_unusual_sdk_shapes_do_not_turn_ordinary_output_into_pending_send(tmp_path, field, value):
    root, protocol = prepare(tmp_path)
    grant(root)
    body = json.loads(answer_response().content)
    body[field] = value
    calls = []
    result = asyncio.run(trial.execute(root=root, delegate_factory=factory(calls, first=httpx.Response(200, json=body))))
    assert len(calls) == 24
    assert not result["ledger"]["pending"] and not result["ledger"]["paused"]
    assert not (root / "stopped.json").exists()


def test_private_writer_preserves_escaped_unicode_without_rewriting_model_output(tmp_path):
    root = (tmp_path / "private-output").resolve()
    root.mkdir(mode=0o700)
    trial._write(root, "observed.json", {"text": "surrogate:\ud800"})
    assert bench.read_json(root / "observed.json")["text"] == "surrogate:\ud800"


@pytest.mark.parametrize("failure,code", [
    (httpx.Response(401, text=PRIVATE), "auth"),
    (httpx.Response(429, text=PRIVATE), "rate_limit"),
    (httpx.Response(503, text=PRIVATE), "network"),
    (httpx.ConnectError(PRIVATE), "network"),
    (httpx.ReadTimeout(PRIVATE), "timeout"),
    (asyncio.CancelledError(), "cancelled"),
])
def test_execution_failures_occupy_one_and_pause_without_auto_retry(tmp_path, failure, code):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    with pytest.raises(trial.TrialError, match="trial_paused"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls, first=failure)))
    state = trial._ledger(protocol, root).status()
    assert len(calls) == state["attempts"] == 1 and state["paused"]
    observation = root / trial._observation_name(protocol["operations"][0])
    raw = observation.read_text()
    assert PRIVATE not in raw
    assert json.loads(raw)["transport"]["error_code"] == code
    with pytest.raises(trial.TrialError, match="trial_paused"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert len(calls) == 1
    result = trial.report(root)
    assert result["scores"]["llm_only"]["accuracy"]["denominator"] == 24
    assert result["scores"]["llm_only"]["not_executed"] == 23


def test_pending_attempt_after_crash_is_unknown_not_unexecuted_or_resendable(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    ledger = trial._ledger(protocol, root)
    op = protocol["operations"][0]
    ledger.reserve(op["operation_id"], op["payload_sha256"])
    calls = []
    with pytest.raises(trial.TrialError, match="trial_paused"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert not calls and ledger.status()["attempts"] == 1
    result = trial.report(root)
    assert result["paired_counts"]["execution_unknown"] == 1
    assert result["scores"]["llm_only"]["not_executed"] == 23
    assert result["shared_resources"]["usage"]["input_tokens"]["unknown_count"] == 1


def test_stop_record_created_during_run_blocks_the_next_physical_send(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    def create(op):
        async def handler(request):
            calls.append(op["operation_id"])
            trial._stop(root, "controlled_stop")
            return answer_response()
        return httpx.MockTransport(handler)
    with pytest.raises(trial.TrialError, match="trial_paused"):
        asyncio.run(trial.execute(root=root, delegate_factory=create))
    assert len(calls) == 1 and trial._ledger(protocol, root).status()["attempts"] == 1


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_completed_evidence_damage_blocks_later_dispatch_not_only_final_report(tmp_path, damage):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    op = protocol["operations"][0]
    delegate = factory(calls)(op)
    asyncio.run(trial._execute_operation(protocol, op, trial._development_cases()[0], root=root,
                                        config={"api_key": PRIVATE}, delegate=delegate))
    path = root / trial._observation_name(op)
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(trial.TrialError, match="completed_observation_missing|observation_hash_changed"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert len(calls) == 1


def test_controlled_observations_cannot_be_followed_by_real_transport(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    op = protocol["operations"][0]
    asyncio.run(trial._execute_operation(protocol, op, trial._development_cases()[0], root=root,
                                        config={"api_key": PRIVATE}, delegate=factory(calls)(op)))
    with pytest.raises(trial.TrialError, match="mixed_or_unknown_evidence"):
        asyncio.run(trial.execute(root=root))
    assert len(calls) == 1


def test_copied_trial_directory_does_not_create_a_new_allowance(tmp_path):
    root, _ = prepare(tmp_path)
    other = (tmp_path / "copied-trial").resolve()
    shutil.copytree(root, other)
    with pytest.raises(trial.TrialError, match="protocol_version"):
        trial.verify(other)


def test_missing_ledger_cannot_be_reinitialized_from_existing_protocol(tmp_path):
    root, _ = prepare(tmp_path)
    (root / "ledger" / "journal.jsonl").unlink()
    with pytest.raises(trial.TrialError, match="ledger_missing"):
        trial.verify(root)
    assert not (root / "ledger" / "journal.jsonl").exists()


def test_extension_version_drift_prevents_any_send(tmp_path, monkeypatch):
    root, protocol = prepare(tmp_path)
    grant(root)
    actual = bench.sha256
    def changed(path):
        return "0" * 64 if str(path).endswith("evaluation/intent_trial.py") else actual(path)
    monkeypatch.setattr(bench, "sha256", changed)
    calls = []
    with pytest.raises(trial.TrialError, match="extension_artifacts_changed"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    assert not calls


def test_payload_changes_do_not_refresh_the_freeze_or_start_calls(tmp_path):
    root, _ = prepare(tmp_path)
    path = root / "protocol.json"
    protocol = bench.read_json(path)
    protocol["operations"][0]["payload"]["max_tokens"] = 1024
    path.write_text(json.dumps(protocol))
    with pytest.raises(trial.TrialError, match="operation_binding"):
        trial.verify(root)


def test_finished_observation_is_bound_to_ledger_hash(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    calls = []
    asyncio.run(trial.execute(root=root, delegate_factory=factory(calls)))
    path = root / trial._observation_name(protocol["operations"][0])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(trial.TrialError, match="observation_hash_changed"):
        trial.report(root)


def test_completed_but_missing_observation_cannot_silently_disappear(tmp_path):
    root, protocol = prepare(tmp_path)
    grant(root)
    asyncio.run(trial.execute(root=root, delegate_factory=factory([])))
    (root / trial._observation_name(protocol["operations"][0])).unlink()
    with pytest.raises(trial.TrialError, match="completed_observation_missing"):
        trial.report(root)


def test_quiet_scope_suppresses_child_sdk_logs_and_restores_settings(caplog):
    previous = logging.root.manager.disable
    with caplog.at_level(logging.DEBUG):
        with trial._quiet_clients():
            logging.getLogger("httpcore.http11").debug(PRIVATE)
            logging.getLogger("anthropic._base_client").debug(PRIVATE)
    assert PRIVATE not in caplog.text and logging.root.manager.disable == previous


def test_run_without_authorization_reports_safe_error_and_zero_calls(tmp_path, capsys):
    root, protocol = prepare(tmp_path)
    with pytest.raises(trial.TrialError, match="explicit_authorization_required"):
        asyncio.run(trial.execute(root=root, delegate_factory=factory([])))
    assert trial._ledger(protocol, root).status()["attempts"] == 0
    assert not (root / "authorization.json").exists()
