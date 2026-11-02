"""Post-fix preparation and controlled sends; no live services or private data."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace

import httpx
import pytest

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation import intent_validation as engine
from evaluation import intent_recheck as recheck


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
KEY = "fake-controlled-key-do-not-output"
PROOF = {"offline_manifest_sha256": "a" * 64, "batches": [{"synthetic": True}]}
REAL_ARCHIVED_PROOF = recheck.archived_evidence_proof


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("recheck tests forbid every real socket")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(recheck, "_configuration", lambda secret=False: {
        "base_url": BASE, "model": base.MODEL, **({"api_key": KEY} if secret else {})})
    monkeypatch.setattr(recheck, "archived_evidence_proof", lambda: deepcopy(PROOF))


def prepare(tmp_path):
    trial = recheck.RecheckTrial(tmp_path / "recheck")
    protocol = asyncio.run(trial.prepare())
    return trial, protocol


def grant(trial):
    trial.authorize("CONTROLLED fixture grant only; zero real model authorization.", confirmed=True)


def response(label="other", text=None):
    return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
        "model": base.MODEL, "content": [{"type": "text", "text": text if text is not None else json.dumps({
            "intent": label, "confidence": .95, "reasoning": "controlled"})}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5}})


def factory(calls, first=None):
    expected = {case["id"]: case["expected"]["intent"] for case in recheck.cases()}
    def create(operation):
        async def handle(request):
            calls.append(operation["operation_id"])
            assert KEY not in request.content.decode()
            if len(calls) == 1 and first is not None:
                if isinstance(first, BaseException):
                    raise first
                return first
            return response(expected[operation["case_id"]])
        return httpx.MockTransport(handle)
    return create


def test_preparation_is_new_24_case_identity_with_no_authorization(tmp_path):
    trial, protocol = prepare(tmp_path)
    assert protocol["batch"] == "serveflow-intent-recheck-v2"
    assert protocol["proposed_model_requests"] == len(protocol["operations"]) == 24
    assert protocol["authorized_model_requests"] == 0 and trial.status()["attempts"] == 0
    assert [op["case_id"] for op in protocol["operations"]] == ["i24", "i43"] + [f"n{i:02d}" for i in range(1, 23)]
    assert protocol["archived_evidence_proof"] == PROOF
    assert not (trial.root / "authorization.json").exists()
    assert trial.verify() == protocol
    report = trial.report()
    assert report["paired_counts"]["not_executed"] == 24
    assert report["partitions"]["seen_regression"]["classification_scores"]["fusion"]["planned_total"] == 2
    assert report["partitions"]["new_challenge"]["classification_scores"]["fusion"]["planned_total"] == 22
    with pytest.raises(engine.ValidationError, match="trial_already_prepared"):
        asyncio.run(trial.prepare())


def test_no_authorization_never_loads_key_or_sends(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path)
    def config(secret=False):
        assert not secret
        return {"base_url": BASE, "model": base.MODEL}
    monkeypatch.setattr(recheck, "_configuration", config)
    calls = []
    with pytest.raises(engine.ValidationError, match="explicit_authorization_required"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert not calls and trial.status()["attempts"] == 0


def test_model_inputs_have_no_labels_partition_or_rationale(tmp_path):
    trial, protocol = prepare(tmp_path)
    for case, operation in zip(recheck.cases(), protocol["operations"]):
        original = asyncio.run(base._capture(case, BASE))
        assert operation["payload"] == original["payload"]
        assert operation["request_bytes_sha256"] == original["request_bytes_sha256"]
        body = json.dumps(operation["payload"], ensure_ascii=False)
        assert case["expected"]["rationale"] not in body
        assert case["partition"] not in body
        assert operation["operation_id"] not in body
        assert operation["payload"]["thinking"] == {"type": "disabled"}
        assert operation["payload"]["max_tokens"] == 256 and operation["payload"]["temperature"] == .1


def test_controlled_run_uses_shared_observation_and_never_resends(tmp_path):
    trial, _ = prepare(tmp_path)
    grant(trial)
    calls = []
    result = asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == len(set(calls)) == 24
    assert result["evidence_kind"] == "controlled-shared-observation"
    assert result["shared_resources"]["usage"]["physical_requests"]["known_sum"] == 24
    assert result["shared_resources"]["usage"]["input_tokens"]["known_sum"] == 240
    for partition, count in [("seen_regression", 2), ("new_challenge", 22)]:
        for arm in ("llm_only", "fusion"):
            assert result["partitions"][partition]["classification_scores"][arm]["planned_total"] == count
    assert all(result["scores"][arm]["resources"]["usage"]["physical_requests"]["known_sum"] is None
               for arm in ("llm_only", "fusion"))
    again = asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == 24 and again["ledger"]["chain_hash"] == result["ledger"]["chain_hash"]
    with pytest.raises(engine.ValidationError, match="cannot_authorize_started_trial"):
        grant(trial)


@pytest.mark.parametrize("first", [response(text="broken JSON"), response("other")])
def test_ordinary_structure_or_label_failure_kept_and_fixed_plan_continues(tmp_path, first):
    trial, _ = prepare(tmp_path); grant(trial)
    calls = []
    result = asyncio.run(trial.execute(delegate_factory=factory(calls, first)))
    assert len(calls) == 24 and not result["ledger"]["paused"]
    assert result["scores"]["llm_only"]["accuracy"]["denominator"] == 24
    assert result["scores"]["llm_only"]["accuracy"]["numerator"] == 23
    assert result["partitions"]["seen_regression"]["classification_scores"]["llm_only"]["accuracy"]["numerator"] == 1


@pytest.mark.parametrize("first,code", [
    (httpx.Response(401, text=KEY), "auth"), (httpx.Response(429, text=KEY), "rate_limit"),
    (httpx.ConnectError(KEY), "network"), (httpx.ReadTimeout(KEY), "timeout"),
    (asyncio.CancelledError(), "cancelled"),
])
def test_execution_errors_consume_one_and_pause_without_retries_or_secret(tmp_path, first, code, capsys):
    trial, protocol = prepare(tmp_path); grant(trial)
    calls = []
    with pytest.raises(engine.ValidationError, match="trial_paused"):
        asyncio.run(trial.execute(delegate_factory=factory(calls, first)))
    output = bench.read_json(trial.root / "i24-observation.json")
    assert output["transport"]["error_code"] == code and output["fusion"] is None
    assert trial.status()["attempts"] == len(calls) == 1 and trial.status()["paused"]
    assert KEY not in json.dumps(output) + capsys.readouterr().out
    with pytest.raises(engine.ValidationError, match="trial_paused"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert len(calls) == 1


def test_pending_reservation_is_unknown_not_unexecuted_and_never_resent(tmp_path):
    trial, protocol = prepare(tmp_path); grant(trial)
    operation = protocol["operations"][0]
    trial._ledger(protocol).reserve(operation["operation_id"], operation["payload_sha256"])
    base._write(trial.root, "i24-observation.json", {"unsealed": True})
    before = (trial.root / "i24-observation.json").read_bytes()
    restarted = recheck.RecheckTrial(trial.root)
    result = restarted.report()
    assert result["paired_counts"]["execution_unknown"] == 1 and result["paired_counts"]["not_executed"] == 23
    assert result["shared_resources"]["usage"]["physical_requests"]["unknown_count"] == 1
    with pytest.raises(engine.ValidationError, match="trial_paused"):
        asyncio.run(restarted.execute(delegate_factory=factory([])))
    assert (trial.root / "i24-observation.json").read_bytes() == before


def test_old_grant_or_copied_directory_cannot_authorize(tmp_path):
    trial, protocol = prepare(tmp_path); grant(trial)
    auth = bench.read_json(trial.root / "authorization.json")
    auth["schema_version"] = "serveflow-intent-authorization-v1"
    # Fault injection, not a normal writer; ordinary API never overwrites.
    (trial.root / "authorization.json").write_text(json.dumps(auth))
    with pytest.raises(engine.ValidationError, match="authorization_binding"):
        asyncio.run(trial.execute(delegate_factory=factory([])))
    copied = tmp_path / "copied"; shutil.copytree(trial.root, copied)
    with pytest.raises(engine.ValidationError, match="protocol_version"):
        recheck.RecheckTrial(copied).verify()


def test_frozen_version_archive_and_cases_drift_stop_before_send(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path); grant(trial)
    monkeypatch.setattr(recheck, "archived_evidence_proof", lambda: {"changed": True})
    with pytest.raises(engine.ValidationError, match="archived_evidence_changed"):
        trial.verify()
    monkeypatch.setattr(recheck, "archived_evidence_proof", lambda: deepcopy(PROOF))
    original = recheck._artifacts
    monkeypatch.setattr(recheck, "_artifacts", lambda: {**original(), "unexpected": "a" * 64})
    with pytest.raises(engine.ValidationError, match="frozen_artifacts_changed"):
        trial.verify()


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_finished_observation_hash_is_checked_before_next_send(tmp_path, damage):
    trial, protocol = prepare(tmp_path); grant(trial)
    operation = protocol["operations"][0]
    asyncio.run(trial._execute_operation(protocol, operation, recheck.cases()[0],
                config={"api_key": KEY}, delegate=factory([])(operation)))
    output = trial.root / "i24-observation.json"
    if damage == "missing": output.unlink()
    else: output.write_bytes(output.read_bytes() + b" ")
    calls = []
    with pytest.raises(engine.ValidationError, match="completed_observation_missing|observation_hash_changed"):
        asyncio.run(trial.execute(delegate_factory=factory(calls)))
    assert not calls and trial.status()["attempts"] == 1


def test_controlled_factory_cannot_fall_through_to_live(tmp_path):
    trial, _ = prepare(tmp_path); grant(trial)
    with pytest.raises(engine.ValidationError, match="execution_stopped"):
        asyncio.run(trial.execute(delegate_factory=lambda operation: None))
    assert trial.status()["attempts"] == 0


def test_archive_source_hashes_use_git_blob_not_changed_current_file(monkeypatch):
    # Local deterministic subprocess fixture; not current private evidence.
    expected = hashlib.sha256(b"original blob").hexdigest(); calls = []
    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=b"original blob")
    monkeypatch.setattr(recheck.subprocess, "run", command)
    recheck._archived_sources({"git_head": "fixed", "artifacts": {"core/x.py": expected}}, "artifacts")
    assert calls == [["git", "show", "fixed:backend/core/x.py"]]
    with pytest.raises(engine.ValidationError, match="archived_sources_changed"):
        recheck._archived_sources({"git_head": "fixed", "artifacts": {"core/x.py": "a" * 64}}, "artifacts")


@pytest.mark.parametrize("damage", ["count", "unknown", "order", "label", "seen_input", "duplicate_new"])
def test_case_manifest_rejects_count_unknown_fields_ids_labels_and_seen_drift(tmp_path, monkeypatch, damage):
    data = bench.read_json(recheck.CASES_PATH)
    if damage == "count": data["cases"].pop()
    elif damage == "unknown": data["cases"][2]["new_field"] = "forbidden"
    elif damage == "order": data["cases"][2]["id"] = "n22"
    elif damage == "label": data["cases"][2]["expected"]["intent"] = "billing"
    elif damage == "seen_input": data["cases"][0]["input"]["message"] = "changed historical question"
    else: data["cases"][3]["input"] = deepcopy(data["cases"][2]["input"])
    path = tmp_path / "cases.json"; path.write_text(json.dumps(data))
    monkeypatch.setattr(recheck, "CASES_PATH", path)
    with pytest.raises(engine.ValidationError): recheck.cases()


def test_archived_audit_refuses_missing_ledger_without_initializing(tmp_path, monkeypatch):
    # Exercise the real audit with only a synthetic, temporary archive root.
    backend = tmp_path / "backend"; backend.mkdir(mode=0o700)
    manifest = backend / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"synthetic.py": "a" * 64}, "git_head": "synthetic"}))
    manifest.chmod(0o600)
    monkeypatch.setattr(bench, "BACKEND_ROOT", backend)
    monkeypatch.setattr(recheck, "OLD_MANIFEST", manifest)
    monkeypatch.setattr(recheck, "OLD_MANIFEST_HASH", bench.sha256(manifest))
    monkeypatch.setattr(recheck, "_archived_sources", lambda *_: None)
    class NeverConstruct:
        def __init__(self, *_):
            raise AssertionError("missing archived identity must never be initialized")
    monkeypatch.setattr(recheck, "OLD_BATCHES", (("missing", 24, NeverConstruct, "a" * 64, "b" * 64),))
    with pytest.raises(engine.ValidationError, match="archived_evidence_missing"):
        REAL_ARCHIVED_PROOF()
    assert not (backend / "data").exists()


@pytest.mark.parametrize(("count", "writer", "reader"), [
    (24, recheck.TrialLedger, recheck._ArchivedDevelopmentLedger),
    (36, recheck.ValidationLedger, recheck._ArchivedValidationLedger),
])
@pytest.mark.parametrize("moved", ["directory", "components"])
def test_archive_race_after_precheck_cannot_recreate_moved_state(tmp_path, monkeypatch, count, writer, reader, moved):
    # All state is synthetic. Move it after the audit's path/hash prechecks but
    # before the real inherited constructor runs; the constructor must not repair.
    backend = tmp_path / "backend"; backend.mkdir(mode=0o700)
    folder = backend / "data/benchmarks/synthetic"
    folder.mkdir(mode=0o700, parents=True)
    operations = [{"operation_id": f"synthetic-{i}", "case_id": f"case-{i}"} for i in range(count)]
    protocol = folder / "protocol.json"
    protocol.write_text(json.dumps({"operations": operations, "git_head": "synthetic", "extension_artifacts": {}}))
    protocol.chmod(0o600)
    protocol_hash = bench.sha256(protocol)
    ledger_root = folder / "ledger"
    writer(ledger_root, protocol_hash, [op["operation_id"] for op in operations], count)
    original = {str(path.relative_to(ledger_root)): path.read_bytes()
                for path in ledger_root.rglob("*") if path.is_file()}
    approval = folder / "authorization.json"; approval.write_text("{}"); approval.chmod(0o600)
    manifest = backend / "manifest.json"; manifest.write_text("{}"); manifest.chmod(0o600)
    moved_root = folder / "moved-ledger"

    class MoveBeforeRead(reader):
        def __init__(self, root, *args, **kwargs):
            if moved == "directory":
                root.rename(moved_root)
            else:
                moved_root.mkdir(mode=0o700)
                for entry in tuple(root.iterdir()):
                    entry.rename(moved_root / entry.name)
            super().__init__(root, *args, **kwargs)

    monkeypatch.setattr(bench, "BACKEND_ROOT", backend)
    monkeypatch.setattr(recheck, "OLD_MANIFEST", manifest)
    monkeypatch.setattr(recheck, "OLD_MANIFEST_HASH", bench.sha256(manifest))
    monkeypatch.setattr(recheck, "_archived_sources", lambda *_: None)
    monkeypatch.setattr(recheck, "OLD_BATCHES", (("synthetic", count, MoveBeforeRead,
        protocol_hash, bench.sha256(ledger_root / "journal.jsonl")),))
    monkeypatch.setattr(recheck, "OLD_AUXILIARY_HASHES", {
        "synthetic": {"authorization": bench.sha256(approval)}})
    with pytest.raises(recheck.TrialLedgerError):
        REAL_ARCHIVED_PROOF()
    if moved == "directory":
        assert not ledger_root.exists()
    else:
        assert ledger_root.is_dir() and not list(ledger_root.iterdir())
    assert original == {str(path.relative_to(moved_root)): path.read_bytes()
                        for path in moved_root.rglob("*") if path.is_file()}
    assert protocol_hash == bench.sha256(protocol) and approval.read_text() == "{}"


@pytest.mark.parametrize(("count", "writer", "reader"), [
    (24, recheck.TrialLedger, recheck._ArchivedDevelopmentLedger),
    (36, recheck.ValidationLedger, recheck._ArchivedValidationLedger),
])
def test_archived_wrappers_cannot_reserve_or_finish(tmp_path, count, writer, reader):
    root = tmp_path / "ledger"
    operations = [f"synthetic-{i}" for i in range(count)]
    writer(root, "a" * 64, operations, count)
    before = {str(path.relative_to(root)): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    archived = reader(root, "a" * 64, operations, count)
    for method, args in ((archived.reserve, (operations[0], "b" * 64)),
                         (archived.finish, (operations[0], "received"))):
        with pytest.raises(recheck.TrialLedgerError, match="protocol"):
            method(*args)
    assert archived.status()["attempts"] == 0
    assert before == {str(path.relative_to(root)): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}


def test_recheck_engine_inherits_same_send_and_recovery_logic():
    for name in ("authorize", "_approval", "_sealed", "_execute_operation", "_stop", "execute"):
        assert getattr(recheck.RecheckTrial, name) is getattr(engine.ValidationTrial, name)
    assert engine.ValidationTrial.max_requests == 36 and recheck.RecheckTrial.max_requests == 24
    assert engine.ValidationTrial.batch == engine.BATCH


def test_cli_has_no_arbitrary_root_model_endpoint_or_quota():
    for option in ("--root", "--base-url", "--model", "--max-requests"):
        with pytest.raises(SystemExit):
            recheck.main(["status", option, "untrusted"])
