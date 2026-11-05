"""Explicit zero-send continuation in temporary roots, never real model scores.

The old protocol, authorization, ledger and stop are real immutable fixture
files. SDK calls use MockTransport and copied synthetic business records. No
test prepares, authorizes, or executes the live repair root.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
import stat
import sys

import httpx
import pytest

from evaluation import benchmark as bench, repair_execution_trial as original
from evaluation import repair_execution_continuation as module
from evaluation import repair_execution_runtime as runtime, task_supervisor as supervisor
from evaluation.phase_deadline_transport import PHASE_DEADLINES
from evaluation.task_transport import CURRENT
from test_repair_execution_trial import prepared as prior_prepared, approve, review, Data
from test_repair_execution_runtime import SDK


REFERENCE = "Controlled explicit continuation of one zero-send startup only; no real authorization"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("continuation tests cannot contact services or read credentials")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for name in tuple(os.environ):
        if name.startswith(("ANTHROPIC_", "ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_", "SERVEFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(original.task_data, "build_live_data", denied)
    monkeypatch.setattr(original.engine, "_mysql_url", denied)


def bytes_under(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def assert_unchanged(root, before):
    for name, raw in before.items():
        assert (root / name).read_bytes() == raw, name


def continuation_command():
    return [sys.executable, "-B", "-u", "-m", "evaluation.repair_execution_continuation", "_worker"]


@pytest.fixture
def paused(prior_prepared, monkeypatch):
    assert prior_prepared.root != original.ROOT
    approve(prior_prepared)
    original.base._write(prior_prepared.root, "worker-stop.json", {
        "reason": "execution_stopped_no_auto_resume", "recorded_at": "2026-11-05T00:00:00+00:00"})
    # Stub only the source attestation; the actual immutable manifest, approval,
    # ledger replay, request audit, SDK and application paths still execute.
    def source_proof(protocol):
        return {"original_source_commit": "controlled-original-commit",
            "original_artifacts_sha256": original.canonical_payload_hash(protocol["artifacts"]),
            "current_artifacts": deepcopy(original.ancestry._current_sources()),
            "exact_supervisor_hook_sha256": "e" * 64,
            "closed_additions": ["evaluation/repair_execution_continuation.py", "tests/test_repair_execution_continuation.py"]}
    monkeypatch.setattr(module, "source_proof", source_proof)
    trial = module.ContinuationTrial(prior_prepared.root)
    return trial, bytes_under(trial.root)


@pytest.fixture
def continued(paused):
    trial, before = paused
    asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    assert_unchanged(trial.root, before)
    return trial, before


def manifest_path(trial, original_files):
    documents = [path for path in trial.root.glob("*.json") if path.name not in original_files]
    assert len(documents) == 1, "preparation must create only its immutable continuation manifest"
    return documents[0]


def native_run(trial, *, failure=None, refused_case=None):
    assert trial.root != original.ROOT
    data, calls = Data(), []
    oracles = {row["id"]: SDK(row, refuse=row["id"] == refused_case) for row in runtime.rows()}
    async def handle(request):
        frame = CURRENT.get()
        raw = bytes(request.content)
        calls.append((frame, raw, deepcopy(request.extensions)))
        if failure is not None:
            raise failure
        result = oracles[frame.case_id].response(**json.loads(raw))
        return httpx.Response(200, json={"id": "controlled-continuation-native", "type": "message", "role": "assistant",
            "model": original.base.MODEL, **result, "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 2}})
    return trial.execute(delegate_factory=lambda frame: httpx.MockTransport(handle), controlled_data=data), data, calls


def test_only_explicit_continuation_can_be_prepared_and_old_sender_stays_paused(paused):
    trial, before = paused
    with pytest.raises(Exception, match="explicit|confirm"):
        asyncio.run(trial.prepare_continuation(REFERENCE))
    assert bytes_under(trial.root) == before
    with pytest.raises(Exception, match="paused"):
        asyncio.run(original.RepairExecutionTrial(trial.root).execute(
            delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("old sender cannot resume")),
            controlled_data=Data()))
    assert bytes_under(trial.root) == before


def test_unprepared_validate_is_read_only_and_does_not_create_a_continuation(paused):
    trial, before = paused
    assert trial.validate()
    assert bytes_under(trial.root) == before
    assert not (trial.root / module.CONTINUATION_FILE).exists()
    assert not (trial.root / module.STARTED_FILE).exists()
    with pytest.raises(Exception):
        trial.verify()
    assert bytes_under(trial.root) == before


def test_prepared_validate_is_read_only_and_does_not_claim_the_start(continued):
    trial, _ = continued
    before = bytes_under(trial.root)
    assert trial.validate()
    assert bytes_under(trial.root) == before
    assert not (trial.root / module.STARTED_FILE).exists()


@pytest.mark.parametrize("reference", ["", " ", None, 96, "x" * 501])
def test_continuation_requires_a_bounded_explicit_reference_without_writing(paused, reference):
    trial, before = paused
    with pytest.raises(Exception):
        asyncio.run(trial.prepare_continuation(reference, confirmed=True))
    assert bytes_under(trial.root) == before


def test_manifest_is_immutable_and_binds_the_same_approval_original_stop_prefix_and_version(continued):
    trial, before = continued
    path = manifest_path(trial, before)
    manifest, raw = bench.read_json(path), path.read_bytes()
    encoded = json.dumps(manifest, ensure_ascii=True, sort_keys=True)
    assert REFERENCE in encoded
    for name in ("protocol.json", "authorization.json", "worker-stop.json"):
        assert hashlib.sha256(before[name]).hexdigest() in encoded
    for name in ("ledger/identity.json", "ledger/journal.jsonl", "ledger/run.lock"):
        assert hashlib.sha256(before[name]).hexdigest() in encoded
    assert trial.ledger(trial.verify()).initial_chain_hash in encoded
    assert original.base._root_hash(trial.root) in encoded
    assert module._git_head() in encoded
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    state = trial.ledger(trial.verify()).status()
    assert state["attempts"] == 0 and not state["pending"] and state["active_case"] is None
    assert trial._approval(trial.verify())["max_requests"] == 96
    with pytest.raises(Exception, match="exists|already|prepared"):
        asyncio.run(trial.prepare_continuation("must not supersede", confirmed=True))
    assert path.read_bytes() == raw
    assert_unchanged(trial.root, before)


@pytest.mark.parametrize("evidence", ["case_start", "pending", "unknown", "request", "physical", "case", "data", "data_seal"])
def test_any_prior_execution_evidence_prevents_continuation_without_altering_it(paused, evidence):
    trial, _ = paused
    p = original.RepairExecutionTrial(trial.root).verify()
    ledger = trial.ledger(p)
    if evidence in {"case_start", "pending", "unknown"}:
        with ledger.exclusive_run() as lease:
            ledger.start_case("RT01", lease=lease)
            if evidence != "case_start":
                claim = ledger.reserve("RT01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
                if evidence == "unknown":
                    ledger.finish(claim["operation_id"], "unknown", lease=lease)
    else:
        name = {"request": "request-" + "a" * 64 + ".json", "physical": "physical-001.json",
                "case": "case-RT01.json", "data": "data-proof.json", "data_seal": "data-proof-seal.json"}[evidence]
        original.base._write(trial.root, name, {"controlled_orphan_evidence": True})
    before = bytes_under(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    assert bytes_under(trial.root) == before


@pytest.mark.parametrize("damage", ["authorization", "protocol", "worker_stop", "journal", "identity", "source", "version"])
def test_manifest_rejects_binding_changes_before_resources_or_reservation(continued, monkeypatch, damage):
    trial, _ = continued
    if damage == "source":
        monkeypatch.setattr(original.ancestry, "_current_sources", lambda: {"unreviewed": "f" * 64})
    elif damage == "version":
        monkeypatch.setattr(module, "_git_head", lambda: "another-version")
    else:
        path = trial.root / {"authorization": "authorization.json", "protocol": "protocol.json",
            "worker_stop": "worker-stop.json", "journal": "ledger/journal.jsonl", "identity": "ledger/identity.json"}[damage]
        if damage == "journal":
            path.write_bytes(path.read_bytes() + b" ")
        else:
            value = bench.read_json(path)
            if damage == "authorization":
                value["reference"] = "changed approval"
            elif damage == "protocol":
                value["git_head"] = "changed original version"
            elif damage == "worker_stop":
                value["recorded_at"] = "changed historical receipt"
            else:
                value["max_requests"] = 95
            path.write_text(json.dumps(value))
    before = bytes_under(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == before


@pytest.mark.parametrize("damage", ["no_stop", "wrong_stop", "stopped", "supervisor_stop", "no_authorization"])
def test_continuation_requires_exact_original_worker_stop_and_existing_authorization(paused, damage):
    trial, _ = paused
    if damage == "no_stop":
        (trial.root / "worker-stop.json").unlink()
    elif damage == "wrong_stop":
        path = trial.root / "worker-stop.json"
        value = bench.read_json(path)
        value["reason"] = "cleanup_failed_no_auto_resume"
        path.write_text(json.dumps(value))
    elif damage == "no_authorization":
        (trial.root / "authorization.json").unlink()
    else:
        original.base._write(trial.root, "stopped.json" if damage == "stopped" else "supervisor-stop.json", {
            "reason": "controlled prior failure"})
    before = bytes_under(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    assert bytes_under(trial.root) == before


def test_continuation_manifest_rejects_unknown_fields_without_reserving(continued):
    trial, before = continued
    path = manifest_path(trial, before)
    value = bench.read_json(path)
    value["allow_another_continuation"] = True
    path.write_text(json.dumps(value))
    changed = bytes_under(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == changed


@pytest.mark.parametrize("name", ["ledger/identity.json", "ledger/journal.jsonl", "ledger/run.lock"])
def test_replacing_original_ledger_files_with_equal_bytes_does_not_restore_a_start(continued, name):
    trial, _ = continued
    path = trial.root / name
    replacement = path.with_name("controlled-replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    replacement.replace(path)
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="ledger_or_stop_changed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == before


@pytest.mark.parametrize("kind", ["initial_intent_requests", "initial_rewrite_requests"])
def test_changed_first_payload_and_recomputed_hashes_cannot_be_refrozen(paused, kind):
    trial, _ = paused
    path = trial.root / "protocol.json"
    value = bench.read_json(path)
    first = value[kind][0]
    first["payload"]["messages"][0]["content"] += " controlled mutation"
    raw = json.dumps(first["payload"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    first["payload_sha256"] = original.canonical_payload_hash(first["payload"])
    first["request_bytes_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(value))
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="original_context_changed|first_request_changed|approval_changed"):
        asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    assert bytes_under(trial.root) == before


@pytest.mark.parametrize("pair", ["only_data", "only_delegate", "neither"])
def test_temporary_root_needs_both_controlled_data_and_transport_before_claim(continued, pair):
    trial, _ = continued
    kwargs = ({"controlled_data": Data()} if pair == "only_data" else
              {"delegate_factory": lambda frame: (_ for _ in ()).throw(AssertionError("no transport"))}
              if pair == "only_delegate" else {})
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="controlled_pair|required|root_mismatch"):
        asyncio.run(trial.execute(**kwargs))
    assert bytes_under(trial.root) == before


def test_controlled_transport_cannot_seal_live_data_or_retry_a_rejected_data_kind(continued):
    trial, before = continued
    data = Data()
    data.proof["kind"] = "live-mysql-and-isolated-chroma"
    with pytest.raises(Exception, match="controlled_data_required"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                  controlled_data=data))
    assert data.closed
    assert not (trial.root / "data-proof.json").exists()
    assert (trial.root / module.STARTED_FILE).exists()
    assert trial.ledger(trial.verify()).status()["attempts"] == 0
    assert (trial.root / "worker-stop.json").read_bytes() == before["worker-stop.json"]
    with pytest.raises(Exception, match="already_started|paused"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no retry")),
                                  controlled_data=Data()))


def test_full_sixteen_sdk_jobs_finish_with_old_stop_preserved_and_semantics_unreviewed(continued, monkeypatch):
    trial, before = continued
    coro, data, calls = native_run(trial)
    report = asyncio.run(coro)
    assert report == trial.report()
    assert report["completed_jobs"] == 16 and report["execution_complete"] and not report["paused"]
    assert report["reserved_requests"] == report["sealed_physical"] == len(calls) < 96
    assert report["authorized_requests"] == report["maximum_requests"] == 96
    assert report["pending_physical"] == 0 and not report["unknown_jobs"] and not report["not_executed_jobs"]
    assert report["artifact_evidence_kinds"] == {"controlled-transport": 16} and data.closed
    assert report["semantic_task_reviews"] == 0 and report["review_ready"]
    assert report["comparison_ready"] is report["real_comparison_ready"] is report["quality_pass_claim"] is False
    assert all(row["status"] == "pending_review" for row in report["task_rows"])
    assert all(group["correct_handling"] == {"numerator": 0, "denominator": 4}
               for group in report["task_groups"].values())
    assert all(row["status"] == "recorded" and row["physical_structure_valid"]
               and row["physical_outcomes"] == ["received", "received"] for row in report["retrieval_rows"])
    assert {frame.case_id for frame, _, _ in calls} == set(runtime.TASK_IDS + runtime.RETRIEVAL_IDS)
    assert not any(frame.phase in {"judge", "embedding", "memory"} for frame, _, _ in calls)
    p = trial.verify()
    for name in ("initial_intent_requests", "initial_rewrite_requests"):
        for frozen in p[name]:
            first = next(raw for frame, raw, _ in calls if frame.case_id == frozen["case_id"])
            assert hashlib.sha256(first).hexdigest() == frozen["request_bytes_sha256"]
    for frame, raw, extensions in calls:
        assert extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), float(PHASE_DEADLINES[frame.phase]))
        sealed = bench.read_json(trial.root / f"request-{hashlib.sha256(raw).hexdigest()}.json")
        assert sealed["wire_text"].encode() == raw
        assert sealed["payload_sha256"] == original.canonical_payload_hash(json.loads(raw))
        assert frame.case_id not in raw.decode() and "required_facts" not in raw.decode()
    for identity in runtime.TASK_IDS:
        artifact = bench.read_json(trial.root / f"case-{identity}.json")
        assert artifact["semantic_review"] is artifact["judge"] is None and artifact["quality_pass_claim"] is False
        for call in artifact["turns"][0]["tool_evidence"]["calls"]:
            assert call["tool_use_id"] and call["trusted_customer_id"] == "demo-customer-a"
            assert call["request_id"] == f"repair-task-{identity}-1"
    assert (trial.root / "ledger/journal.jsonl").read_bytes().startswith(before["ledger/journal.jsonl"])
    assert_unchanged(trial.root, {name: raw for name, raw in before.items() if name != "ledger/journal.jsonl"})
    frozen = bytes_under(trial.root)
    with pytest.raises(Exception, match="already_started|consumed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no replay")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == frozen
    for identity in runtime.TASK_IDS:
        review(trial, identity)
    controlled_review = trial.report()
    assert controlled_review["semantic_task_reviews"] == 8 and controlled_review["comparison_ready"]
    assert controlled_review["real_comparison_ready"] is controlled_review["quality_pass_claim"] is False
    assert controlled_review["additional_authorized_requests"] == 0
    assert (trial.root / "authorization.json").read_bytes() == before["authorization.json"]
    # A later README-only commit can inspect completed evidence. The Git-only
    # helper is stubbed, while completion, cleanup, free lease and strict send
    # admission remain actual.
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    read_only_before = bytes_under(trial.root)
    assert trial.verify() == p and trial.report() == controlled_review
    assert bytes_under(trial.root) == read_only_before
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no new version send")),
                                  controlled_data=Data()))
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: False)
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        trial.report()
    assert bytes_under(trial.root) == read_only_before


def test_missing_native_execution_remains_failed_while_the_fixed_plan_finishes(continued):
    trial, _ = continued
    coro, data, calls = native_run(trial, refused_case="RT02")
    report = asyncio.run(coro)
    failed = next(row for row in report["task_rows"] if row["id"] == "RT02")
    assert failed["status"] == failed["tool_status"] == "failed"
    assert failed["checks"]["checks"]["required_tools"] is False
    assert report["completed_jobs"] == 16 and not report["paused"] and data.closed
    assert any(frame.case_id == "RR08" for frame, _, _ in calls)
    assert report["quality_pass_claim"] is False


def test_unknown_send_stays_spent_and_new_failure_receipt_preserves_original_stop(continued):
    trial, before = continued
    coro, data, calls = native_run(trial, failure=httpx.ReadTimeout("controlled unknown send"))
    with pytest.raises(Exception):
        asyncio.run(coro)
    report = trial.report()
    assert report["reserved_requests"] == len(calls) == 1
    assert report["paused"] and report["completed_jobs"] == 0 and data.closed
    assert report["quality_pass_claim"] is report["comparison_ready"] is False
    assert (trial.root / "worker-stop.json").read_bytes() == before["worker-stop.json"]
    receipts = [path for path in trial.root.glob("*stop*.json") if path.name != "worker-stop.json"]
    assert receipts, "new continuation failure needs its own immutable stop"
    frozen = bytes_under(trial.root)
    with pytest.raises(Exception, match="already_started|paused|consumed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no retry")),
                                  controlled_data=Data()))
    with pytest.raises(Exception):
        asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    assert bytes_under(trial.root) == frozen


def test_claim_is_consumed_even_if_data_fails_before_the_first_model_request(continued, monkeypatch):
    trial, before = continued
    data, sends = Data(), []
    data.fault = "controlled_zero_send_data_failure"
    with pytest.raises(Exception, match="controlled_zero_send_data_failure"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: sends.append(frame), controlled_data=data))
    assert sends == [] and trial.ledger(trial.verify()).status()["attempts"] == 0
    assert (trial.root / "worker-stop.json").read_bytes() == before["worker-stop.json"]
    frozen = bytes_under(trial.root)
    with pytest.raises(Exception, match="already_started|paused|consumed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no reopen")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == frozen
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    report = trial.report()
    assert report["reserved_requests"] == 0 and report["paused"] and not report["execution_complete"]
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no new version retry")),
                                  controlled_data=Data()))
    assert bytes_under(trial.root) == frozen


def test_readme_only_head_cannot_relax_a_continuation_before_it_starts(continued, monkeypatch):
    trial, _ = continued
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        trial.verify()
    assert bytes_under(trial.root) == before


def seal_started_cleanup(trial):
    protocol = trial.verify()
    original.base._write(trial.root, module.STARTED_FILE, {
        **trial._started_binding(protocol), "recorded_at": "2026-11-05T00:00:00+00:00"})
    original.base._write(trial.root, "cleanup-complete.json", {"owned_resources_closed": True,
        "protocol_sha256": bench.sha256(trial.root / "protocol.json")})
    return protocol


@pytest.mark.parametrize("pending", [False, True])
def test_readme_only_head_cannot_hide_an_active_case_or_unknown_pending_send(continued, monkeypatch, pending):
    trial, _ = continued
    p = seal_started_cleanup(trial)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("RT01", lease=lease)
        if pending:
            ledger.reserve("RT01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        trial.report()
    assert bytes_under(trial.root) == before
    state = ledger.status()
    assert state["active_case"] == "RT01" and state["attempts"] == int(pending)


def test_readme_only_head_requires_the_original_worker_lease_to_be_free(continued, monkeypatch):
    trial, _ = continued
    p = seal_started_cleanup(trial)
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    before = bytes_under(trial.root)
    with trial.ledger(p).exclusive_run():
        with pytest.raises(Exception):
            trial.verify()
    assert bytes_under(trial.root) == before


def test_readme_only_head_does_not_relax_a_started_worker_without_cleanup(continued, monkeypatch):
    trial, _ = continued
    seal_started_cleanup(trial)
    (trial.root / "cleanup-complete.json").unlink()
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        trial.report()
    assert bytes_under(trial.root) == before


def test_run_never_uses_read_only_documentation_head_compatibility(continued, monkeypatch):
    trial, _ = continued
    seal_started_cleanup(trial)
    monkeypatch.setattr(module, "_git_head", lambda: "controlled-readme-only-head")
    monkeypatch.setattr(module, "_documentation_head_changed_only", lambda manifest: True)
    # Give only this temporary-root fixture the trusted CLI identity. This
    # never opens or invokes run/execute against the actual repair ROOT.
    monkeypatch.setattr(module, "ROOT", trial.root)
    monkeypatch.setattr(module, "ORIGINAL_PROTOCOL_SHA256", bench.sha256(trial.root / "protocol.json"))
    monkeypatch.setattr(module.Path, "cwd", classmethod(lambda cls: bench.BACKEND_ROOT))
    monkeypatch.setattr(supervisor, "supervise", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no worker")))
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="binding_changed|head_changed"):
        trial.run()
    assert bytes_under(trial.root) == before


def test_continuation_cannot_replace_or_extend_the_original_96_request_grant(paused):
    trial, _ = paused
    path = trial.root / "authorization.json"
    approval = bench.read_json(path)
    approval["max_requests"] = 1
    path.write_text(json.dumps(approval))
    before = bytes_under(trial.root)
    with pytest.raises(Exception, match="original_96_required"):
        asyncio.run(trial.prepare_continuation(REFERENCE, confirmed=True))
    with pytest.raises(Exception, match="original_authorization_only"):
        trial.authorize("no supplemental grant", max_requests=96, confirmed=True)
    assert bytes_under(trial.root) == before
    assert len(list(trial.root.glob("*authorization*.json"))) == 1


@pytest.mark.parametrize("deadline", [900, .1])
def test_continuation_supervisor_has_one_fixed_root_and_at_most_900_seconds(tmp_path, monkeypatch, deadline):
    # Only the fixed-root identity is examined. The actual root is never opened
    # and a process is never spawned.
    opened, launched, waits = [], [], []
    def controlled_open(root):
        opened.append(root)
        return os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    class Process:
        def wait(self, *, timeout):
            waits.append(timeout)
            return 0
    def controlled_popen(command, **kwargs):
        launched.append((command, kwargs))
        return Process()
    monkeypatch.setattr(supervisor, "_open_root", controlled_open)
    monkeypatch.setattr(supervisor.subprocess, "Popen", controlled_popen)
    assert supervisor.supervise(continuation_command(), root=original.ROOT, deadline_s=deadline) == {
        "exit_code": 0, "timed_out": False, "cancelled": False}
    assert opened == [original.ROOT] and waits == [float(deadline)]
    assert launched == [(continuation_command(), {"shell": False, "start_new_session": True,
        "stdout": supervisor.subprocess.DEVNULL, "stderr": supervisor.subprocess.DEVNULL, "close_fds": True})]


@pytest.mark.parametrize("deadline", [901, 3600, 0, -1, float("nan"), float("inf"), True])
def test_continuation_supervisor_rejects_invalid_deadlines_before_resources(monkeypatch, deadline):
    def denied(*args, **kwargs):
        raise AssertionError("invalid worker must not open roots or spawn")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_deadline_invalid"):
        supervisor.supervise(continuation_command(), root=original.ROOT, deadline_s=deadline)


@pytest.mark.parametrize("damage", ["other_root", "extra_arg", "different_python", "missing_flag"])
def test_continuation_supervisor_rejects_altered_command_or_root(tmp_path, monkeypatch, damage):
    def denied(*args, **kwargs):
        raise AssertionError("invalid worker must not open roots or spawn")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    command, root = continuation_command(), original.ROOT
    if damage == "other_root":
        root = tmp_path
    elif damage == "extra_arg":
        command.append("--resume")
    elif damage == "different_python":
        command[0] = "/usr/bin/python3"
    else:
        command.remove("-B")
    with pytest.raises(supervisor.SupervisorError, match="root_invalid|fixed_worker_required"):
        supervisor.supervise(command, root=root, deadline_s=900)


@pytest.mark.parametrize("worker", ["task_trial", "task_deadline_recheck", "repair_execution_trial"])
def test_continuation_hook_keeps_earlier_worker_deadlines_at_900(monkeypatch, worker):
    def denied(*args, **kwargs):
        raise AssertionError("invalid deadline cannot reach root or process")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    command = [sys.executable, "-B", "-u", "-m", f"evaluation.{worker}", "_worker"]
    with pytest.raises(supervisor.SupervisorError, match="supervisor_deadline_invalid"):
        supervisor.supervise(command, root=original.ROOT, deadline_s=901)


@pytest.mark.parametrize("worker", ["current_task_ablation", "expanded_trial", "repair_execution_trial"])
def test_continuation_hook_keeps_earlier_fixed_root_guards(tmp_path, monkeypatch, worker):
    def denied(*args, **kwargs):
        raise AssertionError("invalid old root cannot be opened")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    command = [sys.executable, "-B", "-u", "-m", f"evaluation.{worker}", "_worker"]
    with pytest.raises(supervisor.SupervisorError, match="worker_root_invalid"):
        supervisor.supervise(command, root=tmp_path, deadline_s=900)
