"""Trial freeze/admission/report tests with fake proofs and controlled transport.

No MySQL, Chroma, model service, dotenv or application endpoint is contacted.
Runtime behavior is separately checked by test_task_runtime; mock case results
here exercise reporting gates, never establish answer quality.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation import task_trial as module
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import Frame, context

ORIGINAL_PRIOR_PROOF = module.prior_proof

BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
KEY = "controlled-only-key-never-persist"
OFFLINE = {"schema_version": "controlled-offline-proof", "mysql_preflight": "required-not-performed",
           "fixture_sha256": "a" * 64, "corpus_sha256": "b" * 64}
PRIOR = {"controlled_prior_chain": "c" * 64}


@pytest.fixture(autouse=True)
def controlled(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("trial tests prohibit external sockets")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    config_reads = []
    def configuration(*, secret=False):
        config_reads.append(secret)
        return {"base_url": BASE, "model": base.MODEL, **({"api_key": KEY} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"model": base.MODEL, "max_tokens": 256, "temperature": .1,
                   "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": case["input"]["message"]}]}
        return {"case_id": case["id"], "operation_id": "controlled-offline-capture",
                "input_sha256": canonical_payload_hash(case["input"]), "payload": payload,
                "payload_sha256": canonical_payload_hash(payload),
                "request_bytes_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_dependencies", lambda: {"controlled-dependency": "1"})
    monkeypatch.setattr(module, "_dependencies", lambda: {"controlled-runtime-dependency": "1"})
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled.py": "d" * 64})
    monkeypatch.setattr(module, "offline_data_proof", lambda: deepcopy(OFFLINE))
    monkeypatch.setattr(module, "prior_proof", lambda: deepcopy(PRIOR))
    monkeypatch.setattr(module, "_mysql_url", lambda: None)
    return SimpleNamespace(config_reads=config_reads)


def prepare(tmp_path):
    trial = module.TaskTrial(tmp_path / "batch")
    return trial, asyncio.run(trial.prepare())


def grant(trial):
    return trial.authorize("Controlled test authorization only; not a real send allowance.", confirmed=True)


def corrupt(path, value):
    # Fault injection only; the actual writer is intentionally non-overwriting.
    path.write_text(json.dumps(value), encoding="utf-8")


class ControlledData:
    proof = {"kind": "controlled-fixture", "source": "trial-report-gate-test"}
    def __init__(self):
        self.closed = False
    def assert_healthy(self):
        assert not self.closed
    async def close(self):
        self.closed = True


def controlled_run(trial, monkeypatch, *, blocked=None):
    """Real physical ledger/SDK path, mock report artifacts, zero external sends."""
    sends, executions = [], []
    async def execute_case(case, sdk, data, *, ensure_healthy):
        executions.append(case["id"])
        turns = []
        for number, question in enumerate(case["turns"], 1):
            ensure_healthy()
            with context(Frame(case["id"], f"turn-{number}", "intent")):
                await sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": question}],
                    extra_body={"thinking": {"type": "disabled"}})
            turns.append({"turn_id": f"turn-{number}", "judge_overall": 1.,
                          "judge": {"judge_failed": False}})
        failed = case["id"] == blocked
        return {"schema_version": "serveflow-task-case-result-v1", "case_id": case["id"],
                "execution_status": "success", "deterministic": {
                    "checks": {"business_evidence": not failed},
                    "blockers": ["business_evidence"] if failed else []},
                "turns": turns, "judge_ready": True, "data_kind": "controlled-fixture",
                "semantic_review": None, "answer": "Controlled result; no semantic quality claim."}
    def factory(frame):
        async def handle(request):
            sends.append(frame)
            assert KEY.encode() not in request.content
            return httpx.Response(200, json={"id": "controlled-response", "type": "message", "role": "assistant",
                "model": base.MODEL, "content": [{"type": "text", "text": "controlled"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2}})
        return httpx.MockTransport(handle)
    monkeypatch.setattr(module, "execute_case", execute_case)
    data = ControlledData()
    output = asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert data.closed
    return output, sends, executions


def review(trial, case_id, *, digest=None):
    return {"case_id": case_id,
            "artifact_sha256": digest or bench.sha256(trial.root / f"case-{case_id}.json"),
            "reviewer": "controlled-test-reviewer", "method": "controlled-fixture",
            "dimensions": dict.fromkeys(module.REVIEW_FIELDS, True), "reason": "Controlled input for report gate.",
            "blocking_errors": [], "reading_burden": "none"}


def test_prepare_has_zero_authorization_and_preserves_all_six_denominators(tmp_path):
    trial, protocol = prepare(tmp_path)
    assert protocol["authorized_model_requests"] == 0
    assert protocol["proposed_model_requests"] == 168
    assert protocol["offline_data_proof"] == OFFLINE and protocol["prior_proof"] == PRIOR
    assert len(protocol["plan"]) == len(protocol["initial_intent_requests"]) == 6
    assert sum(len(p["turn_ids"]) for p in protocol["plan"]) == 7
    assert not (trial.root / "authorization.json").exists()
    assert trial.verify() == protocol
    output = trial.report()
    assert output["ledger"]["attempts"] == output["authorized_requests"] == 0
    assert output["counts"]["not_executed"] == output["planned_cases"] == 6
    assert output["planned_turns"] == 7 and sum(output["counts"].values()) == 6
    for case, item in zip(module.cases(), protocol["initial_intent_requests"]):
        body = json.dumps(item["payload"], ensure_ascii=False)
        assert case["expected"]["rationale"] not in body and "required_facts" not in body
    before = (trial.root / "protocol.json").read_bytes()
    with pytest.raises(base.TrialError, match="task_root_already_exists"):
        asyncio.run(trial.prepare())
    assert (trial.root / "protocol.json").read_bytes() == before


def test_execute_without_authorization_never_loads_secret_or_data(tmp_path, monkeypatch, controlled):
    trial, protocol = prepare(tmp_path)
    data = AsyncMock()
    model = AsyncMock()
    monkeypatch.setattr(module, "build_live_data", data)
    monkeypatch.setattr(module, "execute_case", model)
    controlled.config_reads.clear()
    with pytest.raises(FileNotFoundError):
        asyncio.run(trial.execute())
    assert not any(controlled.config_reads)
    data.assert_not_called()
    model.assert_not_called()
    assert trial.ledger(protocol).status()["attempts"] == 0


def test_missing_independent_eval_database_preflight_and_run_have_zero_calls(tmp_path, monkeypatch):
    trial, protocol = prepare(tmp_path)
    data, model = AsyncMock(), AsyncMock()
    monkeypatch.setattr(module, "build_live_data", data)
    monkeypatch.setattr(module, "execute_case", model)
    assert asyncio.run(trial.preflight()) == {
        "ready": False, "reason": "dedicated_eval_mysql_not_configured", "model_requests": 0}
    grant(trial)
    with pytest.raises(base.TrialError, match="dedicated_eval_mysql_not_configured"):
        asyncio.run(trial.execute())
    data.assert_not_called()
    model.assert_not_called()
    assert trial.ledger(protocol).status()["attempts"] == 0
    assert (trial.root / "worker-stop.json").exists()


@pytest.mark.parametrize("reason,expected", [
    ("evidence_invalid_json", "evidence_invalid_json"),
    ("private payload, customer text or SQL must not be persisted", "evidence_invalid_json"),
])
def test_capture_failure_persists_only_safe_reason_and_prevents_restart(tmp_path, monkeypatch, reason, expected):
    from evaluation.task_evidence import EvidenceError
    trial, protocol = prepare(tmp_path)
    grant(trial)
    async def fail(*_args, **_kwargs):
        raise EvidenceError(reason)
    monkeypatch.setattr(module, "execute_case", fail)
    data = ControlledData()
    with pytest.raises(EvidenceError):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None),
                                 controlled_data=data))
    stopped = bench.read_json(trial.root / "worker-stop.json")
    assert stopped["evidence_error"] == expected
    assert "private payload" not in json.dumps(stopped)
    assert data.closed and trial.ledger(protocol).status()["attempts"] == 0
    with pytest.raises(base.TrialError):
        asyncio.run(trial.execute(controlled_data=ControlledData()))


@pytest.mark.parametrize("kind", ["artifacts", "dependencies", "offline_proof", "prior_proof", "initial_payload", "initial_wire_hash", "authorization"])
def test_frozen_or_approval_tampering_blocks_execution_before_data(tmp_path, monkeypatch, kind):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    before_ledger = {str(path.relative_to(trial.root)): path.read_bytes()
                     for path in (trial.root / "ledger").rglob("*") if path.is_file()}
    data = AsyncMock()
    monkeypatch.setattr(module, "build_live_data", data)
    if kind == "artifacts":
        monkeypatch.setattr(module, "_artifacts", lambda: {"controlled.py": "e" * 64})
    elif kind == "dependencies":
        monkeypatch.setattr(module, "_dependencies", lambda: {"controlled-runtime-dependency": "2"})
    elif kind == "offline_proof":
        monkeypatch.setattr(module, "offline_data_proof", lambda: {**OFFLINE, "fixture_sha256": "e" * 64})
    elif kind == "prior_proof":
        monkeypatch.setattr(module, "prior_proof", lambda: {"controlled_prior_chain": "e" * 64})
    elif kind == "authorization":
        path = trial.root / "authorization.json"
        value = bench.read_json(path)
        value["max_requests"] = 169
        corrupt(path, value)
    else:
        path = trial.root / "protocol.json"
        value = deepcopy(protocol)
        if kind == "initial_payload":
            value["initial_intent_requests"][0]["payload"]["messages"][0]["content"] = "tampered question"
        else:
            value["initial_intent_requests"][0]["request_bytes_sha256"] = "e" * 64
        corrupt(path, value)
    with pytest.raises((base.TrialError, TaskBudgetError)):
        asyncio.run(trial.execute())
    data.assert_not_called()
    # The original ledger remains untouched even if its protocol hash no longer
    # matches the deliberately corrupted protocol file.
    assert {str(path.relative_to(trial.root)): path.read_bytes()
            for path in (trial.root / "ledger").rglob("*") if path.is_file()} == before_ledger


def test_six_mock_results_judge_full_score_remains_pending_and_hard_failure_wins(tmp_path, monkeypatch):
    trial, protocol = prepare(tmp_path)
    grant(trial)
    output, sends, executions = controlled_run(trial, monkeypatch, blocked="e05")
    assert executions == [f"e{number:02d}" for number in range(1, 7)]
    assert len(sends) == output["ledger"]["attempts"] == 7
    assert all(row["judge_scores"] == [1.] * row["planned_turns"] for row in output["cases"])
    assert output["counts"] == {"passed": 0, "failed": 1, "evaluation_failed": 0,
                                 "pending_review": 5, "unknown": 0, "not_executed": 0}
    assert sum(output["counts"].values()) == 6
    for value in module.cases():
        trial.review(review(trial, value["id"]))
    after = trial.report()
    assert after["counts"]["passed"] == 5 and after["counts"]["failed"] == 1
    assert next(row for row in after["cases"] if row["case_id"] == "e05")["status"] == "failed"
    assert KEY not in "\n".join(p.read_text() for p in trial.root.rglob("*") if p.is_file())


@pytest.mark.parametrize("kind", ["physical", "case"])
def test_sealed_output_hash_is_rechecked_and_rejects_mutation(tmp_path, monkeypatch, kind):
    trial, _ = prepare(tmp_path)
    grant(trial)
    controlled_run(trial, monkeypatch)
    path = trial.root / ("physical-001.json" if kind == "physical" else "case-e01.json")
    value = bench.read_json(path)
    value["controlled_tamper"] = True
    corrupt(path, value)
    with pytest.raises(base.TrialError, match="task_physical_changed" if kind == "physical" else "task_case_changed"):
        trial.verify()


def test_review_is_bound_to_case_artifact_and_cannot_be_overwritten(tmp_path, monkeypatch):
    trial, _ = prepare(tmp_path)
    grant(trial)
    controlled_run(trial, monkeypatch)
    with pytest.raises(base.TrialError, match="task_review_binding"):
        trial.review(review(trial, "e01", digest="e" * 64))
    accepted = review(trial, "e01")
    trial.review(accepted)
    assert trial.report()["counts"]["passed"] == 1
    with pytest.raises(FileExistsError):
        trial.review(accepted)
    path = trial.root / "review-e01.json"
    value = bench.read_json(path)
    value["artifact_sha256"] = "e" * 64
    corrupt(path, value)
    with pytest.raises(base.TrialError, match="task_review_binding"):
        trial.report()


def test_old_batch_or_copied_directory_cannot_become_this_trial(tmp_path):
    historical = bench.BACKEND_ROOT / "data/benchmarks/intent-recheck-v2"
    with pytest.raises(base.TrialError, match="fixed_task_root_required"):
        module.TaskTrial(historical)
    trial, _ = prepare(tmp_path)
    grant(trial)
    copied = tmp_path / "copied"
    shutil.copytree(trial.root, copied)
    with pytest.raises(base.TrialError, match="task_protocol_changed"):
        module.TaskTrial(copied).verify()


def test_prior_proof_reads_existing_chain_named_report_without_writing(tmp_path, monkeypatch):
    from evaluation import intent_recheck, intent_validation
    root = tmp_path / "prior"; root.mkdir(mode=0o700)
    state = {"attempts": 24, "pending": [], "paused": False, "chain_hash": "f" * 64}
    report = {"ledger": state, "paired_cases": []}
    protocol = {"archived_evidence_proof": {"older_ledgers": "unchanged"}}
    for name, value in (("protocol.json", protocol), ("authorization.json", {"old_allowance": 24}),
                        ("report-" + "f" * 64 + ".json", report)):
        base._write(root, name, value)
    monkeypatch.setattr(intent_recheck, "ROOT", root)
    monkeypatch.setattr(intent_recheck, "RecheckTrial", lambda path: SimpleNamespace(
        verify=lambda: deepcopy(protocol), report=lambda: deepcopy(report)))
    monkeypatch.setattr(intent_validation, "_ledger_snapshot", lambda path: "a" * 64)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    # Keep this read-only proof check independent of the fake prior proof used
    # by other admission tests; do not reload production modules or old ledgers.
    proof = ORIGINAL_PRIOR_PROOF()
    assert proof["chain_hash"] == "f" * 64
    assert "report-" + "f" * 64 + ".json" in proof["hashes"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    bad = deepcopy(report); bad["paired_cases"] = ["tampered"]
    saved = root / ("report-" + "f" * 64 + ".json")
    saved.write_text(json.dumps(bad)); saved.chmod(0o600)
    with pytest.raises(base.TrialError, match="prior_report_changed"):
        ORIGINAL_PRIOR_PROOF()


def test_review_dimensions_burden_and_explicit_blockers_are_separate(tmp_path, monkeypatch):
    from evaluation.benchmark_metrics import REVIEW_DIMENSIONS
    trial, _ = prepare(tmp_path)
    grant(trial)
    controlled_run(trial, monkeypatch)
    value = review(trial, "e01")
    value["blocking_errors"] = ["controlled factual contradiction"]
    value["reading_burden"] = "heavy"
    trial.review(value)
    result = trial.report()
    row = next(c for c in result["cases"] if c["case_id"] == "e01")
    assert row["status"] == "failed" and row["judge_scores"] == [1.0]
    assert row["semantic_review"]["reading_burden"] == "heavy"
    assert row["semantic_review"]["method"] == "controlled-fixture"
    assert set(result["review_dimensions"]) == set(REVIEW_DIMENSIONS)
    assert all(v == {"true_count": 1, "reviewed_count": 1, "planned_denominator": 6}
               for v in result["review_dimensions"].values())
    assert result["physical_resources"]["latency_ms"]["method"] == "nearest-rank"
