"""Controlled local-retrieval runner checks; never index or contact real services."""
from copy import deepcopy
import hashlib
import json
import socket

import pytest

from evaluation import retrieval_baseline as module
from evaluation import benchmark as bench, intent_trial as private


@pytest.fixture(autouse=True)
def controlled(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("baseline unit tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    proof = {name: "controlled-" + name for name in module.KNOWLEDGE_FIELDS}
    monkeypatch.setattr(module, "_knowledge_proof", lambda: deepcopy(proof))
    monkeypatch.setattr(module, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled-source.py": "a" * 64})
    return proof


def row(title="配送说明"):
    content = next(doc["content"] for doc in module.task_data._policies() if doc["title"] == title)
    identity = json.dumps(["title", title], ensure_ascii=True)
    return {"title": title, "content": content, "score": .123,
            "document_id": hashlib.sha256(identity.encode()).hexdigest(),
            "version": hashlib.sha256(content.encode()).hexdigest(), "chunk": 0}


class Runtime:
    def __init__(self, proof, handler=None):
        self.proof = deepcopy(proof)
        self.kb = self
        self.calls, self.closed = [], False
        self.handler = handler or (lambda query: [row()])

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.handler(query)

    def close(self):
        self.closed = True


def prepared(tmp_path):
    value = module.RetrievalBaseline(tmp_path / "new-baseline")
    return value, value.prepare()


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_prepare_and_empty_report_are_zero_inference_with_fixed_split_and_original_queries(tmp_path):
    baseline, protocol = prepared(tmp_path)
    assert protocol["model_api_requests"] == protocol["external_business_requests"] == 0
    assert len(protocol["plan"]) == 24 and protocol["top_k"] == 3
    assert [p["case_id"] for p in protocol["plan"]] == [f"r{i:02d}" for i in range(1, 25)]
    report = baseline.report()
    assert all(part["planned_total"] == part["not_executed"] == 12 for part in report["by_split"].values())
    assert not (baseline.root / "measurement-start.json").exists()
    before = tree(baseline.root)
    with pytest.raises(module.RetrievalBaselineError, match="root_already_exists"):
        baseline.prepare()
    assert tree(baseline.root) == before


def test_actual_handler_results_and_false_retrievals_are_kept_separate_from_answer_quality(tmp_path, controlled):
    baseline, p = prepared(tmp_path)
    runtime = Runtime(controlled)
    result = baseline.measure(knowledge_factory=lambda: runtime)
    assert runtime.closed and len(runtime.calls) == 24
    assert runtime.calls == [(case["input"]["message"], 3) for case in module._cases()]
    assert len(result["observations"]) == 24
    assert all(value["evidence_kind"] == "controlled-fixture" for value in result["observations"])
    assert all(value["results"][0]["content_sha256"] == value["results"][0]["version"] for value in result["observations"])
    for split in result["by_split"].values():
        assert split["recorded_total"] == 12 and split["not_executed"] == 0
        # All 3 out-of-scope items still retrieve an unrelated policy. No
        # threshold, label-driven rejection or evidence omission is introduced.
        assert split["no_evidence"]["mis_hit"]["numerator"] == 3
        assert split["no_evidence"]["mis_hit"]["denominator"] == 3
    assert result["model_api_requests"] == 0
    assert result["execution_state"] == "completed" and result["owned_collection_cleanup_complete"]
    before = tree(baseline.root)
    with pytest.raises(module.RetrievalBaselineError, match="already_started"):
        baseline.measure(knowledge_factory=lambda: pytest.fail("cannot rebuild or repeat"))
    assert tree(baseline.root) == before


@pytest.mark.parametrize("damage", ["source", "case", "copied_root", "model_cache"])
def test_frozen_changes_reject_before_index_or_query(tmp_path, monkeypatch, damage):
    baseline, p = prepared(tmp_path)
    if damage == "source":
        monkeypatch.setattr(module, "_artifacts", lambda: {"changed.py": "b" * 64})
    elif damage == "model_cache":
        monkeypatch.setattr(module, "_knowledge_proof", lambda: {"changed": "cache"})
    elif damage == "case":
        value = deepcopy(p); value["plan"][0]["input_sha256"] = "b" * 64
        (baseline.root / "protocol.json").write_text(json.dumps(value))
    else:
        import shutil
        dest = tmp_path / "copied"; shutil.copytree(baseline.root, dest)
        baseline = module.RetrievalBaseline(dest)
    with pytest.raises(module.RetrievalBaselineError, match="frozen_protocol_changed"):
        baseline.measure(knowledge_factory=lambda: pytest.fail("no index on invalid protocol"))
    assert not (baseline.root / "measurement-start.json").exists()


@pytest.mark.parametrize("change", [{"score": float("nan")}, {"chunk": True},
    {"document_id": "fake-id"}, {"version": "a" * 64}, {"title": "其他来源"},
    {"content": "改写后的正文"}, {"extra": "must not ignore"}])
def test_invalid_source_rows_are_rejected_without_repair(change):
    source = row()
    source.update(change)
    with pytest.raises(module.RetrievalBaselineError):
        module._rows([source])


def test_query_failure_keeps_denominator_and_does_not_save_exception_message(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    count = 0
    def handler(query):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("private SQL password=do-not-store")
        return []
    runtime = Runtime(controlled, handler)
    result = baseline.measure(knowledge_factory=lambda: runtime)
    first = result["observations"][0]
    assert first["status"] == "failed" and first["error_code"] == "retrieval_query_failed"
    assert result["by_split"]["development"]["status_counts"]["failed"] == 1
    assert result["by_split"]["development"]["planned_total"] == 12
    assert "private SQL" not in json.dumps(result) and runtime.closed


def test_interruption_retains_unknown_query_and_prevents_reexecution(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    runtime = Runtime(controlled, lambda query: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        baseline.measure(knowledge_factory=lambda: runtime)
    assert runtime.closed and (baseline.root / "measurement-stop.json").exists()
    report = baseline.report()
    dev = report["by_split"]["development"]
    assert dev["status_counts"]["unknown"] == 1 and dev["not_executed"] == 11
    with pytest.raises(module.RetrievalBaselineError, match="already_started"):
        baseline.measure(knowledge_factory=lambda: runtime)


def test_changed_sealed_output_cannot_be_reported_as_new_measurement(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    baseline.measure(knowledge_factory=lambda: Runtime(controlled))
    path = baseline.root / "case-r01.json"
    value = bench.read_json(path); value["ranked_titles"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(module.RetrievalBaselineError, match="record_changed"):
        baseline.report()


def test_network_access_is_rejected_with_no_retry_and_collection_cleanup(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    runtime = Runtime(controlled, lambda query: socket.create_connection(("127.0.0.1", 1)))
    with pytest.raises(module.RetrievalBaselineError, match="network_forbidden"):
        baseline.measure(knowledge_factory=lambda: runtime)
    assert len(runtime.calls) == 1 and runtime.closed
    assert (baseline.root / "measurement-stop.json").exists()


def test_real_measure_requires_disabled_native_telemetry_before_claim_or_index(tmp_path, monkeypatch):
    baseline, _ = prepared(tmp_path)
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(module.task_data, "_build_knowledge", lambda: pytest.fail("must not start index"))
    with pytest.raises(module.RetrievalBaselineError, match="disabled_onnx_telemetry_required"):
        baseline.measure()
    assert not (baseline.root / "measurement-start.json").exists()


def test_changed_index_proof_stops_before_queries_and_cleans_owned_collection(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    runtime = Runtime({**controlled, "corpus_sha256": "changed-corpus"})
    with pytest.raises(module.RetrievalBaselineError, match="index_identity_changed"):
        baseline.measure(knowledge_factory=lambda: runtime)
    assert runtime.closed and runtime.calls == []
    assert (baseline.root / "measurement-stop.json").exists()
    assert all(part["not_executed"] == 12 for part in baseline.report()["by_split"].values())


def test_cleanup_failure_is_safe_and_cannot_claim_successful_completed_report(tmp_path, controlled):
    baseline, _ = prepared(tmp_path)
    runtime = Runtime(controlled)
    def failure():
        raise RuntimeError("private path or driver credentials")
    runtime.close = failure
    with pytest.raises(module.RetrievalBaselineError, match="retrieval_cleanup_failed") as caught:
        baseline.measure(knowledge_factory=lambda: runtime)
    assert "private" not in str(caught.value)
    assert (baseline.root / "cleanup-failed.json").exists() and not (baseline.root / "report.json").exists()
    assert baseline.report()["execution_state"] == "cleanup_failed"
