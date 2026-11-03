"""Historical-response replay and local methods only; no service access."""
import asyncio
from copy import deepcopy
import json
import socket

import pytest

from core.intent_recognizer import IntentCategory
from evaluation import benchmark as bench, intent_ablation as module, intent_trial as base


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("intent ablation tests prohibit external services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(base, "_configuration", lambda **_: pytest.fail("do not read model configuration"))


def case(message="请核对退款进度", expected="refund", partition="new_challenge", case_id="c1"):
    return {"id": case_id, "partition": partition, "input": {"message": message, "history": []},
            "expected": {"intent": expected}}


def controlled_observation(value, label="refund", confidence=.95):
    """Prepare a genuine SDK-memory payload, then pure local recorded decisions."""
    async def prepare():
        operation = await base._capture(value, "https://dashscope.aliyuncs.com/apps/anthropic")
        text = json.dumps({"intent": label, "confidence": confidence, "reasoning": "controlled fixture"})
        client = type("Client", (), {"base_url": "https://dashscope.aliyuncs.com/apps/anthropic"})()
        recognizer = module._local_recognizer(client)
        embedding = await recognizer._embedding_recognize(value["input"]["message"])
        pattern = recognizer._pattern_recognize(value["input"]["message"])
        llm = {"intent": IntentCategory(label), "confidence": confidence}
        intent, score, sources = recognizer._vote(llm, embedding, pattern)
        observation = {"response": {"text": text},
            "llm": {"intent": label, "confidence": confidence, "parser_failed": False},
            "fusion": {"intent": intent.value, "confidence": score,
                       "source_scores": sources, "embedding_source": "local-ngram"}}
        return operation, observation
    return asyncio.run(prepare())


def test_four_arms_reuse_production_parser_and_local_fallbacks_without_mutating_inputs():
    value = case()
    operation, observation = controlled_observation(value)
    before = deepcopy((value, operation, observation))
    output = asyncio.run(module.replay(value, operation, observation,
        base_url="https://dashscope.aliyuncs.com/apps/anthropic"))
    assert set(output) == set(module.ARMS)
    assert output["llm_only"] == output["full_fusion"] == "refund"
    assert output["keyword_fallback"] == "refund"
    assert output["local_template_fallback"] in {c.value for c in IntentCategory}
    assert (value, operation, observation) == before


@pytest.mark.parametrize("damage", ["input", "raw_text", "llm_label", "fusion_label", "fusion_score"])
def test_changed_input_or_recorded_result_is_rejected_not_silently_rescored(damage):
    value = case()
    operation, observation = controlled_observation(value)
    if damage == "input": value["input"]["message"] += "changed"
    elif damage == "raw_text": observation["response"]["text"] = "bad JSON"
    elif damage == "llm_label": observation["llm"]["intent"] = "invoice"
    elif damage == "fusion_label": observation["fusion"]["intent"] = "invoice"
    else: observation["fusion"]["confidence"] = .123
    with pytest.raises(base.TrialError, match="ablation_"):
        asyncio.run(module.replay(value, operation, observation,
            base_url="https://dashscope.aliyuncs.com/apps/anthropic"))


def test_local_fallback_does_not_apply_fusion_threshold_or_new_weight_normalization(monkeypatch):
    recognizer = module._local_recognizer(None)
    low_embedding = {"intent": IntentCategory.INVOICE, "confidence": .01}
    low_pattern = {"intent": IntentCategory.REFUND, "confidence": .01}
    empty = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    failed = {**empty, "failed": True}
    assert recognizer.threshold == .5
    assert recognizer._vote(failed, low_embedding, empty)[0] is IntentCategory.INVOICE
    assert recognizer._vote(failed, empty, low_pattern)[0] is IntentCategory.REFUND


def test_partition_denominators_and_zero_llm_gain_remain_explicit(monkeypatch):
    values = [case(case_id="a", partition="seen_regression"), case(case_id="b")]
    pairs = [controlled_observation(v) for v in values]
    protocol = {"operations": [p[0] for p in pairs], "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
                "model": base.MODEL, "git_head": "a" * 40}
    observations = [p[1] for p in pairs]
    monkeypatch.setattr(module, "verified_history", lambda: (values, protocol, observations, {"controlled_proof": True}))
    output = asyncio.run(module.measure())
    assert output["evidence_kind"] == "historical-model-output-replay+actual-local-inference"
    assert output["new_real_model_requests"] == output["new_paid_embedding_requests"] == 0
    assert output["old_scores_rewritten"] is False
    assert output["historical_shared_observations"] == 2
    for partition, count in (("seen_regression", 1), ("new_challenge", 1), ("all", 2)):
        for arm in module.ARMS:
            assert output["partitions"][partition][arm]["accuracy"]["denominator"] == count
    assert output["paired_vs_full_fusion"]["llm_only"] == {
        "both_correct": 2, "only_full_fusion_correct": 0, "only_comparator_correct": 0,
        "both_incorrect": 0, "denominator": 2, "full_fusion_accuracy_difference_pp": 0.0}
    resource = output["partitions"]["all"]["full_fusion"]["resources"]
    assert resource["usage"]["physical_requests"]["known_sum"] == 0
    assert resource["usage"]["input_tokens"]["known_sum"] is None
    assert resource["latency_ms"]["known_count"] == 0


def test_source_change_rejects_before_replay(tmp_path, monkeypatch):
    root = tmp_path / "history"
    root.mkdir(mode=0o700)
    data = tmp_path / "source"
    data.mkdir()
    cases_path = data / "cases.json"
    cases_path.write_text("{}")
    protocol = {"cases_sha256": bench.sha256(cases_path), "model": base.MODEL,
                "operations": [{} for _ in range(24)], "artifacts": {name: "a" * 64 for name in module.SOURCE_FILES}}
    base._write(root, "protocol.json", protocol)
    monkeypatch.setattr(module.history, "ROOT", root)
    monkeypatch.setattr(module.history, "CASES_PATH", cases_path)
    monkeypatch.setattr(module.history, "cases", lambda: [{} for _ in range(24)])
    monkeypatch.setattr(module.archive, "_tree", lambda root: "controlled-tree")
    monkeypatch.setattr(module.archive, "archived_proof", lambda: {"recheck": {
        "protocol_sha256": bench.sha256(root / "protocol.json")}})
    monkeypatch.setattr(module.bench, "BACKEND_ROOT", data)
    for name in module.SOURCE_FILES:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed source")
    before = (root / "protocol.json").read_bytes()
    with pytest.raises(base.TrialError, match="ablation_production_method_changed"):
        module.verified_history()
    assert (root / "protocol.json").read_bytes() == before


def test_cli_cannot_write_into_historical_or_arbitrary_directory(tmp_path, monkeypatch, capsys):
    async def result(): return {}
    monkeypatch.setattr(module, "measure", result)
    assert module.main(["--output", str(tmp_path / "old" / "result.json")]) == 2
    assert not (tmp_path / "old").exists()
    assert "no model requests" in capsys.readouterr().out
