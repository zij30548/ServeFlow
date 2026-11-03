"""Offline source ablation using 24 sealed historical LLM observations.

No clients, configuration files, authorization or model services are opened.
The two local arms are the existing production failure fallbacks, not tuned
standalone classifiers. All arms share the same cases and unchanged methods.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from core.intent_embedding import IntentEmbedding
from core.intent_recognizer import IntentCategory, INTENT_POLICY_VERSION
from evaluation import benchmark as bench, intent_trial as base, intent_recheck as history
from evaluation import task_recheck as archive
from evaluation.benchmark_metrics import score_intent
from evaluation.intent_transport import canonical_payload_hash


SOURCE_FILES = (
    "core/intent_recognizer.py", "core/intent_embedding.py", "core/provider_options.py",
    "core/llm_utils.py", "core/safe_diagnostics.py", "evaluation/benchmark/intent-recheck-v2.json",
    "evaluation/intent_trial.py", "evaluation/intent_recheck.py", "evaluation/benchmark_metrics.py",
)
ARMS = ("llm_only", "keyword_fallback", "local_template_fallback", "full_fusion")
OUTPUT_ROOT = bench.BACKEND_ROOT / "data/benchmarks/intent-source-ablation-v1"
ARM_POLICY = {
    "llm_only": "unchanged production parsing of one sealed historical LLM response",
    "keyword_fallback": "production _vote with llm.failed and no embedding evidence; no added threshold",
    "local_template_fallback": "production _vote with llm.failed and no keyword evidence; no added threshold",
    "full_fusion": "unchanged production _vote; local-ngram enabled, threshold 0.5, weights 0.7/0.2/0.1",
}


def _fail(code):
    raise base.TrialError(code)


def verified_history():
    """Read old Git-bound evidence, never use old current-source verify/report."""
    proof = archive.archived_proof()["recheck"]
    root = history.ROOT
    before = archive._tree(root)
    path = base._path(root, "protocol.json")
    if bench.sha256(path) != proof["protocol_sha256"]:
        _fail("ablation_protocol_changed")
    protocol = bench.read_json(path)
    cases = history.cases()
    if (len(cases) != 24 or protocol.get("cases_sha256") != bench.sha256(history.CASES_PATH)
            or protocol.get("model") != base.MODEL or len(protocol.get("operations", [])) != 24):
        _fail("ablation_cases_or_model_changed")
    sources = {}
    for name in SOURCE_FILES:
        digest = bench.sha256(bench.BACKEND_ROOT / name)
        if protocol.get("artifacts", {}).get(name) != digest:
            _fail("ablation_production_method_changed")
        sources[name] = digest
    observations = []
    for case, operation in zip(cases, protocol["operations"]):
        case_id = case["id"]
        path = base._path(root, case_id + "-observation.json")
        if bench.sha256(path) != proof["observation_sha256"].get(case_id):
            _fail("ablation_observation_changed")
        value = bench.read_json(path)
        if (operation.get("case_id") != case_id
                or operation.get("input_sha256") != canonical_payload_hash(bench.model_input(case))
                or operation.get("payload_sha256") != canonical_payload_hash(operation.get("payload"))
                or value.get("evidence_kind") != "live-model"
                or value.get("protocol_sha256") != proof["protocol_sha256"]
                or value.get("operation_id") != operation.get("operation_id")
                or value.get("case_id") != case_id
                or value.get("payload_sha256") != operation.get("payload_sha256")
                or value.get("request_bytes_sha256") != operation.get("request_bytes_sha256")
                or value.get("request_model") != base.MODEL
                or value.get("transport", {}).get("sent") is not True
                or value.get("transport", {}).get("http_status") != 200
                or value.get("transport", {}).get("error_code") is not None
                or value.get("response", {}).get("stop_reason") != "end_turn"
                or type(value.get("response", {}).get("text")) is not str
                or len(value["response"]["text"]) > 16384
                or value["response"].get("text_omitted_over_limit") is not False):
            _fail("ablation_observation_binding_changed")
        observations.append(value)
    if archive._tree(root) != before:
        _fail("ablation_history_changed_during_read")
    return cases, protocol, observations, {**proof, "current_source_sha256": sources,
                                          "historical_tree_sha256": before}


def _local_recognizer(client):
    # Match the real trial's explicit local-only setup. Do not run either
    # constructor: they read environment and can create network-capable clients.
    recognizer = base._recognizer(client)
    embedding = IntentEmbedding.__new__(IntentEmbedding)
    embedding.enabled, embedding.key, embedding.provider = False, "", "local"
    embedding.retry_at, embedding.last_error = 0.0, None
    embedding.local_templates, embedding.remote_templates = {}, {}
    embedding.lock = asyncio.Lock()
    recognizer._embedding = embedding
    return recognizer


async def replay(case, operation, observation, *, base_url):
    """Parse saved text through production code, with input binding in memory."""
    invocations = []
    async def create(**kwargs):
        invocations.append(True)
        payload = deepcopy(kwargs)
        extension = payload.pop("extra_body", {})
        if type(extension) is not dict or set(payload) & set(extension):
            _fail("ablation_replay_payload_changed")
        payload.update(extension)
        if len(invocations) != 1 or canonical_payload_hash(payload) != operation["payload_sha256"]:
            _fail("ablation_replay_payload_changed")
        return SimpleNamespace(content=[{"type": "text", "text": observation["response"]["text"]}])
    client = SimpleNamespace(base_url=base_url, messages=SimpleNamespace(create=create))
    recognizer = _local_recognizer(client)
    llm = await recognizer._llm_recognize(**bench.model_input(case))
    saved = observation["llm"]
    if (len(invocations) != 1 or llm["intent"].value != saved["intent"]
            or bool(llm.get("failed")) != saved["parser_failed"]
            or not math.isclose(llm["confidence"], saved["confidence"], rel_tol=0, abs_tol=1e-12)):
        _fail("ablation_historical_parser_mismatch")
    embedding = await recognizer._embedding_recognize(case["input"]["message"])
    pattern = recognizer._pattern_recognize(case["input"]["message"])
    full, confidence, scores = recognizer._vote(llm, embedding, pattern)
    saved = observation["fusion"]
    if (full.value != saved["intent"] or embedding.get("source") != saved["embedding_source"]
            or not math.isclose(confidence, saved["confidence"], rel_tol=0, abs_tol=1e-12)
            or set(scores) != set(saved["source_scores"])
            or any(not math.isclose(value, saved["source_scores"][name], rel_tol=0, abs_tol=1e-12)
                   for name, value in scores.items())):
        _fail("ablation_historical_fusion_mismatch")
    empty = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    failed = {**empty, "failed": True}
    keyword = recognizer._vote(failed, empty, pattern)[0]
    template = recognizer._vote(failed, embedding, empty)[0]
    return {"llm_only": llm["intent"].value, "keyword_fallback": keyword.value,
            "local_template_fallback": template.value, "full_fusion": full.value}


def _record(case_id, predicted):
    return {"case_id": case_id, "status": "success", "predicted_intent": predicted,
            "latency_ms": None,
            "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 0}}


async def measure():
    cases, protocol, observations, proof = verified_history()
    records = {arm: [] for arm in ARMS}
    per_case = []
    for case, operation, observation in zip(cases, protocol["operations"], observations):
        predictions = await replay(case, operation, observation, base_url=protocol["base_url"])
        for arm, predicted in predictions.items():
            records[arm].append(_record(case["id"], predicted))
        per_case.append({"case_id": case["id"], "partition": case["partition"],
                         "expected": case["expected"]["intent"], "predictions": predictions})
    partitions = {}
    for partition in ("seen_regression", "new_challenge", "all"):
        subset = cases if partition == "all" else [c for c in cases if c["partition"] == partition]
        ids = {c["id"] for c in subset}
        partitions[partition] = {arm: score_intent(subset, [r for r in rows if r["case_id"] in ids])
                                 for arm, rows in records.items()}
    paired = {}
    for arm in ARMS[:-1]:
        cells = {"both_correct": 0, "only_full_fusion_correct": 0,
                 "only_comparator_correct": 0, "both_incorrect": 0}
        for row in per_case:
            full_ok = row["predictions"]["full_fusion"] == row["expected"]
            other_ok = row["predictions"][arm] == row["expected"]
            key = ("both_correct" if full_ok and other_ok else "only_full_fusion_correct" if full_ok
                   else "only_comparator_correct" if other_ok else "both_incorrect")
            cells[key] += 1
        paired[arm] = {**cells, "denominator": len(cases),
            "full_fusion_accuracy_difference_pp": 100 * (cells["only_full_fusion_correct"]
                                                        - cells["only_comparator_correct"]) / len(cases)}
    # Re-read all signed historical bytes after local processing; do not write
    # results into their directory or silently accept a concurrent change.
    if verified_history()[3] != proof:
        _fail("ablation_history_changed_during_measurement")
    return {"schema_version": "serveflow-intent-source-ablation-v1",
        "evidence_kind": "historical-model-output-replay+actual-local-inference",
        "created_at": base._now(), "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT, capture_output=True,
            text=True, check=True).stdout.strip(),
        "runner_sha256": bench.sha256(Path(__file__)), "candidate": INTENT_POLICY_VERSION,
        "historical_execution_commit": protocol["git_head"], "historical_model": protocol["model"],
        "historical_shared_observations": len(observations), "new_real_model_requests": 0,
        "new_paid_embedding_requests": 0, "old_scores_rewritten": False, "proof": proof,
        "arm_policy": deepcopy(ARM_POLICY), "partitions": partitions,
        "paired_vs_full_fusion": paired, "per_case": per_case,
        "limitations": ["same 24 historical outputs, not 96 new model calls",
            "two seen regressions and 22 author-visible challenges are separate",
            "local arms are existing fallbacks, not separately tuned strongest baselines",
            "fine-label accuracy: a legal coarse fallback label can still be incorrect for the frozen specific label",
            "no new thresholds, weight renormalization, samples or expected-label changes",
            "all-zero LLM/fusion gain must remain a no-gain result",
            "source classification only, not RAG, routing execution or end-to-end task success",
            "no new physical latency/token/cost observation; no generic accuracy claim"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(measure())
        if args.output is not None:
            output = args.output.absolute()
            if output.parent != OUTPUT_ROOT:
                _fail("ablation_historical_output_forbidden")
            base._write(output.parent, output.name, result)
        summary = {"evidence_kind": result["evidence_kind"], "new_real_model_requests": 0,
            "arms": {arm: result["partitions"]["all"][arm]["accuracy"] for arm in ARMS},
            "full_fusion_vs_llm": result["paired_vs_full_fusion"]["llm_only"]}
        import json
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except Exception:
        print("STOP: offline_ablation_failed; no model requests or automatic repair")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
