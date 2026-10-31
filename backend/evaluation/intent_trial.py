"""Prepare an admitted, one-observation-per-question intent ablation.

Preparation serializes the existing production request through an in-memory
transport. It does not send requests. A separate, explicit authorization receipt
is required for execution; an old allowance or a different directory is invalid.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from evaluation import benchmark as bench
from evaluation.intent_transport import (
    GuardedIntentTransport, TrialTransportError, canonical_payload_hash,
    messages_url, validated_endpoint,
)
from evaluation.trial_ledger import TrialLedger, TrialLedgerError


BATCH = "serveflow-intent-development-observation-v1"
TRIAL_ROOT = bench.BACKEND_ROOT / "data" / "benchmarks" / "intent-development-v1"
MODEL = "qwen3.7-flash-2026-07-15"
MAX_REQUESTS = 24
LIMITS = {"physical_requests": 24, "per_operation": 1, "timeout_s": 8,
          "request_bytes": 65536, "response_bytes": 262144, "retries": 0}
DECISIONS = {
    "llm_only": "production parsed intent, without an additional confidence threshold",
    "fusion": "production _vote with embedding enabled, local-ngram and threshold 0.5",
    "timing": "shared LLM observation plus separately measured local fusion work; not independent end-to-end latency A/B",
    "structure": "production parsing is preserved; failed parsing is recorded, not repaired",
}
STOP_POLICY = {
    "continue": ["incorrect_label", "structure_failed"],
    "pause": ["version", "configuration", "storage_failure", "auth", "quota", "rate_limit",
              "network", "timeout", "cancelled", "response_limit", "protocol", "unknown"],
}
EXTENSION_FILES = ["evaluation/intent_trial.py", "evaluation/intent_transport.py", "evaluation/trial_ledger.py",
                   "tests/test_intent_trial.py", "tests/test_intent_transport.py", "tests/test_trial_ledger.py"]
PROTOCOL_FIELDS = {"schema_version", "batch", "phase", "created_at", "git_head", "root_binding_sha256",
                   "offline_manifest_path", "offline_manifest_sha256", "extension_artifacts", "dependencies",
                   "model", "base_url", "limits", "decision_policy", "stop_policy", "operations",
                   "prior_evidence_sha256", "authorized_model_requests", "proposed_model_requests"}
APPROVAL_FIELDS = {"schema_version", "batch", "protocol_sha256", "root_binding_sha256", "max_requests",
                   "user_authorization_reference", "created_at", "initial_ledger_chain_hash"}


class TrialError(ValueError):
    pass


def _fail(code):
    raise TrialError(code)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _root_hash(root):
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()


def _path(root, name):
    root = Path(root).absolute()
    if root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
        _fail("unsafe_trial_path")
    if root.exists() and (not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
        _fail("unsafe_trial_permissions")
    path = root / name
    if path.exists() and (not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
                          or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077):
        _fail("unsafe_trial_file")
    return path


def _write(root, name, value):
    path = _path(root, name)
    # ASCII JSON preserves even an upstream escaped lone surrogate verbatim;
    # replacing the model's text would hide a malformed-output diagnostic.
    payload = (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    if len(payload) > bench.LIMIT:
        _fail("private_output_limit")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _configuration(*, secret=False):
    from dotenv import dotenv_values
    raw = dotenv_values(bench.BACKEND_ROOT / ".env")
    names = ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ECHOMIND_INTENT_TIMEOUT_S", "INTENT_EMBEDDING_PROVIDER"]
    values = {name: os.environ.get(name, raw.get(name)) for name in names}
    base = validated_endpoint(values["ANTHROPIC_BASE_URL"] or "")
    try:
        timeout = float(values["ECHOMIND_INTENT_TIMEOUT_S"] or 8)
    except (TypeError, ValueError):
        _fail("configuration")
    if values["ANTHROPIC_MODEL"] != MODEL or timeout != 8 or values["INTENT_EMBEDDING_PROVIDER"] != "local":
        _fail("configuration")
    result = {"base_url": base, "model": MODEL}
    if secret:
        key = os.environ.get("ANTHROPIC_API_KEY", raw.get("ANTHROPIC_API_KEY"))
        if (type(key) is not str or not key or key.strip() != key
                or any(ord(character) <= 32 or ord(character) >= 127 for character in key)):
            _fail("configuration")
        result["api_key"] = key
    return result


def _dependencies():
    versions = {name: metadata.version(name) for name in ("anthropic", "httpx")}
    if versions != {"anthropic": "0.40.0", "httpx": "0.28.1"}:
        _fail("dependency_version")
    return versions


@contextmanager
def _quiet_clients():
    names = ("anthropic", "anthropic._base_client", "httpx", "httpcore")
    previous = {name: logging.getLogger(name).disabled for name in names}
    threshold = logging.root.manager.disable
    try:
        # Disabling a parent logger alone does not suppress child loggers.
        # This runs only in the isolated trial process, never the web runtime.
        logging.disable(logging.CRITICAL)
        for name in names:
            logging.getLogger(name).disabled = True
        yield
    finally:
        logging.disable(threshold)
        for name, value in previous.items():
            logging.getLogger(name).disabled = value


def _recognizer(client):
    from core.intent_recognizer import IntentRecognizer
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.client = client
    recognizer.model = MODEL
    recognizer.threshold = 0.5
    recognizer._embedding_enabled = True
    return recognizer


async def _capture(case, base_url):
    import httpx
    from anthropic import AsyncAnthropic
    captured = []

    async def handler(request):
        body = request.content
        payload = json.loads(body)
        captured.append({"url": str(request.url), "payload": payload,
                         "request_bytes_sha256": hashlib.sha256(body).hexdigest()})
        return httpx.Response(200, json={"id": "offline_capture", "type": "message", "role": "assistant",
            "model": MODEL, "content": [{"type": "text", "text": '{"intent":"other","confidence":0.0,"reasoning":"offline capture"}'}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False,
                             timeout=8, headers={"Accept-Encoding": "identity"})
    client = AsyncAnthropic(api_key="offline-capture-only", base_url=base_url,
                            max_retries=0, timeout=8, http_client=http)
    try:
        with _quiet_clients():
            result = await _recognizer(client)._llm_recognize(**bench.model_input(case))
        if result.get("failed") or len(captured) != 1:
            _fail("request_capture_failed")
        request = captured[0]
        payload = request["payload"]
        if (request["url"] != messages_url(base_url) or payload.get("model") != MODEL
                or payload.get("max_tokens") != 256 or payload.get("temperature") != 0.1
                or payload.get("thinking") != {"type": "disabled"} or payload.get("stream", False) is not False
                or set(payload) != {"model", "max_tokens", "temperature", "messages", "thinking"}):
            _fail("unexpected_production_payload")
        return {"case_id": case["id"], "operation_id": BATCH + ":" + case["id"] + ":observation-1",
                "input_sha256": canonical_payload_hash(bench.model_input(case)),
                "payload": payload, "payload_sha256": canonical_payload_hash(payload),
                "request_bytes_sha256": request["request_bytes_sha256"]}
    finally:
        await client.close()


def _development_cases():
    return [case for case in bench.load_suites()["intent"]["cases"] if case["split"] == "development"]


async def prepare(offline_manifest: Path, *, root: Path = TRIAL_ROOT, base_url=None, prior_records=()):
    root = Path(root).absolute()
    path = _path(root, "protocol.json")
    if path.exists() or (root / "ledger").exists():
        _fail("trial_already_prepared")
    offline_manifest = offline_manifest.resolve(strict=True)
    bench.verify_manifest(bench.read_json(offline_manifest))
    config = _configuration()
    if base_url is not None and validated_endpoint(base_url) != config["base_url"]:
        _fail("configuration")
    operations = [await _capture(case, config["base_url"]) for case in _development_cases()]
    if len(operations) != MAX_REQUESTS:
        _fail("operation_count")
    protocol = {
        "schema_version": "serveflow-intent-trial-v1", "batch": BATCH, "phase": "awaiting-explicit-authorization",
        "created_at": _now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
            capture_output=True, text=True, check=True).stdout.strip(),
        "root_binding_sha256": _root_hash(root), "offline_manifest_path": str(offline_manifest),
        "offline_manifest_sha256": bench.sha256(offline_manifest),
        "extension_artifacts": {name: bench.sha256(bench.BACKEND_ROOT / name) for name in EXTENSION_FILES},
        "dependencies": _dependencies(), "model": MODEL, "base_url": config["base_url"],
        "limits": deepcopy(LIMITS), "decision_policy": deepcopy(DECISIONS), "stop_policy": deepcopy(STOP_POLICY),
        "operations": operations, "prior_evidence_sha256": [bench.sha256(Path(p)) for p in prior_records],
        "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
    }
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write(root, "protocol.json", protocol)
    ledger = _ledger(protocol, root)
    if ledger.status()["attempts"] != 0:
        _fail("unexpected_existing_attempts")
    return protocol


def _ledger(protocol, root):
    return TrialLedger(Path(root) / "ledger", bench.sha256(_path(root, "protocol.json")),
                       [op["operation_id"] for op in protocol["operations"]], MAX_REQUESTS)


def verify(root: Path = TRIAL_ROOT):
    root = Path(root).absolute()
    protocol = bench.read_json(_path(root, "protocol.json"))
    if type(protocol) is not dict or set(protocol) != PROTOCOL_FIELDS:
        _fail("invalid_protocol")
    if (protocol["schema_version"] != "serveflow-intent-trial-v1" or protocol["batch"] != BATCH
            or protocol["phase"] != "awaiting-explicit-authorization" or protocol["model"] != MODEL
            or type(protocol["authorized_model_requests"]) is not int or protocol["authorized_model_requests"] != 0
            or type(protocol["proposed_model_requests"]) is not int or protocol["proposed_model_requests"] != MAX_REQUESTS
            or protocol["limits"] != LIMITS or protocol["decision_policy"] != DECISIONS or protocol["stop_policy"] != STOP_POLICY
            or protocol["root_binding_sha256"] != _root_hash(root) or protocol["dependencies"] != _dependencies()):
        _fail("protocol_version")
    config = _configuration()
    if protocol["base_url"] != config["base_url"]:
        _fail("configuration")
    previous = Path(protocol["offline_manifest_path"])
    if bench.sha256(previous) != protocol["offline_manifest_sha256"]:
        _fail("offline_manifest_changed")
    bench.verify_manifest(bench.read_json(previous))
    if type(protocol["extension_artifacts"]) is not dict or set(protocol["extension_artifacts"]) != set(EXTENSION_FILES):
        _fail("extension_artifacts_changed")
    if any(bench.sha256(bench.BACKEND_ROOT / name) != digest for name, digest in protocol["extension_artifacts"].items()):
        _fail("extension_artifacts_changed")
    operations = protocol["operations"]
    cases = _development_cases()
    if type(operations) is not list or len(operations) != len(cases):
        _fail("operation_count")
    for op, case in zip(operations, cases):
        if (type(op) is not dict or set(op) != {"case_id", "operation_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                or op["case_id"] != case["id"] or op["operation_id"] != BATCH + ":" + case["id"] + ":observation-1"
                or op["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                or op["payload_sha256"] != canonical_payload_hash(op["payload"])
                or type(op["request_bytes_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", op["request_bytes_sha256"])):
            _fail("operation_binding")
    # Missing or changed ledger files are an error, never an invitation to init.
    if not (root / "ledger" / "identity.json").is_file() or not (root / "ledger" / "journal.jsonl").is_file():
        _fail("ledger_missing")
    _ledger(protocol, root).status()
    return protocol


def authorize(reference: str, *, root: Path = TRIAL_ROOT, confirmed=False):
    if confirmed is not True or type(reference) is not str or not 1 <= len(reference.strip()) <= 500:
        _fail("explicit_authorization_required")
    protocol = verify(root)
    state = _ledger(protocol, root).status()
    if state["attempts"] or state["pending"] or state["paused"] or (Path(root) / "stopped.json").exists():
        _fail("cannot_authorize_started_trial")
    receipt = {"schema_version": "serveflow-intent-authorization-v1", "batch": BATCH,
               "protocol_sha256": bench.sha256(_path(root, "protocol.json")), "root_binding_sha256": _root_hash(Path(root)),
               "max_requests": MAX_REQUESTS, "user_authorization_reference": reference.strip(),
               "created_at": _now(), "initial_ledger_chain_hash": state["chain_hash"]}
    _write(root, "authorization.json", receipt)
    return receipt


def _approval(root):
    path = _path(root, "authorization.json")
    if not path.is_file():
        _fail("explicit_authorization_required")
    value = bench.read_json(path)
    if (type(value) is not dict or set(value) != APPROVAL_FIELDS
            or value["schema_version"] != "serveflow-intent-authorization-v1" or value["batch"] != BATCH
            or type(value["max_requests"]) is not int or value["max_requests"] != MAX_REQUESTS
            or value["protocol_sha256"] != bench.sha256(_path(root, "protocol.json"))
            or value["root_binding_sha256"] != _root_hash(Path(root))
            or type(value["user_authorization_reference"]) is not str or not value["user_authorization_reference"].strip()
            or value["initial_ledger_chain_hash"] != _ledger(bench.read_json(_path(root, "protocol.json")), root).initial_chain_hash):
        _fail("authorization_binding")
    return value


def _observation_name(op):
    return op["case_id"] + "-observation.json"


def _token(value):
    return value if type(value) is int and 0 <= value <= 10_000_000 else None


def _response_metadata(response):
    text = ""
    content = getattr(response, "content", None)
    blocks = content if isinstance(content, (list, tuple)) else []
    for block in blocks:
        kind = block.get("type") if type(block) is dict else getattr(block, "type", None)
        value = block.get("text") if type(block) is dict else getattr(block, "text", None)
        if kind == "text" and type(value) is str:
            text += value
    model = getattr(response, "model", None)
    if type(model) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", model):
        model = None
    stop = getattr(response, "stop_reason", None)
    if stop is not None and (type(stop) is not str or stop not in {"end_turn", "max_tokens", "tool_use", "stop_sequence"}):
        stop = "unknown"
    usage = getattr(response, "usage", None)
    return {"upstream_model": model, "stop_reason": stop, "text": text if len(text) <= 16384 else None,
            "text_omitted_over_limit": len(text) > 16384,
            "usage": {"input_tokens": _token(getattr(usage, "input_tokens", None)),
                      "output_tokens": _token(getattr(usage, "output_tokens", None))}}


async def _local_fusion(recognizer, case, llm):
    from core.intent_embedding import IntentEmbedding
    embedding = IntentEmbedding.__new__(IntentEmbedding)
    embedding.enabled = False
    embedding.key = ""
    embedding.provider = "local"
    embedding.retry_at = 0.0
    embedding.last_error = None
    embedding.local_templates = {}
    embedding.remote_templates = {}
    embedding.lock = asyncio.Lock()
    recognizer._embedding = embedding
    started = time.perf_counter()
    emb = await recognizer._embedding_recognize(case["input"]["message"])
    pattern = recognizer._pattern_recognize(case["input"]["message"])
    intent, confidence, scores = recognizer._vote(llm, emb, pattern)
    elapsed = (time.perf_counter() - started) * 1000
    return {"intent": intent.value, "confidence": confidence, "source_scores": scores,
            "embedding_source": emb.get("source"), "local_fusion_ms": elapsed}


def _error(code):
    return {"authentication_error": "auth", "payment_required": "quota", "rate_limited": "rate_limit",
            "timeout": "timeout", "cancelled": "cancelled", "response_limit": "response_limit",
            "before_send_failed": "storage_failure", "payload_mismatch": "version",
            "invalid_request": "protocol", "invalid_payload": "protocol"}.get(code, "network")


def _verify_completed_outputs(protocol, root, ledger, expected_kind=None):
    sealed = ledger.results()
    operations = {op["operation_id"]: op for op in protocol["operations"]}
    for operation_id, result in sealed.items():
        op = operations.get(operation_id)
        if op is None:
            _fail("observation_binding")
        path = _path(root, _observation_name(op))
        if not path.exists():
            _fail("completed_observation_missing")
        if result["output_sha256"] != bench.sha256(path):
            _fail("observation_hash_changed")
        if expected_kind is not None and bench.read_json(path).get("evidence_kind") != expected_kind:
            _fail("mixed_or_unknown_evidence")
    return sealed


async def _execute_operation(protocol, op, case, *, root, config, delegate=None):
    import httpx
    from anthropic import AsyncAnthropic
    ledger = _ledger(protocol, root)
    reserved = False
    response_holder = {}
    guard = None

    def admit():
        nonlocal reserved
        verify(root)
        _approval(root)
        if (Path(root) / "stopped.json").exists():
            _fail("trial_paused")
        _verify_completed_outputs(protocol, root, ledger,
                                  "controlled-transport" if delegate is not None else "live-model")
        if guard.request_bytes_sha256 != op["request_bytes_sha256"]:
            _fail("serialized_request_changed")
        ledger.reserve(op["operation_id"], op["payload_sha256"])
        reserved = True

    guard = GuardedIntentTransport(protocol["base_url"], op["payload_sha256"], admit, transport=delegate)
    http = httpx.AsyncClient(transport=guard, follow_redirects=False, trust_env=False,
                             timeout=8, headers={"Accept-Encoding": "identity"})
    sdk = AsyncAnthropic(api_key=config["api_key"], base_url=protocol["base_url"],
                         max_retries=0, timeout=8, http_client=http)

    async def create(**kwargs):
        response = await sdk.messages.create(**kwargs)
        response_holder["response"] = response
        return response

    client = SimpleNamespace(base_url=sdk.base_url, messages=SimpleNamespace(create=create))
    recognizer = _recognizer(client)
    started = time.perf_counter()
    llm = None
    try:
        with _quiet_clients():
            async with asyncio.timeout(8):
                llm = await recognizer._llm_recognize(**bench.model_input(case))
    except TimeoutError:
        guard.failure_code = guard.failure_code or "timeout"
    except asyncio.CancelledError:
        guard.failure_code = guard.failure_code or "cancelled"
    finally:
        await sdk.close()
    elapsed = (time.perf_counter() - started) * 1000
    if not reserved or not guard.sent:
        _fail("send_admission_failed")
    error_code = _error(guard.failure_code) if guard.failure_code else None
    response = _response_metadata(response_holder.get("response"))
    if llm is None:
        from core.intent_recognizer import IntentCategory
        llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
    fusion = None if error_code else await _local_fusion(recognizer, case, llm)
    output = {"schema_version": "serveflow-intent-observation-v1", "batch": BATCH,
        "evidence_kind": "controlled-transport" if delegate is not None else "live-model",
        "protocol_sha256": bench.sha256(_path(root, "protocol.json")), "operation_id": op["operation_id"],
        "case_id": op["case_id"], "payload_sha256": op["payload_sha256"],
        "request_bytes_sha256": guard.request_bytes_sha256, "request_model": MODEL,
        "transport": {"sent": guard.sent, "http_status": guard.http_status, "request_id": guard.request_id, "error_code": error_code},
        "response": response,
        "llm": {"intent": llm["intent"].value, "confidence": llm.get("confidence", 0.0), "parser_failed": bool(llm.get("failed"))},
        "fusion": fusion, "shared_latency_ms": elapsed}
    name = _observation_name(op)
    _write(root, name, output)
    outcome = "cancelled" if error_code == "cancelled" else "execution_error" if error_code else "structure_failed" if llm.get("failed") else "received"
    ledger.finish(op["operation_id"], outcome, bench.sha256(_path(root, name)), error_code)
    return output


def _stop(root, code):
    try:
        _write(root, "stopped.json", {"schema_version": "serveflow-intent-stop-v1", "error_code": code, "recorded_at": _now()})
    except (OSError, ValueError):
        pass


async def execute(*, root: Path = TRIAL_ROOT, delegate_factory=None):
    protocol = verify(root)
    _approval(root)
    ledger = _ledger(protocol, root)
    state = ledger.status()
    if state["paused"] or state["pending"] or (Path(root) / "stopped.json").exists():
        _fail("trial_paused")
    _verify_completed_outputs(protocol, root, ledger,
                              "controlled-transport" if delegate_factory is not None else "live-model")
    config = _configuration(secret=True)
    for op, case in zip(protocol["operations"], _development_cases()):
        if (Path(root) / "stopped.json").exists():
            _fail("trial_paused")
        if op["operation_id"] in state["completed"]:
            continue
        try:
            output = await _execute_operation(protocol, op, case, root=root, config=config,
                delegate=delegate_factory(op) if delegate_factory else None)
        except (Exception, asyncio.CancelledError):
            _stop(root, "execution_stopped")
            _fail("execution_stopped")
        state = ledger.status()
        print(json.dumps({"case_id": op["case_id"], "attempts": state["attempts"], "paused": state["paused"]}))
        if output["transport"]["error_code"]:
            _fail("trial_paused")
    return report(root)


def report(root: Path = TRIAL_ROOT):
    from evaluation.benchmark_metrics import aggregate_resources, score_intent
    protocol = verify(root)
    ledger = _ledger(protocol, root)
    state = ledger.status()
    sealed = _verify_completed_outputs(protocol, root, ledger)
    if state["attempts"]:
        _approval(root)
    cases = _development_cases()
    records = {"llm_only": [], "fusion": []}
    resources, paired, observation_hashes, kinds = [], [], [], set()
    for op in protocol["operations"]:
        path = _path(root, _observation_name(op))
        if not path.exists():
            if op["operation_id"] in sealed:
                _fail("completed_observation_missing")
            if op["operation_id"] in state["pending"]:
                for arm in records:
                    records[arm].append({"case_id": op["case_id"], "status": "unknown", "predicted_intent": None,
                        "latency_ms": None, "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None}})
                resources.append({"case_id": op["case_id"], "status": "unknown", "latency_ms": None,
                    "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None}})
            continue
        if op["operation_id"] not in sealed or sealed[op["operation_id"]]["output_sha256"] != bench.sha256(path):
            _fail("observation_hash_changed")
        output = bench.read_json(path)
        if (output["protocol_sha256"] != bench.sha256(_path(root, "protocol.json")) or output["operation_id"] != op["operation_id"]
                or output["payload_sha256"] != op["payload_sha256"] or op["operation_id"] not in state["completed"]):
            _fail("observation_binding")
        observation_hashes.append(bench.sha256(path))
        kinds.add(output["evidence_kind"])
        valid_execution = output["transport"]["error_code"] is None
        for arm, result in (("llm_only", output["llm"]), ("fusion", output["fusion"])):
            successful = valid_execution and result is not None and (arm != "llm_only" or not result["parser_failed"])
            records[arm].append({"case_id": op["case_id"], "status": "success" if successful else "failed",
                                 "predicted_intent": result["intent"] if result else None,
                                 "latency_ms": None, "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None}})
        resources.append({"case_id": op["case_id"], "status": "success" if valid_execution else "failed",
                          "latency_ms": output["shared_latency_ms"],
                          "usage": {**output["response"]["usage"], "physical_requests": 1}})
    scores = {arm: score_intent(cases, rows) for arm, rows in records.items()}
    if len(kinds) > 1 or not kinds <= {"controlled-transport", "live-model"}:
        _fail("mixed_or_unknown_evidence")
    matrix = {key: 0 for key in ("both_correct", "llm_only_correct", "fusion_only_correct", "both_incorrect", "execution_unknown", "not_executed")}
    for a, b in zip(scores["llm_only"]["per_case"], scores["fusion"]["per_case"]):
        key = "not_executed" if a["status"] == b["status"] == "not_executed" else "execution_unknown" if "unknown" in (a["status"], b["status"]) else "both_correct" if a["correct"] and b["correct"] else "llm_only_correct" if a["correct"] else "fusion_only_correct" if b["correct"] else "both_incorrect"
        matrix[key] += 1
        paired.append({"case_id": a["case_id"], "outcome": key})
    if ledger.status()["chain_hash"] != state["chain_hash"]:
        _fail("ledger_changed_during_report")
    return {"schema_version": "serveflow-intent-trial-report-v1", "batch": BATCH,
            "evidence_kind": "controlled-shared-observation" if kinds == {"controlled-transport"} else "live-shared-observation" if kinds == {"live-model"} else "pending-send" if state["attempts"] else "not-run",
            "protocol_sha256": bench.sha256(_path(root, "protocol.json")), "ledger": state,
            "scores": scores, "paired_counts": matrix, "paired_cases": paired,
            "shared_resources": aggregate_resources(resources), "observation_sha256": observation_hashes,
            "limits": ["development cases only; no holdout performance claim", "shared observation ablation, not independent A/B",
                       "production parser retained; parsed label accuracy is not full response-schema validation",
                       "shared usage counted once; per-arm resource fields intentionally unknown",
                       "single measurement; no stability, generalization or end-to-end latency improvement claim"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--offline-manifest", type=Path, required=True)
    p.add_argument("--prior-record", type=Path, action="append", default=[])
    commands.add_parser("verify"); commands.add_parser("status")
    p = commands.add_parser("authorize")
    p.add_argument("--reference", required=True)
    p.add_argument("--confirm-user-authorization", action="store_true")
    commands.add_parser("run"); commands.add_parser("report")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            asyncio.run(prepare(args.offline_manifest, prior_records=args.prior_record))
            value = {"prepared": True, "proposed_physical_requests": 24, "authorized_model_requests": 0}
        elif args.command == "authorize":
            authorize(args.reference, confirmed=args.confirm_user_authorization)
            value = {"authorization_recorded": True, "max_physical_requests": 24}
        elif args.command == "verify":
            verify(); value = {"verified": True, "physical_requests": _ledger(verify(), TRIAL_ROOT).status()["attempts"]}
        elif args.command == "status":
            value = _ledger(verify(), TRIAL_ROOT).status()
        else:
            value = asyncio.run(execute()) if args.command == "run" else report()
            name = "report-" + value["ledger"]["chain_hash"] + ".json"
            path = _path(TRIAL_ROOT, name)
            if not path.exists():
                _write(TRIAL_ROOT, name, value)
            elif bench.read_json(path) != value:
                _fail("saved_report_changed")
            value = {"attempts": value["ledger"]["attempts"], "paused": value["ledger"]["paused"],
                     "evidence_kind": value["evidence_kind"], "paired_counts": value["paired_counts"]}
        print(json.dumps(value, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, TrialLedgerError, metadata.PackageNotFoundError) as exc:
        code = exc.code if isinstance(exc, (TrialTransportError, TrialLedgerError)) else str(exc) if isinstance(exc, TrialError) else "trial_input_or_storage_failure"
        print(json.dumps({"error": code, "no_automatic_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
