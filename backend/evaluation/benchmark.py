"""Offline benchmark preparation and scoring; no clients or live execution.

Labels and evidence stay in the case files. ``model_input`` is an explicit
projection, not an instruction to send a request. Every CLI command is local.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = BACKEND_ROOT / "evaluation" / "benchmark"
SUITES = {"intent": (60, 24, 36), "retrieval": (24, 12, 12), "tasks": (12, 6, 6)}
BENCHMARK_INTENTS = {"order_status", "logistics", "refund", "invoice", "payment_issue",
                     "account_security", "technical_login", "technical_crash", "human_handoff",
                     "request", "greeting", "other"}
ARMS = {"intent": {"llm_only", "fusion"}, "retrieval": {"vector_only", "rewrite_rerank"}, "tasks": {"current"}}
COMPARISON_PLAN = {"intent": ["llm_only", "fusion"], "retrieval": ["vector_only", "rewrite_rerank"], "tasks": ["current"]}
AGENTS = {"general", "technical", "billing", "escalation"}
TOOLS = {"search_knowledge_base", "read_demo_order", "inspect_request_context",
         "suggest_required_fields", "lookup_error_code", "build_diagnostic_plan",
         "check_billing_fields", "compare_amounts", "create_handoff_summary"}
TABLE_IDS = {"orders": "order_id", "items": "item_id", "payments": "payment_id",
             "shipments": "shipment_id", "entitlements": "entitlement_id", "refunds": "refund_id"}
LIMIT = 2 * 1024 * 1024
MODEL_PLAN = {"model": "qwen3.7-flash-2026-07-15", "provider": "bailian-beijing-anthropic-compatible",
              "thinking": "disabled", "parameters": "existing component settings; live payloads not yet frozen"}
SCORING_POLICY = "all planned cases remain in denominators; no automatic semantic grading"
LIMITATIONS = ["synthetic authored cases, not independent human blind evaluation",
               "validation is held out from tuning, not hidden from dataset authors",
               "source policies and order fixtures are shared; t05 and t11 are an unauthorized/authorized pair on DEMO-1005",
               "six synthetic policy documents and ten synthetic orders do not establish enterprise accuracy",
               "existing mechanisms require ablation; a current-only task run is not evidence of improvement",
               "live payloads, operation order, persistent send ledger and a new call allowance remain prerequisites"]


class BenchmarkError(ValueError):
    """Safe error code; do not echo submitted answers, paths or unknown keys."""


def _fail(code: str) -> None:
    raise BenchmarkError(code)


def _exact(value: Any, keys: set[str], code: str) -> None:
    if type(value) is not dict or set(value) != keys:
        _fail(code)


def _text(value: Any, maximum: int = 2000) -> None:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        _fail("invalid_text")


def _strings(value: Any, *, allow_empty: bool = True, maximum: int = 20) -> None:
    if type(value) is not list or len(value) > maximum or (not allow_empty and not value):
        _fail("invalid_string_list")
    for item in value:
        _text(item)
    if len(set(value)) != len(value):
        _fail("duplicate_list_item")


def _finite(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        _fail("nonfinite_json")
    if type(value) is dict:
        for item in value.values():
            _finite(item)
    elif type(value) is list:
        for item in value:
            _finite(item)


def read_json(path: Path) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        _fail("json_size_limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda _: _fail("nonfinite_json"))
    except (UnicodeError, json.JSONDecodeError):
        _fail("invalid_json")
    _finite(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def intent_labels(root: Path = BACKEND_ROOT) -> set[str]:
    tree = ast.parse((root / "core" / "intent_recognizer.py").read_text())
    enum = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IntentCategory")
    return {node.value.value for node in enum.body if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}


def _built_in_questions(root: Path) -> set[str]:
    """Extract known examples using AST only; importing runtime is unnecessary."""
    banned = set()
    tree = ast.parse((root / "core" / "intent_recognizer.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_TEMPLATES":
            banned.update(_normal(child.value) for child in ast.walk(node.value)
                          if isinstance(child, ast.Constant) and isinstance(child.value, str))
    tree = ast.parse((root / "evaluation" / "evaluator.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEFAULT_INTENT_CASES":
            banned.update(_normal(call.args[0].value) for call in ast.walk(node.value)
                          if isinstance(call, ast.Call) and call.args
                          and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str))
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEFAULT_DIALOG_CASES":
            for example in ast.literal_eval(node.value):
                banned.update(_normal(text) for text in ([example["question"]] if "question" in example else example.get("turns", [])))
    tree = ast.parse((root / "evaluation" / "few_shot.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "JUDGE_FEW_SHOT_EXAMPLES" for target in node.targets):
            banned.update(_normal(example["question"]) for example in ast.literal_eval(node.value))
    return banned


def _sources(root: Path):
    docs = read_json(root / "knowledge" / "serveflow-demo-v2.json")
    corpus = {doc["title"]: doc["content"] for doc in docs}
    if len(corpus) != len(docs):
        _fail("duplicate_source_title")
    fixture = read_json(root / "demo_business" / "fixtures.json")
    tables = {table: {row[key]: row for row in fixture[table]} for table, key in TABLE_IDS.items()}
    return corpus, tables


def _references(refs: Any, corpus: dict, tables: dict) -> None:
    if type(refs) is not list or len(refs) > 24:
        _fail("invalid_references")
    for ref in refs:
        if type(ref) is not dict:
            _fail("invalid_reference")
        if ref.get("source") == "knowledge":
            _exact(ref, {"source", "title", "quote"}, "invalid_knowledge_reference")
            _text(ref["quote"])
            if type(ref["title"]) is not str or ref["title"] not in corpus or ref["quote"] not in corpus[ref["title"]]:
                _fail("unsupported_knowledge_quote")
        elif ref.get("source") == "fixture":
            _exact(ref, {"source", "table", "row_id", "fields"}, "invalid_fixture_reference")
            _text(ref["table"], 40)
            _text(ref["row_id"], 100)
            row = tables.get(ref["table"], {}).get(ref["row_id"])
            fields = ref["fields"]
            if row is None or type(fields) is not dict or not fields:
                _fail("unknown_fixture_reference")
            for name, value in fields.items():
                if name not in row or type(value) is not type(row[name]) or value != row[name]:
                    _fail("unsupported_fixture_value")
        else:
            _fail("unknown_reference_source")


def model_input(case: dict) -> dict:
    """Only conversational input; labels, reviewer notes and identity stay out."""
    return deepcopy({key: case["input"][key] for key in ("message", "history")})


def fixture_context(case: dict) -> dict:
    """Trusted test harness binding; never append this to a model prompt."""
    return {"customer_id": case["input"]["customer_id"]} if "customer_id" in case["input"] else {}


def validate_suite(data: Any, root: Path = BACKEND_ROOT) -> dict:
    _exact(data, {"schema_version", "suite", "dataset_id", "provenance", "cases"}, "invalid_dataset_fields")
    suite = data["suite"]
    if data["schema_version"] != "serveflow-benchmark-cases-v1" or type(suite) is not str or suite not in SUITES or data["dataset_id"] != suite + "-v1":
        _fail("unsupported_dataset_version")
    if data["provenance"] != "synthetic-authored-not-independent-blind-test":
        _fail("invalid_provenance")
    cases = data["cases"]
    if type(cases) is not list or len(cases) != SUITES[suite][0]:
        _fail("unexpected_case_count")
    corpus, tables = _sources(root)
    labels, banned = intent_labels(root), _built_in_questions(root)
    ids, inputs, families, splits = set(), set(), {}, Counter()
    class_counts, class_splits, kinds = Counter(), Counter(), Counter()
    for case in cases:
        common = {"id", "split", "family_id", "input", "expected", "tags"}
        _exact(case, common if suite == "intent" else common | {"references"}, "invalid_case_fields")
        if type(case["id"]) is not str or not re.fullmatch({"intent": "i", "retrieval": "r", "tasks": "t"}[suite] + r"[0-9]{2}", case["id"]) or case["id"] in ids:
            _fail("invalid_or_duplicate_case_id")
        ids.add(case["id"])
        split = case["split"]
        if type(split) is not str or split not in {"development", "validation"}:
            _fail("invalid_split")
        _text(case["family_id"], 100)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", case["family_id"]):
            _fail("invalid_family_id")
        family = case["family_id"]
        if family in families and families[family] != split:
            _fail("family_crosses_split")
        families[family] = split
        splits[split] += 1
        _strings(case["tags"], allow_empty=False)
        _exact(case["input"], {"message", "history"} | ({"customer_id"} if suite == "tasks" else set()), "invalid_input_fields")
        _text(case["input"]["message"])
        history = case["input"]["history"]
        if type(history) is not list or len(history) > 6:
            _fail("invalid_history")
        for turn in history:
            _exact(turn, {"role", "content"}, "invalid_history_turn")
            if type(turn["role"]) is not str or turn["role"] not in {"user", "assistant"}:
                _fail("invalid_history_role")
            _text(turn["content"])
        normalized = json.dumps({"message": _normal(case["input"]["message"]),
                                 "history": [{"role": t["role"], "content": _normal(t["content"])} for t in history]}, sort_keys=True)
        if normalized in inputs:
            _fail("duplicate_case_input")
        inputs.add(normalized)
        expected = case["expected"]
        if suite == "intent":
            _exact(expected, {"intent", "rationale"}, "invalid_intent_expectation")
            if type(expected["intent"]) is not str or expected["intent"] not in labels or expected["intent"] not in BENCHMARK_INTENTS:
                _fail("unknown_intent_label")
            if _normal(case["input"]["message"]) in banned:
                _fail("built_in_example_overlap")
            class_counts[expected["intent"]] += 1
            class_splits[(expected["intent"], split)] += 1
        elif suite == "retrieval":
            _exact(expected, {"kind", "relevant_titles", "required_facts", "missing_facts", "forbidden_claims", "rationale"}, "invalid_retrieval_expectation")
            kind = expected["kind"]
            if type(kind) is not str or kind not in {"answerable", "insufficient", "out_of_scope"}:
                _fail("unknown_retrieval_kind")
            kinds[(split, kind)] += 1
            for field in ("relevant_titles", "required_facts", "missing_facts", "forbidden_claims"):
                _strings(expected[field])
            if any(title not in corpus for title in expected["relevant_titles"]):
                _fail("unknown_expected_document")
            if (kind == "answerable" and (not expected["relevant_titles"] or not expected["required_facts"] or expected["missing_facts"])) or (kind == "insufficient" and (not expected["relevant_titles"] or not expected["missing_facts"])) or (kind == "out_of_scope" and (expected["relevant_titles"] or expected["required_facts"] or case["references"] or not expected["missing_facts"])):
                _fail("contradictory_retrieval_expectation")
            if kind != "out_of_scope":
                cited_titles = {r.get("title") for r in case["references"] if type(r) is dict and r.get("source") == "knowledge"}
                if not set(expected["relevant_titles"]) <= cited_titles:
                    _fail("missing_document_reference")
        else:
            _exact(expected, {"required_agents", "required_tools", "required_facts", "forbidden_claims", "rationale"}, "invalid_task_expectation")
            if type(case["input"]["customer_id"]) is not str or case["input"]["customer_id"] not in {"demo-customer-a", "demo-customer-b"}:
                _fail("invalid_fixture_identity")
            for field in ("required_agents", "required_tools", "required_facts", "forbidden_claims"):
                _strings(expected[field], allow_empty=field in {"required_agents", "required_tools"})
            if not set(expected["required_agents"]) <= AGENTS or not set(expected["required_tools"]) <= TOOLS:
                _fail("unknown_agent_or_tool")
            if not case["references"]:
                _fail("missing_task_evidence")
        _text(expected["rationale"])
        if suite != "intent":
            _references(case["references"], corpus, tables)
    if (splits["development"], splits["validation"]) != SUITES[suite][1:]:
        _fail("unexpected_split_counts")
    if suite == "intent" and (set(class_counts) != BENCHMARK_INTENTS or set(class_counts.values()) != {5} or any(class_splits[(label, "development")] != 2 or class_splits[(label, "validation")] != 3 for label in class_counts)):
        _fail("unexpected_intent_balance")
    if suite == "retrieval" and any(kinds[(split, kind)] != count for split in splits for kind, count in {"answerable": 6, "insufficient": 3, "out_of_scope": 3}.items()):
        _fail("unexpected_retrieval_balance")
    return {"suite": suite, "total": len(cases), "splits": dict(splits),
            "provenance": data["provenance"], "families": len(families)}


def load_suites(root: Path = BACKEND_ROOT) -> dict:
    return {suite: read_json(root / "evaluation" / "benchmark" / f"{suite}-v1.json") for suite in SUITES}


def artifact_paths(root: Path = BACKEND_ROOT) -> list[str]:
    fixed = ["core/intent_recognizer.py", "core/intent_embedding.py", "core/provider_options.py",
             "core/skill_loader.py", "core/conversation_state.py", "core/concurrency.py", "core/safe_diagnostics.py", "core/llm_utils.py",
             "core/policy_grounding.py",
             "agents/agent_orchestrator.py", "agents/tools.py", "tooling/knowledge_base.py",
             "tooling/tool_manager.py", "tooling/mcp_client.py", "tooling/mcp_server.py",
             "memory/conversation_memory.py", "monitor/performance_monitor.py", "api/main.py",
             "evaluation/evaluator.py", "evaluation/few_shot.py", "evaluation/benchmark.py",
             "evaluation/benchmark_metrics.py", "knowledge/serveflow-demo-v2.json",
             "demo_business/fixtures.json", "demo_business/store.py", "demo_business/session.py",
             "config/mcp_servers.json", "requirements.txt", "requirements-dev.txt", "requirements-demo.txt",
             "tests/test_benchmark.py", "tests/test_benchmark_metrics.py",
             "tests/test_policy_grounding.py", "tests/test_policy_runtime.py",
             "tests/test_mysql_collection_contract.py"]
    fixed.extend(f"evaluation/benchmark/{suite}-v1.json" for suite in SUITES)
    fixed.extend(str(path.relative_to(root)) for path in sorted((root / "skills").rglob("SKILL.md")))
    return sorted(fixed)


def freeze(root: Path = BACKEND_ROOT) -> dict:
    suites = load_suites(root)
    summaries = [validate_suite(data, root) for data in suites.values()]
    files = {name: sha256(root / name) for name in artifact_paths(root)}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    return {
        "schema_version": "serveflow-benchmark-freeze-v1", "phase": "offline-preparation",
        "created_at": datetime.now(timezone.utc).isoformat(), "git_head": revision,
        "authorized_model_requests": 0, "executed_model_requests": 0,
        "live_trial_ready": False, "artifacts": files, "datasets": summaries,
        "model_plan": deepcopy(MODEL_PLAN),
        "comparison_plan": deepcopy(COMPARISON_PLAN),
        "scoring_policy": SCORING_POLICY, "limitations": list(LIMITATIONS),
    }


def verify_manifest(manifest: Any, root: Path = BACKEND_ROOT) -> None:
    expected = {"schema_version", "phase", "created_at", "git_head", "authorized_model_requests",
                "executed_model_requests", "live_trial_ready", "artifacts", "datasets", "model_plan",
                "comparison_plan", "scoring_policy", "limitations"}
    _exact(manifest, expected, "invalid_manifest_fields")
    if manifest["schema_version"] != "serveflow-benchmark-freeze-v1" or manifest["phase"] != "offline-preparation" or type(manifest["authorized_model_requests"]) is not int or manifest["authorized_model_requests"] != 0 or type(manifest["executed_model_requests"]) is not int or manifest["executed_model_requests"] != 0 or manifest["live_trial_ready"] is not False:
        _fail("invalid_offline_freeze")
    if (type(manifest["git_head"]) is not str or not re.fullmatch(r"[0-9a-f]{40,64}", manifest["git_head"])
            or manifest["model_plan"] != MODEL_PLAN or manifest["comparison_plan"] != COMPARISON_PLAN
            or manifest["scoring_policy"] != SCORING_POLICY or manifest["limitations"] != LIMITATIONS):
        _fail("frozen_protocol_changed")
    try:
        created = datetime.fromisoformat(manifest["created_at"])
        if created.tzinfo is None:
            _fail("invalid_manifest_time")
    except (TypeError, ValueError):
        _fail("invalid_manifest_time")
    if type(manifest["artifacts"]) is not dict or set(manifest["artifacts"]) != set(artifact_paths(root)):
        _fail("artifact_set_changed")
    for name, expected_hash in manifest["artifacts"].items():
        if not (root / name).is_file() or sha256(root / name) != expected_hash:
            _fail("frozen_artifact_changed")
    summaries = [validate_suite(data, root) for data in load_suites(root).values()]
    if summaries != manifest["datasets"]:
        _fail("frozen_dataset_summary_changed")


def write_new(path: Path, data: dict) -> None:
    """Private, exclusive output; never overwrite previous evidence."""
    payload = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > LIMIT:
        _fail("output_size_limit")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def result_template(manifest_path: Path, suite: str, split: str, arm: str) -> dict:
    verify_manifest(read_json(manifest_path))
    if type(suite) is not str or suite not in SUITES or type(split) is not str or split not in {"development", "validation"} or type(arm) is not str or arm not in ARMS[suite]:
        _fail("invalid_result_scope")
    return {"schema_version": "serveflow-benchmark-results-v1", "manifest_sha256": sha256(manifest_path),
            "suite": suite, "split": split, "arm": arm, "evidence_kind": "controlled", "records": []}


def score_results(manifest_path: Path, results: Any) -> dict:
    verify_manifest(read_json(manifest_path))
    _exact(results, {"schema_version", "manifest_sha256", "suite", "split", "arm", "evidence_kind", "records"}, "invalid_results_fields")
    if results["schema_version"] != "serveflow-benchmark-results-v1" or results["manifest_sha256"] != sha256(manifest_path):
        _fail("result_manifest_mismatch")
    suite, split = results["suite"], results["split"]
    if type(suite) is not str or suite not in SUITES or type(split) is not str or split not in {"development", "validation"} or type(results["arm"]) is not str or results["arm"] not in ARMS[suite]:
        _fail("invalid_result_scope")
    if type(results["evidence_kind"]) is not str or results["evidence_kind"] not in {"controlled", "saved-output-replay"}:
        _fail("live_evidence_not_supported_by_offline_tool")
    if type(results["records"]) is not list or len(results["records"]) > SUITES[suite][0]:
        _fail("invalid_result_records")
    if suite == "tasks" and results["evidence_kind"] == "controlled":
        for record in results["records"]:
            if type(record) is dict and record.get("review") is not None and (type(record["review"]) is not dict or record["review"].get("method") != "controlled-fixture"):
                _fail("controlled_review_mislabeled")
    from evaluation.benchmark_metrics import score_intent, score_retrieval, score_tasks
    scorer = {"intent": score_intent, "retrieval": score_retrieval, "tasks": score_tasks}[suite]
    cases = [case for case in load_suites()[suite]["cases"] if case["split"] == split]
    report = scorer(cases, results["records"])
    return {"schema_version": "serveflow-benchmark-report-v1", "manifest_sha256": sha256(manifest_path),
            "suite": suite, "split": split, "arm": results["arm"], "evidence_kind": results["evidence_kind"],
            "live_quality_claim_supported": False, "metrics": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    p = commands.add_parser("freeze"); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify"); p.add_argument("--manifest", type=Path, required=True)
    p = commands.add_parser("template")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--split", choices=["development", "validation"], required=True); p.add_argument("--arm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = {"datasets": [validate_suite(data) for data in load_suites().values()], "model_requests": 0}
        elif args.command == "freeze":
            value = freeze(); write_new(args.output, value)
            result = {"frozen": True, "manifest_sha256": sha256(args.output), "model_requests": 0, "live_trial_ready": False}
        elif args.command == "verify":
            verify_manifest(read_json(args.manifest)); result = {"verified": True, "model_requests": 0}
        elif args.command == "template":
            write_new(args.output, result_template(args.manifest, args.suite, args.split, args.arm))
            result = {"template_created": True, "executed_cases": 0, "model_requests": 0}
        else:
            report = score_results(args.manifest, read_json(args.results)); write_new(args.output, report)
            result = {"scored": True, "evidence_kind": report["evidence_kind"], "live_quality_claim_supported": False, "model_requests": 0}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        code = str(exc) if isinstance(exc, BenchmarkError) else "offline_input_or_output_invalid"
        print(json.dumps({"error": code, "model_requests": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
