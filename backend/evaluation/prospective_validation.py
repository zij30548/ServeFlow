"""Offline question/rubric freeze; no live runner, clients, dotenv or databases.

The new questions share the existing synthetic sources. This freezes a design,
not complete SDK payloads or an authorization to execute it. Old datasets,
reports and ledgers are read-only. CLI: validate / prepare / verify.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess

from evaluation import benchmark as bench


ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-validation-v1"
SCHEMA = "serveflow-prospective-cases-v1"
PROVENANCE = "new-authored-questions-shared-sources-not-blind"
SUITES = ("tasks", "retrieval")
DOMAINS = ("orders", "refunds", "entitlements", "technical", "compound", "boundary")
DIMENSIONS = ("evidence_relevance", "coverage", "conditions_preserved", "missing_correct",
              "status_correct", "citations_correct", "tool_permissions")


def _cases(root, suite):
    return bench.read_json(root / f"evaluation/benchmark/prospective-{suite}-v1.json")


def _prior(root):
    """Only source cases, never old model outputs, are used to check overlap."""
    values = []
    for name in ("intent-v1", "intent-recheck-v2", "retrieval-v1", "tasks-v1", "task-e2e-v1"):
        values.extend(bench.read_json(root / f"evaluation/benchmark/{name}.json")["cases"])
    ids, messages = set(), set()
    for case in values:
        ids.add(case["id"])
        text = case.get("input", {}).get("message")
        if text:
            messages.add(bench._normal(text))
        for turn in case.get("turns", []):
            messages.add(bench._normal(turn))
    return ids, messages


def validate_suite(data, root=bench.BACKEND_ROOT):
    bench._exact(data, {"schema_version", "suite", "dataset_id", "provenance", "cases"}, "invalid_prospective_dataset")
    suite = data["suite"]
    if (type(suite) is not str or suite not in SUITES or data["schema_version"] != SCHEMA
            or data["dataset_id"] != f"prospective-{suite}-v1" or data["provenance"] != PROVENANCE):
        bench._fail("invalid_prospective_version")
    cases = data["cases"]
    if type(cases) is not list or len(cases) != 12:
        bench._fail("unexpected_prospective_count")
    corpus, tables = bench._sources(root)
    old_ids, old_messages = _prior(root)
    prefix, messages, domains, kinds = ("p" if suite == "tasks" else "q"), set(), Counter(), Counter()
    tools, boundaries = 0, 0
    for index, case in enumerate(cases, 1):
        fields = {"id", "input", "expected", "references", "related_prior_cases"}
        bench._exact(case, fields | ({"domain"} if suite == "tasks" else set()), "invalid_prospective_case")
        if case["id"] != f"{prefix}{index:02d}":
            bench._fail("invalid_prospective_order_or_id")
        bench._strings(case["related_prior_cases"])
        if not set(case["related_prior_cases"]) <= old_ids:
            bench._fail("unknown_prior_case")
        inp = case["input"]
        bench._exact(inp, {"message", "history"} | ({"customer_id"} if suite == "tasks" else set()), "invalid_prospective_input")
        bench._text(inp["message"])
        # This first new set is explicitly single-turn. Do not invent model history.
        if type(inp["history"]) is not list or inp["history"]:
            bench._fail("prospective_single_turn_required")
        normalized = bench._normal(inp["message"])
        if normalized in old_messages or normalized in messages:
            bench._fail("prospective_question_overlap")
        messages.add(normalized)
        exp = case["expected"]
        common = {"required_facts", "missing_facts", "forbidden_claims", "rationale"}
        extra = {"required_tools", "expected_status", "tool_task", "boundary"} if suite == "tasks" else {"kind", "relevant_titles"}
        bench._exact(exp, common | extra, "invalid_prospective_expectation")
        for name in ("required_facts", "missing_facts", "forbidden_claims"):
            bench._strings(exp[name], allow_empty=name != "forbidden_claims")
        bench._text(exp["rationale"])
        bench._references(case["references"], corpus, tables)
        if suite == "tasks":
            if type(case["domain"]) is not str or case["domain"] not in DOMAINS:
                bench._fail("unknown_prospective_domain")
            domains[case["domain"]] += 1
            if type(inp["customer_id"]) is not str or inp["customer_id"] not in {"demo-customer-a", "demo-customer-b"}:
                bench._fail("invalid_fixture_identity")
            bench._strings(exp["required_tools"])
            if not set(exp["required_tools"]) <= bench.TOOLS:
                bench._fail("unknown_prospective_tool")
            if (type(exp["tool_task"]) is not bool or type(exp["boundary"]) is not bool
                    or exp["tool_task"] != bool(exp["required_tools"])
                    or exp["boundary"] != (case["domain"] == "boundary")
                    or type(exp["expected_status"]) is not str
                    or exp["expected_status"] not in {"success", "partial", "not-prescribed"}
                    or not exp["required_facts"] or not case["references"]):
                bench._fail("contradictory_prospective_task")
            tools += exp["tool_task"]
            boundaries += exp["boundary"]
        else:
            kind = exp["kind"]
            if type(kind) is not str or kind not in {"answerable", "insufficient", "out_of_scope"}:
                bench._fail("unknown_retrieval_kind")
            kinds[kind] += 1
            bench._strings(exp["relevant_titles"])
            cited = {ref.get("title") for ref in case["references"] if ref.get("source") == "knowledge"}
            if (set(exp["relevant_titles"]) != cited or not cited <= set(corpus)
                    or any(ref["source"] != "knowledge" for ref in case["references"])
                    or (kind == "answerable" and (not cited or not exp["required_facts"] or exp["missing_facts"]))
                    or (kind == "insufficient" and (not cited or not exp["missing_facts"]))
                    or (kind == "out_of_scope" and (cited or case["references"] or exp["required_facts"] or not exp["missing_facts"]))):
                bench._fail("contradictory_prospective_retrieval")
    if suite == "tasks":
        if domains != Counter({name: 2 for name in DOMAINS}):
            bench._fail("unexpected_prospective_domain_balance")
        return {"suite": suite, "total": 12, "supported_completion": 12-boundaries,
                "tool_tasks": tools, "boundary_handling": boundaries, "domains": dict(domains)}
    if kinds != Counter(answerable=6, insufficient=3, out_of_scope=3):
        bench._fail("unexpected_prospective_retrieval_balance")
    return {"suite": suite, "total": 12, "kinds": dict(kinds), "evidence_denominator": 9,
            "out_of_scope_denominator": 3}


def validate(root=bench.BACKEND_ROOT):
    return [validate_suite(_cases(root, suite), root) for suite in SUITES]


def design():
    """Shared safety, rubric and stopping rules, not arm-specific pass conditions."""
    return {
        "tasks": {"arms": {"A": {"force_fresh_read": False, "typed_order_projection": False},
                           "D": {"force_fresh_read": True, "typed_order_projection": True}},
            "common": ["same current required-tool menu fix, model, prompts, skills, routing and tool permissions",
                       "same trusted identity, actual tools, static fixture, source data and phase budgets",
                       "isolated case/arm histories and caches; A may voluntarily read, never deliberately blocked"],
            "primary_gate": ["bound complete execution and actual permitted tool/source evidence",
                             "all seven semantic dimensions and no blocking error"],
            "dimensions": list(DIMENSIONS),
            "not_primary": ["verbatim program text", "required agent names", "using the enhancement itself", "Judge threshold"],
            "status": "not-prescribed permits success or partial only with a truthful complete scope; requested unsupported mutations require partial",
            "authorization_boundary": "p11 may safely refuse before tools; if it reads, trusted B identity and no disclosure still apply; no read is required merely to pass",
            "scope": "combined mechanism contrast, not individual factor attribution or single-agent vs multi-agent comparison"},
        "retrieval": {"arms": {"A": "original query local vector Top3, no model API",
                               "B": "existing rewrite + original query first + per-query Top5 + dedup + rerank Top3"},
            "common": ["same six full documents/order/chunks and cached local embedding/index settings",
                       "same original query, no tuned threshold, same 200-character rerank window and existing parser/fallback"],
            "metrics": ["Hit@1", "Hit@3", "macro document Recall@3", "all gold in Top3", "MRR", "out-of-scope mis-hit"],
            "scope": "retrieval only; insufficient hits do not answer missing facts; false retrieval is not observed answer hallucination"},
        "review": {"method": "executor rubric review with source and actual native/handler cross-check; not independent human blind review",
            "include": ["all planned cases, failures, unknowns, unexecuted and pending semantic reviews",
                        "paired both/only-A/only-enhanced/neither counts and each dimension separately",
                        "reading burden, extra conservative reminders, usage/unknowns and timing scope"],
            "statistics": "descriptive small-sample results first; any paired test is exploratory, with method and all tested contrasts disclosed; no target p-value or sample extension for significance"},
        "stop": ["no live sends in this preparation; authorization and executed calls remain zero",
                 "before later execution freeze actual SDK input, model/parameters, order, phase budgets and durable shared ledger",
                 "planned case/arm at most once, no retry or selecting best samples; ordinary quality errors stay in denominator",
                 "stop on version/accounting/auth/quota/network/unknown-send failure; no unplanned continuation",
                 "do not tune on these validation outputs or replace failures; later repairs require separate development and new validation"],
        "limitations": ["new authored expressions/combinations, shared known source facts; not independent human or representative population test",
                        "all 24 questions are single-turn; memory, dynamic record freshness, browser and production performance are not tested",
                        "current RAG has no rejection threshold; out-of-scope failures must remain visible, not removed to obtain a good score",
                        "no implication of production quotas, enterprise accuracy, cost savings or new architecture efficacy"],
    }


def _sources(root):
    # Dynamic enumeration detects additions as well as edits. Never include data,
    # credentials, caches, compiled media or a user's private environment.
    names = set(bench.artifact_paths(root))
    for folder in ("agents", "core", "tooling", "memory", "monitor", "api", "demo_business", "evaluation", "skills", "config"):
        if (root / folder).is_symlink():
            bench._fail("prospective_source_symlink")
        for path in (root / folder).rglob("*"):
            if path.is_symlink():
                bench._fail("prospective_source_symlink")
            if "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md", ".txt"} and path.is_file():
                if path.resolve() != path:
                    bench._fail("prospective_source_symlink")
                names.add(str(path.relative_to(root)))
    names.add("tests/test_prospective_validation.py")
    for name in names:
        if (root / name).is_symlink() or (root / name).resolve() != root / name:
            bench._fail("prospective_source_symlink")
    return {name: bench.sha256(root / name) for name in sorted(names)}


def _plan(root):
    output = []
    for suite in SUITES:
        for index, case in enumerate(_cases(root, suite)["cases"]):
            arms = ("A", "D") if suite == "tasks" else ("A", "B")
            order = list(arms if index % 2 == 0 else arms[::-1])
            output.append({"suite": suite, "case_id": case["id"], "arm_order": order,
                           "conversation_input": bench.model_input(case),
                           "trusted_context": bench.fixture_context(case),
                           "case_sha256": _hash(case)})
    return output


def _hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _dependencies():
    return {"python": platform.python_version(), **{name: metadata.version(name)
            for name in ("anthropic", "httpx", "chromadb", "onnxruntime", "numpy")}}


def prepare(directory=ROOT, root=bench.BACKEND_ROOT):
    directory = Path(directory).absolute()
    if directory.exists() or directory.resolve() != directory:
        bench._fail("prospective_directory_already_exists_or_symlink")
    summaries = validate(root)
    value = {"schema_version": "serveflow-prospective-design-freeze-v1", "phase": "offline-preparation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                        capture_output=True, text=True).stdout.strip(),
        "root_binding_sha256": _hash(str(directory)), "sources": _sources(root), "dependencies": _dependencies(),
        "suites": summaries, "design": design(), "plan": _plan(root),
        "authorized_model_requests": 0, "executed_model_requests": 0, "external_business_requests": 0,
        "live_payloads_frozen": False, "live_trial_ready": False,
        "future_budget": {"retrieval_B_physical_upper_bound": 24, "task_physical_upper_bound": None,
                          "status": "proposal only; task multi-stage accounting and execution freeze still required"}}
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    bench.write_new(directory / "protocol.json", value)
    return value


def verify(directory=ROOT, root=bench.BACKEND_ROOT):
    directory = Path(directory).absolute()
    if directory.resolve() != directory or (directory / "protocol.json").is_symlink():
        bench._fail("prospective_manifest_symlink")
    value = bench.read_json(directory / "protocol.json")
    fields = {"schema_version", "phase", "created_at", "source_commit", "root_binding_sha256", "sources", "dependencies",
              "suites", "design", "plan", "authorized_model_requests", "executed_model_requests", "external_business_requests",
              "live_payloads_frozen", "live_trial_ready", "future_budget"}
    bench._exact(value, fields, "invalid_prospective_freeze")
    if (value["schema_version"] != "serveflow-prospective-design-freeze-v1" or value["phase"] != "offline-preparation"
            or type(value["source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None
            or value["root_binding_sha256"] != _hash(str(directory)) or value["sources"] != _sources(root)
            or value["dependencies"] != _dependencies() or value["suites"] != validate(root)
            or value["design"] != design() or value["plan"] != _plan(root)
            or any(type(value[name]) is not int or value[name] != 0
                   for name in ("authorized_model_requests", "executed_model_requests", "external_business_requests"))
            or value["live_payloads_frozen"] is not False or value["live_trial_ready"] is not False
            or value["future_budget"] != {"retrieval_B_physical_upper_bound": 24, "task_physical_upper_bound": None,
                "status": "proposal only; task multi-stage accounting and execution freeze still required"}):
        bench._fail("prospective_frozen_design_changed")
    bench._text(value["created_at"], 100)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "verify"))
    command = parser.parse_args().command
    try:
        result = validate() if command == "validate" else prepare() if command == "prepare" else verify()
    except (bench.BenchmarkError, OSError, metadata.PackageNotFoundError):
        # Do not print submitted data, environment values or private paths.
        parser.exit(2, "Offline validation stopped; inspect frozen inputs without overwriting them.\n")
    if type(result) is dict:
        result = {"schema_version": result["schema_version"], "suites": result["suites"],
                  "authorized_model_requests": 0, "executed_model_requests": 0, "live_trial_ready": False}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
