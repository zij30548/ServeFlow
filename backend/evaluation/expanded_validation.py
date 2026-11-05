"""Offline expansion of evaluation questions; no clients, database or sends.

``prepare`` freezes question/source hashes only. It does not freeze SDK payloads,
grant a model allowance, execute the questions or create a quality score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

from evaluation import benchmark as bench


ROOT = bench.BACKEND_ROOT / "data/benchmarks/expanded-validation-v1"
SUITES = {"intents": (60, 24, 36), "tasks": (24, 8, 16), "retrieval": (24, 8, 16)}
SCHEMA = "serveflow-expanded-question-freeze-v1"
DIMENSIONS = ("evidence_relevance", "coverage", "conditions_preserved",
              "missing_correct", "status_correct", "citations_correct", "tool_permissions")


def _path(root, suite):
    return root / f"evaluation/benchmark/expanded-{suite}-v1.json"


def prior_cases(root):
    """Only source questions, never prior model observations or grades."""
    values = []
    for path in sorted((root / "evaluation/benchmark").glob("*.json")):
        if path.name.startswith("expanded-"):
            continue
        values.extend(bench.read_json(path).get("cases", []))
    return values


def _question_key(case):
    inp = bench.model_input(case)
    return (bench._normal(inp["message"]), tuple(
        (row["role"], bench._normal(row["content"])) for row in inp["history"]))


def _input(case, suite):
    value = case["input"]
    bench._exact(value, {"message", "history"} | ({"customer_id"} if suite == "tasks" else set()),
                 "expanded_input_fields")
    bench._text(value["message"])
    if type(value["history"]) is not list or len(value["history"]) > 12:
        bench._fail("expanded_history_limit")
    for turn in value["history"]:
        bench._exact(turn, {"role", "content"}, "expanded_history_fields")
        if turn["role"] not in ("user", "assistant"):
            bench._fail("expanded_history_role")
        bench._text(turn["content"])
    if suite == "tasks" and value["customer_id"] not in ("demo-customer-a", "demo-customer-b"):
        bench._fail("expanded_trusted_identity")


def _collection_assertions(case, tables):
    assertions = case.get("collection_assertions", [])
    if type(assertions) is not list or len(assertions) > 12:
        bench._fail("expanded_collection_assertions")
    for assertion in assertions:
        bench._exact(assertion, {"source", "table", "filters", "expected_count"}, "expanded_collection_fields")
        table, filters, count = assertion["table"], assertion["filters"], assertion["expected_count"]
        if (assertion["source"] != "fixture" or type(table) is not str or table not in tables
                or type(filters) is not dict or not filters or type(count) is not int or count < 0):
            bench._fail("expanded_collection_fields")
        rows = list(tables[table].values())
        if any(not any(k in row for row in rows) for k in filters):
            bench._fail("expanded_collection_filter")
        actual = sum(all(k in row and type(row[k]) is type(v) and row[k] == v for k, v in filters.items()) for row in rows)
        if actual != count:
            bench._fail("expanded_collection_count")


def _sources(data, root):
    """Validate declared source hashes; only repository source paths are allowed."""
    declarations = data.get("sources", data.get("source_bindings", [data["source"]] if "source" in data else []))
    if type(declarations) is not list or not declarations:
        bench._fail("expanded_source_bindings")
    result = {}
    for value in declarations:
        if type(value) is not dict:
            bench._fail("expanded_source_binding")
        name, digest = value.get("path"), value.get("sha256")
        if type(name) is not str or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            bench._fail("expanded_source_binding")
        if name.startswith("backend/"):
            name = name[len("backend/"):]
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] not in ("core", "agents", "evaluation", "knowledge", "demo_business", "skills")
                or path.suffix not in (".py", ".json", ".md") or name in result
                or (root / path).is_symlink() or (root / path).resolve() != root.resolve() / path):
            bench._fail("expanded_source_path")
        if bench.sha256(root / path) != digest:
            bench._fail("expanded_source_changed")
        result[name] = digest
    return result


def validate(root=bench.BACKEND_ROOT):
    root = Path(root)
    corpus, tables = bench._sources(root)
    old = prior_cases(root)
    old_ids = {v["id"] for v in old if "id" in v}
    old_inputs = {_question_key(v) for v in old
                  if isinstance(v.get("input"), dict) and "message" in v["input"]}
    all_ids, all_inputs, summaries = set(), set(), []
    for suite, (total, development, validation) in SUITES.items():
        data = bench.read_json(_path(root, suite))
        if type(data) is not dict or data.get("schema_version") != f"serveflow-expanded-{suite}-v1":
            bench._fail("expanded_version")
        bindings = _sources(data, root)
        cases = data.get("cases")
        if type(cases) is not list or len(cases) != total:
            bench._fail("expanded_count")
        families, splits, statuses, labels, kinds = {}, Counter(), Counter(), Counter(), Counter()
        for case in cases:
            if type(case) is not dict:
                bench._fail("expanded_case_fields")
            for field in ("id", "split", "family_id", "novelty", "review_status"):
                bench._text(case.get(field), 4000)
            if case["id"] in all_ids or case["id"] in old_ids:
                bench._fail("expanded_duplicate_id")
            all_ids.add(case["id"])
            if case["split"] not in ("development", "validation"):
                bench._fail("expanded_split")
            previous = families.setdefault(case["family_id"], case["split"])
            if previous != case["split"]:
                bench._fail("expanded_family_leakage")
            splits[case["split"]] += 1
            if case["review_status"] not in ("reviewed", "needs_adjudication"):
                bench._fail("expanded_review_status")
            statuses[case["review_status"]] += 1
            _input(case, suite)
            key = _question_key(case)
            if key in all_inputs or key in old_inputs:
                bench._fail("expanded_duplicate_question")
            all_inputs.add(key)
            if type(case.get("expected")) is not dict:
                bench._fail("expanded_expected_fields")
            expected = case["expected"]
            if suite == "intents":
                if type(expected.get("intent")) is not str or expected["intent"] not in bench.BENCHMARK_INTENTS:
                    bench._fail("expanded_intent_label")
                refs = case.get("references")
                if type(refs) is not list or not refs:
                    bench._fail("expanded_intent_reference")
                for ref in refs:
                    bench._exact(ref, {"source", "symbol", "quote"}, "expanded_intent_reference")
                    if (ref["source"] != "backend/core/intent_recognizer.py"
                            or "core/intent_recognizer.py" not in bindings
                            or ref["symbol"] != "IntentCategory." + expected["intent"].upper()
                            or type(ref["quote"]) is not str or not ref["quote"].strip()
                            or ref["quote"] not in (root / "core/intent_recognizer.py").read_text()):
                        bench._fail("expanded_intent_reference")
                labels[(case["split"], expected["intent"])] += 1
            else:
                bench._references(case.get("references"), corpus, tables)
                for field in ("required_facts", "missing_facts", "forbidden_claims"):
                    bench._strings(expected.get(field))
                if suite == "retrieval":
                    kind = expected.get("kind")
                    if kind not in ("answerable", "insufficient", "out_of_scope"):
                        bench._fail("expanded_retrieval_kind")
                    titles = expected.get("relevant_titles")
                    bench._strings(titles)
                    cited = {ref["title"] for ref in case["references"] if ref["source"] == "knowledge"}
                    if set(titles) != cited or any(ref["source"] != "knowledge" for ref in case["references"]):
                        bench._fail("expanded_retrieval_gold_reference")
                    if (kind == "out_of_scope" and (titles or expected["required_facts"] or not expected["missing_facts"])):
                        bench._fail("expanded_retrieval_empty_gold")
                    if kind != "out_of_scope" and not titles:
                        bench._fail("expanded_retrieval_missing_gold")
                    if ((kind == "answerable" and (not expected["required_facts"] or expected["missing_facts"]))
                            or (kind == "insufficient" and not expected["missing_facts"])):
                        bench._fail("expanded_retrieval_applicability")
                    kinds[(case["split"], kind)] += 1
                else:
                    _collection_assertions(case, tables)
                    plan = case.get("execution_plan")
                    if (type(plan) is not dict or plan.get("role_names_are_scoring_gate") is not False
                            or "required_agents" in expected):
                        bench._fail("expanded_role_not_scoring_gate")
                    bench._strings(expected.get("required_tools"))
                    if not set(expected["required_tools"]) <= bench.TOOLS:
                        bench._fail("expanded_unknown_tool")
                    if expected.get("expected_status") not in ("success", "partial", "not-prescribed"):
                        bench._fail("expanded_expected_status")
                    if any(type(expected.get(k)) is not bool for k in ("tool_task", "boundary")):
                        bench._fail("expanded_task_applicability")
        if splits != Counter(development=development, validation=validation):
            bench._fail("expanded_split_count")
        if suite == "intents" and labels != Counter({(s, label): n for s, n in (("development", 2), ("validation", 3)) for label in bench.BENCHMARK_INTENTS}):
            bench._fail("expanded_intent_balance")
        if suite == "retrieval" and any(kinds[(split, kind)] == 0 for split in splits for kind in ("answerable", "insufficient", "out_of_scope")):
            bench._fail("expanded_retrieval_negative_coverage")
        summaries.append({"suite": suite, "total": total, "splits": dict(splits),
                          "families": len(families), "review_status": dict(statuses),
                          "kinds": {f"{s}:{k}": n for (s, k), n in kinds.items()}})
    return summaries


def artifacts(root):
    existing = set(bench.artifact_paths(root))
    existing.update(str(p.relative_to(root)) for p in (root / "evaluation/benchmark").glob("*.json"))
    existing.update({"evaluation/expanded_validation.py", "tests/test_expanded_validation.py"})
    for suite in SUITES:
        existing.update(_sources(bench.read_json(_path(root, suite)), root))
    return {name: bench.sha256(root / name) for name in sorted(existing)}


def freeze(root=bench.BACKEND_ROOT):
    root = Path(root)
    summaries = validate(root)
    if any(s["review_status"].get("needs_adjudication", 0) for s in summaries):
        bench._fail("expanded_unresolved_labels")
    return {"schema_version": SCHEMA, "phase": "offline-question-preparation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "artifacts": artifacts(root), "datasets": summaries,
            "authorized_model_requests": 0, "executed_model_requests": 0,
            "sdk_payloads_frozen": False, "live_trial_ready": False,
            "development_before_validation": True,
            "comparison": {"intents": "same model observation: LLM-only versus fusion",
                           "tasks": "current common safety base: A versus D; every selected case/arm once",
                           "retrieval": "same corpus: vector Top3 versus rewrite/recall/deduplicate/rerank"},
            "rubric": {"tasks": list(DIMENSIONS),
                       "retrieval": ["Hit@1", "Hit@3", "MRR", "macro Recall@3", "all-gold Top3", "out-of-scope false hits"],
                       "intents": ["Accuracy", "fixed-label Macro-F1"]},
            "limits": ["new authored wording and combinations, existing synthetic facts; no independent blind claim",
                       "exact deduplication and family separation do not prove semantic independence",
                       "scripted histories are test inputs, not evidence of live multi-turn memory",
                       "case review is source-based; current system outputs were not used to choose labels",
                       "no new quality scores; no reuse of old outputs or allowances",
                       "actual bounded SDK payload/operation plan and fresh applicable authorization still required"]}


def verify(value, root=bench.BACKEND_ROOT):
    current = freeze(root)
    bench._exact(value, set(current), "expanded_freeze_fields")
    if any(value[k] != v for k, v in current.items() if k not in ("created_at", "git_head")):
        bench._fail("expanded_freeze_changed")
    return {"verified": True, "question_cases": sum(s["total"] for s in current["datasets"]),
            "authorized_model_requests": 0, "executed_model_requests": 0, "live_trial_ready": False}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "verify"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "questions.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = {"datasets": validate(), "model_requests": 0, "live_trial_ready": False}
        elif args.command == "prepare":
            result = freeze(); bench.write_new(args.manifest, result)
        else:
            result = verify(bench.read_json(args.manifest))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (bench.BenchmarkError, OSError) as error:
        print(json.dumps({"error": str(error) if isinstance(error, bench.BenchmarkError) else "expanded_io_error"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
