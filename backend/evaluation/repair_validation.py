"""Offline repair acceptance preparation: questions, sources and SDK captures.

There is deliberately no authorize/run command, credential use/export, database,
index construction, Judge or real send. A future bounded sender needs its own
execution freeze and explicit allowance; closed historical runners stay closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from evaluation import benchmark as bench, expanded_validation as questions
from evaluation import intent_trial as base, diagnostic_repair_proof as ancestry
from evaluation import task_data, task_trial as engine
from evaluation import task_recheck as history
from evaluation.intent_transport import canonical_payload_hash
from evaluation.prospective_repaired_proof import _git_blobs
from evaluation.task_transport import PHASE_PARAMETERS, AGENT_PARAMETERS
from evaluation.expanded_budget import TASK_PHASE_CAPS
from evaluation.phase_deadline_transport import PHASE_HTTP_LIMITS


BASELINE = "d0d65f25e344eb133a614c8bf6b142e10a50620f"
DATASET = "evaluation/benchmark/repair-validation-v1/cases.json"
ROOT = bench.BACKEND_ROOT / "data/benchmarks/repair-validation-v1"
PREFLIGHT = bench.BACKEND_ROOT / "data/benchmarks/repair-validation-preflight-v1"
NEW_FILES = (DATASET, "evaluation/repair_validation.py", "tests/test_repair_validation.py")
PROPOSED_CEILING = 96
MATHEMATICAL_ENVELOPE = 8 * sum(TASK_PHASE_CAPS.values()) + 8 * 2
SCHEMA = "serveflow-repair-validation-preparation-v1"
OLD_DATASETS = {
    "tasks": ("evaluation/benchmark/expanded-tasks-v1.json",
              "4af1d5050b22a4302e473dc2ce9c2bb642e255e7b3e06827a8a8ac420a03ee2c",
              ("et01", "et06", "et20", "et22")),
    "retrieval": ("evaluation/benchmark/expanded-retrieval-v1.json",
                  "f410be09251a8691d82bd49607fcd7794a23a4e89c7ea71acb76bbd23e3f92b0",
                  ("er12", "er18", "er22", "er24")),
}
TASK_KEYS = {"id", "split", "family_id", "domain", "review_status", "input", "references",
             "collection_assertions", "expected", "execution_plan", "novelty", "related_prior_cases",
             "source_independence"}
RAG_KEYS = {"id", "split", "family_id", "input", "expected", "references", "tags", "novelty",
            "related_prior_cases", "review_status"}
RUBRIC = {
    "tasks": list(questions.DIMENSIONS),
    "task_checks": "actual native tool/handler, bound input, trusted identity, required calls and original status expectation before seven-dimension executor review",
    "task_gate": "all eight tasks pass actual execution and all applicable original questions; no blocking fact/condition/source/permission/coverage error, no unresolved review",
    "task_boundary": "correct refusal/clarification is not an executed write; unknown can be a complete evidence-boundary answer; no role-name or fixed-phrase gate",
    "unasked_fields": "do not require unasked fields; any extra facts must remain accurate; RT05 no compensation promise does not require a stock warning, RT08 unknown address is not a request to collect it",
    "retrieval_metrics": ["Hit@1", "Hit@3", "all-gold Top3", "macro document Recall@3", "MRR", "healthy no-material rejection", "source/structure validity"],
    "retrieval_gate": "five evidence questions Hit@3 and all-gold Top3 both 5/5; three out-of-scope questions healthy empty 3/3; all eight source/structure checks valid; failures stay in denominators",
    "retrieval_boundary": "three answerable and two insufficient questions separate; insufficient hit is not a complete answer, answerable empty is a miss, failure-empty is not a correct rejection",
    "reading_burden": "record necessary context and irrelevant material separately; no arbitrary word-count success gate",
    "comparison": "current production repair verification only; same-query vector versus enhanced RAG; historical scores separate, no concurrent old/new task A/B or general accuracy claim",
    "reviewer": "executor source review with cross-check, not independent human blind evaluation; no scoring model",
    "stop": "ordinary answer/structure failures remain and continue fixed order; accounting/version/data/auth/quota/rate-limit/network/timeout/cancel/unknown sends pause without retry or repair",
}
LIMITATIONS = [
    "eight known regressions plus eight author-written new phrasings; same six synthetic policies and ten synthetic orders",
    "new wording is not independent facts, unseen-to-authors data, blind evaluation or repeated stability evidence",
    "provided history is scripted input, not live memory or task recovery evidence",
    "no HTTP/browser, real merchant transaction, background memory, remote embedding or Judge acceptance",
    "only sixteen first SDK payloads can be captured now; later native/rerank payloads depend on outputs and require canonical+wire binding before each real send",
    "an execution adapter and independent sixteen-position durable budget remain to be implemented and tested; old 132-position or six-case ledgers cannot be repurposed",
    "96 is a proposed shared hard ceiling, not authorization or expected spending; mathematical 200 is not permission; stop if lower authorized ceiling is exhausted",
    "old 237/480 batch and unused 243 stay closed; no old output fills a new position or changes historical scores",
    "et08 label ambiguity and old D-p02 sum-comparison failure are not regraded or claimed repaired by this subset",
]


def validate(root=bench.BACKEND_ROOT):
    root = Path(root)
    data = bench.read_json(root / DATASET)
    bench._exact(data, {"schema_version", "candidate_business_commit", "provenance", "state",
                        "old_cases_preserved", "tasks", "retrieval"}, "repair_dataset_fields")
    if (data["schema_version"] != "serveflow-repair-validation-cases-v1"
            or data["candidate_business_commit"] != BASELINE
            or data["provenance"] != "author-source-reviewed-existing-synthetic-facts-not-independent-blind"
            or data["state"] != "offline-preparation-no-live-sender" or data["old_cases_preserved"] is not True):
        bench._fail("repair_dataset_version")
    corpus, tables = bench._sources(root)
    prior_inputs = set()
    for path in (root / "evaluation/benchmark").glob("*.json"):
        if path.name == Path(DATASET).name:
            continue
        for case in bench.read_json(path).get("cases", []):
            if type(case) is dict and type(case.get("input")) is dict:
                prior_inputs.add(questions._question_key(case))
    seen, inputs, summaries = set(), set(), {}
    for suite, prefix in (("tasks", "RT"), ("retrieval", "RR")):
        rows = data[suite]
        if type(rows) is not list or len(rows) != 8:
            bench._fail("repair_case_count")
        path, digest, old_ids = OLD_DATASETS[suite]
        if bench.sha256(root / path) != digest:
            bench._fail("repair_prior_dataset_changed")
        originals = {c["id"]: c for c in bench.read_json(root / path)["cases"]}
        kinds = Counter()
        for index, row in enumerate(rows, 1):
            bench._exact(row, {"id", "group", "legacy", "case"}, "repair_row_fields")
            if row["id"] != f"{prefix}{index:02d}" or row["id"] in seen:
                bench._fail("repair_identity_or_order")
            seen.add(row["id"])
            case = row["case"]
            bench._exact(case, TASK_KEYS if suite == "tasks" else RAG_KEYS, "repair_case_fields")
            group = "known_regression" if index <= 4 else "new_wording"
            if row["group"] != group or case["review_status"] != "reviewed":
                bench._fail("repair_group_or_review")
            if group == "known_regression":
                original = originals[old_ids[index-1]]
                expected = {"path": path, "dataset_sha256": digest, "case_id": original["id"],
                            "case_sha256": canonical_payload_hash(original)}
                if row["legacy"] != expected or case != original:
                    bench._fail("repair_old_case_rewritten")
            elif (row["legacy"] is not None or case["split"] != "validation"
                  or case["id"] != f"{'nt' if suite == 'tasks' else 'nr'}{index-4:02d}"):
                bench._fail("repair_new_case_identity")
            questions._input(case, suite)
            key = questions._question_key(case)
            if key in inputs or group == "new_wording" and key in prior_inputs:
                bench._fail("repair_duplicate_question")
            inputs.add(key)
            bench._references(case["references"], corpus, tables)
            expected = case["expected"]
            expected_keys = ({"required_tools", "required_facts", "missing_facts", "forbidden_claims",
                              "expected_status", "tool_task", "boundary", "rationale"} if suite == "tasks" else
                             {"kind", "relevant_titles", "required_facts", "missing_facts", "forbidden_claims", "rationale"})
            bench._exact(expected, expected_keys, "repair_expected_fields")
            for field in ("required_facts", "missing_facts", "forbidden_claims"):
                bench._strings(expected.get(field))
            if suite == "tasks":
                bench._exact(expected, {"required_tools", "required_facts", "missing_facts", "forbidden_claims",
                                       "expected_status", "tool_task", "boundary", "rationale"}, "repair_expected_fields")
                questions._collection_assertions(case, tables)
                bench._strings(expected["required_tools"])
                if (not set(expected["required_tools"]) <= bench.TOOLS
                        or case["execution_plan"].get("role_names_are_scoring_gate") is not False
                        or "required_agents" in expected
                        or expected["tool_task"] != bool(expected["required_tools"])
                        or any(type(expected[k]) is not bool for k in ("tool_task", "boundary"))
                        or expected["expected_status"] not in {"success", "partial", "not-prescribed"}):
                    bench._fail("repair_task_expectations")
            else:
                bench._exact(expected, {"kind", "relevant_titles", "required_facts", "missing_facts",
                                       "forbidden_claims", "rationale"}, "repair_expected_fields")
                kind, titles = expected["kind"], expected["relevant_titles"]
                bench._strings(titles)
                if (kind not in {"answerable", "insufficient", "out_of_scope"}
                        or any(ref["source"] != "knowledge" for ref in case["references"])
                        or set(titles) != {ref["title"] for ref in case["references"]}):
                    bench._fail("repair_retrieval_gold")
                if (kind == "out_of_scope" and (titles or expected["required_facts"] or not expected["missing_facts"])
                        or kind != "out_of_scope" and not titles
                        or kind == "answerable" and (not expected["required_facts"] or expected["missing_facts"])
                        or kind == "insufficient" and not expected["missing_facts"]):
                    bench._fail("repair_retrieval_applicability")
                kinds[kind] += 1
        summaries[suite] = {"total": 8, "known_regression": 4, "new_wording": 4, "kinds": dict(kinds)}
    if summaries["retrieval"]["kinds"] != {"answerable": 3, "insufficient": 2, "out_of_scope": 3}:
        bench._fail("repair_retrieval_balance")
    return summaries


def source_proof():
    """Current business bytes stay exactly d0d65f2; only this prep is added."""
    result = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
        cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15)
    if len(result.stdout) > 2*1024*1024:
        bench._fail("repair_source_paths")
    tracked = {p.removeprefix("backend/") for p in result.stdout.decode().split("\0") if p}
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    original = {p for p in tracked if (Path(p).parts[0] in ancestry._RUNTIME_FOLDERS
        and Path(p).suffix in ancestry._SOURCE_SUFFIXES and "__pycache__" not in Path(p).parts)
        or Path(p).parts[0] == "tests" and Path(p).suffix == ".py" or p in fixed}
    current = ancestry._current_sources()
    if set(current) != original | set(NEW_FILES) or set(NEW_FILES) & tracked:
        bench._fail("repair_unapproved_source_set")
    names, blobs = ["backend/"+p for p in sorted(original)], {}
    for offset in range(0, len(names), 256):
        blobs.update(_git_blobs(BASELINE, names[offset:offset+256]))
    import hashlib
    if any(current[name.removeprefix("backend/")] != hashlib.sha256(raw).hexdigest()
           for name, raw in blobs.items()):
        bench._fail("repair_business_changed")
    return {"baseline": BASELINE, "baseline_files": len(original), "current_files": len(current),
            "closed_additions": list(NEW_FILES), "business_modified": False, "artifacts": current}


def prior_proof():
    """Hash existing evidence, without replaying or reopening a closed sender."""
    snapshot_path = PREFLIGHT / "before-preservation.json"
    snapshot = bench.read_json(snapshot_path)
    paths = {str(p.resolve()) for p in (bench.BACKEND_ROOT / "data/benchmarks").rglob("*")
             if p.is_file() and not p.is_relative_to(ROOT) and not p.is_relative_to(PREFLIGHT)}
    if paths != set(snapshot["old_benchmark_files"]):
        bench._fail("repair_old_evidence_set_changed")
    for group in ("old_benchmark_files", "original_caches"):
        for name, digest in snapshot[group].items():
            path = Path(name)
            if path.is_symlink() or bench.sha256(path) != digest:
                bench._fail("repair_old_evidence_changed")
    old = bench.BACKEND_ROOT / "data/benchmarks/expanded-evaluation-v1"
    journal = (old / "ledger/journal.jsonl").read_bytes()
    if len(journal) > bench.LIMIT:
        bench._fail("repair_old_journal_limit")
    events = [json.loads(line) for line in journal.splitlines()]
    if (not events or sum(row.get("event") == "reserve" for row in events) != 237
            or sum(row.get("event") == "finish" for row in events) != 237
            or events[-1].get("chain_hash") != "c77ba84fdc7998d645efd7bcfc6f7cee7e91ccc79a1bb913979f1c4ac11b63b1"):
        bench._fail("repair_old_batch_state_changed")
    return {"snapshot_sha256": bench.sha256(snapshot_path), "files": len(paths),
        "caches": len(snapshot["original_caches"]),
        "previous_batch": {name: bench.sha256(old / name) for name in (
            "protocol.json", "authorization.json", "ledger/identity.json", "ledger/journal.jsonl")},
        "previous_attempts": 237, "previous_ceiling": 480,
        "previous_chain_tail": events[-1]["chain_hash"],
        "unspent_243_not_reused": True, "old_scores_changed": False}


def expected():
    config = base._configuration()  # public options only; never secret=True
    return {"schema_version": SCHEMA, "phase": "offline-preparation",
        "candidate_business_commit": BASELINE, "dataset_sha256": bench.sha256(bench.BACKEND_ROOT / DATASET),
        "datasets": validate(), "source_proof": source_proof(), "prior_proof": prior_proof(),
        "model": base.MODEL, "base_url": config["base_url"], "dependencies": engine._dependencies(),
        "effective_environment": history.effective_environment(), "http_limits": PHASE_HTTP_LIMITS,
        "phase_parameters": {k: v for k, v in PHASE_PARAMETERS.items() if k != "judge"},
        "agent_parameters": AGENT_PARAMETERS, "offline_data_proof": task_data.offline_data_proof(),
        "rubric": RUBRIC, "limitations": LIMITATIONS,
        "budget": {"proposed_shared_ceiling": PROPOSED_CEILING, "mathematical_phase_envelope": MATHEMATICAL_ENVELOPE,
                   "task_phase_caps": dict(TASK_PHASE_CAPS), "retrieval_phase_caps": {"rag_rewrite": 1, "rag_rerank": 1},
                   "maximum_in_flight": 3, "each_position_once": True, "automatic_retry": False},
        "authorized_model_requests": 0, "executed_model_requests": 0, "database_connections": 0,
        "judge_requests": 0, "external_business_requests": 0, "live_sender_ready": False,
        "plan": [{"id": row["id"], "suite": suite, "group": row["group"], "source_case_id": row["case"]["id"],
                  "case_sha256": canonical_payload_hash(row["case"])}
                 for suite in ("tasks", "retrieval") for row in bench.read_json(bench.BACKEND_ROOT / DATASET)[suite]]}


async def capture_requests(p):
    """MockTransport only; no credentials, real transport or data runtime."""
    data = bench.read_json(bench.BACKEND_ROOT / DATASET)
    captures = []
    from evaluation.retrieval_trial import capture_rewrite
    for suite in ("tasks", "retrieval"):
        for row in data[suite]:
            capture = await (base._capture(row["case"], p["base_url"]) if suite == "tasks"
                             else capture_rewrite(row["case"], p["base_url"]))
            capture.pop("operation_id", None)
            capture.update(position_id=row["id"], source_case_id=row["case"]["id"],
                           suite=suite, evidence_kind="offline-sdk-capture-not-live")
            captures.append(capture)
    return captures


async def prepare(root=ROOT):
    root = Path(root).absolute()
    base._path(root, "protocol.json")
    if root.exists():
        bench._fail("repair_preparation_exists")
    p = expected()
    captures = await capture_requests(p)
    # Detect changes during even this entirely local capture process.
    if p != expected():
        bench._fail("repair_preparation_changed_during_capture")
    value = json.loads(json.dumps({**p, "created_at": base._now(), "sdk_first_captures": captures,
        "prepared_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT, text=True).strip()}))
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    base._write(root, "protocol.json", value)
    return value


async def verify(root=ROOT):
    p = bench.read_json(base._path(Path(root), "protocol.json"))
    current = json.loads(json.dumps(expected()))
    bench._exact(p, set(current) | {"created_at", "prepared_git_head", "sdk_first_captures"}, "repair_protocol_fields")
    if any(p[k] != v for k, v in current.items()):
        bench._fail("repair_protocol_changed")
    captures = p["sdk_first_captures"]
    if type(captures) is not list or [r.get("position_id") for r in captures] != [r["id"] for r in current["plan"]]:
        bench._fail("repair_capture_order")
    data = bench.read_json(bench.BACKEND_ROOT / DATASET)
    by_id = {row["id"]: row for suite in ("tasks", "retrieval") for row in data[suite]}
    for capture, planned in zip(captures, current["plan"]):
        bench._exact(capture, {"case_id", "position_id", "source_case_id", "suite", "evidence_kind",
            "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}, "repair_capture_fields")
        row = by_id[planned["id"]]
        if (capture["case_id"] != row["case"]["id"] or capture["source_case_id"] != row["case"]["id"]
                or capture["suite"] != planned["suite"] or capture["evidence_kind"] != "offline-sdk-capture-not-live"
                or capture["input_sha256"] != canonical_payload_hash(bench.model_input(row["case"]))
                or capture["payload_sha256"] != canonical_payload_hash(capture["payload"])):
            bench._fail("repair_capture_binding")
        # HTTPX's actual captured serializer uses compact UTF-8 JSON. Verify
        # these exact first-send bytes independently of the canonical hash.
        raw = json.dumps(capture["payload"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        import hashlib
        if capture["request_bytes_sha256"] != hashlib.sha256(raw).hexdigest():
            bench._fail("repair_capture_wire_hash")
    # A changed payload cannot be legitimised by just updating both stored
    # hashes: compare with freshly captured, still wholly local SDK bytes.
    if captures != await capture_requests(current):
        bench._fail("repair_capture_changed")
    return {"verified": True, "positions": 16, "first_sdk_captures": 16,
            "authorized_model_requests": 0, "executed_model_requests": 0,
            "live_sender_ready": False, "quality_scores": None}


def main(argv=None):
    import asyncio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "verify"))
    args = parser.parse_args(argv)
    try:
        result = validate() if args.command == "validate" else asyncio.run(verify()) if args.command == "verify" else asyncio.run(prepare())
        print(json.dumps(result if args.command != "prepare" else {
            "prepared": True, "positions": 16, "first_sdk_captures": 16,
            "authorized_model_requests": 0, "executed_model_requests": 0, "live_sender_ready": False}, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"error": "repair_preparation_stopped_inspect_locally", "real_model_requests": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
