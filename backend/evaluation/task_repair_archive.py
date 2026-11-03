"""Read-only binding to the failed v2 trial, including its uncertain send.

This is historical integrity evidence, never authorization, recovery or rescoring.
No database, model, application configuration or old current-code verify is used.
"""
from __future__ import annotations

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_recheck as previous
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_budget import TaskBudget


ROOT = previous.ROOT
PROTOCOL_SHA256 = "dcb95517c17a164605acc5f0ded2c9867e4d03a57fa7eee480696e9d9be0bffc"
AUTHORIZATION_SHA256 = "5d7b131192ef4afb2faaf420e1e95565222cbcc4a982a85f83c918b3aea4672f"
JOURNAL_SHA256 = "2413d8061e4fb8199b64ca38688a0b541b868a68f7414e1335795cc5fb428627"
CHAIN_SHA256 = "7b21633b62451921b0095a9b749665affccf35487a6e72a92fca5bce8a53c568"
TREE_SHA256 = "dfa70ff5cdb89532b40633d1dac2881ab3c468df62d526b173a0412f0531db20"
EXECUTION_COMMIT = "3236161afa0174ba4e266f615c40446d0e7e8e9a"
PENDING = ["e04:turn-2:agent:general:1"]
STOP_FILES = ("operator-stop-v2.json", "supervisor-stop.json")


def paused_predecessor_proof():
    """Verify fixed old bytes and chains without interpreting them as new scores."""
    before = previous._tree(ROOT)
    if before != TREE_SHA256:
        base._fail("task_repair_paused_archive_changed")
    expected = {"protocol.json": PROTOCOL_SHA256, "authorization.json": AUTHORIZATION_SHA256,
                "ledger/journal.jsonl": JOURNAL_SHA256}
    if any(bench.sha256(base._path(ROOT, name)) != digest for name, digest in expected.items()):
        base._fail("task_repair_paused_archive_changed")
    protocol = bench.read_json(base._path(ROOT, "protocol.json"))
    if (protocol.get("schema_version") != "serveflow-task-protocol-v2"
            or protocol.get("batch") != previous.BATCH or protocol.get("git_head") != EXECUTION_COMMIT
            or protocol.get("root_binding_sha256") != base._root_hash(ROOT)
            or protocol.get("authorized_model_requests") != 0):
        base._fail("task_repair_paused_protocol_changed")
    # Bind the actual historical Git files, not today's repaired sources.
    previous.history._archived_sources(protocol, "artifacts")
    legacy = previous.archived_proof()
    if protocol.get("prior_proof") != legacy:
        base._fail("task_repair_paused_history_changed")
    budget = TaskBudget(ROOT / "ledger", PROTOCOL_SHA256, protocol["plan"])
    state, results = budget.status(), budget.results()
    if (state["attempts"] != 28 or state["chain_hash"] != CHAIN_SHA256
            or state["completed_cases"] != ["e01", "e02", "e03"]
            or state["active_case"] != "e04" or state["pending"] != PENDING
            or state["paused"] is not False or len(results["operations"]) != 27
            or set(results["cases"]) != {"e01", "e02", "e03"}):
        base._fail("task_repair_paused_ledger_changed")
    approval = bench.read_json(base._path(ROOT, "authorization.json"))
    if (approval.get("protocol_sha256") != PROTOCOL_SHA256 or approval.get("max_requests") != 168
            or approval.get("root_binding_sha256") != base._root_hash(ROOT)
            or approval.get("initial_chain_hash") != budget.initial_chain_hash
            or approval.get("plan") != protocol["plan"]):
        base._fail("task_repair_paused_authorization_changed")
    physical = {}
    for operation in results["operations"].values():
        name = f"physical-{operation['attempt']:03d}.json"
        digest = bench.sha256(base._path(ROOT, name))
        if operation["output_sha256"] != digest:
            base._fail("task_repair_paused_physical_changed")
        physical[name] = digest
    cases = {}
    for case_id, sealed in results["cases"].items():
        name = f"case-{case_id}.json"
        digest = bench.sha256(base._path(ROOT, name))
        if sealed["artifact_sha256"] != digest:
            base._fail("task_repair_paused_case_changed")
        cases[name] = digest
    stop = {name: bench.sha256(base._path(ROOT, name)) for name in STOP_FILES}
    if (ROOT / "physical-028.json").exists():
        base._fail("task_repair_unknown_send_was_rewritten")
    if previous._tree(ROOT) != before:
        base._fail("task_repair_archive_changed_during_read")
    return {"schema_version": "serveflow-paused-task-proof-v1", "tree_sha256": before,
        "protocol_sha256": PROTOCOL_SHA256, "authorization_sha256": AUTHORIZATION_SHA256,
        "journal_sha256": JOURNAL_SHA256, "chain_hash": CHAIN_SHA256,
        "old_reserved_requests": 28, "old_sealed_physical_results": 27,
        "old_unknown_operations": list(PENDING), "old_active_case": "e04",
        "old_completed_cases": list(state["completed_cases"]),
        "physical_sha256": physical, "case_sha256": cases, "stop_sha256": stop,
        "prior_history_sha256": canonical_payload_hash(legacy), "old_allowance_reused": False,
        "old_scores_recomputed": False, "new_authorized_requests": 0}
