"""Stage-clock runner checks; controlled preparation, never real sends/stores."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.model_deadline import ModelDeadlineContextError, assert_model_active, await_model, overhead
from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as business, task_deadline_recheck as module
from evaluation import task_deadline_transport
from evaluation.intent_transport import canonical_payload_hash


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PROOF = {"kind": "controlled-fixture"}
ORIGINAL_ARCHIVED, ORIGINAL_PRIOR, ORIGINAL_ARTIFACTS = module._archived, module.prior_proof, module._artifacts


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def deny(*args, **kwargs): raise AssertionError("deadline tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for key in tuple(os.environ):
        if key.startswith("ECHOMIND_"): monkeypatch.delenv(key)
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    reads = []
    def configuration(*, secret=False):
        reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": "controlled-not-a-key"} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"model": base.MODEL, "max_tokens": 256, "temperature": .1, "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": case["input"]["message"]}]}
        return {"case_id": case["id"], "operation_id": "controlled", "input_sha256": canonical_payload_hash(case["input"]),
                "payload": payload, "payload_sha256": canonical_payload_hash(payload),
                "request_bytes_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(business.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(business.engine, "offline_data_proof", lambda: {"controlled": True})
    monkeypatch.setattr(business, "_artifacts", lambda: {"core/controlled.py": "a" * 64})
    monkeypatch.setattr(module, "_artifacts", lambda: {"core/controlled.py": "a" * 64, "evaluation/clock.py": "b" * 64})
    monkeypatch.setattr(business, "prior_proof", lambda: {"controlled-older": "c" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-v5": "d" * 64, "old_allowance_reused": False})
    monkeypatch.setattr(business.repair, "data_binding", lambda: canonical_payload_hash(PROOF))
    return reads


def prepared(tmp_path):
    trial = module.DeadlineRecheck(tmp_path / "new-clock")
    return trial, asyncio.run(trial.prepare())


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def approved(trial):
    return trial.authorize("Controlled test permission, not real authorization", confirmed=True)


def test_prepare_keeps_questions_payload_parameters_and_rubric_and_grants_zero_sends(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-task-business-protocol-v6-stage-clock"
    assert p["batch"] == module.BATCH and p["candidate"] == module.CANDIDATE
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 312
    assert p["limits"] == business.LIMITS and p["http_limits"] == business.engine.HTTP_LIMITS
    assert p["http_limits"]["request_deadline_s"] == 8
    assert p["plan"] == business.fixed_plan() and len(p["initial_intent_requests"]) == 12
    assert sum(len(c["turn_ids"]) for c in p["plan"]) == 13
    assert p["cases_sha256"] == canonical_payload_hash({"cases": business.cases()})
    assert p["review_policy"] == business.REVIEW_POLICY
    assert p["policy"] == {**business.POLICY, "stage_clock": module.CLOCK_POLICY}
    assert trial.verify() == p and trial.ledger(p).status()["attempts"] == 0
    assert not any(isolated)


def test_new_authorization_binds_root_protocol_and_genesis_without_overwriting(tmp_path):
    trial, p = prepared(tmp_path)
    with pytest.raises(base.TrialError, match="explicit_authorization"): trial.authorize("not confirmed")
    auth = approved(trial)
    assert auth["schema_version"] == "serveflow-task-business-authorization-v6-stage-clock"
    assert auth["max_requests"] == 312 and auth["plan"] == p["plan"]
    assert auth["protocol_sha256"] == bench.sha256(trial.root / "protocol.json")
    assert auth["root_binding_sha256"] == base._root_hash(trial.root)
    assert auth["initial_chain_hash"] == trial.ledger(p).initial_chain_hash
    before = tree(trial.root)
    with pytest.raises(FileExistsError): approved(trial)
    assert tree(trial.root) == before


def test_old_v5_authorization_cannot_create_store_or_client(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    auth = approved(trial)
    auth["schema_version"] = "serveflow-task-business-authorization-v5-observation"
    (trial.root / "authorization.json").write_text(json.dumps(auth))
    store = AsyncMock(); monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError, match="authorization_changed"): asyncio.run(trial.execute())
    store.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0


def test_pending_new_attempt_stays_occupied_and_cannot_be_replayed_or_reinitialized(tmp_path):
    trial, p = prepared(tmp_path); approved(trial)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        ledger.reserve("e01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    with pytest.raises(business.accounting.TaskBudgetError, match="pending"):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None),
            controlled_data=SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=AsyncMock())))
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())
    assert ledger.status()["attempts"] == 1 and len(ledger.status()["pending"]) == 1
    assert tree(trial.root) == before


@pytest.mark.parametrize("damage", ["clock_scope", "stage_budget", "source", "history", "data", "labels", "model"])
def test_changed_freeze_stops_before_model_or_store(tmp_path, monkeypatch, damage):
    trial, p = prepared(tmp_path); approved(trial)
    if damage == "clock_scope": monkeypatch.setitem(module.CLOCK_POLICY, "excluded_scopes", ["network"])
    elif damage == "stage_budget": monkeypatch.setitem(module.CLOCK_POLICY, "model_stage_budgets", {"intent": 60})
    elif damage == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"changed": "f" * 64})
    elif damage == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    elif damage == "data": monkeypatch.setattr(business.repair, "data_binding", lambda: "f" * 64)
    else:
        value = deepcopy(p); value["cases_sha256" if damage == "labels" else "model"] = "changed-after-freeze"
        (trial.root / "protocol.json").write_text(json.dumps(value))
    store = AsyncMock(); monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    store.assert_not_called()
    assert not (trial.root / "worker-stop.json").exists()


def test_only_new_runner_opts_in_and_parent_does_not_enable_stage_exclusions(monkeypatch, tmp_path):
    async def actual_case(*args, **kwargs):
        async def controlled_model():
            async with overhead("admission"): await asyncio.sleep(0)
            assert_model_active()
            return "controlled-coro-with-no-model"
        return await await_model(controlled_model(), timeout_s=.1)
    monkeypatch.setattr(business.engine, "execute_case", actual_case)
    assert asyncio.run(module.DeadlineRecheck(tmp_path / "new").run_case({}, None, None, ensure_healthy=lambda: None))
    with pytest.raises(ModelDeadlineContextError):
        asyncio.run(business.BusinessRegression(tmp_path / "old").run_case({}, None, None, ensure_healthy=lambda: None))


def test_factory_selects_only_new_transport_and_keeps_old_parent_default(tmp_path, monkeypatch):
    old, new = [], []
    monkeypatch.setattr(business.engine, "TaskHTTPTransport", lambda *a, **k: old.append((a, k)) or "inline")
    monkeypatch.setattr(task_deadline_transport, "DeadlineTaskHTTPTransport", lambda *a, **k: new.append((a, k)) or "clock")
    assert business.BusinessRegression(tmp_path / "old").make_transport("controlled") == "inline"
    assert module.DeadlineRecheck(tmp_path / "new").make_transport("controlled") == "clock"
    assert len(old) == len(new) == 1
    with pytest.raises(TypeError): module.DeadlineRecheck(tmp_path / "bad", transport="arbitrary")
    with pytest.raises(base.TrialError): module.DeadlineRecheck(business.ROOT)


def test_report_retains_v4_grade_and_is_not_an_ablation_uplift(tmp_path, monkeypatch):
    value = {"grading_version": "task-business-v4", "business_counts": {"unknown": 12}}
    monkeypatch.setattr(business.BusinessRegression, "report", lambda self: deepcopy(value))
    report = module.DeadlineRecheck(tmp_path / "new").report()
    assert report["grading_version"] == "task-business-v4" and report["business_counts"] == value["business_counts"]
    assert report["stage_clock_policy"] == module.CLOCK_POLICY
    assert report["is_ablation_uplift_report"] is False and report["historical_scores_rewritten"] is False


@pytest.mark.parametrize("inner", [
    "client.messages.create(model='fixed', max_tokens=256, temperature=.1, messages=[{'role':'user','content':'同一问题'}])",
    "client.messages.create(model='fixed', max_tokens=1100, temperature=0.0, tools=[{'name':'read_demo_order','input_schema':{'type':'object'}}], tool_choice={'type':'tool','name':'read_demo_order'}, messages=[{'role':'user','content':'DEMO-1004'}])",
])
def test_wrapper_ast_preserves_text_and_native_tool_payload_parameters(inner):
    old = f"import asyncio\nasync def call(client):\n return await asyncio.wait_for({inner}, timeout=8)\n"
    new = f"import asyncio\nfrom core.model_deadline import await_model\nasync def call(client):\n return await await_model({inner}, timeout_s=8)\n"
    assert module._normalized(old) == module._normalized(new)
    assert module._normalized(old) != module._normalized(new.replace("model='fixed'", "model='changed'"))
    assert module._normalized(old) != module._normalized(new.replace("timeout_s=8", "timeout_s=80"))


@pytest.mark.parametrize("bad_import", ["from core.model_deadline import await_model as wrong_name\n",
                                       "def unrelated():\n from core.model_deadline import await_model\n"])
def test_wrong_import_alias_or_scope_is_not_erased(bad_import):
    old = "import asyncio\nasync def call(client):\n return await asyncio.wait_for(client.messages.create(model='fixed'),timeout=8)\n"
    new = f"import asyncio\n{bad_import}async def call(client):\n return await await_model(client.messages.create(model='fixed'),timeout_s=8)\n"
    assert module._normalized(old) != module._normalized(new)


def clock_sources(tmp_path, monkeypatch):
    root = tmp_path / "controlled-backend"
    allowed = {"core/intent_recognizer.py", "agents/agent_orchestrator.py", "tooling/tool_manager.py"}
    before = {name: "a" * 64 for name in allowed} | {"core/unchanged.py": "b" * 64}
    after = {name: "c" * 64 for name in allowed} | {"core/unchanged.py": "b" * 64, "core/model_deadline.py": "d" * 64}
    old_sources = {}
    for name in allowed:
        count = 1 if name == "core/intent_recognizer.py" else 2
        old = "import asyncio\n" + "".join(f"async def run_{i}(client):\n return await asyncio.wait_for(client.messages.create(model='fixed',temperature=.1),timeout=8)\n" for i in range(count))
        new = "import asyncio\nfrom core.model_deadline import await_model\n" + "".join(f"async def run_{i}(client):\n return await await_model(client.messages.create(model='fixed',temperature=.1),timeout_s=8)\n" for i in range(count))
        old_sources[name] = old
        (root / name).parent.mkdir(parents=True, exist_ok=True); (root / name).write_text(new)
    monkeypatch.setattr(bench, "BACKEND_ROOT", root)
    monkeypatch.setattr(business, "_artifacts", lambda: after)
    def git(args, **kwargs):
        assert args[:2] == ["git", "show"] and args[2].startswith("a" * 40 + ":backend/")
        return SimpleNamespace(stdout=old_sources[args[2].split(":backend/", 1)[1]].encode())
    monkeypatch.setattr(module.subprocess, "run", git)
    return {"git_head": "a" * 40, "artifacts": before}, after, root, allowed


def test_delta_accepts_three_wrapper_only_changes_and_exact_new_core(tmp_path, monkeypatch):
    old, _, _, allowed = clock_sources(tmp_path, monkeypatch)
    proof = module.business_clock_delta(old)
    assert proof["clock_wrapper_only_changes"] == sorted(allowed)
    assert proof["prompts_and_domain_logic_ast_unchanged"] is True


@pytest.mark.parametrize("damage", ["extra_file", "removed_file", "other_source", "model", "tool_args", "missing_import", "extra_wrapper"])
def test_delta_rejects_non_clock_business_change(tmp_path, monkeypatch, damage):
    old, current, root, _ = clock_sources(tmp_path, monkeypatch)
    if damage == "extra_file": current["core/feature.py"] = "e" * 64
    elif damage == "removed_file": del current["core/unchanged.py"]
    elif damage == "other_source": current["core/unchanged.py"] = "e" * 64
    else:
        path = root / "core/intent_recognizer.py"
        text = path.read_text()
        if damage == "model": text = text.replace("model='fixed'", "model='different'")
        elif damage == "tool_args": text = text.replace("temperature=.1", "temperature=.9")
        elif damage == "missing_import": text = text.replace("from core.model_deadline import await_model\n", "")
        else: text += "\nasync def extra(client):\n return await await_model(client.messages.create(model='fixed'), timeout_s=8)\n"
        path.write_text(text)
    with pytest.raises(base.TrialError, match="task_clock_(business_file_set_changed|non_clock_business_change|payload_or_logic_changed)"):
        module.business_clock_delta(old)


def archived_fixture(tmp_path, monkeypatch):
    root = tmp_path / "old-v5"; root.mkdir(mode=0o700)
    spec = {**deepcopy(module.V5), "root": root}
    p = {"git_head": spec["commit"], "root_binding_sha256": base._root_hash(root),
         "artifacts": {"core/archived.py": "a" * 64}, "plan": business.fixed_plan()}
    base._write(root, "protocol.json", p)
    genesis = "d" * 64
    base._write(root, "authorization.json", {"protocol_sha256": bench.sha256(root / "protocol.json"),
        "max_requests": 312, "plan": p["plan"], "root_binding_sha256": base._root_hash(root), "initial_chain_hash": genesis})
    ledger = root / "ledger"; ledger.mkdir(mode=0o700)
    base._write(ledger, "journal.jsonl", {"controlled": True})
    operations, cases = {}, {}
    for number in range(1, 26):
        name = f"physical-{number:03d}.json"; base._write(root, name, {"controlled": number})
        operations[str(number)] = {"attempt": number, "output_sha256": bench.sha256(root / name)}
    for key in spec["completed"]:
        name = f"case-{key}.json"; base._write(root, name, {"controlled": key})
        cases[key] = {"artifact_sha256": bench.sha256(root / name)}
    base._write(root, "reviewed-paused-report.json", {"business_counts": spec["business_counts"]})
    for name, key in (("protocol.json", "protocol"), ("authorization.json", "authorization"), ("ledger/journal.jsonl", "journal")):
        spec[key] = bench.sha256(root / name)
    spec["tree"] = module._tree(root)
    state = {"attempts": 25, "chain_hash": spec["chain"], "pending": [], "paused": True,
             "active_case": "e06", "completed_cases": spec["completed"]}
    fake = SimpleNamespace(status=lambda: state, results=lambda: {"operations": operations, "cases": cases}, initial_chain_hash=genesis)
    def readonly(*args, **kwargs):
        assert not kwargs.get("initialize")
        return fake
    monkeypatch.setattr(business, "BusinessTaskBudget", readonly)
    sources = []; monkeypatch.setattr(module.history, "_archived_sources", lambda p, key: sources.append((p, key)))
    return root, spec, state, sources


def test_old_v5_archive_git_and_ledger_are_checked_without_writes_or_regrade(tmp_path, monkeypatch):
    root, spec, _, sources = archived_fixture(tmp_path, monkeypatch)
    before = tree(root); p, proof = ORIGINAL_ARCHIVED(spec)
    assert proof["reserved_requests"] == proof["sealed_requests"] == 25
    assert proof["original_business_counts"] == spec["business_counts"] and proof["old_scores_recomputed"] is False
    assert sources == [(p, "artifacts")] and tree(root) == before
    (root / "physical-025.json").write_text('{"changed":"old cancellation"}')
    with pytest.raises(base.TrialError, match="old_tree_changed"): ORIGINAL_ARCHIVED(spec)


def test_unknown_old_attempt_cannot_be_removed_or_refunded(tmp_path, monkeypatch):
    root, spec, state, _ = archived_fixture(tmp_path, monkeypatch)
    state["attempts"] = 24; before = tree(root)
    with pytest.raises(base.TrialError, match="old_ledger_changed"): ORIGINAL_ARCHIVED(spec)
    assert tree(root) == before


def test_prior_proof_chains_both_paused_batches_without_new_allowance(monkeypatch):
    older = business.prior_proof()
    p4 = {"prior_proof": older, "artifacts": {"core/fixed.py": "a" * 64}}
    p5 = {"prior_proof": {"tree_sha256": module.V4["tree"], "older_proof_sha256": canonical_payload_hash(older)},
          "artifacts": deepcopy(p4["artifacts"])}
    seen = []
    def archived(spec):
        seen.append(spec)
        return (p4, {"reserved_requests": 12}) if spec is module.V4 else (p5, {"reserved_requests": 25})
    monkeypatch.setattr(module, "_archived", archived)
    monkeypatch.setattr(module, "business_clock_delta", lambda p: {"clock_wrapper_only_changes": ["controlled.py"]})
    result = ORIGINAL_PRIOR()
    assert seen == [module.V4, module.V5]
    assert result["v4"]["reserved_requests"] == 12 and result["v5"]["reserved_requests"] == 25
    assert result["old_allowance_reused"] is False and result["new_authorized_requests"] == 0
    p5["prior_proof"]["tree_sha256"] = "wrong"
    with pytest.raises(base.TrialError, match="old_history_changed"): ORIGINAL_PRIOR()


def test_clock_runner_and_boundary_tests_are_frozen_in_new_scope(monkeypatch):
    monkeypatch.setattr(bench, "sha256", lambda path: "e" * 64)
    assert {"evaluation/task_deadline_recheck.py", "evaluation/task_deadline_transport.py", "core/model_deadline.py",
            "tests/test_model_deadline.py", "tests/test_task_deadline_transport.py", "tests/test_task_deadline_recheck.py"} <= set(ORIGINAL_ARTIFACTS())
