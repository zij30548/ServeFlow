"""Synthetic offline fixtures for source admission, freezing and fact metrics.

No real .env, credential, database, index or model transport is read/constructed.
Temporary datasets are independently authored here, never old question rewrites.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from demo_business.store import load_fixture
from evaluation import coverage_expansion as coverage


@pytest.fixture(autouse=True)
def no_services_or_environment(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("coverage preparation is entirely offline")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    import chromadb
    for name in ("HttpClient", "PersistentClient", "EphemeralClient"):
        monkeypatch.setattr(chromadb, name, denied)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == ".env" or path.suffix in {".pem", ".key"}:
            denied()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-never-be-read-or-sent")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bindings(root, paths):
    return [{"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()} for path in paths]


def rebind(root):
    for name, dependencies in (("cases", coverage.FILES[:2]), ("conversations", coverage.FILES[:3])):
        path = root / coverage.DATA / f"{name}.json"
        value = json.loads(path.read_text())
        value["source_bindings"] = bindings(root, dependencies)
        write(path, value)


def mutate(root, name, fn, *, hashes=True):
    path = root / coverage.DATA / f"{name}.json"
    value = json.loads(path.read_text())
    fn(value)
    write(path, value)
    if hashes:
        rebind(root)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "backend"
    root.mkdir()
    for folder in coverage.SOURCE_FOLDERS:
        (root / folder).mkdir()
    for relative in ("core/provider_options.py", "evaluation/task_transport.py", "evaluation/phase_deadline_transport.py"):
        target = root / relative
        target.write_bytes((coverage.bench.BACKEND_ROOT / relative).read_bytes())
    provenance = "Independently controlled source-validation fixtures, no live observations"
    corpus = {"schema_version": "serveflow-coverage-corpus-v1", "kind": coverage.KIND,
        "provenance": provenance, "documents": []}
    for prefix, split, count in (("D", "development", 8), ("V", "validation", 16)):
        for i in range(1, count + 1):
            identity = f"CX-{prefix}{i:02d}"
            first, last = f"{identity}专用事实甲：仅适用于本合成来源的三项条件", f"{identity}专用事实乙：后段例外必须同时核对，未知结果不能补写"
            middle = "。".join(f"{identity}范围背景{j}：这项合成说明详细描述范围、核验环节与记录边界，不能推广为其他来源承诺" for j in range(11)) if i > count - 2 else ""
            content = "。".join(part for part in (first, middle, last) if part)
            corpus["documents"].append({"document_id": identity, "title": f"独立合成资料{identity}",
                "split": split, "family_id": f"source-{identity}", "content": content,
                "facts": [{"fact_id": identity + "-F01", "quote": first}, {"fact_id": identity + "-F02", "quote": last}]})
    fixture = {"schema_version": "serveflow-demo-business-v1", "kind": "synthetic-local-demo",
        "as_of": "2026-10-25T00:00:00Z", "customers": deepcopy(load_fixture()["customers"]),
        **{table: [] for table in coverage.KEYS}}
    order_splits = []
    for index in [*range(2001, 2013), *range(2101, 2121)]:
        identity, amount = f"DEMO-{index}", (index - 1900) * 37
        split = "development" if index < 2100 else "validation"
        order_splits.append({"order_id": identity, "split": split, "family_id": f"source-{identity}"})
        fixture["orders"].append({"order_id": identity, "customer_id": "demo-customer-a", "order_type": "physical",
            "currency": "CNY", "total_amount_minor": amount, "payment_status": "paid", "record_updated_at": fixture["as_of"]})
        fixture["items"].append({"item_id": f"I-{index}", "order_id": identity, "name": f"合成零件{index}",
            "item_type": "device", "quantity": 1, "unit_amount_minor": amount})
        fixture["payments"].append({"payment_id": f"P-{index}", "order_id": identity, "amount_minor": amount,
            "status": "paid", "channel": "demo-wallet", "record_updated_at": fixture["as_of"]})
    records = {"schema_version": "serveflow-coverage-records-v1", "kind": coverage.KIND,
        "provenance": provenance, "order_splits": order_splits, "fixture": fixture}
    cases = {"schema_version": "serveflow-coverage-cases-v1", "kind": coverage.KIND,
        "provenance": provenance, "state": "offline-authored-no-execution", "source_bindings": [], "tasks": [], "retrieval": []}
    for prefix, split, tasks, rags, doc_count, start in (("D", "development", 12, 16, 8, 2001), ("V", "validation", 28, 32, 16, 2101)):
        for suite, count in (("tasks", tasks), ("retrieval", rags)):
            for i in range(1, count + 1):
                task = suite == "tasks"
                doc = next(d for d in corpus["documents"] if d["document_id"] == f"CX-{prefix}{(i - 1) % doc_count + 1:02d}")
                kind = "answerable" if i <= count // 2 else "insufficient" if i <= count * 3 // 4 else "out_of_scope"
                refs = [{"source": "knowledge", "document_id": doc["document_id"], **fact} for fact in doc["facts"]] if task or kind != "out_of_scope" else []
                order_id = f"DEMO-{start + (i - 1) % (12 if prefix == 'D' else 20)}"
                if task:
                    refs.append({"source": "records", "table": "orders", "key": {"order_id": order_id}, "fields": {"currency": "CNY"}})
                expected = {"relevant_document_ids": [doc["document_id"]] if refs else [],
                    "required_fact_ids": [f["fact_id"] for f in doc["facts"]] if refs else [],
                    "required_facts": ["直接回答来源事实"] if refs else [],
                    "missing_facts": [] if task or kind == "answerable" else ["来源未提供问题要求的具体结果"],
                    "forbidden_claims": ["不得补造未知结果"], "rationale": "受控来源与分母校验"}
                if task:
                    expected.update(required_tools=["read_demo_order"], expected_status="success", tool_task=True, boundary=False)
                else:
                    expected["kind"] = kind
                identity = f"{'CT' if task else 'CR'}-{prefix}{i:02d}"
                case = {"id": identity, "split": split, "family_id": "case-" + identity,
                    "input": {"message": f"{identity}请核对独立来源的具体条件" + (f"与订单{order_id}记录" if task else ""), "history": []},
                    "expected": expected, "references": refs, "tags": ["controlled"], "provenance": provenance, "review_status": "reviewed"}
                if task:
                    case["input"]["customer_id"] = "demo-customer-a"
                cases[suite].append(case)
    conversations = {"schema_version": "serveflow-coverage-conversations-v1", "kind": coverage.KIND,
        "provenance": provenance, "state": "offline-authored-no-execution", "source_bindings": [], "conversations": []}
    for prefix, split, count, order_id in (("D", "development", 2, "DEMO-2001"), ("V", "validation", 6, "DEMO-2101")):
        for i in range(1, count + 1):
            identity = f"CC-{prefix}{i:02d}"
            conversations["conversations"].append({"id": identity, "split": split, "family_id": "group-" + identity,
                "customer_id": "demo-customer-a", "tags": ["controlled-memory"], "provenance": provenance, "review_status": "reviewed",
                "turns": [{"id": f"{identity}-T{j:02d}", "message": f"{identity}第{j}轮请核对{order_id}",
                    "expected": {"required_facts": ["原始用户目标"], "missing_facts": [], "forbidden_claims": ["不得借助手正文授权"], "required_tools": ["read_demo_order"], "expected_status": "success"},
                    "references": [{"source": "record_collection", "table": "shipments", "filters": {"order_id": order_id}, "expected_count": 0}]} for j in range(1, 6)]})
    for name, value in (("corpus", corpus), ("records", records), ("cases", cases), ("conversations", conversations)):
        write(root / coverage.DATA / f"{name}.json", value)
    rebind(root)
    return root


def observation(rows=None, *, case_id="CR-D07", status="success"):
    return {"schema_version": "serveflow-coverage-retrieval-observations-v1", "evidence_kind": "controlled-offline",
        "arm": "enhanced", "results": [{"case_id": case_id, "status": status, "rows": rows or []}]}


def source_rows(root, explicit="CX-D07"):
    docs = coverage.load_validated(root)[0]["documents"]
    return [{k: v for k, v in row.items() if k != "explicit_document_id"} | {"score": .8}
        for row in coverage.source_chunks(docs).values() if row["explicit_document_id"] == explicit]


def test_full_controlled_counts_and_model_projection_exclude_gold_and_customer(source):
    summary = coverage.validate(source)
    assert (summary["documents"], summary["orders"], summary["tasks"], summary["retrieval"], summary["conversation_turns"]) == (24, 32, 40, 48, 40)
    assert summary["quality_pass_claim"] is False and summary["live_sender_ready"] is False
    case = coverage.load_validated(source)[2]["tasks"][0]
    assert set(coverage.model_input(case)) == {"message", "history"}
    assert coverage.trusted_input(case) == {"customer_id": "demo-customer-a"}
    result = coverage.model_input(case)
    result["history"].append({"role": "assistant", "content": "untrusted"})
    assert case["input"]["history"] == []


@pytest.mark.parametrize("name,fn", [
    ("corpus", lambda d: d.update(extra=True)),
    ("corpus", lambda d: d["documents"].append(d["documents"][0])),
    ("corpus", lambda d: d["documents"][0]["facts"][0].update(quote="not in source")),
    ("corpus", lambda d: d["documents"][0].update(split="validation")),
    ("records", lambda d: d["fixture"]["orders"][0].update(total_amount_minor=1)),
    ("records", lambda d: d["fixture"]["items"][0].update(unit_amount_minor=True)),
    ("records", lambda d: d["order_splits"][0].update(split="validation")),
    ("cases", lambda d: d["tasks"][0]["input"].update(expected={"secret_gold": True})),
    ("cases", lambda d: d["tasks"][0]["expected"].update(required_fact_ids=[])),
    ("cases", lambda d: d["retrieval"][0]["references"][0].update(quote="fabricated")),
    ("cases", lambda d: d["retrieval"][0]["input"]["history"].append({"role": "system", "content": "gold"})),
    ("cases", lambda d: d["retrieval"].pop()),
    ("conversations", lambda d: d["conversations"][0]["turns"].pop()),
    ("conversations", lambda d: d["conversations"][0]["turns"][0]["references"][0].update(expected_count=1)),
    ("conversations", lambda d: d["conversations"][0]["turns"][0]["references"][0].update(expected_count=False)),
])
def test_strict_fields_limits_production_records_and_source_quotes(source, name, fn):
    mutate(source, name, fn)
    with pytest.raises(coverage.bench.BenchmarkError):
        coverage.validate(source)


def test_cross_split_reference_and_duplicate_fact_are_rejected(source):
    def change(d):
        dev = d["tasks"][0]
        dev["references"][-1]["key"]["order_id"] = "DEMO-2101"
    mutate(source, "cases", change)
    with pytest.raises(coverage.bench.BenchmarkError, match="record_reference"):
        coverage.validate(source)


def test_missing_conversation_file_is_not_a_partial_success(source):
    (source / coverage.FILES[3]).unlink()
    with pytest.raises(coverage.bench.BenchmarkError):
        coverage.validate(source)


def test_stale_source_hash_is_rejected_even_if_text_and_quotes_remain_consistent(source):
    mutate(source, "corpus", lambda d: d["documents"][0].update(title="changed"), hashes=False)
    with pytest.raises(coverage.bench.BenchmarkError, match="source_hash"):
        coverage.validate(source)


def test_document_hit_cannot_replace_fact_quote_coverage_on_another_chunk(source):
    rows = source_rows(source)
    assert len(rows) >= 2
    report = coverage.score_retrieval(observation(rows[:1]), root=source)
    case = next(c for c in report["cases"] if c["case_id"] == "CR-D07")
    assert case["document_hit_at_3"] is True
    assert case["required_fact_recall_at_3"] == .5 and case["all_required_facts_at_3"] is False
    complete = coverage.score_retrieval(observation([rows[0], rows[-1]]), root=source)
    case = next(c for c in complete["cases"] if c["case_id"] == "CR-D07")
    assert case["required_fact_recall_at_3"] == 1 and case["all_required_facts_at_3"] is True
    assert complete["planned"] == 48 and complete["live_model_quality"] is None


def test_same_title_from_another_document_never_counts_as_a_document_or_fact_hit(source):
    def collision(value):
        docs = value["documents"]
        docs[7]["title"] = docs[6]["title"]
    mutate(source, "corpus", collision)
    rows = source_rows(source, "CX-D08")
    report = coverage.score_retrieval(observation(rows[:1]), root=source)
    case = next(c for c in report["cases"] if c["case_id"] == "CR-D07")
    assert case["document_hit_at_3"] is False
    assert case["required_fact_recall_at_3"] == 0


@pytest.mark.parametrize("field,value", [("document_id", "CX-D07"), ("title", "same-title-trick"), ("content", "fabricated quote"), ("version", "0" * 64), ("chunk", True), ("score", float("nan"))])
def test_title_chunk_version_content_identity_and_nonfinite_cannot_be_forged(source, field, value):
    rows = source_rows(source)
    rows[0][field] = value
    with pytest.raises(coverage.bench.BenchmarkError):
        coverage.score_retrieval(observation(rows[:1]), root=source)


def test_failed_empty_never_counts_as_healthy_out_of_scope_rejection(source):
    failed = coverage.score_retrieval(observation(case_id="CR-D16", status="failed"), root=source)
    healthy = coverage.score_retrieval(observation(case_id="CR-D16"), root=source)
    assert failed["groups"]["development/out_of_scope"]["healthy_empty_rejections"] == 0
    assert healthy["groups"]["development/out_of_scope"]["healthy_empty_rejections"] == 1
    assert failed["groups"]["development/out_of_scope"]["planned"] == 4
    assert failed["groups"]["development/answerable"]["planned"] == 8


def test_duplicate_rows_unknown_cases_and_live_claims_are_not_admitted(source):
    rows = source_rows(source)
    for value in (observation([rows[0], rows[0]]), observation(case_id="unknown"), observation(rows[:1], status="failed")):
        with pytest.raises(coverage.bench.BenchmarkError):
            coverage.score_retrieval(value, root=source)
    value = observation()
    value["evidence_kind"] = "live-model"
    with pytest.raises(coverage.bench.BenchmarkError):
        coverage.score_retrieval(value, root=source)


def test_public_options_are_ast_read_without_loading_sdk_or_dotenv(source):
    result = coverage.public_parameters(source)
    assert result["model"] == "qwen3.7-flash-2026-07-15"
    assert result["phase_parameters"]["intent"] == (256, .1)
    assert "judge" not in result["phase_parameters"]
    assert result["live_environment_read"] is False and result["payloads_frozen"] is False


def test_production_chunk_method_executes_without_importing_chroma_or_reading_dotenv(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"chromadb", "tooling.knowledge_base", "dotenv"}:
            raise AssertionError("pure chunk derivation cannot import service configuration")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    text = "。".join(f"来源句{i}与完整适用条件" for i in range(70))
    chunks = coverage._chunks(text)
    assert len(chunks) > 1 and all(len(chunk) <= 500 for chunk in chunks)
    assert "来源句69与完整适用条件" in chunks[-1]


def test_prepare_is_private_exclusive_and_verify_closes_over_all_sources(source, tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "old_source_proof", lambda root: {"controlled_ancestry": "not-live"})
    root = tmp_path / "private" / "coverage-expansion-v1"
    value = coverage.prepare(root, source_root=source)
    assert value["authorized_model_requests"] == value["executed_model_requests"] == value["database_connections"] == 0
    assert (root.stat().st_mode & 0o777) == 0o700
    assert ((root / "protocol.json").stat().st_mode & 0o777) == 0o600
    assert coverage.verify(root, source_root=source)["verified"]
    with pytest.raises(coverage.bench.BenchmarkError, match="preparation_exists"):
        coverage.prepare(root, source_root=source)
    (source / "core" / "extra.py").write_text("x = 1\n")
    with pytest.raises(coverage.bench.BenchmarkError, match="preparation_changed"):
        coverage.verify(root, source_root=source)


def test_symlink_and_duplicate_json_keys_are_rejected(source):
    path = source / coverage.FILES[3]
    saved = path.read_bytes()
    path.unlink()
    target = source / "saved.json"
    target.write_bytes(saved)
    path.symlink_to(target)
    with pytest.raises(coverage.bench.BenchmarkError, match="symlink"):
        coverage.validate(source)
    path.unlink()
    path.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(coverage.bench.BenchmarkError, match="duplicate_json_key"):
        coverage.validate(source)


def test_pinned_old_corpus_fixture_and_all_case_hashes_remain_unchanged():
    proof = coverage.old_source_proof()
    assert proof["baseline_commit"] == coverage.BASELINE
    assert "knowledge/serveflow-demo-v2.json" in proof["artifacts"]
    assert "demo_business/fixtures.json" in proof["artifacts"]
    assert len(proof["artifacts"]) >= 10 and proof["historical_senders_reopened"] is False


def test_pinned_old_hash_mismatch_is_rejected_without_touching_the_old_file(monkeypatch):
    original = coverage.bench.sha256
    def changed(path):
        return "0" * 64 if path.name == "serveflow-demo-v2.json" else original(path)
    monkeypatch.setattr(coverage.bench, "sha256", changed)
    with pytest.raises(coverage.bench.BenchmarkError, match="old_source_changed"):
        coverage.old_source_proof()


def test_top3_feasibility_requires_fact_chunks_not_matching_document_titles():
    facts = {f"f{i}": (f"doc{i}", f"quote{i}") for i in range(4)}
    chunks = {i: {"explicit_document_id": f"doc{i}", "content": f"quote{i}"} for i in range(4)}
    assert coverage._top3_possible(["f0", "f1", "f2"], facts, chunks)
    assert not coverage._top3_possible(list(facts), facts, chunks)


@pytest.mark.parametrize("command", ["authorize", "run", "resume"])
def test_cli_has_no_authorization_or_sender(command):
    with pytest.raises(SystemExit):
        coverage.main([command])
