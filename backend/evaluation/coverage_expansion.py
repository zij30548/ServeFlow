"""Offline source validation, immutable preparation and fact-level RAG checks.

No sender, authorization, SDK, index, database or credential access exists here.
Controlled observations establish metric behavior, never live answer quality.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess

from demo_business.store import CUSTOMER_IDS, TABLE_COLUMNS, DemoStoreError, validate_fixture
from evaluation import benchmark as bench


DATA = "evaluation/benchmark/coverage-expansion-v1"
FILES = tuple(f"{DATA}/{name}.json" for name in ("corpus", "records", "cases", "conversations"))
ROOT = bench.BACKEND_ROOT / "data/benchmarks/coverage-expansion-v1"
BASELINE = "59180f64d13be7a89d78518e3d7bea2cb9f7020a"
KIND = "author-synthetic-evaluation-only"
SCHEMA = "serveflow-coverage-expansion-preparation-v1"
SOURCE_FOLDERS = frozenset({"agents", "api", "config", "core", "demo_business", "evaluation", "knowledge", "memory", "monitor", "skills", "tests", "tooling"})
SOURCE_SUFFIXES = frozenset({".py", ".json", ".md", ".txt"})
KEYS = {table: columns[0] for table, columns in TABLE_COLUMNS.items() if table != "customers"}
CASE_KEYS = {"id", "split", "family_id", "input", "expected", "references", "tags", "provenance", "review_status"}
COMMON_EXPECTED = {"relevant_document_ids", "required_fact_ids", "required_facts", "missing_facts", "forbidden_claims", "rationale"}
TASK_EXPECTED = COMMON_EXPECTED | {"required_tools", "expected_status", "tool_task", "boundary"}
TURN_EXPECTED = {"required_facts", "missing_facts", "forbidden_claims", "expected_status", "required_tools"}
SPLITS = {"development", "validation"}


def _fail(code):
    raise bench.BenchmarkError("coverage_" + code)


def _exact(value, fields):
    if type(value) is not dict or set(value) != fields:
        _fail("fields")


def _text(value, maximum=2000):
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(ord(c) < 32 and c not in "\n\t" or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        _fail("text")


def _list(value, maximum=100, *, nonempty=False):
    if type(value) is not list or len(value) > maximum or nonempty and not value:
        _fail("list")


def _strings(value, maximum=100, *, nonempty=False):
    _list(value, maximum, nonempty=nonempty)
    for item in value:
        _text(item)
    if len(value) != len(set(value)):
        _fail("duplicate_string")


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        _fail("path")
    path = root / relative
    if root.is_symlink() or any((root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)):
        _fail("symlink")
    if not path.is_file() or path.stat().st_size > bench.LIMIT:
        _fail("file")
    return path


def _read(root, relative):
    return bench.read_json(_safe(root, relative))


def _bindings(value, paths, root):
    _list(value, len(paths))
    if len(value) != len(paths):
        _fail("bindings")
    found = {}
    for row in value:
        _exact(row, {"path", "sha256"})
        if row["path"] not in paths or row["path"] in found:
            _fail("bindings")
        digest = bench.sha256(_safe(root, row["path"]))
        if row["sha256"] != digest:
            _fail("source_hash")
        found[row["path"]] = digest
    return found


def _header(value, schema, fields, *, state=False):
    _exact(value, {"schema_version", "kind", "provenance"} | fields | ({"state"} if state else set()))
    if value["schema_version"] != schema or value["kind"] != KIND:
        _fail("schema")
    _text(value["provenance"])
    if state and value["state"] != "offline-authored-no-execution":
        _fail("state")


@lru_cache(maxsize=1)
def _production_chunker(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KnowledgeBase")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_chunk_text")
    namespace = {"List": list}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "tooling/knowledge_base.py", "exec"), namespace)
    return namespace["_chunk_text"]


def _chunks(content):
    # Execute the actual production method without importing Chroma, whose
    # import-time Settings can read a real .env. Cache by current source bytes,
    # never by path alone; source inventory still rejects any runtime drift.
    source = _safe(bench.BACKEND_ROOT, "tooling/knowledge_base.py").read_text()
    return _production_chunker(source)(None, content, chunk_size=500)


def source_chunks(documents):
    """Exact metadata/content produced by KnowledgeBase.add_documents/search."""
    result = {}
    for doc in documents:
        identity = json.dumps(["explicit", doc["document_id"]], ensure_ascii=True)
        document_id = hashlib.sha256(identity.encode()).hexdigest()
        version = hashlib.sha256(doc["content"].encode()).hexdigest()
        for index, content in enumerate(_chunks(doc["content"])):
            result[(document_id, index, version)] = {"document_id": document_id,
                "explicit_document_id": doc["document_id"], "title": doc["title"],
                "version": version, "chunk": index, "content": content}
    return result


def model_input(case):
    """Gold, source metadata and trusted customer identity never enter text."""
    value = case["input"]
    return {"message": value["message"], "history": json.loads(json.dumps(value["history"]))}


def trusted_input(case):
    return {"customer_id": case["input"]["customer_id"]} if "customer_id" in case["input"] else {}


def _input(value, task):
    _exact(value, {"message", "history"} | ({"customer_id"} if task else set()))
    _text(value["message"])
    _list(value["history"], 12)
    for row in value["history"]:
        _exact(row, {"role", "content"})
        if row["role"] not in {"user", "assistant"}:
            _fail("history_role")
        _text(row["content"], 4000)
    if task and (type(value["customer_id"]) is not str or value["customer_id"] not in CUSTOMER_IDS):
        _fail("customer")


def _references(refs, split, docs, facts, fixture, order_splits):
    _list(refs, 24)
    seen, doc_ids, fact_ids = set(), set(), set()
    for ref in refs:
        if type(ref) is not dict or type(ref.get("source")) is not str:
            _fail("reference")
        token = _hash(ref)
        if token in seen:
            _fail("duplicate_reference")
        seen.add(token)
        if ref["source"] == "knowledge":
            _exact(ref, {"source", "document_id", "fact_id", "quote"})
            for key in ("document_id", "fact_id", "quote"):
                _text(ref[key])
            if (ref["document_id"] not in docs or ref["fact_id"] not in facts
                    or facts[ref["fact_id"]] != (ref["document_id"], ref["quote"])
                    or docs[ref["document_id"]]["split"] != split):
                _fail("knowledge_reference")
            doc_ids.add(ref["document_id"]); fact_ids.add(ref["fact_id"])
        elif ref["source"] in {"records", "record_collection"}:
            collection = ref["source"] == "record_collection"
            _exact(ref, {"source", "table", "filters", "expected_count"} if collection
                else {"source", "table", "key", "fields"})
            table = ref["table"]
            if type(table) is not str or table not in KEYS:
                _fail("record_table")
            filters = ref["filters"] if collection else ref["key"]
            if (type(filters) is not dict or not filters or not set(filters) <= set(TABLE_COLUMNS[table])
                    or not collection and set(filters) != {KEYS[table]}):
                _fail("record_filter")
            if any(type(v) not in {str, int, bool, type(None)} for v in filters.values()):
                _fail("record_filter")
            rows = [row for row in fixture[table]
                    if all(type(row[k]) is type(v) and row[k] == v for k, v in filters.items())]
            if collection:
                if ("order_id" not in filters or filters["order_id"] not in order_splits
                        or order_splits[filters["order_id"]]["split"] != split
                        or type(ref["expected_count"]) is not int or not 0 <= ref["expected_count"] <= 100
                        or len(rows) != ref["expected_count"]):
                    _fail("record_count")
            else:
                fields = ref["fields"]
                if (len(rows) != 1 or type(fields) is not dict or not fields
                        or not set(fields) <= set(TABLE_COLUMNS[table])
                        or any(type(v) is not type(rows[0][k]) or rows[0][k] != v for k, v in fields.items())
                        or order_splits[rows[0]["order_id"]]["split"] != split):
                    _fail("record_reference")
        else:
            _fail("reference_source")
    return doc_ids, fact_ids


def _expected(value, task, doc_ids, fact_ids):
    _exact(value, TASK_EXPECTED if task else COMMON_EXPECTED | {"kind"})
    for field in ("required_facts", "missing_facts", "forbidden_claims", "relevant_document_ids", "required_fact_ids"):
        _strings(value[field], 32)
    _text(value["rationale"])
    if set(value["relevant_document_ids"]) != doc_ids or set(value["required_fact_ids"]) != fact_ids:
        _fail("gold_reference_binding")
    if task:
        _strings(value["required_tools"], 12)
        if (not set(value["required_tools"]) <= bench.TOOLS or type(value["tool_task"]) is not bool
                or value["tool_task"] != bool(value["required_tools"]) or type(value["boundary"]) is not bool
                or type(value["expected_status"]) is not str
                or value["expected_status"] not in {"success", "partial", "not-prescribed"}):
            _fail("task_expectation")
    else:
        kind = value["kind"]
        if (type(kind) is not str or kind not in {"answerable", "insufficient", "out_of_scope"}
                or kind == "out_of_scope" and (doc_ids or fact_ids or value["required_facts"] or not value["missing_facts"])
                or kind != "out_of_scope" and (not doc_ids or not fact_ids)
                or kind == "answerable" and (not value["required_facts"] or value["missing_facts"])
                or kind == "insufficient" and not value["missing_facts"]):
            _fail("retrieval_expectation")


def _top3_possible(required, facts, chunks):
    if not required:
        return True
    target = (1 << len(required)) - 1
    masks = {sum(1 << i for i, fact in enumerate(required)
        if row["explicit_document_id"] == facts[fact][0] and facts[fact][1] in row["content"])
        for row in chunks.values()}
    masks.discard(0)
    # Exact small set-cover check, not a title-based approximation.
    for count in range(1, min(3, len(masks)) + 1):
        for selected in combinations(masks, count):
            combined = 0
            for mask in selected:
                combined |= mask
            if combined == target:
                return True
    return False


def load_validated(root=bench.BACKEND_ROOT):
    corpus, records, cases, conversations = [_read(root, path) for path in FILES]
    _header(corpus, "serveflow-coverage-corpus-v1", {"documents"})
    _header(records, "serveflow-coverage-records-v1", {"order_splits", "fixture"})
    _header(cases, "serveflow-coverage-cases-v1", {"source_bindings", "tasks", "retrieval"}, state=True)
    _header(conversations, "serveflow-coverage-conversations-v1", {"source_bindings", "conversations"}, state=True)
    _bindings(cases["source_bindings"], FILES[:2], root)
    _bindings(conversations["source_bindings"], FILES[:3], root)
    _list(corpus["documents"], 24)
    docs, facts, fact_splits, families = {}, {}, {}, {}
    for doc in corpus["documents"]:
        _exact(doc, {"document_id", "title", "split", "family_id", "content", "facts"})
        for key in ("document_id", "title", "family_id"):
            _text(doc[key], 160)
        _text(doc["content"], 20_000)
        if (type(doc["split"]) is not str or doc["split"] not in SPLITS or doc["document_id"] in docs
                or doc["split"] != ("development" if doc["document_id"].startswith("CX-D") else "validation")):
            _fail("document_identity")
        if families.setdefault(doc["family_id"], doc["split"]) != doc["split"]:
            _fail("source_family_split")
        docs[doc["document_id"]] = doc
        _list(doc["facts"], 32, nonempty=True)
        chunks = _chunks(doc["content"])
        for fact in doc["facts"]:
            _exact(fact, {"fact_id", "quote"})
            _text(fact["fact_id"], 160); _text(fact["quote"], 500)
            normalized = bench._normal(fact["quote"])
            if (not re.fullmatch(re.escape(doc["document_id"]) + r"-F[0-9]{2}", fact["fact_id"])
                    or fact["fact_id"] in facts or fact["quote"] not in doc["content"]
                    or not any(fact["quote"] in chunk for chunk in chunks)
                    or fact_splits.setdefault(normalized, doc["split"]) != doc["split"]):
                _fail("fact_quote_or_split")
            facts[fact["fact_id"]] = (doc["document_id"], fact["quote"])
    if (Counter(doc["split"] for doc in docs.values()) != {"development": 8, "validation": 16}
            or set(docs) != {f"CX-{s}{i:02d}" for s, count in (("D", 8), ("V", 16)) for i in range(1, count + 1)}
            or sum(len(doc["content"]) > 500 for doc in docs.values()) != 4):
        _fail("corpus_counts")
    chunks = source_chunks(corpus["documents"])
    fixture = records["fixture"]
    try:
        validate_fixture(fixture)
    except (ValueError, TypeError, KeyError, DemoStoreError):
        # Driver-free production validation; expose only a fixed safe code.
        _fail("fixture")
    _list(records["order_splits"], 32)
    order_splits = {}
    for row in records["order_splits"]:
        _exact(row, {"order_id", "split", "family_id"})
        _text(row["order_id"], 160)
        _text(row["family_id"], 160)
        if row["order_id"] in order_splits or type(row["split"]) is not str or row["split"] not in SPLITS:
            _fail("order_identity")
        if families.setdefault(row["family_id"], row["split"]) != row["split"]:
            _fail("source_family_split")
        order_splits[row["order_id"]] = row
    expected_orders = {f"DEMO-{i}": "development" if i < 2100 else "validation"
        for i in [*range(2001, 2013), *range(2101, 2121)]}
    if ({k: v["split"] for k, v in order_splits.items()} != expected_orders
            or {row["order_id"] for row in fixture["orders"]} != set(expected_orders)):
        _fail("order_counts")
    seen, inputs = set(), set()
    for suite, task, prefix, counts in (("tasks", True, "CT", (12, 28)), ("retrieval", False, "CR", (16, 32))):
        _list(cases[suite], sum(counts))
        expected_ids = {f"{prefix}-{s}{i:02d}" for s, count in zip(("D", "V"), counts) for i in range(1, count + 1)}
        for case in cases[suite]:
            _exact(case, CASE_KEYS)
            _text(case["id"], 160)
            if case["id"] in seen or case["id"] not in expected_ids or type(case["split"]) is not str or case["split"] not in SPLITS:
                _fail("case_identity")
            if case["split"] != ("development" if case["id"].startswith(prefix + "-D") else "validation"):
                _fail("case_split")
            seen.add(case["id"])
            for field in ("family_id", "provenance"):
                _text(case[field])
            _strings(case["tags"], 24)
            if case["review_status"] != "reviewed":
                _fail("review_status")
            _input(case["input"], task)
            normalized = _hash(model_input(case))
            if normalized in inputs:
                _fail("duplicate_input")
            inputs.add(normalized)
            doc_ids, fact_ids = _references(case["references"], case["split"], docs, facts, fixture, order_splits)
            if not task and any(ref["source"] != "knowledge" for ref in case["references"]):
                _fail("retrieval_reference_source")
            _expected(case["expected"], task, doc_ids, fact_ids)
            if not _top3_possible(case["expected"]["required_fact_ids"], facts, chunks):
                _fail("required_facts_exceed_top3")
        if {case["id"] for case in cases[suite]} != expected_ids:
            _fail("case_counts")
    for split, expected in (("development", {"answerable": 8, "insufficient": 4, "out_of_scope": 4}),
                            ("validation", {"answerable": 16, "insufficient": 8, "out_of_scope": 8})):
        if Counter(case["expected"]["kind"] for case in cases["retrieval"] if case["split"] == split) != expected:
            _fail("retrieval_balance")
    _list(conversations["conversations"], 8)
    for group in conversations["conversations"]:
        _exact(group, {"id", "split", "family_id", "customer_id", "turns", "tags", "provenance", "review_status"})
        _text(group["id"], 160)
        if (group["id"] in seen or type(group["split"]) is not str or group["split"] not in SPLITS
                or type(group["customer_id"]) is not str or group["customer_id"] not in CUSTOMER_IDS
                or group["review_status"] != "reviewed"):
            _fail("conversation_identity")
        seen.add(group["id"])
        for field in ("id", "family_id", "provenance"):
            _text(group[field])
        _strings(group["tags"], 24)
        _list(group["turns"], 5)
        if len(group["turns"]) != 5:
            _fail("turn_count")
        for turn in group["turns"]:
            _exact(turn, {"id", "message", "expected", "references"})
            _text(turn["id"], 160); _text(turn["message"])
            if turn["id"] in seen:
                _fail("turn_identity")
            seen.add(turn["id"])
            _exact(turn["expected"], TURN_EXPECTED)
            for field in ("required_facts", "missing_facts", "forbidden_claims", "required_tools"):
                _strings(turn["expected"][field], 32)
            if (type(turn["expected"]["expected_status"]) is not str
                    or turn["expected"]["expected_status"] not in {"success", "partial", "not-prescribed"}
                    or not set(turn["expected"]["required_tools"]) <= bench.TOOLS):
                _fail("turn_expectation")
            _references(turn["references"], group["split"], docs, facts, fixture, order_splits)
    if Counter(group["split"] for group in conversations["conversations"]) != {"development": 2, "validation": 6}:
        _fail("conversation_counts")
    return corpus, records, cases, conversations


def validate(root=bench.BACKEND_ROOT):
    corpus, records, cases, conversations = load_validated(root)
    used = {split: set() for split in SPLITS}
    for case in [*cases["tasks"], *cases["retrieval"]]:
        used[case["split"]].update(case["expected"]["required_fact_ids"])
    for group in conversations["conversations"]:
        for turn in group["turns"]:
            used[group["split"]].update(ref["fact_id"] for ref in turn["references"] if ref["source"] == "knowledge")
    return {"documents": 24, "orders": 32, "tasks": 40, "retrieval": 48,
        "conversations": 8, "conversation_turns": 40, "long_documents": 4,
        "source_fact_coverage": {split: {"available": sum(len(d["facts"]) for d in corpus["documents"] if d["split"] == split),
            "referenced": len(used[split])} for split in sorted(SPLITS)},
        "authorized_model_requests": 0, "executed_model_requests": 0, "live_sender_ready": False,
        "quality_pass_claim": False}


def _literal(root, path, name):
    tree = ast.parse(_safe(root, path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    _fail("public_parameter")


def public_parameters(root=bench.BACKEND_ROOT):
    phases = _literal(root, "evaluation/task_transport.py", "PHASE_PARAMETERS")
    deadlines = _literal(root, "evaluation/phase_deadline_transport.py", "PHASE_DEADLINES")
    return {"model": _literal(root, "core/provider_options.py", "FLASH_SNAPSHOT"),
        "phase_parameters": {k: v for k, v in phases.items() if k != "judge"},
        "agent_parameters": _literal(root, "evaluation/task_transport.py", "AGENT_PARAMETERS"),
        "phase_deadlines": {k: v for k, v in deadlines.items() if k != "judge"},
        "http_limits": _literal(root, "evaluation/task_transport.py", "HTTP_LIMITS"),
        "sdk_timeout_s": _literal(root, "evaluation/phase_deadline_transport.py", "SDK_TIMEOUT_S"),
        "live_environment_read": False, "payloads_frozen": False}


def inventory(root=bench.BACKEND_ROOT):
    root, values = Path(root), {}
    for folder in sorted(SOURCE_FOLDERS):
        directory = root / folder
        if not directory.is_dir() or directory.is_symlink():
            _fail("source_directory")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                _fail("source_symlink")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                relative = str(path.relative_to(root))
                values[relative] = bench.sha256(_safe(root, relative))
    for path in sorted(root.glob("*.py")) + sorted(root.glob("requirements*.txt")):
        values[path.name] = bench.sha256(_safe(root, path.name))
    if len(values) > 2000:
        _fail("inventory_limit")
    return values


def old_source_proof(root=bench.BACKEND_ROOT):
    """Pinned old six-policy corpus, fixture and every old question file."""
    root = Path(root)
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", BASELINE, "--",
        "backend/knowledge/serveflow-demo-v2.json", "backend/demo_business/fixtures.json", "backend/evaluation/benchmark"],
        cwd=root.parent, check=True, capture_output=True, timeout=15).stdout.decode().splitlines()
    if not names or len(names) > 256:
        _fail("old_source_inventory")
    result = {}
    for name in names:
        if not name.startswith("backend/") or not name.endswith(".json"):
            continue
        raw = subprocess.run(["git", "show", f"{BASELINE}:{name}"], cwd=root.parent,
            check=True, capture_output=True, timeout=15).stdout
        relative = name.removeprefix("backend/")
        digest = hashlib.sha256(raw).hexdigest()
        if bench.sha256(_safe(root, relative)) != digest:
            _fail("old_source_changed")
        result[relative] = digest
    return {"baseline_commit": BASELINE, "artifacts": result, "old_scores_changed": False,
            "historical_senders_reopened": False}


def expected(root=bench.BACKEND_ROOT):
    _, _, cases, conversations = load_validated(root)
    return {"schema_version": SCHEMA, "phase": "offline-preparation", "summary": validate(root),
        "data_sha256": {path: bench.sha256(_safe(root, path)) for path in FILES},
        "source_inventory": inventory(root), "old_sources": old_source_proof(root),
        "model_input_manifest": [{"case_id": case["id"], "suite": suite, "split": case["split"],
            "model_input_sha256": _hash(model_input(case)), "trusted_context_sha256": _hash(trusted_input(case))}
            for suite in ("tasks", "retrieval") for case in cases[suite]],
        "conversation_user_messages": [{"conversation_id": group["id"], "turn_id": turn["id"],
            "message_sha256": _hash({"message": turn["message"]}), "runtime_history_not_yet_captured": True}
            for group in conversations["conversations"] for turn in group["turns"]],
        "public_parameters": public_parameters(root), "authorized_model_requests": 0,
        "executed_model_requests": 0, "database_connections": 0, "sdk_captures": 0,
        "live_sender_ready": False, "quality_pass_claim": False,
        "limitations": ["authored synthetic sources; validation visible to authors; no blind or live quality evidence",
            "exact fact/source split checks do not establish semantic independence",
            "no SDK payload, execution adapter, persistent send budget or live environment frozen",
            "document hits and required fact quotes are separate; insufficient evidence is not a complete answer",
            "historical evidence, scores, caches and unused allowances are not reused"]}


def prepare(root=ROOT, *, source_root=bench.BACKEND_ROOT):
    root = Path(root).absolute()
    if root.exists() or any(p.is_symlink() for p in [root, *root.parents]):
        _fail("preparation_exists")
    value = expected(source_root)
    if value != expected(source_root):
        _fail("preparation_changed")
    value["created_at"] = datetime.now(timezone.utc).isoformat()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = root / "protocol.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return value


def verify(root=ROOT, *, source_root=bench.BACKEND_ROOT):
    value = _read(root, "protocol.json")
    current = json.loads(json.dumps(expected(source_root)))
    _exact(value, set(current) | {"created_at"})
    _text(value["created_at"], 80)
    if any(value[k] != v for k, v in current.items()):
        _fail("preparation_changed")
    return {"verified": True, **current["summary"]}


def validate_source_rows(rows, *, root=bench.BACKEND_ROOT):
    """Pure source admission, without asserting how observations were obtained."""
    corpus, _, _, _ = load_validated(root)
    chunks = source_chunks(corpus["documents"])
    return _validated_rows(rows, chunks)


def _validated_rows(rows, chunks):
    _list(rows, 20)
    seen = set()
    for row in rows:
        _exact(row, {"document_id", "title", "content", "version", "chunk", "score"})
        for key, limit in (("document_id", 64), ("title", 160), ("content", 500), ("version", 64)):
            _text(row[key], limit)
        if type(row["chunk"]) is not int or type(row["score"]) not in {int, float} or not math.isfinite(row["score"]):
            _fail("source_row")
        token = (row["document_id"], row["chunk"], row["version"])
        source = chunks.get(token)
        if token in seen or source is None or any(row[k] != source[k] for k in ("title", "content")):
            _fail("source_row_binding")
        seen.add(token)
    return json.loads(json.dumps(rows))


def retrieval_metrics(case, rows, status, *, root=bench.BACKEND_ROOT):
    """One bound case's deterministic metrics; no evidence-kind/quality claim."""
    corpus, _, cases, _ = load_validated(root)
    if case not in cases["retrieval"] or type(status) is not str or status not in {"success", "failed", "cancelled", "unknown", "not_run"}:
        _fail("metric_case")
    rows = validate_source_rows(rows, root=root)
    if status != "success" and rows:
        _fail("failure_rows")
    chunks = source_chunks(corpus["documents"])
    facts = {fact["fact_id"]: (doc["document_id"], fact["quote"])
        for doc in corpus["documents"] for fact in doc["facts"]}
    return _metric(case, rows, status, chunks, facts)


def _metric(case, rows, status, chunks, facts):
    top = [chunks[(row["document_id"], row["chunk"], row["version"])] for row in rows[:3]]
    required = case["expected"]["required_fact_ids"]
    covered = [fact for fact in required if any(row["explicit_document_id"] == facts[fact][0]
        and facts[fact][1] in row["content"] for row in top)]
    ranks = [i for i, row in enumerate(top, 1) if row["explicit_document_id"] in case["expected"]["relevant_document_ids"]]
    healthy = status == "success"
    return {"case_id": case["id"], "split": case["split"], "kind": case["expected"]["kind"],
        "status": status, "document_hit_at_3": bool(ranks),
        "required_fact_count": len(required), "covered_fact_ids_at_3": covered,
        "required_fact_recall_at_3": len(covered) / len(required) if required else None,
        "all_required_facts_at_3": bool(required) and len(covered) == len(required),
        "reciprocal_rank_at_3": 1 / ranks[0] if ranks else 0,
        "healthy_empty_rejection": healthy and not top and case["expected"]["kind"] == "out_of_scope",
        "out_of_scope_false_hit": healthy and bool(top) and case["expected"]["kind"] == "out_of_scope",
        "top3_characters": sum(len(row["content"]) for row in top)}


def score_retrieval(observations, *, root=bench.BACKEND_ROOT):
    corpus, _, cases, _ = load_validated(root)
    _exact(observations, {"schema_version", "evidence_kind", "arm", "results"})
    if (observations["schema_version"] != "serveflow-coverage-retrieval-observations-v1"
            or observations["evidence_kind"] != "controlled-offline"
            or type(observations["arm"]) is not str or observations["arm"] not in {"vector_only", "enhanced"}):
        _fail("observation_kind")
    _list(observations["results"], 48)
    by_id, chunks = {}, source_chunks(corpus["documents"])
    case_ids = {case["id"] for case in cases["retrieval"]}
    for result in observations["results"]:
        _exact(result, {"case_id", "status", "rows"})
        if (type(result["case_id"]) is not str or result["case_id"] not in case_ids or result["case_id"] in by_id
                or type(result["status"]) is not str or result["status"] not in {"success", "failed", "cancelled", "unknown", "not_run"}):
            _fail("observation_identity")
        _list(result["rows"], 20)
        if result["status"] != "success" and result["rows"]:
            _fail("failure_rows")
        _validated_rows(result["rows"], chunks)
        by_id[result["case_id"]] = result
    facts = {fact["fact_id"]: (doc["document_id"], fact["quote"])
        for doc in corpus["documents"] for fact in doc["facts"]}
    per_case = []
    for case in cases["retrieval"]:
        result = by_id.get(case["id"], {"status": "not_run", "rows": []})
        per_case.append(_metric(case, result["rows"], result["status"], chunks, facts))
    groups = {}
    for split in sorted(SPLITS):
        for kind in ("answerable", "insufficient", "out_of_scope"):
            rows = [r for r in per_case if r["split"] == split and r["kind"] == kind]
            groups[f"{split}/{kind}"] = {"planned": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                "document_hits_at_3": sum(r["document_hit_at_3"] for r in rows),
                "all_required_facts_at_3": sum(r["all_required_facts_at_3"] for r in rows),
                "macro_required_fact_recall_at_3": sum(r["required_fact_recall_at_3"] or 0 for r in rows) / len(rows) if kind != "out_of_scope" else None,
                "healthy_empty_rejections": sum(r["healthy_empty_rejection"] for r in rows),
                "out_of_scope_false_hits": sum(r["out_of_scope_false_hit"] for r in rows)}
    return {"schema_version": "serveflow-coverage-retrieval-report-v1", "evidence_kind": "controlled-offline",
        "arm": observations["arm"], "planned": 48, "groups": groups, "cases": per_case,
        "quality_pass_claim": False, "live_model_quality": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "verify", "score"))
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            if args.observations is None:
                _fail("observations_required")
            result = score_retrieval(bench.read_json(args.observations))
        else:
            result = {"validate": validate, "prepare": prepare, "verify": verify}[args.command]()
        print(json.dumps({"prepared": True, **result["summary"]} if args.command == "prepare" else result, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"error": "coverage_offline_check_failed", "executed_model_requests": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
