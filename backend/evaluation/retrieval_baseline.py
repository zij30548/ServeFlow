"""One fixed, local-vector retrieval measurement; no model API or application DB.

The standalone CLI denies Python sockets and uses only the existing byte-checked
ONNX cache plus a new ephemeral Chroma collection. It never loads dotenv, creates
an LLM client, rewrites queries or reranks results. This baseline alone cannot
establish an improvement of the enhanced retrieval pipeline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from unittest.mock import patch

from evaluation import benchmark as bench, intent_trial as private
from evaluation import task_data
from evaluation.benchmark_metrics import score_retrieval
from evaluation.intent_transport import canonical_payload_hash


ROOT = bench.BACKEND_ROOT / "data/benchmarks/retrieval-vector-baseline-v1"
TOP_K = 3
SOURCE_FILES = (
    "evaluation/retrieval_baseline.py", "tests/test_retrieval_baseline.py",
    "evaluation/benchmark.py", "evaluation/benchmark_metrics.py", "evaluation/task_data.py",
    "evaluation/intent_trial.py", "evaluation/intent_transport.py", "evaluation/trial_ledger.py",
    "tooling/knowledge_base.py", "knowledge/serveflow-demo-v2.json",
    "demo_business/store.py", "demo_business/initialize.py", "demo_business/fixtures.json",
    "evaluation/benchmark/retrieval-v1.json", "requirements.txt",
)
KNOWLEDGE_FIELDS = ("chromadb_version", "embedding", "embedding_implementation_sha256",
                    "corpus_sha256", "document_count", "chunk_count", "chunk_sha256")
LIMITATIONS = [
    "24 authored known questions against 6 short synthetic policy documents; not blind or enterprise evaluation",
    "development and validation remain separate; answerability labels are not search inputs",
    "local vector retrieval only; no query rewrite, rerank, generated answer, model API or paid embedding",
    "Hit@3 is evidence retrieval, not answer correctness or complete coverage",
    "top3 covers half of this six-document corpus; high Hit@3 alone is weak evidence of broad retrieval quality",
    "out-of-scope nonempty results are false retrievals, not demonstrated generated hallucinations",
    "no tuned threshold: ordinary vector nearest-neighbor search may return irrelevant materials",
    "one query per planned item; no retry, replacement or inferred result after interruption",
]


class RetrievalBaselineError(ValueError):
    """Fixed safe codes only; never arbitrary model, DB or exception content."""


def _fail(code):
    raise RetrievalBaselineError(code)


def _cases():
    dataset = bench.load_suites()["retrieval"]
    bench.validate_suite(dataset)
    return dataset["cases"]


def _knowledge_proof():
    value = task_data.offline_data_proof()["knowledge"]
    return {name: value[name] for name in KNOWLEDGE_FIELDS}


def _artifacts():
    return {name: bench.sha256(bench.BACKEND_ROOT / name) for name in SOURCE_FILES}


def _dependencies():
    return {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(), **{
        name: metadata.version(name) for name in ("chromadb", "onnxruntime", "numpy", "tokenizers")}}


def _plan(cases):
    return [{"case_id": case["id"], "split": case["split"],
             "input_sha256": canonical_payload_hash(bench.model_input(case))} for case in cases]


@contextmanager
def no_network():
    """Standalone measurement only: no DNS/socket escape or model download."""
    def denied(*_args, **_kwargs):
        _fail("retrieval_network_forbidden")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
            patch.object(socket, "create_connection", denied), patch.object(socket, "getaddrinfo", denied):
        yield


def _rows(value):
    """Validate returned identities and full source bodies without repairing rows."""
    if type(value) is not list or len(value) > TOP_K:
        _fail("retrieval_result_invalid")
    docs = {doc["title"]: doc["content"] for doc in task_data._policies()}
    output, seen = [], set()
    for row in value:
        if type(row) is not dict or set(row) != {"title", "content", "score", "chunk", "document_id", "version"}:
            _fail("retrieval_result_invalid")
        title, content = row["title"], row["content"]
        if type(title) is not str or title not in docs or type(content) is not str or content != docs[title]:
            _fail("retrieval_source_mismatch")
        identity = json.dumps(["title", title], ensure_ascii=True)
        document_id = hashlib.sha256(identity.encode()).hexdigest()
        digest = hashlib.sha256(content.encode()).hexdigest()
        if (row["document_id"] != document_id or row["version"] != digest
                or type(row["chunk"]) is not int or row["chunk"] != 0
                or type(row["score"]) not in (int, float) or not math.isfinite(row["score"])
                or document_id in seen):
            _fail("retrieval_source_mismatch")
        seen.add(document_id)
        output.append({**row, "content_sha256": digest})
    return output


class RetrievalBaseline:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        private._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            _fail("fixed_retrieval_baseline_root_required")

    def prepare(self):
        if self.root.exists():
            _fail("retrieval_root_already_exists")
        cases = _cases()
        value = {"schema_version": "serveflow-local-retrieval-protocol-v1", "created_at": private._now(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": private._root_hash(self.root), "artifacts": _artifacts(),
            "dependencies": _dependencies(), "knowledge": _knowledge_proof(), "plan": _plan(cases),
            "top_k": TOP_K, "model_api_requests": 0, "external_business_requests": 0,
            "method": "production KnowledgeBase.search, original query, no threshold/rewrite/rerank",
            "limitations": list(LIMITATIONS)}
        self.root.mkdir(mode=0o700, parents=True)
        private._write(self.root, "protocol.json", value)
        return value

    def verify(self):
        p = bench.read_json(private._path(self.root, "protocol.json"))
        expected = {"schema_version", "created_at", "git_head", "root_binding_sha256", "artifacts", "dependencies",
                    "knowledge", "plan", "top_k", "model_api_requests", "external_business_requests", "method", "limitations"}
        if (type(p) is not dict or set(p) != expected or p["schema_version"] != "serveflow-local-retrieval-protocol-v1"
                or p["root_binding_sha256"] != private._root_hash(self.root) or type(p["top_k"]) is not int or p["top_k"] != TOP_K
                or type(p["model_api_requests"]) is not int or p["model_api_requests"] != 0
                or type(p["external_business_requests"]) is not int or p["external_business_requests"] != 0
                or p["artifacts"] != _artifacts() or p["dependencies"] != _dependencies()
                or p["knowledge"] != _knowledge_proof() or p["plan"] != _plan(_cases())
                or p["method"] != "production KnowledgeBase.search, original query, no threshold/rewrite/rerank"
                or p["limitations"] != LIMITATIONS):
            _fail("retrieval_frozen_protocol_changed")
        return p

    def measure(self, *, knowledge_factory=None):
        p = self.verify()
        if (self.root / "measurement-start.json").exists():
            _fail("retrieval_measurement_already_started")
        if knowledge_factory is None and os.environ.get("ORT_DISABLE_TELEMETRY") != "1":
            _fail("retrieval_disabled_onnx_telemetry_required")
        kind = "controlled-fixture" if knowledge_factory is not None else "real-local-vector-retrieval"
        protocol_hash = bench.sha256(self.root / "protocol.json")
        private._write(self.root, "measurement-start.json", {"protocol_sha256": protocol_hash,
            "evidence_kind": kind, "model_api_requests": 0, "recorded_at": private._now()})
        runtime = None
        try:
            with no_network():
                startup = time.perf_counter()
                runtime = (knowledge_factory or task_data._build_knowledge)()
                proof = {name: runtime.proof[name] for name in KNOWLEDGE_FIELDS}
                if proof != p["knowledge"]:
                    _fail("retrieval_index_identity_changed")
                private._write(self.root, "index-proof.json", {"proof": proof, "evidence_kind": kind,
                    "index_startup_ms": (time.perf_counter()-startup)*1000})
                for case, planned in zip(_cases(), p["plan"]):
                    self.verify()
                    claim_name = f"case-{case['id']}-start.json"
                    claim = {**planned, "protocol_sha256": protocol_hash, "recorded_at": private._now()}
                    private._write(self.root, claim_name, claim)
                    start = time.perf_counter()
                    try:
                        rows = _rows(runtime.kb.search(case["input"]["message"], top_k=TOP_K))
                        status, error = "success", None
                    except RetrievalBaselineError as exc:
                        if str(exc) == "retrieval_network_forbidden":
                            raise
                        rows, status, error = [], "failed", "retrieval_result_rejected"
                    except task_data.TaskDataError:
                        raise
                    except Exception:
                        rows, status, error = [], "failed", "retrieval_query_failed"
                    value = {"case_id": case["id"], "split": case["split"], "input_sha256": planned["input_sha256"],
                        "status": status, "latency_ms": (time.perf_counter()-start)*1000,
                        "ranked_titles": [row["title"] for row in rows], "results": rows, "error_code": error,
                        "evidence_kind": kind, "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 0}}
                    name = f"case-{case['id']}.json"
                    private._write(self.root, name, value)
                    private._write(self.root, f"case-{case['id']}-seal.json", {"protocol_sha256": protocol_hash,
                        "claim_sha256": bench.sha256(self.root / claim_name), "output_sha256": bench.sha256(self.root / name)})
        except BaseException:
            private._write(self.root, "measurement-stop.json", {"reason": "execution_stopped", "model_api_requests": 0})
            raise
        finally:
            if runtime is not None:
                try:
                    with no_network():
                        runtime.close()
                    private._write(self.root, "cleanup-complete.json", {"owned_collection_deleted": True,
                        "model_api_requests": 0})
                except BaseException as error:
                    private._write(self.root, "cleanup-failed.json", {"reason": "owned_collection_cleanup_failed",
                        "model_api_requests": 0})
                    if isinstance(error, Exception):
                        raise RetrievalBaselineError("retrieval_cleanup_failed") from None
                    raise
        report = self.report()
        private._write(self.root, "report.json", report)
        return report

    def report(self):
        p = self.verify()
        cases = _cases()
        records, observations = [], []
        protocol_hash = bench.sha256(self.root / "protocol.json")
        start_path = private._path(self.root, "measurement-start.json")
        started = bench.read_json(start_path) if start_path.exists() else None
        if started is not None and (
                type(started) is not dict or set(started) != {"protocol_sha256", "evidence_kind", "model_api_requests", "recorded_at"}
                or started["protocol_sha256"] != protocol_hash
                or started["evidence_kind"] not in {"controlled-fixture", "real-local-vector-retrieval"}
                or type(started["model_api_requests"]) is not int or started["model_api_requests"] != 0):
            _fail("retrieval_record_changed")
        for case, plan in zip(cases, p["plan"]):
            name = f"case-{case['id']}.json"
            seal_path = private._path(self.root, f"case-{case['id']}-seal.json")
            claim_path = private._path(self.root, f"case-{case['id']}-start.json")
            if not seal_path.exists():
                if claim_path.exists():
                    records.append({"case_id": case["id"], "status": "unknown", "latency_ms": None,
                        "ranked_titles": [], "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 0}})
                continue
            seal = bench.read_json(seal_path)
            if (started is None or type(seal) is not dict or set(seal) != {"protocol_sha256", "claim_sha256", "output_sha256"}
                    or seal["protocol_sha256"] != protocol_hash
                    or seal["claim_sha256"] != bench.sha256(claim_path)
                    or seal["output_sha256"] != bench.sha256(private._path(self.root, name))):
                _fail("retrieval_record_changed")
            value = bench.read_json(private._path(self.root, name))
            fields = {"case_id", "split", "input_sha256", "status", "latency_ms", "ranked_titles", "results", "error_code", "evidence_kind", "usage"}
            if (type(value) is not dict or set(value) != fields
                    or value.get("case_id") != case["id"] or value.get("split") != case["split"]
                    or value.get("input_sha256") != plan["input_sha256"]
                    or value["evidence_kind"] != started["evidence_kind"]):
                _fail("retrieval_record_changed")
            observations.append(value)
            records.append({key: value[key] for key in ("case_id", "status", "latency_ms", "ranked_titles", "usage")})
        return {"schema_version": "serveflow-local-retrieval-report-v1", "protocol_sha256": protocol_hash,
            "model_api_requests": 0, "external_business_requests": 0,
            "evidence_kind": started["evidence_kind"] if started else "not-executed",
            "execution_state": ("cleanup_failed" if (self.root / "cleanup-failed.json").exists() else
                "execution_stopped" if (self.root / "measurement-stop.json").exists() else
                "completed" if len(observations) == len(cases) and (self.root / "cleanup-complete.json").exists() else
                "incomplete" if started else "prepared"),
            "owned_collection_cleanup_complete": (self.root / "cleanup-complete.json").exists(),
            "by_split": {split: score_retrieval([c for c in cases if c["split"] == split],
                [r for r in records if any(c["id"] == r["case_id"] and c["split"] == split for c in cases)])
                for split in ("development", "validation")},
            "observations": observations, "limitations": list(LIMITATIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "measure", "report"))
    args = parser.parse_args(argv)
    trial = RetrievalBaseline()
    try:
        if args.operation == "prepare":
            p = trial.prepare()
            value = {"prepared": True, "planned_queries": len(p["plan"]), "model_api_requests": 0}
        elif args.operation == "verify":
            trial.verify()
            value = {"verified": True, "model_api_requests": 0}
        else:
            result = trial.measure() if args.operation == "measure" else trial.report()
            value = {"model_api_requests": 0, "by_split": result["by_split"]}
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        print("STOP: retrieval_baseline_failed; inspect private records, do not overwrite or rerun")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
