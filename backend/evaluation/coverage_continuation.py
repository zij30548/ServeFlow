"""One explicitly approved continuation of the original 20/480 coverage trial.

No new allowance, task, model, prompt or score. Old immutable evidence and the
journal prefix are retained. A sent failed operation is never executed again.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from core.model_deadline import EvaluationDeadlinePolicy, ModelDeadlineExceeded, eval_deadlines
from evaluation import benchmark as bench, coverage_expansion as preparation, intent_trial as base
from evaluation import coverage_execution as original, task_data
from evaluation.coverage_budget import ROOT, MAX_REQUESTS, fixed_plan, job_specs
from evaluation.intent_transport import TrialTransportError, canonical_payload_hash
from evaluation.phase_deadline_transport import PHASE_HTTP_LIMITS, SDK_TIMEOUT_S


SCHEMA = "serveflow-coverage-continuation-v1"
MANIFEST = "continuation-v1.json"
APPROVAL = "continuation-v1-authorization.json"
STARTED = "continuation-v1-worker-started.json"
STOP = "continuation-v1-worker-stop.json"
PREFLIGHT = original.PREFLIGHT / "continuation-v1"
ORIGINAL_HEAD = "d5c2fa0"
ORIGINAL_PROTOCOL_SHA = "a49f864c00d9647517a98c9617856346510b4e3b99c45c0803cde70e9ff7a3fc"
POLICY = {
    "grant": "same original480; already20; at most460 more; no new or recycled allowance",
    "plan": "original136 positions/168 turns; old6 results unchanged; old D-CT-D04 failed and never resent; remaining129 positions once in original order",
    "soft_failure": "one actually sent timeout/network failure ends that case; bounded drain, zero pending and sealed outputs required before next independent case",
    "global_stop": "third consecutive network/timeout case (including original failed case), auth, quota, rate limit, storage, version, source, cancellation, pending or uncertain cleanup stops without restart",
    "unchanged": "same business source, questions, expected labels, model, prompt, SDK payloads, phase deadlines, generation options, source corpus and rendering",
    "review": "frozen seven dimensions; five real turns per conversation conjunctively; all planned denominators; no judge, score-target tuning or post hoc output repair",
}
NEW_SOURCE_PATHS = frozenset({
    f"evaluation/coverage_continuation{suffix}.py" for suffix in ("", "_budget", "_transport", "_report")
} | {f"tests/test_coverage_continuation{suffix}.py" for suffix in ("", "_budget", "_transport", "_report")})
CHANGED_SOURCE_PATHS = frozenset({"evaluation/task_supervisor.py"})
MANIFEST_FIELDS = {"schema_version", "created_at", "root_binding_sha256", "git_head", "original_code_commit",
    "original_files", "budget", "previous_active_case", "remaining_jobs", "plan", "source_inventory",
    "configuration", "public_parameters", "effective_environment", "dependencies", "redis_binding",
    "policy", "review_plan_sha256", "authorization_maximum_total", "additional_allowance"}
APPROVAL_FIELDS = {"schema_version", "recorded_at", "reference", "continuation_sha256",
    "original_authorization_sha256", "prefix_sha256", "original_genesis", "max_total_requests",
    "already_spent", "max_additional_requests"}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(root, name):
    return bench.read_json(base._path(root, name))


def _files(root):
    values = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            base._fail("continuation_evidence_symlink")
        if path.is_file() and path.relative_to(root).as_posix() != "ledger/journal.jsonl":
            values[path.relative_to(root).as_posix()] = bench.sha256(path)
    return values


def _original_sources(p):
    current = preparation.inventory()
    changed = {name for name, digest in p["source_inventory"].items() if current.get(name) != digest}
    added = set(current) - set(p["source_inventory"])
    if not changed <= CHANGED_SOURCE_PATHS or not added <= NEW_SOURCE_PATHS:
        base._fail("continuation_unrelated_source_change")
    for name in changed:
        before = subprocess.run(["git", "show", f"{ORIGINAL_HEAD}:backend/{name}"],
            cwd=bench.BACKEND_ROOT, check=True, capture_output=True, timeout=15).stdout
        if _hash(before) != p["source_inventory"][name]:
            base._fail("continuation_original_source_identity")
        # Only one new fixed worker and its exact ROOT check may be added.
        old_lines = before.decode().splitlines()
        new_lines = (bench.BACKEND_ROOT / name).read_text().splitlines()
        import difflib
        delta = list(difflib.unified_diff(old_lines, new_lines, n=0))
        altered = [line[1:] for line in delta if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))]
        if len(altered) > 14 or not any("coverage_continuation" in line for line in altered):
            base._fail("continuation_supervisor_scope")
    return current


def _original_valid(root=ROOT):
    p = _read(root, "protocol.json")
    if (bench.sha256(root / "protocol.json") != ORIGINAL_PROTOCOL_SHA
            or p["plan"] != fixed_plan() or p["jobs"] != job_specs()
            or p["cases"] != original.source_cases()
            or p["configuration"] != original.configuration()
            or p["public_parameters"] != json.loads(json.dumps(preparation.public_parameters()))
            or p["effective_environment"] != original.history.effective_environment()
            or p["dependencies"] != original.engine._dependencies()
            or p["preparation"] != original.frozen_preparation()
            or p["prior_evidence"] != original.preserved_evidence()
            or p["http_limits"] != PHASE_HTTP_LIMITS):
        base._fail("continuation_original_context_changed")
    legacy = original.CoverageExecutionTrial(root)
    if (root / MANIFEST).exists():
        # Old classes intentionally reject the new journal vocabulary. Keep
        # the old grant checks, obtaining its unchanged genesis via the one
        # manifest-bound continuation interpreter once it exists.
        legacy.ledger = lambda ignored=None: CoverageContinuation(root).ledger()
    if legacy._approval(p)["max_requests"] != MAX_REQUESTS:
        base._fail("continuation_original_grant_changed")
    _original_sources(p)
    return p


class CoverageContinuation:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, MANIFEST)

    def _original(self):
        return _original_valid(self.root)

    def ledger(self, p=None, *, manifest=None):
        from evaluation.coverage_continuation_budget import ContinuationBudget
        manifest = manifest or _read(self.root, MANIFEST)
        b = manifest["budget"]
        return ContinuationBudget(self.root / "ledger", b["original_protocol_sha256"], fixed_plan(),
            continuation_sha256=bench.sha256(self.root / MANIFEST), prefix_size=b["prefix_size"],
            prefix_sha256=b["prefix_sha256"], prefix_chain_hash=b["prefix_chain_hash"], send_limit=MAX_REQUESTS)

    async def prepare(self):
        if self.root != ROOT or (self.root / MANIFEST).exists():
            base._fail("continuation_fixed_once_root")
        p = self._original()
        legacy = original.CoverageExecutionTrial(self.root)
        state = legacy.ledger(p).status()
        sealed = legacy._sealed(p)
        prefix = (self.root / "ledger/journal.jsonl").read_bytes()
        if (state["attempts"] != 20 or len(prefix.splitlines()) != 53 or not state["paused"]
                or state["active_case"] != "D-CT-D04" or state["pending"]
                or state["completed_cases"] != [r["case_id"] for r in job_specs()[:6]]
                or set(sealed["cases"]) != set(state["completed_cases"])):
            base._fail("continuation_exact_twenty_prefix_required")
        capture = await original.capture_first_requests(p["configuration"], p["cases"])
        if any(capture[k] != p[k] for k in capture):
            base._fail("continuation_payload_changed")
        for spec in job_specs()[6:]:
            if any((self.root / f"{prefix_name}-{spec['case_id']}.json").exists() for prefix_name in ("case", "review")):
                base._fail("continuation_unexpected_preexisting_result")
        value = {
            "schema_version": SCHEMA, "created_at": base._now(),
            "root_binding_sha256": base._root_hash(self.root), "git_head": original.git_head(),
            "original_code_commit": p["git_head"], "original_files": _files(self.root),
            "budget": {"original_protocol_sha256": bench.sha256(self.root / "protocol.json"),
                "prefix_size": len(prefix), "prefix_sha256": _hash(prefix),
                "prefix_chain_hash": state["chain_hash"], "already_spent": 20, "max_requests": MAX_REQUESTS},
            "previous_active_case": "D-CT-D04", "remaining_jobs": job_specs()[7:],
            "plan": p["plan"], "source_inventory": _original_sources(p),
            "configuration": p["configuration"], "public_parameters": p["public_parameters"],
            "effective_environment": p["effective_environment"], "dependencies": p["dependencies"],
            "redis_binding": original.redis_binding(), "policy": POLICY,
            "review_plan_sha256": bench.sha256(original.PREFLIGHT / "review-plan.json"),
            "authorization_maximum_total": MAX_REQUESTS, "additional_allowance": 0,
        }
        if p != self._original() or prefix != (self.root / "ledger/journal.jsonl").read_bytes():
            base._fail("continuation_prepare_context_changed")
        base._write(self.root, MANIFEST, value)
        self.ledger(p, manifest=value).status()
        return value

    def verify(self):
        m = _read(self.root, MANIFEST)
        bench._exact(m, MANIFEST_FIELDS, "continuation_manifest_fields")
        if (m.get("schema_version") != SCHEMA or m.get("root_binding_sha256") != base._root_hash(self.root)
                or m.get("policy") != POLICY or m.get("git_head") != original.git_head()
                or m.get("source_inventory") != preparation.inventory()
                or m.get("remaining_jobs") != job_specs()[7:] or m.get("plan") != fixed_plan()
                or m.get("redis_binding") != original.redis_binding()
                or m.get("review_plan_sha256") != bench.sha256(original.PREFLIGHT / "review-plan.json")):
            base._fail("continuation_manifest_or_version_changed")
        p = self._original()
        for name, digest in m["original_files"].items():
            if bench.sha256(base._path(self.root, name)) != digest:
                base._fail("continuation_old_evidence_changed")
        if (m["configuration"] != p["configuration"] or m["public_parameters"] != p["public_parameters"]
                or m["effective_environment"] != p["effective_environment"] or m["dependencies"] != p["dependencies"]
                or m["budget"]["original_protocol_sha256"] != ORIGINAL_PROTOCOL_SHA
                or m["budget"]["already_spent"] != 20 or m["budget"]["max_requests"] != MAX_REQUESTS
                or m["authorization_maximum_total"] != MAX_REQUESTS or m["additional_allowance"] != 0):
            base._fail("continuation_scope_changed")
        self.ledger(p, manifest=m).status()
        self._sealed(p)
        return p, m

    def _sealed(self, p):
        legacy = original.CoverageExecutionTrial(self.root)
        legacy.ledger = lambda ignored=None: self.ledger(p)
        return legacy._sealed(p)

    def authorize(self, reference, *, confirmed=False):
        p, m = self.verify()
        if confirmed is not True or type(reference) is not str or not 1 <= len(reference) <= 500:
            base._fail("continuation_explicit_approval_required")
        if self.ledger(p).status()["attempts"] != 20 or (self.root / STARTED).exists():
            base._fail("continuation_approval_before_start_only")
        value = {"schema_version": SCHEMA + "-authorization", "recorded_at": base._now(),
            "reference": reference, "continuation_sha256": bench.sha256(self.root / MANIFEST),
            "original_authorization_sha256": bench.sha256(self.root / "authorization.json"),
            "prefix_sha256": m["budget"]["prefix_sha256"], "original_genesis": self.ledger(p).initial_chain_hash,
            "max_total_requests": MAX_REQUESTS, "already_spent": 20, "max_additional_requests": 460}
        base._write(self.root, APPROVAL, value)
        return value

    def _approval(self, p, m):
        a = _read(self.root, APPROVAL)
        bench._exact(a, APPROVAL_FIELDS, "continuation_authorization_fields")
        bench._text(a["reference"], 500)
        if (a.get("schema_version") != SCHEMA + "-authorization"
                or a.get("continuation_sha256") != bench.sha256(self.root / MANIFEST)
                or a.get("original_authorization_sha256") != bench.sha256(self.root / "authorization.json")
                or a.get("prefix_sha256") != m["budget"]["prefix_sha256"]
                or a.get("original_genesis") != self.ledger(p).initial_chain_hash
                or (a.get("max_total_requests"), a.get("already_spent"), a.get("max_additional_requests")) != (480, 20, 460)):
            base._fail("continuation_approval_changed")
        return a

    def _unstarted(self, p):
        s = self.ledger(p).status()
        if (s["attempts"] != 20 or s.get("continuation_started") or
                any((self.root / name).exists() for name in (STARTED, STOP, "continuation-v1-stop.json", "supervisor-stop.json"))):
            base._fail("continuation_no_auto_restart")

    def _failure_artifact(self, spec, error_code, p, *, origin, evidence_kind="live-model"):
        ops = [row for row in self.ledger(p).results()["operations"].values() if row["case_id"] == spec["case_id"]]
        seen = {op["turn_id"] for op in ops}
        return {"schema_version": "serveflow-coverage-continuation-case-failure-v1",
            "case_id": spec["case_id"], "frame_case_id": spec["case_id"], "source_case_id": spec["source_case_id"],
            "logical_case_id": spec["source_case_id"], "suite": spec["suite"], "arm": spec["arm"],
            "protocol_sha256": ORIGINAL_PROTOCOL_SHA, "continuation_sha256": bench.sha256(self.root / MANIFEST),
            "data_proof_sha256": bench.sha256(self.root / "data-proof.json"), "trial_kind": original.KIND,
            "evidence_kind": evidence_kind, "execution_status": "failed", "quality_pass_claim": False,
            "continuation_failure": {"error_code": error_code, "reason": origin,
                "unknown_upstream_result": True, "abandoned_user_turns": 5 - len(seen) if spec["suite"] == "conversations" else 0},
            "partial_attempts": [{k: op[k] for k in ("attempt", "operation_id", "turn_id", "phase", "agent", "outcome", "error_code") if k in op} for op in ops],
            "semantic_review": None, "judge": None}

    async def execute(self):
        from anthropic import AsyncAnthropic, APIConnectionError
        import httpx
        from redis.asyncio import Redis
        from evaluation.coverage_continuation_transport import CoverageContinuationTransport, CaseExecutionFailure
        from evaluation.coverage_runtime import build_data
        p, m = self.verify()
        self._approval(p, m)
        self._unstarted(p)
        ledger = self.ledger(p)
        data = redis = sdk = http = transport = None
        faults = []
        with ledger.exclusive_run() as lease:
            try:
                base._write(self.root, STARTED, {"continuation_sha256": bench.sha256(self.root / MANIFEST), "recorded_at": base._now()})
                config = original.configuration(secret=True)
                data = await build_data()
                if data.proof != _read(self.root, "data-proof.json"):
                    base._fail("continuation_source_data_proof_changed")
                base._write(self.root, "continuation-v1-data-proof.json", data.proof)
                redis = Redis.from_url(os.environ["SERVEFLOW_EVAL_REDIS_URL"], socket_connect_timeout=3, socket_timeout=3, decode_responses=True)
                await original.verify_redis(redis, m["redis_binding"])
                data.redis = redis
                approval_hash = bench.sha256(self.root / APPROVAL)

                def healthy():
                    try:
                        if faults or transport is not None and transport.fault or (self.root / "continuation-v1-stop.json").exists():
                            base._fail("continuation_global_fault")
                        self.verify()
                        self._approval(p, m)
                        if bench.sha256(self.root / APPROVAL) != approval_hash or data.proof != _read(self.root, "data-proof.json"):
                            base._fail("continuation_run_binding_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise

                def job_healthy():
                    transport.assert_case_healthy()
                    healthy()

                async def redis_guard():
                    await original.verify_redis(redis, m["redis_binding"])

                transport = CoverageContinuationTransport(config["base_url"], self.root, ledger, lease, healthy,
                    continuation_sha256=bench.sha256(self.root / MANIFEST), redis_guard=redis_guard)
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False,
                    trust_env=False, headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config["api_key"], base_url=config["base_url"], max_retries=0,
                    timeout=SDK_TIMEOUT_S, http_client=http)
                original_failure = self._failure_artifact(job_specs()[6], "timeout", p,
                    origin="original attempt20 retained, never resent", evidence_kind=transport.kind)
                base._write(self.root, "case-D-CT-D04.json", original_failure)
                ledger.resume_after_timeout(bench.sha256(self.root / "case-D-CT-D04.json"), lease=lease)
                for spec in m["remaining_jobs"]:
                    healthy()
                    await redis_guard()
                    ledger.start_case(spec["case_id"], lease=lease)
                    transport.begin_case(spec["case_id"])
                    artifact = error = None
                    try:
                        with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                            artifact = await original.CoverageExecutionTrial(self.root)._job(spec, p["cases"][spec["source_case_id"]], sdk, data, job_healthy)
                    except BaseException as ex:
                        error = ex
                    failure = await transport.finish_case()
                    healthy()
                    if failure:
                        if (faults or transport.fault or error is not None and not isinstance(error, (CaseExecutionFailure, TrialTransportError, ModelDeadlineExceeded, APIConnectionError))):
                            if error is not None:
                                raise error
                            base._fail("continuation_unclassified_case_fault")
                        artifact = self._failure_artifact(spec, failure["error_code"], p,
                            origin="isolated sent execution failure; no retry", evidence_kind=transport.kind)
                    elif error is not None:
                        raise error
                    else:
                        artifact.update(frame_case_id=spec["case_id"], trial_kind=original.KIND,
                            protocol_sha256=ORIGINAL_PROTOCOL_SHA, data_proof_sha256=bench.sha256(self.root / "data-proof.json"),
                            continuation_sha256=bench.sha256(self.root / MANIFEST), evidence_kind=transport.kind, quality_pass_claim=False)
                    name = f"case-{spec['case_id']}.json"
                    base._write(self.root, name, artifact)
                    if failure:
                        ledger.recover_case_failure(spec["case_id"], bench.sha256(self.root / name), lease=lease)
                    else:
                        outcome = artifact.get("execution_status", artifact.get("status"))
                        ledger.finish_case(spec["case_id"], bench.sha256(self.root / name), outcome=outcome, lease=lease)
                    if ledger.status()["paused"]:
                        base._fail("continuation_consecutive_execution_errors")
            except BaseException:
                if not (self.root / STOP).exists():
                    base._write(self.root, STOP, {"reason": "stopped_no_auto_restart", "recorded_at": base._now(),
                        "continuation_sha256": bench.sha256(self.root / MANIFEST)})
                raise
            finally:
                errors = []
                closers = [sdk.close] if sdk is not None else [http.aclose] if http is not None else []
                closers += ([redis.aclose] if redis is not None else []) + ([data.close] if data is not None else [])
                for close in closers:
                    try:
                        await task_data._owned(close())
                    except BaseException as ex:
                        errors.append(ex)
                if errors:
                    if not (self.root / STOP).exists():
                        base._write(self.root, STOP, {"reason": "cleanup_failed", "recorded_at": base._now()})
                    raise errors[0]
                base._write(self.root, "continuation-v1-cleanup.json", {"owned_resources_closed": True,
                    "continuation_sha256": bench.sha256(self.root / MANIFEST)})
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("continuation_fixed_worker_only")
        p, m = self.verify()
        self._approval(p, m)
        self._unstarted(p)
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.coverage_continuation", "_worker"], root=ROOT, deadline_s=3600)

    def report(self):
        from evaluation.coverage_continuation_report import build_report
        p, m = self.verify()
        return build_report(p, self.ledger(p).status(), self._sealed(p), root=self.root,
            authorized_requests=MAX_REQUESTS, continuation=m)

    def review(self, value):
        from evaluation.coverage_continuation_report import validate_review
        p, _ = self.verify()
        validate_review(value, p, root=self.root)
        if value["frame_case_id"] not in self._sealed(p)["cases"]:
            base._fail("continuation_review_unsealed")
        base._write(self.root, f"review-{value['frame_case_id']}.json", value)
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "authorize", "run", "_worker", "report", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-explicit-continuation", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args, trial = parser.parse_args(argv), CoverageContinuation()
    try:
        if args.command == "prepare":
            value = asyncio.run(trial.prepare())
        elif args.command == "authorize":
            value = trial.authorize(args.reference, confirmed=args.confirm_explicit_continuation)
        elif args.command == "run":
            value = trial.run()
            if value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled"):
                base._fail("continuation_supervised_worker_failed")
        elif args.command == "_worker":
            value = asyncio.run(trial.execute())
        elif args.command == "review":
            value = trial.review(bench.read_json(args.review_file))
        else:
            value = trial.report() if args.command == "report" else trial.verify()[1]
        print(json.dumps(value if args.command == "report" else {"result": args.command}, ensure_ascii=False))
        return 0
    except BaseException as ex:
        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({"error": "coverage_continuation_stopped_inspect_private_evidence", "no_auto_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
