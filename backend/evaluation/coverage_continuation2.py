"""Fixed suffix after attempt201, preserving every prior claim and response.

Only the evaluation fault-provenance adapter changes. Business source, inputs,
model payloads, deadlines and scoring stay fixed. This is not a new allowance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, intent_trial as base, task_data
from evaluation import coverage_execution as initial, coverage_expansion as preparation
from evaluation import coverage_continuation as prior
from evaluation.coverage_budget import ROOT, MAX_REQUESTS, fixed_plan, job_specs
from evaluation.phase_deadline_transport import PHASE_HTTP_LIMITS, SDK_TIMEOUT_S


SCHEMA = "serveflow-coverage-continuation-v2"
MANIFEST = "continuation-v2.json"
APPROVAL = "continuation-v2-authorization.json"
STARTED = "continuation-v2-worker-started.json"
STOP = "continuation-v2-worker-stop.json"
ORIGINAL_V1_SHA = "ec20f29b597ca394cd24b32667df0aff717ac6d62e607d759be3259c31f2661c"
POLICY = {**prior.POLICY,
    "grant": "same480; already201; at most279 more; no reset or new allowance",
    "plan": "original136/168; original69 finished positions unchanged; A-CT-V14 fails without resend; remaining66 positions once",
    "provenance": "a wrapped tool_handler_failed is recoverable only with typed original model-execution cause, same sent failed case, intact data/version/accounting and no other capture faults",
}
NEW_PATHS = frozenset({f"evaluation/coverage_continuation2{s}.py" for s in ("", "_budget", "_transport")}
    | {f"tests/test_coverage_continuation2{s}.py" for s in ("", "_budget", "_transport")})


class CoverageContinuation2(prior.CoverageContinuation):
    def ledger(self, p=None, *, manifest=None):
        from evaluation.coverage_continuation2_budget import Continuation2Budget
        return Continuation2Budget(self.root / "ledger", prior.ORIGINAL_PROTOCOL_SHA, fixed_plan(),
            continuation2_sha256=bench.sha256(self.root / MANIFEST))

    def _base(self):
        p = prior._read(self.root, "protocol.json")
        old = prior._read(self.root, prior.MANIFEST)
        if (bench.sha256(self.root / "protocol.json") != prior.ORIGINAL_PROTOCOL_SHA
                or bench.sha256(self.root / prior.MANIFEST) != ORIGINAL_V1_SHA
                or p["cases"] != initial.source_cases() or p["plan"] != fixed_plan() or p["jobs"] != job_specs()
                or p["configuration"] != initial.configuration()
                or p["public_parameters"] != json.loads(json.dumps(preparation.public_parameters()))
                or p["effective_environment"] != initial.history.effective_environment()
                or p["dependencies"] != initial.engine._dependencies()
                or p["preparation"] != initial.frozen_preparation()
                or p["prior_evidence"] != initial.preserved_evidence() or p["http_limits"] != PHASE_HTTP_LIMITS):
            base._fail("continuation2_original_context_changed")
        current = preparation.inventory()
        changed = {n for n, h in old["source_inventory"].items() if current.get(n) != h}
        if not changed <= {"evaluation/task_supervisor.py"} or not (set(current) - set(old["source_inventory"])) <= NEW_PATHS:
            base._fail("continuation2_business_or_unrelated_source_changed")
        legacy = initial.CoverageExecutionTrial(self.root)
        legacy.ledger = lambda ignored=None: (self.ledger(p) if (self.root / MANIFEST).exists()
            else prior.CoverageContinuation(self.root).ledger(p))
        if legacy._approval(p)["max_requests"] != MAX_REQUESTS:
            base._fail("continuation2_original_grant_changed")
        for name, digest in old["original_files"].items():
            if bench.sha256(base._path(self.root, name)) != digest:
                base._fail("continuation2_historical_evidence_changed")
        return p, old, current

    async def prepare(self):
        if self.root != ROOT or (self.root / MANIFEST).exists():
            base._fail("continuation2_fixed_once_root")
        p, old, current = self._base()
        ledger = prior.CoverageContinuation(self.root).ledger(p)
        state = ledger.status()
        prefix = (self.root / "ledger/journal.jsonl").read_bytes()
        if (state["attempts"] != 201 or len(prefix.splitlines()) != 541 or len(state["completed_cases"]) != 69
                or state["completed_cases"] != [r["case_id"] for r in job_specs()[:69]]
                or state["active_case"] != "A-CT-V14" or not state["paused"] or state["pending"]
                or (self.root / "continuation-v1-stop.json").exists()):
            base._fail("continuation2_exact201_prefix_required")
        latest = prior._read(self.root, "physical-201.json")
        if latest["error_code"] != "timeout" or latest["case_id"] != "A-CT-V14" or not latest["delegate_entered"]:
            base._fail("continuation2_expected_execution_failure_required")
        for spec in job_specs()[69:]:
            if (self.root / f"case-{spec['case_id']}.json").exists():
                base._fail("continuation2_unexpected_completed_result")
        capture = await initial.capture_first_requests(p["configuration"], p["cases"])
        if any(capture[k] != p[k] for k in capture):
            base._fail("continuation2_payload_changed")
        m = {"schema_version": SCHEMA, "created_at": base._now(), "git_head": initial.git_head(),
            "root_binding_sha256": base._root_hash(self.root), "original_continuation_sha256": ORIGINAL_V1_SHA,
            "budget": {"original_protocol_sha256": prior.ORIGINAL_PROTOCOL_SHA, "prefix_size": len(prefix),
                "prefix_sha256": prior._hash(prefix), "prefix_chain_hash": state["chain_hash"], "prefix_events": 541,
                "already_spent": 201, "max_requests": 480},
            "previous_active_case": "A-CT-V14", "remaining_jobs": job_specs()[70:], "plan": fixed_plan(),
            "original_files": prior._files(self.root), "source_inventory": current, "policy": POLICY,
            "redis_binding": initial.redis_binding(), "configuration": p["configuration"],
            "public_parameters": p["public_parameters"], "dependencies": p["dependencies"],
            "effective_environment": p["effective_environment"], "review_plan_sha256": old["review_plan_sha256"]}
        if prefix != (self.root / "ledger/journal.jsonl").read_bytes() or current != preparation.inventory():
            base._fail("continuation2_prepare_changed")
        base._write(self.root, MANIFEST, m)
        self.ledger(p).status()
        return m

    def verify(self):
        m = prior._read(self.root, MANIFEST)
        p, old, current = self._base()
        if (m.get("schema_version") != SCHEMA or m.get("git_head") != initial.git_head()
                or m.get("root_binding_sha256") != base._root_hash(self.root)
                or m.get("source_inventory") != current or m.get("policy") != POLICY
                or m.get("remaining_jobs") != job_specs()[70:] or m.get("plan") != fixed_plan()
                or m.get("redis_binding") != initial.redis_binding()
                or m.get("original_continuation_sha256") != ORIGINAL_V1_SHA
                or m.get("configuration") != p["configuration"] or m.get("public_parameters") != p["public_parameters"]
                or m.get("dependencies") != p["dependencies"] or m.get("effective_environment") != p["effective_environment"]
                or m.get("review_plan_sha256") != bench.sha256(initial.PREFLIGHT / "review-plan.json")):
            base._fail("continuation2_manifest_changed")
        for name, digest in m["original_files"].items():
            if bench.sha256(base._path(self.root, name)) != digest:
                base._fail("continuation2_original_bytes_changed")
        self.ledger(p).status()
        self._sealed(p)
        return p, m

    def authorize(self, reference, *, confirmed=False):
        p, m = self.verify()
        if confirmed is not True or type(reference) is not str or not 1 <= len(reference) <= 500:
            base._fail("continuation2_explicit_instruction_required")
        self._unstarted(p)
        a = {"schema_version": SCHEMA + "-authorization", "reference": reference, "recorded_at": base._now(),
            "continuation_sha256": bench.sha256(self.root / MANIFEST), "original_authorization_sha256": bench.sha256(self.root / "authorization.json"),
            "prior_authorization_sha256": bench.sha256(self.root / prior.APPROVAL), "original_genesis": self.ledger(p).initial_chain_hash,
            "max_total_requests": 480, "already_spent": 201, "max_additional_requests": 279}
        base._write(self.root, APPROVAL, a)
        return a

    def _approval(self, p, m):
        a = prior._read(self.root, APPROVAL)
        if (a.get("schema_version") != SCHEMA + "-authorization" or a.get("continuation_sha256") != bench.sha256(self.root / MANIFEST)
                or a.get("original_authorization_sha256") != bench.sha256(self.root / "authorization.json")
                or a.get("prior_authorization_sha256") != bench.sha256(self.root / prior.APPROVAL)
                or a.get("original_genesis") != self.ledger(p).initial_chain_hash
                or (a.get("max_total_requests"), a.get("already_spent"), a.get("max_additional_requests")) != (480, 201, 279)):
            base._fail("continuation2_approval_changed")
        return a

    def _unstarted(self, p):
        s = self.ledger(p).status()
        if s["attempts"] != 201 or any((self.root / n).exists() for n in (STARTED, STOP, "continuation-v2-stop.json", "supervisor-stop.json")):
            base._fail("continuation2_no_auto_restart")

    async def execute(self):
        from anthropic import AsyncAnthropic
        import httpx
        from redis.asyncio import Redis
        from evaluation.coverage_runtime import build_data
        from evaluation.coverage_continuation2_transport import (CoverageContinuation2Transport,
            observe_case_errors, is_recoverable_case_error)
        p, m = self.verify()
        self._approval(p, m)
        self._unstarted(p)
        ledger = self.ledger(p)
        data = redis = sdk = http = transport = None
        faults = []
        with ledger.exclusive_run() as lease:
            try:
                base._write(self.root, STARTED, {"continuation_sha256": bench.sha256(self.root / MANIFEST), "recorded_at": base._now()})
                config = initial.configuration(secret=True)
                data = await build_data()
                if data.proof != prior._read(self.root, "data-proof.json"):
                    base._fail("continuation2_data_proof_changed")
                base._write(self.root, "continuation-v2-data-proof.json", data.proof)
                redis = Redis.from_url(os.environ["SERVEFLOW_EVAL_REDIS_URL"], socket_connect_timeout=3, socket_timeout=3, decode_responses=True)
                await initial.verify_redis(redis, m["redis_binding"])
                data.redis = redis
                approval_hash = bench.sha256(self.root / APPROVAL)

                def healthy():
                    try:
                        if faults or transport is not None and transport.fault or (self.root / "continuation-v2-stop.json").exists():
                            base._fail("continuation2_global_fault")
                        self.verify()
                        self._approval(p, m)
                        if approval_hash != bench.sha256(self.root / APPROVAL) or data.proof != prior._read(self.root, "data-proof.json"):
                            base._fail("continuation2_live_identity_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise

                def job_healthy():
                    transport.assert_case_healthy()
                    healthy()

                async def redis_guard():
                    await initial.verify_redis(redis, m["redis_binding"])

                transport = CoverageContinuation2Transport(config["base_url"], self.root, ledger, lease, healthy,
                    continuation_sha256=bench.sha256(self.root / MANIFEST), redis_guard=redis_guard)
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False, trust_env=False,
                    headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config["api_key"], base_url=config["base_url"], max_retries=0,
                    timeout=SDK_TIMEOUT_S, http_client=http)
                first = self._failure_artifact(job_specs()[69], "timeout", p,
                    origin="original attempt201 retained, never resent", evidence_kind=transport.kind)
                first["continuation_sha256"] = bench.sha256(self.root / MANIFEST)
                base._write(self.root, "case-A-CT-V14.json", first)
                ledger.recover_case_failure("A-CT-V14", bench.sha256(self.root / "case-A-CT-V14.json"), lease=lease)
                for spec in m["remaining_jobs"]:
                    healthy()
                    await redis_guard()
                    ledger.start_case(spec["case_id"], lease=lease)
                    transport.begin_case(spec["case_id"])
                    artifact = error = None
                    with observe_case_errors(transport) as observer:
                        try:
                            with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                                artifact = await initial.CoverageExecutionTrial(self.root)._job(spec,
                                    p["cases"][spec["source_case_id"]], sdk, data, job_healthy)
                        except BaseException as ex:
                            error = ex
                        failure = await transport.finish_case()
                    healthy()
                    if failure:
                        if faults or transport.fault or not is_recoverable_case_error(error, transport, observer):
                            if error is not None:
                                raise error
                            base._fail("continuation2_unclassified_failure")
                        artifact = self._failure_artifact(spec, failure["error_code"], p,
                            origin="isolated sent failure; original typed cause verified; no retry",
                            evidence_kind=transport.kind)
                        artifact["failure_provenance"] = observer.summary()
                    elif error is not None:
                        raise error
                    else:
                        artifact.update(frame_case_id=spec["case_id"], trial_kind=initial.KIND,
                            protocol_sha256=prior.ORIGINAL_PROTOCOL_SHA,
                            data_proof_sha256=bench.sha256(self.root / "data-proof.json"),
                            evidence_kind=transport.kind, quality_pass_claim=False)
                    artifact["continuation_sha256"] = bench.sha256(self.root / MANIFEST)
                    name = f"case-{spec['case_id']}.json"
                    base._write(self.root, name, artifact)
                    if failure:
                        ledger.recover_case_failure(spec["case_id"], bench.sha256(self.root / name), lease=lease)
                    else:
                        ledger.finish_case(spec["case_id"], bench.sha256(self.root / name),
                            outcome=artifact.get("execution_status", artifact.get("status")), lease=lease)
                    if ledger.status()["paused"]:
                        base._fail("continuation2_three_consecutive_faults")
            except BaseException as ex:
                if not (self.root / STOP).exists():
                    base._write(self.root, STOP, {"reason": "stopped_no_auto_restart", "error_type": type(ex).__name__,
                        "recorded_at": base._now(), "continuation_sha256": bench.sha256(self.root / MANIFEST)})
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
                    raise errors[0]
                base._write(self.root, "continuation-v2-cleanup.json", {"owned_resources_closed": True,
                    "continuation_sha256": bench.sha256(self.root / MANIFEST)})
        return self.report()

    def _failure_artifact(self, *args, **kwargs):
        # Keep the original data/protocol identity, replacing only the explicit
        # execution receipt reference for this newly frozen suffix.
        value = super()._failure_artifact(*args, **kwargs)
        value["continuation_sha256"] = bench.sha256(self.root / MANIFEST)
        return value

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("continuation2_fixed_worker_only")
        p, m = self.verify()
        self._approval(p, m)
        self._unstarted(p)
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.coverage_continuation2", "_worker"], root=ROOT, deadline_s=3600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "authorize", "run", "_worker", "report", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-explicit-continuation", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args, trial = parser.parse_args(argv), CoverageContinuation2()
    try:
        if args.command == "prepare": value = asyncio.run(trial.prepare())
        elif args.command == "authorize": value = trial.authorize(args.reference, confirmed=args.confirm_explicit_continuation)
        elif args.command == "run":
            value = trial.run()
            if value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled"):
                base._fail("continuation2_supervised_worker_failed")
        elif args.command == "_worker": value = asyncio.run(trial.execute())
        elif args.command == "review": value = trial.review(bench.read_json(args.review_file))
        else: value = trial.report() if args.command == "report" else trial.verify()[1]
        print(json.dumps(value if args.command == "report" else {"result": args.command}, ensure_ascii=False))
        return 0
    except Exception as ex:
        print(json.dumps({"error": "coverage_continuation2_stopped", "error_type": type(ex).__name__, "no_auto_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
