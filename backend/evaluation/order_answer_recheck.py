"""Prepare a fixed six-case recheck of bb19's bounded order explanations.

No model/root/URL overrides and no automatic authorization. Old trials remain
immutable. The cases are all seen regressions, not A/D or generalization data.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from evaluation import diagnostic_repair_trial as previous
from evaluation import order_answer_recheck_proof as proof


base, bench = previous.base, previous.bench
ROOT = proof.ROOT
cases, fixed_plan = previous.cases, previous.fixed_plan
MAX_REQUESTS = 24
MATHEMATICAL_REQUESTS = previous.MAX_REQUESTS
TRIAL_KIND = "order-answer-recheck-v1"
POLICY = {**previous.POLICY,
    "scope": "same six once-only production-D single turns; every case is now a seen regression; no A/D or RAG gain, held-out or generalization claim",
    "development_exposure": "all six previously run and used for development; bb19 adds finite order-question explanations; previous strict2/6 and tool6/6 remain sealed",
    "budget": "trial hard limit24 and each lower explicit authorization; phase envelope6x23=138 does not authorize or increase this cap; exhaustion preserves incomplete cases, no retry or top-up",
    "business_version": "bb19 business, prompts, schemas, sources and parameters unchanged; only a closed new runner/proof/test set and worker whitelist added",
    "artifact_contract": "existing diagnostic-repair-artifact-v1 with a required new trial_kind and current protocol/data seals",
    "acceptance": "all six must pass all seven frozen review dimensions with no blocker, unknown, pending or unexecuted item; native/tool success cannot replace complete direct answers; equivalent wording allowed, explicit date gaps and payment/fulfilment/permission independence required; correctly explaining an unknown does not by itself force partial",
}


async def execute_case(case, sdk, data, healthy):
    artifact = await previous.execute_case(case, sdk, data, healthy)
    artifact["trial_kind"] = TRIAL_KIND
    return artifact


class OrderAnswerRecheck(previous.FocusedTrial):
    protocol_schema = "serveflow-order-answer-recheck-protocol-v1"
    authorization_schema = "serveflow-order-answer-recheck-authorization-v1"

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT/"data/benchmarks"):
            base._fail("order_answer_recheck_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        if type(send_limit) is not int or not 1 <= send_limit <= MAX_REQUESTS:
            base._fail("order_answer_recheck_limit_invalid")
        return super().ledger(p, initialize=initialize, send_limit=send_limit)

    def expected(self):
        config = base._configuration()
        return {"schema_version": self.protocol_schema, "root_binding_sha256": base._root_hash(self.root),
            "source_proof": proof.source_proof(), "artifacts": proof.artifacts(), "prior_proof": proof.prior_proof(),
            "cases": cases(), "plan": fixed_plan(), "dependencies": previous.engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": previous.messages_url(config["base_url"]),
            "phase_parameters": {k:v for k,v in previous.PHASE_PARAMETERS.items() if k != "judge"},
            "agent_parameters": previous.AGENT_PARAMETERS,
            "limits": {**previous.LIMITS, "physical_requests": MAX_REQUESTS, "mathematical_phase_envelope": MATHEMATICAL_REQUESTS},
            "http_limits": previous.PHASE_HTTP_LIMITS, "policy": POLICY,
            "offline_data_proof": previous.task_data.offline_data_proof(),
            "effective_environment": previous.history.effective_environment(), "authorized_model_requests": 0,
            "proposed_model_requests": MAX_REQUESTS, "judge_requests": 0, "external_business_requests": 0}

    def authorize(self, reference, *, max_requests, confirmed=False):
        if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
            base._fail("order_answer_recheck_limit_invalid")
        return super().authorize(reference, max_requests=max_requests, confirmed=confirmed)

    def _approval(self, p):
        approval = super()._approval(p)
        if approval["max_requests"] > MAX_REQUESTS:
            base._fail("order_answer_recheck_limit_invalid")
        return approval

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p = self.verify(); approval = self._approval(p)
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("order_answer_recheck_controlled_pair_required")
        if delegate_factory is None and self.root != ROOT:
            base._fail("order_answer_recheck_fixed_live_root_required")
        if any((self.root/n).exists() for n in previous.STOP_FILES):
            base._fail("order_answer_recheck_paused")
        if self.ledger(p).status()["completed_cases"] == [c["case_id"] for c in p["plan"]]:
            return self.report()
        if (self.root/"data-proof.json").exists():
            base._fail("order_answer_recheck_previous_run_requires_reconciliation")
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        data, sdk, transport = controlled_data, None, None
        approval_hash = bench.sha256(self.root/"authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                config = base._configuration(secret=delegate_factory is None)
                if data is None:
                    data = await previous.task_data.build_live_data(previous.engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("order_answer_recheck_controlled_data_required")
                if controlled_data is None:
                    previous._binding(data.proof, p["offline_data_proof"])
                data.assert_healthy()
                base._write(self.root, "data-proof.json", data.proof)
                proof_hash = bench.sha256(self.root/"data-proof.json")
                base._write(self.root, "data-proof-seal.json", {"protocol_sha256": bench.sha256(self.root/"protocol.json"), "data_proof_sha256": proof_hash})
                def healthy():
                    if transport is not None and transport.fault:
                        base._fail("order_answer_recheck_transport_stopped")
                    self.verify(); self._approval(p)
                    if bench.sha256(self.root/"authorization.json") != approval_hash:
                        base._fail("order_answer_recheck_run_identity_changed")
                    if bench.sha256(self.root/"data-proof.json") != proof_hash:
                        base._fail("order_answer_recheck_data_changed")
                    data.assert_healthy()
                transport = previous.FocusedTransport(config["base_url"], self.root, ledger, lease, healthy, delegate_factory=delegate_factory)
                import httpx
                from anthropic import AsyncAnthropic
                http = httpx.AsyncClient(transport=transport, timeout=previous.SDK_TIMEOUT_S,
                    follow_redirects=False, trust_env=False, headers={"Accept-Encoding":"identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                    max_retries=0, timeout=previous.SDK_TIMEOUT_S, http_client=http)
                for case in p["cases"]:
                    key = "D-"+case["id"]
                    healthy(); ledger.start_case(key, lease=lease)
                    with base._quiet_clients(), previous.eval_deadlines(previous.EvaluationDeadlinePolicy()):
                        artifact = await execute_case(case, sdk, data, healthy)
                    healthy()
                    artifact.update(logical_case_id=key, protocol_sha256=bench.sha256(self.root/"protocol.json"),
                                    evidence_kind=transport.kind, data_proof_sha256=proof_hash)
                    base._write(self.root, f"case-{key}.json", artifact)
                    ledger.finish_case(key, bench.sha256(self.root/f"case-{key}.json"), outcome=artifact["execution_status"], lease=lease)
            except BaseException:
                if not (self.root/"worker-stop.json").exists():
                    base._write(self.root, "worker-stop.json", {"reason":"execution_stopped_no_auto_resume", "recorded_at":base._now()})
                raise
            finally:
                try:
                    try:
                        if sdk is not None: await previous.task_data._owned(sdk.close())
                    finally:
                        if data is not None: await previous.task_data._owned(data.close())
                    if data is not None:
                        base._write(self.root, "cleanup-complete.json", {"owned_resources_closed":True, "protocol_sha256":bench.sha256(self.root/"protocol.json")})
                except BaseException:
                    if not (self.root/"worker-stop.json").exists():
                        base._write(self.root, "worker-stop.json", {"reason":"cleanup_failed_no_auto_resume", "recorded_at":base._now()})
                    raise
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("order_answer_recheck_fixed_worker_required")
        p = self.verify(); self._approval(p)
        if any((self.root/n).exists() for n in previous.STOP_FILES):
            base._fail("order_answer_recheck_paused")
        if self.ledger(p).status()["completed_cases"] == [c["case_id"] for c in p["plan"]]:
            return {"exit_code":0, "already_complete":True}
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.order_answer_recheck", "_worker"], root=ROOT, deadline_s=900)

    def report(self):
        result = super().report()
        for key in self._sealed(self.verify())["cases"]:
            if bench.read_json(self.root/f"case-{key}.json").get("trial_kind") != TRIAL_KIND:
                base._fail("order_answer_recheck_artifact_identity_changed")
        result.update(schema_version="serveflow-order-answer-recheck-progress-v1", trial_kind=TRIAL_KIND,
            candidate_baseline=proof.BASELINE, trial_request_limit=MAX_REQUESTS,
            all_cases_previously_exposed=True,
            note="all six are seen regressions; original group labels describe historical authorship only; no A/D, RAG or generalization gain; full denominator retained")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","verify","status","report","authorize","run","_worker","review"))
    parser.add_argument("--reference"); parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-explicit-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(); trial = OrderAnswerRecheck()
    try:
        if args.command == "prepare": value = asyncio.run(trial.prepare())
        elif args.command == "authorize": value = trial.authorize(args.reference, max_requests=args.max_requests, confirmed=args.confirm_explicit_authorization)
        elif args.command == "run": raise SystemExit(0 if trial.run()["exit_code"] == 0 else 2)
        elif args.command == "_worker": value = asyncio.run(trial.execute())
        elif args.command == "review": value = trial.review(bench.read_json(args.review_file))
        elif args.command in {"status","report"}: value = trial.report()
        else: value = trial.verify()
        print(json.dumps({"result":args.command,"reserved_requests":value.get("reserved_requests",0),
            "proposed_upper_bound":MAX_REQUESTS,"authorized_by_prepare":0,"completed_jobs":value.get("completed_jobs",0)}))
    except Exception:
        parser.exit(2,"Order-answer recheck stopped; inspect private evidence; no overwrite, retry or reused allowance.\n")


if __name__ == "__main__": main()
