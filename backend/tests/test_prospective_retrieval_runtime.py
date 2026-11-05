"""Controlled production RAG execution only; no model/network/index downloads."""
import asyncio
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from evaluation import prospective_retrieval_runtime as runtime
from evaluation import benchmark as bench, intent_trial as base, task_data
from tooling.retrieval_grounding import query_units


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("prospective_retrieval_tests_forbid_network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def cases():
    return bench.read_json(bench.DATASET_ROOT / "prospective-retrieval-v1.json")["cases"]


def rows():
    return [{"document_id": hashlib.sha256(json.dumps(["title", doc["title"]], ensure_ascii=True).encode()).hexdigest(),
             "title": doc["title"], "content": doc["content"], "chunk": 0, "score": .5,
             "version": hashlib.sha256(doc["content"].encode()).hexdigest()}
            for doc in task_data._policies()]


def fixture(*, rewrite=None, rerank='[5,1,0,2,3,4]',
            empty=False, delay_original=False, damage=None, failure=None):
    source = rows()
    searches, sends, factory_calls, closed = [], [], [], []
    unhealthy = []

    async def search(params, context):
        searches.append(deepcopy(params))
        if empty:
            return []
        first_unit = query_units(cases()[0]["input"]["message"])[0]
        if delay_original and params["top_k"] == 5 and params["query"] != first_unit:
            await asyncio.sleep(.01)
        answer = source[5:] if params["query"] == first_unit else source[:params["top_k"]]
        answer = deepcopy(answer)
        if damage == "content":
            answer[0]["content"] = "private-invalid-source-must-not-be-accepted"
        if damage == "duplicate":
            answer = [source[0], source[0]]
        return answer

    def client_factory(case_id, stage):
        factory_calls.append((case_id, stage))

        async def create(**kwargs):
            sends.append((case_id, stage, deepcopy(kwargs)))
            if failure == stage:
                unhealthy.append("controlled_transport_failure")
                raise RuntimeError("untrusted-private-error-must-not-become-result")
            if stage == "rewrite":
                data = json.loads(kwargs["messages"][0]["content"].rsplit("\n", 1)[-1])
                text = rewrite if rewrite is not None else json.dumps(data["allowed_units"][:3], ensure_ascii=False)
            else:
                text = rerank
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")

        return SimpleNamespace(base_url="https://dashscope.aliyuncs.com/apps/anthropic",
                               messages=SimpleNamespace(create=create))

    def healthy():
        if unhealthy:
            raise ValueError(unhealthy[0])

    knowledge = SimpleNamespace(kb=SimpleNamespace(search_handler=search),
                                close=lambda: closed.append(True))
    return knowledge, client_factory, healthy, searches, sends, factory_calls, closed


def execute(case=None, *, arm_order=("A", "B"), callback=None, **options):
    knowledge, factory, healthy, searches, sends, factories, closed = fixture(**options)
    result = asyncio.run(runtime.execute_retrieval_case(case or cases()[0],
        knowledge=knowledge, client_factory=factory, ensure_healthy=healthy,
        arm_order=arm_order, on_observation=callback))
    return result, searches, sends, factories, closed


@pytest.mark.parametrize("order", [("A", "B"), ["B", "A"]])
def test_production_methods_preserve_frozen_order_and_only_two_counted_model_stages(order):
    case = deepcopy(cases()[0])
    case["expected"]["rationale"] = "LABEL-MUST-NOT-ENTER-ACTOR-CONTEXT"
    result, searches, sends, factories, closed = execute(case, arm_order=order)
    assert result["arm_order"] == list(order)
    assert [event["arm"] for event in result["events"] if event["event"] == "arm_started"] == list(order)
    assert factories == [("q01", "rewrite"), ("q01", "rerank")]
    assert len(sends) == 2 and len(searches) == 5
    assert sum(row["top_k"] == 3 for row in searches) == 1
    assert sum(row["top_k"] == 5 for row in searches) == 4
    assert not closed  # Collection and injected SDK ownership stays with caller.
    assert result["measurement_kind"] == "runtime-observation-not-live-classified"
    assert result["arms"]["A"]["record"]["usage"]["physical_requests"] == 0
    assert result["arms"]["B"]["record"]["usage"]["physical_requests"] is None
    assert "LABEL-MUST-NOT-ENTER-ACTOR-CONTEXT" not in json.dumps(sends)
    assert sends[0][2]["model"] == sends[1][2]["model"] == base.MODEL
    assert (sends[0][2]["max_tokens"], sends[0][2]["temperature"]) == (512, .3)
    assert (sends[1][2]["max_tokens"], sends[1][2]["temperature"]) == (2048, .0)
    assert sends[0][2]["extra_body"] == sends[1][2]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_actual_full_sources_and_tail_conditions_reach_rerank_without_200_character_window():
    result, _, sends, _, _ = execute()
    method = result["method_observation"]
    assert len(method["merged_sources"]) == 6 and method["rerank_input_count"] == 6
    assert method["rewrite_recall_top3"] == [doc["title"] for doc in rows()[:3]]
    assert result["arms"]["B"]["record"]["ranked_titles"] == ["配送说明", "订单查询", "退款政策"]
    assert all(row["content_sha256"] == hashlib.sha256(row["content"].encode()).hexdigest()
               for row in method["merged_sources"])
    prompt = sends[1][2]["messages"][0]["content"]
    for index, source in enumerate(rows()):
        assert f"{index}. {json.dumps(source, ensure_ascii=False)}" in prompt
    assert "本演示不设全国统一送达天数" in prompt
    assert "本演示不设全国统一送达天数" in method["merged_sources"][5]["content"]


def test_query_and_merge_order_do_not_follow_parallel_completion_order():
    result, _, _, _, _ = execute(delay_original=True)
    method = result["method_observation"]
    assert method["rewrite_queries"] == [cases()[0]["input"]["message"], *query_units(cases()[0]["input"]["message"])]
    assert [row["query"] for row in method["recalls"]] == method["rewrite_queries"]
    assert [row["query_index"] for row in method["recalls"]] == [0, 1, 2, 3]
    assert [row["title"] for row in method["merged_sources"]] == [doc["title"] for doc in rows()]
    assert [event["query_index"] for event in result["events"] if event["event"] == "recall_completed"][0] != 0


def test_duplicate_subqueries_use_existing_production_dedup_without_extra_calls():
    unit = query_units(cases()[0]["input"]["message"])[0]
    result, searches, sends, _, _ = execute(rewrite=json.dumps([unit,unit,unit], ensure_ascii=False))
    assert result["method_observation"]["rewrite_queries"] == [cases()[0]["input"]["message"], unit]
    assert len(searches) == 3 and len(sends) == 2


def test_original_rewrite_fallback_is_observed_not_silently_repaired():
    result, searches, sends, _, _ = execute(rewrite="not json", rerank="[0,1,2,3,4]")
    assert result["method_observation"]["rewrite_queries"] == [cases()[0]["input"]["message"]]
    assert len(searches) == 2 and len(sends) == 2
    assert result["arms"]["B"]["record"]["status"] == "success"


def test_new_support_rerank_does_not_complete_missing_or_duplicate_indices():
    fallback, _, sends, _, _ = execute(rerank="no ranking")
    assert len(sends) == 2 and not fallback["method_observation"]["rerank_applied"]
    assert fallback["arms"]["B"]["record"]["ranked_titles"] == []
    assert fallback["arms"]["B"]["record"]["status"] == "failed"
    completed, _, _, _, _ = execute(rerank="[5,5]")
    assert not completed["method_observation"]["rerank_applied"]
    assert completed["arms"]["B"]["record"]["ranked_titles"] == []
    assert completed["arms"]["B"]["record"]["status"] == "failed"


def test_empty_collection_does_not_force_a_rerank_request_or_claim_answer_success():
    result, searches, sends, factories, _ = execute(empty=True)
    assert len(sends) == 1 and factories == [("q01", "rewrite")]
    assert result["arms"]["A"]["record"]["status"] == "success"
    assert result["arms"]["A"]["sources"] == []
    assert result["arms"]["B"]["record"]["status"] == "success"
    assert result["method_observation"]["rerank_input_count"] is None


def test_support_rerank_can_report_no_matching_material_without_more_model_steps():
    result, _, sends, factories, _ = execute(rerank="[]")
    assert len(sends) == 2 and factories == [("q01", "rewrite"), ("q01", "rerank")]
    assert result["arms"]["A"]["sources"]  # Raw vector neighbours still retained separately.
    assert result["arms"]["B"]["sources"] == []
    assert result["arms"]["B"]["record"]["status"] == "success"
    assert result["method_observation"]["rerank_applied"] is True


@pytest.mark.parametrize("stage", ["rewrite", "rerank"])
def test_transport_or_accounting_failure_cannot_hide_behind_production_fallback(stage):
    knowledge, factory, healthy, _, sends, factories, _ = fixture(failure=stage)
    events = []
    with pytest.raises(ValueError, match="controlled_transport_failure"):
        asyncio.run(runtime.execute_retrieval_case(cases()[0], knowledge=knowledge,
            client_factory=factory, ensure_healthy=healthy, arm_order=("B", "A"),
            on_observation=events.append))
    assert len(sends) == (1 if stage == "rewrite" else 2)
    assert not any(event["event"] == "arm_started" and event["arm"] == "A" for event in events)


@pytest.mark.parametrize("damage", ["content", "duplicate"])
def test_b_source_error_is_not_swallowed_by_production_handler_fallback(damage):
    knowledge, factory, healthy, _, sends, _, _ = fixture(damage=damage)
    with pytest.raises(ValueError, match="prospective_retrieval_local_source_failure"):
        asyncio.run(runtime.execute_retrieval_case(cases()[0], knowledge=knowledge,
            client_factory=factory, ensure_healthy=healthy, arm_order=("B", "A")))
    assert len(sends) == 1  # No rerank is started on invalid source rows.


def test_cancel_is_propagated_without_continuing_arms_or_closing_caller_collection():
    knowledge, factory, _, _, sends, _, closed = fixture()
    def unhealthy():
        raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime.execute_retrieval_case(cases()[0], knowledge=knowledge,
            client_factory=factory, ensure_healthy=unhealthy))
    assert sends == [] and closed == []


def test_callback_can_seal_partial_a_before_a_later_accounting_error():
    knowledge, factory, healthy, _, sends, _, _ = fixture()
    sealed = []
    async def callback(event):
        sealed.append(deepcopy(event))
        if event["event"] == "arm_started" and event["arm"] == "B":
            raise ValueError("controlled_storage_failure")
    with pytest.raises(ValueError, match="controlled_storage_failure"):
        asyncio.run(runtime.execute_retrieval_case(cases()[0], knowledge=knowledge,
            client_factory=factory, ensure_healthy=healthy, on_observation=callback))
    a = next(event for event in sealed if event["event"] == "arm_completed")
    assert a["arm"] == "A" and a["result"]["sources"] and sends == []


@pytest.mark.parametrize("change", ["old_id", "history", "empty", "order", "model"])
def test_invalid_identity_or_configuration_stops_before_any_local_or_model_work(change):
    case, kwargs = deepcopy(cases()[0]), {}
    if change == "old_id": case["id"] = "r01"
    elif change == "history": case["input"]["history"] = [{"role": "user", "content": "old"}]
    elif change == "empty": case["input"]["message"] = ""
    elif change == "order": kwargs["arm_order"] = ["A", "A"]
    else: kwargs["model"] = "unapproved-model"
    knowledge, factory, healthy, searches, sends, _, _ = fixture()
    with pytest.raises(ValueError):
        asyncio.run(runtime.execute_retrieval_case(case, knowledge=knowledge,
            client_factory=factory, ensure_healthy=healthy, **kwargs))
    assert not searches and not sends


def test_shared_metrics_keep_all_planned_cases_and_distinguish_no_evidence():
    case = cases()[0]
    sample, _, _, _, _ = execute(case)
    report = runtime.summarize_retrieval(cases(), [sample])
    for arm in ("A", "B"):
        assert report[arm]["planned_total"] == 12
        assert report[arm]["evidence"]["planned_total"] == 9
        assert report[arm]["evidence"]["not_executed"] == 8
        assert report[arm]["no_evidence"]["planned_total"] == 3
    assert report["B"]["evidence"]["hit_at_3"]["numerator"] == 1
    assert report["B"]["evidence"]["document_recall_at_3"]["numerator"] == 1
    with pytest.raises(ValueError, match="sample_identity"):
        runtime.summarize_retrieval(cases(), [sample, sample])


def test_empty_summary_creates_no_send_permission_and_preserves_all_unexecuted():
    report = runtime.summarize_retrieval(cases(), [])
    assert runtime.MODEL_REQUEST_UPPER_BOUND == 24
    assert all(report[arm]["not_executed"] == 12 for arm in ("A", "B"))
