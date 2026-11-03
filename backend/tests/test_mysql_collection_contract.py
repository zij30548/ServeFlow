"""Explicit real MySQL read-only integration; never initialize or reuse app DB.

Ordinary tests skip this file's cases unless the dedicated opt-in URL is set.
"""
import os

import pytest

from demo_business.store import load_fixture
from evaluation.task_data import parse_mysql_url, _verify_mysql
from evaluation.task_evidence import _clone, fixture_order_result


@pytest.fixture(scope='module')
def actual_readonly_store():
    value = os.environ.get('SERVEFLOW_EVAL_READONLY_TEST_MYSQL_URL')
    if not value:
        pytest.skip('explicit isolated SELECT-only MySQL integration URL not configured')
    # Exact evaluation name, literal schema grant, full synthetic marker/content,
    # SELECT-only permissions: no DDL or fixture writes on this path.
    return _verify_mysql(parse_mysql_url(value))[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('order_id', ['DEMO-1001', 'DEMO-1003', 'DEMO-1004', 'DEMO-1010'])
async def test_real_driver_collections_satisfy_strict_evidence_without_fact_changes(actual_readonly_store, order_id):
    result = await actual_readonly_store.read_order('demo-customer-a', order_id)
    assert all(type(result['order'][table]) is list for table in (
        'items', 'payments', 'shipments', 'entitlements', 'refunds'))
    copied, _ = _clone(result)
    assert copied == fixture_order_result(load_fixture(), 'demo-customer-a', order_id)


@pytest.mark.asyncio
async def test_real_collection_projection_preserves_cross_customer_rejection(actual_readonly_store):
    foreign = await actual_readonly_store.read_order('demo-customer-a', 'DEMO-1005')
    missing = await actual_readonly_store.read_order('demo-customer-a', 'DEMO-9999')
    assert foreign == missing and foreign['status'] == 'not_found' and foreign['order'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('case_id', ['e01', 'e02'])
async def test_actual_mysql_passes_original_tool_handler_and_capture_with_controlled_model(actual_readonly_store, case_id):
    # The database/driver/native business handler/evidence capture are real.
    # Intent, actor, and Judge outputs are controlled; no live model is created.
    from test_task_runtime import ControlledData, ControlledSDK, case
    from evaluation.task_runtime import execute_case
    data = ControlledData()
    # Keep the evaluation's controlled label: Chroma and model are NOT live.
    data.proof = {'kind': 'controlled-fixture', 'mysql': 'isolated-real-driver'}
    data.order_reader = actual_readonly_store.read_order
    result = await execute_case(case(case_id), ControlledSDK(), data,
                                ensure_healthy=data.assert_healthy)
    assert result['execution_status'] == 'success'
    assert result['deterministic']['blockers'] == []
    assert result['semantic_review'] is None  # Never turn this into a quality score.
    call = result['turns'][0]['tool_evidence']['calls'][0]
    assert call['name'] == 'read_demo_order' and call['success']
    order = call['result']['order']
    assert type(order['entitlements']) is list and order['entitlements'] == []
    if case_id == 'e02':
        assert order['refunds'][0]['status'] == 'pending_review'
        assert order['refunds'][0]['amount_minor'] is None
