import asyncio
from copy import deepcopy
import os
import threading

import pytest

from demo_business import DemoBusinessStore, DemoStoreError
from demo_business.initialize import EXPECTED_TABLES, initialize_database, verify_database
from demo_business.store import (
    DemoDatabaseConfig, KIND, SCHEMA_VERSION, SOURCE, TABLE_COLUMNS,
    connect, fixture_hash, load_fixture, sample_orders, validate_fixture,
)


URL = 'mysql+pymysql://app:private-test-password@127.0.0.1:13306/serveflow_demo_unit'


class FakeCursor:
    def __init__(self, fixture=None, marker=None, extra_rows=0):
        self.fixture = deepcopy(fixture or load_fixture())
        self.marker = marker or {'schema_version': SCHEMA_VERSION, 'kind': KIND,
                                 'seed_sha256': fixture_hash(), 'as_of': self.fixture['as_of']}
        self.calls = []
        self.result = []
        self.extra_rows = extra_rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, parameters=None):
        self.calls.append((statement, parameters))
        if 'FROM sf_demo_meta WHERE' in statement:
            self.result = [deepcopy(self.marker)]
        elif 'FROM sf_demo_orders WHERE' in statement:
            order_id, customer_id = parameters
            self.result = [deepcopy(r) for r in self.fixture['orders'] if r['order_id'] == order_id and r['customer_id'] == customer_id]
        elif ' WHERE order_id = %s ORDER BY ' in statement:
            table = statement.split('FROM sf_demo_', 1)[1].split(' ', 1)[0]
            self.result = [deepcopy(r) for r in self.fixture[table] if r['order_id'] == parameters[0]]
            if table == 'items' and self.extra_rows:
                self.result *= self.extra_rows
        elif statement == 'START TRANSACTION READ ONLY':
            self.result = []
        else:
            raise AssertionError('Unexpected SQL')

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return self.result


class FakeConnection:
    def __init__(self, cursor):
        self.reader = cursor
        self.closed = False

    def cursor(self):
        return self.reader

    def close(self):
        self.closed = True


def make_store(**kwargs):
    cursor = FakeCursor(**kwargs)
    connection = FakeConnection(cursor)
    return DemoBusinessStore(URL, connector=lambda config: connection), cursor, connection


@pytest.mark.parametrize('url', [
    'sqlite:///tmp/demo.sqlite3',
    'mysql://app:secret@db.example.com/serveflow_demo_a',
    'mysql://app:secret@127.0.0.1/trippilot',
    'mysql://app:secret@127.0.0.1/serveflow_demo_a?charset=utf8',
    'mysql://app:secret@127.0.0.1/serveflow_demo_a#fragment',
    'mysql://app:secret@127.0.0.1/serveflow_demo_a/other',
    'mysql://app:secret@127.0.0.1/serveflow_demo_',
    'mysql://app:secret@127.0.0.1:99999/serveflow_demo_a',
    'mysql://app:secret@127.0.0.1/%2fserveflow_demo_a',
    'mysql://app:secret@127.0.0.1/serveflow_demo_a ',
])
def test_invalid_database_targets_are_rejected_before_connect(url):
    with pytest.raises(DemoStoreError, match='demo_store_invalid_configuration'):
        DemoBusinessStore(url, connector=lambda config: pytest.fail('must not connect'))


def test_url_is_not_represented_with_credentials_and_absent_feature_is_optional(monkeypatch):
    config = DemoDatabaseConfig.parse(URL)
    assert 'private-test-password' not in repr(config)
    assert config.database == 'serveflow_demo_unit'
    monkeypatch.delenv('SERVEFLOW_DEMO_MYSQL_URL', raising=False)
    assert DemoBusinessStore.from_env() is None


def test_configured_feature_reports_missing_connector_without_connecting(monkeypatch):
    monkeypatch.setenv('SERVEFLOW_DEMO_MYSQL_URL', URL)
    monkeypatch.setattr('demo_business.store.importlib.util.find_spec', lambda name: None)
    with pytest.raises(DemoStoreError, match='demo_store_dependency_missing'):
        DemoBusinessStore.from_env()


def test_fixture_is_complete_synthetic_and_referentially_valid():
    fixture = validate_fixture(deepcopy(load_fixture()))
    assert len(fixture['customers']) == 2
    assert len(fixture['orders']) == 10
    assert len({r['order_id'] for r in fixture['orders']}) == 10
    assert all(type(row['total_amount_minor']) is int for row in fixture['orders'])
    assert all('演示' in row['name'] for row in fixture['items'])


@pytest.mark.parametrize('corruption', ['foreign_order', 'foreign_item', 'duplicate', 'money_float', 'unknown_field', 'missing_amount'])
def test_fixture_rejects_inconsistent_or_ambiguous_records(corruption):
    fixture = deepcopy(load_fixture())
    if corruption == 'foreign_order':
        fixture['payments'][0]['order_id'] = 'DEMO-9999'
    elif corruption == 'foreign_item':
        fixture['shipments'][0]['item_id'] = 'I-1002'
    elif corruption == 'duplicate':
        fixture['orders'].append(deepcopy(fixture['orders'][0]))
    elif corruption == 'money_float':
        fixture['orders'][0]['total_amount_minor'] = 169800.0
    elif corruption == 'unknown_field':
        fixture['orders'][0]['owner_id'] = 'pretend-owner'
    else:
        fixture['refunds'][1]['amount_minor'] = None
    with pytest.raises(DemoStoreError, match='demo_fixture_invalid'):
        validate_fixture(fixture)


@pytest.mark.asyncio
async def test_owned_order_projects_actual_rows_without_extra_business_claims():
    store, cursor, connection = make_store()
    result = await store.read_order('demo-customer-a', 'DEMO-1001')
    assert result['source'] == SOURCE
    assert result['is_demo'] is True
    assert result['status'] == 'found'
    assert result['as_of'] == '2026-10-25T00:00:00Z'
    assert result['order']['total_amount_minor'] == 169800
    assert [r['status'] for r in result['order']['shipments']] == ['in_transit', 'awaiting_dispatch']
    assert result['order']['shipments'][1]['estimated_delivery'] is None
    assert connection.closed
    assert all(statement.startswith(('SELECT ', 'START TRANSACTION READ ONLY')) for statement, _ in cursor.calls)
    order_query = next(call for call in cursor.calls if 'FROM sf_demo_orders WHERE' in call[0])
    assert order_query[1] == ('DEMO-1001', 'demo-customer-a')
    assert 'DEMO-1001' not in order_query[0]


@pytest.mark.asyncio
async def test_other_customer_and_absent_order_are_indistinguishable():
    store, _, _ = make_store()
    other = await store.read_order('demo-customer-a', 'DEMO-1005')
    store2, _, _ = make_store()
    absent = await store2.read_order('demo-customer-a', 'DEMO-9999')
    assert other == absent
    assert other['status'] == 'not_found'
    assert other['order'] is None


@pytest.mark.asyncio
async def test_pending_refund_is_unknown_not_zero_and_cancelled_renewal_is_not_refund():
    store, _, _ = make_store()
    pending = (await store.read_order('demo-customer-a', 'DEMO-1004'))['order']
    assert pending['refunds'][0]['status'] == 'pending_review'
    assert pending['refunds'][0]['amount_minor'] is None
    store2, _, _ = make_store()
    renewal = (await store2.read_order('demo-customer-a', 'DEMO-1010'))['order']
    assert renewal['entitlements'][0]['status'] == 'active'
    assert renewal['entitlements'][0]['auto_renew'] is False
    assert renewal['refunds'] == []


def test_bad_input_does_not_reach_database():
    store = DemoBusinessStore(URL, connector=lambda config: pytest.fail('must not connect'))
    with pytest.raises(DemoStoreError, match='demo_invalid_order'):
        store.read_order_sync('demo-customer-a', "DEMO-1001' OR 1=1")
    with pytest.raises(DemoStoreError, match='demo_invalid_customer'):
        store.read_order_sync('arbitrary-customer', 'DEMO-1001')


def test_schema_marker_and_oversize_related_rows_fail_closed_and_close_connection():
    store, _, connection = make_store(marker={'schema_version': 'other'})
    with pytest.raises(DemoStoreError, match='demo_schema_mismatch'):
        store.read_order_sync('demo-customer-a', 'DEMO-1001')
    assert connection.closed
    store, _, connection = make_store(extra_rows=7)
    with pytest.raises(DemoStoreError, match='demo_store_invalid_record'):
        store.read_order_sync('demo-customer-a', 'DEMO-1001')
    assert connection.closed


def test_driver_errors_do_not_leak_sql_credentials_or_private_paths():
    def failure(config):
        raise RuntimeError('SELECT secret /private/file password=private-test-password')
    store = DemoBusinessStore(URL, connector=failure)
    with pytest.raises(DemoStoreError) as caught:
        store.read_order_sync('demo-customer-a', 'DEMO-1001')
    assert str(caught.value) == 'demo_store_unavailable'


@pytest.mark.asyncio
async def test_cancelled_read_drains_thread_before_releasing_connection():
    entered, release = threading.Event(), threading.Event()
    cursor = FakeCursor()
    connection = FakeConnection(cursor)

    def connector(config):
        entered.set()
        assert release.wait(2)
        return connection

    store = DemoBusinessStore(URL, connector=connector)
    task = asyncio.create_task(store.read_order('demo-customer-a', 'DEMO-1001'))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not connection.closed
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection.closed


def test_sample_catalogue_never_lists_other_customer_orders():
    a = sample_orders('demo-customer-a')
    b = sample_orders('demo-customer-b')
    assert len(a) == 9 and len(b) == 1
    assert {r['order_id'] for r in a}.isdisjoint({r['order_id'] for r in b})
    assert sample_orders('not-a-customer') == []


class InitCursor(FakeCursor):
    def __init__(self, existing=False, *, fail_create=False, fail_insert=False):
        super().__init__()
        self.tables = set(EXPECTED_TABLES) if existing else set()
        if not existing:
            self.fixture = {**self.fixture, **{table: [] for table in TABLE_COLUMNS}}
        self.fail_create = fail_create
        self.fail_insert = fail_insert

    def execute(self, statement, parameters=None):
        if statement == 'SHOW FULL TABLES':
            self.calls.append((statement, parameters))
            self.result = [{'Tables_in_demo': table, 'Table_type': 'BASE TABLE'} for table in sorted(self.tables)]
        elif statement.startswith('CREATE TABLE '):
            self.calls.append((statement, parameters))
            if self.fail_create and len(self.tables) == 2:
                raise RuntimeError('DDL fail with private path')
            self.tables.add(statement.split()[2])
        elif statement.startswith('INSERT INTO sf_demo_meta '):
            self.calls.append((statement, parameters))
            _, version, kind, seed, as_of = parameters
            self.marker = {'schema_version': version, 'kind': kind, 'seed_sha256': seed, 'as_of': as_of}
        elif statement.startswith('SELECT COUNT(*)'):
            self.calls.append((statement, parameters))
            self.result = [{'count': 1}]
        elif statement.startswith('SELECT ') and ' ORDER BY ' in statement and ' WHERE ' not in statement:
            self.calls.append((statement, parameters))
            table = statement.split('FROM sf_demo_', 1)[1].split(' ', 1)[0]
            self.result = deepcopy(self.fixture[table])
        else:
            return super().execute(statement, parameters)

    def executemany(self, statement, rows):
        self.calls.append((statement, rows))
        table = statement.split('INSERT INTO sf_demo_', 1)[1].split(' ', 1)[0]
        if self.fail_insert and table == 'payments':
            raise RuntimeError('INSERT failure with password')
        self.fixture[table] = [dict(zip(TABLE_COLUMNS[table], row)) for row in rows]


class InitConnection(FakeConnection):
    def __init__(self, cursor, uncertain_commit=False):
        super().__init__(cursor)
        self.commits = self.rollbacks = self.begins = 0
        self.uncertain_commit = uncertain_commit

    def begin(self):
        self.begins += 1

    def commit(self):
        self.commits += 1
        if self.uncertain_commit:
            raise RuntimeError('COMMIT acknowledgement lost')

    def rollback(self):
        self.rollbacks += 1


def test_existing_matching_seed_is_verified_without_writes():
    cursor = InitCursor(existing=True)
    connection = InitConnection(cursor)
    result = initialize_database(URL, connector=lambda config: connection)
    assert result['operation'] == 'verified-existing-no-write'
    assert connection.commits == connection.begins == 0
    assert not any(sql.startswith(('CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP')) for sql, _ in cursor.calls)
    assert connection.closed


def test_unrelated_existing_tables_are_rejected_before_ddl_or_seed():
    cursor = InitCursor(existing=True)
    cursor.tables.add('unrelated_user_data')
    connection = InitConnection(cursor)
    with pytest.raises(DemoStoreError, match='demo_schema_mismatch'):
        initialize_database(URL, connector=lambda config: connection)
    assert cursor.calls and all(call == ('SHOW FULL TABLES', None) for call in cursor.calls)
    assert connection.commits == 0


def test_existing_modified_seed_is_not_overwritten():
    cursor = InitCursor(existing=True)
    cursor.fixture['orders'][0]['payment_status'] = 'unpaid'
    connection = InitConnection(cursor)
    with pytest.raises(DemoStoreError, match='demo_seed_mismatch'):
        initialize_database(URL, connector=lambda config: connection)
    assert not any(sql.startswith(('CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP')) for sql, _ in cursor.calls)


def test_failed_ddl_leaves_partial_schema_for_inspection_without_cleanup():
    cursor = InitCursor(fail_create=True)
    connection = InitConnection(cursor)
    with pytest.raises(DemoStoreError, match='demo_initialize_failed_inspect_no_retry'):
        initialize_database(URL, connector=lambda config: connection)
    assert len(cursor.tables) == 2
    assert connection.commits == 0
    assert not any(sql.startswith(('INSERT', 'DROP', 'DELETE')) for sql, _ in cursor.calls)
    assert connection.closed


def test_seed_failure_rolls_back_and_never_retries_or_drops_schema():
    cursor = InitCursor(fail_insert=True)
    connection = InitConnection(cursor)
    with pytest.raises(DemoStoreError, match='demo_initialize_failed_inspect_no_retry'):
        initialize_database(URL, connector=lambda config: connection)
    assert connection.rollbacks == 1 and connection.commits == 0
    assert sum(sql.startswith('INSERT INTO sf_demo_payments ') for sql, _ in cursor.calls) == 1
    assert not any(sql.startswith(('DROP', 'DELETE')) for sql, _ in cursor.calls)


def test_uncertain_commit_is_not_retried_or_claimed_success():
    cursor = InitCursor()
    connection = InitConnection(cursor, uncertain_commit=True)
    with pytest.raises(DemoStoreError, match='demo_initialize_failed_inspect_no_retry'):
        initialize_database(URL, connector=lambda config: connection)
    assert connection.commits == 1
    assert sum(sql.startswith('INSERT INTO sf_demo_orders ') for sql, _ in cursor.calls) == 1


def test_successful_seed_inserts_once_and_commits_one_transaction():
    cursor = InitCursor()
    connection = InitConnection(cursor)
    result = initialize_database(URL, connector=lambda config: connection)
    assert result['operation'] == 'initialized-empty-demo-database'
    assert result['counts']['orders'] == 10
    assert connection.begins == connection.commits == 1
    assert connection.rollbacks == 0


@pytest.fixture(scope='module')
def actual_demo_url():
    url = os.getenv('SERVEFLOW_DEMO_TEST_MYSQL_URL')
    if not url:
        pytest.skip('independent MySQL demo test database not configured')
    DemoDatabaseConfig.parse(url)
    return url


def test_mysql_explicit_initialization_is_repeatable_without_reseeding(actual_demo_url):
    first = initialize_database(actual_demo_url)
    second = initialize_database(actual_demo_url)
    assert first['operation'] in {'initialized-empty-demo-database', 'verified-existing-no-write'}
    assert second['operation'] == 'verified-existing-no-write'
    assert verify_database(actual_demo_url)['counts']['orders'] == 10


@pytest.mark.asyncio
async def test_mysql_persistence_and_owner_checks_after_new_store(actual_demo_url):
    initialize_database(actual_demo_url)
    first = await DemoBusinessStore(actual_demo_url).read_order('demo-customer-a', 'DEMO-1003')
    reopened = await DemoBusinessStore(actual_demo_url).read_order('demo-customer-a', 'DEMO-1003')
    assert first == reopened
    assert first['order']['entitlements'][0]['status'] == 'activation_failed'
    assert first['order']['payments'][0]['amount_minor'] == 3900
    assert (await DemoBusinessStore(actual_demo_url).read_order('demo-customer-a', 'DEMO-1005'))['status'] == 'not_found'


def test_mysql_failed_related_insert_rolls_back_the_whole_new_order(actual_demo_url):
    from pymysql import IntegrityError
    initialize_database(actual_demo_url)
    connection = connect(DemoDatabaseConfig.parse(actual_demo_url))
    try:
        connection.begin()
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO sf_demo_orders (order_id,customer_id,order_type,currency,total_amount_minor,payment_status,record_updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                ('DEMO-1999', 'demo-customer-a', 'physical', 'CNY', 100, 'unpaid', '2026-10-24T00:00:00Z'),
            )
            with pytest.raises(IntegrityError):
                cursor.execute(
                    'INSERT INTO sf_demo_items (item_id,order_id,name,item_type,quantity,unit_amount_minor) VALUES (%s,%s,%s,%s,%s,%s)',
                    ('I-ROLLBACK', 'DEMO-9999', '演示回滚测试', 'device', 1, 100),
                )
        connection.rollback()
    finally:
        connection.close()
    assert DemoBusinessStore(actual_demo_url).read_order_sync('demo-customer-a', 'DEMO-1999')['status'] == 'not_found'
    assert verify_database(actual_demo_url)['counts']['orders'] == 10
