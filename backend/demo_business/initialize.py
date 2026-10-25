"""Explicit initialization of an already-created, empty MySQL demo database.

MySQL DDL commits implicitly. Failed DDL is preserved for inspection, never
dropped or silently resumed. All fixture INSERTs and the final marker share one
transaction. An uncertain commit is not retried by this module.
"""
from __future__ import annotations

import json
from typing import Any

from .store import (
    DemoDatabaseConfig, DemoStoreError, KIND, SCHEMA_VERSION, TABLE_COLUMNS,
    TABLE_PREFIX, connect, fixture_hash, load_fixture, verify_marker,
)


DDL = (
    "CREATE TABLE sf_demo_meta (id TINYINT NOT NULL PRIMARY KEY, schema_version VARCHAR(64) NOT NULL, kind VARCHAR(64) NOT NULL, seed_sha256 CHAR(64) NOT NULL, as_of CHAR(20) NOT NULL) ENGINE=InnoDB",
    "CREATE TABLE sf_demo_customers (customer_id VARCHAR(32) NOT NULL PRIMARY KEY, display_name VARCHAR(80) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_orders (order_id VARCHAR(16) NOT NULL PRIMARY KEY, customer_id VARCHAR(32) NOT NULL, order_type VARCHAR(16) NOT NULL, currency CHAR(3) NOT NULL, total_amount_minor BIGINT NOT NULL, payment_status VARCHAR(24) NOT NULL, record_updated_at CHAR(20) NOT NULL, INDEX customer_orders(customer_id,order_id), FOREIGN KEY(customer_id) REFERENCES sf_demo_customers(customer_id), CHECK(total_amount_minor >= 0)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_items (item_id VARCHAR(32) NOT NULL PRIMARY KEY, order_id VARCHAR(16) NOT NULL, name VARCHAR(160) NOT NULL, item_type VARCHAR(24) NOT NULL, quantity INT NOT NULL, unit_amount_minor BIGINT NOT NULL, UNIQUE KEY order_item(order_id,item_id), FOREIGN KEY(order_id) REFERENCES sf_demo_orders(order_id), CHECK(quantity > 0), CHECK(unit_amount_minor >= 0)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_payments (payment_id VARCHAR(32) NOT NULL PRIMARY KEY, order_id VARCHAR(16) NOT NULL, amount_minor BIGINT NOT NULL, status VARCHAR(24) NOT NULL, channel VARCHAR(24) NOT NULL, record_updated_at CHAR(20) NOT NULL, FOREIGN KEY(order_id) REFERENCES sf_demo_orders(order_id), CHECK(amount_minor >= 0)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_shipments (shipment_id VARCHAR(32) NOT NULL PRIMARY KEY, order_id VARCHAR(16) NOT NULL, item_id VARCHAR(32) NOT NULL, status VARCHAR(32) NOT NULL, tracking_id VARCHAR(64) NULL, estimated_delivery CHAR(10) NULL, record_updated_at CHAR(20) NOT NULL, FOREIGN KEY(order_id) REFERENCES sf_demo_orders(order_id), FOREIGN KEY(order_id,item_id) REFERENCES sf_demo_items(order_id,item_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_entitlements (entitlement_id VARCHAR(32) NOT NULL PRIMARY KEY, order_id VARCHAR(16) NOT NULL, status VARCHAR(32) NOT NULL, package_name VARCHAR(160) NOT NULL, auto_renew BOOLEAN NOT NULL, expires_at CHAR(20) NULL, record_updated_at CHAR(20) NOT NULL, FOREIGN KEY(order_id) REFERENCES sf_demo_orders(order_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
    "CREATE TABLE sf_demo_refunds (refund_id VARCHAR(32) NOT NULL PRIMARY KEY, order_id VARCHAR(16) NOT NULL, status VARCHAR(32) NOT NULL, amount_minor BIGINT NULL, record_updated_at CHAR(20) NOT NULL, FOREIGN KEY(order_id) REFERENCES sf_demo_orders(order_id), CHECK(amount_minor >= 0)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
)
EXPECTED_TABLES = {TABLE_PREFIX + name for name in TABLE_COLUMNS} | {"sf_demo_meta"}


def _tables(cursor) -> set[str]:
    cursor.execute("SHOW FULL TABLES")
    rows = cursor.fetchall()
    names = set()
    for row in rows:
        values = tuple(row.values())
        if len(values) != 2 or values[1] != "BASE TABLE":
            raise DemoStoreError("demo_database_not_empty")
        names.add(values[0])
    return names


def _normal_rows(rows: list[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        values = dict(row)
        if table == "entitlements":
            if values.get("auto_renew") not in (0, 1, False, True):
                raise DemoStoreError("demo_seed_mismatch")
            values["auto_renew"] = bool(values["auto_renew"])
        normalized.append(values)
    return sorted(normalized, key=lambda row: row[TABLE_COLUMNS[table][0]])


def _verify_cursor(cursor) -> dict[str, Any]:
    if _tables(cursor) != EXPECTED_TABLES:
        raise DemoStoreError("demo_schema_mismatch")
    verify_marker(cursor)
    fixture = load_fixture()
    counts = {}
    for table, names in TABLE_COLUMNS.items():
        cursor.execute(f"SELECT {', '.join(names)} FROM {TABLE_PREFIX}{table} ORDER BY {names[0]} LIMIT 102")
        rows = _normal_rows(cursor.fetchall(), table)
        expected = _normal_rows(fixture[table], table)
        if rows != expected:
            raise DemoStoreError("demo_seed_mismatch")
        counts[table] = len(rows)
    cursor.execute("SELECT COUNT(*) AS count FROM sf_demo_meta")
    if cursor.fetchone() != {"count": 1}:
        raise DemoStoreError("demo_schema_mismatch")
    return {"schema_version": SCHEMA_VERSION, "seed_sha256": fixture_hash(), "is_demo": True,
            "as_of": fixture["as_of"], "counts": counts}


def verify_database(url: str | DemoDatabaseConfig, *, connector=None) -> dict[str, Any]:
    config = DemoDatabaseConfig.parse(url) if isinstance(url, str) else url
    connection = None
    try:
        connection = (connector or connect)(config)
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            return _verify_cursor(cursor)
    except DemoStoreError:
        raise
    except Exception:
        raise DemoStoreError("demo_store_unavailable") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def initialize_database(url: str | DemoDatabaseConfig, *, connector=None) -> dict[str, Any]:
    # Validate all fixture relationships before any DDL or INSERT.
    fixture = load_fixture()
    config = DemoDatabaseConfig.parse(url) if isinstance(url, str) else url
    connection = None
    try:
        connection = (connector or connect)(config)
        with connection.cursor() as cursor:
            tables = _tables(cursor)
            if tables:
                result = _verify_cursor(cursor)
                return {**result, "operation": "verified-existing-no-write"}
            for statement in DDL:
                cursor.execute(statement)
            connection.begin()
            for table, names in TABLE_COLUMNS.items():
                placeholders = ", ".join("%s" for _ in names)
                cursor.executemany(
                    f"INSERT INTO {TABLE_PREFIX}{table} ({', '.join(names)}) VALUES ({placeholders})",
                    [tuple(row[name] for name in names) for row in fixture[table]],
                )
            cursor.execute(
                "INSERT INTO sf_demo_meta (id,schema_version,kind,seed_sha256,as_of) VALUES (%s,%s,%s,%s,%s)",
                (1, SCHEMA_VERSION, KIND, fixture_hash(), fixture["as_of"]),
            )
            # No retries here. If COMMIT acknowledgement is lost, inspect/verify.
            connection.commit()
            return {**_verify_cursor(cursor), "operation": "initialized-empty-demo-database"}
    except DemoStoreError:
        raise
    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        raise DemoStoreError("demo_initialize_failed_inspect_no_retry") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
