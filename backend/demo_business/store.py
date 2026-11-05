"""Readonly, bounded access to an explicitly initialized MySQL demo database.

This does not query a merchant, charge/refund money, or generate sample records at
runtime. The optional connector is imported only after a demo URL is configured.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import unquote, urlsplit


SCHEMA_VERSION = "serveflow-demo-business-v1"
SOURCE = "serveflow-mysql-demo-v1"
KIND = "synthetic-local-demo"
IO_TIMEOUT_SECONDS = 3
CUSTOMER_IDS = frozenset({"demo-customer-a", "demo-customer-b"})
ORDER_PATTERN = re.compile(r"DEMO-[0-9]{4}")
DATABASE_PATTERN = re.compile(r"serveflow_demo_[a-z0-9_]{1,45}")
FIXTURE_PATH = Path(__file__).with_name("fixtures.json")

# Explicit columns are also the only columns projected into tool responses.
TABLE_COLUMNS = {
    "customers": ("customer_id", "display_name"),
    "orders": ("order_id", "customer_id", "order_type", "currency", "total_amount_minor", "payment_status", "record_updated_at"),
    "items": ("item_id", "order_id", "name", "item_type", "quantity", "unit_amount_minor"),
    "payments": ("payment_id", "order_id", "amount_minor", "status", "channel", "record_updated_at"),
    "shipments": ("shipment_id", "order_id", "item_id", "status", "tracking_id", "estimated_delivery", "record_updated_at"),
    "entitlements": ("entitlement_id", "order_id", "status", "package_name", "auto_renew", "expires_at", "record_updated_at"),
    "refunds": ("refund_id", "order_id", "status", "amount_minor", "record_updated_at"),
}
def _allowed_hosts():
    # A fixed Compose service name is permitted only by the explicit perimeter
    # mode. Ordinary local/production callers retain the original loopback set.
    return {"127.0.0.1", "localhost", "mysql"} if os.getenv("ECHOMIND_ENV", "development").lower() == "protected-demo" else {"127.0.0.1", "localhost"}


TABLE_PREFIX = "sf_demo_"
MAX_RELATED_ROWS = 12


class DemoStoreError(RuntimeError):
    """A public-safe error code, never a driver exception/SQL/credential."""

    def __init__(self, code: str = "demo_store_unavailable"):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DemoDatabaseConfig:
    host: str
    port: int
    database: str
    username: str = field(repr=False)
    password: str = field(repr=False)

    def __post_init__(self):
        if (self.host not in _allowed_hosts()
                or type(self.port) is not int or not 1 <= self.port <= 65535
                or not isinstance(self.database, str) or not DATABASE_PATTERN.fullmatch(self.database)
                or not isinstance(self.username, str) or not self.username
                or not isinstance(self.password, str)):
            raise DemoStoreError("demo_store_invalid_configuration")

    @classmethod
    def parse(cls, value: str) -> "DemoDatabaseConfig":
        try:
            if not isinstance(value, str) or not value or value != value.strip() or any(c.isspace() for c in value):
                raise ValueError
            parsed = urlsplit(value)
            database = unquote(parsed.path.removeprefix("/"))
            if (parsed.scheme not in {"mysql", "mysql+pymysql"}
                    or parsed.hostname not in _allowed_hosts()
                    or parsed.query or parsed.fragment
                    or not parsed.username
                    or not DATABASE_PATTERN.fullmatch(database)):
                raise ValueError
            port = parsed.port or 3306
            if not 1 <= port <= 65535:
                raise ValueError
            return cls(parsed.hostname, port, database, unquote(parsed.username), unquote(parsed.password or ""))
        except (TypeError, ValueError):
            raise DemoStoreError("demo_store_invalid_configuration") from None


def _validate_row(table: str, row: Any) -> None:
    if not isinstance(row, dict) or set(row) != set(TABLE_COLUMNS[table]):
        raise DemoStoreError("demo_fixture_invalid")
    for key, value in row.items():
        if key.endswith("_minor"):
            if value is None and table == "refunds":
                continue
            if type(value) is not int or not 0 <= value <= 1_000_000_000:
                raise DemoStoreError("demo_fixture_invalid")
        elif key == "quantity":
            if type(value) is not int or not 1 <= value <= 100:
                raise DemoStoreError("demo_fixture_invalid")
        elif key == "auto_renew":
            if type(value) is not bool:
                raise DemoStoreError("demo_fixture_invalid")
        elif value is None:
            if key not in {"tracking_id", "estimated_delivery", "expires_at"}:
                raise DemoStoreError("demo_fixture_invalid")
        elif not isinstance(value, str) or not 0 < len(value) <= 160:
            raise DemoStoreError("demo_fixture_invalid")
        if isinstance(value, str) and (key == "record_updated_at" or key == "expires_at"):
            try:
                if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value):
                    raise ValueError
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise DemoStoreError("demo_fixture_invalid") from None
        if key == "estimated_delivery" and value is not None:
            try:
                if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                    raise ValueError
                date.fromisoformat(value)
            except (TypeError, ValueError):
                raise DemoStoreError("demo_fixture_invalid") from None
    if "order_id" in row and not ORDER_PATTERN.fullmatch(row["order_id"]):
        raise DemoStoreError("demo_fixture_invalid")
    if "customer_id" in row and row["customer_id"] not in CUSTOMER_IDS:
        raise DemoStoreError("demo_fixture_invalid")
    allowed = {
        ("orders", "order_type"): {"physical", "digital", "bundle"},
        ("orders", "currency"): {"CNY"},
        ("orders", "payment_status"): {"paid", "unpaid", "refunded"},
        ("items", "item_type"): {"device", "accessory", "subscription", "digital_content"},
        ("payments", "status"): {"paid", "refunded"},
        ("payments", "channel"): {"demo-wallet", "demo-store"},
        ("shipments", "status"): {"in_transit", "awaiting_dispatch", "preorder", "delivered", "delivery_disputed"},
        ("entitlements", "status"): {"activation_failed", "revoked_after_refund", "active"},
        ("refunds", "status"): {"pending_review", "completed"},
    }
    for (name, key), values in allowed.items():
        if table == name and row[key] not in values:
            raise DemoStoreError("demo_fixture_invalid")


def validate_fixture(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"schema_version", "kind", "as_of", *TABLE_COLUMNS}:
        raise DemoStoreError("demo_fixture_invalid")
    if data["schema_version"] != SCHEMA_VERSION or data["kind"] != KIND or data["as_of"] != "2026-10-25T00:00:00Z":
        raise DemoStoreError("demo_fixture_invalid")
    for table, columns in TABLE_COLUMNS.items():
        rows = data[table]
        if not isinstance(rows, list) or not 0 <= len(rows) <= 100:
            raise DemoStoreError("demo_fixture_invalid")
        ids = set()
        for row in rows:
            _validate_row(table, row)
            if row[columns[0]] in ids:
                raise DemoStoreError("demo_fixture_invalid")
            ids.add(row[columns[0]])
    customers = {row["customer_id"] for row in data["customers"]}
    orders = {row["order_id"]: row for row in data["orders"]}
    items = {row["item_id"]: row for row in data["items"]}
    if customers != CUSTOMER_IDS or not orders:
        raise DemoStoreError("demo_fixture_invalid")
    for order in orders.values():
        if order["customer_id"] not in customers:
            raise DemoStoreError("demo_fixture_invalid")
        total = sum(row["quantity"] * row["unit_amount_minor"] for row in data["items"] if row["order_id"] == order["order_id"])
        if total != order["total_amount_minor"]:
            raise DemoStoreError("demo_fixture_invalid")
    for table in ("items", "payments", "shipments", "entitlements", "refunds"):
        for row in data[table]:
            if row["order_id"] not in orders:
                raise DemoStoreError("demo_fixture_invalid")
            if table == "shipments" and (row["item_id"] not in items or items[row["item_id"]]["order_id"] != row["order_id"]):
                raise DemoStoreError("demo_fixture_invalid")
            if table == "refunds" and row["status"] == "completed" and row["amount_minor"] is None:
                raise DemoStoreError("demo_fixture_invalid")
    for order in orders.values():
        packet = {**order, **{table: [row for row in data[table] if row["order_id"] == order["order_id"]]
                               for table in ("items", "payments", "shipments", "entitlements", "refunds")}}
        try:
            _validate_order_relations(packet)
        except DemoStoreError:
            raise DemoStoreError("demo_fixture_invalid") from None
    return data


def _validate_order_relations(order: dict[str, Any]) -> None:
    total = order["total_amount_minor"]
    if not order["items"] or sum(row["quantity"] * row["unit_amount_minor"] for row in order["items"]) != total:
        raise DemoStoreError("demo_store_invalid_record")
    paid = sum(row["amount_minor"] for row in order["payments"])
    if order["payment_status"] == "unpaid":
        if order["payments"] or order["refunds"]:
            raise DemoStoreError("demo_store_invalid_record")
    elif paid != total or any(row["status"] != order["payment_status"] for row in order["payments"]):
        raise DemoStoreError("demo_store_invalid_record")
    for refund in order["refunds"]:
        amount = refund["amount_minor"]
        if amount is not None and amount > paid:
            raise DemoStoreError("demo_store_invalid_record")
        if refund["status"] == "completed" and amount is None:
            raise DemoStoreError("demo_store_invalid_record")
    if order["payment_status"] == "refunded" and sum(
        row["amount_minor"] for row in order["refunds"] if row["status"] == "completed" and row["amount_minor"] is not None
    ) != total:
        raise DemoStoreError("demo_store_invalid_record")
    item_ids = {row["item_id"] for row in order["items"]}
    if any(row["item_id"] not in item_ids for row in order["shipments"]):
        raise DemoStoreError("demo_store_invalid_record")
    if order["order_type"] == "digital" and order["shipments"]:
        raise DemoStoreError("demo_store_invalid_record")


@lru_cache(maxsize=1)
def load_fixture() -> dict[str, Any]:
    try:
        raw = FIXTURE_PATH.read_bytes()
        if len(raw) > 65_536:
            raise DemoStoreError("demo_fixture_invalid")
        return validate_fixture(json.loads(raw))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DemoStoreError("demo_fixture_invalid") from None


def fixture_hash() -> str:
    return hashlib.sha256(json.dumps(load_fixture(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sample_orders(customer_id: str) -> list[dict[str, str]]:
    """A bounded synthetic catalogue, not a list of another customer's orders."""
    scenarios = {
        "DEMO-1001": "设备已发货，配件待发", "DEMO-1002": "预售尚未出库",
        "DEMO-1003": "已扣款但数字权益开通失败", "DEMO-1004": "退款申请待审核",
        "DEMO-1005": "演示客户B的设备订单", "DEMO-1006": "数字内容已退款",
        "DEMO-1007": "订单待支付", "DEMO-1008": "签收状态有争议",
        "DEMO-1009": "团队订阅权益有效", "DEMO-1010": "已关闭自动续费，当前权益仍有效",
    }
    if not isinstance(customer_id, str) or customer_id not in CUSTOMER_IDS:
        return []
    return [{"order_id": row["order_id"], "scenario": scenarios[row["order_id"]]}
            for row in load_fixture()["orders"] if row["customer_id"] == customer_id]


def connect(config: DemoDatabaseConfig):
    try:
        import pymysql
        return pymysql.connect(
            host=config.host, port=config.port, user=config.username,
            password=config.password, database=config.database, charset="utf8mb4",
            connect_timeout=IO_TIMEOUT_SECONDS, read_timeout=IO_TIMEOUT_SECONDS,
            write_timeout=IO_TIMEOUT_SECONDS, autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception:
        raise DemoStoreError("demo_store_unavailable") from None


def verify_marker(cursor) -> None:
    cursor.execute("SELECT schema_version, kind, seed_sha256, as_of FROM sf_demo_meta WHERE id = 1")
    row = cursor.fetchone()
    if row != {"schema_version": SCHEMA_VERSION, "kind": KIND, "seed_sha256": fixture_hash(), "as_of": load_fixture()["as_of"]}:
        raise DemoStoreError("demo_schema_mismatch")


class DemoBusinessStore:
    def __init__(self, url: str | DemoDatabaseConfig, *, connector: Callable | None = None):
        self.config = DemoDatabaseConfig.parse(url) if isinstance(url, str) else url
        if not isinstance(self.config, DemoDatabaseConfig):
            raise DemoStoreError("demo_store_invalid_configuration")
        self._connector = connector or connect

    @classmethod
    def from_env(cls) -> "DemoBusinessStore | None":
        value = os.getenv("SERVEFLOW_DEMO_MYSQL_URL", "")
        if not value:
            return None
        store = cls(value)
        if importlib.util.find_spec("pymysql") is None:
            raise DemoStoreError("demo_store_dependency_missing")
        return store

    @staticmethod
    def list_sample_orders(customer_id: str) -> list[dict[str, str]]:
        return sample_orders(customer_id)

    async def read_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        worker = asyncio.create_task(asyncio.to_thread(self.read_order_sync, customer_id, order_id))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Cancelling an await cannot kill a running Python/MySQL thread.
            # Keep the owner alive until its bounded socket work closes; never
            # publish a late result or orphan an exception after cancellation.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if worker.done():
                try:
                    worker.result()
                except BaseException:
                    pass
            raise

    def read_order_sync(self, customer_id: str, order_id: str) -> dict[str, Any]:
        if not isinstance(customer_id, str) or customer_id not in CUSTOMER_IDS:
            raise DemoStoreError("demo_invalid_customer")
        if not isinstance(order_id, str) or not ORDER_PATTERN.fullmatch(order_id):
            raise DemoStoreError("demo_invalid_order")
        base = {"source": SOURCE, "is_demo": True, "as_of": load_fixture()["as_of"], "status": "not_found", "order": None}
        connection = None
        deadline = time.monotonic() + IO_TIMEOUT_SECONDS

        def deadline_check():
            if time.monotonic() >= deadline:
                raise DemoStoreError("demo_store_timeout")

        try:
            connection = self._connector(self.config)
            with connection.cursor() as cursor:
                deadline_check()
                cursor.execute("START TRANSACTION READ ONLY")
                verify_marker(cursor)
                deadline_check()
                columns = ", ".join(TABLE_COLUMNS["orders"])
                cursor.execute(f"SELECT {columns} FROM sf_demo_orders WHERE order_id = %s AND customer_id = %s", (order_id, customer_id))
                order = cursor.fetchone()
                if order is None:
                    return base
                _validate_row("orders", order)
                if order["order_id"] != order_id or order["customer_id"] != customer_id:
                    raise DemoStoreError("demo_store_invalid_record")
                for table in ("items", "payments", "shipments", "entitlements", "refunds"):
                    deadline_check()
                    names = TABLE_COLUMNS[table]
                    cursor.execute(f"SELECT {', '.join(names)} FROM {TABLE_PREFIX}{table} WHERE order_id = %s ORDER BY {names[0]} LIMIT {MAX_RELATED_ROWS + 1}", (order_id,))
                    raw_rows = cursor.fetchall()
                    # DictCursor may return an empty tuple while nonempty
                    # results are lists. Normalize only these legal driver
                    # containers at the business boundary; unknown/invalid
                    # results are not an empty collection.
                    if type(raw_rows) not in (list, tuple) or len(raw_rows) > MAX_RELATED_ROWS:
                        raise DemoStoreError("demo_store_invalid_record")
                    rows = []
                    for raw_row in raw_rows:
                        if type(raw_row) is not dict:
                            raise DemoStoreError("demo_store_invalid_record")
                        row = dict(raw_row)
                        if table == "entitlements" and type(row.get("auto_renew")) is int and row["auto_renew"] in (0, 1):
                            row["auto_renew"] = bool(row["auto_renew"])
                        _validate_row(table, row)
                        if row["order_id"] != order_id:
                            raise DemoStoreError("demo_store_invalid_record")
                        rows.append(row)
                    order[table] = rows
                deadline_check()
                _validate_order_relations(order)
                return {**base, "status": "found", "order": order}
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
