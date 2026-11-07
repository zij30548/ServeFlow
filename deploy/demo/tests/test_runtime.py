"""Offline bootstrap/operation checks; never setuid, exec, network or real SQL."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("protected_demo_runtime_tests", ROOT / "deploy/demo/runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("ordinary runtime tests forbid privileged/external operations")
    monkeypatch.setattr(runtime.os, "environ", {})
    for name in ("setuid", "setgid", "setgroups", "chown", "fchown", "execv"):
        monkeypatch.setattr(runtime.os, name, denied)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(sys, "path", [str(ROOT / "backend"), *sys.path])
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})


@pytest.fixture
def configuration():
    return {"version": 1, "public_origin": "https://demo.example.test", "gateway_bearer": "a" * 64,
        "api_key": "controlled-not-a-real-key", "model_origin": "https://dashscope.aliyuncs.com/apps/anthropic",
        "model": "qwen3.7-flash-2026-07-15", "session_key_base64": base64.b64encode(b"s" * 32).decode(),
        "mysql_reader_password": "r" * 40, "model_limit": 100}


def fake_owner(monkeypatch, path, *, uid, gid):
    original = Path.stat
    def controlled(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if self != path:
            return value
        return SimpleNamespace(st_mode=value.st_mode, st_uid=uid, st_gid=gid)
    monkeypatch.setattr(Path, "stat", controlled)


def test_only_empty_root_owned_volume_is_adjusted_chmod_before_chown(tmp_path, monkeypatch):
    target = tmp_path / "new-volume"
    target.mkdir(mode=0o755)
    fake_owner(monkeypatch, target, uid=0, gid=0)
    events = []
    chmod = os.chmod
    monkeypatch.setattr(runtime.os, "chmod", lambda path, mode: (events.append(("chmod", path, mode)), chmod(path, mode)))
    monkeypatch.setattr(runtime.os, "chown", lambda path, uid, gid: events.append(("chown", path, uid, gid)))
    runtime._owned_directory(target, 10001, 10001, new_empty=True)
    assert events == [("chmod", target, 0o700), ("chown", target, 10001, 10001)]
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_existing_owned_private_volume_is_verified_without_metadata_changes(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir(mode=0o700)
    (target / "counter").write_bytes(b"already-used-permits")
    fake_owner(monkeypatch, target, uid=10001, gid=10001)
    monkeypatch.setattr(runtime.os, "chmod", lambda *a: pytest.fail("existing metadata changed"))
    runtime._owned_directory(target, 10001, 10001, new_empty=True)
    assert (target / "counter").read_bytes() == b"already-used-permits"


@pytest.mark.parametrize("uid,gid,mode,occupied", [
    (0, 0, 0o755, True), (501, 20, 0o700, False), (10001, 10001, 0o755, True),
])
def test_nonempty_or_wrong_owner_volume_is_never_recursively_repaired(tmp_path, monkeypatch, uid, gid, mode, occupied):
    target = tmp_path / "existing"
    target.mkdir(mode=mode)
    if occupied:
        (target / "old-data").write_bytes(b"keep")
    fake_owner(monkeypatch, target, uid=uid, gid=gid)
    monkeypatch.setattr(runtime.os, "chmod", lambda *a: pytest.fail("old permissions changed"))
    with pytest.raises(runtime.ConfigurationError, match="inspection"):
        runtime._owned_directory(target, 10001, 10001, new_empty=True)
    if occupied:
        assert (target / "old-data").read_bytes() == b"keep"


def test_symlink_storage_root_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(runtime.ConfigurationError, match="unsafe"):
        runtime._owned_directory(link, 10001, 10001, new_empty=True)


def test_private_secret_is_exclusive_0600_before_ownership_drop(tmp_path, monkeypatch):
    target = tmp_path / "session.key"
    events = []
    def chown(fd, uid, gid):
        events.append((stat.S_IMODE(os.fstat(fd).st_mode), uid, gid))
    monkeypatch.setattr(runtime.os, "fchown", chown)
    runtime._private_write(target, b"s" * 32, 10001, 10001)
    assert target.read_bytes() == b"s" * 32
    assert events == [(0o600, 10001, 10001)]
    with pytest.raises(FileExistsError):
        runtime._private_write(target, b"replacement", 10001, 10001)
    assert target.read_bytes() == b"s" * 32


def test_private_write_does_not_follow_or_replace_existing_link(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"keep")
    target = tmp_path / "link"
    target.symlink_to(original)
    with pytest.raises(OSError):
        runtime._private_write(target, b"replacement", 10001, 10001)
    assert original.read_bytes() == b"keep"


def test_drop_clears_groups_and_gid_before_uid(monkeypatch):
    events = []
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime.os, "setgroups", lambda groups: events.append(("groups", groups)))
    monkeypatch.setattr(runtime.os, "setgid", lambda value: events.append(("gid", value)))
    monkeypatch.setattr(runtime.os, "setuid", lambda value: events.append(("uid", value)))
    runtime._drop(10001, 10001)
    assert events == [("groups", []), ("gid", 10001), ("uid", 10001)]


def test_nonroot_bootstrap_and_failed_group_clear_never_start_server(monkeypatch):
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 501)
    with pytest.raises(runtime.ConfigurationError, match="container_root"):
        runtime._drop(10001, 10001)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime.os, "setgroups", Mock(side_effect=PermissionError("controlled")))
    with pytest.raises(PermissionError):
        runtime._drop(10001, 10001)


@pytest.fixture
def prepare(monkeypatch, tmp_path, configuration):
    monkeypatch.setattr(runtime, "RUN", tmp_path / "run")
    monkeypatch.setattr(runtime, "DATA", tmp_path / "data")
    runtime.os.environ["DEMO_PUBLIC_ORIGIN"] = configuration["public_origin"]
    events = []
    monkeypatch.setattr(runtime, "_owned_directory", lambda path, uid, gid, **kw: events.append(("directory", path, uid, gid, kw)))
    monkeypatch.setattr(runtime, "_private_write", lambda path, raw, uid, gid: events.append(("private", path, raw, uid, gid)))
    monkeypatch.setattr(runtime, "_drop", lambda uid, gid: events.append(("drop", uid, gid)))
    monkeypatch.setattr(runtime.os, "chdir", lambda value: events.append(("chdir", value)))
    return events


def test_api_bootstrap_never_reads_root_mysql_secret_and_drops_before_application(prepare, monkeypatch, configuration):
    monkeypatch.setattr(runtime, "private_read", lambda *a, **k: pytest.fail("API read root secret"))
    result = runtime.prepare_api(configuration)
    assert "SERVEFLOW_DEMO_ADMIN_MYSQL_URL" not in result
    assert result["ANTHROPIC_API_KEY"] == configuration["api_key"]
    assert result["HOME"] == "/home/serveflow" and result["MCP_EXTERNAL_ENABLED"] == "false"
    assert result["SERVEFLOW_DEMO_MYSQL_URL"].startswith("mysql+pymysql://sf_demo_reader:")
    assert "@mysql:3306/serveflow_demo_customer" in result["SERVEFLOW_DEMO_MYSQL_URL"]
    assert json.loads(result["ECHOMIND_API_TOKENS"])[configuration["gateway_bearer"]]["role"] == "user"
    assert prepare[2][0] == "private" and prepare[2][2] == b"s" * 32
    assert [event[0] for event in prepare] == ["directory", "directory", "private", "drop", "chdir"]


def test_api_bootstrap_removes_inherited_mysql_admin_url(prepare, configuration):
    runtime.os.environ["SERVEFLOW_DEMO_ADMIN_MYSQL_URL"] = "mysql://old-root:old-private@localhost/old"
    runtime.prepare_api(configuration)
    assert "SERVEFLOW_DEMO_ADMIN_MYSQL_URL" not in runtime.os.environ


@pytest.mark.parametrize("model_origin", ["https://dashscope.aliyuncs.com/apps/anthropic",
    "https://workspace-test.cn-beijing.maas.aliyuncs.com/apps/anthropic"])
def test_explicit_qwen_opt_in_derives_same_workspace_and_keeps_config_and_quota(prepare, configuration, model_origin):
    configuration["model_origin"] = model_origin
    before = dict(configuration)
    runtime.os.environ["DEMO_INTENT_EMBEDDING_PROVIDER"] = "qwen"
    result = runtime.prepare_api(configuration)
    from core.qwen_embedding import canonical_base_url
    assert result["INTENT_EMBEDDING_PROVIDER"] == "qwen"
    assert result["QWEN_EMBEDDING_API_KEY"] == configuration["api_key"]
    assert canonical_base_url(result["QWEN_EMBEDDING_BASE_URL"]) == model_origin.removesuffix("/apps/anthropic") + "/compatible-mode/v1"
    assert result["ANTHROPIC_MODEL"] == configuration["model"]
    assert result["ANTHROPIC_BASE_URL"] == model_origin
    assert result["SERVEFLOW_DEMO_QUOTA_DIR"] == str(runtime.DATA / "usage")
    assert result["SERVEFLOW_DEMO_MODEL_LIMIT"] == "100" and configuration == before
    assert result["CHROMA_PERSIST_DIRECTORY"] == str(runtime.DATA / "chroma")


def test_default_api_keeps_remote_embedding_disabled(prepare, configuration):
    result = runtime.prepare_api(configuration)
    assert result["INTENT_EMBEDDING_PROVIDER"] == "local"
    assert "QWEN_EMBEDDING_API_KEY" not in result and "QWEN_EMBEDDING_BASE_URL" not in result


def test_qwen_cache_uses_existing_private_app_volume(prepare, configuration):
    runtime.os.environ['DEMO_INTENT_EMBEDDING_PROVIDER']='qwen'
    result=runtime.prepare_api(configuration)
    assert result['QWEN_INTENT_CACHE_DIRECTORY']==str(runtime.DATA/'intent-templates')


@pytest.mark.parametrize("key,value", [("DEMO_INTENT_EMBEDDING_PROVIDER", "siliconflow"),
    ("DEMO_INTENT_EMBEDDING_PROVIDER", "QWEN"), ("DEMO_INTENT_EMBEDDING_PROVIDER", ""),
    ("QWEN_EMBEDDING_BASE_URL", "https://evil.test/compatible-mode/v1"),
    ("QWEN_EMBEDDING_API_KEY", "unexpected-key")])
def test_unreviewed_embedding_configuration_stops_before_private_state_work(prepare, configuration, key, value):
    runtime.os.environ[key] = value
    with pytest.raises(runtime.ConfigurationError, match="optional_integration"):
        runtime.prepare_api(configuration)
    assert prepare == []


def test_initializer_alone_receives_explicit_new_root_secret(prepare, monkeypatch, configuration):
    read = Mock(return_value=b"p" * 40)
    monkeypatch.setattr(runtime, "private_read", read)
    result = runtime.prepare_api(configuration, admin_secret=True)
    read.assert_called_once_with(Path("/run/secrets/mysql_root_password"), maximum=128)
    assert result["SERVEFLOW_DEMO_ADMIN_MYSQL_URL"] == "mysql+pymysql://root:" + "p" * 40 + "@mysql:3306/serveflow_demo_customer"


@pytest.mark.parametrize("key,value", [("DEMO_PUBLIC_ORIGIN", "https://other.example.test"),
    ("MCP_EXTERNAL_ENABLED", "true"), ("ALERT_WEBHOOK_URL", "https://example.invalid/hook"),
    ("SILICONFLOW_API_KEY", "unexpected"), ("INTENT_EMBEDDING_PROVIDER", "remote")])
def test_unreviewed_origin_or_paid_integrations_stop_before_private_state_work(prepare, configuration, key, value):
    runtime.os.environ[key] = value
    with pytest.raises(runtime.ConfigurationError, match="optional_integration"):
        runtime.prepare_api(configuration)
    assert prepare == []


def test_web_secrets_are_separate_and_drop_precedes_nginx_exec(tmp_path, monkeypatch, configuration):
    gateway = {key: configuration[key] for key in ("version", "public_origin", "gateway_bearer")}
    runtime.os.environ["DEMO_PUBLIC_ORIGIN"] = gateway["public_origin"]
    template = tmp_path / "nginx.conf.template"
    template.write_text((ROOT / "deploy/demo/nginx.conf.template").read_text())
    monkeypatch.setattr(runtime, "HERE", tmp_path)
    loaded, reads, writes, events = [], [], [], []
    def load(path, **kw):
        loaded.append((path, kw));return gateway
    monkeypatch.setattr(runtime, "load", load)
    monkeypatch.setattr(runtime, "private_read", lambda path, **kw: (reads.append(path) or b"customer:$2y$10$" + b"A" * 53 + b"\n"))
    monkeypatch.setattr(runtime, "_owned_directory", lambda *a, **k: events.append("directory"))
    monkeypatch.setattr(runtime, "_private_write", lambda path, raw, *a: writes.append((path, raw)))
    monkeypatch.setattr(runtime, "_drop", lambda uid, gid: events.append(("drop", uid, gid)))
    monkeypatch.setattr(runtime.os, "execv", lambda path, args: events.append(("exec", path, args)))
    runtime.web()
    assert loaded == [(Path("/run/secrets/gateway_config"), {"api": False})]
    assert reads == [Path("/run/secrets/demo_htpasswd")]
    assert [p.name for p, raw in writes] == ["htpasswd", "nginx.conf", "proxy.inc"]
    assert configuration["api_key"].encode() not in b"".join(raw for p, raw in writes)
    proxy = writes[-1][1].decode()
    assert 'proxy_set_header Authorization "Bearer ' + configuration["gateway_bearer"] + '";' in proxy
    assert "X-ServeFlow-Demo-User $remote_user" in proxy
    assert "proxy_ignore_client_abort off" in proxy and "proxy_request_buffering off" in proxy
    assert events[-2] == ("drop", 101, 101) and events[-1][0:2] == ("exec", "/usr/sbin/nginx")
    assert events[-1][2] == ["nginx", "-e", "stderr", "-c", str(runtime.RUN / "nginx.conf"), "-g", "daemon off;"]


@pytest.mark.parametrize("rows", [b"", b"customer:not-a-bcrypt-hash\n",
    (b"customer:$2y$10$" + b"A" * 53 + b"\n") * 2,
    b"customer:$2y$12$" + b"A" * 53 + b"\n"])
def test_invalid_or_duplicate_htpasswd_never_writes_or_executes(configuration, monkeypatch, rows):
    runtime.os.environ["DEMO_PUBLIC_ORIGIN"] = configuration["public_origin"]
    monkeypatch.setattr(runtime, "load", lambda *a, **k: configuration)
    monkeypatch.setattr(runtime, "private_read", lambda *a, **k: rows)
    monkeypatch.setattr(runtime, "_owned_directory", lambda *a, **k: pytest.fail("invalid users reached filesystem"))
    with pytest.raises(runtime.ConfigurationError, match="password_file"):
        runtime.web()


class Cursor:
    def __init__(self, *, existed=False, users=(), grants=(), fail=None):
        self.existed, self.users, self.grants, self.fail = existed, users, grants, fail
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fail and sql.startswith(self.fail):
            raise RuntimeError("controlled SQL exception with sensitive input")
    def fetchone(self): return (runtime.DATABASE,) if self.existed else None
    def fetchall(self):
        return self.grants if self.calls[-1][0].startswith("SHOW GRANTS") else self.users


class Connection:
    def __init__(self, cursor): self.value, self.closed = cursor, False
    def cursor(self): return self.value
    def close(self): self.closed = True


@pytest.fixture
def database(monkeypatch, configuration):
    runtime.os.environ["ECHOMIND_ENV"] = "protected-demo"
    from demo_business import initialize, store
    import pymysql
    environment = runtime._api_environment(configuration)
    environment["SERVEFLOW_DEMO_ADMIN_MYSQL_URL"] = "mysql+pymysql://root:" + "p" * 40 + "@mysql:3306/serveflow_demo_customer"
    cursor = Cursor()
    connection = Connection(cursor)
    connect = Mock(return_value=connection)
    seed = Mock(return_value={"controlled": "fixture initialized"})
    verify = Mock(return_value={"controlled": "reader verified"})
    monkeypatch.setattr(pymysql, "connect", connect)
    monkeypatch.setattr(initialize, "initialize_database", seed)
    monkeypatch.setattr(runtime, "_verify_reader", verify)
    return environment, cursor, connection, connect, seed, verify


def test_existing_mysql_is_only_verified_not_created_reseeded_or_recredentialed(database, configuration):
    environment, cursor, connection, connect, seed, verify = database
    cursor.existed = True
    cursor.users = ((runtime.READER, "%"),)
    assert runtime.initialize_mysql(environment, configuration) == {"operation": "verified-existing-no-write"}
    assert len(cursor.calls) == 2 and all(sql.startswith("SELECT") for sql, _ in cursor.calls)
    seed.assert_not_called()
    verify.assert_called_once_with(environment)
    assert connection.closed and connect.call_count == 1


def test_new_mysql_explicitly_creates_fixed_schema_reader_then_initializes_and_verifies(database, configuration):
    environment, cursor, connection, connect, seed, verify = database
    assert runtime.initialize_mysql(environment, configuration) == {"controlled": "fixture initialized"}
    assert [sql.split()[0] for sql, _ in cursor.calls] == ["SELECT", "SELECT", "CREATE", "CREATE", "GRANT"]
    create_user = cursor.calls[3]
    assert create_user[1] == (runtime.READER, configuration["mysql_reader_password"])
    assert configuration["mysql_reader_password"] not in create_user[0]
    assert all("DROP" not in sql and "ALTER USER" not in sql for sql, _ in cursor.calls)
    assert connection.closed and seed.call_count == verify.call_count == 1


@pytest.mark.parametrize("existed,users", [(False, ((runtime.READER, "%"),)),
    (True, ((runtime.READER, "localhost"),)), (True, ((runtime.READER, "%"), (runtime.READER, "localhost")))])
def test_ambiguous_existing_reader_stops_without_writes_or_repair(database, configuration, existed, users):
    environment, cursor, connection, connect, seed, verify = database
    cursor.existed, cursor.users = existed, users
    with pytest.raises(runtime.ConfigurationError, match="requires_inspection"):
        runtime.initialize_mysql(environment, configuration)
    assert all(sql.startswith("SELECT") for sql, _ in cursor.calls)
    seed.assert_not_called()
    verify.assert_not_called()
    assert connection.closed


def test_partial_mysql_ddl_failure_is_preserved_never_dropped_or_retried(database, configuration):
    environment, cursor, connection, connect, seed, verify = database
    cursor.fail = "CREATE USER"
    with pytest.raises(RuntimeError):
        runtime.initialize_mysql(environment, configuration)
    assert connection.closed and connect.call_count == 1
    assert sum(sql.startswith("CREATE DATABASE") for sql, _ in cursor.calls) == 1
    assert not any(sql.startswith(("DROP", "ALTER")) for sql, _ in cursor.calls)
    seed.assert_not_called()
    verify.assert_not_called()


@pytest.mark.parametrize("privileges,accepted", [
    (["GRANT USAGE ON *.* TO `reader`@`%`", "GRANT SELECT ON `serveflow_demo_customer`.* TO `reader`@`%`"], True),
    (["GRANT ALL PRIVILEGES ON `serveflow_demo_customer`.* TO `reader`@`%`"], False),
    (["GRANT SELECT ON *.* TO `reader`@`%`"], False),
    (["GRANT SELECT ON `serveflow_demo_customer`.* TO `reader`@`%` WITH GRANT OPTION"], False),
    (["GRANT SELECT ON `other_database`.* TO `reader`@`%`"], False),
    (["GRANT USAGE ON *.* TO `reader`@`%`"], False),
])
def test_reader_grants_require_exact_select_scope_and_always_close(configuration, monkeypatch, privileges, accepted):
    runtime.os.environ["ECHOMIND_ENV"] = "protected-demo"
    from demo_business import initialize, store
    verified = Mock(return_value={"schema": "controlled verified"})
    monkeypatch.setattr(initialize, "verify_database", verified)
    connection = Connection(Cursor(grants=[{"grant": text} for text in privileges]))
    monkeypatch.setattr(store, "connect", lambda *a: connection)
    if accepted:
        assert runtime._verify_reader(runtime._api_environment(configuration)) == {"schema": "controlled verified"}
    else:
        with pytest.raises(runtime.ConfigurationError, match="select_only"):
            runtime._verify_reader(runtime._api_environment(configuration))
    assert connection.closed and verified.call_count == 1


@pytest.fixture
def dispatch(tmp_path, monkeypatch, configuration):
    monkeypatch.setattr(runtime, "DATA", tmp_path)
    monkeypatch.setattr(runtime, "load", lambda *a, **k: configuration)
    environment = runtime._api_environment(configuration)
    monkeypatch.setattr(runtime, "prepare_api", lambda *a, **k: environment)
    from core import protected_demo_quota as quota
    return tmp_path, quota, environment


def test_repeated_quota_initialization_verifies_existing_counter_instead_of_reset(dispatch, monkeypatch):
    directory, quota, _ = dispatch
    (directory / "usage").mkdir()
    marker = directory / "usage/keep-counter"
    marker.write_bytes(b"used=17")
    verify = Mock(return_value={"used": 17})
    monkeypatch.setattr(quota, "verify", verify)
    monkeypatch.setattr(quota, "initialize", lambda *a, **k: pytest.fail("existing counter reset"))
    assert runtime.main(["quota-init"]) == 0
    verify.assert_called_once_with()
    assert marker.read_bytes() == b"used=17"


def test_corrupted_existing_quota_is_not_reinitialized(dispatch, monkeypatch, capsys):
    directory, quota, _ = dispatch
    (directory / "usage").mkdir()
    marker = directory / "usage/keep-counter"
    marker.write_bytes(b"corrupt evidence, preserve")
    monkeypatch.setattr(quota, "verify", Mock(side_effect=RuntimeError("controlled corrupt ledger")))
    monkeypatch.setattr(quota, "initialize", lambda *a, **k: pytest.fail("corrupt counter reset"))
    assert runtime.main(["quota-init"]) == 2
    assert marker.read_bytes() == b"corrupt evidence, preserve"
    assert "controlled corrupt ledger" not in capsys.readouterr().err


def test_quota_initializer_runs_once_only_for_absent_directory(dispatch, monkeypatch):
    _, quota, _ = dispatch
    initialize = Mock(return_value={"used": 0})
    monkeypatch.setattr(quota, "initialize", initialize)
    monkeypatch.setattr(quota, "verify", lambda *a, **k: pytest.fail("absent counter verified"))
    assert runtime.main(["quota-init"]) == 0
    initialize.assert_called_once_with()


def test_template_ops_are_explicit_and_never_initialize_quota_or_touch_sql(dispatch, monkeypatch, capsys):
    _,quota,_=dispatch
    from core import intent_templates
    calls=[]
    async def prepare_templates(*,max_requests,diagnostics):
        calls.append(max_requests)
        return {'ready':True,'new_requests':3}
    monkeypatch.setattr(intent_templates,'prepare_templates',prepare_templates)
    monkeypatch.setattr(intent_templates,'template_status',lambda:{'state':'unprepared'})
    monkeypatch.setattr(quota,'initialize',lambda:pytest.fail('quota reset'))
    monkeypatch.setattr(runtime,'_verify_reader',lambda *a:pytest.fail('unexpected SQL'))
    assert runtime.main(['templates-status'])==0 and calls==[]
    assert runtime.main(['prepare-templates','--max-requests','3'])==0 and calls==[3]
    assert 'new_requests' in capsys.readouterr().out


def test_prepare_templates_requires_cap_before_any_private_bootstrap(monkeypatch):
    monkeypatch.setattr(runtime,'load',lambda *a,**k:pytest.fail('private bootstrap before explicit cap'))
    with pytest.raises(SystemExit):runtime.main(['prepare-templates'])


def test_api_starts_one_worker_only_after_schema_quota_and_cache_checks(dispatch, monkeypatch):
    _, quota, _ = dispatch
    events = []
    monkeypatch.setattr(runtime, "_verify_reader", lambda env: events.append("schema") or {})
    monkeypatch.setattr(quota, "verify", lambda: events.append("quota") or {})
    import model_cache
    monkeypatch.setattr(model_cache, "verify", lambda path: events.append("cache"))
    class ExecIntercept(BaseException): pass
    def execv(path, args):
        events.append(("exec", path, args));raise ExecIntercept()
    monkeypatch.setattr(runtime.os, "execv", execv)
    with pytest.raises(ExecIntercept):
        runtime.main(["api"])
    assert events[:3] == ["schema", "quota", "cache"]
    args = events[3][2]
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--port") + 1] == "8816"
    assert args[args.index("--workers") + 1] == "1" and "--no-access-log" in args


def test_operation_failure_does_not_leak_credentials_or_original_exception(dispatch, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "initialize_mysql", Mock(side_effect=RuntimeError("mysql://root:private-secret@mysql/private")))
    assert runtime.main(["init"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "private-secret" not in output.err and "mysql://" not in output.err
    assert "inspect_no_automatic_retry" in output.err


@pytest.mark.parametrize("web_mode,status,accepted", [(False, 200, True), (False, 503, False),
    (True, 401, True), (True, 200, False)])
def test_health_checks_no_auth_or_model_and_require_web_challenge(configuration, monkeypatch, web_mode, status, accepted):
    calls = []
    connection = SimpleNamespace(
        request=lambda method, path, headers: calls.append((method, path, headers)),
        getresponse=lambda: SimpleNamespace(status=status, read=lambda limit: calls.append(("read", limit))),
        close=lambda: calls.append("closed"))
    monkeypatch.setattr(runtime.http.client, "HTTPConnection", lambda host, port, timeout: (calls.append((host, port, timeout)) or connection))
    def load(path, *, api):
        assert web_mode and path == Path("/run/secrets/gateway_config") and api is False
        return configuration
    monkeypatch.setattr(runtime, "load", load)
    if accepted:
        runtime.health(web_mode)
    else:
        with pytest.raises(runtime.ConfigurationError, match="not_ready"):
            runtime.health(web_mode)
    assert calls[0] == ("127.0.0.1", 8080 if web_mode else 8816, 3)
    method, path, headers = calls[1]
    assert method == "GET" and path == ("/" if web_mode else "/health")
    assert headers == ({"Host": "demo.example.test"} if web_mode else {})
    assert calls[-1] == "closed"
