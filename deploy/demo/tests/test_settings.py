"""Strict config inputs on temporary files; no credentials or external calls."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import socket

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("demo_deployment_settings_tests", ROOT / "deploy/demo/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs): raise AssertionError("deployment settings tests cannot contact a service")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture
def config():
    return {"version": 1, "public_origin": "https://demo.example.test", "gateway_bearer": "a" * 64,
        "api_key": "controlled-placeholder-only", "model_origin": "https://dashscope.aliyuncs.com/apps/anthropic",
        "model": settings.MODEL, "session_key_base64": base64.b64encode(b"x" * 32).decode(),
        "mysql_reader_password": "controlled-" + "p" * 32, "model_limit": 100}


@pytest.fixture
def secret(tmp_path, config):
    path = tmp_path.resolve() / "api.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("value", ["https://demo.example.test", "https://demo.example.test:9443", "https://localhost:8443"])
def test_canonical_https_origins_only(value):
    assert settings.origin(value) == value


@pytest.mark.parametrize("value", ["http://demo.example.test", "https://user:pass@demo.example.test",
    "https://demo.example.test/", "https://demo.example.test/x", "https://demo.example.test?x=1",
    "https://demo.example.test#x", "https://Demo.example.test", "https://demo.example.test:443",
    "https://demo.example.test:99999", "https://demo.example.test.", "https://*.example.test", "https://demo.example.test\n", None])
def test_bad_origins_reject_without_normalizing_or_guessing(value):
    with pytest.raises(settings.ConfigurationError, match="demo_https_origin_required"):
        settings.origin(value)


@pytest.mark.parametrize("value", ["https://dashscope.aliyuncs.com/apps/anthropic",
    "https://workspace-123.cn-beijing.maas.aliyuncs.com/apps/anthropic"])
def test_official_beijing_workspace_and_standard_endpoints(value):
    assert settings.model_origin(value) == value


@pytest.mark.parametrize("value", ["http://dashscope.aliyuncs.com/apps/anthropic",
    "https://example.test/apps/anthropic", "https://a.cn-shanghai.maas.aliyuncs.com/apps/anthropic",
    "https://dashscope.aliyuncs.com:443/apps/anthropic", "https://u:p@dashscope.aliyuncs.com/apps/anthropic",
    "https://dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope.aliyuncs.com/apps/anthropic/",
    "https://dashscope.aliyuncs.com/apps/anthropic?x=1", "https://dashscope.aliyuncs.com/apps/anthropic#x"])
def test_deployment_endpoint_must_be_canonical(value):
    with pytest.raises(settings.ConfigurationError, match="demo_beijing_endpoint_required"):
        settings.model_origin(value)


def test_api_and_gateway_whitelists_keep_model_key_out_of_gateway(config, secret):
    assert settings.load(secret, api=True, host_owner=True) == config
    gateway = {k: config[k] for k in settings.GATEWAY_FIELDS}
    assert settings.validate(gateway, api=False) == gateway
    assert "api_key" not in gateway and "mysql_reader_password" not in gateway
    with pytest.raises(settings.ConfigurationError): settings.validate(config, api=False)
    with pytest.raises(settings.ConfigurationError): settings.validate(gateway, api=True)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra="unknown"), lambda c: c.pop("model_limit"),
    lambda c: c.update(version=True), lambda c: c.update(model="arbitrary-model"),
    lambda c: c.update(model_limit=0), lambda c: c.update(model_limit=1001),
    lambda c: c.update(model_limit=True), lambda c: c.update(gateway_bearer="z" * 64),
    lambda c: c.update(session_key_base64="not base64"), lambda c: c.update(session_key_base64=base64.b64encode(b"x"*31).decode()),
    lambda c: c.update(api_key="short"), lambda c: c.update(api_key="private-header\nvalue"),
    lambda c: c.update(mysql_reader_password="short"), lambda c: c.update(mysql_reader_password="p"*31+" ")])
def test_unknown_fields_wrong_types_and_resource_limits_reject(config, mutate):
    mutate(config)
    with pytest.raises(settings.ConfigurationError): settings.validate(config, api=True)


@pytest.mark.parametrize("raw", [b'{"version":1,"version":1}', b'{"version":NaN}',
    b'{"version":Infinity}', b'{"version":1e999}', b'{', b'\xff'])
def test_json_duplicates_nonfinite_and_invalid_encoding_are_rejected(secret, raw):
    secret.write_bytes(raw)
    with pytest.raises(settings.ConfigurationError): settings.load(secret, api=True, host_owner=True)


def test_file_limits_relative_paths_and_missing_files_reject(secret, tmp_path):
    with pytest.raises(settings.ConfigurationError): settings.private_read(Path("relative.json"), host_owner=True)
    with pytest.raises(settings.ConfigurationError): settings.private_read(tmp_path / "missing", host_owner=True)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, maximum=8, host_owner=True)
    secret.write_bytes(b"")
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=True)


def test_host_private_file_permissions_and_owner_are_enforced(secret, monkeypatch):
    secret.chmod(0o640)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=True)
    secret.chmod(0o600)
    actual = os.geteuid()
    monkeypatch.setattr(settings.os, "geteuid", lambda: actual + 1)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=True)


def test_bind_secret_may_be_readable_but_never_group_writable(secret):
    secret.chmod(0o444)
    assert settings.private_read(secret, host_owner=False)
    secret.chmod(0o664)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=False)


def test_links_hardlinks_and_special_files_reject(secret, tmp_path):
    symbolic = tmp_path.resolve() / "symbolic"
    symbolic.symlink_to(secret)
    with pytest.raises(settings.ConfigurationError): settings.private_read(symbolic, host_owner=True)
    hard = tmp_path.resolve() / "hard"
    os.link(secret, hard)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=True)
    hard.unlink()
    fifo = tmp_path.resolve() / "fifo"
    os.mkfifo(fifo, mode=0o600)
    with pytest.raises(settings.ConfigurationError): settings.private_read(fifo, host_owner=True)


def test_host_owner_secret_rejects_ancestor_symlink(secret, tmp_path):
    parent_link = tmp_path.resolve() / "linked-parent"
    parent_link.symlink_to(secret.parent, target_is_directory=True)
    with pytest.raises(settings.ConfigurationError): settings.private_read(parent_link / secret.name, host_owner=True)


def test_secret_special_permission_bits_reject(secret):
    secret.chmod(0o4600)
    with pytest.raises(settings.ConfigurationError): settings.private_read(secret, host_owner=True)
