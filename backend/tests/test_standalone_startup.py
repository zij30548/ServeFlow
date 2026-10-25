"""Local startup/configuration checks; no native client or network is started."""
import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import run


ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_guards_native_import_and_uses_backend_dotenv_and_loopback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9999")
    events = []
    app = object()
    previous_import = builtins.__import__

    def configured_dotenv(path, *, override):
        assert path == ROOT / ".env" and override is False
        assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        assert sys.dont_write_bytecode is True
        assert Path.cwd() == ROOT
        events.append("dotenv")

    def start(application, **options):
        assert application is app
        assert options == {"host": "127.0.0.1", "port": 8816, "workers": 1, "access_log": False}
        events.append("server")

    def controlled_import(name, *args, **kwargs):
        if name == "dotenv":
            return SimpleNamespace(load_dotenv=configured_dotenv)
        if name == "api.main":
            assert events == ["dotenv"]
            assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
            events.append("api")
            return SimpleNamespace(app=app)
        if name == "uvicorn":
            return SimpleNamespace(run=start)
        return previous_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    run.main()
    assert events == ["dotenv", "api", "server"]


@pytest.mark.parametrize("configured", ["0", "false", "", "2"])
def test_conflicting_native_guard_stops_before_configuration_or_app(monkeypatch, configured):
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", configured)
    with pytest.raises(RuntimeError, match="must be 1"):
        run.main()


def test_late_native_guard_stops_without_trying_to_reinitialize(monkeypatch):
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", object())
    with pytest.raises(RuntimeError, match="initialized before"):
        run.main()


def _defaults(name):
    """Evaluate only the actual getenv default expression, without importing API."""
    tree = ast.parse((ROOT / "api/main.py").read_text())
    results = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
                and node.func.attr == "getenv" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == name):
            expression = ast.Expression(node.args[1])
            results.append(eval(compile(expression, "<configuration-default>", "eval"), {
                "str": str, "pathlib": SimpleNamespace(Path=Path), "_ROOT": str(ROOT),
            }))
    return results


def test_both_api_and_cli_disable_external_mcp_by_default_and_ship_no_remote_servers():
    assert _defaults("MCP_EXTERNAL_ENABLED") == ["false", "false"]
    assert json.loads((ROOT / "config/mcp_servers.json").read_text()) == {"servers": []}


def test_data_defaults_follow_migrated_backend_not_cwd_or_old_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _defaults("CHROMA_PERSIST_DIRECTORY") == [str(ROOT / "data/chroma")] * 4
    assert _defaults("EVAL_BASELINE_PATH") == [str(ROOT / "data/eval/baseline.json")]
    assert _defaults("REDIS_URL") == ["redis://127.0.0.1:6379/0"] * 2
    assert _defaults("CHROMA_HOST") == ["127.0.0.1"] * 4


def test_direct_api_entry_also_defaults_to_loopback_and_the_same_port():
    assert _defaults("API_HOST") == ["127.0.0.1"]
    assert _defaults("API_PORT") == ["8816"]


@pytest.mark.parametrize("name", ["agents", "api", "core", "memory", "monitor", "evaluation",
                                  "tooling", "demo_business", "knowledge", "skills", "config"])
def test_local_packages_have_explicit_identity(name):
    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert Path(spec.origin) == ROOT / name / "__init__.py"
    assert list(spec.submodule_search_locations) == [str(ROOT / name)]
