"""Static deployment boundary checks; not a Docker/Linux runtime acceptance."""
from pathlib import Path
import shlex

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy/demo"


def compose():
    return yaml.safe_load((DEPLOY / "compose.yaml").read_text())


def test_only_loopback_gateway_is_published_and_data_network_is_internal():
    document = compose()
    services = document["services"]
    assert set(services) == {"web", "api", "redis", "mysql", "ops"}
    assert services["web"]["ports"] == ["127.0.0.1:8080:8080"]
    assert all("ports" not in item for name, item in services.items() if name != "web")
    assert services["web"]["networks"] == ["gateway"]
    assert set(services["api"]["networks"]) == {"gateway", "data"}
    assert all(services[name]["networks"] == ["data"] for name in ("mysql", "redis", "ops"))
    assert document["networks"]["data"]["internal"] is True


def test_browser_gateway_has_no_model_or_database_secrets():
    document = compose()
    services = document["services"]
    assert set(services["web"]["secrets"]) == {"gateway_config", "demo_htpasswd"}
    assert services["api"]["secrets"] == ["api_config"]
    assert set(services["ops"]["secrets"]) == {"api_config", "mysql_root_password"}
    assert services["mysql"]["secrets"] == ["mysql_root_password"]
    for name in ("api", "web", "ops"):
        assert set(services[name]["environment"]) == {"DEMO_PUBLIC_ORIGIN"}
    for row in document["secrets"].values():
        assert set(row) == {"file"}
        assert row["file"].startswith("${DEMO_PRIVATE_DIR:?")
    assert {name: Path(row["file"]).name for name, row in document["secrets"].items()} == {
        "api_config": "api.json", "gateway_config": "gateway.json",
        "demo_htpasswd": "htpasswd", "mysql_root_password": "mysql-root-password"}


def test_business_initialization_is_explicit_and_not_a_startup_hook():
    document = compose()
    services = document["services"]
    assert services["api"]["command"] == ["api"]
    assert services["web"]["command"] == ["web"]
    assert services["ops"]["profiles"] == ["tools"]
    assert services["ops"]["restart"] == "no"
    assert services["ops"]["command"] == ["verify"]
    assert set(services["mysql"]["environment"]) == {"MYSQL_ROOT_PASSWORD_FILE", "MYSQL_ROOT_HOST"}
    assert "MYSQL_DATABASE" not in services["mysql"]["environment"]
    assert "MYSQL_USER" not in services["mysql"]["environment"]


def test_runtime_code_is_readonly_and_only_dedicated_named_data_is_writable():
    document = compose()
    services = document["services"]
    for name in ("api", "web", "ops"):
        row = services[name]
        assert row["read_only"] is True
        assert row["cap_drop"] == ["ALL"]
        assert set(row["cap_add"]) == {"CHOWN", "SETUID", "SETGID", "DAC_OVERRIDE"}
        assert row["security_opt"] == ["no-new-privileges:true"]
        assert all("noexec" in mount and "nosuid" in mount and "nodev" in mount for mount in row["tmpfs"])
    assert services["api"]["volumes"] == services["ops"]["volumes"] == ["app_data:/var/lib/serveflow"]
    assert services["redis"]["volumes"] == ["redis_data:/data"]
    assert services["mysql"]["volumes"] == ["mysql_data:/var/lib/mysql"]
    assert set(document["volumes"]) == {"app_data", "redis_data", "mysql_data"}
    assert "volumes" not in services["web"]


def test_health_and_logs_do_not_put_credentials_in_commands():
    for name, row in compose()["services"].items():
        assert row["logging"] == {"driver": "json-file", "options": {"max-size": "2m", "max-file": "3"}}
        assert row["pids_limit"] > 0 and row["cpus"] > 0 and row["mem_limit"]
        health = row.get("healthcheck")
        if health is None:
            assert name == "ops"
            continue
        assert health["test"][0] == "CMD"
        assert not any("password" in value.lower() or "bearer" in value.lower() for value in health["test"])
    assert compose()["services"]["web"]["healthcheck"]["test"][-1] == "web-health"
    assert compose()["services"]["api"]["healthcheck"]["test"][-1] == "api-health"


def test_fixed_images_and_linux_build_instead_of_host_native_artifacts():
    document = compose()
    for row in document["services"].values():
        assert row["platform"] == "linux/amd64"
        assert ":" in row["image"] and not row["image"].endswith(":latest")
    api = (DEPLOY / "Dockerfile.api").read_text()
    web = (DEPLOY / "Dockerfile.web").read_text()
    assert "python -m venv /opt/venv" in api and "COPY --from=dependencies /opt/venv /opt/venv" in api
    assert "pip freeze --all" in api and "/opt/serveflow/requirements-resolved.txt" in api
    assert "libgomp1" in api and "model_cache.py fetch" in api
    assert "npm ci" in web and "npm run build" in web
    assert "COPY --from=frontend-build /build/frontend/dist" in web
    assert "vite preview" not in web and "npm run dev" not in web
    assert ".venv" not in "\n".join(line for line in api.splitlines() if line.startswith("COPY"))


def test_image_public_code_readability_and_source_notices_are_explicit():
    rules = (ROOT / ".dockerignore").read_text().splitlines()
    assert "!LICENSE" in rules and "!NOTICE.md" in rules
    for name in ("Dockerfile.api", "Dockerfile.web"):
        source = (DEPLOY / name).read_text()
        assert "COPY LICENSE NOTICE.md /opt/serveflow/" in source
        assert "-type d -exec chmod 0755 {} +" in source
        assert "-type f -exec chmod 0644 {} +" in source
        assert "chmod 0644 /opt/serveflow/LICENSE /opt/serveflow/NOTICE.md" in source
        permission_commands = [line for line in source.splitlines() if "find /opt/serveflow" in line]
        assert permission_commands
        assert all("-L " not in line and "-R " not in line for line in permission_commands)
        assert not any(token in line for line in permission_commands
                       for token in ("/run/secrets", "/var/lib", "/opt/venv", "model-cache", "/home/"))


def test_default_deny_context_excludes_private_data_and_required_sources_are_present():
    patterns = (ROOT / ".dockerignore").read_text().splitlines()
    rules = [line.strip() for line in patterns if line.strip() and not line.startswith("#")]
    assert rules[0] == "**"
    for denied in ("**/.env", "**/.env.*", "**/.venv/", "**/node_modules/", "**/__pycache__/",
                   "**/data/", "**/*.key", "**/*.pem", "**/*.db", "**/*.sqlite*", "**/*.log",
                   "**/api_config.json", "**/gateway_config.json", "**/mysql_root_password", "**/htpasswd",
                   "**/api.json", "**/gateway.json", "**/mysql-root-password"):
        assert denied in rules
    for name in ("runtime.py", "settings.py", "model_cache.py", "nginx.conf.template"):
        assert "!deploy/demo/" + name in rules
    for path in (DEPLOY / "Dockerfile.api", DEPLOY / "Dockerfile.web"):
        for line in path.read_text().splitlines():
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            for token in shlex.split(line)[1:-1]:
                assert (ROOT / token).exists(), f"Missing Docker build input: {token}"
