"""Closed container bootstrap/operations; no shell and no automatic reset.

Root reads this container's explicitly mounted secrets, prepares owned empty
storage or verifies existing ownership, then drops uid/gid before execution.
Web mode imports only the Python standard library.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import quote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from settings import ConfigurationError, DATABASE, READER, load, private_read

RUN = Path("/run/serveflow")
DATA = Path("/var/lib/serveflow")


def _private_write(path, raw, uid, gid):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.fchown(fd, uid, gid)
    finally:
        os.close(fd)


def _owned_directory(path, uid, gid, *, new_empty=False):
    if path.is_symlink():
        raise ConfigurationError("demo_unsafe_storage_path")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o7000:
        raise ConfigurationError("demo_unsafe_storage_path")
    if info.st_uid == uid and info.st_gid == gid and not info.st_mode & 0o077:
        return
    if new_empty and info.st_uid == 0 and not any(path.iterdir()):
        # Only this newly empty volume root; never recurse or repair old data.
        os.chmod(path, 0o700)
        os.chown(path, uid, gid)
        return
    raise ConfigurationError("demo_existing_storage_permissions_require_inspection")


def _drop(uid, gid):
    if os.geteuid() != 0:
        raise ConfigurationError("demo_bootstrap_requires_container_root")
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


def _check_mode_inputs(config):
    # Prevent env_file or inherited flags enabling unrelated paid integrations.
    if (os.getenv("DEMO_PUBLIC_ORIGIN") != config["public_origin"]
            or os.getenv("MCP_EXTERNAL_ENABLED", "false").lower() not in {"false", "0"}
            or os.getenv("ALERT_WEBHOOK_URL") or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("INTENT_EMBEDDING_PROVIDER", "local") != "local"):
        raise ConfigurationError("demo_public_config_or_optional_integration_invalid")


def _api_environment(config):
    return {
        "ECHOMIND_ENV": "protected-demo",
        "ECHOMIND_API_TOKENS": json.dumps({config["gateway_bearer"]: {"role": "user", "user_id": "protected-demo-proxy"}}),
        "ANTHROPIC_API_KEY": config["api_key"], "ANTHROPIC_BASE_URL": config["model_origin"],
        "ANTHROPIC_MODEL": config["model"], "SERVEFLOW_DEMO_PUBLIC_ORIGIN": config["public_origin"],
        "SERVEFLOW_DEMO_SESSION_KEY_FILE": str(RUN / "session.key"),
        "SERVEFLOW_DEMO_MYSQL_URL": f"mysql+pymysql://{READER}:{quote(config['mysql_reader_password'], safe='')}@mysql:3306/{DATABASE}",
        "SERVEFLOW_DEMO_QUOTA_DIR": str(DATA / "usage"),
        "SERVEFLOW_DEMO_MODEL_LIMIT": str(config["model_limit"]),
        "REDIS_URL": "redis://redis:6379/0", "CHROMA_PERSIST_DIRECTORY": str(DATA / "chroma"),
        "INTENT_EMBEDDING_PROVIDER": "local", "MCP_EXTERNAL_ENABLED": "false", "ALERT_WEBHOOK_URL": "",
        "ORT_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
        "ANONYMIZED_TELEMETRY": "False", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
        "PROFILE_UPDATE_WORKERS": "1", "PROMETHEUS_PORT": "0", "HOME": "/home/serveflow",
    }


def prepare_api(config, *, admin_secret=False):
    _check_mode_inputs(config)
    _owned_directory(RUN, 10001, 10001, new_empty=True)
    _owned_directory(DATA, 10001, 10001, new_empty=True)
    _private_write(RUN / "session.key", base64.b64decode(config["session_key_base64"], validate=True), 10001, 10001)
    environment = _api_environment(config)
    if admin_secret:
        raw = private_read(Path("/run/secrets/mysql_root_password"), maximum=128)
        if not re.fullmatch(rb"[A-Za-z0-9_-]{32,64}", raw):
            raise ConfigurationError("demo_mysql_root_secret_invalid")
        environment["SERVEFLOW_DEMO_ADMIN_MYSQL_URL"] = f"mysql+pymysql://root:{quote(raw.decode(), safe='')}@mysql:3306/{DATABASE}"
    os.environ.pop("SERVEFLOW_DEMO_ADMIN_MYSQL_URL", None)
    os.environ.update(environment)
    os.environ.pop("SILICONFLOW_API_KEY", None)
    _drop(10001, 10001)
    sys.dont_write_bytecode = True
    os.chdir(ROOT / "backend")
    sys.path.insert(0, str(ROOT / "backend"))
    return environment


def render_web(config, template):
    authority = urlsplit(config["public_origin"]).netloc
    host = urlsplit(config["public_origin"]).hostname
    return template.replace("__PUBLIC_AUTHORITY__", authority).replace("__PUBLIC_HOST__", host)


def web():
    config = load(Path("/run/secrets/gateway_config"), api=False)
    if os.getenv("DEMO_PUBLIC_ORIGIN") != config["public_origin"]:
        raise ConfigurationError("demo_gateway_origin_mismatch")
    raw = private_read(Path("/run/secrets/demo_htpasswd"), maximum=16384)
    rows, users = raw.splitlines(), set()
    for row in rows:
        match = re.fullmatch(rb"([A-Za-z0-9][A-Za-z0-9_.-]{0,63}):\$2[aby]\$10\$[./A-Za-z0-9]{53}", row)
        if match is None or match[1] in users:
            raise ConfigurationError("demo_password_file_invalid")
        users.add(match[1])
    if not 1 <= len(rows) <= 10:
        raise ConfigurationError("demo_password_file_invalid")
    _owned_directory(RUN, 101, 101, new_empty=True)
    _private_write(RUN / "htpasswd", raw, 101, 101)
    template = (HERE / "nginx.conf.template").read_text()
    _private_write(RUN / "nginx.conf", render_web(config, template).encode(), 101, 101)
    # Browser Authorization and proxy identity headers are overwritten here.
    proxy = f'''proxy_set_header Authorization "Bearer {config["gateway_bearer"]}";
proxy_set_header X-ServeFlow-Demo-User $remote_user;
proxy_set_header X-Forwarded-Proto https;
proxy_set_header Host "{urlsplit(config["public_origin"]).netloc}";
proxy_set_header Connection "";
proxy_http_version 1.1;
proxy_connect_timeout 5s;
proxy_read_timeout 60s;
proxy_send_timeout 60s;
proxy_request_buffering off;
proxy_buffering off;
proxy_ignore_client_abort off;
'''
    _private_write(RUN / "proxy.inc", proxy.encode(), 101, 101)
    _drop(101, 101)
    # -e applies before config parsing, so privilege-dropped startup never
    # attempts to open the image's root-owned default error log.
    os.execv("/usr/sbin/nginx", ["nginx", "-e", "stderr", "-c", str(RUN / "nginx.conf"), "-g", "daemon off;"])


def _verify_reader(environment):
    from demo_business.initialize import verify_database
    from demo_business.store import DemoDatabaseConfig, connect
    config = DemoDatabaseConfig.parse(environment["SERVEFLOW_DEMO_MYSQL_URL"])
    result = verify_database(config)
    connection = connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
            grants = cursor.fetchall()
        has_select = False
        for row in grants:
            statement = next(iter(row.values()))
            if statement.startswith("GRANT USAGE ON *.* TO "):
                continue
            if statement.startswith(f"GRANT SELECT ON `{DATABASE}`.* TO ") and "WITH GRANT OPTION" not in statement:
                has_select = True
                continue
            raise ConfigurationError("demo_mysql_reader_must_be_select_only")
        if not has_select:
            raise ConfigurationError("demo_mysql_reader_must_be_select_only")
    finally:
        connection.close()
    return result


def initialize_mysql(environment, config):
    from demo_business.initialize import initialize_database
    from demo_business.store import DemoDatabaseConfig, connect
    admin = DemoDatabaseConfig.parse(environment["SERVEFLOW_DEMO_ADMIN_MYSQL_URL"])
    import pymysql
    connection = pymysql.connect(host="mysql", port=3306, user="root", password=admin.password,
        connect_timeout=5, read_timeout=5, write_timeout=5, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s", (DATABASE,))
            existed = cursor.fetchone() is not None
            cursor.execute("SELECT User,Host FROM mysql.user WHERE User=%s", (READER,))
            users = cursor.fetchall()
            if users and tuple(users) != ((READER, "%"),):
                raise ConfigurationError("demo_existing_mysql_reader_requires_inspection")
            if existed:
                # Existing databases/credentials are never recreated or reset.
                _verify_reader(environment)
                return {"operation": "verified-existing-no-write"}
            if users:
                raise ConfigurationError("demo_existing_mysql_reader_requires_inspection")
            cursor.execute(f"CREATE DATABASE `{DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
            cursor.execute("CREATE USER %s@'%%' IDENTIFIED BY %s", (READER, config["mysql_reader_password"]))
            cursor.execute(f"GRANT SELECT ON `{DATABASE}`.* TO %s@'%%'", (READER,))
    finally:
        connection.close()
    initialized = initialize_database(admin)
    _verify_reader(environment)
    return initialized


def health(web_mode=False):
    connection = http.client.HTTPConnection("127.0.0.1", 8080 if web_mode else 8816, timeout=3)
    try:
        headers = {}
        if web_mode:
            config = load(Path("/run/secrets/gateway_config"), api=False)
            headers["Host"] = urlsplit(config["public_origin"]).netloc
        connection.request("GET", "/" if web_mode else "/health", headers=headers)
        response = connection.getresponse()
        expected = 401 if web_mode else 200
        if response.status != expected:
            raise ConfigurationError("demo_health_not_ready")
        response.read(65537)
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("api", "web", "api-health", "web-health", "init", "verify", "quota-init", "quota-status"))
    args = parser.parse_args(argv)
    try:
        if args.command.endswith("-health"):
            health(args.command == "web-health")
            return 0
        if args.command == "web":
            web()
            return 0
        config = load(Path("/run/secrets/api_config"), api=True)
        environment = prepare_api(config, admin_secret=args.command == "init")
        from core import protected_demo_quota as quota
        if args.command == "quota-init":
            result = quota.verify() if (DATA / "usage").exists() else quota.initialize()
        elif args.command == "quota-status":
            result = quota.status()
        elif args.command == "init":
            result = initialize_mysql(environment, config)
        else:
            result = _verify_reader(environment)
            quota.verify()
            if args.command == "api":
                from model_cache import verify as verify_cache
                verify_cache(Path("/opt/serveflow/model-cache/onnx_models/all-MiniLM-L6-v2"))
                os.execv(sys.executable, [sys.executable, "-B", "-m", "uvicorn", "api.main:app",
                    "--host", "0.0.0.0", "--port", "8816", "--workers", "1", "--no-access-log",
                    "--timeout-graceful-shutdown", "30"])
        print(json.dumps({"result": "completed", "operation": args.command, "details": result}, ensure_ascii=True))
        return 0
    except Exception:
        # SQL, credentials, raw URLs and filesystem exception text stay private.
        print("STOP: demo_operation_failed_inspect_no_automatic_retry", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
