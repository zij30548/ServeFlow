"""Explicit host-side private configuration; does not run Docker or contact APIs."""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import ConfigurationError, MODEL, USER, load, model_origin, origin, private_read, validate

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"api.json", "gateway.json", "htpasswd", "mysql-root-password"}


def directory(value, *, exists):
    path = Path(value)
    if (not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents))
            or path.is_relative_to(ROOT) or exists and not path.is_dir()
            or path.exists() and (not path.is_dir() or path.stat().st_mode & 0o7077 or path.stat().st_uid != os.geteuid())):
        raise ConfigurationError("demo_private_directory_required_outside_checkout")
    return path


def write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def password_hash(username, password):
    if not USER.fullmatch(username) or not 12 <= len(password.encode()) <= 64 or any(ord(c) < 32 for c in password):
        raise ConfigurationError("demo_username_or_password_invalid")
    binary = shutil.which("htpasswd")
    if binary is None:
        raise ConfigurationError("demo_install_apache2_utils_first")
    try:
        # Never use -b: a password must not appear in process arguments.
        result = subprocess.run([str(Path(binary).resolve()), "-nBi", "-C", "10", username],
            input=(password + "\n").encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20, check=True)
        import re
        # Apache -n commonly emits an extra blank line after its one record.
        # Validate the entire output, then canonicalize only this framing;
        # arbitrary stripping would also accept extra records or other text.
        match = re.fullmatch(rb"(?P<record>" + re.escape(username.encode())
            + rb":\$2[aby]\$10\$[./A-Za-z0-9]{53})(?:\r?\n){1,2}", result.stdout)
        if match is None:
            raise ValueError
        return match["record"] + b"\n"
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ConfigurationError("demo_password_hash_failed") from None


def initialize(path, public_origin, username, endpoint, limit, *, read_secret=getpass.getpass):
    path = directory(path, exists=False)
    origin(public_origin)
    model_origin(endpoint)
    if not USER.fullmatch(username) or type(limit) is not int or not 1 <= limit <= 1000:
        raise ConfigurationError()
    if path.exists():
        raise ConfigurationError("demo_existing_configuration_not_overwritten")
    if not path.parent.is_dir():
        raise ConfigurationError("demo_parent_directory_missing")
    password = read_secret("演示访问密码（12–64字节，不回显）：")
    if password != read_secret("再次输入访问密码："):
        raise ConfigurationError("demo_password_confirmation_mismatch")
    hashed = password_hash(username, password)
    key = read_secret("百炼API密钥（不回显，不会发起调用）：")
    bearer = secrets.token_hex(32)
    api = validate({"version": 1, "public_origin": public_origin, "gateway_bearer": bearer,
        "api_key": key, "model_origin": endpoint, "model": MODEL,
        "session_key_base64": base64.b64encode(secrets.token_bytes(32)).decode(),
        "mysql_reader_password": secrets.token_urlsafe(32), "model_limit": limit}, api=True)
    gateway = {k: api[k] for k in ("version", "public_origin", "gateway_bearer")}
    path.mkdir(mode=0o700)
    # Failures preserve a partial directory for inspection; never delete or rerun
    # initialization over an existing directory.
    for name, value in (("api.json", api), ("gateway.json", gateway)):
        write_new(path / name, (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode())
    write_new(path / "htpasswd", hashed)
    write_new(path / "mysql-root-password", secrets.token_urlsafe(36).encode())
    verify(path)


def verify(path):
    path = directory(path, exists=True)
    if {p.name for p in path.iterdir()} != NAMES:
        raise ConfigurationError("demo_private_file_manifest_invalid")
    api = load(path / "api.json", api=True, host_owner=True)
    gateway = load(path / "gateway.json", api=False, host_owner=True)
    if any(gateway[k] != api[k] for k in gateway):
        raise ConfigurationError("demo_gateway_configuration_mismatch")
    root_password = private_read(path / "mysql-root-password", maximum=128, host_owner=True)
    import re
    if not re.fullmatch(rb"[A-Za-z0-9_-]{32,64}", root_password):
        raise ConfigurationError("demo_mysql_root_secret_invalid")
    rows = private_read(path / "htpasswd", maximum=16384, host_owner=True).splitlines()
    seen = set()
    for row in rows:
        match = re.fullmatch(rb"([A-Za-z0-9][A-Za-z0-9_.-]{0,63}):\$2[aby]\$10\$[./A-Za-z0-9]{53}", row)
        if match is None or match[1] in seen:
            raise ConfigurationError("demo_password_file_invalid")
        seen.add(match[1])
    if not 1 <= len(rows) <= 10:
        raise ConfigurationError("demo_password_file_invalid")
    return {"result": "private_configuration_verified", "users": len(rows),
            "model_limit": api["model_limit"], "model_requests": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--origin")
    parser.add_argument("--username")
    parser.add_argument("--model-origin", default="https://dashscope.aliyuncs.com/apps/anthropic")
    parser.add_argument("--model-limit", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize(args.directory, args.origin, args.username or "", args.model_origin, args.model_limit)
        print(json.dumps(verify(args.directory), ensure_ascii=False))
        return 0
    except (ConfigurationError, OSError, EOFError, KeyboardInterrupt) as error:
        code = error.code if isinstance(error, ConfigurationError) else "demo_configuration_stopped_inspect_no_retry"
        print("STOP: " + code, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
