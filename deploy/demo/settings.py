"""Strict deployment inputs; stdlib only, no dotenv or network operations."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit


MODEL = "qwen3.7-flash-2026-07-15"
DATABASE = "serveflow_demo_customer"
READER = "sf_demo_reader"
API_FIELDS = frozenset({"version", "public_origin", "gateway_bearer", "api_key", "model_origin",
    "model", "session_key_base64", "mysql_reader_password", "model_limit"})
GATEWAY_FIELDS = frozenset({"version", "public_origin", "gateway_bearer"})
USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_HOST = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*)\Z")
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z")


class ConfigurationError(ValueError):
    def __init__(self, code="demo_configuration_invalid"):
        self.code = code
        super().__init__(code)


def origin(value):
    try:
        if (type(value) is not str or not 1 <= len(value) <= 300
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value) or any(c in value for c in "?#\\")):
            raise ValueError
        u = urlsplit(value)
        if (value != "https://" + u.netloc or u.scheme != "https" or u.hostname is None or not _HOST.fullmatch(u.hostname)
                or u.hostname.endswith(".") or ".." in u.hostname or u.port == 443
                or u.port is not None and not 1 <= u.port <= 65535
                or u.username is not None or u.password is not None or u.path or u.query or u.fragment
                or u.netloc != u.hostname + (f":{u.port}" if u.port is not None else "")):
            raise ValueError
        return value
    except (TypeError, ValueError):
        raise ConfigurationError("demo_https_origin_required") from None


def model_origin(value):
    try:
        if (type(value) is not str or not 1 <= len(value) <= 512
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value) or any(c in value for c in "?#\\")):
            raise ValueError
        u = urlsplit(value)
        if (u.scheme != "https" or u.hostname is None or u.netloc != u.hostname or u.port is not None
                or u.username is not None or u.password is not None or u.path != "/apps/anthropic"
                or not (u.hostname == "dashscope.aliyuncs.com" or _WORKSPACE.fullmatch(u.hostname))):
            raise ValueError
        return value
    except (TypeError, ValueError):
        raise ConfigurationError("demo_beijing_endpoint_required") from None


def _pairs(values):
    result = {}
    for k, v in values:
        if k in result:
            raise ConfigurationError()
        result[k] = v
    return result


def private_read(path, *, maximum=32768, host_owner=False):
    """Compose bind-secret ownership is host-controlled; runtime copies privately.

    Host CLI files must be owned by the operator and mode0600. Container input
    mounts may be root-readable mode0444, but never writable by group/others.
    """
    fd = None
    try:
        path = Path(path)
        if (not path.is_absolute() or path.is_symlink()
                or host_owner and (path.resolve() != path or any(p.is_symlink() for p in path.parents))):
            raise ValueError
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= maximum
                or info.st_mode & 0o7022 or host_owner and (info.st_uid != os.geteuid() or info.st_mode & 0o077)):
            raise ValueError
        raw = os.read(fd, maximum + 1)
        if len(raw) != info.st_size:
            raise ValueError
        return raw
    except (OSError, ValueError, TypeError):
        raise ConfigurationError("demo_secret_file_invalid") from None
    finally:
        if fd is not None:
            os.close(fd)


def load(path, *, api, host_owner=False):
    try:
        value = json.loads(private_read(path, host_owner=host_owner), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        return validate(value, api=api)
    except ConfigurationError:
        raise
    except (ValueError, UnicodeError, TypeError):
        raise ConfigurationError() from None


def validate(value, *, api):
    if (type(value) is not dict or set(value) != (API_FIELDS if api else GATEWAY_FIELDS)
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["gateway_bearer"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["gateway_bearer"])):
        raise ConfigurationError()
    origin(value["public_origin"])
    if api:
        model_origin(value["model_origin"])
        if (value["model"] != MODEL or type(value["model_limit"]) is not int or not 1 <= value["model_limit"] <= 1000
                or type(value["api_key"]) is not str or not 16 <= len(value["api_key"]) <= 512
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value["api_key"])
                or type(value["mysql_reader_password"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9_-]{32,64}", value["mysql_reader_password"])
                or type(value["session_key_base64"]) is not str):
            raise ConfigurationError()
        try:
            if len(base64.b64decode(value["session_key_base64"], validate=True)) != 32:
                raise ValueError
        except (ValueError, TypeError):
            raise ConfigurationError() from None
    return value
