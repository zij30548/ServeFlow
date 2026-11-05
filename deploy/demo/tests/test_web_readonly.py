"""Nginx writes must fit the reviewed tmpfs mounts after privilege drop.

These inspect the shipped deployment boundary. They do not claim a Linux
container was started, and never read secrets, user data or project dotenv.
"""
from pathlib import Path, PurePosixPath
import re
import shlex

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy/demo"


def directives(name):
    source = (DEPLOY / "nginx.conf.template").read_text()
    return [shlex.split(value) for value in re.findall(r"(?m)^\s*" + re.escape(name) + r"\s+([^;]+);", source)]


def web_service():
    return yaml.safe_load((DEPLOY / "compose.yaml").read_text())["services"]["web"]


@pytest.mark.parametrize("name", ["client_body_temp_path", "proxy_temp_path",
    "fastcgi_temp_path", "uwsgi_temp_path", "scgi_temp_path"])
def test_all_compiled_http_temp_modules_have_explicit_writable_locations(name):
    # Nginx initializes even unused FastCGI/uWSGI/SCGI modules at startup;
    # omission would select the image's read-only /var/cache/nginx default.
    values = directives(name)
    assert len(values) == 1 and len(values[0]) == 1
    path = PurePosixPath(values[0][0])
    assert path.is_absolute() and ".." not in path.parts and "$" not in str(path)
    assert path != PurePosixPath("/tmp") and path.is_relative_to("/tmp")
    mounts = {row.split(":", 1)[0]: row.split(":", 1)[1].split(",")
              for row in web_service()["tmpfs"]}
    assert "rw" in mounts["/tmp"]
    assert {"nosuid", "nodev", "noexec"} <= set(mounts["/tmp"])


def test_error_log_uses_inherited_stderr_not_a_privileged_path_reopen():
    assert directives("error_log") == [["stderr", "crit"]]
    source = (DEPLOY / "nginx.conf.template").read_text()
    assert "/dev/stderr" not in source and "/var/log/nginx" not in source
    assert directives("access_log") == [["off"]]


def test_fix_does_not_make_root_filesystem_writable_or_disable_access_protection():
    web = web_service()
    assert web["read_only"] is True
    assert {row.split(":", 1)[0] for row in web["tmpfs"]} == {"/run/serveflow", "/tmp"}
    assert "volumes" not in web  # no broad writable /var or root bind mount
    assert web["ports"] == ["127.0.0.1:8080:8080"]
    assert directives("auth_basic") == [["Protected customer demonstration"]]
    assert directives("auth_basic_user_file") == [["/run/serveflow/htpasswd"]]
