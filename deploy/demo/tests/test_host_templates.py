"""Host-side bootstrap/TLS boundary; no real certificate or cloud operations."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy/demo"


def source(name):
    # Drop comments so instructions cannot satisfy a missing directive.
    return re.sub(r"(?m)#.*$", "", (DEPLOY / name).read_text())


def blocks(text, label):
    """Return complete bodies without assuming nested locations end a server."""
    result = []
    for match in re.finditer(label + r"\s*\{", text):
        start = match.end(); depth = 1
        for index in range(start, len(text)):
            if text[index] == "{": depth += 1
            elif text[index] == "}":
                depth -= 1
                if not depth:
                    result.append(text[start:index]); break
        else: raise AssertionError("unclosed template block")
    return result


def acme(server):
    found = blocks(server, r"location\s+\^~\s+/\.well-known/acme-challenge/")
    assert len(found) == 1
    challenge = found[0]
    assert re.search(r"\broot\s+/var/lib/serveflow-acme\s*;", challenge)
    assert re.search(r"\btry_files\s+\$uri\s+=404\s*;", challenge)
    assert "proxy_pass" not in challenge and "return 30" not in challenge
    assert re.findall(r"auth_basic\s+([^;]+);", server) == ["off"]
    assert "auth_basic off;" in challenge  # only public verification tokens bypass inherited auth
    denied = blocks(challenge, r"limit_except\s+GET\s+HEAD")
    assert len(denied) == 1 and re.search(r"\bdeny\s+all\s*;", denied[0])
    return challenge


def test_bootstrap_never_requires_a_certificate_or_exposes_the_application():
    text = source("host-http.conf.example")
    servers = blocks(text, r"\bserver")
    assert len(servers) == 1
    assert re.search(r"\blisten\s+80\s*;", servers[0])
    assert "ssl_certificate" not in text and "proxy_pass" not in text
    assert not re.search(r"\blisten\s+443\b", text)
    acme(servers[0])
    root = blocks(servers[0], r"location\s+/")
    assert len(root) == 1 and re.search(r"\breturn\s+503\s*;", root[0])


def test_final_http_challenge_stays_accessible_for_webroot_renewal():
    servers = blocks(source("host-tls.conf.example"), r"\bserver")
    assert len(servers) == 2
    http = next(s for s in servers if re.search(r"\blisten\s+80\s*;", s))
    challenge = acme(http)
    root = blocks(http, r"location\s+/")
    assert len(root) == 1
    assert re.search(r"\breturn\s+308\s+https://DEMO_HOST\$request_uri\s*;", root[0])
    # A server-level return would run before ACME location matching.
    top = http.replace(challenge, "").replace(root[0], "")
    assert not re.search(r"\breturn\b", top)


def test_tls_keeps_fixed_loopback_proxy_cancellation_and_body_boundaries():
    text = source("host-tls.conf.example")
    tls = next(s for s in blocks(text, r"\bserver") if re.search(r"\blisten\s+443\s+ssl\s*;", s))
    assert "ssl_certificate /etc/letsencrypt/live/DEMO_HOST/fullchain.pem;" in tls
    assert "ssl_certificate_key /etc/letsencrypt/live/DEMO_HOST/privkey.pem;" in tls
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in tls
    assert "client_max_body_size 64k;" in tls
    assert "proxy_pass http://127.0.0.1:8080;" in tls
    assert "proxy_set_header Host DEMO_HOST;" in tls
    assert "proxy_set_header X-Forwarded-Proto https;" in tls
    for directive in ("proxy_request_buffering off;", "proxy_buffering off;", "proxy_ignore_client_abort off;",
                      "proxy_connect_timeout 5s;", "proxy_read_timeout 65s;", "proxy_send_timeout 65s;"):
        assert directive in tls
    assert not re.search(r"proxy_pass\s+[^;]*\$", tls)


@pytest.mark.parametrize("name", ["host-http.conf.example", "host-tls.conf.example"])
def test_every_host_server_disables_access_logs_and_has_explicit_domain(name):
    for server in blocks(source(name), r"\bserver"):
        assert "server_name DEMO_HOST;" in server
        assert "access_log off;" in server
        assert re.search(r"error_log\s+/var/log/nginx/serveflow-demo-error.log\s+crit\s*;", server)
