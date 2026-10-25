import json

import pytest

from demo_business.session import DemoSessionError, DemoSessionService


def service(clock=lambda: 1000):
    return DemoSessionService(b'x' * 32, ('http://127.0.0.1:5173',), clock)


def test_demo_context_is_private_random_and_cannot_change_customer():
    sessions = service()
    a, claims = sessions.issue()
    b, other = sessions.issue()
    assert a != b
    assert sessions.workspace_id(claims) != sessions.workspace_id(other)
    assert sessions.read(a) == claims
    assert claims['customer'] == 'demo-customer-a'
    body, signature = a.split('.')
    import base64
    tampered = dict(claims, customer='demo-customer-b')
    encoded = base64.urlsafe_b64encode(json.dumps(tampered).encode()).decode().rstrip('=')
    with pytest.raises(DemoSessionError):
        sessions.read(encoded + '.' + signature)


@pytest.mark.parametrize('token', [None, '', 'bad', 'x.y.z', 'x.%', 'a.' + 'x' * 2000])
def test_bad_or_missing_demo_context_rejected(token):
    with pytest.raises(DemoSessionError):
        service().read(token)


def test_expiry_and_csrf_are_bound_to_the_same_session():
    now = [1000]
    sessions = service(lambda: now[0])
    token, claims = sessions.issue()
    _, other = sessions.issue()
    sessions.verify_write(claims, 'http://127.0.0.1:5173', sessions.csrf(claims))
    for origin, csrf in [('https://evil.example', sessions.csrf(claims)),
                         ('http://127.0.0.1:5173', sessions.csrf(other))]:
        with pytest.raises(DemoSessionError):
            sessions.verify_write(claims, origin, csrf)
    now[0] += 28800
    with pytest.raises(DemoSessionError, match='expired'):
        sessions.read(token)


@pytest.mark.parametrize('origin', ['https://example.com:443', 'http://127.0.0.1:5173/',
                                  'http://user@localhost:5173', 'http://127.0.0.1:5173?q=x',
                                  'http://localhost', ' http://localhost:5173'])
def test_only_explicit_local_origins_allowed(origin):
    with pytest.raises(DemoSessionError):
        DemoSessionService(b'x' * 32, (origin,))


def test_key_file_is_explicit_private_and_not_overwritten(monkeypatch, tmp_path):
    key = tmp_path / 'demo.key'
    key.write_bytes(b'x' * 32)
    key.chmod(0o600)
    monkeypatch.setenv('SERVEFLOW_DEMO_SESSION_KEY_FILE', str(key))
    monkeypatch.setenv('SERVEFLOW_DEMO_ORIGINS', 'http://127.0.0.1:5173')
    monkeypatch.setenv('ECHOMIND_ENV', 'development')
    assert DemoSessionService.from_env().origins == ('http://127.0.0.1:5173',)
    assert key.read_bytes() == b'x' * 32
    key.chmod(0o644)
    with pytest.raises(DemoSessionError):
        DemoSessionService.from_env()
    key.chmod(0o600)
    monkeypatch.setenv('ECHOMIND_ENV', 'production')
    with pytest.raises(DemoSessionError, match='local_development'):
        DemoSessionService.from_env()
