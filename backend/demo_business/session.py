"""Signed, browser-local demo context. Not a real customer login system."""
from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from urllib.parse import urlsplit


COOKIE_NAME = 'serveflow_demo_session'
LIFETIME = 8 * 60 * 60


class DemoSessionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode('ascii').rstrip('=')


def _decode(value: str) -> bytes:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise DemoSessionError('demo_session_invalid')
    return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)


class DemoSessionService:
    def __init__(self, key: bytes, origins: tuple[str, ...], clock=time.time):
        if len(key) != 32 or not origins:
            raise DemoSessionError('demo_session_configuration_invalid')
        checked = []
        for origin in origins:
            try:
                parsed = urlsplit(origin)
                valid_host = parsed.hostname == 'localhost' or ipaddress.ip_address(parsed.hostname or '').is_loopback
                if (not valid_host or parsed.scheme not in {'http', 'https'} or parsed.path
                        or parsed.query or parsed.fragment or parsed.username is not None
                        or parsed.password is not None or parsed.port is None
                        or origin != f'{parsed.scheme}://{parsed.netloc}'
                        or any(c.isspace() for c in origin)):
                    raise ValueError()
            except (TypeError, ValueError):
                raise DemoSessionError('demo_session_origin_invalid') from None
            checked.append(origin)
        if len(set(checked)) != len(checked):
            raise DemoSessionError('demo_session_origin_invalid')
        self._key = key
        self.origins = tuple(checked)
        self._clock = clock

    @classmethod
    def from_env(cls):
        if os.getenv('ECHOMIND_ENV', 'development').lower() == 'production':
            raise DemoSessionError('demo_requires_local_development')
        filename = os.getenv('SERVEFLOW_DEMO_SESSION_KEY_FILE', '')
        if not filename:
            raise DemoSessionError('demo_session_key_required')
        path = Path(filename)
        try:
            info = path.lstat()
            if (not path.is_absolute() or not stat.S_ISREG(info.st_mode)
                    or info.st_mode & 0o077 or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_size != 32):
                raise OSError()
            key = path.read_bytes()
        except OSError:
            raise DemoSessionError('demo_session_key_invalid') from None
        origins = tuple(os.getenv('SERVEFLOW_DEMO_ORIGINS', '').split(','))
        return cls(key, origins)

    def issue(self, customer_id='demo-customer-a'):
        if customer_id not in {'demo-customer-a', 'demo-customer-b'}:
            raise DemoSessionError('demo_customer_invalid')
        now = int(self._clock())
        claims = {'v': 1, 'sid': secrets.token_hex(16), 'customer': customer_id,
                  'iat': now, 'exp': now + LIFETIME}
        body = _encode(json.dumps(claims, sort_keys=True, separators=(',', ':')).encode())
        signature = _encode(hmac.digest(self._key, ('session:' + body).encode(), 'sha256'))
        return body + '.' + signature, claims

    def read(self, token: str | None):
        if not token:
            raise DemoSessionError('demo_session_required')
        try:
            if not isinstance(token, str) or len(token) > 1024:
                raise ValueError()
            body, signature = token.split('.')
            expected = hmac.digest(self._key, ('session:' + body).encode(), 'sha256')
            if not hmac.compare_digest(_decode(signature), expected):
                raise ValueError()
            claims = json.loads(_decode(body))
            if (not isinstance(claims, dict) or set(claims) != {'v', 'sid', 'customer', 'iat', 'exp'}
                    or type(claims['v']) is not int or claims['v'] != 1
                    or not isinstance(claims['sid'], str) or not re.fullmatch('[a-f0-9]{32}', claims['sid'])
                    or claims['customer'] not in {'demo-customer-a', 'demo-customer-b'}
                    or type(claims['iat']) is not int or type(claims['exp']) is not int
                    or claims['exp'] != claims['iat'] + LIFETIME
                    or claims['iat'] > self._clock() + 30):
                raise ValueError()
            if claims['exp'] <= self._clock():
                raise DemoSessionError('demo_session_expired')
            return claims
        except DemoSessionError:
            raise
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise DemoSessionError('demo_session_invalid') from None

    def csrf(self, claims):
        return _encode(hmac.digest(self._key, ('csrf:' + claims['sid']).encode(), 'sha256'))

    def verify_write(self, claims, origin, csrf):
        if origin not in self.origins:
            raise DemoSessionError('demo_origin_rejected')
        if not isinstance(csrf, str) or not hmac.compare_digest(csrf, self.csrf(claims)):
            raise DemoSessionError('demo_csrf_rejected')

    def workspace_id(self, claims):
        return 'sf_demo_' + claims['sid']
