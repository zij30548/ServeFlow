"""可选 Bearer 身份映射。生产模式必须配置，开发模式限本机部署。"""
import hashlib
import hmac
import json
import os


def validate_server_access():
    mode = os.getenv("SERVEFLOW_ENV", "").lower()
    if mode not in {"development", "production", "protected-demo"}:
        raise ValueError("启动服务前必须将 SERVEFLOW_ENV 设置为 development、production 或 protected-demo")
    identity("")


def identity(authorization):
    raw = os.getenv('SERVEFLOW_API_TOKENS', '').strip()
    if not raw:
        if os.getenv('SERVEFLOW_ENV', 'development').lower() in {'production', 'protected-demo'}:
            raise ValueError('受保护模式必须设置 SERVEFLOW_API_TOKENS')
        return None
    entries = json.loads(raw)
    if not isinstance(entries, dict) or not entries:
        raise ValueError('SERVEFLOW_API_TOKENS 必须为非空对象')
    for token, account in entries.items():
        if len(token) < 24 or not isinstance(account, dict) or not isinstance(account.get('user_id'), str) or not account.get('user_id') or account.get('role') not in ('user', 'admin'):
            raise ValueError('API Token 配置无效')
    supplied = authorization[7:] if authorization.startswith('Bearer ') else ''
    digest = hashlib.sha256(supplied.encode()).digest()
    for token, account in entries.items():
        if hmac.compare_digest(hashlib.sha256(token.encode()).digest(), digest):
            return account
    return {}
