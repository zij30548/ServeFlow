"""Narrow provider compatibility options; never change arbitrary endpoints.

Only the explicitly supported Beijing Flash snapshot uses Bailian's native
Anthropic thinking switch. No URL, credential or provider error is logged here.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

FLASH_SNAPSHOT = "qwen3.7-flash-2026-07-15"
_WORKSPACE_HOST = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z"
)


def nonthinking_options(client: Any, model: str) -> dict[str, Any]:
    """Return request options only for the real client's exact official origin.

    A prompt, environment name, selected role or caller-supplied model alias
    cannot opt another provider into this compatibility behavior.
    """
    if model != FLASH_SNAPSHOT:
        return {}
    try:
        base = urlsplit(str(getattr(client, "base_url", "")))
        host = base.hostname
        if (base.scheme != "https" or host is None
                or not (host == "dashscope.aliyuncs.com" or _WORKSPACE_HOST.fullmatch(host))
                or base.port not in {None, 443} or base.username is not None or base.password is not None
                or base.query or base.fragment or base.path not in {"/apps/anthropic", "/apps/anthropic/"}):
            return {}
    except (TypeError, ValueError, UnicodeError):
        return {}
    # The pinned Anthropic 0.40 SDK predates the typed ``thinking`` keyword.
    # extra_body is its supported extension hook and serializes this as the
    # exact top-level HTTP JSON field, without upgrading the original SDK.
    return {"extra_body": {"thinking": {"type": "disabled"}}}
