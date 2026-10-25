"""Bounded LLM failure metadata; never render exception messages or payloads."""

import asyncio
import json
import math

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)


_PHASES = frozenset({
    "intent.transport", "intent.parse", "intent.validate",
    "composer.transport", "composer.extract",
    "retrieval.rewrite", "retrieval.rerank",
    "memory.profile", "memory.summary", "memory.merge",
})
_ERRORS = (
    (asyncio.CancelledError, "CancelledError", "cancelled"),
    (TimeoutError, "TimeoutError", "deadline_exceeded"),
    (APITimeoutError, "APITimeoutError", "deadline_exceeded"),
    (AuthenticationError, "AuthenticationError", "authentication_failed"),
    (PermissionDeniedError, "PermissionDeniedError", "permission_denied"),
    (RateLimitError, "RateLimitError", "rate_limited"),
    (BadRequestError, "BadRequestError", "provider_request_rejected"),
    (APIConnectionError, "APIConnectionError", "connection_failed"),
    (APIStatusError, "APIStatusError", "provider_status_error"),
    (json.JSONDecodeError, "JSONDecodeError", "invalid_json"),
    (TypeError, "TypeError", "invalid_structure"),
    (KeyError, "KeyError", "invalid_structure"),
    (ValueError, "ValueError", "invalid_semantics"),
)


def record_llm_failure(logger, phase, error, elapsed_ms, *, empty_output=False):
    """Log fixed vocabulary and bounded timing, excluding exception repr/str.

    These categories describe an observed failure, not upstream root causality.
    Unknown exception classes are deliberately collapsed to OtherError.
    """
    safe_phase = phase if phase in _PHASES else "unknown"
    error_type, error_code = "OtherError", "unexpected_error"
    if empty_output:
        error_type, error_code = "None", "empty_output"
    else:
        for kind, label, code in _ERRORS:
            if isinstance(error, kind):
                error_type, error_code = label, code
                break
    safe_elapsed = 0
    if isinstance(elapsed_ms, (int, float)) and not isinstance(elapsed_ms, bool) and math.isfinite(elapsed_ms):
        safe_elapsed = int(min(3_600_000, max(0, elapsed_ms)))
    diagnostic = {
        "phase": safe_phase,
        "error_code": error_code,
        "error_type": error_type,
        "elapsed_ms": safe_elapsed,
    }
    logger.warning(
        "llm_failure phase=%s error_code=%s error_type=%s elapsed_ms=%d",
        safe_phase, error_code, error_type, safe_elapsed,
    )
    return diagnostic
