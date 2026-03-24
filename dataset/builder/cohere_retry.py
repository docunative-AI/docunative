"""
Shared Cohere chat retry with exponential backoff for HTTP 429 (rate limits).

Used by dataset.builder.writer and dataset.builder.qa_factory_llm.
Configure via COHERE_429_MAX_RETRIES, COHERE_429_INITIAL_DELAY_SEC, COHERE_429_MAX_DELAY_SEC.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional

from tenacity import Retrying, RetryCallState, RetryError, retry_if_exception_type, stop_after_attempt


def _cohere_429_log_detail_enabled() -> bool:
    """Set COHERE_LOG_429_DETAILS=0 to hide body/headers on 429 (default: on)."""
    return os.getenv("COHERE_LOG_429_DETAILS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def format_cohere_429_for_log(exc: BaseException) -> str | None:
    """
    Extract API message + headers from Cohere TooManyRequestsError for debugging.
    Body often states whether the limit is per-minute vs daily quota; headers may include
    Retry-After and rate-limit metadata.
    """
    try:
        from cohere.errors import TooManyRequestsError
    except ImportError:
        return None
    if not isinstance(exc, TooManyRequestsError):
        return None
    parts: list[str] = []
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(f"body={body!r}")
    hdrs = getattr(exc, "headers", None)
    if isinstance(hdrs, dict) and hdrs:
        safe = {
            k: v
            for k, v in hdrs.items()
            if str(k).lower() not in ("authorization", "x-api-key", "api-key")
        }
        parts.append(f"headers={safe!r}")
    sc = getattr(exc, "status_code", None)
    if sc is not None:
        parts.append(f"status_code={sc}")
    return " | ".join(parts) if parts else None


def _cohere_rate_limit_delay_seconds(attempt_idx: int, headers: Optional[dict[str, str]]) -> float:
    """
    Sleep duration before retrying after HTTP 429 from Cohere.
    Exponential backoff from COHERE_429_INITIAL_DELAY_SEC, capped at COHERE_429_MAX_DELAY_SEC.
    Honors Retry-After when present. Small jitter reduces synchronized retries across threads.
    """
    initial = float(os.getenv("COHERE_429_INITIAL_DELAY_SEC", "10"))
    max_delay = float(os.getenv("COHERE_429_MAX_DELAY_SEC", "120"))
    base = min(initial * (2**attempt_idx), max_delay)

    retry_after: float | None = None
    if isinstance(headers, dict):
        ra_raw = headers.get("retry-after") or headers.get("Retry-After")
        if ra_raw is not None:
            try:
                retry_after = float(ra_raw)
            except (TypeError, ValueError):
                pass

    delay = max(base, retry_after) if retry_after is not None else base
    cap_jitter = min(5.0, delay * 0.15)
    if cap_jitter > 0:
        delay += random.uniform(0, cap_jitter)
    return delay


def _wait_cohere_429(retry_state: RetryCallState) -> float:
    """Tenacity wait strategy: backoff + Retry-After + jitter (see env COHERE_429_*)."""
    attempt_idx = max(0, retry_state.attempt_number - 1)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    hdrs = getattr(exc, "headers", None)
    return _cohere_rate_limit_delay_seconds(
        attempt_idx, hdrs if isinstance(hdrs, dict) else None
    )


def _cohere_before_sleep(log: logging.Logger, max_429_retries: int):
    def _log(retry_state: RetryCallState) -> None:
        if _cohere_429_log_detail_enabled() and retry_state.outcome:
            exc = retry_state.outcome.exception()
            if exc is not None:
                detail = format_cohere_429_for_log(exc)
                if detail:
                    log.warning("Cohere HTTP 429 — %s", detail)
        log.warning(
            "Cohere 429 Too Many Requests — sleeping %.1fs (attempt %d, max retries %d)",
            retry_state.upcoming_sleep,
            retry_state.attempt_number,
            max_429_retries,
        )

    return _log


def cohere_chat_with_429_retry(
    client: Any,
    log: logging.Logger,
    *,
    exhausted_message: str,
    **chat_kwargs: Any,
) -> Any | None:
    """
    Call client.chat(**chat_kwargs) with retries on TooManyRequestsError.

    Returns the chat response object, or None if 429 retries are exhausted.
    Other exceptions propagate to the caller.
    """
    from cohere.errors import TooManyRequestsError

    max_429_retries = max(0, int(os.getenv("COHERE_429_MAX_RETRIES", "10")))
    max_attempts = max_429_retries + 1

    try:
        retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            retry=retry_if_exception_type(TooManyRequestsError),
            wait=_wait_cohere_429,
            before_sleep=_cohere_before_sleep(log, max_429_retries),
        )
        return retrying(client.chat, **chat_kwargs)
    except RetryError:
        log.error("%s — exhausted %d 429 retries", exhausted_message, max_429_retries)
        return None
