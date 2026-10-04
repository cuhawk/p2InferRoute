"""Logging with mandatory secret redaction.

Every log sink passes through a ``RedactionFilter`` that replaces configured secrets
(API keys, bearer tokens) with ``[REDACTED]``. This is defense in depth: adapters must
still avoid logging credentials in the first place (see ``docs/THREAT_MODEL.md``).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/=]+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9\-_]{8,}\b")
_AUTH_URL_RE = re.compile(r"(?i)(https?://)[^/\s:@]+:([^@\s]+)@")


def redact_text(text: str, secrets: list[str] | None = None) -> str:
    """Replace known secrets and credential-like patterns with [REDACTED]."""
    out = _BEARER_RE.sub("Bearer [REDACTED]", text)
    out = _SK_RE.sub("sk-[REDACTED]", out)
    out = _AUTH_URL_RE.sub(r"\1[REDACTED]:[REDACTED]@", out)
    for secret in secrets or []:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "[REDACTED]")
    return out


class RedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from the record message and args."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = secrets or []

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(str(record.msg), self.secrets)
            if record.args:
                record.args = (
                    tuple(
                        redact_text(str(a), self.secrets) if isinstance(a, str) else a
                        for a in record.args
                    )
                    if not isinstance(record.args, dict)
                    else {
                        k: redact_text(str(v), self.secrets) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                )
        except Exception:  # never break logging because of redaction
            pass
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for structured log collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO", json_enabled: bool = False, secrets: list[str] | None = None
) -> None:
    """Configure root logging once; safe to call multiple times (idempotent-ish)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_enabled
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactionFilter(secrets))
    # Replace handlers on first call; subsequent calls reconfigure level/secrets.
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_inferroute", False)
        for h in root.handlers
    ):
        handler._inferroute = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    else:
        for h in root.handlers:
            if getattr(h, "_inferroute", False):
                h.setLevel(level.upper())
                h.addFilter(RedactionFilter(secrets))
    # HTTP client libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
