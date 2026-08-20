"""Structured application logging without secrets or document payloads."""

from __future__ import annotations

import logging
import uuid


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
    )


class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("request_id", self.extra.get("request_id", "-"))
        return msg, kwargs


def request_logger(name: str, request_id: str | None = None) -> RequestLoggerAdapter:
    return RequestLoggerAdapter(logging.getLogger(name), {"request_id": request_id or str(uuid.uuid4())})

