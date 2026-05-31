import logging
import os
from contextvars import ContextVar
from typing import Optional


_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_route: ContextVar[str] = ContextVar("route", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.route = _route.get()
        return True


def _has_stream_handler(logger: logging.Logger) -> bool:
    return any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers)


def _has_file_handler(logger: logging.Logger, log_file: str) -> bool:
    target = os.path.abspath(log_file)
    return any(isinstance(handler, logging.FileHandler) and os.path.abspath(getattr(handler, "baseFilename", "")) == target for handler in logger.handlers)


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "shoppinggpt.txt")

    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [req=%(request_id)s route=%(route)s] %(name)s: %(message)s"
    )

    root_logger.setLevel(level)

    if not _has_stream_handler(root_logger):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(_ContextFilter())
        root_logger.addHandler(stream_handler)

    if not _has_file_handler(root_logger, log_file):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_ContextFilter())
        root_logger.addHandler(file_handler)


def set_request_context(request_id: Optional[str] = None, route: str = "-") -> None:
    _request_id.set(request_id or "-")
    _route.set(route)


def clear_request_context() -> None:
    _request_id.set("-")
    _route.set("-")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
