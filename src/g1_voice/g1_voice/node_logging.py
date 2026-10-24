"""
Shared logging helpers for ``g1_voice`` ROS 2 nodes.

The helpers in this module never replace ``Node.get_logger``.  They delegate to
it, so ROS nodes keep their native logger while gaining a small, consistent
formatting and throttling layer that is also easy to exercise in unit tests.
"""

import time
from typing import Any, Dict

_LEVEL_FALLBACKS = {"warn": ("warning",), "warning": ("warn",)}


def get_logger(node: Any) -> Any:
    """
    Return the logger exposed by *node*, or ``None`` if it has none.

    Test doubles that only implement ``get_logger`` are supported, as are
    objects that do not implement it at all.
    """
    getter = getattr(node, "get_logger", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - defensive for partial doubles
        return None


def _format_message(message: str, fields: Dict[str, Any]) -> str:
    """Append ``key=value`` fields to *message* in a stable order."""
    if not fields:
        return message
    suffix = " ".join(
        "{}={}".format(key, value) for key, value in fields.items()
    )
    return "{} | {}".format(message, suffix)


def log(
        node: Any,
        level: str = "info",
        message: str = "",
        **fields: Any) -> None:
    """
    Log *message* through *node*'s ROS logger.

    Unknown levels fall back to ``info``.  Both ``warn`` and ``warning`` are
    accepted; the first method that exists on the logger is used, which keeps
    the helpers compatible with both rclpy loggers and the minimal doubles used
    in the test suite.
    """
    logger = get_logger(node)
    if logger is None:
        return

    msg = _format_message(message, fields)
    level = (level or "info").lower()

    # Each severity must be invoked from its own source line. rclpy caches a
    # context keyed by (file, function, line) and raises
    # "Logger severity cannot be changed between calls" when two different
    # severities are emitted from the same call site, so do NOT route
    # everything through a single shared `method(msg)` line.
    if level == "debug":
        debug_method = getattr(logger, "debug", None)
        if callable(debug_method):
            debug_method(msg)
        return

    if level in ("warn", "warning"):
        # Prefer `warn` when available, then fall back to `warning`.
        warn_method = getattr(logger, "warn", None)
        if callable(warn_method):
            warn_method(msg)
            return
        warning_method = getattr(logger, "warning", None)
        if callable(warning_method):
            warning_method(msg)
            return
        # Last resort: keep the message visible.
        info_method = getattr(logger, "info", None)
        if callable(info_method):
            info_method(msg)
        return

    if level == "error":
        error_method = getattr(logger, "error", None)
        if callable(error_method):
            error_method(msg)
        return

    # info (and any unknown level).
    info_method = getattr(logger, "info", None)
    if callable(info_method):
        info_method(msg)
    return


def info(node: Any, message: str = "", **fields: Any) -> None:
    """Log *message* at info level."""
    log(node, "info", message, **fields)


def warn(node: Any, message: str = "", **fields: Any) -> None:
    """Log *message* at warn level."""
    log(node, "warn", message, **fields)


def warning(node: Any, message: str = "", **fields: Any) -> None:
    """Log *message* at warning level."""
    log(node, "warning", message, **fields)


def error(node: Any, message: str = "", **fields: Any) -> None:
    """Log *message* at error level."""
    log(node, "error", message, **fields)


def debug(node: Any, message: str = "", **fields: Any) -> None:
    """Log *message* at debug level."""
    log(node, "debug", message, **fields)


def should_log(
    node: Any, key: str, interval_sec: float, now: float = None
) -> bool:
    """
    Return whether a throttled log line may be emitted now.

    State is stored lazily on *node* in ``_log_throttle_times``, so callers do
    not need to initialise anything in their constructors.
    """
    if now is None:
        now = time.monotonic()

    bucket = getattr(node, "_log_throttle_times", None)
    if not isinstance(bucket, dict):
        bucket = {}
        try:
            setattr(node, "_log_throttle_times", bucket)
        except Exception:  # pragma: no cover - defensive for frozen doubles
            return True

    last = bucket.get(key, 0.0)
    if now - float(last) >= float(interval_sec):
        bucket[key] = now
        return True
    return False


def throttle(
    node: Any, key: str, interval_sec: float, now: float = None
) -> bool:
    """Alias for :func:`should_log`."""
    return should_log(node, key, interval_sec, now=now)
