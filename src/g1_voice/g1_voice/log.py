"""Compatibility alias for :mod:`node_logging`."""

try:
    from .node_logging import *  # noqa: F401,F403
except Exception:  # pragma: no cover - supports direct file loading in tests
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import node_logging as _node_logging

    for _name in dir(_node_logging):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_node_logging, _name)
