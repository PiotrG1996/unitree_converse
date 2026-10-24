"""Tests for the shared ``node_logging`` helper module."""

import importlib.util
import os
import sys
import unittest


def _load(name, filename):
    """Load a module from the g1_voice source tree."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'g1_voice', filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_NODE_LOGGING = _load('node_logging_under_test', 'node_logging.py')
_LOG = _load('log_alias_under_test', 'log.py')
_LOGGING_UTILS = _load('logging_utils_alias_under_test', 'logging_utils.py')


class _Logger:
    """Minimal logger double recording every call."""

    def __init__(self):
        self.calls = []

    def info(self, message):
        self.calls.append(('info', message))

    def warn(self, message):
        self.calls.append(('warn', message))

    def warning(self, message):
        self.calls.append(('warning', message))

    def error(self, message):
        self.calls.append(('error', message))

    def debug(self, message):
        self.calls.append(('debug', message))


class _Node:
    """Minimal node double exposing ``get_logger``."""

    def __init__(self, logger=None):
        self._logger = logger

    def get_logger(self):
        return self._logger


class TestNodeLogging(unittest.TestCase):
    """Behaviour of the shared logging helpers."""

    def test_get_logger_returns_node_logger(self):
        """The helper delegates to ``Node.get_logger``."""
        logger = _Logger()
        node = _Node(logger)
        self.assertIs(_NODE_LOGGING.get_logger(node), logger)

    def test_get_logger_handles_missing_logger(self):
        """A node without a logger yields ``None``."""
        node = _Node(None)
        self.assertIsNone(_NODE_LOGGING.get_logger(node))

    def test_info_delegates_with_message(self):
        """``info`` forwards the message unchanged."""
        logger = _Logger()
        node = _Node(logger)
        _NODE_LOGGING.info(node, 'hello')
        self.assertEqual(logger.calls, [('info', 'hello')])

    def test_structured_fields_are_appended(self):
        """Keyword arguments are appended as ``key=value`` pairs."""
        logger = _Logger()
        node = _Node(logger)
        _NODE_LOGGING.info(node, 'hello', count=2)
        self.assertEqual(logger.calls, [('info', "hello | count=2")])

    def test_warn_prefers_warn_method(self):
        """``warn`` uses ``warn`` when the logger provides it."""
        logger = _Logger()
        node = _Node(logger)
        _NODE_LOGGING.warn(node, 'careful')
        self.assertEqual(logger.calls, [('warn', 'careful')])

    def test_warn_falls_back_to_warning(self):
        """``warn`` uses ``warning`` when ``warn`` is unavailable."""
        logger = _Logger()
        logger.warn = None
        node = _Node(logger)
        _NODE_LOGGING.warn(node, 'careful')
        self.assertEqual(logger.calls, [('warning', 'careful')])

    def test_warning_falls_back_to_warn(self):
        """``warning`` uses ``warn`` when ``warning`` is unavailable."""
        logger = _Logger()
        logger.warning = None
        node = _Node(logger)
        _NODE_LOGGING.warning(node, 'careful')
        self.assertEqual(logger.calls, [('warn', 'careful')])

    def test_unknown_level_falls_back_to_info(self):
        """An unknown level is logged through ``info``."""
        logger = _Logger()
        node = _Node(logger)
        _NODE_LOGGING.log(node, 'verbose', 'hello')
        self.assertEqual(logger.calls, [('info', 'hello')])

    def test_no_logger_is_a_no_op(self):
        """Logging without a logger never raises."""
        node = _Node(None)
        _NODE_LOGGING.info(node, 'hello')
        self.assertEqual(node, node)

    def test_should_log_throttles(self):
        """Repeated calls within the interval are suppressed."""
        node = _Node(None)
        self.assertTrue(
            _NODE_LOGGING.should_log(node, 'key', 1.0, now=10.0))
        self.assertFalse(
            _NODE_LOGGING.should_log(node, 'key', 1.0, now=10.5))
        self.assertTrue(
            _NODE_LOGGING.should_log(node, 'key', 1.0, now=11.5))

    def test_aliases_expose_same_functions(self):
        """The alias modules re-export the helper functions."""
        self.assertEqual(_LOG.info.__name__, _NODE_LOGGING.info.__name__)
        self.assertEqual(_LOGGING_UTILS.warn.__name__, _NODE_LOGGING.warn.__name__)


if __name__ == '__main__':
    unittest.main()
