"""
Regression tests for the false-accept backoff session logic.

The 2026-10-16 incident: sessions re-opened every ~25s on hallucination
turns while the backoff never escalated, because
  * ``_touch_session`` cleared ``session_opened_by_wake`` on the first
    STT/LLM/TTS busy touch (~1s into every session), so the close-time
    backoff condition was never true, and
  * ``_fire_trigger`` reset cooldown+count on every fire, so even a
    successful escalation would restart from level 1.
The fix tracks a per-session "LLM reply happened" signal and only resets
the streak on a real reply (or a manual pad press).
"""

import importlib.util
import os
import sys
import time
import types
import unittest


def _load_wake_module():
    """Import wake_word_node.py without a live ROS install (stub rclpy)."""
    for name in ('rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    class _Node:
        pass

    sys.modules['rclpy'].ok = lambda: False
    sys.modules['rclpy.node'].Node = _Node

    class _Msg:
        def __init__(self, data=''):
            self.data = data

    sys.modules['std_msgs.msg'].String = _Msg
    sys.modules['std_msgs.msg'].Bool = _Msg
    path = os.path.join(
        os.path.dirname(__file__), '..', 'g1_voice', 'wake_word_node.py')
    spec = importlib.util.spec_from_file_location(
        'wake_word_node_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['wake_word_node_under_test'] = module
    spec.loader.exec_module(module)
    return module


_wake_module = _load_wake_module()


class _FakePub:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeDetector:
    """Minimal openWakeWord stand-in: only last_score is observed."""

    def __init__(self, peak=0.0):
        self.last_score = peak


class _FakeWakeNode:
    """Minimal harness binding WakeWordNode's real session helpers (no ROS)."""

    def __init__(self):
        self.logged = []
        # Session state (mirrors WakeWordNode.__init__)
        self.session_open = False
        self.session_opened_by_wake = False
        self.session_had_llm_reply = False
        self.last_session_activity = 0.0
        self.session_silence_timeout = 20.0
        # Duration cap + empty-turn limit (session-close conditions).
        self.session_max_duration_sec = 300.0
        self.max_empty_turns = 4
        # Backoff state
        self.base_cooldown = 8.0
        self.cooldown = 8.0
        self.false_accept_count = 0
        self.false_accept_backoff_multiplier = 2.0
        self.max_false_accept_cooldown = 60.0
        # Trigger/pipeline state
        self.last_trigger_time = 0.0
        self.listening = True
        self.stt_busy = False
        self.llm_busy = False
        self.tts_busy = False
        self._wake_socket = None
        # Missed-wake telemetry state (mirrors WakeWordNode.__init__)
        self.aletta_threshold = 0.85
        self.telemetry_log_interval_sec = 30.0
        self._last_telemetry_log = 0.0
        self._blocked_by_cooldown_log = 0.0
        self.oww_detector = _FakeDetector()
        self.oww_detector = _FakeDetector()
        self.trigger_pub = _FakePub()
        self.status_pub = _FakePub()
        self.continuous_start_pub = _FakePub()
        self.continuous_stop_pub = _FakePub()
        # Bind the real production helpers.
        cls = _wake_module.WakeWordNode
        self._touch_session = lambda: cls._touch_session(self)
        self._open_session = lambda source: cls._open_session(self, source)
        self._close_session = lambda reason: cls._close_session(self, reason)
        self._can_trigger = lambda: cls._can_trigger(self)
        self._fire_trigger = lambda source: cls._fire_trigger(self, source)
        self._in_session_refresh = (
            lambda score: cls._in_session_refresh(self, score))
        self._log_burst_telemetry = lambda: cls._log_burst_telemetry(self)
        self._resume_when_idle = lambda: cls._resume_when_idle(self)
        self._check_session_timeout = (
            lambda: cls._check_session_timeout(self))
        self._check_session_timeout = (
            lambda: cls._check_session_timeout(self))
        self.llm_status_callback = (
            lambda msg: cls.llm_status_callback(self, msg))

    def get_logger(self):
        harness = self

        class _Log:
            def info(self, msg):
                harness.logged.append(('info', msg))

            def warn(self, msg):
                harness.logged.append(('warn', msg))

            def warning(self, msg):
                harness.logged.append(('warn', msg))

            def error(self, msg):
                harness.logged.append(('error', msg))

            def debug(self, msg):
                harness.logged.append(('debug', msg))

        return _Log()

    def _wake_open(self):
        """Simulate a wake trigger firing (at t=0 so cooldown never blocks)."""
        self.last_trigger_time = 0.0
        self._fire_trigger('aletta_wake_word')


class TestBackoffEscalation(unittest.TestCase):

    def test_wake_attribution_survives_activity(self):
        node = _FakeWakeNode()
        node._wake_open()
        node._touch_session()  # first STT busy touch, ~1s into the session
        node._touch_session()
        self.assertTrue(node.session_opened_by_wake)

    def test_no_reply_timeout_escalates_cooldown(self):
        node = _FakeWakeNode()
        node._wake_open()
        node._touch_session()  # activity must NOT disarm the backoff
        node._close_session('20s of silence (>20s)')
        self.assertEqual(node.false_accept_count, 1)
        self.assertEqual(node.cooldown, 16.0)
        self.assertTrue(any('no reply' in m for _lvl, m in node.logged))

    def test_streak_accumulates_across_no_reply_sessions(self):
        node = _FakeWakeNode()
        for expected_count, expected_cooldown in ((1, 16.0), (2, 32.0),
                                                  (3, 60.0)):
            node._wake_open()  # last_trigger_time reset (elapsed time passes)
            node._touch_session()
            node._close_session('20s of silence (>20s)')
            self.assertEqual(node.false_accept_count, expected_count)
            self.assertEqual(node.cooldown, expected_cooldown)

    def test_reply_session_does_not_escalate(self):
        node = _FakeWakeNode()
        node._wake_open()
        node._touch_session()
        node.llm_status_callback(_wake_module.String('thinking'))
        node._close_session('20s of silence (>20s)')
        self.assertEqual(node.false_accept_count, 0)
        self.assertEqual(node.cooldown, node.base_cooldown)

    def test_reply_resets_existing_streak(self):
        node = _FakeWakeNode()
        node.session_open = True
        node.false_accept_count = 3
        node.cooldown = 60.0
        node.llm_status_callback(_wake_module.String('thinking'))
        self.assertEqual(node.false_accept_count, 0)
        self.assertEqual(node.cooldown, node.base_cooldown)
        self.assertTrue(node.session_had_llm_reply)

    def test_close_clears_reply_flag_for_next_session(self):
        node = _FakeWakeNode()
        node._wake_open()
        node.llm_status_callback(_wake_module.String('thinking'))
        node._close_session('20s of silence (>20s)')
        self.assertFalse(node.session_had_llm_reply)
        self.assertFalse(node.session_opened_by_wake)

    def test_pad_trigger_clears_backoff(self):
        node = _FakeWakeNode()
        node.false_accept_count = 3
        node.cooldown = 60.0
        node._fire_trigger('pad_f1')
        self.assertEqual(node.false_accept_count, 0)
        self.assertEqual(node.cooldown, node.base_cooldown)
        self.assertTrue(node.session_open)

    def test_wake_trigger_keeps_backoff_streak(self):
        node = _FakeWakeNode()
        node.false_accept_count = 2
        node.cooldown = 32.0
        node._wake_open()
        self.assertEqual(node.false_accept_count, 2)
        self.assertEqual(node.cooldown, 32.0)

    def test_cooldown_cap_respected(self):
        node = _FakeWakeNode()
        node.false_accept_count = 10
        node.cooldown = 60.0
        node._wake_open()
        node._touch_session()
        node._close_session('20s of silence (>20s)')
        self.assertEqual(node.cooldown, node.max_false_accept_cooldown)

    def test_blocked_by_cooldown_between_false_accepts(self):
        node = _FakeWakeNode()
        node._wake_open()  # last_trigger_time = now
        node._touch_session()
        node._close_session('20s of silence (>20s)')  # cooldown -> 16
        # Immediately after the fire, the elevated cooldown must block a
        # re-trigger (this is the slowdown that was missing entirely).
        node.last_trigger_time = time.time()
        self.assertFalse(node._can_trigger())

    def test_in_session_refresh_extends_window(self):
        node = _FakeWakeNode()
        node._wake_open()
        node.last_session_activity = time.time() - 15.0
        node._in_session_refresh(0.99)
        self.assertTrue(node.session_open)
        self.assertAlmostEqual(
            node.last_session_activity, time.time(), delta=2.0)
        self.assertTrue(any('refreshed' in m for _lvl, m in node.logged))

    def test_resume_ignores_cooldown_listens_immediately(self):
        # Missed-wake regression: after a phantom timeout the escalated
        # cooldown must not keep the mic closed. The listener resumes at
        # once; only firing stays gated. (The historic `listening` pause
        # flag was removed — the mic now stays open by design — so this
        # asserts the session/timeout bookkeeping instead.)
        node = _FakeWakeNode()
        node._wake_open()  # last_trigger_time = now
        node._touch_session()
        node._close_session('20s of silence (>20s)')  # cooldown -> 16
        self.assertEqual(node.cooldown, 16.0)
        self.assertFalse(node.session_open)
        node._resume_when_idle()
        self.assertFalse(node.session_open)

    def test_resume_keeps_session_closed_state(self):
        node = _FakeWakeNode()
        node._wake_open()
        node._resume_when_idle()  # session still open -> stays open
        self.assertTrue(node.session_open)

    def test_fire_still_gated_by_cooldown(self):
        node = _FakeWakeNode()
        node._wake_open()  # last_trigger_time = now, cooldown 8
        triggered_before = len(node.trigger_pub.published)
        node._fire_trigger('aletta_wake_word')  # must be blocked
        self.assertEqual(len(node.trigger_pub.published), triggered_before)

    def test_telemetry_throttles_bursts(self):
        node = _FakeWakeNode()
        node.oww_detector.last_score = 0.3
        node._log_burst_telemetry()
        first = len(node.logged)
        node._log_burst_telemetry()  # within interval -> silent
        self.assertEqual(len(node.logged), first)


if __name__ == '__main__':
    unittest.main()
