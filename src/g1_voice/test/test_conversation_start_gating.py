"""
Conversation-start gating matrix tests (Aletta vs pad vs START).

Contract verified here (see the conversation-start analysis):
  * ``wake_word_node`` is the SOLE owner of continuous mode: only a
    valid "Aletta" wake detection opens a session, publishing
    ``/g1/voice/continuous_start`` exactly once per session (a refresh
    never re-publishes it).
  * The pad's F1 is push-to-talk ONLY while no session is open; during
    an Aletta session the button node suppresses F1 triggers.
  * START is a kill switch: it can END a session but never opens one.
  * SELECT speaks a fixed phrase via TTS without touching STT/LLM.
  * ``stt_node`` starts recording on ``/g1/voice/trigger`` and DEFERS
    (never drops) the trigger while the TTS echo guard is active.

No ROS runtime is required: modules are loaded from source with rclpy
stubbed and real methods are bound onto minimal harnesses (same
convention as test_wake_word_session.py).
"""

import importlib.util
import os
import sys
import threading
import time
import types
import unittest


def _install_ros_stubs():
    """Stub rclpy/std_msgs so node modules import without a ROS install."""
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


def _load_module(module_name, filename):
    """Load a g1_voice node module from source under a unique name."""
    _install_ros_stubs()
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    path = os.path.join(os.path.dirname(__file__), '..', 'g1_voice', filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_wake_module = _load_module(
    'wake_word_gating_under_test', 'wake_word_node.py')
_button_module = _load_module(
    'button_trigger_under_test', 'button_trigger_node.py')
_stt_module = _load_module('stt_gating_under_test', 'stt_node.py')

# Pad key bitfield (Unitree WirelessController) — mirrors the constants in
# button_trigger_node.py so a rename there fails loudly here instead of
# silently drifting.
F1_KEY = 64
START_KEY = 4
SELECT_KEY = 8


class _FakePub:
    """Records published messages for later assertion."""

    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _Msg:
    """Minimal std_msgs stand-in (data carrier)."""

    def __init__(self, data=''):
        self.data = data


class _Keys:
    """Minimal WirelessController stand-in (only the keys field is read)."""

    def __init__(self, keys):
        self.keys = keys


def _make_logger(sink):
    """Build a logger whose output lands in ``sink`` for assertions."""
    class _Log:
        def info(self, msg):
            sink.append(('info', msg))

        def warn(self, msg):
            sink.append(('warn', msg))

        warning = warn

        def error(self, msg):
            sink.append(('error', msg))

        def debug(self, msg):
            sink.append(('debug', msg))

    return _Log()


class _FakeWakeNode:
    """Harness binding WakeWordNode's real session/trigger helpers."""

    def __init__(self):
        self.logged = []
        # Session state (mirrors WakeWordNode.__init__)
        self.session_open = False
        self.session_opened_by_wake = False
        self.session_had_llm_reply = False
        self.last_session_activity = 0.0
        self.session_opened_at = 0.0
        self.empty_turns = 0
        self.session_max_duration_sec = 300.0
        self.max_empty_turns = 4
        # Backoff state
        self.base_cooldown = 8.0
        self.cooldown = 8.0
        self.false_accept_count = 0
        self.false_accept_backoff_multiplier = 2.0
        self.max_false_accept_cooldown = 60.0
        # Session state observability (mirrors WakeWordNode.__init__)
        self._session_state = 'idle'
        # Trigger/pipeline state
        self.last_trigger_time = 0.0
        self.stt_busy = False
        self.llm_busy = False
        self.tts_busy = False
        # Turn-in-flight guard (mirrors WakeWordNode.__init__).
        self.turn_in_flight = False
        self._turn_in_flight_since = 0.0
        # Thread-safety lock (mirrors WakeWordNode.__init__).
        self._state_lock = threading.RLock()
        # Publishers observed by the tests
        self.trigger_pub = _FakePub()
        self.status_pub = _FakePub()
        self.session_state_pub = _FakePub()
        self.continuous_start_pub = _FakePub()
        self.continuous_stop_pub = _FakePub()
        cls = _wake_module.WakeWordNode
        self._touch_session = lambda: cls._touch_session(self)
        self._open_session = lambda source: cls._open_session(self, source)
        self._close_session = lambda reason: cls._close_session(self, reason)
        self._can_trigger = lambda: cls._can_trigger(self)
        self._fire_trigger = lambda source: cls._fire_trigger(self, source)
        self._publish_session_state = lambda: cls._publish_session_state(self)
        self.manual_stop_callback = (
            lambda msg: cls.manual_stop_callback(self, msg))

    def get_logger(self):
        return _make_logger(self.logged)


class _FakeButtonNode:
    """Harness binding ButtonTriggerNode's real controller handlers."""

    def __init__(self):
        self.logged = []
        self.continuous_mode = False
        self.speak_enabled = True
        self.recording_active = False
        self.prev_keys = 0
        self.last_f3_time = 0.0
        self.last_start_time = 0.0
        self.last_select_time = 0.0
        self.custom_say_text = 'ping'
        self.f1_pressed = False
        self.f3_pressed = False
        self.start_pressed = False
        self.select_pressed = False
        # Publishers observed by the tests
        self.trigger_pub = _FakePub()
        self.stop_pub = _FakePub()
        self.continuous_start_pub = _FakePub()
        self.continuous_stop_pub = _FakePub()
        self.say_pub = _FakePub()
        self.status_pub = _FakePub()
        cls = _button_module.ButtonTriggerNode
        self.controller_callback = (
            lambda msg: cls.controller_callback(self, msg))
        self._on_continuous_start = (
            lambda msg: cls._on_continuous_start(self, msg))
        self._on_continuous_stop = (
            lambda msg: cls._on_continuous_stop(self, msg))
        self._toggle_continuous = lambda: cls._toggle_continuous(self)
        self._publish_trigger = (
            lambda value: cls._publish_trigger(self, value))
        self._publish_stop = lambda: cls._publish_stop(self)
        self._publish_status = (
            lambda status: cls._publish_status(self, status))
        self._publish_say_text = lambda: cls._publish_say_text(self)

    def get_logger(self):
        return _make_logger(self.logged)


class _FakeSttTriggerNode:
    """Harness binding STTNode's real trigger/echo-guard helpers."""

    def __init__(self):
        self.logged = []
        self.is_recording = False
        self._recording_lock = threading.Lock()
        self._watchdog_lock = threading.Lock()
        self._watchdog = None
        # TTS echo-guard state (mirrors STTNode.__init__)
        self.tts_busy = False
        self._last_tts_speaking_time = 0.0
        self.tts_echo_guard_sec = 0.05
        # Mic path logging flag (mirrors STTNode.__init__)
        self._mic_path_logged = False
        self.use_udp_mic = True
        self.udp_multicast_group = '239.168.123.161'
        self.udp_port = 5555
        self.udp_local_ip = '192.168.123.164'
        # Thread-target fakes: record invocations instead of running them.
        self.record_calls = []
        self.delayed_calls = []
        self._record_and_transcribe = lambda: self.record_calls.append(1)
        self._delayed_trigger = lambda delay, deadline=None: (
            self.delayed_calls.append(delay))
        cls = _stt_module.STTNode
        self.trigger_callback = (
            lambda msg: cls.trigger_callback(self, msg))
        self._retrigger = lambda: cls._retrigger(self)
        self._try_start_recording = lambda: cls._try_start_recording(self)
        self._tts_echo_active = lambda: cls._tts_echo_active(self)
        self._echo_guard_remaining = (
            lambda: cls._echo_guard_remaining(self))

    def get_logger(self):
        return _make_logger(self.logged)


class TestAlettaIsSoleSessionOpener(unittest.TestCase):
    """wake_word_node owns continuous mode; only Aletta opens a session."""

    def test_aletta_trigger_fires_and_opens_session(self):
        node = _FakeWakeNode()
        node._fire_trigger('aletta_wake_word')
        self.assertEqual(len(node.trigger_pub.published), 1)
        self.assertTrue(node.trigger_pub.published[0].data)
        self.assertTrue(node.session_open)
        self.assertTrue(node.session_opened_by_wake)
        # The session opener must announce continuous mode downstream:
        # this is what disables pad F1 and enables TTS chained turns.
        self.assertEqual(len(node.continuous_start_pub.published), 1)
        self.assertTrue(node.continuous_start_pub.published[0].data)

    def test_session_refresh_does_not_republish_continuous_start(self):
        node = _FakeWakeNode()
        node.cooldown = 0.0  # isolate refresh semantics from throttling
        node._fire_trigger('aletta_wake_word')
        node._fire_trigger('aletta_wake_word')
        # Trigger fires per detection, but continuous_start is edge-like:
        # subscribers must see it exactly once per opened session.
        self.assertEqual(len(node.trigger_pub.published), 2)
        self.assertEqual(len(node.continuous_start_pub.published), 1)

    def test_trigger_blocked_by_cooldown(self):
        node = _FakeWakeNode()
        node.last_trigger_time = time.time()
        node.cooldown = 8.0
        node._fire_trigger('aletta_wake_word')
        self.assertEqual(node.trigger_pub.published, [])
        self.assertFalse(node.session_open)

    def test_trigger_blocked_while_pipeline_busy(self):
        for busy_field in ('stt_busy', 'llm_busy', 'tts_busy'):
            node = _FakeWakeNode()
            setattr(node, busy_field, True)
            node._fire_trigger('aletta_wake_word')
            self.assertEqual(
                node.trigger_pub.published, [],
                f'trigger fired while {busy_field} was active')
            self.assertFalse(node.session_open)

    def test_wake_fire_keeps_backoff_but_pad_fire_clears_it(self):
        # A wake (possibly false-accept) fire must NOT restart the
        # escalation streak...
        node = _FakeWakeNode()
        node.cooldown = 60.0
        node.false_accept_count = 3
        node._fire_trigger('aletta_wake_word')
        self.assertEqual(node.cooldown, 60.0)
        self.assertEqual(node.false_accept_count, 3)
        # ...while a human pad press is authoritative: clear the backoff.
        pad = _FakeWakeNode()
        pad.cooldown = 60.0
        pad.false_accept_count = 3
        pad._fire_trigger('pad_button')
        self.assertEqual(pad.cooldown, pad.base_cooldown)
        self.assertEqual(pad.false_accept_count, 0)

    def test_manual_stop_closes_open_session(self):
        node = _FakeWakeNode()
        node._fire_trigger('aletta_wake_word')
        node.manual_stop_callback(_Msg(True))
        self.assertFalse(node.session_open)
        self.assertEqual(len(node.continuous_stop_pub.published), 1)
        self.assertTrue(node.continuous_stop_pub.published[0].data)


class TestPadButtonGating(unittest.TestCase):
    """button_trigger_node: F1 push-to-talk, START kill switch, SELECT."""

    def test_f1_press_release_cycle_publishes_trigger_then_stop(self):
        node = _FakeButtonNode()
        node.controller_callback(_Keys(F1_KEY))
        self.assertEqual(len(node.trigger_pub.published), 1)
        self.assertTrue(node.trigger_pub.published[0].data)
        self.assertTrue(node.recording_active)
        node.controller_callback(_Keys(0))
        self.assertEqual(len(node.stop_pub.published), 1)
        self.assertTrue(node.stop_pub.published[0].data)
        self.assertFalse(node.recording_active)

    def test_f1_works_during_open_session(self):
        node = _FakeButtonNode()
        # Aletta opened a session — F1 is still allowed so the user can
        # force a turn manually.
        node._on_continuous_start(_Msg(True))
        self.assertTrue(node.continuous_mode)
        node.controller_callback(_Keys(F1_KEY))
        self.assertEqual(len(node.trigger_pub.published), 1)
        self.assertTrue(node.trigger_pub.published[0].data)
        self.assertTrue(node.recording_active)
        node.controller_callback(_Keys(0))
        self.assertEqual(len(node.stop_pub.published), 1)
        self.assertTrue(node.stop_pub.published[0].data)
        self.assertFalse(node.recording_active)

    def test_f1_release_stops_during_open_session(self):
        node = _FakeButtonNode()
        node._on_continuous_start(_Msg(True))
        self.assertTrue(node.continuous_mode)
        node.recording_active = True
        node.prev_keys = F1_KEY
        node.controller_callback(_Keys(0))
        self.assertEqual(len(node.stop_pub.published), 1)
        self.assertTrue(node.stop_pub.published[0].data)
        self.assertFalse(node.recording_active)

    def test_start_cannot_open_a_session(self):
        node = _FakeButtonNode()
        node.controller_callback(_Keys(START_KEY))
        self.assertEqual(
            node.continuous_start_pub.published, [],
            'START must never open a continuous session (Aletta-only)')
        self.assertEqual(node.continuous_stop_pub.published, [])
        self.assertEqual(
            node.trigger_pub.published, [],
            'START must never fire a voice trigger either')

    def test_start_closes_open_session(self):
        node = _FakeButtonNode()
        node.continuous_mode = True  # an Aletta session is open
        node.controller_callback(_Keys(START_KEY))
        self.assertEqual(len(node.continuous_stop_pub.published), 1)
        self.assertTrue(node.continuous_stop_pub.published[0].data)
        self.assertFalse(node.continuous_mode)

    def test_select_says_custom_phrase_without_starting_turn(self):
        node = _FakeButtonNode()
        node.controller_callback(_Keys(SELECT_KEY))
        self.assertEqual(len(node.say_pub.published), 1)
        self.assertEqual(node.say_pub.published[0].data, 'ping')
        # SELECT bypasses STT/LLM: it must not touch the turn pipeline.
        self.assertEqual(node.trigger_pub.published, [])
        self.assertEqual(node.stop_pub.published, [])


class TestSttTriggerConsumption(unittest.TestCase):
    """stt_node: the trigger is the only recording entry point."""

    def _drain(self, seconds=0.1):
        # trigger_callback spawns daemon threads; give them a moment.
        time.sleep(seconds)

    def test_trigger_starts_recording_when_idle(self):
        node = _FakeSttTriggerNode()
        node.trigger_callback(_Msg(True))
        self._drain()
        self.assertEqual(len(node.record_calls), 1)
        self.assertEqual(node.delayed_calls, [])

    def test_trigger_deferred_while_tts_echo_active(self):
        node = _FakeSttTriggerNode()
        node.tts_busy = True
        node.trigger_callback(_Msg(True))
        self._drain()
        # Deferred, not dropped: the delayed path runs instead of an
        # immediate recording that would transcribe the robot's own reply.
        self.assertEqual(node.record_calls, [])
        self.assertEqual(len(node.delayed_calls), 1)
        self.assertGreater(node.delayed_calls[0], 0.0)

    def test_trigger_ignored_while_already_recording(self):
        node = _FakeSttTriggerNode()
        node.is_recording = True
        node.trigger_callback(_Msg(True))
        self._drain()
        self.assertEqual(node.record_calls, [])
        self.assertEqual(node.delayed_calls, [])

    def test_trigger_false_is_noop(self):
        node = _FakeSttTriggerNode()
        node.trigger_callback(_Msg(False))
        self._drain()
        self.assertEqual(node.record_calls, [])
        self.assertEqual(node.delayed_calls, [])


class TestSttRetriggerConsumption(unittest.TestCase):
    """stt_node._retrigger: the always-on self-advance path is race-free."""

    def _drain(self, seconds=0.1):
        # _retrigger spawns daemon threads; give them a moment to run.
        time.sleep(seconds)

    def test_retrigger_starts_recording_when_idle(self):
        node = _FakeSttTriggerNode()
        node._retrigger()
        self._drain()
        self.assertEqual(len(node.record_calls), 1)

    def test_retrigger_ignored_while_already_recording(self):
        node = _FakeSttTriggerNode()
        node.is_recording = True
        node._retrigger()
        self._drain()
        self.assertEqual(node.record_calls, [])

    def test_retrigger_race_only_one_recording(self):
        # The lock must make the check-and-set atomic: even if two callers
        # race, only one recording thread is spawned (Bug 4 fix). The mock
        # recording target stays alive long enough (via the event) to keep
        # is_recording=True through the race window.
        node = _FakeSttTriggerNode()
        node.recording_release = threading.Event()
        node._record_and_transcribe = lambda: (
            node.record_calls.append(1),
            node.recording_release.wait(timeout=2.0))
        node.recording_release.set()
        t1 = threading.Thread(target=node._retrigger)
        t2 = threading.Thread(target=node._retrigger)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        # Let the spawned daemon recording thread finish appending.
        time.sleep(0.1)
        self.assertEqual(len(node.record_calls), 1)


if __name__ == '__main__':
    unittest.main()
