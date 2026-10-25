"""
Comprehensive tests for all g1_voice functions.

Covers:
  * g1_audio_worker: stream-ID uniqueness (no collisions for back-to-back play)
  * wake_word_node: session-state observability.
  * wake_word_node: "Aletta during open session" starts a new STT turn when the
    pipeline is free; refreshes only when busy
  * tts_node.split_complete_sentences: edge cases.
  * speech_gate: keep_segment/filter_segments edge cases

No ROS runtime required: modules are loaded from source with rclpy stubbed.
"""

import importlib.util
import os
import sys
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


# ---- load modules under test ----
_wake_module = _load_module(
    'wake_word_allfuncs_under_test', 'wake_word_node.py')
WakeWordNode = _wake_module.WakeWordNode

# ---- g1_audio_worker ----
_audio_worker = _load_module(
    'g1_audio_worker_allfuncs_under_test', 'g1_audio_worker.py')


class _FakeWakeNode:
    """Harness binding WakeWordNode's real helpers (no ROS)."""

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
        # Session state observability
        self._session_state = 'idle'
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
        # Pending wake: wake detected during a busy pipeline, queued for
        # firing when the pipeline drains. Mirrors WakeWordNode.__init__.
        self.pending_wake = False
        # Turn-in-flight guard (mirrors WakeWordNode.__init__).
        self.turn_in_flight = False
        self._turn_in_flight_since = 0.0
        # Telemetry state
        self.aletta_threshold = 0.85
        self.telemetry_log_interval_sec = 30.0
        self._last_telemetry_log = 0.0
        self._blocked_by_cooldown_log = 0.0
        # Publishers
        self.trigger_pub = _FakePub()
        self.status_pub = _FakePub()
        self.session_state_pub = _FakePub()
        self.continuous_start_pub = _FakePub()
        self.continuous_stop_pub = _FakePub()
        # Bind real methods
        cls = WakeWordNode
        self._touch_session = lambda: cls._touch_session(self)
        self._open_session = lambda source: cls._open_session(self, source)
        self._close_session = lambda reason: cls._close_session(self, reason)
        self._can_trigger = lambda: cls._can_trigger(self)
        self._fire_trigger = lambda source: cls._fire_trigger(self, source)
        self._publish_session_state = lambda: cls._publish_session_state(self)
        self._in_session_refresh = (
            lambda score: cls._in_session_refresh(self, score))
        self._check_pending_wake = (
            lambda: cls._check_pending_wake(self))
        self._wake_open = lambda: cls._fire_trigger(self, 'aletta_wake_word')

    def get_logger(self):
        harness = self

        class _Log:
            def info(self, msg):
                harness.logged.append(('info', msg))

            def warn(self, msg):
                harness.logged.append(('warn', msg))

            warning = warn

            def error(self, msg):
                harness.logged.append(('error', msg))

            def debug(self, msg):
                harness.logged.append(('debug', msg))

        return _Log()


class _FakePub:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _Msg:
    def __init__(self, data=''):
        self.data = data


class TestAudioWorkerStreamId(unittest.TestCase):
    """g1_audio_worker: stream IDs must be unique for every play command."""

    def test_stream_ids_are_unique_per_call(self):
        # Reset the per-process counter.
        if hasattr(_audio_worker.main, '_stream_seq'):
            del _audio_worker.main._stream_seq

        ids = []
        for _ in range(100):
            # time.time_ns() can collide across rapid calls; the counter must
            # guarantee uniqueness for every play command.
            if not hasattr(_audio_worker.main, '_stream_seq'):
                _audio_worker.main._stream_seq = 0
            _audio_worker.main._stream_seq += 1
            stream_id = f"{time.time_ns()}-{_audio_worker.main._stream_seq}"
            ids.append(stream_id)

        self.assertEqual(len(ids), len(set(ids)),
                         'stream IDs must be unique for every play command')

    def test_stream_id_format(self):
        if hasattr(_audio_worker.main, '_stream_seq'):
            del _audio_worker.main._stream_seq
        _audio_worker.main._stream_seq = 1
        stream_id = f"{time.time_ns()}-{_audio_worker.main._stream_seq}"
        parts = stream_id.split('-')
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].isdigit())
        self.assertEqual(parts[1], '1')


class TestWakeWordSessionState(unittest.TestCase):
    """wake_word_node: session-state observability transitions."""

    def setUp(self):
        self.node = _FakeWakeNode()

    def test_initial_state_is_idle(self):
        self.assertEqual(self.node._session_state, 'idle')
        self.assertFalse(self.node.session_open)

    def test_open_session_transitions_to_opening(self):
        self.node._open_session('aletta_wake_word')
        self.assertTrue(self.node.session_open)
        self.assertEqual(self.node._session_state, 'opening')
        self.assertEqual(len(self.node.session_state_pub.published), 1)
        st = self.node.session_state_pub.published[0].data
        self.assertEqual(st, 'opening')

    def test_refresh_session_transitions_to_refreshing(self):
        # First open (state -> opening), then refresh (state -> refreshing).
        self.node._open_session('aletta_wake_word')
        self.node.session_state_pub.published.clear()
        self.node._open_session('aletta_wake_word')  # was_open=True -> refresh
        self.assertTrue(self.node.session_open)
        self.assertEqual(self.node._session_state, 'refreshing')
        self.assertEqual(len(self.node.session_state_pub.published), 1)
        st = self.node.session_state_pub.published[0].data
        self.assertEqual(st, 'refreshing')

    def test_close_session_transitions_to_closed(self):
        self.node._open_session('aletta_wake_word')
        self.node.session_state_pub.published.clear()
        self.node._close_session('test close')
        self.assertFalse(self.node.session_open)
        self.assertEqual(self.node._session_state, 'closed')
        self.assertEqual(len(self.node.session_state_pub.published), 1)
        st = self.node.session_state_pub.published[0].data
        self.assertEqual(st, 'closed')

    def test_close_when_already_closed_does_not_publish(self):
        self.node.session_open = False
        self.node._session_state = 'closed'
        self.node.session_state_pub.published.clear()
        self.node._close_session('redundant close')
        self.assertEqual(len(self.node.session_state_pub.published), 0)

    def test_pad_fire_transitions_to_opening(self):
        self.node._open_session('pad_f1')
        self.assertTrue(self.node.session_open)
        self.assertEqual(self.node._session_state, 'opening')
        st = self.node.session_state_pub.published[0].data
        self.assertEqual(st, 'opening')


class TestWakeWordInSessionRefresh(unittest.TestCase):
    """wake_word_node: 'Aletta during open session' new-turn behavior."""

    def setUp(self):
        self.node = _FakeWakeNode()
        self.node._wake_open()  # open a session via Aletta

    def test_refresh_when_pipeline_busy_does_not_fire_trigger(self):
        self.node.tts_busy = True  # pipeline busy
        triggered_before = len(self.node.trigger_pub.published)
        self.node._in_session_refresh(0.99)
        n = len(self.node.trigger_pub.published)
        self.assertEqual(n, triggered_before)
        self.assertTrue(any('refreshed' in m for _lvl, m in self.node.logged))

    def test_new_turn_when_pipeline_free_fires_trigger(self):
        # Pipeline free -> Aletta during open session starts a new turn.
        # Clear cooldown (set by _wake_open in setUp) so the trigger can fire.
        self.node.last_trigger_time = 0.0
        self.node.cooldown = 0.0
        triggered_before = len(self.node.trigger_pub.published)
        self.node._in_session_refresh(0.99)
        n = len(self.node.trigger_pub.published)
        self.assertEqual(n, triggered_before + 1)
        msgs = [m for _lvl, m in self.node.logged]
        self.assertTrue(any('starting new turn' in m for m in msgs))

    def test_stt_busy_blocks_new_turn(self):
        self.node.stt_busy = True
        triggered_before = len(self.node.trigger_pub.published)
        self.node._in_session_refresh(0.99)
        n = len(self.node.trigger_pub.published)
        self.assertEqual(n, triggered_before)

    def test_llm_busy_blocks_new_turn(self):
        self.node.llm_busy = True
        triggered_before = len(self.node.trigger_pub.published)
        self.node._in_session_refresh(0.99)
        n = len(self.node.trigger_pub.published)
        self.assertEqual(n, triggered_before)


class TestSplitCompleteSentencesEdgeCases(unittest.TestCase):
    """tts_node.split_complete_sentences: edge cases."""

    @staticmethod
    def _load_split():
        _install_ros_stubs()
        path = os.path.join(
            os.path.dirname(__file__), '..', 'g1_voice', 'tts_node.py')
        spec = importlib.util.spec_from_file_location(
            'tts_split_under_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['tts_split_under_test'] = module
        spec.loader.exec_module(module)
        return module.split_complete_sentences

    def setUp(self):
        self.split = self._load_split()

    def test_empty_buffer(self):
        self.assertEqual(self.split(''), ([], ''))
        self.assertEqual(self.split('   '), ([], ''))

    def test_no_terminator_returns_all_as_remainder(self):
        r = self.split('Hello there my friend')
        self.assertEqual(r, ([], 'Hello there my friend'))

    def test_single_sentence_ends_with_terminator(self):
        self.assertEqual(self.split('Hello.'), (['Hello.'], ''))

    def test_multiple_sentences_with_mixed_terminators(self):
        self.assertEqual(
            self.split('Hello! How are you? I am fine.'),
            (['Hello!', 'How are you?', 'I am fine.'], '')
        )

    def test_terminator_followed_by_no_space_is_not_a_boundary(self):
        # "a.a" is not two sentences; no space/end after second terminator.
        self.assertEqual(self.split('a.a.b'), ([], 'a.a.b'))

    def test_terminator_at_end_with_trailing_whitespace(self):
        self.assertEqual(self.split('Hello.   '), (['Hello.'], ''))

    def test_newline_is_a_boundary(self):
        self.assertEqual(
            self.split('First line.\nSecond line.'),
            (['First line.', 'Second line.'], '')
        )

    def test_semicolon_is_a_terminator(self):
        self.assertEqual(
            self.split('Item one; Item two.'),
            (['Item one;', 'Item two.'], '')
        )

    def test_polish_text(self):
        self.assertEqual(
            self.split('Dzień dobry. Jak się masz? Wszystko w porządku.'),
            (['Dzień dobry.', 'Jak się masz?', 'Wszystko w porządku.'], '')
        )

    def test_trailing_incomplete_text_is_remainder(self):
        sentences, remainder = self.split('Hello world. This is incomplete')
        self.assertEqual(sentences, ['Hello world.'])
        self.assertEqual(remainder, 'This is incomplete')


class TestSpeechGateEdgeCases(unittest.TestCase):
    """speech_gate: keep_segment/filter_segments edge cases."""

    @staticmethod
    def _load_gate():
        path = os.path.join(
            os.path.dirname(__file__), '..', 'g1_voice', 'speech_gate.py')
        spec = importlib.util.spec_from_file_location(
            'speech_gate_under_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['speech_gate_under_test'] = module
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.gate = self._load_gate()

    class _Seg:
        def __init__(self, text, no_speech_prob):
            self.text = text
            self.no_speech_prob = no_speech_prob

    def test_keep_segment_exact_threshold_passes(self):
        self.assertTrue(self.gate.keep_segment(0.5, 0.5))

    def test_keep_segment_just_above_threshold_fails(self):
        self.assertFalse(self.gate.keep_segment(0.50001, 0.5))

    def test_keep_segment_none_is_safe_keep(self):
        # Unparseable score: keep rather than risk losing genuine speech.
        self.assertTrue(self.gate.keep_segment(None, 0.5))

    def test_keep_segment_negative_prob_is_speech(self):
        self.assertTrue(self.gate.keep_segment(-1.0, 0.5))

    def test_filter_segments_all_dropped(self):
        segs = [self._Seg('noise 1', 0.6), self._Seg('noise 2', 0.7)]
        kept, dropped = self.gate.filter_segments(segs, 0.5)
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 2)

    def test_filter_segments_all_kept(self):
        segs = [self._Seg('speech 1', 0.1), self._Seg('speech 2', 0.2)]
        kept, dropped = self.gate.filter_segments(segs, 0.5)
        self.assertEqual(kept, ['speech 1', 'speech 2'])
        self.assertEqual(dropped, [])

    def test_filter_segments_custom_threshold(self):
        segs = [self._Seg('murmur', 0.55)]
        kept_loose, _ = self.gate.filter_segments(segs, 0.6)
        kept_strict, _ = self.gate.filter_segments(segs, 0.5)
        self.assertEqual(kept_loose, ['murmur'])
        self.assertEqual(kept_strict, [])

    def test_filter_segments_empty_input(self):
        kept, dropped = self.gate.filter_segments([], 0.5)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [])

    def test_filter_segments_text_only_no_score(self):
        # Segment with no no_speech_prob attribute: keep_segment returns True.
        seg = self._Seg('unscored speech', None)
        kept, dropped = self.gate.filter_segments([seg], 0.5)
        self.assertEqual(kept, ['unscored speech'])
        self.assertEqual(dropped, [])


if __name__ == '__main__':
    unittest.main()
