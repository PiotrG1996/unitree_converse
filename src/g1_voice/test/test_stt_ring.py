"""STT rolling mic buffer: wake turns start right after "Aletta"."""

import os
import sys
import threading
import time
import types
import unittest
from collections import deque

from g1_voice.logic import EchoGuard  # noqa: E402  (pure Python, no ROS)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _stub_ros():
    """Let stt_node import without a ROS install."""
    stubs = {
        'rclpy': {'ok': lambda: False},
        'rclpy.node': {'Node': object},
        'rclpy.qos': {'QoSProfile': object, 'ReliabilityPolicy': object,
                      'DurabilityPolicy': object},
        'std_msgs': {},
        'std_msgs.msg': {'Bool': object, 'Float64': object, 'String': object},
    }
    for name, attrs in stubs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for key, value in attrs.items():
            if not hasattr(module, key):
                setattr(module, key, value)


_stub_ros()
from g1_voice import stt_node  # noqa: E402

RATE = 16000
CHUNK = 2560  # 160 ms, like the G1 mic packets


class FakeSTT:
    def __init__(self):
        self.sample_rate = RATE
        self.wake_audio_offset = 0.35
        self._mic_lock = threading.Lock()
        self._ring = deque()
        self._live = None
        self._wakes = deque(maxlen=8)
        self._last_audio_end = 0.0
        self._start_wake = None
        self.mic_ring_sec = 4.0
        self.echo_tail = 0.3
        self._echo_lock = threading.Lock()
        self.clock = [1000.0]
        self._echo = EchoGuard(2.5, clock=lambda: self.clock[0])
        for name in ('_start_live', '_stop_live', '_turn_start'):
            setattr(self, name, getattr(stt_node.STTNode, name).__get__(self))

    def fill(self, end_time, n_chunks):
        """Ring of numbered chunks; chunk k holds the value k."""
        for k in range(n_chunks):
            arrived = end_time - (n_chunks - 1 - k) * CHUNK / RATE
            self._ring.append((arrived, np.full(CHUNK, float(k), dtype=np.float32)))


class RingTest(unittest.TestCase):

    def test_start_mid_chunk_keeps_only_later_samples(self):
        s = FakeSTT()
        s.fill(end_time=100.0, n_chunks=10)  # chunk 9 ends at 100.0
        pre, live = s._start_live(100.0 - 0.24)  # 1.5 chunks before the end
        audio = np.concatenate(pre)
        self.assertEqual(audio.size, int(0.24 * RATE))
        self.assertEqual(audio[0], 8.0)   # second half of chunk 8
        self.assertEqual(audio[-1], 9.0)
        self.assertIs(s._live, live)
        s._stop_live()
        self.assertIsNone(s._live)

    def test_start_now_has_no_preroll(self):
        s = FakeSTT()
        s.fill(end_time=100.0, n_chunks=10)
        pre, _ = s._start_live(100.0)
        self.assertEqual(pre, [])

    def test_stale_wake_is_ignored(self):
        s = FakeSTT()
        s._wakes.append(time.time() - 10.0)
        self.assertAlmostEqual(s._turn_start(), time.time(), delta=0.05)

    def test_wake_inside_previous_turn_is_ignored(self):
        s = FakeSTT()
        wake = time.time() - 1.0
        s._wakes.append(wake)
        s._last_audio_end = wake + 0.5  # already recorded by the last turn
        self.assertAlmostEqual(s._turn_start(), time.time(), delta=0.05)
        self.assertIsNone(s._start_wake)


class EchoStartTest(unittest.TestCase):
    """A turn after speaker playback starts echo_tail_sec after it stopped."""

    def played(self, s, quiet_for):
        s._echo.set('tts', True)
        s.clock[0] += 3.0
        s._echo.set('tts', False)
        s.clock[0] += quiet_for

    def test_answer_during_echo_guard_is_kept(self):
        s = FakeSTT()
        self.played(s, 2.5)  # the worker waited out the 2.5 s guard
        self.assertAlmostEqual(s._turn_start(), time.time() - 2.2, delta=0.05)

    def test_old_playback_is_ignored(self):
        s = FakeSTT()
        self.played(s, 10.0)  # no longer in the 4 s buffer
        self.assertAlmostEqual(s._turn_start(), time.time(), delta=0.05)

    def test_never_reaches_into_previous_turn(self):
        s = FakeSTT()
        self.played(s, 2.5)
        s._last_audio_end = time.time() - 1.0
        self.assertAlmostEqual(s._turn_start(), s._last_audio_end, delta=0.05)

    def test_wake_word_takes_priority(self):
        s = FakeSTT()
        self.played(s, 2.5)
        wake = time.time() - 0.2
        s._wakes.append(wake)
        self.assertAlmostEqual(s._turn_start(), wake - 0.35, places=6)


class NameCutRefireTest(unittest.TestCase):
    """The wake model scores one "Aletta" twice, ~1.1-1.3 s apart (21 Nov 2026)."""

    class Endpoint:
        onset = 16000  # speech 1 s into the audio

    def make(self):
        s = FakeSTT()
        s.wake_trim_offset = 0.2
        s.wake_trim_max = 1.5
        s.silence_threshold = 0.010
        s.get_logger = lambda: types.SimpleNamespace(info=lambda *a: None)
        s._trim_name = stt_node.STTNode._trim_name.__get__(s)
        return s

    def test_refire_of_the_starting_word_is_not_used(self):
        s = self.make()
        end = 1000.0
        audio = np.zeros(int(4.0 * RATE), dtype=np.float32)  # 996.0 .. 1000.0
        s._start_wake = 996.6                    # opened the session
        s._wakes.extend([996.6, 997.9])          # 997.9 = same word again
        trimmed, cut = s._trim_name(audio, self.Endpoint, end)
        # Cut at the starting word (looked back 1.5 s), never at the repeat.
        self.assertAlmostEqual(cut, 0.4, places=2)
        self.assertEqual(trimmed.size, audio.size - int(0.4 * RATE))

    def test_a_new_name_later_is_still_cut(self):
        s = self.make()
        audio = np.zeros(int(4.0 * RATE), dtype=np.float32)
        s._start_wake = None                     # turn started after playback
        s._wakes.append(997.6)
        trimmed, cut = s._trim_name(audio, self.Endpoint, 1000.0)
        self.assertAlmostEqual(cut, 1.4, places=2)


class WakeOffsetTest(unittest.TestCase):

    def test_offset_applied(self):
        s = FakeSTT()
        wake = time.time() - 0.2
        s._wakes.append(wake)
        self.assertAlmostEqual(s._turn_start(), wake - 0.35, places=6)
        self.assertEqual(s._start_wake, wake)

    def test_start_never_reaches_into_previous_turn(self):
        s = FakeSTT()
        wake = time.time() - 0.2
        s._wakes.append(wake)
        s._last_audio_end = wake - 0.1
        self.assertAlmostEqual(s._turn_start(), wake - 0.1, places=6)


if __name__ == '__main__':
    unittest.main()
