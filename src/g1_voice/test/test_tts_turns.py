"""TTS turn routing: streamed deltas and replies map to the right user turn."""

import os
import queue
import sys
import threading
import types
import unittest
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _stub_ros():
    """Let tts_node import without a ROS install."""
    class Msg:
        def __init__(self, data=None):
            self.data = data

    stubs = {
        'rclpy': {'ok': lambda: False},
        'rclpy.node': {'Node': object},
        'rclpy.qos': {'QoSProfile': object, 'DurabilityPolicy': object,
                      'ReliabilityPolicy': object},
        'std_msgs': {},
        'std_msgs.msg': {'Bool': Msg, 'String': Msg},
    }
    for name, attrs in stubs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for key, value in attrs.items():
            if not hasattr(module, key):
                setattr(module, key, value)
    return Msg


Msg = _stub_ros()
from g1_voice import tts_node  # noqa: E402


class FakeTTS:
    """Binds the real TTSNode turn handlers to plain state (no ROS)."""

    def __init__(self):
        self._turns = deque()
        self._queue = queue.Queue()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self.turn_timeout = 90.0
        self.first_clause_words = 0  # plain sentence splitting unless a test sets it
        for name in ('_on_transcript', '_on_stream', '_on_response', '_on_say',
                     '_enqueue_text', '_put', '_split', '_emit'):
            setattr(self, name, getattr(tts_node.TTSNode, name).__get__(self))
        self._guess_language = tts_node.TTSNode._guess_language

    def get_logger(self):
        return types.SimpleNamespace(warn=lambda msg: None)

    def drain(self, markers=False):
        items = []
        while not self._queue.empty():
            item = self._queue.get()
            if isinstance(item, tts_node._FirstAudio):
                if markers:
                    items.append('FIRST')
                continue
            items.append('END' if item is tts_node._TURN_END else item)
        return items


class TurnRoutingTest(unittest.TestCase):

    def setUp(self):
        self.tts = FakeTTS()

    def test_streamed_reply_spoken_once(self):
        t = self.tts
        t._on_transcript(Msg('[USER_LANGUAGE:pl]\nCześć'))
        t._on_stream(Msg('Cześć! Jak '))
        t._on_stream(Msg('się masz? Dobrze'))
        t._on_response(Msg('Cześć! Jak się masz? Dobrze'))
        self.assertEqual(t.drain(), [('pl', 'Cześć!'), ('pl', 'Jak się masz?'),
                                     ('pl', 'Dobrze'), 'END'])

    def test_non_streamed_reply(self):
        t = self.tts
        t._on_transcript(Msg('[USER_LANGUAGE:en]\nhi'))
        t._on_response(Msg('Hello. I am Aletta'))
        self.assertEqual(t.drain(), [('en', 'Hello.'), ('en', 'I am Aletta'), 'END'])

    def test_empty_reply_still_ends_turn(self):
        t = self.tts
        t._on_transcript(Msg('[USER_LANGUAGE:en]\nhi'))
        t._on_response(Msg(''))
        self.assertEqual(t.drain(), ['END'])

    def test_overlapping_turns_keep_their_own_replies(self):
        t = self.tts
        t._on_transcript(Msg('[USER_LANGUAGE:en]\nfirst'))
        t._on_stream(Msg('Reply one. '))
        t._on_transcript(Msg('[USER_LANGUAGE:pl]\ndrugi'))  # barge-in
        t._on_stream(Msg('Still one.'))
        t._on_response(Msg('Reply one. Still one.'))
        t._on_stream(Msg('Odpowiedź dwa.'))
        t._on_response(Msg('Odpowiedź dwa.'))
        self.assertEqual(t.drain(), [
            ('en', 'Reply one.'), ('en', 'Still one.'), 'END',
            ('pl', 'Odpowiedź dwa.'), 'END'])

    def test_reply_without_transcript_is_spoken_without_turn_end(self):
        t = self.tts
        t._on_response(Msg('Dzień dobry.'))
        self.assertEqual(t.drain(), [('pl', 'Dzień dobry.')])

    def test_say_text(self):
        t = self.tts
        t._on_say(Msg('Hello! I am Aletta.'))
        self.assertEqual(t.drain(), [('en', 'Hello!'), ('en', 'I am Aletta.')])

    def test_first_audio_marker_once_per_turn(self):
        t = self.tts
        t._on_transcript(Msg('[USER_LANGUAGE:en]\nhi'))
        t._on_stream(Msg('One. Two. '))
        t._on_stream(Msg('Three.'))
        t._on_response(Msg('One. Two. Three.'))
        self.assertEqual(t.drain(markers=True), [
            'FIRST', ('en', 'One.'), ('en', 'Two.'), ('en', 'Three.'), 'END'])
        self.assertEqual(t._pending, 5)  # every queued item is counted

    def test_streamed_first_clause_is_spoken_early(self):
        t = self.tts
        t.first_clause_words = 5
        t._on_transcript(Msg('[USER_LANGUAGE:en]\nbattery?'))
        t._on_stream(Msg('My battery status is 30%, and I am'))
        self.assertEqual(t.drain(), [('en', 'My battery status is 30%,')])
        t._on_stream(Msg(' not charging. Anything else?'))
        t._on_response(Msg('My battery status is 30%, and I am not charging. Anything else?'))
        self.assertEqual(t.drain(), [('en', 'and I am not charging.'),
                                     ('en', 'Anything else?'), 'END'])


if __name__ == '__main__':
    unittest.main()
