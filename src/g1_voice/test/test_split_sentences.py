import importlib.util
import os
import queue
import sys
import threading
import types
import unittest


def _load_tts_module():
    """Import tts_node.py without a live ROS install (stub rclpy/std_msgs)."""
    for name in ('rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['rclpy'].ok = lambda: False

    class _Node:  # minimal stand-in for rclpy.node.Node
        pass

    sys.modules['rclpy.node'].Node = _Node

    class _Msg:
        def __init__(self, data=''):
            self.data = data

    sys.modules['std_msgs.msg'].String = _Msg
    sys.modules['std_msgs.msg'].Bool = _Msg
    path = os.path.join(
        os.path.dirname(__file__), '..', 'g1_voice', 'tts_node.py')
    spec = importlib.util.spec_from_file_location('tts_node_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['tts_node_under_test'] = module
    spec.loader.exec_module(module)
    return module


_tts_module = _load_tts_module()
split_complete_sentences = _tts_module.split_complete_sentences
_TURN_END = _tts_module._TURN_END


class _FakeNode:
    """Minimal harness binding TTSNode's real unbound methods (no ROS)."""

    def __init__(self, module):
        self._module = module
        self._speech_queue = queue.Queue()
        self._response_language = None
        self._response_language_lock = threading.Lock()
        self._stream_text = ''
        self._stream_remaining = ''
        self._stream_saw_chunk = False
        self._turn_language = 'en'
        self.continuous_mode = False
        self.trigger_pub = None
        self.spoken = []
        self.logged = []

    def get_logger(self):
        harness = self

        class _Log:
            def info(self, msg):
                harness.logged.append(('info', msg))

            def warning(self, msg):
                harness.logged.append(('warning', msg))

            def error(self, msg):
                harness.logged.append(('error', msg))

        return _Log()

    def drain_sentences(self):
        """Return queued (language, text) pairs, skipping the turn sentinel."""
        items = []
        while True:
            try:
                item = self._speech_queue.get_nowait()
            except queue.Empty:
                return items
            if item is _TURN_END:
                continue
            items.append(item)

    # Bind the real unbound helpers so missing attrs never mask regressions.
    _is_polish_text = staticmethod(_tts_module.TTSNode._is_polish_text)
    _reset_turn = _tts_module.TTSNode._reset_turn

    def _speak(self, text, language):
        self.spoken.append((language, text))


class TestSplitCompleteSentences(unittest.TestCase):
    def test_splits_on_punctuation_space(self):
        self.assertEqual(
            split_complete_sentences('A. B. C.'),
            (['A.', 'B.', 'C.'], '')
        )

    def test_returns_all_complete_sentences_through_end_of_buffer(self):
        self.assertEqual(
            split_complete_sentences('Dzień dobry. Jak masz?'),
            (['Dzień dobry.', 'Jak masz?'], '')
        )


class TestLanguagePipeline(unittest.TestCase):
    """
    Verify speech_queue language selection for both Polish and English.

    Uses the real TTSNode methods bound to a stubbed harness: the STT
    USER_LANGUAGE marker must select the matching voice for each reply.
    """

    def setUp(self):
        self.node = _FakeNode(_tts_module)

    def test_language_marker_pl_goes_to_pl_voice(self):
        # STT publishes [USER_LANGUAGE:pl]<text>; TTS should queue with 'pl'.
        language, reply = _tts_module.TTSNode._language_and_text(
            self.node, '[USER_LANGUAGE:pl]\nDzień dobry! Jak masz?')
        self.assertEqual(language, 'pl')
        self.assertEqual(reply, 'Dzień dobry! Jak masz?')
        _tts_module.TTSNode._reset_turn(self.node)
        _tts_module.TTSNode._on_stream_chunk(self.node, reply, language)
        _tts_module.TTSNode._finish_stream(self.node, language)
        items = self.node.drain_sentences()
        self.assertTrue(items)
        for lang, _text in items:
            self.assertEqual(lang, 'pl')

    def test_language_marker_en_goes_to_en_voice(self):
        language, reply = _tts_module.TTSNode._language_and_text(
            self.node, '[USER_LANGUAGE:en]\nHello! How are you?')
        self.assertEqual(language, 'en')
        self.assertEqual(reply, 'Hello! How are you?')
        _tts_module.TTSNode._reset_turn(self.node)
        _tts_module.TTSNode._on_stream_chunk(self.node, reply, language)
        _tts_module.TTSNode._finish_stream(self.node, language)
        items = self.node.drain_sentences()
        self.assertTrue(items)
        for lang, _text in items:
            self.assertEqual(lang, 'en')

    def test_no_marker_falls_back_to_polish_heuristic(self):
        language, reply = _tts_module.TTSNode._language_and_text(
            self.node, 'Cześć, jak się masz?')
        self.assertEqual(language, 'pl')
        _tts_module.TTSNode._reset_turn(self.node)
        _tts_module.TTSNode._on_stream_chunk(self.node, reply, language)
        _tts_module.TTSNode._finish_stream(self.node, language)
        items = self.node.drain_sentences()
        self.assertTrue(items)
        for lang, _text in items:
            self.assertEqual(lang, 'pl')

    def test_queue_items_drive_turn_end_without_crash(self):
        _tts_module.TTSNode._reset_turn(self.node)
        _tts_module.TTSNode._on_stream_chunk(
            self.node, 'Hi there. How are you?', 'en')
        _tts_module.TTSNode._finish_stream(self.node, 'en')
        seen_end = False
        pairs = []
        while True:
            try:
                item = self.node._speech_queue.get_nowait()
            except queue.Empty:
                break
            if _tts_module.TTSNode._process_queue_item(self.node, item):
                seen_end = True
            else:
                pairs.append(item)
        self.assertTrue(seen_end)
        self.assertEqual(
            pairs, [('en', 'Hi there.'), ('en', 'How are you?')])
        self.assertTrue(seen_end)
        self.assertEqual(
            pairs, [('en', 'Hi there.'), ('en', 'How are you?')])

    def test_transcript_latches_stream_voice_language(self):
        # Regression: _transcript_language_callback must latch _turn_language
        # at TURN START — streamed llm_stream deltas arrive before the final
        # response_callback, so without this latch the first streamed
        # sentence of each turn is spoken with the previous turn's voice.
        msg = types.SimpleNamespace(data='[USER_LANGUAGE:pl]\nDzień dobry!')
        _tts_module.TTSNode._transcript_language_callback(self.node, msg)
        self.assertEqual(self.node._turn_language, 'pl')
        # A streamed delta queued after the transcript uses the NEW language.
        _tts_module.TTSNode._on_stream_chunk(
            self.node, 'Dzień dobry.', self.node._turn_language)
        items = self.node.drain_sentences()
        self.assertEqual(items, [('pl', 'Dzień dobry.')])

    def test_reply_language_mismatch_is_logged(self):
        # Warn-only observability: an English-looking reply on a Polish turn
        # must surface a warning; a matching pair must not.
        _tts_module.TTSNode._warn_on_language_mismatch(
            self.node, 'Dzień dobry, jak się masz?', 'pl')
        self.assertFalse(
            [m for m in self.node.logged if 'mismatch' in m[1]])
        _tts_module.TTSNode._warn_on_language_mismatch(
            self.node, 'Hello there, how are you today my friend?', 'pl')
        self.assertTrue(
            [m for m in self.node.logged if 'mismatch' in m[1]])


class TestStreamingDedup(unittest.TestCase):
    """
    Regression: each streamed sentence must be enqueued exactly once.

    The old _on_stream_chunk re-split the whole cumulative buffer on every
    delta and re-enqueued every already-spoken sentence, so a short farewell
    was repeated once per later chunk. Feed deltas incrementally and assert
    no sentence is duplicated.
    """

    def setUp(self):
        self.node = _FakeNode(_tts_module)

    def _stream_words(self, text, language='en'):
        _tts_module.TTSNode._reset_turn(self.node)
        for word in text.split(' '):
            _tts_module.TTSNode._on_stream_chunk(self.node, word + ' ', language)
        _tts_module.TTSNode._finish_stream(self.node, language)
        return self.node.drain_sentences()

    def test_multi_sentence_reply_spoken_once_each(self):
        items = self._stream_words(
            'It was nice chatting with you. Goodbye and take care!')
        texts = [text for _lang, text in items]
        self.assertEqual(texts, [
            'It was nice chatting with you.',
            'Goodbye and take care!',
        ])

    def test_single_sentence_across_many_deltas_not_repeated(self):
        # Stream a one-sentence reply across many deltas; it must be spoken
        # once (flushed by _finish_stream), not once per delta.
        items = self._stream_words('Hello there my friend.')
        texts = [text for _lang, text in items]
        self.assertEqual(texts, ['Hello there my friend.'])

    def test_polish_reply_spoken_once_each(self):
        items = self._stream_words(
            'Dzień dobry. Jak się masz dzisiaj?', language='pl')
        texts = [text for _lang, text in items]
        self.assertEqual(texts, ['Dzień dobry.', 'Jak się masz dzisiaj?'])
        for lang, _text in items:
            self.assertEqual(lang, 'pl')
