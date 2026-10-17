import importlib.util
import os
import sys
import types
import unittest


def _load_stt_module():
    """Import stt_node.py without a live ROS install (stub rclpy/std_msgs)."""
    for name in ('rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['rclpy'].ok = lambda: False

    class _Node:
        pass

    sys.modules['rclpy.node'].Node = _Node

    class _String:
        def __init__(self, data=''):
            self.data = data

    sys.modules['std_msgs.msg'].String = _String
    sys.modules['std_msgs.msg'].Bool = _String
    path = os.path.join(
        os.path.dirname(__file__), '..', 'g1_voice', 'stt_node.py')
    spec = importlib.util.spec_from_file_location('stt_node_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['stt_node_under_test'] = module
    spec.loader.exec_module(module)
    return module


_stt_module = _load_stt_module()


class _Seg:
    """Stand-in for a faster-whisper Segment."""

    def __init__(self, text='', no_speech_prob=0.0, avg_logprob=-0.3):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class _Info:
    def __init__(self, language='', language_probability=0.0):
        self.language = language
        self.language_probability = language_probability


class _FakeSttNode:
    """Minimal harness binding STTNode's real unbound helpers (no ROS)."""

    def __init__(self):
        self.logged = []
        self.max_no_speech_prob = 0.5
        self.language = 'auto'
        self.language_detection_min_probability = 0.75
        self.language_duel_override_margin = 0.25
        self._transcribe_calls = []
        # Bind the real STTNode helpers as instance methods so
        # _transcribe_with_language_fallback exercises the production code.
        self._has_speech = lambda segments: (
            _stt_module.STTNode._has_speech(self, segments))
        self._transcription_score = lambda segments: (
            _stt_module.STTNode._transcription_score(segments))
        self._looks_like_noise = lambda text: (
            _stt_module.STTNode._looks_like_noise(text))

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

    def _transcribe(self, audio, language=None):
        self._transcribe_calls.append(language)
        return [], _Info()


class TestHasSpeech(unittest.TestCase):
    """
    Regression: idle/quiet-room noise turns must be recognised as non-speech.

    Before the fix every ambient-noise turn ran three GPU decodes (auto +
    forced-pl + forced-en) only to reach an empty transcript, because the
    no-speech gate downstream dropped every segment. ``_has_speech`` reuses the
    exact same ``keep_segment`` rule so the duel is skipped on noise, saving two
    decodes per idle turn and removing the misleading "selected=pl" log.
    """

    def _has_speech(self, segments, max_no_speech_prob=0.5):
        node = _FakeSttNode()
        node.max_no_speech_prob = max_no_speech_prob
        return _stt_module.STTNode._has_speech(node, segments)

    def test_noise_segments_have_no_speech(self):
        # Ambient-noise hallucinations measure no_speech_prob 0.53-0.80.
        segments = [_Seg('Wytrzymaj, wytrzymaj.', 0.71),
                    _Seg('Sprawdź sieć.', 0.80)]
        self.assertFalse(self._has_speech(segments))

    def test_genuine_speech_has_speech(self):
        segments = [_Seg('Jak się masz?', 0.10)]
        self.assertTrue(self._has_speech(segments))

    def test_one_good_segment_among_noise_counts_as_speech(self):
        # A real turn can contain one hallucinated + one genuine segment; the
        # genuine one must keep the duel enabled.
        segments = [_Seg('szum', 0.85), _Seg('Cześć, dzień dobry.', 0.12)]
        self.assertTrue(self._has_speech(segments))

    def test_empty_segments_have_no_speech(self):
        self.assertFalse(self._has_speech([]))

    def test_blank_text_segments_have_no_speech(self):
        self.assertFalse(self._has_speech([_Seg('   ', 0.10)]))

    def test_missing_no_speech_prob_is_treated_as_speech(self):
        # keep_segment keeps segments with an unparseable score; _has_speech
        # must agree (never lose a real turn to a missing attribute).
        seg = _Seg('Hello there', None)
        self.assertTrue(self._has_speech([seg]))


class TestSkipDuelOnNoSpeech(unittest.TestCase):
    """
    The scored pl/en duel must be skipped when the automatic pass is noise.

    This is the user-valuable fix: when the automatic pass yields no
    speech-quality segment the duel is skipped, avoiding two wasted GPU decodes
    per idle turn while leaving every real (speech) turn unchanged.
    """

    def _run_fallback(self, auto_segments, auto_info):
        node = _FakeSttNode()

        def _fake_transcribe(audio, language=None):
            node._transcribe_calls.append(language)
            # The automatic pass returns the supplied noise/speech segments; a
            # forced-language pass returns empty (only reached if the duel runs).
            if language is None:
                return auto_segments, auto_info
            return [], _Info()

        node._transcribe = _fake_transcribe
        segments, info = _stt_module.STTNode._transcribe_with_language_fallback(
            node, 'audio')
        return node, segments, info

    def test_noise_turn_skips_duel(self):
        # Low-confidence auto (would trigger the duel) but no speech-quality
        # segment: the duel must NOT run, and the auto result is returned.
        noise = [_Seg('halucynacja', 0.78)]
        info = _Info('pl', 0.55)
        node, segments, _info_out = self._run_fallback(noise, info)
        # Only the automatic pass ran (a single None-language call).
        self.assertEqual(node._transcribe_calls, [None])
        self.assertEqual(segments, noise)

    def test_borderline_speech_turn_still_runs_duel(self):
        # Low-confidence auto WITH a genuine segment: the duel must still run so
        # a real borderline turn is classified by the scored pl/en comparison.
        speech = [_Seg('Jakiś dźwięk mowy', 0.10)]
        info = _Info('en', 0.56)
        node, _segments, _info_out = self._run_fallback(speech, info)
        # auto (None) + forced 'pl' + forced 'en' => three decodes.
        self.assertEqual(node._transcribe_calls, [None, 'pl', 'en'])

    def test_confident_auto_turn_never_runs_duel(self):
        # Confident detection short-circuits before the noise check entirely.
        speech = [_Seg('Hello there', 0.10)]
        info = _Info('en', 0.98)
        node, _segments, _info_out = self._run_fallback(speech, info)
        self.assertEqual(node._transcribe_calls, [None])


class TestDuelMarginAndNoiseGuard(unittest.TestCase):
    """
    Regression: the pl/en duel must never flip a noise/auto turn to Polish.

    Two guards, both added after the 2026-10-16 incident (5/7 duels returned
    pl when auto said en; one at en probability=0.75):
      * hallucination-like candidates (forced-pl loops on non-speech) are
        disqualified before scoring — the loop wins length AND avg_logprob,
        so scoring alone can never be trusted;
      * a clean candidate may replace the auto pass only when it beats the
        auto score by ``language_duel_override_margin``.
    """

    def _run_fallback(self, auto_segments, auto_info, forced):
        node = _FakeSttNode()

        def _fake_transcribe(audio, language=None):
            node._transcribe_calls.append(language)
            if language is None:
                return auto_segments, auto_info
            return forced.get(language, []), _Info()

        node._transcribe = _fake_transcribe
        segments, info = _stt_module.STTNode._transcribe_with_language_fallback(
            node, 'audio')
        return node, segments, info

    def test_noisy_pl_loop_disqualified_auto_kept(self):
        # Auto pass: clean-ish English below the confidence gate -> duel runs.
        auto = [_Seg('Hello there my friend', 0.10, -0.60)]
        info = _Info('en', 0.56)
        # Forced pl returns the classic repetition loop; forced en is empty.
        pl_loop = [_Seg('że to jest to, że to jest to, że to jest to,', 0.10,
                        -0.20)]
        node, segments, _info_out = self._run_fallback(auto, info, {'pl': pl_loop})
        self.assertEqual(node._transcribe_calls, [None, 'pl', 'en'])
        self.assertEqual(segments, auto)
        self.assertTrue(any('keeping auto' in m for _lvl, m in node.logged))

    def test_clean_pl_beats_auto_by_margin(self):
        # Even a clean Polish decode must NOT override an English auto
        # detection: "Aletta" is English, and Whisper on noisy mic audio
        # often confuses English for Polish. English is kept.
        auto = [_Seg('the of and', 0.10, -1.40)]
        info = _Info('en', 0.56)
        pl = [_Seg('Cześć, jak się masz dzisiaj?', 0.10, -0.30)]
        node, segments, _info_out = self._run_fallback(auto, info, {'pl': pl})
        self.assertEqual(segments, auto)
        self.assertTrue(any('keeping auto' in m for _lvl, m in node.logged))

    def test_candidate_within_margin_keeps_auto(self):
        # A marginally better pl decode must NOT override a decent auto pass
        # (the flip that produced Polish answers from English speech/noise).
        auto = [_Seg('Hello there my friend', 0.10, -0.60)]
        info = _Info('en', 0.56)
        pl = [_Seg('Witaj, przyjacielu', 0.10, -0.70)]
        node, segments, _info_out = self._run_fallback(auto, info, {'pl': pl})
        self.assertEqual(segments, auto)
        self.assertTrue(any('keeping auto' in m for _lvl, m in node.logged))

    def test_all_candidates_empty_keeps_auto(self):
        auto = [_Seg('Hello there my friend', 0.10, -0.60)]
        info = _Info('en', 0.56)
        node, segments, _info_out = self._run_fallback(auto, info, {})
        self.assertEqual(segments, auto)
        self.assertEqual(node._transcribe_calls, [None, 'pl', 'en'])


class TestWithinWordRepetition(unittest.TestCase):
    """
    Regression: Whisper loops one syllable with no space between repeats.

    Run-ons like "nienienienie", "lalala", "yeahyeahyeah", "aaaaaah" each
    tokenize as a SINGLE word, so the word-level repetition checks never see
    the repetition. A word that starts with a short unit repeated 3+ times
    (allowing a truncated final repeat) is flagged. The length floor and the
    "starts-with unit" comparison keep real words ("murmur", "chihuahua",
    "rehearsal", "beriberi") safe.
    """

    def _noise(self, text):
        return _stt_module.STTNode._looks_like_noise(text)

    def test_run_on_single_syllable_is_noise(self):
        for text in (
            'nienienienie',
            'lalalala',
            'yeahyeahyeah',
            'aaaaaah',
            'murmurmurmur',
            'alexandraalexandraalexandra',
        ):
            self.assertTrue(self._noise(text), msg=text)

    def test_run_on_polish_word_is_noise(self):
        # A real Polish word looped, both ASCII-tokenized and with diacritics
        # (the diacritic case is also caught by the phrase-repetition check).
        self.assertTrue(self._noise('wytrzymajwytrzymajwytrzymaj'))
        self.assertTrue(self._noise('sobórzęsobórzęsobórzę'))

    def test_truncated_final_repeat_is_noise(self):
        # Leading window of 3 full units still matches even if the tail is cut.
        self.assertTrue(self._noise('lalalal'))
        self.assertTrue(self._noise('nienienien'))

    def test_real_repetitive_words_are_kept(self):
        # Genuine words containing internal repetition must NOT be dropped.
        for text in (
            'murmur', 'banana', 'murmuring', 'bananas', 'papa', 'mama',
            'coco', 'kayak', 'couscous', 'chihuahua', 'rehearsal', 'beriberi',
            'froufrou', 'hotshots', 'tartar', 'cuscus', 'alpapapa',
        ):
            self.assertFalse(self._noise(text), msg=text)

    def test_real_sentences_are_kept(self):
        for text in (
            'Hello robot, can you tell me what time it is please?',
            'Cześć robocie, która jest godzina?',
            'The big brown fox jumps over the lazy dog again.',
            'I would like to understand how this works exactly.',
            'Piotr Gąski prowadzi badania na politechnice.',
        ):
            self.assertFalse(self._noise(text), msg=text)


if __name__ == '__main__':
    unittest.main()
