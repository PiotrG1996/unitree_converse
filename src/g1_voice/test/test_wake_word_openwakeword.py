"""
Unit tests for the custom "Aletta" openWakeWord detector and its helpers.

The detection logic lives in :mod:`wake_word_openwakeword`, which defers the
``openwakeword`` import until ``init``. These tests exercise the pure scoring
helpers directly, and drive :class:`OpenWakeWordDetector` with a fake
``openwakeword`` module injected into ``sys.modules`` — so no real
ONNX/tflite model file is required.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest

import numpy as np


def _load_module():
    """Import the module under test without a live ROS/openwakeword install."""
    for name in ('rclpy', 'rclpy.node'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    path = os.path.join(
        os.path.dirname(__file__), '..', 'g1_voice',
        'wake_word_openwakeword.py')
    spec = importlib.util.spec_from_file_location('wake_word_oww_under_test',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['wake_word_oww_under_test'] = module
    spec.loader.exec_module(module)
    return module


_oww = _load_module()
to_int16 = _oww.to_int16
is_detection = _oww.is_detection
peak_score = _oww.peak_score
OpenWakeWordDetector = _oww.OpenWakeWordDetector
FRAME_SAMPLES = _oww.FRAME_SAMPLES


class _FakeModel:
    """Stand-in for an openWakeWord ``Model`` object."""

    def __init__(self, score=0.0):
        self._score = score
        self.seen_dtypes = []
        self.reset_count = 0
        self.models = {'aletta_v0.1': object()}

    def predict(self, x):
        self.seen_dtypes.append(x.dtype)
        return {'aletta_v0.1': self._score}

    def reset(self):
        self.reset_count += 1


def _inject_fake_openwakeword(factory):
    """Install a fake ``openwakeword.model`` returning ``factory()`` as Model."""
    oww_pkg = types.ModuleType('openwakeword')
    model_mod = types.ModuleType('openwakeword.model')
    model_mod.Model = factory
    oww_pkg.model = model_mod
    sys.modules['openwakeword'] = oww_pkg
    sys.modules['openwakeword.model'] = model_mod


class TestToInt16(unittest.TestCase):
    def test_int16_passthrough(self):
        raw = np.array([0, 1000, -1000], dtype=np.int16)
        out = to_int16(raw)
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, raw)

    def test_float_scaled_back_to_full_range(self):
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        out = to_int16(audio)
        self.assertEqual(out.dtype, np.int16)
        # 1.0 scales to the int16 maximum (32767, clipped) and -1.0 to the
        # minimum (-32768); neither wraps around.
        self.assertEqual(int(out[1]), 32767)
        self.assertEqual(int(out[2]), -32768)

    def test_float_round_trip_preserves_values(self):
        # What the node feeds Whisper (float /32768) must recover the int16.
        raw = np.array([0, 12345, -23456], dtype=np.int16)
        normalized = raw.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(to_int16(normalized), raw)


class TestIsDetection(unittest.TestCase):
    def test_above_or_equal_threshold_detects(self):
        self.assertTrue(is_detection(0.9, 0.5))
        self.assertTrue(is_detection(0.5, 0.5))

    def test_below_threshold_does_not_detect(self):
        self.assertFalse(is_detection(0.49, 0.5))
        self.assertFalse(is_detection(0.0, 0.5))

    def test_unparseable_score_is_safe(self):
        # A stray None from the model must never trigger a false wake.
        self.assertFalse(is_detection(None, 0.5))
        self.assertFalse(is_detection('oops', 0.5))


class TestPeakScore(unittest.TestCase):
    def test_returns_max(self):
        self.assertEqual(peak_score({'a': 0.2, 'b': 0.7, 'c': 0.1}), 0.7)

    def test_empty_or_none_is_zero(self):
        self.assertEqual(peak_score({}), 0.0)
        self.assertEqual(peak_score(None), 0.0)

    def test_ignores_unparseable(self):
        self.assertEqual(peak_score({'a': None, 'b': 0.3}), 0.3)


class TestBaseModelPaths(unittest.TestCase):
    """
    Base feature models must match the selected inference framework.

    openWakeWord raises "The onnx inference framework is selected, but tflite
    models were provided!" on a mismatch, which would silently disable wake on
    an ONNX-only install — so that is asserted explicitly.
    """

    def test_onnx_framework_uses_onnx_base_models(self):
        melspec, embedding = _oww.base_model_paths('onnx')
        self.assertTrue(melspec.endswith('melspectrogram.onnx'))
        self.assertTrue(embedding.endswith('embedding_model.onnx'))

    def test_tflite_framework_uses_tflite_base_models(self):
        melspec, embedding = _oww.base_model_paths('tflite')
        self.assertTrue(melspec.endswith('melspectrogram.tflite'))
        self.assertTrue(embedding.endswith('embedding_model.tflite'))

    def test_detector_resolves_matching_paths(self):
        detector = OpenWakeWordDetector(model_path='x.tflite',
                                        framework='tflite')
        self.assertTrue(detector.melspec_model_path.endswith(
            'melspectrogram.tflite'))
        self.assertTrue(detector.embedding_model_path.endswith(
            'embedding_model.tflite'))

    def test_detector_onnx_defaults_to_onnx_model(self):
        # openWakeWord 0.6.0 exports ONNX only, so that is the built-in default.
        detector = OpenWakeWordDetector(model_path='', framework='onnx')
        self.assertTrue(detector.model_path.endswith('aletta_v0.1.onnx'))
        self.assertTrue(detector.melspec_model_path.endswith('.onnx'))

    def test_explicit_paths_are_honoured(self):
        melspec, embedding = _oww.base_model_paths(
            'onnx', '/custom/mel.tflite', '/custom/emb.tflite')
        self.assertEqual(melspec, '/custom/mel.tflite')
        self.assertEqual(embedding, '/custom/emb.tflite')


class TestDetectorInitFallback(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop('openwakeword', None)
        sys.modules.pop('openwakeword.model', None)

    def test_missing_model_file_returns_false(self):
        detector = OpenWakeWordDetector(model_path='/no/such/model.tflite')
        self.assertFalse(detector.init())
        self.assertIsNone(detector.model)

    def test_missing_package_returns_false(self):
        with tempfile.NamedTemporaryFile(suffix='.tflite') as tmp:
            detector = OpenWakeWordDetector(model_path=tmp.name)
            # Force the deferred import to raise by breaking the import hook.
            import builtins
            real_import = builtins.__import__

            def _boom(name, *a, **k):
                if name.startswith('openwakeword'):
                    raise ImportError('no openwakeword')
                return real_import(name, *a, **k)

            builtins.__import__ = _boom
            try:
                self.assertFalse(detector.init())
            finally:
                builtins.__import__ = real_import
            self.assertIsNone(detector.model)

    def test_load_error_returns_false(self):
        def _boom(*a, **k):
            raise RuntimeError('bad model')

        with tempfile.NamedTemporaryFile(suffix='.tflite') as tmp:
            _inject_fake_openwakeword(_boom)
            detector = OpenWakeWordDetector(model_path=tmp.name)
            self.assertFalse(detector.init())
            self.assertIsNone(detector.model)

    def test_successful_init_records_model_name(self):
        def _factory(*a, **k):
            return _FakeModel()

        with tempfile.NamedTemporaryFile(suffix='.tflite') as tmp:
            _inject_fake_openwakeword(_factory)
            detector = OpenWakeWordDetector(model_path=tmp.name, threshold=0.6)
            self.assertTrue(detector.init())
            self.assertEqual(detector._model_name, 'aletta_v0.1')


class TestDetectorProcess(unittest.TestCase):
    def _make_ready(self, score, threshold=0.5, min_frames=1):
        # min_frames=1 keeps these plumbing tests (dtype conversion,
        # threshold semantics) independent of the confirmation feature,
        # which TestConsecutiveConfirmation covers separately.
        model = _FakeModel(score)
        detector = OpenWakeWordDetector(
            model_path='x.tflite', threshold=threshold,
            min_consecutive_frames=min_frames)
        detector.model = model
        detector._model_name = 'aletta_v0.1'
        return detector, model

    def test_detection_fires_and_uses_int16(self):
        detector, model = self._make_ready(0.9)
        chunk = (np.random.randn(FRAME_SAMPLES * 2) * 3000).astype(np.int16)
        self.assertTrue(detector.process(chunk))
        # The model must receive int16, never float32.
        self.assertEqual(model.seen_dtypes[-1], np.int16)

    def test_no_detection_below_threshold(self):
        detector, _ = self._make_ready(0.2)
        chunk = (np.random.randn(FRAME_SAMPLES * 2) * 3000).astype(np.int16)
        self.assertFalse(detector.process(chunk))

    def test_float_audio_converted_before_predict(self):
        detector, model = self._make_ready(0.9)
        audio = np.random.randn(FRAME_SAMPLES * 2).astype(np.float32) / 2.0
        self.assertTrue(detector.process(audio))
        self.assertEqual(model.seen_dtypes[-1], np.int16)

    def test_short_chunk_is_ignored(self):
        detector, model = self._make_ready(0.99)
        # Fewer samples than one 80 ms frame -> no prediction at all.
        self.assertFalse(detector.process(np.zeros(10, dtype=np.int16)))
        self.assertEqual(model.seen_dtypes, [])

    def test_process_without_model_is_safe(self):
        detector = OpenWakeWordDetector(model_path='x.tflite')
        self.assertFalse(
            detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)))

    def test_reset_and_close(self):
        detector, model = self._make_ready(0.9)
        detector.reset()
        self.assertEqual(model.reset_count, 1)
        detector.close()
        self.assertIsNone(detector.model)


class _SequenceModel:
    """Fake model returning a scripted sequence of scores (last one repeats)."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.models = {'aletta_v0.1': object()}

    def predict(self, x):
        if len(self._scores) > 1:
            return {'aletta_v0.1': self._scores.pop(0)}
        return {'aletta_v0.1': self._scores[0]}


class TestConsecutiveConfirmation(unittest.TestCase):
    """The min_consecutive_frames gate (default 2) rejects single spikes."""

    @staticmethod
    def _ready(scores, min_frames=2):
        detector = OpenWakeWordDetector(
            model_path='x.tflite', threshold=0.5,
            min_consecutive_frames=min_frames)
        detector.model = _SequenceModel(scores)
        detector._model_name = 'aletta_v0.1'
        return detector

    @staticmethod
    def _frame():
        return np.zeros(FRAME_SAMPLES, dtype=np.int16)

    def test_single_over_threshold_frame_does_not_fire(self):
        detector = self._ready([0.9])
        self.assertFalse(detector.process(self._frame()))
        self.assertTrue(detector.process(self._frame()))

    def test_below_threshold_resets_counter(self):
        detector = self._ready([0.9, 0.2, 0.9, 0.9])
        self.assertFalse(detector.process(self._frame()))  # 1st hit
        self.assertFalse(detector.process(self._frame()))  # below → reset
        self.assertFalse(detector.process(self._frame()))  # 1st hit again
        self.assertTrue(detector.process(self._frame()))   # 2nd consecutive

    def test_reset_clears_counter(self):
        detector = self._ready([0.9])
        self.assertFalse(detector.process(self._frame()))
        detector.reset()
        self.assertFalse(detector.process(self._frame()))
        self.assertTrue(detector.process(self._frame()))

    def test_min_frames_one_fires_immediately(self):
        detector = self._ready([0.9], min_frames=1)
        self.assertTrue(detector.process(self._frame()))

    def test_min_frames_coerced_to_at_least_one(self):
        detector = self._ready([0.9], min_frames=0)
        self.assertEqual(detector.min_consecutive_frames, 1)
        self.assertTrue(detector.process(self._frame()))

    def test_short_chunk_does_not_advance_counter(self):
        detector = self._ready([0.9])
        self.assertFalse(detector.process(np.zeros(10, dtype=np.int16)))
        self.assertFalse(detector.process(self._frame()))  # 1st counted hit
        self.assertTrue(detector.process(self._frame()))   # 2nd consecutive


if __name__ == '__main__':
    unittest.main()
