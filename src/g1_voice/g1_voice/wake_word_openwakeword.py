"""
openWakeWord-based wake word detection for the robot's real name, "Aletta".

This is the pipeline's only voice trigger (pad buttons via
``button_trigger_node`` are the other path). It scores precomputed
melspectrogram/embedding features instead of running a full ASR decoder, so
it reacts with much lower latency than Whisper phrase matching and keeps no
second model resident.

This module wraps the openWakeWord API so ``wake_word_node`` stays import-light
and unit-testable:

* The ``openwakeword`` import is deferred to :meth:`OpenWakeWordDetector.init`,
  so importing this module (and the node) never requires the package to be
  present. When the package, its base feature models, or the custom "Aletta"
  model are unavailable, ``init`` returns ``False`` and the node logs a loud
  error and leaves wake disabled (there is no fallback backend).
* The scoring helpers (``is_detection``, ``peak_score``, ``to_int16``) are pure
  functions with no framework dependency, so the detection logic is exercised
  directly in tests with a fake model — no real ONNX/tflite file is needed.
"""

import logging
import os

import numpy as np

# openWakeWord consumes 80 ms frames of 16-bit PCM at 16 kHz (1280 samples).
# ``Model.predict`` buffers internally, so feeding a longer chunk still works
# and returns the peak score across every 80 ms frame in that chunk.
FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000
# Only used to scale normalized float audio back to 16-bit PCM.
_INT16_FULL_SCALE = 32768.0


def to_int16(audio):
    """
    Return ``audio`` as a 16-bit PCM ``numpy`` array.

    openWakeWord requires ``int16`` data (feeding float32 audio is either
    rejected or, worse, silently scale-wrong). Callers can hand us either the
    raw int16 straight off the microphone or the normalized float32 already
    produced for Whisper; both are converted correctly.
    """
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        return arr
    # Normalized float audio lives in roughly [-1, 1]; scale it back to the
    # full int16 range before casting. Clip to the valid int16 range first:
    # 1.0 * 32768 would overflow int16 and wrap to -32768, corrupting a loud
    # positive sample. ``astype`` copies, so the caller's buffer is untouched.
    scaled = arr.astype(np.float32) * _INT16_FULL_SCALE
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def is_detection(score, threshold):
    """
    Return True when a model score crosses the wake-word threshold.

    A missing or unparseable score is treated as "no detection" so a stray
    ``None`` from the model can never trigger a false wake.
    """
    try:
        return float(score) >= float(threshold)
    except (TypeError, ValueError):
        return False


def peak_score(scores):
    """
    Return the highest score in an openWakeWord ``{model_name: score}`` dict.

    openWakeWord returns one score per loaded model; a single "Aletta" model
    yields one entry, but taking the peak keeps the helper correct if several
    custom models are ever bundled together.
    """
    best = 0.0
    for value in (scores or {}).values():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > best:
            best = value
    return best


# Default user-writable location for openWakeWord models (base features +
# custom "Aletta" model). Populated by setup.sh via download_models().
DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser('~'), '.local', 'share', 'openwakeword')


def _resolve_path(value, default_filename=''):
    """Expand ``~``/env vars; fall back to ``DEFAULT_MODEL_DIR`` when empty."""
    if not value:
        if not default_filename:
            return ''
        return os.path.join(DEFAULT_MODEL_DIR, default_filename)
    # An explicit path (even relative) is honoured as given: it is resolved
    # against the CWD by openWakeWord/the caller. Only EMPTY paths fall back to
    # the default dir — prepending it to relative paths broke
    # `eval_aletta_model.py --model models/aletta_v0.1.onnx`, which silently
    # rewrote the path to ~/.local/share/openwakeword/models/... and made the
    # eval error out with "model not found" (so nothing got deploy-gated).
    return os.path.expandvars(os.path.expanduser(str(value)))


def base_model_paths(framework, melspec_path='', embedding_path=''):
    """
    Resolve the melspectrogram/embedding base-model paths for ``framework``.

    openWakeWord refuses to mix frameworks (ONNX runtime + ``.tflite`` base
    models raises ``ValueError``), so the default filenames must follow the
    selected ``framework`` — an ONNX wake-word model needs the ``.onnx`` base
    models, and vice versa. setup.sh downloads both variants.
    """
    ext = 'tflite' if framework == 'tflite' else 'onnx'
    return (
        _resolve_path(melspec_path, 'melspectrogram.' + ext),
        _resolve_path(embedding_path, 'embedding_model.' + ext),
    )


class OpenWakeWordDetector:
    """
    A single "Aletta" wake word model with graceful, logged degradation.

    Construction never loads anything; call :meth:`init` once to load the
    model. ``init`` swallows every import/download/load error and returns
    ``False`` so the caller can fall back to another detector.
    """

    def __init__(self, model_path, framework='tflite', threshold=0.5,
                 melspec_model_path='', embedding_model_path='',
                 min_consecutive_frames=2):
        """
        Store configuration only; no model is loaded until ``init``.

        ``model_path`` is the trained "Aletta" model (.tflite or .onnx).
        ``framework`` selects openWakeWord's inference backend. The
        ``melspec_model_path``/``embedding_model_path`` base feature models are
        forwarded to openWakeWord because its bundled ``resources`` directory
        is often empty on a fresh install. Empty paths resolve at ``init``
        time to ``~/.local/share/openwakeword/`` (populated by setup.sh), so
        configs stay portable across dev PC and robot users.

        ``min_consecutive_frames`` requires that many consecutive
        ``process()`` calls (one 80 ms frame each, as the node feeds them) to
        score at/above the threshold before reporting a detection. Real
        utterances hold high scores for several frames, while single-spike
        false accepts (noise transients, playback bleed) rarely do — so this
        trades ~80 ms of latency per extra frame for far fewer false wakes.
        Coerced to at least 1 (1 = fire on the first over-threshold frame,
        the old behaviour).
        """
        self.model_path = _resolve_path(model_path, 'aletta_v0.1.onnx')
        self.framework = framework
        self.threshold = threshold
        # Base feature models must match the framework (see base_model_paths).
        self.melspec_model_path, self.embedding_model_path = base_model_paths(
            framework, melspec_model_path, embedding_model_path)
        self.model = None
        self._model_name = None
        self.logger = None
        # Peak score from the most recent process() call — surfaced so the
        # node can log it and the threshold can be calibrated on the robot.
        self.last_score = 0.0
        self.min_consecutive_frames = max(1, int(min_consecutive_frames))
        # Running count of consecutive over-threshold process() calls.
        self._consecutive = 0

    def init(self, logger=None):
        """
        Load openWakeWord and the custom model. Return True on success.

        Never raises. Any failure (missing model file, missing ``openwakeword``
        package, missing base feature models) is logged and returns ``False``.
        """
        self.logger = logger
        if not self.model_path or not os.path.exists(self.model_path):
            self._log(logger, 'error',
                      'Aletta wake word model not found at '
                      f'{self.model_path!r} — wake disabled (no fallback; '
                      'pad buttons still work)')
            return False
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            self._log(logger, 'error',
                      f'openwakeword package unavailable ({exc}) — '
                      'wake disabled (no fallback; pad buttons still work)')
            return False
        try:
            kwargs = {}
            if self.melspec_model_path:
                kwargs['melspec_model_path'] = self.melspec_model_path
            if self.embedding_model_path:
                kwargs['embedding_model_path'] = self.embedding_model_path
            self.model = Model(
                wakeword_models=[self.model_path],
                inference_framework=self.framework, **kwargs)
            self._model_name = next(iter(self.model.models))
            self._log(logger, 'info',
                      'Aletta openWakeWord model loaded '
                      f'({self._model_name}, framework={self.framework}, '
                      f'threshold={self.threshold})')
            return True
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self._log(logger, 'error',
                      f'Failed to load Aletta wake word model: {exc} — '
                      'wake disabled (no fallback; pad buttons still work)')
            self.model = None
            return False

    def process(self, audio):
        """
        Feed int16 (or normalized float) PCM and return True on detection.

        ``audio`` is a chunk of 16 kHz mono samples. ``Model.predict`` handles
        the 80 ms framing internally and returns the peak score over the whole
        chunk, so one call here covers the entire chunk.

        A detection requires ``min_consecutive_frames`` consecutive calls at/
        above the threshold (the node feeds one 80 ms frame per call). One
        call counts as one step regardless of chunk length — feeding longer
        chunks weakens the confirmation, so the node streams 80 ms frames.
        """
        if self.model is None:
            return False
        frame = to_int16(audio)
        if frame.size < FRAME_SAMPLES:
            return False
        scores = self.model.predict(frame)
        if not isinstance(scores, dict):
            return False
        score = scores.get(self._model_name)
        if score is None:
            score = peak_score(scores)
        try:
            self.last_score = float(score)
        except (TypeError, ValueError):
            self.last_score = 0.0
        if is_detection(score, self.threshold):
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.min_consecutive_frames

    def reset(self):
        """Clear the model's rolling buffers (call after a trigger)."""
        self._consecutive = 0
        if self.model is not None:
            try:
                self.model.reset()
            except Exception as exc:  # noqa: BLE001 - never crash the loop
                self._log(self.logger, 'debug', f'openWakeWord reset failed: {exc}')

    def close(self):
        """Drop the loaded model so its memory can be reclaimed."""
        self.model = None
        self._model_name = None

    @staticmethod
    def _log(logger, level, message):
        """Log via a ROS logger when given, else the stdlib ``logging``."""
        if logger is not None:
            getattr(logger, level)(message)
        else:
            getattr(logging, level)(message)
