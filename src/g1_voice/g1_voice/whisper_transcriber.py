"""faster-whisper transcription wrapper."""


import ctypes
import os
import re
from collections import Counter

import numpy as np


_CTRANSLATE2_LIB_DIRS = (
    os.environ.get('CTRANSLATE2_LIB_DIR', ''),
    '/home/unitree/ct2_install/lib',
    '/home/unitree/CTranslate2/build',
    os.path.expanduser('~/ct2_install/lib'),
    os.path.expanduser('~/CTranslate2/build'),
)


def preload_ctranslate2(logger=None):
    """Preload a hand-built CUDA libctranslate2 by absolute path."""
    for lib_dir in _CTRANSLATE2_LIB_DIRS:
        if not lib_dir:
            continue
        for name in ('libctranslate2.so.4', 'libctranslate2.so'):
            path = os.path.join(lib_dir, name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
                    if logger:
                        logger.info('Preloaded CUDA runtime lib: %s' % path)
                except OSError as exc:
                    if logger:
                        logger.warn(
                            'Found %s but failed to preload: %s' % (path, exc))
                return


class WhisperTranscriber:
    """Wrapper around faster-whisper with language fallback."""

    def __init__(
        self,
        model_size='base',
        language='auto',
        device='cpu',
        beam_size=5,
        initial_prompt='',
        vad_threshold=0.5,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=200,
        normalize_audio=False,
        max_no_speech_prob=0.5,
        language_detection_min_probability=0.5,
        language_duel_override_margin=0.25,
        min_publish_avg_logprob=-2.5,
        logger=None,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.vad_threshold = vad_threshold
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.normalize_audio = normalize_audio
        self.max_no_speech_prob = max_no_speech_prob
        self.language_detection_min_probability = (
            language_detection_min_probability)
        self.language_duel_override_margin = language_duel_override_margin
        self.min_publish_avg_logprob = min_publish_avg_logprob
        self.logger = logger
        self.model = None
        self._load_model()

    def _load_model(self):
        from faster_whisper import WhisperModel

        compute_type = 'float16' if self.device == 'cuda' else 'int8'
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=compute_type)

    def transcribe(self, audio_data, language=None):
        audio_for_model = audio_data
        if self.normalize_audio:
            arr = np.asarray(audio_data, dtype=np.float32)
            if arr.size:
                peak = float(np.max(np.abs(arr)))
                if 0.0 < peak < 0.95:
                    audio_for_model = arr * (0.9 / peak)
        segments, info = self.model.transcribe(
            audio_for_model,
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt or None,
            vad_filter=True,
            vad_parameters=dict(
                onset=self.vad_threshold,
                min_silence_duration_ms=self.vad_min_silence_ms,
                speech_pad_ms=self.vad_speech_pad_ms),
        )
        return list(segments), info

    def transcribe_with_language_fallback(self, audio_data):
        configured = self.language.lower()
        if configured not in ('', 'auto'):
            segments, info = self.transcribe(audio_data, configured)
            return segments, info
        auto_segments, auto_info = self.transcribe(audio_data)
        detected = (getattr(auto_info, 'language', '') or '').lower()
        probability = getattr(auto_info, 'language_probability', 0.0)
        if (detected in ('pl', 'en')
                and probability >= self.language_detection_min_probability):
            return auto_segments, auto_info
        candidates = {}
        for lang in ('en', 'pl'):
            segs, inf = self.transcribe(audio_data, lang)
            candidates[lang] = (segs, inf)
        scored = {}
        auto_score = self.transcription_score(auto_segments)
        allow_pl_win = detected != 'en'
        for language, (segments, _) in candidates.items():
            joined = ' '.join(s.text.strip() for s in segments).strip()
            if self._looks_like_noise(joined):
                scored[language] = None
                continue
            if language == 'pl' and not allow_pl_win:
                scored[language] = None
                continue
            scored[language] = self.transcription_score(segments)
        best_language = None
        best_score = float('-inf')
        for language, score in scored.items():
            if score is not None and score > best_score:
                best_language, best_score = language, score
        margin = self.language_duel_override_margin
        if best_language is None or best_score < auto_score + margin:
            return auto_segments, auto_info
        best_segments, best_info = candidates[best_language]
        return best_segments, best_info

    def has_speech(self, segments):
        from g1_voice.speech_gate import keep_segment

        for segment in segments:
            if not (segment.text or '').strip():
                continue
            nsp = getattr(segment, 'no_speech_prob', None)
            if nsp is None:
                return True
            if keep_segment(nsp, self.max_no_speech_prob):
                return True
        return False

    def _looks_like_noise(self, text):
        t = text.strip()
        if not t:
            return True
        if not re.search(r'[A-Za-z0-9]', t):
            return True
        words = re.findall(r'[A-Za-z0-9]+', t.lower())
        if not words:
            return True
        for p in range(2, len(words) // 2 + 1):
            for i in range(len(words) - 2 * p + 1):
                if words[i:i + p] == words[i + p:i + 2 * p]:
                    repeated_words = 2 * p
                    j = i + 2 * p
                    while (j + p <= len(words)
                           and words[j:j + p] == words[i:i + p]):
                        repeated_words += p
                        j += p
                    if repeated_words / len(words) >= 0.7:
                        return True
        if len(words) >= 4:
            top, count = Counter(words).most_common(1)[0]
            if count / len(words) >= 0.5:
                return True
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
        for word in words:
            if len(word) < 6:
                continue
            for unit_len in range(1, len(word) // 3 + 1):
                unit = word[:unit_len]
                if word[:unit_len * 3] == unit * 3:
                    return True
        sentence_parts = re.split(r'[.!?;]+', text)
        sentences = [s.strip().lower() for s in sentence_parts if s.strip()]
        if len(sentences) >= 2:
            sent_counts = Counter(sentences)
            repeated = sum(c for c in sent_counts.values() if c > 1)
            if repeated / len(sentences) >= 0.5:
                return True
        return False

    @staticmethod
    def transcription_score(segments):
        text = ' '.join(segment.text.strip() for segment in segments).strip()
        if not text:
            return float('-inf')
        confidence = np.mean(
            [getattr(segment, 'avg_logprob', -10) for segment in segments])
        return float(confidence) + min(len(text), 80) * 0.001
