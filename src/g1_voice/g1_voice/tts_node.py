import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import threading
import os
import re
import sys
import io
import wave
import subprocess
import tempfile
import time
import queue
import numpy as np


# Sentence boundaries for incremental TTS. A sentence is "complete" once a
# terminator (. ! ? ;) is followed by whitespace, a newline, or end-of-string.
# This guarantees we never start speaking a half-word when the next streamed
# chunk would have continued it. Used by split_complete_sentences() below.
#
# NOTE: `(?<=[.!?;])` is a fixed-width lookbehind, so the two alternatives
# `(?<=[.!?;])\s+` and `(?<=[.!?;])$` never overlap — the $ only matches the
# single position at the very end of the buffer. That means a run like
# `a.a.a...` yields one sentence per terminator only when separators exist.
# The function below is the authoritative version; it handles the
# end-of-string terminator case deterministically (e.g. "a." * 500 ->
# 500 complete sentences, not 1).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?;])\s+|\n+|(?<=[.!?;]$)')


def split_complete_sentences(buffer):
    """
    Split streamed text into complete sentences plus a trailing remainder.

    Returns (sentences, remainder):
      * sentences — stripped, non-empty pieces that end on a terminator and are
        therefore safe to speak right now.
      * remainder — the not-yet-terminated trailing text, kept for the next
        chunk (and flushed by the caller when the turn ends).

    Pure function (no ROS state) so it is unit-testable in isolation, mirroring
    speech_gate.py.
    """
    buffer = buffer.strip()
    if not buffer:
        return [], ""

    sentences = []
    start = 0
    n = len(buffer)
    i = 0
    while i < n:
        if buffer[i] in '.!?;':
            # End of a sentence when the terminator is followed by whitespace,
            # a newline, or the end of the buffer itself.
            if i + 1 >= n or buffer[i + 1] in ' \t\n\r':
                piece = buffer[start:i + 1].strip()
                if piece:
                    sentences.append(piece)
                start = i + 1
                # Skip any separating whitespace so the next sentence starts
                # at the next non-space character.
                while start < n and buffer[start] in ' \t\n\r':
                    start += 1
        i += 1

    remainder = buffer[start:].strip()
    return sentences, remainder


# Sentinel placed on the speech queue to mark "the whole LLM turn is enqueued
# and spoken". The worker loop fires the continuous-listening trigger ONCE when
# it drains this — instead of firing after every sentence — so a streamed,
# multi-sentence reply never re-triggers the microphone mid-answer.
_TURN_END = object()


class TTSNode(Node):
    def _param(self, name):
        """Return a declared parameter's value (short form of get_parameter)."""
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('tts_node')

        self.declare_parameter(
            'voice_model_path',
            os.path.expanduser('~/.local/share/piper/en_US-lessac-medium.onnx'))
        self.declare_parameter(
            'polish_voice_model_path',
            os.path.expanduser(
                '~/.local/share/piper/pl_PL-gosia-medium/pl_PL-gosia-medium.onnx'))
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('output_device', -1)
        self.declare_parameter('continuous_mode', True)
        self.declare_parameter('output_sample_rate', 44100)
        self.declare_parameter('network_interface', 'eth0')
        self.declare_parameter('tts_mode', 'binary')  # 'python' or 'binary'
        self.declare_parameter('piper_binary', os.path.expanduser('~/.local/piper-bin/piper'))
        self.declare_parameter('piper_lib_dir', os.path.expanduser('~/.local/piper-bin'))
        # Piper speaking rate. 1.0 = the model's natural baseline; lower speaks
        # faster, higher speaks slower. Exposed as a parameter so "make it feel
        # faster" is a config tweak, not a code change.
        self.declare_parameter('length_scale', 0.85)
        # Streaming flag lives in this package's own YAML block so the TTS
        # node can read it directly (previously it reached for the llm_node
        # block's `stream` key, which raised at startup). Keep the default in
        # sync with /llm_node's `stream: true`.
        self.declare_parameter('stream', True)

        self.model_path = self._param('voice_model_path')
        self.polish_model_path = self._param('polish_voice_model_path')
        self.use_cuda = self._param('use_cuda')
        self.output_device = self._param('output_device')
        self.output_sample_rate = self._param('output_sample_rate')
        self.network_interface = self._param('network_interface')
        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'g1_audio_worker.py'
        )
        self._audio_worker = subprocess.Popen(
            [sys.executable, worker_path, self.network_interface],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1
        )
        ready_line = self._audio_worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(
                'G1 audio worker exited during startup; check its error output'
            )
        ready = json.loads(ready_line)
        if not ready.get('ready'):
            self._audio_worker.terminate()
            raise RuntimeError(
                'G1 audio worker initialization failed: '
                + ready.get('error', 'unknown error')
            )
        self.get_logger().info(
            f'Native G1 audio worker initialized on {self.network_interface}'
        )
        self.tts_mode = self._param('tts_mode')
        self.piper_binary = self._param('piper_binary')
        self.piper_lib_dir = self._param('piper_lib_dir')
        self.length_scale = self._param('length_scale')

        # Playback always goes through the native g1_audio_worker (Unitree AudioHub),
        # so sounddevice output-device selection is not used. The output_device param
        # is retained only for config compatibility with older profiles.
        self.continuous_mode = self._param('continuous_mode')

        # Streaming enabled (from voice_params_real.yaml `/llm_node` block).
        self._streaming_enabled = self._param('stream')
        self._response_language = None
        self.speak_enabled = True

        # Incremental-stream state. `_turn_language` is latched once per LLM
        # turn (from the STT USER_LANGUAGE marker carried on the transcript)
        # so every streamed sentence of one reply is spoken with the same
        # voice, in either language. `_stream_saw_chunk` records that at least
        # one live llm_stream delta arrived, so response_callback can skip
        # re-enqueueing the already-streamed full reply (which would speak
        # every sentence twice).
        self._stream_text = ''
        self._stream_remaining = ''
        self._stream_saw_chunk = False
        # Count of complete sentences already enqueued this turn, so a
        # cumulative re-split only speaks the not-yet-spoken tail.
        self._stream_spoken = 0
        self._turn_language = 'en'
        # True while a turn is being streamed: stream_callback may enqueue.
        # response_callback closes the turn; any straggler delta arriving
        # after that is ignored so a late chunk can never leak into the
        # next turn (which would speak stale text, in either language).
        self._turn_open = False

        self.voice = None
        self.polish_voice = None
        self._response_language = None
        self._response_language_lock = threading.Lock()
        self._speech_queue = queue.Queue()
        self.is_speaking = False
        self.speak_lock = threading.Lock()
        self._speech_worker = threading.Thread(
            target=self._speech_worker_loop, daemon=True)
        self._speech_worker.start()

        if self.tts_mode == 'python':
            self._load_voice_python()
        else:
            self._check_binary()

        self.response_sub = self.create_subscription(
            String, 'llm_response', self.response_callback, 10)
        self.stream_sub = self.create_subscription(
            String, 'llm_stream', self.stream_callback, 10)
        self.create_subscription(
            String, '/g1/stt/transcript', self._transcript_language_callback, 10)
        self.create_subscription(Bool, '/g1/voice/continuous_start', self._continuous_start_cb, 10)
        self.create_subscription(Bool, '/g1/voice/continuous_stop', self._continuous_stop_cb, 10)
        self.create_subscription(Bool, '/g1/voice/speak_enable', self._speak_enable_cb, 10)

        self.status_pub = self.create_publisher(String, '/g1/tts/status', 10)
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)

        self.get_logger().info(f'TTS node ready — mode: {self.tts_mode}, model: {self.model_path}')

    def _load_voice_python(self):
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Piper model not found: {self.model_path}')
            return
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
            self.get_logger().info(
                f'Piper English voice loaded — native rate: '
                f'{self.voice.config.sample_rate}Hz')
            if not os.path.isfile(self.polish_model_path):
                self.get_logger().error(
                    f'Piper Polish model not found: {self.polish_model_path}')
                return
            self.polish_voice = PiperVoice.load(
                self.polish_model_path, use_cuda=self.use_cuda)
            self.get_logger().info(
                f'Piper Polish voice loaded — native rate: '
                f'{self.polish_voice.config.sample_rate}Hz')
        except Exception as e:
            self.get_logger().error(f'Failed to load Piper Python voice: {e}')

    def _check_binary(self):
        if not os.path.exists(self.piper_binary):
            self.get_logger().error(f'Piper binary not found: {self.piper_binary}')
            return
        self.get_logger().info(f'Piper binary found: {self.piper_binary}')
        for language, model_path in (
                ('English', self.model_path),
                ('Polish', self.polish_model_path)):
            if not os.path.isfile(model_path):
                self.get_logger().error(
                    f'{language} Piper voice model not found: {model_path}')
            elif not os.access(model_path, os.R_OK):
                self.get_logger().error(
                    f'{language} Piper voice model is not readable: {model_path}')
            else:
                self.get_logger().info(
                    f'{language} Piper voice ready: {model_path}')

    def _play_wav_bytes(self, wav_bytes: bytes):
        """Play a 16-bit PCM WAV through the native G1 speaker (g1_audio_worker)."""
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            n_channels = wf.getnchannels()
            src_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        audio = audio.astype(np.float32) / 32768.0

        # Piper output frequently peaks at full scale (peak ≈ 1.0), which clips
        # on the speaker and sounds distorted/crackly. Normalise to a safe peak
        # so the voice is clean and loudness is consistent.
        audio = self._normalize(audio)

        if src_rate != self.output_sample_rate:
            audio = self._resample(audio, src_rate, self.output_sample_rate)

        self._play_audio(audio, self.output_sample_rate)

    def response_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            reply = data.get('reply', '').strip() if isinstance(data, dict) else msg.data.strip()
        except (json.JSONDecodeError, KeyError):
            reply = msg.data.strip()

        if not reply:
            return

        language, reply = self._language_and_text(reply)
        if not reply:
            self.get_logger().warning('Ignoring empty response after language marker')
            return

        # Latch the turn language FIRST: live llm_stream deltas carry plain
        # text (no marker), so stream_callback speaks them with whatever
        # language is latched for this turn. Close the turn after flushing so
        # straggler deltas cannot leak sentences into the next turn.
        self._turn_language = language
        self._warn_on_language_mismatch(reply, language)

        # In streaming mode the live `llm_stream` deltas (stream_callback) have
        # already enqueued this turn's sentences as they arrived. If none
        # arrived (older llm_node, dropped chunks), sentence-split the full
        # reply here so the turn is still spoken exactly once, in the
        # language latched from the STT USER_LANGUAGE marker.
        if self._streaming_enabled:
            if not self._stream_saw_chunk:
                self._stream_text = ''
                self._stream_remaining = ''
                self._on_stream_chunk(reply, language)
            else:
                # Reconcile tail: the buffer already reflects every live
                # chunk; only refresh the unspoken remainder view.
                _, self._stream_remaining = split_complete_sentences(
                    self._stream_text)
            self._finish_stream(language)
            self._turn_open = False
        else:
            preview = f'"{reply[:80]}..."' if len(reply) > 80 else f'"{reply}"'
            self.get_logger().info(f'Speaking: {preview}')
            self._speech_queue.put((language, reply))
            self._speech_queue.put(_TURN_END)

    def _normalize(self, audio: np.ndarray, target: float = 0.9) -> np.ndarray:
        """
        Scale audio so its peak sits at `target`.

        Prevents hard clipping (Piper peaks at ≈1.0) and keeps TTS loudness
        consistent.
        """
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            audio = audio * (target / peak)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    def _play_audio(self, audio: np.ndarray, src_rate: int):
        """Send synthesized PCM to the separate native G1 audio worker."""
        temp_path = None
        try:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if src_rate != 16000:
                audio = self._resample(audio, src_rate, 16000)

            audio = np.clip(audio, -1.0, 1.0)
            pcm_data = (audio * 32767.0).astype(np.int16).tobytes()
            if not pcm_data:
                self.get_logger().warning("No audio samples to play")
                return

            with tempfile.NamedTemporaryFile(
                prefix="g1_tts_", suffix=".pcm", delete=False
            ) as temp_file:
                temp_file.write(pcm_data)
                temp_path = temp_file.name

            if self._audio_worker.poll() is not None:
                raise RuntimeError("G1 audio worker is no longer running")

            self._audio_worker.stdin.write(json.dumps({
                "command": "play",
                "path": temp_path
            }) + "\n")
            self._audio_worker.stdin.flush()

            response_line = self._audio_worker.stdout.readline()
            if not response_line:
                raise RuntimeError("G1 audio worker closed its response pipe")

            response = json.loads(response_line)
            if not response.get("ok"):
                raise RuntimeError(
                    response.get("error", "Unknown worker playback error")
                )

            self.get_logger().info("Playback complete via native G1 speaker")

        except Exception as e:
            self.get_logger().error(f"Native G1 audio playback failed: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _process_queue_item(self, item) -> bool:
        """Consume one speech-queue item. Returns True if it was _TURN_END."""
        if item is _TURN_END:
            # End of turn: re-trigger continuous listening once.
            if self.continuous_mode:
                time.sleep(0.5)
                trigger = Bool()
                trigger.data = True
                self.trigger_pub.publish(trigger)
                self.get_logger().info('Continuous mode — auto-triggering next turn')
            return True
        language, text = item
        self._speak(text, language)
        return False

    def _speech_worker_loop(self):
        while rclpy.ok():
            item = self._speech_queue.get()
            try:
                self._process_queue_item(item)
            finally:
                self._speech_queue.task_done()

    def _speak(self, text: str, language: str):
        with self.speak_lock:
            self.is_speaking = True
            self._publish_status('speaking')

            try:
                if not self.speak_enabled:
                    # F3 muted: process the reply but do not play it. Skip synthesis
                    # entirely (it would be discarded anyway) and skip straight to
                    # idle so the continuous-mode loop keeps turning.
                    self.get_logger().info(
                        'Speaking MUTED (F3) — reply received but not played.')
                elif self.tts_mode == 'python':
                    self._speak_python(text, language)
                else:
                    self._speak_binary(text, language)
            except Exception as e:
                self.get_logger().error(f'TTS error: {e}')
                self._publish_status('error')
                self.is_speaking = False
                return

            self.is_speaking = False
            self._publish_status('idle')

    def _transcript_language_callback(self, msg: String):
        """
        Remember the language of the user turn for the upcoming LLM reply.

        The transcript arriving means a NEW user turn started, so the stream
        state is reset here: the next streamed sentence belongs to this
        turn's reply, in this turn's language (pl or en).
        """
        match = re.match(
            r'^\s*\[USER_LANGUAGE:(pl|en)\]\s*\n',
            msg.data,
            flags=re.IGNORECASE,
        )
        if match:
            language = match.group(1).lower()
            with self._response_language_lock:
                self._response_language = language
            # Latch the streamed-sentence voice HERE, at turn start:
            # llm_stream deltas arrive while generation is still running, long
            # before response_callback latches the final reply. Without this,
            # the first streamed sentence(s) of each turn are spoken with the
            # PREVIOUS turn's language voice.
            self._turn_language = language
            self._reset_turn()

    def _language_and_text(self, text: str):
        """Return the explicit or remembered language and clean response text."""
        match = re.match(
            r'^\s*\[USER_LANGUAGE:(pl|en)\]\s*\n',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            with self._response_language_lock:
                self._response_language = None
            return match.group(1).lower(), text[match.end():].strip()

        with self._response_language_lock:
            language = self._response_language
            self._response_language = None

        if language is None:
            language = "pl" if self._is_polish_text(text) else "en"
        self.get_logger().info(
            f'TTS language selected from latest user turn: {language}')
        return language, text.strip()

    def _warn_on_language_mismatch(self, reply: str, language: str):
        """
        Log when the reply text's language disagrees with the turn language.

        The LLM is instructed to reply in the user's language, but it can
        drift (mixed-language input, history pull). Speaking a reply in the
        wrong voice sounds broken, so surface the mismatch in the logs.
        Warn-only by design: the diacritics/keyword heuristic is not perfect
        (e.g. English text mentioning Polish proper names like "Poznań"
        contains diacritics), so it must not override the latched voice.
        """
        looks_polish = self._is_polish_text(reply)
        expected_polish = (language == 'pl')
        if looks_polish != expected_polish:
            actual = 'pl' if looks_polish else 'en'
            self.get_logger().warning(
                f'Reply language mismatch: turn language is {language!r} '
                f'but reply text looks {actual!r}: {reply[:60]!r}...')

    # --- streamed-llm state machine (llm_stream -> incremental sentences) ---

    # Cumulative buffer of the current LLM turn. Each `llm_stream` chunk appends
    # here; we split out complete sentences for the queue. `_remaining` keeps the
    # not-yet-terminated tail between chunks.
    def _reset_turn(self):
        self._stream_text = ''
        self._stream_remaining = ''
        self._stream_saw_chunk = False
        self._stream_spoken = 0
        self._turn_open = True

    def stream_callback(self, msg: String):
        """
        Enqueue one incremental `llm_stream` delta as complete sentences.

        The llm_node publishes raw incremental text on `llm_stream` (cumulative
        chunks flushed on word boundaries, plain text) plus the final full
        reply on `llm_response` via response_callback. This path speaks the
        first sentence while generation is still running, in either language —
        the turn language was latched by response_callback/_on_stream_chunk
        from the STT USER_LANGUAGE marker. If no live chunks arrive (e.g. an
        older llm_node), response_callback falls back to sentence-splitting
        the full reply.
        """
        if not self._turn_open:
            self.get_logger().warning(
                'Ignoring llm_stream delta for a closed turn')
            return
        self._on_stream_chunk(msg.data, self._turn_language)

    def _on_stream_chunk(self, chunk: str, language: str = 'en'):
        """
        Push an incremental `llm_stream` delta into the sentence queue.

        Args:
            chunk: Streamed text delta from the LLM.
            language: The target language ('pl' or 'en') for this turn, taken
                from the STT USER_LANGUAGE marker. Passed explicitly so a
                streamed reply in either language is spoken with the correct
                voice, even though the marker is consumed from the text.
        """
        if not chunk:
            return
        self._turn_language = language
        self._stream_saw_chunk = True

        # Drop a leading language marker if it somehow arrives in-stream.
        if not self._stream_text and chunk.startswith('[USER_LANGUAGE:'):
            parts = chunk.split(']', 1)
            chunk = parts[1].lstrip() if len(parts) == 2 else ''
            if not chunk:
                return

        self._stream_text += chunk

        # The buffer is cumulative, so re-splitting returns every complete
        # sentence each time. Enqueue only the sentences beyond the count we
        # have already spoken so each is uttered exactly once. Slicing the
        # full-buffer result (rather than concatenating a stripped remainder
        # with the next chunk) preserves inter-word whitespace at boundaries.
        sentences, self._stream_remaining = split_complete_sentences(
            self._stream_text)

        for sentence in sentences[self._stream_spoken:]:
            self._speech_queue.put((language, sentence))
        self._stream_spoken = len(sentences)

    def _finish_stream(self, language: str = 'en'):
        """
        Flush the not-yet-terminated tail as the final sentence of the turn.

        This is the end-of-turn handler: once the buffer's last sentence is
        enqueued, we publish the continuous-mode trigger exactly once so the
        mic doesn't re-open mid-reply. The trigger is also queued as an item so
        the turn-ordered drain order is preserved.

        Args:
            language: Target language ('pl' or 'en') for the final sentence,
                passed explicitly to keep it in sync with the rest of the turn.
        """
        if self._stream_remaining.strip():
            self._speech_queue.put(
                (language, self._stream_remaining.strip()))
        self._speech_queue.put(_TURN_END)

    def _speak_python(self, text: str, language: str):
        voice = self.polish_voice if language == 'pl' else self.voice
        if voice is None:
            self.get_logger().error(f'No Python {language} voice loaded')
            return

        chunks = []
        native_sr = voice.config.sample_rate
        for audio_chunk in voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)

        if not chunks:
            self.get_logger().warn('Piper produced no audio')
            return

        audio = np.concatenate(chunks).astype(np.float32)
        self._play_audio(audio, native_sr)

    @staticmethod
    def _is_polish_text(text: str) -> bool:
        if any(ch in text for ch in 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'):
            return True
        tokens = set(re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE))
        polish_words = {
            'cześć', 'dzień', 'dobry', 'dziękuję', 'proszę', 'tak', 'nie',
            'jestem', 'jesteś', 'jest', 'możesz', 'mogę', 'pomóc', 'jak',
            'się', 'niech', 'mój', 'moja', 'moje', 'twoje', 'to', 'czy',
            'oraz', 'ale', 'dlaczego', 'ponieważ', 'robot', 'bateria'
        }
        return len(tokens & polish_words) >= 2

    def _speak_binary(self, text: str, language: str):
        """Synthesize text with Piper and play it through the G1 speaker."""
        text = text.strip()
        if not text:
            self.get_logger().warning("Empty TTS text; skipping synthesis")
            return

        if not os.path.isfile(self.piper_binary):
            self.get_logger().error(
                f"Piper binary not found: {self.piper_binary}"
            )
            return

        if not text:
            self.get_logger().warning("No text left for synthesis")
            return

        selected_model = (
            self.polish_model_path if language == "pl" else self.model_path
        )

        # Don't silently speak Polish with an English voice (or vice versa).
        if not os.path.isfile(selected_model):
            self.get_logger().error(
                f"Selected {language} Piper model not found: {selected_model}"
            )
            return

        if not os.access(selected_model, os.R_OK):
            self.get_logger().error(
                f"Piper model is not readable: {selected_model}"
            )
            return

        env = os.environ.copy()

        if self.piper_lib_dir:
            env["LD_LIBRARY_PATH"] = (
                self.piper_lib_dir
                + os.pathsep
                + env.get("LD_LIBRARY_PATH", "")
            )
            env.setdefault(
                "ESPEAK_DATA_PATH",
                os.path.join(self.piper_lib_dir, "espeak-ng-data"),
            )

        self.get_logger().info(
            f"Selected Piper language: {language}; model: {selected_model}"
        )

        # A value of 1.0 is the model's usual baseline speed.
        # Lower values speak faster; higher values speak slower.
        # Tuned via the `length_scale` parameter (default 0.85).
        cmd = [
            self.piper_binary,
            "--model", selected_model,
            "--length_scale", f"{self.length_scale:g}",
        ]

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tf:
                wav_path = tf.name

            start = time.monotonic()
            result = subprocess.run(
                cmd + ["--output_file", wav_path],
                input=text + "\n",
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                env=env,
                check=False,
            )

            self.get_logger().info(
                f"Piper synthesis took {time.monotonic() - start:.2f}s"
            )

            if result.returncode != 0:
                error_text = result.stderr.strip()[-1500:]
                self.get_logger().error(
                    f"Piper failed (exit {result.returncode}): {error_text}"
                )
                return

            if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
                self.get_logger().error("Piper produced an empty WAV file")
                return

            with open(wav_path, "rb") as wav_file:
                wav_bytes = wav_file.read()

            # Validate the WAV before passing it to the playback pipeline.
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                    if wav_file.getnframes() == 0:
                        self.get_logger().error("Piper WAV contains no audio frames")
                        return
            except (wave.Error, EOFError) as exc:
                self.get_logger().error(f"Piper produced an invalid WAV: {exc}")
                return

        except subprocess.TimeoutExpired:
            self.get_logger().error("Piper synthesis timed out after 60 seconds")
            return
        except OSError as exc:
            self.get_logger().error(f"Could not run Piper: {exc}")
            return
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError as exc:
                    self.get_logger().warning(
                        f"Could not remove temporary WAV: {exc}"
                    )

        playback_start = time.monotonic()
        self._play_wav_bytes(wav_bytes)
        self.get_logger().info(
            f"Audio playback processing took "
            f"{time.monotonic() - playback_start:.2f}s"
        )

    # def _speak_binary(self, text: str):
    #     """Synthesize with the standalone Piper binary and play through the
    #     G1 on-device speaker (ALSA default). The upstream g1_piper_tts helper
    #     (AudioHub PlayStream API) is not buildable with the installed
    #     unitree_sdk2 (no AudioClient::PlayStream), so we render a WAV locally
    #     with Piper and play it via sounddevice/ALSA instead."""
    #     if not os.path.exists(self.piper_binary):
    #         self.get_logger().error(f'Piper binary not found: {self.piper_binary}')
    #         return

    #     env = os.environ.copy()
    #     # The tarball ships libpiper_phonemize/libonnxruntime next to the binary.
    #     if self.piper_lib_dir:
    #         env['LD_LIBRARY_PATH'] = (
    #             self.piper_lib_dir + os.pathsep + env.get('LD_LIBRARY_PATH', ''))
    #     env.setdefault('ESPEAK_DATA_PATH',
    #                    os.path.join(self.piper_lib_dir, 'espeak-ng-data'))

    #     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
    #         wav_path = tf.name
    #     try:
    #         is_polish = self._is_polish_text(text)
    #         selected_model = self.model_path
    #         selected_language = 'en'
    #         if is_polish:
    #             if os.path.isfile(self.polish_model_path):
    #                 selected_model = self.polish_model_path
    #                 selected_language = 'pl'
    #             else:
    #                 self.get_logger().error(
    #                     f'Polish Piper model not found: {self.polish_model_path}; '
    #                     'falling back to English voice')
    #         self.get_logger().info(
    #             f'Selected Piper language: {selected_language}; model: {selected_model}')
    #         cmd = [self.piper_binary, '--model', selected_model,
    #             '--length_scale', '0.85',
    #             '--output-file', wav_path]

    #         # result = subprocess.run(
    #         #     cmd, input=text.encode('utf-8'), capture_output=True,
    #         #     timeout=60, env=env)

    #         t0 = time.monotonic()
    #         result = subprocess.run(
    #             cmd, input=text.encode('utf-8'), capture_output=True,
    #             timeout=60, env=env)
    #         self.get_logger().info(
    #             f'Piper synthesis took {time.monotonic() - t0:.2f}s'
    #         )

    #         if result.returncode != 0:
    #             self.get_logger().error(
    #                 f'Piper error: {result.stderr.decode(errors="ignore").strip()}')
    #             return
    #         with open(wav_path, 'rb') as f:
    #             wav_bytes = f.read()
    #     except FileNotFoundError:
    #         self.get_logger().error(f'Piper binary not found: {self.piper_binary}')
    #         return
    #     except subprocess.TimeoutExpired:
    #         self.get_logger().error('Piper synthesis timed out')
    #         return
    #     finally:
    #         if os.path.exists(wav_path):
    #             os.unlink(wav_path)

    #     # self._play_wav_bytes(wav_bytes)

    #     t1 = time.monotonic()
    #     self._play_wav_bytes(wav_bytes)
    #     self.get_logger().info(
    #         f'Audio processing and playback took {time.monotonic() - t1:.2f}s'
    #     )

    def _continuous_start_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = True
            self.get_logger().info('Continuous mode enabled')

    def _continuous_stop_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = False
            self.get_logger().info('Continuous mode disabled')

    def _speak_enable_cb(self, msg: Bool):
        # F3 button toggles this. When False (muted), the robot still receives and
        # processes replies but does not play them through the speaker.
        self.speak_enabled = bool(msg.data)
        state = 'ENABLED' if self.speak_enabled else 'MUTED'
        self.get_logger().info(f'Speaking {state} (via F3 / /g1/voice/speak_enable)')

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
