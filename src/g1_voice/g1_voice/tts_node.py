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

try:
    from .node_logging import (
        error as log_error,
        info as log_info,
        warning as log_warning,
        warn as log_warn,
    )
except ImportError:
    try:
        from g1_voice.node_logging import (
            error as log_error,
            info as log_info,
            warning as log_warning,
            warn as log_warn,
        )
    except ImportError:
        import os as _os
        import sys as _sys

        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import node_logging as _node_logging

        log_error = _node_logging.error
        log_info = _node_logging.info
        log_warning = _node_logging.warning
        log_warn = _node_logging.warn


class TTSNode(Node):
    def _param(self, name):
        """Return a declared parameter's value (short form of get_parameter)."""
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('tts_node')

        self.declare_parameter('voice_model_path', os.path.expanduser(
            '~/.local/share/piper/en_US-lessac-medium.onnx'))
        self.declare_parameter('polish_voice_model_path', os.path.expanduser(
            '~/.local/share/piper/pl_PL-gosia-medium/pl_PL-gosia-medium.onnx'))
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('output_device', -1)
        # Default OFF to match stt_node: after a reply the pipeline goes back
        # to waiting for "Aletta" (or a pad press). START re-arms the loop.
        self.declare_parameter('continuous_mode', False)
        self.declare_parameter('output_sample_rate', 44100)
        self.declare_parameter('network_interface', 'eth0')
        self.declare_parameter('tts_mode', 'binary')  # 'python' or 'binary'
        self.declare_parameter('piper_binary',
                               os.path.expanduser('~/.local/piper-bin/piper'))
        self.declare_parameter('piper_lib_dir',
                               os.path.expanduser('~/.local/piper-bin'))
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
        # Ask the main loop to spawn the native G1 audio worker subprocess.
        # The worker is a separate long-lived process; if it dies (or is
        # never present), the speech pipeline is un-blocked and continues
        # to work, while replay/backlog is handled by re-launching the
        # worker. This is done asynchronously so a dead worker can never
        # stall the single speech-queue thread.
        self._audio_worker = None
        self._worker_relaunch_requested = True
        self.create_timer(0.5, self._maybe_launch_worker)
        # Remaining init (params, streaming state, subscriptions) needs
        # get_logger()/create_* so it stays in the node lifecycle, just
        # after the worker hand-off above.
        self._init_after_worker()

    def _maybe_launch_worker(self):
        """
        Spawn the native G1 audio worker if it is missing/dead.

        The speech thread (_play_audio) only reads/writes an already
        installed handle; installing or replacing the subprocess happens
        exclusively here, on the executor thread, so two threads can
        never race to spawn workers.
        """
        if not self._worker_relaunch_requested:
            return
        # A worker is considered alive if it has been spawned and is
        # still running (hasn't been replaced). If it is None or dead,
        # relaunch it now.
        if self._audio_worker is not None and self._audio_worker.poll() is None:
            self._worker_relaunch_requested = False
            return

        self._worker_relaunch_requested = False
        # Drop the dead handle's pipes so repeated failures don't leak fds.
        old = self._audio_worker
        if old is not None:
            for pipe in (old.stdin, old.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'g1_audio_worker.py'
        )
        self._audio_worker = subprocess.Popen(
            [sys.executable, worker_path, self.network_interface],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0
        )
        # Confirm the worker is up before returning. If it dies during
        # startup the next _maybe_launch_worker pass catches it (poll() !=
        # None) and relaunch again instead of stalling on a broken pipe.
        try:
            ready_line = self._audio_worker.stdout.readline()
            if not ready_line:
                log_error(
                    self, 'G1 audio worker exited during startup; check its error output')
                self._audio_worker = None
                self._worker_relaunch_requested = True
                return
            ready = json.loads(ready_line.decode('utf-8'))
            if not ready.get('ready'):
                log_error(self,
                          'G1 audio worker initialization failed: '
                          + ready.get('error', 'unknown error')
                          )
                try:
                    self._audio_worker.terminate()
                except Exception:
                    pass
                self._audio_worker = None
                self._worker_relaunch_requested = True
                return
            log_info(
                self,
                f'Native G1 audio worker initialized on {self.network_interface}')
        except Exception as e:
            log_error(self,
                      f'Failed to read G1 audio worker startup message: {e}'
                      )
            try:
                self._audio_worker.terminate()
            except Exception:
                pass
            self._audio_worker = None
            self._worker_relaunch_requested = True

    def _init_after_worker(self):
        """Finish TTSNode setup once the audio-worker hand-off is staged."""
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
        # Turn generation counter: bumped on every transcript reset. Each
        # streamed chunk stamps the current generation, so response_callback
        # can tell live chunks of THIS turn apart from a stale buffer left
        # by a previous turn (which must be discarded, not re-spoken).
        self._turn_seq = 0
        self._stream_seq = -1
        # Count of complete sentences already enqueued this turn, so a
        # cumulative re-split only speaks the not-yet-spoken tail.
        self._stream_spoken = 0
        self._turn_language = 'en'
        # True while a turn is being streamed: stream_callback may enqueue.
        # response_callback closes the turn; any straggler delta arriving
        # after that is ignored so a late chunk can never leak into the
        # next turn (which would speak stale text, in either language).
        self._turn_open = False
        self._turn_had_transcript = False

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
            String,
            '/g1/stt/transcript',
            self._transcript_language_callback,
            10)
        self.create_subscription(
            Bool,
            '/g1/voice/continuous_start',
            self._continuous_start_cb,
            10)
        self.create_subscription(
            Bool,
            '/g1/voice/continuous_stop',
            self._continuous_stop_cb,
            10)
        self.create_subscription(
            Bool,
            '/g1/voice/speak_enable',
            self._speak_enable_cb,
            10)
        self.say_sub = self.create_subscription(
            String, '/g1/voice/say_text', self.say_text_callback, 10)

        self.status_pub = self.create_publisher(String, '/g1/tts/status', 10)
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)

        log_info(
            self,
            f'TTS node ready — mode: {self.tts_mode}, model: {self.model_path}')

    def _load_voice_python(self):
        if not os.path.exists(self.model_path):
            log_error(self, f'Piper model not found: {self.model_path}')
            return
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(
                self.model_path, use_cuda=self.use_cuda)
            log_info(self,
                     f'Piper English voice loaded — native rate: '
                     f'{self.voice.config.sample_rate}Hz')
            if not os.path.isfile(self.polish_model_path):
                log_error(
                    self, f'Piper Polish model not found: {self.polish_model_path}')
                return
            self.polish_voice = PiperVoice.load(
                self.polish_model_path, use_cuda=self.use_cuda)
            log_info(self,
                     f'Piper Polish voice loaded — native rate: '
                     f'{self.polish_voice.config.sample_rate}Hz')
        except Exception as e:
            log_error(self, f'Failed to load Piper Python voice: {e}')

    def _check_binary(self):
        if not os.path.exists(self.piper_binary):
            log_error(self, f'Piper binary not found: {self.piper_binary}')
            return
        log_info(self, f'Piper binary found: {self.piper_binary}')
        for language, model_path in (
                ('English', self.model_path),
                ('Polish', self.polish_model_path)):
            if not os.path.isfile(model_path):
                log_error(
                    self, f'{language} Piper voice model not found: {model_path}')
            elif not os.access(model_path, os.R_OK):
                log_error(
                    self, f'{language} Piper voice model is not readable: {model_path}')
            else:
                log_info(self,
                         f'{language} Piper voice ready: {model_path}')

    def _play_wav_bytes(self, wav_bytes: bytes) -> bool:
        """Play WAV via G1 audio worker; returns True on success."""
        try:
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

            timeout = 30.0 + 3.0 * len(audio) / float(self.output_sample_rate)
            return self._play_audio_with_timeout(
                audio, self.output_sample_rate, timeout_sec=timeout)
        except Exception as exc:
            log_error(self, f"WAV playback preparation failed: {exc}")
            return False

    def response_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            reply = data.get(
                'reply', '').strip() if isinstance(
                data, dict) else msg.data.strip()
        except (json.JSONDecodeError, KeyError):
            reply = msg.data.strip()

        if not reply:
            return

        language, reply = self._language_and_text(reply)
        if not reply:
            log_warning(self, 'Ignoring empty response after language marker')
            return

        # Only trust previously streamed chunks when they belong to THIS
        # turn: the transcript reset bumps _turn_seq, and stream_callback
        # stamps each chunk with it. A final reply arriving without a
        # transcript reset (re-prompt, missed transcript, out-of-order
        # delivery) must NOT re-split a stale buffer from a previous turn —
        # that spoke the old reply again.
        streamed_this_turn = (self._stream_saw_chunk and
                              self._stream_seq == self._turn_seq)

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
            if not streamed_this_turn:
                # No live chunks for THIS turn (or stale chunks from a past
                # turn): speak the full reply exactly once, discarding any
                # stale buffer.
                self._stream_text = ''
                self._stream_remaining = ''
                self._stream_saw_chunk = False
                self._stream_spoken = 0
                self._turn_open = True
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
            log_info(self, f'Speaking: {preview}')
            self._speech_queue.put((language, reply))
            self._speech_queue.put(_TURN_END)

    def say_text_callback(self, msg: String):
        """
        Speak exact text published on /g1/voice/say_text.

        Manual speech path (e.g. SELECT pad button): bypasses STT/LLM and the
        streaming turn state entirely. The line is queued behind any in-flight
        reply and never emits _TURN_END, so continuous mode is not retriggered.
        Mute (F3 / speak_enable) is honoured downstream in _speak(). The
        latched reply language is left untouched: consuming it here would
        steal the voice routing of an in-flight LLM reply.
        """
        text = (msg.data or '').strip()
        if not text:
            return
        # Non-consuming read: SELECT must not steal the voice routing of an
        # in-flight LLM reply (see _language_and_text).
        language, text = self._language_and_text(text, consume_latched=False)
        if not text:
            log_warning(self, 'Ignoring empty say_text after language marker')
            return
        self._warn_on_language_mismatch(text, language)
        preview = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
        log_info(self, f'Speaking custom say_text: {preview}')
        self._speech_queue.put((language, text))

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

    def _resample(
            self,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int) -> np.ndarray:
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
        """
        Send synthesized PCM to the separate native G1 audio worker.

        On any failure (dead worker, closed pipe, error reply) this method
        tears down only the worker handle IT was using and asks the node's
        timer to spawn a fresh one, so a stale thread can never kill a
        newer worker. A dead worker closes the pipe, so readline() returns
        immediately; a worker that hangs alive is bounded by the caller's
        _play_audio_with_timeout() deadline.
        """
        worker = None
        try:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if src_rate != 16000:
                audio = self._resample(audio, src_rate, 16000)

            audio = np.clip(audio, -1.0, 1.0)
            pcm_data = (audio * 32767.0).astype(np.int16).tobytes()
            if not pcm_data:
                log_warning(self, "No audio samples to play")
                return

            # Pin the handle: a late failure below must tear down exactly
            # this worker, not a newer one spawned in the meantime.
            worker = self._audio_worker
            if worker is None or worker.poll() is not None:
                log_error(self,
                          "G1 audio worker is not running — relaunching")
                raise RuntimeError("G1 audio worker not running")

            # Send PCM bytes in-memory (no temp file): command carries the
            # byte length, followed by the raw PCM on stdin.
            worker.stdin.write((json.dumps({
                "command": "play",
                "length": len(pcm_data)
            }) + "\n").encode('utf-8'))
            worker.stdin.flush()
            worker.stdin.write(pcm_data)
            worker.stdin.flush()

            # Block on the worker's single completion line. The worker
            # plays the whole utterance paced to realtime and only then
            # writes {"ok": true}, so this wait is sentence-length by
            # design. A dead worker closes the pipe -> readline() returns
            # "" immediately -> handled below. A worker that hangs alive
            # is bounded by _play_audio_with_timeout()'s outer deadline.
            response_line = worker.stdout.readline()
            if not response_line:
                raise RuntimeError(
                    "G1 audio worker closed its response pipe")

            response = json.loads(response_line.decode('utf-8'))
            if not response.get("ok"):
                raise RuntimeError(
                    response.get("error", "Unknown worker playback error")
                )

            log_info(self, "Playback complete via native G1 speaker")

        except Exception as e:
            log_error(self,
                      f"Native G1 audio playback failed: {e}")
            # Tear down exactly the handle this call pinned, and flag the
            # node's timer to spawn a fresh worker. Without the flag, the
            # (gated) _maybe_launch_worker() would never run and TTS would
            # stay dead until the whole node restarts. If a newer worker
            # was already installed, leave it alone (only ours dies).
            if worker is not None:
                if self._audio_worker is worker:
                    self._audio_worker = None
                self._worker_relaunch_requested = True
                # Closing stdin ends the worker's command loop cleanly; the
                # pipes are closed too so we don't leak fds across failures.
                for pipe in (worker.stdin, worker.stdout):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except Exception:
                            pass
                try:
                    worker.terminate()
                except Exception:
                    pass
            raise
        finally:
            pass

    def _process_queue_item(self, item) -> bool:
        """Consume one speech-queue item. Returns True if it was _TURN_END."""
        if item is _TURN_END:
            # End of turn: re-trigger continuous listening once.
            # Re-arm on all turns (not just valid transcripts) so the robot
            # does not stay deaf for up to the watchdog interval after a
            # failed/noise turn. Session closure is handled by wake_word_node's
            # max_empty_turns counter (4 consecutive empty turns), which
            # prevents infinite noise-recording loops.
            if self.continuous_mode:
                time.sleep(0.5)
                trigger = Bool()
                trigger.data = True
                self.trigger_pub.publish(trigger)
                log_info(self, 'Continuous mode — auto-triggering next turn')
            self._turn_had_transcript = False
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
                    log_info(
                        self, 'Speaking MUTED (F3) — reply received but not played.')
                elif self.tts_mode == 'python':
                    self._speak_python(text, language)
                else:
                    self._speak_binary(text, language)
            except Exception as e:
                log_error(self, f'TTS error: {e}')
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
            # Set AFTER _reset_turn(): the reset clears this flag, and setting
            # it first (the old order) left it False, so the continuous-mode
            # _TURN_END auto-trigger never fired and chained turns stalled.
            self._turn_had_transcript = True

    def _language_and_text(self, text: str, consume_latched: bool = True):
        """
        Return the explicit or remembered language and clean response text.

        Args:
            text: Raw reply, optionally prefixed with a [USER_LANGUAGE:xx]
                marker.
            consume_latched: When True (LLM reply path), a previously latched
                language is consumed (one-shot). When False (SELECT say_text
                path), the latch is left intact so an in-flight reply keeps
                its voice routing.
        """
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
            if consume_latched:
                self._response_language = None

        if language is None:
            language = "pl" if self._is_polish_text(text) else "en"
        log_info(self,
                 f'TTS language selected from latest user turn: {language}')
        return language, text.strip()

    # Backwards-compatible alias used by older call sites/tests.
    def _language_and_text_preserve(self, text: str):
        return self._language_and_text(text, consume_latched=False)

    def _play_audio_with_timeout(
            self,
            audio,
            src_rate,
            timeout_sec: float = 60.0) -> bool:
        """
        Run _play_audio in a bounded thread and bound its wall-clock time.

        _play_audio blocks on the audio worker's stdout with no timeout: a
        dead worker would block the speech-queue thread forever. Running it
        in a daemon thread lets us enforce timeout_sec and skip ahead.
        Returns True on clean playback, False on timeout/failure.
        """
        done = threading.Event()
        outcome = {'ok': False}

        def _run():
            try:
                self._play_audio(audio, src_rate)
                outcome['ok'] = True
            except Exception:
                outcome['ok'] = False
            finally:
                done.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        if not done.wait(timeout=timeout_sec):
            log_error(self,
                      f'TTS playback timed out after {timeout_sec:.0f}s — '
                      'skipping to next queue item')
            self._publish_status('error')
            return False
        return outcome['ok']

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
            log_warning(
                self, f'Reply language mismatch: turn language is {language!r} '
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
        self._turn_had_transcript = False
        self._turn_seq += 1
        self._stream_seq = -1

    def _reset_turn_state_only(self):
        """
        Clear stream buffers without touching the transcript flag.

        Kept for the unit-test harness and external callers that need a
        buffer-only reset. Bumps the generation counter exactly like
        _reset_turn() so stale llm_stream chunks of the previous turn are
        still detected and discarded (differing only in the transcript
        bookkeeping flag).
        """
        self._stream_text = ''
        self._stream_remaining = ''
        self._stream_saw_chunk = False
        self._stream_spoken = 0
        self._turn_open = True
        self._turn_seq += 1
        self._stream_seq = -1

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
            log_warning(self,
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
        self._stream_seq = self._turn_seq

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
            log_error(self, f'No Python {language} voice loaded')
            return

        chunks = []
        native_sr = voice.config.sample_rate
        for audio_chunk in voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)

        if not chunks:
            log_warn(self, 'Piper produced no audio')
            return

        audio = np.concatenate(chunks).astype(np.float32)
        # Bound playback time: a stuck audio worker must not stall the
        # single speech-queue thread forever (see _play_audio_with_timeout).
        timeout = 30.0 + 3.0 * len(audio) / float(native_sr)
        if not self._play_audio_with_timeout(audio, native_sr, timeout_sec=timeout):
            log_error(self, f"Piper playback failed — TTS turn incomplete")
            self._publish_status('error')

    @staticmethod
    def _is_polish_text(text: str) -> bool:
        # Function words and inflections that exist ONLY in Polish. Words that
        # are spelled identically in English ('robot', 'bateria', 'to', 'jak',
        # 'jest', 'tak', 'nie', 'oraz', 'ale') must NOT be listed: an English
        # reply such as "I am a robot, able to help you" matched 'robot'+'to'
        # and was misdetected as Polish. Diacritics remain the strong signal.
        if any(ch in text for ch in 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'):
            return True
        tokens = set(re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE))
        polish_words = {
            'cześć', 'dzień', 'dobry', 'dziękuję', 'proszę',
            'jestem', 'jesteś', 'możesz', 'mogę', 'pomóc',
            'się', 'niech', 'mój', 'moja', 'moje', 'twoje', 'czy',
            'dlaczego', 'ponieważ', 'witaj', 'przepraszam'
        }
        return len(tokens & polish_words) >= 2

    def _speak_binary(self, text: str, language: str):
        """Synthesize text with Piper and play it through the G1 speaker."""
        text = text.strip()
        if not text:
            log_warning(self, "Empty TTS text; skipping synthesis")
            return

        if not os.path.isfile(self.piper_binary):
            log_error(self,
                      f"Piper binary not found: {self.piper_binary}"
                      )
            return

        if not text:
            log_warning(self, "No text left for synthesis")
            return

        selected_model = (
            self.polish_model_path if language == "pl" else self.model_path
        )

        # Don't silently speak Polish with an English voice (or vice versa).
        if not os.path.isfile(selected_model):
            log_error(
                self,
                f"Selected {language} Piper model not found: {selected_model}")
            return

        if not os.access(selected_model, os.R_OK):
            log_error(self,
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

        log_info(
            self,
            f"Selected Piper language: {language}; model: {selected_model}")

        # A value of 1.0 is the model's usual baseline speed.
        # Lower values speak faster; higher values speak slower.
        # Tuned via the `length_scale` parameter (default 0.85).
        cmd = [
            self.piper_binary,
            "--model", selected_model,
            "--length_scale", f"{self.length_scale:g}",
        ]

        # Stream WAV bytes directly to stdout (no temp file) for in-memory playback.
        cmd = cmd + ["--output_file", "-"]

        wav_bytes = None
        try:
            start = time.monotonic()
            result = subprocess.run(
                cmd,
                input=(text + "\n").encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                env=env,
                check=False,
            )

            log_info(self,
                     f"Piper synthesis took {time.monotonic() - start:.2f}s"
                     )

            if result.returncode != 0:
                error_text = result.stderr.decode('utf-8', errors='replace').strip()[-1500:]
                log_error(
                    self, f"Piper failed (exit {result.returncode}): {error_text}")
                return

            wav_bytes = result.stdout
            if not wav_bytes:
                log_error(self, "Piper produced empty WAV output")
                return

            # Validate the WAV before passing it to the playback pipeline.
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                    if wav_file.getnframes() == 0:
                        log_error(self, "Piper WAV contains no audio frames")
                        return
            except (wave.Error, EOFError) as exc:
                log_error(self, f"Piper produced an invalid WAV: {exc}")
                return

        except subprocess.TimeoutExpired:
            log_error(self, "Piper synthesis timed out after 60 seconds")
            return
        except OSError as exc:
            log_error(self, f"Could not run Piper: {exc}")
            return

        playback_start = time.monotonic()
        if not self._play_wav_bytes(wav_bytes):
            log_error(self, "Audio playback failed — TTS turn incomplete")
            self._publish_status('error')
            self.is_speaking = False
            return
        log_info(self,
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
    #         log_error(self, f'Piper binary not found: {self.piper_binary}')
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
    #                 log_error(self,
    #                     f'Polish Piper model not found: {self.polish_model_path}; '
    #                     'falling back to English voice')
    #         log_info(self,
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
    #         log_info(self,
    #             f'Piper synthesis took {time.monotonic() - t0:.2f}s'
    #         )

    #         if result.returncode != 0:
    #             log_error(self,
    #                 f'Piper error: {result.stderr.decode(errors="ignore").strip()}')
    #             return
    #         with open(wav_path, 'rb') as f:
    #             wav_bytes = f.read()
    #     except FileNotFoundError:
    #         log_error(self, f'Piper binary not found: {self.piper_binary}')
    #         return
    #     except subprocess.TimeoutExpired:
    #         log_error(self, 'Piper synthesis timed out')
    #         return
    #     finally:
    #         if os.path.exists(wav_path):
    #             os.unlink(wav_path)

    #     # self._play_wav_bytes(wav_bytes)

    #     t1 = time.monotonic()
    #     self._play_wav_bytes(wav_bytes)
    #     log_info(self,
    #         f'Audio processing and playback took {time.monotonic() - t1:.2f}s'
    #     )

    def _continuous_start_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = True
            log_info(self, 'Continuous mode enabled')

    def _continuous_stop_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = False
            log_info(self, 'Continuous mode disabled')

    def _speak_enable_cb(self, msg: Bool):
        # F3 button toggles this. When False (muted), the robot still receives and
        # processes replies but does not play them through the speaker.
        self.speak_enabled = bool(msg.data)
        state = 'ENABLED' if self.speak_enabled else 'MUTED'
        log_info(self, f'Speaking {state} (via F3 / /g1/voice/speak_enable)')

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    # args=None lets rclpy parse sys.argv, which is how launch files pass
    # --ros-args/--params-file into the node; forcing [] would silently
    # drop every parameter override from voice_real.launch.py.
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
