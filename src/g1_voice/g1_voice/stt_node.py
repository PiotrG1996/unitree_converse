"""
Speech-to-text node: one recorded and transcribed turn per trigger.

On /g1/voice/trigger it records, transcribes with faster-whisper and
publishes the marker-prefixed text (see logic.tag_language) on
/g1/stt/transcript.

Turn status on /g1/stt/status: recording -> transcribing -> (empty_turn) ->
idle. ``idle`` is always published, even if the turn crashes.

In continuous mode (an open "Aletta" session) the next turn starts by itself
after an empty turn; after a real transcript tts_node triggers it once the
reply has been spoken, and a watchdog covers a reply that never comes.

With the UDP mic the node keeps the last ``mic_ring_sec`` of audio. A turn
started by the wake word (/g1/wake/detected carries the detection time)
begins right after "Aletta" instead of when the trigger arrives, so
"Aletta, what's your battery?" works in one breath and the name itself never
reaches Whisper.

Inside a session people say the name again before each question, while a
turn is already recording. wake_word_node still reports those detections,
and the audio up to the end of the name is cut off before Whisper sees it
(see logic.name_cut): the misheard name used to garble the question
("Aletta, what is two times two" -> "All that for both is two times two").

Echo guard: no recording starts while the G1 speaker plays or within
``tts_echo_guard_sec`` after it. Playback is known from our own
/g1/tts/status and from the audio board's ``play_state`` on /audio_msg,
which also covers Unitree's built-in assistant (its male English voice).

While /g1/voice/assistant_mode is ``unitree`` (Unitree's GPT is the
default answerer) triggers outside an Aletta session are ignored; an
"Aletta" session works as usual.
"""

import ctypes
import glob
import os
import queue
import socket
import threading
import time
import wave
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String

from g1_voice.logic import (EchoGuard, Endpointer, board_play_state, choose_transcription,
                            filter_segments, has_speech, is_only_name, looks_like_noise,
                            echoes_prompt, name_cut, open_multicast_socket,
                            snap_cut_to_pause, tag_language)

# Vocabulary hints (Whisper initial_prompt): words people ask the robot about
# that Whisper does not expect. On recorded G1 turns this turned "all of your
# battle status" into "What was your battery status?" and "running across
# nodes" into "running ROS nodes" (with model medium). Words only, no
# ready-made questions, so the hint can't put sentences in people's mouths.
PROMPT_EN = ('Aletta, Unitree G1, battery, charging, ROS 2, nodes, topics, Wi-Fi, '
             'network, Ollama, Whisper, Piper, Poznan University of Technology.')
# Polish: off by default. On 6 recorded Polish turns it fixed nothing and
# once echoed itself ("Aletta, Ropis, KF1, stu"); try it with initial_prompt_pl.
PROMPT_PL = ''

# Wake detections older than this when the turn starts are not used.
WAKE_MAX_AGE_SEC = 3.0
# A detection this soon after the one that started the turn is the same word
# scored again (the model's ~1.3 s window), never a new name.
WAKE_REFIRE_SEC = 1.5

# Hand-built CUDA CTranslate2 (the Jetson pip wheel ships without it).
_CT2_DIRS = (os.environ.get('CTRANSLATE2_LIB_DIR', ''),
             os.path.expanduser('~/ct2_install/lib'),
             os.path.expanduser('~/CTranslate2/build'))


def _preload_ctranslate2(logger):
    """
    Load libctranslate2 by absolute path before faster_whisper imports it.

    systemd's LD_LIBRARY_PATH does not contain it, and changing the variable
    at runtime has no effect. RTLD_LOCAL matters: RTLD_GLOBAL corrupted the
    heap inside the ROS process ("free(): invalid pointer").
    """
    for lib_dir in filter(None, _CT2_DIRS):
        path = os.path.join(lib_dir, 'libctranslate2.so.4')
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
            logger.info('Preloaded {}'.format(path))
            return


class STTNode(Node):

    def __init__(self):
        super().__init__('stt_node')
        p = self._declare
        # A faster-whisper name ('small', 'large-v3-turbo') or a model folder.
        self.model_size = os.path.expanduser(p('model_size', 'small'))
        # '' = float16 on CUDA, int8 on CPU; 'int8_float16' halves GPU memory.
        self.compute_type = p('compute_type', '')
        self.language = p('language', 'auto')
        self.device = p('device', 'cuda')
        # Allow environment variable override: STT_DEVICE=cpu
        env_device = os.environ.get('STT_DEVICE')
        if env_device is not None:
            self.device = env_device
            self.get_logger().info(f"STT device overridden by env: {self.device}")

        # Check GPU memory pressure when using CUDA
        if self.device == 'cuda':
            from g1_voice.logic import check_gpu_pressure, log_gpu_pressure
            is_pressure, free_gb, total_gb = check_gpu_pressure(free_gb_threshold=2.0)
            log_gpu_pressure('stt_node', free_gb, total_gb)
            if is_pressure:
                self.get_logger().warn(
                    f"GPU VRAM under pressure ({free_gb:.1f}/{total_gb:.1f} GB free). "
                    "STT will proceed but may be slow; consider CPU mode."
                )
        self.sample_rate = p('sample_rate', 16000)
        self.recording_duration = p('recording_duration', 10.0)
        self.silence_threshold = p('silence_threshold', 0.010)
        # A turn only starts on clearly voiced sound: room ~0.005, the robot's
        # fading echo <= 0.013, a person talking 0.03-0.13 (int16 RMS / 32768).
        self.speech_start_threshold = p('speech_start_threshold', 0.02)
        self.silence_duration = p('silence_duration', 1.2)
        # Max silence before the user starts talking (then an empty turn).
        self.max_wait = p('max_wait_for_speech', 8.0)
        self.use_udp_mic = p('use_udp_mic', True)
        self.udp_group = p('udp_multicast_group', '239.168.123.161')
        self.udp_port = p('udp_port', 5555)
        self.udp_local_ip = p('udp_local_ip', '192.168.123.164')
        self.continuous_mode = p('continuous_mode', False)
        self.watchdog_interval = p('watchdog_interval', 20.0)
        self.reply_timeout = p('reply_timeout_sec', 60.0)
        self.max_no_speech_prob = p('max_no_speech_prob', 0.85)
        # Drop segments Whisper itself is unsure of (0.0 disables).
        self.min_avg_logprob = p('min_avg_logprob', -1.0) or None
        # Whisper's combined rule: a likely-silent segment must also be confident.
        self.noisy_no_speech_prob = p('noisy_no_speech_prob', 0.5)
        self.noisy_min_avg_logprob = p('noisy_min_avg_logprob', -0.7)
        self.beam_size = p('beam_size', 5)
        # Audio kept before a turn starts, and how far a wake-started turn
        # looks back: the detection comes 0.3-1.0 s after "Aletta" (later when
        # people don't pause), and the name is cut at the pause after it.
        self.mic_ring_sec = p('mic_ring_sec', 4.0)
        self.wake_audio_offset = p('wake_audio_offset_sec', 1.5)
        # Name said inside a recording turn: cut the audio this long before
        # the detection, if the detection is within wake_trim_max_sec of the
        # speech onset. Smaller than wake_audio_offset_sec: the detector lag
        # varies 0.27-0.6 s, and cutting into the pause after the name is
        # harmless while cutting into the question is not.
        self.wake_trim_offset = p('wake_trim_offset_sec', 0.2)
        self.wake_trim_max = p('wake_trim_max_sec', 1.5)
        # Per-language vocabulary hint ('' = none). Safe now that silent turns
        # never reach Whisper (speech_start_threshold); a transcript that only
        # repeats the hint is dropped (logic.echoes_prompt).
        self.prompts = {'en': p('initial_prompt_en', PROMPT_EN),
                        'pl': p('initial_prompt_pl', PROMPT_PL)}
        self.hotwords = p('hotwords', '')
        # Optional smaller model for the language-ID pass only (e.g. 'small'
        # with model_size 'medium': medium's accuracy for +1 s, not +1.8 s).
        self.language_id_model = os.path.expanduser(p('language_id_model', ''))
        # Cap tokens per decode: a hallucination loop ("Thank you." x28) took
        # 3.0 s to decode; 10 s of speech needs well under 100 tokens.
        self.max_new_tokens = p('max_new_tokens', 100)
        self.vad_threshold = p('vad_threshold', 0.3)
        self.vad_min_silence_ms = p('vad_min_silence_ms', 500)
        self.vad_speech_pad_ms = p('vad_speech_pad_ms', 400)
        self.normalize_audio = p('normalize_audio', True)
        self.echo_guard_sec = p('tts_echo_guard_sec', 2.5)
        # A turn after speaker playback still waits out the echo guard, but
        # its audio starts this long after the speaker stopped (from the
        # rolling buffer): people answer right away, and the G1 mic's echo
        # suppression leaves only a faint tail (<= 420 int16 RMS) after 0.1 s.
        self.echo_tail = p('echo_tail_sec', 0.3)
        # Audio board speaker state ({"play_state": 0|1}); '' = our TTS only.
        self.board_topic = p('board_play_state_topic', '/audio_msg')
        # Debug: save every turn's audio + decode details here ('' = off).
        self.debug_audio_dir = os.path.expanduser(p('debug_audio_dir', ''))

        compute_type = self.compute_type or (
            'float16' if self.device == 'cuda' else 'int8')
        self.get_logger().info('Loading faster-whisper {} on {} ({})'.format(
            self.model_size, self.device, compute_type))
        _preload_ctranslate2(self.get_logger())
        from faster_whisper import WhisperModel
        # No CPU fallback: a broken GPU path must fail loudly.
        self.model = WhisperModel(self.model_size, device=self.device,
                                  compute_type=compute_type)
        self.lid_model = self.model
        if self.language_id_model and self.language_id_model != self.model_size:
            self.get_logger().info('Language ID with faster-whisper {}'.format(
                self.language_id_model))
            self.lid_model = WhisperModel(
                self.language_id_model, device=self.device,
                compute_type='float16' if self.device == 'cuda' else 'int8')
        # The first language-ID call takes ~3.5 s (CUDA/cuDNN init); pay it
        # now, not on the first turn.
        try:
            warm = np.random.default_rng(0).normal(0.0, 0.01, self.sample_rate).astype(np.float32)
            self._detect(warm)
            self._transcribe(warm, 'en')
        except Exception as exc:  # noqa: BLE001 - the first turn will retry
            self.get_logger().warn('Whisper warm-up failed: {}'.format(exc))

        self._trigger = threading.Event()
        self._stop = threading.Event()
        self._abort = False
        self._busy = False
        self._echo = EchoGuard(self.echo_guard_sec)
        self._echo_lock = threading.Lock()
        self._awaiting_reply_since = None
        self._unitree_mode = False
        self._ignored_trigger_at = -1e9
        self._last_turn_end = time.monotonic()
        self._wakes = deque(maxlen=8)  # wall times of recent wake detections
        self._last_audio_end = 0.0     # wall time the previous turn's audio ended
        self._start_wake = None        # detection the current turn started from
        self._session_language = None  # language of the last turn this session
        # Rolling mic buffer (UDP mic): [(arrival wall time, float32 chunk)].
        self._mic_lock = threading.Lock()
        self._ring = deque()
        self._live = None  # queue.Queue fed while a turn records

        self.transcript_pub = self.create_publisher(
            String, '/g1/stt/transcript', 10)
        self.status_pub = self.create_publisher(String, '/g1/stt/status', 10)
        self.create_subscription(Bool, '/g1/voice/trigger', self._on_trigger, 10)
        self.create_subscription(
            Bool, '/g1/voice/stop_recording', self._on_stop, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_start', self._on_continuous_start, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_stop', self._on_continuous_stop, 10)
        self.create_subscription(String, '/g1/tts/status', self._on_tts, 10)
        self.create_subscription(
            String, '/g1/voice/assistant_mode', self._on_assistant_mode,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, 'llm_response', self._on_reply, 10)
        self.create_subscription(Float64, '/g1/wake/detected', self._on_wake, 10)
        self.create_subscription(
            String, '/g1/wake/session_state', self._on_session_state,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if self.board_topic:
            # Published by the board's own DDS app: best effort matches any QoS.
            self.create_subscription(
                String, self.board_topic, self._on_board, QoSProfile(
                    depth=20, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0, self._watchdog)
        if self.use_udp_mic:
            threading.Thread(target=self._mic_loop, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info('STT ready (mic: {}, continuous={})'.format(
            'UDP {}:{}'.format(self.udp_group, self.udp_port)
            if self.use_udp_mic else 'sounddevice', self.continuous_mode))

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    # ---------------------------------------------------------------- inputs
    def _on_trigger(self, msg):
        if not msg.data:
            return
        if self._unitree_mode and not self.continuous_mode:
            # The session start (continuous_start) may arrive just after the
            # trigger: _on_continuous_start replays a trigger ignored here.
            self._ignored_trigger_at = time.monotonic()
            self.get_logger().info('Trigger ignored: Unitree assistant is answering')
            return
        if self._busy:
            self.get_logger().info('Trigger ignored: turn already running')
            return
        self._stop.clear()
        self._abort = False
        self._trigger.set()

    def _on_stop(self, msg):
        if msg.data:
            self._stop.set()

    def _on_assistant_mode(self, msg):
        self._unitree_mode = msg.data == 'unitree'

    def _on_wake(self, msg):
        self._wakes.append(msg.data)

    def _on_continuous_start(self, msg):
        if msg.data:
            self.continuous_mode = True
            self._session_language = None
            if (time.monotonic() - self._ignored_trigger_at < 2.0
                    and not self._busy):
                self._ignored_trigger_at = -1e9
                self._stop.clear()
                self._abort = False
                self._trigger.set()

    def _on_session_state(self, msg):
        """Latched session state: catches a session opened before this node started."""
        if msg.data == 'open' and not self.continuous_mode:
            self.get_logger().info('Joining the open conversation session')
            self._on_continuous_start(Bool(data=True))
        elif msg.data == 'closed' and self.continuous_mode:
            self._on_continuous_stop(Bool(data=True))

    def _on_continuous_stop(self, msg):
        # START kill switch / session closed: stop and discard the turn.
        if msg.data:
            self.continuous_mode = False
            self._session_language = None
            self._trigger.clear()
            if self._busy:
                self._abort = True
                self._stop.set()

    def _on_tts(self, msg):
        with self._echo_lock:
            self._echo.set('tts', msg.data == 'speaking')

    def _on_board(self, msg):
        playing = board_play_state(msg.data)
        if playing is None:
            return  # an ASR result of Unitree's assistant: not ours to use
        with self._echo_lock:
            # Our TTS reports 'speaking' before its first PlayStream, so
            # playback without it is someone else (Unitree's assistant).
            # Expected in Unitree mode: chat_go is the one talking then.
            if playing and not self._echo.playing() and not self._unitree_mode:
                self.get_logger().info(
                    'Speaker playing without our TTS (Unitree assistant?): '
                    'not listening until it stops')
            self._echo.set('board', playing)

    def _on_reply(self, _msg):
        self._awaiting_reply_since = None

    def _echo_active(self):
        with self._echo_lock:
            return self._echo.active()

    def _awaiting_reply(self):
        since = self._awaiting_reply_since
        return since is not None and time.monotonic() - since < self.reply_timeout

    def _watchdog(self):
        """Restart listening in a session when no reply re-triggered us."""
        if (self.continuous_mode and not self._busy
                and not self._trigger.is_set() and not self._echo_active()
                and not self._awaiting_reply()
                and time.monotonic() - self._last_turn_end > self.watchdog_interval):
            self.get_logger().info('Watchdog: listening again')
            self._trigger.set()

    # ----------------------------------------------------------------- worker
    def _worker(self):
        while rclpy.ok():
            if not self._trigger.wait(timeout=0.5):
                continue
            # Never open the mic on the robot's own voice.
            deadline = time.monotonic() + 30.0
            while self._echo_active() and time.monotonic() < deadline:
                time.sleep(0.1)
            # Cancelled while waiting out the echo guard: the session closed
            # (continuous_stop clears the trigger) or F1 handed over to the
            # Unitree GPT. Seen 21 Nov 22:34: such a turn still recorded 7 s.
            if not self._trigger.is_set() or (
                    self._unitree_mode and not self.continuous_mode):
                self._trigger.clear()
                self.get_logger().info('Turn cancelled before recording')
                continue
            self._busy = True
            published = False
            try:
                published = self._run_turn()
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self.get_logger().error('Turn failed: {}'.format(exc))
            finally:
                if not published:
                    self._publish_status('empty_turn')
                self._publish_status('idle')
                self._last_turn_end = time.monotonic()
                self._trigger.clear()
                self._busy = False
            # In a session, an empty turn listens again right away; the
            # session closes after max_empty_turns (wake_word_node).
            if self.continuous_mode and not published and not self._abort:
                self._trigger.set()

    def _turn_start(self):
        """
        Wall time the turn's audio starts (never inside the previous turn).

        Right after a recent wake word that no earlier turn has recorded;
        else ``echo_tail_sec`` after the speaker last stopped, if that is
        still in the rolling buffer (an answer begun during the echo guard);
        else now.
        """
        now = time.time()
        fresh = [w for w in self._wakes
                 if 0.0 <= now - w < WAKE_MAX_AGE_SEC and w > self._last_audio_end]
        if fresh:
            self._start_wake = max(fresh)
            return max(self._start_wake - self.wake_audio_offset, self._last_audio_end)
        self._start_wake = None
        with self._echo_lock:
            quiet = self._echo.quiet_for()
        if quiet is not None and self.echo_tail <= quiet < self.mic_ring_sec - 0.2:
            return max(now - quiet + self.echo_tail, self._last_audio_end)
        return now

    def _trim_name(self, audio, endpoint, end_wall):
        """Cut a spoken "Aletta" off the start of the turn (see logic.name_cut)."""
        duration = audio.size / self.sample_rate
        onset = min(endpoint.onset, self.sample_rate) / self.sample_rate
        # A wake-started turn looks back wake_audio_offset_sec (1.5 s), so its
        # own detection is a cut candidate too; detections right after it are
        # the same word scored again.
        start = self._start_wake
        wakes = [w for w in self._wakes
                 if start is None or w == start or not 0.0 < w - start < WAKE_REFIRE_SEC]
        cut = name_cut(end_wall - duration, duration, wakes, onset,
                       self.wake_trim_offset, self.wake_trim_max)
        if cut is None:
            return audio, None
        # Only ever cut inside a pause: the detection can come up to ~1 s
        # after the name, i.e. already inside the question.
        snapped = snap_cut_to_pause(audio, self.sample_rate, cut, self.silence_threshold)
        if snapped is None:
            self.get_logger().info(
                'Name detected but no pause after it: keeping the whole turn')
            return audio, None
        cut = snapped
        self.get_logger().info('Cut the name from the turn ({:.2f}s of {:.2f}s)'.format(
            cut, duration))
        return audio[int(cut * self.sample_rate):], cut

    def _run_turn(self):
        self._publish_status('recording')
        audio, endpoint, end_wall = self._record(self._turn_start())
        self._last_audio_end = end_wall
        if self._abort or audio is None or audio.size < self.sample_rate // 4:
            return False  # no speech: an empty turn
        full_audio = audio
        audio, cut = self._trim_name(audio, endpoint, end_wall)
        if not has_speech(audio, self.sample_rate, self.silence_threshold,
                          onset_threshold=self.speech_start_threshold):
            self.get_logger().info('Only the name was said; still listening')
            return False

        self._publish_status('transcribing')
        timing = {'detect': 0.0, 'decode': 0.0}

        def timed(name, func):
            def call(*args):
                start = time.monotonic()
                try:
                    return func(*args)
                finally:
                    timing[name] += time.monotonic() - start
            return call

        segments, language = choose_transcription(
            timed('decode', self._transcribe), timed('detect', self._detect),
            audio, self.language, self._session_language,
            log=self.get_logger().info)
        self.get_logger().info(
            'STT timing: end-of-speech wait {:.2f}s, language ID {:.2f}s, '
            'decode {:.2f}s'.format(endpoint.trailing_silence(),
                                    timing['detect'], timing['decode']))
        kept, dropped = filter_segments(
            segments, self.max_no_speech_prob, self.min_avg_logprob,
            self.noisy_no_speech_prob, self.noisy_min_avg_logprob)
        if dropped:
            self.get_logger().info('Dropped unsure segments (no-speech score): {}'.format(
                '; '.join('[{:.2f}] {}'.format(s, t) for s, t in dropped)))
        text = ' '.join(kept).strip()
        if not text:
            verdict = 'empty'
            self.get_logger().info('Empty transcript')
        elif looks_like_noise(text):
            verdict = 'hallucination'
            self.get_logger().info('Ignored hallucination: "{}"'.format(text))
        elif is_only_name(text):
            verdict = 'name only'
            self.get_logger().info('Only the name was said; still listening')
        elif echoes_prompt(text, self.prompts.get(language)):
            verdict = 'hallucination'
            self.get_logger().info('Ignored echo of the vocabulary hint: "{}"'.format(text))
        elif self._abort:
            verdict = 'aborted'
        else:
            verdict = 'published'
        self._dump_turn(full_audio, segments, language, verdict, text, cut)
        if verdict != 'published':
            return False

        self.get_logger().info('Transcript [{}]: "{}"'.format(language, text))
        if self.continuous_mode:
            self._session_language = language
        self._awaiting_reply_since = time.monotonic()
        self.transcript_pub.publish(String(data=tag_language(text, language)))
        return True

    def _dump_turn(self, audio, segments, language, verdict, text, cut=None, keep=50):
        """Save the turn's raw audio and decode details (debug_audio_dir)."""
        if not self.debug_audio_dir:
            return
        try:
            os.makedirs(self.debug_audio_dir, exist_ok=True)
            base = os.path.join(self.debug_audio_dir, time.strftime('%Y%m%d_%H%M%S'))
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(base + '.wav', 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.sample_rate)
                f.writeframes(pcm.tobytes())
            frames = pcm[:len(pcm) // 1600 * 1600].reshape(-1, 1600).astype(np.float32)
            rms = np.sqrt((frames ** 2).mean(axis=1)) if len(frames) else np.zeros(1)
            lines = ['verdict: {}'.format(verdict), 'language: {}'.format(language),
                     'text: {}'.format(text),
                     'duration: {:.2f}s, rms per 100 ms: median {:.0f}, max {:.0f}'.format(
                         len(pcm) / self.sample_rate, np.median(rms), rms.max()),
                     'name cut at: {}'.format('{:.2f}s (saved audio is uncut)'.format(cut)
                                              if cut is not None else 'none')]
            lines += ['segment {:.2f}-{:.2f}s no_speech={:.2f} logprob={:.2f}: {}'.format(
                s.start, s.end, s.no_speech_prob, s.avg_logprob, s.text.strip())
                for s in segments]
            with open(base + '.txt', 'w') as f:
                f.write('\n'.join(lines) + '\n')
            for old in sorted(glob.glob(os.path.join(self.debug_audio_dir, '*.wav')))[:-keep]:
                for path in (old, old[:-4] + '.txt'):
                    if os.path.exists(path):
                        os.remove(path)
        except Exception as exc:  # noqa: BLE001 - debugging must never break STT
            self.get_logger().warn('Turn dump failed: {}'.format(exc))

    def _normalized(self, audio):
        if self.normalize_audio:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if 0.0 < peak < 0.95:
                audio = audio * (0.9 / peak)
        return audio

    def _detect(self, audio):
        """Whisper language probabilities {code: p} (one encoder pass)."""
        _, _, probs = self.lid_model.detect_language(self._normalized(audio))
        return dict(probs)

    def _transcribe(self, audio, language):
        segments, info = self.model.transcribe(
            self._normalized(audio), language=language, beam_size=self.beam_size,
            temperature=0.0, condition_on_previous_text=False,
            initial_prompt=self.prompts.get(language) or None,
            hotwords=self.hotwords or None, vad_filter=True,
            max_new_tokens=self.max_new_tokens or None,
            vad_parameters=dict(onset=self.vad_threshold,
                                min_silence_duration_ms=self.vad_min_silence_ms,
                                speech_pad_ms=self.vad_speech_pad_ms))
        return list(segments), info

    # -------------------------------------------------------------- recording
    # -------------------------------------------------------------- recording
    def _mic_loop(self):
        """Keep the last ``mic_ring_sec`` of the UDP mic; feed a live turn."""
        sock = None
        while rclpy.ok():
            try:
                if sock is None:
                    sock = open_multicast_socket(
                        self.udp_group, self.udp_port, self.udp_local_ip, 0.5)
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                now = time.time()
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                with self._mic_lock:
                    self._ring.append((now, chunk))
                    while self._ring and now - self._ring[0][0] > self.mic_ring_sec:
                        self._ring.popleft()
                    if self._live is not None:
                        self._live.put(chunk)
            except OSError as exc:
                self.get_logger().error(
                    'Mic socket error ({}); retrying'.format(exc),
                    throttle_duration_sec=30.0)
                if sock is not None:
                    sock.close()
                sock = None
                time.sleep(1.0)

    def _start_live(self, start_from):
        """Begin a live feed; return the buffered audio from ``start_from`` on."""
        live = queue.Queue()
        with self._mic_lock:
            pre = []
            for arrived, chunk in self._ring:
                skip = int(round((start_from - (arrived - chunk.size / self.sample_rate))
                                 * self.sample_rate))
                if skip < chunk.size:
                    pre.append(chunk[max(0, skip):])
            self._live = live
        return pre, live

    def _stop_live(self):
        with self._mic_lock:
            self._live = None

    def _record(self, start_from):
        """
        Record one utterance (see logic.Endpointer) or until F1 release.

        With the UDP mic the audio starts at wall time ``start_from`` (taken
        from the rolling buffer when that is in the past). The G1 mic floor
        peaks near RMS 0.009; speech is 0.03-0.1.
        Returns ``(float32 audio trimmed to 1 s before the speech onset or
        None, endpointer, wall time the audio ends)``.
        """
        pending = []
        if self.use_udp_mic:
            pending, live = self._start_live(start_from)

            def read():
                if pending:
                    return pending.pop(0)
                try:
                    return live.get(timeout=0.2)
                except queue.Empty:
                    return None

            close = self._stop_live
        else:
            import sounddevice as sd
            source = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                    dtype='float32', blocksize=1024)
            source.start()

            def read():
                return source.read(1024)[0].reshape(-1)

            close = source.close

        chunks = []
        endpoint = Endpointer(self.sample_rate, self.silence_threshold,
                              self.recording_duration, self.max_wait,
                              self.silence_duration,
                              onset_threshold=self.speech_start_threshold)
        deadline = time.monotonic() + self.max_wait + self.recording_duration + 2.0
        try:
            while not self._stop.is_set() and time.monotonic() < deadline:
                chunk = read()
                if chunk is None or not chunk.size:
                    continue
                chunks.append(chunk)
                if not endpoint.feed(float(np.sqrt(np.mean(chunk ** 2))), chunk.size):
                    break
        finally:
            close()
        end_wall = time.time()
        if not chunks:
            if not self._stop.is_set():  # a stopped/aborted turn may read nothing
                self.get_logger().warn('No microphone audio received')
            return None, endpoint, end_wall
        if endpoint.onset is None:
            # Nobody spoke: never let Whisper "transcribe" silence.
            self.get_logger().info(
                'No speech in {:.1f}s (loudest {:.3f}, start threshold {})'.format(
                    endpoint.total / self.sample_rate,
                    max(float(np.sqrt(np.mean(c ** 2))) for c in chunks),
                    self.speech_start_threshold))
            return None, endpoint, end_wall
        audio = np.concatenate(chunks)
        # 1 s pre-roll before the onset.
        return audio[max(0, endpoint.onset - self.sample_rate):], endpoint, end_wall

    def _publish_status(self, status):
        self.status_pub.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
