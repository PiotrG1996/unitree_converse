import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import numpy as np
import socket
import struct
import threading
import re
import os
import ctypes


# Directories searched for a hand-built (CUDA-enabled) libctranslate2. The pip
# ctranslate2 wheel on this Jetson ships without the native shared library, so
# the user compiled CTranslate2 from source and installed it here. The dynamic
# linker does not know about these paths, so we preload the .so by absolute
# path with ctypes before importing faster_whisper (see _preload_ctranslate2).
_CTRANSLATE2_LIB_DIRS = (
    os.environ.get('CTRANSLATE2_LIB_DIR', ''),
    '/home/unitree/ct2_install/lib',
    '/home/unitree/CTranslate2/build',
    os.path.expanduser('~/ct2_install/lib'),
    os.path.expanduser('~/CTranslate2/build'),
)


def _preload_ctranslate2(logger=None):
    """
    Preload a hand-built CUDA libctranslate2 by absolute path.

    faster_whisper -> ctranslate2 needs libctranslate2.so.4 at import time. On
    this Jetson that library lives outside the default loader search path, so a
    bare `import faster_whisper` raises
        ImportError: libctranslate2.so.4: cannot open shared object file
    under the minimal systemd environment (which sets LD_LIBRARY_PATH=/usr/local/lib
    only). Setting os.environ['LD_LIBRARY_PATH'] at runtime does NOT help because
    glibc caches the search path at process start. Loading the library explicitly
    with ctypes (RTLD_LOCAL) makes its symbols available to the subsequently
    imported ctranslate2 extension regardless of the launcher.

    Best-effort: if the library is already loadable (e.g. on the dev machine or
    a normally-installed wheel) this is a harmless no-op.
    """
    for lib_dir in _CTRANSLATE2_LIB_DIRS:
        if not lib_dir:
            continue
        for name in ('libctranslate2.so.4', 'libctranslate2.so'):
            path = os.path.join(lib_dir, name)
            if os.path.exists(path):
                try:
                    # RTLD_LOCAL, NOT RTLD_GLOBAL: ctranslate2's C extension links
                    # against libctranslate2.so.4 by soname, so the dynamic linker
                    # reuses this already-loaded instance without needing its symbols
                    # exported globally. Forcing RTLD_GLOBAL pollutes the process
                    # symbol namespace and corrupts the heap inside a full ROS node
                    # ("free(): invalid pointer" abort during CUDA model load).
                    ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
                    if logger:
                        logger.info(f'Preloaded CUDA runtime lib: {path}')
                except OSError as e:
                    if logger:
                        logger.warn(f'Found {path} but failed to preload: {e}')
                return
    # Nothing to preload — assume the system-provided library is resolvable.


class STTNode(Node):
    def _param(self, name):
        """Return a declared parameter's value (short form of get_parameter)."""
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('stt_node')

        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'auto')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('recording_duration', 8.0)
        self.declare_parameter('silence_threshold', 0.008)
        self.declare_parameter('silence_duration', 2.0)
        self.declare_parameter('use_udp_mic', True)
        self.declare_parameter('udp_multicast_group', '239.168.123.161')
        self.declare_parameter('udp_port', 5555)
        self.declare_parameter('udp_local_ip', '192.168.123.164')
        self.declare_parameter('continuous_mode', True)
        self.declare_parameter('watchdog_interval', 2.0)
        self.declare_parameter('boot_kick_delay', 6.0)
        self.declare_parameter('max_no_speech_prob', 0.5)
        self.declare_parameter('language_detection_min_probability', 0.55)

        self.model_size = self._param('model_size')
        self.language = self._param('language')
        self.device = self._param('device')
        self.sample_rate = self._param('sample_rate')
        self.recording_duration = self._param('recording_duration')
        self.silence_threshold = self._param('silence_threshold')
        self.silence_duration = self._param('silence_duration')
        self.use_udp_mic = self._param('use_udp_mic')
        self.udp_multicast_group = self._param('udp_multicast_group')
        self.udp_port = self._param('udp_port')
        self.udp_local_ip = self._param('udp_local_ip')
        self.continuous_mode = self._param('continuous_mode')
        self.watchdog_interval = self._param('watchdog_interval')
        self.boot_kick_delay = self._param('boot_kick_delay')
        self.max_no_speech_prob = self._param('max_no_speech_prob')
        self.language_detection_min_probability = self._param(
            'language_detection_min_probability')

        self.is_recording = False
        self.stop_requested = False
        self._transcript_published = False
        self.llm_busy = False

        # Pick a compute_type that is valid for the selected device. On CUDA,
        # plain 'int8' is not a supported quantisation; faster-whisper wants
        # 'float16' (or 'int8_float16') for GPU, while CPU keeps 'int8'.
        compute_type = 'float16' if self.device == 'cuda' else 'int8'

        self.get_logger().info(
            f'Loading faster-whisper model: {self.model_size} '
            f'(device={self.device}, compute_type={compute_type})')
        # Make the hand-built CUDA libctranslate2 resolvable before faster_whisper
        # imports it. Harmless no-op when the library is already on the loader path.
        _preload_ctranslate2(logger=self.get_logger())
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=compute_type)
        except Exception as e:
            # GPU-only target: if the CUDA load fails (no CUDA build of
            # ctranslate2, driver issue, etc.), surface it loudly rather than
            # silently falling back to CPU.
            self.get_logger().error(
                f'faster-whisper failed to load on device="{self.device}" '
                f'(compute_type={compute_type}): {e}')
            raise
        self.get_logger().info('faster-whisper model loaded')

        self.trigger_sub = self.create_subscription(
            Bool, '/g1/voice/trigger', self.trigger_callback, 10)
        self.stop_sub = self.create_subscription(
            Bool, '/g1/voice/stop_recording', self.stop_callback, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_start', self._continuous_start_cb, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_stop', self._continuous_stop_cb, 10)
        self.create_subscription(
            String, '/g1/llm/status', self._llm_status_callback, 10)

        self.transcript_pub = self.create_publisher(String, '/g1/stt/transcript', 10)
        self.status_pub = self.create_publisher(String, '/g1/stt/status', 10)

        mic_mode = 'UDP multicast' if self.use_udp_mic else 'sounddevice'
        self.get_logger().info(f'STT node ready — mic: {mic_mode}')
        self.get_logger().info(f'Continuous mode: {self.continuous_mode}')

        # Always-on (continuous) machinery is OPT-IN and only armed when
        # continuous_mode is True (e.g. enabled at runtime via the START button).
        # In the default push-to-talk mode (continuous_mode False) the robot only
        # listens after F1 is held, and never re-arms itself -- so it will NOT
        # answer on its own, which is the desired default behaviour.
        self._watchdog = None
        self._watchdog_lock = threading.Lock()
        if self.continuous_mode:
            # Keep the always-on conversation loop alive: after each recording
            # pass, if nobody advanced the loop (no reply came back through the
            # TTS node), re-trigger ourselves so the robot keeps listening. This
            # makes the "answer all the time" behaviour robust even when a turn is
            # muted, empty, or the LLM/robot-state hop produces no reply.
            self._schedule_self_retrigger(self.watchdog_interval)

            # Kick off the very first turn shortly after startup so the robot
            # starts answering immediately, without waiting for any button press.
            self._start_kick = self.create_timer(
                self.boot_kick_delay, self._boot_kick_cb)
        else:
            self._start_kick = None

    def _boot_kick_cb(self):
        # One-shot: disarm the timer, then start the first turn.
        self._start_kick.cancel()
        if not self.is_recording:
            self.get_logger().info('Always-on mode — starting first listening turn')
            self._retrigger()

    def _retrigger(self):
        """
        Start a new listening turn directly (self-advance).

        Unlike the TTS node, the STT node does not own the trigger *publisher* --
        it subscribes to /g1/voice/trigger. So to advance the loop ourselves we
        start the recording thread the same way trigger_callback does, rather
        than publishing on a publisher we don't have.
        """
        if self.is_recording:
            return
        self.get_logger().info('Always-on — starting next listening turn')
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def _schedule_self_retrigger(self, delay=None):
        """
        Arm a watchdog that re-triggers listening after `delay` seconds.

        Cancelled by the normal flow path, so in the healthy case the TTS node's
        re-trigger drives the loop and the watchdog never fires. This keeps the
        loop moving when a turn yields no reply (muted, empty transcript, LLM
        hiccup).
        """
        if delay is None:
            delay = self.watchdog_interval
        with self._watchdog_lock:
            if self._watchdog is not None:
                self._watchdog.cancel()
            self._watchdog = self.create_timer(delay, self._watchdog_fire)

    def _watchdog_fire(self):
        self._watchdog.cancel()
        if self.llm_busy:
            self.get_logger().info(
                'Continuous watchdog deferred while LLM is generating')
            self._schedule_self_retrigger(self.watchdog_interval)
        elif not self.is_recording:
            self.get_logger().info('Always-on watchdog — no reply received, listening again')
            self._retrigger()

    def trigger_callback(self, msg: Bool):
        """Handle an external /g1/voice/trigger message to start recording."""
        if msg.data and not self.is_recording:
            self.get_logger().info('Trigger received — starting recording')
            # An external trigger means the loop is being driven normally; cancel
            # the self-retrigger watchdog for this turn.
            with self._watchdog_lock:
                if self._watchdog is not None:
                    self._watchdog.cancel()
            threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def stop_callback(self, msg: Bool):
        if msg.data:
            self.stop_requested = True
            self.get_logger().info('Stop recording requested — F1 released')

    def _continuous_start_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = True
            self.get_logger().info('Continuous mode enabled')

    def _continuous_stop_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = False
            self.get_logger().info('Continuous mode disabled')

    def _llm_status_callback(self, msg: String):
        self.llm_busy = msg.data == 'thinking'

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        """
        Detect Whisper hallucination/noise transcripts so they never reach the LLM.

        Continuous mode records silence/ambient noise, and faster-whisper reliably
        turns that into short looping garbage (e.g. "I'm sorry. I'm sorry.",
        "Thank you very much." repeated, single repeated words). Sending those to
        the LLM makes it roleplay nonsense like "check the internet connection".
        We drop:
          * text with no letters/digits at all (punctuation/artifacts only), and
          * heavy word repetition: the same word/phrase repeated 3+ times, or a
            very short transcript whose words are almost all identical.
        """
        t = text.strip()
        if not t:
            return True
        # Must contain at least one letter or digit, else it's pure noise/punctuation.
        if not re.search(r'[A-Za-z0-9]', t):
            return True

        words = re.findall(r"[A-Za-z0-9']+", t.lower())
        if not words:
            return True

        # Immediate phrase repetition: a run of >=2 words that is immediately
        # repeated (e.g. "thank you very much thank you very much", "it's a
        # mouthful it's a mouthful"). This is the classic Whisper hallucination
        # loop and the single-word ratio check above misses it.
        for p in range(2, len(words) // 2 + 1):
            for i in range(0, len(words) - 2 * p + 1):
                if words[i:i + p] == words[i + p:i + 2 * p]:
                    return True

        # Repetition ratio: (occurrences of most common word) / total words.
        # Loops like "sorry sorry sorry sorry" score high. Require >=4 words so a
        # legit short reply like "hello there" or "yes please" is never dropped.
        if len(words) >= 4:
            from collections import Counter
            top, count = Counter(words).most_common(1)[0]
            if count / len(words) >= 0.5:
                return True

        # Any single word repeated 3+ times in a row is a classic Whisper loop.
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True

        return False

    def _record_and_transcribe(self):
        self.is_recording = True
        self._transcript_published = False
        self._publish_status('recording')

        if self.use_udp_mic:
            audio_data = self._record_via_udp()
        else:
            audio_data = self._record_via_sounddevice()

        if audio_data is None or len(audio_data) == 0:
            self.get_logger().warn('No audio recorded')
            self._publish_status('idle')
            self.is_recording = False
            if self.continuous_mode:
                self._schedule_self_retrigger()
            return

        self._publish_status('transcribing')

        try:
            segments, info = self._transcribe_with_language_fallback(audio_data)

            # Confidence gate: drop segments Whisper flags as likely non-speech
            # (see speech_gate.py). Quiet-room noise hallucinations measure
            # no_speech_prob 0.53-0.59 while genuine speech is well below 0.5,
            # so this removes invented transcripts (e.g. "please check the
            # network") before they reach the LLM.
            try:
                from g1_voice.speech_gate import filter_segments
                kept, dropped = filter_segments(
                    segments, self.max_no_speech_prob)
            except Exception:
                self.get_logger().warn(
                    'speech_gate unavailable; keeping raw transcript')
                kept = [s.text.strip() for s in segments if s.text.strip()]
                dropped = []

            if dropped:
                self.get_logger().info(
                    f'Filtered {len(dropped)} no-speech segment(s): '
                    + '; '.join(f'[{nsp:.2f}] "{t}"' for nsp, t in dropped))

            transcript = ' '.join(kept).strip()
            detected_language = (getattr(info, 'language', '') or '').lower()
            language_probability = getattr(info, 'language_probability', 0.0)
            self.get_logger().info(
                f'Whisper detected language: {detected_language} '
                f'(probability={language_probability:.2f})')
            if detected_language not in ('pl', 'en'):
                self.get_logger().warn(
                    f'Ignoring speech detected as unsupported language: '
                    f'{detected_language or "unknown"}')
                transcript = ''

            if transcript and self._looks_like_noise(transcript):
                self.get_logger().info(
                    f'Ignored noise/hallucination transcript: "{transcript}"')
            elif transcript:
                self.get_logger().info(f'Transcript: "{transcript}"')
                msg = String()
                msg.data = f'[USER_LANGUAGE:{detected_language}]\n{transcript}'
                self.transcript_pub.publish(msg)
                self._transcript_published = True
            else:
                self.get_logger().warn('Empty transcript — nothing detected')

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            self._publish_status('error')
            self.is_recording = False
            if self.continuous_mode:
                self._schedule_self_retrigger()
            return

        self._publish_status('idle')
        self.is_recording = False
        # TTS owns the healthy continuous-mode handoff after a valid transcript.
        # Do not let the watchdog start a second turn while the LLM is still
        # generating the response for this turn.
        if self.continuous_mode and not self._transcript_published:
            self._schedule_self_retrigger()

    def _transcribe(self, audio_data, language=None):
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return list(segments), info

    @staticmethod
    def _transcription_score(segments):
        text = ' '.join(segment.text.strip() for segment in segments).strip()
        if not text:
            return float('-inf')
        # avg_logprob is a model confidence signal; prefer longer supported
        # speech only when confidence is comparable.
        confidence = np.mean([
            getattr(segment, 'avg_logprob', -10.0) for segment in segments])
        return float(confidence) + min(len(text), 80) * 0.001

    def _transcribe_with_language_fallback(self, audio_data):
        configured = self.language.lower()
        if configured not in ('', 'auto'):
            segments, info = self._transcribe(audio_data, configured)
            return segments, info

        auto_segments, auto_info = self._transcribe(audio_data)
        detected = (getattr(auto_info, 'language', '') or '').lower()
        probability = getattr(auto_info, 'language_probability', 0.0)

        if detected in ('pl', 'en') and probability >= self.language_detection_min_probability:
            return auto_segments, auto_info

        candidates = {}
        for language in ('pl', 'en'):
            candidates[language] = self._transcribe(audio_data, language)

        best_language = max(
            candidates,
            key=lambda language: self._transcription_score(candidates[language][0]))
        best_segments, best_info = candidates[best_language]
        self.get_logger().info(
            f'Language fallback: auto={detected or "unknown"} '
            f'(probability={probability:.2f}), selected={best_language}')
        return best_segments, best_info

    def _record_via_udp(self):
        self.stop_requested = False
        BYTES_PER_SAMPLE = 2
        target_bytes = int(self.sample_rate * BYTES_PER_SAMPLE * self.recording_duration)
        silence_limit = int(self.silence_duration * self.sample_rate * BYTES_PER_SAMPLE)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(('', self.udp_port))

        mreq = struct.pack(
            '4s4s',
            socket.inet_aton(self.udp_multicast_group),
            socket.inet_aton(self.udp_local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(0.1)

        frames = []
        total_bytes = 0
        silence_bytes = 0
        speech_detected = False

        self.get_logger().info(f'Recording via UDP {self.udp_multicast_group}:{self.udp_port}...')

        try:
            while total_bytes < target_bytes and not self.stop_requested:
                try:
                    data, _ = sock.recvfrom(65535)
                    frames.append(data)
                    total_bytes += len(data)

                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(samples**2))

                    if rms >= self.silence_threshold:
                        speech_detected = True
                        silence_bytes = 0
                    else:
                        if speech_detected:
                            silence_bytes += len(data)

                    # Early silence-stop ONLY in continuous mode.
                    # In push-to-talk (F1) the user controls the endpoint by
                    # releasing the button, so we must keep recording until the
                    # release (or the recording_duration cap) — otherwise a brief
                    # pause between words truncates the clip and yields an empty
                    # transcript.
                    if (self.continuous_mode and speech_detected
                            and silence_bytes >= silence_limit):
                        self.get_logger().info('Silence detected — stopping early')
                        break

                except socket.timeout:
                    continue

        except Exception as e:
            self.get_logger().error(f'UDP recording error: {e}')
            return None
        finally:
            sock.close()

        if self.stop_requested:
            self.get_logger().info('Recording stopped by F1 release')

        if not frames:
            return None

        raw = b''.join(frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _record_via_sounddevice(self):
        import sounddevice as sd

        self.stop_requested = False
        audio_frames = []
        silence_frames = 0
        silence_limit = int(self.silence_duration * self.sample_rate)
        chunk_size = 1024
        speech_detected = False

        def audio_callback(indata, frames, time, status):
            audio_frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=chunk_size,
                callback=audio_callback
            ):
                self.get_logger().info('Recording via sounddevice...')
                total_samples = 0
                max_samples = int(self.recording_duration * self.sample_rate)

                while total_samples < max_samples and not self.stop_requested:
                    sd.sleep(100)
                    total_samples = sum(len(f) for f in audio_frames)

                    if audio_frames:
                        tail = audio_frames[-10:]
                        recent = np.concatenate(tail)
                        rms = np.sqrt(np.mean(recent**2))
                        if rms >= self.silence_threshold:
                            speech_detected = True
                            silence_frames = 0
                        else:
                            if speech_detected:
                                silence_frames += chunk_size

                        if speech_detected and silence_frames >= silence_limit:
                            self.get_logger().info('Silence detected — stopping early')
                            break

        except Exception as e:
            self.get_logger().error(f'Recording error: {e}')
            return None

        if not audio_frames:
            return None

        return np.concatenate(audio_frames).flatten()

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


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
