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
        # WAKE-WORD GATED BY DEFAULT: the robot waits for "Aletta"
        # (wake_word_node) or an explicit pad press (F1/START). It must NOT
        # auto-listen on boot and must NOT answer background speech.
        # Continuous always-on listening is opt-in at runtime via START.
        self.declare_parameter('continuous_mode', False)
        # Idle watchdog interval (continuous mode only). In gated mode no
        # watchdog is armed, so the wait costs ~zero CPU.
        self.declare_parameter('watchdog_interval', 15.0)
        # Boot auto-listen delay. 0/disabled by default so the robot boots
        # into "waiting for Aletta" instead of recording immediately.
        # Only honoured when continuous_mode is True at startup.
        self.declare_parameter('boot_kick_delay', 0.0)
        self.declare_parameter('max_no_speech_prob', 0.5)
        self.declare_parameter('language_detection_min_probability', 0.55)
        # Duel override margin: a forced-language candidate may replace the
        # automatic pass only when its score beats the auto pass by this much
        # (logprob units). Without a margin the blind max() flips noise turns
        # to Polish: on 2026-10-16, 5 of 7 duels returned pl even when
        # auto said en (once at en probability=0.75). Real forced-language
        # separation on genuine speech is far larger than 0.25.
        self.declare_parameter('language_duel_override_margin', 0.25)
        # Publish floor: drop transcripts whose kept segments average below
        # this avg_logprob — only extreme Whisper garbage trips it. The value
        # is logged on every publish so it can be tightened with data.
        self.declare_parameter('min_publish_avg_logprob', -2.5)

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
        self.language_duel_override_margin = self._param(
            'language_duel_override_margin')
        self.min_publish_avg_logprob = self._param('min_publish_avg_logprob')

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

        # Always-on (continuous) machinery is strictly OPT-IN: it is armed ONLY
        # when continuous_mode is True (e.g. enabled at runtime via the START
        # button). In the default wake-word-gated mode (False) the robot only
        # listens after "Aletta" or F1 is held, and NEVER re-arms itself — so
        # it will NOT answer on its own. Boot auto-listen is disabled unless
        # boot_kick_delay > 0 AND continuous_mode is True.
        self._watchdog = None
        self._watchdog_lock = threading.Lock()
        if self.continuous_mode and self.boot_kick_delay > 0:
            # Keep the always-on conversation loop alive: after each recording
            # pass, if nobody advanced the loop (no reply came back through the
            # TTS node), re-trigger ourselves so the robot keeps listening. This
            # makes the "answer all the time" behaviour robust even when a turn is
            # muted, empty, or the LLM/robot-state hop produces no reply.
            self._schedule_self_retrigger(self.watchdog_interval)

            # Kick off the very first turn shortly after startup so the robot
            # starts answering immediately, without waiting for any button press
            # (continuous mode only — never in wake-gated mode).
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
        # loop and the single-word ratio check above misses it. Only flag when
        # the repetition dominates the transcript (>=70% of words): real speech
        # can contain a repeated phrase for emphasis without being noise.
        for p in range(2, len(words) // 2 + 1):
            for i in range(0, len(words) - 2 * p + 1):
                if words[i:i + p] == words[i + p:i + 2 * p]:
                    repeated_words = 2 * p
                    j = i + 2 * p
                    while j + p <= len(words) and words[j:j + p] == words[i:i + p]:
                        repeated_words += p
                        j += p
                    if repeated_words / len(words) >= 0.70:
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

        # Run-on repetition WITHIN a single word (no spaces): Whisper often
        # loops one syllable without a break, e.g. "nienienie...", "lalala",
        # "yeahyeahyeah", "aaaaaah". Each whole run tokenizes as ONE word, so
        # the word-level checks above never see the repetition. Detect a word
        # that STARTS with a short unit repeated 3+ times (the final repeat may
        # be truncated, so we compare the leading window, not an exact tiling).
        # The length floor keeps real short words ("murmur", "banana") safe.
        for word in words:
            if len(word) < 6:
                continue
            for unit_len in range(1, len(word) // 3 + 1):
                unit = word[:unit_len]
                if word[:unit_len * 3] == unit * 3:
                    return True

        return False

    def _record_and_transcribe(self):
        self.is_recording = True
        self._transcript_published = False
        self._turn_had_transcript = False
        self._publish_status('recording')

        if self.use_udp_mic:
            audio_data = self._record_via_udp()
        else:
            audio_data = self._record_via_sounddevice()

        if audio_data is None or len(audio_data) == 0:
            self.get_logger().warn('No audio recorded')
            # Contract with wake_word_node.stt_status_callback: a turn with
            # no audio is an empty turn (counts toward max_empty_turns).
            self._publish_status('empty_turn')
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
                # Whisper's auto language detection is unreliable on noisy
                # mic audio (e.g. 'es' at probability=0.54 for real English
                # speech). Don't throw away a valid transcript: default to
                # English for TTS/LTS routing. The text itself is what the
                # LLM sees.
                self.get_logger().warn(
                    f'Unsupported language {detected_language or "unknown"}, '
                    f'defaulting to English')
                detected_language = 'en'

            if transcript and self._looks_like_noise(transcript):
                self.get_logger().info(
                    f'Ignored noise/hallucination transcript: "{transcript}"')
            elif transcript:
                mean_logprob = float(np.mean([
                    getattr(s, 'avg_logprob', -10.0) for s in kept]))
                if mean_logprob < self.min_publish_avg_logprob:
                    # Extreme-garbage floor; the value is logged on every
                    # publish so the threshold can be tightened from data.
                    self.get_logger().warn(
                        f'Dropped low-confidence transcript '
                        f'(avg_logprob={mean_logprob:.2f} < '
                        f'{self.min_publish_avg_logprob:.2f}): '
                        f'"{transcript}"')
                    transcript = ''
                else:
                    self.get_logger().info(
                        f'Transcript: "{transcript}" '
                        f'(avg_logprob={mean_logprob:.2f})')
                    msg = String()
                    msg.data = (
                        f'[USER_LANGUAGE:{detected_language}]\n{transcript}')
                    self.transcript_pub.publish(msg)
                    self._transcript_published = True
                    # Continuous-mode handoff: a valid user transcript
                    # (not a noise/empty turn) lets the continuous-loop
                    # re-arm. The LLM/TTS path closes the turn via
                    # _TURN_END, which re-triggers listening.
                    self._turn_had_transcript = True
            else:
                self.get_logger().warn('Empty transcript — nothing detected')

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            self._publish_status('error')
            # No transcript came out of this turn — report it as empty so
            # the session can still close crisply via max_empty_turns
            # instead of lingering until the duration cap.
            self._publish_status('empty_turn')
            self.is_recording = False
            if self.continuous_mode:
                self._schedule_self_retrigger()
            return

        if not self._transcript_published:
            # Contract with wake_word_node.stt_status_callback: every turn
            # with no published transcript (silence, noise, unsupported
            # language, dropped low-confidence decode) reports 'empty_turn'
            # so the max_empty_turns close can function. Without this the
            # wake node's empty-turn counter never increments and sessions
            # only die on the 300s duration cap.
            self._publish_status('empty_turn')
        self._publish_status('idle')
        self.is_recording = False
        # Re-arm continuous listening after the turn ends.
        #
        # TTS owns the healthy handoff after a valid transcript (it publishes
        # _TURN_END -> /g1/voice/trigger once the reply finishes speaking).
        # The watchdog below is the safety net: after an empty/noise turn
        # there is no LLM reply coming, so nothing else will re-trigger
        # listening. Without this the node goes deaf after one empty turn
        # and the wake loop keeps firing wake-detects with no recording.
        # After a valid transcript we still arm the watchdog so a muted/
        # failed TTS turn cannot stall the loop either; the external
        # trigger_callback cancels it when TTS drives the next turn normally.
        if self.continuous_mode:
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

        # Skip the scored pl/en duel when the automatic pass contains no
        # speech-quality segment: the input is ambient noise/silence, and
        # forcing each language would only re-decode that same noise. Measured
        # on the G1, 130 of 153 such idle turns resolved to an empty transcript
        # anyway (the no-speech gate drops them). Bailing out here avoids two
        # extra GPU decodes per idle turn and stops the misleading
        # "selected=pl" log on pure noise. If the gate is unavailable we assume
        # speech so the duel still runs (never lose a real turn to a missing
        # import).
        if not self._has_speech(auto_segments):
            self.get_logger().info(
                'No speech in auto pass '
                f'(auto={detected or "unknown"} probability={probability:.2f}); '
                'skipping language duel')
            return auto_segments, auto_info

        candidates = {}
        for language in ('pl', 'en'):
            candidates[language] = self._transcribe(audio_data, language)

        # Score the auto pass too: a forced candidate may replace it only
        # with a clear margin (language_duel_override_margin). The old blind
        # max() picked Polish for noise turns — forced-pl decodes of non-
        # speech loop into long, confident (high-avg_logprob) text that beat
        # short English garbage on BOTH terms of _transcription_score
        # (observed 2026-10-16: 5/7 duels returned pl when auto said en).
        #
        # English-locked duel: when Whisper's auto pass detected English,
        # Polish must never override it. The user spoke English ("Aletta"
        # is English); on noisy mic audio Whisper often confuses English
        # for Polish. Only an auto detection of Polish or unknown can
        # produce a Polish winner.
        auto_score = self._transcription_score(auto_segments)

        allow_pl_win = detected != 'en'

        scored = {}
        for language, (segments, _info) in candidates.items():
            joined = ' '.join(s.text.strip() for s in segments).strip()
            if self._looks_like_noise(joined):
                # Hallucination loops never win — they would win on length
                # and predictability, exactly the noise-vs-en failure mode.
                scored[language] = None
                continue
            if language == 'pl' and not allow_pl_win:
                # Polish cannot win when auto said English.
                scored[language] = None
                continue
            scored[language] = self._transcription_score(segments)

        best_language = None
        best_score = float('-inf')
        for language, score in scored.items():
            if score is not None and score > best_score:
                best_language, best_score = language, score

        scores_desc = ' | '.join(
            f'{language}=' + ('disqualified' if score is None
                              else f'{score:.2f}')
            for language, score in scored.items())
        margin = self.language_duel_override_margin

        if best_language is None or best_score < auto_score + margin:
            self.get_logger().info(
                f'Duel: auto={detected or "unknown"} '
                f'(probability={probability:.2f}) score={auto_score:.2f} | '
                f'{scores_desc} -> keeping auto (margin {margin:.2f})')
            return auto_segments, auto_info

        best_segments, best_info = candidates[best_language]
        self.get_logger().info(
            f'Duel: auto={detected or "unknown"} '
            f'(probability={probability:.2f}) score={auto_score:.2f} | '
            f'{scores_desc} -> selected={best_language}')
        return best_segments, best_info

    def _has_speech(self, segments) -> bool:
        """Return True when at least one segment is genuine speech, not noise.

        Reuses ``speech_gate.keep_segment`` so this decision uses exactly the
        same confidence rule the transcript path applies downstream. A segment
        passes the no-speech gate when its no_speech_prob is at/below threshold.
        """
        try:
            from g1_voice.speech_gate import keep_segment
        except Exception:
            # Gate unavailable: err on the side of speech so the duel runs.
            return True
        for segment in segments:
            if not (segment.text or '').strip():
                continue
            nsp = getattr(segment, 'no_speech_prob', None)
            if nsp is None:
                # Missing score: keep_segment keeps it (err on the side of
                # speech so the duel runs and a real turn isn't lost).
                return True
            if keep_segment(nsp, self.max_no_speech_prob):
                return True
        return False

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
        # Mic diagnostic (2026-10-17, live G1): room floor sits at RMS
        # ~0.0069 with speech peaks only ~0.013 (6/76 packets above the
        # 0.008 threshold in a 12s spoken capture). A single above-threshold
        # packet therefore means noise, not speech: require SUSTAINED
        # energy (min_speech_bytes above threshold) before the early-stop
        # can arm, and smooth RMS over a short sliding window so one hot
        # packet cannot latch speech_detected on its own.
        min_speech_bytes = int(0.2 * self.sample_rate * BYTES_PER_SAMPLE)
        speech_bytes = 0
        rms_window = []
        rms_window_max = 5

        self.get_logger().info(f'Recording via UDP {self.udp_multicast_group}:{self.udp_port}...')

        try:
            while total_bytes < target_bytes and not self.stop_requested:
                try:
                    data, _ = sock.recvfrom(65535)
                    frames.append(data)
                    total_bytes += len(data)

                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(samples**2))
                    rms_window.append(float(rms))
                    if len(rms_window) > rms_window_max:
                        rms_window.pop(0)
                    # Median (not mean): a single loud packet cannot drag the
                    # window over threshold on its own; sustained energy is
                    # required. Measured floor 0.0069 vs blips 0.013: a mean
                    # window crosses at 0.0081 on one blip, a median window
                    # stays at floor until 3+ of 5 packets are hot.
                    smooth_rms = float(np.median(rms_window))

                    if smooth_rms >= self.silence_threshold:
                        speech_bytes += len(data)
                        if speech_bytes >= min_speech_bytes:
                            speech_detected = True
                        silence_bytes = 0
                    else:
                        speech_bytes = 0
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
        # Mirror the UDP path: require sustained (not single-chunk) energy
        # before the early-stop can arm (see _record_via_udp).
        min_speech_frames = int(0.2 * self.sample_rate)
        speech_frames = 0

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
                            speech_frames += chunk_size
                            if speech_frames >= min_speech_frames:
                                speech_detected = True
                            silence_frames = 0
                        else:
                            speech_frames = 0
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
