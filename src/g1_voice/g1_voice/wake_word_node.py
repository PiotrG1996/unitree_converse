import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import numpy as np
import os
import socket
import struct
import threading
import time


class WakeWordNode(Node):
    """
    Wake on the name "Aletta" (openWakeWord) or pad buttons only.

    Voice trigger: the custom openWakeWord model hears "Aletta" on the UDP
    mic stream and fires ``/g1/voice/trigger``. Pad buttons drive the same
    pipeline through ``button_trigger_node`` (F1 push-to-talk, START
    continuous; SELECT speaks a fixed phrase directly via TTS). There is no
    Whisper phrase matching, no "hey robot", and no keyboard trigger — by
    design. If the Aletta model cannot load, wake stays disabled (loud
    error) but the pad keeps working.
    """

    def _param(self, name):
        """Return a declared parameter's value (short form of get_parameter)."""
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('sample_rate', 16000)
        # openWakeWord 0.6.0 exports ONNX only, and its base feature models
        # must match the framework (see wake_word_openwakeword).
        self.declare_parameter(
            'aletta_model_path',
            '~/.local/share/openwakeword/aletta_v0.1.onnx')
        self.declare_parameter('aletta_framework', 'onnx')
        self.declare_parameter('aletta_threshold', 0.85)
        # Confirmation: fire only after N consecutive 80 ms frames at/above
        # the threshold. Real "Aletta" utterances hold high scores for several
        # frames; single-spike false accepts (noise transients, playback
        # bleed) rarely do. Costs ~80 ms per extra frame.
        self.declare_parameter('min_consecutive_frames', 2)
        # Echo guard: while TTS is speaking — and for this long afterwards —
        # the mic hears the robot's own Piper voice, which the (synthetic-
        # trained) model scores highly. Drain audio without scoring so the
        # model cannot self-trigger on the robot's playback.
        self.declare_parameter('echo_guard_sec', 1.5)
        # Session: "Aletta" opens a continuous conversation window whose turns
        # are chained via the TTS _TURN_END auto-trigger (see tts_node).
        # There is deliberately NO silence timeout: a dropped/noisy turn must
        # not end the conversation (2026-10-16: the 20s close fired right
        # after a low-confidence transcript was dropped, killing sessions the
        # user was still in). The window closes only on explicit evidence:
        #   * START pad kill-switch (/g1/voice/continuous_stop),
        #   * session_max_duration_sec of wall-clock time (phantom backstop),
        #   * max_empty_turns consecutive turns with no published transcript
        #     (STT reports these on /g1/stt/status as 'empty_turn').
        # Only a fresh "Aletta" re-opens it.
        self.declare_parameter('session_max_duration_sec', 300.0)
        self.declare_parameter('max_empty_turns', 4)
        # While the session is CLOSED, the wake listener stays throttled:
        # short scored bursts separated by idle sleeps. This keeps the wait
        # state near zero CPU (no ONNX embedding work between bursts).
        self.declare_parameter('idle_listen_burst_sec', 1.5)
        # Idle (session closed) gap between scored bursts. Was 1.5 s - a ~50%
        # duty cycle that swallowed real "Aletta" attempts by pure timing luck
        # (missed-wake incident, 2026-10-16). 0.3 s keeps ~85% coverage while
        # still throttling ONNX embedding work between bursts.
        self.declare_parameter('idle_sleep_sec', 0.3)
        # RMS energy gate: frames below this int16 RMS are skipped before
        # ONNX scoring. Raised from 100 to 300 so quiet room noise/HVAC
        # never reaches the synthetic model, which scores everything ~0.99.
        self.declare_parameter('silence_rms', 300.0)
        # False-accept backoff: when a wake-opened session times out with no
        # speech, multiply the cooldown by this factor. Caps at
        # max_false_accept_cooldown. Resets on any real session activity.
        self.declare_parameter('false_accept_backoff_multiplier', 2.0)
        self.declare_parameter('max_false_accept_cooldown', 60.0)
        # Base openWakeWord feature models (melspectrogram + embedding). The
        # package's own resources dir is often empty on a fresh install, so we
        # pass these explicitly (see wake_word_openwakeword and setup.sh).
        # Empty = resolved at runtime to ~/.local/share/openwakeword/.
        self.declare_parameter('oww_melspec_model_path', '')
        self.declare_parameter('oww_embedding_model_path', '')
        self.declare_parameter('use_udp_mic', True)
        # Log the mic path that will actually be used at startup, so a
        # session-open with zero Aletta detections in the logs is
        # diagnosable: either the openWakeWord model failed to load, or the
        # UDP multicast stream is not reaching the wake listener.
        self._wake_mic_path_logged = False
        self.declare_parameter('udp_multicast_group', '239.168.123.161')
        self.declare_parameter('udp_port', 5555)
        self.declare_parameter('udp_local_ip', '192.168.123.164')
        # Base cooldown between trigger firings. False accepts escalate it
        # exponentially (up to max_false_accept_cooldown); a real LLM reply
        # resets it here. Was a hardcoded 8.0 s — now tunable.
        self.declare_parameter('base_cooldown_sec', 8.0)

        self.sample_rate = self._param('sample_rate')
        self.aletta_model_path = os.path.expanduser(
            self._param('aletta_model_path'))
        self.aletta_framework = self._param('aletta_framework')
        self.aletta_threshold = self._param('aletta_threshold')
        self.min_consecutive_frames = int(
            self._param('min_consecutive_frames'))
        self.echo_guard_sec = float(self._param('echo_guard_sec'))
        self.oww_melspec_model_path = os.path.expanduser(
            self._param('oww_melspec_model_path'))
        self.oww_embedding_model_path = os.path.expanduser(
            self._param('oww_embedding_model_path'))
        self.use_udp_mic = self._param('use_udp_mic')
        self.udp_multicast_group = self._param('udp_multicast_group')
        self.udp_port = self._param('udp_port')
        self.udp_local_ip = self._param('udp_local_ip')

        self.session_max_duration_sec = float(
            self._param('session_max_duration_sec'))
        self.max_empty_turns = int(self._param('max_empty_turns'))
        self.idle_listen_burst_sec = float(
            self._param('idle_listen_burst_sec'))
        self.idle_sleep_sec = float(self._param('idle_sleep_sec'))
        self.silence_rms = float(self._param('silence_rms'))
        self.false_accept_backoff_multiplier = float(
            self._param('false_accept_backoff_multiplier'))
        self.max_false_accept_cooldown = float(
            self._param('max_false_accept_cooldown'))
        # Per-burst wake telemetry (missed-wake observability, 2026-10-16):
        # limit idle-path chatter to one debug line per interval when bursts
        # score nothing above threshold but saw audio energy - tells us
        # empirically whether real attempts are near-misses (low peak score)
        # or never reach the detector (silence_rms gate / no frames scored).
        self.declare_parameter('telemetry_log_interval_sec', 30.0)
        self.telemetry_log_interval_sec = float(
            self._param('telemetry_log_interval_sec'))
        self._last_telemetry_log = 0.0
        self._blocked_by_cooldown_log = 0.0

        self.last_trigger_time = 0.0
        self.base_cooldown = float(self._param('base_cooldown_sec'))
        self.cooldown = self.base_cooldown
        self.false_accept_count = 0
        # NOTE: the historic `listening` pause flag was removed — the mic
        # stays open during sessions by design (see _wake_word_loop).
        # True when the current open session was opened by a wake detection
        # (not by a pad button). Attribution survives session activity touches
        # and is only cleared on close — the false-accept backoff needs it at
        # close time, when deciding whether the wake session was productive.
        self.session_opened_by_wake = False
        # True when the LLM started generating a reply during this session —
        # proof of a real conversation (a hallucination turn with no LLM reply
        # leaves it False, so consecutive no-reply wake sessions escalate the
        # cooldown instead of being treated as activity).
        self.session_had_llm_reply = False
        # Conversation session: False = idle wait (throttled listener, no
        # answers). "Aletta" opens it (START can only end it); 15 s of turn
        # silence closes it. Turn activity = trigger fired / STT/LLM/TTS busy.
        self.session_open = False
        self._session_state = 'idle'
        self._publish_session_state()
        self.last_session_activity = 0.0
        self.session_opened_at = 0.0
        self.empty_turns = 0
        # openWakeWord detector for the "Aletta" wake word. The only voice
        # trigger: if it fails to load, wake stays disabled (pad still works).
        self.oww_detector = None
        self._wake_socket = None

        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.status_pub = self.create_publisher(String, '/g1/wake/status', 10)
        # The wake-word node is the SOLE owner of continuous mode: only a
        # valid "Aletta" detection opens the chained-turn window; it closes
        # on the 15 s silence timeout (START can only force it shut early via
        # /g1/voice/continuous_stop from button_trigger_node).
        self.continuous_start_pub = self.create_publisher(
            Bool, '/g1/voice/continuous_start', 10)
        self.continuous_stop_pub = self.create_publisher(
            Bool, '/g1/voice/continuous_stop', 10)
        # Session-state observability: idle / opening / refreshing / closed.
        # Operators can query this to tell whether "Aletta" is actually
        # opening a new session or just refreshing an open one.
        self.session_state_pub = self.create_publisher(
            String, '/g1/wake/session_state', 10)
        self._session_state = 'idle'

        self.stt_busy = False
        self.llm_busy = False
        self.tts_busy = False
        # Last moment TTS was observed busy — drives the echo guard window
        # after playback ends (room/enclosure keeps ringing for a moment).
        self._last_tts_busy = 0.0
        self.stt_status_sub = self.create_subscription(
            String, '/g1/stt/status', self.stt_status_callback, 10)
        self.llm_status_sub = self.create_subscription(
            String, '/g1/llm/status', self.llm_status_callback, 10)
        self.tts_status_sub = self.create_subscription(
            String, '/g1/tts/status', self.tts_status_callback, 10)
        # START pad kill-switch: button_trigger_node publishes
        # /g1/voice/continuous_stop, which STT/TTS already honour. The wake
        # node must hear it too so its session bookkeeping (session_open,
        # continuous re-arming, empty-turn counting) closes immediately
        # instead of lingering until the duration cap. Without this the
        # session stayed open (but deaf) after a manual stop.
        self.manual_stop_sub = self.create_subscription(
            Bool, '/g1/voice/continuous_stop',
            self.manual_stop_callback, 10)

        if self.use_udp_mic:
            self._start_wake_listener()

        self.get_logger().info('Wake word node ready (Aletta + pad only)')

    def stt_status_callback(self, msg: String):
        self.stt_busy = msg.data in ('recording', 'transcribing')
        if self.stt_busy and self.session_open:
            self._touch_session()
        if msg.data == 'empty_turn' and self.session_open:
            # A turn produced no publishable transcript (silence, noise, or
            # a dropped low-confidence decode). Not activity — but also not
            # a reason to close on its own: the STT watchdog re-arms
            # listening, and the session survives until max_empty_turns.
            self.empty_turns += 1
            self.get_logger().info(
                f'Empty turn {self.empty_turns}/{self.max_empty_turns} '
                '(no transcript published)')
            if self.empty_turns >= self.max_empty_turns:
                self._close_session(
                    f'{self.empty_turns} consecutive empty turns')
        elif msg.data in ('recording', 'transcribing'):
            # A fresh turn is starting — handled above for the touch; the
            # empty counter resets only on a published transcript (see
            # robot_state/transcript path via llm busy) to avoid resetting
            # on back-to-back silence.
            pass

    def manual_stop_callback(self, msg: Bool):
        """START pad kill-switch: close the session immediately."""
        if msg.data and self.session_open:
            self._close_session('manual stop (START pad)')

    def llm_status_callback(self, msg: String):
        self.llm_busy = msg.data == 'thinking'
        if self.llm_busy and self.session_open:
            # A reply is being generated: this session is a real
            # conversation, not a false accept. End the backoff streak.
            self.session_had_llm_reply = True
            self.false_accept_count = 0
            self.cooldown = self.base_cooldown
            # A published transcript broke the empty streak — reset the
            # consecutive-empty counter so one noisy turn mixed into a
            # real conversation cannot close the session.
            self.empty_turns = 0
            self._touch_session()

    def tts_status_callback(self, msg: String):
        self.tts_busy = msg.data in ('speaking', 'synthesizing', 'playing')
        if self.tts_busy:
            self._last_tts_busy = time.time()
        if self.tts_busy and self.session_open:
            self._touch_session()

    def _tts_echo_active(self):
        """
        Report whether the mic is likely hearing the robot's own playback.

        Covers TTS busy time plus ``echo_guard_sec`` of room/enclosure ring-
        down after playback ends. The wake model was trained on the robot's
        own Piper voice, so scoring during this window produces near-certain
        false accepts (observed 2026-10-15: detection storms exactly while
        TTS chunks were playing).
        """
        if self.tts_busy:
            return True
        return (time.time() - self._last_tts_busy) < self.echo_guard_sec

    def _publish_session_state(self):
        """Publish the current session state for observability."""
        msg = String()
        msg.data = self._session_state
        self.session_state_pub.publish(msg)

    def _touch_session(self):
        """Mark turn activity (keeps the conversation window open)."""
        self.last_session_activity = time.time()

    def _open_session(self, source: str):
        """Open (or refresh) the conversation window on a valid entry."""
        was_open = self.session_open
        self.session_open = True
        self._touch_session()
        if not was_open:
            self.session_opened_at = time.time()
            self.empty_turns = 0
            self._session_state = 'opening'
            self._publish_session_state()
            # Enable chained turns downstream (STT continuous_mode + TTS
            # _TURN_END auto-trigger). Without this the window is local-only.
            msg = Bool()
            msg.data = True
            self.continuous_start_pub.publish(msg)
            self.get_logger().info(
                f'Conversation session OPEN ({source}) — closes on START, '
                f'{self.max_empty_turns} empty turns, or '
                f'{self.session_max_duration_sec:.0f}s max duration')
        else:
            # Session already open: this is a refresh, not a new opening.
            self._session_state = 'refreshing'
            self._publish_session_state()

    def _close_session(self, reason: str):
        """Close the window: back to throttled wait, no answers."""
        was_open = self.session_open
        if self.session_open:
            msg = Bool()
            msg.data = True
            self.continuous_stop_pub.publish(msg)
        self.session_open = False
        self.last_session_activity = 0.0
        self.session_opened_at = 0.0
        self.empty_turns = 0
        if was_open:
            self._session_state = 'closed'
            self._publish_session_state()
        # False-accept backoff: a wake-opened session that ends with NO LLM
        # reply was almost certainly a false accept (or hallucination turns
        # nobody asked for). Exponentially increase the cooldown so the idle
        # storm slows instead of re-firing every few seconds. A session that
        # produced a reply is a real conversation: session_had_llm_reply
        # stays True and no escalation happens (count was already reset when
        # the reply started). Closes are explicit now (START / empty-turn
        # limit / duration cap) — there is no silence timeout anymore.
        if (self.session_opened_by_wake and not self.session_had_llm_reply):
            self.false_accept_count += 1
            new_cooldown = self.base_cooldown * (
                self.false_accept_backoff_multiplier ** self.false_accept_count)
            self.cooldown = min(new_cooldown, self.max_false_accept_cooldown)
            self.get_logger().warn(
                f'False accept #{self.false_accept_count} — session ended '
                f'({reason}) with no reply, cooldown back off to '
                f'{self.cooldown:.0f}s')
        self.session_opened_by_wake = False
        self.session_had_llm_reply = False
        self.get_logger().info(
            f'Conversation session CLOSED ({reason}) — '
            'say "Aletta" to start again')

    def _can_trigger(self):
        now = time.time()
        if self.stt_busy or self.llm_busy or self.tts_busy:
            return False
        if (now - self.last_trigger_time) < self.cooldown:
            return False
        return True

    def _check_session_timeout(self):
        """Enforce the session duration cap (no silence timeout)."""
        """Kept under the historical name so the listener call sites and the
        unit-test harness keep working. Silence/empty turns never close the
        session here: STT re-arms listening via its watchdog, and the
        empty-turn counter (stt_status_callback) closes after
        max_empty_turns. This method enforces only the wall-clock backstop
        against phantom sessions, and never interrupts an in-flight turn."""
        if not self.session_open:
            return
        if self.stt_busy or self.llm_busy or self.tts_busy:
            # A turn is in flight — touch and never cap mid-turn.
            self._touch_session()
            return
        open_for = time.time() - self.session_opened_at
        if open_for >= self.session_max_duration_sec:
            self._close_session(
                f'max duration ({open_for:.0f}s >= '
                f'{self.session_max_duration_sec:.0f}s)')

    def _in_session_refresh(self, score):
        """Refresh an open session on a fresh wake-word detection.

        When the pipeline is free (not recording / transcribing / thinking /
            speaking),
        "Aletta" during an open session starts a fresh STT turn — just like F1
        push-to-talk. When the pipeline is busy, we only refresh the session
        window and log it, avoiding a race with the TTS _TURN_END chain.
        """
        self._touch_session()
        if not (self.stt_busy or self.llm_busy or self.tts_busy):
            self.get_logger().info(
                'Aletta during open session — starting new turn '
                f'(score={score:.2f})')
            self._fire_trigger('aletta_wake_word')
        else:
            self.get_logger().info(
                'Aletta heard during open session '
                f'(score={score:.2f}) - session refreshed '
                '(pipeline busy, no new turn)')

    def _fire_trigger(self, source: str):
        if not self._can_trigger():
            self.get_logger().debug(
                'Trigger blocked — pipeline busy or cooldown active')
            return
        self.last_trigger_time = time.time()
        if source != 'aletta_wake_word':
            # Pad button = a human is present and wants the robot: clear the
            # backoff so a manual press is never throttled. Wake fires keep
            # the streak: consecutive no-reply wake sessions accumulate
            # (8→16→32→60s). The old unconditional reset here restarted the
            # streak on every false accept, so the cooldown could never
            # exceed 16s — the observed ~25s session cadence of 2026-10-16.
            # The streak resets on a real reply (llm_status_callback) instead.
            self.cooldown = self.base_cooldown
            self.false_accept_count = 0
        self._open_session(source)
        self.session_opened_by_wake = (source == 'aletta_wake_word')
        # NOTE (missed-wake fix, 2026-10-16): the wake socket now STAYS OPEN
        # during sessions. Multicast UDP allows the STT and wake sockets to
        # coexist, so a repeated "Aletta" refreshes the window (see
        # _in_session_refresh) instead of hitting a closed socket. The firing
        # itself stays gated by _can_trigger (cooldown + pipeline-busy).
        self.get_logger().info(f'Trigger fired from: {source}')
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)
        status = String()
        status.data = f'triggered:{source}'
        self.status_pub.publish(status)

    def _init_openwakeword(self):
        """
        Load the "Aletta" openWakeWord model. Return True on success.

        Delegates to :class:`OpenWakeWordDetector`, which never raises: on any
        failure it logs the reason and leaves ``self.oww_detector`` as None.
        There is no fallback — wake stays disabled and the pad keeps working.
        """
        from g1_voice.wake_word_openwakeword import OpenWakeWordDetector
        detector = OpenWakeWordDetector(
            model_path=self.aletta_model_path,
            framework=self.aletta_framework,
            threshold=self.aletta_threshold,
            melspec_model_path=self.oww_melspec_model_path,
            embedding_model_path=self.oww_embedding_model_path,
            min_consecutive_frames=self.min_consecutive_frames)
        if detector.init(self.get_logger()):
            self.oww_detector = detector
            return True
        return False

    def _start_wake_listener(self):
        """
        Start the Aletta-only UDP mic loop.

        The custom openWakeWord model is the sole voice trigger. If it cannot
        load, wake stays disabled (loud error) — pad buttons via
        ``button_trigger_node`` keep working. The mic loop only starts once
        the detector is ready.
        """
        if self._init_openwakeword():
            if not self._wake_mic_path_logged:
                self._wake_mic_path_logged = True
                self.get_logger().info(
                    f'Wake mic path = UDP multicast '
                    f'(group={self.udp_multicast_group}, port={self.udp_port})')
            threading.Thread(target=self._wake_word_loop, daemon=True).start()
            self.get_logger().info(
                'Wake listener started — "Aletta" openWakeWord model '
                f'(threshold={self.aletta_threshold})')
        else:
            self.get_logger().error(
                'Wake listener disabled — Aletta model did not load; '
                'use pad buttons (F1/START/SELECT) to talk to the robot')

    def _wake_word_loop(self):
        # openWakeWord streams on 80 ms frames (1280 samples @16kHz).
        # Only this tiny ONNX model runs while waiting — Whisper/LLM/TTS stay
        # fully idle. A cheap RMS energy gate skips scoring on silence so the
        # idle wait costs ~zero CPU (one socket read + RMS per packet).
        oww_frame_bytes = 1280 * 2
        # RMS floor: below this the 80 ms frame is silence/room noise — skip
        # the ONNX score. Configurable via silence_rms parameter (default 300
        # int16 LSB ≈ quiet room; speech is 10-100x louder).
        # NOTE: keep the wake listener socket on its own port (or use
        # SO_REUSEPORT where available). The old code bound both the wake
        # listener and the STT socket to the same 5555 on the same interface,
        # which let one socket's multicast membership dominate the port and
        # starve the other of datagrams — the classic cause of "the trigger
        # fires but no audio is ever received" in a dual-socket UDP pipeline.
        while rclpy.ok():
            # NOTE: the socket intentionally stays open during sessions too
            # (missed-wake fix, 2026-10-16): a repeated "Aletta" refreshes the
            # window via _in_session_refresh instead of hitting a closed
            # socket. There is no paused state — the old `listening` flag was
            # write-only (always True) and its pause branch dead code.
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._wake_socket = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # NOTE: no SO_REUSEPORT — on Linux it load-balances datagrams
            # across same-port sockets, starving the STT socket (see stt_node).
            sock.bind(('', self.udp_port))
            mreq = struct.pack(
                '4s4s', socket.inet_aton(self.udp_multicast_group),
                socket.inet_aton(self.udp_local_ip))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(0.5)
            audio = bytearray()
            self.get_logger().info(
                'Aletta mic open on UDP '
                f'{self.udp_multicast_group}:{self.udp_port}')
            try:
                while rclpy.ok():
                    try:
                        data, _ = sock.recvfrom(65535)
                    except socket.timeout:
                        # Also runs when the mic goes quiet: enforces the
                        # session duration cap (silence never closes).
                        self._check_session_timeout()
                        continue
                    self._check_session_timeout()
                    # Idle wait: no open session — score only a short burst,
                    # then sleep. Keeps the wait state near zero CPU while
                    # still reacting to "Aletta" within ~a second.
                    if not self.session_open:
                        burst_end = time.time() + self.idle_listen_burst_sec
                        burst = bytearray(data)
                        while time.time() < burst_end:
                            try:
                                chunk, _ = sock.recvfrom(65535)
                            except socket.timeout:
                                break
                            burst.extend(chunk)
                            if len(burst) > 8 * oww_frame_bytes:
                                break
                        audio.extend(burst)
                        scored = self._score_buffered_audio(
                            audio, oww_frame_bytes)
                        # Always discard the scored burst: scored frames are
                        # consumed inside, and leftovers are stale by the next
                        # burst (after idle_sleep_sec). The old code only
                        # cleared on detection, leaking ~13 KB/s of idle audio
                        # into an ever-growing buffer.
                        audio.clear()
                        if scored:
                            break
                        # Firing itself stays gated inside _fire_trigger's
                        # _can_trigger check (cooldown + pipeline-busy): the
                        # mic keeps scoring through the cooldown so a real
                        # "Aletta" held past expiry fires at expiry
                        # (missed-wake, 2026-10-16).
                        self._log_burst_telemetry()
                        self._check_session_timeout()
                        time.sleep(self.idle_sleep_sec)
                        continue
                    audio.extend(data)
                    # Stream 80 ms frames: score each frame as it arrives.
                    while len(audio) >= oww_frame_bytes:
                        frame = np.frombuffer(
                            memoryview(audio)[:oww_frame_bytes],
                            dtype=np.int16).copy()
                        del audio[:oww_frame_bytes]
                        # Echo guard: while the robot is speaking (or the
                        # room is still ringing down) the mic hears the
                        # Piper voice the model was trained on — drain the
                        # audio WITHOUT scoring so it cannot self-trigger
                        # and the model's rolling window stays clean.
                        if self._tts_echo_active():
                            continue
                        # Energy gate: skip ONNX scoring on silence/room
                        # noise. float32 math on 1280 samples is ~microseconds;
                        # the ONNX embedding + classifier is ~1000x that.
                        rms = float(np.sqrt(np.mean(
                            frame.astype(np.float32) ** 2)))
                        if rms < self.silence_rms:
                            continue
                        if self.oww_detector.process(frame):
                            score = self.oww_detector.last_score
                            self.get_logger().info(
                                'Aletta wake word detected (openWakeWord, '
                                f'score={score:.2f})')
                            if self.session_open:
                                self._in_session_refresh(score)
                            else:
                                self._fire_trigger('aletta_wake_word')
                            self.oww_detector.reset()
                            audio.clear()
                            break
            except OSError:
                self.get_logger().error(
                    'Aletta UDP socket closed unexpectedly')
            except Exception as e:
                self.get_logger().error(f'Aletta UDP listener error: {e}')
            finally:
                if self._wake_socket is sock:
                    self._wake_socket = None
                sock.close()

    def _score_buffered_audio(self, audio, oww_frame_bytes):
        """
        Score buffered frames; return True if the wake word fired.

        Used by the idle-burst path (no open session): the whole burst is
        scored in one go and the caller clears the buffer. Detection fires
        the same trigger as the streaming path.
        """
        detected = False
        while len(audio) >= oww_frame_bytes:
            frame = np.frombuffer(
                memoryview(audio)[:oww_frame_bytes],
                dtype=np.int16).copy()
            del audio[:oww_frame_bytes]
            # Echo guard: drain without scoring while the robot's own
            # playback (or its ring-down) is on the mic (see _tts_echo_active).
            if self._tts_echo_active():
                continue
            # Energy gate: skip ONNX scoring on silence/room noise.
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms < self.silence_rms:
                continue
            if self.oww_detector.process(frame):
                self.get_logger().info(
                    'Aletta wake word detected (openWakeWord, '
                    f'score={self.oww_detector.last_score:.2f})')
                self._fire_trigger('aletta_wake_word')
                self.oww_detector.reset()
                audio.clear()
                detected = True
                break
        return detected

    def _log_burst_telemetry(self):
        """Throttled idle-burst observability (missed-wake, 2026-10-16)."""
        now = time.time()
        peak = getattr(self.oww_detector, 'last_score', 0.0) or 0.0
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = 0.0
        remaining = self.cooldown - (now - self.last_trigger_time)
        if remaining > 0 and peak >= self.aletta_threshold * 0.5:
            if now - self._blocked_by_cooldown_log >= 10.0:
                self._blocked_by_cooldown_log = now
                self.get_logger().warn(
                    'Wake-like audio during cooldown '
                    f'(peak={peak:.2f}, fires in {remaining:.0f}s) - '
                    'mic kept scoring, will fire at expiry if held')
            return
        if now - self._last_telemetry_log < self.telemetry_log_interval_sec:
            return
        self._last_telemetry_log = now
        self.get_logger().debug(
            'Wake idle burst: no detection '
            f'(peak score={peak:.2f} vs threshold={self.aletta_threshold:.2f})')

    def _resume_when_idle(self):
        # Kept for the unit-test harness (test_wake_word_session drives the
        # session/timeout logic through it). The live listener loop no longer
        # pauses — the mic stays open during sessions (see _wake_word_loop) —
        # so this only enforces the duration cap and logs the idle prompt.
        if self.stt_busy or self.llm_busy:
            return
        # Enforce the session duration cap so a wake-opened session
        # closes even if the mic path stalls, and the listener can resume.
        if self.session_open:
            self._check_session_timeout()
        if self.session_open:
            return
        # NOTE (missed-wake, 2026-10-16): only *firing* is
        # cooldown-gated (in _fire_trigger); the mic resumes immediately
        # so a real "Aletta" held past expiry fires at expiry.
        self.get_logger().info('Pipeline idle — say "Aletta" or use pad')


def main(args=None):
    rclpy.init(args=args)
    node = WakeWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
