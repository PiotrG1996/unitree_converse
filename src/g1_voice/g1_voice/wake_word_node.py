"""
Wake-word node: "Aletta" opens a conversation session.

Detection uses the custom openWakeWord model.

* Session closed: a detection opens it (subject to the false-accept
  cooldown), publishes /g1/voice/continuous_start and triggers a turn.
* Session open: a detection while the pipeline is idle starts a new turn.
  A detection while a turn is recording is still published on
  /g1/wake/detected (no trigger) so stt_node can cut the name off the turn.
* The session closes on START (/g1/voice/continuous_stop), on
  ``max_empty_turns`` consecutive turns without a transcript, or after
  ``session_max_duration_sec``; see ``logic.Session``.

Every 80 ms mic frame is fed to the model so its rolling feature window stays
continuous. The score peaks ~1 s *after* the word (the model classifies a
~1.3 s window), when the room is already quiet again, so the loudness gate
looks at the peak of the last ``energy_window_sec``, not the current frame.
Detections while that window was quiet, or inside the echo guard (our TTS
or anything else on the G1 speaker, from the audio board's ``play_state`` on
/audio_msg, e.g. Unitree's built-in assistant), are vetoed.
While /g1/voice/assistant_mode is ``unitree`` (robot on Wi-Fi: the official
Unitree GPT answers) detections are ignored, so neither "Aletta" nor a false
wake can interrupt it. With ``takeover_from_unitree`` true, "Aletta" instead
opens a session that hands the conversation to Aletta until it closes.
If the model cannot load, wake stays disabled (logged) and the pad buttons
still work.
"""

import socket
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# The model scores a ~1.3 s window: after a hit and reset it fires again on
# the same word ~1.1-1.3 s later (measured). Hits this close are one word.

from std_msgs.msg import Bool, Float64, String

from g1_voice.logic import EchoGuard, Session, board_play_state, open_multicast_socket
from g1_voice.wake_word_openwakeword import FRAME_SAMPLES, OpenWakeWordDetector

FRAME_BYTES = FRAME_SAMPLES * 2


class WakeWordNode(Node):

    def __init__(self):
        super().__init__('wake_word_node')
        p = self._declare
        model_path = p('aletta_model_path',
                       '~/.local/share/openwakeword/aletta_v0.1.onnx')
        framework = p('aletta_framework', 'onnx')
        self.threshold = p('aletta_threshold', 0.8)
        min_frames = p('min_consecutive_frames', 3)
        self.refractory = p('refractory_sec', 1.5)
        self.takeover_from_unitree = p('takeover_from_unitree', False)
        self._last_hit = -1e9
        melspec = p('oww_melspec_model_path', '')
        embedding = p('oww_embedding_model_path', '')
        # int16 RMS that the loudest frame of the last energy_window_sec must
        # reach. Measured on the G1 mic: quiet room p99 ~350, voice 600-1500.
        self.min_rms = p('min_rms', 400.0)
        self._rms_window = deque(maxlen=max(1, int(p('energy_window_sec', 2.0) / 0.08)))
        self.echo_guard_sec = p('echo_guard_sec', 2.5)
        board_topic = p('board_play_state_topic', '/audio_msg')
        self.udp_group = p('udp_multicast_group', '239.168.123.161')
        self.udp_port = p('udp_port', 5555)
        self.udp_local_ip = p('udp_local_ip', '192.168.123.164')
        self.session = Session(
            max_duration=p('session_max_duration_sec', 300.0),
            max_empty_turns=p('max_empty_turns', 4),
            base_cooldown=p('base_cooldown_sec', 4.0),
            backoff=p('false_accept_backoff_multiplier', 1.5),
            max_cooldown=p('max_false_accept_cooldown', 30.0))
        self._lock = threading.Lock()
        self._echo = EchoGuard(self.echo_guard_sec)
        self._stt_recording = False
        self._unitree_mode = False

        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.start_pub = self.create_publisher(Bool, '/g1/voice/continuous_start', 10)
        self.stop_pub = self.create_publisher(Bool, '/g1/voice/continuous_stop', 10)
        # Latched, so a node that starts after the session opened still learns
        # it (stt_node once missed continuous_start by one second at boot).
        self.state_pub = self.create_publisher(
            String, '/g1/wake/session_state',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.state_pub.publish(String(data='closed'))
        # Wall time of each detection that starts a turn (stt_node begins the
        # turn's audio right after the word) or lands in a recording turn
        # (stt_node cuts the name off).
        self.detected_pub = self.create_publisher(Float64, '/g1/wake/detected', 10)
        # New session = new conversation: misheard turns from an earlier
        # session must not steer the LLM's next answers.
        self.llm_reset_pub = self.create_publisher(Bool, 'llm_reset', 10)
        self.create_subscription(String, '/g1/stt/status', self._on_stt, 10)
        self.create_subscription(String, '/g1/stt/transcript', self._on_transcript, 10)
        self.create_subscription(String, 'llm_response', self._on_reply, 10)
        self.create_subscription(String, '/g1/tts/status', self._on_tts, 10)
        self.create_subscription(
            String, '/g1/voice/assistant_mode', self._on_assistant_mode,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Bool, '/g1/voice/continuous_stop', self._on_stop, 10)
        # assistant_mode_node opens a session after F1 switched to Aletta.
        self.create_subscription(Bool, '/g1/voice/session_request', self._on_session_request, 10)
        if board_topic:
            self.create_subscription(
                String, board_topic, self._on_board, QoSProfile(
                    depth=20, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0, self._check_expiry)

        self.detector = OpenWakeWordDetector(
            model_path=model_path, framework=framework, threshold=self.threshold,
            melspec_model_path=melspec, embedding_model_path=embedding,
            min_consecutive_frames=min_frames)
        if self.detector.init(self.get_logger()):
            threading.Thread(target=self._mic_loop, daemon=True).start()
            self.get_logger().info('Listening for "Aletta" on UDP {}:{}'.format(
                self.udp_group, self.udp_port))
        else:
            self.get_logger().error('Wake word disabled; use the pad buttons')

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    # -------------------------------------------------------- pipeline state
    def _on_stt(self, msg):
        with self._lock:
            self._stt_recording = msg.data == 'recording'
            self.session.set_busy('stt', msg.data in ('recording', 'transcribing'))
            if msg.data == 'empty_turn' and self.session.on_empty_turn():
                self._close('{} empty turns'.format(self.session.empty_turns))

    def _on_transcript(self, _msg):
        with self._lock:
            self.session.on_transcript()
            self.session.set_busy('reply', True)

    def _on_reply(self, _msg):
        with self._lock:
            self.session.set_busy('reply', False)

    def _on_tts(self, msg):
        speaking = msg.data == 'speaking'
        with self._lock:
            self._echo.set('tts', speaking)
            self.session.set_busy('tts', speaking)

    def _on_board(self, msg):
        playing = board_play_state(msg.data)
        if playing is not None:
            with self._lock:
                self._echo.set('board', playing)

    def _on_stop(self, msg):
        with self._lock:
            if msg.data and self.session.is_open:
                self._close('START')

    def _check_expiry(self):
        with self._lock:
            if self.session.expired():
                self._close('max duration')

    def _close(self, reason):
        """Close the session (caller holds the lock)."""
        if not self.session.close():
            return
        self.stop_pub.publish(Bool(data=True))
        self.state_pub.publish(String(data='closed'))
        self.get_logger().info(
            'Session CLOSED ({}); next wake cooldown {:.0f}s'.format(
                reason, self.session.cooldown))

    # ------------------------------------------------------------ detection
    def _on_assistant_mode(self, msg):
        with self._lock:
            self._unitree_mode = msg.data == 'unitree'

    def _on_wake(self, score, detected_at):
        with self._lock:
            if self._unitree_mode and not self.takeover_from_unitree:
                self.get_logger().debug(
                    'Aletta ({:.2f}) ignored: Unitree GPT answers while online'.format(score))
                return
            session = self.session
            if session.is_open:
                if self._stt_recording:
                    self.get_logger().info(
                        'Aletta ({:.2f}) during recording: '
                        'sent to STT to cut the name'.format(score))
                    self.detected_pub.publish(Float64(data=detected_at))
                    return
                if session.busy():
                    self.get_logger().info(
                        'Aletta ({:.2f}) ignored: turn in progress'.format(score))
                    return
                self.get_logger().info('Aletta ({:.2f}) in session: new turn'.format(score))
            elif session.can_open():
                session.open()
                if self._unitree_mode:
                    self.get_logger().info('Aletta takes over from Unitree GPT')
                self.llm_reset_pub.publish(Bool(data=True))
                self.start_pub.publish(Bool(data=True))
                self.state_pub.publish(String(data='open'))
                self.get_logger().info('Session OPEN (Aletta {:.2f})'.format(score))
            else:
                self.get_logger().info(
                    'Aletta ({:.2f}) ignored: cooldown/busy'.format(score))
                return
            self.detected_pub.publish(Float64(data=detected_at))
            self.trigger_pub.publish(Bool(data=True))

    def _on_session_request(self, msg):
        """Open a session without the wake word (F1 switched to Aletta)."""
        if not msg.data:
            return
        with self._lock:
            if self.session.is_open:
                return
            self.session.open()
            self.llm_reset_pub.publish(Bool(data=True))
            self.start_pub.publish(Bool(data=True))
            self.state_pub.publish(String(data='open'))
            self.get_logger().info('Session OPEN (F1)')
            self.trigger_pub.publish(Bool(data=True))

    def _echo_active(self):
        with self._lock:
            return self._echo.active()

    def _mic_loop(self):
        buffer = bytearray()
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
                arrived = time.time()
                buffer.extend(data)
                while len(buffer) >= FRAME_BYTES:
                    frame = np.frombuffer(bytes(buffer[:FRAME_BYTES]), dtype=np.int16)
                    del buffer[:FRAME_BYTES]
                    # Wall time at the end of this frame's audio.
                    end = arrived - (len(buffer) // 2) / 16000.0
                    self._process(frame, end)
            except OSError as exc:
                self.get_logger().error(
                    'Mic socket error ({}); retrying'.format(exc),
                    throttle_duration_sec=30.0)
                if sock is not None:
                    sock.close()
                sock = None
                time.sleep(1.0)

    def _process(self, frame, arrived):
        hit = self.detector.process(frame)
        self._rms_window.append(float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))))
        if max(self._rms_window) < self.min_rms or self._echo_active():
            self.detector.veto()
            return
        if hit:
            self.detector.reset()
            now = time.monotonic()
            if now - self._last_hit < self.refractory:
                return  # the same "Aletta" scored again
            self._last_hit = now
            self._on_wake(self.detector.last_score, arrived)


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
