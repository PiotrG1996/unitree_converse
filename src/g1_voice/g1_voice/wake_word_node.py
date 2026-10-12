import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import numpy as np
import socket
import struct
import threading
import time
import re
import unicodedata


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter(
            'wake_phrases', ['hey robot', 'cześć robot'])
        self.declare_parameter(
            'wake_phrase_aliases', ['czesc robot', 'cześ robot'])
        self.declare_parameter('wake_model_size', 'base')
        self.declare_parameter('wake_language', 'auto')
        self.declare_parameter(
            'wake_initial_prompt', 'Hey robot. Cześć robot. Czesc robot.')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('wake_chunk_duration', 2.0)
        self.declare_parameter('use_udp_mic', True)
        self.declare_parameter('udp_multicast_group', '239.168.123.161')
        self.declare_parameter('udp_port', 5555)
        self.declare_parameter('udp_local_ip', '192.168.123.164')
        self.declare_parameter('use_select_button', False)
        self.declare_parameter('select_key', 's')

        self.wake_phrases = list(
            self.get_parameter('wake_phrases').get_parameter_value().string_array_value)
        self.wake_phrase_aliases = list(
            self.get_parameter('wake_phrase_aliases')
            .get_parameter_value().string_array_value)
        self.wake_model_size = self.get_parameter('wake_model_size').get_parameter_value().string_value
        self.wake_language = self.get_parameter('wake_language').get_parameter_value().string_value
        self.wake_initial_prompt = self.get_parameter(
            'wake_initial_prompt').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.wake_chunk_duration = self.get_parameter('wake_chunk_duration').get_parameter_value().double_value
        self.use_udp_mic = self.get_parameter('use_udp_mic').get_parameter_value().bool_value
        self.udp_multicast_group = self.get_parameter('udp_multicast_group').get_parameter_value().string_value
        self.udp_port = self.get_parameter('udp_port').get_parameter_value().integer_value
        self.udp_local_ip = self.get_parameter('udp_local_ip').get_parameter_value().string_value
        self.use_select_button = self.get_parameter('use_select_button').get_parameter_value().bool_value
        self.select_key = self.get_parameter('select_key').get_parameter_value().string_value

        self.last_trigger_time = 0.0
        self.cooldown = 2.0
        self.listening = True
        self.model = None
        self._wake_socket = None

        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.status_pub = self.create_publisher(String, '/g1/wake/status', 10)

        self.stt_busy = False
        self.llm_busy = False
        self.stt_status_sub = self.create_subscription(
            String, '/g1/stt/status', self.stt_status_callback, 10)
        self.llm_status_sub = self.create_subscription(
            String, '/g1/llm/status', self.llm_status_callback, 10)

        if self.use_udp_mic:
            self._init_whisper()
            if self.model is not None:
                threading.Thread(target=self._wake_word_loop, daemon=True).start()
                self.get_logger().info(
                    f'Wake phrase listener started — phrases: '
                    f'{", ".join(self.wake_phrases)}')
            else:
                self.get_logger().error('Wake phrase listener disabled — Whisper failed to load')

        if self.use_select_button:
            # Use a ROS2 timer-based keyboard poller instead of raw tty
            # This works even when stdin is not a terminal (e.g. ros2 launch)
            threading.Thread(target=self._select_button_loop, daemon=True).start()
            self.get_logger().info(f'SELECT button listener started — press "{self.select_key}" + Enter to trigger')

        self.get_logger().info('Wake word node ready')

    def stt_status_callback(self, msg: String):
        self.stt_busy = msg.data in ('recording', 'transcribing')

    def llm_status_callback(self, msg: String):
        self.llm_busy = msg.data == 'thinking'

    def _can_trigger(self):
        now = time.time()
        if self.stt_busy or self.llm_busy:
            return False
        if (now - self.last_trigger_time) < self.cooldown:
            return False
        return True

    def _fire_trigger(self, source: str):
        if not self._can_trigger():
            self.get_logger().info('Trigger blocked — pipeline busy or cooldown active')
            return
        self.last_trigger_time = time.time()
        self.listening = False
        if self._wake_socket is not None:
            self._wake_socket.close()
            self._wake_socket = None
        self.get_logger().info(f'Trigger fired from: {source}')
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)
        status = String()
        status.data = f'triggered:{source}'
        self.status_pub.publish(status)

    def _init_whisper(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.wake_model_size, device='cpu', compute_type='int8')
            self.get_logger().info('Wake phrase Whisper model loaded')
        except Exception as e:
            self.get_logger().error(f'Failed to load wake phrase model: {e}')
            self.model = None

    def _wake_word_loop(self):
        bytes_per_chunk = int(self.sample_rate * self.wake_chunk_duration * 2)
        while rclpy.ok():
            if not self.listening:
                self._resume_when_idle()
                time.sleep(0.2)
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._wake_socket = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(('', self.udp_port))
            mreq = struct.pack(
                '4s4s', socket.inet_aton(self.udp_multicast_group),
                socket.inet_aton(self.udp_local_ip))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(0.5)
            audio = bytearray()
            self.get_logger().info(
                f'Wake phrase mic open on UDP {self.udp_multicast_group}:{self.udp_port}')
            try:
                while rclpy.ok() and self.listening:
                    try:
                        data, _ = sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    audio.extend(data)
                    if len(audio) < bytes_per_chunk:
                        continue
                    chunk = np.frombuffer(
                        bytes(audio[:bytes_per_chunk]), dtype=np.int16)
                    del audio[:bytes_per_chunk]
                    audio_chunk = chunk.astype(np.float32) / 32768.0
                    transcript = self._transcribe_wake(audio_chunk)
                    if transcript:
                        self.get_logger().info(
                            f'Wake phrase audio: "{transcript}"')
                    detected_phrase = self._matching_wake_phrase(transcript)
                    if detected_phrase is None and self.wake_language.lower() == 'auto':
                        for language in ('pl', 'en'):
                            fallback = self._transcribe_wake(audio_chunk, language)
                            detected_phrase = self._matching_wake_phrase(fallback)
                            if detected_phrase is not None:
                                self.get_logger().info(
                                    f'Wake phrase audio ({language} fallback): '
                                    f'"{fallback}"')
                                break
                    if detected_phrase is not None:
                        self.get_logger().info(
                            f'Wake phrase detected: "{detected_phrase}"')
                        self._fire_trigger('wake_phrase')
                        audio.clear()
            except OSError:
                if self.listening:
                    self.get_logger().error('Wake phrase UDP socket closed unexpectedly')
            except Exception as e:
                self.get_logger().error(f'Wake phrase UDP listener error: {e}')
            finally:
                if self._wake_socket is sock:
                    self._wake_socket = None
                sock.close()

    def _transcribe_wake(self, audio, language=None):
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.wake_initial_prompt,
            vad_filter=True)
        return ' '.join(segment.text for segment in segments).strip()

    def _resume_when_idle(self):
        if self.listening or self.stt_busy or self.llm_busy:
            return
        if time.time() - self.last_trigger_time >= self.cooldown:
            self.listening = True
            self.get_logger().info('Pipeline idle — waiting for wake phrase')

    @staticmethod
    def _normalize_phrase(value: str) -> str:
        value = unicodedata.normalize('NFKD', value.lower())
        value = ''.join(char for char in value
                        if not unicodedata.combining(char))
        return re.sub(r'[^a-z0-9 ]+', ' ', value).strip()

    def _matching_wake_phrase(self, transcript: str):
        words = self._normalize_phrase(transcript)
        for phrase in self.wake_phrases + self.wake_phrase_aliases:
            normalized = self._normalize_phrase(phrase)
            if normalized and re.search(rf'\b{re.escape(normalized)}\b', words):
                return phrase
        return None

    def _select_button_loop(self):
        """
        Reads lines from stdin. Works both in terminal and via ros2 launch.
        Press Enter (or 's' + Enter) to trigger.
        On real robot: replace this with a subscriber to the Unitree remote topic.
        """
        import sys
        self.get_logger().info(
            f'Keyboard trigger ready — press Enter (or "{self.select_key}" + Enter) in this terminal')
        try:
            for line in sys.stdin:
                line = line.strip()
                if line == '' or line == self.select_key:
                    self._fire_trigger('select_button')
        except Exception as e:
            self.get_logger().error(f'SELECT button listener error: {e}')


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
