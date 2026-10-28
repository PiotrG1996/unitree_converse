import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import time

# Unitree WirelessController bitfield (from unitree_sdk2 advanced_gamepad):
#   bit0 R1, bit1 L1, bit2 start, bit3 select, bit4 R2, bit5 L2,
#   bit6 F1, bit7 F2, bit8 A, bit9 B, bit10 X, bit11 Y,
#   bit12 up, bit13 right, bit14 down, bit15 left
# NOTE: there is no physical "F3" on the pad — bit 7 is F2 (referred to as "F3"
# in older Unitree docs); we keep the F3 name here to match that convention.
# bit 6 F1: push-to-talk (hold to record, release to send)
F1_KEY = 64
# bit 7 F2 (called "F3" in older Unitree docs): toggle speaking enabled/muted
F3_KEY = 128
# bit 2 START: end the conversation session (kill switch — sessions open only
# via "Aletta"; moved off F3 so F3 could be dedicated to the mute control)
START_KEY = 4
# bit 3 SELECT: speak one fixed custom phrase (custom_say_text) via TTS,
# bypassing STT/LLM. Free in this pipeline (F1/F3/START are taken).
SELECT_KEY = 8

DEBOUNCE_TIME = 0.25  # seconds — de-bounce the switch but allow fast double-taps

try:
    from .node_logging import info as log_info, warn as log_warn
except ImportError:
    try:
        from g1_voice.node_logging import info as log_info, warn as log_warn
    except ImportError:
        import os as _os
        import sys as _sys

        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import node_logging as _node_logging

        log_info = _node_logging.info
        log_warn = _node_logging.warn


class ButtonTriggerNode(Node):
    def __init__(self):
        super().__init__('button_trigger_node')

        self.continuous_mode = False
        self.speak_enabled = True
        self.f1_pressed = False
        self.f3_pressed = False
        self.start_pressed = False
        self.select_pressed = False
        self.recording_active = False
        self.prev_keys = 0
        self.last_f3_time = 0.0    # for debouncing the speaking mute toggle
        self.last_start_time = 0.0  # for debouncing the continuous-mode toggle
        self.last_select_time = 0.0  # for debouncing the custom-say trigger

        # Long default split across string literals to stay under the 99-col
        # flake8 limit; the YAML configs override it anyway.
        self.declare_parameter(
            'custom_say_text', 'Hello! I am Aletta, a humanoid robot assistant at '
            'Poznan University of Technology. I am here to support '
            'researchers, students, and academic staff in their daily '
            'activities. I can assist with research-related tasks, provide '
            'information, answer questions, and help you explore new ideas '
            'in science and technology. Whether you are conducting research, '
            'working on a project, or simply curious about robotics and '
            'artificial intelligence, I am happy to help. My goal is to make '
            'your work easier, support scientific collaboration, and contribute '
            'to an innovative research environment. Welcome to Poznan '
            'University of Technology! How can I assist you today?')
        self.custom_say_text = self.get_parameter('custom_say_text').value

        # Publishers
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.stop_pub = self.create_publisher(
            Bool, '/g1/voice/stop_recording', 10)
        self.continuous_start_pub = self.create_publisher(
            Bool, '/g1/voice/continuous_start', 10)
        self.continuous_stop_pub = self.create_publisher(
            Bool, '/g1/voice/continuous_stop', 10)
        self.speak_enable_pub = self.create_publisher(
            Bool, '/g1/voice/speak_enable', 10)
        self.say_pub = self.create_publisher(String, '/g1/voice/say_text', 10)
        self.status_pub = self.create_publisher(
            String, '/g1/button/status', 10)

        # Track the real session state that wake_word_node owns. It publishes
        # /g1/voice/continuous_start when "Aletta" opens a session, and any
        # close (START kill-switch, empty-turn limit, duration cap) publishes
        # /g1/voice/continuous_stop. Before this, the button node's own
        # continuous_mode flag stayed False forever, so START was always
        # "ignored" and could never end an open Aletta session.
        self.create_subscription(
            Bool, '/g1/voice/continuous_start', self._on_continuous_start, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_stop', self._on_continuous_stop, 10)

        from unitree_go.msg import WirelessController
        self.controller_sub = self.create_subscription(
            WirelessController,
            '/wirelesscontroller',
            self.controller_callback,
            10
        )

        log_info(self, 'Button trigger node ready')
        log_info(self, 'Hold F1 → push-to-talk (release to send)')
        log_info(self, 'F3 → toggle speaking enabled/muted (robot TTS output)')
        log_info(self, 'START → end conversation session (kill switch)')
        log_info(self, 'SELECT → speak custom phrase (custom_say_text)')

        # Publish the initial speaking-enabled state so subscribers can sync
        # (TTS defaults to enabled too, so this is belt-and-suspenders).
        self._publish_speak_enable(self.speak_enabled)

    def controller_callback(self, msg):
        keys = msg.keys
        prev = self.prev_keys
        self.prev_keys = keys

        f1_now = bool(keys & F1_KEY)
        f3_now = bool(keys & F3_KEY)
        start_now = bool(keys & START_KEY)
        select_now = bool(keys & SELECT_KEY)
        f1_prev = bool(prev & F1_KEY)
        f3_prev = bool(prev & F3_KEY)
        start_prev = bool(prev & START_KEY)
        select_prev = bool(prev & SELECT_KEY)

        self.f1_pressed = f1_now
        self.f3_pressed = f3_now
        self.start_pressed = start_now
        self.select_pressed = select_now

        # START rising edge with debounce → end the session (kill switch: the
        # press only stops, never starts). Handled synchronously so it is
        # never dropped by a slow worker.
        if start_now and not start_prev:
            now = time.time()
            if now - self.last_start_time > DEBOUNCE_TIME:
                self.last_start_time = now
                self._toggle_continuous()
            return

        # SELECT rising edge with debounce → speak the fixed custom phrase.
        if select_now and not select_prev:
            now = time.time()
            if now - self.last_select_time > DEBOUNCE_TIME:
                self.last_select_time = now
                self._publish_say_text()
            return

        # F3 rising edge with debounce → toggle speaking enabled/muted.
        if f3_now and not f3_prev:
            now = time.time()
            if now - self.last_f3_time > DEBOUNCE_TIME:
                self.last_f3_time = now
                self._toggle_speak_enable()
            return

        # F1 pressed (rising edge) → start recording. Allowed in push-to-talk
        # mode and during an open session so the user can force a turn.
        if f1_now and not f1_prev:
            log_info(self, 'F1 held → start recording')
            self.recording_active = True
            self._publish_trigger(True)
            self._publish_status('recording')

        # F1 released (falling edge) -> stop recording and transcribe.
        if not f1_now and f1_prev and self.recording_active:
            log_info(self, 'F1 released → stop recording, transcribing...')
            self.recording_active = False
            self._publish_stop()
            self._publish_status('transcribing')

    def _on_continuous_start(self, msg: Bool):
        r"""Session opened (e.g. via \"Aletta\") — suppress F1 push-to-talk."""
        if msg.data:
            self.continuous_mode = True
            log_info(
                self,
                'Continuous mode ON (session opened) — F1 push-to-talk disabled')

    def _on_continuous_stop(self, msg: Bool):
        """Session ended — re-enable F1 push-to-talk."""
        if msg.data:
            self.continuous_mode = False
            log_info(
                self,
                'Continuous mode OFF (session closed) — F1 push-to-talk re-enabled')

    def _toggle_continuous(self):
        # Continuous conversation is Aletta-only (wake_word_node owns
        # /g1/voice/continuous_start). START can no longer START a session —
        # it only ends one, acting as a manual kill switch mid-conversation.
        if self.continuous_mode:
            self.continuous_mode = False
            msg = Bool()
            msg.data = True
            self.continuous_stop_pub.publish(msg)
            log_info(self, 'Continuous mode OFF (manual stop)')
            self._publish_status('continuous_off')
        else:
            log_info(
                self,
                'START ignored — continuous mode starts only via "Aletta"')
            self._publish_status('continuous_blocked')

    def _toggle_speak_enable(self):
        self.speak_enabled = not self.speak_enabled
        self._publish_speak_enable(self.speak_enabled)
        if self.speak_enabled:
            log_info(self, 'Speaking ENABLED (robot will speak replies)')
            self._publish_status('speaking_enabled')
        else:
            log_info(self, 'Speaking MUTED (robot will stay silent)')
            self._publish_status('speaking_muted')

    def _publish_trigger(self, value: bool):
        msg = Bool()
        msg.data = value
        self.trigger_pub.publish(msg)

    def _publish_say_text(self):
        text = (self.custom_say_text or '').strip()
        if not text:
            log_warn(self, 'SELECT pressed but custom_say_text is empty')
            return
        msg = String()
        msg.data = text
        self.say_pub.publish(msg)
        log_info(self, f'SELECT → speaking custom phrase ({len(text)} chars)')
        self._publish_status('custom_say')

    def _publish_speak_enable(self, value: bool):
        msg = Bool()
        msg.data = value
        self.speak_enable_pub.publish(msg)

    def _publish_stop(self):
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ButtonTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
