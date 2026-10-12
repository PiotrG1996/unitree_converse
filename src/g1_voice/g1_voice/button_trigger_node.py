import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import threading
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
# bit 2 START: toggle continuous listening mode (moved off F3 so F3 can be
# dedicated to the speaking mute control)
START_KEY = 4

DEBOUNCE_TIME = 0.25  # seconds — de-bounce the switch but allow fast double-taps


class ButtonTriggerNode(Node):
    def __init__(self):
        super().__init__('button_trigger_node')

        self.continuous_mode = False
        self.speak_enabled = True
        self.f1_pressed = False
        self.f3_pressed = False
        self.start_pressed = False
        self.recording_active = False
        self.prev_keys = 0
        self.last_f3_time = 0.0    # for debouncing the speaking mute toggle
        self.last_start_time = 0.0  # for debouncing the continuous-mode toggle

        # Publishers
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.stop_pub = self.create_publisher(Bool, '/g1/voice/stop_recording', 10)
        self.continuous_start_pub = self.create_publisher(Bool, '/g1/voice/continuous_start', 10)
        self.continuous_stop_pub = self.create_publisher(Bool, '/g1/voice/continuous_stop', 10)
        self.speak_enable_pub = self.create_publisher(Bool, '/g1/voice/speak_enable', 10)
        self.status_pub = self.create_publisher(String, '/g1/button/status', 10)

        from unitree_go.msg import WirelessController
        self.controller_sub = self.create_subscription(
            WirelessController,
            '/wirelesscontroller',
            self.controller_callback,
            10
        )

        self.get_logger().info('Button trigger node ready')
        self.get_logger().info('Hold F1 → push-to-talk (release to send)')
        self.get_logger().info('F3 → toggle speaking enabled/muted (robot TTS output)')
        self.get_logger().info('START → toggle continuous listening mode')

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
        f1_prev = bool(prev & F1_KEY)
        f3_prev = bool(prev & F3_KEY)
        start_prev = bool(prev & START_KEY)

        self.f1_pressed = f1_now
        self.f3_pressed = f3_now
        self.start_pressed = start_now

        # START rising edge with debounce → toggle continuous listening mode. The
        # toggle runs synchronously so a press is never dropped by a slow worker.
        if start_now and not start_prev:
            now = time.time()
            if now - self.last_start_time > DEBOUNCE_TIME:
                self.last_start_time = now
                self._toggle_continuous()
            return

        # F3 rising edge with debounce → toggle speaking enabled/muted.
        if f3_now and not f3_prev:
            now = time.time()
            if now - self.last_f3_time > DEBOUNCE_TIME:
                self.last_f3_time = now
                self._toggle_speak_enable()
            return

        # F1 pressed (rising edge) → start recording, only in push-to-talk mode
        if f1_now and not f1_prev and not self.continuous_mode:
            self.get_logger().info('F1 held → start recording')
            self.recording_active = True
            self._publish_trigger(True)
            self._publish_status('recording')

        # F1 released (falling edge) → stop recording
        if not f1_now and f1_prev and self.recording_active:
            self.get_logger().info('F1 released → stop recording, transcribing...')
            self.recording_active = False
            self._publish_stop()
            self._publish_status('transcribing')

    def _toggle_continuous(self):
        if self.continuous_mode:
            self.continuous_mode = False
            msg = Bool()
            msg.data = True
            self.continuous_stop_pub.publish(msg)
            self.get_logger().info('Continuous mode OFF')
            self._publish_status('continuous_off')
        else:
            self.continuous_mode = True
            msg = Bool()
            msg.data = True
            self.continuous_start_pub.publish(msg)
            self.get_logger().info('Continuous mode ON')
            self._publish_status('continuous_on')
            # Fire the first trigger shortly after, off the ROS callback thread.
            threading.Thread(target=self._delayed_first_trigger, daemon=True).start()

    def _delayed_first_trigger(self):
        time.sleep(0.5)
        self._publish_trigger(True)

    def _toggle_speak_enable(self):
        self.speak_enabled = not self.speak_enabled
        self._publish_speak_enable(self.speak_enabled)
        if self.speak_enabled:
            self.get_logger().info('Speaking ENABLED (robot will speak replies)')
            self._publish_status('speaking_enabled')
        else:
            self.get_logger().info('Speaking MUTED (robot will stay silent)')
            self._publish_status('speaking_muted')

    def _publish_trigger(self, value: bool):
        msg = Bool()
        msg.data = value
        self.trigger_pub.publish(msg)

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
