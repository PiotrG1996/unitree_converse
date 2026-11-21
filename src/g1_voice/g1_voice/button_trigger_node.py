"""
Maps the Unitree wireless controller (/wirelesscontroller) to voice actions.

  F1 (bit 6)      robot online (Unitree GPT mode available): press = switch
                  between the Unitree GPT and Aletta; offline: hold =
                  push-to-talk (release ends the recording)
  F2 "F3" (bit 7) toggle mute of spoken replies
  START (bit 2)   end the conversation session (sessions open only via "Aletta")
  SELECT (bit 3)  speak ``custom_say_text`` (no STT/LLM)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

F1, F3, START, SELECT = 64, 128, 4, 8


class ButtonTriggerNode(Node):

    def __init__(self):
        super().__init__('button_trigger_node')
        self.custom_say_text = self.declare_parameter(
            'custom_say_text', 'Hello! I am Aletta.').value
        self.prev_keys = 0
        self.speak_enabled = True

        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.stop_pub = self.create_publisher(Bool, '/g1/voice/stop_recording', 10)
        self.continuous_stop_pub = self.create_publisher(
            Bool, '/g1/voice/continuous_stop', 10)
        self.speak_enable_pub = self.create_publisher(
            Bool, '/g1/voice/speak_enable', 10)
        self.say_pub = self.create_publisher(String, '/g1/voice/say_text', 10)
        self.toggle_pub = self.create_publisher(Bool, '/g1/voice/assistant_toggle', 10)
        self.online = False
        self._f1_toggled = False
        self.create_subscription(
            Bool, '/g1/voice/assistant_online', self._on_online,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        from unitree_go.msg import WirelessController
        self.create_subscription(
            WirelessController, '/wirelesscontroller', self.on_controller, 10)
        self.get_logger().info(
            'Pad: F1 switch Unitree/Aletta (online) or hold to talk (offline) | '
            'F3 mute | START end session | SELECT intro')

    def _on_online(self, msg):
        self.online = msg.data

    def on_controller(self, msg):
        keys, prev = msg.keys, self.prev_keys
        self.prev_keys = keys

        def pressed(bit):
            return bool(keys & bit) and not prev & bit

        if pressed(F1):
            self._f1_toggled = self.online
            if self.online:
                self.get_logger().info('F1: switch Unitree GPT / Aletta')
                self.toggle_pub.publish(Bool(data=True))
            else:
                self.trigger_pub.publish(Bool(data=True))
        elif prev & F1 and not keys & F1 and not self._f1_toggled:
            self.stop_pub.publish(Bool(data=True))
        if pressed(START):
            self.get_logger().info('START: ending conversation session')
            self.continuous_stop_pub.publish(Bool(data=True))
        if pressed(F3):
            self.speak_enabled = not self.speak_enabled
            self.speak_enable_pub.publish(Bool(data=self.speak_enabled))
            self.get_logger().info('Speech {}'.format(
                'enabled' if self.speak_enabled else 'MUTED'))
        if pressed(SELECT) and self.custom_say_text.strip():
            self.get_logger().info('SELECT: speaking the introduction')
            self.say_pub.publish(String(data=self.custom_say_text.strip()))


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
