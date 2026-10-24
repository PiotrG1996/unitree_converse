import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import subprocess

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


class RobotStateNode(Node):
    """
    Enrich STT transcripts with live robot state before the LLM sees them.

    Subscribes to /g1/stt/transcript, enriches it with live robot state,
    and republishes to llm_prompt (which llm_node listens to).

    State collected:
    - Battery: SOC, SOH, current, temperature
    - Network: WiFi IP, ethernet IP
    - System: uptime, ROS2 nodes count
    - IMU: orientation (roll/pitch/yaw)
    """

    def __init__(self):
        super().__init__('robot_state_node')

        # Internal state cache
        self.battery_soc = None
        self.battery_soh = None
        self.battery_current = None
        self.battery_temp = None
        self.battery_voltage = None
        self.battery_cycles = None
        self.imu_rpy = None
        self.mode = None

        # Cached ROS2 node count. Counting nodes requires spawning `ros2 node
        # list`, which performs full DDS discovery and can take hundreds of ms
        # to seconds. It is refreshed periodically in the background instead of
        # on every user turn so it never blocks the transcript -> LLM path.
        self._cached_ros_nodes = 'unknown'

        # Subscribe to robot state topics
        try:
            from unitree_hg.msg import BmsState
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=10
            )
            self.create_subscription(
                BmsState, '/lf/bmsstate', self._bms_callback, qos)
            log_info(self, 'Subscribed to /lf/bmsstate')
        except Exception as e:
            log_warn(self, f'Could not subscribe to BmsState: {e}')

        try:
            from unitree_go.msg import SportModeState
            self.create_subscription(
                SportModeState,
                '/odommodestate',
                self._sport_callback,
                10)
            log_info(self, 'Subscribed to /odommodestate')
        except Exception as e:
            log_warn(self, f'Could not subscribe to SportModeState: {e}')

        # Subscribe to STT transcript
        self.transcript_sub = self.create_subscription(
            String, '/g1/stt/transcript', self._transcript_callback, 10)

        # Publish enriched prompt to LLM
        self.prompt_pub = self.create_publisher(String, 'llm_prompt', 10)
        self.status_pub = self.create_publisher(String, '/g1/robot_state', 10)

        # Publish state summary every 30 seconds
        self.create_timer(30.0, self._publish_state_summary)

        # Refresh the (expensive) ROS2 node count in the background, decoupled
        # from the per-turn prompt path.
        self.create_timer(30.0, self._update_ros_node_count)

        log_info(self, 'Robot state node ready')

    def _update_ros_node_count(self):
        """Count ROS2 nodes in the background; cache for per-turn reuse."""
        try:
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True, text=True, timeout=5
            )
        except Exception:
            # Timeout/spawn failure: keep the previous cached value.
            return
        if result.returncode != 0:
            # CLI/daemon errors return empty stdout; caching that would
            # overwrite a good count with a bogus 0.
            return
        nodes = [n for n in result.stdout.strip().split('\n') if n]
        if not nodes:
            # An empty graph is never true (this node itself is always in it);
            # treat as a discovery race and keep the previous value.
            return
        self._cached_ros_nodes = len(nodes)

    def _bms_callback(self, msg):
        self.battery_soc = msg.soc
        self.battery_soh = msg.soh
        self.battery_current = msg.current / 1000.0  # mA to A
        temps = [t for t in msg.temperature if t > 0]
        self.battery_temp = max(temps) if temps else None
        self.battery_voltage = msg.bmsvoltage[0] / \
            1000.0 if len(msg.bmsvoltage) > 0 else None
        self.battery_cycles = msg.cycle

    def _sport_callback(self, msg):
        if msg.imu_state:
            rpy = msg.imu_state.rpy
            self.imu_rpy = (
                round(rpy[0] * 57.3, 1),  # rad to deg
                round(rpy[1] * 57.3, 1),
                round(rpy[2] * 57.3, 1)
            )
        self.mode = msg.mode

    def _get_system_info(self):
        """Collect system info: IPs, uptime."""
        info = {}

        # Network IPs
        try:
            info['eth_ip'] = '192.168.123.164'
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            info['wifi_ip'] = s.getsockname()[0]
            s.close()
        except Exception:
            info['wifi_ip'] = 'not connected'

        # Uptime
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            info['uptime'] = f'{hours}h {minutes}m'
        except Exception:
            info['uptime'] = 'unknown'

        # ROS2 nodes (use the background-cached count; never block the turn
        # here)
        info['ros_nodes'] = self._cached_ros_nodes

        return info

    def _build_state_context(self):
        """Build a concise state string to prepend to LLM prompts."""
        lines = ['[ROBOT STATE]']

        # Battery
        if self.battery_soc is not None:
            charging = self.battery_current and self.battery_current > 0
            status = 'charging' if charging else 'discharging'
            lines.append(
                f'Battery: {self.battery_soc}% (health {self.battery_soh}%, '
                f'{abs(self.battery_current):.1f}A {status}, '
                f'temp {self.battery_temp}°C, {self.battery_cycles} cycles)'
            )
        else:
            lines.append('Battery: unknown')

        # IMU
        if self.imu_rpy:
            r, p, y = self.imu_rpy
            lines.append(f'Orientation: roll={r}° pitch={p}° yaw={y}°')

        # Mode
        if self.mode is not None:
            mode_names = {
                0: 'idle',
                1: 'balancing',
                2: 'walking',
                3: 'running'}
            mode_str = mode_names.get(self.mode, f'mode {self.mode}')
            lines.append(f'Motion mode: {mode_str}')

        # System
        sys_info = self._get_system_info()
        lines.append(
            f'Network: eth {sys_info["eth_ip"]}, wifi {sys_info["wifi_ip"]}')
        lines.append(f'Uptime: {sys_info["uptime"]}')
        lines.append(f'Active ROS2 nodes: {sys_info["ros_nodes"]}')
        lines.append(f'LLM: Ollama LLaMA 3.2 3B (local)')
        lines.append(f'STT: faster-whisper base')
        lines.append(f'TTS: Piper en_US-lessac-medium (Aletta voice)')
        lines.append('[/ROBOT STATE]')

        return '\n'.join(lines)

    def _transcript_callback(self, msg: String):
        """Intercept transcript, enrich with robot state, forward to LLM."""
        raw_text = msg.data.strip()
        language_code = ''
        transcript = raw_text

        # Read language metadata from STT while preserving the spoken text.
        first_line, separator, remaining = raw_text.partition('\n')
        if separator and first_line.startswith(
                '[USER_LANGUAGE:') and first_line.endswith(']'):
            language_code = first_line[len('[USER_LANGUAGE:'):-1].lower()
            transcript = remaining.strip()
        if not transcript:
            return

        state_context = self._build_state_context()
        if language_code == 'pl':
            language_name = 'Polish'
        elif language_code == 'en':
            language_name = 'English'
        else:
            language_name = ''

        if language_name:
            language_instruction = (
                f"LATEST USER TURN LANGUAGE: {language_name} ({language_code}). "
                f"Reply in {language_name} for this turn, regardless of earlier "
                "conversation turns. This instruction overrides the language "
                "used in previous turns, unless the user explicitly requests "
                "a different language.")
            enriched = (
                f"{state_context}\n\n{language_instruction}"
                f"\n\nUser: {transcript}"
            )
        else:
            enriched = f"{state_context}\n\nUser: {transcript}"

        out = String()
        out.data = enriched
        self.prompt_pub.publish(out)

        log_info(self, f'Forwarded enriched prompt to LLM')

    def _publish_state_summary(self):
        """Publish robot state summary every 30s."""
        state = self._build_state_context()
        msg = String()
        msg.data = state
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
