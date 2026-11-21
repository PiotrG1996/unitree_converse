"""
Robot state node: adds live robot state to each STT transcript.

The enriched prompt goes to the LLM on ``llm_prompt``. The state block is
also published on /g1/robot_state every 30 s for diagnostics.

When the user asks about the software (nodes, topics, settings, ...; see
``logic.asks_about_system``) the block also summarizes the ROS graph, the
assistant mode, key settings, and every parameter of the voice nodes whose
name matches the question (``logic.match_parameters``): all parameters of
all pipeline nodes are listed and read via their ROS parameter services.
This is strictly read-only: the graph comes from local discovery, the
parameters from each node's ``list_parameters`` / ``get_parameters``
services (refreshed every 30 s in the background); nothing is ever set,
called with side effects, or published to another node's topics.
"""

import threading
import time

import rclpy
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from g1_voice.logic import (ROBOT_IP, VOICE_NODES, asks_about_system, format_local_time,
                            match_parameters,
                            format_parameters, interface_for_ip, interface_ip,
                            parameter_value, split_language, summarize_graph)

MOTION_MODES = {0: 'idle', 1: 'balancing', 2: 'walking', 3: 'running'}
LANGUAGE_NAMES = {'en': 'English', 'pl': 'Polish'}


def uptime():
    with open('/proc/uptime') as f:
        seconds = int(float(f.read().split()[0]))
    return '{}h {}m'.format(seconds // 3600, seconds % 3600 // 60)


class RobotStateNode(Node):

    def __init__(self):
        super().__init__('robot_state_node')
        self.battery = None
        self.rpy = None
        self.mode = None

        try:
            from unitree_hg.msg import BmsState
            # BmsState lives in unitree_hg and is published RELIABLE.
            self.create_subscription(
                BmsState, '/lf/bmsstate', self._on_bms,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        except ImportError as exc:
            self.get_logger().warn('No battery state: {}'.format(exc))
        try:
            from unitree_go.msg import SportModeState
            self.create_subscription(
                SportModeState, '/odommodestate', self._on_sport, 10)
        except ImportError as exc:
            self.get_logger().warn('No IMU state: {}'.format(exc))

        self.create_subscription(String, '/g1/stt/transcript', self._on_transcript, 10)
        self.prompt_pub = self.create_publisher(String, 'llm_prompt', 10)
        self.state_pub = self.create_publisher(String, '/g1/robot_state', 10)
        self.create_timer(
            30.0, lambda: self.state_pub.publish(String(data=self.state_block())))

        # Read-only cache of ALL parameters of the voice nodes:
        # {(node, param): value}, refreshed every 30 s.
        self._params = {}
        self._param_clients = {}
        for node in VOICE_NODES:
            if node == self.get_name():
                continue  # our own parameters: nothing useful, and no self-calls
            self._param_clients[node] = (
                self.create_client(ListParameters, '/{}/list_parameters'.format(node)),
                self.create_client(GetParameters, '/{}/get_parameters'.format(node)))
        self.assistant_mode = None
        self.create_subscription(
            String, '/g1/voice/assistant_mode', self._on_assistant_mode,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(30.0, self._refresh_parameters)
        self.create_timer(3.0, self._first_refresh)

        # Load the LLM now: after a reboot Ollama loads the model on the first
        # question, which then took 5.4 s to its first token (later: 0.3-0.6 s).
        warmup_url = self.declare_parameter(
            'llm_warmup_url', 'http://localhost:11434/api/generate').value
        warmup_model = self.declare_parameter('llm_warmup_model', 'llama3.2').value
        if warmup_url:
            threading.Thread(target=self._warm_llm, args=(warmup_url, warmup_model),
                             daemon=True).start()

    def _warm_llm(self, url, model):
        """Ask Ollama to load ``model`` (empty prompt) so the first reply is fast."""
        import json
        import urllib.request
        time.sleep(2.0)  # let the latched assistant mode arrive first
        if self.assistant_mode == 'unitree':
            return  # Unitree's GPT answers; assistant_mode_node loads it when needed
        start = time.monotonic()
        body = json.dumps({'model': model, 'keep_alive': -1}).encode()
        for _ in range(10):  # Ollama may still be starting
            try:
                req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=60).read()
                self.get_logger().info('LLM {} loaded in {:.1f}s'.format(
                    model, time.monotonic() - start))
                return
            except Exception as exc:  # noqa: BLE001 - only a speed-up
                last = exc
                time.sleep(3.0)
        self.get_logger().warn('LLM warm-up failed: {}'.format(last))

    def _first_refresh(self):
        if not self._params:
            self._refresh_parameters()

    def _on_assistant_mode(self, msg):
        self.assistant_mode = msg.data

    def _refresh_parameters(self):
        """List, then read, every parameter of each voice node (async, never blocks)."""
        for node, (lister, getter) in self._param_clients.items():
            if not lister.service_is_ready() or not getter.service_is_ready():
                continue
            future = lister.call_async(ListParameters.Request())
            future.add_done_callback(
                lambda f, node=node, getter=getter: self._on_list(node, getter, f))

    def _on_list(self, node, getter, future):
        try:
            names = list(future.result().result.names)
        except Exception as exc:  # noqa: BLE001 - the summary just omits them
            self.get_logger().debug('list_parameters on {} failed: {}'.format(node, exc))
            return
        names = [n for n in names if n != 'use_sim_time']
        if names:
            done = getter.call_async(GetParameters.Request(names=names))
            done.add_done_callback(
                lambda f, node=node, names=names: self._on_parameters(node, names, f))

    def _on_parameters(self, node, names, future):
        try:
            values = future.result().values
        except Exception as exc:  # noqa: BLE001 - the summary just omits them
            self.get_logger().debug('get_parameters on {} failed: {}'.format(node, exc))
            return
        for name, value in zip(names, values):
            self._params[(node, name)] = parameter_value(value)

    def system_lines(self, text=''):
        """ROS graph, mode, settings and the parameters a question names (read-only)."""
        nodes = [name for name, _ in self.get_node_names_and_namespaces()]
        topics = [name for name, _ in self.get_topic_names_and_types()]
        lines = summarize_graph(nodes, topics)
        if self.assistant_mode:
            lines.append('Assistant mode: {} (Unitree GPT answers by default while the '
                         'robot is online; saying "Aletta" switches to Aletta)'.format(
                             self.assistant_mode))
        settings = format_parameters(self._params)
        if settings:
            lines.append(settings)
        matched = match_parameters(self._params, text)
        if matched:
            lines.append('Parameters named in the question (read-only): ' + '; '.join(matched))
        lines.append('Parameters readable: {} on {} nodes. They can be read, not changed, '
                     'by voice.'.format(len(self._params),
                                        len({n for n, _ in self._params})))
        return lines

    def _on_bms(self, msg):
        # Only what is useful in conversation: a 3B model otherwise volunteers
        # currents, temperatures and cycle counts nobody asked about.
        self.battery = 'Battery: {}%, {}, health {}%'.format(
            msg.soc, 'charging' if msg.current > 0 else 'not charging', msg.soh)

    def _on_sport(self, msg):
        self.rpy = [round(v * 57.2958, 1) for v in msg.imu_state.rpy]
        self.mode = msg.mode

    def state_block(self, system=False, text=''):
        lines = ['[ROBOT STATE]', self.battery or 'Battery: unknown']
        if self.rpy:
            lines.append('Orientation: roll={} deg pitch={} deg yaw={} deg'.format(*self.rpy))
        if self.mode is not None:
            lines.append('Motion mode: {}'.format(
                MOTION_MODES.get(self.mode, self.mode)))
        lines += [
            'Network: robot bus {} ({}), wifi {}'.format(
                ROBOT_IP, interface_for_ip() or 'DOWN',
                interface_ip('wlan0') or 'not connected'),
            'Local time: {}'.format(format_local_time()),
            'Uptime: {}'.format(uptime()),
            'Software: local LLM (Ollama), faster-whisper, Piper TTS (all on-robot)',
        ]
        if system:
            try:
                lines += self.system_lines(text)
            except Exception as exc:  # noqa: BLE001 - never lose the turn
                self.get_logger().warn('System summary failed: {}'.format(exc))
        lines.append('[/ROBOT STATE]')
        return '\n'.join(lines)

    def _on_transcript(self, msg):
        language, text = split_language(msg.data)
        if not text:
            return
        parts = [self.state_block(system=asks_about_system(text), text=text)]
        if language in LANGUAGE_NAMES:
            name = LANGUAGE_NAMES[language]
            parts.append(
                'LATEST USER TURN LANGUAGE: {0} ({1}). Reply in {0} for this '
                'turn, unless the user explicitly asks for another '
                'language.'.format(name, language))
        parts.append('User: ' + text)
        self.prompt_pub.publish(String(data='\n\n'.join(parts)))


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
