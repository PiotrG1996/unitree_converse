"""
Assistant mode node: Aletta or Unitree's built-in GPT assistant.

The G1 runs its own cloud voice assistant (robot service ``chat_go``, male
English voice, plays as app "gpt"). It listens to the same microphone, so
with both active a question gets two answers. This node decides who answers
(``logic.AssistantArbiter``, parameter ``unitree_assistant``):

* ``off`` (default): Aletta only; ``chat_go`` is switched off and kept off.
* ``auto``: the official Unitree GPT while the robot reports Wi-Fi/internet
  (``/public_network_status``), Aletta on the Jetson when offline. Online,
  the Jetson's Ollama model is unloaded (``unload_llm_online``) and loaded
  again when going offline; Whisper stays loaded for an instant switch. Unitree's assistant
  then does everything (its ASR, cloud GPT, its English voice through the
  robot's audio_player_service); Aletta stays silent.
* ``leave``: Aletta only; ``chat_go`` is never touched.

The mode is published latched on /g1/voice/assistant_mode (``aletta`` or
``unitree``); wake_word_node and stt_node stand down while it is
``unitree``, except that "Aletta" opens an Aletta session, which switches
the mode to ``aletta`` and chat_go off at once (until the session closes).
The service is switched with the robot_state service API
(``ServiceSwitch``, api 1001) sent as plain ROS messages, so no Unitree SDK
(and no second DDS participant) is loaded in this process. Whether chat_go
runs is read from /gpt_state, which it publishes every ~5 s.
"""

import json
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from unitree_api.msg import Request, Response

from g1_voice.logic import AssistantArbiter, gpt_state, network_online

SERVICE_SWITCH = 1001        # robot_state API: {"name": ..., "switch": 0|1}
SWITCH_RETRY_SEC = 20.0      # a switch takes up to ~3 s; /gpt_state lags ~7 s
SERVICE_PROTECTED = 5        # robot_state "status" for a protected service
GPT_SPEECH = 1018            # gpt API: one recorded question (MP3, base64) for chat_go


class AssistantModeNode(Node):

    def __init__(self):
        super().__init__('assistant_mode_node')
        p = self._declare
        policy = p('unitree_assistant', 'off')
        # YAML reads an unquoted off/on as a boolean.
        if isinstance(policy, bool):
            policy = 'auto' if policy else 'off'
        self.service = p('gpt_service_name', 'chat_go')
        self.arbiter = AssistantArbiter(
            policy=policy, online_stable=p('online_stable_sec', 10.0),
            network_timeout=p('network_timeout_sec', 30.0),
            gpt_timeout=p('gpt_state_timeout_sec', 12.0))
        # Free the Jetson while Unitree's GPT answers: unload the Ollama model
        # online (~2-3 GB), load it again when Aletta is needed. '' = off.
        self.ollama_url = p('ollama_url', 'http://localhost:11434/api/generate')
        self.ollama_model = p('ollama_model', 'llama3.2')
        self.unload_llm_online = p('unload_llm_online', True)
        self._pending = None       # (request id, wanted state)
        self._last_switch = -1e9
        self._last_target = None
        self._mode = None
        self._chat_state = None

        best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, '/g1/voice/assistant_mode', latched)
        # F1 switching (online only): button_trigger_node sends a toggle when
        # /g1/voice/assistant_online is true, else F1 stays push-to-talk.
        self.online_pub = self.create_publisher(Bool, '/g1/voice/assistant_online', latched)
        self.say_pub = self.create_publisher(String, '/g1/voice/say_text', 10)
        self.session_pub = self.create_publisher(Bool, '/g1/voice/session_request', 10)
        self.stop_pub = self.create_publisher(Bool, '/g1/voice/continuous_stop', 10)
        self.create_subscription(Bool, '/g1/voice/assistant_toggle', self._on_toggle, 10)
        self._online = None
        self._switching_until = 0.0  # ignore F1 while a switch is loading
        self._announce_ready = False
        self.request_pub = self.create_publisher(Request, '/api/robot_state/request', 10)
        self.create_subscription(Response, '/api/robot_state/response', self._on_response, best)
        self.create_subscription(String, '/public_network_status', self._on_network, best)
        self.create_subscription(String, '/gpt_state', self._on_gpt, best)
        self.create_subscription(String, '/g1/wake/session_state', self._on_session, 10)
        # What chat_go says and hears, logged in Unitree mode so the journal
        # shows both conversations.
        self.create_subscription(String, '/gpt_msg', self._on_gpt_msg, best)
        # chat_go's real activity: a Unitree process sends each recorded
        # question as speech (api 1018), then chat_go plays its answer as app
        # "gpt". Logged so the journal shows when Unitree's GPT is talking.
        self.create_subscription(Request, '/api/gpt/request', self._on_gpt_request, best)
        self.create_timer(p('check_interval_sec', 2.0), self._tick)
        self.get_logger().info('Unitree GPT assistant policy: {} (service {})'.format(
            policy, self.service))
        self._tick()

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    # ---------------------------------------------------------------- inputs
    def _on_network(self, msg):
        online = network_online(msg.data)
        if online is not None:
            self.arbiter.on_network(online)

    def _on_gpt(self, msg):
        state = gpt_state(msg.data)
        self.arbiter.on_gpt_state(state)
        chat = self.arbiter.chat_state
        if chat != self._chat_state:
            if self._chat_state is None:
                self.get_logger().info('Unitree assistant (chat_go) running')
            self._chat_state = chat
            # Not a "voice chat enabled" flag (it answers while false): debug only.
            self.get_logger().debug('chat_go chat_state: {}'.format(chat))

    def _on_gpt_request(self, msg):
        if msg.header.identity.api_id != GPT_SPEECH:
            return
        try:
            data = json.loads(msg.parameter)
        except ValueError:
            return
        if data.get('state') == 'finish':
            self.get_logger().info('Unitree GPT: question received ({:.1f} kB of speech)'.format(
                data.get('totalLength', 0) / 1024.0))

    def _on_gpt_msg(self, msg):
        if self._mode == 'unitree':
            self.get_logger().info('Unitree GPT: {}'.format(msg.data[:300]))

    def _on_toggle(self, msg):
        if not msg.data:
            return
        if time.monotonic() < self._switching_until:
            # Ignored on purpose: a double press must not flip straight back.
            self.say_pub.publish(String(data='Still switching to {}, please wait.'.format(
                'Aletta' if self.arbiter.forced_aletta else 'the Unitree assistant')))
            return
        was_session = self.arbiter.session_open
        target = self.arbiter.toggle()
        if target is None:
            return
        self.get_logger().info('F1: switching to {}'.format(
            'Aletta' if target == 'aletta' else 'Unitree GPT'))
        if target == 'aletta':
            # Unitree's GPT goes off at once; the LLM loads (~5 s), then
            # _set_llm_loaded says "ready" and opens a session.
            self.say_pub.publish(String(data='Switching to Aletta, one moment, please.'))
            self._switching_until = time.monotonic() + 20.0
            self._announce_ready = True
            # Load (or confirm) the LLM even when the mode was already
            # 'aletta' (an open conversation): no mode change would trigger it.
            threading.Thread(target=self._set_llm_loaded, args=(True,), daemon=True).start()
        else:
            if was_session:
                self.stop_pub.publish(Bool(data=True))  # end Aletta's conversation
            # chat_go needs ~3 s to switch on plus a few seconds to start.
            # (Never say its wake phrase here: it would wake on our voice.)
            self.say_pub.publish(String(
                data='Switching to the Unitree assistant. It will be ready in a few seconds.'))
            self._switching_until = time.monotonic() + 10.0
            self._announce_ready = False
        self._tick()

    def _aletta_ready(self):
        """After an F1 switch to Aletta: announce, then open a conversation."""
        if not self._announce_ready or self.arbiter.mode() != 'aletta':
            return
        self._announce_ready = False
        self._switching_until = 0.0
        self.say_pub.publish(String(data='Aletta is ready. How can I help you?'))
        self.session_pub.publish(Bool(data=True))

    def _on_session(self, msg):
        is_open = msg.data == 'open'
        if is_open != self.arbiter.session_open:
            self.arbiter.session_open = is_open
            self._tick()  # hand over now, not at the next 2 s tick

    def _on_response(self, msg):
        if self._pending is None or msg.header.identity.id != self._pending[0]:
            return
        wanted = self._pending[1]
        self._pending = None
        code = msg.header.status.code
        status = None
        try:
            status = json.loads(msg.data).get('status')
        except (ValueError, AttributeError):
            pass
        if code != 0 or status == SERVICE_PROTECTED:
            self.get_logger().error('Switching {} {} failed (code {}, status {})'.format(
                self.service, 'on' if wanted else 'off', code, status))
        else:
            self.get_logger().info('Unitree assistant ({}) switched {}'.format(
                self.service, 'ON' if wanted else 'OFF'))

    # ----------------------------------------------------------------- logic
    def _tick(self):
        mode = self.arbiter.mode()
        online = self.arbiter.policy == 'auto' and self.arbiter.online()
        if online != self._online:
            self._online = online
            self.online_pub.publish(Bool(data=online))
        if mode != self._mode:
            self._mode = mode
            self.mode_pub.publish(String(data=mode))
            reason = ''
            if mode == 'aletta' and self.arbiter.policy == 'auto':
                if self.arbiter.network() == 'offline':
                    reason = ' (robot offline)'
                elif self.arbiter.forced_aletta:
                    reason = ' (F1)'
            self.get_logger().info('Assistant mode: {}{}'.format(
                'Unitree GPT (robot online)' if mode == 'unitree' else 'Aletta', reason))
            if self.unload_llm_online and self.ollama_url and self.arbiter.policy == 'auto':
                threading.Thread(target=self._set_llm_loaded, args=(mode != 'unitree',),
                                 daemon=True).start()
            elif mode == 'aletta':
                self._aletta_ready()  # LLM already loaded
        wanted = self.arbiter.wanted_gpt()
        if wanted is None or wanted == self.arbiter.gpt_running():
            return
        # Give a switch and /gpt_state time to settle before repeating it; a
        # switch the other way (Aletta took over) goes out at once.
        if (wanted == self._last_target
                and time.monotonic() - self._last_switch < SWITCH_RETRY_SEC):
            return
        self._switch(wanted)

    def _set_llm_loaded(self, loaded):
        """Load (keep_alive -1) or unload (keep_alive 0) the Ollama model."""
        body = json.dumps({'model': self.ollama_model,
                           'keep_alive': -1 if loaded else 0}).encode()
        start = time.monotonic()
        try:
            req = urllib.request.Request(self.ollama_url, body,
                                         {'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=90).read()
            self.get_logger().info('LLM {} {} in {:.1f}s'.format(
                self.ollama_model, 'loaded' if loaded else 'unloaded (Jetson memory freed)',
                time.monotonic() - start))
        except Exception as exc:  # noqa: BLE001 - only memory/speed, not function
            self.get_logger().warn('LLM {} failed: {}'.format(
                'load' if loaded else 'unload', exc))
        if loaded:
            self._aletta_ready()  # Ollama also loads on demand if this failed

    def _switch(self, on):
        req = Request()
        req.header.identity.id = time.monotonic_ns()
        req.header.identity.api_id = SERVICE_SWITCH
        req.parameter = json.dumps({'name': self.service, 'switch': int(on)})
        self._pending = (req.header.identity.id, on)
        self._last_switch = time.monotonic()
        self._last_target = on
        self.get_logger().info('Switching Unitree assistant ({}) {}'.format(
            self.service, 'on' if on else 'off'))
        self.request_pub.publish(req)


def main(args=None):
    rclpy.init(args=args)
    node = AssistantModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
