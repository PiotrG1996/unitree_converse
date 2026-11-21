"""
Text-to-speech node: speaks LLM replies with Piper on the G1 speaker.

* Every /g1/stt/transcript opens a pending turn (FIFO). bob_llm answers
  prompts in order, so ``llm_stream`` deltas and the final ``llm_response``
  always belong to the oldest pending turn. Complete sentences are spoken as
  they stream in; the response flushes the rest and ends the turn.
* At the end of a turn in continuous mode (open "Aletta" session) a trigger
  on /g1/voice/trigger starts the next listening turn.
* /g1/voice/say_text speaks exact text (SELECT button); /g1/voice/speak_enable
  mutes/unmutes (F3).
* /g1/tts/status is ``speaking`` while audio plays and ``idle`` once the
  queue is drained.

Synthesis and playback run in separate threads, so the next sentence is
synthesized while the current one plays; the first piece of a reply may end
at a clause (``logic.split_speakable``) so speech starts sooner. Each reply
logs its latency from the transcript to the first token and first audio.
"""

import io
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

from g1_voice.logic import (ROBOT_IP, interface_for_ip, is_polish_text,
                            split_language, split_speakable)

PLAYBACK_RATE = 16000
_TURN_END = object()


class _FirstAudio:
    """Queue marker placed before a turn's first sentence (timing log)."""

    __slots__ = ('turn',)

    def __init__(self, turn):
        self.turn = turn


def resample(audio, src_rate, dst_rate=PLAYBACK_RATE):
    """Resample to the speaker rate with a band-limited polyphase filter."""
    if src_rate == dst_rate or not audio.size:
        return audio.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def wav_to_float(wav_bytes):
    """Decode 16-bit PCM WAV bytes to (mono float32, sample_rate)."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        channels, rate = wf.getnchannels(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class AudioPlayer:
    """
    Plays 16 kHz s16le PCM through the g1_audio_worker subprocess.

    Owned by the playback thread only. Any failure or timeout kills the worker
    so the pipe protocol can never desync; the next play() starts a fresh
    one. A worker without unitree_sdk2py (dev machine) falls back to aplay.
    """

    def __init__(self, interface, logger):
        self.interface = interface  # '' = whichever interface holds ROBOT_IP
        self.logger = logger
        self.proc = None
        self.native = False

    def _readline(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line.decode('utf-8')) if line else None

    def _start(self):
        interface = self.interface or interface_for_ip(ROBOT_IP)
        if not interface:
            raise RuntimeError('no interface has the robot IP {}: is the G1 '
                               'network cable connected?'.format(ROBOT_IP))
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'g1_audio_worker.py')
        self.proc = subprocess.Popen([sys.executable, worker, interface],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        msg = self._readline(timeout=20.0)
        if not msg or not msg.get('ready'):
            self.close()
            raise RuntimeError('audio worker failed to start: {}'.format(
                (msg or {}).get('error', 'no ready message')))
        self.native = bool(msg.get('native_audio'))
        self.logger.info('Audio worker ready on {} (native_audio={})'.format(
            interface, self.native))

    def play(self, pcm):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        if not self.native:
            subprocess.run(['aplay', '-q', '-f', 'S16_LE', '-r', str(PLAYBACK_RATE),
                            '-c', '1', '-'], input=pcm, timeout=60, check=False)
            return
        timeout = 15.0 + 2.0 * len(pcm) / (2.0 * PLAYBACK_RATE)
        try:
            self.proc.stdin.write((json.dumps(
                {'command': 'play', 'length': len(pcm)}) + '\n').encode())
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
            reply = self._readline(timeout)
        except (OSError, ValueError) as exc:
            reply = {'error': str(exc)}
        if not reply or not reply.get('ok'):
            self.close()
            raise RuntimeError('playback failed: {}'.format(
                (reply or {}).get('error', 'timeout')))

    def close(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        self.proc = None


class PiperProcess:
    """
    One long-running Piper binary per voice (``--json-input``).

    Starting Piper loads the ONNX voice (~0.5 s); a resident process
    synthesizes a short sentence in ~0.1-0.2 s. Each request writes a WAV to
    a RAM-disk file; Piper prints the path when it is done. Any failure or
    timeout kills the process and the next call starts a fresh one.
    """

    def __init__(self, binary, model, lib_dir, length_scale):
        self.cmd = [binary, '--model', model, '--json-input',
                    '--length_scale', '{:g}'.format(length_scale)]
        self.env = dict(os.environ)
        self.env['LD_LIBRARY_PATH'] = lib_dir + os.pathsep + self.env.get(
            'LD_LIBRARY_PATH', '')
        self.env.setdefault('ESPEAK_DATA_PATH', os.path.join(lib_dir, 'espeak-ng-data'))
        shm = '/dev/shm' if os.path.isdir('/dev/shm') else None
        self.wav_path = os.path.join(tempfile.mkdtemp(prefix='piper_', dir=shm), 'out.wav')
        self.proc = None

    def synthesize(self, text, timeout=30.0):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=self.env)
        try:
            request = {'text': ' '.join(text.split()), 'output_file': self.wav_path}
            self.proc.stdin.write((json.dumps(request) + '\n').encode('utf-8'))
            self.proc.stdin.flush()
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not ready or not self.proc.stdout.readline():
                raise RuntimeError('piper produced no audio')
            with open(self.wav_path, 'rb') as f:
                return wav_to_float(f.read())
        except Exception:
            self.close()
            raise

    def close(self, remove_files=False):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        if remove_files:
            shutil.rmtree(os.path.dirname(self.wav_path), ignore_errors=True)


class _Turn:
    __slots__ = ('language', 'text', 'spoken', 'streamed', 'created',
                 'first_token', 'marked')

    def __init__(self, language):
        self.language = language
        self.text = ''
        self.spoken = 0
        self.streamed = False
        self.created = time.monotonic()
        self.first_token = None
        self.marked = False


class TTSNode(Node):

    def __init__(self):
        super().__init__('tts_node')
        p = self._declare
        home = os.path.expanduser('~')
        self.voices = {
            'en': p('voice_model_path', home + '/.local/share/piper/'
                    'en_US-lessac-medium.onnx'),
            'pl': p('polish_voice_model_path', home + '/.local/share/piper/'
                    'pl_PL-gosia-medium/pl_PL-gosia-medium.onnx'),
        }
        self.tts_mode = p('tts_mode', 'binary')
        # Allow environment variable override: TTS_MODE=python (forces CPU, no VRAM)
        env_tts_mode = os.environ.get('TTS_MODE')
        if env_tts_mode is not None:
            self.tts_mode = env_tts_mode
            self.get_logger().info(f"TTS mode overridden by env: {self.tts_mode}")

        # Check GPU memory pressure when using binary mode (GPU-accelerated Piper)
        if self.tts_mode == 'binary':
            from g1_voice.logic import check_gpu_pressure, log_gpu_pressure
            is_pressure, free_gb, total_gb = check_gpu_pressure(free_gb_threshold=2.0)
            log_gpu_pressure('tts_node', free_gb, total_gb)
            if is_pressure:
                self.get_logger().warn(
                    f"GPU VRAM under pressure ({free_gb:.1f}/{total_gb:.1f} GB free). "
                    "TTS will use CPU python mode to avoid OOM. Set TTS_MODE=binary to override."
                )
                self.tts_mode = 'python'
        self.piper_binary = p('piper_binary', home + '/.local/piper-bin/piper')
        self.piper_lib_dir = p('piper_lib_dir', home + '/.local/piper-bin')
        self.length_scale = p('length_scale', 0.85)
        self.continuous_mode = p('continuous_mode', False)
        self.turn_timeout = p('turn_timeout_sec', 90.0)
        # Words before a ',' or ':' that may end the first spoken piece (0 = off).
        self.first_clause_words = p('first_clause_words', 5)
        interface = p('network_interface', '')

        for lang, path in self.voices.items():
            if not os.path.isfile(path):
                self.get_logger().error('{} Piper voice missing: {}'.format(lang, path))
        self._python_voices = {}
        self._piper = {}
        self.speak_enabled = True
        self._turns = deque()
        self._queue = queue.Queue()  # text items: (language, text) / markers
        self._audio = queue.Queue()  # synthesized items for the player
        self._pending = 0            # items queued but not yet played
        self._pending_lock = threading.Lock()
        self._speaking = False
        self._player = AudioPlayer(interface, self.get_logger())

        self.status_pub = self.create_publisher(String, '/g1/tts/status', 10)
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)
        self.create_subscription(String, '/g1/stt/transcript', self._on_transcript, 10)
        self.create_subscription(String, 'llm_stream', self._on_stream, 10)
        self.create_subscription(String, 'llm_response', self._on_response, 10)
        self.create_subscription(String, '/g1/voice/say_text', self._on_say, 10)
        self.create_subscription(Bool, '/g1/voice/speak_enable', self._on_speak_enable, 10)
        self.create_subscription(
            Bool, '/g1/voice/continuous_start', self._on_continuous_start, 10)
        self.create_subscription(
            String, '/g1/wake/session_state', self._on_session_state,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            Bool, '/g1/voice/continuous_stop', self._on_continuous_stop, 10)
        threading.Thread(target=self._synth_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()
        self.get_logger().info('TTS ready (mode: {})'.format(self.tts_mode))

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    # ------------------------------------------------------------ turn logic
    def _on_transcript(self, msg):
        language, _ = split_language(msg.data)
        now = time.monotonic()
        while self._turns and now - self._turns[0].created > self.turn_timeout:
            self.get_logger().warn('Dropping a turn whose reply never came')
            self._turns.popleft()
        self._turns.append(_Turn(language or 'en'))

    def _put(self, item):
        with self._pending_lock:
            self._pending += 1
        self._queue.put(item)

    def _split(self, text):
        return split_speakable(text, self.first_clause_words)

    def _emit(self, turn, language, pieces):
        if pieces and turn is not None and not turn.marked:
            turn.marked = True
            self._put(_FirstAudio(turn))
        for piece in pieces:
            self._put((language, piece))

    def _on_stream(self, msg):
        if not self._turns or not msg.data:
            return
        turn = self._turns[0]
        if turn.first_token is None:
            turn.first_token = time.monotonic()
        turn.text += msg.data
        turn.streamed = True
        sentences, _ = self._split(turn.text)
        self._emit(turn, turn.language, sentences[turn.spoken:])
        turn.spoken = len(sentences)

    def _on_response(self, msg):
        try:
            data = json.loads(msg.data)
            reply = data.get('reply', '') if isinstance(data, dict) else msg.data
        except ValueError:
            reply = msg.data
        marker, reply = split_language(str(reply))

        if not self._turns:
            # Prompt that did not come from STT (e.g. manual llm_prompt).
            if reply:
                self._enqueue_text(marker or self._guess_language(reply), reply)
            return

        turn = self._turns.popleft()
        if turn.streamed:
            sentences, rest = self._split(turn.text)
            self._emit(turn, turn.language,
                       sentences[turn.spoken:] + ([rest] if rest else []))
        elif reply:
            self._enqueue_text(marker or turn.language, reply, turn)
        # Even an empty/failed reply ends the turn, so the session keeps going.
        self._put(_TURN_END)

    def _on_say(self, msg):
        marker, text = split_language(msg.data)
        if text:
            self._enqueue_text(marker or self._guess_language(text), text)

    def _enqueue_text(self, language, text, turn=None):
        sentences, rest = self._split(text)
        self._emit(turn, language, sentences + ([rest] if rest else []))

    @staticmethod
    def _guess_language(text):
        return 'pl' if is_polish_text(text) else 'en'

    def _on_speak_enable(self, msg):
        self.speak_enabled = bool(msg.data)
        self.get_logger().info('Speech {}'.format(
            'enabled' if self.speak_enabled else 'MUTED'))

    def _on_continuous_start(self, msg):
        if msg.data:
            self.continuous_mode = True

    def _on_continuous_stop(self, msg):
        if msg.data:
            self.continuous_mode = False

    def _on_session_state(self, msg):
        """Latched session state: catches a session opened before this node started."""
        self.continuous_mode = msg.data == 'open'

    # ----------------------------------------------------------- speech loop
    def _synth_worker(self):
        """Text -> PCM. Runs ahead of playback by one sentence or more."""
        try:
            for language in self.voices:
                self._synthesize('Ready.', language)  # load both voices
        except Exception as exc:  # noqa: BLE001 - first real reply will retry
            self.get_logger().warn('Piper warm-up failed: {}'.format(exc))
        failed = False  # synthesis failed: skip the rest of this reply
        while rclpy.ok():
            item = self._queue.get()
            if item is _TURN_END or isinstance(item, _FirstAudio):
                failed = failed and item is not _TURN_END
                self._audio.put(item)
                continue
            language, text = item
            if not self.speak_enabled or failed:
                self._audio.put(None)  # skipped: keeps the pending count
                continue
            self._set_speaking(True)
            try:
                start = time.monotonic()
                pcm = self._render(text, language)
                self._audio.put((pcm, time.monotonic() - start, language, text))
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.get_logger().error(
                    'TTS failed, skipping the rest of this reply: {}'.format(exc))
                failed = True
                self._audio.put(None)

    def _play_worker(self):
        """PCM -> G1 speaker; owns the AudioPlayer."""
        try:
            self._player.play(b'\x00\x00' * 1600)  # start the worker: 0.1 s silence
            self.get_logger().info('Audio output ready')
        except Exception as exc:  # noqa: BLE001 - first real reply will retry
            self.get_logger().warn('Audio warm-up failed: {}'.format(exc))
        failed = False  # playback failed: skip the rest of this reply
        timing_turn = None
        while rclpy.ok():
            item = self._audio.get()
            if isinstance(item, _FirstAudio):
                timing_turn = item.turn
            elif item is _TURN_END:
                failed = False
            elif item is not None and not failed:
                pcm, synth_time, language, text = item
                self._set_speaking(True)
                if timing_turn is not None:
                    self._log_reply_timing(timing_turn)
                    timing_turn = None
                start = time.monotonic()
                try:
                    self._player.play(pcm)
                    self.get_logger().info('Spoke [{}] {:.2f}s (synth {:.2f}s): "{}"'.format(
                        language, time.monotonic() - start, synth_time, text))
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    self.get_logger().error(
                        'Playback failed, skipping the rest of this reply: {}'.format(exc))
                    failed = True
            self._item_done(item is _TURN_END)

    def _item_done(self, turn_end):
        with self._pending_lock:
            self._pending -= 1
            drained = self._pending <= 0
        if drained:
            self._set_speaking(False)
        if turn_end and self.continuous_mode:
            self.trigger_pub.publish(Bool(data=True))

    def _log_reply_timing(self, turn):
        now = time.monotonic()
        token = ('first token {:.2f}s, '.format(turn.first_token - turn.created)
                 if turn.first_token is not None else '')
        self.get_logger().info('Reply timing: {}first audio {:.2f}s after transcript'.format(
            token, now - turn.created))

    def _set_speaking(self, speaking):
        with self._pending_lock:
            if speaking == self._speaking:
                return
            self._speaking = speaking
        self.status_pub.publish(String(data='speaking' if speaking else 'idle'))

    def _render(self, text, language):
        """Synthesize ``text`` to 16 kHz s16le PCM bytes."""
        audio, rate = self._synthesize(text, language)
        audio = resample(audio, rate)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-4:
            audio = audio * (0.9 / peak)  # Piper peaks at full scale: avoid clipping
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def _synthesize(self, text, language):
        model = self.voices.get(language, self.voices['en'])
        if self.tts_mode == 'python':
            voice = self._python_voices.get(model)
            if voice is None:
                from piper.voice import PiperVoice
                voice = self._python_voices[model] = PiperVoice.load(model)
            chunks = [c.audio_float_array for c in voice.synthesize(text)]
            return np.concatenate(chunks), voice.config.sample_rate

        piper = self._piper.get(model)
        if piper is None:
            piper = self._piper[model] = PiperProcess(
                self.piper_binary, model, self.piper_lib_dir, self.length_scale)
        return piper.synthesize(text)

    def destroy_node(self):
        self._player.close()
        for piper in self._piper.values():
            piper.close(remove_files=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
