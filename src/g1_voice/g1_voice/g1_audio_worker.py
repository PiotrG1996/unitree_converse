#!/usr/bin/env python3
"""
Subprocess that plays PCM on the G1 speaker via the Unitree AudioHub API.

Kept out of the ROS process because unitree_sdk2py brings its own DDS
participant. Protocol (stdin/stdout, one JSON line each):

  -> {"ready": true, "native_audio": bool}          on start
  <- {"command": "play", "length": N} + N bytes of 16 kHz mono s16le PCM
  -> {"ok": true} | {"ok": false, "error": "..."}  after playback finished

Without unitree_sdk2py (dev machine) it reports native_audio=false and the
caller plays audio locally instead.
"""

import json
import os
import sys
import time

APP_NAME = 'unitree_converse'
BYTES_PER_SECOND = 32000  # 16 kHz * 2 bytes

# unitree_sdk2py (and CycloneDDS) print diagnostics to stdout, which would
# corrupt the JSON replies. Keep a private copy of stdout for the protocol and
# point fd 1 (Python and C-level prints) at stderr.
_PROTOCOL = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(message):
    _PROTOCOL.write(json.dumps(message) + '\n')
    _PROTOCOL.flush()


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else 'eth0'
    client = None
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    except ImportError:
        pass
    else:
        try:
            ChannelFactoryInitialize(0, interface)
            client = AudioClient()
            client.SetTimeout(10.0)
            client.Init()
        except Exception as exc:  # noqa: BLE001
            send({'ready': False, 'error': str(exc)})
            return 1
    send({'ready': True, 'native_audio': client is not None})

    stdin = sys.stdin.buffer
    seq = 0
    for line in stdin:
        command = json.loads(line)
        if command.get('command') != 'play':
            break
        pcm = stdin.read(int(command['length']))
        if client is None:
            send({'ok': True})
            continue
        seq += 1
        stream_id = '{}-{}'.format(time.time_ns(), seq)  # unique per utterance
        try:
            for offset in range(0, len(pcm), BYTES_PER_SECOND):
                chunk = pcm[offset:offset + BYTES_PER_SECOND]
                start = time.monotonic()
                code, _ = client.PlayStream(APP_NAME, stream_id, chunk)
                if code != 0:
                    raise RuntimeError('PlayStream returned {}'.format(code))
                # Pace to real time; PlayStream itself took part of it.
                time.sleep(max(0.0, len(chunk) / BYTES_PER_SECOND
                               - (time.monotonic() - start)))
            send({'ok': True})
        except Exception as exc:  # noqa: BLE001
            send({'ok': False, 'error': str(exc)})
        finally:
            client.PlayStop(APP_NAME)
    return 0


if __name__ == '__main__':
    sys.exit(main())
