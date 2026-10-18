#!/usr/bin/env python3
"""Separate-process worker for native Unitree G1 speaker playback."""

import json
import sys
import time
import traceback


def send_message(message):
    print(json.dumps(message), flush=True)


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

        ChannelFactoryInitialize(0, interface)
        client = AudioClient()
        client.SetTimeout(10.0)
        client.Init()
        send_message({"ready": True})
    except Exception as exc:
        send_message({"ready": False, "error": str(exc)})
        return 1

    for line in sys.stdin:
        try:
            command = json.loads(line)

            if command.get("command") == "quit":
                break

            if command.get("command") != "play":
                send_message({"ok": False, "error": "Unknown command"})
                continue

            path = command["path"]
            app_name = "unitree_converse"
            # Monotonic per-process counter: wall-clock ms collided for
            # back-to-back sentences (same ms + same seq within one
            # process restart), mixing two playbacks in one stream handle.
            # time_ns() alone can still collide across rapid calls, so the
            # process counter guarantees uniqueness for every play command.
            if not hasattr(main, "_stream_seq"):
                main._stream_seq = 0
            main._stream_seq += 1
            stream_id = f"{time.time_ns()}-{main._stream_seq}"

            with open(path, "rb") as audio_file:
                pcm_data = audio_file.read()

            if not pcm_data:
                send_message({"ok": False, "error": "Empty PCM audio"})
                continue

            # 16 kHz, mono, signed 16-bit PCM: 32,000 bytes per second.
            chunk_size = 32000

            try:
                for offset in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[offset:offset + chunk_size]
                    t0 = time.monotonic()
                    ret_code, _ = client.PlayStream(
                        app_name, stream_id, chunk
                    )
                    send_time = time.monotonic() - t0

                    if ret_code != 0:
                        raise RuntimeError(
                            f"PlayStream returned code {ret_code}"
                        )

                    # Pace to realtime: PlayStream already consumed
                    # send_time, so only sleep the remainder. The old code
                    # slept the full chunk duration AFTER the blocking
                    # call, running slower than realtime with gaps.
                    sleep_time = max(
                        0.0, len(chunk) / 32000.0 - send_time)
                    print(
                        f"Audio chunk: {len(chunk) / 32000.0:.2f}s; "
                        f"PlayStream call: {send_time:.2f}s; "
                        f"pacing sleep: {sleep_time:.2f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(sleep_time)

                client.PlayStop(app_name)
                send_message({"ok": True})
            except Exception:
                try:
                    client.PlayStop(app_name)
                except Exception:
                    pass
                raise

        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send_message({"ok": False, "error": str(exc)})

    try:
        client.PlayStop("unitree_converse")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
