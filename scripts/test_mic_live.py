#!/usr/bin/env python3
"""
test_mic_live.py — interactive live microphone test for the G1.

Captures the RockChip UDP mic stream (the exact path stt_node uses) while YOU
speak, then reports:
  * peak / RMS level per second (visual proof speech reaches the mic)
  * whether speech cleared stt_node's silence_threshold (0.008)
  * a Whisper transcript of what it heard

Run it and SPEAK as soon as you see the prompt:

    python3 scripts/test_mic_live.py
    python3 scripts/test_mic_live.py --seconds 8   # longer window

PASS means the mic delivers usable speech to the STT stage. FAIL with a high
peak but empty transcript points at Whisper; FAIL with a low peak points at a
muted/too-quiet mic (check the RockChip mic gain / hardware mute).
"""
import argparse
import socket
import struct
import sys
import time

import numpy as np

GROUP = "239.168.123.161"
PORT = 5555
LOCAL = "192.168.123.164"
SR = 16000
SILENCE_THRESHOLD = 0.008  # keep in sync with voice_params_real.yaml


def capture(seconds):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", PORT))
    mreq = struct.pack(
        "4s4s", socket.inet_aton(GROUP), socket.inet_aton(LOCAL))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.settimeout(0.5)

    frames = []
    npkts = 0
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < seconds:
        try:
            d, _ = s.recvfrom(65535)
            frames.append(d)
            npkts += 1
        except socket.timeout:
            d = b""
        elapsed = time.time() - t0
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(f"  ... {elapsed:4.1f}s elapsed", flush=True)
    s.close()
    raw = b"".join(frames)
    a = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype=np.int16)
    return a.astype(np.float32) / 32768.0, npkts


def main():
    ap = argparse.ArgumentParser(description="Live G1 microphone test")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument(
        "--language", choices=("auto", "en", "pl"), default="auto",
        help="Whisper language for the test (default: auto)")
    ap.add_argument(
        "--initial-prompt",
        default="Hey robot. Cześć robot. Czesc robot.",
        help="Optional Whisper vocabulary hint")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("  G1 LIVE MICROPHONE TEST", flush=True)
    print(f"  Capturing {args.seconds:.0f}s from {GROUP}:{PORT}", flush=True)
    print("  >>> SPEAK NOW, CLEARLY, INTO THE ROBOT <<<", flush=True)
    print("=" * 60, flush=True)

    # Countdown so the operator is ready.
    for i in (3, 2, 1):
        print(f"  starting in {i}...", flush=True)
        time.sleep(1)

    audio, npkts = capture(args.seconds)

    if audio.size == 0:
        print("\n[FAIL] No UDP packets received — mic stream is dead.")
        print("  Is the RockChip MCU streaming? Are you on the robot?")
        return 2

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    # Fraction of 1s windows whose RMS clears the speech threshold.
    win = SR
    voiced = 0
    total = 0
    print("\nPer-second level (vs silence_threshold=0.008):")
    for i in range(0, audio.size, win):
        chunk = audio[i:i + win]
        if chunk.size == 0:
            continue
        r = float(np.sqrt(np.mean(chunk ** 2)))
        total += 1
        cleared = r >= SILENCE_THRESHOLD
        if cleared:
            voiced += 1
        bar = "#" * min(40, int(r * 200))
        flag = "SPEECH" if cleared else "quiet"
        print(f"  t={i//win}s RMS={r:.4f} {bar} [{flag}]")

    print(f"\n[LEVEL] packets={npkts} peak={peak:.4f} overall_rms={rms:.4f}")
    print(f"[LEVEL] {voiced}/{total} one-second windows cleared "
          f"threshold {SILENCE_THRESHOLD}")

    cleared_threshold = voiced >= 1 and peak >= SILENCE_THRESHOLD

    # Whisper on the captured speech.
    transcript = ""
    try:
        from faster_whisper import WhisperModel
        print("\n[STT] running Whisper 'base' ...", flush=True)
        m = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(
            audio, language=None if args.language == "auto" else args.language,
            beam_size=5, initial_prompt=args.initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500))
        parts = []
        print("[STT] per-segment confidence "
              "(noise hallucinations measured at no_speech_prob 0.53-0.59):")
        for x in segs:
            print(f"    no_speech_prob={x.no_speech_prob:.3f} "
                  f"avg_logprob={x.avg_logprob:.3f}  {x.text!r}")
            parts.append(x.text.strip())
        transcript = " ".join(parts).strip()
    except Exception as e:
        print(f"[STT] whisper failed: {e}")

    print(f"[STT] transcript: {transcript!r}")

    print("\n" + "=" * 60)
    if not cleared_threshold:
        print("FAIL — speech never cleared the level threshold.")
        print("  Likely cause: mic muted / gain too low / not near the mic.")
        print("  Check the RockChip mic hardware mute and mic volume.")
        rc = 1
    elif not transcript:
        print("FAIL — mic captured speech-level audio but Whisper got nothing")
        print("  Level OK; inspect STT (model/language) for the problem.")
        rc = 1
    else:
        print("PASS — mic delivers speech and it transcribes.")
        rc = 0
    print("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
