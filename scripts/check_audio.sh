#!/usr/bin/env bash
# check_audio.sh — verify the G1 speaker + microphones are working.
#
# WHAT IT CHECKS
#   1. PulseAudio is up and the onboard sink/source are present.
#   2. SPEAKER:  plays a short tone through the same PulseAudio sink the TTS
#      node uses (alsa_output.platform-sound.analog-stereo) and reads it back
#      through the sink's .monitor source to confirm audio actually came out.
#   3. MIC (ALSA/PulseAudio input): records ~3s from the onboard mic source and
#      reports the RMS level so you can confirm the mic captures sound
#      (speak during the capture for a live check).
#   4. MIC (UDP multicast, RockChip): joins 239.168.123.161:5555 for a few
#      seconds and reports how many packets/bytes arrived — this is the path
#      stt_node actually uses on the real robot.
#
# USAGE
#   ./scripts/check_audio.sh              # speaker + mic + UDP checks
#   SKIP_UDP=1 ./scripts/check_audio.sh   # skip the multicast check
#   SKIP_MIC=1 ./scripts/check_audio.sh   # speaker-only (no capture)
#
# Exit code: 0 if every enabled check passed, non-zero otherwise.

set -u

PULSE_RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export XDG_RUNTIME_DIR="$PULSE_RUNTIME"
export PULSE_SERVER="unix:$PULSE_RUNTIME/pulse/native"

SINK="alsa_output.platform-sound.analog-stereo"
SRC="alsa_input.platform-sound.analog-stereo"
MON="${SINK}.monitor"
UDP_GROUP="239.168.123.161"
UDP_PORT="5555"
UDP_LOCAL_IP="192.168.123.164"

SKIP_MIC="${SKIP_MIC:-0}"
SKIP_UDP="${SKIP_UDP:-0}"

PASS=0
FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "  [INFO] $*"; }
hdr()  { echo; echo "== $* =="; }

# RMS of a raw s16le mono capture, via python3 (no numpy CLI dependency).
rms_of_raw() {
    python3 - "$1" <<'PY'
import sys, array
data = open(sys.argv[1], 'rb').read()
if not data:
    print("0.0"); raise SystemExit
a = array.array('h'); a.frombytes(data[:len(data) - (len(data) % 2)])
if not a:
    print("0.0"); raise SystemExit
rms = (sum(x * x for x in a) / len(a)) ** 0.5
print(f"{rms/32768.0:.5f}")
PY
}

hdr "PulseAudio server"
if pactl info >/dev/null 2>&1; then
    ok "PulseAudio reachable ($(pactl info 2>/dev/null | awk -F': ' '/Default Sink/{print $2}'))"
else
    bad "PulseAudio NOT reachable at $PULSE_SERVER (run ensure_pulse.sh / log in)"
    echo "        Skipping speaker + ALSA-mic checks (no PulseAudio)."
    SKIP_MIC=1
fi

# ── 1. SPEAKER ──────────────────────────────────────────────────────────────
hdr "Speaker (PulseAudio sink: $SINK)"
if ! pactl info >/dev/null 2>&1; then
    info "speaker check skipped (no PulseAudio)"
else
    if ! pactl list short sinks 2>/dev/null | grep -q "$SINK"; then
        bad "sink $SINK not present"
    else
        TONE="$(mktemp /tmp/g1_tone.XXXXXX.raw)"
        MON_RAW="$(mktemp /tmp/g1_mon.XXXXXX.raw)"
        # 1000 Hz tone, 0.4s, 44100 Hz mono s16le -> the sink's native format.
        if ! sox -n -r 44100 -c 1 -b 16 -e signed-integer "$TONE" \
                synth 0.4 sine 1000 gain -3 >/dev/null 2>&1; then
            bad "sox failed to generate test tone"
        fi
        # Read the sink monitor in the background while we play. `timeout` (8s)
        # guarantees parecord self-terminates even if our kill misses it.
        timeout 8 parecord -d "$MON" --rate=44100 --channels=1 \
            --format=s16le "$MON_RAW" >/dev/null 2>&1 &
        MON_PID=$!
        sleep 0.3
        # `timeout` on paplay so a stuck sink can't hang the whole check.
        if timeout 8 paplay -d "$SINK" --raw --rate=44100 --channels=1 \
                --format=s16le "$TONE" >/dev/null 2>&1; then
            ok "tone played on $SINK"
        else
            bad "paplay failed or timed out on $SINK"
        fi
        sleep 0.2
        kill "$MON_PID" 2>/dev/null
        # Kill any lingering parecord for this monitor, then reap.
        pkill -f "parecord.*$MON" 2>/dev/null
        wait "$MON_PID" 2>/dev/null
        MON_RMS="$(rms_of_raw "$MON_RAW")"
        # A monitor read-back above ~0.005 RMS means audio truly left the sink.
        if python3 -c "import sys; sys.exit(0 if float('$MON_RMS') > 0.005 else 1)"; then
            ok "speaker loopback detected (monitor RMS=$MON_RMS)"
        else
            bad "no audio on $SINK monitor (RMS=$MON_RMS) — speaker may be muted"
        fi
        rm -f "$TONE" "$MON_RAW"
    fi
fi


# ── 2. MIC (PulseAudio input) ────────────────────────────────────────────────
hdr "Microphone (PulseAudio source: $SRC)"
if [ "$SKIP_MIC" = "1" ]; then
    info "ALSA/PulseAudio mic check skipped (SKIP_MIC=1)"
elif ! pactl info >/dev/null 2>&1; then
    info "ALSA/PulseAudio mic check skipped (no PulseAudio)"
else
    if ! pactl list short sources 2>/dev/null | grep -q "$SRC"; then
        bad "source $SRC not present"
    else
        CAP="$(mktemp /tmp/g1_mic.XXXXXX.raw)"
        echo "  ... speak into the mic now (capturing 3s) ..."
        # `timeout` caps the record: parecord has no built-in duration here and
        # would otherwise block forever. Exit code 124 (killed) is expected.
        timeout 4 parecord -d "$SRC" --rate=16000 --channels=1 --format=s16le \
            "$CAP" >/dev/null 2>&1
        MIC_RMS="$(rms_of_raw "$CAP")"
        # Informational: quiet rooms are legitimately near-silent, so this is a
        # PASS as long as the stream opened and produced samples (non-empty).
        if [ -s "$CAP" ]; then
            ok "mic capture streamed samples (RMS=$MIC_RMS)"
            info "if RMS is ~0 the room was silent; speak during capture to confirm."
        else
            bad "mic capture produced no samples ($SRC)"
        fi
        rm -f "$CAP"
    fi
fi

# ── 3. MIC (UDP multicast — the real stt_node path) ──────────────────────────
hdr "Microphone (UDP multicast: $UDP_GROUP:$UDP_PORT)"
if [ "$SKIP_UDP" = "1" ]; then
    info "UDP multicast mic check skipped (SKIP_UDP=1)"
elif python3 - "$UDP_GROUP" "$UDP_PORT" "$UDP_LOCAL_IP" <<'PY'
import socket, struct, sys, time
group, port, local = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(local))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.settimeout(0.5)
except Exception as e:
    print(f"  [FAIL] could not join multicast group: {e}")
    raise SystemExit(1)
pkts = 0
nbytes = 0
end = time.time() + 3.0
try:
    while time.time() < end:
        try:
            d, _ = s.recvfrom(65535)
            pkts += 1
            nbytes += len(d)
        except socket.timeout:
            continue
except Exception as e:
    print(f"  [FAIL] error receiving: {e}")
    raise SystemExit(1)
if pkts > 0:
    print(f"  [PASS] received {pkts} UDP packets ({nbytes} bytes) from {group}:{port}")
    print("  [INFO] RockChip mic multicast is live — stt_node can capture.")
    raise SystemExit(0)
print(f"  [FAIL] no UDP packets on {group}:{port} in 3s "
      f"(is the RockChip MCU streaming? mic muted? not on robot?)")
raise SystemExit(1)
PY
then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────
hdr "Audio check summary"
echo "  PASS=$PASS  FAIL=$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: OK — speaker and microphones look healthy."
    exit 0
fi
echo "  RESULT: one or more audio checks FAILED (see [FAIL] lines above)."
exit 1

