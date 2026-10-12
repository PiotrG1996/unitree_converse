#!/usr/bin/env bash
# ensure_pulse.sh — make sure the user's PulseAudio session (which drives the G1
# onboard speaker via the pulseaudio 'platform-sound.analog-stereo' sink) is up
# before the voice pipeline starts.
#
# Run from unitree_converse.service as User=unitree (ExecStartPre). It is always
# non-fatal: if PulseAudio cannot be brought up, we exit 0 and let tts_node fall
# back to the ALSA 'default' device.
set -u

RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SOCK="$RUNTIME/pulse/native"
export XDG_RUNTIME_DIR="$RUNTIME"
export PULSE_SERVER="unix:$SOCK"
# Best-effort DBus session address so `systemctl --user` can reach the user manager.
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$RUNTIME/bus}"

wait_sock() {
    # Wait up to $1 seconds for the PA socket to accept connections.
    local tries="${1:-15}"
    for _ in $(seq 1 "$tries"); do
        if [ -S "$SOCK" ] && pactl info >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# 1) Already available? Nothing to do.
if wait_sock 3; then
    echo "ensure_pulse: PulseAudio already available at $SOCK"
    exit 0
fi

# 2) Try the user manager (works when lingering/login session is present).
echo "ensure_pulse: starting PulseAudio via user manager..."
systemctl --user start pulseaudio.service pulseaudio.socket >/dev/null 2>&1 || true
if wait_sock 10; then
    echo "ensure_pulse: PulseAudio started via user manager"
    exit 0
fi

# 3) Last resort: start the daemon directly (survives even without a session).
echo "ensure_pulse: user manager start unavailable; trying 'pulseaudio --start'..."
pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1 || true
if wait_sock 10; then
    echo "ensure_pulse: PulseAudio started via 'pulseaudio --start'"
    exit 0
fi

# Non-fatal: tts_node will fall back to ALSA.
echo "ensure_pulse: WARNING — PulseAudio not available; TTS will use ALSA fallback"
exit 0
