#!/usr/bin/env bash
# wait_for_robot_ready.sh — ExecStartPre gate for unitree_converse.service.
#
# WHY THIS EXISTS (act-mode fix):
#   The pipeline was dying at boot with every node reporting
#       _rclpy.RCLError: rcl node's rmw handle is invalid
#   Root cause: ros2 launch was started before eth0 had its 192.168.123.164
#   address AND before CycloneDDS finished bringing its transport up, so
#   rmw_cyclonedds_cpp could not create the domain/node. A flat `sleep 5` only
#   papers over this; here we *wait for the actual preconditions* so the launch
#   starts cleanly the first time.
#
# Preconditions checked (all must pass):
#   1. The CycloneDDS config file referenced by CYCLONEDDS_URI exists.
#   2. The robot bus interface (default eth0) exists and has an IPv4 address.
#   3. (Best effort) the loopback + eth0 are up so DDS multicast can be sent.
#
# Always exits 0. This is a *readiness gate*, not a hard failure: if something
# never comes up we still let the launch try (Restart=always will retry the whole
# unit), rather than blocking the boot sequence forever.

set -u

IFACE="${UNITREE_DDS_IFACE:-eth0}"
URI="${CYCLONEDDS_URI:-$HOME/cyclonedds_ws/cyclonedds.xml}"
TIMEOUT="${1:-30}"   # max seconds to wait (passed by the service, default 30)

log() { echo "wait_for_robot_ready: $*"; }

wait_for() {
    # wait_for <seconds> <description> <command...>
    local secs="$1"; shift
    local desc="$1"; shift
    local i
    for ((i = 0; i < secs; i++)); do
        if "$@" >/dev/null 2>&1; then
            log "$desc OK"
            return 0
        fi
        sleep 1
    done
    return 1
}

log "waiting up to ${TIMEOUT}s — iface=$IFACE uri=$URI"

# 1) CycloneDDS config must exist (service sets CYCLONEDDS_URI to an absolute path).
if [ ! -f "$URI" ]; then
    log "WARNING: CYCLONEDDS_URI file not found: $URI (continuing anyway)"
else
    log "CYCLONEDDS config present: $URI"
fi

# 2) Interface exists and has an IPv4 address.
has_ip() {
    ip -4 -o addr show dev "$IFACE" 2>/dev/null | grep -q 'inet '
}
if wait_for "$TIMEOUT" "iface $IFACE has IPv4" has_ip; then
    ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print "  ->", $2, $4}'
else
    log "WARNING: iface $IFACE has no IPv4 after ${TIMEOUT}s (continuing anyway)"
fi

# 3) Best-effort: link is up (carrier present).
if ip -br link show "$IFACE" 2>/dev/null | grep -q 'UP'; then
    log "link $IFACE is UP"
fi

log "readiness gate done — launching pipeline"
exit 0
