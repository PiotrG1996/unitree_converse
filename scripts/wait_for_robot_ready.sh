#!/bin/bash
# ExecStartPre gate: wait (up to $1 s, default 30) until some interface holds
# the robot IP and the G1 audio/locomotion board answers. Anchored to the IP,
# not to "eth0": interface names on this Jetson change when a USB Ethernet
# dongle is plugged in. Never fails the unit; it only logs what is missing.
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
BOARD_IP="${BOARD_IP:-192.168.123.161}"
TIMEOUT="${1:-30}"

iface_with_ip() {
    ip -4 -o addr show | awk -v ip="$ROBOT_IP" '$4 ~ "^"ip"/" {print $2; exit}'
}

for ((i = 0; i < TIMEOUT; i++)); do
    IFACE="$(iface_with_ip)"
    if [ -n "$IFACE" ] && ping -c 1 -W 1 -I "$IFACE" "$BOARD_IP" >/dev/null 2>&1; then
        echo "wait_for_robot_ready: $ROBOT_IP on $IFACE, board $BOARD_IP reachable"
        exit 0
    fi
    sleep 1
done

IFACE="$(iface_with_ip)"
if [ -z "$IFACE" ]; then
    echo "wait_for_robot_ready: WARNING no interface has $ROBOT_IP (NetworkManager profile 'unitree1')"
else
    echo "wait_for_robot_ready: WARNING $BOARD_IP not reachable via $IFACE:" \
         "robot off, or the G1 cable is on the other Ethernet port"
fi
exit 0
