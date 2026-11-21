#!/bin/bash
# Keeps the robot's Wi-Fi online. Runs as root (wifi-watchdog.service).
#
# Every CHECK_INTERVAL seconds it tests real internet access *through wlan0*
# (ping two anycast resolvers, then an HTTP check). NetworkManager only notices
# a dropped association; the rtl8852bu USB driver can also stay "connected"
# while passing no traffic, which NM never repairs. After FAILS_PER_STEP
# consecutive failures it escalates, one step at a time:
#   1. nmcli device connect   (NM picks the best known network)
#   2. try every known, autoconnect-enabled network in range, by priority
#   3. Wi-Fi radio off/on
#   4. reload the Wi-Fi driver module
# then waits COOLDOWN seconds and starts over. Any success resets everything.
#
# DRY_RUN=1 only prints the actions.

IFACE="${WIFI_IFACE:-wlan0}"
DRIVER="${WIFI_DRIVER:-8852bu}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
FAILS_PER_STEP="${FAILS_PER_STEP:-3}"
COOLDOWN="${COOLDOWN:-120}"
PING_HOSTS="1.1.1.1 8.8.8.8"
HTTP_CHECK="http://connectivity-check.ubuntu.com/"   # answers 204

log() { echo "wifi_watchdog: $*"; }
run() {
    log "action: $*"
    [ -n "$DRY_RUN" ] || "$@"
}

online() {
    ip link show "$IFACE" >/dev/null 2>&1 || return 1
    for host in $PING_HOSTS; do
        ping -I "$IFACE" -c 1 -W 3 "$host" >/dev/null 2>&1 && return 0
    done
    [ "$(curl --interface "$IFACE" -s -o /dev/null -w '%{http_code}' -m 6 "$HTTP_CHECK")" = 204 ]
}

# Known Wi-Fi profiles with autoconnect=yes, highest priority first.
known_profiles() {
    nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY con show |
        awk -F: '$2 == "802-11-wireless" && $3 == "yes" {print $4 ":" $1}' |
        sort -t: -k1,1nr | cut -d: -f2-
}

try_known_networks() {
    nmcli dev wifi rescan ifname "$IFACE" >/dev/null 2>&1
    sleep 5
    local visible
    visible="$(nmcli -t -f SSID dev wifi list ifname "$IFACE" --rescan no 2>/dev/null)"
    while IFS= read -r profile; do
        [ -n "$profile" ] || continue
        ssid="$(nmcli -g 802-11-wireless.ssid con show "$profile" 2>/dev/null)"
        if grep -Fxq -- "$ssid" <<<"$visible"; then
            run timeout 45 nmcli con up id "$profile" ifname "$IFACE" && online && return 0
        else
            log "known network '$profile' ($ssid) not in range"
        fi
    done <<<"$(known_profiles)"
    return 1
}

reload_driver() {
    run modprobe -r "$DRIVER"
    sleep 3
    run modprobe "$DRIVER"
    sleep 15   # USB re-enumeration + NM autoconnect
}

fails=0
step=0
was_online=1
log "watching $IFACE (driver $DRIVER), check every ${CHECK_INTERVAL}s"

# Boot: bring Wi-Fi up ourselves instead of waiting for the first failures.
run nmcli radio wifi on
for _ in $(seq 30); do
    ip link show "$IFACE" >/dev/null 2>&1 && break
    sleep 2
done
if [ "$(nmcli -g GENERAL.STATE dev show "$IFACE" 2>/dev/null | cut -d' ' -f1)" != 100 ]; then
    run timeout 45 nmcli device connect "$IFACE" || try_known_networks
fi
log "$IFACE: $(nmcli -g GENERAL.CONNECTION dev show "$IFACE" 2>/dev/null || echo down)"
while true; do
    if online; then
        [ "$was_online" = 1 ] || log "internet via $IFACE restored"
        fails=0; step=0; was_online=1
        sleep "$CHECK_INTERVAL"
        continue
    fi

    fails=$((fails + 1))
    [ "$was_online" = 0 ] || log "no internet via $IFACE ($(nmcli -g GENERAL.CONNECTION dev show "$IFACE" 2>/dev/null))"
    was_online=0
    if ! ip link show "$IFACE" >/dev/null 2>&1; then
        log "$IFACE is missing: the driver has hung"
        reload_driver
        continue
    fi
    if [ $((fails % FAILS_PER_STEP)) -ne 0 ]; then
        sleep "$CHECK_INTERVAL"
        continue
    fi

    step=$((step + 1))
    case "$step" in
        1) run timeout 45 nmcli device connect "$IFACE" ;;
        2) try_known_networks ;;
        3) run nmcli radio wifi off; sleep 5; run nmcli radio wifi on; sleep 15 ;;
        4) reload_driver ;;
        *) log "all recovery steps failed; retrying in ${COOLDOWN}s"
           step=0
           sleep "$COOLDOWN" ;;
    esac
    sleep 5
done
