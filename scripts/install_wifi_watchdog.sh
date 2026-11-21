#!/bin/bash
# One-time setup for a stable robot Wi-Fi link. Run with sudo.
#   sudo ./scripts/install_wifi_watchdog.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Known networks, highest priority first: "profile:priority".
# tmobile-24 / T-Mobile-WiFi-Backup are duplicates of the main SSID and stay off.
PROFILES="RoboticsGap-WiFi:30 T-Mobile_Swiatlowod_8439_2.4GHz:20 OnePlus12:5"

systemctl enable --now NetworkManager.service

for entry in $PROFILES; do
    name="${entry%:*}"; prio="${entry##*:}"
    if nmcli -t -f NAME con show | grep -Fxq "$name"; then
        # System level: usable by every user and at boot without a login
        # (permissions empty); the PSK must be stored in the system keyfile
        # (flags 0), not in a desktop keyring that only unlocks on login.
        if [ "$(nmcli -g 802-11-wireless-security.psk-flags con show "$name")" != 0 ]; then
            echo "WARNING: $name keeps its password in a user keyring; re-enter it with:"
            echo "  sudo nmcli con mod '$name' 802-11-wireless-security.psk-flags 0 802-11-wireless-security.psk '<password>'"
        fi
        # retries 0 = retry forever; powersave 2 = off (power save drops links)
        nmcli con mod "$name" connection.permissions "" \
            connection.autoconnect yes \
            connection.autoconnect-priority "$prio" \
            connection.autoconnect-retries 0 \
            802-11-wireless.powersave 2
        echo "configured $name (priority $prio)"
    fi
done

# Let NetworkManager re-check internet access every 60 s.
cat > /etc/NetworkManager/conf.d/99-robot-wifi.conf <<'EOF'
[connection]
wifi.powersave = 2

[connectivity]
uri=http://connectivity-check.ubuntu.com/
interval=60
EOF
systemctl reload NetworkManager

chmod +x "$REPO/scripts/wifi_watchdog.sh"
cp "$REPO/wifi-watchdog.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wifi-watchdog.service
echo "Done. Follow it with: journalctl -u wifi-watchdog -f"
