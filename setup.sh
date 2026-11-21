#!/bin/bash
# One-shot setup for the Aletta voice pipeline.
#   ./setup.sh robot   # Unitree G1 Jetson Orin NX (Ubuntu 20.04, ROS 2 Foxy)
#   ./setup.sh dev     # dev machine (Ubuntu 22.04, ROS 2 Humble)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    read -r -p "Set up for 'robot' (G1 Jetson) or 'dev' machine? " TARGET
fi
case "$TARGET" in
    robot) ROS_DISTRO=foxy ;;
    dev)   ROS_DISTRO=humble ;;
    *) echo "usage: $0 robot|dev"; exit 1 ;;
esac

PIPER_VOICES=https://huggingface.co/rhasspy/piper-voices/resolve/main
PIPER_RELEASE=https://github.com/rhasspy/piper/releases/download/2023.11.14-2
VOICE_DIR="$HOME/.local/share/piper"
OWW_DIR="$HOME/.local/share/openwakeword"
PIPER_DIR="$HOME/.local/piper-bin"

echo "[1/6] Submodules and Python packages"
git -C "$REPO" submodule update --init --recursive
sudo apt-get install -y portaudio19-dev alsa-utils
PKGS="faster-whisper sounddevice soundfile scipy openwakeword"
[ "$TARGET" = dev ] && PKGS="$PKGS piper-tts"
pip3 install --user $PKGS

echo "[2/6] Piper voices (English + Polish)"
mkdir -p "$VOICE_DIR/pl_PL-gosia-medium"
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
    wget -q -nc -O "$VOICE_DIR/$f" "$PIPER_VOICES/en/en_US/lessac/medium/$f"
done
for f in pl_PL-gosia-medium.onnx pl_PL-gosia-medium.onnx.json; do
    wget -q -nc -O "$VOICE_DIR/pl_PL-gosia-medium/$f" \
        "$PIPER_VOICES/pl/pl_PL/gosia/medium/$f"
done

echo "[3/6] Whisper + openWakeWord models"
# medium transcribes, small does the language-ID pass (voice_params_real.yaml);
# both are cached so the first start needs no internet.
python3 -c "from faster_whisper.utils import download_model; download_model('medium'); download_model('small')"
mkdir -p "$OWW_DIR"
python3 -c "from openwakeword.utils import download_models; download_models(target_directory='$OWW_DIR')"
MODEL="$REPO/src/wakeword_training/models/aletta_v0.1.onnx"
if [ -f "$MODEL" ]; then
    cp -n "$MODEL" "$OWW_DIR/"
else
    echo "WARNING: $MODEL missing; wake word stays off until you train one"
fi

if [ "$TARGET" = robot ]; then
    echo "[4/6] Piper binary -> $PIPER_DIR"
    TMP="$(mktemp -d)"
    wget -q -O "$TMP/piper.tar.gz" "$PIPER_RELEASE/piper_linux_aarch64.tar.gz"
    tar -xzf "$TMP/piper.tar.gz" -C "$TMP"
    mkdir -p "$PIPER_DIR"
    cp -r "$TMP/piper/." "$PIPER_DIR/"
    rm -rf "$TMP"
    echo "     GPU Whisper also needs CTranslate2 built with CUDA in ~/ct2_install (see README)."
fi

echo "[5/6] Ollama + llama3.2"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if [ "$TARGET" = robot ]; then
    sudo cp "$REPO/ollama.service" /etc/systemd/system/ollama.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now ollama.service
fi
ollama pull llama3.2

echo "[6/6] Build"
set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
set -u
cd "$REPO"
colcon build --symlink-install

if [ "$TARGET" = robot ]; then
    sudo cp "$REPO/unitree_converse.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now unitree_converse.service
    echo "Done. Logs: journalctl -u unitree_converse -f"
else
    echo "Done. Run: source install/setup.bash && ros2 launch g1_voice voice_sim.launch.py"
fi
