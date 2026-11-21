#!/bin/bash
# Hardware diagnostic script for NVIDIA Jetson Orin NX
# Records system info for Phonon TTS deployment

REPORT_FILE="/home/unitree/phonon-jetson/hardware_report.txt"

echo "=== NVIDIA Jetson Phonon TTS Hardware Report ===" > "$REPORT_FILE"
echo "Generated: $(date -u)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "--- System Identification ---" >> "$REPORT_FILE"
MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "unknown")
echo "Model: $MODEL" >> "$REPORT_FILE"
ARCH=$(uname -m)
echo "Architecture: $ARCH" >> "$REPORT_FILE"
KERNEL=$(uname -r)
echo "Kernel: $KERNEL" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "--- OS and L4T ---" >> "$REPORT_FILE"
cat /etc/nv_tegra_release 2>/dev/null >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "--- Python ---" >> "$REPORT_FILE"
python3 --version >> "$REPORT_FILE" 2>&1
python3 -c "import sys; print('Platform:', sys.platform)" >> "$REPORT_FILE" 2>&1
echo "" >> "$REPORT_FILE"

echo "--- CUDA ---" >> "$REPORT_FILE"
if ls /usr/local/cuda-11.4/bin/nvcc 2>/dev/null; then
    echo "CUDA 11.4: available" >> "$REPORT_FILE"
else
    echo "CUDA: not found" >> "$REPORT_FILE"
fi
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" >> "$REPORT_FILE" 2>&1
echo "" >> "$REPORT_FILE"

echo "--- cuDNN ---" >> "$REPORT_FILE"
if ls /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so 2>/dev/null; then
    echo "cuDNN: available" >> "$REPORT_FILE"
else
    echo "cuDNN: not checked" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

echo "--- TensorRT ---" >> "$REPORT_FILE"
if which trtexec 2>/dev/null; then
    echo "TensorRT: available" >> "$REPORT_FILE"
else
    echo "TensorRT: not installed" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

echo --- Memory and CPU --- >> "$REPORT_FILE"
free -h >> "$REPORT_FILE" 2>&1 || true
lscpu >> "$REPORT_FILE" 2>&1 || true
echo "" >> "$REPORT_FILE"

echo --- Power and Thermal --- >> "$REPORT_FILE"
tegrastats 2>/dev/null | head -5 >> "$REPORT_FILE" 2>&1 || true
ZONE_TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
if [ -n "$ZONE_TEMP" ]; then
    echo "Thermal zone0: $((ZONE_TEMP / 1000))°C" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

echo --- Audio Devices --- >> "$REPORT_FILE"
arecord -l 2>/dev/null >> "$REPORT_FILE" 2>&1 || echo "Audio capture: check" >> "$REPORT_FILE"
aplay -l 2>/dev/null >> "$REPORT_FILE" 2>&1 || echo "Audio playback: check" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "=== Report written to $REPORT_FILE ==="
