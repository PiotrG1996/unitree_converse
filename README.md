# unitree_converse

A complete voice conversation pipeline for the **Unitree G1 humanoid robot** ("Aletta"), running fully on-robot with no cloud dependency. The robot listens, understands, and responds using a smooth English female voice through its built-in speaker.

---

## Demo

- **Hold F1** on the Unitree remote → robot listens → release F1 → robot responds
- **Press START** → toggle continuous conversation mode (robot keeps listening after each response)
- **Press F3** → mute/unmute the robot's spoken replies (processing continues while muted)
- Aletta knows her battery level, orientation, network status, uptime, and software stack

---

## Quick Start

```bash
git clone --recurse-submodules https://github.com/SaxionMechatronics/unitree_converse.git
cd unitree_converse
./setup.sh
```

The setup script will ask whether you are setting up on the **Unitree G1 Jetson** or a **dev machine**, then handle everything automatically:

- Downloads Piper voice model (`en_US-lessac-medium`)
- Downloads faster-whisper base model
- Installs Piper binary (Jetson) or piper-tts Python package (dev machine)
- Installs and starts Ollama, pulls LLaMA 3.2 3B
- Builds the ROS2 workspace
- Installs and enables systemd services (Jetson only)

> **Note:** If `setup.sh` fails at any step, see the [Manual Installation](#manual-installation) section below for step-by-step instructions.

---

## Architecture

![unitree_converse architecture](unitree_converse_architecture.svg)

---

## Hardware

| Component | Details |
|-----------|---------|
| Robot | Unitree G1 Edu (29-DOF + Dex3-L hands) |
| Onboard compute | NVIDIA Jetson Orin NX 16GB (`192.168.123.164`) |
| Dev machine | Ubuntu 22.04, RTX Pro 5000 Blackwell |
| Microphone | G1 built-in mic via RockChip UDP multicast `239.168.123.161:5555` |
| Speaker | G1 built-in speaker via Unitree AudioHub API |
| Remote | Unitree wireless controller (`/wirelesscontroller`) |
| Network | Ethernet: dev `192.168.123.100` ↔ Jetson `192.168.123.164` |

---

## Software Stack

| Component | Technology |
|-----------|-----------|
| ROS2 | Foxy (Jetson) / Humble (dev machine) |
| DDS | CycloneDDS via `~/cyclonedds_ws` |
| LLM | Ollama + LLaMA 3.2 3B |
| LLM ROS2 node | bob_llm |
| Speech-to-text | faster-whisper (base, CPU) |
| Text-to-speech | Piper TTS (`en_US-lessac-medium` for English, `pl_PL-gosia-medium` for Polish) |
| Audio output | Unitree AudioHub API via `g1_audio_worker` (`AudioClient.PlayStream`) |
| Wake phrase | Whisper transcript gate (`hey_robot`) over the G1 UDP microphone |

---

## Package Structure

```
unitree_converse/
├── setup.sh                        # Interactive setup script
├── unitree_converse.service        # Systemd service file
├── ollama.service                  # Ollama systemd service reference
├── ensure_pulse.sh                 # ExecStartPre: bring up PulseAudio
├── source_ros.sh                   # Source ROS2 + CycloneDDS env for manual use
├── scripts/
│   ├── wait_for_robot_ready.sh     # ExecStartPre: DDS/eth0 readiness gate
│   ├── check_audio.sh              # Speaker + mic diagnostics (Pulse/ALSA/UDP)
│   └── check_pipeline.sh           # Ollama + Piper + Whisper stage checks
├── README.md
└── src/
    ├── g1_voice/
    │   ├── g1_voice/
    │   │   ├── wake_word_node.py       # openWakeWord + keyboard trigger (sim)
    │   │   ├── stt_node.py             # faster-whisper + UDP mic + stop signal
    │   │   ├── tts_node.py             # Piper TTS with language-specific voices
    │   │   ├── button_trigger_node.py  # F1/F3/START remote button mapping
    │   │   └── robot_state_node.py     # Live robot state injection into LLM
    │   ├── launch/
    │   │   ├── voice_sim.launch.py     # Dev machine (sounddevice mic, Python TTS)
    │   │   └── voice_real.launch.py    # On-robot (UDP mic, piper binary)
    │   ├── config/
    │   │   ├── voice_params.yaml       # Dev machine params
    │   │   └── voice_params_real.yaml  # Jetson robot params
    │   └── test/
    │       ├── test_audio_hw.py        # Speaker/microphone hardware tests
    │       └── test_llm_pipeline.py    # Ollama/Piper/Whisper stage tests
    └── bob_llm/                        # LLM ROS2 node (submodule, Foxy-patched fork)
```

---

## Key Discovery: Audio Routing

The G1's audio is **not** handled by the Jetson's ALSA/PulseAudio. It is managed by a separate **RockChip MCU** at `192.168.123.161`.

### Microphone
The RockChip streams raw **16-bit mono 16kHz PCM** via UDP multicast. The `stt_node` joins the multicast group to receive mic audio.

```python
sock.bind(('', 5555))
mreq = struct.pack('4s4s',
    socket.inet_aton('239.168.123.161'),
    socket.inet_aton('192.168.123.164'))
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
```

### Speaker
The G1 speaker is driven by the **Unitree AudioHub API** through the
`unitree_sdk2py` `AudioClient` (`PlayStream()`), reached over the G1 DDS channel.
The `tts_node` synthesises speech with the standalone **Piper** binary, then hands
the resulting 16-bit PCM to a dedicated `g1_audio_worker` subprocess which plays
it on the onboard speaker:

```
tts_node._speak_binary(text)
        │
        ├── piper --model <voice> --output-file /tmp/xxx.wav   (local render)
        ├── read WAV → normalise → resample → 16-bit PCM @ 16 kHz
        └── g1_audio_worker (subprocess, stdin/stdout JSON protocol)
                 │
                 └── AudioClient.PlayStream(...)  →  G1 AudioHub speaker
```

Because playback is routed through `AudioClient::PlayStream`, the Jetson's
ALSA/PulseAudio output devices are **not** used; the `output_device` parameter is
retained only for config compatibility and has no effect. The `tts_mode`
parameter still selects how speech is *rendered* — `python` (piper-tts Python
package) or `binary` (standalone Piper) — but both render paths play back through
the same native audio worker.

> **Historical note:** the earliest revisions piped Piper → sox → the Jetson
> PulseAudio/ALSA stack (sink `alsa_output.platform-sound.analog-stereo`,
> tried via sounddevice device candidates `['pulse', 'default', None]`). That
> PulseAudio output path is no longer used on the robot; playback now goes
> straight to the AudioHub. The `ensure_pulse.sh` `ExecStartPre` hook is still
> kept for the dev-machine audio diagnostics.

---

## ROS2 Topics

| Topic | Type | Purpose |
|-------|------|---------|
| `/wirelesscontroller` | `unitree_go/msg/WirelessController` | Remote button input |
| `/g1/voice/trigger` | `std_msgs/Bool` | Start recording |
| `/g1/voice/stop_recording` | `std_msgs/Bool` | Stop recording (F1 released) |
| `/g1/voice/continuous_start` | `std_msgs/Bool` | Enable continuous mode |
| `/g1/voice/continuous_stop` | `std_msgs/Bool` | Disable continuous mode |
| `/g1/stt/transcript` | `std_msgs/String` | Raw transcribed speech |
| `/g1/stt/status` | `std_msgs/String` | recording/transcribing/idle |
| `/g1/robot_state` | `std_msgs/String` | Live robot state |
| `llm_prompt` | `std_msgs/String` | Enriched prompt (state + transcript) |
| `llm_response` | `std_msgs/String` | LLM reply |
| `/g1/tts/status` | `std_msgs/String` | speaking/idle |
| `/g1/button/status` | `std_msgs/String` | Button node status |

---

## Button Mapping

The `button_trigger_node` maps the Unitree wireless controller bitfield
(`unitree_go/msg/WirelessController`) to pipeline actions:

| Button | Bit | keys Value | Action |
|--------|-----|-----------|--------|
| **Hold F1** | bit 6 | 64 | Push-to-talk: records while held, transcribes on release |
| **F3** (F2 on pad) | bit 7 | 128 | Toggle spoken output: mute/unmute the robot's replies |
| **START** | bit 2 | 4 | Toggle continuous conversation mode on/off |

> **Note:** There is no physical "F3" label on the pad — bit 7 is F2 in hardware
> docs but is referred to as "F3" in older Unitree docs; this project keeps the
> F3 name for that convention. Continuous mode was moved from F3 to START so F3
> could be dedicated to the mute control.

---

## Conversation Modes

### Wake phrase (real robot)
The real-robot configuration currently runs in continuous conversation mode.
It starts listening automatically after startup and listens for the next turn
after each response. The wake listener is disabled in this profile so it cannot
compete with STT for the UDP microphone stream.

The systemd service is configured with `Restart=always`, waits for the robot
network and PulseAudio, and validates the installed ROS configuration, Piper
binary, and both language voice models before launching. After changing source
or configuration files, rebuild and reload the service registration before
testing a reboot.

The wake phrase is transcript-based because openWakeWord's built-in models do
not include these phrases; changing a model name in YAML alone cannot create a
new wake-word model.

> **Mutually exclusive modes:** the wake-phrase workflow and the F1/START remote
> workflow cannot run at the same time, because both need the single G1 UDP
> microphone stream. The real profile runs purely in continuous (remote-managed)
> mode with `wake_word_node` disabled (`use_udp_mic: false`); to use wake phrases
> instead, enable the wake listener and drive it from the same mic rather than
> also starting STT's UDP capture.

Whisper's wake listener logs each non-empty recognition as `Wake phrase audio`.
This makes it possible to distinguish a microphone/Whisper problem from a
phrase-matching problem. The real-robot configuration also includes common
accent-free Polish transcription variants such as `czesc robot`.

For conversation turns, automatic language detection is confidence-aware:
high-confidence Polish or English results are accepted directly. Ambiguous
results are decoded once in each supported language and the stronger result is
selected, avoiding accidental English-voice output for short Polish speech.

### Push-to-Talk (optional)
Hold F1 while speaking → release → robot responds → done.

### Continuous Mode
Press **START** to toggle on → robot listens → responds → automatically listens again.
Press **START** again to stop.

### Mute (F3)
Press **F3** at any time to mute/unmute the robot's spoken replies. While muted the
pipeline still records, transcribes, and runs the LLM — it simply does not play the
audio, so the continuous loop keeps turning silently.

---

## Systemd Services

The pipeline uses two systemd services that auto-start at boot:

```
ollama.service              ← runs LLaMA 3.2 3B on Jetson GPU
unitree_converse.service    ← voice pipeline (depends on ollama)
```

```bash
# Check status
sudo systemctl status unitree_converse.service
sudo systemctl status ollama.service

# Follow live logs
journalctl -u unitree_converse.service -f

# Restart after config changes
sudo systemctl restart unitree_converse.service

# Stop before manual launch
sudo systemctl stop unitree_converse.service
```

> **Critical:** Never run `ros2 launch` manually while the service is running. Both instances compete for CycloneDDS shared memory and cause `bad_alloc` crashes. Always stop the service first:
> ```bash
> sudo systemctl stop unitree_converse.service
> sleep 2
> sudo rm -f /dev/shm/fastrtps_*
> ros2 launch g1_voice voice_real.launch.py
> ```

---

## Testing

### Automated checks (scripts)

```bash
# Speaker + microphones (PulseAudio/ALSA mic + RockChip UDP multicast mic)
./scripts/check_audio.sh

# LLM pipeline stages: Ollama -> Piper -> Whisper (no mic/speaker needed)
./scripts/check_pipeline.sh

# Also probe the running ROS2 graph (start the service first)
ROS_FULL=1 ./scripts/check_pipeline.sh
```

Both scripts print `[PASS]`/`[FAIL]` per check and exit non-zero on failure,
so they are suitable for CI or a post-deploy smoke test.

### Pytest suite

```bash
# Audio hardware tests (devices open + stream). Add G1_INTERACTIVE=1 to also
# require you to hear a tone / speak into the mic.
cd src/g1_voice && python3 -m pytest test/test_audio_hw.py -v

# LLM pipeline tests (Ollama / Piper / Whisper round trip)
cd src/g1_voice && python3 -m pytest test/test_llm_pipeline.py -v

# bob_llm integration test (requires tests/.env with LLM_API_KEY etc.)
cd src/bob_llm && python3 -m pytest tests/test_llm_integration.py -v
```

### Manual ROS2 checks

```bash
source source_ros.sh   # sets the exact service environment

# Manual transcript — bypasses mic, tests LLM + TTS
ros2 topic pub --once /g1/stt/transcript std_msgs/msg/String "data: 'What is your battery level?'"

# Manual trigger — tests full pipeline including mic
ros2 topic pub --once /g1/voice/trigger std_msgs/msg/Bool "data: true"

# Enable continuous mode
ros2 topic pub --once /g1/voice/continuous_start std_msgs/msg/Bool "data: true"

# Check live robot state
ros2 topic echo /g1/robot_state

# Test the TTS stage directly (renders via Piper, plays via native AudioHub)
./scripts/check_pipeline.sh
```

---

## Configuration (voice_params_real.yaml)

```yaml
/llm_node:
  ros__parameters:
    api_url: "http://localhost:11434/v1"
    api_model: "llama3.2"
    system_prompt: "You are Aletta, a friendly humanoid robot by Unitree Robotics at Saxion University. You have access to your live robot state in [ROBOT STATE] blocks. Use this to answer questions about your battery, orientation, network, and software. Keep ALL responses under 2 sentences. Be concise."

/stt_node:
  ros__parameters:
    use_udp_mic: true
    udp_multicast_group: "239.168.123.161"
    udp_port: 5555
    udp_local_ip: "192.168.123.164"
    silence_threshold: 0.008
    silence_duration: 2.0
    recording_duration: 8.0
    continuous_mode: false

/tts_node:
  ros__parameters:
    tts_mode: "binary"
    continuous_mode: false
```

---

## Sim-to-Real Differences

| Parameter | Simulation (dev machine) | Real Robot (Jetson) |
|-----------|--------------------------|---------------------|
| `use_udp_mic` | `false` (sounddevice) | `true` (UDP multicast) |
| `tts_mode` | `python` (piper-tts lib) | `binary` (g1_piper_tts) |
| ROS2 distro | Humble | Foxy |
| Network interface | `lo` or `wlp132s0f0` | `eth0` |
| CycloneDDS interface | `wlan0` (dev) | `eth0` (robot) |

---

## Manual Installation

> Use these steps if `setup.sh` fails at any point.

### Jetson Orin NX (Ubuntu 20.04, ROS2 Foxy)

```bash
# Python deps
pip3 install faster-whisper sounddevice soundfile tqdm filelock openwakeword
sudo apt-get install -y sox portaudio19-dev

# Piper binary
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
sudo cp piper/piper /usr/local/bin/piper

# Piper voice model
mkdir -p ~/.local/share/piper && cd ~/.local/share/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# faster-whisper base model
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Build g1_piper_tts C++ binary
git clone https://github.com/unitreerobotics/unitree_sdk2.git unitree_sdk2_latest
# Add g1_piper_tts.cpp to example/g1/audio/ and update CMakeLists.txt
cd unitree_sdk2_latest/build && cmake .. && make g1_piper_tts -j$(nproc)

# unitree_ros2 messages
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd unitree_ros2/cyclonedds_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select unitree_go unitree_api unitree_hg

# Build workspace
cd ~/unitree_converse
source /opt/ros/foxy/setup.bash
source ~/cyclonedds_ws/install/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash

# Install services
sudo cp unitree_converse.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ollama.service unitree_converse.service
sudo systemctl start ollama.service unitree_converse.service
```

### Dev Machine (Ubuntu 22.04, ROS2 Humble)

```bash
pip install faster-whisper sounddevice soundfile tqdm filelock openwakeword piper-tts
sudo apt-get install -y sox portaudio19-dev

mkdir -p ~/.local/share/piper && cd ~/.local/share/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

source /opt/ros/humble/setup.bash
cd ~/UNITREE/unitree_converse
colcon build --symlink-install
source install/setup.bash
ros2 launch g1_voice voice_sim.launch.py
```

---

## Known Issues

| Issue | Fix |
|-------|-----|
| `bad_alloc` on launch | Service already running. Stop it, clear `/dev/shm/fastrtps_*`, relaunch. Or reboot. |
| `cyclonedds.xml` must use `eth0` | If set to `wlan0` and WiFi not up at boot, DDS fails. Change `NetworkInterface` to `eth0`. |
| openWakeWord uses ~10GB RAM | Disable with `use_wake_word: false`. Use F1 button instead. |
| Battery showing unknown | `BmsState` uses `unitree_hg` (not `unitree_go`) and needs QoS `RELIABLE`. |
| Piper Python unavailable on aarch64 | Use standalone binary + sox pipeline (`g1_piper_tts`). |
| Saxion WiFi AP isolation | Cannot SSH over WiFi. Use ethernet (`192.168.123.164`) or portable router. |

---

## Future Work

- Custom "Aletta" wake word model
- Voice-controlled motion commands
- Persistent conversation history across reboots
- Upgrade Jetson to ROS2 Humble
- Map additional remote buttons to actions

---

## Credits

- [bob_llm](https://github.com/bob-ros2/bob_llm) — ROS2 LLM node
- [Piper TTS](https://github.com/rhasspy/piper) — Fast local TTS
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Efficient Whisper implementation
- Unitree Robotics — G1 SDK and AudioHub API

---

*SMART Research Group — Saxion University of Applied Sciences, Enschede, Netherlands*
