# unitree_converse

A complete voice conversation pipeline for the **Unitree G1 humanoid robot** ("Aletta"), running fully on-robot with no cloud dependency. The robot listens, understands, and responds using a smooth English female voice through its built-in speaker.

---

## Demo

- **Say "Aletta"** → robot wakes and opens a conversation session (turns chain automatically; closes after 15 s of silence)
- **Hold F1** on the Unitree remote → robot listens → release F1 → robot responds
- **Press START** → end the conversation session immediately (kill switch — sessions start only via "Aletta")
- **Press F3** → mute/unmute the robot's spoken replies (processing continues while muted)
- Aletta knows her battery level, orientation, network status, uptime, and software stack

---

## Quick Start

Clone the repository, then run the setup script:

```bash
git clone https://github.com/PiotrG1996/unitree_converse.git
cd unitree_converse
./setup.sh
```

> This repository is a fork of
> [SaxionMechatronics/unitree_converse](https://github.com/SaxionMechatronics/unitree_converse).
> Use the fork for your own changes and push to your fork when you're ready to
> share work:
>
> ```bash
git remote add myfork https://github.com/PiotrG1996/unitree_converse.git
git push -u myfork main
> ```

The setup script will ask whether you are setting up on the **Unitree G1 Jetson** or a **dev machine**, then handle everything automatically:

- Downloads Piper voice model (`en_US-lessac-medium`)
- Downloads faster-whisper base model
- Initializes git submodules (`src/bob_llm`, `src/wakeword_training`)
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
| Speech-to-text | faster-whisper (`base`, **GPU** — CUDA float16) |
| Text-to-speech | Piper TTS (`en_US-lessac-medium` for English, `pl_PL-gosia-medium` for Polish) |
| Audio output | Unitree AudioHub API via `g1_audio_worker` (`AudioClient.PlayStream`) |
| Wake word | **"Aletta"** — custom openWakeWord model (ONNX) over the G1 UDP microphone (pad buttons only otherwise) |

---

## GPU-Only Speech-to-Text (faster-whisper on CUDA)

Speech recognition runs **entirely on the Jetson Orin iGPU** — there is no CPU
fallback. On this platform the CPU is too slow for a natural turn cadence, so
the pipeline is deliberately GPU-only.

### How it is wired

Three things make GPU-only transcription work on this Jetson:

1. **Device + compute type come from config.** `stt_node` reads
   `device: "cuda"` from `voice_params_real.yaml` and derives the matching
   quantisation automatically — `float16` for CUDA, `int8` for CPU. Plain
   `int8` is not a valid CUDA quantisation for faster-whisper.

   ```yaml
   /stt_node:
     ros__parameters:
       device: "cuda"      # GPU-only: float16 on the Jetson iGPU
   ```

2. **The CUDA CTranslate2 engine is preloaded by absolute path.** The pip
   `ctranslate2` wheel on this Jetson ships *without* the native
   `libctranslate2.so.4`, so the user compiled CTranslate2 from source and
   installed it outside the dynamic-linker search path. Under the minimal
   systemd environment (`LD_LIBRARY_PATH=/usr/local/lib` only) a bare
   `import faster_whisper` would fail with
   `ImportError: libctranslate2.so.4: cannot open shared object file`.
   `stt_node._preload_ctranslate2()` loads the hand-built library with
   `ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)` before importing
   faster-whisper. `RTLD_LOCAL` (not `RTLD_GLOBAL`) is essential — forcing
   global symbols corrupts the heap inside a full ROS node
   (`free(): invalid pointer` during CUDA model load). This is a harmless
   no-op on machines where CTranslate2 is already on the loader path.

3. **A CUDA load failure fails loudly instead of silently dropping to CPU.**
   If the model cannot load on the GPU, the node logs the device/compute type
   and re-raises — a broken GPU path is never papered over by a slow CPU run
   that would go unnoticed.

> **No second Whisper on the wake channel.** Wake detection uses the custom
> openWakeWord "Aletta" model (TFLite/onnx, sub-watt CPU inference), not a
> second ASR instance — so it adds no unified-memory pressure on the GPU
> (a prior OOM on this robot killed the LLM mid-reply).

### CPU vs GPU benchmark

`scripts/bench_whisper_gpu.py` transcribes the **same audio** with the same
`base` model on both backends (CPU = stock pip `ctranslate2` int8; GPU =
hand-built CUDA `ctranslate2` float16) and reports load time, transcription
time, and the real-time factor (transcribe time ÷ audio duration; `<1` means
faster than real time).

Measured on the G1 Jetson Orin NX, `base` model, 5.53 s utterance, 5 passes:

| Backend | Compute type | Model load | Transcribe (median) | Real-time factor | Transcript |
|---------|--------------|-----------:|--------------------:|-----------------:|------------|
| **CPU** (previous) | int8 | 0.79 s | 1.217 s | 0.220 | correct |
| **GPU** (current)  | float16 | 0.75 s | **0.367 s** | **0.066** | correct |

**Result: ~3.3× faster end-to-end transcription**, with the same transcript.
Because the real-time factor falls from `0.22` to `0.066`, transcribing a
5.5-second utterance drops from ~1.2 s to ~0.37 s, and the first token of the
reply starts noticeably sooner. Reproduce it with:

```bash
./scripts/check_pipeline.sh          # also synthesises a WAV to transcribe
python3 scripts/bench_whisper_gpu.py base /tmp/bench_speech.wav
```

---

## Package Structure

Top-level layout (as produced by `tree -L 1`):

```
unitree_converse/
├── build                            # colcon build dir (generated, git-ignored)
├── install                          # colcon install dir (generated, git-ignored)
├── log                              # colcon build/test logs (generated)
├── ensure_pulse.sh                  # ExecStartPre: bring up PulseAudio
├── setup.sh                         # Interactive setup script
├── source_ros.sh                    # Source ROS2 + CycloneDDS env for manual use
├── ollama.service                   # Ollama systemd service reference
├── unitree_converse.service         # Voice-pipeline systemd service file
├── unitree_converse_architecture.svg  # Architecture diagram
├── README.md
├── scripts/
│   ├── wait_for_robot_ready.sh     # ExecStartPre: DDS/eth0 readiness gate
│   ├── check_audio.sh              # Speaker + mic diagnostics (Pulse/ALSA/UDP)
│   ├── check_pipeline.sh           # Ollama + Piper + Whisper stage checks
│   ├── bench_whisper_gpu.py        # CPU vs GPU Whisper transcription benchmark
│   └── test_mic_live.py            # Capture from the UDP mic and transcribe live
└── src/
    ├── g1_voice/                       # This package (ROS2 ament_python)
    │   ├── package.xml                 # ROS2 package manifest
    │   ├── setup.py                    # Console-script entry points for each node
    │   ├── setup.cfg                   # setuptools develop/install config
    │   ├── resource/g1_voice           # ament resource index marker
    │   ├── g1_voice/
    │   │   ├── __init__.py                 # (empty) marks the Python package
    │   │   ├── wake_word_node.py           # wake listener: "Aletta" openWakeWord only (no phrases, no keyboard)
    │   │   ├── wake_word_openwakeword.py   # openWakeWord detector wrapper (import-light, loud-error)
    │   │   ├── stt_node.py                 # faster-whisper STT on GPU + UDP mic + speech gate
    │   │   ├── speech_gate.py              # No-speech confidence filter (rclpy-free helpers)
    │   │   ├── tts_node.py                 # Piper TTS, bilingual streamed via llm_stream
    │   │   ├── g1_audio_worker.py          # Native G1 AudioHub playback (separate process)
    │   │   ├── action_dispatcher_node.py   # (placeholder, empty) future robot-action node
    │   │   ├── button_trigger_node.py      # F1/F3/START/SELECT remote button mapping
    │   │   └── robot_state_node.py         # Live robot state injection into the LLM
    │   ├── launch/
    │   │   ├── voice_sim.launch.py     # Dev machine (sounddevice mic, Python TTS)
    │   │   └── voice_real.launch.py    # On-robot (UDP mic, piper binary, all 6 nodes)
    │   ├── config/
    │   │   ├── voice_params.yaml       # Dev machine params
    │   │   └── voice_params_real.yaml  # Jetson robot params (GPU STT, tuned prompt)
    │   └── test/
    │       ├── test_audio_hw.py            # Speaker/microphone hardware tests
    │       ├── test_llm_pipeline.py        # Ollama/Piper/Whisper stage tests
    │       ├── test_speech_gate.py         # No-speech confidence-gate unit tests
    │       ├── test_split_sentences.py     # TTS sentence split / bilingual voice-latch tests
    │       ├── test_stt_language_gate.py   # STT language-duel + noise-skip unit tests
    │       ├── test_copyright.py           # ament_copyright linter (skipped repo-wide)
    │       ├── test_flake8.py              # ament_flake8 linter
    │       └── test_pep257.py              # ament_pep257 (docstring) linter
    ├── wakeword_training/                # Git submodule: wake-word training
    │   ├── scripts/                      # generate / record / train / eval Aletta
    │   ├── data/aletta/                  # Labeled training clips (WAV, 16 kHz)
    │   └── models/                       # Trained ONNX models (aletta_v0.1.onnx)
    └── bob_llm/                        # LLM ROS2 node (submodule, Foxy-patched fork)
```

> The six nodes launched by `voice_real.launch.py` are `wake_word_node`,
> `stt_node`, `llm_node.py` (from `bob_llm`), `tts_node`,
> `button_trigger_node`, and `robot_state_node`. `speech_gate.py`,
> `g1_audio_worker.py`, and `__init__.py` are library/helper modules, not
> standalone nodes.

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
| `/g1/voice/say_text` | `std_msgs/String` | Speak exact text (SELECT button / manual) |
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
| **START** | bit 2 | 4 | End the conversation session (kill switch — sessions start only via "Aletta") |
| **SELECT** | bit 3 | 8 | Speak fixed custom phrase (`custom_say_text`, no STT/LLM) |
> **Note:** There is no physical "F3" label on the pad — bit 7 is F2 in hardware
> docs but is referred to as "F3" in older Unitree docs; this project keeps the
> F3 name for that convention. Continuous mode was moved from F3 to START so F3
> could be dedicated to the mute control.

### Custom speech (SELECT)

Pressing **SELECT** on the pad speaks one fixed phrase without involving speech recognition or the LLM. The text comes from the `custom_say_text` parameter (`/button_trigger_node` block in `voice_params*.yaml`) — edit the text there, no code change needed. The phrase travels on `/g1/voice/say_text`, is spoken with the correct PL/EN Piper voice, honours the F3 mute switch, and never re-triggers continuous listening. Test it without the pad:

```bash
ros2 topic pub --once /g1/voice/say_text std_msgs/msg/String "data: 'Hello! I am Aletta.'"
```

## Conversation Modes

### Wake word (real robot) — "Aletta" only
**By default the robot waits for the wake word "Aletta"** (or a controller-pad
button) before it listens. It does not auto-listen at startup and there is no
always-on mode: saying "Aletta" opens a **conversation session** — a window in
which turns chain automatically (the end of one reply re-arms listening) — and
the window closes itself after `session_silence_timeout` seconds (15 s by
default) with no turn activity. START cannot open a session; it only ends one.

The only voice trigger is the name **"Aletta"**, detected by a custom
openWakeWord model on the G1 UDP microphone. There is no Whisper phrase
matching (`hey robot` / `cześć robot` are gone) and no keyboard trigger —
the only other way to drive the pipeline is the controller pad
(F1 push-to-talk, SELECT custom phrase, F3 mute, START = session kill switch).

The systemd service is configured with `Restart=always`, waits for the robot
network and PulseAudio, and validates the installed ROS configuration, Piper
binary, and both language voice models before launching. After changing source
or configuration files, rebuild and reload the service registration before
testing a reboot.

If the Aletta model file or the `openwakeword` package is missing,
`wake_word_node` logs a loud error and **leaves wake disabled** — it never
falls back to another backend. The pad buttons keep working, so the robot is
still reachable (e.g. SELECT speaks the intro phrase).

> **Mic-stream note:** the wake listener and STT share the single G1 UDP
> microphone stream by taking turns: the wake listener holds the socket while
> the pipeline is idle, closes it when a trigger fires, and reopens it once
> STT/LLM are done. While a conversation session is open, STT owns the
> stream most of the time and the wake word is effectively quiet — the
> session closes itself after 15 s of silence (or press START to end it
> immediately), returning to wake-word-first operation.

For conversation turns, automatic language detection is confidence-aware:
high-confidence Polish or English results are accepted directly. Ambiguous
results are decoded once in each supported language and the stronger result is
selected, avoiding accidental English-voice output for short Polish speech.

### Custom "Aletta" wake word

The name **"Aletta" is the sole voice trigger.** `wake_word_node` loads a
**custom openWakeWord model** trained on the robot's real name and streams
80 ms frames of the UDP mic into it (no `wake_detector` switch — the Whisper
phrase path was removed). openWakeWord scores precomputed
melspectrogram/embedding features instead of running an ASR decoder, so it
reacts faster and uses far less CPU.

openWakeWord does not ship a pre-trained "Aletta" model, so you train one.
All training tooling lives in the **`src/wakeword_training` git submodule**
(scripts, labeled data, trained models) — `setup.sh` initializes it
automatically, or run `git submodule update --init --recursive` yourself.
Run every command below from the submodule root:

```bash
cd src/wakeword_training
```

1. **Synthesize clips** with the robot's own Piper voice — positives *and*
   negatives. Negatives are mandatory: without them the model has nothing to
   separate the wake word from and training aborts.

   ```bash
   python3 scripts/generate_aletta_samples.py --count 200 --negatives 300
   ```

   Writes 16 kHz mono WAVs to `data/aletta/positive/` (several phrasings of
   "Aletta", lightly augmented) and `data/aletta/negative/` (phonetically
   similar words like "Alexa"/"Aleta", plus ordinary speech).

   > **Synthetic negatives only get you so far.** Every clip comes from one
   > Piper voice, so a model trained on synthetic-only data learns "the robot's
   > voice is speaking" as much as it learns the *name* — it wakes on ordinary
   > speech. Record real audio (next step) and re-train.

1b. **Record real audio** — this is what actually fixes the false-accept rate.
   `record_aletta_audio.py` captures from the G1's UDP microphone into the same
   directories, auto-trimming each take to the training window length:

   ```bash
   sudo systemctl stop unitree_converse.service   # MANDATORY, see below
   python3 scripts/record_aletta_audio.py --label negative --count 20
   python3 scripts/record_aletta_audio.py --label positive --count 20
   sudo systemctl start unitree_converse.service
   ```

   For negatives, record whatever the robot really hears: room noise, your
   voice, colleagues talking, near-misses like "Alexa"/"Aleta". For positives,
   have several people say "Aletta" at different distances and speeds.

   > **Stop the service first.** `wake_word_node` and `stt_node` bind the same
   > multicast port with `SO_REUSEPORT`; with two sockets on one port the kernel
   > *distributes* packets between them, so recording while the service runs
   > corrupts both the pipeline's audio and your recordings.

   Files land as `real_pos_NNNN.wav` / `real_neg_NNNN.wav`, numbered after any
   existing clips, so synthetic and real audio coexist.

2. **Train** (needs `torch` and openWakeWord's training extras):

   ```bash
   pip install openwakeword[full] torch torchvision torchaudio
   python3 scripts/train_aletta_wakeword.py --steps 20000 \
       --out ~/.local/share/openwakeword
   ```

   Training runs on CPU (a few minutes for 20k steps); raise `--steps` once you
   have real negatives. The script prints a **held-out** recall / false-accept
   report at the end — read it before shipping the model.

   openWakeWord 0.6.0 exports **ONNX only** (`convert_onnx_to_tflite` needs
   `tensorflow` + `onnx_tf`, which are not installed on the robot), so the
   output is `aletta_v0.1.onnx`. The default `aletta_model_path` and
   `aletta_framework: "onnx"` in the configs already point at it. If you switch
   to tflite later, keep `aletta_framework` and the model file extension in
   sync — the base feature models must match the framework.

3. `setup.sh` downloads openWakeWord's **base** feature models
   (melspectrogram + embedding — not bundled with the pip package) into
   `~/.local/share/openwakeword/`, where the detector finds them at runtime
   (no paths are baked into the YAML configs). Both the `.onnx` and `.tflite`
   variants are fetched, so either `aletta_framework` works.

If the custom model or `openwakeword` package is unavailable, `wake_word_node`
logs a loud error and **leaves wake disabled — there is no fallback backend**.
Pad buttons (F1/START/SELECT) keep working. Tune `aletta_threshold`
(higher = fewer false wakes) using the held-out report from step 2.

#### Runtime false-accept mitigations

Independently of the model quality, `wake_word_node` applies two guards
(both configurable in `voice_params*.yaml`):

- **Consecutive-frame confirmation** (`min_consecutive_frames`, default 2):
  a detection requires that many consecutive 80 ms frames at/above
  `aletta_threshold`. Real utterances hold high scores for several frames;
  single-spike false accepts rarely do. Costs ~80 ms per extra frame.
- **TTS echo guard** (`echo_guard_sec`, default 1.5): while the robot is
  speaking — and for the ring-down afterwards — the wake listener drains
  mic audio **without scoring it**. The model is trained on the robot's own
  Piper voice, so scoring its playback would be a guaranteed self-trigger.

#### Evaluating a model before deploying it

`scripts/eval_aletta_model.py` replays labeled clips through the exact
detector the robot runs and prints recall / false-accept rates per
threshold:

```bash
python3 scripts/eval_aletta_model.py \
    --model ~/.local/share/openwakeword/aletta_v0.1.onnx \
    --data data/aletta --sample 50
```

Deploy only when the negative false-accept column is ~0 at the chosen
threshold; otherwise record real negatives and retrain (step 1b).

### Push-to-Talk (optional)
Hold F1 while speaking → release → robot responds → done.

### Conversation session (chained turns)
Saying **"Aletta"** opens a session → robot listens → responds → automatically listens again, turn after turn. The session closes itself after 15 s without turn activity (`session_silence_timeout`), or immediately when you press **START**. START cannot open a session — only "Aletta" can.

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

/wake_word_node:
  ros__parameters:
    aletta_model_path: "~/.local/share/openwakeword/aletta_v0.1.onnx"
    aletta_framework: "onnx"
    aletta_threshold: 0.85          # higher = fewer false wakes
    min_consecutive_frames: 2       # reject single-spike false accepts
    echo_guard_sec: 1.5             # ignore mic while TTS plays (+ ring-down)
    session_silence_timeout: 15.0   # seconds of silence before the session closes
    idle_listen_burst_sec: 1.5      # idle listener throttle (near-zero CPU)
    idle_sleep_sec: 1.5

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

# faster-whisper base model (pip wheel + CUDA CTranslate2 — see GPU section above)
pip3 install faster-whisper
# The pip ctranslate2 wheel ships WITHOUT the native CUDA libctranslate2.so.4,
# so build CTranslate2 from source with CUDA and install it outside the loader
# path (stt_node preloads it by absolute path). Then pre-download the weights:
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cuda', compute_type='float16')"
# If that raises 'libctranslate2.so.4: cannot open shared object file', set the
# preload dir stt_node searches (ct2_install/lib or CTRANSLATE2_LIB_DIR).

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
| openWakeWord fails to load / wake should be off | Set `use_udp_mic: false` in `voice_params*.yaml` (or drop `wake_word_node` from the launch file). F1/START/SELECT pad buttons still work. |
| Battery showing unknown | `BmsState` uses `unitree_hg` (not `unitree_go`) and needs QoS `RELIABLE`. |
| Piper Python unavailable on aarch64 | Use standalone binary + sox pipeline (`g1_piper_tts`). |
| Saxion WiFi AP isolation | Cannot SSH over WiFi. Use ethernet (`192.168.123.164`) or portable router. |

---

## Future Work

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

RoboticsGap Systems - Piotr Gapski, PhD Candidate, Poland
