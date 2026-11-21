# g1_voice

ROS 2 package for spoken conversation on the **Unitree G1** humanoid, with and
without internet:

| Robot state | Who answers | Runs on |
|-------------|-------------|---------|
| **Online** (the G1 reports Wi-Fi/internet) | The **official Unitree GPT** (`chat_go`): its ASR, cloud LLM, male English voice and gestures | The robot's own computer; the Jetson pipeline idles and its LLM is unloaded |
| **Offline**, or **F1** pressed while online | **Aletta**: wake word, Whisper, Ollama llama3.2, Piper (English and Polish, female voices) | The Jetson Orin NX, fully offline |

`assistant_mode_node` decides between the two, `wake_word_node` -> `stt_node` ->
`robot_state_node` -> `llm_node` (from [bob_llm](../bob_llm)) -> `tts_node` form
Aletta's pipeline. Target: ROS 2 Foxy on the G1's Jetson (Ubuntu 20.04,
CycloneDDS); the simulation launch also runs on Humble.

The repository [README](../../README.md) explains the design, measurements and
operations. This file is the package reference.

## Launch

```bash
ros2 launch g1_voice voice_real.launch.py [log_level:=DEBUG]   # G1 (Jetson)
ros2 launch g1_voice voice_sim.launch.py  [log_level:=DEBUG]   # dev machine
```

| Launch file | Config | Nodes |
|-------------|--------|-------|
| `voice_real.launch.py` | `config/voice_params_real.yaml` | wake_word, stt, robot_state, llm (bob_llm), tts, button_trigger, assistant_mode |
| `voice_sim.launch.py` | `config/voice_params.yaml` | the same without assistant_mode (local mic, CPU Whisper, local playback) |

On the robot, `unitree_converse.service` runs the real launch file.

## Nodes

### assistant_mode_node: Unitree GPT or Aletta
Switches the robot service `chat_go` with the robot_state service API
(`ServiceSwitch`, api 1001, sent as plain ROS messages) and publishes who
answers. F1 (online) toggles Unitree GPT / Aletta with spoken progress;
Ollama's model is unloaded while Unitree answers (frees 2.2 GB) and loaded
again for Aletta (~5 s). Decision logic: `logic.AssistantArbiter` (unit-tested).

| | Topic | Type |
|---|---|---|
| pub | `/g1/voice/assistant_mode` (latched) | String: `unitree` / `aletta` |
| pub | `/g1/voice/assistant_online` (latched) | Bool: F1 switches (true) or is push-to-talk |
| pub | `/g1/voice/say_text`, `/g1/voice/session_request`, `/g1/voice/continuous_stop` | String, Bool, Bool |
| pub | `/api/robot_state/request` | unitree_api/Request |
| sub | `/public_network_status`, `/gpt_state`, `/gpt_msg` | String (robot) |
| sub | `/api/gpt/request` (logged: questions chat_go receives), `/api/robot_state/response` | unitree_api |
| sub | `/g1/voice/assistant_toggle`, `/g1/wake/session_state` | Bool, String |

### wake_word_node: "Aletta" and the conversation session
openWakeWord on every 80 ms frame of the G1 mic (UDP multicast), loudness
gate, echo guard (our TTS and the board's `play_state`), refractory period,
false-wake backoff; owns the session (`logic.Session`). Ignores the wake word
while Unitree GPT answers unless `takeover_from_unitree` is true.

| | Topic | Type |
|---|---|---|
| pub | `/g1/voice/trigger`, `/g1/voice/continuous_start`, `/g1/voice/continuous_stop`, `llm_reset` | Bool |
| pub | `/g1/wake/detected` | Float64 (wall time) |
| pub | `/g1/wake/session_state` (latched) | String: `open` / `closed` |
| sub | `/g1/stt/status`, `/g1/stt/transcript`, `llm_response`, `/g1/tts/status`, `/audio_msg` | String |
| sub | `/g1/voice/assistant_mode`, `/g1/voice/session_request`, `/g1/voice/continuous_stop` | String, Bool |

### stt_node: speech recognition
4 s ring buffer of the G1 mic; one worker thread, one turn per trigger.
Endpointing (`logic.Endpointer`), name cut at the pause after "Aletta",
faster-whisper `medium` (int8) with `small` for language ID, Polish/English
selection, no-speech/confidence/hallucination filters.

| | Topic | Type |
|---|---|---|
| pub | `/g1/stt/transcript` | String: `"[USER_LANGUAGE:en\|pl]\n<text>"` |
| pub | `/g1/stt/status` | String: `recording`, `transcribing`, `empty_turn`, `idle` |
| sub | `/g1/voice/trigger`, `/g1/voice/stop_recording`, `/g1/voice/continuous_start`, `/g1/voice/continuous_stop` | Bool |
| sub | `/g1/wake/detected`, `/g1/wake/session_state`, `/g1/voice/assistant_mode` | Float64, String |
| sub | `/g1/tts/status`, `llm_response`, `/audio_msg` | String |

### robot_state_node: prompt enrichment
Prepends a `[ROBOT STATE]` block (battery, orientation, motion mode, network,
local time in 24 h and 12 h form, uptime) and a reply-language instruction.
For questions about the software it adds the ROS graph, the assistant mode and
every parameter of the voice nodes whose name matches the question (all ~100
read via `list_parameters`/`get_parameters`, read-only). Loads the LLM at start.

| | Topic | Type |
|---|---|---|
| pub | `llm_prompt` | String |
| pub | `/g1/robot_state` (every 30 s) | String |
| sub | `/g1/stt/transcript`, `/g1/voice/assistant_mode` | String |
| sub | `/lf/bmsstate`, `/odommodestate` | unitree_hg/BmsState, unitree_go/SportModeState |
| clients | `/<voice node>/list_parameters`, `/<voice node>/get_parameters` | rcl_interfaces |

### tts_node: speech output
Pending-turn FIFO, sentence streaming, persistent Piper process per voice
(`en_US-lessac-medium`, `pl_PL-gosia-medium`), playback through
`g1_audio_worker.py` (separate process for Unitree AudioHub `PlayStream`).

| | Topic | Type |
|---|---|---|
| pub | `/g1/tts/status` | String: `speaking` / `idle` |
| pub | `/g1/voice/trigger` (next turn in a session) | Bool |
| sub | `llm_stream`, `llm_response`, `/g1/stt/transcript`, `/g1/voice/say_text` | String |
| sub | `/g1/voice/speak_enable`, `/g1/voice/continuous_start`, `/g1/voice/continuous_stop`, `/g1/wake/session_state` | Bool, String |

### button_trigger_node: wireless controller
| Button | Action |
|--------|--------|
| F1 | online: switch Unitree GPT <-> Aletta (`/g1/voice/assistant_toggle`); offline: hold to talk |
| F3 (labelled F2) | mute / unmute replies |
| START | end the conversation session |
| SELECT | speak `custom_say_text` |

Subscribes `/wirelesscontroller` (unitree_go) and `/g1/voice/assistant_online`.

## Parameters

Every parameter set on the robot is listed, with a comment, in
[`config/voice_params_real.yaml`](config/voice_params_real.yaml). Parameters
that run on their code defaults:

| Node | Parameter | Default | Meaning |
|------|-----------|---------|---------|
| stt_node | `initial_prompt_en` | robot/lab vocabulary | Whisper hint for English (words only) |
| stt_node | `initial_prompt_pl` | `""` | Whisper hint for Polish (off: did not help) |
| stt_node | `hotwords` | `""` | Whisper hotwords (off: hallucinated on silence) |
| stt_node | `reply_timeout_sec` | 60.0 | Give up waiting for a reply before re-listening |
| stt_node | `sample_rate` | 16000 | Mic sample rate (G1 mic) |
| stt_node | `vad_min_silence_ms` | 500 | Whisper VAD silence split |
| tts_node | `turn_timeout_sec` | 90.0 | Drop a pending turn that never got a reply |
| wake_word_node | `oww_melspec_model_path`, `oww_embedding_model_path` | `""` | openWakeWord feature models (`""` = bundled) |
| robot_state_node | `llm_warmup_url`, `llm_warmup_model` | Ollama `/api/generate`, `llama3.2` | LLM warm-up at start (`""` URL = off) |
| assistant_mode_node | `gpt_service_name` | `chat_go` | Robot service of the Unitree GPT |
| assistant_mode_node | `check_interval_sec` | 2.0 | Decision tick |
| assistant_mode_node | `gpt_state_timeout_sec` | 12.0 | chat_go counts as stopped after this without `/gpt_state` |
| assistant_mode_node | `ollama_url`, `ollama_model` | Ollama `/api/generate`, `llama3.2` | Model loaded/unloaded with the mode |

`llm_node` parameters are documented in [bob_llm](../bob_llm/README.md); the
robot values (system prompt, model, `max_history_length: 1`) are in the same
YAML file.

## Tests

```bash
cd src/g1_voice
python3 -m pytest -q --ignore=test/test_llm_pipeline.py   # unit tests + flake8 + pep257
python3 -m pytest -q test/test_llm_pipeline.py            # needs Ollama and Piper
```

Unit tests cover the ROS-free logic in `logic.py` (endpointing, name cut,
filters, language choice, echo guard, session, assistant arbiter, parameter
matching) and the STT ring buffer, TTS turn queue and wake-word wrapper with
ROS stubbed out, using timings and scores recorded on the robot.

## Dependencies

ROS: `rclpy`, `std_msgs`, `rcl_interfaces`, `bob_llm`, `unitree_go`,
`unitree_hg`, `unitree_api` (from unitree_ros2). Python: `faster-whisper`
(CTranslate2 built with CUDA on the Jetson), `openwakeword`, `numpy`, `scipy`,
`sounddevice`; `unitree_sdk2py` in `g1_audio_worker.py`. External: Ollama with
`llama3.2`, the Piper binary and voices. `setup.sh` in the repository root
installs all of them.
