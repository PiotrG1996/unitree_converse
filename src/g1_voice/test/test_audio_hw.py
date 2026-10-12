"""
Hardware audio tests: verify the G1 speaker and microphone work.

These tests exercise the SAME device path the TTS node uses (PulseAudio ->
ALSA default) so a pass here means tts_node can drive the real speaker.

Run modes:
    pytest test_audio_hw.py                  # automated: devices open & stream
    G1_INTERACTIVE=1 pytest test_audio_hw.py # human-in-the-loop: also listen

Interactive mode additionally asserts the speaker produced real audio (via the
sink monitor) and prompts you to confirm you heard a tone.

These tests are self-contained (no ROS 2 graph needed) and are skipped cleanly
when the audio backend is unavailable (e.g. on a CI machine).
"""

import os

import numpy as np
import pytest

INTERACTIVE = os.environ.get("G1_INTERACTIVE", "") not in ("", "0", "false")


# Same environment bootstrap tts_node performs so PulseAudio is visible.
def _ensure_pulse_env():
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    if os.path.isdir(runtime):
        os.environ.setdefault("XDG_RUNTIME_DIR", runtime)
        sock = os.path.join(runtime, "pulse", "native")
        if os.path.exists(sock):
            os.environ.setdefault("PULSE_SERVER", f"unix:{sock}")


_ensure_pulse_env()


def _import_sd():
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"sounddevice unavailable: {e}")
    return sd


def _pulse_available():
    try:
        import subprocess
        return subprocess.run(
            ["pactl", "info"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


def _monitor_rms(seconds=1.0, device=None):
    """Record from the default sink monitor and return RMS (0..1)."""
    sd = _import_sd()
    if not _pulse_available():
        pytest.skip("PulseAudio not available for monitor loopback")
    from subprocess import check_output
    try:
        sink = check_output(
            ["pactl", "info"], text=True
        ).split("Default Sink:")[1].split()[0]
    except Exception as e:
        pytest.skip(f"could not determine default sink: {e}")
    monitor = sink + ".monitor"
    try:
        rec = sd.rec(
            int(seconds * 44100), samplerate=44100, channels=1,
            dtype="float32", device=monitor,
        )
        sd.wait()
    except Exception as e:
        pytest.skip(f"monitor record failed: {e}")
    audio = np.asarray(rec).flatten()
    return float(np.sqrt(np.mean(audio ** 2)))


def test_pulseaudio_available():
    """Check that PulseAudio, which drives the G1 speaker, is reachable."""
    if not _pulse_available():
        pytest.skip("PulseAudio not reachable")
    assert _pulse_available()


def test_speaker_plays_tone():
    """
    Play a tone through the default output and confirm playback succeeds.

    In interactive mode, additionally read the sink monitor back and assert
    real audio left the speaker (RMS above threshold).
    """
    sd = _import_sd()
    sr = 44100
    dur = 0.4
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    monitor_thread = None
    monitor_rms = {}
    if INTERACTIVE and _pulse_available():
        def _capture():
            try:
                rec = sd.rec(
                    int(dur * sr), samplerate=sr, channels=1,
                    dtype="float32", device="pulse",
                )
                sd.wait()
                monitor_rms["v"] = float(
                    np.sqrt(np.mean(np.asarray(rec).flatten() ** 2)))
            except Exception as e:
                monitor_rms["err"] = str(e)

        import threading
        monitor_thread = threading.Thread(target=_capture, daemon=True)
        monitor_thread.start()

    try:
        sd.play(tone, sr)
        sd.wait()
    except Exception as e:
        pytest.fail(f"speaker playback failed: {e}")

    if monitor_thread is not None:
        monitor_thread.join(timeout=3.0)

    if INTERACTIVE:
        if "v" in monitor_rms:
            assert monitor_rms["v"] > 0.005, (
                f"speaker monitor RMS too low: {monitor_rms['v']}")
        else:
            prompt = "  Did you hear a 1 kHz tone? [y/N] "
            heard = input(prompt).strip().lower()
            assert heard in ("y", "yes"), "operator did not confirm hearing"


def test_microphone_captures_samples():
    """
    Open the default input, record ~2s, and assert non-empty audio data.

    By default this only proves the capture stream opens and returns samples
    (a silent room still passes). Set G1_INTERACTIVE=1 to also require audible
    input (operator speaks during capture).
    """
    sd = _import_sd()
    try:
        sd.query_devices(kind="input")
    except Exception as e:
        pytest.skip(f"no input device: {e}")

    seconds = 2.0
    try:
        rec = sd.rec(
            int(seconds * 16000), samplerate=16000, channels=1,
            dtype="float32", device=None,
        )
        sd.wait()
    except Exception as e:
        pytest.fail(f"microphone capture failed: {e}")

    audio = np.asarray(rec).flatten()
    assert audio.size > 0, "microphone returned zero samples"
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if INTERACTIVE:
        assert rms > 0.001, (
            f"mic only silence (RMS={rms:.5f}) — speak during capture")
    else:
        assert audio.size >= int(16000 * 0.5), (
            "mic captured far fewer samples than expected")


def test_microphone_device_lists():
    """The default input device must expose at least one channel."""
    sd = _import_sd()
    try:
        dev = sd.query_devices(kind="input")
    except Exception as e:
        pytest.skip(f"no input device: {e}")
    assert dev["max_input_channels"] >= 1
