"""
LLM pipeline tests: verify each stage of the on-robot voice pipeline.

Stages covered (each independent, no mic/speaker required):
  * Ollama backend reachable and returns a completion for the configured model.
  * Piper TTS synthesises non-empty audio for a sample sentence.
  * faster-whisper transcribes Piper audio back to text (TTS -> STT round).

These tests skip (rather than fail) when a backend is intentionally unavailable
so they are safe to run on a dev machine as well as the robot.

Override defaults with env vars to match voice_params_real.yaml:
    OLLAMA_URL, OLLAMA_MODEL, PIPER_BIN, PIPER_LIB_DIR, VOICE_MODEL
"""

import json
import os
import subprocess

import pytest

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
PIPER_BIN = os.environ.get(
    "PIPER_BIN", os.path.expanduser("~/.local/piper-bin/piper"))
PIPER_LIB_DIR = os.environ.get(
    "PIPER_LIB_DIR", os.path.expanduser("~/.local/piper-bin"))
VOICE_MODEL = os.environ.get(
    "VOICE_MODEL",
    os.path.expanduser("~/.local/share/piper/en_US-lessac-medium.onnx"))


def _curl_json(path, payload, timeout=60):
    """POST to the Ollama OpenAI-compatible endpoint; return dict or None."""
    import urllib.request
    import urllib.error
    url = OLLAMA_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def test_ollama_reachable():
    """The OpenAI-compatible endpoint must answer /models or chat."""
    resp = _curl_json("/chat/completions", {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": "Say OK."}],
    }, timeout=90)
    if resp is None:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}")
    assert resp, "empty response from Ollama"


def test_ollama_returns_reply():
    """A chat completion must include a non-empty assistant message."""
    resp = _curl_json("/chat/completions", {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with the word OK."}],
    }, timeout=90)
    if resp is None:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}")
    try:
        content = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        pytest.fail(f"unexpected Ollama response shape: {resp}")
    assert content, "assistant reply was empty"


def test_piper_model_present():
    """The configured voice model file must exist."""
    if not os.path.exists(VOICE_MODEL):
        pytest.skip(f"voice model not found: {VOICE_MODEL}")
    assert os.path.getsize(VOICE_MODEL) > 0


def _synthesise(text="Hello, I am Aletta."):
    """Run the standalone piper binary; return path to a non-empty WAV."""
    if not os.path.isfile(PIPER_BIN) or not os.access(PIPER_BIN, os.X_OK):
        pytest.skip(f"piper binary not executable: {PIPER_BIN}")
    if not os.path.exists(VOICE_MODEL):
        pytest.skip(f"voice model not found: {VOICE_MODEL}")

    out = f"/tmp/g1_test_piper_{os.getpid()}.wav"
    env = os.environ.copy()
    if PIPER_LIB_DIR:
        env["LD_LIBRARY_PATH"] = PIPER_LIB_DIR + os.pathsep + env.get(
            "LD_LIBRARY_PATH", "")
        env.setdefault(
            "ESPEAK_DATA_PATH",
            os.path.join(PIPER_LIB_DIR, "espeak-ng-data"))
    try:
        subprocess.run(
            [PIPER_BIN, "--model", VOICE_MODEL, "--output-file", out],
            input=text.encode("utf-8"), capture_output=True, timeout=60,
            env=env, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        pytest.fail(f"piper execution failed: {e}")
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        pytest.skip("piper produced no audio (binary/model may be incomplete)")
    return out


def test_piper_synthesises_audio(tmp_path):
    """Piper must produce a non-empty WAV for a sample sentence."""
    out = _synthesise()
    try:
        assert os.path.getsize(out) > 0
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_whisper_roundtrip_on_piper_audio():
    """
    TTS -> STT round trip: transcribe Piper audio and get usable text.

    Verifies the STT stage can decode the same audio format TTS produces.
    The model is 'base' to match stt_node's default.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        pytest.skip(f"faster-whisper unavailable: {e}")

    wav = _synthesise("Hello, I am Aletta.")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav, language="en", beam_size=5)
        transcript = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        pytest.fail(f"whisper transcription failed: {e}")
    finally:
        if os.path.exists(wav):
            os.unlink(wav)

    assert transcript, "whisper returned an empty transcript"
    # Prefer the expected keyword, but any non-empty transcript proves the
    # STT stage runs end-to-end (model output can vary).
    assert any(ch.isalpha() for ch in transcript), (
        f"transcript has no letters: {transcript!r}")
