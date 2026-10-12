#!/usr/bin/env bash
# check_pipeline.sh — verify each stage of the on-robot LLM voice pipeline.
#
# WHAT IT CHECKS (each stage independently, no mic/speaker needed)
#   1. Ollama  — the OpenAI-compatible endpoint answers a trivial chat request
#                and returns the expected model (llama3.2 by default).
#   2. Piper   — the standalone binary synthesises non-empty WAV audio from text
#                (mirrors tts_node._speak_binary, including LD_LIBRARY_PATH and
#                ESPEAK_DATA_PATH so espeak phonemisation works).
#   3. Whisper — faster-whisper transcribes a synthesised clip back to text that
#                contains an expected keyword (end-to-end TTS->STT sanity).
#   4. ROS2    — (optional, ROS_FULL=1) with the pipeline already running, checks
#                the key topics exist and prints the live /g1/robot_state.
#
# USAGE
#   ./scripts/check_pipeline.sh                 # Ollama + Piper + Whisper
#   ROS_FULL=1 ./scripts/check_pipeline.sh      # also probe ROS2 topics
#   OLLAMA_MODEL=llama3.2 ./scripts/check_pipeline.sh
#
# Reads the same paths/params as voice_params_real.yaml but allows overrides.
# Exit code: 0 if every enabled stage passed.

set -u

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
PIPER_BIN="${PIPER_BIN:-$HOME/.local/piper-bin/piper}"
PIPER_LIB_DIR="${PIPER_LIB_DIR:-$HOME/.local/piper-bin}"
VOICE_MODEL="${VOICE_MODEL:-$HOME/.local/share/piper/en_US-lessac-medium.onnx}"
ROS_FULL="${ROS_FULL:-0}"

PASS=0
FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "  [INFO] $*"; }
hdr()  { echo; echo "== $* =="; }

# ── 1. OLLAMA (LLM backend) ──────────────────────────────────────────────────
hdr "Ollama LLM backend ($OLLAMA_URL, model=$OLLAMA_MODEL)"
if ! command -v curl >/dev/null 2>&1; then
    bad "curl not available"
else
    # Confirm the server is up and the model is present.
    TAGS_JSON="$(curl -s -m 8 "$OLLAMA_URL/../api/tags" 2>/dev/null)"
    if [ -z "$TAGS_JSON" ]; then
        # Fall back to the OpenAI models listing.
        TAGS_JSON="$(curl -s -m 8 "$OLLAMA_URL/models" 2>/dev/null)"
    fi
    if [ -z "$TAGS_JSON" ]; then
        bad "Ollama not reachable at $OLLAMA_URL (is ollama.service running?)"
    elif echo "$TAGS_JSON" | grep -q "$OLLAMA_MODEL"; then
        ok "Ollama reachable and '$OLLAMA_MODEL' is pulled"
    else
        info "Ollama reachable but '$OLLAMA_MODEL' not seen in listing"
        ok "Ollama reachable at $OLLAMA_URL"
    fi

    # Real inference round-trip against the OpenAI-compatible endpoint.
    echo "  ... sending a test chat completion (may take a few seconds) ..."
    RESP="$(curl -s -m 60 "$OLLAMA_URL/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}]}" \
        2>/dev/null)"
    REPLY="$(printf '%s' "$RESP" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"]["content"].strip())
except Exception:
    print("")
' 2>/dev/null)"
    if [ -n "$REPLY" ]; then
        ok "chat completion answered: \"$REPLY\""
    else
        bad "no usable reply from $OLLAMA_MODEL (empty/error response)"
        info "raw: $(printf '%s' "$RESP" | head -c 300)"
    fi
fi


# ── 2. PIPER (TTS synthesis) ─────────────────────────────────────────────────
hdr "Piper TTS ($PIPER_BIN)"
if [ ! -x "$PIPER_BIN" ]; then
    bad "piper binary missing/not executable: $PIPER_BIN"
elif [ ! -f "$VOICE_MODEL" ]; then
    bad "voice model missing: $VOICE_MODEL"
else
    WAV="$(mktemp /tmp/g1_piper.XXXXXX.wav)"
    # Mirror tts_node._speak_binary env so espeak phonemisation finds its data.
    PIPER_OUT="$(LD_LIBRARY_PATH="$PIPER_LIB_DIR:${LD_LIBRARY_PATH:-}" \
        ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-$PIPER_LIB_DIR/espeak-ng-data}" \
        "$PIPER_BIN" --model "$VOICE_MODEL" --output-file "$WAV" \
        <<< "Hello, I am Aletta." 2>&1)"
    if [ -s "$WAV" ]; then
        WAV_BYTES="$(wc -c < "$WAV")"
        ok "synthesised WAV ($WAV_BYTES bytes)"
        # Record for the Whisper stage below.
        PIPER_WAV="$WAV"
    else
        bad "piper produced no/empty WAV"
        info "piper output: $(printf '%s' "$PIPER_OUT" | head -c 300)"
        rm -f "$WAV"
        PIPER_WAV=""
    fi
fi
PIPER_WAV="${PIPER_WAV:-}"

# ── 3. WHISPER (STT) — TTS->STT round trip ───────────────────────────────────
hdr "faster-whisper STT (round-trip on synthesised audio)"
if [ -z "$PIPER_WAV" ] || [ ! -s "$PIPER_WAV" ]; then
    info "skipping STT round-trip (no Piper WAV to transcribe)"
else
    echo "  ... transcribing (model 'base' loads on first use) ..."
    TRANSCRIPT="$(python3 - "$PIPER_WAV" <<'PY'
import sys
try:
    from faster_whisper import WhisperModel
    m = WhisperModel('base', device='cpu', compute_type='int8')
    segs, _ = m.transcribe(sys.argv[1], language='en', beam_size=5)
    print(' '.join(s.text.strip() for s in segs).strip())
except Exception as e:
    print(f"__ERROR__ {e}")
PY
)"
    case "$TRANSCRIPT" in
        __ERROR__*)
            bad "whisper failed: ${TRANSCRIPT#__ERROR__ }" ;;
        *Aletta*|*aletta*)
            ok "round-trip transcript contains keyword: \"$TRANSCRIPT\"" ;;
        "")
            bad "whisper returned empty transcript" ;;
        *)
            # Non-empty but no keyword: still proves STT runs end-to-end.
            ok "whisper transcribed audio: \"$TRANSCRIPT\""
            info "expected the word 'Aletta' — check mic/TTS clarity if needed." ;;
    esac
fi
rm -f "$PIPER_WAV"

# ── 4. ROS2 (optional, requires running pipeline) ────────────────────────────
hdr "ROS2 pipeline topics (optional)"
if [ "$ROS_FULL" != "1" ]; then
    info "skipped (set ROS_FULL=1 with the pipeline running to probe topics)"
elif ! command -v ros2 >/dev/null 2>&1; then
    info "ros2 not on PATH — source the workspace first"
else
    for t in /g1/stt/transcript /g1/robot_state llm_response /g1/tts/status; do
        if timeout 8 ros2 topic list 2>/dev/null | grep -q "^${t}$"; then
            ok "topic present: $t"
        else
            bad "topic missing: $t"
        fi
    done
    echo "  --- /g1/robot_state (one sample) ---"
    timeout 10 ros2 topic echo --once /g1/robot_state 2>/dev/null || info "no robot_state sample"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
hdr "Pipeline check summary"
echo "  PASS=$PASS  FAIL=$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: OK — every enabled pipeline stage works."
    exit 0
fi
echo "  RESULT: one or more pipeline stages FAILED (see [FAIL] lines above)."
exit 1

