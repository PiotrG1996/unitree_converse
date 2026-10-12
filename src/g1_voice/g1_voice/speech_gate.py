"""Pure helpers for the STT no-speech confidence gate.

Kept import-free (no rclpy/numpy) so pytest can exercise the filtering logic
directly. ``stt_node`` calls into here after Whisper transcription.

Background: in a quiet room Whisper hallucinates coherent sentences from
ambient noise. Those measure ``no_speech_prob`` 0.53-0.59 while genuine
speech is well below 0.5. Dropping high ``no_speech_prob`` segments stops
invented transcripts (e.g. "please check the network") from reaching the LLM.
"""


def keep_segment(no_speech_prob, max_no_speech_prob=0.5):
    """Return True when a Whisper segment is trustworthy speech."""
    try:
        return float(no_speech_prob) <= float(max_no_speech_prob)
    except (TypeError, ValueError):
        # Missing/unparseable score: keep the text rather than risking the
        # loss of genuine speech.
        return True


def filter_segments(segments, max_no_speech_prob=0.5):
    """Split Whisper segments into kept texts and dropped (score, text).

    Args:
        segments: iterable of objects with ``.text`` and
            ``.no_speech_prob`` (e.g. faster-whisper Segment).
        max_no_speech_prob: segments scoring above this are dropped.

    Returns:
        (kept_texts, dropped) where kept_texts is a list of stripped
        non-empty strings and dropped is a list of (score, text) tuples.
    """
    kept = []
    dropped = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not keep_segment(
                getattr(seg, "no_speech_prob", None), max_no_speech_prob):
            try:
                score = float(getattr(seg, "no_speech_prob", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            dropped.append((score, text))
        elif text:
            kept.append(text)
    return kept, dropped
