"""
Unit tests for the STT no-speech confidence gate (speech_gate.py).

The gate drops Whisper segments it flags as likely non-speech so invented
transcripts from ambient noise never reach the LLM ("please check the
network" roleplay). Thresholds below are grounded in measured evidence from
the G1: quiet-room hallucinations score no_speech_prob 0.53-0.59, while real
speech scores well below 0.5.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'g1_voice'))

from speech_gate import filter_segments, keep_segment  # noqa: E402


class _Seg:
    def __init__(self, text, no_speech_prob):
        self.text = text
        self.no_speech_prob = no_speech_prob


def test_noise_hallucinations_are_dropped():
    # Measured quiet-room hallucinations: no_speech_prob 0.535 / 0.594.
    segs = [_Seg(' Do you want to do the honey?', 0.535),
            _Seg(' What honey?', 0.594)]
    kept, dropped = filter_segments(segs, 0.5)
    assert kept == []
    assert len(dropped) == 2
    assert [s for s, _ in dropped] == [0.535, 0.594]


def test_real_speech_is_kept():
    segs = [_Seg(' What is your battery level?', 0.12)]
    kept, dropped = filter_segments(segs, 0.5)
    assert kept == ['What is your battery level?']
    assert dropped == []


def test_boundary_score_passes():
    # Equal to the threshold counts as speech (strictly-greater drops).
    assert keep_segment(0.5, 0.5) is True
    assert keep_segment(0.50001, 0.5) is False


def test_missing_score_defaults_to_keep():
    # Segments the backend did not score must never be lost.
    kept, dropped = filter_segments([_Seg(' Hello?', None)], 0.5)
    assert kept == ['Hello?']
    assert dropped == []


def test_empty_and_blank_segments_ignored():
    kept, dropped = filter_segments(
        [_Seg('   ', 0.1), _Seg('', 0.1)], 0.5)
    assert kept == []
    assert dropped == []


def test_mixed_clip_keeps_only_speech():
    segs = [_Seg(' Hello Aletta.', 0.21),
            _Seg(' Thank you very much.', 0.61)]
    kept, dropped = filter_segments(segs, 0.5)
    assert kept == ['Hello Aletta.']
    assert len(dropped) == 1


def test_threshold_is_tunable():
    segs = [_Seg(' Quiet murmur.', 0.55)]
    kept_loose, _ = filter_segments(segs, 0.6)
    kept_strict, _ = filter_segments(segs, 0.5)
    assert kept_loose == ['Quiet murmur.']
    assert kept_strict == []
