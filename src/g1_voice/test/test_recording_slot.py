"""
Regression tests for the STT recording-slot lifecycle.

Covers the bugs fixed in the recording-slot rework:
  * __init__ used to call _finish_recording() before _recording_lock was
    created -> AttributeError at node construction.
  * _finish_recording() used to call itself recursively -> RecursionError.

No ROS runtime required: rclpy is stubbed (matching the suite convention, see
test_all_functions.py) and the real slot methods are exercised against a
stand-in object.
"""

import os
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'g1_voice'))

# Stub rclpy/std_msgs so stt_node imports without a live ROS install, mirroring
# the convention used by the rest of the suite (see test_all_functions.py).
for _name in ('rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg'):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules['rclpy.node'].Node = type('_Node', (), {})


class _Msg:
    def __init__(self, data=''):
        self.data = data


sys.modules['std_msgs.msg'].String = _Msg
sys.modules['std_msgs.msg'].Bool = _Msg

from stt_node import STTNode  # noqa: E402


def _standin():
    """STTNode-like stand-in with a fresh recording slot (no ROS needed)."""
    obj = object.__new__(STTNode)
    obj._recording_lock = threading.Lock()
    obj.is_recording = False
    return obj


def test_try_start_recording_claims_slot_once():
    s = _standin()
    assert STTNode._try_start_recording(s) is True
    assert s.is_recording is True
    # Re-entering while recording is in flight must be refused.
    assert STTNode._try_start_recording(s) is False
    assert s.is_recording is True


def test_finish_recording_releases_slot_and_is_idempotent():
    s = _standin()
    assert STTNode._try_start_recording(s) is True
    # Calling _finish_recording() repeatedly must not recurse or raise.
    STTNode._finish_recording(s)
    assert s.is_recording is False
    STTNode._finish_recording(s)
    assert s.is_recording is False
    # The slot is free again after release.
    assert STTNode._try_start_recording(s) is True
    assert s.is_recording is True
    STTNode._finish_recording(s)
