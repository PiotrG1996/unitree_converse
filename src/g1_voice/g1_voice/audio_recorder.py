"""Audio recording for STT: UDP multicast mic or local sounddevice."""


import socket
import struct

import numpy as np


class AudioRecorder:
    """Capture mono 16 kHz audio from UDP multicast or sounddevice mic."""

    def __init__(
        self,
        sample_rate=16000,
        recording_duration=10.0,
        silence_threshold=0.008,
        silence_duration=2.0,
        use_udp_mic=True,
        udp_multicast_group='239.168.123.161',
        udp_port=5555,
        udp_local_ip='192.168.123.164',
    ):
        self.sample_rate = sample_rate
        self.recording_duration = recording_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.use_udp_mic = use_udp_mic
        self.udp_multicast_group = udp_multicast_group
        self.udp_port = udp_port
        self.udp_local_ip = udp_local_ip
        self.stop_requested = False

    def record(self):
        self.stop_requested = False
        if self.use_udp_mic:
            return self._record_via_udp()
        return self._record_via_sounddevice()

    def request_stop(self):
        self.stop_requested = True

    def _record_via_udp(self):
        bytes_per_sample = 2
        target_bytes = int(
            self.sample_rate * bytes_per_sample * self.recording_duration)
        silence_limit = int(
            self.silence_duration * self.sample_rate * bytes_per_sample)
        min_speech_bytes = int(0.2 * self.sample_rate * bytes_per_sample)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.udp_port))

        mreq = struct.pack(
            '4s4s',
            socket.inet_aton(self.udp_multicast_group),
            socket.inet_aton(self.udp_local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(0.1)

        frames = []
        total_bytes = 0
        silence_bytes = 0
        speech_detected = False
        speech_bytes = 0
        rms_window = []
        rms_window_max = 5

        try:
            while total_bytes < target_bytes and not self.stop_requested:
                try:
                    data, _ = sock.recvfrom(65535)
                    frames.append(data)
                    total_bytes += len(data)

                    samples = np.frombuffer(
                        data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    rms_window.append(rms)
                    if len(rms_window) > rms_window_max:
                        rms_window.pop(0)
                    smooth_rms = float(np.median(rms_window))

                    if smooth_rms >= self.silence_threshold:
                        speech_bytes += len(data)
                        if speech_bytes >= min_speech_bytes:
                            speech_detected = True
                        silence_bytes = 0
                    else:
                        speech_bytes = 0
                        if speech_detected:
                            silence_bytes += len(data)

                    if (speech_detected
                            and silence_bytes >= silence_limit):
                        break

                except socket.timeout:
                    continue

        except Exception:
            return None
        finally:
            sock.close()

        if not frames:
            return None

        raw = b''.join(frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _record_via_sounddevice(self):
        import sounddevice as sd

        self.stop_requested = False
        audio_frames = []
        silence_frames = 0
        silence_limit = int(self.silence_duration * self.sample_rate)
        chunk_size = 1024
        speech_detected = False
        min_speech_frames = int(0.2 * self.sample_rate)
        speech_frames = 0

        def audio_callback(indata, frames, time_info, status):
            audio_frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=chunk_size,
                callback=audio_callback,
            ):
                total_samples = 0
                max_samples = int(
                    self.recording_duration * self.sample_rate)

                while (total_samples < max_samples
                        and not self.stop_requested):
                    sd.sleep(100)
                    total_samples = sum(len(f) for f in audio_frames)

                    if audio_frames:
                        tail = audio_frames[-10:]
                        recent = np.concatenate(tail)
                        rms = float(np.sqrt(np.mean(recent ** 2)))
                        if rms >= self.silence_threshold:
                            speech_frames += chunk_size
                            if speech_frames >= min_speech_frames:
                                speech_detected = True
                            silence_frames = 0
                        else:
                            speech_frames = 0
                            if speech_detected:
                                silence_frames += chunk_size

                        if (speech_detected
                                and silence_frames >= silence_limit):
                            break

        except Exception:
            return None

        if not audio_frames:
            return None

        return np.concatenate(audio_frames).flatten()
