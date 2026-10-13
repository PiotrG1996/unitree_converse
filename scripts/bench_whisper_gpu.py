#!/usr/bin/env python3
"""
bench_whisper_gpu.py — head-to-head CPU vs GPU benchmark for faster-whisper.

Measures, for the SAME audio and the SAME model size:
  * model load time (first construction)
  * transcription time over N passes (reports median and best)
  * real-time factor (transcribe_time / audio_duration)

CPU  uses device='cpu',  compute_type='int8'   (what the pipeline used before)
GPU  uses device='cuda', compute_type='float16' (what it uses now)

The CUDA libctranslate2 on this Jetson lives outside the loader search path, so
it is preloaded by absolute path (RTLD_LOCAL) exactly like stt_node does.

Usage:
    python3 scripts/bench_whisper_gpu.py [model_size] [audio.wav]

Defaults: model_size=base, audio=/tmp/bench_speech.wav
"""
import os
import sys
import time
import ctypes
import wave
import subprocess

# --- Preload the hand-built CUDA libctranslate2 (RTLD_LOCAL, see stt_node) ----
for _d in (
    os.environ.get('CTRANSLATE2_LIB_DIR', ''),
    '/home/unitree/ct2_install/lib',
    '/home/unitree/CTranslate2/build',
    os.path.expanduser('~/ct2_install/lib'),
    os.path.expanduser('~/CTranslate2/build'),
):
    if _d:
        for _n in ('libctranslate2.so.4', 'libctranslate2.so'):
            _p = os.path.join(_d, _n)
            if os.path.exists(_p):
                try:
                    ctypes.CDLL(_p, mode=ctypes.RTLD_LOCAL)
                except OSError:
                    pass
                break
        else:
            continue
        break

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def load_audio(path):
    """Read a 16-bit PCM WAV into a float32 numpy array + duration."""
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr, n / sr


def load_audio_duration(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def _parse_child_output(stdout):
    """Parse the single 'RESULT\\t...' line printed by run_child()."""
    for line in stdout.splitlines():
        if line.startswith('RESULT\t'):
            parts = line.split('\t')
            # RESULT, device, compute_type, load_s, best, median, transcript...
            return {
                'device': parts[1],
                'compute_type': parts[2],
                'load_s': float(parts[3]),
                'best': float(parts[4]),
                'median': float(parts[5]),
                'transcript': parts[6] if len(parts) > 6 else '',
            }
    return None


def bench(model_size, audio_path, passes=5):
    audio, sr, dur = load_audio(audio_path)
    print(f'\n=== faster-whisper benchmark ===')
    print(f'model_size      : {model_size}')
    print(f'audio           : {audio_path}')
    print(f'sample_rate     : {sr} Hz')
    print(f'audio_duration  : {dur:.2f} s')
    print(f'timed passes    : {passes}\n')

    # Each device runs in its OWN subprocess to avoid shared-library conflicts:
    #   * GPU : uses the hand-built CUDA ctranslate2 4.1.0 (preloaded .so), which
    #           has NO CPU GEMM backend (cannot run CPU at all).
    #   * CPU : uses the stock pip ctranslate2 4.5.0 (bundled in ctranslate2.libs),
    #           which has a working CPU int8 backend.
    # Running them in one process would let the two libctranslate2 versions clash.
    audio_dur = load_audio_duration(audio_path)

    def run(device, compute_type, use_handbuilt):
        cmd = [sys.executable, __file__, '--child', model_size, audio_path,
               device, compute_type, str(passes), '1' if use_handbuilt else '0']
        env = dict(os.environ)
        if use_handbuilt:
            env['LD_LIBRARY_PATH'] = ('/home/unitree/ct2_install/lib:'
                                      + env.get('LD_LIBRARY_PATH', ''))
        else:
            # Ensure the stock pip ctranslate2 (4.5.0) is found first, NOT the
            # hand-built 4.1.0 in ~/.local.
            env['PYTHONPATH'] = '/usr/local/lib/python3.8/dist-packages'
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return _parse_child_output(out.stdout), out.stderr

    print(f'\n=== faster-whisper benchmark (CPU vs GPU) ===')
    print(f'model_size      : {model_size}')
    print(f'audio           : {audio_path}')
    print(f'audio_duration  : {audio_dur:.2f} s')
    print(f'timed passes    : {passes}')
    print(f'CPU backend     : stock pip ctranslate2 4.5.0 (int8)')
    print(f'GPU backend     : hand-built CUDA ctranslate2 4.1.0 (float16)\n')

    cpu_res, _ = run('cpu', 'int8', use_handbuilt=False)
    gpu_res, _ = run('cuda', 'float16', use_handbuilt=True)

    for tag, r in (('CPU (int8)', cpu_res), ('GPU (float16)', gpu_res)):
        if r is None:
            print(f'--- {tag}: FAILED ---\n')
            continue
        print(f'--- {tag}  (device={r["device"]}, compute_type={r["compute_type"]}) ---')
        print(f'  model load     : {r["load_s"]:6.2f} s')
        print(f'  transcribe min : {r["best"]:6.3f} s')
        print(f'  transcribe med : {r["median"]:6.3f} s')
        print(f'  real-time fact : {r["median"] / audio_dur:6.3f}  '
              f'(<1 = faster than realtime)')
        print(f'  transcript     : "{r["transcript"][:80]}"')
        print()

    print('=== SUMMARY ===')
    if cpu_res and gpu_res:
        print(f'CPU median : {cpu_res["median"]:6.3f} s')
        print(f'GPU median : {gpu_res["median"]:6.3f} s')
        if gpu_res['median'] > 0:
            print(f'speedup    : {cpu_res["median"] / gpu_res["median"]:5.2f}x'
                  f'  (GPU vs CPU, higher = GPU faster)')
    else:
        print('one side failed; see per-device output above')
    print()


def run_child(model_size, audio_path, device, compute_type, passes, use_handbuilt):
    """Run a single device's benchmark and print one RESULT line (parsed by parent)."""
    if use_handbuilt == 1:
        for _d in (
            os.environ.get('CTRANSLATE2_LIB_DIR', ''),
            '/home/unitree/ct2_install/lib',
            '/home/unitree/CTranslate2/build',
        ):
            if _d:
                for _n in ('libctranslate2.so.4', 'libctranslate2.so'):
                    _p = os.path.join(_d, _n)
                    if os.path.exists(_p):
                        try:
                            ctypes.CDLL(_p, mode=ctypes.RTLD_LOCAL)
                        except OSError:
                            pass
                        break
                else:
                    continue
                break

    audio, _sr, dur = load_audio(audio_path)
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    load_s = time.perf_counter() - t0
    dev = getattr(model.model, 'device', '?')
    ct = getattr(model.model, 'compute_type', '?')

    segs, _ = model.transcribe(audio, language='en', beam_size=1, vad_filter=True)
    _ = list(segs)  # warm-up

    times = []
    transcript = ''
    for _ in range(passes):
        t0 = time.perf_counter()
        segs, _ = model.transcribe(audio, language='en', beam_size=1, vad_filter=True)
        transcript = ' '.join(s.text.strip() for s in segs).strip()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    # Single tab-separated line the parent can parse.
    print('RESULT\t%s\t%s\t%.4f\t%.4f\t%.4f\t%s' % (
        dev, ct, load_s, times[0], median, transcript.replace('\t', ' ')))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--child':
        # child mode: argv = --child model audio device compute passes use_handbuilt
        run_child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                  int(sys.argv[6]), int(sys.argv[7]))
        sys.exit(0)
    model_size = sys.argv[1] if len(sys.argv) > 1 else 'base'
    audio_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bench_speech.wav'
    if not os.path.exists(audio_path):
        sys.exit(f'audio not found: {audio_path}')
    bench(model_size, audio_path)
