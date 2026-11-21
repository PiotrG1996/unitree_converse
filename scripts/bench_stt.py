#!/usr/bin/env python3
"""
Compare Whisper models on the robot, through the same pipeline stt_node uses.

    python3 scripts/bench_stt.py small ~/models/faster-whisper-large-v3-turbo:int8_float16

A model is a faster-whisper name or folder, optionally ':compute_type'
(default float16).

Synthetic set: English and Polish sentences rendered with the Piper voices,
then degraded like the G1 far-field mic (room reverb, speech at the measured
level, the measured noise floor) and scored by word error rate (WER).
Real set: turns recorded by stt_node (debug_audio_dir) that contain speech;
printed side by side for inspection (no reference text).
Each model runs in its own process so the memory numbers are clean.
"""

import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys
import time
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'g1_voice'))
from g1_voice.logic import choose_transcription, filter_segments  # noqa: E402
from g1_voice.tts_node import resample, wav_to_float  # noqa: E402

PIPER = os.path.expanduser('~/.local/piper-bin/piper')
VOICES = {
    'en': os.path.expanduser('~/.local/share/piper/en_US-lessac-medium.onnx'),
    'pl': os.path.expanduser('~/.local/share/piper/pl_PL-gosia-medium/pl_PL-gosia-medium.onnx'),
}
SENTENCES = [
    ('en', 'Aletta, what is your battery level?'),
    ('en', 'Can you tell me where the laboratory is?'),
    ('en', 'What is the capital of France?'),
    ('en', 'How many legs does a spider have?'),
    ('en', 'Please introduce yourself to our guests.'),
    ('pl', 'Aletta, jaki jest poziom baterii?'),
    ('pl', 'Gdzie znajduje się laboratorium robotyki?'),
    ('pl', 'Jaka jest stolica Francji?'),
    ('pl', 'Ile nóg ma pająk?'),
    ('pl', 'Przedstaw się proszę naszym gościom.'),
]


def mem_available_mb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable'):
                return int(line.split()[1]) // 1024
    return 0


def words(text):
    return re.findall(r'[^\W_]+', text.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def far_field(audio, rng):
    """Room reverb (RT60 ~0.4 s), speech RMS ~1200/32768, G1 noise floor."""
    n = int(0.4 * 16000)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 4.0
    wet = np.convolve(audio, ir / np.abs(ir).sum() * 4)[:len(audio) + n]
    wet = wet * (1200 / 32768) / np.sqrt(np.mean(wet ** 2))
    pad = np.zeros(8000)
    x = np.concatenate([pad, wet, pad])
    return (x + rng.standard_normal(len(x)) * 170 / 32768).astype(np.float32)


def synth_set():
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(PIPER),
               ESPEAK_DATA_PATH=os.path.join(os.path.dirname(PIPER), 'espeak-ng-data'))
    rng = np.random.default_rng(0)
    out = []
    for lang, text in SENTENCES:
        wav = subprocess.run([PIPER, '--model', VOICES[lang], '--output_file', '-'],
                             input=(text + '\n').encode(), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, env=env, check=True).stdout
        a, r = wav_to_float(wav)
        out.append((lang, text, far_field(resample(a, r), rng)))
    return out


def real_set(folder):
    out = []
    for txt in sorted(glob.glob(os.path.join(folder, '*.txt'))):
        info = open(txt).read()
        m = re.search(r'max (\d+)', info)
        if not m or int(m.group(1)) < 1500:  # skip turns without real speech
            continue
        with wave.open(txt[:-4] + '.wav') as f:
            a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
        out.append((os.path.basename(txt[:-4]), re.search(r'text: (.*)', info).group(1), a))
    return out


def run_model(spec, real_dir):
    size, _, compute = spec.partition(':')
    size = os.path.expanduser(size)
    for d in (os.environ.get('CTRANSLATE2_LIB_DIR', ''), os.path.expanduser('~/ct2_install/lib')):
        if d and os.path.exists(os.path.join(d, 'libctranslate2.so.4')):
            ctypes.CDLL(os.path.join(d, 'libctranslate2.so.4'), mode=ctypes.RTLD_LOCAL)
            break
    from faster_whisper import WhisperModel
    synth = synth_set()
    real = real_set(real_dir) if real_dir else []
    before = mem_available_mb()
    t = time.time()
    model = WhisperModel(size, device='cuda', compute_type=compute or 'float16')
    load = time.time() - t

    def transcribe(audio, language):
        audio = audio * (0.9 / max(1e-4, float(np.max(np.abs(audio)))))
        segs, info = model.transcribe(
            audio, language=language, beam_size=5, temperature=0.0,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters=dict(onset=0.3, min_silence_duration_ms=500, speech_pad_ms=400))
        return list(segs), info

    def detect(audio):
        audio = audio * (0.9 / max(1e-4, float(np.max(np.abs(audio)))))
        return dict(model.detect_language(audio)[2])

    def pipeline(audio):
        t = time.time()
        segs, lang = choose_transcription(transcribe, detect, audio, 'auto')
        kept, _ = filter_segments(segs, 0.85)
        return ' '.join(kept), lang, time.time() - t

    pipeline(synth[0][2])  # warm-up
    used = before - mem_available_mb()
    result = {'model': os.path.basename(spec), 'load_s': load, 'mem_mb': used, 'synth': [], 'real': []}
    for lang, ref, audio in synth:
        hyp, got, dt = pipeline(audio)
        result['synth'].append({'lang': lang, 'ref': ref, 'hyp': hyp, 'lang_ok': got == lang,
                                'wer': wer(ref, hyp), 'sec': dt})
    for name, before_text, audio in real:
        hyp, got, dt = pipeline(audio)
        result['real'].append({'turn': name, 'robot_heard': before_text, 'hyp': hyp,
                               'lang': got, 'sec': dt})
    print('RESULT ' + json.dumps(result))


def report(results):
    for r in results:
        s = r['synth']
        by = {lang: [x for x in s if x['lang'] == lang] for lang in ('en', 'pl')}
        print('\n=== {model}: load {load_s:.1f}s, memory +{mem_mb} MB'.format(**r))
        for lang, xs in by.items():
            print('  {}: WER {:.0%}, language right {}/{}, {:.2f}s per turn'.format(
                lang, np.mean([x['wer'] for x in xs]), sum(x['lang_ok'] for x in xs),
                len(xs), np.mean([x['sec'] for x in xs])))
        for x in s:
            if x['wer'] > 0:
                print('    {:.0%}  {!r}\n         -> {!r}'.format(x['wer'], x['ref'], x['hyp']))
    if results and results[0]['real']:
        print('\n=== real recordings (robot heard -> each model)')
        for i, turn in enumerate(results[0]['real']):
            print('  {}  robot: {!r}'.format(turn['turn'], turn['robot_heard']))
            for r in results:
                x = r['real'][i]
                print('    {:15} [{}] {!r} ({:.1f}s)'.format(r['model'], x['lang'], x['hyp'], x['sec']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('models', nargs='+')
    ap.add_argument('--real', default='/tmp/stt_turns', help="stt_node debug_audio_dir ('' = skip)")
    ap.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        run_model(args.models[0], args.real)
        return
    results = []
    for size in args.models:
        print('running', size, '...', flush=True)
        out = subprocess.run([sys.executable, __file__, size, '--one', '--real', args.real],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
        line = [ln for ln in out.splitlines() if ln.startswith('RESULT ')]
        if not line:
            print('  failed:', out[-500:])
            continue
        results.append(json.loads(line[0][7:]))
    report(results)


if __name__ == '__main__':
    main()
