#!/usr/bin/env python3
"""
Compare Ollama models with the robot's real prompt (voice_params_real.yaml).

    python3 scripts/bench_llm.py llama3.2 hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q4_K_M

Each question is asked RUNS times as a fresh conversation (max_history_length: 1)
and checked automatically: correct content, reply language, refusing physical
actions, asking to repeat garbled input. Also measures time to first token,
generation speed and the model's memory (ollama ps). Every model is unloaded
afterwards, so only one is resident at a time.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'g1_voice'))

OLLAMA = 'http://localhost:11434'
STATE = """[ROBOT STATE]
Battery: 80%, not charging, health 98%
Orientation: roll=0.3 deg pitch=-1.2 deg yaw=45.0 deg
Motion mode: idle
Network: robot bus 192.168.123.164 (eth0), wifi 192.168.1.115
Uptime: 0h 47m
Software: Ollama, faster-whisper, Piper TTS (all on-robot)
[/ROBOT STATE]"""
REFUSE = (r"nie mogę|nie potrafię|nie jestem w stanie|nie mam możliwości"
          r"|can't|cannot|not able|unable|not capable")
REPEAT = (r"powt[óo]rz|nie rozumiem|nie zrozumia|niezrozumia|repeat"
          r"|didn't (catch|understand)|not sure what|unclear|clarify|rephrase")
CASES = [
    ('en', 'Aletta, what is two times two?', r'\b4\b|four'),
    ('en', 'What is three times seven?', r'\b21\b|twenty-one'),
    ('en', 'What is the capital of France?', r'paris'),
    ('en', 'What is your battery level?', r'\b80\b|eighty'),
    ('en', 'Aletta, move the box to the left.', REFUSE),
    ('en', 'is free the window seven', REPEAT),
    ('en', 'Who are you?', r'aletta'),
    ('pl', 'Aletta ile to jest 2x2?', r'\b4\b|cztery'),
    ('pl', 'Ile to jest trzy razy siedem?', r'\b21\b|dwadzieścia jeden'),
    ('pl', 'Jaka jest stolica Francji?', r'paryż'),
    ('pl', 'Jaki jest poziom baterii?', r'\b80\b|osiemdziesiąt'),
    ('pl', 'Aletta, przesuń pudełko w lewo.', REFUSE),
    ('pl', 'I na peryski 2x2', REPEAT),
    ('pl', 'Kim jesteś?', r'aletta'),
]


_PL = set('jest nie się na w z jestem jesteś proszę mogę czy co jak że ale do od dla moja mój mam to'.split())
_EN = set('the is i you a an of to my am not can are it this that what please your and'.split())


def reply_language(text):
    """Guess 'pl'/'en' from function words (diacritics break ties)."""
    words = re.findall(r'[^\W_]+', text.lower())
    pl = sum(w in _PL for w in words) + 2 * any(c in 'ąćęłńóśźż' for c in text.lower())
    en = sum(w in _EN for w in words)
    return 'pl' if pl > en else 'en'


def chat(model, system, user, keep_alive='5m', max_tokens=90):
    body = {'model': model, 'stream': True, 'keep_alive': keep_alive,
            'options': {'temperature': 0.4, 'num_predict': max_tokens, 'num_ctx': 2048},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    req = urllib.request.Request(OLLAMA + '/api/chat', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0, first, text, stats = time.time(), None, '', {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            msg = json.loads(line)
            piece = msg.get('message', {}).get('content', '')
            if piece and first is None:
                first = time.time() - t0
            text += piece
            if msg.get('done'):
                stats = msg
    tps = stats.get('eval_count', 0) / max(1e-9, stats.get('eval_duration', 1) / 1e9)
    return text.strip().replace('\n', ' '), first or 0.0, time.time() - t0, tps


def prompt(lang, text):
    name = {'pl': 'Polish', 'en': 'English'}[lang]
    return ('{}\n\nLATEST USER TURN LANGUAGE: {} ({}). Reply in {} for this turn, unless the '
            'user explicitly asks for another language.\n\nUser: {}').format(STATE, name, lang, name, text)


def resident_mb(model):
    out = subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        if line.split() and line.split()[0] in (model, model + ':latest'):
            m = re.search(r'([\d.]+) (GB|MB)', line)
            return float(m.group(1)) * (1024 if m.group(2) == 'GB' else 1) if m else 0
    return 0


def bench(model, system, runs):
    print('\n=== ' + model, flush=True)
    t = time.time()
    chat(model, system, 'Hi', max_tokens=1)  # load
    print('  load {:.1f}s, resident {:.0f} MB'.format(time.time() - t, resident_mb(model)))
    score = {'en': [0, 0], 'pl': [0, 0]}
    lang_ok = {'en': [0, 0], 'pl': [0, 0]}
    firsts, speeds = [], []
    for lang, text, expect in CASES:
        for i in range(runs):
            reply, first, total, tps = chat(model, system, prompt(lang, text))
            ok = bool(re.search(expect, reply.lower()))
            right_lang = reply_language(reply) == lang
            score[lang][0] += ok
            score[lang][1] += 1
            lang_ok[lang][0] += right_lang
            lang_ok[lang][1] += 1
            firsts.append(first)
            speeds.append(tps)
            if i == 0 or not ok:
                print('  {} {} {!r:38} -> {}'.format('OK ' if ok else 'BAD', lang, text, reply[:100]))
    print('  correct: en {}/{}, pl {}/{} | reply language right: en {}/{}, pl {}/{}'.format(
        *score['en'], *score['pl'], *lang_ok['en'], *lang_ok['pl']))
    print('  first token median {:.2f}s, {:.1f} tok/s'.format(
        sorted(firsts)[len(firsts) // 2], sorted(speeds)[len(speeds) // 2]))
    chat(model, system, 'Hi', keep_alive=0, max_tokens=1)  # unload


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('models', nargs='+')
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO, 'src/g1_voice/config/voice_params_real.yaml')))
    system = cfg['/llm_node']['ros__parameters']['system_prompt']
    for model in args.models:
        bench(model, system, args.runs)


if __name__ == '__main__':
    main()
