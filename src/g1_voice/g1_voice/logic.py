"""
ROS-free logic shared by the g1_voice nodes.

Everything here is pure Python (numpy optional) so it is unit-tested directly
without a ROS install:

* language markers carried from STT to TTS,
* sentence splitting for streamed TTS,
* Whisper hallucination / no-speech filtering and the pl/en language duel,
* end-of-turn detection and trimming the spoken name off a turn,
* the echo guard (our TTS and anything else on the G1 speaker),
* the arbiter between Aletta and Unitree's built-in GPT assistant,
* the wake-word conversation ``Session`` state machine.
"""

import fcntl
import json
import re
import socket
import struct
import time
import unicodedata
from collections import Counter


ROBOT_IP = '192.168.123.164'  # this Jetson's address on the G1 internal network


def interface_ip(name):
    """IPv4 address of network interface ``name``, or None."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack('256s', name[:15].encode())
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def interface_for_ip(ip=ROBOT_IP):
    """
    Name of the interface that holds ``ip`` (e.g. 'eth1'), or None.

    eth0/eth1 names are not stable on this Jetson (a USB Ethernet dongle can
    take eth0), so everything robot-side is anchored to the robot IP instead.
    """
    for _, name in socket.if_nameindex():
        if interface_ip(name) == ip:
            return name
    return None


def open_multicast_socket(group, port, local_ip, timeout=0.1):
    """
    Join the G1 microphone multicast stream.

    SO_REUSEADDR (not SO_REUSEPORT) lets wake_word_node and stt_node bind the
    same port and *both* receive every multicast datagram.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', port))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                        struct.pack('4s4s', socket.inet_aton(group),
                                    socket.inet_aton(local_ip)))
        sock.settimeout(timeout)
    except Exception:
        sock.close()
        raise
    return sock


# --------------------------------------------------------------------------
# Language marker: STT prefixes every transcript with "[USER_LANGUAGE:xx]\n"
# so robot_state_node can instruct the LLM and tts_node can pick the voice.
# --------------------------------------------------------------------------

LANGUAGES = ('en', 'pl')
_MARKER_RE = re.compile(r'^\s*\[USER_LANGUAGE:(pl|en)\]\s*\n?', re.IGNORECASE)


def tag_language(text, language):
    """Prefix ``text`` with the language marker."""
    return '[USER_LANGUAGE:{}]\n{}'.format(language, text)


def split_language(text):
    """Return ``(language or None, text without marker)``."""
    match = _MARKER_RE.match(text or '')
    if not match:
        return None, (text or '').strip()
    return match.group(1).lower(), text[match.end():].strip()


_POLISH_CHARS = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')
# Words that exist only in Polish (no English homographs like 'to', 'robot').
_POLISH_WORDS = {
    'cześć', 'dzień', 'dobry', 'dziękuję', 'proszę', 'jestem', 'jesteś',
    'możesz', 'mogę', 'pomóc', 'się', 'niech', 'mój', 'moja', 'moje',
    'twoje', 'czy', 'dlaczego', 'ponieważ', 'witaj', 'przepraszam',
}


def is_polish_text(text):
    """Heuristic: diacritics, or at least two Polish-only words."""
    if any(ch in _POLISH_CHARS for ch in text):
        return True
    tokens = set(re.findall(r'[^\W_]+', text.lower()))
    return len(tokens & _POLISH_WORDS) >= 2


# --------------------------------------------------------------------------
# Sentence splitting for incremental (streamed) TTS.
# --------------------------------------------------------------------------

_TERMINATORS = '.!?;'
_SPACE = ' \t\n\r'


def split_complete_sentences(buffer):
    """
    Split text into complete sentences plus the unterminated remainder.

    A sentence ends at ``. ! ? ;`` followed by whitespace or end of buffer,
    so a streamed chunk is never spoken mid-word.
    Returns ``(sentences, remainder)``.
    """
    buffer = (buffer or '').strip()
    sentences = []
    start = 0
    n = len(buffer)
    for i, ch in enumerate(buffer):
        if ch in _TERMINATORS and (i + 1 >= n or buffer[i + 1] in _SPACE):
            piece = buffer[start:i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    return sentences, buffer[start:].strip()


_CLAUSE_MARKS = ',:'


def split_speakable(buffer, first_clause_words=5):
    """
    Like ``split_complete_sentences``, but the first piece may end early.

    The first piece ends at the first ``,`` or ``:`` followed by whitespace
    once at least ``first_clause_words`` words precede it, if that comes
    before the first sentence end. Speaking that clause while the LLM still
    streams the rest starts the reply ~0.5-1 s sooner. The split depends only
    on the text before it, so it is stable while the buffer grows.
    """
    buffer = (buffer or '').strip()
    if first_clause_words > 0:
        for i, ch in enumerate(buffer):
            if ch in _TERMINATORS and (i + 1 >= len(buffer) or buffer[i + 1] in _SPACE):
                break
            if (ch in _CLAUSE_MARKS and i + 1 < len(buffer) and buffer[i + 1] in _SPACE
                    and len(buffer[:i].split()) >= first_clause_words):
                rest, remainder = split_complete_sentences(buffer[i + 1:])
                return [buffer[:i + 1]] + rest, remainder
    return split_complete_sentences(buffer)


# --------------------------------------------------------------------------
# Whisper output filtering.
# --------------------------------------------------------------------------

def _score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def keep_segment(no_speech_prob, max_no_speech_prob=0.5, avg_logprob=None,
                 min_avg_logprob=None, noisy_no_speech_prob=None,
                 noisy_min_avg_logprob=None):
    """
    Return True when a Whisper segment is trustworthy speech.

    Dropped when (each rule only if its threshold is set):

    * ``no_speech_prob`` > ``max_no_speech_prob``;
    * ``avg_logprob`` < ``min_avg_logprob``: on the G1 mic real turns decode
      at -0.66 to -0.88, invented ones ("God bless you all, God bless you
      all", "Thank you, speak with knowledge") at -1.01 to -1.22;
    * ``no_speech_prob`` > ``noisy_no_speech_prob`` and ``avg_logprob`` <
      ``noisy_min_avg_logprob`` (Whisper's own combined rule, tuned on the
      G1): distant speech and noise decoded as "That's the end of the
      episode..." at no_speech 0.58-0.65 / logprob -0.79 to -0.93, while
      real turns scored no_speech 0.04-0.27.

    A missing score never drops a segment.
    """
    nsp, logprob = _score(no_speech_prob), _score(avg_logprob)
    if nsp is not None and nsp > float(max_no_speech_prob):
        return False
    if min_avg_logprob is not None and logprob is not None and logprob < min_avg_logprob:
        return False
    if (noisy_no_speech_prob is not None and noisy_min_avg_logprob is not None
            and nsp is not None and logprob is not None
            and nsp > noisy_no_speech_prob and logprob < noisy_min_avg_logprob):
        return False
    return True


def filter_segments(segments, max_no_speech_prob=0.5, min_avg_logprob=None,
                    noisy_no_speech_prob=None, noisy_min_avg_logprob=None):
    """Return ``(kept_texts, dropped)``; dropped is ``[(no_speech_prob, text)]``."""
    kept, dropped = [], []
    for seg in segments:
        text = (seg.text or '').strip()
        nsp = getattr(seg, 'no_speech_prob', None)
        if not keep_segment(nsp, max_no_speech_prob, getattr(seg, 'avg_logprob', None),
                            min_avg_logprob, noisy_no_speech_prob, noisy_min_avg_logprob):
            try:
                score = float(nsp)
            except (TypeError, ValueError):
                score = 0.0
            dropped.append((score, text))
        elif text:
            kept.append(text)
    return kept, dropped


# Whisper learned these from subtitled videos and "hears" them in noise.
_SUBTITLE_PHRASES = (
    'thanks for watching', 'thank you for watching', 'subscribe',
    'subtitles by', 'transcribed by', 'amara.org', 'napisy stworzone',
    'napisy wykonane', 'dzięki za obejrzenie',
    # Heard on the G1 in noise or on clipped turns (21 Nov 2026).
    'next video', 'this video', 'my channel', 'see you next time',
    'end of the episode', 'see you in the next one', 'thank you for joining us',
)


def is_latin_text(text):
    """Return True when most letters are Latin script (English/Polish)."""
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(1 for ch in letters if 'LATIN' in unicodedata.name(ch, ''))
    return bool(letters) and latin >= 0.8 * len(letters)


def is_only_name(text, name='aletta'):
    """Return True when the transcript is just the robot's name (no question)."""
    words = re.findall(r'[^\W_]+', (text or '').lower())
    return bool(words) and all(w == name for w in words)


def looks_like_noise(text):
    """
    Detect Whisper hallucination loops so they never reach the LLM.

    Flags: no letters at all, a phrase repeated 3+ times back-to-back that
    dominates the text (people do repeat a question twice), one word making up
    half of a 4+ word text, a word repeated 3x in a row, a syllable looped
    inside one word ("lalala"), 3+ sentences that are mostly repeats,
    non-Latin script, or a known subtitle phrase ("Thanks for watching").
    Unicode-aware, so Polish diacritics are letters.
    """
    t = (text or '').strip()
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", t.lower())
    if not words or not is_latin_text(t):
        return True  # empty, or e.g. the Japanese "thanks for watching" subtitle
    if any(phrase in t.lower() for phrase in _SUBTITLE_PHRASES):
        return True

    n = len(words)
    for p in range(2, n // 3 + 1):
        for i in range(0, n - 3 * p + 1):
            unit = words[i:i + p]
            if unit != words[i + p:i + 2 * p] or unit != words[i + 2 * p:i + 3 * p]:
                continue
            j = i + 3 * p
            while words[j:j + p] == unit:
                j += p
            if (j - i) / n >= 0.70:
                return True

    if n >= 4 and Counter(words).most_common(1)[0][1] / n >= 0.5:
        return True

    if any(words[i] == words[i + 1] == words[i + 2] for i in range(n - 2)):
        return True

    for word in words:
        if len(word) >= 6:
            for unit_len in range(1, len(word) // 3 + 1):
                if word[:unit_len * 3] == word[:unit_len] * 3:
                    return True

    sentences = [s.strip().lower() for s in re.split(r'[.!?;]+', t) if s.strip()]
    if len(sentences) >= 3:
        repeated = sum(c for c in Counter(sentences).values() if c > 1)
        if repeated / len(sentences) >= 0.5:
            return True
    return False


def segments_text(segments):
    return ' '.join((s.text or '').strip() for s in segments).strip()


def transcription_score(segments):
    """Mean avg_logprob with a small bonus for longer text (max +0.08)."""
    text = segments_text(segments)
    if not text:
        return float('-inf')
    logprobs = [getattr(s, 'avg_logprob', -10.0) for s in segments]
    return sum(logprobs) / len(logprobs) + min(len(text), 80) * 0.001


def pick_language(probs, previous=None, en_min=0.25, en_ratio=10.0,
                  pl_min=0.2, pl_ratio=2.0):
    """
    Choose 'en' or 'pl' from Whisper's language probabilities, or None.

    Whisper detects English reliably on the G1 mic (p 0.27-0.97, Polish
    <= 0.01 on real turns) but Polish poorly: short Polish phrases came out
    as hi/zh/en with p(pl) 0.05-0.06. So English needs to be clearly ahead
    of Polish, Polish needs to lead English, and anything else falls back to
    ``previous`` (the language of the last turn in this session), or None
    when there is none: the caller then decodes both languages.
    """
    en, pl = probs.get('en', 0.0), probs.get('pl', 0.0)
    if en >= en_min and en >= en_ratio * pl:
        return 'en'
    if pl >= pl_min and pl >= pl_ratio * en:
        return 'pl'
    return previous if previous in LANGUAGES else None


def choose_transcription(transcribe, detect, audio, language='auto',
                         previous=None, log=None):
    """
    Transcribe ``audio`` in Polish or English with as few decodes as possible.

    ``transcribe(audio, language) -> (segments, info)``;
    ``detect(audio) -> {language: probability}``.

    * A fixed ``language`` decodes once.
    * Otherwise one language-ID pass (~0.5 s) and ``pick_language`` decide,
      then one decode (~0.65 s).
    * Only when that is ambiguous and there is no session language, both
      languages are decoded and the better-scoring non-hallucination wins.

    Returns ``(segments, language)``.
    """
    log = log or (lambda msg: None)
    if language not in ('', 'auto'):
        segments, _ = transcribe(audio, language)
        return segments, language

    probs = detect(audio)
    chosen = pick_language(probs, previous)
    summary = 'en={:.2f} pl={:.2f}'.format(probs.get('en', 0.0), probs.get('pl', 0.0))
    if chosen is not None:
        from_session = pick_language(probs) is None  # unclear: session language used
        log('Language {} ({}{})'.format(chosen, summary, ', session' if from_session else ''))
        segments, _ = transcribe(audio, chosen)
        return segments, chosen

    candidates = []
    for lang in LANGUAGES:
        segments, _ = transcribe(audio, lang)
        noise = looks_like_noise(segments_text(segments))
        candidates.append((not noise, transcription_score(segments), lang, segments))
    _, _, lang, segments = max(candidates, key=lambda c: (c[0], c[1]))
    log('Language duel ({}) -> {}'.format(summary, lang))
    return segments, lang


# --------------------------------------------------------------------------
# Utterance endpointing for one STT turn.
# --------------------------------------------------------------------------

class Endpointer:
    """
    Decide when a recording turn is complete, from per-chunk RMS.

    * Speech starts after ``min_speech`` s of median-smoothed RMS at or above
      ``onset_threshold`` (default ``threshold``; one loud click is not
      speech); ``onset`` is its sample index. A higher onset level keeps room
      noise and the robot's fading echo from starting a turn, while the
      lower ``threshold`` still hears quiet word endings.
    * No speech within ``max_wait`` s: stop (an empty turn).
    * After speech: stop on ``silence`` s of trailing quiet, or
      ``max_utterance`` s after the onset. The limit counts from when the
      user starts talking, so a question that begins late is never cut off.
    """

    def __init__(self, rate, threshold, max_utterance=10.0, max_wait=8.0,
                 silence=2.5, min_speech=0.2, smooth=5, onset_threshold=None):
        self.rate = rate
        self.threshold = threshold
        self.onset_threshold = max(threshold, onset_threshold or threshold)
        self.max_utterance = int(max_utterance * rate)
        self.max_wait = int(max_wait * rate)
        self.silence_limit = int(silence * rate)
        self.min_speech = int(min_speech * rate)
        self.smooth = smooth
        self.total = 0
        self.onset = None
        self._rms = []
        self._speech_run = 0
        self._silence = 0
        self.last_speech = 0  # sample index where speech was last heard

    def trailing_silence(self):
        """Seconds of audio after the last speech (the end-of-turn wait)."""
        return (self.total - self.last_speech) / self.rate if self.onset is not None else 0.0

    def feed(self, rms, n_samples):
        """Add one chunk; return False when the turn should end."""
        self.total += n_samples
        self._rms = (self._rms + [rms])[-self.smooth:]
        level = self.threshold if self.onset is not None else self.onset_threshold
        if sorted(self._rms)[len(self._rms) // 2] >= level:
            self._speech_run += n_samples
            self._silence = 0
            self.last_speech = self.total
            if self.onset is None and self._speech_run >= self.min_speech:
                self.onset = self.total - self._speech_run
        else:
            self._speech_run = 0
            if self.onset is not None:
                self._silence += n_samples
                if self._silence >= self.silence_limit:
                    return False
        if self.onset is None:
            # Never give up while sound is coming in: confirming speech
            # through the smoothing window takes ~0.5 s.
            return (self.total < self.max_wait or rms >= self.onset_threshold
                    or self._speech_run > 0)
        return self.total - self.onset < self.max_utterance


def snap_cut_to_pause(audio, rate, cut, threshold, frame_sec=0.05, min_gap=0.1,
                      max_back=1.0):
    """
    Move a name cut back to the end of a real pause, or None for no cut.

    The wake model's score peaks 0.3-1.0 s after "Aletta", so ``cut`` (from
    the detection time) can land inside the question when people don't pause
    (4 of 4 recorded cuts did). Returns the end of the latest quiet gap of at
    least ``min_gap`` s (every ``frame_sec`` frame RMS below ``threshold``)
    that ends at or before ``cut`` and no more than ``max_back`` s before it,
    so the question always starts after the cut; None when there is no such
    pause (better to keep the name than to lose words).
    """
    step = max(1, int(frame_sec * rate))
    n = len(audio) // step
    if n == 0:
        return None
    frames = audio[:n * step].reshape(n, step).astype('float64')
    quiet = (frames ** 2).mean(axis=1) ** 0.5 < threshold
    need = max(1, int(round(min_gap / frame_sec)))
    end = min(int(round(cut / frame_sec)), n)
    lowest = max(need, end - int(round(max_back / frame_sec)))
    for e in range(end, lowest - 1, -1):
        if quiet[e - need:e].all():
            return e * frame_sec
    return None


def echoes_prompt(text, prompt):
    """
    Tell whether a transcript only repeats words of the Whisper ``prompt``.

    With a vocabulary hint Whisper can "hear" the hint itself on a
    near-silent turn ("Unitree G1, battery."). A real question has words of
    its own (at least two not in the prompt, or half of its words).
    """
    words = re.findall(r"[^\W_]+", (text or '').lower())
    vocab = set(re.findall(r"[^\W_]+", (prompt or '').lower()))
    if not words or not vocab:
        return False
    own = [w for w in words if w not in vocab]
    return len(own) < 2 and len(own) < 0.5 * len(words)


def has_speech(audio, rate, threshold, frame_sec=0.1, onset_threshold=None):
    """Return True when ``audio`` (float32) holds speech by ``Endpointer``'s rule."""
    endpoint = Endpointer(rate, threshold, max_wait=len(audio) / rate + 1.0,
                          onset_threshold=onset_threshold)
    step = max(1, int(frame_sec * rate))
    for i in range(0, len(audio), step):
        frame = audio[i:i + step]
        endpoint.feed(float((frame.astype('float64') ** 2).mean()) ** 0.5, frame.size)
        if endpoint.onset is not None:
            return True
    return False


def name_cut(audio_start, duration, detections, onset=0.0, offset=0.2,
             max_after_onset=1.5):
    """
    Seconds into a turn's audio where a spoken "Aletta" ends, or None.

    In a session people start every question with the name. Whisper mishears
    it ("Aletta, what is" -> "All that for both is") and the garbage becomes
    part of the question, so the audio before the name's end is cut off.

    ``detections`` are wake-word wall times; the detector fires ``offset`` s
    or more after the word ends. Only the first detection inside the audio
    (which starts at wall time ``audio_start`` and lasts ``duration`` s)
    counts, and only while it is within ``max_after_onset`` s of the speech
    onset (``onset`` s into the audio): the name opens the question, and a
    later detection is the model misfiring on the question itself.
    """
    inside = sorted(w for w in detections if audio_start < w <= audio_start + duration)
    if not inside:
        return None
    cut = max(0.0, inside[0] - offset - audio_start)
    if cut - onset > max_after_onset:
        return None
    return min(cut, duration)


# --------------------------------------------------------------------------
# Echo guard: never listen to the robot's own speaker.
# --------------------------------------------------------------------------

def board_play_state(text):
    """
    Speaker state from the G1 audio board's ``/audio_msg``, or None.

    The board publishes ``{"play_state": 1}`` when anything starts playing on
    the speaker and ``{"play_state": 0}`` when it stops: our PlayStream, but
    also Unitree's built-in assistant (its male English voice), which we
    cannot switch off. Other messages on the topic are its own ASR results.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or 'play_state' not in data:
        return None
    return bool(data['play_state'])


class EchoGuard:
    """
    Tell whether the speaker plays, or stopped less than ``guard`` s ago.

    Several sources report playback (``tts`` = our /g1/tts/status, ``board``
    = the audio board's play_state). A "playing" report expires after
    ``max_playing`` s, so a lost "stopped" message can't mute the
    microphone forever. Not thread-safe: callers serialize access.
    """

    def __init__(self, guard=2.5, max_playing=60.0, clock=time.monotonic):
        self.guard = guard
        self.max_playing = max_playing
        self.clock = clock
        self._playing_since = {}
        self._quiet_since = -1e9

    def set(self, source, playing):
        now = self.clock()
        if playing:
            self._playing_since.setdefault(source, now)
        elif self._playing_since.pop(source, None) is not None:
            self._quiet_since = now

    def playing(self, source=None):
        now = self.clock()
        for name, since in list(self._playing_since.items()):
            if now - since >= self.max_playing:
                del self._playing_since[name]  # stale: "stopped" was lost
                self._quiet_since = now
        if source is not None:
            return source in self._playing_since
        return bool(self._playing_since)

    def active(self):
        return self.playing() or self.clock() - self._quiet_since < self.guard

    def quiet_for(self):
        """Seconds since the speaker last stopped, or None while it plays."""
        if self.playing():
            return None
        return self.clock() - self._quiet_since


# --------------------------------------------------------------------------
# Aletta vs. Unitree's built-in GPT assistant (service "chat_go").
# --------------------------------------------------------------------------

ASSISTANT_POLICIES = ('off', 'auto', 'leave')


def network_online(text):
    """
    Robot internet state from ``/public_network_status``, or None.

    The robot publishes e.g. ``{"network_status": "NetworkStatus.ON_WIFI_CONNECTED"}``.
    Any ``..._CONNECTED`` state counts as online.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    status = data.get('network_status') if isinstance(data, dict) else None
    if not isinstance(status, str):
        return None
    status = status.upper()
    return status.endswith('_CONNECTED') and 'DISCONNECTED' not in status


def gpt_state(text):
    """
    Parse ``/gpt_state`` (published every ~5 s while chat_go runs), or None.

    E.g. ``{"state": true, "llm_name": "clound", "chat_state": true}``;
    ``chat_state`` is not an on/off switch (chat_go answers while it is false).
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class AssistantArbiter:
    """
    Decide who answers: Aletta or Unitree's GPT assistant.

    Policies:

    * ``off``: Aletta always; Unitree's assistant is kept switched off.
    * ``auto``: Unitree's assistant is the default while the robot has been
      online for ``online_stable`` s. Saying "Aletta" opens an Aletta session,
      which takes over at once (chat_go is switched off so it can't answer
      too); when the session closes, Unitree's assistant is the default again.
      Going offline (or no network report for ``network_timeout`` s) hands
      back to Aletta at once. While the network state is still undecided
      (just came online, or no report yet after start) the service is left
      as it is, so it is never switched off and straight back on.
      ``chat_state`` in /gpt_state is NOT a "voice chat enabled" flag: on
      21 Nov chat_go answered six questions in a row while it was false, so
      it is only logged, never used to decide.
    * ``leave``: Aletta always; Unitree's service is never touched.

    ``gpt_running`` follows ``/gpt_state``, which stops within
    ``gpt_timeout`` s of the service being switched off.
    Not thread-safe: callers serialize access.
    """

    def __init__(self, policy='off', online_stable=10.0, network_timeout=30.0,
                 gpt_timeout=12.0, clock=time.monotonic):
        if policy not in ASSISTANT_POLICIES:
            raise ValueError('unitree_assistant must be one of {}'.format(
                ', '.join(ASSISTANT_POLICIES)))
        self.policy = policy
        self.online_stable = online_stable
        self.network_timeout = network_timeout
        self.gpt_timeout = gpt_timeout
        self.clock = clock
        self._network_at = None    # last network report
        self._online_since = None  # start of the current online stretch
        self._gpt_at = None        # last /gpt_state
        self.chat_state = None
        self.session_open = False
        self._mode = 'aletta'
        self._started = clock()
        self.forced_aletta = False  # online, switched to Aletta with F1

    def on_network(self, online):
        now = self.clock()
        self._network_at = now
        if not online:
            self._online_since = None
        elif self._online_since is None:
            self._online_since = now

    def on_gpt_state(self, state):
        self._gpt_at = self.clock()
        self.chat_state = bool(state.get('chat_state')) if state else None

    def network(self):
        """'online' (stable), 'pending' (undecided) or 'offline'."""
        now = self.clock()
        if self._network_at is None:
            return 'pending' if now - self._started < self.network_timeout else 'offline'
        if now - self._network_at > self.network_timeout or self._online_since is None:
            return 'offline'
        return 'online' if now - self._online_since >= self.online_stable else 'pending'

    def online(self):
        return self.network() == 'online'

    def gpt_running(self):
        return self._gpt_at is not None and self.clock() - self._gpt_at <= self.gpt_timeout

    def mode(self):
        """'aletta' or 'unitree': an open Aletta session or F1 means Aletta."""
        if self.network() == 'offline':
            self.forced_aletta = False  # back online -> Unitree by default
        if (self.policy != 'auto' or not self.online() or self.session_open
                or self.forced_aletta):
            self._mode = 'aletta'
        else:
            self._mode = 'unitree'
        return self._mode

    def toggle(self):
        """
        F1 while online: Unitree -> Aletta, or back to Unitree.

        Returns the new mode, or None when F1 has no switching role (robot
        offline, or policy not ``auto``).
        """
        if self.policy != 'auto' or not self.online():
            return None
        # Decide by the F1 state, not by the current mode: with an Aletta
        # conversation still open the mode is already 'aletta', and F1 must
        # still mean "stay with Aletta" (the session ending must not hand
        # back to Unitree; seen 21 Nov 20:48).
        self.forced_aletta = not self.forced_aletta
        return 'aletta' if self.forced_aletta else 'unitree'

    def wanted_gpt(self):
        """True/False: chat_go should run / be off; None: leave it alone."""
        if self.policy == 'leave':
            return None
        if self.policy == 'off':
            return False
        if self.session_open or self.forced_aletta:
            return False  # Aletta is talking: chat_go must not answer too
        if self.mode() == 'unitree':
            return True
        if self.network() == 'offline':
            return False
        return None  # undecided, or online but an Aletta session is still open


# --------------------------------------------------------------------------
# Wake-word conversation session.
# --------------------------------------------------------------------------

class Session:
    """
    Conversation window opened by "Aletta".

    Closes on START, ``max_empty_turns`` consecutive turns without a
    transcript, or ``max_duration`` (deferred while a turn is in flight).
    A wake-opened session that produced no transcript at all counts as a
    false accept and multiplies the re-open cooldown (capped); any real
    transcript resets the cooldown.

    Pipeline busy flags (``stt``, ``reply``, ``tts``) expire after
    ``busy_timeout`` so a lost status message can never wedge the session.
    Not thread-safe: callers serialize access.
    """

    BUSY = ('stt', 'reply', 'tts')

    def __init__(self, max_duration=300.0, max_empty_turns=4,
                 base_cooldown=4.0, backoff=1.5, max_cooldown=30.0,
                 busy_timeout=60.0, clock=time.monotonic):
        self.max_duration = max_duration
        self.max_empty_turns = max_empty_turns
        self.base_cooldown = base_cooldown
        self.backoff = backoff
        self.max_cooldown = max_cooldown
        self.busy_timeout = busy_timeout
        self.clock = clock
        self.is_open = False
        self.opened_at = 0.0
        self.empty_turns = 0
        self.had_transcript = False
        self.cooldown = base_cooldown
        self.false_accepts = 0
        self.last_open = -1e9
        self.last_close = -1e9
        self._busy_since = {}

    def set_busy(self, name, busy):
        if busy:
            self._busy_since[name] = self.clock()
        else:
            self._busy_since.pop(name, None)

    def busy(self):
        now = self.clock()
        return any(now - since < self.busy_timeout
                   for since in self._busy_since.values())

    def can_open(self):
        # Counted from the close: a false session (4 empty turns) lasts ~31 s,
        # longer than the 30 s cooldown cap, so counting from the open made
        # the backoff a no-op (sessions reopened 2-3 s after closing).
        return (not self.is_open and not self.busy()
                and self.clock() - self.last_close >= self.cooldown)

    def open(self):
        self.is_open = True
        self.opened_at = self.last_open = self.clock()
        self.empty_turns = 0
        self.had_transcript = False

    def close(self):
        """Close the window. Returns True if it was open."""
        if not self.is_open:
            return False
        self.last_close = self.clock()
        self.is_open = False
        if self.had_transcript:
            self.false_accepts = 0
            self.cooldown = self.base_cooldown
        else:
            self.false_accepts += 1
            self.cooldown = min(
                self.base_cooldown * self.backoff ** self.false_accepts,
                self.max_cooldown)
        return True

    def on_transcript(self):
        self.empty_turns = 0
        self.had_transcript = True

    def on_empty_turn(self):
        """Count an empty turn; True when the session should close."""
        if not self.is_open:
            return False
        self.empty_turns += 1
        return self.empty_turns >= self.max_empty_turns

    def expired(self):
        if not self.is_open:
            return False
        age = self.clock() - self.opened_at
        if age >= self.max_duration + self.busy_timeout:
            return True
        return age >= self.max_duration and not self.busy()


# --------------------------------------------------------------------------
# GPU / Memory Health Monitoring
# --------------------------------------------------------------------------
# When running concurrently with other autonomous applications (navigation,
# perception, etc.), GPU VRAM may be shared. This module provides a cheap
# check that can be called at node init to decide whether to force CPU
# fallback paths or raise a warning.

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def check_gpu_pressure(free_gb_threshold=3.0):
    """
    Return (is_pressure, free_gb, total_gb).

    If CUDA is available but free VRAM < threshold GB, we consider this
    "pressure" and recommend CPU-only mode for voice components.

    Returns (False, total, total) if torch or CUDA is unavailable.
    """
    if not _HAS_TORCH:
        return False, 0.0, 0.0
    if not torch.cuda.is_available():
        return False, 0.0, 0.0

    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    if free_gb < free_gb_threshold:
        return True, free_gb, total_gb

    return False, free_gb, total_gb


def log_gpu_pressure(node_name, free_gb, total_gb):
    """Log a warning if GPU VRAM is under pressure."""
    if free_gb is None:
        return
    if free_gb < 3.0:
        print(f"[GPU Pressure] [{node_name}] Low GPU VRAM: {free_gb:.1f}/{total_gb:.1f} GB free. "
              "Consider CPU fallback for STT/TTS to avoid OOM crashes.")


# Nodes launched by voice_real.launch.py.
VOICE_NODES = ('wake_word_node', 'stt_node', 'robot_state_node', 'llm_node',
               'tts_node', 'button_trigger_node', 'assistant_mode_node')

# (node, parameter, spoken label) read via each node's get_parameters service.
VOICE_PARAMETERS = (
    ('llm_node', 'api_model', 'language model'),
    ('stt_node', 'model_size', 'speech recognition model'),
    ('stt_node', 'language', 'speech recognition language'),
    ('wake_word_node', 'aletta_threshold', 'wake word threshold'),
    ('tts_node', 'tts_mode', 'speech synthesis mode'),
    ('tts_node', 'length_scale', 'speech length scale'),
)

# Word stems that mean the user asks about the software (en + pl).
_SYSTEM_STEMS = (
    'node', 'topic', 'system', 'software', 'parameter', 'setting', 'config',
    'diagnos', 'pipeline', 'status', 'model', 'ros', 'threshold', 'mode',
    # Polish ('system', 'status', 'model' above also match Polish forms).
    'węz', 'topik', 'oprogramow', 'parametr', 'ustawie', 'konfigur', 'próg',
    'tryb',
)


def asks_about_system(text):
    """Return True when the user asks about nodes, topics, settings, etc."""
    words = re.findall(r'[^\W_]+', (text or '').lower())
    return any(w.startswith(stem) for w in words for stem in _SYSTEM_STEMS)


def summarize_graph(node_names, topic_names, expected=VOICE_NODES):
    """
    Spoken-size summary of the ROS graph; names may carry a namespace.

    Returns a list of lines for the [ROBOT STATE] block.
    """
    running = {name.rsplit('/', 1)[-1] for name in node_names}
    missing = [n for n in expected if n not in running]
    if missing:
        voice = 'Voice pipeline nodes: {} of {} running, missing: {}'.format(
            len(expected) - len(missing), len(expected), ', '.join(missing))
    else:
        voice = 'Voice pipeline nodes: all {} running ({})'.format(
            len(expected), ', '.join(n.replace('_node', '') for n in expected))
    others = len(running - set(expected))
    return [voice, 'ROS 2 graph: {} nodes ({} outside the voice pipeline), {} topics'.format(
        len(running), others, len(set(topic_names)))]


def parameter_value(value):
    """Python value of an rcl_interfaces ParameterValue, or None if unset."""
    kind = getattr(value, 'type', 0)
    field = {1: 'bool_value', 2: 'integer_value', 3: 'double_value',
             4: 'string_value'}.get(kind)
    if field is None:
        return None
    result = getattr(value, field)
    return round(result, 3) if isinstance(result, float) else result


def format_local_time(t=None):
    """
    Local time for the LLM in 24 h *and* 12 h form, e.g. "20:06 (8:06 PM), ...".

    Given only "20:06" the 3B model answered "It is currently 6:06 PM"; with
    both forms it has nothing to convert.
    """
    t = time.localtime(t)
    hour12 = t.tm_hour % 12 or 12
    return '{:02d}:{:02d} ({}:{:02d} {}), {}'.format(
        t.tm_hour, t.tm_min, hour12, t.tm_min, 'AM' if t.tm_hour < 12 else 'PM',
        time.strftime('%A %d %B %Y', t))


def match_parameters(values, text, limit=10, max_len=60):
    """
    Lines ``node.param = value`` for parameters named in a question.

    ``values`` is ``{(node, param): value}`` (all parameters of the voice
    nodes). A parameter matches when a word of ``text`` (4+ letters, or a
    stem of it) appears in its name, e.g. "threshold" ->
    ``wake_word_node.aletta_threshold``, "model" -> every ``*model*``.
    Long values (the system prompt) are shortened.
    """
    words = {w for w in re.findall(r'[a-z0-9]+', (text or '').lower()) if len(w) >= 4}
    stems = {w[:-1] if w.endswith('s') else w for w in words}
    lines = []
    for (node, param), value in sorted(values.items()):
        name = param.lower()
        if value in (None, '') or not any(s in name for s in stems):
            continue
        shown = str(value)
        if len(shown) > max_len:
            shown = shown[:max_len] + '...'
        lines.append('{}.{} = {}'.format(node, param, shown))
        if len(lines) >= limit:
            break
    return lines


def format_parameters(values, spec=VOICE_PARAMETERS):
    """One line from ``{(node, param): value}``; parameters not read are skipped."""
    parts = ['{} {}'.format(label, values[(node, param)])
             for node, param, label in spec
             if values.get((node, param)) not in (None, '')]
    return 'Voice settings: ' + ', '.join(parts) if parts else None
