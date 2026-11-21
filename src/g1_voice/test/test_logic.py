"""Unit tests for the ROS-free pipeline logic (g1_voice/logic.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from g1_voice.logic import (  # noqa: E402
    AssistantArbiter, EchoGuard, Endpointer, Session, VOICE_NODES, asks_about_system,
    board_play_state, choose_transcription, echoes_prompt, filter_segments,
    format_local_time, format_parameters, gpt_state, has_speech, is_only_name, is_polish_text,
    looks_like_noise, match_parameters, name_cut, network_online, parameter_value, pick_language,
    snap_cut_to_pause, split_complete_sentences, split_language, split_speakable,
    summarize_graph, tag_language)


class Seg:
    def __init__(self, text, avg_logprob=-0.3, no_speech_prob=0.1):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class Info:
    def __init__(self, language, probability):
        self.language = language
        self.language_probability = probability


class LanguageMarkerTest(unittest.TestCase):

    def test_round_trip(self):
        self.assertEqual(split_language(tag_language('Cześć', 'pl')), ('pl', 'Cześć'))

    def test_no_marker(self):
        self.assertEqual(split_language('  hello '), (None, 'hello'))

    def test_polish_detection(self):
        self.assertTrue(is_polish_text('Dzień dobry'))
        self.assertTrue(is_polish_text('czy możesz mi pomóc'))
        self.assertFalse(is_polish_text('I am a robot, able to help you'))


class SentenceSplitTest(unittest.TestCase):

    def test_complete_and_remainder(self):
        self.assertEqual(split_complete_sentences('Hi there. How are'),
                         (['Hi there.'], 'How are'))

    def test_terminator_at_end(self):
        self.assertEqual(split_complete_sentences('One! Two?'), (['One!', 'Two?'], ''))

    def test_decimal_is_not_a_boundary(self):
        self.assertEqual(split_complete_sentences('Battery is 3.5 volts'),
                         ([], 'Battery is 3.5 volts'))

    def test_first_clause(self):
        text = 'My battery status is 30%, and I am fine. Yes, really.'
        self.assertEqual(split_speakable(text, 5),
                         (['My battery status is 30%,', 'and I am fine.', 'Yes, really.'], ''))

    def test_first_clause_needs_enough_words(self):
        self.assertEqual(split_speakable("Yes, I'm Aletta, nice", 5),
                         ([], "Yes, I'm Aletta, nice"))
        self.assertEqual(split_speakable('Hello, I am fine. Next', 5),
                         (['Hello, I am fine.'], 'Next'))

    def test_first_clause_waits_for_following_space(self):
        # A streamed "1," may continue as "1,5": split only once a space follows.
        self.assertEqual(split_speakable('The value is about 1,', 3),
                         ([], 'The value is about 1,'))
        self.assertEqual(split_speakable('The value is about 1,5 volts', 3),
                         ([], 'The value is about 1,5 volts'))

    def test_many_terminators(self):
        self.assertEqual(len(split_complete_sentences('a. ' * 500)[0]), 500)


class NoiseFilterTest(unittest.TestCase):

    def test_hallucinations(self):
        for text in ['', '...', 'sorry sorry sorry', 'Thank you very much. ' * 3,
                     ('\u3054\u8996\u8074\u3042\u308a\u304c\u3068'
                      '\u3046\u3054\u3056\u3044\u307e\u3057\u305f\u3002'),  # Japanese subtitle
                     'Thanks for watching!',
                     'Napisy stworzone przez społeczność Amara.org',
                     "I can't help it, I can't help it, I can't help it",
                     'lalalalala', 'you you you you',
                     "What's up everyone? What's up everybody? What's up everybody?"]:
            self.assertTrue(looks_like_noise(text), text)

    def test_real_speech(self):
        for text in ['What is your battery level?', 'hello there', 'yes please',
                     'Jak się masz?', 'Żółć', 'Tak. Tak.', 'banana murmur',
                     'What are you doing? What are you doing?']:
            self.assertFalse(looks_like_noise(text), text)


class LanguageChoiceTest(unittest.TestCase):

    # Probabilities measured on the G1 mic (real turns + far-field Piper speech).
    def test_pick_language_measured(self):
        cases = [
            ({'en': 0.97, 'pl': 0.00}, 'en'), ({'en': 0.29, 'pl': 0.01}, 'en'),
            ({'en': 0.27, 'pl': 0.00}, 'en'),  # detected as Portuguese overall
            ({'en': 0.08, 'pl': 0.26}, 'pl'), ({'en': 0.02, 'pl': 0.89}, 'pl'),
            ({'en': 0.12, 'pl': 0.06}, None),  # Polish heard as Hindi
            ({'en': 0.31, 'pl': 0.05}, None),  # Polish "Tak."
            ({'en': 0.07, 'pl': 0.05}, None),  # Polish heard as Chinese
        ]
        for probs, expected in cases:
            self.assertEqual(pick_language(probs), expected, probs)

    def test_ambiguous_uses_session_language(self):
        self.assertEqual(pick_language({'en': 0.12, 'pl': 0.06}, previous='pl'), 'pl')
        self.assertEqual(pick_language({'en': 0.9, 'pl': 0.0}, previous='pl'), 'en')

    def make(self, results, probs):
        calls = []

        def transcribe(audio, language):
            calls.append(language)
            return results[language], None

        def detect(audio):
            calls.append('detect')
            return probs
        return transcribe, detect, calls

    def test_fixed_language_decodes_once(self):
        transcribe, detect, calls = self.make({'en': [Seg('hi')]}, {})
        segments, lang = choose_transcription(transcribe, detect, None, 'en')
        self.assertEqual((lang, calls), ('en', ['en']))

    def test_clear_language_decodes_once(self):
        transcribe, detect, calls = self.make({'en': [Seg('hello')]}, {'en': 0.7})
        self.assertEqual(choose_transcription(transcribe, detect, None)[1], 'en')
        self.assertEqual(calls, ['detect', 'en'])

    def test_session_language_decodes_once(self):
        transcribe, detect, calls = self.make({'pl': [Seg('tak')]}, {'en': 0.3, 'pl': 0.05})
        self.assertEqual(choose_transcription(transcribe, detect, None, previous='pl')[1], 'pl')
        self.assertEqual(calls, ['detect', 'pl'])

    def test_log_marks_only_real_session_fallback(self):
        transcribe, detect, _ = self.make({'en': [Seg('hi')]}, {'en': 0.81, 'pl': 0.01})
        logs = []
        choose_transcription(transcribe, detect, None, previous='en', log=logs.append)
        self.assertEqual(logs, ['Language en (en=0.81 pl=0.01)'])
        transcribe, detect, _ = self.make({'pl': [Seg('tak')]}, {'en': 0.3, 'pl': 0.05})
        logs = []
        choose_transcription(transcribe, detect, None, previous='pl', log=logs.append)
        self.assertEqual(logs, ['Language pl (en=0.30 pl=0.05, session)'])

    def test_ambiguous_duel_prefers_better_score(self):
        transcribe, detect, calls = self.make({
            'en': [Seg('jean dobree', -0.9)], 'pl': [Seg('dzień dobry', -0.2)]},
            {'en': 0.1, 'pl': 0.06})
        segments, lang = choose_transcription(transcribe, detect, None)
        self.assertEqual((lang, segments[0].text), ('pl', 'dzień dobry'))
        self.assertEqual(calls, ['detect', 'en', 'pl'])

    def test_duel_never_picks_a_hallucination(self):
        transcribe, detect, _ = self.make({
            'en': [Seg('yes', -0.8)], 'pl': [Seg('tak tak tak tak', -0.1)]},
            {'en': 0.1, 'pl': 0.06})
        self.assertEqual(choose_transcription(transcribe, detect, None)[1], 'en')


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SessionTest(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.s = Session(max_duration=300, max_empty_turns=3, base_cooldown=4,
                         backoff=2, max_cooldown=30, busy_timeout=60,
                         clock=self.clock)

    def test_empty_turns_close_session(self):
        self.s.open()
        self.assertFalse(self.s.on_empty_turn())
        self.assertFalse(self.s.on_empty_turn())
        self.assertTrue(self.s.on_empty_turn())

    def test_transcript_resets_empty_streak(self):
        self.s.open()
        self.s.on_empty_turn()
        self.s.on_empty_turn()
        self.s.on_transcript()
        self.assertFalse(self.s.on_empty_turn())

    def test_false_accept_backoff_and_reset(self):
        self.s.open()
        self.s.close()
        self.assertEqual(self.s.cooldown, 8)
        self.clock.now += 5
        self.assertFalse(self.s.can_open())
        self.clock.now += 5
        self.s.open()
        self.s.close()
        self.assertEqual(self.s.cooldown, 16)
        self.clock.now += 20
        self.s.open()
        self.s.on_transcript()
        self.s.close()
        self.assertEqual(self.s.cooldown, 4)

    def test_cooldown_counts_from_close(self):
        # A false session runs 4 empty turns (~31 s), longer than the cap.
        for _ in range(6):
            self.s.open()
            self.clock.now += 31
            self.s.close()
            self.clock.now += 1000
        self.s.open()
        self.clock.now += 31
        self.s.close()
        self.assertEqual(self.s.cooldown, 30)
        self.clock.now += 3
        self.assertFalse(self.s.can_open())  # reopened after 2-3 s before the fix
        self.clock.now += 27
        self.assertTrue(self.s.can_open())

    def test_cooldown_capped(self):
        for _ in range(10):
            self.clock.now += 100
            self.s.open()
            self.s.close()
        self.assertEqual(self.s.cooldown, 30)

    def test_busy_blocks_open_and_expires(self):
        self.s.set_busy('stt', True)
        self.assertFalse(self.s.can_open())
        self.clock.now += 61  # a lost 'idle' status cannot wedge the session
        self.assertTrue(self.s.can_open())

    def test_duration_cap_waits_for_turn_then_forces(self):
        self.s.open()
        self.clock.now += 301
        self.s.set_busy('tts', True)
        self.assertFalse(self.s.expired())
        self.s.set_busy('tts', False)
        self.assertTrue(self.s.expired())
        self.s.set_busy('stt', True)
        self.clock.now += 59
        self.s.set_busy('stt', True)  # refreshed busy flag
        self.clock.now += 2
        self.assertTrue(self.s.expired())

    def test_close_is_idempotent(self):
        self.assertFalse(self.s.close())
        self.s.open()
        self.assertTrue(self.s.close())
        self.assertFalse(self.s.close())


class NameOnlyTest(unittest.TestCase):

    def test_name_only(self):
        self.assertTrue(is_only_name('Aletta'))
        self.assertTrue(is_only_name('Aletta, Aletta.'))
        self.assertFalse(is_only_name('Aletta, what is two times two?'))
        self.assertFalse(is_only_name(''))


class ConfidenceFilterTest(unittest.TestCase):
    """Scores of real G1 turns from 21 Nov 2026."""

    def test_unsure_segments_are_dropped(self):
        segs = [Seg('God bless you all, God bless you all.', -1.05, 0.44),
                Seg('Thank you, speak with knowledge.', -1.22, 0.29),
                Seg('Yeah, he is my mom, huh?', -1.01, 0.44)]
        kept, dropped = filter_segments(segs, 0.85, -1.0)
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 3)

    def test_real_turns_are_kept(self):
        segs = [Seg('What is 3 times 11?', -0.66, 0.27),
                Seg('Czyli jedziemy już po polsku.', -0.88, 0.04)]
        kept, _ = filter_segments(segs, 0.85, -1.0)
        self.assertEqual(len(kept), 2)

    def test_distant_speech_and_noise_are_dropped(self):
        segs = [Seg("He was never here, but he's here now.", -0.93, 0.65),
                Seg('I hope you understand a lot of that stuff.', -0.90, 0.58),
                Seg("That's the end of the episode.", -0.79, 0.65)]
        kept, _ = filter_segments(segs, 0.85, -1.0, 0.5, -0.7)
        self.assertEqual(kept, [])

    def test_combined_rule_keeps_real_turns(self):
        segs = [Seg('What is 3 times 11?', -0.66, 0.27),
                Seg('Czyli jedziemy już po polsku.', -0.88, 0.04),
                Seg('Confident but noisy.', -0.5, 0.7)]
        kept, _ = filter_segments(segs, 0.85, -1.0, 0.5, -0.7)
        self.assertEqual(len(kept), 3)

    def test_logprob_check_is_optional(self):
        kept, _ = filter_segments([Seg('Hello', -1.5, 0.1)], 0.85)
        self.assertEqual(kept, ['Hello'])

    def test_video_phrases_are_noise(self):
        for text in ('And we will see you in the next video.',
                     'I will show you my web store long before the end of this video.'):
            self.assertTrue(looks_like_noise(text), text)
        self.assertFalse(looks_like_noise('Can you show me a video of the lab?'))


class BoardPlayStateTest(unittest.TestCase):

    def test_play_state_messages(self):
        self.assertTrue(board_play_state('{"play_state":1}'))
        self.assertFalse(board_play_state('{"play_state":0}'))

    def test_asr_results_and_garbage_are_ignored(self):
        asr = ('{"index":294,"type":0,"text":"\u30a2\u30ec\u3055\u3093\u3002",'
               '"language":"<|ja|>","is_final":false}')
        for text in (asr, '', 'not json', '[1, 2]', None):
            self.assertIsNone(board_play_state(text))


class EchoGuardTest(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.echo = EchoGuard(guard=2.5, max_playing=60.0, clock=self.clock)

    def test_quiet_at_start(self):
        self.assertFalse(self.echo.active())

    def test_guard_after_playback(self):
        self.echo.set('board', True)
        self.assertTrue(self.echo.active())
        self.clock.now += 3.0
        self.echo.set('board', False)
        self.clock.now += 2.0
        self.assertTrue(self.echo.active())
        self.clock.now += 0.6
        self.assertFalse(self.echo.active())

    def test_any_source_keeps_it_active(self):
        self.echo.set('tts', True)
        self.echo.set('board', True)
        self.echo.set('board', False)  # gap between our sentences
        self.clock.now += 10.0
        self.assertTrue(self.echo.active())
        self.assertTrue(self.echo.playing('tts'))
        self.assertFalse(self.echo.playing('board'))

    def test_unknown_stop_does_not_restart_guard(self):
        self.echo.set('board', False)
        self.assertFalse(self.echo.active())

    def test_quiet_for(self):
        self.assertGreater(self.echo.quiet_for(), 1e6)  # never played
        self.echo.set('tts', True)
        self.assertIsNone(self.echo.quiet_for())
        self.clock.now += 4.0
        self.echo.set('tts', False)
        self.clock.now += 2.5
        self.assertAlmostEqual(self.echo.quiet_for(), 2.5)

    def test_lost_stop_message_expires(self):
        self.echo.set('board', True)
        self.clock.now += 59.0
        self.assertTrue(self.echo.active())
        self.clock.now += 1.0
        self.assertFalse(self.echo.playing())
        self.clock.now += 2.6
        self.assertFalse(self.echo.active())


class UnitreeStatusParseTest(unittest.TestCase):

    def test_network_online(self):
        self.assertTrue(network_online(
            '{"network_status": "NetworkStatus.ON_WIFI_CONNECTED"}'))
        self.assertTrue(network_online('{"network_status": "NetworkStatus.ON_4G_CONNECTED"}'))
        self.assertFalse(network_online('{"network_status": "NetworkStatus.DISCONNECTED"}'))
        self.assertFalse(network_online('{"network_status": "NetworkStatus.OFFLINE"}'))
        for bad in ('', 'x', '{"other": 1}', '[1]', None):
            self.assertIsNone(network_online(bad))

    def test_gpt_state(self):
        state = gpt_state('{"state": true, "llm_name": "clound", "chat_state": false}')
        self.assertEqual(state['llm_name'], 'clound')
        self.assertIsNone(gpt_state('nope'))


class AssistantArbiterTest(unittest.TestCase):

    def make(self, policy):
        self.clock = FakeClock()
        return AssistantArbiter(policy, online_stable=10.0, network_timeout=30.0,
                                gpt_timeout=12.0, clock=self.clock)

    def test_off_keeps_gpt_off_even_online(self):
        a = self.make('off')
        a.on_network(True)
        self.clock.now += 60
        a.on_network(True)
        self.assertEqual(a.mode(), 'aletta')
        self.assertFalse(a.wanted_gpt())

    def test_leave_never_touches_gpt(self):
        a = self.make('leave')
        self.assertIsNone(a.wanted_gpt())
        self.assertEqual(a.mode(), 'aletta')

    def test_auto_hands_over_after_stable_online(self):
        a = self.make('auto')
        self.assertIsNone(a.wanted_gpt())  # nothing known yet: hands off
        a.on_network(True)
        self.clock.now += 9
        a.on_network(True)
        self.assertEqual(a.mode(), 'aletta')
        self.assertIsNone(a.wanted_gpt())  # not stable yet: hands off
        self.clock.now += 1
        self.assertEqual(a.mode(), 'unitree')
        self.assertTrue(a.wanted_gpt())

    def test_auto_offline_returns_to_aletta_at_once(self):
        a = self.make('auto')
        a.on_network(True)
        self.clock.now += 20
        self.assertEqual(a.mode(), 'unitree')
        a.on_network(False)
        self.assertEqual(a.mode(), 'aletta')
        a.on_network(True)  # back online: wait for stability again
        self.assertEqual(a.mode(), 'aletta')

    def test_auto_switches_off_when_offline(self):
        a = self.make('auto')
        a.on_network(False)
        self.assertFalse(a.wanted_gpt())

    def test_auto_no_report_after_start_counts_as_offline(self):
        a = self.make('auto')
        self.clock.now += 31
        self.assertEqual(a.network(), 'offline')
        self.assertFalse(a.wanted_gpt())

    def test_auto_silent_network_counts_as_offline(self):
        a = self.make('auto')
        a.on_network(True)
        self.clock.now += 20
        self.assertEqual(a.mode(), 'unitree')
        self.clock.now += 31
        self.assertEqual(a.mode(), 'aletta')

    def test_auto_aletta_session_takes_over(self):
        a = self.make('auto')
        a.on_network(True)
        self.clock.now += 20
        self.assertEqual(a.mode(), 'unitree')       # online: Unitree by default
        self.assertTrue(a.wanted_gpt())
        a.session_open = True                       # "Aletta" said
        self.assertEqual(a.mode(), 'aletta')
        self.assertFalse(a.wanted_gpt())            # chat_go must not answer too
        a.session_open = False                      # session closed
        self.assertEqual(a.mode(), 'unitree')
        self.assertTrue(a.wanted_gpt())

    def test_auto_offline_session_is_plain_aletta(self):
        a = self.make('auto')
        a.on_network(False)
        a.session_open = True
        self.assertEqual(a.mode(), 'aletta')
        self.assertFalse(a.wanted_gpt())

    def test_gpt_running_follows_state_messages(self):
        a = self.make('off')
        self.assertFalse(a.gpt_running())
        a.on_gpt_state({'state': True, 'chat_state': False})
        self.assertTrue(a.gpt_running())
        self.assertFalse(a.chat_state)
        self.clock.now += 13
        self.assertFalse(a.gpt_running())

    def online_auto(self):
        a = self.make('auto')
        a.on_network(True)
        self.clock.now += 20
        a.on_network(True)
        return a

    def test_auto_chat_state_does_not_decide(self):
        # 21 Nov 2026: chat_go answered six questions while chat_state was false.
        a = self.online_auto()
        a.on_gpt_state({'state': True, 'chat_state': False})
        self.assertEqual(a.mode(), 'unitree')
        self.assertTrue(a.wanted_gpt())

    def test_f1_toggles_between_unitree_and_aletta(self):
        a = self.online_auto()
        self.assertEqual(a.mode(), 'unitree')
        self.assertEqual(a.toggle(), 'aletta')       # F1: to Aletta
        self.assertFalse(a.wanted_gpt())              # chat_go off
        a.session_open = True                          # conversation opened
        a.session_open = False                         # ...and ended
        self.assertEqual(a.mode(), 'aletta')          # stays Aletta until F1
        self.assertEqual(a.toggle(), 'unitree')      # F1: back to Unitree
        self.assertTrue(a.wanted_gpt())

    def test_f1_back_to_unitree_waits_for_session_close(self):
        a = self.online_auto()
        a.toggle()
        a.session_open = True
        self.assertEqual(a.toggle(), 'unitree')      # F1: target is Unitree...
        self.assertEqual(a.mode(), 'aletta')          # ...but the conversation is open
        self.assertFalse(a.wanted_gpt())              # chat_go stays off meanwhile
        a.session_open = False                         # node closed it (continuous_stop)
        self.assertEqual(a.mode(), 'unitree')
        self.assertTrue(a.wanted_gpt())

    def test_f1_during_open_conversation_keeps_aletta(self):
        # 21 Nov 20:48: F1 pressed while an Aletta conversation was open; when
        # it ended the robot fell back to Unitree.
        a = self.online_auto()
        a.session_open = True
        self.assertEqual(a.toggle(), 'aletta')
        a.session_open = False
        self.assertEqual(a.mode(), 'aletta')
        self.assertEqual(a.toggle(), 'unitree')

    def test_f1_has_no_role_offline(self):
        a = self.make('auto')
        a.on_network(False)
        self.assertIsNone(a.toggle())
        self.assertIsNone(self.make('off').toggle())

    def test_going_offline_clears_f1(self):
        a = self.online_auto()
        a.toggle()
        a.on_network(False)
        self.assertEqual(a.mode(), 'aletta')
        a.on_network(True)
        self.clock.now += 20
        a.on_network(True)
        self.assertEqual(a.mode(), 'unitree')          # online again: Unitree default

    def test_bad_policy(self):
        with self.assertRaises(ValueError):
            AssistantArbiter('maybe')


class NameCutTest(unittest.TestCase):
    """Timings from a real session (21 Nov 2026), relative to each turn's audio."""

    def test_name_then_pause_then_question(self):
        # "Aletta" 0.7-1.4 s, pause, "what is two times two" 2.2-3.5 s; the
        # model also misfired after the question (4.23 s).
        cut = name_cut(0.0, 5.16, [2.01, 4.23], onset=0.8)
        self.assertAlmostEqual(cut, 1.81, places=2)

    def test_turn_started_mid_name(self):
        # Mic opened on the tail of "Aletta"; question 0.3-1.7 s.
        self.assertAlmostEqual(name_cut(0.0, 3.36, [0.37, 1.81, 3.89]), 0.17, places=2)

    def test_name_only(self):
        self.assertAlmostEqual(name_cut(0.0, 2.40, [1.12]), 0.92, places=2)

    def test_late_first_detection_is_a_misfire(self):
        self.assertIsNone(name_cut(0.0, 5.0, [3.5], onset=1.0))

    def test_detections_outside_the_audio(self):
        self.assertIsNone(name_cut(10.0, 3.0, [9.0, 13.5]))
        self.assertIsNone(name_cut(10.0, 3.0, []))

    def test_cut_clamped_to_the_audio(self):
        self.assertEqual(name_cut(10.0, 3.0, [10.1]), 0.0)


class EchoesPromptTest(unittest.TestCase):
    PROMPT = 'Aletta, Unitree G1, battery, charging, ROS 2, nodes, topics, Wi-Fi.'

    def test_hint_echo(self):
        for text in ('Unitree G1, battery.', 'ROS 2 nodes', 'Aletta.'):
            self.assertTrue(echoes_prompt(text, self.PROMPT), text)

    def test_real_questions(self):
        for text in ('What was your battery status?', 'What are all running ROS nodes?',
                     'What time is it?'):
            self.assertFalse(echoes_prompt(text, self.PROMPT), text)
        self.assertFalse(echoes_prompt('battery', ''))


class LocalTimeTest(unittest.TestCase):

    def test_both_forms(self):
        import time as _time
        t = _time.mktime((2026, 11, 21, 20, 6, 0, 0, 0, -1))
        self.assertTrue(format_local_time(t).startswith('20:06 (8:06 PM), Saturday 21 November'))
        t = _time.mktime((2026, 11, 21, 0, 5, 0, 0, 0, -1))
        self.assertTrue(format_local_time(t).startswith('00:05 (12:05 AM)'))
        t = _time.mktime((2026, 11, 21, 12, 30, 0, 0, 0, -1))
        self.assertTrue(format_local_time(t).startswith('12:30 (12:30 PM)'))


class MatchParametersTest(unittest.TestCase):
    PARAMS = {
        ('wake_word_node', 'aletta_threshold'): 0.8,
        ('stt_node', 'model_size'): 'medium',
        ('stt_node', 'language_id_model'): 'small',
        ('stt_node', 'speech_start_threshold'): 0.02,
        ('llm_node', 'system_prompt'): 'You are Aletta, a friendly humanoid robot ' * 5,
        ('assistant_mode_node', 'unitree_assistant'): 'auto',
        ('tts_node', 'voice_model_path'): '',
    }

    def test_question_words_select_parameters(self):
        lines = match_parameters(self.PARAMS, 'What thresholds are set?')
        self.assertEqual(lines, ['stt_node.speech_start_threshold = 0.02',
                                 'wake_word_node.aletta_threshold = 0.8'])
        lines = match_parameters(self.PARAMS, 'Which speech recognition model do you use?')
        self.assertIn('stt_node.model_size = medium', lines)
        self.assertIn('stt_node.language_id_model = small', lines)

    def test_long_values_are_shortened_and_empty_skipped(self):
        line = match_parameters(self.PARAMS, 'show the system prompt')[0]
        self.assertTrue(line.startswith('llm_node.system_prompt = You are Aletta'))
        self.assertTrue(line.endswith('...'))
        self.assertEqual(match_parameters(self.PARAMS, 'voice model path'), [
            'stt_node.language_id_model = small', 'stt_node.model_size = medium'])

    def test_no_match(self):
        self.assertEqual(match_parameters(self.PARAMS, 'How are you?'), [])


class SnapCutToPauseTest(unittest.TestCase):
    """50 ms loudness profiles of real turns (21 Nov 2026) with the detection-based cut."""

    LEVEL = {'.': 0.005, '-': 0.012, '+': 0.025, '#': 0.05}
    CASES = [
        ('.........---....####-.+####+..-++########'
         '#+.........................-+-...', 2.07, 1.50),
        ('........--..#++-...+#-+-.++' '+.........................+++.....', 1.35, 0.95),
        ('++++--.-' '+-+.++...++--+#++-.--'
         '............................-.-+', 0.41, None),
        ('..........++++..--+++.+#+---....' '+--...-......-#+.............-....-+', 1.60, 1.60),
    ]

    def audio(self, profile):
        import numpy as np
        return np.concatenate([np.full(800, self.LEVEL[c], dtype=np.float32) for c in profile])

    def test_recorded_cuts(self):
        for profile, cut, expected in self.CASES:
            got = snap_cut_to_pause(self.audio(profile), 16000, cut, 0.010)
            if expected is None:
                self.assertIsNone(got, profile)
            else:
                self.assertAlmostEqual(got, expected, places=2, msg=profile)

    def test_cut_already_in_a_pause_is_kept(self):
        got = snap_cut_to_pause(self.audio('..####....###'), 16000, 0.45, 0.010)
        self.assertAlmostEqual(got, 0.45, places=2)


class HasSpeechTest(unittest.TestCase):
    RATE = 16000

    def test_silence_click_and_speech(self):
        import numpy as np
        quiet = np.full(self.RATE, 0.005, dtype=np.float32)
        self.assertFalse(has_speech(quiet, self.RATE, 0.01))
        click = quiet.copy()
        click[8000:9600] = 0.5
        self.assertFalse(has_speech(click, self.RATE, 0.01))
        voice = quiet.copy()
        voice[3200:11200] = 0.05
        self.assertTrue(has_speech(voice, self.RATE, 0.01))
        self.assertFalse(has_speech(quiet[:0], self.RATE, 0.01))


class EndpointerTest(unittest.TestCase):
    CHUNK = 2560  # 160 ms at 16 kHz, like the G1 mic packets

    def run_turn(self, rms_values, silence=2.5, **kw):
        ep = Endpointer(16000, 0.0072, max_utterance=10.0, max_wait=8.0, silence=silence, **kw)
        for i, rms in enumerate(rms_values):
            if not ep.feed(rms, self.CHUNK):
                return ep, (i + 1) * self.CHUNK / 16000
        return ep, None

    def test_echo_tail_and_noise_do_not_start_a_turn(self):
        # Measured after the robot stops speaking: <= 420 int16 RMS = 0.013.
        ep, stopped = self.run_turn([0.013] * 60, onset_threshold=0.02)
        self.assertIsNone(ep.onset)
        self.assertIsNotNone(stopped)

    def test_voice_starts_and_quiet_word_ending_continues(self):
        # Speech at 0.05, then a quiet ending (0.012) above the end threshold.
        rms = [0.005] * 5 + [0.05] * 8 + [0.012] * 10 + [0.005] * 30
        ep, stopped = self.run_turn(rms, silence=1.2, onset_threshold=0.02)
        # Confirmed through the 5-packet median, within the loud stretch.
        self.assertTrue(5 * self.CHUNK <= ep.onset <= 13 * self.CHUNK)
        # The quiet ending still counted as speech: the turn ends ~1.2 s after it.
        end_of_ending = 23 * self.CHUNK / 16000
        self.assertGreater(stopped, end_of_ending + 0.9)
        self.assertLess(stopped, end_of_ending + 1.8)

    def test_has_speech_uses_the_start_threshold(self):
        import numpy as np
        tail = np.full(16000, 0.013, dtype=np.float32)
        self.assertTrue(has_speech(tail, 16000, 0.010))
        self.assertFalse(has_speech(tail, 16000, 0.010, onset_threshold=0.02))

    def test_silence_only_ends_after_max_wait(self):
        ep, stopped = self.run_turn([0.005] * 200)
        self.assertIsNone(ep.onset)
        self.assertAlmostEqual(stopped, 8.0, delta=0.2)

    def test_late_speech_is_not_cut_off(self):
        # 7.8 s of silence, then 4 s of speech, then silence: the old fixed
        # 10 s window cut this utterance at 10 s.
        ep, stopped = self.run_turn([0.005] * 49 + [0.1] * 25 + [0.005] * 40)
        self.assertAlmostEqual(ep.onset / 16000, 7.84 + 0.32, delta=0.4)
        self.assertGreater(stopped, 7.8 + 4.0 + 2.0)

    def test_long_speech_capped_from_onset(self):
        ep, stopped = self.run_turn([0.005] * 10 + [0.1] * 200)
        self.assertAlmostEqual(stopped - ep.onset / 16000, 10.0, delta=0.2)

    def test_trailing_silence_reported(self):
        ep, _ = self.run_turn([0.005] * 5 + [0.1] * 10 + [0.005] * 40, silence=1.2)
        self.assertAlmostEqual(ep.trailing_silence(), 1.2, delta=0.2)

    def test_single_click_is_not_speech(self):
        ep, _ = self.run_turn([0.005] * 10 + [0.5] + [0.005] * 60)
        self.assertIsNone(ep.onset)


class SystemSummaryTest(unittest.TestCase):

    def test_asks_about_system(self):
        for text in ('Which nodes are running?', 'How many topics do you have?',
                     'What language model do you use?', 'Tell me your settings',
                     'Ile masz węzłów?', 'Jakie masz ustawienia?',
                     'Jaki model językowy działa w systemie?'):
            self.assertTrue(asks_about_system(text), text)
        for text in ('What is the weather like?', 'How is your battery?',
                     'Opowiedz mi coś na temat Poznania', ''):
            self.assertFalse(asks_about_system(text), text)

    def test_graph_all_running(self):
        nodes = list(VOICE_NODES) + ['lowstate_bridge']
        lines = summarize_graph(nodes, ['/a', '/b', '/b'])
        self.assertIn('all 7 running', lines[0])
        self.assertEqual(lines[1], 'ROS 2 graph: 8 nodes (1 outside the voice pipeline), 2 topics')

    def test_graph_missing_node_with_namespace(self):
        nodes = ['/' + n for n in VOICE_NODES if n != 'tts_node']
        lines = summarize_graph(nodes, [])
        self.assertIn('6 of 7 running, missing: tts_node', lines[0])

    def test_parameter_value(self):
        class Value:
            def __init__(self, kind, **fields):
                self.type = kind
                self.bool_value, self.integer_value = False, 0
                self.double_value, self.string_value = 0.0, ''
                for k, v in fields.items():
                    setattr(self, k, v)
        self.assertEqual(parameter_value(Value(4, string_value='small')), 'small')
        self.assertEqual(parameter_value(Value(3, double_value=0.80000001)), 0.8)
        self.assertEqual(parameter_value(Value(2, integer_value=5)), 5)
        self.assertIsNone(parameter_value(Value(0)))  # unset parameter

    def test_format_parameters(self):
        self.assertIsNone(format_parameters({}))
        line = format_parameters({('llm_node', 'api_model'): 'llama3.2',
                                  ('stt_node', 'model_size'): 'small',
                                  ('stt_node', 'language'): ''})
        self.assertEqual(line, 'Voice settings: language model llama3.2, '
                               'speech recognition model small')


if __name__ == '__main__':
    unittest.main()
