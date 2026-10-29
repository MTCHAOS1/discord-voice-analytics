"""Ritmo de fala e silêncio com dados sintéticos e resultados calculáveis."""
from contextlib import redirect_stdout
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dashboard.analytics import (daily_metrics, daily_timelapse, merge_intervals, silence_stats, speech_timing,
                                     words_per_minute)
from src.dashboard.report import combine_dashboards, demo_dashboard, load_dashboard, write_dashboard
from src.quality.validate_session import check_words, validate_session
from src.transformation.analyze import analyze
class EmbeddedData(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.active = False
        self.chunks = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = dict(attrs).get('id') == 'dashboard-data'

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, text):
        if self.active:
            self.chunks.append(text)


def words_frame(rows):
    return pd.DataFrame(rows, columns=['session_id', 'speaker', 'start_s', 'end_s', 'word'])


class TestRhythmCalculations(unittest.TestCase):
    def test_merge_intervals_joins_overlaps_and_small_gaps(self):
        intervals = [(5, 6), (0, 1), (1.4, 2), (0.5, 0.8)]
        self.assertEqual(merge_intervals(intervals), [(0.0, 1.0), (1.4, 2.0), (5.0, 6.0)])
        self.assertEqual(merge_intervals(intervals, bridge=0.5), [(0.0, 2.0), (5.0, 6.0)])
        self.assertEqual(merge_intervals([]), [])

    def test_overlap_is_counted_once_per_person(self):
        frame = pd.DataFrame([dict(speaker='A', start_s=0, end_s=4), dict(speaker='A', start_s=2, end_s=6),
                              dict(speaker='B', start_s=0, end_s=1)])
        result = speech_timing(frame).set_index('speaker').measured_speech_seconds
        self.assertEqual(result['A'], 6)
        self.assertEqual(result['B'], 1)

    def test_words_per_minute_is_none_without_time(self):
        self.assertIsNone(words_per_minute(10, 0))
        self.assertEqual(words_per_minute(30, 30), 60)

    def test_silence_includes_start_and_end_gaps_with_known_duration(self):
        frame = words_frame([('s', 'A', 10, 12, 'oi'), ('s', 'B', 12.3, 14, 'tudo'), ('s', 'A', 30, 32, 'bem')])
        segments = pd.DataFrame([dict(session_id='s', speaker='A', start_s=10, end_s=14, text='oi tudo'),
                                 dict(session_id='s', speaker='A', start_s=30, end_s=32, text='bem')])
        stats = silence_stats(frame, 60, min_gap=3, segments=segments)
        self.assertEqual(stats['basis'], 'gravação inteira')
        self.assertAlmostEqual(stats['spoken_seconds'], 6)
        self.assertAlmostEqual(stats['silence_seconds'], 54)
        self.assertAlmostEqual(stats['silence_share'], 90)
        gaps = stats['gaps']
        self.assertEqual(sorted(gaps.start_s.tolist()), [0, 14, 32])
        self.assertEqual(gaps.gap_seconds.tolist(), sorted(gaps.gap_seconds, reverse=True))
        middle = gaps[gaps.start_s == 14].iloc[0]
        self.assertEqual((middle.before_text, middle.after_text), ('oi tudo', 'bem'))
        self.assertEqual(gaps[gaps.start_s == 0].iloc[0].before_text, '')
        self.assertEqual(gaps[gaps.start_s == 32].iloc[0].after_text, '')

    def test_silence_without_duration_uses_first_and_last_speech(self):
        frame = words_frame([('s', 'A', 10, 12, 'oi'), ('s', 'A', 20, 22, 'bem')])
        stats = silence_stats(frame, None, min_gap=3)
        self.assertEqual(stats['basis'], 'entre a primeira e a última fala')
        self.assertEqual(stats['duration_seconds'], 12)
        self.assertEqual(stats['silence_seconds'], 8)
        self.assertEqual(stats['gaps'].start_s.tolist(), [12])
        self.assertEqual(silence_stats(frame, float('nan'))['basis'], 'entre a primeira e a última fala')

    def test_pause_inside_a_segment_has_that_segment_on_both_sides(self):
        frame = words_frame([('s', 'A', 100, 105, 'x'), ('s', 'A', 125, 130, 'y')])
        segments = pd.DataFrame([dict(session_id='s', speaker='A', start_s=90, end_s=98, text='anterior'),
                                 dict(session_id='s', speaker='B', start_s=100, end_s=130, text='dentro'),
                                 dict(session_id='s', speaker='C', start_s=140, end_s=150, text='posterior')])
        gaps = silence_stats(frame, None, min_gap=3, segments=segments)['gaps']
        row = gaps.iloc[0]
        self.assertEqual((row.start_s, row.end_s), (105, 125))
        self.assertEqual((row.before_speaker, row.before_text, row.after_speaker, row.after_text),
                         ('B', 'dentro', 'B', 'dentro'))

    def test_neighbors_outside_segments_use_closest_ones(self):
        frame = words_frame([('s', 'A', 10, 12, 'a'), ('s', 'A', 30, 32, 'b')])
        segments = pd.DataFrame([dict(session_id='s', speaker='A', start_s=5, end_s=12, text='antes'),
                                 dict(session_id='s', speaker='B', start_s=1, end_s=8, text='mais antigo'),
                                 dict(session_id='s', speaker='C', start_s=30, end_s=40, text='depois'),
                                 dict(session_id='s', speaker='D', start_s=60, end_s=70, text='longe')])
        row = silence_stats(frame, None, min_gap=3, segments=segments)['gaps'].iloc[0]
        self.assertEqual((row.before_text, row.after_text), ('antes', 'depois'))

    def test_large_synthetic_session_finishes_without_quadratic_loop(self):
        rng = np.random.default_rng(1)
        starts = np.sort(rng.uniform(0, 14400, 20000))
        frame = pd.DataFrame(dict(session_id='s', speaker='A', start_s=starts, end_s=starts + 0.2, word='x'))
        seg_starts = np.sort(rng.uniform(0, 14400, 1600))
        segments = pd.DataFrame(dict(session_id='s', speaker='A', start_s=seg_starts, end_s=seg_starts + 6, text='t'))
        began = time.perf_counter()
        stats = silence_stats(frame, 14400, min_gap=1, segments=segments)
        self.assertLess(time.perf_counter() - began, 5)
        self.assertFalse(stats['gaps'].empty)

    def test_speech_past_the_recording_end_is_clipped_to_the_window(self):
        frame = words_frame([('s', 'A', 5, 8, 'a'), ('s', 'A', 95, 130, 'b')])
        stats = silence_stats(frame, 100, min_gap=3)
        self.assertEqual(stats['duration_seconds'], 100)
        self.assertAlmostEqual(stats['spoken_seconds'], 3 + 5)
        self.assertAlmostEqual(stats['silence_seconds'], 92)
        self.assertEqual(stats['gaps'].end_s.max(), 95)
        self.assertTrue((stats['gaps'].end_s <= 100).all())

    def test_silence_with_empty_frame(self):
        stats = silence_stats(words_frame([]), 100)
        self.assertEqual((stats['duration_seconds'], stats['spoken_seconds'], stats['silence_seconds']), (0, 0, 0))
        self.assertTrue(stats['gaps'].empty)

    def test_combined_wpm_is_recomputed_from_sums(self):
        def dataset(session, words, seconds):
            return dict(demo_dashboard(), session_id=session, recording_seconds=100, timing_source='palavras',
                        participants=[dict(speaker='A', segments=1, estimated_speech_seconds=10, content_word_count=5,
                                           spoken_word_count=words, measured_speech_seconds=seconds,
                                           words_per_minute=words / seconds * 60)])
        combined = combine_dashboards([dataset('s1', 60, 60), dataset('s2', 300, 60)])
        person = combined['participants'][0]
        self.assertEqual((person['spoken_word_count'], person['measured_speech_seconds']), (360, 120))
        self.assertEqual(person['words_per_minute'], 180)
        self.assertEqual(combined['recordings'], {'s1': 100, 's2': 100})
        self.assertEqual(combined['timing_sources'], {'s1': 'palavras', 's2': 'palavras'})

    def test_combine_accepts_participants_with_only_old_fields(self):
        old = dict(speaker='A', segments=1, estimated_speech_seconds=10, content_word_count=5)
        person = combine_dashboards([dict(demo_dashboard(), participants=[old])])['participants'][0]
        self.assertEqual(person['spoken_word_count'], 0)
        self.assertIsNone(person['words_per_minute'])

    def test_daily_wpm_is_computed_after_grouping(self):
        def person(words, seconds):
            return dict(speaker='A', segments=1, estimated_speech_seconds=10, content_word_count=5,
                        spoken_word_count=words, measured_speech_seconds=seconds)
        old = dict(speaker='A', segments=1, estimated_speech_seconds=10, content_word_count=5)
        datasets = [dict(day='2026-10-26', participants=[person(60, 60)], words=[]),
                    dict(day='2026-10-26', participants=[person(300, 60)], words=[]),
                    dict(day='2026-10-27', participants=[person(30, 60)], words=[]),
                    dict(day='2026-10-28', participants=[old], words=[])]
        result = daily_metrics(datasets).set_index('day').words_per_minute
        self.assertEqual(result['2026-10-26'], 180)
        self.assertEqual(result['2026-10-27'], 30)
        self.assertTrue(pd.isna(result['2026-10-28']))


class TestDailyTimelapse(unittest.TestCase):
    def test_participant_without_content_words_gives_no_frames(self):
        datasets = [dict(day='2026-10-26', words=[dict(speaker='A', word='mapa', count=2)]),
                    dict(day='2026-10-27', words=[dict(speaker='A', word='jogo', count=1)])]
        self.assertEqual(daily_timelapse(datasets, 'B'), ([], []))
        days, frames = daily_timelapse(datasets, 'A')
        self.assertEqual(days, ['2026-10-26', '2026-10-27'])
        self.assertEqual(frames[-1][1], {('A', 'mapa'): 2, ('A', 'jogo'): 1})


class TestCheckWords(unittest.TestCase):
    def test_reports_divergent_session_and_words_outside_the_track(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / 'words.parquet'
            rows = [('s', 'A', '1-a.wav', 1.0, 2.0), ('outra', 'A', '1-a.wav', 3.0, 4.0),
                    ('s', 'A', '1-a.wav', 10.0, 11.0), ('s', 'A', '1-a.wav', 8.0, 12.5)]
            pd.DataFrame([dict(session_id=a, speaker=b, source_file=c, start_s=d, end_s=e, word='x', probability=1.0)
                          for a, b, c, d, e in rows]).to_parquet(source, index=False)
            errors = []
            self.assertEqual(check_words(source, {'1-a.wav': 'A'}, errors, 's', {'1-a.wav': 10.0}), 4)
            text = '\n'.join(errors)
            self.assertIn('1 palavra(s) com session_id divergente', text)
            self.assertIn('1 palavra(s) terminam mais de 2s depois do fim do audio', text)
            self.assertIn('1 palavra(s) comecam depois do fim do audio', text)

    def test_nullable_times_are_invalid_not_a_crash(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / 'words.parquet'
            pd.DataFrame(dict(session_id='s', speaker='A', source_file='1-a.wav',
                              start_s=pd.array([1, None], dtype='Int64'), end_s=pd.array([2, 3], dtype='Int64'),
                              word='x', probability=1.0)).to_parquet(source, index=False)
            errors = []
            check_words(source, {'1-a.wav': 'A'}, errors, 's', {'1-a.wav': 60.0})
            self.assertTrue(any('timestamp invalido' in e for e in errors))


class TestLoadWithWords(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = 'synthetic_rhythm'
        self.enterContext(redirect_stdout(StringIO()))

    def prepare(self, with_words):
        raw = self.root / 'data/raw' / self.session
        raw.mkdir(parents=True)
        (raw / '1-alpha.wav').write_bytes(b'synthetic fixture, never decoded')
        processed = self.root / 'data/processed' / self.session
        processed.mkdir(parents=True)
        terms = ['termo' + chr(97 + i // 26) + chr(97 + i % 26) for i in range(35)]
        pd.DataFrame([dict(session_id=self.session, speaker='alpha', source_file='1-alpha.wav',
                           start_s=0.0, end_s=4.0, text=' '.join(terms))]).to_parquet(
                               processed / 'segments.parquet', index=False)
        if with_words:
            pd.DataFrame([dict(session_id=self.session, speaker='alpha', source_file='1-alpha.wav',
                               start_s=i * 0.1, end_s=i * 0.1 + 0.1, word=term, probability=0.9)
                          for i, term in enumerate(terms)]).to_parquet(processed / 'words.parquet', index=False)
        analyze(self.session, base_dir=self.root)
        with patch('src.quality.validate_session.audio_duration', return_value=4.0):
            return validate_session(self.session, self.root)

    def test_without_words_parquet_uses_segments(self):
        report = self.prepare(with_words=False)
        self.assertIsNone(report['summary']['word_timestamps'])
        self.assertEqual(report['status'], 'PASS')
        data = load_dashboard(self.session, self.root, include_segments=True)
        person = data['participants'][0]
        self.assertEqual(data['timing_source'], 'trechos')
        self.assertIsNone(data['recording_seconds'])
        self.assertEqual(person['spoken_word_count'], 35)
        self.assertEqual(person['measured_speech_seconds'], 4)
        self.assertEqual(person['words_per_minute'], 35 / 4 * 60)
        self.assertNotIn('word_timings', data)

    def test_with_words_parquet_measures_by_word(self):
        report = self.prepare(with_words=True)
        self.assertEqual(report['summary']['word_timestamps'], 35)
        self.assertEqual(report['status'], 'PASS')
        metadata = self.root / 'data/processed' / self.session / 'metadata.json'
        metadata.write_text('{"tracks": [{"duration_s": 4.0}, {"duration_s": NaN}, {"duration_s": 7.5}]}',
                            encoding='utf-8')
        data = load_dashboard(self.session, self.root, include_segments=True)
        person = data['participants'][0]
        self.assertEqual(data['timing_source'], 'palavras')
        self.assertEqual(data['recording_seconds'], 7.5)
        self.assertAlmostEqual(person['measured_speech_seconds'], 3.5)
        self.assertAlmostEqual(person['words_per_minute'], 600)
        self.assertEqual(len(data['word_timings']), 35)
        self.assertEqual(person['estimated_speech_seconds'], 4)
        html = write_dashboard(data, self.root / 'dashboard.html').read_text(encoding='utf-8')
        embedded = json.loads(''.join(EmbeddedData(html).chunks))
        self.assertNotIn('word_timings', embedded)
        self.assertNotIn('segments', embedded)
        metadata.write_text('{ invalido', encoding='utf-8')
        self.assertIsNone(load_dashboard(self.session, self.root)['recording_seconds'])

    def test_empty_words_parquet_falls_back_to_segments(self):
        self.prepare(with_words=False)
        processed = self.root / 'data/processed' / self.session
        pd.DataFrame(columns=['session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'word',
                              'probability']).to_parquet(processed / 'words.parquet', index=False)
        data = load_dashboard(self.session, self.root, include_segments=True)
        self.assertEqual(data['timing_source'], 'trechos')
        self.assertEqual(data['participants'][0]['measured_speech_seconds'], 4)
        self.assertNotIn('word_timings', data)
