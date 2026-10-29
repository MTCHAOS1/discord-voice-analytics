"""Semântica das análises com exemplos sintéticos e resultados calculáveis."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.dashboard.analytics import (activity_windows, characteristic_words, compare_people,
    daily_metrics, demo_analytics, frequent_expressions, search_context, segment_frame, session_summary)
from src.dashboard.report import write_dashboard


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.data, self.datasets = demo_analytics()
        self.frame = segment_frame(self.data)

    def test_daily_totals_and_vocabulary_union_same_day(self):
        first, second = self.datasets
        second = dict(second, day=first['day'])
        frame = daily_metrics([first, second])
        self.assertEqual(frame.segments.sum(), 10)
        self.assertAlmostEqual(frame.share.sum(), 100)
        aurora = frame[frame.speaker == 'Aurora'].iloc[0]
        words = {r['word'] for r in first['words'] if r['speaker'] == 'Aurora'}
        self.assertEqual(aurora.distinct_words, len(words))
        self.assertEqual(aurora.estimated_speech_seconds, 60)

    def test_comparison_common_and_exclusive_words(self):
        data = {'words': [dict(speaker='A', word='mapa', count=2), dict(speaker='B', word='mapa', count=3),
                          dict(speaker='A', word='jogo', count=1)]}
        frame, common, similarity = compare_people(data, 'A', 'B')
        self.assertEqual(common, 1)
        self.assertEqual(similarity, 50)
        self.assertEqual(frame.set_index('word').loc['jogo', 'second'], 0)

    def test_daily_missing_person_is_zero_and_not_forward_filled(self):
        first = dict(self.datasets[0], participants=[self.datasets[0]['participants'][0]])
        second = dict(self.datasets[1], participants=[self.datasets[1]['participants'][1]])
        result = daily_metrics([first, second])
        missing = result[(result.day == second['day']) & (result.speaker == first['participants'][0]['speaker'])]
        self.assertEqual(missing.iloc[0].estimated_speech_seconds, 0)
        self.assertEqual(missing.iloc[0].share, 0)

    def test_characteristic_rates_are_normalized_against_rest(self):
        data = {'words': [dict(speaker='A', word='mapa', count=8), dict(speaker='A', word='jogo', count=2),
                          dict(speaker='B', word='mapa', count=10), dict(speaker='B', word='jogo', count=90)]}
        result = characteristic_words(data, 'A')
        self.assertEqual(result.word.tolist(), ['mapa'])
        self.assertAlmostEqual(result.iloc[0].difference, 700)
        self.assertTrue(characteristic_words(data, 'A', minimum=9).empty)
        self.assertTrue(characteristic_words({'words': data['words'][:2]}, 'A').empty)

    def test_search_context_is_literal_and_does_not_cross_sessions(self):
        result = search_context(self.frame, 'MAPA', speaker='Aurora')
        self.assertEqual(len(result), 4)
        for row in result.itertuples():
            self.assertEqual(row.speaker, 'Aurora')
        # Primeiro trecho de cada sessão não herda o último da sessão anterior.
        self.assertTrue((result[result.start_s == 0].before == '').all())
        self.assertTrue(search_context(self.frame, '[').empty)
        self.assertTrue(search_context(self.frame, 'map').empty)
        self.assertGreater(len(search_context(self.frame, 'map', whole=False)), 0)

    def test_expressions_repeated_and_no_cross_segment_or_sentence_join(self):
        result = frequent_expressions(self.frame, 3, minimum=2)
        row = result[(result.speaker == 'Bento') & (result.phrase == 'mapa tem uma')].iloc[0]
        self.assertEqual(row['count'], 4)
        self.assertIn(row.text, self.frame.text.tolist())
        frame = pd.DataFrame([
            dict(session_id='x', speaker='A', start_s=0, end_s=1, text='um. dois'),
            dict(session_id='x', speaker='A', start_s=1, end_s=2, text='três'),
        ])
        self.assertTrue(frequent_expressions(frame, 2, minimum=1).empty)

    def test_activity_uses_union_and_splits_at_window_boundary(self):
        frame = pd.DataFrame([dict(session_id='x', start_s=0, end_s=20),
                              dict(session_id='x', start_s=10, end_s=40),
                              dict(session_id='x', start_s=290, end_s=310)])
        result = activity_windows(frame)
        self.assertEqual(result.active_seconds.tolist(), [50, 10])
        self.assertEqual(result.start_s.tolist(), [0, 300])

    def test_extractive_summary_and_topics_reference_exact_sources(self):
        frame = self.frame[self.frame.session_id == 'demo_01']
        highlights, topics = session_summary(frame, maximum=3)
        self.assertLessEqual(len(highlights), 3)
        self.assertTrue(highlights.start_s.is_monotonic_increasing)
        self.assertTrue(set(highlights.text).issubset(set(frame.text)))
        for row in topics.itertuples():
            source = frame[(frame.session_id == row.session_id) & (frame.start_s == row.start_s)]
            self.assertIn(row.text, source.text.tolist())

    def test_html_export_does_not_embed_context_transcripts(self):
        data = dict(self.data)
        data['segments'] = [dict(text='SECRET_PHRASE_SHOULD_NOT_BE_EXPORTED')]
        with TemporaryDirectory() as temp:
            output = Path(temp) / 'dashboard.html'
            write_dashboard(data, output)
            self.assertNotIn('SECRET_PHRASE_SHOULD_NOT_BE_EXPORTED', output.read_text())

    def test_empty_analytics_are_supported(self):
        empty = segment_frame({})
        self.assertTrue(frequent_expressions(empty).empty)
        self.assertTrue(activity_windows(empty).empty)
        highlights, topics = session_summary(empty)
        self.assertTrue(highlights.empty)
        self.assertTrue(topics.empty)
        self.assertTrue(search_context(empty, '').empty)
