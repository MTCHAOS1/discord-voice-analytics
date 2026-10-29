import unittest

import pandas as pd

from src.dashboard.charts import participation_donut


class TestParticipationChart(unittest.TestCase):
    def test_highlight_preserves_full_denominator_and_all_speakers(self):
        frame = pd.DataFrame([dict(speaker='Aurora', estimated_speech_seconds=30),
                              dict(speaker='Bento', estimated_speech_seconds=70)])
        spec = participation_donut(frame, 'Aurora').to_dict()
        rows = next(iter(spec['datasets'].values()))
        by_person = {row['speaker']: row for row in rows}
        self.assertEqual(by_person['Aurora']['percentage'], 30)
        self.assertEqual(by_person['Bento']['percentage'], 70)
        self.assertEqual(by_person['Aurora']['emphasis'], 1)
        self.assertLess(by_person['Bento']['emphasis'], 1)
        self.assertEqual(by_person['Aurora']['legend_label'], 'Aurora · 30.0%')
        self.assertEqual(frame.columns.tolist(), ['speaker', 'estimated_speech_seconds'])

    def test_zero_total_has_finite_zero_percentages(self):
        frame = pd.DataFrame([dict(speaker='Aurora', estimated_speech_seconds=0)])
        spec = participation_donut(frame).to_dict()
        rows = next(iter(spec['datasets'].values()))
        self.assertEqual(rows[0]['percentage'], 0)
