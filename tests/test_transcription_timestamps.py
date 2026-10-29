"""Limites dos áudios com Whisper simulado, sem download ou gravações reais."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.transcription.transcribe import bound_timestamps, transcribe
from src.transformation.analyze import analyze
from src.quality.validate_session import validate_session


class TestTranscriptionTimestamps(unittest.TestCase):
    def test_only_valid_intersections_are_clipped(self):
        self.assertEqual(bound_timestamps(7, 30, 10)[:2], (10, 'clipped'))
        self.assertEqual(bound_timestamps(7, 10, 10), (10, None, None))
        for start, end, duration in [(10, 11, 10), (-1, 3, 10), (4, 4, 10),
                                     (7, 30, None), (7, 30, float('nan')),
                                     (float('nan'), 30, 10), (7, float('inf'), 10),
                                     (9.999, 12, 9.9994)]:
            with self.subTest(start=start, end=end, duration=duration):
                self.assertEqual(bound_timestamps(start, end, duration)[1], 'review')

    def process_fixture(self, root, start, end, duration=10):
        # Inicializa extensões Arrow antes de isolar o módulo Whisper.
        pd.io.parquet.get_engine('pyarrow')
        raw = root / 'data/raw/fixture'
        raw.mkdir(parents=True)
        (raw / '1-Aurora.flac').touch()
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(start=start, end=end, text=' teste final ')], None)
        whisper = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with patch.dict('sys.modules', {'faster_whisper': whisper}), \
             patch('src.transcription.transcribe.audio_duration', return_value=duration):
            output = transcribe('fixture', base_dir=root)
        analyze('fixture', base_dir=root)
        with patch('src.quality.validate_session.audio_duration', return_value=duration):
            report = validate_session('fixture', base_dir=root)
        return output, report

    def test_clipping_updates_exports_metrics_and_warns_with_audit(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output, report = self.process_fixture(root, 7, 35.822)
            self.assertEqual(pd.read_parquet(output).iloc[0].end_s, 10)
            self.assertEqual(pd.read_csv(output.with_suffix('.csv')).iloc[0].end_s, 10)
            self.assertIn('teste final', output.with_name('transcricao.txt').read_text())
            metrics = pd.read_csv(root / 'data/curated/fixture/speaker_metrics.csv')
            self.assertEqual(metrics.iloc[0].estimated_speech_seconds, 3)
            self.assertEqual(report['status'], 'WARN')
            event = report['timestamp_events'][0]
            self.assertEqual(event['original_end_s'], 35.822)
            self.assertEqual(event['action'], 'clipped')
            metadata = json.loads(output.with_name('metadata.json').read_text())
            self.assertEqual(metadata['timestamp_events'], report['timestamp_events'])

    def test_outside_audio_is_preserved_and_fails_even_within_tolerance(self):
        with TemporaryDirectory() as temp:
            output, report = self.process_fixture(Path(temp), 10, 11)
            self.assertEqual(pd.read_parquet(output).iloc[0].end_s, 11)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['timestamp_events'][0]['action'], 'review')

    def test_unknown_duration_warns_without_modifying_interval(self):
        with TemporaryDirectory() as temp:
            output, report = self.process_fixture(Path(temp), 7, 35, None)
            self.assertEqual(pd.read_parquet(output).iloc[0].end_s, 35)
            self.assertEqual(report['status'], 'WARN')

    def test_valid_timestamps_remain_pass(self):
        with TemporaryDirectory() as temp:
            _, report = self.process_fixture(Path(temp), 7, 10)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['timestamp_events'], [])
