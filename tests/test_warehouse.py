import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from src.database.load_warehouse import build_plan, load_session


class TestWarehouse(unittest.TestCase):
    def setup_data(self, root, status='PASS'):
        p = root / 'data/processed/session_001'
        g = root / 'data/curated/session_001'
        q = root / 'data/quality/session_001'
        p.mkdir(parents=True); g.mkdir(parents=True); q.mkdir(parents=True)
        segments = pd.DataFrame([
            dict(session_id='session_001', speaker='participante_a', source_file='1-participante_a.flac', start_s=1.0, end_s=2.0, text='Vamos jogar'),
            dict(session_id='session_001', speaker='participante_b', source_file='2-participante_b.flac', start_s=1.4, end_s=2.2, text='Bora'),
        ])
        (p/'segments.parquet').write_bytes(b'parquet-fixture')
        metrics = pd.DataFrame([dict(speaker='participante_a', segments=1, estimated_speech_seconds=1.0, content_word_count=2),
                                dict(speaker='participante_b', segments=1, estimated_speech_seconds=.8, content_word_count=1)])
        metrics.to_csv(g/'speaker_metrics.csv', index=False)
        words = pd.DataFrame([dict(speaker='participante_a', word='jogar', count=1),
                              dict(speaker='participante_b', word='bora', count=1)])
        words.to_csv(g/'top_words.csv', index=False)
        (q/'validation_report.json').write_text(json.dumps(dict(status=status, session_id='session_001', summary={
            'audio_tracks':2,'transcript_segments':2,'participants_with_segments':2})), encoding='utf-8')
        return segments

    def test_dry_run_never_connects(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); seg=self.setup_data(root)
            with patch('src.database.load_warehouse.pd.read_parquet', return_value=seg), \
                 patch('src.database.load_warehouse._pg_connection', side_effect=AssertionError('DB access in dry-run')):
                result=load_session('session_001', base_dir=root)
            self.assertTrue(result['dry_run'])
            self.assertEqual(result['segments'], 2)

    def test_reject_failed_report(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);seg=self.setup_data(root, 'FAIL')
            with patch('src.database.load_warehouse.pd.read_parquet', return_value=seg):
                with self.assertRaisesRegex(ValueError, 'FAIL'):
                    build_plan('session_001', base_dir=root)

    def test_reject_stale_report(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);seg=self.setup_data(root)
            seg=seg.iloc[:1].copy()
            with patch('src.database.load_warehouse.pd.read_parquet', return_value=seg):
                with self.assertRaisesRegex(ValueError, 'Quantidade de trechos'):
                    build_plan('session_001', base_dir=root)

    def test_reject_wrong_word_speaker(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);seg=self.setup_data(root)
            pd.DataFrame([dict(speaker='someone_else',word='x',count=1)]).to_csv(
                root/'data/curated/session_001/top_words.csv', index=False)
            with patch('src.database.load_warehouse.pd.read_parquet', return_value=seg):
                with self.assertRaisesRegex(ValueError, 'desconhecido'):
                    build_plan('session_001', base_dir=root)

    def test_warn_requires_opt_in(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);seg=self.setup_data(root,'WARN')
            with patch('src.database.load_warehouse.pd.read_parquet', return_value=seg):
                with self.assertRaisesRegex(ValueError, 'WARN'):
                    build_plan('session_001', base_dir=root)
                result=build_plan('session_001', base_dir=root, allow_warnings=True)
                self.assertEqual(result['counts']['segments'],2)


if __name__ == '__main__':
    unittest.main()
