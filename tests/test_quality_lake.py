import json
import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.quality.validate_session import validate_session
from src.ingestion.sync_data_lake import list_session_objects, sync_session


class TestQualityAndLake(unittest.TestCase):
    def prepare(self, root: Path):
        raw = root / 'data/raw/session_001'
        raw.mkdir(parents=True)
        (raw / '1-participante_a.flac').write_bytes(b'fake-audio-for-test')
        (raw / '2-participante_b.flac').write_bytes(b'fake-audio-for-test')
        processed = root / 'data/processed/session_001'
        processed.mkdir(parents=True)
        rows = [
            dict(session_id='session_001', speaker='participante_a', source_file='1-participante_a.flac',
                 start_s=1.0, end_s=3.0, text='Vamos jogar hoje'),
            dict(session_id='session_001', speaker='participante_b', source_file='2-participante_b.flac',
                 start_s=2.0, end_s=4.0, text='Boa ideia'),
        ]
        pd.DataFrame(rows).to_parquet(processed / 'segments.parquet', index=False)
        return processed

    def test_validation_pass(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            with patch('src.quality.validate_session.audio_duration', return_value=300.0):
                report = validate_session('session_001', root)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['summary']['transcript_segments'], 2)
            out = root / 'data/quality/session_001/validation_report.json'
            self.assertEqual(json.loads(out.read_text())['summary']['audio_tracks'], 2)

    def test_wrong_speaker_is_error(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            processed = self.prepare(root)
            parquet = processed / 'segments.parquet'
            data = pd.read_parquet(parquet)
            data.loc[0, 'speaker'] = 'participante_b'
            data.to_parquet(parquet, index=False)
            with patch('src.quality.validate_session.audio_duration', return_value=300.0):
                report = validate_session('session_001', root)
            self.assertEqual(report['status'], 'FAIL')
            self.assertTrue(any('errado' in e for e in report['errors']))

    def test_dry_run_never_uploads(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            objects = list_session_objects('session_001', root)
            self.assertEqual(len(objects), 3)
            self.assertTrue(objects[0][1].startswith('bronze/session_001/'))
            result = sync_session('session_001', apply=False, base_dir=root)
            self.assertEqual(result['uploaded'], 0)
            self.assertTrue(result['dry_run'])

    def test_upload_and_idempotency_with_fake_minio(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class FakeS3Error(Exception):
            def __init__(self, code):
                self.code = code
        class FakeClient:
            store = {}
            def __init__(self, *args, **kwargs):
                pass
            def bucket_exists(self, name):
                return True
            def stat_object(self, bucket, key):
                if key not in self.store:
                    raise FakeS3Error('NoSuchKey')
                return self.store[key]
            def fput_object(self, bucket, key, file_path, content_type, metadata):
                self.store[key] = SimpleNamespace(
                    size=Path(file_path).stat().st_size,
                    metadata={'x-amz-meta-sha256': metadata['sha256']})
        FakeClient.store.clear()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            mod = types.ModuleType('minio')
            mod.Minio = FakeClient
            errors = types.ModuleType('minio.error')
            errors.S3Error = FakeS3Error
            config = {'MINIO_ACCESS_KEY': 'testuser', 'MINIO_SECRET_KEY': 'testsecret'}
            with patch.dict(sys.modules, {'minio': mod, 'minio.error': errors}), \
                 patch.dict(os.environ, config, clear=False):
                r1 = sync_session('session_001', apply=True, base_dir=root)
                r2 = sync_session('session_001', apply=True, base_dir=root)
            self.assertEqual(r1['uploaded'], 3)
            self.assertEqual(r2['uploaded'], 0)
            self.assertEqual(r2['skipped'], 3)


if __name__ == '__main__':
    unittest.main()
