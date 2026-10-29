"""Regressao MinIO com SDK real e I/O remoto inteiramente simulado."""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minio.error import S3Error

from src.ingestion.sync_data_lake import file_sha256, sync_session


class TestSyncDataLake(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'data/processed/test_session/segments.parquet'
        self.path.parent.mkdir(parents=True)
        self.payload = b'synthetic parquet fixture'
        self.path.write_bytes(self.payload)
        self.digest = sha256(self.payload).hexdigest()
        self.enterContext(redirect_stdout(StringIO()))
        self.env_loader = self.enterContext(patch('dotenv.load_dotenv'))
        self.enterContext(patch.dict(os.environ, {
            'MINIO_ACCESS_KEY': 'test-access', 'MINIO_SECRET_KEY': 'test-secret',
            'MINIO_BUCKET': 'test-bucket',
        }, clear=True))
        # Autospec rejeita keywords incompativeis com a versao instalada.
        self.minio = self.enterContext(patch('minio.Minio', autospec=True))
        self.client = self.minio.return_value
        self.client.bucket_exists.return_value = True

    def sync(self):
        return sync_session('test_session', apply=True, base_dir=self.root)

    def test_upload_uses_sdk_metadata_and_preserves_sha256(self):
        self.client.stat_object.side_effect = S3Error(
            response=None, code='NoSuchKey', message='missing',
            resource='resource', request_id='request', host_id='host',
        )
        result = self.sync()
        self.client.fput_object.assert_called_once_with(
            'test-bucket', 'silver/test_session/segments.parquet', str(self.path),
            content_type='application/vnd.apache.parquet',
            metadata={'sha256': self.digest},
        )
        self.assertEqual(result['uploaded'], 1)
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertEqual(file_sha256(self.path), self.digest)

    def test_same_size_and_sha256_skip_upload(self):
        self.client.stat_object.return_value = SimpleNamespace(
            size=len(self.payload), metadata={'X-Amz-Meta-Sha256': self.digest},
        )
        result = self.sync()
        self.assertEqual(result['skipped'], 1)
        self.client.fput_object.assert_not_called()

    def test_changed_or_missing_metadata_triggers_upload(self):
        for metadata in ({'x-amz-meta-sha256': '0' * 64}, {}, None):
            with self.subTest(metadata=metadata):
                self.client.fput_object.reset_mock()
                self.client.stat_object.return_value = SimpleNamespace(
                    size=len(self.payload), metadata=metadata,
                )
                self.assertEqual(self.sync()['uploaded'], 1)
                self.assertEqual(
                    self.client.fput_object.call_args.kwargs['metadata'],
                    {'sha256': self.digest},
                )

    def test_dry_run_does_not_initialize_client_or_load_env(self):
        result = sync_session('test_session', base_dir=self.root)
        self.assertTrue(result['dry_run'])
        self.minio.assert_not_called()
        self.env_loader.assert_not_called()

    def test_sha256_streams_multiple_chunks(self):
        payload = b'abc123' * 400_000
        self.path.write_bytes(payload)
        self.assertEqual(file_sha256(self.path), sha256(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
