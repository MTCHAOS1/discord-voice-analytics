from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.verify_minio_integration import verify


class TestMinioIntegrationCheck(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.bucket_exists.return_value = True
        self.first = dict(planned=1, uploaded=1, skipped=0, dry_run=False)
        self.second = dict(planned=1, uploaded=0, skipped=1, dry_run=False)
        self.payload = b'Discord Voice Analytics: synthetic MinIO integration fixture.\n'
        self.digest = sha256(self.payload).hexdigest()
        self.client.stat_object.return_value = SimpleNamespace(
            size=len(self.payload), metadata={'X-Amz-Meta-Sha256': self.digest},
        )
        self.response = Mock()
        self.response.read.return_value = self.payload
        self.client.get_object.return_value = self.response

    def test_success_checks_remote_content_and_second_sync(self):
        with patch('scripts.verify_minio_integration.sync_session',
                   side_effect=[self.first, self.second]) as sync:
            result = verify(self.client, 'test-bucket')
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['remote_sha256'], self.digest)
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(sync.call_args_list[0], sync.call_args_list[1])
        self.client.get_object.assert_called_once_with('test-bucket', result['object_name'])
        self.response.close.assert_called_once()
        self.response.release_conn.assert_called_once()
        self.client.remove_object.assert_not_called()

    def test_remote_content_mismatch_fails_even_with_matching_metadata(self):
        self.client.get_object.return_value.read.return_value = b'corrupt'
        with patch('scripts.verify_minio_integration.sync_session', return_value=self.first):
            with self.assertRaisesRegex(RuntimeError, 'conteudo remoto'):
                verify(self.client, 'test-bucket')
        self.client.get_object.return_value.release_conn.assert_called_once()

    def test_second_upload_is_rejected(self):
        with patch('scripts.verify_minio_integration.sync_session',
                   side_effect=[self.first, self.first]):
            with self.assertRaisesRegex(RuntimeError, 'segunda sincronizacao'):
                verify(self.client, 'test-bucket')

    def test_missing_bucket_never_syncs_or_creates_bucket(self):
        self.client.bucket_exists.return_value = False
        with patch('scripts.verify_minio_integration.sync_session') as sync:
            with self.assertRaisesRegex(RuntimeError, 'bucket configurado'):
                verify(self.client, 'test-bucket')
        sync.assert_not_called()
        self.client.make_bucket.assert_not_called()
