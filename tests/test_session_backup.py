"""Preservação de resultados usando apenas arquivos temporários sintéticos."""
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.transcription.backup import backup_session
from src.transcription.transcribe import transcribe


class TestSessionBackup(unittest.TestCase):
    def fixture(self, root):
        for layer in ('processed', 'curated', 'quality'):
            directory = root / 'data' / layer / 'fixture'
            directory.mkdir(parents=True)
            (directory / 'original.txt').write_bytes(layer.encode())

    def test_first_session_has_no_backup(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(backup_session('fixture', root))
            self.assertFalse((root / 'data/backups').exists())

    def test_unique_complete_backups_can_restore_originals(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            backup = backup_session('fixture', root)
            second = backup_session('fixture', root)
            self.assertNotEqual(backup, second)
            manifest = json.loads((backup / 'manifest.json').read_text())
            self.assertEqual(len(manifest['files']), 3)
            for entry in manifest['files']:
                self.assertEqual(hashlib.sha256((backup / entry['path']).read_bytes()).hexdigest(),
                                 entry['sha256'])
            for layer in manifest['layers']:
                target = root / 'data' / layer / 'fixture'
                shutil.rmtree(target)
                shutil.copytree(backup / layer, target)
                self.assertEqual((target / 'original.txt').read_bytes(), layer.encode())

    def test_failed_copy_does_not_publish_backup_or_change_originals(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with patch('src.transcription.backup.shutil.copytree', side_effect=OSError('disco cheio')):
                with self.assertRaises(OSError):
                    backup_session('fixture', root)
            self.assertEqual(list((root / 'data/backups/fixture').iterdir()), [])
            self.assertEqual((root / 'data/processed/fixture/original.txt').read_bytes(), b'processed')

    def test_transcribe_aborts_before_model_or_overwrite_if_backup_fails(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            raw = root / 'data/raw/fixture'
            raw.mkdir(parents=True)
            (raw / '1-Aurora.flac').touch()
            model = Mock()
            with patch.dict('sys.modules', {'faster_whisper': SimpleNamespace(WhisperModel=model)}), \
                 patch('src.transcription.transcribe.backup_session', side_effect=OSError('disco cheio')):
                with self.assertRaisesRegex(RuntimeError, 'Backup falhou'):
                    transcribe('fixture', base_dir=root)
            model.assert_not_called()
            self.assertEqual((root / 'data/processed/fixture/original.txt').read_bytes(), b'processed')

    def test_symlinks_are_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'data/processed/fixture/link').symlink_to(root / 'data/quality/fixture')
            with self.assertRaisesRegex(ValueError, 'simbolico'):
                backup_session('fixture', root)
