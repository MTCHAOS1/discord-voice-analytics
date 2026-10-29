import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.ingestion.session_catalog import session_catalog


class TestSessionCatalog(unittest.TestCase):
    def test_craig_date_uses_sao_paulo_and_unknown_dates_are_preserved(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for session in ['known', 'unknown', 'metadata']:
                processed = root / 'data/processed' / session
                processed.mkdir(parents=True)
                (processed / 'segments.parquet').touch()
            raw = root / 'data/raw/known'
            raw.mkdir(parents=True)
            (raw / 'info.txt').write_text('Recording test\nStart time:\t2026-10-28T01:30:00Z\n')
            (root / 'data/processed/metadata/metadata.json').write_text(json.dumps(
                {'recording_started_at': '2026-10-29T01:00:00+00:00'}))
            catalog = {row['session_id']: row for row in session_catalog(root)}
            self.assertEqual(catalog['known']['day'], '2026-10-27')
            self.assertEqual(catalog['metadata']['day'], '2026-10-28')
            self.assertIsNone(catalog['unknown']['day'])

    def test_invalid_or_naive_timestamp_does_not_invent_date(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            processed = root / 'data/processed/fixture'
            processed.mkdir(parents=True)
            (processed / 'segments.parquet').touch()
            for value in ['2026-10-27T10:00:00', 'invalid']:
                (processed / 'metadata.json').write_text(json.dumps({'recording_started_at': value}))
                self.assertIsNone(session_catalog(root)[0]['day'])
