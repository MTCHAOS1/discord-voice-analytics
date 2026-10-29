"""Pipeline com WAVs sinteticos; nenhum modelo ou gravacao privada e aberto."""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import pandas as pd

from src.database.load_warehouse import load_session
from src.quality.validate_session import validate_session
from src.transcription.transcribe import transcribe
from src.transformation.analyze import analyze

PROJECT = Path(__file__).resolve().parents[1]


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='dva-pipeline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = 'synthetic_pipeline'
        raw = self.root / 'data/raw' / self.session
        raw.mkdir(parents=True)
        for name in ('1-alpha.wav', '2-beta.wav'):
            with wave.open(str(raw / name), 'wb') as audio:
                audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * 16000 * 4)
        self.enterContext(redirect_stdout(StringIO()))
        module = ModuleType('faster_whisper')
        self.model = MagicMock()
        module.WhisperModel = self.model
        previous = sys.modules.get('faster_whisper')
        sys.modules['faster_whisper'] = module
        def restore_whisper():
            if previous is None:
                sys.modules.pop('faster_whisper', None)
            else:
                sys.modules['faster_whisper'] = previous
        self.addCleanup(restore_whisper)
        self.enterContext(patch('src.transcription.transcribe.audio_duration', return_value=4.0))
        self.duration = self.enterContext(patch('src.quality.validate_session.audio_duration', return_value=4.0))
        sql = self.root / 'sql/warehouse'
        sql.mkdir(parents=True)
        shutil.copyfile(PROJECT / 'sql/warehouse/001_init.sql', sql / '001_init.sql')

    def prepare(self):
        self.model.return_value.transcribe.side_effect = [
            (iter([SimpleNamespace(start=0, end=1, text=' Jogo divertido '),
                   SimpleNamespace(start=2, end=3, text='Jogo novo')]), None),
            (iter([SimpleNamespace(start=0.5, end=2.5, text='Jogo divertido')]), None),
        ]
        self.parquet = transcribe(self.session, base_dir=self.root)
        analyze(self.session, base_dir=self.root)
        return validate_session(self.session, self.root)

    def load(self, **kwargs):
        return load_session(self.session, base_dir=self.root, **kwargs)


class TestPipeline(PipelineFixture):
    def test_end_to_end_outputs_and_repeat_without_duplicates(self):
        report = self.prepare()
        self.assertEqual(report['status'], 'PASS')
        original = pd.read_parquet(self.parquet)
        self.assertEqual(original['start_s'].tolist(), [0, 0.5, 2])
        metrics_path = self.root / 'data/curated' / self.session / 'speaker_metrics.csv'
        metrics = pd.read_csv(metrics_path).set_index('speaker')
        self.assertEqual(metrics.loc['alpha', 'segments'], 2)
        self.assertEqual(metrics.loc['alpha', 'estimated_speech_seconds'], 2)
        self.assertEqual(metrics.loc['alpha', 'content_word_count'], 4)
        terms = pd.read_csv(metrics_path.with_name('top_words.csv'))
        self.assertEqual(int(terms.query("speaker == 'alpha' and word == 'jogo'")['count'].iloc[0]), 2)
        with patch('src.database.load_warehouse._pg_connection') as connect:
            result = self.load()
            connect.assert_not_called()
        self.assertEqual(result, dict(dry_run=True, participants=2, segments=3, words=5, audio_tracks=2))
        self.prepare()
        pd.testing.assert_frame_equal(original, pd.read_parquet(self.parquet))
        pd.testing.assert_frame_equal(metrics, pd.read_csv(metrics_path).set_index('speaker'))

    def test_fail_blocks_database_then_corrected_run_succeeds(self):
        self.prepare()
        data = pd.read_parquet(self.parquet)
        data.loc[0, 'speaker'] = 'unknown'
        data.to_parquet(self.parquet, index=False)
        self.assertEqual(validate_session(self.session, self.root)['status'], 'FAIL')
        with patch('src.database.load_warehouse._pg_connection') as connect:
            with self.assertRaisesRegex(ValueError, 'FAIL'):
                self.load(apply=True)
            connect.assert_not_called()
        self.assertEqual(self.prepare()['status'], 'PASS')
        self.assertEqual(self.load()['segments'], 3)

    def test_warn_requires_explicit_override(self):
        self.prepare()
        self.duration.return_value = None
        self.assertEqual(validate_session(self.session, self.root)['status'], 'WARN')
        with self.assertRaisesRegex(ValueError, 'WARN'):
            self.load()
        self.assertTrue(self.load(allow_warnings=True)['dry_run'])

    def test_empty_transcription_cannot_replace_loaded_data(self):
        self.model.return_value.transcribe.side_effect = [(iter([]), None), (iter([]), None)]
        transcribe(self.session, base_dir=self.root)
        analyze(self.session, base_dir=self.root)
        self.assertEqual(validate_session(self.session, self.root)['status'], 'WARN')
        with patch('src.database.load_warehouse._pg_connection') as connect:
            with self.assertRaisesRegex(ValueError, 'vazios'):
                self.load(apply=True, allow_warnings=True)
            connect.assert_not_called()

    def test_stale_metrics_same_segment_count_block_load(self):
        self.prepare()
        data = pd.read_parquet(self.parquet)
        data.loc[0, 'end_s'] = 1.5
        data.to_parquet(self.parquet, index=False)
        self.assertEqual(validate_session(self.session, self.root)['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'Metricas divergem'):
            self.load()
        analyze(self.session, base_dir=self.root)
        self.assertTrue(self.load()['dry_run'])

    def test_model_failure_preserves_previous_transcription(self):
        self.prepare()
        original = self.parquet.read_bytes()
        self.model.return_value.transcribe.side_effect = RuntimeError('synthetic model failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic model failure'):
            transcribe(self.session, base_dir=self.root)
        self.assertEqual(self.parquet.read_bytes(), original)

    def test_infinite_timestamps_block_load(self):
        self.prepare()
        data = pd.read_parquet(self.parquet)
        data.loc[0, 'end_s'] = float('inf')
        data.to_parquet(self.parquet, index=False)
        with self.assertRaisesRegex(ValueError, 'Timestamps invalidos'):
            self.load()

    def test_invalid_metric_numbers_block_load(self):
        self.prepare()
        path = self.root / 'data/curated' / self.session / 'speaker_metrics.csv'
        original = pd.read_csv(path)
        for field, value in [('estimated_speech_seconds', float('inf')),
                             ('content_word_count', 1.5)]:
            with self.subTest(field=field):
                data = original.astype({field: float})
                data.loc[0, field] = value
                data.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'Metrica'):
                    self.load()

    def test_database_failure_propagates_to_transaction_context(self):
        self.prepare()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.executemany.side_effect = RuntimeError('synthetic database failure')
        with patch('src.database.load_warehouse._pg_connection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, 'synthetic database failure'):
                self.load(apply=True)
        # O contexto recebe a excecao: a implementacao psycopg decide o rollback.
        self.assertIs(connection.__exit__.call_args.args[0], RuntimeError)
        self.assertIn('DELETE FROM warehouse.sessions', cursor.execute.call_args_list[1].args[0])
        self.assertEqual(cursor.execute.call_args_list[1].args[1], (self.session,))


@unittest.skipUnless(os.environ.get('RUN_TEMP_POSTGRES_TESTS') == '1',
                     'PostgreSQL temporario requer RUN_TEMP_POSTGRES_TESTS=1')
class TestPipelinePostgres(PipelineFixture):
    def test_real_reload_and_rollback(self):
        import psycopg
        binaries = sorted(Path('/usr/lib/postgresql').glob('*/bin/initdb'))
        initdb = shutil.which('initdb') or (str(binaries[-1]) if binaries else None)
        if not initdb:
            self.fail('Instale os binarios PostgreSQL (initdb e pg_ctl).')
        pg_ctl = str(Path(initdb).with_name('pg_ctl'))
        cluster = self.root / 'pgdata'
        sockets = self.root / 'socket'
        sockets.mkdir()
        subprocess.run([initdb, '-D', str(cluster), '-A', 'trust', '--no-locale', '-E', 'UTF8'],
                       capture_output=True, text=True, check=True)
        log = self.root / 'postgres.log'
        start = subprocess.run([pg_ctl, '-D', str(cluster), '-l', str(log),
                                '-o', f"-h '' -k {sockets}", '-w', 'start'],
                               capture_output=True, text=True)
        if start.returncode:
            self.fail('PostgreSQL temporario nao iniciou: ' + log.read_text())
        self.addCleanup(subprocess.run, [pg_ctl, '-D', str(cluster), '-m', 'fast', '-w', 'stop'],
                        capture_output=True, check=True)

        def connect():
            return psycopg.connect(host=str(sockets), dbname='postgres')

        self.prepare()
        with patch('src.database.load_warehouse._pg_connection', side_effect=connect):
            self.load(apply=True)
            self.load(apply=True)
            with connect() as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM warehouse.sessions').fetchone()[0], 1)
                self.assertEqual(conn.execute('SELECT count(*) FROM warehouse.transcript_segments').fetchone()[0], 3)
                tables = ('sessions', 'participants', 'transcript_segments', 'word_frequency')
                before = {table: conn.execute(f'SELECT * FROM warehouse.{table} ORDER BY 1, 2').fetchall()
                          for table in tables}
                conn.execute("CREATE FUNCTION warehouse.reject_test() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'synthetic failure'; END; $$")
                conn.execute('CREATE TRIGGER reject_test BEFORE INSERT ON warehouse.word_frequency FOR EACH ROW EXECUTE FUNCTION warehouse.reject_test()')
            with self.assertRaisesRegex(psycopg.Error, 'synthetic failure'):
                self.load(apply=True)
            with connect() as conn:
                for table in tables:
                    self.assertEqual(conn.execute(f'SELECT * FROM warehouse.{table} ORDER BY 1, 2').fetchall(), before[table])
                self.assertEqual(conn.execute('SELECT count(*) FROM warehouse.transcript_segments').fetchone()[0], 3)
                conn.execute('DROP TRIGGER reject_test ON warehouse.word_frequency')
            self.load(apply=True)
