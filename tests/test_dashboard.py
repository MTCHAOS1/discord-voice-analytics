from contextlib import redirect_stdout
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.dashboard.report import demo_dashboard, load_dashboard, write_dashboard, display_name
from src.quality.validate_session import validate_session
from src.transformation.analyze import analyze

PROJECT = Path(__file__).resolve().parents[1]


class EmbeddedData(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.active = False
        self.chunks = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = dict(attrs).get('id') == 'dashboard-data'

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, text):
        if self.active:
            self.chunks.append(text)


class TestDashboard(unittest.TestCase):
    def test_participant_names_in_metrics_and_word_frequencies(self):
        self.enterContext(patch.dict('src.dashboard.report.PARTICIPANT_NAMES',
                                    {'participante_b': 'Participante B'}, clear=True))
        self.assertEqual(display_name('unknown'), 'unknown')
        self.prepare()
        from src.database.load_warehouse import build_plan
        plan = build_plan(self.session, self.root)
        plan['data']['metrics']['speaker'] = 'participante_b'
        plan['data']['segments']['speaker'] = 'participante_b'
        with patch('src.dashboard.report.build_plan', return_value=plan):
            data = load_dashboard(self.session, self.root)
        self.assertEqual(data['participants'][0]['speaker'], 'Participante B')
        self.assertEqual({row['speaker'] for row in data['words']}, {'Participante B'})
        self.assertEqual(plan['data']['metrics'].iloc[0]['speaker'], 'participante_b')

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = 'synthetic_dashboard'
        self.enterContext(redirect_stdout(StringIO()))

    def prepare(self):
        raw = self.root / 'data/raw' / self.session
        raw.mkdir(parents=True)
        (raw / '1-alpha.wav').write_bytes(b'synthetic fixture, never decoded')
        processed = self.root / 'data/processed' / self.session
        processed.mkdir(parents=True)
        # Mais de 30 palavras distintas para testar o limite do CSV de analise.
        terms = ['termo' + chr(97 + i // 26) + chr(97 + i % 26) for i in range(35)]
        pd.DataFrame([dict(session_id=self.session, speaker='alpha', source_file='1-alpha.wav',
                           start_s=0.0, end_s=3.0, text=' '.join(terms))]).to_parquet(
                               processed / 'segments.parquet', index=False)
        analyze(self.session, base_dir=self.root)
        with patch('src.quality.validate_session.audio_duration', return_value=4.0):
            validate_session(self.session, self.root)

    def test_load_validated_data_without_embedding_transcript_or_audio(self):
        self.prepare()
        data = load_dashboard(self.session, self.root)
        self.assertEqual(data['quality_status'], 'PASS')
        self.assertEqual(len(data['words']), 35)
        self.assertEqual(len(data['vocabulary']), 35)
        self.assertEqual(data['participants'][0]['content_word_count'], 35)
        self.assertNotIn('segments', data)
        self.assertNotIn('text', data)
        self.assertNotIn('source_file', data['participants'][0])
        detailed = load_dashboard(self.session, self.root, include_segments=True)
        self.assertEqual(detailed['segments'][0]['speaker'], 'alpha')
        self.assertIn('text', detailed['segments'][0])
        self.assertNotIn('source_file', detailed['segments'][0])
        html = write_dashboard(data, self.root / 'dashboard.html').read_text()
        self.assertNotIn('synthetic fixture, never decoded', html)
        self.assertNotIn('1-alpha.wav', html)
        embedded = json.loads(''.join(EmbeddedData(html).chunks))
        self.assertEqual(embedded['participants'], data['participants'])

    def test_fail_and_warn_gates(self):
        self.prepare()
        report_path = self.root / 'data/quality' / self.session / 'validation_report.json'
        report = json.loads(report_path.read_text())
        for status in ('FAIL', 'WARN'):
            with self.subTest(status=status):
                report['status'] = status
                report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    load_dashboard(self.session, self.root)
        self.assertEqual(load_dashboard(self.session, self.root, allow_warnings=True)['quality_status'], 'WARN')
        report['status'] = 'FAIL'
        report_path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            load_dashboard(self.session, self.root, allow_warnings=True)

    def test_missing_and_stale_data_block_generation(self):
        with self.assertRaises(FileNotFoundError):
            load_dashboard(self.session, self.root)
        self.prepare()
        path = self.root / 'data/curated' / self.session / 'speaker_metrics.csv'
        metrics = pd.read_csv(path)
        metrics.loc[0, 'estimated_speech_seconds'] = 20
        metrics.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'Metricas divergem'):
            load_dashboard(self.session, self.root)

    def test_embedded_json_cannot_break_out_of_script(self):
        data = demo_dashboard()
        hostile = '</script><script>alert("x")</script>&'
        data['participants'][0]['speaker'] = hostile
        html = write_dashboard(data, self.root / 'escaped.html').read_text()
        self.assertNotIn(hostile, html)
        embedded = json.loads(''.join(EmbeddedData(html).chunks))
        self.assertEqual(embedded['participants'][0]['speaker'], hostile)

    def test_does_not_overwrite_existing_file(self):
        path = self.root / 'existing.html'
        path.write_text('original')
        with self.assertRaises(FileExistsError):
            write_dashboard(demo_dashboard(), path)
        self.assertEqual(path.read_text(), 'original')
        with self.assertRaises(ValueError):
            write_dashboard(demo_dashboard(), self.root / 'existing.py')

    def test_cli_demo_runs_outside_project_without_reading_sessions(self):
        output = self.root / 'demo.html'
        result = subprocess.run([sys.executable, str(PROJECT / 'main.py'), 'dashboard', '--demo',
                                 '--output', str(output)], cwd=self.root,
                                capture_output=True, text=True, check=True)
        self.assertIn('Dashboard gerado', result.stdout)
        data = json.loads(''.join(EmbeddedData(output.read_text()).chunks))
        self.assertTrue(data['demo'])
        self.assertEqual(data['session_id'], 'demo_sintetica')
        self.assertFalse((self.root / 'data').exists())
