"""Orquestração do CLI sem baixar modelos ou acessar gravações."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from main import main, print_process_summary


class TestProcessCli(unittest.TestCase):
    def run_process(self, status='PASS', fail_stage=None, extra=()):
        calls = Mock()
        output = StringIO()
        with patch('sys.argv', ['main.py', 'process', '--session', 'fixture', *extra]), \
             patch('src.transcription.transcribe.transcribe') as transcribe, \
             patch('src.transformation.analyze.analyze') as analyze, \
             patch('src.quality.validate_session.validate_session') as validate, \
             redirect_stdout(output), redirect_stderr(output):
            calls.attach_mock(transcribe, 'transcribe')
            calls.attach_mock(analyze, 'analyze')
            calls.attach_mock(validate, 'validate')
            validate.return_value = {'status': status,
                                     'summary': dict(audio_tracks=7, transcript_segments=2748,
                                                     participants_with_segments=7),
                                     'timestamp_events': [dict(action='clipped'), dict(action='review')],
                                     'warnings': ['Confira o trecho final'] if status == 'WARN' else [],
                                     'errors': ['Intervalo invalido'] if status == 'FAIL' else []}
            if fail_stage:
                getattr(calls, fail_stage).side_effect = RuntimeError('falha controlada')
            code = 0
            try:
                main()
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), calls

    def test_success_order_and_defaults(self):
        code, output, calls = self.run_process()
        self.assertEqual(code, 0)
        self.assertEqual([c[0] for c in calls.mock_calls], ['transcribe', 'analyze', 'validate'])
        calls.transcribe.assert_called_once_with('fixture', 'large-v3', 'cpu')
        calls.analyze.assert_called_once_with('fixture')
        calls.validate.assert_called_once_with('fixture')
        self.assertIn('Pronta para selecionar', output)
        self.assertIn('Faixas: 7 | Trechos: 2748 | Participantes com fala: 7', output)
        self.assertIn('Timestamps ajustados: 1 | Ocorrencias para revisao: 1', output)
        self.assertIn('Dashboard: pronto.', output)

    def test_model_and_device_overrides(self):
        code, _, calls = self.run_process(extra=['--model', 'small', '--device', 'cuda'])
        self.assertEqual(code, 0)
        calls.transcribe.assert_called_once_with('fixture', 'small', 'cuda')

    def test_failure_stops_subsequent_stages(self):
        for stage, expected in [('transcribe', 1), ('analyze', 2), ('validate', 3)]:
            with self.subTest(stage=stage):
                code, output, calls = self.run_process(fail_stage=stage)
                self.assertEqual(code, 1)
                self.assertEqual(len(calls.mock_calls), expected)
                self.assertIn('falha controlada', output)
                self.assertNotIn('Pronta para selecionar', output)
                self.assertIn(f'{dict(transcribe="transcricao", analyze="analise", validate="validacao")[stage]}: falhou', output)
                self.assertIn('Dashboard: processamento incompleto', output)
                if stage == 'transcribe':
                    self.assertIn('analise: nao executada', output)

    def test_quality_fail_blocks_and_warn_requires_review(self):
        for status, expected in [('FAIL', 1), ('WARN', 0)]:
            with self.subTest(status=status):
                code, output, _ = self.run_process(status=status)
                self.assertEqual(code, expected)
                self.assertIn(status, output)
                self.assertIn('validation_report.json', output)
                self.assertNotIn('Pronta para selecionar', output)
                self.assertIn('Dashboard: bloqueado' if status == 'FAIL' else 'Dashboard: revisao pendente', output)
                self.assertIn('Intervalo invalido' if status == 'FAIL' else 'Confira o trecho final', output)

    def test_summary_shows_current_backup_and_elapsed_time(self):
        for backup in ['/tmp/snapshot', None]:
            with self.subTest(backup=backup):
                output = StringIO()
                with redirect_stdout(output), patch('main.monotonic', return_value=15):
                    print_process_summary(SimpleNamespace(session='fixture', model='large-v3', device='cpu'),
                                          {'transcricao': 'concluida'}, 10,
                                          metadata={'backup_path': backup})
                self.assertIn('Tempo: 5.0s', output.getvalue())
                self.assertIn(backup or 'sem resultados anteriores', output.getvalue())


if __name__ == '__main__':
    unittest.main()
