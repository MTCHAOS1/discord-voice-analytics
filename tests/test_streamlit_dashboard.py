"""Interacoes Streamlit com dados sinteticos, sem servidor ou gravações."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.dashboard.report import demo_dashboard


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Instale requirements-dashboard.txt')
class TestStreamlitDashboard(unittest.TestCase):
    def setUp(self):
        catalog = [dict(session_id='session_001', day='2026-10-26'),
                   dict(session_id='session_002', day='2026-10-27'),
                   dict(session_id='session_003', day='2026-10-27')]
        patcher = patch('src.ingestion.session_catalog.session_catalog', return_value=catalog)
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self, demo=True):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'))
        if demo:
            app.session_state['source'] = 'Demonstração'
        return app

    def test_starts_with_local_session_instead_of_synthetic_data(self):
        data = demo_dashboard()
        data.update(session_id='local_fixture', demo=False, quality_status='PASS')
        data['participants'] = [dict(speaker='Local', segments=69,
                                    estimated_speech_seconds=926.06, content_word_count=331)]
        data['words'] = []
        with patch('src.dashboard.report.load_dashboard', return_value=data) as load:
            app = self.app(demo=False).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='source').value, 'Sessão local')
            self.assertEqual(load.call_count, 3)
            self.assertEqual(app.selectbox(key='day_filter').value, 'Todos os dias')
            self.assertEqual([m.value for m in app.metric], ['46min 18s', '1', '207', '993'])
            self.assertTrue(app.success)

    def test_demo_filters_and_reset(self):
        with patch('src.dashboard.report.load_dashboard') as load:
            app = self.app().run()
            self.assertFalse(app.exception)
            load.assert_not_called()
            self.assertEqual([m.value for m in app.metric], ['7min 0s', '4', '60', '860'])

            app.selectbox(key='speaker').select('Aurora').run()
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.metric], ['3min 0s', '1', '24', '360'])
            self.assertAlmostEqual(app.dataframe[0].value.iloc[0]['Participação (%)'], 180 / 420 * 100)
            for limit in [5, 10, 20]:
                app.selectbox(key='limit').select(limit).run()
                self.assertFalse(app.exception)
            app.selectbox(key='speaker').select(None).run()
            self.assertEqual([m.value for m in app.metric], ['7min 0s', '4', '60', '860'])

    def test_participation_can_switch_between_donut_and_bars(self):
        app = self.app().run()
        self.assertEqual(app.radio(key='participation_chart').value, 'Rosca')
        app.selectbox(key='speaker').select('Aurora').run()
        for mode in ['Barras', 'Rosca']:
            app.radio(key='participation_chart').set_value(mode).run()
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.metric], ['3min 0s', '1', '24', '360'])
            self.assertAlmostEqual(app.dataframe[0].value.iloc[0]['Participação (%)'], 180 / 420 * 100)

    def test_word_cloud_navigation_and_participant_counts(self):
        app = self.app().run()
        app.radio(key='page').set_value('Nuvem de palavras').run()
        self.assertFalse(app.exception)
        table = app.dataframe[0].value
        counts = table.groupby('Palavra')['Ocorrências'].sum()
        self.assertEqual(counts['equipe'], 34)
        self.assertEqual(set(table['Participante']), {'Aurora', 'Bento', 'Cora', 'Davi'})
        from io import BytesIO
        from PIL import Image
        from src.dashboard.word_cloud import render_word_cloud
        image = Image.open(BytesIO(render_word_cloud(counts.to_dict())))
        self.assertEqual(image.size, (1200, 650))
        self.assertGreater(len(image.getcolors(maxcolors=1000000)), 1)
        from src.dashboard.word_cloud import participant_colors
        colors = participant_colors(table['Participante'])
        image = Image.open(BytesIO(render_word_cloud({('Aurora', 'equipe'): 16,
                                                      ('Bento', 'equipe'): 10}, colors)))
        from PIL import ImageColor
        pixels = set(image.getdata())
        for color in colors.values():
            self.assertIn(ImageColor.getrgb(color), pixels)
        app.selectbox(key='cloud_speaker').select('Aurora').run()
        self.assertFalse(app.exception)
        counts = app.dataframe[0].value.set_index('Palavra')['Ocorrências']
        self.assertEqual(counts['equipe'], 16)
        self.assertNotIn('mapa', counts)
        app.radio(key='page').set_value('Visão geral').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)

    def test_word_cloud_empty_words(self):
        data = demo_dashboard()
        data['words'] = []
        with patch('src.dashboard.report.demo_dashboard', return_value=data):
            app = self.app().run()
            app.radio(key='page').set_value('Nuvem de palavras').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.info[-1].value, 'Nenhuma palavra de conteúdo para este filtro.')
            self.assertFalse(app.dataframe)

    def test_all_sessions_sum_metrics_and_words_and_block_invalid_session(self):
        def load(session, **kwargs):
            data = demo_dashboard()
            data.update(session_id=session, demo=False, quality_status='PASS')
            return data

        with patch('src.dashboard.report.load_dashboard', side_effect=load):
            app = self.app(demo=False).run()
            count = 3
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[1].value, '4')
            self.assertEqual(app.metric[2].value, str(60 * count))
            app.radio(key='page').set_value('Nuvem de palavras').run()
            self.assertFalse(app.exception)
            table = app.dataframe[0].value
            self.assertEqual(table.loc[table['Palavra'] == 'equipe', 'Ocorrências'].sum(), 34 * count)
            app.selectbox(key='day_filter').select('2026-10-27').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.loc[lambda t: t['Palavra'] == 'equipe', 'Ocorrências'].sum(), 68)
            app.radio(key='page').set_value('Visão geral').run()
            self.assertEqual(app.selectbox(key='day_filter').value, '2026-10-27')
            self.assertEqual(app.metric[2].value, '120')
            app.selectbox(key='day_filter').select('2026-10-26').run()
            self.assertEqual(app.metric[2].value, '60')
            app.selectbox(key='day_filter').select('Todos os dias').run()
            self.assertEqual(app.metric[2].value, '180')
        with patch('src.dashboard.report.load_dashboard', side_effect=ValueError('Relatorio FAIL')):
            app.run()
            self.assertTrue(app.error)
            self.assertFalse(app.metric)

    def test_local_load_fail_clears_metrics(self):
        with patch('src.dashboard.report.load_dashboard', side_effect=ValueError('Relatorio FAIL')):
            app = self.app().run()
            app.radio(key='source').set_value('Sessão local').run()
            self.assertFalse(app.exception)
            self.assertIn('Relatorio FAIL', app.error[0].value)
            self.assertEqual(len(app.metric), 0)

    def test_unknown_date_remains_in_unified_view_and_can_be_filtered(self):
        catalog = [dict(session_id='unknown', day=None), dict(session_id='known', day='2026-10-27')]
        def load(session, **kwargs):
            data = demo_dashboard()
            data.update(session_id=session, demo=False, quality_status='PASS')
            return data
        with patch('src.ingestion.session_catalog.session_catalog', return_value=catalog), \
             patch('src.dashboard.report.load_dashboard', side_effect=load):
            app = self.app(demo=False).run()
            self.assertEqual(app.metric[2].value, '120')
            app.selectbox(key='day_filter').select('Sem data identificada').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[2].value, '60')

    def test_no_sessions_shows_guidance(self):
        with patch('src.ingestion.session_catalog.session_catalog', return_value=[]):
            app = self.app(demo=False).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.metric)
            self.assertIn('Nenhuma sessão processada', app.info[0].value)

    def test_dictionary_person_search_sort_and_empty_result(self):
        app = self.app().run()
        app.radio(key='page').set_value('Dicionário pessoal').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='dictionary_speaker').value, 'Aurora')
        self.assertEqual(app.metric[0].value, '3')
        app.selectbox(key='dictionary_speaker').select('Bento').run()
        self.assertEqual(set(app.dataframe[0].value['Participante']), {'Bento'})
        app.selectbox(key='dictionary_speaker').select('Todos os participantes').run()
        self.assertEqual(app.metric[0].value, '5')
        app.selectbox(key='dictionary_speaker').select('Aurora').run()
        self.assertEqual(app.metric[0].value, '3')
        self.assertEqual(app.metric[1].value, '36')
        self.assertEqual(set(app.dataframe[0].value['Participante']), {'Aurora'})
        app.selectbox(key='dictionary_order').select('A–Z').run()
        words = app.dataframe[0].value['Palavra'].tolist()
        self.assertEqual(words, sorted(words))
        app.text_input(key='dictionary_search').set_value('EQUI').run()
        self.assertEqual(app.dataframe[0].value['Palavra'].tolist(), ['equipe'])
        app.text_input(key='dictionary_search').set_value('[').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.dataframe)
        self.assertEqual(app.metric[1].value, '0')

    def test_dictionary_includes_common_words_and_tracks_day_filter(self):
        def load(session, **kwargs):
            data = demo_dashboard()
            data.update(session_id=session, demo=False, quality_status='PASS')
            data['vocabulary'] = [*data['words'], dict(speaker='Aurora', word='eu', count=7)]
            return data
        with patch('src.dashboard.report.load_dashboard', side_effect=load):
            app = self.app(demo=False).run()
            app.radio(key='page').set_value('Dicionário pessoal').run()
            app.selectbox(key='dictionary_speaker').select('Aurora').run()
            app.text_input(key='dictionary_search').set_value('eu').run()
            self.assertEqual(app.metric[1].value, '21')
            app.selectbox(key='day_filter').select('2026-10-27').run()
            self.assertEqual(app.metric[1].value, '14')
            app.checkbox(key='dictionary_common').uncheck().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)

    def test_warn_requires_review_and_empty_words_are_supported(self):
        data = demo_dashboard()
        data.update(demo=False, quality_status='WARN', words=[])

        def load(session, *, base_dir, allow_warnings):
            if not allow_warnings:
                raise ValueError('Relatorio WARN: revise os alertas')
            return data

        with patch('src.dashboard.report.load_dashboard', side_effect=load):
            app = self.app().run()
            app.radio(key='source').set_value('Sessão local').run()
            self.assertTrue(app.error)
            app.checkbox(key='allow_warnings').check().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.warning)
            self.assertEqual(app.info[0].value, 'Nenhuma palavra de conteúdo para este filtro.')
            self.assertEqual(len(app.metric), 4)

    def test_all_new_analysis_pages_render_demo_without_reading_local_data(self):
        from src.dashboard.analysis_pages import PAGES
        with patch('src.dashboard.report.load_dashboard') as load:
            app = self.app().run()
            for page in PAGES:
                with self.subTest(page=page):
                    app.radio(key='page').set_value(page).run()
                    self.assertFalse(app.exception)
            load.assert_not_called()

    def test_timelapse_navigation_and_gif_generation(self):
        app = self.app().run()
        app.radio(key='page').set_value('Timelapse da nuvem').run()
        self.assertFalse(app.exception)
        app.radio(key='timelapse_mode').set_value('Dia a dia').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.slider), 1)
        app.slider[0].set_value(1).run()
        self.assertFalse(app.exception)
        app.button(key='timelapse_gif_button').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['timelapse_gif'][1][:6], b'GIF89a')
        app.radio(key='page').set_value('Ritmo e silêncio').run()
        self.assertFalse(app.exception)
        app.radio(key='page').set_value('Evolução por dia').run()
        app.selectbox(key='evolution_metric').select('Palavras por minuto').run()
        self.assertFalse(app.exception)

    def test_timelapse_inside_recording_with_several_frames_and_participant_without_words(self):
        from src.dashboard.analytics import demo_analytics
        data, datasets = demo_analytics()
        data['word_timings'] = [dict(row, start_s=row['start_s'] * 5, end_s=row['end_s'] * 5)
                                for row in data['word_timings']]
        data['recordings'] = {key: 400 for key in data['recordings']}
        data['participants'] = [*data['participants'], dict(speaker='Davi', segments=0, estimated_speech_seconds=0,
                                                            content_word_count=0)]
        with patch('src.dashboard.analytics.demo_analytics', return_value=(data, datasets)):
            app = self.app().run()
            app.radio(key='page').set_value('Timelapse da nuvem').run()
            app.selectbox(key='timelapse_step').select(1).run()
            self.assertFalse(app.exception)
            frames = app.slider[0].max
            self.assertGreater(frames, 1)
            app.slider[0].set_value(2).run()
            self.assertFalse(app.exception)
            app.button(key='timelapse_gif_button').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['timelapse_gif'][1][:6], b'GIF89a')
            app.selectbox(key='timelapse_speaker').select('Davi').run()
            self.assertEqual(app.info[-1].value, 'Nenhuma palavra de conteúdo para este filtro.')
            app.radio(key='timelapse_mode').set_value('Dia a dia').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.info[-1].value, 'Nenhuma palavra de conteúdo para este filtro.')

    def test_context_search_and_comparison_interactions(self):
        app = self.app().run()
        app.radio(key='page').set_value('Busca com contexto').run()
        app.text_input(key='context_query').set_value('mapa').run()
        self.assertFalse(app.exception)
        self.assertGreater(int(app.metric[0].value), 0)
        app.selectbox(key='context_speaker').select('Aurora').run()
        self.assertEqual(set(app.dataframe[0].value['Participante']), {'Aurora'})
        app.text_input(key='context_query').set_value('[').run()
        self.assertEqual(app.metric[0].value, '0')
        app.radio(key='page').set_value('Comparar participantes').run()
        app.selectbox(key='compare_mode').select('Em comum').run()
        self.assertFalse(app.exception)
        self.assertTrue((app.dataframe[1].value.iloc[:, 1:3] > 0).all().all())

    def test_overlap_ranking_filters_and_empty_state(self):
        app = self.app().run()
        app.radio(key='page').set_value('Sobreposições de fala').run()
        self.assertFalse(app.exception)
        total = int(app.metric[0].value)
        self.assertGreater(total, 0)
        app.selectbox(key='overlap_session').select('demo_01').run()
        self.assertEqual(int(app.metric[0].value), total // 2)
        app.selectbox(key='overlap_speaker').select('Aurora').run()
        self.assertFalse(app.exception)
        context = app.dataframe[-1].value
        self.assertTrue(((context['Participante'] == 'Aurora') | (context['Quem já falava'] == 'Aurora')).all())
        app.slider(key='overlap_minimum').set_value(5.0).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '0')

    def test_dictionary_opens_context_for_same_person_and_term(self):
        from src.dashboard.analytics import demo_analytics
        data, _ = demo_analytics()
        def load(session, **kwargs):
            return dict(data, demo=False, quality_status='PASS', session_id=session)
        with patch('src.dashboard.report.load_dashboard', side_effect=load):
            app = self.app(demo=False).run()
            app.radio(key='page').set_value('Dicionário pessoal').run()
            app.selectbox(key='dictionary_speaker').select('Aurora').run()
            app.selectbox(key='dictionary_context_word').select('mapa').run()
            app.button(key='dictionary_open_context').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='page').value, 'Busca com contexto')
            self.assertEqual(app.text_input(key='context_query').value, 'mapa')
            self.assertEqual(set(app.dataframe[0].value['Participante']), {'Aurora'})
