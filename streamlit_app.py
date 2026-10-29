"""Dashboard local: execute com streamlit run streamlit_app.py."""
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

from src.dashboard.report import demo_dashboard, load_dashboard, combine_dashboards
from src.ingestion.session_catalog import session_catalog
from src.dashboard.charts import labeled_bars, participation_donut
from src.dashboard.analysis_pages import PAGES, CONTEXT_PAGES, render_analysis

PROJECT_ROOT = Path(__file__).resolve().parent


def open_context(word, person):
    st.session_state['context_query'] = word
    st.session_state['context_speaker'] = person or 'Todos os participantes'
    st.session_state['page'] = 'Busca com contexto'


def format_duration(seconds):
    seconds = int(float(seconds) + 0.5)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f'{hours}h {minutes}min {rest}s' if hours else f'{minutes}min {rest}s'


def main():
    st.set_page_config(page_title='Discord Voice Analytics', page_icon='🎙️', layout='wide')
    st.markdown('''<style>
    .stMainBlockContainer {max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -.04em; font-weight: 750 !important;}
    h2, h3 {letter-spacing: -.02em;}
    [data-testid="stCaptionContainer"] {color: #b7c5d8; font-size: .95rem;}
    [data-testid="stMetric"] {background: #172333; border: 1px solid #34465e;
        border-radius: 14px; padding: 18px; min-height: 126px;}
    [data-testid="stMetricLabel"] {color: #cbd8e8; font-size: 1rem;}
    [data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2.1vw, 2.3rem); color: #ffffff;}
    [data-testid="stSidebar"] {border-right: 1px solid #2b3d53;}
    [data-testid="stDataFrame"] {border: 1px solid #34465e; border-radius: 10px;}
    @media (max-width: 640px) {
        .stMainBlockContainer {padding: 1.2rem 1rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
    }
    </style>''', unsafe_allow_html=True)
    st.caption('DISCORD VOICE ANALYTICS')
    st.title('Cada voz conta.')
    st.caption('Discord Voice Analytics · participação e vocabulário da conversa')

    with st.sidebar:
        st.header('Seu painel')
        page = st.radio('Página', ['Visão geral', 'Nuvem de palavras', 'Dicionário pessoal', *PAGES], key='page')
        source = st.radio('Origem', ['Sessão local', 'Demonstração'], key='source')
        if source == 'Sessão local':
            allow_warnings = st.checkbox('Revisei os alertas e permito relatório WARN', key='allow_warnings')
        st.caption('A demonstração usa apenas dados fictícios. Sessões locais exigem análise e validação prévias.')

    if source == 'Demonstração':
        data = demo_dashboard()
        if page in PAGES:
            from src.dashboard.analytics import demo_analytics
            data, datasets = demo_analytics()
        st.info('Demonstração com dados fictícios. Nenhuma gravação foi utilizada.')
        if page in PAGES:
            st.caption('Estas análises usam uma amostra sintética com frases e dois dias de exemplo.')
    else:
        try:
            catalog = session_catalog(PROJECT_ROOT)
            if not catalog:
                st.info('Nenhuma sessão processada encontrada. Transcreva uma sessão ou selecione Demonstração.')
                st.stop()
            st.subheader('Período da análise')
            days = sorted({entry['day'] for entry in catalog if entry['day']}, reverse=True)
            options = ['Todos os dias', *days]
            if any(entry['day'] is None for entry in catalog):
                options.append('Sem data identificada')
            day = st.selectbox('Dia da gravação', options, key='day_filter',
                               format_func=lambda value: date.fromisoformat(value).strftime('%d/%m/%Y')
                               if value in days else value)
            selected_sessions = [entry['session_id'] for entry in catalog
                                 if day == 'Todos os dias' or entry['day'] == day or
                                 (day == 'Sem data identificada' and entry['day'] is None)]
            if any(entry['day'] is None for entry in catalog):
                st.caption('Sessões sem data estão incluídas em Todos os dias e disponíveis no filtro Sem data identificada.')
            datasets = []
            for session_id in selected_sessions:
                try:
                    extra = {'include_segments': True} if page in CONTEXT_PAGES else {}
                    dataset = dict(load_dashboard(session_id, base_dir=PROJECT_ROOT,
                                                  allow_warnings=allow_warnings, **extra))
                    dataset['day'] = next(entry['day'] for entry in catalog if entry['session_id'] == session_id)
                    datasets.append(dataset)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{session_id}: {exc}') from exc
            data = combine_dashboards(datasets)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f'Não foi possível carregar a sessão: {exc}')
            st.caption('Execute analyze e validate para a sessão antes de abri-la aqui.')
            st.stop()
        if data['quality_status'] == 'WARN':
            st.warning('Sessão com alertas, exibida após confirmação de revisão.')
        else:
            st.success('Dados passaram pelas verificações locais de consistência.')

    st.subheader('Demonstração' if data['demo'] else f'Análise · {day if day not in days else date.fromisoformat(day).strftime("%d/%m/%Y")}')
    if data.get('sessions'):
        st.caption(f'{len(data["sessions"])} gravação(ões) reunida(s) · Participantes agrupados pelo apelido')
    if not data['demo']:
        with st.expander('Arquivos utilizados'):
            st.caption('Sessões incluídas: ' + ', '.join(data['sessions']))
            st.caption('Indicadores calculados dos resultados locais desta sessão. '
                       'Palavras são recontadas a partir dos segmentos para não limitar o ranking ao CSV top 30.')
            st.code('\n'.join(str(PROJECT_ROOT / path / session_id / filename)
                              for session_id in selected_sessions for path, filename in [
                ('data/processed', 'segments.parquet'),
                ('data/curated', 'speaker_metrics.csv'),
                ('data/curated', 'top_words.csv'),
                ('data/quality', 'validation_report.json'),
            ]), language=None)
    if data.get('validated_at'):
        st.caption(f'Validação: {data["validated_at"]}')
    participants = pd.DataFrame(data['participants'])
    total = float(participants['estimated_speech_seconds'].sum())
    participants['share'] = participants['estimated_speech_seconds'] / total * 100 if total else 0.0
    participants = participants.sort_values(['estimated_speech_seconds', 'speaker'], ascending=[False, True])

    if page in PAGES:
        render_analysis(page, data, datasets)
        return

    if page == 'Dicionário pessoal':
        st.subheader('Dicionário pessoal')
        st.caption('Explore todas as palavras transcritas por participante no período selecionado, sem limite de Top 30. '
                   'As contagens refletem a transcrição e podem conter erros do reconhecimento de voz.')
        person = st.selectbox('Participante', ['Todos os participantes', *sorted(participants['speaker'].tolist())],
                              index=1, key='dictionary_speaker')
        if person == 'Todos os participantes':
            person = None
        st.markdown(f'**Dicionário de {person}**' if person else '**Dicionário de todos os participantes**')
        controls = st.columns([2, 1])
        search = controls[0].text_input('Buscar palavra', placeholder='Digite parte de uma palavra', key='dictionary_search')
        order = controls[1].selectbox('Ordenar por', ['Mais faladas', 'A–Z'], key='dictionary_order')
        include_common = st.checkbox('Incluir palavras comuns e curtas', value=True, key='dictionary_common')
        rows = data.get('vocabulary', data['words']) if include_common else data['words']
        vocabulary = pd.DataFrame(rows, columns=['speaker', 'word', 'count'])
        if person is not None:
            vocabulary = vocabulary[vocabulary['speaker'] == person]
        if search.strip():
            import unicodedata
            query = unicodedata.normalize('NFC', search.strip()).lower()
            vocabulary = vocabulary[vocabulary['word'].str.contains(query, regex=False, na=False)]
        vocabulary = vocabulary.groupby(['speaker', 'word'], as_index=False)['count'].sum()
        vocabulary = vocabulary.sort_values(['count', 'word', 'speaker'] if order == 'Mais faladas'
                                            else ['word', 'speaker'],
                                            ascending=[False, True, True] if order == 'Mais faladas' else True)
        metrics = st.columns(3)
        metrics[0].metric('Palavras distintas', vocabulary['word'].nunique())
        metrics[1].metric('Ocorrências', int(vocabulary['count'].sum()))
        metrics[2].metric('Participantes', vocabulary['speaker'].nunique())
        st.caption('Indicadores e download acompanham os filtros. Letras são preservadas, incluindo acentos; '
                   'maiúsculas e minúsculas são unificadas. Números e pontuação não entram no dicionário.')
        if data['demo']:
            st.caption('A demonstração contém apenas um vocabulário sintético reduzido.')
        if vocabulary.empty:
            st.info('Nenhuma palavra encontrada para os filtros selecionados.')
        else:
            table = vocabulary.rename(columns={'speaker': 'Participante', 'word': 'Palavra', 'count': 'Ocorrências'})
            st.dataframe(table, hide_index=True, width='stretch', height=520)
            period = 'demo' if data['demo'] else (day if day in days else 'todos_os_dias' if day == 'Todos os dias' else 'sem_data')
            st.download_button('Baixar dicionário em CSV', table.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'dicionario_{period}.csv', mime='text/csv')
            if not data['demo']:
                term = st.selectbox('Ver uma palavra no contexto da conversa',
                                    sorted(vocabulary['word'].unique()), key='dictionary_context_word')
                st.button('Buscar essa palavra nos trechos', key='dictionary_open_context',
                          on_click=open_context, args=(term, person))
        return

    if page == 'Nuvem de palavras':
        from src.dashboard.word_cloud import render_word_cloud, participant_colors

        st.subheader('Nuvem de palavras')
        st.caption('Quanto mais frequente a palavra, maior ela aparece. '
                   'A contagem exclui palavras comuns, como artigos e preposições.')
        st.caption('Cada cor identifica um participante na legenda. Uma palavra dita por várias '
                   'pessoas aparece uma vez por pessoa, com tamanho proporcional à contagem individual.')
        colors = participant_colors(participants['speaker'].tolist())
        cloud_controls = st.columns([2, 1])
        speaker = cloud_controls[0].selectbox('Participante', [None, *participants['speaker'].tolist()],
                               format_func=lambda value: 'Todos os participantes' if value is None else value,
                               key='cloud_speaker')
        limit = cloud_controls[1].select_slider('Máximo de palavras', options=[20, 50, 100], value=50, key='cloud_limit')
        frequency = pd.DataFrame(data['words'], columns=['speaker', 'word', 'count'])
        if speaker is not None:
            frequency = frequency[frequency['speaker'] == speaker]
        ranking = (frequency.groupby(['speaker', 'word'], as_index=False)['count'].sum()
                   .sort_values(['count', 'speaker', 'word'], ascending=[False, True, True]).head(limit))
        if ranking.empty:
            st.info('Nenhuma palavra de conteúdo para este filtro.')
        else:
            legend = {name: color for name, color in colors.items() if name in set(ranking['speaker'])}
            from html import escape
            st.markdown('<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">' + ''.join(
                f'<span style="background:#172333;border:1px solid {color};border-radius:8px;padding:7px 12px;'
                f'color:#f0f4fa"><span style="color:{color}" aria-hidden="true">●</span> {escape(name)}</span>'
                for name, color in legend.items()) + '</div>', unsafe_allow_html=True)
            png = render_word_cloud({(row.speaker, row.word): row.count
                                     for row in ranking.itertuples(index=False)}, legend)
            st.image(png, caption='Palavras mais faladas · ' + (speaker or 'Todos os participantes'),
                     width='stretch')
            period = 'demo' if data['demo'] else (day if day in days else 'todos_os_dias' if day == 'Todos os dias' else 'sem_data')
            st.download_button('Baixar nuvem em PNG', png, file_name=f'nuvem_{period}.png',
                               mime='image/png')
            with st.expander('Ver contagens'):
                st.dataframe(ranking.rename(columns={'speaker': 'Participante', 'word': 'Palavra', 'count': 'Ocorrências'}),
                             hide_index=True, width='stretch')
        return

    controls = st.columns(2)
    selected = controls[0].selectbox('Participante', [None, *participants['speaker'].tolist()],
                                    format_func=lambda value: 'Todos os participantes' if value is None else value,
                                    key='speaker')
    limit = controls[1].selectbox('Palavras no ranking', [5, 10, 20], index=1,
                                 format_func=lambda value: f'Top {value}', key='limit')
    visible = participants if selected is None else participants[participants['speaker'] == selected]
    metrics = st.columns(4)
    metrics[0].metric('Fala estimada', format_duration(visible['estimated_speech_seconds'].sum()))
    metrics[1].metric('Participantes', len(visible))
    metrics[2].metric('Segmentos', int(visible['segments'].sum()))
    metrics[3].metric('Palavras de conteúdo', int(visible['content_word_count'].sum()))
    st.caption('A participação usa a soma do tempo de fala de toda a sessão, mesmo ao filtrar uma pessoa.')

    frequency = pd.DataFrame(data['words'], columns=['speaker', 'word', 'count'])
    if selected is not None:
        frequency = frequency[frequency['speaker'] == selected]
    ranking = (frequency.groupby('word', as_index=False)['count'].sum()
               .sort_values(['count', 'word'], ascending=[False, True]).head(limit))
    left, right = st.columns(2)
    with left:
        st.subheader('Participação na conversa')
        st.caption('Percentual do tempo estimado de fala no período.')
        chart_type = st.radio('Formato do gráfico', ['Rosca', 'Barras'], horizontal=True, key='participation_chart')
        if total <= 0:
            st.info('Sem tempo estimado de fala para calcular a participação.')
        elif chart_type == 'Rosca':
            st.altair_chart(participation_donut(participants, selected), width='stretch')
            if selected is not None:
                st.caption(f'{selected} em destaque. As demais pessoas permanecem no total do período.')
        else:
            st.altair_chart(labeled_bars(visible, 'speaker', 'share', 'Participação (%)', '#69c9a0',
                                        percentage=True), width='stretch')
    with right:
        st.subheader('Palavras em destaque')
        st.caption('Palavras mais frequentes, com o número de ocorrências.')
        if ranking.empty:
            st.info('Nenhuma palavra de conteúdo para este filtro.')
        else:
            st.altair_chart(labeled_bars(ranking, 'word', 'count', 'Ocorrências', '#ad9cf5'), width='stretch')

    st.subheader('Participação por pessoa')
    details = visible[['speaker', 'estimated_speech_seconds', 'share', 'segments', 'content_word_count']].copy()
    details['estimated_speech_seconds'] = details['estimated_speech_seconds'].map(format_duration)
    details.columns = ['Participante', 'Fala estimada', 'Participação (%)', 'Segmentos', 'Palavras de conteúdo']
    st.dataframe(details, hide_index=True, width='stretch',
                 column_config={'Participação (%)': st.column_config.ProgressColumn(
                     format='%.1f%%', min_value=0, max_value=100)})
    st.caption('O tempo estimado pelos segmentos Whisper pode incluir silêncios e sobreposições; '
               'a soma não representa a duração da sessão nem mede voz ativa exata.')
    st.caption('Nomes e palavras da sessão podem ser privados. Compartilhe somente com autorização.')


if __name__ == '__main__':
    main()
