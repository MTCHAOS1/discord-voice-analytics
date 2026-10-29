"""Páginas de análise do Dashboard; não acessam rede nem alteram sessões."""
import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.analytics import (activity_windows, characteristic_words, compare_people,
                                     daily_metrics, frequent_expressions, search_context,
                                     segment_frame, session_summary)

PAGES = ['Evolução por dia', 'Comparar participantes', 'Palavras características',
         'Busca com contexto', 'Expressões frequentes', 'Linha do tempo', 'Resumo e assuntos', 'Sobreposições de fala']
CONTEXT_PAGES = set(PAGES[3:])

LABELS = {'speaker': 'Participante', 'word': 'Palavra', 'count': 'Ocorrências',
          'session_id': 'Sessão', 'start_s': 'Início (s)', 'end_s': 'Fim (s)', 'text': 'Trecho',
          'before': 'Contexto anterior', 'after': 'Contexto seguinte', 'phrase': 'Expressão',
          'day': 'Dia', 'segments': 'Trechos', 'estimated_speech_seconds': 'Fala estimada (s)',
          'content_word_count': 'Palavras de conteúdo', 'share': 'Participação (%)',
          'distinct_words': 'Vocabulário distinto', 'term': 'Assunto (termo)',
          'active_seconds': 'Atividade estimada (s)', 'target': 'Quem já falava',
          'entries': 'Entradas durante outra fala', 'received': 'Entradas recebidas',
          'overlap_seconds': 'Sobreposição estimada (s)', 'speech_entries': 'Entradas de fala',
          'per_100_entries': 'Por 100 entradas de fala', 'kind': 'Tipo',
          'ongoing_start_s': 'Início do trecho anterior (s)', 'ongoing_end_s': 'Fim do trecho anterior (s)',
          'ongoing_text': 'Texto de quem já falava', 'lexical_similarity': 'Semelhança lexical (%)',
          'topic_signal': 'Indício sobre o assunto'}


def table_download(frame, key):
    table = frame.rename(columns=LABELS)
    if key == 'expressoes_frequentes':
        st.caption('Arraste a barra horizontal no rodapé da tabela para ver todas as colunas. '
                   'Dê dois cliques em uma célula para consultar o texto completo.')
        st.dataframe(table, hide_index=True, width='stretch', height=520,
                     column_config={
                         'Participante': st.column_config.TextColumn(width=150, pinned=True),
                         'Expressão': st.column_config.TextColumn(width=520),
                         'Ocorrências': st.column_config.NumberColumn(width=120),
                         'Sessão': st.column_config.TextColumn(width=160),
                         'Início (s)': st.column_config.NumberColumn(width=120),
                         'Trecho': st.column_config.TextColumn(width=750),
                     })
    elif key == 'contexto_sobreposicoes':
        st.caption('Use a rolagem horizontal para comparar os textos completos das duas pessoas.')
        st.dataframe(table, hide_index=True, width='stretch', height=520,
                     column_config={
                         'Participante': st.column_config.TextColumn(width=150, pinned=True),
                         'Quem já falava': st.column_config.TextColumn(width=150),
                         'Texto de quem já falava': st.column_config.TextColumn(width=650),
                         'Trecho': st.column_config.TextColumn(width=650),
                         'Indício sobre o assunto': st.column_config.TextColumn(width=350),
                     })
    else:
        st.dataframe(table, hide_index=True, width='stretch')
    st.download_button('Baixar tabela em CSV', table.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'{key}.csv', mime='text/csv', key=f'export_{key}')


def choose_person(data, key):
    return st.selectbox('Participante', ['Todos os participantes', *sorted(p['speaker'] for p in data['participants'])], key=key)


def render_analysis(page, data, datasets):
    st.subheader(page)
    names = sorted(p['speaker'] for p in data['participants'])
    if page == 'Evolução por dia':
        st.caption('Sessões do mesmo dia são somadas. A participação usa todo o tempo estimado de fala daquele dia.')
        frame = daily_metrics(datasets)
        person = choose_person(data, 'evolution_speaker')
        if person != 'Todos os participantes':
            frame = frame[frame.speaker == person]
        if frame['day'].nunique() < 2:
            st.info('O período selecionado contém apenas um dia. Novos dias permitirão acompanhar a evolução.')
        label = st.selectbox('Indicador', ['Participação (%)', 'Fala estimada (s)', 'Palavras de conteúdo', 'Vocabulário distinto'], key='evolution_metric')
        field = {v: k for k, v in LABELS.items()}[label]
        dated = frame[frame.day != 'Sem data']
        if not dated.empty:
            chart = alt.Chart(dated).mark_line(point=True).encode(
                x=alt.X('day:T', title='Dia da gravação'), y=alt.Y(f'{field}:Q', title=label),
                color=alt.Color('speaker:N', title='Participante'),
                tooltip=[alt.Tooltip('day:T', title='Dia'), 'speaker:N', alt.Tooltip(f'{field}:Q', title=label)]
            ).properties(height=350)
            st.altair_chart(chart, width='stretch')
        if (frame.day == 'Sem data').any():
            st.caption('Sessões sem data aparecem na tabela e não entram no gráfico cronológico.')
        table_download(frame, 'evolucao_diaria')
        return

    if page == 'Comparar participantes':
        if len(names) < 2:
            st.info('São necessários pelo menos dois participantes para comparar.')
            return
        controls = st.columns(2)
        first = controls[0].selectbox('Primeira pessoa', names, key='compare_first')
        second = controls[1].selectbox('Segunda pessoa', [name for name in names if name != first], key='compare_second')
        frame, common, similarity = compare_people(data, first, second)
        metrics = st.columns(3)
        metrics[0].metric('Palavras em comum', common)
        metrics[1].metric('Vocabulário em comum (%)', f'{similarity:.1f}')
        metrics[2].metric('Palavras exclusivas', int(((frame['first'] == 0) | (frame['second'] == 0)).sum()))
        st.caption('Similaridade = palavras distintas em comum ÷ união dos vocabulários. '
                   'As taxas por mil palavras de conteúdo permitem comparar pessoas que falam quantidades diferentes.')
        metrics_rows = pd.DataFrame([row for row in data['participants'] if row['speaker'] in (first, second)])
        table_download(metrics_rows, 'comparacao_metricas')
        mode = st.selectbox('Vocabulário', ['Todas as palavras', 'Em comum', f'Só {first}', f'Só {second}'], key='compare_mode')
        for field in ('first', 'second'):
            total = frame[field].sum()
            frame[f'{field}_rate'] = frame[field] / total * 1000 if total else 0
        if mode == 'Em comum':
            frame = frame[(frame['first'] > 0) & (frame['second'] > 0)]
        elif mode == f'Só {first}':
            frame = frame[(frame['first'] > 0) & (frame['second'] == 0)]
        elif mode == f'Só {second}':
            frame = frame[(frame['second'] > 0) & (frame['first'] == 0)]
        frame['total'] = frame['first'] + frame['second']
        frame = frame.sort_values(['total', 'word'], ascending=[False, True]).drop(columns='total')
        table_download(frame.rename(columns={'first': f'{first} · ocorrências', 'second': f'{second} · ocorrências',
                                             'first_rate': f'{first} · por mil', 'second_rate': f'{second} · por mil'}),
                       'comparacao_vocabulario')
        return

    if page == 'Palavras características':
        person = st.selectbox('Participante', names, key='characteristic_speaker')
        minimum = st.number_input('Mínimo de ocorrências', min_value=1, value=2, step=1, key='characteristic_min')
        st.caption('Compara a frequência por mil palavras de conteúdo da pessoa com a do restante do grupo. '
                   'A diferença positiva define a ordem; exclusividade vale apenas para o período selecionado.')
        frame = characteristic_words(data, person, int(minimum))
        if frame.empty:
            st.info('Sem palavras características com esses filtros. É necessário vocabulário de outras pessoas para comparar.')
        else:
            table_download(frame.rename(columns={'rest_count': 'Ocorrências no grupo', 'own_rate': 'Pessoa · por mil',
                                                  'rest_rate': 'Grupo · por mil', 'difference': 'Diferença por mil'}),
                           'palavras_caracteristicas')
        return

    frame = segment_frame(data)
    if frame.empty:
        st.info('Nenhum trecho disponível no período selecionado.')
        return
    if page == 'Sobreposições de fala':
        from src.dashboard.overlap import overlap_events, overlap_ranking
        from src.dashboard.charts import labeled_bars

        st.caption('Quem começa a falar enquanto outra voz ainda tem um trecho ativo. '
                   'São indícios pelos timestamps do Whisper: não comprovam interrupção, intenção ou mudança de assunto.')
        controls = st.columns(2)
        minimum = controls[0].slider('Sobreposição mínima (segundos)', 0.1, 5.0, 1.0, 0.1, key='overlap_minimum')
        entry = controls[1].slider('Duração mínima da entrada (segundos)', 0.0, 10.0, 2.0, 0.5, key='overlap_entry')
        session = st.selectbox('Gravação', ['Todas as gravações', *sorted(frame.session_id.unique())], key='overlap_session')
        if session != 'Todas as gravações':
            frame = frame[frame.session_id == session]
        events = overlap_events(frame, minimum, entry)
        ranking = overlap_ranking(frame, events)
        metrics = st.columns(3)
        metrics[0].metric('Entradas durante outra fala', int(ranking.entries.sum()))
        metrics[1].metric('Pares com início simultâneo', int((events.kind == 'Início simultâneo').sum()))
        metrics[2].metric('Participantes com entradas', int((ranking.entries > 0).sum()))
        st.caption('Trechos contínuos da mesma voz são unidos. Entrar sobre várias pessoas conta uma vez '
                   'no ranking; os detalhes mostram cada par. Inícios simultâneos ficam fora do ranking. '
                   'A taxa por 100 entradas ajuda a comparar pessoas com quantidades diferentes de fala.')
        st.altair_chart(labeled_bars(ranking, 'speaker', 'entries', 'Entradas durante outra fala', '#f9a8d4'), width='stretch')
        table_download(ranking, 'ranking_sobreposicoes')
        st.markdown('**Quem entrou sobre quem**')
        directed = events[events.kind == 'Entrada durante outra fala']
        pairs = directed.groupby(['speaker', 'target']).size().reset_index(name='entries')
        if pairs.empty:
            st.info('Nenhuma entrada durante outra fala com os limites selecionados.')
        else:
            table_download(pairs.sort_values('entries', ascending=False), 'pares_sobreposicoes')
        st.markdown('**Revisar contexto e possíveis mudanças de assunto**')
        st.caption('Semelhança lexical compara palavras de conteúdo dos dois trechos. '
                   'Menos de 15% gera um aviso para revisão, somente se ambos têm ao menos três palavras distintas. '
                   'Poucas palavras em comum também podem ocorrer dentro do mesmo assunto. '
                   'Compare os textos e confira a gravação antes de concluir que alguém interrompeu um assunto.')
        person = choose_person(data, 'overlap_speaker')
        if person != 'Todos os participantes':
            events = events[(events.speaker == person) | (events.target == person)]
        only_topic = st.checkbox('Mostrar apenas baixa semelhança lexical', key='overlap_topic')
        if only_topic:
            events = events[events.topic_signal == 'Baixa semelhança lexical — revisar contexto']
        if events.empty:
            st.info('Nenhuma ocorrência para revisar com esses filtros.')
        else:
            table_download(events, 'contexto_sobreposicoes')
        return
    if page == 'Busca com contexto':
        st.caption('Busca literal na transcrição. Os horários são relativos ao início de cada gravação. '
                   'O contexto mostra o trecho anterior e o seguinte da mesma sessão.')
        person = choose_person(data, 'context_speaker')
        session = st.selectbox('Gravação', ['Todas as gravações', *sorted(frame.session_id.unique())], key='context_session')
        query = st.text_input('Palavra ou expressão', key='context_query')
        whole = st.checkbox('Somente palavra ou expressão inteira', value=True, key='context_whole')
        if not query.strip():
            st.info('Digite uma palavra ou expressão para localizar os trechos.')
            return
        results = search_context(frame, query, None if person == 'Todos os participantes' else person, whole)
        if session != 'Todas as gravações':
            results = results[results.session_id == session]
        st.metric('Trechos encontrados', len(results))
        if results.empty:
            st.info('Nenhum trecho encontrado para esses filtros.')
        else:
            table_download(results, 'busca_contexto')
        return

    if page == 'Expressões frequentes':
        st.caption('Combinações de palavras consecutivas dentro de uma mesma frase e trecho. '
                   'A contagem inclui palavras comuns; cada linha traz uma ocorrência de referência.')
        person = choose_person(data, 'expressions_speaker')
        if person != 'Todos os participantes':
            frame = frame[frame.speaker == person]
        size = st.selectbox('Palavras por expressão', [2, 3, 4], key='expressions_size')
        minimum = st.number_input('Mínimo de ocorrências', min_value=1, value=2, step=1, key='expressions_min')
        result = frequent_expressions(frame, size, int(minimum))
        if result.empty:
            st.info('Nenhuma expressão repetida com os filtros selecionados.')
        else:
            table_download(result, 'expressoes_frequentes')
        return

    sessions = sorted(frame.session_id.unique())
    session = st.selectbox('Gravação', sessions, key='timeline_session' if page == 'Linha do tempo' else 'summary_session')
    frame = frame[frame.session_id == session]
    if page == 'Linha do tempo':
        st.caption('Intervalos estimados pelo Whisper, em segundos desde o início da gravação. '
                   'Sobreposições podem aparecer; elas não comprovam interrupções ou voz ativa.')
        person = choose_person(data, 'timeline_speaker')
        if person != 'Todos os participantes':
            frame = frame[frame.speaker == person]
        if frame.empty:
            st.info('Essa pessoa não tem trechos na gravação selecionada.')
            return
        maximum = max(1, int(frame.end_s.max()) + 1)
        extent = st.slider('Intervalo da gravação (segundos)', 0, maximum, (0, maximum), key=f'timeline_range_{session}_{person}')
        frame = frame[(frame.end_s > extent[0]) & (frame.start_s < extent[1])].copy()
        if frame.empty:
            st.info('Nenhum trecho nesse intervalo.')
            return
        frame['start_s'] = frame.start_s.clip(lower=extent[0])
        frame['end_s'] = frame.end_s.clip(upper=extent[1])
        chart = alt.Chart(frame).mark_bar(size=20).encode(
            x=alt.X('start_s:Q', title='Segundos desde o início'), x2='end_s:Q',
            y=alt.Y('speaker:N', title='Participante'), color=alt.Color('speaker:N', legend=None),
            tooltip=['speaker:N', 'start_s:Q', 'end_s:Q', 'text:N']).properties(height=max(230, frame.speaker.nunique() * 55))
        st.altair_chart(chart, width='stretch')
        st.markdown('**Períodos com maior atividade estimada**')
        st.caption('Janelas de cinco minutos, ordenadas pelo tempo coberto por trechos. Sobreposições contam uma vez.')
        activity = activity_windows(frame).sort_values('active_seconds', ascending=False)
        table_download(activity, 'atividade_temporal')
        with st.expander('Trechos da linha do tempo'):
            table_download(frame, 'linha_do_tempo')
        return

    st.caption('Resumo extrativo local: reúne trechos originais relevantes e termos recorrentes. '
               'Os assuntos são pistas lexicais, não uma interpretação semântica ou conclusões verificadas.')
    maximum = st.selectbox('Trechos no resumo', [3, 5, 10], index=1, key='summary_limit')
    highlights, topics = session_summary(frame, maximum)
    if highlights.empty:
        st.info('Não há palavras de conteúdo suficientes para resumir essa gravação.')
        return
    st.markdown('**Trechos em destaque · ordem cronológica**')
    for row in highlights.itertuples():
        with st.container(border=True):
            st.caption(f'{row.session_id} · {row.speaker} · {row.start_s:.1f}s–{row.end_s:.1f}s')
            st.text(row.text)
    st.markdown('**Assuntos sugeridos pelos termos mais recorrentes**')
    table_download(topics, 'assuntos_com_referencias')
    with st.expander('Exportar trechos do resumo'):
        table_download(highlights, 'resumo_extrativo')
