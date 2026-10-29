"""Gráficos horizontais com valores legíveis e altura proporcional às linhas."""
import altair as alt

from src.dashboard.word_cloud import participant_colors


def participation_donut(frame, selected=None):
    """Mantém todas as pessoas no denominador, mesmo ao destacar uma delas."""
    data = frame.copy()
    total = float(data['estimated_speech_seconds'].sum())
    data['percentage'] = data['estimated_speech_seconds'] / total * 100 if total else 0.0
    data['legend_label'] = data.apply(lambda row: f'{row.speaker} · {row.percentage:.1f}%', axis=1)
    data['percentage_label'] = data['percentage'].map(lambda value: f'{value:.1f}%' if value >= 4 else '')
    data['emphasis'] = data['speaker'].map(lambda name: 1.0 if selected is None or name == selected else 0.25)
    colors = participant_colors(data['speaker'].tolist())
    base = alt.Chart(data).encode(
        theta=alt.Theta('estimated_speech_seconds:Q', stack=True),
        order=alt.Order('speaker:N', sort='ascending'),
        tooltip=[alt.Tooltip('speaker:N', title='Participante'),
                 alt.Tooltip('percentage:Q', title='Participação (%)', format='.1f'),
                 alt.Tooltip('estimated_speech_seconds:Q', title='Fala estimada (s)', format='.1f')])
    arcs = base.mark_arc(innerRadius=65, outerRadius=112, stroke='#0c1017', strokeWidth=2).encode(
        color=alt.Color('legend_label:N', title=None,
                        scale=alt.Scale(domain=data['legend_label'].tolist(),
                                        range=[colors[name] for name in data['speaker']]),
                        legend=alt.Legend(orient='bottom', columns=2, labelFontSize=13, labelLimit=280)),
        opacity=alt.Opacity('emphasis:Q', scale=None, legend=None))
    labels = base.mark_text(radius=132, fontSize=13, color='#f0f4fa').encode(text='percentage_label:N')
    return (arcs + labels).properties(height=310).configure_view(strokeWidth=0)


def labeled_bars(frame, category, value, title, color, percentage=False):
    data = frame.copy()
    data['label'] = data[value].map(lambda number: f'{number:.1f}%' if percentage else f'{number:,.0f}'.replace(',', '.'))
    maximum = float(data[value].max()) if not data.empty else 1
    base = alt.Chart(data).encode(
        y=alt.Y(f'{category}:N', sort=data[category].tolist(), title=None,
                axis=alt.Axis(labelFontSize=14, labelLimit=180, labelPadding=10)),
        x=alt.X(f'{value}:Q', title=title, scale=alt.Scale(domain=[0, max(maximum * 1.23, 1)]),
                axis=alt.Axis(labelFontSize=12, titleFontSize=12, gridOpacity=0.12)),
        tooltip=[alt.Tooltip(f'{category}:N', title='Nome'),
                 alt.Tooltip('label:N', title=title)])
    bars = base.mark_bar(color=color, cornerRadiusEnd=5, size=22)
    labels = base.mark_text(align='left', dx=8, fontSize=14, color='#f0f4fa').encode(text='label:N')
    return (bars + labels).properties(height=max(230, len(data) * 38)).configure_view(strokeWidth=0)
