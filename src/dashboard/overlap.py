"""Entradas durante intervalos de outra voz: indícios temporais, não intenção."""
import pandas as pd

from src.transformation.analyze import words

EVENT_COLUMNS = ['session_id', 'speaker', 'target', 'start_s', 'end_s', 'overlap_seconds',
                 'kind', 'ongoing_start_s', 'ongoing_end_s', 'ongoing_text', 'text',
                 'lexical_similarity', 'topic_signal']


def voice_intervals(frame):
    """Une trechos sobrepostos/contíguos da mesma voz, sem cruzar sessões."""
    intervals = []
    for (session, speaker), group in frame.groupby(['session_id', 'speaker']):
        current = None
        for row in group.sort_values(['start_s', 'end_s']).itertuples():
            part = dict(start=row.start_s, end=row.end_s, text=row.text)
            if current is not None and row.start_s <= current['end']:
                current['end'] = max(current['end'], row.end_s)
                current['parts'].append(part)
            else:
                current = dict(session_id=session, speaker=speaker, start=row.start_s,
                               end=row.end_s, parts=[part])
                intervals.append(current)
    return intervals


def overlap_events(frame, minimum_overlap=1.0, minimum_entry=2.0):
    events = []
    active = []
    session = None
    for entrant in sorted(voice_intervals(frame), key=lambda row: (row['session_id'], row['start'], row['speaker'])):
        if entrant['session_id'] != session:
            active = []
            session = entrant['session_id']
        active = [row for row in active if row['end'] > entrant['start']]
        if entrant['end'] - entrant['start'] >= minimum_entry:
            for ongoing in active:
                if ongoing['speaker'] == entrant['speaker']:
                    continue
                end = min(entrant['end'], ongoing['end'])
                overlap = end - entrant['start']
                if overlap < minimum_overlap:
                    continue
                simultaneous = entrant['start'] == ongoing['start']
                # Em início simultâneo, só inclui o par se ambas as entradas passam o filtro.
                if simultaneous and ongoing['end'] - ongoing['start'] < minimum_entry:
                    continue
                part = max((p for p in ongoing['parts'] if p['start'] <= entrant['start'] < p['end']),
                           key=lambda p: p['start'])
                text = entrant['parts'][0]['text']
                previous_terms, new_terms = set(words(part['text'])), set(words(text))
                similarity = None
                signal = 'Vocabulário insuficiente para comparar'
                if len(previous_terms) >= 3 and len(new_terms) >= 3:
                    similarity = len(previous_terms & new_terms) / len(previous_terms | new_terms) * 100
                    signal = 'Baixa semelhança lexical — revisar contexto' if similarity < 15 else 'Vocabulário relacionado'
                events.append(dict(session_id=session, speaker=entrant['speaker'], target=ongoing['speaker'],
                                   start_s=entrant['start'], end_s=end, overlap_seconds=overlap,
                                   kind='Início simultâneo' if simultaneous else 'Entrada durante outra fala',
                                   ongoing_start_s=part['start'], ongoing_end_s=part['end'], ongoing_text=part['text'],
                                   text=text, lexical_similarity=similarity, topic_signal=signal))
        active.append(entrant)
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def _union_seconds(group):
    total = 0
    start = end = None
    for row in group.sort_values(['start_s', 'end_s']).itertuples():
        if start is None:
            start, end = row.start_s, row.end_s
        elif row.start_s <= end:
            end = max(end, row.end_s)
        else:
            total += end - start
            start, end = row.start_s, row.end_s
    return total + (end - start if start is not None else 0)


def overlap_ranking(frame, events):
    directed = events[events.kind == 'Entrada durante outra fala']
    intervals = voice_intervals(frame)
    rows = []
    for speaker in sorted(frame.speaker.unique()):
        own = directed[directed.speaker == speaker]
        received = directed[directed.target == speaker]
        # Entrar sobre duas vozes ao mesmo tempo conta como uma entrada no ranking.
        entries = len(own.drop_duplicates(['session_id', 'start_s']))
        received_entries = len(received.drop_duplicates(['session_id', 'speaker', 'start_s']))
        episodes = sum(row['speaker'] == speaker for row in intervals)
        seconds = sum(_union_seconds(group) for _, group in own.groupby('session_id'))
        rows.append(dict(speaker=speaker, entries=entries, received=received_entries,
                         overlap_seconds=seconds, speech_entries=episodes,
                         per_100_entries=entries / episodes * 100 if episodes else 0))
    return pd.DataFrame(rows, columns=['speaker', 'entries', 'received', 'overlap_seconds',
                                        'speech_entries', 'per_100_entries']).sort_values(
        ['entries', 'overlap_seconds', 'speaker'], ascending=[False, False, True])
