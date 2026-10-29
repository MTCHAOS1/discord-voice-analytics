"""Análises locais rastreáveis, calculadas somente sobre dados já validados."""
from collections import Counter
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from src.transformation.analyze import dictionary_words, words

SEGMENT_COLUMNS = ['session_id', 'speaker', 'start_s', 'end_s', 'text']


def segment_frame(data):
    return pd.DataFrame(data.get('segments', []), columns=SEGMENT_COLUMNS).sort_values(
        ['session_id', 'start_s', 'speaker', 'end_s']).reset_index(drop=True)


def merge_intervals(intervals, bridge=0.0):
    """Une intervalos sobrepostos e os separados por lacuna de até `bridge` segundos."""
    merged = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals):
        if merged and start - merged[-1][1] <= bridge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def speech_timing(frame, bridge=0.5):
    """Fala medida por pessoa: união dos intervalos, sem contar sobreposição duas vezes."""
    rows = []
    for speaker, group in frame.groupby('speaker'):
        merged = merge_intervals(zip(group.start_s, group.end_s), bridge)
        rows.append(dict(speaker=speaker, measured_speech_seconds=sum(end - start for start, end in merged)))
    return pd.DataFrame(rows, columns=['speaker', 'measured_speech_seconds'])


def words_per_minute(word_count, seconds):
    return None if seconds <= 0 else word_count / seconds * 60


GAP_COLUMNS = ['session_id', 'start_s', 'end_s', 'gap_seconds',
               'before_speaker', 'before_text', 'after_speaker', 'after_text']


def _neighbors(source, moments, before):
    """Trecho vizinho de cada instante, com busca binária em vez de varrer todos os trechos.

    Antes: o trecho que contém o instante (o de maior fim) ou, senão, o que termina mais perto antes.
    Depois: o trecho que contém o instante (start <= m < end) ou, senão, o que começa mais perto depois.
    """
    if source.empty or len(moments) == 0:
        return [('', '')] * len(moments)
    ordered = source.sort_values(['start_s', 'end_s'], kind='stable')
    starts = ordered.start_s.to_numpy(dtype=float)
    ends = ordered.end_s.to_numpy(dtype=float)
    speakers = ordered.speaker.astype(str).to_numpy()
    texts = ordered['text'].astype(str).to_numpy() if 'text' in ordered.columns else None
    # Maior fim entre os trechos já iniciados e a posição de quem o atinge.
    running = np.maximum.accumulate(ends)
    holder = np.maximum.accumulate(np.where(ends == running, np.arange(len(ends)), 0))
    moments = np.asarray(moments, dtype=float)
    started = np.searchsorted(starts, moments, side='right')
    rows = []
    for moment, count in zip(moments, started):
        row = None
        if before:
            if count:
                row = holder[count - 1]
        elif count and running[count - 1] > moment:
            row = holder[count - 1]
        else:
            later = np.searchsorted(starts, moment, side='left')
            row = later if later < len(starts) else None
        rows.append(('', '') if row is None else (speakers[row], texts[row] if texts is not None else ''))
    return rows


def silence_stats(frame, recording_seconds=None, min_gap=3.0, bridge=0.5, segments=None):
    """Silêncio de uma sessão: tempo fora da união das falas e as pausas mais longas."""
    if frame.empty:
        return dict(window_start=0.0, window_end=0.0, duration_seconds=0.0, spoken_seconds=0.0,
                    silence_seconds=0.0, silence_share=0.0, basis='gravação inteira',
                    gaps=pd.DataFrame(columns=GAP_COLUMNS))
    whole = recording_seconds is not None and math.isfinite(recording_seconds) and recording_seconds > 0
    if whole:
        start, end = 0.0, float(recording_seconds)
    else:
        start, end = float(frame.start_s.min()), float(frame.end_s.max())
    clipped = [(max(a, start), min(b, end)) for a, b in merge_intervals(zip(frame.start_s, frame.end_s), bridge)
               if min(b, end) > max(a, start)]
    spoken = sum(b - a for a, b in clipped)
    duration = end - start
    edges = [(start, start), *clipped, (end, end)] if whole else clipped
    session = str(frame.session_id.iloc[0]) if 'session_id' in frame.columns else ''
    source = segments if segments is not None and not segments.empty else frame
    previous = np.array([prev for (_, prev), _ in zip(edges, edges[1:])])
    following = np.array([nxt for _, (nxt, _) in zip(edges, edges[1:])])
    found = np.flatnonzero(following - previous >= min_gap)
    before = _neighbors(source, previous[found], True)
    after = _neighbors(source, following[found], False)
    rows = [dict(session_id=session, start_s=previous[i], end_s=following[i], gap_seconds=following[i] - previous[i],
                 before_speaker=b[0], before_text=b[1], after_speaker=a[0], after_text=a[1])
            for i, b, a in zip(found, before, after)]
    gaps = pd.DataFrame(rows, columns=GAP_COLUMNS).sort_values('gap_seconds', ascending=False).reset_index(drop=True)
    share = (duration - spoken) / duration * 100 if duration > 0 else 0.0
    return dict(window_start=start, window_end=end, duration_seconds=duration, spoken_seconds=spoken,
                silence_seconds=duration - spoken, silence_share=share,
                basis='gravação inteira' if whole else 'entre a primeira e a última fala', gaps=gaps)


def timelapse_counts(frame, step_seconds=300, speaker=None, limit=50):
    """Contagens acumuladas por quadro, sempre sobre o mesmo conjunto de palavras."""
    if speaker is not None:
        frame = frame[frame.speaker == speaker]
    if frame.empty:
        return []
    ordered = frame.sort_values(['start_s', 'speaker']).reset_index(drop=True)
    tokens = [[(row.speaker, word) for word in words(str(row.text))] for row in ordered.itertuples()]
    totals = Counter(key for row in tokens for key in row)
    chosen = {key for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]}
    count = max(1, math.ceil(float(ordered.start_s.max()) / step_seconds))
    frames, running, position = [], Counter(), 0
    for index in range(1, count + 1):
        end = index * step_seconds
        while position < len(ordered) and ordered.start_s.iloc[position] <= end:
            running.update(key for key in tokens[position] if key in chosen)
            position += 1
        frames.append((end, dict(running)))
    return frames


def daily_timelapse(datasets, speaker=None, limit=50):
    """Um quadro por dia com data, acumulando as palavras de conteúdo dos conjuntos."""
    by_day = {}
    for data in datasets:
        if data.get('day'):
            for row in data['words']:
                if speaker is None or row['speaker'] == speaker:
                    by_day.setdefault(data['day'], Counter())[row['speaker'], row['word']] += row['count']
    days = sorted(by_day)
    totals = sum((by_day[day] for day in days), Counter())
    chosen = {key for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]}
    frames, running = [], Counter()
    for index, day in enumerate(days, 1):
        running.update({key: value for key, value in by_day[day].items() if key in chosen})
        frames.append((index, dict(running)))
    return days, frames


def daily_metrics(datasets):
    rows = []
    for data in datasets:
        for person in data['participants']:
            rows.append(dict(day=data.get('day') or 'Sem data', **person))
    extra = ['spoken_word_count', 'measured_speech_seconds']
    frame = pd.DataFrame(rows, columns=['day', 'speaker', 'segments', 'estimated_speech_seconds',
                                        'content_word_count', *extra])
    frame[extra] = frame[extra].fillna(0)
    if frame.empty:
        return frame.assign(share=pd.Series(dtype=float), distinct_words=pd.Series(dtype=int),
                            words_per_minute=pd.Series(dtype=float))
    result = frame.groupby(['day', 'speaker'], as_index=False).sum(numeric_only=True)
    # Ausência em um dia conhecido significa zero, não uma linha que salta o dia.
    index = pd.MultiIndex.from_product([sorted(frame.day.unique()), sorted(frame.speaker.unique())],
                                       names=['day', 'speaker'])
    result = result.set_index(['day', 'speaker']).reindex(index, fill_value=0).reset_index()
    totals = result.groupby('day')['estimated_speech_seconds'].transform('sum')
    result['share'] = result['estimated_speech_seconds'].div(totals.where(totals > 0)).fillna(0) * 100
    vocabularies = {}
    for data in datasets:
        for row in data['words']:
            vocabularies.setdefault((data.get('day') or 'Sem data', row['speaker']), set()).add(row['word'])
    result['distinct_words'] = [len(vocabularies.get((r.day, r.speaker), set())) for r in result.itertuples()]
    # PPM nunca é somado: recalculado após agrupar por dia.
    seconds = result['measured_speech_seconds']
    result['words_per_minute'] = result['spoken_word_count'] / seconds.where(seconds > 0) * 60
    return result


def compare_people(data, first, second):
    counts = {name: Counter() for name in (first, second)}
    for row in data['words']:
        if row['speaker'] in counts:
            counts[row['speaker']][row['word']] += row['count']
    left, right = counts[first], counts[second]
    union = set(left) | set(right)
    common = set(left) & set(right)
    table = pd.DataFrame([dict(word=word, first=left[word], second=right[word]) for word in sorted(union)],
                         columns=['word', 'first', 'second'])
    return table, len(common), (100 * len(common) / len(union) if union else 0)


def characteristic_words(data, speaker, minimum=2):
    own, rest = Counter(), Counter()
    for row in data['words']:
        (own if row['speaker'] == speaker else rest)[row['word']] += row['count']
    total_own, total_rest = sum(own.values()), sum(rest.values())
    rows = []
    if total_own and total_rest:
        for word, count in own.items():
            if count < minimum:
                continue
            own_rate = count / total_own * 1000
            rest_rate = rest[word] / total_rest * 1000
            if own_rate > rest_rate:
                rows.append(dict(word=word, count=count, rest_count=rest[word], own_rate=own_rate,
                                 rest_rate=rest_rate, difference=own_rate - rest_rate))
    return pd.DataFrame(rows, columns=['word', 'count', 'rest_count', 'own_rate', 'rest_rate', 'difference']).sort_values(
        ['difference', 'count', 'word'], ascending=[False, False, True])


def search_context(frame, query, speaker=None, whole=True):
    query = unicodedata.normalize('NFC', query.strip())
    if not query:
        return frame.iloc[:0].assign(before='', after='')
    pattern = re.escape(query)
    if whole:
        pattern = r'(?<!\w)' + pattern + r'(?!\w)'
    normalized = frame['text'].map(lambda text: unicodedata.normalize('NFC', str(text)))
    matches = normalized.str.contains(pattern, case=False, regex=True, na=False)
    if speaker:
        matches &= frame['speaker'].eq(speaker)
    result = frame.loc[matches].copy()
    before, after = {}, {}
    for _, group in frame.groupby('session_id', sort=False):
        records = list(group.index)
        for position, index in enumerate(records):
            def neighbor(offset):
                target = position + offset
                if 0 <= target < len(records):
                    row = frame.loc[records[target]]
                    return f'{row.speaker}: {row.text}'
                return ''
            before[index], after[index] = neighbor(-1), neighbor(1)
    result['before'] = [before[index] for index in result.index]
    result['after'] = [after[index] for index in result.index]
    return result


def frequent_expressions(frame, size=2, minimum=2):
    counts, references = Counter(), {}
    for row in frame.itertuples():
        for sentence in re.split(r'[.!?;\n]+', row.text):
            tokens = dictionary_words(sentence)
            for index in range(len(tokens) - size + 1):
                phrase = ' '.join(tokens[index:index + size])
                key = row.speaker, phrase
                counts[key] += 1
                references.setdefault(key, dict(session_id=row.session_id, start_s=row.start_s, text=row.text))
    rows = [dict(speaker=speaker, phrase=phrase, count=count, **references[speaker, phrase])
            for (speaker, phrase), count in counts.items() if count >= minimum]
    return pd.DataFrame(rows, columns=['speaker', 'phrase', 'count', 'session_id', 'start_s', 'text']).sort_values(
        ['count', 'phrase', 'speaker'], ascending=[False, True, True])


def activity_windows(frame, seconds=300):
    """União dos intervalos: sobreposição não duplica o tempo ativo por janela."""
    windows = {}
    for row in frame.itertuples():
        for index in range(math.floor(row.start_s / seconds), math.ceil(row.end_s / seconds)):
            start, end = max(row.start_s, index * seconds), min(row.end_s, (index + 1) * seconds)
            if end > start:
                windows.setdefault((row.session_id, index), []).append((start, end))
    rows = []
    for (session, index), intervals in sorted(windows.items()):
        intervals.sort()
        start, end = intervals[0]
        covered = 0
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        covered += end - start
        rows.append(dict(session_id=session, start_s=index * seconds, end_s=(index + 1) * seconds,
                         active_seconds=covered))
    return pd.DataFrame(rows, columns=['session_id', 'start_s', 'end_s', 'active_seconds'])


def session_summary(frame, maximum=5):
    """Resumo extrativo: trechos originais relevantes e temas lexicais ancorados."""
    tokens = [words(text) for text in frame['text']]
    counts = Counter(word for row in tokens for word in row)
    ranked = []
    for position, row in enumerate(frame.itertuples()):
        unique = set(tokens[position])
        if unique:
            score = sum(math.log1p(counts[word]) for word in unique) / math.sqrt(len(tokens[position]))
            ranked.append((score, position))
    chosen, seen = [], set()
    for _, position in sorted(ranked, key=lambda item: (-item[0], item[1])):
        text = frame.iloc[position]['text'].strip().lower()
        if text not in seen:
            chosen.append(position)
            seen.add(text)
        if len(chosen) >= maximum:
            break
    highlights = frame.iloc[sorted(chosen)].copy()
    topics = []
    for term, count in counts.most_common(8):
        position = next(i for i, row in enumerate(tokens) if term in row)
        ref = frame.iloc[position]
        topics.append(dict(term=term, count=count, session_id=ref.session_id, speaker=ref.speaker,
                           start_s=ref.start_s, text=ref.text))
    return highlights, pd.DataFrame(topics, columns=['term', 'count', 'session_id', 'speaker', 'start_s', 'text'])


def demo_word_timings(segments):
    """Palavras sintéticas distribuídas uniformemente dentro de cada trecho."""
    rows = []
    for segment in segments:
        tokens = segment['text'].split()
        span = (segment['end_s'] - segment['start_s']) / len(tokens)
        rows.extend(dict(session_id=segment['session_id'], speaker=segment['speaker'], word=token,
                         start_s=segment['start_s'] + index * span, end_s=segment['start_s'] + (index + 1) * span)
                    for index, token in enumerate(tokens))
    return rows


def demo_analytics():
    """Amostra sintética coerente para páginas que exibem frases e evolução."""
    from src.dashboard.report import combine_dashboards
    datasets = []
    for day, session in [('2026-10-26', 'demo_01'), ('2026-10-27', 'demo_02')]:
        segments = [dict(session_id=session, speaker=speaker, start_s=index * 12.0,
                         end_s=index * 12.0 + 15, text=text)
                    for index, (speaker, text) in enumerate([
                        ('Aurora', 'A equipe precisa estudar o mapa. Vamos jogar em equipe.'),
                        ('Bento', 'Vamos jogar em equipe. O mapa tem uma passagem.'),
                        ('Aurora', 'Minha estratégia é defender a passagem e estudar o mapa.'),
                        ('Cora', 'A vitória depende da equipe. Vamos jogar em equipe.'),
                        ('Bento', 'O mapa tem uma passagem. A estratégia depende da equipe.'),
                    ])]
        frame = pd.DataFrame(segments)
        timings = demo_word_timings(segments)
        measured = speech_timing(pd.DataFrame(timings)).set_index('speaker').measured_speech_seconds
        participants, vocabulary, frequencies = [], [], []
        for speaker, group in frame.groupby('speaker'):
            content = Counter(word for text in group.text for word in words(text))
            all_words = Counter(word for text in group.text for word in dictionary_words(text))
            spoken = sum(len(dictionary_words(row['word'])) for row in timings if row['speaker'] == speaker)
            participants.append(dict(speaker=speaker, segments=len(group), estimated_speech_seconds=len(group) * 15,
                                     content_word_count=sum(content.values()), spoken_word_count=spoken,
                                     measured_speech_seconds=float(measured[speaker]),
                                     words_per_minute=words_per_minute(spoken, float(measured[speaker]))))
            frequencies.extend(dict(speaker=speaker, word=word, count=count) for word, count in content.items())
            vocabulary.extend(dict(speaker=speaker, word=word, count=count) for word, count in all_words.items())
        datasets.append(dict(session_id=session, day=day, demo=True, quality_status='DEMO', validated_at=None,
                             participants=participants, words=frequencies, vocabulary=vocabulary, segments=segments,
                             recording_seconds=75, timing_source='palavras', word_timings=timings))
    data = combine_dashboards(datasets)
    data.update(demo=True, quality_status='DEMO')
    return data, datasets
