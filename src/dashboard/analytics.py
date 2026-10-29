"""Análises locais rastreáveis, calculadas somente sobre dados já validados."""
from collections import Counter
import math
import re
import unicodedata

import pandas as pd

from src.transformation.analyze import dictionary_words, words

SEGMENT_COLUMNS = ['session_id', 'speaker', 'start_s', 'end_s', 'text']


def segment_frame(data):
    return pd.DataFrame(data.get('segments', []), columns=SEGMENT_COLUMNS).sort_values(
        ['session_id', 'start_s', 'speaker', 'end_s']).reset_index(drop=True)


def daily_metrics(datasets):
    rows = []
    for data in datasets:
        for person in data['participants']:
            rows.append(dict(day=data.get('day') or 'Sem data', **person))
    frame = pd.DataFrame(rows, columns=['day', 'speaker', 'segments', 'estimated_speech_seconds', 'content_word_count'])
    if frame.empty:
        return frame.assign(share=pd.Series(dtype=float), distinct_words=pd.Series(dtype=int))
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
        participants, vocabulary, frequencies = [], [], []
        for speaker, group in frame.groupby('speaker'):
            content = Counter(word for text in group.text for word in words(text))
            all_words = Counter(word for text in group.text for word in dictionary_words(text))
            participants.append(dict(speaker=speaker, segments=len(group), estimated_speech_seconds=len(group) * 15,
                                     content_word_count=sum(content.values())))
            frequencies.extend(dict(speaker=speaker, word=word, count=count) for word, count in content.items())
            vocabulary.extend(dict(speaker=speaker, word=word, count=count) for word, count in all_words.items())
        datasets.append(dict(session_id=session, day=day, demo=True, quality_status='DEMO', validated_at=None,
                             participants=participants, words=frequencies, vocabulary=vocabulary, segments=segments))
    data = combine_dashboards(datasets)
    data.update(demo=True, quality_status='DEMO')
    return data, datasets
