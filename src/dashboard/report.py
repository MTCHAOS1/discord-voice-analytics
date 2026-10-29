"""Gera um HTML autonomo, sem audio ou trechos de transcricao."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

import pandas as pd

from src.database.load_warehouse import build_plan
from src.transformation.analyze import words, dictionary_words
from src.dashboard.analytics import speech_timing, words_per_minute

ASSETS = Path(__file__).with_name('assets')
# Sem identificadores pessoais fixos; os nomes vêm dos dados locais.
PARTICIPANT_NAMES: dict[str, str] = {}


def display_name(speaker: str) -> str:
    return PARTICIPANT_NAMES.get(speaker, speaker)


WORD_FIELDS = ['session_id', 'speaker', 'start_s', 'end_s', 'word']


def recording_seconds(session_id: str, base_dir: Path):
    """Maior duração de faixa em metadata.json; None se ausente ou inválida."""
    try:
        metadata = json.loads((Path(base_dir) / 'data/processed' / session_id / 'metadata.json')
                              .read_text(encoding='utf-8'))
        durations = [track['duration_s'] for track in metadata['tracks']]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    valid = [d for d in durations if isinstance(d, (int, float)) and math.isfinite(d) and d > 0]
    return max(valid) if valid else None


def timing_metrics(segments, timed_words):
    """Palavras ditas, fala medida e PPM por pessoa; usa os trechos se não há tempo por palavra."""
    if timed_words is not None:
        source, texts = timed_words, timed_words.assign(text=timed_words['word'])
    else:
        source, texts = segments, segments
    measured = speech_timing(source).set_index('speaker')['measured_speech_seconds']
    spoken = {speaker: sum(len(dictionary_words(str(text))) for text in group['text'])
              for speaker, group in texts.groupby('speaker')}
    result = {}
    for speaker in segments['speaker'].unique():
        count, seconds = spoken.get(speaker, 0), float(measured.get(speaker, 0.0))
        result[speaker] = dict(spoken_word_count=count, measured_speech_seconds=seconds,
                               words_per_minute=words_per_minute(count, seconds))
    return result


def load_dashboard(session_id: str, base_dir: Path = Path('.'),
                   allow_warnings: bool = False, include_segments: bool = False) -> dict:
    plan = build_plan(session_id, base_dir=base_dir, allow_warnings=allow_warnings)
    data = plan['data']
    # Recontar a partir dos segmentos evita somar listas truncadas ao top 30.
    frequencies = []
    vocabulary = []
    for speaker, group in data['segments'].groupby('speaker'):
        counts = Counter(word for text in group['text'] for word in words(str(text)))
        frequencies.extend(dict(speaker=display_name(str(speaker)), word=word, count=count)
                           for word, count in counts.items())
        all_counts = Counter(word for text in group['text'] for word in dictionary_words(str(text)))
        vocabulary.extend(dict(speaker=display_name(str(speaker)), word=word, count=count)
                          for word, count in all_counts.items())
    words_path = Path(base_dir) / 'data/processed' / session_id / 'words.parquet'
    timed_words = pd.read_parquet(words_path) if words_path.exists() else None
    if timed_words is not None and timed_words.empty:
        timed_words = None
    timing = timing_metrics(data['segments'], timed_words)
    result = dict(session_id=session_id, demo=False,
                quality_status=data['report']['status'],
                validated_at=data['report'].get('generated_at_utc'),
                participants=[dict(row, speaker=display_name(str(row['speaker'])),
                                   **timing.get(row['speaker'], {}))
                              for row in data['metrics'].to_dict(orient='records')], words=frequencies,
                vocabulary=vocabulary, recording_seconds=recording_seconds(session_id, base_dir),
                timing_source='palavras' if timed_words is not None else 'trechos')
    if include_segments:
        result['segments'] = [dict(row, speaker=display_name(str(row['speaker']))) for row in
                              data['segments'][['session_id', 'speaker', 'start_s', 'end_s', 'text']].to_dict(orient='records')]
        if timed_words is not None:
            result['word_timings'] = [dict(row, speaker=display_name(str(row['speaker']))) for row in
                                      timed_words[WORD_FIELDS].to_dict(orient='records')]
    return result


def demo_dashboard() -> dict:
    return dict(
        session_id='demo_sintetica', demo=True, quality_status='DEMO', validated_at=None,
        participants=[
            dict(speaker='Aurora', segments=24, estimated_speech_seconds=180, content_word_count=360,
                 spoken_word_count=450, measured_speech_seconds=170, words_per_minute=450 / 170 * 60),
            dict(speaker='Bento', segments=18, estimated_speech_seconds=120, content_word_count=250,
                 spoken_word_count=290, measured_speech_seconds=115, words_per_minute=290 / 115 * 60),
            dict(speaker='Cora', segments=12, estimated_speech_seconds=90, content_word_count=190,
                 spoken_word_count=210, measured_speech_seconds=85, words_per_minute=210 / 85 * 60),
            dict(speaker='Davi', segments=6, estimated_speech_seconds=30, content_word_count=60,
                 spoken_word_count=60, measured_speech_seconds=28, words_per_minute=60 / 28 * 60),
        ],
        recording_seconds=480, timing_source='palavras',
        words=[dict(speaker=speaker, word=word, count=count)
               for speaker, terms in {
                   'Aurora': [('equipe', 16), ('jogo', 12), ('estratégia', 8)],
                   'Bento': [('jogo', 15), ('equipe', 10), ('mapa', 7)],
                   'Cora': [('mapa', 12), ('equipe', 8), ('vitória', 6)],
                   'Davi': [('jogo', 5), ('vitória', 4), ('mapa', 3)],
               }.items() for word, count in terms],
    )


def combine_dashboards(datasets: list[dict]) -> dict:
    """Soma sessões já validadas; apelidos iguais representam a mesma pessoa."""
    if not datasets:
        raise ValueError('Nenhuma sessão processada encontrada.')
    participants = {}
    counts = Counter()
    vocabulary = Counter()
    for data in datasets:
        for row in data['participants']:
            target = participants.setdefault(row['speaker'], dict(
                speaker=row['speaker'], segments=0, estimated_speech_seconds=0,
                content_word_count=0, spoken_word_count=0, measured_speech_seconds=0))
            for field in ('segments', 'estimated_speech_seconds', 'content_word_count'):
                target[field] += row[field]
            for field in ('spoken_word_count', 'measured_speech_seconds'):
                target[field] += row.get(field, 0)
        for row in data['words']:
            counts[row['speaker'], row['word']] += row['count']
        for row in data.get('vocabulary', data['words']):
            vocabulary[row['speaker'], row['word']] += row['count']
    for target in participants.values():
        target['words_per_minute'] = words_per_minute(target['spoken_word_count'], target['measured_speech_seconds'])
    result = dict(session_id='todas_as_sessoes', demo=False,
                quality_status='WARN' if any(d['quality_status'] == 'WARN' for d in datasets) else 'PASS',
                validated_at=None, sessions=[d['session_id'] for d in datasets],
                participants=list(participants.values()),
                words=[dict(speaker=speaker, word=word, count=count)
                       for (speaker, word), count in sorted(counts.items())],
                vocabulary=[dict(speaker=speaker, word=word, count=count)
                            for (speaker, word), count in sorted(vocabulary.items())],
                recordings={d['session_id']: d.get('recording_seconds') for d in datasets},
                timing_sources={d['session_id']: d.get('timing_source', 'trechos') for d in datasets})
    if any('segments' in data for data in datasets):
        result['segments'] = [row for data in datasets for row in data.get('segments', [])]
    if any('word_timings' in data for data in datasets):
        result['word_timings'] = [row for data in datasets for row in data.get('word_timings', [])]
    return result


def write_dashboard(data: dict, output: Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != '.html':
        raise ValueError('O destino do dashboard deve ter extensao .html.')
    # O HTML independente continua contendo apenas agregações, nunca frases.
    payload = dict({key: value for key, value in data.items() if key not in ('segments', 'word_timings')},
                   generated_at=datetime.now(timezone.utc).isoformat())
    # JSON dentro de script precisa impedir fechamento antecipado por dados.
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False)
               .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e'))
    template = (ASSETS / 'dashboard.html').read_text(encoding='utf-8')
    html = template.replace('__TITLE__', escape(str(data['session_id'])))
    html = html.replace('__STYLE__', (ASSETS / 'dashboard.css').read_text(encoding='utf-8'))
    html = html.replace('__SCRIPT__', (ASSETS / 'dashboard.js').read_text(encoding='utf-8'))
    html = html.replace('__DATA__', encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Nunca sobrescrever silenciosamente um relatorio existente.
    with output.open('x', encoding='utf-8') as file:
        file.write(html)
    return output
