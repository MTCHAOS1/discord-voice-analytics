'''Carrega a sessao local e suas metricas para PostgreSQL.

Dry-run por padrao. Com --apply, substitui apenas a sessao escolhida em
uma unica transacao, incluindo tabelas dependentes por DELETE CASCADE.
'''
from __future__ import annotations
import json
import math
import os
import re
from pathlib import Path

import pandas as pd

SEGMENT_FIELDS = {'session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'text'}
METRIC_FIELDS = {'speaker', 'segments', 'estimated_speech_seconds', 'content_word_count'}
WORD_FIELDS = {'speaker', 'word', 'count'}


def _read_rows(session_id: str, base_dir: Path, allow_warnings: bool) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('Identificador de sessao invalido.')
    report_path = base_dir / 'data/quality' / session_id / 'validation_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Faltando relatorio: {report_path}. Execute validate antes.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    status = report.get('status')
    if report.get('session_id') != session_id:
        raise ValueError('Relatorio de qualidade pertence a outra sessao.')
    if status not in ('PASS', 'WARN'):
        raise ValueError(f'Relatorio de qualidade {status!r}. Nao carregar com FAIL.')
    if status == 'WARN' and not allow_warnings:
        raise ValueError('Relatorio WARN. Revise e use --allow-warnings se desejar prosseguir.')

    processed = base_dir / 'data/processed' / session_id
    curated = base_dir / 'data/curated' / session_id
    segments = pd.read_parquet(processed / 'segments.parquet')
    metrics = pd.read_csv(curated / 'speaker_metrics.csv', encoding='utf-8-sig')
    words = pd.read_csv(curated / 'top_words.csv', encoding='utf-8-sig')
    for name, df, required in [('segments', segments, SEGMENT_FIELDS),
                               ('speaker_metrics', metrics, METRIC_FIELDS),
                               ('top_words', words, WORD_FIELDS)]:
        absent = required - set(df.columns)
        if absent:
            raise ValueError(f'{name} sem colunas: {sorted(absent)}')
    if segments.empty or metrics.empty:
        raise ValueError('Dados vazios. Nao substituiremos os dados de uma sessao por dados vazios.')
    if segments['session_id'].astype(str).ne(session_id).any():
        raise ValueError('segments.parquet contem sessao divergente.')
    if len(segments) != int(report['summary']['transcript_segments']):
        raise ValueError('Quantidade de trechos difere do relatorio. Execute validate novamente.')
    if metrics['speaker'].isna().any() or metrics['speaker'].duplicated().any():
        raise ValueError('Participantes duplicados ou invalidos no CSV de metricas.')
    speakers = set(metrics['speaker'].astype(str))
    if speakers != set(segments['speaker'].astype(str)):
        raise ValueError('Participantes divergem entre metricas e transcricoes.')
    if len(speakers) != int(report['summary']['participants_with_segments']):
        raise ValueError('Quantidade de participantes diverge do relatorio.')
    if not set(words['speaker'].astype(str)).issubset(speakers):
        raise ValueError('Palavras atribuidas a participante desconhecido.')
    if words.duplicated(['speaker', 'word']).any():
        raise ValueError('Frequencia duplicada para participante/palavra.')
    if segments.duplicated(['speaker', 'source_file', 'start_s', 'end_s']).any():
        raise ValueError('Trechos com chave de tempo duplicada.')
    for field in ['start_s', 'end_s']:
        segments[field] = pd.to_numeric(segments[field], errors='coerce')
        if not segments[field].map(math.isfinite).all():
            raise ValueError(f'Timestamps invalidos: {field}')
    if ((segments['start_s'] < 0) | (segments['end_s'] <= segments['start_s'])).any():
        raise ValueError('Intervalos invalidos no Parquet.')
    if segments['text'].isna().any() or segments['text'].astype(str).str.strip().eq('').any():
        raise ValueError('Trechos com texto vazio.')
    if metrics['segments'].sum() != len(segments):
        raise ValueError('Soma dos trechos por participante nao coincide com transcricoes.')
    for field in ['segments', 'estimated_speech_seconds', 'content_word_count']:
        v = pd.to_numeric(metrics[field], errors='coerce')
        if not v.map(math.isfinite).all() or (v < 0).any():
            raise ValueError(f'Metrica invalida: {field}')
        if field in ('segments', 'content_word_count') and (v % 1 != 0).any():
            raise ValueError(f'Metrica deve ser inteira: {field}')
    if not words.empty:
        v = pd.to_numeric(words['count'], errors='coerce')
        if not v.map(math.isfinite).all() or (v <= 0).any() or (v % 1 != 0).any():
            raise ValueError('Contagem de palavras invalida.')
    from src.transformation.analyze import words as content_words
    for row in metrics.itertuples(index=False):
        group = segments[segments['speaker'] == row.speaker]
        duration = round(float((group['end_s'] - group['start_s']).sum()), 2)
        word_count = sum(len(content_words(str(text))) for text in group['text'])
        if (row.segments != len(group)
                or not math.isclose(float(row.estimated_speech_seconds), duration, abs_tol=0.0001)
                or row.content_word_count != word_count):
            raise ValueError('Metricas divergem da transcricao. Execute analyze novamente.')
    source_by_speaker = segments.groupby('speaker')['source_file'].nunique()
    if (source_by_speaker != 1).any():
        raise ValueError('Mais de um arquivo associado ao mesmo speaker nesta sessao.')

    return {'report': report, 'segments': segments, 'metrics': metrics, 'words': words}


def build_plan(session_id: str, base_dir: Path = Path('.'), allow_warnings: bool = False) -> dict:
    data = _read_rows(session_id, Path(base_dir), allow_warnings)
    counts = {'participants': len(data['metrics']), 'segments': len(data['segments']),
              'words': len(data['words']), 'audio_tracks': int(data['report']['summary']['audio_tracks'])}
    return {'session_id': session_id, 'counts': counts, 'data': data}


def _pg_connection():
    from dotenv import load_dotenv
    import psycopg
    load_dotenv('.env')
    dsn = os.environ.get('DATABASE_URL', '').strip()
    if dsn:
        return psycopg.connect(dsn, connect_timeout=10)
    password = os.environ.get('PGPASSWORD', '')
    if not password or password.startswith('COLOQUE_'):
        raise ValueError('Configure PGPASSWORD no .env antes do --apply.')
    return psycopg.connect(
        host=os.environ.get('PGHOST', '127.0.0.1'),
        port=os.environ.get('PGPORT', '5432'),
        dbname=os.environ.get('PGDATABASE', 'discord_voice_analytics'),
        user=os.environ.get('PGUSER', 'postgres'), password=password, connect_timeout=10,
    )


def load_session(session_id: str, apply: bool = False,
                 allow_warnings: bool = False, base_dir: Path = Path('.')) -> dict:
    plan = build_plan(session_id, base_dir=base_dir, allow_warnings=allow_warnings)
    counts = plan['counts']
    print(f'{"CARGA" if apply else "SIMULACAO"} PostgreSQL: {session_id} '
          f'| {counts["participants"]} participantes, {counts["segments"]} trechos, '
          f'{counts["words"]} frequencias de palavras')
    if not apply:
        print('Nenhuma conexao ou alteracao no PostgreSQL. Use --apply para gravar.')
        return {'dry_run': True, **counts}

    data = plan['data']
    schema_file = Path(base_dir) / 'sql/warehouse/001_init.sql'
    schema_sql = schema_file.read_text(encoding='utf-8')

    speaker_file = {
        str(speaker): str(group['source_file'].iloc[0])
        for speaker, group in data['segments'].groupby('speaker')
    }
    participant_rows = [
        (session_id, str(r.speaker), speaker_file[str(r.speaker)],
         int(r.segments), float(r.estimated_speech_seconds), int(r.content_word_count))
        for r in data['metrics'].itertuples(index=False)
    ]
    transcript_rows = [
        (session_id, str(r.speaker), str(r.source_file),
         float(r.start_s), float(r.end_s), str(r.text))
        for r in data['segments'].itertuples(index=False)
    ]
    frequency_rows = [
        (session_id, str(r.speaker), str(r.word), int(r.count))
        for r in data['words'].itertuples(index=False)
    ]

    # Todas as alteracoes em uma transacao. Um erro faz ROLLBACK, sem carga parcial.
    with _pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            cur.execute('DELETE FROM warehouse.sessions WHERE session_id = %s', (session_id,))
            cur.execute('''INSERT INTO warehouse.sessions
                (session_id, audio_tracks, transcript_segments) VALUES (%s, %s, %s)''',
                (session_id, counts['audio_tracks'], counts['segments']))
            cur.executemany('''INSERT INTO warehouse.participants
                (session_id, speaker, source_file, segments, estimated_speech_seconds, content_word_count)
                VALUES (%s, %s, %s, %s, %s, %s)''', participant_rows)
            cur.executemany('''INSERT INTO warehouse.transcript_segments
                (session_id, speaker, source_file, start_s, end_s, text)
                VALUES (%s, %s, %s, %s, %s, %s)''', transcript_rows)
            if frequency_rows:
                cur.executemany('''INSERT INTO warehouse.word_frequency
                    (session_id, speaker, word, count) VALUES (%s, %s, %s, %s)''', frequency_rows)
    print('Concluido: sessao carregada em warehouse.* com transacao atomica.')
    return {'dry_run': False, **counts}
