"""Validacao dos segmentos Whisper frente aos arquivos individuais do Craig."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from src.ingestion.inspect_files import audio_files, audio_duration, get_speaker

REQUIRED = {'session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'text'}
WORD_REQUIRED = {'session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'word', 'probability'}


def check_words(source: Path, track_map: dict, errors: list[str], session_id: str | None = None,
                durations: dict | None = None):
    """Valida words.parquet quando existir; sessoes antigas nao o possuem."""
    if not source.exists():
        return None
    try:
        words = pd.read_parquet(source)
    except Exception as exc:
        errors.append(f'Nao foi possivel ler words.parquet: {type(exc).__name__}: {exc}')
        return None
    missing = WORD_REQUIRED - set(words.columns)
    if missing:
        errors.append('words.parquet sem colunas obrigatorias: ' + ', '.join(sorted(missing)))
        return int(len(words))
    start = pd.to_numeric(words['start_s'], errors='coerce')
    end = pd.to_numeric(words['end_s'], errors='coerce')
    start, end = start.astype(float), end.astype(float)
    invalid = (~np.isfinite(start)) | (~np.isfinite(end)) | (start < 0) | (end < start)
    if invalid.any():
        errors.append(f'{int(invalid.sum())} palavra(s) com timestamp invalido.')
    unknown = ~words['source_file'].isin(track_map)
    if unknown.any():
        errors.append(f'{int(unknown.sum())} palavra(s) referenciam audio inexistente.')
    if session_id is not None:
        divergent = words['session_id'].astype(str).ne(session_id)
        if divergent.any():
            errors.append(f'{int(divergent.sum())} palavra(s) com session_id divergente.')
    wrong = (~unknown) & words['speaker'].ne(words['source_file'].map(track_map))
    if wrong.any():
        errors.append(f'{int(wrong.sum())} palavra(s) associadas ao participante errado.')
    exceeded = outside = 0
    for filename, duration in (durations or {}).items():
        if duration is None or not math.isfinite(duration) or duration <= 0:
            continue
        in_file = words['source_file'] == filename
        exceeded += int((in_file & (end > duration + 2)).sum())
        outside += int((in_file & (start >= duration)).sum())
    if exceeded:
        errors.append(f'{exceeded} palavra(s) terminam mais de 2s depois do fim do audio.')
    if outside:
        errors.append(f'{outside} palavra(s) comecam depois do fim do audio; revisao manual necessaria.')
    return int(len(words))


def validate_session(session_id: str, base_dir: Path = Path('.')) -> dict:
    base_dir = Path(base_dir)
    raw_dir = base_dir / 'data' / 'raw' / session_id
    processed_dir = base_dir / 'data' / 'processed' / session_id
    quality_dir = base_dir / 'data' / 'quality' / session_id
    quality_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    timestamp_events = []
    metadata_path = processed_dir / 'metadata.json'
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            timestamp_events = metadata.get('timestamp_events', [])
            if not isinstance(timestamp_events, list) or any(
                    not isinstance(event, dict) or event.get('action') not in ('clipped', 'review')
                    for event in timestamp_events):
                raise ValueError('timestamp_events invalido')
            clipped = sum(event['action'] == 'clipped' for event in timestamp_events)
            review = sum(event['action'] == 'review' for event in timestamp_events)
            if clipped:
                warnings.append(f'{clipped} trecho(s) tiveram o fim limitado ao audio; '
                                'revise o texto e os detalhes em metadata.json (timestamp_events).')
            if review:
                warnings.append(f'{review} ocorrencia(s) de timestamps exigem revisao; consulte metadata.json.')
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f'Nao foi possivel ler os registros de timestamps: {exc}')
    tracks = audio_files(raw_dir) if raw_dir.is_dir() else []
    track_map = {p.name: get_speaker(p.name) for p in tracks}
    durations = {p.name: audio_duration(p) for p in tracks}
    if not tracks:
        errors.append(f'Nao existem faixas de audio em {raw_dir}.')
    if len(set(track_map.values())) < len(track_map):
        errors.append('Mais de uma faixa identificada com o mesmo participante; confira os nomes.')

    source = processed_dir / 'segments.parquet'
    if not source.exists():
        errors.append(f'Arquivo processado nao encontrado: {source}.')
        df = pd.DataFrame()
    else:
        try:
            df = pd.read_parquet(source)
        except Exception as exc:
            errors.append(f'Nao foi possivel ler o Parquet: {type(exc).__name__}: {exc}')
            df = pd.DataFrame()

    missing = REQUIRED - set(df.columns)
    if source.exists() and missing:
        errors.append('Colunas obrigatorias ausentes: ' + ', '.join(sorted(missing)))
    participants = {}
    count = int(len(df))
    if not missing and source.exists():
        if df.empty:
            warnings.append('Nenhum trecho foi transcrito; verifique os audios e a deteccao de voz.')
        data = df.copy()
        data['start_num'] = pd.to_numeric(data['start_s'], errors='coerce')
        data['end_num'] = pd.to_numeric(data['end_s'], errors='coerce')
        invalid_time = (~data['start_num'].map(math.isfinite)) | (~data['end_num'].map(math.isfinite))
        bad_interval = invalid_time | (data['start_num'] < 0) | (data['end_num'] <= data['start_num'])
        if bad_interval.any():
            errors.append(f'{int(bad_interval.sum())} trecho(s) com timestamp invalido.')
        text_empty = data['text'].isna() | data['text'].astype(str).str.strip().eq('')
        if text_empty.any():
            errors.append(f'{int(text_empty.sum())} trecho(s) com texto vazio.')
        wrong_session = data['session_id'].astype(str).ne(session_id)
        if wrong_session.any():
            errors.append(f'{int(wrong_session.sum())} trecho(s) com session_id divergente.')
        unknown_files = ~data['source_file'].isin(track_map)
        if unknown_files.any():
            errors.append(f'{int(unknown_files.sum())} trecho(s) referenciam audio inexistente.')
        expected_speaker = data['source_file'].map(track_map)
        wrong_speaker = (~unknown_files) & data['speaker'].ne(expected_speaker)
        if wrong_speaker.any():
            errors.append(f'{int(wrong_speaker.sum())} trecho(s) associados ao participante errado.')
        exceeded = 0
        outside = 0
        for filename, duration in durations.items():
            if duration is None or not math.isfinite(duration) or duration <= 0:
                warnings.append(f'Nao foi possivel medir a duracao de {filename}; use ffprobe.')
                continue
            exceeded += int(((data['source_file'] == filename) &
                             (data['end_num'] > duration + 2)).sum())
            outside += int(((data['source_file'] == filename) &
                            (data['start_num'] >= duration)).sum())
        if exceeded:
            errors.append(f'{exceeded} trecho(s) terminam mais de 2s depois do fim do audio.')
        if outside:
            errors.append(f'{outside} trecho(s) comecam depois do fim do audio; revisao manual necessaria.')
        duplicates = int(data.duplicated(subset=['source_file', 'start_s', 'end_s', 'text']).sum())
        if duplicates:
            warnings.append(f'{duplicates} trecho(s) duplicados no Parquet.')
        for speaker in sorted(set(track_map.values())):
            group = data[data['speaker'] == speaker].copy()
            valid = group['start_num'].notna() & group['end_num'].notna()
            speech_seconds = (group.loc[valid, 'end_num'] - group.loc[valid, 'start_num']).clip(lower=0).sum()
            participants[speaker] = {
                'segments': int(len(group)),
                'estimated_segment_seconds': round(float(speech_seconds), 2),
                'raw_word_count': int(group['text'].fillna('').astype(str).str.split().map(len).sum())
            }
            if group.empty:
                warnings.append(f'{speaker} sem trechos transcritos; verifique se falou na gravacao.')

    word_count = check_words(processed_dir / 'words.parquet', track_map, errors, session_id, durations)

    report = {
        'session_id': session_id,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FAIL' if errors else ('WARN' if warnings else 'PASS'),
        'summary': {
            'audio_tracks': len(tracks),
            'transcript_segments': count,
            'participants_with_segments': sum(p['segments'] > 0 for p in participants.values()),
            'word_timestamps': word_count
        },
        'participants': participants,
        'errors': errors,
        'warnings': warnings,
        'timestamp_events': timestamp_events,
        'notes': [
            'estimated_segment_seconds soma duracoes dos trechos Whisper, nao tempo exato de voz ativa.',
            'As transcricoes podem conter erros; valide manualmente uma amostra antes de interpretar resultados.'
        ]
    }
    out = quality_dir / 'validation_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'VALIDACAO [{report["status"]}] sessao={session_id}: '
          f'{len(tracks)} faixas, {count} trechos, '
          f'{report["summary"]["participants_with_segments"]} participantes com fala')
    for msg in errors:
        print('  ERRO: ' + msg)
    for msg in warnings:
        print('  AVISO: ' + msg)
    print(f'Relatorio: {out}')
    return report
