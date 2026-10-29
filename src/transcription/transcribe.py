"""Transcribe separate Craig tracks with timestamps, entirely on-device."""
from pathlib import Path
import json
import math
import pandas as pd
from src.ingestion.inspect_files import audio_files, get_speaker, audio_duration
from src.transcription.backup import backup_session
from src.ingestion.session_catalog import recording_start

COLUMNS = ['session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'text']
WORD_COLUMNS = ['session_id', 'speaker', 'source_file', 'start_s', 'end_s', 'word', 'probability']


def bound_timestamps(start: float, end: float, duration: float | None):
    """Limita apenas intervalos válidos que ainda interceptam o áudio."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return end, 'review', 'Intervalo invalido; preservado para revisao.'
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return end, 'review', 'Duracao do audio indisponivel; timestamps preservados.'
    if start >= duration:
        return end, 'review', 'Trecho inteiramente fora do audio; preservado para revisao.'
    if end > duration:
        bounded = round(duration, 3)
        if bounded <= start:
            return end, 'review', 'Ajuste produziria intervalo vazio; preservado para revisao.'
        return bounded, 'clipped', 'Fim limitado a duracao do audio; confira o texto final.'
    return end, None, None


def transcribe(session_id: str, model_name: str = 'large-v3', device: str = 'cpu',
               base_dir: Path = Path('.')) -> Path:
    from faster_whisper import WhisperModel

    input_dir = Path(base_dir) / 'data/raw' / session_id
    output_dir = Path(base_dir) / 'data/processed' / session_id
    tracks = audio_files(input_dir)
    if not tracks:
        raise RuntimeError(f'Nenhum áudio válido encontrado em {input_dir}. Execute inspect primeiro.')
    try:
        backup = backup_session(session_id, base_dir)
    except Exception as exc:
        raise RuntimeError(f'Backup falhou; resultados anteriores preservados. {exc}') from exc
    if backup is not None:
        print(f'Backup dos resultados anteriores: {backup.resolve()}', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_type = 'float16' if device == 'cuda' else 'int8'
    print(f'Carregando Faster-Whisper: modelo={model_name}, dispositivo={device}, precisão={compute_type}')
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    rows = []
    word_rows = []
    metadata = {'session_id': session_id, 'model': model_name, 'device': device, 'tracks': [],
                'timestamp_events': [], 'word_timestamps': True, 'backup_path': str(backup.resolve()) if backup else None,
                'recording_started_at': recording_start(input_dir)}
    for path in tracks:
        speaker = get_speaker(path.name)
        duration = audio_duration(path)
        print(f'\nTranscrevendo {speaker}: {path.name}')
        metadata['tracks'].append({'speaker': speaker, 'file': path.name, 'duration_s': duration})
        segments, _ = model.transcribe(
            str(path), language='pt', beam_size=5, vad_filter=True,
            condition_on_previous_text=False, word_timestamps=True
        )
        segment_count = 0
        for seg in segments:
            transcript_text = seg.text.strip()
            if not transcript_text:
                continue
            start = round(float(seg.start), 3)
            original_end = round(float(seg.end), 3)
            end, action, reason = bound_timestamps(start, original_end, duration)
            if action:
                metadata['timestamp_events'].append(dict(
                    source_file=path.name, speaker=speaker, track_segment=segment_count + 1,
                    start_s=start if math.isfinite(start) else str(start),
                    original_end_s=original_end if math.isfinite(original_end) else str(original_end),
                    end_s=end if math.isfinite(end) else str(end),
                    duration_s=duration if duration is None or math.isfinite(duration) else str(duration),
                    action=action, reason=reason))
            rows.append({
                'session_id': session_id,
                'speaker': speaker,
                'source_file': path.name,
                'start_s': start,
                'end_s': end,
                'text': transcript_text,
            })
            segment_count += 1
            for item in getattr(seg, 'words', None) or []:
                word = str(item.word).strip()
                if not word:
                    continue
                word_start = round(float(item.start), 3)
                word_end = round(float(item.end), 3)
                word_end, _, _ = bound_timestamps(word_start, word_end, duration)
                word_rows.append({
                    'session_id': session_id, 'speaker': speaker, 'source_file': path.name,
                    'start_s': word_start, 'end_s': word_end, 'word': word,
                    'probability': getattr(item, 'probability', None),
                })
        print(f'  Trechos transcritos: {segment_count}')

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df = df.sort_values(['start_s', 'speaker', 'end_s']).reset_index(drop=True)
    df.to_parquet(output_dir / 'segments.parquet', index=False)
    words_df = pd.DataFrame(word_rows, columns=WORD_COLUMNS)
    if not words_df.empty:
        words_df['probability'] = pd.to_numeric(words_df['probability'], errors='coerce').astype(float)
        words_df = words_df.sort_values(['start_s', 'speaker']).reset_index(drop=True)
    else:
        words_df = words_df.astype({'start_s': float, 'end_s': float, 'probability': float})
    words_df.to_parquet(output_dir / 'words.parquet', index=False)
    df.to_csv(output_dir / 'segments.csv', index=False, encoding='utf-8-sig')
    with (output_dir / 'transcricao.txt').open('w', encoding='utf-8') as file:
        for row in df.itertuples(index=False):
            file.write(f'[{row.start_s:07.2f}s] {row.speaker}: {row.text}\n')
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'\nFinalizado: {len(df)} trechos em {output_dir}')
    if metadata['timestamp_events']:
        print(f'Timestamps: {len(metadata["timestamp_events"])} ocorrencia(s) registradas em metadata.json; '
              'execute validate e revise os apontamentos.')
    return output_dir / 'segments.parquet'
