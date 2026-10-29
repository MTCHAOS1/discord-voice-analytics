-- Sessao e quantidade de falas
SELECT session_id, audio_tracks, transcript_segments, imported_at
FROM warehouse.sessions ORDER BY imported_at DESC;

-- Ranking de trechos por participante (nao equivale ao tempo exato de voz ativa)
SELECT speaker, segments, estimated_speech_seconds, content_word_count
FROM warehouse.participants
WHERE session_id = 'session_001'
ORDER BY segments DESC;

-- Top 10 palavras de cada participante
SELECT speaker, word, count
FROM (
    SELECT speaker, word, count,
           ROW_NUMBER() OVER (PARTITION BY speaker ORDER BY count DESC, word) AS position
    FROM warehouse.word_frequency
    WHERE session_id = 'session_001'
) AS ranked
WHERE position <= 10
ORDER BY speaker, position;

-- Visualizar a conversa em ordem cronologica (falas sobrepostas sao preservadas)
SELECT start_s, end_s, speaker, text
FROM warehouse.transcript_segments
WHERE session_id = 'session_001'
ORDER BY start_s, speaker;
