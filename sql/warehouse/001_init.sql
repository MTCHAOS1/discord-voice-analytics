-- Discord Voice Analytics - etapa 5. Schema separado, sem alterar outros projetos.
CREATE SCHEMA IF NOT EXISTS warehouse;

CREATE TABLE IF NOT EXISTS warehouse.sessions (
    session_id TEXT PRIMARY KEY,
    audio_tracks INTEGER NOT NULL CHECK (audio_tracks >= 0),
    transcript_segments INTEGER NOT NULL CHECK (transcript_segments >= 0),
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Um apelido do Craig identifica o falante DENTRO da sessao; nao e identidade global.
CREATE TABLE IF NOT EXISTS warehouse.participants (
    session_id TEXT NOT NULL REFERENCES warehouse.sessions(session_id) ON DELETE CASCADE,
    speaker TEXT NOT NULL,
    source_file TEXT NOT NULL,
    segments INTEGER NOT NULL CHECK (segments >= 0),
    estimated_speech_seconds NUMERIC(12,3) NOT NULL CHECK (estimated_speech_seconds >= 0),
    content_word_count INTEGER NOT NULL CHECK (content_word_count >= 0),
    PRIMARY KEY (session_id, speaker)
);

CREATE TABLE IF NOT EXISTS warehouse.transcript_segments (
    session_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    source_file TEXT NOT NULL,
    start_s NUMERIC(12,3) NOT NULL CHECK (start_s >= 0),
    end_s NUMERIC(12,3) NOT NULL CHECK (end_s > start_s),
    text TEXT NOT NULL CHECK (length(btrim(text)) > 0),
    PRIMARY KEY (session_id, speaker, source_file, start_s, end_s),
    FOREIGN KEY (session_id, speaker)
        REFERENCES warehouse.participants(session_id, speaker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS warehouse.word_frequency (
    session_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    word TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (session_id, speaker, word),
    FOREIGN KEY (session_id, speaker)
        REFERENCES warehouse.participants(session_id, speaker) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcript_by_session_time
    ON warehouse.transcript_segments (session_id, start_s);
