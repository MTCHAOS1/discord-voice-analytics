"""Basic descriptive metrics; speech duration is an approximation from Whisper segments."""
from collections import Counter
from pathlib import Path
import re
import unicodedata
import pandas as pd

STOPWORDS = set('''a o as os de do da dos das e em no na nos nas um uma uns umas para por com que se ao aos à às é foi são ser como mas ou eu você vocês ele ela eles elas nós meu minha seus suas seu sua esse essa isso isto aquele aquela aqui ali lá já não sim tem tinha ter muito mais menos até quando onde então porque pra pro tá tô né ué hum ah ahn eh'''.split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r'[^\W\d_]+', text.lower(), re.UNICODE) if len(w) >= 3 and w not in STOPWORDS]


def dictionary_words(text: str) -> list[str]:
    """Todas as palavras alfabéticas, inclusive curtas e stopwords."""
    return re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', text).lower(), re.UNICODE)


def analyze(session_id: str, base_dir: Path = Path('.')):
    source = Path(base_dir) / 'data/processed' / session_id / 'segments.parquet'
    if not source.exists():
        raise FileNotFoundError(f'Transcreva a sessão antes: {source}')
    out = Path(base_dir) / 'data/curated' / session_id
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(source)
    if df.empty:
        print('Nenhuma fala encontrada; gerando arquivos vazios.')
    records = []
    term_rows = []
    for speaker, group in df.groupby('speaker'):
        all_words = [word for text in group['text'] for word in words(str(text))]
        # Não equivale a cronômetro perfeito; trechos podem incluir silêncio.
        speech_seconds = (group['end_s'] - group['start_s']).clip(lower=0).sum()
        records.append({
            'speaker': speaker,
            'segments': int(len(group)),
            'estimated_speech_seconds': round(float(speech_seconds), 2),
            'content_word_count': len(all_words),
        })
        for word, count in Counter(all_words).most_common(30):
            term_rows.append({'speaker': speaker, 'word': word, 'count': count})
    metrics = pd.DataFrame(records, columns=['speaker', 'segments', 'estimated_speech_seconds', 'content_word_count'])
    top_words = pd.DataFrame(term_rows, columns=['speaker', 'word', 'count'])
    metrics.to_csv(out / 'speaker_metrics.csv', index=False, encoding='utf-8-sig')
    top_words.to_csv(out / 'top_words.csv', index=False, encoding='utf-8-sig')
    print('\nMétricas por participante (duração de fala aproximada):')
    print(metrics.to_string(index=False))
    print(f'\nArquivos criados em {out.resolve()}')
