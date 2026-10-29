"""Datas das gravações; nunca usa a data de processamento como substituta."""
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo('America/Sao_Paulo')


def recording_start(raw_dir: Path) -> str | None:
    info = raw_dir / 'info.txt'
    if not info.is_file():
        return None
    match = re.search(r'^Start time:\s*([^\r\n]+)', info.read_text(encoding='utf-8-sig'), re.MULTILINE)
    if not match:
        return None
    return normalize_start(match.group(1).strip())


def normalize_start(value) -> str | None:
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.isoformat() if dt.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def session_catalog(base_dir: Path = Path('.')) -> list[dict]:
    root = Path(base_dir)
    catalog = []
    for path in sorted((root / 'data/processed').glob('*/segments.parquet')):
        session = path.parent.name
        if not re.fullmatch(r'[A-Za-z0-9_-]+', session):
            continue
        metadata_path = path.with_name('metadata.json')
        start = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if not isinstance(metadata, dict):
                raise ValueError(f'{session}: metadata.json deve conter um objeto.')
            start = normalize_start(metadata.get('recording_started_at'))
        start = start or recording_start(root / 'data/raw' / session)
        day = datetime.fromisoformat(start).astimezone(TIMEZONE).date().isoformat() if start else None
        catalog.append(dict(session_id=session, recording_started_at=start, day=day))
    return catalog
