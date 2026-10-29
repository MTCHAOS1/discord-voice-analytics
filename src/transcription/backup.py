"""Snapshots locais dos resultados antes de uma nova transcrição."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4


def backup_session(session_id: str, base_dir: Path = Path('.')) -> Path | None:
    """Publica o backup apenas após copiar e conferir todos os arquivos."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('Identificador de sessao invalido.')
    data = Path(base_dir) / 'data'
    sources = {layer: data / layer / session_id for layer in ('processed', 'curated', 'quality')
               if (data / layer / session_id).exists()}
    if not sources:
        return None
    parent = data / 'backups' / session_id
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    staging = parent / ('.' + name + '.partial')
    destination = parent / name
    staging.mkdir()
    try:
        files = []
        for layer, source in sources.items():
            if source.is_symlink() or not source.is_dir():
                raise ValueError(f'Diretorio de resultados invalido: {source}')
            for path in source.rglob('*'):
                if path.is_symlink():
                    raise ValueError(f'Link simbolico nao permitido no backup: {path}')
            shutil.copytree(source, staging / layer)
            for path in sorted(source.rglob('*')):
                if not path.is_file():
                    continue
                relative = Path(layer) / path.relative_to(source)
                with path.open('rb') as original:
                    original_hash = hashlib.file_digest(original, 'sha256').hexdigest()
                with (staging / relative).open('rb') as copied:
                    copied_hash = hashlib.file_digest(copied, 'sha256').hexdigest()
                if original_hash != copied_hash:
                    raise OSError(f'Backup divergente: {relative}')
                files.append(dict(path=relative.as_posix(), sha256=copied_hash,
                                  size_bytes=path.stat().st_size))
        (staging / 'manifest.json').write_text(json.dumps(dict(
            session_id=session_id, created_at_utc=datetime.now(timezone.utc).isoformat(),
            layers=list(sources), files=files), indent=2), encoding='utf-8')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
