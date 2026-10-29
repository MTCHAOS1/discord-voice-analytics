"""Sincroniza arquivos locais para MinIO via S3, somente com --apply.

Mantem arquivos originais intactos. Nao realiza chamadas a rede no modo de simulacao.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import mimetypes
import os

LAYERS = {
    'bronze': Path('data/raw'),
    'silver': Path('data/processed'),
    'gold': Path('data/curated'),
    'quality': Path('data/quality')
}


def list_session_objects(session_id: str, base_dir: Path = Path('.')) -> list[tuple[Path, str]]:
    result = []
    for layer, relative_root in LAYERS.items():
        session_dir = Path(base_dir) / relative_root / session_id
        if not session_dir.exists():
            continue
        for path in sorted(session_dir.rglob('*')):
            if path.is_file() and not path.is_symlink():
                subpath = path.relative_to(session_dir).as_posix()
                result.append((path, f'{layer}/{session_id}/{subpath}'))
    return result


def file_sha256(path: Path) -> str:
    checksum = sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def sync_session(session_id: str, apply: bool = False, base_dir: Path = Path('.')) -> dict:
    objects = list_session_objects(session_id, base_dir)
    if not objects:
        raise FileNotFoundError('Nenhum arquivo para sincronizar. Confira a sessao e as pastas data/.')
    total = sum(path.stat().st_size for path, _ in objects)
    print(f'{"ENVIO" if apply else "SIMULACAO"} session={session_id} | '
          f'{len(objects)} arquivos | {total / 1024**2:.1f} MiB')
    for path, object_name in objects:
        print(f'  {path} -> {object_name}')
    if not apply:
        print('Nenhum arquivo foi enviado. Confira a lista antes de usar --apply --consent-confirmed.')
        return {'planned': len(objects), 'uploaded': 0, 'skipped': 0, 'dry_run': True}

    from dotenv import load_dotenv
    from minio import Minio
    from minio.error import S3Error
    load_dotenv(Path(base_dir) / '.env')
    endpoint = os.environ.get('MINIO_ENDPOINT', '127.0.0.1:9000')
    access_key = os.environ.get('MINIO_ACCESS_KEY')
    secret_key = os.environ.get('MINIO_SECRET_KEY')
    bucket = os.environ.get('MINIO_BUCKET', 'discord-voice-analytics')
    secure = os.environ.get('MINIO_SECURE', 'false').lower() in ('1', 'true', 'yes')
    if not access_key or not secret_key or 'COLOQUE_' in access_key or 'COLOQUE_' in secret_key:
        raise ValueError('Configure MINIO_ACCESS_KEY e MINIO_SECRET_KEY reais no arquivo .env.')
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f'Bucket criado: {bucket}')
    uploaded = 0
    skipped = 0
    for path, object_name in objects:
        fingerprint = file_sha256(path)
        try:
            previous = client.stat_object(bucket, object_name)
        except S3Error as exc:
            if exc.code in ('NoSuchKey', 'NoSuchObject', 'NotFound'):
                previous = None
            else:
                raise
        if previous is not None:
            metadata = {str(k).lower(): v for k, v in (previous.metadata or {}).items()}
            if previous.size == path.stat().st_size and metadata.get('x-amz-meta-sha256') == fingerprint:
                print(f'  SKIP {object_name} (mesmo SHA-256)')
                skipped += 1
                continue
        mime = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        if path.suffix == '.parquet':
            mime = 'application/vnd.apache.parquet'
        # MinIO 7.x recebe metadados via metadata e adiciona x-amz-meta- no S3.
        client.fput_object(
            bucket, object_name, str(path),
            content_type=mime, metadata={'sha256': fingerprint},
        )
        print(f'  OK   {object_name}')
        uploaded += 1
    print(f'Concluido: {uploaded} enviados, {skipped} ja existentes no bucket {bucket}.')
    return {'planned': len(objects), 'uploaded': uploaded, 'skipped': skipped, 'dry_run': False}
