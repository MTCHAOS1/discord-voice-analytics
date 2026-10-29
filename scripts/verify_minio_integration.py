"""Valida MinIO local usando exclusivamente um arquivo sintetico isolado."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sync_data_lake import sync_session


def verify(client, bucket: str) -> dict:
    """Nao remove objetos; cada execucao usa uma chave unica de teste."""
    session = f'synthetic_minio_{uuid4().hex}'
    key = f'bronze/{session}/integration.txt'
    payload = b'Discord Voice Analytics: synthetic MinIO integration fixture.\n'
    digest = sha256(payload).hexdigest()
    # Exigir bucket existente evita criacao acidental de infraestrutura.
    if not client.bucket_exists(bucket):
        raise RuntimeError('O bucket configurado precisa existir antes do teste.')
    with TemporaryDirectory(prefix='dva-minio-') as temp:
        root = Path(temp)
        fixture = root / 'data/raw' / session / 'integration.txt'
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(payload)
        first = sync_session(session, apply=True, base_dir=root)
        info = client.stat_object(bucket, key)
        metadata = {str(k).lower(): v for k, v in (info.metadata or {}).items()}
        if metadata.get('x-amz-meta-sha256') != digest or info.size != len(payload):
            raise RuntimeError('SHA-256 ou tamanho remoto divergente.')
        # Ler somente a chave sintetica criada por esta execucao.
        response = client.get_object(bucket, key)
        try:
            remote_digest = sha256(response.read()).hexdigest()
        finally:
            response.close()
            response.release_conn()
        if remote_digest != digest:
            raise RuntimeError('SHA-256 do conteudo remoto divergente.')
        second = sync_session(session, apply=True, base_dir=root)
        if first != dict(planned=1, uploaded=1, skipped=0, dry_run=False):
            raise RuntimeError('A primeira sincronizacao deveria enviar um arquivo.')
        if second != dict(planned=1, uploaded=0, skipped=1, dry_run=False):
            raise RuntimeError('A segunda sincronizacao deveria ignorar o arquivo.')
        if fixture.read_bytes() != payload:
            raise RuntimeError('O arquivo sintetico local foi alterado.')
    return dict(status='PASS', bucket=bucket, object_name=key, sha256=digest,
                remote_sha256=remote_digest, first_sync=first, second_sync=second)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--consent-confirmed', action='store_true')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports/minio_integration.json')
    args = parser.parse_args()
    if not (args.apply and args.consent_confirmed):
        parser.error('O upload sintetico requer --apply --consent-confirmed.')

    from dotenv import load_dotenv
    from minio import Minio
    from urllib3 import PoolManager
    from urllib3.util import Retry, Timeout

    load_dotenv(PROJECT_ROOT / '.env')
    endpoint = os.environ.get('MINIO_ENDPOINT', '127.0.0.1:9000')
    parsed = urlsplit('//' + endpoint)
    if (parsed.hostname not in ('localhost', '127.0.0.1', '::1')
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        parser.error('Este teste exige endpoint MinIO local, sem esquema nem credenciais na URL.')
    access = os.environ.get('MINIO_ACCESS_KEY', '')
    secret = os.environ.get('MINIO_SECRET_KEY', '')
    if not access or not secret or 'COLOQUE_' in access or 'COLOQUE_' in secret:
        parser.error('Configure credenciais MinIO validas no ambiente ou .env.')
    bucket = os.environ.get('MINIO_BUCKET', 'discord-voice-analytics')
    client = Minio(
        endpoint, access_key=access, secret_key=secret,
        secure=os.environ.get('MINIO_SECURE', 'false').lower() in ('1', 'true', 'yes'),
        http_client=PoolManager(timeout=Timeout(connect=3, read=10), retries=Retry(total=0)),
    )
    result = verify(client, bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: SHA-256 remoto e idempotencia confirmados. Relatorio: {args.output}')
    print('O objeto sintetico foi preservado no bucket; nenhuma exclusao foi realizada.')


if __name__ == '__main__':
    main()
