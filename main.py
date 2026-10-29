"""Discord Voice Analytics - pipeline local, verificacao e Data Lake."""
import argparse
import json
import re
from pathlib import Path
from time import monotonic


def print_process_summary(args, stages, started, report=None, metadata=None):
    print(f'\nResumo do processamento: {args.session}')
    print(f'Modelo: {args.model} | Dispositivo: {args.device} | Tempo: {monotonic() - started:.1f}s')
    for stage, status in stages.items():
        print(f'  {stage}: {status}')
    if metadata is not None:
        print('Backup: ' + (metadata.get('backup_path') or 'nao necessario (sem resultados anteriores).'))
    else:
        print('Backup: consulte a mensagem de backup da transcricao; resumo indisponivel.')
    if report is None:
        print('Dashboard: processamento incompleto; confira o erro antes de usar os resultados.')
        return
    summary = report.get('summary', {})
    print(f'Faixas: {summary.get("audio_tracks", "indisponivel")} | '
          f'Trechos: {summary.get("transcript_segments", "indisponivel")} | '
          f'Participantes com fala: {summary.get("participants_with_segments", "indisponivel")}')
    events = report.get('timestamp_events', [])
    print(f'Timestamps ajustados: {sum(e.get("action") == "clipped" for e in events)} | '
          f'Ocorrencias para revisao: {sum(e.get("action") == "review" for e in events)}')
    for label, messages in [('ERRO', report.get('errors', [])), ('AVISO', report.get('warnings', []))]:
        for message in messages:
            print(f'  {label}: {message}')
    print(f'Relatorio: data/quality/{args.session}/validation_report.json')
    print(f'Resultados: data/processed/{args.session}/ e data/curated/{args.session}/')
    if report['status'] == 'PASS':
        print('Dashboard: pronto. Atualize a pagina para ver todos os dias ou filtre o dia da gravacao.')
    elif report['status'] == 'WARN':
        print('Dashboard: revisao pendente. Revise os alertas e confirme o relatorio WARN na interface.')
    else:
        print('Dashboard: bloqueado. Corrija os erros e execute analyze e validate novamente.')


def main():
    parser = argparse.ArgumentParser(description='Discord Voice Analytics - MVP')
    parser.add_argument('action', choices=('process', 'inspect', 'transcribe', 'analyze', 'validate', 'sync-lake', 'load-warehouse', 'dashboard'))
    parser.add_argument('--session', default='session_001', help='Identificador da sessao, ex. session_001')
    parser.add_argument('--model', default='large-v3', help='Modelo Whisper (padrao: large-v3)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--apply', action='store_true', help='Envia ao MinIO. Sem isso: apenas simula.')
    parser.add_argument('--consent-confirmed', action='store_true', help='Confirma a autorizacao de todos para armazenar os dados.')
    parser.add_argument('--allow-warnings', action='store_true', help='Permite carregar uma sessao com relatorio WARN apos revisao.')
    parser.add_argument('--demo', action='store_true', help='Dashboard com dados ficticios, sem ler sessoes.')
    parser.add_argument('--output', type=Path, help='Novo arquivo .html para o dashboard (nao sobrescreve).')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session):
        parser.error('ID de sessao invalido: use apenas letras, numeros, hifens e underscore.')

    if args.action == 'process':
        from src.transcription.transcribe import transcribe
        from src.transformation.analyze import analyze
        from src.quality.validate_session import validate_session

        stage = 'transcricao'
        started = monotonic()
        stages = dict(transcricao='nao executada', analise='nao executada', validacao='nao executada')
        metadata = None
        try:
            print(f'[1/3] Transcrevendo {args.session} com {args.model} ({args.device})...', flush=True)
            output = transcribe(args.session, args.model, args.device)
            stages[stage] = 'concluida'
            if isinstance(output, Path):
                try:
                    metadata = json.loads(output.with_name('metadata.json').read_text(encoding='utf-8'))
                    if not isinstance(metadata, dict):
                        metadata = None
                except (OSError, ValueError):
                    metadata = None
            stage = 'analise'
            print('[2/3] Gerando metricas...', flush=True)
            analyze(args.session)
            stages[stage] = 'concluida'
            stage = 'validacao'
            print('[3/3] Validando resultados...', flush=True)
            report = validate_session(args.session)
            stages[stage] = report['status']
        except Exception as exc:
            stages[stage] = 'falhou'
            print_process_summary(args, stages, started, metadata=metadata)
            parser.exit(1, f'Processamento interrompido na etapa {stage}: {exc}\n')
        print_process_summary(args, stages, started, report, metadata)
        if report['status'] == 'FAIL':
            parser.exit(1, f'Sessao {args.session}: FAIL. Corrija os erros do relatorio '
                        f'data/quality/{args.session}/validation_report.json antes de abrir no Dashboard.\n')
        if report['status'] == 'WARN':
            print(f'Sessao {args.session}: WARN. Revise data/quality/{args.session}/validation_report.json '
                  'e confirme os alertas no Dashboard antes de exibir os dados.')
        else:
            print(f'Sessao {args.session}: PASS. Pronta para selecionar pelo dia no Dashboard.')
    elif args.action == 'inspect':
        from src.ingestion.inspect_files import inspect
        inspect(Path('data/raw') / args.session)
    elif args.action == 'transcribe':
        from src.transcription.transcribe import transcribe
        transcribe(args.session, args.model, args.device)
    elif args.action == 'analyze':
        from src.transformation.analyze import analyze
        analyze(args.session)
    elif args.action == 'validate':
        from src.quality.validate_session import validate_session
        report = validate_session(args.session)
        if report['status'] == 'FAIL':
            raise SystemExit(1)
    elif args.action == 'sync-lake':
        from src.ingestion.sync_data_lake import sync_session
        if args.apply and not args.consent_confirmed:
            parser.error('Para enviar os arquivos, utilize --apply --consent-confirmed apos obter consentimento.')
        sync_session(args.session, apply=args.apply)
    elif args.action == 'load-warehouse':
        from src.database.load_warehouse import load_session
        load_session(args.session, apply=args.apply, allow_warnings=args.allow_warnings)
    elif args.action == 'dashboard':
        from src.dashboard.report import demo_dashboard, load_dashboard, write_dashboard
        try:
            data = demo_dashboard() if args.demo else load_dashboard(
                args.session, allow_warnings=args.allow_warnings)
            output = args.output or Path('reports') / f'dashboard_{data["session_id"]}.html'
            write_dashboard(data, output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(1, f'Nao foi possivel gerar o dashboard: {exc}\n')
        print(f'Dashboard gerado: {output.resolve()}')
        print('Abra o arquivo no navegador. Nenhum envio ou conexao externa foi realizado.')


if __name__ == '__main__':
    main()
