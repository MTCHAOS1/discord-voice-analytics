# Discord Voice Analytics

Pipeline local para transcrever faixas individuais do Craig (Discord), calcular
estatísticas por participante e explorar os resultados em um dashboard Streamlit.
O repositório contém código e exemplos sintéticos; gravações, transcrições,
relatórios e credenciais devem permanecer no ambiente local.

## Instalação

Recomendado: Python 3.12 no Linux/WSL2. Instale também `ffmpeg`/`ffprobe` para
medir a duração dos áudios. Execute os comandos na raiz do projeto:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

A primeira transcrição precisa de internet para baixar o modelo Whisper.
A transcrição é executada localmente, sem envio dos áudios a uma API.

## Preparar e processar uma sessão

Extraia as faixas individuais da gravação para uma pasta exclusiva por sessão.
Os nomes abaixo são fictícios:

```text
data/raw/session_001/
  1-participante_a.flac
  2-participante_b.flac
  info.txt
```

Preserve o `info.txt` do Craig, se disponível: ele permite identificar o início
da gravação. Não coloque somente o ZIP na pasta; extraia os áudios primeiro.
São reconhecidos FLAC, WAV, OGG, OGA, MP3, M4A, MP4, AAC, OPUS e WebM.

```bash
.venv/bin/python main.py inspect --session session_001
.venv/bin/python main.py process --session session_001
```

`process` executa transcrição, análise e validação. O padrão é `large-v3` na
CPU com int8; escolha outro modelo com `--model small`, por exemplo.
Os resultados ficam em `data/processed/`, `data/curated/` e `data/quality/`.

- `PASS`: resultados prontos para exibição.
- `WARN`: revise `data/quality/<sessao>/validation_report.json` e confirme os alertas no dashboard.
- `FAIL`: exibição bloqueada; corrija os erros antes de continuar.

Falhas de processamento e `FAIL` retornam código 1. `WARN` retorna código 0.
O pipeline não é transacional: arquivos de etapas concluídas podem permanecer
no disco após uma falha. Evite processar a mesma sessão em paralelo ou consultar
seus resultados durante o reprocessamento.

Também é possível executar as etapas separadamente:

```bash
.venv/bin/python main.py transcribe --session session_001
.venv/bin/python main.py analyze --session session_001
.venv/bin/python main.py validate --session session_001
```

## GPU NVIDIA no Linux/WSL

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt
nvidia-smi
bash scripts/process_gpu.sh session_001
```

O script configura os caminhos das bibliotecas para o processo, verifica CUDA e
suporte a float16 e executa o pipeline com `--device cuda`. Se a GPU estiver
indisponível, interrompe antes de transcrever. Os comandos Python diretos
continuam usando CPU por padrão.

São necessárias cuBLAS para CUDA 12 e cuDNN 9, conforme a
[documentação do Faster-Whisper](https://github.com/SYSTRAN/faster-whisper#gpu).
No WSL, o driver NVIDIA do Windows também precisa disponibilizar a GPU.
Instalar os pacotes Python não instala o driver nem garante acesso ao dispositivo.

## Dashboard

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

Abra http://127.0.0.1:8501. Para instalar apenas o dashboard, use
`requirements-dashboard.txt` em vez de `requirements.txt`.

A opção **Sessão local** abre em **Todos os dias**, agregando as sessões
processadas e validadas. Atualize a página depois de processar uma nova sessão.
O filtro **Dia da gravação** usa o início da gravação no fuso America/Sao_Paulo;
sessões sem timestamp aparecem em **Sem data identificada**. A opção
**Demonstração** usa somente dados sintéticos com nomes fictícios.

Estão disponíveis participação, ranking e nuvem de palavras, dicionário pessoal,
evolução diária, comparação de participantes, palavras características, busca
com contexto, expressões frequentes, linha do tempo, resumo extrativo e
sobreposições de fala. Os filtros de pessoa mantêm o total do período como
denominador da participação. Apelidos iguais são agrupados entre sessões.

O código não contém um cadastro pessoal de apelidos. Os identificadores exibidos
nas sessões locais vêm dos próprios dados; isso não anonimiza gravações ou
resultados. As páginas de contexto exibem trechos de transcrição e os downloads
podem conter informações pessoais. Não publique esses arquivos.

Durações e sobreposições são estimadas pelos timestamps do Whisper. Elas não
comprovam interrupções, intenção ou mudança semântica de assunto. O resumo é
extrativo, e os termos recorrentes não substituem revisão da conversa.

Para gerar uma demonstração HTML independente:

```bash
.venv/bin/python main.py dashboard --demo --output reports/demo.html
```

Para uma sessão local, substitua `--demo` por `--session session_001`.
O HTML não inclui áudio nem frases completas da transcrição, mas pode conter
nomes e palavras privadas. O comando não sobrescreve um arquivo já existente.

## Timestamps e backups

A transcrição limita o fim de intervalos válidos à duração medida pelo ffprobe.
Os ajustes são registrados em `metadata.json` e exigem revisão (`WARN`).
Intervalos inválidos ou inteiramente fora do áudio geram `FAIL`; duração
indisponível gera `WARN`. Nenhum texto é descartado automaticamente.

Antes de reprocessar, `transcribe` e `process` preservam os resultados anteriores
em `data/backups/<sessao>/<data-UTC>_<id>/`, com manifesto e hashes SHA-256.
Uma falha no backup interrompe a transcrição. Não há restauração automática
após falhas posteriores; comandos isolados `analyze` e `validate` não criam backup.
Os áudios originais não são copiados para o snapshot.

## Integrações opcionais

O dashboard funciona sem MinIO ou PostgreSQL. Para configurar as integrações,
copie `.env.example` para `.env` e preencha as credenciais localmente.

```bash
# Simular sincronização com MinIO
.venv/bin/python main.py sync-lake --session session_001

# Enviar somente após autorização dos participantes
.venv/bin/python main.py sync-lake --session session_001 --apply --consent-confirmed

# Simular carga no PostgreSQL
.venv/bin/python main.py load-warehouse --session session_001

# Aplicar carga da sessão
.venv/bin/python main.py load-warehouse --session session_001 --apply
```

O schema fica em `sql/warehouse/001_init.sql`, com consultas de exemplo em
`sql/analytics/first_queries.sql`. Resultados `WARN` precisam de revisão antes
de usar `--allow-warnings` na carga. A sincronização e a carga são ações
explícitas e não fazem parte do comando `process`.

## Testes

```bash
.venv/bin/python -m unittest discover -s tests -v
node --test tests/test_dashboard_ui.cjs
```

Os testes usam dados sintéticos. Para incluir PostgreSQL temporário, instale
os binários PostgreSQL e execute a suíte com `RUN_TEMP_POSTGRES_TESTS=1`.
Isso exige permissão para iniciar um servidor local e criar sockets.
A integração real com MinIO pode ser verificada separadamente com
`scripts/verify_minio_integration.py`; testes simulados não comprovam acesso ao serviço.

## Antes de publicar

O `.gitignore` exclui dados locais, relatórios, backups, arquivos de áudio,
exportações, arquivos compactados, credenciais e configurações locais do agente.
Somente os marcadores `.gitkeep` das quatro pastas de dados são versionáveis.
Mantenha as credenciais reais fora de `.env.example`.

Revise `git status --short` e `git diff --cached` antes de confirmar um commit.
Regras de exclusão não removem arquivos já rastreados nem limpam o histórico de
um repositório existente. Não use `git add -f` em arquivos privados.
