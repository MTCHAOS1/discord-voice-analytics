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

Depois de clonar o repositório, você precisa fornecer suas próprias gravações.
Coloque os arquivos na pasta `data/raw/<sessao>/`, dentro da raiz deste projeto.
Crie uma pasta exclusiva para cada gravação; o nome dessa pasta será o valor
usado em `--session`. Por exemplo, para `--session session_001`:

```bash
mkdir -p data/raw/session_001
```

Extraia as faixas individuais do ZIP do Craig para essa pasta.
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
Mantenha os nomes das faixas: o sistema usa o nome do arquivo sem a extensão e
sem o prefixo numérico para identificar o participante. Por exemplo,
`1-participante_a.flac` identifica `participante_a`.

### Onde fica cada arquivo

Todos os caminhos abaixo são relativos à raiz do projeto. Use o mesmo
identificador de sessão em todas as etapas.

| Pasta ou arquivo | Conteúdo e como preencher |
| --- | --- |
| `data/raw/<sessao>/` | Entrada manual: faixas de áudio extraídas e `info.txt` do Craig, quando disponível. |
| `data/processed/<sessao>/` | Gerado por `transcribe`: `segments.parquet`, `segments.csv`, `transcricao.txt` e `metadata.json`. |
| `data/curated/<sessao>/` | Gerado por `analyze`: `speaker_metrics.csv` e `top_words.csv`. |
| `data/quality/<sessao>/` | Gerado por `validate`: `validation_report.json`. |
| `data/backups/<sessao>/` | Gerado automaticamente antes de reprocessar uma sessão que já possui resultados. |
| `reports/` | Destino sugerido para os relatórios HTML exportados com `main.py dashboard --output`. |
| `.env` | Configuração local opcional de MinIO/PostgreSQL; copie de `.env.example` e preencha suas credenciais. |

As pastas de saída são criadas pelo pipeline. Para começar com seus áudios,
preencha apenas `data/raw/<sessao>/` e execute:

```bash
.venv/bin/python main.py inspect --session session_001
.venv/bin/python main.py process --session session_001
```

### Por que os dados não estão no Git?

As gravações e os resultados reais ficaram fora do Git de propósito: podem
conter vozes, apelidos, conversas privadas e informações pessoais. Áudios e
arquivos compactados também podem ser grandes. Credenciais e configurações
locais não devem ser compartilhadas junto com o código.

O `.gitignore` exclui o conteúdo de `data/`, os relatórios em `reports/`, os
arquivos de áudio, exportações CSV/Parquet, arquivos compactados e `.env`.
Somente os arquivos vazios `.gitkeep` das pastas `raw`, `processed`, `curated`
e `quality` são versionados para preservar a estrutura inicial.

Por isso, `git clone` e `git pull` não baixam as sessões reais, e `git push`
não envia esses arquivos locais ignorados. Cada pessoa deve obter os áudios
por um meio privado autorizado e processá-los localmente. Sem uma sessão
processada, use a opção **Demonstração** do dashboard para conhecer a interface.
Não use `git add -f` para incluir dados privados; o Git não substitui o backup
local das gravações e dos resultados.

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

## Ritmo de fala e silêncio

O `transcribe` agora pede ao Whisper o tempo de cada palavra e grava
`data/processed/<sessao>/words.parquet`. A página **Ritmo e silêncio** calcula
palavras por minuto (palavras ditas ÷ fala medida) e o silêncio da gravação. A
fala medida une os intervalos de cada pessoa, e pausas de até 0,5 s contam como
fala. Sobreposições são contadas uma vez por pessoa.

Limitações: são estimativas. Só entram as faixas presentes na pasta, então
participantes ausentes aparecem como silêncio. O Whisper pode inventar texto em
silêncios longos, o que reduz o silêncio medido. Sessões antigas não têm
`words.parquet`: o dashboard usa os trechos do Whisper (que podem incluir
silêncios) e avisa; rode `transcribe` de novo para medir por palavra.

## Timelapse da nuvem

A página **Timelapse da nuvem** mostra a nuvem de palavras crescendo ao longo de
uma gravação (intervalos de 1 a 30 minutos) ou dia a dia. As posições são fixas:
a mesma palavra fica no mesmo lugar em todos os quadros. O botão "Gerar GIF"
cria um arquivo para download; o GIF contém palavras da conversa e deve ser
compartilhado só com autorização. O HTML exportado não inclui tempos por palavra.

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
