#!/usr/bin/env bash
# Execute com: bash scripts/process_gpu.sh session_004
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Uso: bash scripts/process_gpu.sh ID_DA_SESSAO' >&2
    exit 2
fi

python_bin="$project_dir/.venv/bin/python"
cuda_library_dirs="$("$python_bin" - <<'PY'
from pathlib import Path
import sysconfig

base = Path(sysconfig.get_paths()['purelib']) / 'nvidia'
libraries = [base / 'cublas' / 'lib', base / 'cudnn' / 'lib']
if not all(path.is_dir() for path in libraries):
    raise SystemExit('Instale as bibliotecas: .venv/bin/python -m pip install -r requirements-gpu.txt')
libraries.extend(path for path in sorted(base.glob('*/lib')) if path not in libraries)
print(':'.join(map(str, libraries)))
PY
)"
export LD_LIBRARY_PATH="$cuda_library_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$python_bin" - <<'PY'
import ctypes
import ctranslate2

if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit('GPU NVIDIA indisponivel. Verifique nvidia-smi no terminal WSL antes de processar.')
for library in ('libcublas.so.12', 'libcudnn.so.9'):
    ctypes.CDLL(library)
supported = ctranslate2.get_supported_compute_types('cuda')
if 'float16' not in supported:
    raise SystemExit(f'GPU sem suporte a float16. Tipos disponiveis: {sorted(supported)}')
print('CUDA disponivel; bibliotecas carregadas e float16 suportado.', flush=True)
PY

exec "$python_bin" main.py process --session "$1" --device cuda
