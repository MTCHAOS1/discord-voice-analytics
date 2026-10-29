"""Local Craig recordings discovery. Doesn't touch or upload audio."""
from pathlib import Path
import re
import shutil
import subprocess

AUDIO_EXTENSIONS = {'.flac', '.wav', '.ogg', '.oga', '.mp3', '.m4a', '.mp4', '.aac', '.opus', '.webm'}


def get_speaker(filename: str) -> str:
    """Map Craig's '1-participante_a.flac' to 'participante_a'."""
    name = Path(filename).stem
    return re.sub(r'^\d+[-_]', '', name)


def audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Pasta de áudio não encontrada: {directory}")
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda p: p.name.lower()
    )


def audio_duration(path: Path) -> float | None:
    if not shutil.which('ffprobe'):
        return None
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        text=True, capture_output=True, check=False
    )
    try:
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def inspect(directory: Path):
    print(f'Pasta da sessão: {directory.resolve()}')
    if not directory.is_dir():
        raise FileNotFoundError('Crie a pasta e copie os arquivos exportados pelo Craig.')
    found = audio_files(directory)
    if not found:
        print('Nenhum arquivo de áudio reconhecido. Exiba as extensões no Windows ou execute: file *')
    for path in found:
        dur = audio_duration(path)
        duration_label = f'{dur:.1f}s' if dur is not None else 'não medida (instale ffmpeg)'
        print(f'  {path.name} -> participante={get_speaker(path.name)} | duração={duration_label} | {path.stat().st_size:,} bytes')
    other = sorted(p.name for p in directory.iterdir() if p.is_file() and p not in found and p.name != '.gitkeep')
    if other:
        print('Arquivos auxiliares ignorados:', ', '.join(other))
    return found
