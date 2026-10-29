"""Nuvem determinística com frequências locais, sem serviços externos."""
from functools import lru_cache
from io import BytesIO
import math
import colorsys

from PIL import Image, ImageDraw, ImageFont


def participant_colors(speakers):
    """Cores estáveis por nome dentro da sessão, inclusive ao filtrar."""
    return {speaker: '#%02x%02x%02x' % tuple(round(c * 255) for c in
            colorsys.hsv_to_rgb((index * 0.61803398875 + 0.4) % 1, 0.48, 0.95))
            for index, speaker in enumerate(sorted(set(speakers)))}


WIDTH, HEIGHT = 1200, 650
PALETTE = ['#69c9a0', '#ad9cf5', '#7dd3fc', '#fcd34d', '#f9a8d4']


@lru_cache(maxsize=None)
def load_font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def word_size(count, maximum):
    return round(18 + 72 * math.sqrt(count / maximum))


CAPTION_FONT_SIZE = 26


def caption_box(draw):
    """Canto superior direito onde o relógio ou a data é desenhado, com folga para o texto mais largo."""
    width = draw.textlength('00h00 de 00h00', font=ImageFont.load_default(size=CAPTION_FONT_SIZE))
    return (WIDTH - 20 - math.ceil(width), 14, WIDTH - 20, 14 + CAPTION_FONT_SIZE + 12)


def layout_word_cloud(frequencies: dict, reserve_caption: bool = False) -> dict:
    """Posições fixas calculadas com as contagens finais; os quadros só mudam o tamanho.

    Com `reserve_caption`, a área da legenda do relógio fica livre para não cobrir palavras.
    """
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    occupied = [caption_box(draw)] if reserve_caption else []
    items = {}
    terms = sorted(((word, count) for word, count in frequencies.items() if count > 0),
                   key=lambda item: (-item[1], item[0]))
    maximum = max((count for _, count in terms), default=1)
    for index, (word, count) in enumerate(terms):
        speaker, label = word if isinstance(word, tuple) else (None, word)
        size = word_size(count, maximum)
        while size >= 14:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=load_font(size))
            width, height = right - left, bottom - top
            for step in range(1800):
                angle = step * 0.22
                radius = 5 * math.sqrt(step)
                x = round(600 + radius * 2.5 * math.cos(angle) - width / 2)
                y = round(325 + radius * 1.4 * math.sin(angle) - height / 2)
                box = (x - 5, y - 5, x + width + 5, y + height + 5)
                if box[0] < 0 or box[1] < 0 or box[2] > WIDTH or box[3] > HEIGHT:
                    continue
                if any(box[0] < b[2] and box[2] > b[0] and
                       box[1] < b[3] and box[3] > b[1] for b in occupied):
                    continue
                occupied.append(box)
                items[word] = dict(speaker=speaker, label=label, size=size, box=box, index=index)
                break
            else:
                size -= 4
                continue
            break
    return dict(items=items, maximum=maximum)


def draw_word_cloud(layout: dict, counts: dict, speaker_colors: dict | None = None,
                    caption: str | None = None) -> Image.Image:
    """Desenha um quadro: tamanho pela contagem do quadro, posição pela caixa reservada."""
    speaker_colors = speaker_colors or {}
    legend_height = 48 * math.ceil(len(speaker_colors) / 3)
    canvas = Image.new('RGB', (WIDTH, HEIGHT + legend_height), '#101827')
    draw = ImageDraw.Draw(canvas)
    for key, item in layout['items'].items():
        count = counts.get(key, 0)
        if count <= 0:
            continue
        font = load_font(min(word_size(count, layout['maximum']), item['size']))
        left, top, right, bottom = draw.textbbox((0, 0), item['label'], font=font)
        box = item['box']
        x = (box[0] + box[2]) / 2 - (right - left) / 2 - left
        y = (box[1] + box[3]) / 2 - (bottom - top) / 2 - top
        draw.text((round(x), round(y)), item['label'], font=font,
                  fill=speaker_colors.get(item['speaker'], PALETTE[item['index'] % len(PALETTE)]))
    for index, (speaker, color) in enumerate(speaker_colors.items()):
        y = 660 + (index // 3) * 48
        x = 25 + (index % 3) * 395
        draw.rectangle((x, y, x + 20, y + 20), fill=color)
        draw.text((x + 32, y), speaker, font=ImageFont.load_default(size=24), fill=color)
    if caption:
        font = ImageFont.load_default(size=CAPTION_FONT_SIZE)
        draw.text((WIDTH - 20 - draw.textlength(caption, font=font), 14), caption, font=font, fill='#cbd8e8')
    return canvas


def render_word_cloud(frequencies: dict, speaker_colors: dict | None = None) -> bytes:
    """Renderiza termos por frequência em PNG, evitando sobreposição."""
    layout = layout_word_cloud(frequencies)
    output = BytesIO()
    draw_word_cloud(layout, frequencies, speaker_colors).save(output, format='PNG')
    return output.getvalue()


def clock(seconds):
    minutes = int(seconds // 60)
    return f'{minutes // 60:02d}h{minutes % 60:02d}'


MIN_FRAME_MS = 20


def frame_duration(frame_count: int, total_ms: int) -> int:
    """Duração de cada quadro para o GIF somar `total_ms` (o último vale 3 quadros); mínimo de 20 ms, múltiplo de 10 ms."""
    # O formato GIF guarda a duração em centésimos de segundo.
    return max(MIN_FRAME_MS, round(total_ms / (frame_count + 2) / 10) * 10)


def render_timelapse_gif(layout: dict, frames: list, speaker_colors: dict | None, total_seconds: float,
                         duration_ms: int = 600, labels: list | None = None, total_ms: int | None = None) -> bytes:
    """GIF com um quadro por item de `timelapse_counts`; o último dura o triplo.

    Com `total_ms`, a duração por quadro sai dele (ver `frame_duration`) e `duration_ms` é ignorado.
    """
    if total_ms is not None:
        duration_ms = frame_duration(len(frames), total_ms)
    def images():
        # Cada quadro vai para o modo P assim que é desenhado, sem manter os RGB na memória.
        for index, (end, counts) in enumerate(frames):
            caption = labels[index] if labels else f'{clock(min(end, total_seconds))} de {clock(total_seconds)}'
            yield draw_word_cloud(layout, counts, speaker_colors, caption).convert('P', palette=Image.Palette.ADAPTIVE)

    durations = [duration_ms] * len(frames)
    durations[-1] = duration_ms * 3
    sequence = images()
    first = next(sequence)
    output = BytesIO()
    first.save(output, format='GIF', save_all=True, append_images=sequence, loop=0, duration=durations)
    return output.getvalue()
