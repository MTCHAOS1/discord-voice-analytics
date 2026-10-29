"""Nuvem determinística com frequências locais, sem serviços externos."""
from io import BytesIO
import math
import colorsys

from PIL import Image, ImageDraw, ImageFont


def participant_colors(speakers):
    """Cores estáveis por nome dentro da sessão, inclusive ao filtrar."""
    return {speaker: '#%02x%02x%02x' % tuple(round(c * 255) for c in
            colorsys.hsv_to_rgb((index * 0.61803398875 + 0.4) % 1, 0.48, 0.95))
            for index, speaker in enumerate(sorted(set(speakers)))}


def render_word_cloud(frequencies: dict, speaker_colors: dict | None = None) -> bytes:
    """Renderiza termos por frequência em PNG, evitando sobreposição."""
    speaker_colors = speaker_colors or {}
    legend_height = 48 * math.ceil(len(speaker_colors) / 3)
    canvas = Image.new('RGB', (1200, 650 + legend_height), '#101827')
    draw = ImageDraw.Draw(canvas)
    occupied = []
    colors = ['#69c9a0', '#ad9cf5', '#7dd3fc', '#fcd34d', '#f9a8d4']
    terms = sorted(((word, count) for word, count in frequencies.items() if count > 0),
                   key=lambda item: (-item[1], item[0]))
    maximum = max((count for _, count in terms), default=1)
    for index, (word, count) in enumerate(terms):
        speaker, label = word if isinstance(word, tuple) else (None, word)
        size = round(18 + 72 * math.sqrt(count / maximum))
        while size >= 14:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', size)
            except OSError:
                font = ImageFont.load_default(size=size)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            width, height = right - left, bottom - top
            for step in range(1800):
                angle = step * 0.22
                radius = 5 * math.sqrt(step)
                x = round(600 + radius * 2.5 * math.cos(angle) - width / 2)
                y = round(325 + radius * 1.4 * math.sin(angle) - height / 2)
                box = (x - 5, y - 5, x + width + 5, y + height + 5)
                if box[0] < 0 or box[1] < 0 or box[2] > 1200 or box[3] > 650:
                    continue
                if any(box[0] < b[2] and box[2] > b[0] and
                       box[1] < b[3] and box[3] > b[1] for b in occupied):
                    continue
                draw.text((x - left, y - top), label, font=font,
                          fill=speaker_colors.get(speaker, colors[index % len(colors)]))
                occupied.append(box)
                break
            else:
                size -= 4
                continue
            break
    for index, (speaker, color) in enumerate(speaker_colors.items()):
        y = 660 + (index // 3) * 48
        x = 25 + (index % 3) * 395
        draw.rectangle((x, y, x + 20, y + 20), fill=color)
        draw.text((x + 32, y), speaker, font=ImageFont.load_default(size=24), fill=color)
    output = BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()
