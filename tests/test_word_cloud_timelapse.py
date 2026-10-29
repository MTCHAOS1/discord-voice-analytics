"""Nuvem com layout fixo e timelapse, com contagens sintéticas."""
from io import BytesIO
import math
import unittest

import pandas as pd
from PIL import Image, ImageChops, ImageDraw

from src.dashboard.analytics import timelapse_counts
from src.dashboard.word_cloud import (caption_box, draw_word_cloud, layout_word_cloud, participant_colors,
                                      render_timelapse_gif, render_word_cloud)


def frame(rows):
    return pd.DataFrame(rows, columns=['speaker', 'start_s', 'text'])


class TestTimelapse(unittest.TestCase):
    def setUp(self):
        self.frame = frame([('A', 10, 'equipe mapa equipe'), ('B', 400, 'mapa jogo'),
                            ('A', 700, 'equipe vitória'), ('B', 1250, 'estratégia mapa')])

    def test_number_of_frames_and_cumulative_counts(self):
        frames = timelapse_counts(self.frame, 300)
        self.assertEqual(len(frames), math.ceil(1250 / 300))
        self.assertEqual([end for end, _ in frames], [300, 600, 900, 1200, 1500])
        for (_, before), (_, after) in zip(frames, frames[1:]):
            self.assertTrue(all(after.get(key, 0) >= count for key, count in before.items()))
        self.assertEqual(frames[0][1], {('A', 'equipe'): 2, ('A', 'mapa'): 1})
        self.assertEqual(frames[-1][1][('A', 'equipe')], 3)

    def test_word_set_is_fixed_by_limit_and_speaker_filter(self):
        frames = timelapse_counts(self.frame, 300, limit=2)
        keys = {key for _, counts in frames for key in counts}
        self.assertEqual(keys, {('A', 'equipe'), ('B', 'mapa')})
        self.assertEqual(len(frames[-1][1]), 2)
        only = timelapse_counts(self.frame, 300, speaker='B')
        self.assertTrue(all(key[0] == 'B' for _, counts in only for key in counts))
        self.assertEqual(timelapse_counts(frame([]), 300), [])

    def test_same_word_keeps_same_box_in_every_frame(self):
        frames = timelapse_counts(self.frame, 300)
        layout = layout_word_cloud(frames[-1][1])
        boxes = {key: item['box'] for key, item in layout['items'].items()}
        self.assertTrue(boxes)
        for _, counts in frames:
            image = draw_word_cloud(layout, counts)
            self.assertEqual(image.size, (1200, 650))
        self.assertEqual({key: item['box'] for key, item in layout['items'].items()}, boxes)
        for item in layout['items'].values():
            self.assertTrue(0 <= item['box'][0] and item['box'][2] <= 1200)

    def test_empty_frame_does_not_break(self):
        layout = layout_word_cloud({('A', 'equipe'): 3})
        image = draw_word_cloud(layout, {}, caption='00h00 de 00h20')
        self.assertGreater(len(image.getcolors(maxcolors=1000000)), 1)
        self.assertEqual(draw_word_cloud(layout_word_cloud({}), {}).size, (1200, 650))

    def test_gif_is_valid_with_one_image_per_frame(self):
        frames = timelapse_counts(self.frame, 300)
        layout = layout_word_cloud(frames[-1][1])
        colors = participant_colors(['A', 'B'])
        data = render_timelapse_gif(layout, frames, colors, 1250)
        image = Image.open(BytesIO(data))
        self.assertEqual(image.format, 'GIF')
        self.assertEqual(image.n_frames, len(frames))
        labeled = render_timelapse_gif(layout, [(1, frames[0][1]), (2, frames[-1][1])], colors, 2,
                                       labels=['26/10/2026', '27/10/2026'])
        self.assertEqual(Image.open(BytesIO(labeled)).n_frames, 2)

    def test_render_word_cloud_still_generates_png(self):
        colors = participant_colors(['A', 'B'])
        for frequencies in ({('A', 'equipe'): 16, ('B', 'equipe'): 10}, {'equipe': 4, 'mapa': 2}):
            image = Image.open(BytesIO(render_word_cloud(frequencies, colors)))
            self.assertEqual(image.format, 'PNG')
        self.assertEqual(Image.open(BytesIO(render_word_cloud({'equipe': 4}))).size, (1200, 650))


class TestLayoutBoxes(unittest.TestCase):
    def test_drawn_word_stays_inside_its_reserved_box(self):
        final = {('A', 'equipe'): 16, ('B', 'mapa'): 9, ('A', 'estratégia'): 4}
        layout = layout_word_cloud(final)
        background = Image.new('RGB', (1200, 650), '#101827')
        for counts in (final, {key: 1 for key in final}):
            for key, item in layout['items'].items():
                image = draw_word_cloud(layout, {key: counts[key]})
                drawn = ImageChops.difference(image.convert('RGB').crop((0, 0, 1200, 650)), background).getbbox()
                box = item['box']
                self.assertGreaterEqual(drawn[0], box[0])
                self.assertGreaterEqual(drawn[1], box[1])
                self.assertLessEqual(drawn[2], box[2])
                self.assertLessEqual(drawn[3], box[3])

    def test_caption_area_is_free_only_when_reserved(self):
        frequencies = {('A', f'palavra{index}'): 40 - index for index in range(40)}
        reserved = layout_word_cloud(frequencies, reserve_caption=True)
        area = caption_box(ImageDraw.Draw(Image.new('RGB', (1, 1))))
        for item in reserved['items'].values():
            box = item['box']
            self.assertFalse(box[0] < area[2] and box[2] > area[0] and box[1] < area[3] and box[3] > area[1])
        self.assertEqual(layout_word_cloud(frequencies), layout_word_cloud(frequencies, reserve_caption=False))


class TestGifTotalDuration(unittest.TestCase):
    def total_ms(self, days):
        frames = [(index, {('A', 'equipe'): index, ('A', f'palavra{index}'): 1}) for index in range(1, days + 1)]
        layout = layout_word_cloud(frames[-1][1])
        labels = [f'{index:02d}/10/2026' for index in range(1, days + 1)]
        image = Image.open(BytesIO(render_timelapse_gif(layout, frames, participant_colors(['A']), days,
                                                        labels=labels, total_ms=40000)))
        total = 0
        for index in range(image.n_frames):
            image.seek(index)
            total += image.info['duration']
        return total

    def test_daily_gif_lasts_forty_seconds(self):
        for days in (5, 12):
            self.assertAlmostEqual(self.total_ms(days), 40000, delta=5 * (days + 2))

    def test_minimum_frame_duration_may_exceed_total(self):
        from src.dashboard.word_cloud import frame_duration
        self.assertEqual(frame_duration(3000, 40000), 20)
        self.assertEqual(frame_duration(8, 40000), 4000)
