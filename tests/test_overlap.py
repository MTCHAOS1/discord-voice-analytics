import unittest

import pandas as pd

from src.dashboard.overlap import overlap_events, overlap_ranking


def frame(*rows):
    return pd.DataFrame([dict(session_id=session, speaker=speaker, start_s=start,
                              end_s=end, text=text) for session, speaker, start, end, text in rows])


class TestOverlap(unittest.TestCase):
    def test_direction_and_union_avoid_double_counting_multiple_targets(self):
        data = frame(('s', 'A', 0, 10, 'mapa jogo equipe'),
                     ('s', 'B', 2, 9, 'mapa jogo equipe'),
                     ('s', 'C', 3, 8, 'mapa jogo equipe'))
        events = overlap_events(data)
        ranking = overlap_ranking(data, events).set_index('speaker')
        self.assertEqual(len(events), 3)
        self.assertEqual(ranking.loc['C', 'entries'], 1)
        self.assertEqual(ranking.loc['C', 'overlap_seconds'], 5)
        self.assertEqual(ranking.loc['A', 'entries'], 0)
        self.assertEqual(ranking.loc['A', 'received'], 2)
        self.assertEqual(ranking.loc['C', 'per_100_entries'], 100)
        self.assertTrue((events.lexical_similarity == 100).all())

    def test_simultaneous_is_not_attributed_as_an_interruption(self):
        data = frame(('s', 'A', 0, 10, 'mapa jogo equipe'), ('s', 'B', 0, 7, 'mapa jogo equipe'))
        events = overlap_events(data)
        self.assertEqual(events.kind.tolist(), ['Início simultâneo'])
        self.assertEqual(overlap_ranking(data, events).entries.sum(), 0)

    def test_adjacent_same_person_segments_are_one_entry(self):
        data = frame(('s', 'A', 0, 10, 'mapa jogo equipe'),
                     ('s', 'B', 2, 4, 'equipe jogo mapa'),
                     ('s', 'B', 4, 8, 'equipe jogo mapa'))
        events = overlap_events(data)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].overlap_seconds, 6)

    def test_thresholds_and_session_boundaries(self):
        data = frame(('s1', 'A', 0, 10, 'mapa jogo equipe'),
                     ('s1', 'B', 9.5, 10.5, 'sim'),
                     ('s1', 'C', 10.5, 12, 'frase seguinte'),
                     ('s2', 'D', 1, 10, 'frase outra sessao'))
        self.assertTrue(overlap_events(data).empty)
        events = overlap_events(data, minimum_overlap=0.1, minimum_entry=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].target, 'A')
        self.assertEqual(events.iloc[0].speaker, 'B')
        self.assertEqual(events.iloc[0].topic_signal, 'Vocabulário insuficiente para comparar')

    def test_lexical_signal_preserves_both_source_texts(self):
        data = frame(('s', 'A', 0, 10, 'mapa jogo equipe'),
                     ('s', 'B', 3, 8, 'carro viagem estrada'))
        event = overlap_events(data).iloc[0]
        self.assertEqual(event.lexical_similarity, 0)
        self.assertEqual(event.topic_signal, 'Baixa semelhança lexical — revisar contexto')
        self.assertEqual(event.ongoing_text, 'mapa jogo equipe')
        self.assertEqual(event.text, 'carro viagem estrada')
        self.assertEqual(event.ongoing_start_s, 0)

    def test_no_overlaps_still_ranks_each_person_with_zero(self):
        data = frame(('s', 'A', 0, 1, 'sim'), ('s', 'B', 1, 2, 'também'))
        ranking = overlap_ranking(data, overlap_events(data))
        self.assertEqual(len(ranking), 2)
        self.assertEqual(ranking.entries.sum(), 0)
