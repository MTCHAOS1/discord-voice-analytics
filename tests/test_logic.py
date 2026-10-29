import unittest
from src.ingestion.inspect_files import get_speaker
from src.transformation.analyze import words, dictionary_words


class TestMvp(unittest.TestCase):
    def test_craig_filename(self):
        self.assertEqual(get_speaker('1-participante_a.flac'), 'participante_a')
        self.assertEqual(get_speaker('4-participante_e.m4a'), 'participante_e')

    def test_frequencies(self):
        self.assertEqual(words('Mano, mano! Eu não quero.'), ['mano', 'mano', 'quero'])

    def test_dictionary_keeps_short_common_and_accented_words(self):
        self.assertEqual(dictionary_words('Eu, eu! A NÃO ação ac\u0327a\u0303o 123.'),
                         ['eu', 'eu', 'a', 'não', 'ação', 'ação'])


if __name__ == '__main__':
    unittest.main()
