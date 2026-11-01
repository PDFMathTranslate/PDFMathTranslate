"""Tests for spacing between CJK and Latin text."""

import unittest

from pdf2zh.text_spacing import add_cjk_latin_spacing


class TestAddCjkLatinSpacing(unittest.TestCase):
    def test_inserts_one_space_in_both_directions_for_supported_scripts(self):
        cases = (
            ("中文A", "中文 A"),
            ("A中文", "A 中文"),
            ("\U00020000A", "\U00020000 A"),
            ("A\U00020000", "A \U00020000"),
            ("かなA", "かな A"),
            ("Aかな", "A かな"),
            ("カナA", "カナ A"),
            ("Aカナ", "A カナ"),
            ("한글A", "한글 A"),
            ("A한글", "A 한글"),
            ("中文A中文", "中文 A 中文"),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(add_cjk_latin_spacing(text), expected)

    def test_supports_precomposed_and_decomposed_accented_latin(self):
        cases = (
            ("中文é", "中文 é"),
            ("é中文", "é 中文"),
            ("中文cafe\u0301中文", "中文 cafe\u0301 中文"),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(add_cjk_latin_spacing(text), expected)

    def test_preserves_existing_whitespace_and_unrelated_boundaries(self):
        cases = (
            "中文  A\nA\t中文",
            "中文,A",
            "A，中文",
            "中文1A",
            "A2中文",
            "中文Ж",
            "Ж中文",
            "中文Ω",
            "Ω中文",
            "中文{v0}A",
            "plain ASCII text",
            "中文文本",
            "",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(add_cjk_latin_spacing(text), text)

    def test_is_idempotent(self):
        cases = (
            "中文A",
            "A中文",
            "\U00020000cafe\u0301한글",
            "中文 cafe\u0301 中文",
            "中文{v0}A\nカナ B",
        )

        for text in cases:
            with self.subTest(text=text):
                spaced = add_cjk_latin_spacing(text)
                self.assertEqual(add_cjk_latin_spacing(spaced), spaced)


if __name__ == "__main__":
    unittest.main()
