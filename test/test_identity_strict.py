import unittest
from unittest.mock import Mock

from pdfminer.pdfinterp import PDFResourceManager

from pdf2zh.converter import MissingTranslationError, TranslateConverter
from pdf2zh.translator import IdentityTranslator, OpenAITranslator


class TestIdentityTranslator(unittest.TestCase):
    def test_returns_text_unchanged_and_uses_v_placeholders(self):
        translator = IdentityTranslator("en", "ja", None)
        self.assertEqual("Hello {v0}", translator.translate("Hello {v0}"))
        # Same template as OpenAITranslator; the converter formats it into {v3}.
        self.assertEqual(
            OpenAITranslator.get_formular_placeholder(translator, 3),
            translator.get_formular_placeholder(3),
        )
        self.assertEqual("identity", translator.name)


class TestStrictTranslationFile(unittest.TestCase):
    def setUp(self):
        self.rsrcmgr = PDFResourceManager()
        self.layout = {1: Mock()}

    def make(self, translation_map, strict):
        return TranslateConverter(
            self.rsrcmgr,
            layout=self.layout,
            lang_in="en",
            lang_out="ja",
            service="identity",
            translation_map=translation_map,
            strict_translation_file=strict,
        )

    def test_strict_requires_translation_map(self):
        with self.assertRaises(ValueError):
            self.make({}, strict=True)

    def test_strict_raises_on_missing_text(self):
        converter = self.make({"Known text.": "既知の文。"}, strict=True)
        converter.get_collected_texts = lambda: ["Known text.", "Unknown text."]
        converter._pending_pages = [object()]
        with self.assertRaises(MissingTranslationError) as raised:
            converter.flush_batch({})
        self.assertEqual(["Unknown text."], raised.exception.missing)

    def test_non_strict_falls_back_to_identity(self):
        converter = self.make({"Known text.": "既知の文。"}, strict=False)
        converter.get_collected_texts = lambda: ["Known text.", "Unknown text."]
        converter._pending_pages = [object()]
        converter._translate_and_typeset = lambda *a, **k: ""
        converter._pending_pages = []
        converter.flush_batch({})
        self.assertNotIn("Unknown text.", converter.translations)


if __name__ == "__main__":
    unittest.main()
