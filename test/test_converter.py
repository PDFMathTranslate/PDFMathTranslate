import importlib
import typing
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import openai
from pdfminer.layout import LTPage, LTChar, LTLine
from pdfminer.pdfinterp import PDFResourceManager
from tenacity import RetryError

from pdf2zh import cache
from pdf2zh.config import ConfigManager
from pdf2zh.converter import PDFConverterEx, TranslateConverter
from pdf2zh.translator import OpenAITranslator


class TestPDFConverterEx(unittest.TestCase):
    def setUp(self):
        self.rsrcmgr = PDFResourceManager()
        self.converter = PDFConverterEx(self.rsrcmgr)

    def test_begin_page(self):
        mock_page = Mock()
        mock_page.pageno = 1
        mock_page.cropbox = (0, 0, 100, 200)
        mock_ctm = [1, 0, 0, 1, 0, 0]
        self.converter.begin_page(mock_page, mock_ctm)
        self.assertIsNotNone(self.converter.cur_item)
        self.assertEqual(self.converter.cur_item.pageid, 1)

    def test_render_char(self):
        mock_matrix = (1, 2, 3, 4, 5, 6)
        mock_font = Mock()
        mock_font.to_unichr.return_value = "A"
        mock_font.char_width.return_value = 10
        mock_font.char_disp.return_value = (0, 0)
        graphic_state = Mock()
        self.converter.cur_item = Mock()
        result = self.converter.render_char(
            mock_matrix,
            mock_font,
            fontsize=12,
            scaling=1.0,
            rise=0,
            cid=65,
            ncs=None,
            graphicstate=graphic_state,
        )
        self.assertEqual(result, 120.0)  # Expected text width


class TestTranslateConverter(unittest.TestCase):
    def setUp(self):
        self.rsrcmgr = PDFResourceManager()
        self.layout = {1: Mock()}
        self.translator_class = Mock()
        self.converter = TranslateConverter(
            self.rsrcmgr,
            layout=self.layout,
            lang_in="en",
            lang_out="zh",
            service="google",
        )

    def test_translator_initialization(self):
        self.assertIsNotNone(self.converter.translator)
        self.assertEqual(self.converter.translator.lang_in, "en")
        self.assertEqual(self.converter.translator.lang_out, "zh-CN")

    @patch("pdf2zh.converter.TranslateConverter.receive_layout")
    def test_receive_layout(self, mock_receive_layout):
        mock_page = LTPage(1, (0, 0, 100, 200))
        mock_font = Mock()
        mock_font.fontname.return_value = "mock_font"
        mock_page.add(
            LTChar(
                matrix=(1, 2, 3, 4, 5, 6),
                font=mock_font,
                fontsize=12,
                scaling=1.0,
                rise=0,
                text="A",
                textwidth=10,
                textdisp=(1.0, 1.0),
                ncs=Mock(),
                graphicstate=Mock(),
            )
        )
        self.converter.receive_layout(mock_page)
        mock_receive_layout.assert_called_once_with(mock_page)

    def test_receive_layout_with_complex_formula(self):
        ltpage = LTPage(1, (0, 0, 500, 500))
        ltchar = Mock()
        ltchar.fontname.return_value = "mock_font"
        ltline = LTLine(0.1, (0, 0), (10, 20))
        ltpage.add(ltchar)
        ltpage.add(ltline)
        mock_layout = MagicMock()
        mock_layout.shape = (100, 100)
        mock_layout.__getitem__.return_value = -1
        self.converter.layout = [None, mock_layout]
        self.converter.thread = 1
        result = self.converter.receive_layout(ltpage)
        self.assertIsNotNone(result)

    def test_invalid_translation_service(self):
        with self.assertRaises(ValueError):
            TranslateConverter(
                self.rsrcmgr,
                layout=self.layout,
                lang_in="en",
                lang_out="zh",
                service="InvalidService",
            )


# Synthetic endpoint only. Tests mock the SDK call and never connect.
_OPENAI_BASE_URL = "https://example.invalid/v1"
_OPENAI_ENVS = {
    "OPENAI_BASE_URL": _OPENAI_BASE_URL,
    "OPENAI_API_KEY": "test-key",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_STREAM": "false",
    "OPENAI_STOP_TOKENS": "",
    "OPENAI_MAX_TOKENS": "-1",
}
_ATTEMPT_BUDGET = 100


class _AttemptOverflow(BaseException):
    """Fail-fast sentinel once call 101 starts.

    The pre-fix paragraph worker retries every Exception without a stop
    condition. A BaseException still aborts that loop, so an exhausted-budget
    regression fails instead of hanging.
    """


def _http_for_openai_errors():
    """HTTP client module that matches the installed OpenAI SDK response type."""
    response_type = typing.get_type_hints(openai.APIStatusError.__init__)["response"]
    candidates = [
        arg for arg in typing.get_args(response_type) if arg is not type(None)
    ]
    if candidates:
        response_type = candidates[0]
    return importlib.import_module(response_type.__module__.split(".")[0])


def _rate_limit_error(message):
    body = {
        "error": {
            "message": message,
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    http = _http_for_openai_errors()
    request = http.Request("POST", _OPENAI_BASE_URL + "/chat/completions")
    response = http.Response(429, request=request, json=body)
    return openai.RateLimitError(message, response=response, body=body)


def _completion(text):
    message = Mock()
    message.content = text
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]
    return response


def _char(text, x, y, fontname="Helvetica"):
    font = Mock()
    font.fontname = fontname
    font.is_vertical.return_value = False
    font.get_descent.return_value = 0.0
    return LTChar(
        matrix=(1, 0, 0, 1, x, y),
        font=font,
        fontsize=12,
        scaling=1.0,
        rise=0,
        text=text,
        textwidth=0.5,
        textdisp=0,
        ncs=Mock(),
        graphicstate=Mock(),
    )


class TestReceiveLayoutOpenAIRetry(unittest.TestCase):
    def setUp(self):
        self.test_db = cache.init_test_db()
        self._config_patches = (
            patch.object(ConfigManager, "get_translator_by_name", return_value=None),
            patch.object(ConfigManager, "set_translator_by_name"),
        )
        for patcher in self._config_patches:
            patcher.start()
        self.rsrcmgr = PDFResourceManager()
        self.converter = TranslateConverter(
            self.rsrcmgr,
            layout={1: np.ones((300, 300), dtype=int)},
            lang_in="en",
            lang_out="zh",
            service="openai:gpt-4o-mini",
            thread=1,
            envs=dict(_OPENAI_ENVS),
            ignore_cache=False,
        )
        self.assertIsInstance(self.converter.translator, OpenAITranslator)
        self.converter.thread = 1
        latin = Mock()
        latin.to_unichr.side_effect = chr
        latin.char_width.return_value = 0.5
        self.converter.fontmap = {"tiro": latin}
        self.converter.fontid = {}
        self.converter.noto_name = "Noto"
        noto = Mock()
        noto.has_glyph.side_effect = lambda codepoint: codepoint
        noto.char_lengths.return_value = (6.0,)
        self.converter.noto = noto

    def tearDown(self):
        for patcher in reversed(self._config_patches):
            patcher.stop()
        cache.clean_test_db(self.test_db)

    def _page(self, items):
        page = LTPage(1, (0, 0, 300, 300))
        layout = np.ones((300, 300), dtype=int)
        x = 10.0
        for text, y, cls, formula in items:
            char = _char(text, x, y)
            if formula:
                char.cid = ord(text)
                char.font = Mock()
                self.converter.fontid[char.font] = "tiro"
            page.add(char)
            layout[int(char.y0), int(char.x0)] = cls
            x = char.x1
        self.converter.layout = {1: layout}
        return page

    def _text_page(self, text):
        return self._page([(char, 20.0, 1, False) for char in text])

    @patch("tenacity.nap.time.sleep", return_value=None)
    def test_receive_layout_stops_after_openai_rate_limit_budget(self, sleep):
        attempts = OpenAITranslator.do_translate.retry.stop.max_attempt_number
        self.assertEqual(attempts, _ATTEMPT_BUDGET)
        errors = []

        def create(*args, **kwargs):
            attempt = len(errors) + 1
            if attempt > attempts:
                raise _AttemptOverflow(attempt)
            error = _rate_limit_error(f"rate limit {attempt}")
            errors.append(error)
            raise error

        page = self._text_page("Hello")
        output = None
        with patch.object(
            self.converter.translator.client.chat.completions,
            "create",
            side_effect=create,
        ):
            with self.assertRaises(openai.RateLimitError) as caught:
                output = self.converter.receive_layout(page)

        self.assertIsNone(output)
        self.assertEqual(len(errors), attempts)
        self.assertIs(caught.exception, errors[-1])
        self.assertEqual(caught.exception.status_code, 429)
        self.assertNotIsInstance(caught.exception, RetryError)
        self.assertEqual(sleep.call_count, attempts - 1)

    @patch("tenacity.nap.time.sleep", return_value=None)
    def test_receive_layout_retries_ordinary_exception(self, sleep):
        calls = {"n": 0}

        def create(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("temporary upstream failure")
            return _completion("OK")

        page = self._text_page("Hello")
        with patch.object(
            self.converter.translator.client.chat.completions,
            "create",
            side_effect=create,
        ):
            result = self.converter.receive_layout(page)

        self.assertEqual(calls["n"], 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(sleep.call_args.args[0], 1)
        self.assertIn("4f4b", result)

    @patch("tenacity.nap.time.sleep", return_value=None)
    def test_receive_layout_skips_blank_and_formula_paragraphs(self, sleep):
        create = Mock(return_value=_completion("SHOULD_NOT_TRANSLATE"))
        page = self._page(
            [
                (" ", 40.0, 3, False),
                ("α", 80.0, 2, True),
            ]
        )
        with patch.object(
            self.converter.translator.client.chat.completions,
            "create",
            create,
        ):
            result = self.converter.receive_layout(page)

        create.assert_not_called()
        sleep.assert_not_called()
        self.assertNotIn("SHOULD_NOT_TRANSLATE", result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
