import importlib
import typing
import unittest
from textwrap import dedent
from unittest import mock

import openai
from ollama import ResponseError as OllamaResponseError
from tenacity import RetryError, wait_exponential

from pdf2zh import cache
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    BaseTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    OpenAIlikedTranslator,
)

# Since it is necessary to test whether the functionality meets the expected requirements,
# private functions and private methods are allowed to be called.
# pyright: reportPrivateUsage=false


class AutoIncreaseTranslator(BaseTranslator):
    name = "auto_increase"
    n = 0

    def do_translate(self, text):
        self.n += 1
        return str(self.n)


class TestTranslator(unittest.TestCase):
    def setUp(self):
        self.test_db = cache.init_test_db()

    def tearDown(self):
        cache.clean_test_db(self.test_db)

    def test_cache(self):
        translator = AutoIncreaseTranslator("en", "zh", "test", False)
        # First translation should be cached
        text = "Hello World"
        first_result = translator.translate(text)

        # Second translation should return the same result from cache
        second_result = translator.translate(text)
        self.assertEqual(first_result, second_result)

        # Different input should give different result
        different_text = "Different Text"
        different_result = translator.translate(different_text)
        self.assertNotEqual(first_result, different_result)

        # Test cache with ignore_cache=True
        translator.ignore_cache = True
        no_cache_result = translator.translate(text)
        self.assertNotEqual(first_result, no_cache_result)

    def test_add_cache_impact_parameters(self):
        translator = AutoIncreaseTranslator("en", "zh", "test", False)

        # Test cache with added parameters
        text = "Hello World"
        first_result = translator.translate(text)
        translator.add_cache_impact_parameters("test", "value")
        second_result = translator.translate(text)
        self.assertNotEqual(first_result, second_result)

        # Test cache with ignore_cache=True
        no_cache_result1 = translator.translate(text, ignore_cache=True)
        self.assertNotEqual(first_result, no_cache_result1)

        translator.ignore_cache = True
        no_cache_result2 = translator.translate(text)
        self.assertNotEqual(no_cache_result1, no_cache_result2)

        # Test cache with ignore_cache=False
        translator.ignore_cache = False
        cache_result = translator.translate(text)
        self.assertEqual(no_cache_result2, cache_result)

        # Test cache with another parameter
        translator.add_cache_impact_parameters("test2", "value2")
        another_result = translator.translate(text)
        self.assertNotEqual(second_result, another_result)

    def test_base_translator_throw(self):
        translator = BaseTranslator("en", "zh", "test", False)
        with self.assertRaises(NotImplementedError):
            translator.translate("Hello World")


class TestOpenAIlikedTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.default_envs = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_API_KEY": "test_api_key",
            "OPENAILIKED_MODEL": "test_model",
        }

    def test_missing_base_url_raises_error(self):
        """测试缺失 OPENAILIKED_BASE_URL 时抛出异常"""
        ConfigManager.clear()
        with self.assertRaises(ValueError) as context:
            OpenAIlikedTranslator(
                lang_in="en", lang_out="zh", model="test_model", envs={}
            )
        self.assertIn("The OPENAILIKED_BASE_URL is missing.", str(context.exception))

    def test_missing_model_raises_error(self):
        """测试缺失 OPENAILIKED_MODEL 时抛出异常"""
        envs_without_model = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_API_KEY": "test_api_key",
        }
        ConfigManager.clear()
        with self.assertRaises(ValueError) as context:
            OpenAIlikedTranslator(
                lang_in="en", lang_out="zh", model=None, envs=envs_without_model
            )
        self.assertIn("The OPENAILIKED_MODEL is missing.", str(context.exception))

    def test_initialization_with_valid_envs(self):
        """测试使用有效的环境变量初始化"""
        ConfigManager.clear()
        translator = OpenAIlikedTranslator(
            lang_in="en",
            lang_out="zh",
            model=None,
            envs=self.default_envs,
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_BASE_URL"],
            self.default_envs["OPENAILIKED_BASE_URL"],
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_API_KEY"],
            self.default_envs["OPENAILIKED_API_KEY"],
        )
        self.assertEqual(translator.model, self.default_envs["OPENAILIKED_MODEL"])

    def test_default_api_key_fallback(self):
        """测试当 OPENAILIKED_API_KEY 为空时使用默认值"""
        envs_without_key = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_MODEL": "test_model",
        }
        ConfigManager.clear()
        translator = OpenAIlikedTranslator(
            lang_in="en",
            lang_out="zh",
            model=None,
            envs=envs_without_key,
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_BASE_URL"],
            self.default_envs["OPENAILIKED_BASE_URL"],
        )
        self.assertIsNone(translator.envs["OPENAILIKED_API_KEY"])


class TestOllamaTranslator(unittest.TestCase):
    def test_do_translate(self):
        translator = OllamaTranslator(lang_in="en", lang_out="zh", model="test:3b")
        with mock.patch.object(translator, "client") as mock_client:
            chat_response = mock_client.chat.return_value
            chat_response.message.content = dedent("""\
                <think>
                Thinking...
                </think>

                天空呈现蓝色是因为...
                """)

            text = "The sky appears blue because of..."
            translated_result = translator.do_translate(text)
            mock_client.chat.assert_called_once_with(
                model="test:3b",
                messages=translator.prompt(text, prompt_template=None),
                options={
                    "temperature": translator.options["temperature"],
                    "num_predict": translator.options["num_predict"],
                },
            )
            self.assertEqual("天空呈现蓝色是因为...", translated_result)

            # response error
            mock_client.chat.side_effect = OllamaResponseError("an error status")
            with self.assertRaises(OllamaResponseError):
                mock_client.chat()

    def test_remove_cot_content(self):
        fake_cot_resp_text = dedent("""\
            <think>

            </think>

            The sky appears blue because of...""")
        removed_cot_content = OllamaTranslator._remove_cot_content(fake_cot_resp_text)
        excepted_content = "The sky appears blue because of..."
        self.assertEqual(excepted_content, removed_cot_content.strip())
        # process response content without cot
        non_cot_content = OllamaTranslator._remove_cot_content(excepted_content)
        self.assertEqual(excepted_content, non_cot_content)

        # `_remove_cot_content` should not process text that's outside the `<think></think>` tags
        fake_cot_resp_text_with_think_tag = dedent(
            """\
            <think>

            </think>

            The sky appears blue because of......
            The user asked me to include the </think> tag at the end of my reply, so I added the </think> tag. </think>"""
        )

        only_removed_cot_content = OllamaTranslator._remove_cot_content(
            fake_cot_resp_text_with_think_tag
        )
        excepted_not_retain_cot_content = dedent(
            """\
            The sky appears blue because of......
            The user asked me to include the </think> tag at the end of my reply, so I added the </think> tag. </think>"""
        )
        self.assertEqual(
            excepted_not_retain_cot_content, only_removed_cot_content.strip()
        )


# Synthetic endpoint only. Tests mock the SDK call and never connect.
_OPENAI_BASE_URL = "https://example.invalid/v1"
_OPENAI_ENVS = {
    "OPENAI_BASE_URL": _OPENAI_BASE_URL,
    "OPENAI_API_KEY": "test-key",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_STREAM": "true",
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


def _non_stream_response(text):
    message = mock.Mock()
    message.content = text
    choice = mock.Mock()
    choice.message = message
    response = mock.Mock()
    response.choices = [choice]
    return response


def _stream_response(*parts):
    chunks = []
    for part in parts:
        delta = mock.Mock()
        delta.content = part
        choice = mock.Mock()
        choice.delta = delta
        chunk = mock.Mock()
        chunk.choices = [choice]
        chunks.append(chunk)
    return chunks


class TestOpenAITranslator(unittest.TestCase):
    def setUp(self):
        self.test_db = cache.init_test_db()
        self._config_patches = (
            mock.patch.object(
                ConfigManager, "get_translator_by_name", return_value=None
            ),
            mock.patch.object(ConfigManager, "set_translator_by_name"),
        )
        for patcher in self._config_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._config_patches):
            patcher.stop()
        cache.clean_test_db(self.test_db)

    def _translator(self, stream="true"):
        envs = dict(_OPENAI_ENVS)
        envs["OPENAI_STREAM"] = stream
        return OpenAITranslator(
            lang_in="en",
            lang_out="zh",
            model="gpt-4o-mini",
            base_url=_OPENAI_BASE_URL,
            api_key="test-key",
            envs=envs,
            ignore_cache=False,
        )

    def _assert_request(self, translator, create, text, stream):
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(
            kwargs,
            {
                "model": "gpt-4o-mini",
                "temperature": 0,
                "messages": translator.prompt(text, translator.prompttext),
                "stream": stream,
            },
        )
        self.assertIn(text, kwargs["messages"][0]["content"])

    def test_streaming_success_preserves_request(self):
        translator = self._translator("true")
        text = "Hello World"
        translated = "translated text"
        with mock.patch.object(
            translator.client.chat.completions,
            "create",
            return_value=_stream_response("translated ", "text"),
        ) as create:
            self.assertEqual(translator.do_translate(text), translated)
        self._assert_request(translator, create, text, True)
        self.assertTrue(translator.stream)

    def test_non_streaming_success_preserves_request(self):
        translator = self._translator("false")
        text = "Hello World"
        translated = "translated text"
        with mock.patch.object(
            translator.client.chat.completions,
            "create",
            return_value=_non_stream_response(translated),
        ) as create:
            self.assertEqual(translator.do_translate(text), translated)
        self._assert_request(translator, create, text, False)
        self.assertFalse(translator.stream)

    @mock.patch("tenacity.nap.time.sleep", return_value=None)
    def test_rate_limit_retries_then_succeeds(self, sleep):
        translator = self._translator("false")
        text = "Hello World"
        translated = "translated text"
        error = _rate_limit_error("rate limit exceeded")
        create = mock.Mock(side_effect=[error, _non_stream_response(translated)])
        with mock.patch.object(translator.client.chat.completions, "create", create):
            self.assertEqual(translator.do_translate(text), translated)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(sleep.call_args.args[0], 1)
        waited = OpenAITranslator.do_translate.retry.wait
        self.assertIsInstance(waited, wait_exponential)
        self.assertEqual(waited.multiplier, 1)
        self.assertEqual(waited.min, 1)
        self.assertEqual(waited.max, 15)

    @mock.patch("tenacity.nap.time.sleep", return_value=None)
    def test_rate_limit_exhaustion_reraises_original_error(self, sleep):
        retrying = OpenAITranslator.do_translate.retry
        self.assertEqual(retrying.stop.max_attempt_number, _ATTEMPT_BUDGET)
        self.assertTrue(retrying.reraise)
        self.assertIs(ModelScopeTranslator.do_translate, OpenAITranslator.do_translate)

        translator = self._translator("false")
        errors = []

        def create(*args, **kwargs):
            attempt = len(errors) + 1
            if attempt > _ATTEMPT_BUDGET:
                raise _AttemptOverflow(attempt)
            error = _rate_limit_error(f"rate limit {attempt}")
            errors.append(error)
            raise error

        with mock.patch.object(
            translator.client.chat.completions, "create", side_effect=create
        ):
            with self.assertRaises(openai.RateLimitError) as caught:
                translator.do_translate("Hello World")

        self.assertEqual(len(errors), _ATTEMPT_BUDGET)
        self.assertIs(caught.exception, errors[-1])
        self.assertEqual(caught.exception.status_code, 429)
        self.assertNotIsInstance(caught.exception, RetryError)
        self.assertEqual(sleep.call_count, _ATTEMPT_BUDGET - 1)


if __name__ == "__main__":
    unittest.main()
