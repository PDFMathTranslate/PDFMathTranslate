import unittest
from textwrap import dedent
from unittest import mock

from ollama import ResponseError as OllamaResponseError

from pdf2zh import cache
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    BaseTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    RivaTranslator,
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
            chat_response.message.content = dedent(
                """\
                <think>
                Thinking...
                </think>
                    
                天空呈现蓝色是因为...
                """
            )

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
        fake_cot_resp_text = dedent(
            """\
            <think>

            </think>

            The sky appears blue because of..."""
        )
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


class TestRivaTranslator(unittest.TestCase):
    def setUp(self):
        self.mock_auth_cls = mock.Mock(name="Auth")
        self.mock_client_cls = mock.Mock(name="Client")
        self.mock_auth_instance = self.mock_auth_cls.return_value
        self.mock_client_instance = self.mock_client_cls.return_value

    def test_missing_dependency_raises(self):
        with mock.patch(
            "pdf2zh.translator._load_riva_client", side_effect=ImportError("missing")
        ):
            with self.assertRaises(ImportError):
                RivaTranslator(
                    lang_in="en",
                    lang_out="ja",
                    model=None,
                    envs={"RIVA_ENDPOINT": "lab:50051", "RIVA_MODEL": "model"},
                    ignore_cache=True,
                )

    def test_translate_invokes_riva_client(self):
        with mock.patch(
            "pdf2zh.translator._load_riva_client",
            return_value=(self.mock_auth_cls, self.mock_client_cls),
        ):
            self.mock_client_instance.translate.return_value = mock.Mock(
                texts=["こんにちは"]
            )
            translator = RivaTranslator(
                lang_in="en",
                lang_out="ja",
                model=None,
                envs={"RIVA_ENDPOINT": "lab:50051", "RIVA_MODEL": "model"},
            )
            translator.cache = mock.Mock()
            translator.cache.get.return_value = None
            result = translator.translate("Hello world")
            self.mock_auth_cls.assert_called_once_with(uri="lab:50051")
            self.mock_client_cls.assert_called_once_with(self.mock_auth_instance)
            self.mock_client_instance.translate.assert_called_once_with(
                texts=["Hello world"],
                model="model",
                source_language="en-US",
                target_language="ja-JP",
            )
            self.assertEqual("こんにちは", result)

    def test_translate_connection_failure(self):
        with mock.patch(
            "pdf2zh.translator._load_riva_client",
            return_value=(self.mock_auth_cls, self.mock_client_cls),
        ):
            self.mock_client_instance.translate.side_effect = RuntimeError("boom")
            translator = RivaTranslator(
                lang_in="en",
                lang_out="ja",
                model=None,
                envs={"RIVA_ENDPOINT": "lab:50051", "RIVA_MODEL": "model"},
            )
            translator.cache = mock.Mock()
            translator.cache.get.return_value = None
            with self.assertRaises(RuntimeError):
                translator.translate("Hello")

    def test_language_pair_warning(self):
        pair = mock.Mock(source_language="de-DE", target_language="en-US")
        with mock.patch(
            "pdf2zh.translator._load_riva_client",
            return_value=(self.mock_auth_cls, self.mock_client_cls),
        ):
            self.mock_client_instance.get_config.return_value = mock.Mock(
                language_pairs=[pair]
            )
            with self.assertLogs("pdf2zh.translator", level="WARNING") as logs:
                RivaTranslator(
                    lang_in="en",
                    lang_out="ja",
                    model=None,
                    envs={"RIVA_ENDPOINT": "lab:50051", "RIVA_MODEL": "model"},
                )
            self.assertTrue(any("does not advertise" in msg for msg in logs.output))


if __name__ == "__main__":
    unittest.main()
