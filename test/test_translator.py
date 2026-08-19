import unittest
import json
import subprocess
from pathlib import Path
from string import Template
from textwrap import dedent
from unittest import mock

from ollama import ResponseError as OllamaResponseError

from pdf2zh import cache
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    BaseTranslator,
    CodexTranslator,
    OllamaTranslator,
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


class TestCodexTranslator(unittest.TestCase):
    def setUp(self):
        self.test_db = cache.init_test_db()
        ConfigManager.clear()

    def tearDown(self):
        cache.clean_test_db(self.test_db)
        ConfigManager.clear()

    @staticmethod
    def _write_translation_payload(cmd, translation):
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"translation": translation}, f)

    @staticmethod
    def _write_batch_translation_payload(cmd, translations):
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"translations": translations}, f)

    @staticmethod
    def _probe_side_effect(extra=None):
        extra = extra or []

        def side_effect(cmd, **kwargs):
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "codex 0.122.0\n", "")
            if cmd[1:] == ["exec", "--help"]:
                help_text = "\n".join(
                    [
                        "--skip-git-repo-check",
                        "--ephemeral",
                        "--sandbox",
                        "--color",
                        "--output-schema",
                        "--output-last-message",
                        "--ignore-user-config",
                        "--ignore-rules",
                        "--model",
                        "--profile",
                        "--config",
                    ]
                )
                return subprocess.CompletedProcess(cmd, 0, help_text, "")
            if extra:
                return extra.pop(0)(cmd, **kwargs)
            raise AssertionError(f"Unexpected command: {cmd}")

        return side_effect

    def test_do_translate_builds_default_codex_command(self):
        cwd_capture = {}

        def fake_run(cmd, **kwargs):
            cwd_capture["cwd"] = kwargs["cwd"]
            self.assertNotEqual(kwargs["cwd"], ".")
            self.assertTrue("--ephemeral" in cmd)
            self.assertTrue("--skip-git-repo-check" in cmd)
            self.assertTrue("--sandbox" in cmd)
            self.assertTrue("--output-schema" in cmd)
            self.assertTrue("--output-last-message" in cmd)
            self._write_translation_payload(cmd, "你好，世界")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=self._probe_side_effect([fake_run]),
        ) as run:
            translator = CodexTranslator(lang_in="en", lang_out="zh", model=None)
            translated = translator.do_translate("Hello, world")

        self.assertEqual("你好，世界", translated)
        cmd = run.call_args.args[0]
        self.assertEqual("codex", cmd[0])
        self.assertEqual(["exec"], cmd[1:2])
        self.assertNotEqual(cwd_capture["cwd"], None)
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn("--ignore-rules", cmd)
        self.assertIn("-c", cmd)
        self.assertIn('model_reasoning_effort="none"', cmd)
        self.assertIn("gpt-5.4-mini", cmd)

    def test_do_translate_honors_profile_model_and_timeout(self):
        envs = {
            "CODEX_BIN": "/usr/local/bin/codex",
            "CODEX_PROFILE": "translate",
            "CODEX_MODEL": "gpt-5.4-mini",
            "CODEX_TIMEOUT": "37",
        }

        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_translation_payload(cmd, "测试"),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1]
                ]
            )
            translator = CodexTranslator("en", "zh", None, envs=envs)
            translated = translator.do_translate("test")

        self.assertEqual("测试", translated)
        cmd = run.call_args.args[0]
        self.assertEqual("/usr/local/bin/codex", cmd[0])
        self.assertNotIn("--ignore-user-config", cmd)
        self.assertIn("--profile", cmd)
        self.assertIn("translate", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("gpt-5.4-mini", cmd)
        self.assertEqual(run.call_args.kwargs["timeout"], 37)

    def test_init_fails_when_required_codex_flags_missing(self):
        def side_effect(cmd, **kwargs):
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "codex 0.122.0\n", "")
            if cmd[1:] == ["exec", "--help"]:
                return subprocess.CompletedProcess(cmd, 0, "--model\n", "")
            raise AssertionError(f"Unexpected command: {cmd}")

        with mock.patch("pdf2zh.translator.subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as context:
                CodexTranslator("en", "zh", None)

        self.assertIn("--output-schema", str(context.exception))

    def test_do_translate_falls_back_to_compat_path_on_unsupported_flag(self):
        run_calls = []

        def compat_run(cmd, **kwargs):
            run_calls.append(cmd)
            if "--ignore-user-config" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 2, "", "error: unexpected argument '--ignore-user-config'"
                )
            self._write_translation_payload(cmd, "兼容路径")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=self._probe_side_effect([compat_run, compat_run]),
        ):
            translator = CodexTranslator("en", "zh", None)
            translated = translator.do_translate("test")

        self.assertEqual("兼容路径", translated)
        self.assertEqual(2, len(run_calls))
        self.assertIn("--ignore-user-config", run_calls[0])
        self.assertNotIn("--ignore-user-config", run_calls[1])

    def test_translate_batch_returns_cached_and_batched_values(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["甲", "乙"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1]
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            translator.cache.set("cached", "已缓存")
            translated = translator.translate_batch(["cached", "a", "{v1}", "b", " "])

        self.assertEqual(["已缓存", "甲", "{v1}", "乙", " "], translated)
        cmd = run.call_args.args[0]
        self.assertIn("--output-schema", cmd)
        self.assertEqual("甲", translator.cache.get("a"))
        self.assertEqual("乙", translator.cache.get("b"))

    def test_translate_batch_splits_and_retries_on_mismatched_length(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["only-one"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["甲"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["乙"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            translated = translator.translate_batch(["a", "b"])

        self.assertEqual(["甲", "乙"], translated)

    def test_translate_batch_splits_and_retries_on_timeout(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (_ for _ in ()).throw(
                        subprocess.TimeoutExpired(cmd, 120)
                    ),
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["甲"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, ["乙"]),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            translated = translator.translate_batch(["a", "b"])

        self.assertEqual(["甲", "乙"], translated)

    def test_translate_batch_uses_single_translate_when_custom_prompt_present(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_translation_payload(cmd, "一"),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                    lambda cmd, **kwargs: (
                        self._write_translation_payload(cmd, "二"),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                ]
            )
            translator = CodexTranslator(
                "en", "zh", None, prompt=Template("Custom: ${text}")
            )
            translated = translator.translate_batch(["a", "b"])

        self.assertEqual(["一", "二"], translated)
        self.assertEqual(4, run.call_count)

    def test_translate_batch_splits_long_items_and_recombines(self):
        long_text = ("Sentence one. " * 120).strip()
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(
                            cmd, ["甲", "乙", "丙", "丁", "戊", "己"]
                        ),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1]
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            translated = translator.translate_batch([long_text])

        self.assertEqual(["甲乙丙丁戊己"], translated)

    def test_recombine_translated_segments_preserves_whitespace_boundaries(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)

        combined = translator._recombine_translated_segments(
            ["Hello ", "world"], ["你好", "世界"]
        )
        self.assertEqual("你好 世界", combined)

        combined_newline = translator._recombine_translated_segments(
            ["Hello\n", "world"], ["你好", "世界"]
        )
        self.assertEqual("你好\n世界", combined_newline)

        combined_tabs = translator._recombine_translated_segments(
            ["Hello\t\t", "world"], ["你好", "世界"]
        )
        self.assertEqual("你好\t\t世界", combined_tabs)

    def test_normalize_translation_output_removes_chinese_word_spaces(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)

        normalized = translator._normalize_translation_output(
            "本文 提出 了一种 复合 控制 策略 ， 用于 实现 轨迹 跟踪 任务 。"
        )
        self.assertEqual("本文提出了一种复合控制策略，用于实现轨迹跟踪任务。", normalized)

    def test_normalize_translation_output_preserves_english_boundary(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)

        normalized = translator._normalize_translation_output(
            "MPC 跟踪 控制 对象"
        )
        self.assertEqual("MPC 跟踪控制对象", normalized)

    def test_normalize_translation_output_removes_placeholder_spaces(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)

        normalized = translator._normalize_translation_output(
            "代价 函数 {v1} 最小 化 ， 约束 {v2} 满足 。"
        )
        self.assertEqual("代价函数{v1}最小化，约束{v2}满足。", normalized)

    def test_capability_probe_caches_fast_path_support(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)

        self.assertTrue(translator.fast_command_available)
        self.assertTrue(translator.compat_command_available)
        self.assertEqual("fast", translator.preferred_command_mode)
        self.assertIn("--output-schema", translator.supported_exec_flags)

    def test_init_raises_when_codex_binary_missing(self):
        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=FileNotFoundError("codex not found"),
        ):
            with self.assertRaises(RuntimeError) as context:
                CodexTranslator("en", "zh", None)

        self.assertIn("codex executable not found", str(context.exception))

    def test_do_translate_raises_on_non_zero_exit(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_translation_payload(cmd, "ignored"),
                        subprocess.CompletedProcess(cmd, 1, "", "boom"),
                    )[1]
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            with self.assertRaises(RuntimeError) as context:
                translator.do_translate("Hello")

        self.assertIn("boom", str(context.exception))

    def test_do_translate_raises_on_invalid_json(self):
        def fake_run(cmd, **kwargs):
            output_path = cmd[cmd.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("not-json")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=self._probe_side_effect([fake_run]),
        ):
            translator = CodexTranslator("en", "zh", None)
            with self.assertRaises(RuntimeError) as context:
                translator.do_translate("Hello")

        self.assertIn("invalid JSON", str(context.exception))

    def test_do_translate_raises_on_missing_translation_field(self):
        def fake_run(cmd, **kwargs):
            output_path = cmd[cmd.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"message": "missing"}, f)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=self._probe_side_effect([fake_run]),
        ):
            translator = CodexTranslator("en", "zh", None)
            with self.assertRaises(RuntimeError) as context:
                translator.do_translate("Hello")

        self.assertIn("translation", str(context.exception))

    def test_do_translate_raises_on_missing_batch_field(self):
        def fake_run(cmd, **kwargs):
            output_path = cmd[cmd.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch(
            "pdf2zh.translator.subprocess.run",
            side_effect=self._probe_side_effect(
                [fake_run, fake_run, fake_run, fake_run, fake_run]
            ),
        ):
            translator = CodexTranslator("en", "zh", None)
            with self.assertRaises(RuntimeError) as context:
                translator.translate_batch(["Hello", "World"])

        self.assertIn("translation", str(context.exception))

    def test_single_item_batch_falls_back_to_single_translate(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect(
                [
                    lambda cmd, **kwargs: (
                        self._write_batch_translation_payload(cmd, []),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1],
                    lambda cmd, **kwargs: (
                        self._write_translation_payload(cmd, "单条回退"),
                        subprocess.CompletedProcess(cmd, 0, "", ""),
                    )[1]
                ]
            )
            translator = CodexTranslator("en", "zh", None)
            translated = translator.translate_batch(["a"])

        self.assertEqual(["单条回退"], translated)

    def test_codex_uses_llm_placeholders(self):
        with mock.patch("pdf2zh.translator.subprocess.run") as run:
            run.side_effect = self._probe_side_effect()
            translator = CodexTranslator("en", "zh", None)
        self.assertEqual("{{v3}}", translator.get_formular_placeholder(3))
        self.assertEqual("{{v3}}", translator.get_rich_text_left_placeholder(3))
        self.assertEqual("{{v4}}", translator.get_rich_text_right_placeholder(3))

    def test_gui_service_map_includes_codex(self):
        gui_source = Path("pdf2zh/gui.py").read_text(encoding="utf-8")
        self.assertIn('"Codex": CodexTranslator', gui_source)


if __name__ == "__main__":
    unittest.main()
