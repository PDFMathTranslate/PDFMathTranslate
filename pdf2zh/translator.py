import html
import json
import logging
import os
import re
import subprocess
import tempfile
import unicodedata
from copy import copy
from string import Template
from typing import cast
import deepl
import ollama
import openai
import requests
import xinference_client
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from tencentcloud.common import credential
from tencentcloud.tmt.v20180321.models import (
    TextTranslateRequest,
    TextTranslateResponse,
)
from tencentcloud.tmt.v20180321.tmt_client import TmtClient

from pdf2zh.cache import TranslationCache
from pdf2zh.config import ConfigManager


from tenacity import retry, retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    name = "base"
    envs = {}
    lang_map: dict[str, str] = {}
    CustomPrompt = False

    def __init__(self, lang_in: str, lang_out: str, model: str, ignore_cache: bool):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model
        self.ignore_cache = ignore_cache

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        # Detach from self.__class__.envs
        # Cannot use self.envs = copy(self.__class__.envs)
        # because if set_envs called twice, the second call will override the first call
        self.envs = copy(self.envs)
        if ConfigManager.get_translator_by_name(self.name):
            self.envs = ConfigManager.get_translator_by_name(self.name)
        needUpdate = False
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
                needUpdate = True
        if needUpdate:
            ConfigManager.set_translator_by_name(self.name, self.envs)
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]
            ConfigManager.set_translator_by_name(self.name, self.envs)

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text: str, ignore_cache: bool = False) -> str:
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = self.do_translate(text)
        self.cache.set(text, translation)
        return translation

    def translate_batch(
        self, texts: list[str], ignore_cache: bool = False
    ) -> list[str]:
        return [self.translate(text, ignore_cache=ignore_cache) for text in texts]

    def do_translate(self, text: str) -> str:
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    def prompt(
        self, text: str, prompt_template: Template | None = None
    ) -> list[dict[str, str]]:
        try:
            return [
                {
                    "role": "user",
                    "content": cast(Template, prompt_template).safe_substitute(
                        {
                            "lang_in": self.lang_in,
                            "lang_out": self.lang_out,
                            "text": text,
                        }
                    ),
                }
            ]
        except AttributeError:  # `prompt_template` is None
            pass
        except Exception:
            logging.exception("Error parsing prompt, use the default prompt.")

        return [
            {
                "role": "user",
                "content": (
                    "You are a professional, authentic machine translation engine. "
                    "Only Output the translated text, do not include any other text."
                    "\n\n"
                    f"Translate the following markdown source text to {self.lang_out}. "
                    "Keep the formula notation {v*} unchanged. "
                    "Output translation directly without any additional text."
                    "\n\n"
                    f"Source Text: {text}"
                    "\n\n"
                    "Translated Text:"
                ),
            },
        ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, id: int):
        return f"<b{id}>"

    def get_rich_text_right_placeholder(self, id: int):
        return f"</b{id}>"

    def get_formular_placeholder(self, id: int):
        return self.get_rich_text_left_placeholder(
            id
        ) + self.get_rich_text_right_placeholder(id)


class GoogleTranslator(BaseTranslator):
    name = "google"
    lang_map = {"zh": "zh-CN"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"  # noqa: E501
        }

    def do_translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.endpoint,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<', response.text
        )
        if response.status_code == 400:
            result = "IRREPARABLE TRANSLATION ERROR"
        else:
            response.raise_for_status()
            result = html.unescape(re_result[0])
        return remove_control_characters(result)


class BingTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "bing"
    lang_map = {"zh": "zh-Hans"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://www.bing.com/translator"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",  # noqa: E501
        }

    def find_sid(self):
        response = self.session.get(self.endpoint)
        response.raise_for_status()
        url = response.url[:-10]
        ig = re.findall(r"\"ig\":\"(.*?)\"", response.text)[0]
        iid = re.findall(r"data-iid=\"(.*?)\"", response.text)[-1]
        key, token = re.findall(
            r"params_AbusePreventionHelper\s=\s\[(.*?),\"(.*?)\",", response.text
        )[0]
        return url, ig, iid, key, token

    def do_translate(self, text):
        text = text[:1000]  # bing translate max length
        url, ig, iid, key, token = self.find_sid()
        response = self.session.post(
            f"{url}ttranslatev3?IG={ig}&IID={iid}",
            data={
                "fromLang": self.lang_in,
                "to": self.lang_out,
                "text": text,
                "token": token,
                "key": key,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]


class DeepLTranslator(BaseTranslator):
    # https://github.com/DeepLcom/deepl-python
    name = "deepl"
    envs = {
        "DEEPL_AUTH_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        auth_key = self.envs["DEEPL_AUTH_KEY"]
        self.client = deepl.Translator(auth_key)

    def do_translate(self, text):
        response = self.client.translate_text(
            text, target_lang=self.lang_out, source_lang=self.lang_in
        )
        return response.text


class DeepLXTranslator(BaseTranslator):
    # https://deeplx.owo.network/endpoints/free.html
    name = "deeplx"
    envs = {
        "DEEPLX_ENDPOINT": "https://api.deepl.com/translate",
        "DEEPLX_ACCESS_TOKEN": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.endpoint = self.envs["DEEPLX_ENDPOINT"]
        self.session = requests.Session()
        auth_key = self.envs["DEEPLX_ACCESS_TOKEN"]
        if auth_key:
            self.endpoint = f"{self.endpoint}?token={auth_key}"

    def do_translate(self, text):
        response = self.session.post(
            self.endpoint,
            json={
                "source_lang": self.lang_in,
                "target_lang": self.lang_out,
                "text": text,
            },
            verify=False,  # noqa: S506
        )
        response.raise_for_status()
        return response.json()["data"]


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama-python
    name = "ollama"
    envs = {
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "gemma2",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        model: str,
        envs=None,
        prompt: Template | None = None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OLLAMA_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {
            "temperature": 0,  # 随机采样可能会打断公式标记
            "num_predict": 2000,
        }
        self.client = ollama.Client(
            host=self.envs["OLLAMA_HOST"],
        )
        self.prompt_template = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text: str) -> str:
        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token

        response = self.client.chat(
            model=self.model,
            messages=self.prompt(text, self.prompt_template),
            options=self.options,
        )
        content = self._remove_cot_content(response.message.content or "")
        return content.strip()

    @staticmethod
    def _remove_cot_content(content: str) -> str:
        """Remove text content with the thought chain from the chat response

        :param content: Non-streaming text content
        :return: Text without a thought chain
        """
        return re.sub(r"^<think>.+?</think>", "", content, count=1, flags=re.DOTALL)


class XinferenceTranslator(BaseTranslator):
    # https://github.com/xorbitsai/inference
    name = "xinference"
    envs = {
        "XINFERENCE_HOST": "http://127.0.0.1:9997",
        "XINFERENCE_MODEL": "gemma-2-it",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["XINFERENCE_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = xinference_client.RESTfulClient(self.envs["XINFERENCE_HOST"])
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text):
        maxlen = max(2000, len(text) * 5)
        for model in self.model.split(";"):
            try:
                xf_model = self.client.get_model(model)
                xf_prompt = self.prompt(text, self.prompttext)
                xf_prompt = [
                    {
                        "role": "user",
                        "content": xf_prompt[0]["content"]
                        + "\n"
                        + xf_prompt[1]["content"],
                    }
                ]
                response = xf_model.chat(
                    generate_config=self.options,
                    messages=xf_prompt,
                )

                response = response["choices"][0]["message"]["content"].replace(
                    "<end_of_turn>", ""
                )
                if len(response) > maxlen:
                    raise Exception("Response too long")
                return response.strip()
            except Exception as e:
                print(e)
        raise Exception("All models failed")


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"
    envs = {
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_STREAM": "true",  # Configurable: set to "true" (default) or "false"
        "OPENAI_STOP_TOKENS": "",  # Space separated list of stop tokens
        "OPENAI_MAX_TOKENS": -1,  # Specify -1 to call the API without setting max_tokens
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
        stop_tokens=None,
        max_tokens=None,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        stop_tokens = (
            stop_tokens
            if stop_tokens is not None
            else (self.envs.get("OPENAI_STOP_TOKENS") or "").split()
        )
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(self.envs.get("OPENAI_MAX_TOKENS") or -1)
        )
        self.options = {
            "temperature": 0,  # 随机采样可能会打断公式标记
        }
        if stop_tokens:
            self.options["stop"] = stop_tokens
        if max_tokens > 0:
            self.options["max_tokens"] = max_tokens
        self.client = openai.OpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("stop", self.options.get("stop"))
        self.add_cache_impact_parameters("max_tokens", self.options.get("max_tokens"))
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)
        # Parse stream option from config (default to True for OpenAI)
        stream_val = self.envs.get("OPENAI_STREAM", "true").lower()
        self.stream = stream_val == "true"

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
            stream=self.stream,
        )
        if self.stream:
            collected = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            content = "".join(collected).strip()
        else:
            if not response.choices:
                if hasattr(response, "error"):
                    raise ValueError("Error response from Service", response.error)
            content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)


class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"
    envs = {
        "AZURE_OPENAI_BASE_URL": None,  # e.g. "https://xxx.openai.azure.com"
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_MODEL": "gpt-4o-mini",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",  # default api version
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = self.envs["AZURE_OPENAI_BASE_URL"]
        if not model:
            model = self.envs["AZURE_OPENAI_MODEL"]
        api_version = self.envs.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if api_key is None:
            api_key = self.envs["AZURE_OPENAI_API_KEY"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        return response.choices[0].message.content.strip()


class ModelScopeTranslator(OpenAITranslator):
    name = "modelscope"
    envs = {
        "MODELSCOPE_BASE_URL": "https://api-inference.modelscope.cn/v1",
        "MODELSCOPE_API_KEY": None,
        "MODELSCOPE_MODEL": "Qwen/Qwen2.5-32B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = "https://api-inference.modelscope.cn/v1"
        api_key = self.envs["MODELSCOPE_API_KEY"]
        if not model:
            model = self.envs["MODELSCOPE_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class ZhipuTranslator(OpenAITranslator):
    # https://bigmodel.cn/dev/api/thirdparty-frame/openai-sdk
    name = "zhipu"
    envs = {
        "ZHIPU_API_KEY": None,
        "ZHIPU_MODEL": "glm-4-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key = self.envs["ZHIPU_API_KEY"]
        if not model:
            model = self.envs["ZHIPU_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                **self.options,
                messages=self.prompt(text, self.prompttext),
            )
        except openai.BadRequestError as e:
            if (
                json.loads(response.choices[0].message.content.strip())["error"]["code"]
                == "1301"
            ):
                return "IRREPARABLE TRANSLATION ERROR"
            raise e
        return response.choices[0].message.content.strip()


class SiliconTranslator(OpenAITranslator):
    # https://docs.siliconflow.cn/quickstart
    name = "silicon"
    envs = {
        "SILICON_API_KEY": None,
        "SILICON_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.siliconflow.cn/v1"
        api_key = self.envs["SILICON_API_KEY"]
        if not model:
            model = self.envs["SILICON_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class X302AITranslator(OpenAITranslator):
    # https://doc.302.ai/
    name = "302ai"
    envs = {
        "X302AI_API_KEY": None,
        "X302AI_MODEL": "Gemma-7B",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.302.ai/v1"
        api_key = self.envs["X302AI_API_KEY"]
        if not model:
            model = self.envs["X302AI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class GeminiTranslator(OpenAITranslator):
    # https://ai.google.dev/gemini-api/docs/openai
    name = "gemini"
    envs = {
        "GEMINI_API_KEY": None,
        "GEMINI_MODEL": "gemini-1.5-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = self.envs["GEMINI_API_KEY"]
        if not model:
            model = self.envs["GEMINI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class AzureTranslator(BaseTranslator):
    # https://github.com/Azure/azure-sdk-for-python
    name = "azure"
    envs = {
        "AZURE_ENDPOINT": "https://api.translator.azure.cn",
        "AZURE_API_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        endpoint = self.envs["AZURE_ENDPOINT"]
        api_key = self.envs["AZURE_API_KEY"]
        credential = AzureKeyCredential(api_key)
        self.client = TextTranslationClient(
            endpoint=endpoint, credential=credential, region="chinaeast2"
        )
        # https://github.com/Azure/azure-sdk-for-python/issues/9422
        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.WARNING)

    def do_translate(self, text) -> str:
        response = self.client.translate(
            body=[text],
            from_language=self.lang_in,
            to_language=[self.lang_out],
        )
        translated_text = response[0].translations[0].text
        return translated_text


class TencentTranslator(BaseTranslator):
    # https://github.com/TencentCloud/tencentcloud-sdk-python
    name = "tencent"
    envs = {
        "TENCENTCLOUD_SECRET_ID": None,
        "TENCENTCLOUD_SECRET_KEY": None,
    }

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model)
        try:
            cred = credential.DefaultCredentialProvider().get_credential()
        except EnvironmentError:
            cred = credential.Credential(
                self.envs["TENCENTCLOUD_SECRET_ID"],
                self.envs["TENCENTCLOUD_SECRET_KEY"],
            )
        self.client = TmtClient(cred, "ap-beijing")
        self.req = TextTranslateRequest()
        self.req.Source = self.lang_in
        self.req.Target = self.lang_out
        self.req.ProjectId = 0

    # Tencent API limit: 6000 chars per request. Use 5000 as safe threshold.
    _MAX_CHARS = 5000

    def _translate_chunk(self, text):
        self.req.SourceText = text
        resp: TextTranslateResponse = self.client.TextTranslate(self.req)
        return resp.TargetText

    def do_translate(self, text):
        if len(text) <= self._MAX_CHARS:
            return self._translate_chunk(text)

        # Split on newlines, keeping the delimiter
        chunks = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > self._MAX_CHARS and current:
                chunks.append(current)
                current = line
            else:
                current += line
        if current:
            chunks.append(current)

        return "".join(self._translate_chunk(c) for c in chunks)


class AnythingLLMTranslator(BaseTranslator):
    name = "anythingllm"
    envs = {
        "AnythingLLM_URL": None,
        "AnythingLLM_APIKEY": None,
    }
    CustomPrompt = True

    def __init__(
        self, lang_out, lang_in, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["AnythingLLM_URL"]
        self.api_key = self.envs["AnythingLLM_APIKEY"]
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.prompttext = prompt

    def do_translate(self, text):
        messages = self.prompt(text, self.prompttext)
        payload = {
            "message": messages,
            "mode": "chat",
            "sessionId": "translation_expert",
        }

        response = requests.post(
            self.api_url, headers=self.headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()

        if "textResponse" in data:
            return data["textResponse"].strip()


class DifyTranslator(BaseTranslator):
    name = "dify"
    envs = {
        "DIFY_API_URL": None,  # 填写实际 Dify API 地址
        "DIFY_API_KEY": None,  # 替换为实际 API 密钥
    }

    def __init__(
        self, lang_out, lang_in, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["DIFY_API_URL"]
        self.api_key = self.envs["DIFY_API_KEY"]

    def do_translate(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": {
                "lang_out": self.lang_out,
                "lang_in": self.lang_in,
                "text": text,
            },
            "response_mode": "blocking",
            "user": "translator-service",
        }

        # 向 Dify 服务器发送请求
        response = requests.post(
            self.api_url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        response_data = response.json()

        # 解析响应
        return response_data.get("answer", "")


class ArgosTranslator(BaseTranslator):
    name = "argos"

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            logger.warning(
                "argos-translate is not installed, if you want to use argostranslate, please install it. If you don't use argostranslate translator, you can safely ignore this warning."
            )
            raise
        super().__init__(lang_in, lang_out, model, ignore_cache)
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        try:
            available_package = list(
                filter(
                    lambda x: x.from_code == self.lang_in
                    and x.to_code == self.lang_out,
                    available_packages,
                )
            )[0]
        except Exception:
            raise ValueError(
                "lang_in and lang_out pair not supported by Argos Translate."
            )
        download_path = available_package.download()
        argostranslate.package.install_from_path(download_path)

    def translate(self, text: str, ignore_cache: bool = False):
        # Translate
        import argotranslate.translate  # noqa: F401

        installed_languages = (
            argostranslate.translate.get_installed_languages()  # noqa: F821
        )
        from_lang = list(filter(lambda x: x.code == self.lang_in, installed_languages))[
            0
        ]
        to_lang = list(filter(lambda x: x.code == self.lang_out, installed_languages))[
            0
        ]
        translation = from_lang.get_translation(to_lang)
        translatedText = translation.translate(text)
        return translatedText


class GrokTranslator(OpenAITranslator):
    # https://docs.x.ai/docs/overview#getting-started
    name = "grok"
    envs = {
        "GROK_API_KEY": None,
        "GROK_MODEL": "grok-2-1212",
        "GROK_BASE_URL": "https://api.x.ai/v1",  # Configurable base URL
        "GROK_STREAM": "true",  # Configurable: set to "true" (default) or "false"
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = self.envs.get("GROK_BASE_URL", "https://api.x.ai/v1")
        api_key = self.envs["GROK_API_KEY"]
        if not model:
            model = self.envs["GROK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        # Override stream setting from config (default to True)
        stream_val = self.envs.get("GROK_STREAM", "true").lower()
        self.stream = stream_val == "true"


class GroqTranslator(OpenAITranslator):
    name = "groq"
    envs = {
        "GROQ_API_KEY": None,
        "GROQ_MODEL": "llama-3-3-70b-versatile",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.groq.com/openai/v1"
        api_key = self.envs["GROQ_API_KEY"]
        if not model:
            model = self.envs["GROQ_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class DeepseekTranslator(OpenAITranslator):
    name = "deepseek"
    envs = {
        "DEEPSEEK_API_KEY": None,
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.deepseek.com/v1"
        api_key = self.envs["DEEPSEEK_API_KEY"]
        if not model:
            model = self.envs["DEEPSEEK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class MiniMaxTranslator(OpenAITranslator):
    # https://platform.minimaxi.com/document/introduction
    name = "minimax"
    envs = {
        "MINIMAX_API_KEY": None,
        "MINIMAX_MODEL": "MiniMax-M2.7",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.minimax.io/v1"
        api_key = self.envs["MINIMAX_API_KEY"]
        if not model:
            model = self.envs["MINIMAX_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.options = {"temperature": 0.1}
        self.prompttext = prompt


class OpenAIlikedTranslator(OpenAITranslator):
    name = "openailiked"
    envs = {
        "OPENAILIKED_BASE_URL": None,
        "OPENAILIKED_API_KEY": None,
        "OPENAILIKED_MODEL": None,
        "OPENAILIKED_STREAM": "false",  # Configurable: set to "true" or "false"
        "OPENAILIKED_STOP_TOKENS": "",  # Space separated list of stop tokens
        "OPENAILIKED_MAX_TOKENS": -1,  # Specify -1 to call the API without setting max_tokens
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if self.envs["OPENAILIKED_BASE_URL"]:
            base_url = self.envs["OPENAILIKED_BASE_URL"]
        else:
            raise ValueError("The OPENAILIKED_BASE_URL is missing.")
        if not model:
            if self.envs["OPENAILIKED_MODEL"]:
                model = self.envs["OPENAILIKED_MODEL"]
            else:
                raise ValueError("The OPENAILIKED_MODEL is missing.")
        if self.envs["OPENAILIKED_API_KEY"] is None:
            api_key = "openailiked"
        else:
            api_key = self.envs["OPENAILIKED_API_KEY"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
            prompt=prompt,
            stop_tokens=self.envs.get("OPENAILIKED_STOP_TOKENS", "").split(),
            max_tokens=int(self.envs.get("OPENAILIKED_MAX_TOKENS", -1)),
        )
        # Parse stream option from config (default to False for compatibility)
        stream_val = self.envs.get("OPENAILIKED_STREAM", "false").lower()
        self.stream = stream_val == "true"

    def do_translate(self, text) -> str:
        """Override to support configurable streaming."""
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
            stream=self.stream,
        )
        if self.stream:
            collected = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            content = "".join(collected).strip()
        else:
            if not response.choices:
                if hasattr(response, "error"):
                    raise ValueError("Error response from Service", response.error)
            content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content


class CodexTranslator(BaseTranslator):
    name = "codex"
    envs = {
        "CODEX_BIN": "codex",
        "CODEX_PROFILE": None,
        "CODEX_MODEL": "gpt-5.4-mini",
        "CODEX_TIMEOUT": "120",
    }
    CustomPrompt = True
    REQUIRED_EXEC_FLAGS = {
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "--color",
        "--output-schema",
        "--output-last-message",
    }
    FAST_PATH_FLAGS = {"--ignore-user-config", "--ignore-rules", "--model", "-c"}
    COMPAT_PATH_FLAGS = {"--model", "--profile"}
    MAX_BATCH_ITEMS = 8
    MAX_BATCH_CHARS = 2500
    MAX_ITEM_CHARS = 300
    HSPACE_RE = r"[ \t\u00A0]+"
    CJK_CHAR_RE = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    CJK_PUNCT_RE = r"[\u3001-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]"
    PLACEHOLDER_RE = r"(?:\{\{v\d+\}\}|\{v\d+\})"

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["CODEX_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.codex_bin = self.envs["CODEX_BIN"] or "codex"
        self.profile = self.envs["CODEX_PROFILE"]
        self.timeout = int(self.envs.get("CODEX_TIMEOUT") or "120")
        self.prompttext = prompt
        self.single_output_schema = {
            "type": "object",
            "properties": {
                "translation": {"type": "string"},
            },
            "required": ["translation"],
            "additionalProperties": False,
        }
        self.batch_output_schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        }
        self.codex_version = None
        self.supported_exec_flags: set[str] = set()
        self.fast_command_available = False
        self.compat_command_available = False
        self.preferred_command_mode = "fast"
        self._probe_cli()
        self.add_cache_impact_parameters("profile", self.profile)
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        self.add_cache_impact_parameters("command_mode", self.preferred_command_mode)

    def _build_codex_prompt(self, text: str) -> str:
        base_prompt = self.prompt(text, self.prompttext)[0]["content"]
        return (
            f"{base_prompt}\n\n"
            "Additional requirements:\n"
            '- Return valid JSON with exactly one field: {"translation": "..."}.\n'
            '- The "translation" field must contain only the translated text.\n'
            "- Preserve markdown structure and formulas.\n"
            "- Preserve placeholder tokens like {v0} and {{v0}} exactly.\n"
            "- Do not add explanations, comments, or code fences.\n"
        )

    def _build_batch_prompt(self, texts: list[str]) -> str:
        indexed_texts = [
            {"index": idx, "text": text} for idx, text in enumerate(texts, start=1)
        ]
        serialized_texts = json.dumps(indexed_texts, ensure_ascii=False)
        return (
            "You are a professional, authentic machine translation engine. "
            "Only output valid JSON that matches the provided schema.\n\n"
            f"Translate the `text` field of each object in the following JSON array "
            f"from {self.lang_in} to {self.lang_out}. Preserve markdown structure, "
            "formulas, and placeholder tokens like {v0} and {{v0}} exactly. "
            f"There are exactly {len(texts)} items. Return exactly {len(texts)} "
            "translated strings in ascending `index` order. Do not merge, drop, "
            "or reorder items.\n\n"
            f"Source Texts JSON: {serialized_texts}\n\n"
            'Return JSON with exactly one field: {"translations": ["...", "..."]}.'
        )

    def _run_probe_command(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [self.codex_bin, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"codex executable not found: {self.codex_bin}"
            ) from exc

    def _probe_cli(self):
        version_result = self._run_probe_command(["--version"])
        if version_result.returncode != 0:
            detail = version_result.stderr.strip() or version_result.stdout.strip()
            raise RuntimeError(f"Codex CLI version probe failed: {detail}")
        self.codex_version = version_result.stdout.strip() or version_result.stderr.strip()

        help_result = self._run_probe_command(["exec", "--help"])
        if help_result.returncode != 0:
            detail = help_result.stderr.strip() or help_result.stdout.strip()
            raise RuntimeError(f"Codex CLI help probe failed: {detail}")

        help_text = "\n".join([help_result.stdout, help_result.stderr])
        candidate_flags = (
            self.REQUIRED_EXEC_FLAGS
            | self.FAST_PATH_FLAGS
            | self.COMPAT_PATH_FLAGS
            | {"--config"}
        )
        self.supported_exec_flags = {
            flag for flag in candidate_flags if flag in help_text
        }
        if "--config" in self.supported_exec_flags:
            self.supported_exec_flags.add("-c")

        missing_required = self.REQUIRED_EXEC_FLAGS - self.supported_exec_flags
        if missing_required:
            raise RuntimeError(
                "Codex CLI is missing required exec flags: "
                + ", ".join(sorted(missing_required))
            )

        self.compat_command_available = True
        self.fast_command_available = self.FAST_PATH_FLAGS.issubset(
            self.supported_exec_flags
        ) and not self.profile
        if self.profile:
            if "--profile" not in self.supported_exec_flags:
                raise RuntimeError(
                    "Codex CLI does not support --profile, but CODEX_PROFILE was set."
                )
            self.preferred_command_mode = "compat"
        elif self.fast_command_available:
            self.preferred_command_mode = "fast"
        else:
            self.preferred_command_mode = "compat"

    @staticmethod
    def _is_passthrough_text(text: str) -> bool:
        return not text.strip() or re.match(r"^\{v\d+\}$", text) is not None

    @staticmethod
    def _looks_like_unsupported_flag(detail: str) -> bool:
        lowered = detail.lower()
        return any(
            needle in lowered
            for needle in [
                "unexpected argument",
                "unrecognized option",
                "unknown option",
                "found argument",
            ]
        )

    def _build_command(
        self, prompt_text: str, schema_path: str, output_path: str, mode: str
    ) -> list[str]:
        command = [
            self.codex_bin,
            "exec",
        ]
        if mode == "fast":
            command.extend(["--ignore-user-config", "--ignore-rules"])
        if mode == "compat" and self.profile:
            command.extend(["--profile", self.profile])
        command.extend(
            [
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--color",
                "never",
            ]
        )
        if self.model and "--model" in self.supported_exec_flags:
            command.extend(["--model", self.model])
        if mode == "fast" and "-c" in self.supported_exec_flags:
            command.extend(["-c", 'model_reasoning_effort="none"'])
        command.extend(
            [
                "--output-schema",
                schema_path,
                "--output-last-message",
                output_path,
            ]
        )
        command.append(prompt_text)
        return command

    def _load_json_output(self, output_path: str) -> dict:
        if not os.path.exists(output_path):
            raise RuntimeError("Codex translator did not produce an output file.")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Codex translator returned invalid JSON output.") from exc

    def _load_translation(self, output_path: str) -> str:
        payload = self._load_json_output(output_path)
        translation = payload.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise RuntimeError(
                "Codex translator output is missing the required 'translation' field."
            )
        return self._normalize_translation_output(translation.strip())

    def _load_batch_translations(self, output_path: str, expected_count: int) -> list[str]:
        payload = self._load_json_output(output_path)
        translations = payload.get("translations")
        if not isinstance(translations, list):
            raise RuntimeError(
                "Codex translator output is missing the required 'translations' field."
            )
        if len(translations) != expected_count:
            raise RuntimeError(
                "Codex translator batch output must have the same length as the input."
            )
        if any(not isinstance(item, str) or not item.strip() for item in translations):
            raise RuntimeError("Codex translator batch output contains empty items.")
        return [self._normalize_translation_output(item.strip()) for item in translations]

    def _iter_command_modes(self) -> list[str]:
        if self.preferred_command_mode == "compat" or not self.fast_command_available:
            return ["compat"]
        return ["fast", "compat"]

    def _execute_codex_request(
        self, prompt_text: str, schema: dict, response_loader, mode_override: str = None
    ):
        with tempfile.TemporaryDirectory(prefix="pdf2zh-codex-") as workdir:
            schema_path = os.path.join(workdir, "output.schema.json")
            output_path = os.path.join(workdir, "output.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f)

            modes = [mode_override] if mode_override else self._iter_command_modes()
            last_error = None
            for mode in modes:
                command = self._build_command(prompt_text, schema_path, output_path, mode)
                try:
                    result = subprocess.run(
                        command,
                        cwd=workdir,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        check=False,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"codex executable not found: {self.codex_bin}"
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"Codex translator timed out after {self.timeout} seconds."
                    ) from exc

                if result.returncode == 0:
                    if mode == "compat":
                        self.preferred_command_mode = "compat"
                    return response_loader(output_path)

                detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
                last_error = RuntimeError(
                    f"Codex translator failed with exit code {result.returncode}: {detail}"
                )
                if (
                    mode == "fast"
                    and "compat" in modes
                    and self._looks_like_unsupported_flag(detail)
                ):
                    self.preferred_command_mode = "compat"
                    continue
                raise last_error

            raise last_error

    def _run_single_translation(self, text: str) -> str:
        prompt_text = self._build_codex_prompt(text)
        return self._execute_codex_request(
            prompt_text, self.single_output_schema, self._load_translation
        )

    def _chunk_batch(self, texts: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
        batches = []
        current_batch = []
        current_chars = 0
        for item in texts:
            _, text = item
            text_len = len(text)
            if current_batch and (
                len(current_batch) >= self.MAX_BATCH_ITEMS
                or current_chars + text_len > self.MAX_BATCH_CHARS
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(item)
            current_chars += text_len
        if current_batch:
            batches.append(current_batch)
        return batches

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.MAX_ITEM_CHARS:
            return [text]

        sentence_like_parts = []
        cursor = 0
        for match in re.finditer(r".*?(?:[.!?;:](?:\s+|$)|$)", text, flags=re.DOTALL):
            part = match.group(0)
            if not part:
                continue
            sentence_like_parts.append(part)
            cursor = match.end()
            if cursor >= len(text):
                break
        if not sentence_like_parts:
            sentence_like_parts = [text]

        chunks = []
        current = ""
        for part in sentence_like_parts:
            if len(part) > self.MAX_ITEM_CHARS:
                if current:
                    chunks.append(current)
                    current = ""
                remaining = part
                while len(remaining) > self.MAX_ITEM_CHARS:
                    split_at = remaining.rfind(" ", 0, self.MAX_ITEM_CHARS)
                    if split_at <= 0:
                        split_at = self.MAX_ITEM_CHARS
                    chunks.append(remaining[:split_at])
                    remaining = remaining[split_at:]
                if remaining:
                    current = remaining
                continue

            if current and len(current) + len(part) > self.MAX_ITEM_CHARS:
                chunks.append(current)
                current = part
            else:
                current += part

        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk]

    def _run_batch_translation(self, texts: list[str]) -> list[str]:
        prompt_text = self._build_batch_prompt(texts)
        try:
            return self._execute_codex_request(
                prompt_text,
                self.batch_output_schema,
                lambda output_path: self._load_batch_translations(
                    output_path, len(texts)
                ),
            )
        except RuntimeError as exc:
            if len(texts) == 1:
                return [self._run_single_translation(texts[0])]
            detail = str(exc)
            if (
                "translations" not in detail
                and "same length" not in detail
                and "empty items" not in detail
                and "timed out" not in detail
            ):
                raise
            midpoint = len(texts) // 2
            return self._run_batch_translation(texts[:midpoint]) + self._run_batch_translation(
                texts[midpoint:]
            )

    @staticmethod
    def _recombine_translated_segments(
        source_segments: list[str], translated_segments: list[str]
    ) -> str:
        combined = ""
        for idx, translated in enumerate(translated_segments):
            cleaned = translated.strip()
            if idx == 0:
                combined = cleaned
                continue
            previous_source = source_segments[idx - 1]
            current_source = source_segments[idx]
            boundary_match = re.search(r"(\s+)$", previous_source) or re.match(
                r"^(\s+)", current_source
            )
            boundary = boundary_match.group(1) if boundary_match else ""
            if boundary and not combined.endswith(boundary):
                combined += boundary
            combined += cleaned
        return combined

    def _normalize_translation_output(self, text: str) -> str:
        if self.lang_out.lower() not in {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant"}:
            return text

        pairs = [
            (self.CJK_CHAR_RE, self.CJK_CHAR_RE),
            (self.CJK_CHAR_RE, self.CJK_PUNCT_RE),
            (self.CJK_PUNCT_RE, self.CJK_CHAR_RE),
            (self.CJK_CHAR_RE, self.PLACEHOLDER_RE),
            (self.PLACEHOLDER_RE, self.CJK_CHAR_RE),
            (self.CJK_PUNCT_RE, self.PLACEHOLDER_RE),
            (self.PLACEHOLDER_RE, self.CJK_PUNCT_RE),
        ]
        normalized = text
        changed = True
        while changed:
            previous = normalized
            for left, right in pairs:
                normalized = re.sub(
                    fr"({left}){self.HSPACE_RE}({right})", r"\1\2", normalized
                )
            changed = normalized != previous
        return normalized

    def do_translate(self, text: str) -> str:
        return self._run_single_translation(text)

    def translate_batch(
        self, texts: list[str], ignore_cache: bool = False
    ) -> list[str]:
        if self.prompttext:
            return BaseTranslator.translate_batch(self, texts, ignore_cache=ignore_cache)

        results = [None] * len(texts)
        pending_items = []
        for idx, text in enumerate(texts):
            if self._is_passthrough_text(text):
                results[idx] = text
                continue
            if not (self.ignore_cache or ignore_cache):
                cache_result = self.cache.get(text)
                if cache_result is not None:
                    results[idx] = cache_result
                    continue
            pending_items.append((idx, text))

        if pending_items:
            expanded_items = []
            segment_sources: dict[int, list[str]] = {}
            recombine_map: dict[int, list[int]] = {}
            expanded_index = 0
            for original_idx, source_text in pending_items:
                segments = self._split_long_text(source_text)
                segment_sources[original_idx] = segments
                recombine_map[original_idx] = []
                for segment in segments:
                    expanded_items.append((expanded_index, segment))
                    recombine_map[original_idx].append(expanded_index)
                    expanded_index += 1

            expanded_results: dict[int, str] = {}
            for batch in self._chunk_batch(expanded_items):
                batch_texts = [text for _, text in batch]
                translated_batch = self._run_batch_translation(batch_texts)
                for (batch_idx, _source_text), translated_text in zip(batch, translated_batch):
                    expanded_results[batch_idx] = translated_text

            for original_idx, source_text in pending_items:
                segment_indices = recombine_map[original_idx]
                translated_segments = [
                    expanded_results[segment_idx] for segment_idx in segment_indices
                ]
                combined_translation = self._recombine_translated_segments(
                    segment_sources[original_idx], translated_segments
                )
                combined_translation = self._normalize_translation_output(
                    combined_translation
                )
                results[original_idx] = combined_translation
                self.cache.set(source_text, combined_translation)

        return [text if result is None else result for text, result in zip(texts, results)]

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)


class QwenMtTranslator(OpenAITranslator):
    """
    Use Qwen-MT model from Aliyun. it's designed for translating.
    Since Traditional Chinese is not yet supported by Aliyun. it will be also translated to Simplified Chinese, when it's selected.
    There's special parameters in the message to the server.
    """

    name = "qwen-mt"
    envs = {
        "ALI_MODEL": "qwen-mt-turbo",
        "ALI_API_KEY": None,
        "ALI_DOMAINS": "This sentence is extracted from a scientific paper. When translating, please pay close attention to the use of specialized troubleshooting terminologies and adhere to scientific sentence structures to maintain the technical rigor and precision of the original text.",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = self.envs["ALI_API_KEY"]

        if not model:
            model = self.envs["ALI_MODEL"]

        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt

    @staticmethod
    def lang_mapping(input_lang: str) -> str:
        """
        Mapping the language code to the language code that Aliyun Qwen-Mt model supports.
        Since all existings languagues codes used in gui.py are able to be mapped, the original
        languague code will not be checked.
        """
        langdict = {
            "zh": "Chinese",
            "zh-TW": "Chinese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "es": "Spanish",
            "it": "Italian",
        }

        return langdict[input_lang]

    def do_translate(self, text) -> str:
        """
        Qwen-MT Model reqeust to send translation_options to the server.
        domains are options, but suggested. it must be in English.
        """
        translation_options = {
            "source_lang": self.lang_mapping(self.lang_in),
            "target_lang": self.lang_mapping(self.lang_out),
            "domains": self.envs["ALI_DOMAINS"],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=[{"role": "user", "content": text}],
            extra_body={"translation_options": translation_options},
        )
        return response.choices[0].message.content.strip()
