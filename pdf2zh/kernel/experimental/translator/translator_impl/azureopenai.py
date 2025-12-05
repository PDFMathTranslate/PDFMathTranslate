import logging

import openai
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh.kernel.experimental.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.AzureOpenAI(
            azure_endpoint=settings.translate_engine_settings.azure_openai_base_url,
            azure_deployment=settings.translate_engine_settings.azure_openai_model,
            api_version=settings.translate_engine_settings.azure_openai_api_version,
            api_key=settings.translate_engine_settings.azure_openai_api_key,
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = settings.translate_engine_settings.azure_openai_model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "total_tokens"):
                self.token_count.inc(response.usage.total_tokens)
            if hasattr(response.usage, "prompt_tokens"):
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if hasattr(response.usage, "completion_tokens"):
                self.completion_token_count.inc(response.usage.completion_tokens)
        message = response.choices[0].message.content.strip()
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "total_tokens"):
                self.token_count.inc(response.usage.total_tokens)
            if hasattr(response.usage, "prompt_tokens"):
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if hasattr(response.usage, "completion_tokens"):
                self.completion_token_count.inc(response.usage.completion_tokens)
        message = response.choices[0].message.content.strip()
        message = self._remove_cot_content(message)
        return message
