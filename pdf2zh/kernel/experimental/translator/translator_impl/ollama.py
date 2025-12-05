import logging

import ollama
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


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama
    name = "ollama"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.options = {
            "temperature": 0,
            "num_predict": settings.translate_engine_settings.num_predict,
        }  # 随机采样可能会打断公式标记
        self.client = ollama.Client(
            host=settings.translate_engine_settings.ollama_host,
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("num_predict", self.options["num_predict"])
        self.model = settings.translate_engine_settings.ollama_model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token
        response = self.client.chat(
            model=self.model,
            options=self.options,
            messages=self.prompt(text),
        )
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token

        response = self.client.chat(
            model=self.model,
            options=self.options,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message
