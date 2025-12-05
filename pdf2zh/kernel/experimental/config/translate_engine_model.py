import logging
import re
import typing
from dataclasses import dataclass
from inspect import getdoc
from types import NoneType
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

# any field in SENSITIVE_FIELDS will be masked in GUI
GUI_SENSITIVE_FIELDS = []
# any field in GUI_PASSWORD_FIELDS will be masked in GUI and treated as password
GUI_PASSWORD_FIELDS = []

logger = logging.getLogger(__name__)


def _clean_string(value: str | None) -> str | None:
    """Clean string by trimming whitespace"""
    if value is None:
        return None
    return value.strip()


def _clean_url(value: str | None) -> str | None:
    """Clean URL for OpenAI-compatible services"""
    if value is None:
        return None
    cleaned = value.strip().rstrip("/")
    # Remove /chat/completions suffix for OpenAI-compatible APIs
    cleaned = re.sub(r"/chat/completions/?$", "", cleaned)
    return cleaned.rstrip("/")


def _check_if_positive_float(value: str | None, field: str = "Value") -> str | None:
    """Check if a string can be parsed as a positive float"""
    if value is None:
        return None

    try:
        f = float(value)
    except ValueError as e:
        raise ValueError(f"{field} must be a float") from e

    if f <= 0:
        raise ValueError(f"{field} must be greater than 0")

    return value


class TranslateEngineSettingError(Exception):
    """Translate engine setting error"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


## Please add the translator configuration class below this location.

# Please note that all translator configurations must be of string type,
# otherwise the GUI will not function properly!
#
# You should implement validation of the translator configuration in validate_settings.
# And complete type conversion (if any) in the corresponding implementation of the translator.


class OpenAISettings(BaseModel):
    """OpenAI API settings"""

    translate_engine_type: Literal["OpenAI"] = Field(default="OpenAI")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_base_url: str | None = Field(
        default=None, description="Base URL for OpenAI API"
    )
    openai_api_key: str | None = Field(
        default=None, description="API key for OpenAI service"
    )
    openai_timeout: str | None = Field(
        default=None, description="Timeout (seconds) for OpenAI service"
    )
    openai_temperature: str | None = Field(
        default=None, description="Temperature for OpenAI service"
    )
    openai_reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort for OpenAI service (minimal/low/medium/high)",
    )
    openai_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for OpenAI service"
    )

    # This parameter contains a spelling error, but it will not be corrected for compatibility reasons.
    # For details, see: https://github.com/PDFMathTranslate/PDFMathTranslate-next/issues/175#issuecomment-3213568681
    openai_send_temprature: bool | None = Field(
        default=None, description="Send temprature to OpenAI service"
    )
    openai_send_reasoning_effort: bool | None = Field(
        default=None, description="Send reasoning effort to OpenAI service"
    )

    def validate_settings(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.openai_api_key = _clean_string(self.openai_api_key)
        self.openai_base_url = _clean_url(self.openai_base_url)
        self.openai_model = _clean_string(self.openai_model)
        self.openai_timeout = _check_if_positive_float(
            _clean_string(self.openai_timeout),
            field="Timeout",
        )
        self.openai_temperature = _clean_string(self.openai_temperature)
        self.openai_reasoning_effort = _clean_string(self.openai_reasoning_effort)
        if self.openai_send_temprature:
            if not self.openai_temperature:
                raise ValueError(
                    "Temperature is required when send temperature is enabled"
                )
            try:
                float(self.openai_temperature)
            except ValueError as e:
                raise ValueError("Temperature must be a float") from e
        if self.openai_send_reasoning_effort and not self.openai_reasoning_effort:
            raise ValueError(
                "Reasoning effort is required when send reasoning effort is enabled"
            )


GUI_PASSWORD_FIELDS.append("openai_api_key")
GUI_SENSITIVE_FIELDS.append("openai_base_url")


class BingSettings(BaseModel):
    """Bing Translation settings"""

    translate_engine_type: Literal["Bing"] = Field(default="Bing")

    def validate_settings(self) -> None:
        pass


class GoogleSettings(BaseModel):
    """Google Translation settings"""

    translate_engine_type: Literal["Google"] = Field(default="Google")

    def validate_settings(self) -> None:
        pass


class DeepLSettings(BaseModel):
    """Bing Translation settings"""

    translate_engine_type: Literal["DeepL"] = Field(default="DeepL")
    deepl_auth_key: str | None = Field(default=None, description="DeepL auth key")

    def validate_settings(self) -> None:
        if not self.deepl_auth_key:
            raise ValueError("DeepL Auth key is required")
        self.deepl_auth_key = _clean_string(self.deepl_auth_key)


GUI_PASSWORD_FIELDS.append("deepl_auth_key")

# for openai compatibility translator
# You only need to add the corresponding configuration class
# and return the OpenAISettings instance using the transform method.


class DeepSeekSettings(BaseModel):
    """DeepSeek settings"""

    translate_engine_type: Literal["DeepSeek"] = Field(default="DeepSeek")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )
    deepseek_model: str = Field(
        default="deepseek-chat", description="DeepSeek model to use"
    )
    deepseek_api_key: str | None = Field(
        default=None, description="API key for DeepSeek service"
    )
    deepseek_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for DeepSeek service"
    )

    def validate_settings(self) -> None:
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key is required")
        self.deepseek_api_key = _clean_string(self.deepseek_api_key)
        self.deepseek_model = _clean_string(self.deepseek_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.deepseek_model,
            openai_api_key=self.deepseek_api_key,
            openai_base_url="https://api.deepseek.com/v1",
            openai_enable_json_mode=self.deepseek_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("deepseek_api_key")


class OllamaSettings(BaseModel):
    """Ollama API settings"""

    translate_engine_type: Literal["Ollama"] = Field(default="Ollama")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    ollama_model: str = Field(default="gemma2", description="Ollama model to use")
    ollama_host: str | None = Field(
        default="http://localhost:11434", description="Ollama host"
    )
    num_predict: int | None = Field(
        default=2000, description="The max number of token to predict."
    )

    def validate_settings(self) -> None:
        if not self.ollama_host:
            raise ValueError("Ollama host is required")
        self.ollama_host = _clean_string(self.ollama_host)
        self.ollama_model = _clean_string(self.ollama_model)


GUI_SENSITIVE_FIELDS.append("ollama_host")


class XinferenceSettings(BaseModel):
    """Xinference API settings"""

    translate_engine_type: Literal["Xinference"] = Field(default="Xinference")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    xinference_model: str = Field(
        default="gemma-2-it", description="Xinference model to use"
    )
    xinference_host: str | None = Field(default=None, description="Xinference host")

    def validate_settings(self) -> None:
        if not self.xinference_host:
            raise ValueError("Xinference host is required")
        self.xinference_host = _clean_string(self.xinference_host)
        self.xinference_model = _clean_string(self.xinference_model)


GUI_SENSITIVE_FIELDS.append("xinference_host")


class AzureOpenAISettings(BaseModel):
    """AzureOpenAI API settings"""

    translate_engine_type: Literal["AzureOpenAI"] = Field(default="AzureOpenAI")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    azure_openai_model: str = Field(
        default="gpt-4o-mini", description="AzureOpenAI model to use"
    )
    azure_openai_base_url: str | None = Field(
        default=None, description="Base URL for AzureOpenAI API"
    )
    azure_openai_api_key: str | None = Field(
        default=None, description="API key for AzureOpenAI service"
    )
    azure_openai_api_version: str = Field(
        default="2024-06-01", description="API version for AzureOpenAI service"
    )

    def validate_settings(self) -> None:
        if not self.azure_openai_api_key:
            raise ValueError("AzureOpenAI API key is required")
        self.azure_openai_api_key = _clean_string(self.azure_openai_api_key)
        self.azure_openai_base_url = _clean_string(self.azure_openai_base_url)
        self.azure_openai_model = _clean_string(self.azure_openai_model)
        self.azure_openai_api_version = _clean_string(self.azure_openai_api_version)


GUI_PASSWORD_FIELDS.append("azure_openai_api_key")
GUI_SENSITIVE_FIELDS.append("azure_openai_base_url")


class ModelScopeSettings(BaseModel):
    """ModelScope API settings"""

    translate_engine_type: Literal["ModelScope"] = Field(default="ModelScope")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    modelscope_model: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct", description="ModelScope model to use"
    )
    modelscope_api_key: str | None = Field(
        default=None, description="API key for ModelScope service"
    )
    modelscope_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for ModelScope service"
    )

    def validate_settings(self) -> None:
        if not self.modelscope_api_key:
            raise ValueError("ModelScope API key is required")
        self.modelscope_api_key = _clean_string(self.modelscope_api_key)
        self.modelscope_model = _clean_string(self.modelscope_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.modelscope_model,
            openai_api_key=self.modelscope_api_key,
            openai_base_url="https://api-inference.modelscope.cn/v1",
            openai_enable_json_mode=self.modelscope_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("modelscope_api_key")


class ZhipuSettings(BaseModel):
    """Zhipu API settings"""

    translate_engine_type: Literal["Zhipu"] = Field(default="Zhipu")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    zhipu_model: str = Field(default="glm-4-flash", description="Zhipu model to use")
    zhipu_api_key: str | None = Field(
        default=None, description="API key for Zhipu service"
    )
    zhipu_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for Zhipu service"
    )

    def validate_settings(self) -> None:
        if not self.zhipu_api_key:
            raise ValueError("Zhipu API key is required")
        self.zhipu_api_key = _clean_string(self.zhipu_api_key)
        self.zhipu_model = _clean_string(self.zhipu_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.zhipu_model,
            openai_api_key=self.zhipu_api_key,
            openai_base_url="https://open.bigmodel.cn/api/paas/v4",
            openai_enable_json_mode=self.zhipu_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("zhipu_api_key")


class SiliconFlowSettings(BaseModel):
    """SiliconFlow API settings"""

    translate_engine_type: Literal["SiliconFlow"] = Field(default="SiliconFlow")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    siliconflow_base_url: str | None = Field(
        default="https://api.siliconflow.cn/v1",
        description="Base URL for SiliconFlow API",
    )
    siliconflow_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", description="SiliconFlow model to use"
    )
    siliconflow_api_key: str | None = Field(
        default=None, description="API key for SiliconFlow service"
    )
    siliconflow_enable_thinking: bool | None = Field(
        default=False, description="Enable thinking for SiliconFlow service"
    )
    siliconflow_send_enable_thinking_param: bool | None = Field(
        default=False,
        description="Send enable thinking param to SiliconFlow service",
    )
    siliconflow_enable_json_mode: bool | None = Field(
        default=False, description="Enable JSON mode for SiliconFlow service"
    )

    def validate_settings(self) -> None:
        if not self.siliconflow_api_key:
            raise ValueError("SiliconFlow API key is required")
        self.siliconflow_api_key = _clean_string(self.siliconflow_api_key)
        self.siliconflow_base_url = _clean_string(self.siliconflow_base_url)
        self.siliconflow_model = _clean_string(self.siliconflow_model)


GUI_PASSWORD_FIELDS.append("siliconflow_api_key")
GUI_SENSITIVE_FIELDS.append("siliconflow_base_url")


class SiliconFlowFreeSettings(BaseModel):
    """SiliconFlow Free API settings"""

    translate_engine_type: Literal["SiliconFlowFree"] = Field(default="SiliconFlowFree")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    siliconflow_free_enable_json_mode: bool | None = Field(
        default=False, description="Enable JSON mode for SiliconFlow Free service"
    )

    def validate_settings(self) -> None:
        pass


class TencentSettings(BaseModel):
    """Tencent Mechine Translation settings"""

    translate_engine_type: Literal["TencentMechineTranslation"] = Field(
        default="TencentMechineTranslation"
    )
    tencentcloud_secret_id: str | None = Field(
        default=None, description="Tencent Mechine Translation secret ID"
    )
    tencentcloud_secret_key: str | None = Field(
        default=None, description="Tencent Mechine Translation secret Key"
    )

    def validate_settings(self) -> None:
        if not self.tencentcloud_secret_id:
            raise ValueError("Tencent Mechine Translation ID is required")
        if not self.tencentcloud_secret_key:
            raise ValueError("Tencent Mechine Translation Key is required")
        self.tencentcloud_secret_id = _clean_string(self.tencentcloud_secret_id)
        self.tencentcloud_secret_key = _clean_string(self.tencentcloud_secret_key)


GUI_PASSWORD_FIELDS.append("tencentcloud_secret_id")
GUI_PASSWORD_FIELDS.append("tencentcloud_secret_key")


class GeminiSettings(BaseModel):
    """Gemini API settings"""

    translate_engine_type: Literal["Gemini"] = Field(default="Gemini")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    gemini_model: str = Field(
        default="gemini-1.5-flash", description="Gemini model to use"
    )
    gemini_api_key: str | None = Field(
        default=None, description="API key for Gemini service"
    )
    gemini_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for Gemini service"
    )

    def validate_settings(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        self.gemini_api_key = _clean_string(self.gemini_api_key)
        self.gemini_model = _clean_string(self.gemini_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.gemini_model,
            openai_api_key=self.gemini_api_key,
            openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            openai_enable_json_mode=self.gemini_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("gemini_api_key")


class AzureSettings(BaseModel):
    """Azure Translation settings"""

    translate_engine_type: Literal["Azure"] = Field(default="Azure")
    azure_endpoint: str | None = Field(
        default="https://api.translator.azure.cn", description="Azure endpoint"
    )
    azure_api_key: str | None = Field(default=None, description="Azure API Key")

    def validate_settings(self) -> None:
        if not self.azure_api_key:
            raise ValueError("Azure API key is required")
        self.azure_api_key = _clean_string(self.azure_api_key)
        self.azure_endpoint = _clean_string(self.azure_endpoint)


GUI_PASSWORD_FIELDS.append("azure_api_key")
GUI_SENSITIVE_FIELDS.append("azure_endpoint")


class AnythingLLMSettings(BaseModel):
    """AnythingLLM settings"""

    translate_engine_type: Literal["AnythingLLM"] = Field(default="AnythingLLM")
    anythingllm_url: str | None = Field(default=None, description="AnythingLLM url")
    anythingllm_apikey: str | None = Field(
        default=None, description="AnythingLLM API Key"
    )

    def validate_settings(self) -> None:
        if not self.anythingllm_apikey:
            raise ValueError("AnythingLLM API Key is required")
        self.anythingllm_apikey = _clean_string(self.anythingllm_apikey)
        self.anythingllm_url = _clean_string(self.anythingllm_url)


GUI_PASSWORD_FIELDS.append("anythingllm_apikey")
GUI_SENSITIVE_FIELDS.append("anythingllm_url")


class DifySettings(BaseModel):
    """Dify settings"""

    translate_engine_type: Literal["Dify"] = Field(default="Dify")
    dify_url: str | None = Field(default=None, description="Dify url")
    dify_apikey: str | None = Field(default=None, description="Dify API Key")

    def validate_settings(self) -> None:
        if not self.dify_apikey:
            raise ValueError("Dify API Key is required")
        self.dify_apikey = _clean_string(self.dify_apikey)
        self.dify_url = _clean_string(self.dify_url)


GUI_PASSWORD_FIELDS.append("dify_apikey")
GUI_SENSITIVE_FIELDS.append("dify_url")


class GrokSettings(BaseModel):
    """Grok API settings"""

    translate_engine_type: Literal["Grok"] = Field(default="Grok")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    grok_model: str = Field(default="grok-2-1212", description="Grok model to use")
    grok_api_key: str | None = Field(
        default=None, description="API key for Grok service"
    )
    grok_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for Grok service"
    )

    def validate_settings(self) -> None:
        if not self.grok_api_key:
            raise ValueError("Grok API key is required")
        self.grok_api_key = _clean_string(self.grok_api_key)
        self.grok_model = _clean_string(self.grok_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.grok_model,
            openai_api_key=self.grok_api_key,
            openai_base_url="https://api.x.ai/v1",
            openai_enable_json_mode=self.grok_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("grok_api_key")


class GroqSettings(BaseModel):
    """Groq API settings"""

    translate_engine_type: Literal["Groq"] = Field(default="Groq")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    groq_model: str = Field(
        default="llama-3-3-70b-versatile", description="Groq model to use"
    )
    groq_api_key: str | None = Field(
        default=None, description="API key for Groq service"
    )
    groq_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for Groq service"
    )

    def validate_settings(self) -> None:
        if not self.groq_api_key:
            raise ValueError("Groq API key is required")
        self.groq_api_key = _clean_string(self.groq_api_key)
        self.groq_model = _clean_string(self.groq_model)

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.groq_model,
            openai_api_key=self.groq_api_key,
            openai_base_url="https://api.groq.com/openai/v1",
            openai_enable_json_mode=self.groq_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("groq_api_key")


class QwenMtSettings(BaseModel):
    """QwenMt API settings"""

    translate_engine_type: Literal["QwenMt"] = Field(default="QwenMt")
    support_llm: Literal["yes", "no"] = Field(
        default="no", description="Whether the translator supports LLM"
    )

    qwenmt_model: str = Field(default="qwen-mt-plus", description="QwenMt model to use")
    qwenmt_base_url: str | None = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Base URL for QwenMt API",
    )
    qwenmt_api_key: str | None = Field(
        default=None, description="API key for QwenMt service"
    )
    ali_domains: str | None = Field(
        default="This sentence is extracted from a scientific paper. When translating, please pay close attention to the use of specialized troubleshooting terminologies and adhere to scientific sentence structures to maintain the technical rigor and precision of the original text.",
        description="the target domain to guide translation style for QwenMt service",
    )

    def validate_settings(self) -> None:
        logger.warning(
            "The current QwenMT is not fully adapted and does not support the glossary function at this time."
        )
        if not self.qwenmt_api_key:
            raise ValueError("QwenMt API key is required")
        self.qwenmt_api_key = _clean_string(self.qwenmt_api_key)
        self.qwenmt_base_url = _clean_string(self.qwenmt_base_url)
        self.qwenmt_model = _clean_string(self.qwenmt_model)
        self.ali_domains = _clean_string(self.ali_domains)


GUI_PASSWORD_FIELDS.append("qwenmt_api_key")
GUI_SENSITIVE_FIELDS.append("qwenmt_base_url")


class OpenAICompatibleSettings(BaseModel):
    """OpenAICompatible settings"""

    translate_engine_type: Literal["OpenAICompatible"] = Field(
        default="OpenAICompatible"
    )
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    openai_compatible_model: str = Field(
        default="gpt-4o-mini", description="OpenAI Compatible model to use"
    )
    openai_compatible_base_url: str | None = Field(
        default=None, description="Base URL for OpenAI Compatible service"
    )
    openai_compatible_api_key: str | None = Field(
        default=None, description="API key for OpenAI Compatible service"
    )
    openai_compatible_timeout: str | None = Field(
        default=None, description="Timeout (seconds) for OpenAI Compatible service"
    )
    openai_compatible_temperature: str | None = Field(
        default=None, description="Temperature for OpenAI Compatible service"
    )
    openai_compatible_reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort for OpenAI Compatible service (minimal/low/medium/high)",
    )
    openai_compatible_send_temperature: bool | None = Field(
        default=None, description="Send temperature to OpenAI Compatible service"
    )
    openai_compatible_send_reasoning_effort: bool | None = Field(
        default=None, description="Send reasoning effort to OpenAI Compatible service"
    )
    openai_compatible_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for OpenAI Compatible service"
    )

    def validate_settings(self) -> None:
        if not self.openai_compatible_api_key:
            raise ValueError("OpenAI Compatible API key is required")
        if not self.openai_compatible_base_url:
            raise ValueError("OpenAI Compatible base URL is required")
        if not self.openai_compatible_model:
            raise ValueError("OpenAI Compatible model is required")
        self.openai_compatible_api_key = _clean_string(self.openai_compatible_api_key)
        self.openai_compatible_base_url = _clean_url(self.openai_compatible_base_url)
        self.openai_compatible_model = _clean_string(self.openai_compatible_model)
        self.openai_compatible_timeout = _check_if_positive_float(
            _clean_string(self.openai_compatible_timeout), field="Timeout"
        )
        self.openai_compatible_temperature = _clean_string(
            self.openai_compatible_temperature
        )
        self.openai_compatible_reasoning_effort = _clean_string(
            self.openai_compatible_reasoning_effort
        )
        if self.openai_compatible_send_temperature:
            if not self.openai_compatible_temperature:
                raise ValueError(
                    "Temperature is required when send temperature is enabled"
                )
            try:
                float(self.openai_compatible_temperature)
            except ValueError as e:
                raise ValueError("Temperature must be a float") from e
        if (
            self.openai_compatible_send_reasoning_effort
            and not self.openai_compatible_reasoning_effort
        ):
            raise ValueError(
                "Reasoning effort is required when send reasoning effort is enabled"
            )

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.openai_compatible_model,
            openai_api_key=self.openai_compatible_api_key,
            openai_base_url=self.openai_compatible_base_url,
            openai_timeout=self.openai_compatible_timeout,
            openai_temperature=self.openai_compatible_temperature,
            openai_reasoning_effort=self.openai_compatible_reasoning_effort,
            openai_send_temprature=self.openai_compatible_send_temperature,
            openai_send_reasoning_effort=self.openai_compatible_send_reasoning_effort,
            openai_enable_json_mode=self.openai_compatible_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("openai_compatible_api_key")
GUI_SENSITIVE_FIELDS.append("openai_compatible_base_url")


class AliyunDashScopeSettings(BaseModel):
    """Aliyun DashScope settings"""

    translate_engine_type: Literal["AliyunDashScope"] = Field(default="AliyunDashScope")
    support_llm: Literal["yes", "no"] = Field(
        default="yes", description="Whether the translator supports LLM"
    )

    aliyun_dashscope_model: str = Field(
        default="qwen-plus-latest", description="Aliyun DashScope model to use"
    )
    aliyun_dashscope_base_url: str | None = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Base URL for Aliyun DashScope API",
    )
    aliyun_dashscope_api_key: str | None = Field(
        default=None, description="API key for Aliyun DashScope service"
    )
    aliyun_dashscope_timeout: str | None = Field(
        default="500", description="Timeout (seconds) for Aliyun DashScope service"
    )
    aliyun_dashscope_temperature: str | None = Field(
        default="0.0", description="Temperature for Aliyun DashScope service"
    )
    aliyun_dashscope_send_temperature: bool | None = Field(
        default=None, description="Send temperature to Aliyun DashScope service"
    )
    aliyun_dashscope_enable_json_mode: bool | None = Field(
        default=None, description="Enable JSON mode for Aliyun DashScope service"
    )

    def validate_settings(self) -> None:
        if not self.aliyun_dashscope_api_key:
            raise ValueError("Aliyun DashScope API key is required")
        if not self.aliyun_dashscope_base_url:
            raise ValueError("Aliyun DashScope base URL is required")
        if not self.aliyun_dashscope_model:
            raise ValueError("Aliyun DashScope model is required")
        self.aliyun_dashscope_api_key = _clean_string(self.aliyun_dashscope_api_key)
        self.aliyun_dashscope_base_url = _clean_url(self.aliyun_dashscope_base_url)
        self.aliyun_dashscope_model = _clean_string(self.aliyun_dashscope_model)
        self.aliyun_dashscope_timeout = _check_if_positive_float(
            _clean_string(self.aliyun_dashscope_timeout), field="Timeout"
        )
        self.aliyun_dashscope_temperature = _clean_string(
            self.aliyun_dashscope_temperature
        )
        if self.aliyun_dashscope_send_temperature:
            if not self.aliyun_dashscope_temperature:
                raise ValueError(
                    "Temperature is required when send temperature is enabled"
                )
            try:
                float(self.aliyun_dashscope_temperature)
            except ValueError as e:
                raise ValueError("Temperature must be a float") from e

    def transform(self) -> OpenAISettings:
        return OpenAISettings(
            openai_model=self.aliyun_dashscope_model,
            openai_api_key=self.aliyun_dashscope_api_key,
            openai_base_url=self.aliyun_dashscope_base_url,
            openai_timeout=self.aliyun_dashscope_timeout,
            openai_temperature=self.aliyun_dashscope_temperature,
            openai_send_temprature=self.aliyun_dashscope_send_temperature,
            openai_enable_json_mode=self.aliyun_dashscope_enable_json_mode,
        )


GUI_PASSWORD_FIELDS.append("aliyun_dashscope_api_key")


class ClaudeCodeSettings(BaseModel):
    """Claude Code settings"""

    translate_engine_type: Literal["ClaudeCode"] = Field(default="ClaudeCode")
    claude_code_path: str = Field(
        default="claude", description="Path to Claude Code CLI"
    )
    claude_code_model: str = Field(
        default="sonnet", description="Claude Code model to use"
    )

    def validate_settings(self):
        if not self.claude_code_path:
            raise ValueError("Claude Code path is required")


## Please add the translator configuration class above this location.

# 所有翻译引擎
TRANSLATION_ENGINE_SETTING_TYPE: TypeAlias = (
    SiliconFlowFreeSettings
    | OpenAISettings
    | AliyunDashScopeSettings
    | GoogleSettings
    | BingSettings
    | DeepLSettings
    | DeepSeekSettings
    | OllamaSettings
    | XinferenceSettings
    | AzureOpenAISettings
    | ModelScopeSettings
    | ZhipuSettings
    | SiliconFlowSettings
    | TencentSettings
    | GeminiSettings
    | AzureSettings
    | AnythingLLMSettings
    | DifySettings
    | GrokSettings
    | GroqSettings
    | QwenMtSettings
    | OpenAICompatibleSettings
    | ClaudeCodeSettings
)

# 不支持的翻译引擎
NOT_SUPPORTED_TRANSLATION_ENGINE_SETTING_TYPE: TypeAlias = NoneType

# 默认翻译引擎
_DEFAULT_TRANSLATION_ENGINE = SiliconFlowFreeSettings
assert len(_DEFAULT_TRANSLATION_ENGINE.model_fields) == 3, (
    "Default translation engine cannot have detail settings"
)

# The following is magic code,
# if you need to modify it,
# please contact the maintainer!

GUI_SENSITIVE_FIELDS.extend(GUI_PASSWORD_FIELDS)


@dataclass
class TranslationEngineMetadata:
    translate_engine_type: str
    cli_flag_name: str
    cli_detail_field_name: str | None
    setting_model_type: type[BaseModel]
    support_llm: bool

    def __init__(
        self,
        setting_model_type: type[BaseModel],
    ) -> None:
        self.translate_engine_type = setting_model_type.model_fields[
            "translate_engine_type"
        ].default
        self.cli_flag_name = self.translate_engine_type.lower()
        self.cli_detail_field_name = self.cli_flag_name + "_detail"
        self.setting_model_type = setting_model_type
        if len(setting_model_type.model_fields) == 1:
            self.cli_detail_field_name = None
        self.support_llm = (
            (sl := setting_model_type.model_fields.get("support_llm", None))
            and sl.default == "yes"
        ) or False


args = typing.get_args(TRANSLATION_ENGINE_SETTING_TYPE)

TRANSLATION_ENGINE_METADATA = [
    TranslationEngineMetadata(
        setting_model_type=arg,
    )
    for arg in args
]

TRANSLATION_ENGINE_METADATA_MAP = {
    metadata.translate_engine_type: metadata for metadata in TRANSLATION_ENGINE_METADATA
}


# auto check duplicate translation engine metadata
assert len(TRANSLATION_ENGINE_METADATA_MAP) == len(TRANSLATION_ENGINE_METADATA), (
    "Duplicate translation engine metadata"
)

# auto check duplicate cli flag name and cli detail field name
dedup_set = set()
for metadata in TRANSLATION_ENGINE_METADATA:
    if metadata.cli_flag_name in dedup_set:
        raise ValueError(f"Duplicate cli flag name: {metadata.cli_flag_name}")
    dedup_set.add(metadata.cli_flag_name)
    if metadata.cli_detail_field_name and metadata.cli_detail_field_name in dedup_set:
        raise ValueError(
            f"Duplicate cli detail field name: {metadata.cli_detail_field_name}"
        )
    dedup_set.add(metadata.cli_detail_field_name)
del dedup_set


_TERM_EXTRACTION_ENGINE_SETTING_TYPE: type[BaseModel] | None = None
for metadata in TRANSLATION_ENGINE_METADATA:
    if not metadata.support_llm:
        continue
    if _TERM_EXTRACTION_ENGINE_SETTING_TYPE is None:
        _TERM_EXTRACTION_ENGINE_SETTING_TYPE = metadata.setting_model_type
    else:
        _TERM_EXTRACTION_ENGINE_SETTING_TYPE = (
            _TERM_EXTRACTION_ENGINE_SETTING_TYPE | metadata.setting_model_type
        )

assert _TERM_EXTRACTION_ENGINE_SETTING_TYPE is not None, (
    "No LLM-capable translation engines configured"
)

# 术语提取引擎：仅包含 support_llm == \"yes\" 的翻译引擎设置类型
TERM_EXTRACTION_ENGINE_SETTING_TYPE: TypeAlias = _TERM_EXTRACTION_ENGINE_SETTING_TYPE


def _build_term_setting_model(
    setting_model_type: type[BaseModel],
) -> type[BaseModel]:
    """Dynamically build a term-extraction settings model with prefixed fields."""
    fields: dict[str, tuple[typing.Any, Field]] = {}
    base_to_term_field_map: dict[str, str] = {}

    for name, model_field in setting_model_type.model_fields.items():
        # Keep discriminator-related fields unchanged
        if name in ("translate_engine_type", "support_llm"):
            new_name = name
        else:
            new_name = f"term_{name}"

        base_to_term_field_map[name] = new_name

        fields[new_name] = (
            model_field.annotation,
            Field(
                default=model_field.default,
                description=model_field.description,
                default_factory=model_field.default_factory,
                alias=model_field.alias,
                discriminator=model_field.discriminator,
            ),
        )

    term_model_name = f"Term{setting_model_type.__name__}"
    TermModel = create_model(term_model_name, **fields)  # type: ignore[arg-type]  # noqa: N806

    # Set a meaningful docstring for the dynamically created term settings model
    # so that inspect.getdoc(TermModel) returns helpful information in CLI help.
    base_doc = getdoc(setting_model_type) or setting_model_type.__doc__ or ""
    if base_doc:
        TermModel.__doc__ = f"Term settings based on: {base_doc}"
    else:
        TermModel.__doc__ = (
            "Term settings model based on the base engine settings model."
        )

    def to_base_settings(self) -> BaseModel:
        """Convert term settings back to the base engine settings model."""
        data: dict[str, typing.Any] = {}
        for base_name, term_name in base_to_term_field_map.items():
            data[base_name] = getattr(self, term_name)
        return setting_model_type(**data)

    TermModel.to_base_settings = to_base_settings  # type: ignore[attr-defined]
    return TermModel


@dataclass
class TermTranslationEngineMetadata:
    translate_engine_type: str
    cli_flag_name: str
    cli_detail_field_name: str | None
    term_setting_model_type: type[BaseModel]


TERM_EXTRACTION_ENGINE_METADATA: list[TermTranslationEngineMetadata] = []

for metadata in TRANSLATION_ENGINE_METADATA:
    if not metadata.support_llm:
        continue
    term_setting_model_type = _build_term_setting_model(metadata.setting_model_type)
    TERM_EXTRACTION_ENGINE_METADATA.append(
        TermTranslationEngineMetadata(
            translate_engine_type=metadata.translate_engine_type,
            cli_flag_name=metadata.cli_flag_name,
            cli_detail_field_name=metadata.cli_detail_field_name,
            term_setting_model_type=term_setting_model_type,
        )
    )

TERM_EXTRACTION_ENGINE_METADATA_MAP = {
    metadata.translate_engine_type: metadata
    for metadata in TERM_EXTRACTION_ENGINE_METADATA
}


DEFAULT_TRANSLATION_ENGINE_METADATA = TRANSLATION_ENGINE_METADATA_MAP[
    _DEFAULT_TRANSLATION_ENGINE.model_fields["translate_engine_type"].default
]

if __name__ == "__main__":
    print(TRANSLATION_ENGINE_METADATA_MAP)
