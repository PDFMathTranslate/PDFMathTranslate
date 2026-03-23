"""Unified service registry — single source of truth for available translators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceInfo:
    display: str
    name: str
    custom_prompt: bool


# Ordered list of all supported translation services.
# display:       human-readable label for UIs
# name:          internal identifier (must match translator.name in translator.py)
# custom_prompt: whether the service supports user-supplied prompt templates
SERVICES: list[ServiceInfo] = [
    ServiceInfo("Google", "google", False),
    ServiceInfo("Bing", "bing", False),
    ServiceInfo("DeepL", "deepl", False),
    ServiceInfo("DeepLX", "deeplx", False),
    ServiceInfo("Ollama", "ollama", True),
    ServiceInfo("Xinference", "xinference", True),
    ServiceInfo("OpenAI", "openai", True),
    ServiceInfo("AzureOpenAI", "azure-openai", True),
    ServiceInfo("Zhipu", "zhipu", True),
    ServiceInfo("ModelScope", "modelscope", True),
    ServiceInfo("SiliconFlow", "silicon", True),
    ServiceInfo("SiliconFlow Free", "siliconflowfree", True),
    ServiceInfo("Gemini", "gemini", True),
    ServiceInfo("Azure", "azure", False),
    ServiceInfo("Tencent", "tencent", False),
    ServiceInfo("Dify", "dify", False),
    ServiceInfo("AnythingLLM", "anythingllm", True),
    ServiceInfo("Argos Translate", "argos", False),
    ServiceInfo("Grok", "grok", True),
    ServiceInfo("Groq", "groq", True),
    ServiceInfo("DeepSeek", "deepseek", True),
    ServiceInfo("MiniMax", "minimax", True),
    ServiceInfo("OpenAI-compatible", "openailiked", True),
    ServiceInfo("Ali Qwen-Translation", "qwen-mt", True),
    ServiceInfo("302.AI", "302ai", True),
]

SERVICE_BY_NAME: dict[str, ServiceInfo] = {s.name: s for s in SERVICES}
