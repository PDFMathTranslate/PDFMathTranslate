import importlib
import logging

from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.config.translate_engine_model import (
    NOT_SUPPORTED_TRANSLATION_ENGINE_SETTING_TYPE,
)
from pdf2zh.kernel.experimental.config.translate_engine_model import TRANSLATION_ENGINE_METADATA
from pdf2zh.kernel.experimental.config.translate_engine_model import TranslateEngineSettingError
from pdf2zh.kernel.experimental.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh.kernel.experimental.translator.base_translator import BaseTranslator
from pdf2zh.kernel.experimental.translator.rate_limiter.qps_rate_limiter import QPSRateLimiter

logger = logging.getLogger(__name__)


def get_rate_limiter(qps: int | None) -> BaseRateLimiter | None:
    """Create rate limiter based on qps value."""
    if qps and qps > 0:
        return QPSRateLimiter(qps)
    return None


def _create_translator_instance(
    settings: SettingsModel,
    translator_config,
    rate_limiter: BaseRateLimiter | None,
    enforce_glossary_support: bool = True,
) -> (BaseTranslator, int | None, int | None):
    """Create translator instance from translator_config.

    Args:
        settings: Global settings model.
        translator_config: Concrete translation engine settings instance.
        rate_limiter: Rate limiter for this translator.
        enforce_glossary_support: Whether to enforce glossary + LLM support check.
    """
    if isinstance(translator_config, NOT_SUPPORTED_TRANSLATION_ENGINE_SETTING_TYPE):
        raise TranslateEngineSettingError(
            f"{translator_config.translate_engine_type} is not supported, Please use other translator!"
        )

    for metadata in TRANSLATION_ENGINE_METADATA:
        if isinstance(translator_config, metadata.setting_model_type):
            translate_engine_type = metadata.translate_engine_type
            logger.info(f"Using {translate_engine_type} translator")
            model_name = f"pdf2zh.kernel.experimental.translator.translator_impl.{translate_engine_type.lower()}"
            module = importlib.import_module(model_name)

            if (
                enforce_glossary_support
                and settings.translation.glossaries
                and not metadata.support_llm
            ):
                raise TranslateEngineSettingError(
                    f"{translate_engine_type} does not support glossary. Please choose a different translator or remove the glossary."
                )
            temp_settings = settings.model_copy()
            temp_settings.translate_engine_settings = translator_config
            translator = getattr(module, f"{translate_engine_type}Translator")(
                temp_settings, rate_limiter
            )
            recommended_qps = None
            recommended_pool_max_workers = None
            if getattr(translator, "pdf2zh.kernel.experimental_recommended_qps", None):
                recommended_qps = translator.pdf2zh.kernel.experimental_recommended_qps
            if getattr(translator, "pdf2zh.kernel.experimental_recommended_pool_max_workers", None):
                recommended_pool_max_workers = (
                    translator.pdf2zh.kernel.experimental_recommended_pool_max_workers
                )

            # Health check: perform a short translation ignoring cache to validate translator availability
            translator.translate("Hello", ignore_cache=True)
            return translator, recommended_qps, recommended_pool_max_workers

    raise ValueError("No translator found")


def get_translator(settings: SettingsModel) -> BaseTranslator:
    """Get main translator instance according to translate_engine_settings."""
    translator_config = settings.translate_engine_settings
    rate_limiter = get_rate_limiter(settings.translation.qps)
    translator, recommended_qps, recommended_pool_max_workers = (
        _create_translator_instance(
            settings=settings,
            translator_config=translator_config,
            rate_limiter=rate_limiter,
            enforce_glossary_support=True,
        )
    )
    if recommended_qps:
        settings.translation.qps = recommended_qps
        logger.info(f"Updated qps to {recommended_qps}")
    if recommended_pool_max_workers:
        settings.translation.pool_max_workers = recommended_pool_max_workers
        logger.info(f"Updated pool max workers to {recommended_pool_max_workers}")
    return translator


def get_term_translator(settings: SettingsModel) -> BaseTranslator | None:
    """Get term-extraction translator instance if configured.

    This translator uses a potentially different engine and separate rate limit
    from the main translation engine.
    """
    translator_config = settings.term_extraction_engine_settings
    if translator_config is None:
        return None

    # Prefer dedicated term_qps, fallback to main qps when not set
    term_qps = settings.translation.term_qps or settings.translation.qps
    rate_limiter = get_rate_limiter(term_qps)

    translator, recommended_qps, recommended_pool_max_workers = (
        _create_translator_instance(
            settings=settings,
            translator_config=translator_config,
            rate_limiter=rate_limiter,
            enforce_glossary_support=False,
        )
    )
    if recommended_qps:
        settings.translation.term_qps = recommended_qps
        logger.info(f"Updated term qps to {recommended_qps}")
    if recommended_pool_max_workers:
        settings.translation.term_pool_max_workers = recommended_pool_max_workers
        logger.info(f"Updated term pool max workers to {recommended_pool_max_workers}")
    return translator
