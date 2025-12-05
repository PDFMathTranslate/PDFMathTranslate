import logging

import deepl
from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh.kernel.experimental.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class DeepLTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "deepl"
    # Normalize common variants to a canonical internal code before
    # mapping to DeepL specific enums.
    lang_map = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "zh-hk": "zh-HK",
        "pt-br": "pt-BR",
    }

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.client = deepl.Translator(
            settings.translate_engine_settings.deepl_auth_key
        )

    @staticmethod
    def _map_source_lang(lang: str | None) -> str | None:
        """
        Map internal language code to DeepL source_lang.

        If lang is None or "auto", return None to let DeepL auto-detect.
        """
        if lang is None:
            return None

        lang_lower = lang.lower()
        if lang_lower == "auto":
            return None

        # DeepL source languages: AR, BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HE,
        # HU, ID, IT, JA, KO, LT, LV, NB, NL, PL, PT, RO, RU, SK, SL, SV, TH, TR,
        # UK, VI, ZH
        source_map = {
            # Exact DeepL codes
            "ar": "AR",
            "bg": "BG",
            "cs": "CS",
            "da": "DA",
            "de": "DE",
            "el": "EL",
            "en": "EN",
            "es": "ES",
            "et": "ET",
            "fi": "FI",
            "fr": "FR",
            "he": "HE",
            "hu": "HU",
            "id": "ID",
            "it": "IT",
            "ja": "JA",
            "ko": "KO",
            "lt": "LT",
            "lv": "LV",
            "nb": "NB",
            "nl": "NL",
            "pl": "PL",
            "pt": "PT",
            "ro": "RO",
            "ru": "RU",
            "sk": "SK",
            "sl": "SL",
            "sv": "SV",
            "th": "TH",
            "tr": "TR",
            "uk": "UK",
            "vi": "VI",
            "zh": "ZH",
            # Common internal variants
            "zh-cn": "ZH",
            "zh-tw": "ZH",
            "zh-hk": "ZH",
            "pt-br": "PT",
            "pt-pt": "PT",
            "no": "NB",
        }

        mapped = source_map.get(lang_lower)
        if mapped is None:
            raise ValueError(f"Unsupported DeepL source_lang for lang_in={lang}")
        return mapped

    @staticmethod
    def _map_target_lang(lang: str) -> str:
        """
        Map internal language code to DeepL target_lang.

        English will default to EN-US by design.
        """
        if not lang:
            raise ValueError("lang_out must not be empty for DeepL target_lang")

        lang_lower = lang.lower()

        # DeepL target languages:
        # AR, BG, CS, DA, DE, EL, EN-GB, EN-US, ES, ES-419, ET, FI, FR, HE, HU, ID,
        # IT, JA, KO, LT, LV, NB, NL, PL, PT-BR, PT-PT, RO, RU, SK, SL, SV, TH, TR,
        # UK, VI, ZH, ZH-HANS, ZH-HANT
        target_map = {
            # Exact DeepL codes (allow users to set them directly)
            "ar": "AR",
            "bg": "BG",
            "cs": "CS",
            "da": "DA",
            "de": "DE",
            "el": "EL",
            "en-gb": "EN-GB",
            "en-us": "EN-US",
            "es": "ES",
            "es-419": "ES-419",
            "et": "ET",
            "fi": "FI",
            "fr": "FR",
            "he": "HE",
            "hu": "HU",
            "id": "ID",
            "it": "IT",
            "ja": "JA",
            "ko": "KO",
            "lt": "LT",
            "lv": "LV",
            "nb": "NB",
            "nl": "NL",
            "pl": "PL",
            "pt-br": "PT-BR",
            "pt-pt": "PT-PT",
            "ro": "RO",
            "ru": "RU",
            "sk": "SK",
            "sl": "SL",
            "sv": "SV",
            "th": "TH",
            "tr": "TR",
            "uk": "UK",
            "vi": "VI",
            "zh": "ZH",
            "zh-hans": "ZH-HANS",
            "zh-hant": "ZH-HANT",
            # Internal variants that need special mapping
            # English: default to EN-US
            "en": "EN-US",
            # Chinese: map variants to Simplified/Traditional
            "zh-cn": "ZH-HANS",
            "zh-tw": "ZH-HANT",
            "zh-hk": "ZH-HANT",
            # Portuguese: distinguish PT-PT vs PT-BR
            "pt": "PT-PT",
            "no": "NB",
        }

        mapped = target_map.get(lang_lower)
        if mapped is None:
            raise ValueError(f"Unsupported DeepL target_lang for lang_out={lang}")
        return mapped

    @retry(
        retry=retry_if_exception(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None):
        # Map internal language codes to DeepL specific enums.
        target_lang = self._map_target_lang(self.lang_out)
        source_lang = self._map_source_lang(self.lang_in)

        translate_kwargs = {"target_lang": target_lang}
        if source_lang is not None:
            translate_kwargs["source_lang"] = source_lang

        response = self.client.translate_text(text, **translate_kwargs)
        return response.text
