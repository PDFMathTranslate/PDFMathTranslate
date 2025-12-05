import asyncio
import cgi
import csv
import io
import logging
import shutil
import tempfile
import typing
import uuid
from enum import Enum
from pathlib import Path
from string import Template

import chardet
import gradio as gr
import requests
import yaml
from babeldoc import __version__ as babeldoc_version
from gradio_i18n import Translate
from gradio_pdf import PDF

from pdf2zh.kernel.experimental import __version__
from pdf2zh.kernel.experimental.config import ConfigManager
from pdf2zh.kernel.experimental.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.config.translate_engine_model import GUI_PASSWORD_FIELDS
from pdf2zh.kernel.experimental.config.translate_engine_model import GUI_SENSITIVE_FIELDS
from pdf2zh.kernel.experimental.config.translate_engine_model import TERM_EXTRACTION_ENGINE_METADATA
from pdf2zh.kernel.experimental.config.translate_engine_model import (
    TERM_EXTRACTION_ENGINE_METADATA_MAP,
)
from pdf2zh.kernel.experimental.config.translate_engine_model import TRANSLATION_ENGINE_METADATA
from pdf2zh.kernel.experimental.config.translate_engine_model import TRANSLATION_ENGINE_METADATA_MAP
from pdf2zh.kernel.experimental.const import DEFAULT_CONFIG_DIR
from pdf2zh.kernel.experimental.const import DEFAULT_CONFIG_FILE
from pdf2zh.kernel.experimental.high_level import TranslationError
from pdf2zh.kernel.experimental.high_level import do_translate_async_stream
from pdf2zh.kernel.experimental.i18n import LANGUAGES
from pdf2zh.kernel.experimental.i18n import gettext as _
from pdf2zh.kernel.experimental.i18n import update_current_languages

logger = logging.getLogger(__name__)


class SaveMode(Enum):
    """Enum for configuration save behavior."""

    follow_settings = "follow_settings"  # Follow disable_config_auto_save setting
    never = "never"  # Never save
    always = "always"  # Always save regardless of disable_config_auto_save


def get_translation_dic(file_path: Path):
    with file_path.open(encoding="utf-8", newline="\n") as f:
        return yaml.safe_load(f)


__gui_service_arg_names = []
__gui_term_service_arg_names = []
LLM_support_index_map = {}
# The following variables associate strings with specific languages
lang_map = {
    "English": "en",
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese - Hong Kong": "zh-HK",
    "Traditional Chinese - Taiwan": "zh-TW",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Russian": "ru",
    "Spanish": "es",
    "Portuguese": "pt",
    "Brazilian Portuguese": "pt-BR",
    "French": "fr",
    "Malay": "ms",
    "Indonesian": "id",
    "Turkmen": "tk",
    "Filipino (Tagalog)": "tl",
    "Vietnamese": "vi",
    "Kazakh (Latin)": "kk",
    "German": "de",
    "Dutch": "nl",
    "Irish": "ga",
    "Italian": "it",
    "Greek": "el",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Icelandic": "is",
    "Finnish": "fi",
    "Ukrainian": "uk",
    "Czech": "cs",
    "Romanian": "ro",  # Covers Romanian, Moldovan, Moldovan (Cyrillic)
    "Hungarian": "hu",
    "Slovak": "sk",
    "Croatian": "hr",  # Also listed later, keep first
    "Estonian": "et",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Belarusian": "be",
    "Macedonian": "mk",
    "Albanian": "sq",
    "Serbian (Cyrillic)": "sr",  # Covers Serbian (Latin) too
    "Slovenian": "sl",
    "Catalan": "ca",
    "Bulgarian": "bg",
    "Maltese": "mt",
    "Swahili": "sw",
    "Amharic": "am",
    "Oromo": "om",
    "Tigrinya": "ti",
    "Haitian Creole": "ht",
    "Latin": "la",
    "Lao": "lo",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Thai": "th",
    "Burmese": "my",
    "Tamil": "ta",
    "Telugu": "te",
    "Oriya": "or",  # Also listed later, keep first
    "Armenian": "hy",
    "Mongolian (Cyrillic)": "mn",
    "Georgian": "ka",
    "Khmer": "km",
    "Bosnian": "bs",
    "Luxembourgish": "lb",
    "Romansh": "rm",
    "Turkish": "tr",
    "Sinhala": "si",
    "Uzbek": "uz",
    "Kyrgyz": "ky",  # Listed as Kirghiz later, keep this one
    "Tajik": "tg",
    "Abkhazian": "ab",
    "Afar": "aa",
    "Afrikaans": "af",
    "Akan": "ak",
    "Aragonese": "an",
    "Avaric": "av",
    "Ewe": "ee",
    "Aymara": "ay",
    "Ojibwa": "oj",
    "Occitan": "oc",
    "Ossetian": "os",
    "Pali": "pi",
    "Bashkir": "ba",
    "Basque": "eu",
    "Breton": "br",
    "Chamorro": "ch",
    "Chechen": "ce",
    "Chuvash": "cv",
    "Tswana": "tn",
    "Ndebele, South": "nr",
    "Ndonga": "ng",
    "Faroese": "fo",
    "Fijian": "fj",
    "Frisian, Western": "fy",
    "Ganda": "lg",
    "Kongo": "kg",
    "Kalaallisut": "kl",
    "Church Slavic": "cu",
    "Guarani": "gn",
    "Interlingua": "ia",
    "Herero": "hz",
    "Kikuyu": "ki",
    "Rundi": "rn",
    "Kinyarwanda": "rw",
    "Galician": "gl",
    "Kanuri": "kr",
    "Cornish": "kw",
    "Komi": "kv",
    "Xhosa": "xh",
    "Corsican": "co",
    "Cree": "cr",
    "Quechua": "qu",
    "Kurdish (Latin)": "ku",
    "Kuanyama": "kj",
    "Limburgan": "li",
    "Lingala": "ln",
    "Manx": "gv",
    "Malagasy": "mg",
    "Marshallese": "mh",
    "Maori": "mi",
    "Navajo": "nv",
    "Nauru": "na",
    "Nyanja": "ny",
    "Norwegian Nynorsk": "nn",
    "Sardinian": "sc",
    "Northern Sami": "se",
    "Samoan": "sm",
    "Sango": "sg",
    "Shona": "sn",
    "Esperanto": "eo",
    "Scottish Gaelic": "gd",
    "Somali": "so",
    "Southern Sotho": "st",
    "Tatar": "tt",
    "Tahitian": "ty",
    "Tongan": "to",
    "Twi": "tw",
    "Walloon": "wa",
    "Welsh": "cy",
    "Venda": "ve",
    "Volapük": "vo",
    "Interlingue": "ie",
    "Hiri Motu": "ho",
    "Igbo": "ig",
    "Ido": "io",
    "Inuktitut": "iu",
    "Inupiaq": "ik",
    "Sichuan Yi": "ii",
    "Yoruba": "yo",
    "Zhuang": "za",
    "Tsonga": "ts",
    "Zulu": "zu",
}

rev_lang_map = {v: k for k, v in lang_map.items()}

# The following variable associate strings with page ranges
# Page map with fixed internal keys
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Range": None,  # User-defined range
}

def get_page_choices():
    """Get page range choices with translated labels"""
    return [
        (_("All"), "All"),
        (_("First"), "First"),
        (_("First 5 pages"), "First 5 pages"),
        (_("Range"), "Range"),
    ]

# Load configuration
config_manager = ConfigManager()
try:
    # Load configuration from files and environment variables
    settings = config_manager.initialize_cli_config()
    # Check if sensitive inputs should be disabled in GUI
    disable_sensitive_input = settings.gui_settings.disable_gui_sensitive_input
except Exception as e:
    logger.warning(f"Could not load initial config: {e}")
    settings = CLIEnvSettingsModel()
    disable_sensitive_input = False

# Define default values
default_lang_from = rev_lang_map.get(settings.translation.lang_in, "English")

default_lang_to = settings.translation.lang_out
for display_name, code in lang_map.items():
    if code == default_lang_to:
        default_lang_to = display_name
        break
else:
    default_lang_to = "Simplified Chinese"  # Fallback

# Available translation services
# This will eventually be dynamically determined based on available translators
available_services = [x.translate_engine_type for x in TRANSLATION_ENGINE_METADATA]

if settings.gui_settings.enabled_services:
    enabled_services = {
        x.lower() for x in settings.gui_settings.enabled_services.split(",")
    }
    available_services = [
        x for x in available_services if x.lower() in enabled_services
    ]

assert available_services, "No translation service is enabled"


disable_gui_sensitive_input = settings.gui_settings.disable_gui_sensitive_input


def download_with_limit(url: str, save_path: str, size_limit: int = None) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = Path(url).name
        filename = Path(filename).stem + ".pdf"
        save_path = Path(save_path).resolve()
        file_path = save_path / filename
        if not file_path.resolve().is_relative_to(save_path):
            raise gr.Error("Invalid filename")
        with file_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return file_path


def _prepare_input_file(
    file_type: str, file_input: str, link_input: str, output_dir: Path
) -> Path:
    """
    This function prepares the input file for translation.

    Inputs:
        - file_type: The type of file to translate (File or Link)
        - file_input: The path to the file to translate
        - link_input: The link to the file to translate
        - output_dir: The directory to save the file to

    Returns:
        - The path of the input file
    """
    if file_type == "File":
        if not file_input:
            raise gr.Error("No file input provided")
        file_path = shutil.copy(file_input, output_dir)
    else:
        if not link_input:
            raise gr.Error("No link input provided")
        try:
            file_path = download_with_limit(link_input, output_dir)
        except Exception as e:
            raise gr.Error(f"Error downloading file: {e}") from e

    return Path(file_path)


def _validate_rate_limit_inputs(
    true_rate_limit_mode: str, **inputs
) -> tuple[bool, str]:
    """
    Validate rate limit inputs

    Returns:
        tuple: (is_valid, error_message)
    """
    if true_rate_limit_mode == "RPM":
        rpm = inputs.get("rpm_input", 0)
        if not isinstance(rpm, int | float) or rpm <= 0:
            return False, "RPM must be a positive integer"

        if isinstance(rpm, float):
            if not rpm.is_integer():
                return False, "RPM must be a positive integer"

    elif true_rate_limit_mode == "Concurrent Threads":
        threads = inputs.get("concurrent_threads", 0)
        if not isinstance(threads, int | float) or threads <= 0:
            return False, "Concurrent threads must be a positive integer"

        if isinstance(threads, float):
            if not threads.is_integer():
                return False, "Concurrent threads must be a positive integer"

    elif true_rate_limit_mode == "Custom":
        qps = inputs.get("custom_qps", 0)
        pool_workers = inputs.get("custom_pool_workers")

        if not isinstance(qps, int | float) or qps <= 0:
            return False, "QPS must be a positive integer"

        if isinstance(qps, float):
            if not qps.is_integer():
                return False, "QPS must be a positive integer"

        if pool_workers is not None and (
            not isinstance(pool_workers, int | float) or pool_workers < 0
        ):
            return False, "Pool workers must be a non-negative integer"

        if isinstance(pool_workers, float):
            if not pool_workers.is_integer():
                return False, "Pool workers must be a non-negative integer"

    return True, ""


def _calculate_rate_limit_params(
    rate_limit_mode: str, ui_inputs: dict, default_qps: int = 4
) -> tuple[int, int | None]:
    """
    Calculate QPS and pool workers based on rate limit mode

    Args:
        rate_limit_mode: Rate limit mode ("RPM", "Concurrent Threads", "Custom")
        ui_inputs: User input parameters dictionary
        default_qps: Default QPS value

    Returns:
        tuple: (qps, pool_max_workers)

    Raises:
        ValueError: When input parameter validation fails
    """
    # Validate input parameters
    is_valid, error_msg = _validate_rate_limit_inputs(
        true_rate_limit_mode=rate_limit_mode, **ui_inputs
    )
    if not is_valid:
        logger.warning(f"Rate limit validation failed: {error_msg}")
        raise ValueError(error_msg)

    if rate_limit_mode == "RPM":
        rpm: int = ui_inputs.get("rpm_input", 240)
        qps = max(1, rpm // 60)
        pool_workers = min(1000, qps * 10)

    elif rate_limit_mode == "Concurrent Threads":
        threads: int = ui_inputs.get("concurrent_threads_input", 40)
        # Ensure at least 1 worker, at most 1000 workers, using a safer calculation method
        pool_workers = min(1000, max(1, min(int(threads * 0.9), max(1, threads - 20))))
        qps = max(1, pool_workers)

    else:  # Custom
        qps = ui_inputs.get("custom_qps_input", default_qps)
        pool_workers = ui_inputs.get("custom_pool_workers")
        qps = int(qps)
        pool_workers = int(pool_workers) if pool_workers and pool_workers > 0 else None

    logger.info(f"QPS: {qps}, Pool Workers: {pool_workers}")

    return qps, pool_workers if pool_workers and pool_workers > 0 else None


def _build_translate_settings(
    base_settings: CLIEnvSettingsModel,
    file_path: Path,
    output_dir: Path,
    save_mode: SaveMode,
    ui_inputs: dict,
) -> SettingsModel:
    """
    This function builds translation settings from UI inputs.

    Inputs:
        - base_settings: The base settings model to build upon
        - file_path: The path to the input file
        - output_dir: The output directory
        - save_mode: SaveMode enum indicating when to save config
        - ui_inputs: A dictionary of UI inputs

    Returns:
        - A configured SettingsModel instance
    """
    # Clone base settings to avoid modifying the original
    translate_settings = base_settings.clone()
    original_output = translate_settings.translation.output
    original_pages = translate_settings.pdf.pages
    original_gui_settings = config_manager.config_cli_settings.gui_settings

    # Extract UI values
    service = ui_inputs.get("service")
    lang_from = ui_inputs.get("lang_from")
    lang_to = ui_inputs.get("lang_to")
    page_range = ui_inputs.get("page_range")
    page_input = ui_inputs.get("page_input")
    prompt = ui_inputs.get("prompt")
    ignore_cache = ui_inputs.get("ignore_cache")

    # PDF Output Options
    no_mono = ui_inputs.get("no_mono")
    no_dual = ui_inputs.get("no_dual")
    dual_translate_first = ui_inputs.get("dual_translate_first")
    use_alternating_pages_dual = ui_inputs.get("use_alternating_pages_dual")
    watermark_output_mode = ui_inputs.get("watermark_output_mode")

    # Rate Limit Options
    rate_limit_mode = ui_inputs.get("rate_limit_mode")

    # Advanced Translation Options
    min_text_length = ui_inputs.get("min_text_length")
    rpc_doclayout = ui_inputs.get("rpc_doclayout")
    enable_auto_term_extraction = ui_inputs.get("enable_auto_term_extraction")
    primary_font_family = ui_inputs.get("primary_font_family")

    # Advanced PDF Options
    skip_clean = ui_inputs.get("skip_clean")
    disable_rich_text_translate = ui_inputs.get("disable_rich_text_translate")
    enhance_compatibility = ui_inputs.get("enhance_compatibility")
    split_short_lines = ui_inputs.get("split_short_lines")
    short_line_split_factor = ui_inputs.get("short_line_split_factor")
    translate_table_text = ui_inputs.get("translate_table_text")
    skip_scanned_detection = ui_inputs.get("skip_scanned_detection")
    ocr_workaround = ui_inputs.get("ocr_workaround")
    max_pages_per_part = ui_inputs.get("max_pages_per_part")
    formular_font_pattern = ui_inputs.get("formular_font_pattern")
    formular_char_pattern = ui_inputs.get("formular_char_pattern")
    auto_enable_ocr_workaround = ui_inputs.get("auto_enable_ocr_workaround")
    only_include_translated_page = ui_inputs.get("only_include_translated_page")

    # BabelDOC v0.5.1 new options
    merge_alternating_line_numbers = ui_inputs.get("merge_alternating_line_numbers")
    remove_non_formula_lines = ui_inputs.get("remove_non_formula_lines")
    non_formula_line_iou_threshold = ui_inputs.get("non_formula_line_iou_threshold")
    figure_table_protection_threshold = ui_inputs.get(
        "figure_table_protection_threshold"
    )
    skip_formula_offset_calculation = ui_inputs.get("skip_formula_offset_calculation")

    # Term extraction options
    term_service = ui_inputs.get("term_service")
    term_rate_limit_mode = ui_inputs.get("term_rate_limit_mode")
    term_rpm_input = ui_inputs.get("term_rpm_input")
    term_concurrent_threads = ui_inputs.get("term_concurrent_threads")
    term_custom_qps = ui_inputs.get("term_custom_qps")
    term_custom_pool_workers = ui_inputs.get("term_custom_pool_workers")

    # New input for custom_system_prompt
    custom_system_prompt_input = ui_inputs.get("custom_system_prompt_input")
    glossaries = ui_inputs.get("glossaries")
    save_auto_extracted_glossary = ui_inputs.get("save_auto_extracted_glossary")

    # Map UI language selections to language codes
    source_lang = lang_map.get(lang_from, "auto")
    target_lang = lang_map.get(lang_to, "zh")

    # Set up page selection
    if page_range == "Range" and page_input:
        pages = page_input  # The backend parser handles the format
    else:
        # Use predefined ranges from page_map
        selected_pages = page_map[page_range]
        if selected_pages is None:
            pages = None  # All pages
        else:
            # Convert page indices to comma-separated string
            pages = ",".join(
                str(p + 1) for p in selected_pages
            )  # +1 because UI is 1-indexed

    # Update settings with UI values
    translate_settings.basic.input_files = {str(file_path)}
    translate_settings.report_interval = 0.2
    translate_settings.translation.lang_in = source_lang
    translate_settings.translation.lang_out = target_lang
    translate_settings.translation.output = str(output_dir)
    translate_settings.translation.ignore_cache = ignore_cache

    # Update Translation Settings
    if min_text_length is not None:
        translate_settings.translation.min_text_length = int(min_text_length)
    if rpc_doclayout:
        translate_settings.translation.rpc_doclayout = rpc_doclayout

    # UI uses positive switch, config uses negative flag, so we invert here
    if enable_auto_term_extraction is not None:
        translate_settings.translation.no_auto_extract_glossary = (
            not enable_auto_term_extraction
        )
    if primary_font_family:
        if primary_font_family == "Auto":
            translate_settings.translation.primary_font_family = None
        else:
            translate_settings.translation.primary_font_family = primary_font_family

    # Calculate and update rate limit settings
    if service != "SiliconFlowFree":
        qps, pool_workers = _calculate_rate_limit_params(
            rate_limit_mode, ui_inputs, translate_settings.translation.qps or 4
        )

        # Update translation settings
        translate_settings.translation.qps = int(qps)
        translate_settings.translation.pool_max_workers = (
            int(pool_workers) if pool_workers is not None else None
        )

    # Calculate and update term extraction rate limit settings
    if term_rate_limit_mode:
        term_rate_inputs = {
            "rpm_input": term_rpm_input,
            "concurrent_threads": term_concurrent_threads,
            "custom_qps": term_custom_qps,
            "custom_pool_workers": term_custom_pool_workers,
        }
        term_qps, term_pool_workers = _calculate_rate_limit_params(
            term_rate_limit_mode,
            term_rate_inputs,
            translate_settings.translation.term_qps
            or translate_settings.translation.qps
            or 4,
        )
        translate_settings.translation.term_qps = int(term_qps)
        translate_settings.translation.term_pool_max_workers = (
            int(term_pool_workers) if term_pool_workers is not None else None
        )

    # Reset all term extraction engine flags
    for term_metadata in TERM_EXTRACTION_ENGINE_METADATA:
        term_flag_name = f"term_{term_metadata.cli_flag_name}"
        if hasattr(translate_settings, term_flag_name):
            setattr(translate_settings, term_flag_name, False)

    # Configure term extraction engine settings from UI when not following main engine
    if (
        term_service
        and term_service != "Follow main translation engine"
        and not translate_settings.translation.no_auto_extract_glossary
        and term_service in TERM_EXTRACTION_ENGINE_METADATA_MAP
    ):
        term_metadata = TERM_EXTRACTION_ENGINE_METADATA_MAP[term_service]

        # Enable selected term extraction engine flag
        term_flag_name = f"term_{term_metadata.cli_flag_name}"
        if hasattr(translate_settings, term_flag_name):
            setattr(translate_settings, term_flag_name, True)

        # Update term extraction engine detail settings
        if term_metadata.cli_detail_field_name:
            term_detail_field_name = f"term_{term_metadata.cli_detail_field_name}"
            term_detail_settings = getattr(translate_settings, term_detail_field_name)
            term_model_type = term_metadata.term_setting_model_type

            for field_name, field in term_model_type.model_fields.items():
                if field_name in ("translate_engine_type", "support_llm"):
                    continue

                value = ui_inputs.get(field_name)
                if value is None:
                    continue

                type_hint = field.annotation
                original_type = typing.get_origin(type_hint)
                type_args = typing.get_args(type_hint)

                if type_hint is str or str in type_args:
                    pass
                elif type_hint is int or int in type_args:
                    value = int(value)
                elif type_hint is bool or bool in type_args:
                    value = bool(value)
                else:
                    raise Exception(
                        f"Unsupported type {type_hint} for field {field_name} in gui term extraction engine settings"
                    )

                setattr(term_detail_settings, field_name, value)

    # Update PDF Settings
    translate_settings.pdf.pages = pages
    translate_settings.pdf.no_mono = no_mono
    translate_settings.pdf.no_dual = no_dual
    translate_settings.pdf.dual_translate_first = dual_translate_first
    translate_settings.pdf.use_alternating_pages_dual = use_alternating_pages_dual

    # Map watermark mode from UI to enum
    translate_settings.pdf.watermark_output_mode = (
        watermark_output_mode.lower().replace(" ", "_")
    )

    # Update Advanced PDF Settings
    translate_settings.pdf.skip_clean = skip_clean
    translate_settings.pdf.disable_rich_text_translate = disable_rich_text_translate
    translate_settings.pdf.enhance_compatibility = enhance_compatibility
    translate_settings.pdf.split_short_lines = split_short_lines
    translate_settings.pdf.ocr_workaround = ocr_workaround
    if short_line_split_factor is not None:
        translate_settings.pdf.short_line_split_factor = float(short_line_split_factor)

    translate_settings.pdf.translate_table_text = translate_table_text
    translate_settings.pdf.skip_scanned_detection = skip_scanned_detection
    translate_settings.pdf.auto_enable_ocr_workaround = auto_enable_ocr_workaround
    translate_settings.pdf.only_include_translated_page = only_include_translated_page

    if max_pages_per_part is not None and max_pages_per_part > 0:
        translate_settings.pdf.max_pages_per_part = int(max_pages_per_part)

    if formular_font_pattern:
        translate_settings.pdf.formular_font_pattern = formular_font_pattern

    if formular_char_pattern:
        translate_settings.pdf.formular_char_pattern = formular_char_pattern

    # Apply BabelDOC v0.5.1 new options
    translate_settings.pdf.no_merge_alternating_line_numbers = (
        not merge_alternating_line_numbers
    )
    translate_settings.pdf.no_remove_non_formula_lines = not remove_non_formula_lines
    if non_formula_line_iou_threshold is not None:
        translate_settings.pdf.non_formula_line_iou_threshold = float(
            non_formula_line_iou_threshold
        )
    if figure_table_protection_threshold is not None:
        translate_settings.pdf.figure_table_protection_threshold = float(
            figure_table_protection_threshold
        )
    translate_settings.pdf.skip_formula_offset_calculation = (
        skip_formula_offset_calculation
    )

    assert service in TRANSLATION_ENGINE_METADATA_MAP, "UNKNOW TRANSLATION ENGINE!"

    for metadata in TRANSLATION_ENGINE_METADATA:
        cli_flag = metadata.cli_flag_name
        setattr(translate_settings, cli_flag, False)

    metadata = TRANSLATION_ENGINE_METADATA_MAP[service]
    cli_flag = metadata.cli_flag_name
    setattr(translate_settings, cli_flag, True)
    if metadata.cli_detail_field_name:
        detail_setting = getattr(translate_settings, metadata.cli_detail_field_name)
        if metadata.setting_model_type:
            for field_name in metadata.setting_model_type.model_fields:
                if field_name == "translate_engine_type" or field_name == "support_llm":
                    continue
                if disable_gui_sensitive_input:
                    if field_name in GUI_PASSWORD_FIELDS:
                        continue
                    if field_name in GUI_SENSITIVE_FIELDS:
                        continue
                value = ui_inputs.get(field_name)
                type_hint = detail_setting.model_fields[field_name].annotation
                original_type = typing.get_origin(type_hint)
                type_args = typing.get_args(type_hint)
                if type_hint is str or str in type_args:
                    pass
                elif type_hint is int or int in type_args:
                    value = int(value)
                elif type_hint is bool or bool in type_args:
                    value = bool(value)
                else:
                    raise Exception(
                        f"Unsupported type {type_hint} for field {field_name} in gui translation engine settings"
                    )
                setattr(detail_setting, field_name, value)

    # Add custom prompt if provided
    if prompt:
        # This might need adjustment based on how prompt is handled in the new system
        translate_settings.custom_prompt = Template(prompt)

    # Add custom system prompt if provided
    if custom_system_prompt_input:
        translate_settings.translation.custom_system_prompt = custom_system_prompt_input
    else:
        translate_settings.translation.custom_system_prompt = None

    if glossaries:
        translate_settings.translation.glossaries = glossaries
    else:
        translate_settings.translation.glossaries = None

    translate_settings.translation.save_auto_extracted_glossary = (
        save_auto_extracted_glossary
    )

    # Validate settings before proceeding
    try:
        translate_settings.validate_settings()
        temp_settings = translate_settings.to_settings_model()
        translate_settings.translation.output = original_output
        translate_settings.pdf.pages = original_pages
        translate_settings.gui_settings = original_gui_settings
        translate_settings.basic.gui = False
        translate_settings.basic.debug = False
        translate_settings.translation.glossaries = None

        # Determine if config should be saved based on save_mode
        should_save = False
        if save_mode == SaveMode.always:
            should_save = True
        elif save_mode == SaveMode.follow_settings:
            should_save = not temp_settings.gui_settings.disable_config_auto_save
        # SaveMode.never: should_save remains False

        if should_save:
            config_manager.write_user_default_config_file(settings=translate_settings)
            global settings
            settings = translate_settings
        temp_settings.validate_settings()
        return temp_settings
    except ValueError as e:
        raise gr.Error(f"Invalid settings: {e}") from e


def _build_glossary_list(glossary_file, service_name=None):
    if not LLM_support_index_map.get(service_name, False):
        return None
    glossary_list = []
    if glossary_file is None:
        return None
    for file in glossary_file:
        try:
            f = io.StringIO(file.decode(chardet.detect(file)["encoding"]))
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".csv"
            ) as temp_file:
                temp_file.write(f.getvalue())
                f.close()
            glossary_list.append(temp_file.name)
        except (UnicodeDecodeError, csv.Error, KeyError) as e:
            logger.error(f"Error processing glossary file: {e}")
            gr.Error(f"Failed to process glossary file: {e}")
    return ",".join(glossary_list)


def build_ui_inputs(*args):
    """
    Build ui_inputs dictionary from *args.

    Args:
        *args: UI setting controls in the following order:
            service, lang_from, lang_to, page_range, page_input,
            no_mono, no_dual, dual_translate_first, use_alternating_pages_dual, watermark_output_mode,
            rate_limit_mode, rpm_input, concurrent_threads_input, custom_qps_input, custom_pool_max_workers_input,
            prompt, min_text_length, rpc_doclayout, custom_system_prompt_input, glossary_file,
            save_auto_extracted_glossary, enable_auto_term_extraction, primary_font_family, skip_clean,
            disable_rich_text_translate, enhance_compatibility, split_short_lines, short_line_split_factor,
            translate_table_text, skip_scanned_detection, max_pages_per_part, formular_font_pattern,
            formular_char_pattern, ignore_cache, state, ocr_workaround, auto_enable_ocr_workaround,
            only_include_translated_page, merge_alternating_line_numbers, remove_non_formula_lines,
            non_formula_line_iou_threshold, figure_table_protection_threshold, skip_formula_offset_calculation,
            term_service, term_rate_limit_mode, term_rpm_input, term_concurrent_threads_input,
            term_custom_qps_input, term_custom_pool_max_workers_input, *translation_engine_arg_inputs

    Returns:
        dict: ui_inputs dictionary with all UI settings
    """
    # Fixed parameter names in order (excluding translation_engine_arg_inputs)
    fixed_param_names = [
        "service",
        "lang_from",
        "lang_to",
        "page_range",
        "page_input",
        "no_mono",
        "no_dual",
        "dual_translate_first",
        "use_alternating_pages_dual",
        "watermark_output_mode",
        "rate_limit_mode",
        "rpm_input",
        "concurrent_threads",  # mapped from concurrent_threads_input
        "custom_qps",  # mapped from custom_qps_input
        "custom_pool_workers",  # mapped from custom_pool_max_workers_input
        "prompt",
        "min_text_length",
        "rpc_doclayout",
        "custom_system_prompt_input",
        "glossary_file",  # will be converted to glossaries
        "save_auto_extracted_glossary",
        "enable_auto_term_extraction",
        "primary_font_family",
        "skip_clean",
        "disable_rich_text_translate",
        "enhance_compatibility",
        "split_short_lines",
        "short_line_split_factor",
        "translate_table_text",
        "skip_scanned_detection",
        "max_pages_per_part",
        "formular_font_pattern",
        "formular_char_pattern",
        "ignore_cache",
        "state",
        "ocr_workaround",
        "auto_enable_ocr_workaround",
        "only_include_translated_page",
        "merge_alternating_line_numbers",
        "remove_non_formula_lines",
        "non_formula_line_iou_threshold",
        "figure_table_protection_threshold",
        "skip_formula_offset_calculation",
        "term_service",
        "term_rate_limit_mode",
        "term_rpm_input",
        "term_concurrent_threads",
        "term_custom_qps",
        "term_custom_pool_workers",
    ]

    # Split args into fixed params and translation_engine_arg_inputs
    num_fixed = len(fixed_param_names)
    fixed_args = args[:num_fixed]
    translation_engine_arg_inputs = args[num_fixed:]

    # Build ui_inputs dictionary
    ui_inputs = {}
    for param_name, arg_value in zip(fixed_param_names, fixed_args, strict=False):
        ui_inputs[param_name] = arg_value

    # Convert glossary_file to glossaries
    service = ui_inputs["service"]
    glossary_file = ui_inputs["glossary_file"]
    ui_inputs["glossaries"] = _build_glossary_list(glossary_file, service)

    # Add translation engine args (main translator + term translator detail settings)
    main_detail_count = len(__gui_service_arg_names)
    term_detail_count = len(__gui_term_service_arg_names)

    main_detail_inputs = translation_engine_arg_inputs[:main_detail_count]
    term_detail_inputs = translation_engine_arg_inputs[
        main_detail_count : main_detail_count + term_detail_count
    ]

    for arg_name, arg_input in zip(
        __gui_service_arg_names, main_detail_inputs, strict=False
    ):
        ui_inputs[arg_name] = arg_input

    for arg_name, arg_input in zip(
        __gui_term_service_arg_names, term_detail_inputs, strict=False
    ):
        ui_inputs[arg_name] = arg_input

    return ui_inputs


async def _run_translation_task(
    settings: SettingsModel, file_path: Path, state: dict, progress: gr.Progress
) -> tuple[Path | None, Path | None, Path | None, dict | None]:
    """
    This function runs the translation task and handles progress updates.

    Inputs:
        - settings: The translation settings
        - file_path: The path to the input file
        - state: The state dictionary for tracking the task
        - progress: The Gradio progress bar

    Returns:
        - A tuple of (mono_pdf_path, dual_pdf_path, glossary_path, token_usage)
    """
    mono_path = None
    dual_path = None
    glossary_path = None
    token_usage = None

    try:
        settings.basic.input_files = set()
        async for event in do_translate_async_stream(settings, file_path):
            if event["type"] in (
                "progress_start",
                "progress_update",
                "progress_end",
            ):
                # Update progress bar
                desc = event["stage"]
                progress_value = event["overall_progress"] / 100.0
                part_index = event["part_index"]
                total_parts = event["total_parts"]
                stage_current = event["stage_current"]
                stage_total = event["stage_total"]
                desc = f"{desc} ({part_index}/{total_parts}, {stage_current}/{stage_total})"
                logger.info(f"Progress: {progress_value}, {desc}")
                progress(progress_value, desc=desc)
            elif event["type"] == "finish":
                # Extract result paths
                result = event["translate_result"]
                mono_path = result.mono_pdf_path
                dual_path = result.dual_pdf_path
                glossary_path = result.auto_extracted_glossary_path
                token_usage = event.get("token_usage", {})
                progress(1.0, desc=_("Translation complete!"))
                break
            elif event["type"] == "error":
                # Handle error event
                error_msg = event.get("error", "Unknown error")
                error_details = event.get("details", "")
                # error_str = f"{error_msg}" + (
                #     f": {error_details}" if error_details else ""
                # )
                raise gr.Error(f"Translation error: {error_msg}")
    except asyncio.CancelledError:
        # Handle task cancellation - let translate_file handle the UI updates
        logger.info(
            f"Translation for session {state.get('session_id', 'unknown')} was cancelled"
        )
        raise  # Re-raise for the calling function to handle
    except TranslationError as e:
        # Handle structured translation errors
        logger.error(f"Translation error: {e}")
        raise gr.Error(f"Translation error: {e.message}") from e
    except gr.Error as e:
        # Handle Gradio errors
        logger.error(f"Gradio error: {e}")
        raise
    except Exception as e:
        # Handle other exceptions
        logger.error(f"Error in _run_translation_task: {e}", exc_info=True)
        raise gr.Error(f"Translation failed: {e}") from e

    return mono_path, dual_path, glossary_path, token_usage


async def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    if "current_task" not in state or state["current_task"] is None:
        return

    logger.info(
        f"Stopping translation for session {state.get('session_id', 'unknown')}"
    )
    # Cancel the task
    try:
        state["current_task"].cancel()
        # Wait briefly for cancellation to take effect
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error stopping translation: {e}")
    finally:
        state["current_task"] = None


async def translate_file(
    file_type,
    file_input,
    link_input,
    *ui_args,
    progress=None,
):
    """
    This function translates a PDF file from one language to another using the new architecture.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - *ui_args: UI setting controls (see build_ui_inputs for details)
        - progress: The progress bar

    Returns:
        - The translated mono PDF file
        - The preview PDF file
        - The translated dual PDF file
        - The visibility state of the mono PDF output
        - The visibility state of the dual PDF output
        - The visibility state of the output title
    """
    # Setup progress tracking
    if progress is None:
        progress = gr.Progress()

    # Build ui_inputs from *args
    ui_inputs = build_ui_inputs(*ui_args)
    state = ui_inputs["state"]

    # Initialize session and output directory
    session_id = str(uuid.uuid4())
    state["session_id"] = session_id

    # Track progress
    progress(0, desc=_("Starting translation..."))

    # Prepare output directory
    output_dir = Path("pdf2zh_files") / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Step 1: Prepare input file
        file_path = _prepare_input_file(file_type, file_input, link_input, output_dir)

        # Step 2: Build translation settings
        translate_settings = _build_translate_settings(
            settings.clone(), file_path, output_dir, SaveMode.follow_settings, ui_inputs
        )

        # Step 3: Create and run the translation task
        task = asyncio.create_task(
            _run_translation_task(translate_settings, file_path, state, progress)
        )
        state["current_task"] = task

        # Wait for the translation to complete
        mono_path, dual_path, glossary_path, token_usage = await task
        if not mono_path or not mono_path.exists():
            mono_path = None
        else:
            mono_path = mono_path.as_posix()
        if not dual_path or not dual_path.exists():
            dual_path = None
        else:
            dual_path = dual_path.as_posix()

        if not glossary_path or not glossary_path.exists():
            glossary_path = None
        else:
            glossary_path = glossary_path.as_posix()

        token_info = ""
        if token_usage:
            token_info = "\n\n**Token Usage:**\n"  # noqa: S105, this is not a hardcoded password
            total_usage = {
                "total": 0,
                "prompt": 0,
                "cache_hit_prompt": 0,
                "completion": 0,
            }
            if "main" in token_usage:
                m = token_usage["main"]
                token_info += f"- Main: Total {m['total']} (Prompt {m['prompt']}, Cache Hit Prompt {m['cache_hit_prompt']}, Completion {m['completion']})\n"
                total_usage["total"] += m["total"]
                total_usage["prompt"] += m["prompt"]
                total_usage["cache_hit_prompt"] += m["cache_hit_prompt"]
                total_usage["completion"] += m["completion"]
            if "term" in token_usage:
                t = token_usage["term"]
                token_info += f"- Term: Total {t['total']} (Prompt {t['prompt']}, Cache Hit Prompt {t['cache_hit_prompt']}, Completion {t['completion']})\n"
                total_usage["total"] += t["total"]
                total_usage["prompt"] += t["prompt"]
                total_usage["cache_hit_prompt"] += t["cache_hit_prompt"]
                total_usage["completion"] += t["completion"]
            token_info += f"- Total: Total {total_usage['total']} (Prompt {total_usage['prompt']}, Cache Hit Prompt {total_usage['cache_hit_prompt']}, Completion {total_usage['completion']})\n"
            logger.info(f"Token usage: {token_info}")
        # Build success UI updates
        return (
            str(mono_path) if mono_path else None,  # Output mono file
            str(mono_path) if mono_path else dual_path,  # Preview
            str(dual_path) if dual_path else None,  # Output dual file
            str(glossary_path) if glossary_path else None,  # Output dual file
            gr.update(visible=bool(mono_path)),  # Show mono download if available
            gr.update(visible=bool(dual_path)),  # Show dual download if available
            gr.update(
                visible=bool(glossary_path)
            ),  # Show glossary download if available
            gr.update(
                value=f"{_('## Translated')}{token_info}",
                visible=bool(mono_path or dual_path),
            ),  # Show output title if any output
        )
    except asyncio.CancelledError:
        gr.Info(_("Translation cancelled"))
        # Return None for all outputs if cancelled
        return (
            None,
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    except gr.Error:
        # Re-raise Gradio errors without modification
        raise
    except Exception as e:
        # Catch any other errors and wrap in gr.Error
        logger.exception(f"Error in translate_file: {e}")
        raise gr.Error(f"Translation failed: {e}") from e
    finally:
        # Clear task reference
        state["current_task"] = None


def save_config(
    *ui_args,
    progress=None,
):
    """
    This function saves the translation configuration.

    Inputs:
        - *ui_args: UI setting controls (see build_ui_inputs for details)
        - progress: The progress bar
    """
    # Setup progress tracking
    if progress is None:
        progress = gr.Progress()

    # Build ui_inputs from *args
    ui_inputs = build_ui_inputs(*ui_args)

    # Track progress
    progress(0, desc=_("Saving configuration..."))

    # Prepare output directory
    output_dir = Path("pdf2zh_files")

    _build_translate_settings(
        settings.clone(), config_fake_pdf_path, output_dir, SaveMode.always, ui_inputs
    )

    # Show success message
    gr.Info(_("Configuration saved to: {path}").format(path=DEFAULT_CONFIG_FILE))


# Custom theme definition
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }
    """

# Build paths to resources
current_dir = Path(__file__).parent
assets_dir = current_dir / "assets"
logo_path = assets_dir / "powered_by_siliconflow_light.png"
translation_file_path = current_dir / "gui_translation.yaml"
config_fake_pdf_path = DEFAULT_CONFIG_DIR / "config.fake.pdf"

if not config_fake_pdf_path.exists():
    with config_fake_pdf_path.open("w") as f:
        f.write("This is a fake PDF file for configuration saving.")
        f.flush()


tech_details_string = f"""
                    <summary>Technical details</summary>
                    - ⭐ Star at GitHub: <a href="https://github.com/PDFMathTranslate/PDFMathTranslate-next">PDFMathTranslate/PDFMathTranslate-next</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI by: <a href="https://github.com/reycn">Rongxin</a> & <a href="https://github.com/hellofinch">hellofinch</a> & <a href="https://github.com/awwaawwa">awwaawwa</a><br>
                    - pdf2zh Version: {__version__} <br>
                    - BabelDOC Version: {babeldoc_version}<br>
                    - Free translation service provided by <a href="https://siliconflow.cn/" target="_blank" style="text-decoration: none;">SiliconFlow</a><br>
                    <a href="https://siliconflow.cn/" target="_blank" style="text-decoration: none;">
                        <img src="/gradio_api/file={logo_path}" alt="Powered By SiliconFlow" style="height: 40px; margin-top: 10px;">
                    </a>
                    <br>
                """
update_current_languages(settings.gui_settings.ui_lang)
# The following code creates the GUI
with gr.Blocks(
    title="PDFMathTranslate - PDF Translation with preserved formats",
    theme=gr.themes.Default(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
) as demo:
    lang_selector = gr.Dropdown(
        choices=LANGUAGES,
        label=_("UI Language"),
        value=settings.gui_settings.ui_lang,
        render=False,
    )
    with Translate(get_translation_dic(translation_file_path), lang_selector):
        gr.Markdown("# [PDFMathTranslate Next](https://pdf2zh-next.com)")

        translation_engine_arg_inputs = []
        detail_text_inputs = []
        require_llm_translator_inputs = []
        detail_text_input_index_map = {}
        term_detail_text_inputs = []
        term_detail_text_input_index_map = {}
        LLM_support_index_map.clear()
        with gr.Row():
            with gr.Column(scale=1):
                lang_selector.render()
                gr.Markdown(_("## File"))
                file_type = gr.Radio(
                    choices=[("File", "File"), ("Link", "Link")],
                    label="Type",
                    value="File",
                )
                file_input = gr.File(
                    label=_("File"),
                    file_count="single",
                    file_types=[".pdf", ".PDF"],
                    type="filepath",
                    elem_classes=["input-file"],
                )
                link_input = gr.Textbox(
                    label=_("Link"),
                    visible=False,
                    interactive=True,
                )

                gr.Markdown(_("## Translation Options"))

                with gr.Row():
                    lang_from = gr.Dropdown(
                        label=_("Translate from"),
                        choices=list(lang_map.keys()),
                        value=default_lang_from,
                    )
                    lang_to = gr.Dropdown(
                        label=_("Translate to"),
                        choices=list(lang_map.keys()),
                        value=default_lang_to,
                    )

                siliconflow_free_acknowledgement = gr.Markdown(
                    _(
                        "Free translation service provided by [SiliconFlow](https://siliconflow.cn)"
                    ),
                    visible=True,
                )

                detail_index = 0
                term_detail_index = 0
                with gr.Group() as translation_engine_settings:
                    service = gr.Dropdown(
                        label=_("Service"),
                        choices=available_services,
                        value=available_services[0],
                    )

                    __gui_service_arg_names = []
                    for service_name in available_services:
                        metadata = TRANSLATION_ENGINE_METADATA_MAP[service_name]
                        LLM_support_index_map[metadata.translate_engine_type] = (
                            metadata.support_llm
                        )
                        if not metadata.cli_detail_field_name:
                            # no detail field, no need to show
                            continue
                        detail_settings = getattr(
                            settings, metadata.cli_detail_field_name
                        )
                        visible = service.value == metadata.translate_engine_type

                        # OpenAI specific settings (initially visible if OpenAI is default)
                        with gr.Group(visible=True) as service_detail:
                            detail_text_input_index_map[
                                metadata.translate_engine_type
                            ] = []
                            for (
                                field_name,
                                field,
                            ) in metadata.setting_model_type.model_fields.items():
                                if disable_gui_sensitive_input:
                                    if field_name in GUI_SENSITIVE_FIELDS:
                                        continue
                                    if field_name in GUI_PASSWORD_FIELDS:
                                        continue
                                if field.default_factory:
                                    continue

                                if field_name == "translate_engine_type":
                                    continue
                                if field_name == "support_llm":
                                    continue
                                type_hint = field.annotation
                                original_type = typing.get_origin(type_hint)
                                type_args = typing.get_args(type_hint)
                                value = getattr(detail_settings, field_name)
                                if (
                                    type_hint is str
                                    or str in type_args
                                    or type_hint is int
                                    or int in type_args
                                ):
                                    if field_name in GUI_PASSWORD_FIELDS:
                                        field_input = gr.Textbox(
                                            label=field.description,
                                            value=value,
                                            interactive=True,
                                            type="password",
                                            visible=visible,
                                        )
                                    else:
                                        field_input = gr.Textbox(
                                            label=field.description,
                                            value=value,
                                            interactive=True,
                                            visible=visible,
                                        )
                                elif type_hint is bool or bool in type_args:
                                    field_input = gr.Checkbox(
                                        label=field.description,
                                        value=value,
                                        interactive=True,
                                        visible=visible,
                                    )
                                else:
                                    raise Exception(
                                        f"Unsupported type {type_hint} for field {field_name} in gui translation engine settings"
                                    )
                                detail_text_input_index_map[
                                    metadata.translate_engine_type
                                ].append(detail_index)
                                detail_index += 1
                                detail_text_inputs.append(field_input)
                                __gui_service_arg_names.append(field_name)
                                translation_engine_arg_inputs.append(field_input)
                    with gr.Group() as rate_limit_settings:
                        rate_limit_mode = gr.Radio(
                            choices=[
                                ("RPM (Requests Per Minute)", "RPM"),
                                ("Concurrent Requests", "Concurrent Threads"),
                                ("Custom", "Custom"),
                            ],
                            label="Rate Limit Mode",
                            value="Custom",
                            interactive=True,
                            visible=False,
                            info="Select the rate limit mode that best suits your API provider, system will automatically convert the rate limiting values of RPM or Concurrent Requests to QPS and Pool Max Workers when you click the Translate button",
                        )

                        rpm_input = gr.Number(
                            label=_("RPM (Requests Per Minute)"),
                            value=240,  # More conservative default value
                            precision=0,
                            minimum=1,
                            maximum=60000,
                            interactive=True,
                            visible=False,
                            info=_(
                                "Most API providers provide this parameter, such as OpenAI GPT-4: 500 RPM"
                            ),
                        )

                        concurrent_threads_input = gr.Number(
                            label=_("Concurrent Threads"),
                            value=20,  # More conservative default value
                            precision=0,
                            minimum=1,
                            maximum=1000,
                            interactive=True,
                            visible=False,
                            info=_(
                                "Maximum number of requests processed simultaneously"
                            ),
                        )

                        custom_qps_input = gr.Number(
                            label=_("QPS (Queries Per Second)"),
                            value=settings.translation.qps or 4,
                            precision=0,
                            minimum=1,
                            maximum=1000,
                            interactive=True,
                            visible=False,
                            info=_("Number of requests sent per second"),
                        )

                        custom_pool_max_workers_input = gr.Number(
                            label=_("Pool Max Workers"),
                            value=settings.translation.pool_max_workers,
                            precision=0,
                            minimum=0,
                            maximum=1000,
                            interactive=True,
                            visible=False,
                            info=_(
                                "If not set or set to 0, QPS will be used as the number of workers"
                            ),
                        )

                # Term extraction options (engine + rate limit + detail settings)
                with gr.Accordion(_("Auto Term Extraction"), open=True):
                    enable_auto_term_extraction = gr.Checkbox(
                        label=_("Enable auto term extraction"),
                        value=not settings.translation.no_auto_extract_glossary,
                        interactive=True,
                    )

                    term_disabled_info = gr.Markdown(
                        _(
                            "Auto term extraction is disabled. Term extraction settings below will not take effect until it is enabled."
                        ),
                        visible=settings.translation.no_auto_extract_glossary,
                    )

                    with gr.Group(visible=True) as term_settings_group:
                        term_service = gr.Dropdown(
                            label=_("Term extraction engine"),
                            choices=[(_("Follow main translation engine"), "Follow main translation engine")]
                            + [
                                metadata.translate_engine_type
                                for metadata in TERM_EXTRACTION_ENGINE_METADATA
                            ],
                            value="Follow main translation engine",
                        )

                        # Term engine detail settings
                        __gui_term_service_arg_names = []
                        for term_metadata in TERM_EXTRACTION_ENGINE_METADATA:
                            if not term_metadata.cli_detail_field_name:
                                continue
                            term_detail_field_name = (
                                f"term_{term_metadata.cli_detail_field_name}"
                            )
                            term_detail_settings = getattr(
                                settings, term_detail_field_name
                            )

                            # Term engine settings group should stay visible;
                            # visibility is controlled by each field input.
                            with gr.Group() as term_service_detail:
                                term_detail_text_input_index_map[
                                    term_metadata.translate_engine_type
                                ] = []
                                for (
                                    field_name,
                                    field,
                                ) in term_metadata.term_setting_model_type.model_fields.items():
                                    if field_name in (
                                        "translate_engine_type",
                                        "support_llm",
                                    ):
                                        continue
                                    if field.default_factory:
                                        continue

                                    base_field_name = field_name
                                    if base_field_name.startswith("term_"):
                                        base_name = base_field_name[len("term_") :]
                                    else:
                                        base_name = base_field_name

                                    if disable_gui_sensitive_input:
                                        if base_name in GUI_SENSITIVE_FIELDS:
                                            continue
                                        if base_name in GUI_PASSWORD_FIELDS:
                                            continue

                                    type_hint = field.annotation
                                    original_type = typing.get_origin(type_hint)
                                    type_args = typing.get_args(type_hint)
                                    value = getattr(term_detail_settings, field_name)

                                    if (
                                        type_hint is str
                                        or str in type_args
                                        or type_hint is int
                                        or int in type_args
                                    ):
                                        if base_name in GUI_PASSWORD_FIELDS:
                                            field_input = gr.Textbox(
                                                label=field.description,
                                                value=value,
                                                interactive=True,
                                                type="password",
                                                visible=False,
                                            )
                                        else:
                                            field_input = gr.Textbox(
                                                label=field.description,
                                                value=value,
                                                interactive=True,
                                                visible=False,
                                            )
                                    elif type_hint is bool or bool in type_args:
                                        field_input = gr.Checkbox(
                                            label=field.description,
                                            value=value,
                                            interactive=True,
                                            visible=False,
                                        )
                                    else:
                                        raise Exception(
                                            f"Unsupported type {type_hint} for field {field_name} in gui term extraction engine settings"
                                        )

                                    term_detail_text_input_index_map[
                                        term_metadata.translate_engine_type
                                    ].append(term_detail_index)
                                    term_detail_index += 1
                                    term_detail_text_inputs.append(field_input)
                                    __gui_term_service_arg_names.append(field_name)
                                    translation_engine_arg_inputs.append(field_input)

                        term_rate_limit_mode = gr.Radio(
                            choices=[
                                ("RPM (Requests Per Minute)", "RPM"),
                                ("Concurrent Requests", "Concurrent Threads"),
                                ("Custom", "Custom"),
                            ],
                            label="Term rate limit mode",
                            value="Custom",
                            interactive=True,
                        )

                        term_rpm_input = gr.Number(
                            label=_("Term RPM (Requests Per Minute)"),
                            value=240,
                            precision=0,
                            minimum=1,
                            maximum=60000,
                            interactive=True,
                            visible=False,
                        )

                        term_concurrent_threads_input = gr.Number(
                            label=_("Term concurrent threads"),
                            value=20,
                            precision=0,
                            minimum=1,
                            maximum=1000,
                            interactive=True,
                            visible=False,
                        )

                        term_custom_qps_input = gr.Number(
                            label=_("Term QPS (Queries Per Second)"),
                            value=(
                                settings.translation.term_qps
                                or settings.translation.qps
                                or 4
                            ),
                            precision=0,
                            minimum=1,
                            maximum=1000,
                            interactive=True,
                            visible=True,
                        )

                        term_custom_pool_max_workers_input = gr.Number(
                            label=_("Term pool max workers"),
                            value=settings.translation.term_pool_max_workers,
                            precision=0,
                            minimum=0,
                            maximum=1000,
                            interactive=True,
                            visible=True,
                        )

                page_range = gr.Radio(
                    choices=[
                        ("All", "All"),
                        ("First", "First"),
                        ("First 5 pages", "First 5 pages"),
                        ("Range", "Range"),
                    ],
                    label="Pages",
                    value="All",
                )

                page_input = gr.Textbox(
                    label=_("Page range (e.g., 1,3,5-10,-5)"),
                    visible=False,
                    interactive=True,
                    placeholder=_("e.g., 1,3,5-10"),
                )

                only_include_translated_page = gr.Checkbox(
                    label=_("Only include translated pages in the output PDF."),
                    info=_("Effective only when a page range is specified."),
                    value=settings.pdf.only_include_translated_page,
                    interactive=True,
                )

                # PDF Output Options
                gr.Markdown(_("## PDF Output Options"))
                with gr.Row():
                    no_mono = gr.Checkbox(
                        label=_("Disable monolingual output"),
                        value=settings.pdf.no_mono,
                        interactive=True,
                    )
                    no_dual = gr.Checkbox(
                        label=_("Disable bilingual output"),
                        value=settings.pdf.no_dual,
                        interactive=True,
                    )

                with gr.Row():
                    dual_translate_first = gr.Checkbox(
                        label=_("Put translated pages first in dual mode"),
                        value=settings.pdf.dual_translate_first,
                        interactive=True,
                    )
                    use_alternating_pages_dual = gr.Checkbox(
                        label=_("Use alternating pages for dual PDF"),
                        value=settings.pdf.use_alternating_pages_dual,
                        interactive=True,
                    )

                watermark_output_mode = gr.Radio(
                    choices=[("Watermarked", "Watermarked"), ("No Watermark", "No Watermark")],
                    label="Watermark mode",
                    value="Watermarked"
                    if settings.pdf.watermark_output_mode == "watermarked"
                    else "No Watermark",
                )

                # Additional translation options
                with gr.Accordion(_("Advanced Options"), open=False):
                    prompt = gr.Textbox(
                        label=_("Custom prompt for translation"),
                        value="",
                        visible=False,
                        interactive=True,
                        placeholder=_("Custom prompt for the translator"),
                    )

                    # New Textbox for custom_system_prompt
                    custom_system_prompt_input = gr.Textbox(
                        label=_("Custom System Prompt"),
                        value=settings.translation.custom_system_prompt or "",
                        interactive=True,
                        placeholder=_(
                            "e.g. /no_think You are a professional zh-CN native translator who needs to fluently translate text into zh-CN."
                        ),
                    )

                    min_text_length = gr.Number(
                        label=_("Minimum text length to translate"),
                        value=settings.translation.min_text_length,
                        precision=0,
                        minimum=0,
                        interactive=True,
                    )

                    rpc_doclayout = gr.Textbox(
                        label=_("RPC service for document layout analysis (optional)"),
                        value=settings.translation.rpc_doclayout or "",
                        visible=False,
                        interactive=True,
                        placeholder="http://host:port",
                    )

                    save_auto_extracted_glossary = gr.Checkbox(
                        label=_("save automatically extracted glossary"),
                        value=settings.translation.save_auto_extracted_glossary,
                        interactive=True,
                    )

                    primary_font_family = gr.Dropdown(
                        label=_("Primary font family for translated text"),
                        choices=["Auto", "serif", "sans-serif", "script"],
                        value="Auto"
                        if not settings.translation.primary_font_family
                        else settings.translation.primary_font_family,
                        interactive=True,
                    )

                    glossary_file = gr.File(
                        label=_("Glossary File"),
                        file_count="multiple",
                        file_types=[".csv"],
                        type="binary",
                        visible=True,
                    )
                    require_llm_translator_inputs.append(glossary_file)

                    glossary_table = gr.Dataframe(
                        headers=["source", "target"],
                        datatype=["str", "str"],
                        interactive=False,
                        col_count=(2, "fixed"),
                        visible=False,
                    )
                    require_llm_translator_inputs.append(glossary_table)

                    # PDF options section
                    gr.Markdown(_("### PDF Options"))

                    skip_clean = gr.Checkbox(
                        label=_("Skip clean (maybe improve compatibility)"),
                        value=settings.pdf.skip_clean,
                        interactive=True,
                    )

                    disable_rich_text_translate = gr.Checkbox(
                        label=_(
                            "Disable rich text translation (maybe improve compatibility)"
                        ),
                        value=settings.pdf.disable_rich_text_translate,
                        interactive=True,
                    )

                    enhance_compatibility = gr.Checkbox(
                        label=_(
                            "Enhance compatibility (auto-enables skip_clean and disable_rich_text)"
                        ),
                        value=settings.pdf.enhance_compatibility,
                        interactive=True,
                    )

                    split_short_lines = gr.Checkbox(
                        label=_("Force split short lines into different paragraphs"),
                        value=settings.pdf.split_short_lines,
                        interactive=True,
                    )

                    short_line_split_factor = gr.Slider(
                        label=_("Split threshold factor for short lines"),
                        value=settings.pdf.short_line_split_factor,
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        interactive=True,
                        visible=settings.pdf.split_short_lines,
                    )

                    translate_table_text = gr.Checkbox(
                        label=_("Translate table text (experimental)"),
                        value=settings.pdf.translate_table_text,
                        interactive=True,
                    )

                    skip_scanned_detection = gr.Checkbox(
                        label=_("Skip scanned detection"),
                        value=settings.pdf.skip_scanned_detection,
                        interactive=True,
                    )

                    ocr_workaround = gr.Checkbox(
                        label=_(
                            "OCR workaround (experimental, will auto enable Skip scanned detection in backend)"
                        ),
                        value=settings.pdf.ocr_workaround,
                        interactive=True,
                    )

                    auto_enable_ocr_workaround = gr.Checkbox(
                        label=_(
                            "Auto enable OCR workaround (enable automatic OCR workaround for heavily scanned documents)"
                        ),
                        value=settings.pdf.auto_enable_ocr_workaround,
                        interactive=True,
                    )

                    max_pages_per_part = gr.Number(
                        label=_(
                            "Maximum pages per part (for auto-split translation, 0 means no limit)"
                        ),
                        value=settings.pdf.max_pages_per_part,
                        precision=0,
                        minimum=0,
                        interactive=True,
                    )

                    formular_font_pattern = gr.Textbox(
                        label=_(
                            "Font pattern to identify formula text (regex, not recommended to change)"
                        ),
                        value=settings.pdf.formular_font_pattern or "",
                        interactive=True,
                        placeholder="e.g., CMMI|CMR",
                    )

                    formular_char_pattern = gr.Textbox(
                        label=_(
                            "Character pattern to identify formula text (regex, not recommended to change)"
                        ),
                        value=settings.pdf.formular_char_pattern or "",
                        interactive=True,
                        placeholder="e.g., [∫∬∭∮∯∰∇∆]",
                    )

                    ignore_cache = gr.Checkbox(
                        label=_("Ignore cache"),
                        value=settings.translation.ignore_cache,
                        interactive=True,
                    )

                    # BabelDOC v0.5.1 new options
                    gr.Markdown(_("#### BabelDOC Advanced Options"))

                    merge_alternating_line_numbers = gr.Checkbox(
                        label=_("Merge alternating line numbers"),
                        info=_(
                            "Handle alternating line numbers and text paragraphs in documents with line numbers"
                        ),
                        value=not settings.pdf.no_merge_alternating_line_numbers,
                        interactive=True,
                    )

                    remove_non_formula_lines = gr.Checkbox(
                        label=_("Remove non-formula lines"),
                        info=_("Remove non-formula lines within paragraph areas"),
                        value=not settings.pdf.no_remove_non_formula_lines,
                        interactive=True,
                    )

                    non_formula_line_iou_threshold = gr.Slider(
                        label=_("Non-formula line IoU threshold"),
                        info=_("IoU threshold for identifying non-formula lines"),
                        value=settings.pdf.non_formula_line_iou_threshold,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        interactive=True,
                    )

                    figure_table_protection_threshold = gr.Slider(
                        label=_("Figure/table protection threshold"),
                        info=_(
                            "Protection threshold for figures and tables (lines within figures/tables will not be processed)"
                        ),
                        value=settings.pdf.figure_table_protection_threshold,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        interactive=True,
                    )

                    skip_formula_offset_calculation = gr.Checkbox(
                        label=_("Skip formula offset calculation"),
                        info=_("Skip formula offset calculation during processing"),
                        value=settings.pdf.skip_formula_offset_calculation,
                        interactive=True,
                    )

                output_title = gr.Markdown(_("## Translated"), visible=False)
                output_file_mono = gr.File(
                    label=_("Download Translation (Mono)"), visible=False
                )
                output_file_dual = gr.File(
                    label=_("Download Translation (Dual)"), visible=False
                )
                output_file_glossary = gr.File(
                    label=_("Download automatically extracted glossary"), visible=False
                )
                translate_btn = gr.Button(_("Translate"), variant="primary")
                cancel_btn = gr.Button(_("Cancel"), variant="secondary")
                save_btn = gr.Button(_("Save Settings"), variant="secondary")

                tech_details = gr.Markdown(
                    tech_details_string,
                    elem_classes=["secondary-text"],
                )

            with gr.Column(scale=2):
                gr.Markdown(_("## Preview"))
                preview = PDF(label=_("Document Preview"), visible=True, height=2000)

        # Event handlers
        def on_select_filetype(file_type):
            """Update visibility based on selected file type"""
            return (
                gr.update(visible=file_type == "File"),
                gr.update(visible=file_type == "Link"),
            )

        def on_select_page(choice):
            """Update page input visibility based on selection"""
            return gr.update(visible=choice == "Range")

        def on_select_service(service_name):
            """Update service-specific settings visibility"""
            if not detail_text_inputs:
                return
            detail_group_index = detail_text_input_index_map.get(service_name, [])
            llm_support = LLM_support_index_map.get(service_name, False)
            siliconflow_free_acknowledgement_visible = service_name == "SiliconFlowFree"
            siliconflow_update = [
                gr.update(visible=siliconflow_free_acknowledgement_visible)
            ]
            return_list = []
            glossary_updates = [
                gr.update(visible=llm_support)
                for i in range(len(require_llm_translator_inputs))
            ]
            if len(detail_text_inputs) == 1:
                return_list = (
                    siliconflow_update
                    + glossary_updates
                    + [gr.update(visible=(0 in detail_group_index))]
                )
            else:
                return_list = (
                    siliconflow_update
                    + glossary_updates
                    + [
                        gr.update(visible=(i in detail_group_index))
                        for i in range(len(detail_text_inputs))
                    ]
                )
            return return_list

        def on_enhance_compatibility_change(enhance_value):
            """Update skip_clean and disable_rich_text_translate when enhance_compatibility changes"""
            if enhance_value:
                # When enhanced compatibility is enabled, both options are auto-enabled and disabled for user modification
                return (
                    gr.update(value=True, interactive=False),
                    gr.update(value=True, interactive=False),
                )
            else:
                # When disabled, allow user to modify these settings
                return (
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                )

        def on_split_short_lines_change(split_value):
            """Update short_line_split_factor visibility based on split_short_lines value"""
            return gr.update(visible=split_value)

        def on_glossary_file_change(glossary_file):
            if glossary_file is None:
                return gr.update(visible=False)

            glossary_list = []
            for file in glossary_file:
                file_encoding = chardet.detect(file)["encoding"]
                content = file.decode(file_encoding).replace("\r\n", "\n").strip()
                with io.StringIO(content) as f:
                    csvreader = csv.reader(f, delimiter=",", doublequote=True)
                    next(csvreader)  # Skip header
                    for line in csvreader:
                        if line:
                            glossary_list.append(line)
            logger.warning(f"on_glossary_file_delete glossary_list {glossary_list}")
            if not glossary_list:
                glossary_list = ["", "", ""]
            return gr.update(visible=True, value=glossary_list)

        def on_rate_limit_mode_change(mode, service_name):
            """Update rate-limit-specific-settings visibility based on rate_limit_mode value"""
            if service_name == "SiliconFlowFree":
                return [gr.update(visible=False)] * 4  # Hide all options

            rpm_visible = mode == "RPM"
            threads_visible = mode == "Concurrent Threads"
            custom_visible = mode == "Custom"

            return [
                gr.update(visible=rpm_visible),
                gr.update(visible=threads_visible),
                gr.update(visible=custom_visible),
                gr.update(visible=custom_visible),
            ]

        def on_enable_auto_term_extraction_change(enabled: bool):
            """Update term disabled info visibility based on auto term extraction toggle"""
            return gr.update(visible=not enabled)

        def on_term_rate_limit_mode_change(mode: str):
            """Update term rate-limit controls visibility based on mode"""
            rpm_visible = mode == "RPM"
            threads_visible = mode == "Concurrent Threads"
            custom_visible = mode == "Custom"
            return [
                gr.update(visible=rpm_visible),
                gr.update(visible=threads_visible),
                gr.update(visible=custom_visible),
                gr.update(visible=custom_visible),
            ]

        def on_term_service_change(term_service_name: str):
            """Update term engine-specific settings visibility"""
            if not term_detail_text_inputs:
                return
            detail_group_index = term_detail_text_input_index_map.get(
                term_service_name, []
            )
            if len(term_detail_text_inputs) == 1:
                return [gr.update(visible=(0 in detail_group_index))]
            return [
                gr.update(visible=(i in detail_group_index))
                for i in range(len(term_detail_text_inputs))
            ]

        def on_service_change_with_rate_limit(mode, service_name):
            """Expand original on_select_service with rate-limit-UI updated"""
            original_updates = on_select_service(service_name)

            rate_limit_visible = service_name != "SiliconFlowFree"

            detailed_visible = [gr.update(visible=False)] * 4

            if rate_limit_visible:
                detailed_visible = on_rate_limit_mode_change(mode, service_name)

            # Add updates of rate-limit-UI
            rate_limit_updates = [
                gr.update(visible=rate_limit_visible),
            ]

            return original_updates + rate_limit_updates + detailed_visible

        def on_lang_selector_change(lang):
            settings.gui_settings.ui_lang = lang
            update_current_languages(lang)
            config_manager.write_user_default_config_file(settings=settings.clone())
            return

        # UI language change handler

        lang_selector.change(on_lang_selector_change, lang_selector)

        # Default file handler
        file_input.upload(
            lambda x: x,
            inputs=file_input,
            outputs=preview,
        )

        # Event bindings
        file_type.select(
            on_select_filetype,
            file_type,
            [file_input, link_input],
        )

        page_range.select(
            on_select_page,
            page_range,
            page_input,
        )

        on_select_service_outputs = (
            [siliconflow_free_acknowledgement]
            + require_llm_translator_inputs
            + detail_text_inputs
        )

        service.select(
            on_service_change_with_rate_limit,
            [rate_limit_mode, service],
            outputs=(
                on_select_service_outputs
                if len(on_select_service_outputs) > 0
                else None
            )
            + [
                rate_limit_mode,
                rpm_input,
                concurrent_threads_input,
                custom_qps_input,
                custom_pool_max_workers_input,
            ],
        )

        rate_limit_mode.change(
            on_rate_limit_mode_change,
            inputs=[rate_limit_mode, service],
            outputs=[
                rpm_input,
                concurrent_threads_input,
                custom_qps_input,
                custom_pool_max_workers_input,
            ],
        )

        glossary_file.change(
            on_glossary_file_change,
            glossary_file,
            outputs=glossary_table,
        )

        # Add event handler for enhance_compatibility
        enhance_compatibility.change(
            on_enhance_compatibility_change,
            enhance_compatibility,
            [skip_clean, disable_rich_text_translate],
        )

        # Add event handler for split_short_lines
        split_short_lines.change(
            on_split_short_lines_change,
            split_short_lines,
            short_line_split_factor,
        )

        # Auto term extraction toggle handlers
        enable_auto_term_extraction.change(
            on_enable_auto_term_extraction_change,
            enable_auto_term_extraction,
            term_disabled_info,
        )

        # Term rate limit handlers
        term_rate_limit_mode.change(
            on_term_rate_limit_mode_change,
            term_rate_limit_mode,
            [
                term_rpm_input,
                term_concurrent_threads_input,
                term_custom_qps_input,
                term_custom_pool_max_workers_input,
            ],
        )

        # Term service change handler
        term_service.change(
            on_term_service_change,
            term_service,
            outputs=(
                term_detail_text_inputs if len(term_detail_text_inputs) > 0 else None
            ),
        )

        # State for managing translation tasks
        state = gr.State({"session_id": None, "current_task": None})

        # UI setting controls list (shared by translate_btn and save_btn)
        ui_setting_controls = [
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            # PDF Output Options
            no_mono,
            no_dual,
            dual_translate_first,
            use_alternating_pages_dual,
            watermark_output_mode,
            # Rate Limit Options
            rate_limit_mode,
            rpm_input,
            concurrent_threads_input,
            custom_qps_input,
            custom_pool_max_workers_input,
            # Advanced Options
            prompt,
            min_text_length,
            rpc_doclayout,
            custom_system_prompt_input,
            glossary_file,
            save_auto_extracted_glossary,
            # New advanced translation options
            enable_auto_term_extraction,
            primary_font_family,
            skip_clean,
            disable_rich_text_translate,
            enhance_compatibility,
            split_short_lines,
            short_line_split_factor,
            translate_table_text,
            skip_scanned_detection,
            max_pages_per_part,
            formular_font_pattern,
            formular_char_pattern,
            ignore_cache,
            state,
            ocr_workaround,
            auto_enable_ocr_workaround,
            only_include_translated_page,
            # BabelDOC v0.5.1 new options
            merge_alternating_line_numbers,
            remove_non_formula_lines,
            non_formula_line_iou_threshold,
            figure_table_protection_threshold,
            skip_formula_offset_calculation,
            # Term extraction engine options
            term_service,
            term_rate_limit_mode,
            term_rpm_input,
            term_concurrent_threads_input,
            term_custom_qps_input,
            term_custom_pool_max_workers_input,
            *translation_engine_arg_inputs,
            # any UI components that are used by translate/save should be listed above!
            # Extra UI components to be updated on load (not used by translate/save)
            siliconflow_free_acknowledgement,
            glossary_table,
            term_disabled_info,
        ]

        # Translation button click handler
        translate_btn.click(
            translate_file,
            inputs=[
                file_type,
                file_input,
                link_input,
                *ui_setting_controls,
            ],
            outputs=[
                output_file_mono,  # Mono PDF file
                preview,  # Preview
                output_file_dual,  # Dual PDF file
                output_file_glossary,
                output_file_mono,  # Visibility of mono output
                output_file_dual,  # Visibility of dual output
                output_file_glossary,
                output_title,  # Visibility of output title
            ],
        )

        # Cancel button click handler
        cancel_btn.click(
            stop_translate_file,
            inputs=[state],
        )

        # Save button click handler
        save_btn.click(
            save_config,
            inputs=ui_setting_controls,
        )

        def load_saved_config_to_ui(state):
            """Reload all settings from config and update UI components."""
            try:
                fresh_settings = settings
                update_current_languages(settings.gui_settings.ui_lang)

                updates: list = []

                # Determine selected service by cli flag
                selected_service = None
                for metadata in TRANSLATION_ENGINE_METADATA:
                    if getattr(fresh_settings, metadata.cli_flag_name, False):
                        selected_service = metadata.translate_engine_type
                        break
                if not selected_service:
                    selected_service = available_services[0]
                llm_support = LLM_support_index_map.get(selected_service, False)

                # Follow the EXACT order of ui_setting_controls
                # service
                updates.append(gr.update(value=selected_service))
                # lang_from, lang_to
                loaded_lang_from = rev_lang_map.get(
                    fresh_settings.translation.lang_in, "English"
                )
                loaded_lang_to_code = fresh_settings.translation.lang_out
                loaded_lang_to = next(
                    (k for k, v in lang_map.items() if v == loaded_lang_to_code),
                    "Simplified Chinese",
                )
                updates.append(gr.update(value=loaded_lang_from))
                updates.append(gr.update(value=loaded_lang_to))
                # page_range, page_input
                pages_setting = fresh_settings.pdf.pages
                if pages_setting is None or pages_setting == "":
                    updates.append(gr.update(value="All"))
                    updates.append(gr.update(value="", visible=False))
                else:
                    updates.append(gr.update(value="Range"))
                    updates.append(gr.update(value=str(pages_setting), visible=True))
                # PDF Output Options
                updates.append(gr.update(value=fresh_settings.pdf.no_mono))
                updates.append(gr.update(value=fresh_settings.pdf.no_dual))
                updates.append(gr.update(value=fresh_settings.pdf.dual_translate_first))
                updates.append(
                    gr.update(value=fresh_settings.pdf.use_alternating_pages_dual)
                )
                watermark_value = (
                    "Watermarked"
                    if fresh_settings.pdf.watermark_output_mode == "watermarked"
                    else "No Watermark"
                )
                updates.append(gr.update(value=watermark_value))
                # Rate Limit Options
                rate_limit_visible = selected_service != "SiliconFlowFree"
                updates.append(gr.update(value="Custom", visible=rate_limit_visible))
                updates.append(gr.update(visible=False))  # rpm_input
                updates.append(gr.update(visible=False))  # concurrent_threads_input
                updates.append(
                    gr.update(
                        value=fresh_settings.translation.qps or 4,
                        visible=rate_limit_visible,
                    )
                )
                updates.append(
                    gr.update(
                        value=fresh_settings.translation.pool_max_workers,
                        visible=rate_limit_visible,
                    )
                )
                # Advanced Options
                updates.append(gr.update(value=""))  # prompt
                updates.append(
                    gr.update(value=fresh_settings.translation.min_text_length)
                )
                updates.append(
                    gr.update(value=fresh_settings.translation.rpc_doclayout or "")
                )
                updates.append(
                    gr.update(
                        value=fresh_settings.translation.custom_system_prompt or ""
                    )
                )
                updates.append(
                    gr.update(visible=llm_support)
                )  # glossary_file visibility
                updates.append(
                    gr.update(
                        value=fresh_settings.translation.save_auto_extracted_glossary
                    )
                )
                # enable_auto_term_extraction is the inverse of no_auto_extract_glossary
                updates.append(
                    gr.update(
                        value=not fresh_settings.translation.no_auto_extract_glossary
                    )
                )
                primary_font_display = (
                    "Auto"
                    if not fresh_settings.translation.primary_font_family
                    else fresh_settings.translation.primary_font_family
                )
                updates.append(gr.update(value=primary_font_display))
                updates.append(gr.update(value=fresh_settings.pdf.skip_clean))
                updates.append(
                    gr.update(value=fresh_settings.pdf.disable_rich_text_translate)
                )
                updates.append(
                    gr.update(value=fresh_settings.pdf.enhance_compatibility)
                )
                updates.append(gr.update(value=fresh_settings.pdf.split_short_lines))
                updates.append(
                    gr.update(
                        value=fresh_settings.pdf.short_line_split_factor,
                        visible=fresh_settings.pdf.split_short_lines,
                    )
                )
                updates.append(gr.update(value=fresh_settings.pdf.translate_table_text))
                updates.append(
                    gr.update(value=fresh_settings.pdf.skip_scanned_detection)
                )
                updates.append(gr.update(value=fresh_settings.pdf.max_pages_per_part))
                updates.append(
                    gr.update(value=fresh_settings.pdf.formular_font_pattern or "")
                )
                updates.append(
                    gr.update(value=fresh_settings.pdf.formular_char_pattern or "")
                )
                updates.append(gr.update(value=fresh_settings.translation.ignore_cache))
                updates.append(state)  # state, keep unchanged
                updates.append(gr.update(value=fresh_settings.pdf.ocr_workaround))
                updates.append(
                    gr.update(value=fresh_settings.pdf.auto_enable_ocr_workaround)
                )
                updates.append(
                    gr.update(value=fresh_settings.pdf.only_include_translated_page)
                )
                # BabelDOC
                updates.append(
                    gr.update(
                        value=not fresh_settings.pdf.no_merge_alternating_line_numbers
                    )
                )
                updates.append(
                    gr.update(value=not fresh_settings.pdf.no_remove_non_formula_lines)
                )
                updates.append(
                    gr.update(value=fresh_settings.pdf.non_formula_line_iou_threshold)
                )
                updates.append(
                    gr.update(
                        value=fresh_settings.pdf.figure_table_protection_threshold
                    )
                )
                updates.append(
                    gr.update(value=fresh_settings.pdf.skip_formula_offset_calculation)
                )
                # Term extraction engine basic settings
                term_engine_enabled = (
                    not fresh_settings.translation.no_auto_extract_glossary
                )
                selected_term_service = "Follow main translation engine"
                for term_metadata in TERM_EXTRACTION_ENGINE_METADATA:
                    term_flag_name = f"term_{term_metadata.cli_flag_name}"
                    if getattr(fresh_settings, term_flag_name, False):
                        selected_term_service = term_metadata.translate_engine_type
                        break
                updates.append(gr.update(value=selected_term_service))
                # Term rate limit: use Custom mode by default
                updates.append(gr.update(value="Custom"))
                updates.append(gr.update(visible=False))  # term_rpm_input
                updates.append(
                    gr.update(visible=False)
                )  # term_concurrent_threads_input
                updates.append(
                    gr.update(
                        value=(
                            fresh_settings.translation.term_qps
                            or fresh_settings.translation.qps
                            or 4
                        ),
                        visible=True,
                    )
                )
                updates.append(
                    gr.update(
                        value=fresh_settings.translation.term_pool_max_workers,
                        visible=True,
                    )
                )
                # Translation engine detail fields (ordered)
                disable_sensitive_gui = (
                    fresh_settings.gui_settings.disable_gui_sensitive_input
                )
                for service_name in available_services:
                    metadata = TRANSLATION_ENGINE_METADATA_MAP[service_name]
                    if not metadata.cli_detail_field_name:
                        continue
                    detail_settings = getattr(
                        fresh_settings, metadata.cli_detail_field_name
                    )
                    for (
                        field_name,
                        field,
                    ) in metadata.setting_model_type.model_fields.items():
                        if disable_sensitive_gui:
                            if field_name in GUI_SENSITIVE_FIELDS:
                                continue
                            if field_name in GUI_PASSWORD_FIELDS:
                                continue
                        if field.default_factory:
                            continue
                        if (
                            field_name == "translate_engine_type"
                            or field_name == "support_llm"
                        ):
                            continue
                        value = getattr(detail_settings, field_name)
                        visible = metadata.translate_engine_type == selected_service
                        updates.append(gr.update(value=value, visible=visible))

                # Term extraction engine detail fields (ordered)
                for term_metadata in TERM_EXTRACTION_ENGINE_METADATA:
                    if not term_metadata.cli_detail_field_name:
                        continue
                    term_detail_field_name = (
                        f"term_{term_metadata.cli_detail_field_name}"
                    )
                    term_detail_settings = getattr(
                        fresh_settings, term_detail_field_name
                    )
                    for (
                        field_name,
                        field,
                    ) in term_metadata.term_setting_model_type.model_fields.items():
                        if field.default_factory:
                            continue
                        if field_name in ("translate_engine_type", "support_llm"):
                            continue
                        base_field_name = field_name
                        if base_field_name.startswith("term_"):
                            base_name = base_field_name[len("term_") :]
                        else:
                            base_name = base_field_name
                        if disable_sensitive_gui:
                            if base_name in GUI_SENSITIVE_FIELDS:
                                continue
                            if base_name in GUI_PASSWORD_FIELDS:
                                continue
                        value = getattr(term_detail_settings, field_name)
                        visible = (
                            term_metadata.translate_engine_type == selected_term_service
                        )
                        updates.append(gr.update(value=value, visible=visible))

                # Extra UI components at the end of ui_setting_controls
                siliconflow_free_ack_visible = selected_service == "SiliconFlowFree"
                updates.append(gr.update(visible=siliconflow_free_ack_visible))
                updates.append(
                    gr.update(visible=llm_support)
                )  # glossary_table visibility
                updates.append(
                    gr.update(
                        visible=fresh_settings.translation.no_auto_extract_glossary
                    )
                )  # term_disabled_info visibility

                return updates
            except Exception as e:
                logger.warning(f"Could not reload config on page load: {e}")
                return [None] * len(ui_setting_controls)

        # Use ui_setting_controls as outputs for page load
        demo.load(load_saved_config_to_ui, inputs=[state], outputs=ui_setting_controls)


def parse_user_passwd(file_path: str, welcome_page: str) -> tuple[list, str]:
    """
    This function parses a user password file.

    Inputs:
        - file_path: The path to the file

    Returns:
        - A tuple containing the user list and HTML
    """
    content = ""
    tuple_list = None
    if welcome_page:
        try:
            path = Path(welcome_page)
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Error: File '{welcome_page}' not found.")
    if file_path:
        try:
            path = Path(file_path)
            tuple_list = [
                tuple(line.strip().split(","))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except FileNotFoundError:
            tuple_list = None
    return tuple_list, content


def setup_gui(
    share: bool = False,
    auth_file: str | None = None,
    welcome_page: str | None = None,
    server_port=7860,
    inbrowser: bool = True,
) -> None:
    """
    This function sets up the GUI for the application.

    Inputs:
        - share: Whether to share the GUI
        - auth_file: The authentication file
        - server_port: The port to run the server on

    Returns:
        - None
    """

    user_list = None
    html = None

    user_list, html = parse_user_passwd(auth_file, welcome_page)

    if not auth_file or not user_list:
        try:
            demo.launch(
                server_name="0.0.0.0",
                debug=True,
                inbrowser=inbrowser,
                share=share,
                server_port=server_port,
                allowed_paths=[
                    logo_path,
                ],
            )
        except Exception:
            print(
                "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
            )
            try:
                demo.launch(
                    server_name="127.0.0.1",
                    debug=True,
                    inbrowser=inbrowser,
                    share=share,
                    server_port=server_port,
                    allowed_paths=[
                        logo_path,
                    ],
                )
            except Exception:
                print(
                    "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                )
                demo.launch(
                    debug=True,
                    inbrowser=inbrowser,
                    share=True,
                    server_port=server_port,
                    allowed_paths=[
                        logo_path,
                    ],
                )
    else:
        try:
            demo.launch(
                server_name="0.0.0.0",
                debug=True,
                inbrowser=inbrowser,
                share=share,
                auth=user_list,
                auth_message=html,
                server_port=server_port,
                allowed_paths=[
                    logo_path,
                ],
            )
        except Exception:
            print(
                "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
            )
            try:
                demo.launch(
                    server_name="127.0.0.1",
                    debug=True,
                    inbrowser=inbrowser,
                    share=share,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                    allowed_paths=[
                        logo_path,
                    ],
                )
            except Exception:
                print(
                    "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                )
                demo.launch(
                    debug=True,
                    inbrowser=inbrowser,
                    share=True,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                    allowed_paths=[
                        logo_path,
                    ],
                )


# For auto-reloading while developing
if __name__ == "__main__":
    from rich.logging import RichHandler

    # disable httpx, openai, httpcore, http11 logs
    logging.getLogger("httpx").setLevel("CRITICAL")
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai").setLevel("CRITICAL")
    logging.getLogger("openai").propagate = False
    logging.getLogger("httpcore").setLevel("CRITICAL")
    logging.getLogger("httpcore").propagate = False
    logging.getLogger("http11").setLevel("CRITICAL")
    logging.getLogger("http11").propagate = False
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    setup_gui(inbrowser=False)
