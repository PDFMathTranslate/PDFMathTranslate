from __future__ import annotations

import logging

from pydantic import Field
from pydantic import create_model

from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.config.translate_engine_model import _DEFAULT_TRANSLATION_ENGINE
from pdf2zh.kernel.experimental.config.translate_engine_model import TERM_EXTRACTION_ENGINE_METADATA
from pdf2zh.kernel.experimental.config.translate_engine_model import TRANSLATION_ENGINE_METADATA

logger = logging.getLogger(__name__)

# The following is magic code,
# if you need to modify it,
# please contact the maintainer!

__translation_flag_fields = {
    x.cli_flag_name: (
        bool,
        Field(
            default=False, description=f"Use {x.translate_engine_type} for translation"
        ),
    )
    for x in TRANSLATION_ENGINE_METADATA
}

__translation_flag_fields.update(
    {
        x.cli_detail_field_name: (
            x.setting_model_type,
            Field(default_factory=x.setting_model_type),
        )
        for x in TRANSLATION_ENGINE_METADATA
        if x.cli_detail_field_name
    }
)

# Term-extraction engine flags and detail settings (only for LLM-capable engines)
__term_translation_flag_fields = {
    f"term_{x.cli_flag_name}": (
        bool,
        Field(
            default=False,
            description=f"Use {x.translate_engine_type} for term extraction",
        ),
    )
    for x in TERM_EXTRACTION_ENGINE_METADATA
}

__term_translation_flag_fields.update(
    {
        f"term_{x.cli_detail_field_name}": (
            x.term_setting_model_type,
            Field(default_factory=x.term_setting_model_type),
        )
        for x in TERM_EXTRACTION_ENGINE_METADATA
        if x.cli_detail_field_name
    }
)

__translation_flag_fields.update(__term_translation_flag_fields)

__exclude_fields = list(__translation_flag_fields.keys())

# If you want to use more field parameters in `pdf2zh.kernel.experimental/config/model.py`
# please add the corresponding forwarding here!

__cli_env_settings_model_fields = {
    k: (
        v.annotation,
        Field(
            default=v.default,
            description=v.description,
            default_factory=v.default_factory,
            alias=v.alias,
            discriminator=v.discriminator,
        ),
    )
    for k, v in SettingsModel.model_fields.items()
    if k not in ("translate_engine_settings", "term_extraction_engine_settings")
}
__cli_env_settings_model_fields.update(__translation_flag_fields)

CLIEnvSettingsModel = create_model(
    "CLIEnvSettingsModel",
    **__cli_env_settings_model_fields,
)


def to_settings_model(self) -> SettingsModel:
    for metadata in TRANSLATION_ENGINE_METADATA:
        if getattr(self, metadata.cli_flag_name):
            if metadata.cli_detail_field_name:
                translate_engine_settings = metadata.setting_model_type(
                    **getattr(self, metadata.cli_detail_field_name).model_dump()
                )
            else:
                translate_engine_settings = metadata.setting_model_type()
            break
    else:
        logger.warning("No translation engine selected, using SiliconFlow Free")
        translate_engine_settings = _DEFAULT_TRANSLATION_ENGINE()

    # Term extraction engine (optional)
    term_extraction_engine_settings = None
    for metadata in TERM_EXTRACTION_ENGINE_METADATA:
        term_flag_name = f"term_{metadata.cli_flag_name}"
        if getattr(self, term_flag_name, False):
            term_detail_field_name = (
                f"term_{metadata.cli_detail_field_name}"
                if metadata.cli_detail_field_name
                else None
            )
            if term_detail_field_name:
                term_detail_model = getattr(self, term_detail_field_name)
                # Convert term settings back to base engine settings
                term_extraction_engine_settings = term_detail_model.to_base_settings()
            else:
                # No detail settings, use default base settings
                term_extraction_engine_settings = (
                    metadata.term_setting_model_type().to_base_settings()
                )
            break

    return SettingsModel(
        **self.model_dump(exclude=__exclude_fields),
        translate_engine_settings=translate_engine_settings,
        term_extraction_engine_settings=term_extraction_engine_settings,
    )


def validate_settings(self) -> None:
    self.to_settings_model().validate_settings()


def clone(self):
    return self.model_copy(deep=True)


CLIEnvSettingsModel.to_settings_model = to_settings_model
CLIEnvSettingsModel.validate_settings = validate_settings
CLIEnvSettingsModel.clone = clone
