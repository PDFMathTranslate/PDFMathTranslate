from __future__ import annotations

import argparse
import ast
import copy
import logging
import os
import threading
import typing
from inspect import getdoc
from pathlib import Path
from types import NoneType
from typing import Any
from typing import Literal
from typing import get_args
from typing import get_origin

import tomlkit
from pydantic import BaseModel

from pdf2zh.kernel.experimental.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.config.translate_engine_model import TRANSLATION_ENGINE_METADATA
from pdf2zh.kernel.experimental.const import DEFAULT_CONFIG_DIR
from pdf2zh.kernel.experimental.const import DEFAULT_CONFIG_FILE
from pdf2zh.kernel.experimental.const import VERSION_DEFAULT_CONFIG_FILE
from pdf2zh.kernel.experimental.const import WRITE_TEMP_CONFIG_FILE

# The following is magic code,
# if you need to modify it,
# please contact the maintainer!

_translation_engine_flag_names = [x.cli_flag_name for x in TRANSLATION_ENGINE_METADATA]
_term_translation_engine_flag_names = [
    f"term_{x.cli_flag_name}" for x in TRANSLATION_ENGINE_METADATA if x.support_llm
]

log = logging.getLogger(__name__)

_no_duplicate_field_name = set()
_no_duplicate_field_name.add("translate_engine_type")
_no_duplicate_field_name.add("support_llm")


class MagicDefault:
    pass


def build_args_parser(
    parser: argparse.ArgumentParser | None = None,
    settings_model: type[BaseModel] | None = None,
    field_name2type: dict[str, Any] | None = None,
    recursion_depth: int = 0,
    set_count: int = 0,
) -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    if parser is None:
        parser = argparse.ArgumentParser()

    if field_name2type is None:
        field_name2type = {}

    if settings_model is not None:
        parser = parser.add_argument_group(
            title=settings_model.__name__,
            description=getdoc(settings_model),
        )
    else:
        settings_model = CLIEnvSettingsModel

    for field_name, field_detail in settings_model.model_fields.items():
        if field_name not in _no_duplicate_field_name and field_name in field_name2type:
            log.critical(f"duplicate field name: {field_name}")
            raise ValueError(f"duplicate field name: {field_name}")
        field_name2type[field_name] = field_detail.annotation
        if field_detail.default_factory is not None:
            if recursion_depth > 0:
                raise ValueError("not supported nested settings models")
            build_args_parser(
                parser,
                field_detail.default_factory,
                field_name2type,
                recursion_depth + 1,
                set_count,
            )
        else:
            type_hint = typing.get_type_hints(settings_model)[field_name]
            original_type = typing.get_origin(type_hint)
            args = typing.get_args(type_hint)
            log.debug(
                f"field_name: {field_name}, type_hint: {type_hint}, original_type: {original_type}, args: {args}"
            )
            if original_type is Literal:
                continue
            if original_type is None:
                args = [type_hint]
            args_name = field_name.replace("_", "-").lower()

            if original_type is set:
                if set_count > 0:
                    raise ValueError("not supported multiple set arguments")

                if len(args) > 1:
                    raise ValueError("not supported multiple set arguments")

                if args[0] is not str:
                    raise ValueError("set type must be str")

                set_count += 1
                parser.add_argument(
                    f"{args_name}",
                    nargs="*",
                    type=str,
                    help=field_detail.description,
                )
                continue

            for arg in args:
                if arg is bool:
                    parser.add_argument(
                        f"--{args_name}",
                        action="store_true"
                        if field_detail.default is False
                        else "store_false",
                        default=MagicDefault,
                        help=field_detail.description,
                    )
                elif arg == NoneType:
                    continue
                else:
                    parser.add_argument(
                        f"--{args_name}",
                        type=arg,
                        default=MagicDefault,
                        help=field_detail.description,
                    )
    return parser, field_name2type


class ConfigManager:
    """Singleton configuration manager"""

    _instance: ConfigManager | None = None
    _settings: SettingsModel | None = None
    config_cli_settings: CLIEnvSettingsModel | None = None
    _default_config_file_path = DEFAULT_CONFIG_FILE
    _config_file_lock = threading.Lock()

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists"""
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def _read_toml_file(self, file_path: Path) -> dict:
        """Read and parse a TOML file

        Args:
            file_path: Path to the TOML file

        Returns:
            Parsed TOML content as dictionary
        """
        try:
            with self._config_file_lock:
                with file_path.open(encoding="utf-8") as f:
                    content = tomlkit.load(f)
            # Convert "null" strings back to None
            return self._process_toml_content(dict(content))
        except FileNotFoundError:
            log.debug(f"Config file not found: {file_path}")
            return {}
        except Exception as e:
            log.warning(f"Error reading config file {file_path}: {e}")
            return {}

    def _process_toml_content(self, content: dict) -> dict:
        """Process TOML content recursively, converting special values

        Args:
            content: Dictionary from TOML file

        Returns:
            Processed dictionary
        """
        # First convert "null" strings to None
        processed = {}
        for key, value in content.items():
            if isinstance(value, dict):
                processed[key] = self._process_toml_content(value)
            elif isinstance(value, str) and value == "null":
                processed[key] = None
            else:
                processed[key] = value

        return processed

    def _write_toml_file(self, file_path: Path, content: dict) -> None:
        """Write content to a TOML file

        Args:
            file_path: Path to write the TOML file
            content: Content to write as dictionary, can be flattened or nested
        """
        try:
            # Convert None values to "null" strings for tomlkit
            def convert_none_to_null(d):
                result = {}
                for k, v in d.items():
                    if v is None:
                        result[k] = "null"
                    elif isinstance(v, dict):
                        result[k] = convert_none_to_null(v)
                    else:
                        result[k] = v
                return result

            content = convert_none_to_null(content)

            toml_content = tomlkit.document()

            # Handle nested structure directly
            for key, value in content.items():
                if isinstance(value, dict):
                    section = tomlkit.table()
                    for k, v in value.items():
                        section.add(k, v)
                    toml_content.add(key, section)
                else:
                    toml_content.add(key, value)
            with self._config_file_lock:
                with file_path.open("w", encoding="utf-8") as f:
                    tomlkit.dump(content, f)
        except Exception as e:
            log.warning(f"Error writing config file {file_path}: {e}")
            raise

    def _is_file_content_identical(self, file_path: Path, content: dict) -> bool:
        """Check if file content is identical to the given content

        Args:
            file_path: Path to the file to check
            content: Content to compare with

        Returns:
            True if content is identical, False otherwise
        """
        try:
            existing_content = self._read_toml_file(file_path)
            return existing_content == content
        except Exception as e:
            log.warning(f"Error comparing file content: {e}")
            return False

    def _get_default_config(self) -> dict:
        """Get default configuration from model

        Returns:
            Default configuration as dictionary
        """
        default_model = CLIEnvSettingsModel()
        config_dict = default_model.model_dump(mode="json")

        # Flatten the dictionary to match TOML processing behavior
        return config_dict

    def _update_version_default_config(self) -> None:
        """Update version default configuration file if needed"""
        self._ensure_config_dir()
        default_config = self._get_default_config()

        # Always write if file doesn't exist
        if (
            not VERSION_DEFAULT_CONFIG_FILE.exists()
            or not self._is_file_content_identical(
                VERSION_DEFAULT_CONFIG_FILE, default_config
            )
        ):
            self._write_toml_file(VERSION_DEFAULT_CONFIG_FILE, default_config)
            log.debug("Updated version default configuration file")

    def parse_cli_args(self) -> None:
        """Parse command line arguments"""
        parser, _ = build_args_parser()
        args = parser.parse_args()
        cli_args = {
            k.replace("-", "_"): v
            for k, v in vars(args).items()
            if v is not MagicDefault
        }

        cli_parsed_args = self.parse_dict_vars(
            dict_vars=cli_args,
        )
        log.debug(f"CLI args: {cli_parsed_args}")
        self._settings = self._build_model_from_args(
            CLIEnvSettingsModel, cli_parsed_args
        ).to_settings_model()
        log.debug(f"Settings from CLI: {self._settings.model_dump_json()}")

    def parse_env_vars(
        self,
        dedup_field_name: set[str] | None = None,
        recursion_depth: int = 0,
        settings_model: type[BaseModel] | None = None,
    ) -> dict:
        return self.parse_dict_vars(
            dedup_field_name,
            recursion_depth,
            settings_model,
            os.environ,
            prefix="PDF2ZH_",
        )

    def parse_dict_vars(
        self,
        dedup_field_name: set[str] | None = None,
        recursion_depth: int = 0,
        settings_model: type[BaseModel] | None = None,
        dict_vars: dict | None = None,
        prefix: str = "",
    ) -> dict:
        """Parse environment variables into a settings dictionary

        Returns:
            Dictionary with settings derived from environment variables
        """

        env_settings = {}
        if dedup_field_name is None:
            dedup_field_name = set()
        if settings_model is None:
            settings_model = CLIEnvSettingsModel

        dict_vars = {k.replace("-", "_").upper(): v for k, v in dict_vars.items()}
        for field_name, field_detail in settings_model.model_fields.items():
            if (
                field_name not in _no_duplicate_field_name
                and field_name in dedup_field_name
            ):
                log.critical(f"duplicate field name: {field_name}")
                raise ValueError(f"duplicate field name: {field_name}")
            dedup_field_name.add(field_name)
            if field_detail.default_factory is not None:
                if recursion_depth > 0:
                    raise ValueError("not supported nested settings models")
                parsed = self.parse_dict_vars(
                    dedup_field_name,
                    recursion_depth + 1,
                    field_detail.default_factory,
                    dict_vars,
                    prefix=prefix,
                )
                if parsed:
                    env_settings[field_name] = parsed
            else:
                type_hint = typing.get_type_hints(settings_model)[field_name]
                env_name = f"{prefix}{field_name.upper()}"

                if env_name in dict_vars:
                    # Get the value from environment
                    env_value = dict_vars[env_name]

                    # Get type information for proper conversion
                    origin_type = get_origin(type_hint)
                    type_args = get_args(type_hint)

                    # Convert the value to the appropriate type
                    try:
                        converted_value = self._convert_env_value(
                            env_value, type_hint, origin_type, type_args
                        )
                        # Add to settings dict
                        env_settings[field_name] = converted_value
                    except (ValueError, TypeError) as e:
                        log.warning(
                            f"Could not convert environment variable {env_name}: {e}"
                        )
                    else:
                        log.debug(f"Field {field_name} not found in type hints")

        if recursion_depth == 0:
            log.debug(f"Environment settings: {env_settings}")
        return env_settings

    def _convert_env_value(
        self, value: str, type_hint: Any, origin_type: Any, type_args: tuple
    ) -> Any:
        """
        Convert environment variable string value to the appropriate type

        Args:
            value: String value from environment variable
            type_hint: The type hint for the field
            origin_type: The origin type (e.g., list, dict)
            type_args: Type arguments for generic types

        Returns:
            Converted value matching the expected type
        """
        # Handle Union types (including Optional)
        if origin_type is typing.Union:
            # Try each type in the Union until one works
            for arg in type_args:
                if arg is NoneType and value.lower() in ("none", ""):
                    return None
                try:
                    return self._convert_env_value(
                        value, arg, get_origin(arg), get_args(arg)
                    )
                except (ValueError, TypeError):
                    continue
            # If no conversion worked, raise error
            raise ValueError(
                f"Could not convert '{value}' to any of the types: {type_args}"
            )
        elif origin_type is set:
            if isinstance(value, list):
                return set(value)
            elif isinstance(value, set):
                return value
            elif isinstance(value, str):
                literal_eval = ast.literal_eval(value)
                if literal_eval is None:
                    return set()
                if isinstance(literal_eval, set):
                    return literal_eval
                if isinstance(literal_eval, list):
                    return set(literal_eval)
            else:
                raise ValueError(f"Could not convert '{value}' to set")
        # Handle basic types
        if type_hint is bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ("true", "1", "yes", "y", "on")
        elif type_hint is int:
            return int(value)
        elif type_hint is float:
            return float(value)
        elif type_hint is str:
            return value
        args = typing.get_args(type_hint)
        for arg in args:
            if arg is bool:
                if isinstance(value, bool):
                    return value
                return value.lower() in ("true", "1", "yes", "y", "on")
            elif arg is int:
                return int(value)
            elif arg is float:
                return float(value)
            elif arg is str:
                return value
        # For other types, try direct conversion
        return type_hint(value)

    def _deep_merge(self, target: dict, source: dict) -> dict:
        """
        Deep merge two dictionaries

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from

        Returns:
            Merged dictionary
        """
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            elif (
                key in target
                and isinstance(target[key], list)
                and isinstance(value, list)
            ):
                target[key].extend(value)
            else:
                target[key] = value
        return target

    def merge_settings(self, config_dicts: list[dict]) -> dict:
        """
        Merge multiple configuration dictionaries based on priority

        Args:
            config_dicts: List of config dictionaries, ordered by priority (highest first)

        Returns:
            Merged configuration dictionary
        """
        result = {}
        enabled_engine = None
        enabled_term_engine = None

        # Determine which translation and term engines are enabled based on priority.
        # config_dicts is ordered from highest to lowest priority, so once an engine
        # type is selected, it must not be overridden by lower-priority configs.
        for config in config_dicts:
            if enabled_engine is None:
                for engine_name in _translation_engine_flag_names:
                    if config.get(engine_name, False):
                        enabled_engine = engine_name
                        break

            if enabled_term_engine is None:
                for term_engine_name in _term_translation_engine_flag_names:
                    if config.get(term_engine_name, False):
                        enabled_term_engine = term_engine_name
                        break

            # If both engine types have been selected we can stop scanning for flags.
            if enabled_engine is not None and enabled_term_engine is not None:
                break

        # Process from lowest to highest priority
        for config in reversed(config_dicts):
            # Deep merge config into result
            self._deep_merge(result, config)

        if enabled_engine:
            for engine_name in _translation_engine_flag_names:
                if engine_name in result:
                    result[engine_name] = False
            result[enabled_engine] = True

        if enabled_term_engine:
            for engine_name in _term_translation_engine_flag_names:
                if engine_name in result:
                    result[engine_name] = False
            result[enabled_term_engine] = True

        return result

    def initialize_config(self) -> SettingsModel:
        """Initialize configuration from all sources"""
        # Parse CLI arguments (highest priority)
        self._settings = self.initialize_cli_config().to_settings_model()
        log.debug(f"Initialized settings: {self._settings.model_dump_json()}")

        return self._settings

    def test_config(self, args):
        try:
            self._build_model_from_args(CLIEnvSettingsModel, args)
            return True
        except Exception as e:
            log.exception("Error in test_config:")
            return False

    def initialize_cli_config(self) -> CLIEnvSettingsModel:
        parser, _ = build_args_parser()
        args = parser.parse_args()
        cli_args: dict[str, Any] = {
            k.replace("-", "_"): v
            for k, v in vars(args).items()
            if v is not MagicDefault
        }
        cli_parsed_args = self.parse_dict_vars(
            dict_vars=cli_args,
        )
        # Parse environment variables (middle priority)
        env_vars = self.parse_env_vars()
        # Read default configuration file (lower priority)
        default_config_file = self._read_toml_file(self._default_config_file_path)

        if not self.test_config(default_config_file):
            default_config_file = {}
            log.error(
                f"Error in test_config: {self._default_config_file_path}, skip it"
            )

        # Merge all settings by priority
        merged_args = self.merge_settings(
            [
                cli_parsed_args,
                env_vars,
            ]
        )
        config_from_file_dicts = []
        if "config_file" in merged_args:
            user_config = self._read_toml_file(Path(merged_args["config_file"]))
            if not self.test_config(user_config):
                log.error(
                    f"Error in test_config: {merged_args['config_file']}, skip it"
                )
                user_config = {}

            config_from_file_dicts.append(copy.deepcopy(user_config))
            del merged_args["config_file"]
            merged_args = self.merge_settings(
                [merged_args, user_config, default_config_file]
            )
        else:
            merged_args = self.merge_settings([merged_args, default_config_file])
        # Create settings model from merged dictionary
        self._update_version_default_config()
        config_from_file_dicts.append(copy.deepcopy(default_config_file))
        config_from_file_args = self.merge_settings(config_from_file_dicts)

        self.config_cli_settings = self._build_model_from_args(
            CLIEnvSettingsModel, config_from_file_args
        )

        cli_settings = self._build_model_from_args(CLIEnvSettingsModel, merged_args)
        cli_settings.validate_settings()
        return cli_settings

    def write_user_default_config_file(self, settings: CLIEnvSettingsModel):
        # clear input file
        settings.basic.input_files = set()

        content = settings.model_dump(mode="json")
        if self._is_file_content_identical(DEFAULT_CONFIG_FILE, content):
            log.info(
                f"Config file {DEFAULT_CONFIG_FILE} is identical to the settings, skip it"
            )
            return

        self._write_toml_file(WRITE_TEMP_CONFIG_FILE, content)
        WRITE_TEMP_CONFIG_FILE.replace(DEFAULT_CONFIG_FILE)
        log.info(f"Written user default config file to {DEFAULT_CONFIG_FILE}")

    def _build_model_from_args(
        self, model_class: type[BaseModel], args_dict: dict
    ) -> BaseModel:
        """
        Recursively build model instances from arguments dictionary

        Args:
            model_class: Pydantic model class to instantiate
            args_dict: Dictionary of arguments

        Returns:
            Instance of the specified model class
        """

        # Create and return the model instance
        return model_class(**args_dict)

    @property
    def settings(self) -> SettingsModel:
        """Get current settings"""
        if self._settings is None:
            raise RuntimeError("Settings not initialized")
        return self._settings


# disable not necessary log for normal usage
log.setLevel(logging.INFO)

if __name__ == "__main__":
    # only for debug
    log.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    config_manager = ConfigManager()
    # Use the new initialization method that combines CLI and env vars
    config_manager.initialize_config()
