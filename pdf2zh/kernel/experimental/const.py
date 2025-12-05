__version__ = "2.7.1"
__major_version__ = "2"
__config_file_version__ = "3"

from pathlib import Path

# Constants for configuration paths
DEFAULT_CONFIG_DIR = Path("~/.config/pdf2zh").expanduser()
DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / f"config.v{__config_file_version__}.toml"
WRITE_TEMP_CONFIG_FILE = (
    DEFAULT_CONFIG_DIR / f"config.v{__config_file_version__}.temp.toml"
)
VERSION_DEFAULT_CONFIG_DIR = DEFAULT_CONFIG_DIR / "default"
VERSION_DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
VERSION_DEFAULT_CONFIG_FILE = VERSION_DEFAULT_CONFIG_DIR / f"{__version__}.toml"
