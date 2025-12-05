from gradio_i18n import gettext
from gradio_i18n.i18n import TranslateContext

LANGUAGES = [
    ("English", "en"),
    ("简体中文", "zh"),
    ("繁體中文", "zh-TW"),
    ("日本語", "ja"),
    ("한국인", "ko"),
    ("Français", "fr"),
    ("Deutsch", "de"),
    ("Español", "es"),
    ("Русский", "ru"),
    ("Italiano", "it"),
    ("Português", "pt"),
]

_ = gettext


def update_current_languages(lang):
    # Hook explanation:
    # Sometimes Gradio i18n can't get the language for the current request and falls back to the default.
    # The library doesn't expose a way to manually set the default, and it uses a set to store available languages.
    # The default language is the first one in the available list.
    # So in some cases requests randomly pick a language...
    # Here we simply hook it to force a specific language feature.

    if lang not in TranslateContext.available_languages:
        return

    def always_return():
        return lang

    TranslateContext.get_default_language = always_return
