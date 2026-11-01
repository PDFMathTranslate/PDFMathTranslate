"""Spacing for mixed CJK and Latin text before PDF typesetting."""

from functools import lru_cache
import unicodedata


@lru_cache(maxsize=4096)
def _script(char: str) -> str:
    name = unicodedata.name(char, "")
    if unicodedata.category(char).startswith("L"):
        if name.startswith("LATIN "):
            return "latin"
        if name.startswith(
            (
                "CJK ",
                "HIRAGANA ",
                "KATAKANA ",
                "HANGUL ",
                "HALFWIDTH KATAKANA ",
                "HALFWIDTH HANGUL ",
            )
        ):
            return "cjk"
    return ""


def add_cjk_latin_spacing(text: str) -> str:
    """Separate touching CJK and Latin letters, preserving marks and whitespace.

    Punctuation, numbers and formula placeholders remain unchanged. Combining
    marks stay attached to their base letter, including decomposed accents.
    """
    result = []
    previous = ""
    for char in text:
        if unicodedata.category(char).startswith("M"):
            result.append(char)
            continue
        current = _script(char)
        if previous and current and previous != current:
            result.append(" ")
        result.append(char)
        previous = current
    return "".join(result)
