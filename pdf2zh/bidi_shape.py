"""Bidirectional text layout and complex-script shaping.

This module turns a logical-order translated string into positioned glyphs.
It knows nothing about PDF: it takes text, a font, a size and a box, and it
returns lines of runs whose glyphs already carry final advances and offsets.

The pipeline follows UAX #9 (Unicode Bidirectional Algorithm) and delegates
OpenType shaping to HarfBuzz:

    protect placeholders   -> {vN} becomes U+FFFC so bidi cannot reorder it apart
    resolve levels         -> per-character embedding levels (logical order)
    mirror                 -> UAX #9 rule L4, which python-bidi does NOT apply
    segment runs           -> maximal spans of one level / font / script
    shape                  -> HarfBuzz, per run, in that run's direction
    break lines            -> cluster-safe, word-aware, never mid-cluster
    reorder per line       -> UAX #9 rule L2, applied to each line's runs
    place                  -> assign absolute x, starting from the correct edge

See docs/RTL_ARABIC_SUPPORT.md for the full specification and the defect IDs
referenced in the comments below.
"""

from __future__ import annotations

import bisect
import functools
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import uharfbuzz as hb

from pdf2zh import uba

log = logging.getLogger(__name__)

# U+FFFC OBJECT REPLACEMENT CHARACTER.  Bidi class ON, Bidi_Mirrored=No, so it
# behaves as an inline object and survives reordering intact (defect A08).
OBJ = "￼"

# Same tolerance the original renderer used for the right boundary.
EDGE_TOLERANCE_RATIO = 0.1


# ---------------------------------------------------------------------------
# Language tables
# ---------------------------------------------------------------------------

# Scripts written right-to-left.  Keys are lowercase, prefix-matched on "-".
RTL_LANGS = {
    "ar",  # Arabic
    "arc",  # Aramaic
    "ckb",  # Sorani Kurdish
    "dv",  # Divehi
    "fa",  # Persian
    "he",  # Hebrew
    "iw",  # Hebrew (legacy code, used by Google Translate)
    "ps",  # Pashto
    "sd",  # Sindhi
    "syr",  # Syriac
    "ug",  # Uyghur
    "ur",  # Urdu
    "yi",  # Yiddish
}

# Combining marks that belong to ordinary text rather than mathematics.
# vflag() in converter.py must not treat these as formula characters (E01).
TEXT_MARK_RANGES = (
    (0x0300, 0x036F),  # combining diacritics (Latin / Greek / Cyrillic)
    (0x0483, 0x0489),  # Cyrillic combining marks
    (0x0591, 0x05C7),  # Hebrew niqqud and cantillation
    (0x0610, 0x061A),  # Arabic honorifics
    (0x064B, 0x065F),  # Arabic harakat
    (0x0670, 0x0670),  # Arabic superscript alef
    (0x06D6, 0x06ED),  # Quranic annotation marks
    (0x0E31, 0x0E3A),  # Thai
    (0x0E47, 0x0E4E),  # Thai
)

ARABIC_TATWEEL = 0x0640

# Arabic harakat, used to pick the tight vs. safe line-height floor (B01).
_TASHKEEL = frozenset(range(0x064B, 0x0653)) | {0x0670}

WESTERN_DIGITS = "0123456789"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def is_rtl_lang(lang: str) -> bool:
    """True when `lang` is written right-to-left."""
    lang = (lang or "").lower().replace("_", "-")
    if lang in RTL_LANGS:
        return True
    return lang.split("-")[0] in RTL_LANGS


def is_text_mark(cp: int) -> bool:
    """True for combining marks that are text, not mathematics (E01)."""
    return any(lo <= cp <= hi for lo, hi in TEXT_MARK_RANGES)


def has_tashkeel(text: str) -> bool:
    return any(ord(c) in _TASHKEEL for c in text)


def normalize_digits(text: str, form: str) -> str:
    """Apply the configured digit form (D05).  `auto` passes text through."""
    if form == "western":
        return text.translate(str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS))
    if form == "arabic":
        return text.translate(str.maketrans(WESTERN_DIGITS, ARABIC_INDIC_DIGITS))
    return text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ShapedGlyph:
    """One positioned glyph.  Advances and offsets are in points at the shaped size."""

    gid: int  # glyph id; equals the CID under Identity-H
    x_advance: float
    x_offset: float
    y_offset: float
    chars: tuple[int, ...]  # source codepoints of this cluster, for ToUnicode
    nominal_width: float  # unshaped hmtx advance, to detect forced positioning

    @property
    def force_positioning(self) -> bool:
        """True when this glyph cannot ride on the natural advance of the run."""
        return (
            self.x_offset != 0.0
            or self.y_offset != 0.0
            or abs(self.x_advance - self.nominal_width) > 1e-6
        )

    @property
    def is_space(self) -> bool:
        return self.chars == (0x20,)


@dataclass
class TextRun:
    """A maximal span with one embedding level, one font and one script.

    `glyphs` is always stored in LOGICAL order so that line breaking is
    direction-agnostic.  Use `visual_glyphs()` when emitting.
    """

    text: str
    level: int
    font_key: str
    script: str
    language: str
    size: float
    glyphs: list[ShapedGlyph] = field(default_factory=list)

    @property
    def rtl(self) -> bool:
        return bool(self.level % 2)

    @property
    def width(self) -> float:
        return sum(g.x_advance for g in self.glyphs)

    def visual_glyphs(self) -> list[ShapedGlyph]:
        """Glyphs in visual (left-to-right) order, ready for emission."""
        return list(reversed(self.glyphs)) if self.rtl else list(self.glyphs)

    def slice(self, start: int, stop: int) -> "TextRun":
        """A copy carrying glyphs[start:stop].  Used by the line breaker."""
        return TextRun(
            text=self.text,
            level=self.level,
            font_key=self.font_key,
            script=self.script,
            language=self.language,
            size=self.size,
            glyphs=self.glyphs[start:stop],
        )


@dataclass
class PlaceholderRun:
    """A {vN} formula box.  Occupies width but carries no glyphs."""

    vid: int
    width: float
    level: int

    @property
    def rtl(self) -> bool:
        return bool(self.level % 2)


Item = TextRun | PlaceholderRun


@dataclass
class PlacedItem:
    """An item with its absolute left edge resolved."""

    item: Item
    x: float


@dataclass
class LayoutLine:
    items: list[Item]  # logical order; reorder_line() produces visual order
    width: float


@dataclass
class LayoutResult:
    lines: list[list[PlacedItem]]
    base_rtl: bool
    overflow_x: float  # how far the widest line exceeds the box (0 when it fits)
    max_ascent: float  # ink above the baseline on the first line, for B03


# ---------------------------------------------------------------------------
# S1 - placeholder protection (A08)
# ---------------------------------------------------------------------------

# Kept identical to the regex in converter.py so translator quirks such as
# "{ v 0 }" and "{V0}" keep working.  Python's \d is Unicode-aware and int()
# accepts Arabic-Indic digits, so "{v٥}" still resolves to 5.
_VN_RE = re.compile(r"\{\s*v([\d\s]+)\}", re.IGNORECASE)

# Machine translation engines rewrite the marker text itself.  Google, for
# example, reads "v4" as an abbreviation and returns "{aya 4}" in Arabic with
# Arabic-Indic digits.  Such a marker no longer matches _VN_RE, so it used to
# survive into the output as literal braces.  This looser pattern recovers any
# brace group whose content carries exactly one run of digits.
_VN_FUZZY_RE = re.compile(r"\{[^{}]{0,32}\}")
_DIGITS_RE = re.compile(r"\d+")


def _placeholder_vid(marker: str, valid: set[int] | None) -> int | None:
    """Recover the formula id from a marker, tolerating translator damage."""
    exact = _VN_RE.fullmatch(marker)
    if exact:
        try:
            vid = int(exact.group(1).replace(" ", ""))
        except ValueError:
            return None
        # Translators sometimes invent markers that were never in the source.
        if valid is not None and vid not in valid:
            return None
        return vid
    # Fuzzy recovery: exactly one number inside the braces, and it names a
    # formula that actually exists in this paragraph.
    numbers = _DIGITS_RE.findall(marker)
    if len(numbers) != 1:
        return None
    try:
        vid = int(numbers[0])  # int() accepts Arabic-Indic digits
    except ValueError:
        return None
    if valid is not None and vid not in valid:
        return None
    return vid


def protect_placeholders(
    text: str, valid_vids: set[int] | None = None
) -> tuple[str, dict[int, int]]:
    """Replace every {vN} formula marker with U+FFFC.

    Returns (protected_text, {index_in_protected_text: vid}).  Because the bidi
    stage carries original character indices, each sentinel is matched back to
    its vid by index even when several placeholders share a paragraph.

    `valid_vids` is the set of formula ids that exist in this paragraph; it
    gates the fuzzy recovery above so that ordinary braces in the text are not
    mistaken for markers.
    """
    out: list[str] = []
    marks: dict[int, int] = {}
    length = 0
    pos = 0
    for m in _VN_FUZZY_RE.finditer(text):
        vid = _placeholder_vid(m.group(0), valid_vids)
        if vid is None:  # ordinary braces, keep as literal text
            continue
        chunk = text[pos : m.start()]
        out.append(chunk)
        length += len(chunk)
        marks[length] = vid
        out.append(OBJ)
        length += 1
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), marks


# ---------------------------------------------------------------------------
# S2 - bidi resolution (A02, D02, D06)
# ---------------------------------------------------------------------------


@dataclass
class BidiChar:
    ch: str
    level: int
    index: int  # index into the protected text


def detect_base_rtl(text: str, lang_out: str) -> bool:
    """UAX #9 rules P2/P3: first strong character wins, target language breaks ties.

    A pure-Latin citation inside an Arabic document must stay LTR, so the
    paragraph direction is content-derived rather than forced by lang_out.
    """
    for ch in text:
        cls = unicodedata.bidirectional(ch)
        if cls == "L":
            return False
        if cls in ("R", "AL"):
            return True
    return is_rtl_lang(lang_out)


def resolve_levels(text: str, base_rtl: bool) -> list[BidiChar]:
    """UAX #9 levels for `text`, in LOGICAL order.

    Delegates to pdf2zh.uba, which passes the full Unicode BidiCharacterTest
    conformance suite.  We need per-character embedding levels rather than just
    a reordered string: the levels are what let us segment text into shaping
    runs and track formula placeholders through reordering.

    Characters removed by rule X9 (explicit formatting codes) are dropped here
    so they never reach the shaper.

    NOTE: rule L4 (mirroring) is deliberately NOT applied here.  HarfBuzz
    mirrors mirrorable characters itself for any buffer whose direction is
    RTL, and because every run is shaped with the direction implied by its own
    embedding level, that is exactly L4.  Applying it here as well double
    mirrors: the two cancel and brackets come out backwards, e.g. an Arabic
    parenthetical renders as ")text(" instead of "(text)".
    """
    levels, _ = uba.resolve(text, 1 if base_rtl else 0)
    out: list[BidiChar] = []
    for i, ch in enumerate(text):
        if uba.is_removed(ch):
            continue
        out.append(BidiChar(ch=ch, level=levels[i], index=i))
    return out


def reorder_l2(items: list, level_of) -> list:
    """UAX #9 rule L2 over any sequence exposing a level.  Returns visual order."""
    if not items:
        return list(items)
    levels = [level_of(it) for it in items]
    return [items[i] for i in uba.reorder_indices(levels)]


# ---------------------------------------------------------------------------
# S3 - run segmentation (A10, B06, D01, D04)
# ---------------------------------------------------------------------------


def is_common_script(ch: str) -> bool:
    """True for script-neutral characters (spaces, ASCII punctuation, digits).

    These inherit the surrounding run's script so that a word gap does not
    fragment a phrase into three separately positioned runs.
    """
    cp = ord(ch)
    if cp in (0x20, 0x09, 0x00A0, 0x200B):
        return True
    if 0x21 <= cp <= 0x40 or 0x5B <= cp <= 0x60 or 0x7B <= cp <= 0x7E:
        return True
    return False


def script_of(ch: str) -> str:
    """ISO 15924 script tag for HarfBuzz."""
    cp = ord(ch)
    if (
        0x0600 <= cp <= 0x06FF
        or 0x0750 <= cp <= 0x077F
        or 0x08A0 <= cp <= 0x08FF
        or 0xFB50 <= cp <= 0xFDFF
        or 0xFE70 <= cp <= 0xFEFF
    ):
        return "Arab"
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return "Hebr"
    if 0x0700 <= cp <= 0x074F:
        return "Syrc"
    if 0x0780 <= cp <= 0x07BF:
        return "Thaa"
    return "Latn"


def segment_runs(
    chars: list[BidiChar],
    marks: dict[int, int],
    *,
    font_key: str,
    language: str,
    size: float,
    placeholder_width,
) -> list[Item]:
    """Split logical-order characters into runs of one level / font / script.

    RTL paragraphs use a single font for everything - Arabic letters, embedded
    Latin, digits, spaces and punctuation alike - so one harmonised family is
    used throughout (D01, D04, B06).
    """
    items: list[Item] = []
    for c in chars:
        if c.ch == OBJ:
            vid = marks.get(c.index)
            if vid is None:  # sentinel without a recorded vid; drop it
                continue
            items.append(
                PlaceholderRun(
                    vid=vid, width=float(placeholder_width(vid)), level=c.level
                )
            )
            continue
        last = items[-1] if items else None
        if is_common_script(c.ch) and isinstance(last, TextRun):
            script = last.script  # inherit, so a space does not split the run
        else:
            script = script_of(c.ch)
        if (
            isinstance(last, TextRun)
            and last.level == c.level
            and last.font_key == font_key
            and last.script == script
        ):
            last.text += c.ch
        else:
            items.append(
                TextRun(
                    text=c.ch,
                    level=c.level,
                    font_key=font_key,
                    script=script,
                    language=language,
                    size=size,
                )
            )
    return items


# ---------------------------------------------------------------------------
# S4 - shaping (A01, A04, A05, A06)
# ---------------------------------------------------------------------------


class Shaper:
    """Wraps one font file for HarfBuzz shaping.

    Instances are cached per font path; shaping results are cached per
    (text, direction, script, language) because real documents repeat runs
    heavily.
    """

    def __init__(self, font_path: str):
        self.font_path = font_path
        blob = hb.Blob.from_file_path(font_path)
        self.face = hb.Face(blob)
        self.font = hb.Font(self.face)
        self.upem = self.face.upem or 1000
        self._missing: set[int] = set()

    @functools.lru_cache(maxsize=4096)
    def _nominal(self, gid: int) -> float:
        try:
            return float(self.font.get_glyph_h_advance(gid))
        except Exception:
            return 0.0

    @functools.lru_cache(maxsize=8192)
    def _shape_raw(self, text: str, direction: str, script: str, language: str):
        buf = hb.Buffer()
        # Monotone character clusters keep cluster indices mappable back to the
        # source string, which is what makes ligature ToUnicode possible.
        buf.cluster_level = 1
        buf.add_str(text)
        buf.direction = direction
        buf.script = script
        buf.language = language
        hb.shape(self.font, buf, {"kern": True, "liga": True})
        infos = [(gi.codepoint, gi.cluster) for gi in buf.glyph_infos]
        positions = [
            (p.x_advance, p.x_offset, p.y_offset) for p in buf.glyph_positions
        ]
        return infos, positions

    def shape(
        self, text: str, *, rtl: bool, script: str, language: str, size: float
    ) -> list[ShapedGlyph]:
        """Shape `text` (logical order).  Returns glyphs in LOGICAL order."""
        if not text:
            return []
        direction = "rtl" if rtl else "ltr"
        infos, positions = self._shape_raw(text, direction, script, language)
        scale = size / self.upem

        # cluster -> source character indices, so a ligature glyph keeps every
        # codepoint it came from (needed for ToUnicode, defect A11).
        clusters = sorted({c for _, c in infos})
        mapping: dict[int, list[int]] = {}
        for i in range(len(text)):
            pos = bisect.bisect_right(clusters, i) - 1
            mapping.setdefault(clusters[max(pos, 0)], []).append(i)

        glyphs: list[ShapedGlyph] = []
        for (gid, cluster), (x_adv, x_off, y_off) in zip(infos, positions):
            if gid == 0:  # .notdef would render as a visible tofu box
                for i in mapping.get(cluster, ()):
                    cp = ord(text[i])
                    if cp not in self._missing:
                        self._missing.add(cp)
                        log.debug("no glyph for U+%04X in %s", cp, self.font_path)
                continue
            glyphs.append(
                ShapedGlyph(
                    gid=gid,
                    x_advance=x_adv * scale,
                    x_offset=x_off * scale,
                    y_offset=y_off * scale,
                    chars=tuple(ord(text[i]) for i in mapping.get(cluster, ())),
                    nominal_width=self._nominal(gid) * scale,
                )
            )
        # HarfBuzz returns RTL runs in visual order; store logical order so the
        # line breaker can stay direction-agnostic.
        if rtl:
            glyphs.reverse()
        return glyphs

    def glyph_ascent(self, gid: int, size: float) -> float:
        """Ink height above the baseline for one glyph, in points (B03)."""
        try:
            ext = self.font.get_glyph_extents(gid)
        except Exception:
            return 0.0
        if ext is None:
            return 0.0
        # HarfBuzz y_bearing is the top of the ink box, measured from the baseline.
        return float(ext.y_bearing) * size / self.upem


@functools.lru_cache(maxsize=8)
def get_shaper(font_path: str) -> Shaper:
    return Shaper(font_path)


# ---------------------------------------------------------------------------
# S5 - line breaking (A09, C01)
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # hangul
    (0xF900, 0xFAFF),  # CJK compatibility
)


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _break_after(glyph: ShapedGlyph) -> bool:
    """True when a line may break immediately after this glyph."""
    if not glyph.chars:
        return False
    cp = glyph.chars[-1]
    if cp in (0x20, 0x09, 0x200B):  # space, tab, ZWSP
        return True
    if cp in (0x2D, 0x00AD):  # hyphen, soft hyphen
        return True
    return _is_cjk(cp)


def break_lines(
    items: list[Item], first_width: float, body_width: float
) -> list[LayoutLine]:
    """Greedy wrap over runs.  Never splits a shaped cluster.

    Wrapping is unconditional: unlike the original renderer it is not gated on
    the source paragraph having had a line break, so single-line sources such
    as headings and table cells wrap instead of running off the page (C01).

    A word wider than the whole box is left to overflow rather than being cut
    mid-cluster; the caller sees it through LayoutResult.overflow_x and can
    shrink the font.
    """
    # Flatten to atoms: (item index, glyph index or None, width, breakable, space)
    atoms: list[tuple[int, int | None, float, bool, bool]] = []
    for ii, item in enumerate(items):
        if isinstance(item, PlaceholderRun):
            atoms.append((ii, None, item.width, False, False))
        else:
            for gi, glyph in enumerate(item.glyphs):
                atoms.append(
                    (ii, gi, glyph.x_advance, _break_after(glyph), glyph.is_space)
                )
    if not atoms:
        return []

    lines: list[LayoutLine] = []
    start = 0
    width = 0.0
    limit = first_width
    last_opportunity: int | None = None

    i = 0
    while i < len(atoms):
        advance = atoms[i][2]
        if width + advance > limit and i > start:
            if last_opportunity is not None and last_opportunity >= start:
                cut = last_opportunity + 1
            else:
                cut = i  # no word boundary on this line: cut at a cluster edge
            lines.append(_build_line(items, atoms, start, cut))
            start = cut
            while start < len(atoms) and atoms[start][4]:
                start += 1  # a wrapped line never begins with a space
            i = max(i, start)
            width = sum(a[2] for a in atoms[start:i])
            limit = body_width
            last_opportunity = None
            continue
        width += advance
        if atoms[i][3]:
            last_opportunity = i
        i += 1

    if start < len(atoms):
        lines.append(_build_line(items, atoms, start, len(atoms)))
    return [line for line in lines if line.items]


def _build_line(
    items: list[Item],
    atoms: list[tuple[int, int | None, float, bool, bool]],
    start: int,
    stop: int,
) -> LayoutLine:
    """Rebuild runs for atoms[start:stop], trimming trailing spaces."""
    while stop > start and atoms[stop - 1][4]:
        stop -= 1  # trailing spaces must not count toward the line width
    out: list[Item] = []
    k = start
    while k < stop:
        index = atoms[k][0]
        j = k
        while j < stop and atoms[j][0] == index:
            j += 1
        item = items[index]
        if isinstance(item, PlaceholderRun):
            out.append(item)
        else:
            first_glyph = atoms[k][1] or 0
            last_glyph = atoms[j - 1][1]
            assert last_glyph is not None
            out.append(item.slice(first_glyph, last_glyph + 1))
        k = j
    return LayoutLine(items=out, width=sum(_item_width(i) for i in out))

def _item_width(item: Item) -> float:
    return item.width


# ---------------------------------------------------------------------------
# S6 - placement (A03, B04, B05)
# ---------------------------------------------------------------------------


def _level_of(item: Item) -> int:
    return item.level


def place_line(
    line: LayoutLine, x0: float, x1: float, base_rtl: bool, indent: float
) -> list[PlacedItem]:
    """Reorder a line to visual order (L2) and assign absolute left edges.

    Once bidi has produced visual order, placement is a left-to-right sweep in
    both directions.  Direction only changes which edge the line starts from.
    """
    visual = reorder_l2(line.items, _level_of)
    if base_rtl:
        pen = x1 - indent - line.width
    else:
        pen = x0 + indent
    placed: list[PlacedItem] = []
    for item in visual:
        placed.append(PlacedItem(item=item, x=pen))
        pen += _item_width(item)
    return placed


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def layout_paragraph(
    text: str,
    *,
    font_path: str,
    font_key: str,
    size: float,
    x0: float,
    x1: float,
    indent: float = 0.0,
    lang_out: str = "",
    base_rtl: bool | None = None,
    placeholder_width=lambda vid: 0.0,
    valid_vids: set[int] | None = None,
    digit_form: str = "auto",
) -> LayoutResult:
    """Lay out one translated paragraph.

    Returns lines of placed items whose glyphs carry final advances and
    offsets.  The caller is responsible for the vertical position of each line.
    """
    text = normalize_digits(text, digit_form)
    protected, marks = protect_placeholders(text, valid_vids)
    if base_rtl is None:
        base_rtl = detect_base_rtl(protected, lang_out)

    chars = resolve_levels(protected, base_rtl)

    language = (lang_out or "").lower().split("-")[0] or "en"
    items = segment_runs(
        chars,
        marks,
        font_key=font_key,
        language=language,
        size=size,
        placeholder_width=placeholder_width,
    )

    shaper = get_shaper(font_path)
    for item in items:
        if isinstance(item, TextRun):
            item.glyphs = shaper.shape(
                item.text,
                rtl=item.rtl,
                script=item.script,
                language=item.language,
                size=size,
            )
    items = [
        it for it in items if isinstance(it, PlaceholderRun) or it.glyphs
    ]

    body_width = max(x1 - x0, 1e-6)
    first_width = max(body_width - indent, 1e-6)
    lines = break_lines(items, first_width, body_width)

    placed_lines = [place_line(ln, x0, x1, base_rtl, indent) for ln in lines]

    overflow = 0.0
    for ln in lines:
        overflow = max(overflow, ln.width - body_width)

    max_ascent = 0.0
    if placed_lines:
        for placed in placed_lines[0]:
            if isinstance(placed.item, TextRun):
                for g in placed.item.glyphs:
                    max_ascent = max(
                        max_ascent, shaper.glyph_ascent(g.gid, size) + g.y_offset
                    )

    return LayoutResult(
        lines=placed_lines,
        base_rtl=base_rtl,
        overflow_x=max(overflow, 0.0),
        max_ascent=max_ascent,
    )


# ---------------------------------------------------------------------------
# Self-test: guards python-bidi's internal API
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Cheap wiring check for the bidi engine.

    Full UAX #9 conformance is covered by test/test_uba.py, which runs the
    Unicode BidiCharacterTest suite.  This only catches gross misconfiguration.
    """
    # Latin letter followed by two Arabic letters, in an LTR paragraph:
    # the Latin stays at level 0, the Arabic rises to level 1.
    chars = resolve_levels("aاب", base_rtl=False)
    if [c.level for c in chars] != [0, 1, 1]:
        raise RuntimeError(
            "bidi engine misconfigured: unexpected embedding levels "
            f"{[c.level for c in chars]}"
        )
    # Numbers must survive reordering inside RTL text.
    chars = resolve_levels("من 1990", base_rtl=True)
    visual = "".join(c.ch for c in reorder_l2(chars, lambda c: c.level))
    if "1990" not in visual:
        raise RuntimeError("bidi engine misconfigured: digit run was reversed")
