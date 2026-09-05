# RTL & Arabic Support — Implementation Specification

> **Status:** specification / not yet implemented
> **Target:** `pdf2zh` v1 kernel (`pdf2zh/converter.py` rendering path)
> **Scope:** full bidirectional (RTL) support with production-quality Arabic typesetting, without changing the project's existing architecture, op-emission format, or LTR behaviour.

---

## 0. Executive summary

`pdf2zh` cannot render Arabic (or Hebrew, Persian, Urdu) today. This is not a collection of independent bugs — it follows from a single architectural assumption baked into `TranslateConverter.receive_layout` section C: **a character-indexed, left-to-right, one-character-one-glyph, monotonically increasing pen.**

Arabic requires the opposite pipeline:

```
translated text (logical order)
  -> placeholder protection
  -> Unicode Bidirectional Algorithm (UAX #9) -> embedding levels
  -> level runs INTERSECT font runs
  -> HarfBuzz shaping per run (logical order, per-run direction)
  -> glyph clusters with advances + x/y offsets
  -> direction-aware line breaking
  -> direction-aware pen placement
  -> PDF content stream
```

The work is contained. `Paragraph`, the formula stack, the layout model, the translator layer and the `gen_op_txt` / `gen_op_line` emission format are all unchanged. LTR languages route through the identical new path with `direction=ltr` and must produce byte-identical output (guarded by a regression test).

Two enabling facts, both verified against the real assets:

- **The bundled font is already fully capable.** `GoNotoKurrent-Regular.ttf` (what `download_remote_fonts()` selects for `ar`) carries `arab` script in both GSUB (`init, medi, fina, isol, rlig, liga, calt, ccmp`) and GPOS (`mark, mkmk, curs`), 255 Arabic-block codepoints, 752 presentation forms, 64,750 glyphs.
- **No font-embedding change is needed.** PyMuPDF `insert_font()` embeds the font as a Type0 / Identity-H CID font with an identity CID→GID map, and `raw_string()` already writes raw glyph IDs as `%04x`. HarfBuzz glyph IDs can therefore be written straight into the existing op format. 64,750 < 65,536, so the 4-hex-digit encoding stays valid.

---

## 1. Why the current pipeline cannot render Arabic

`pdf2zh/pdfinterp.py:325-340` strips **every** text operator (`name[0] == "T"`) from the original content stream. The translated page therefore contains no text except what `receive_layout` emits, and that emitter (`converter.py:385-386`) produces only:

```
/font size Tf 1 0 0 1 x y Tm [<GID GID GID...>] TJ
```

Absolute `Tm` per run, raw glyph IDs, no `TJ` kerning arrays, no `Tc` / `Tw` / `Tz`, no direction concept. PDF itself has **no bidi engine and no shaper** — the glyph sequence and pen positions we compute are exactly what the reader draws.

Every aspect of Arabic correctness must therefore be computed in Python before emission. None of it currently is.

---

## 2. Defect register

Severity: **B** = blocker (Arabic unusable/incorrect), **M** = major (visibly wrong or degraded), **N** = minor / enhancement, **I** = informational (verified non-issue).


> **Implementation status (updated after Phase 1-3):** 28 done, 2 partial, 4 open (B07, B08, D03, E03), 3 closed as verified non-issues.
> Landed in `pdf2zh/uba.py` (UAX #9, 100% conformant), `pdf2zh/bidi_shape.py`
> (shaping + layout), and the `converter.py` / `high_level.py` integration.

### 2.1 Output rendering — bidi & shaping

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| A01 | No Arabic shaping; every letter emitted in isolated form | `converter.py:368-374` `raw_string` does `has_glyph(ord(c))` per char. Measured `الكتاب`: current GIDs `3921,4541,4432,5151,3921,3975` vs correct `3975,3922,5155,4436,4543,3921` — zero overlap | B | S4 | **done** |
| A02 | No bidi reordering; logical order rendered with an LTR pen, so text reads backwards | `converter.py:409-500` | B | S2 | **done** |
| A03 | Pen is LTR-only: start `x=pstk.x` (`:392`), wrap test `x+adv > x1` (`:441`, `:454`), wrap reset `x=x0` (`:455`), leading-space drop `x==x0` (`:489`) | `converter.py` as cited | B | S6 | **done** |
| A04 | One char = one glyph assumption; cannot represent ligatures. `لا إله` = 6 codepoints → **5 glyphs** | `converter.py:409` loop, `:424-437`, `:368-374` | B | S4 | **done** |
| A05 | Advance widths taken from isolated forms — **+38% to +52% error** | `converter.py:434` `noto.char_lengths(ch,size)[0]`. Measured at 12pt: `الكتاب` 48.47 vs 31.96 (+51.7%); `لا إله` 30.37 vs 21.91 (+38.6%); `مرحبا بالعالم` 81.97 vs 57.82 (+41.8%); `كِتَابٌ` 37.27 vs 24.83 (+50.1%) | B | S4 | **done** |
| A06 | Combining marks (harakat) given a full advance and no offset, so diacritics detach and drift right | `converter.py:434`; the `mod` hack at `:421-422`, `:495` only covers a trailing mark of a formula group. Shaped `كِتَابٌ` yields `x_advance=0` with offsets `(302,-160)`, `(29,27)`, `(21,0)` | B | S4 | **done** |
| A07 | Formula `{vN}` glyphs drawn LTR from the pen, then `x += adv`, so formulas overlap the following text on RTL lines | `converter.py:457-485`, `:497` | B | S7 | **done** |
| A08 | **Formula placeholders are destroyed by bidi.** `{v0}` becomes `}v0{` after reordering, so `re.match(r"\{\s*v([\d\s]+)\}")` never matches | `converter.py:410-411`. Verified by codepoint: logical `007B 0076 0030 007D` → visual `007D 0076 0030 007B` | B | S1 | **done** |
| A09 | Line breaking is per-character, mid-word — severs Arabic joining runs | `converter.py:454` | M | S5 | **done** |
| A10 | Buffer flush keyed on font change only; bidi level runs are never a boundary | `converter.py:438-442` | B | S3 | **done** |
| A11 | No `ToUnicode` for shaped glyphs, so output Arabic is not selectable, searchable, or screen-reader accessible | `high_level.py:203-205`, `:246-248` | M | S10 | partial (cluster data collected) |

### 2.2 Spacing, edges & vertical metrics

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| B01 | Arabic line height `1.0` is below the measured collision threshold; consecutive lines overlap by **3.32pt** at size 12 | `converter.py:379`. Measured ink height: Arabic + tashkeel 15.32pt vs 12pt advance, so it needs **>= 1.28** | B | S8 | **done** |
| B02 | Fit loop can only shrink line height by ~0.05 and never below 1.0; `size` is never scaled | `converter.py:513-516`, `:397` | B | S8 | **done** |
| B03 | First-line ascent overshoot: vocalized Arabic rises **12.79pt** above the baseline versus **9.12pt** for the Latin the baseline was placed for, colliding with the paragraph above | `converter.py:392`; baseline is `LTChar.y0` via the `descent = 0` hack at `pdfinterp.py:99` | M | S8 | **done** |
| B04 | No margin/padding concept; the paragraph box is the raw ink extent of the original text | `converter.py:310-313` | M | S6 | open |
| B05 | First-line indent (`x - x0`) is LTR-only; RTL needs `x1 - (x - x0)` | `converter.py:292`, `:392` | M | S6 | **done** |
| B06 | Inter-word spaces render in **Times-Roman** (`tiro`), not the Arabic font, and split the run three ways per gap | `converter.py:427-429`. `tiro` space = 3.00pt at 12pt, GoNoto = 3.12pt | M | S3 | **done** |
| B07 | Bullets pinned to their original left-hand position (`cls = 0`), orphaned on the wrong side of RTL list text | `converter.py:239-240` | M | S7 | open |
| B08 | No `Tc` / `Tw`, no justification, no kashida — Arabic sets ragged | `converter.py:385-386` | N | S11 | open |
| B09 | Horizontal ink overhang — **verified non-issue** for this font: measured overhang is negative (ink sits inside the advance box) on all samples | `الجملة العربية` ink `[0.36, 62.84]` vs advance `63.62` | I | — | closed |

### 2.3 Overflow & fit

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| C01 | **Wrapping is gated on `brk`**, so headings, captions, table cells and labels never wrap; `x` grows past `x1`, past the page edge | `converter.py:454`; `brk` set only at `:287-289` | B | S5 | **done** |
| C02 | No font-size scaling; the only fit mechanism is the line-height squeeze at `:513-516` | `converter.py:397` (`size` read once, never modified) | B | S8 | **done** |
| C03 | Vertical spill silently overlaps following content; no detection, no warning, no clamp | `converter.py:520` | M | S8 | **done** |
| C04 | No post-layout verification that emitted ops stayed inside `[x0,x1] x [y0,y1]` | — | M | S8 | partial (overflow warned) |
| C05 | Correctly shaped Arabic is usually **narrower** than the source (0.52x–0.87x measured), but the current +40% width error masks this. Fixing shaping changes fit behaviour for every language pair | Appendix A.2 | I | S8 | open |

### 2.4 Mixed script & numbers

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| D01 | Two unrelated typefaces inside one sentence: ASCII / digits / space / punctuation go to Times-Roman, Arabic to Noto | `converter.py:427-432` | M | S3 | **done** |
| D02 | Bracket mirroring (UAX #9 rule L4) is never applied — and note `python-bidi.get_display()` does **not** apply it either | verified: `(تقريباً)` reorders to `)...(` unless L4 is applied explicitly | B | S2 | **done** |
| D03 | No directional isolation (LRI/PDI) for embedded LTR runs; adjacent neutrals leak across script boundaries | — | M | S2 | open |
| D04 | Western digits (`42`, bidi class **EN**) and Arabic-Indic digits (`٤٢`, class **AN**) render in different typefaces within one document | `converter.py:427-432` | M | S3 | **done** |
| D05 | No digit-form policy; Western vs Arabic-Indic vs Extended Arabic-Indic is left to whatever the LLM emits, varying sentence to sentence | — | N | S9 | **done** |
| D06 | **Design constraint:** naive string reversal corrupts every number. Real UAX #9 is mandatory | verified: `من 1990 إلى 2024` naive-reverses to `4202 ىلإ 0991 نم` | B | S2 | **done** |

### 2.5 Arabic as source language (`--lang-in ar`)

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| E01 | `vflag` classifies every Arabic diacritic and tatweel as a formula, so vocalized Arabic is shredded into `{vN}` boxes and never reaches the translator | `converter.py:215-224`. Categories: fatha/damma/kasra/shadda/sukun/tanween = `Mn`; tatweel U+0640 = `Lm`. Also hits Hebrew niqqud (`Mn`) | B | S12 | **done** |
| E02 | Space / line-break detection is direction-inverted: in RTL the pen moves leftward, so real word spaces hit the *line-break* branch and real line wraps hit the *inline-space* branch | `converter.py:285-289` | B | S12 | **done** |
| E03 | No visual→logical recovery and no presentation-form (`U+FE70`–`U+FEFF`) normalization for source PDFs that store Arabic pre-reversed | section A generally | M | S12 | open |

### 2.6 Infrastructure

| ID | Problem | Evidence | Sev | Stage | Status |
|----|---------|----------|-----|-------|--------|
| F01 | No shaping or bidi dependency declared | `pyproject.toml` | B | S0 | **done** |
| F02 | Arabic (and Hebrew / Persian / Urdu) absent from the GUI language list | `gui.py:102-113` | M | S9 | **done** |
| F03 | `subset_fonts(fallback=True)` may not discover glyphs that were never resolved through `has_glyph`, risking dropped glyphs | `high_level.py:246-248` | M (risk) | S10 | **done** |
| F04 | Bundled font capability — **verified sufficient**, no font work required | GSUB `arab`: `init/medi/fina/isol/rlig/liga/calt/ccmp`; GPOS `arab`: `mark/mkmk/curs`; 64,750 glyphs | I | — | closed |

---

## 3. Prior art

### 3.1 Nothing in this project family solves it

| Project | Status |
|---|---|
| `pdf2zh` v1 (this repo) | No RTL support. Grep for `bidi`/`rtl`/`arabic` finds exactly two lines: `"ar": 1.0` in the line-height map and `"ar"` in `noto_list` |
| [PDFMathTranslate issue #1091](https://github.com/PDFMathTranslate/PDFMathTranslate/issues/1091) | "Translart Arabic" — user requests correction of reversed Arabic output. Closed as `enhancement` with no fix, no workaround, no linked PR |
| [BabelDOC](https://github.com/funstory-ai/BabelDOC) (the v2 engine, and a declared dependency of this repo) | Its [typesetting docs](https://funstory-ai.github.io/BabelDOC/ImplementationDetails/Typesetting/Typesetting/) state explicitly: *"this is only implemented for paragraphs, only handles left-to-right writing."* |

So both the v1 and v2 engines lack RTL. This work is genuinely new for the project.

### 3.2 The reference implementation to borrow from: `fpdf2`

[`fpdf2`](https://github.com/py-pdf/fpdf2) (LGPL-3.0, compatible with this repo's AGPL-3.0) is the closest analogue: a Python library that writes PDF content streams directly and added full HarfBuzz + UAX #9 support. Its structure maps almost one-to-one onto what we need.

**`fpdf/fonts.py :: perform_harfbuzz_shaping`** — the shaping call:

```python
buf = hb.Buffer()
buf.cluster_level = 1
buf.add_str(text)
buf.guess_segment_properties()
buf.direction = params["fragment_direction"].value   # "ltr" / "rtl"
buf.script    = params["script"]
buf.language  = params["language"]
hb.shape(hbfont, buf, features)
```

Note `cluster_level = 1` (monotone characters) — required so cluster indices stay usable for mapping back to source text.

**`fpdf/fonts.py :: shape_text`** — cluster back-mapping, which is exactly the `ToUnicode` solution for A11:

```python
cluster_list = sorted(int(gi.cluster) for gi in glyph_infos)
cluster_mapping = {}
for i in range(len(text)):
    cl = get_cluster_from_text_index(cluster_list, i)
    cluster_mapping.setdefault(cl, []).append(i)

for gi in glyph_infos:
    unicode = [ord(text[i]) for i in cluster_mapping.pop(gi.cluster, [])]
    # -> one glyph may map to MULTIPLE source codepoints (ligatures)
```

It also computes a `force_positioning` flag — set when the shaped advance differs from the glyph's nominal `hmtx` width, or when either offset is non-zero. That flag is the trigger for emitting an explicit `Tm`.

**`fpdf/line_break.py :: render_with_text_shaping`** — the emission pattern we should copy: accumulate glyphs into one show-text operator and only break out to a fresh `Tm` when a glyph carries an offset or needs forced positioning:

```python
for ti in font.shape_text(...):
    if ti["x_offset"] != 0 or ti["y_offset"] != 0:
        flush current string           # emit accumulated glyphs
        emit  "1 0 0 1 <ox> <oy> Tm"   # reposition for the offset glyph
    text += char
    pos_x += adjust_pos(ti["x_advance"])
    if ti["force_positioning"]:
        flush; emit new Tm at pos_x
```

**`fpdf/line_break.py :: get_ordered_fragments`** — fragment-level bidi reordering:

```python
# group fragments into directional runs
# if paragraph is RTL: directional_runs = directional_runs[::-1]
# within each run: run[::-1] if run is RTL else run
```

**`fpdf/bidi.py`** — a complete self-contained UAX #9 implementation (`BidiParagraph`, `IsolatingRun`, `resolve_weak_types`, `resolve_neutral_types`, `resolve_implicit_levels`, `pair_brackets`, `get_bidi_fragments() -> ((text, direction), ...)`). Useful as a correctness reference even though we will depend on `python-bidi` instead of vendoring it.

### 3.3 Overflow handling: BabelDOC's adaptive typesetting

BabelDOC solves defects C01–C04 for LTR and its algorithm transfers directly. From its typesetting documentation:

- Goal: *"fit all components within the original paragraph bounding box. If impossible, try to expand the bounding box in writing direction."*
- Scaling factor starts at 1.0, line spacing at 1.5.
- First reduce line spacing in 0.1 steps down to a 1.4 floor.
- Then reduce element scale by 0.05 while above 0.6, by 0.1 below 0.6.
- Below scale 0.7 the minimum line spacing relaxes to 1.1.
- Fail if scale drops below 0.1.
- A separate pass attempts bounding-box expansion before aggressive scaling, accounting for page margins and overlapping content.

We adopt this ladder with Arabic-specific floors (Section 8).

### 3.4 What we are *not* using, and why

| Option | Verdict |
|---|---|
| PyMuPDF `TextWriter(right_to_left=True)` | Handles direction but **not shaping** — MuPDF's `TextWriter` has known Arabic disjoining bugs ([#1719](https://github.com/pymupdf/PyMuPDF/issues/1719), [#897](https://github.com/pymupdf/PyMuPDF/issues/897)). Only PyMuPDF's `Story` class shapes via HarfBuzz, and `Story` is an HTML-flow engine incompatible with our absolute-position op emission |
| `arabic_reshaper` (presentation forms `U+FE70`–`FEFF`) | The cheap route. Produces isolated/initial/medial/final forms via a lookup table but gives **no mark positioning (GPOS), no kerning, no `rlig` beyond lam-alef, and no cluster data for `ToUnicode`**. Rejected: it cannot fix A06 |
| Vendoring `fpdf2/bidi.py` | Viable (LGPL is AGPL-compatible) but adds ~800 lines to maintain. Use `python-bidi` instead; keep `fpdf2` as the correctness oracle in tests |
| `PyICU` (`icu.Bidi`) | Highest-fidelity UBA with native level runs and logical/visual maps, but a heavy binary dependency that is painful on Windows. Document as an **optional** backend, not the default |
| Naive string reversal | Corrupts every number (D06). Never |

### 3.5 Dependency decisions

| Package | Version | License | Role |
|---|---|---|---|
| `uharfbuzz` | `>=0.39` | Apache-2.0 | OpenType shaping (GSUB/GPOS) |
| `python-bidi` | `>=0.6,<0.7` | LGPL-3.0 | UAX #9 embedding levels + mirroring table |
| `fontTools` | already a dependency | MIT | glyph metrics, `hmtx`, cmap, `ToUnicode` generation |

All are AGPL-3.0 compatible. `arabic_reshaper` is **not** required.

---

## 4. Target architecture

### 4.1 Module layout

```
pdf2zh/
  bidi_shape.py     <- NEW. Pure text -> positioned glyphs. No PDF knowledge.
  converter.py      <- section C inner loop replaced; section A fixes
  high_level.py     <- RTL language table, ToUnicode post-pass
  gui.py            <- language list
  pdf2zh.py         <- CLI flags
```

`bidi_shape.py` must have **no imports from `pdf2zh`** and no PDF concepts. It takes text, a font, a size and a box; it returns positioned glyphs. That keeps it unit-testable without a PDF and keeps `converter.py` recognisable.

### 4.2 Pipeline stages

| Stage | Name | Module | Defects closed |
|---|---|---|---|
| S0 | Dependencies & font plumbing | `pyproject.toml`, `high_level.py` | F01 |
| S1 | Placeholder protection | `bidi_shape.py` | A08 |
| S2 | Bidi resolution (levels, L2 reorder, L4 mirroring) | `bidi_shape.py` | A02, D02, D03, D06 |
| S3 | Run segmentation (level INTERSECT font INTERSECT script) | `bidi_shape.py` | A10, B06, D01, D04 |
| S4 | HarfBuzz shaping | `bidi_shape.py` | A01, A04, A05, A06 |
| S5 | Line breaking (cluster-safe, word-aware, ungated) | `bidi_shape.py` | A09, C01 |
| S6 | Direction-aware placement | `bidi_shape.py` | A03, B04, B05 |
| S7 | Formula placement in RTL | `converter.py` | A07, B07 |
| S8 | Vertical metrics & fit ladder | `converter.py` | B01, B02, B03, C02, C03, C04 |
| S9 | Configuration surface | `pdf2zh.py`, `gui.py`, `config.py` | D05, F02 |
| S10 | ToUnicode & subsetting | `high_level.py` | A11, F03 |
| S11 | Justification (kashida) — *phase 4, optional* | `bidi_shape.py` | B08 |
| S12 | Source-side (`lang_in`) fixes | `converter.py` | E01, E02, E03 |

### 4.3 Phasing

- **Phase 1 (S0–S6):** Arabic renders correctly. This is the minimum shippable unit.
- **Phase 2 (S7–S9):** formulas, fit, config. Required for real documents.
- **Phase 3 (S10, S12):** accessibility and Arabic-as-source.
- **Phase 4 (S11):** justification polish.

---

## 5. Module specification: `pdf2zh/bidi_shape.py`

### 5.1 Data model

```python
from dataclasses import dataclass, field


@dataclass
class ShapedGlyph:
    """One positioned glyph. Advances/offsets are in POINTS at the shaped size."""
    gid: int                      # glyph id == CID under Identity-H
    x_advance: float
    x_offset: float
    y_offset: float
    chars: tuple[int, ...]        # source codepoints for this cluster (ToUnicode)
    nominal_width: float          # hmtx advance, to detect forced positioning

    @property
    def force_positioning(self) -> bool:
        return (
            self.x_offset != 0
            or self.y_offset != 0
            or abs(self.x_advance - self.nominal_width) > 1e-6
        )


@dataclass
class TextRun:
    """A maximal span with one direction, one font and one script."""
    text: str
    level: int                    # bidi embedding level
    font_key: str                 # "noto" | "tiro"
    script: str                   # OpenType script tag, e.g. "arab", "latn"
    glyphs: list[ShapedGlyph] = field(default_factory=list)

    @property
    def rtl(self) -> bool:
        return bool(self.level % 2)

    @property
    def width(self) -> float:
        return sum(g.x_advance for g in self.glyphs)


@dataclass
class PlaceholderRun:
    """A {vN} formula box. Occupies width but contains no glyphs."""
    vid: int
    width: float
    level: int

    @property
    def rtl(self) -> bool:
        return bool(self.level % 2)


@dataclass
class LayoutLine:
    items: list[TextRun | PlaceholderRun]   # already in VISUAL order, left to right
    width: float


@dataclass
class LayoutResult:
    lines: list[LayoutLine]
    base_rtl: bool
    overflow_x: float             # how far the widest line exceeds the box (0 if none)
```

### 5.2 Language tables

```python
# Scripts written right-to-left. Keys are lowercase, prefix-matched on "-".
RTL_LANGS = {
    "ar",   # Arabic
    "arc",  # Aramaic
    "az-arab",
    "ckb",  # Sorani Kurdish
    "dv",   # Divehi
    "fa",   # Persian
    "he", "iw",   # Hebrew (iw is the legacy code used by Google)
    "ku-arab",
    "ps",   # Pashto
    "sd",   # Sindhi
    "syr",  # Syriac
    "ug",   # Uyghur
    "ur",   # Urdu
    "yi",   # Yiddish
}


def is_rtl_lang(lang: str) -> bool:
    lang = (lang or "").lower().replace("_", "-")
    if lang in RTL_LANGS:
        return True
    return lang.split("-")[0] in RTL_LANGS
```

Note `iw` and `ur` are already in `high_level.noto_list`, so the font selection path needs no change for them.

### 5.3 S1 — Placeholder protection

**Problem (A08):** `{v0}` is reordered to `}v0{` by bidi, after which the extraction regex never matches.

**Solution:** replace each placeholder with `U+FFFC OBJECT REPLACEMENT CHARACTER` *before* bidi. Verified properties: bidi class `ON` (behaves as a neutral object, which is semantically correct for an inline formula box), `Bidi_Mirrored = No`, and it survives reordering intact.

Because we track original character indices through the whole bidi stage (Section 5.4), each `U+FFFC` is matched back to its `vid` by index — no ambiguity even with several placeholders in one paragraph.

```python
import re

OBJ = "￼"
_VN_RE = re.compile(r"\{\s*v([\d\s]+)\}", re.IGNORECASE)


def protect_placeholders(text: str) -> tuple[str, dict[int, int]]:
    """Replace each {vN} with U+FFFC.

    Returns (protected_text, {index_in_protected_text: vid}).
    """
    out: list[str] = []
    marks: dict[int, int] = {}
    pos = 0
    for m in _VN_RE.finditer(text):
        out.append(text[pos:m.start()])
        try:
            vid = int(m.group(1).replace(" ", ""))
        except ValueError:
            out.append(m.group(0))     # malformed, keep literally
            pos = m.end()
            continue
        marks[sum(len(s) for s in out)] = vid
        out.append(OBJ)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), marks
```

> Keep the existing `re.IGNORECASE` and the `[\d\s]+` tolerance — translators emit `{ v 0 }` and `{V0}`. Python's `\d` is Unicode-aware and `int()` accepts Arabic-Indic digits, so `{v٥}` still resolves to 5.

### 5.4 S2 — Bidi resolution

**Verified approach.** `python-bidi`'s public `get_display()` returns only a string — no index map and, importantly, **it does not apply UAX #9 rule L4 (mirroring)**. We therefore drive the pure-Python primitives in `bidi.algorithm` directly, which gives us per-character embedding levels *and* lets us carry our own metadata (original index) through the resolution.

This was validated: index-tracked levels + our own L2 reorder reproduce `get_display()` **exactly** on every mixed Arabic/Latin/digit/placeholder test string.

```python
import unicodedata

from bidi.algorithm import (
    MIRRORED,
    explicit_embed_and_overrides,
    get_empty_storage,
    resolve_implicit_levels,
    resolve_neutral_types,
    resolve_weak_types,
)


@dataclass
class BidiChar:
    ch: str
    level: int
    index: int          # index into the protected text


def resolve_levels(text: str, base_rtl: bool) -> list[BidiChar]:
    """UAX #9 stages X1-X10, W1-W7, N0-N2, I1-I2. Returns chars in LOGICAL order."""
    base = 1 if base_rtl else 0
    st = get_empty_storage()
    st["base_level"] = base
    st["base_dir"] = "R" if base else "L"
    for i, ch in enumerate(text):
        cls = unicodedata.bidirectional(ch)
        st["chars"].append(
            {"ch": ch, "level": base, "type": cls, "orig": cls, "index": i}
        )
    explicit_embed_and_overrides(st)
    resolve_weak_types(st)
    resolve_neutral_types(st, False)      # NOTE: positional `debug` arg is required
    resolve_implicit_levels(st, False)    # NOTE: positional `debug` arg is required
    return [BidiChar(c["ch"], c["level"], c["index"]) for c in st["chars"]]


def reorder_l2(chars: list[BidiChar]) -> list[BidiChar]:
    """UAX #9 rule L2 -- returns VISUAL order (left to right)."""
    items = list(chars)
    if not items:
        return items
    levels = [c.level for c in items]
    hi = max(levels)
    lo_odd = min((lv for lv in levels if lv % 2), default=hi + 1)
    for lvl in range(hi, lo_odd - 1, -1):
        i = 0
        while i < len(items):
            if items[i].level >= lvl:
                j = i
                while j < len(items) and items[j].level >= lvl:
                    j += 1
                items[i:j] = items[i:j][::-1]
                i = j
            else:
                i += 1
    return items


def apply_mirroring(chars: list[BidiChar]) -> None:
    """UAX #9 rule L4. python-bidi's get_display() does NOT do this."""
    for c in chars:
        if c.level % 2:
            c.ch = MIRRORED.get(c.ch, c.ch)
```

**Base direction per paragraph.** Do not force RTL on every paragraph just because `lang_out` is RTL — a pure-Latin citation or a code block inside an Arabic document must stay LTR. Use UAX #9 rule P2/P3 (first strong character) with the target language as the tie-breaker for direction-neutral paragraphs:

```python
def detect_base_rtl(text: str, lang_out: str) -> bool:
    for ch in text:
        cls = unicodedata.bidirectional(ch)
        if cls in ("L",):
            return False
        if cls in ("R", "AL"):
            return True
    return is_rtl_lang(lang_out)     # no strong character: fall back to target lang
```

**Isolation (D03).** Wrap runs whose direction differs from the paragraph base in `U+2066 LRI` / `U+2067 RLI` … `U+2069 PDI` before resolution, so that neutrals adjacent to an embedded run cannot leak across the boundary. `explicit_embed_and_overrides` handles isolate initiators and strips them from the char list, so they never reach the shaper.

> **Version guard.** `bidi.algorithm` is `python-bidi`'s legacy pure-Python path and is not part of its documented API. Pin `python-bidi>=0.6,<0.7` and add the import-time self-test in Section 12.1 so a breaking upgrade fails loudly at start-up rather than silently producing reversed Arabic.

### 5.5 S3 — Run segmentation

Break the visual-order character sequence wherever **any** of level, font or script changes. This is what closes A10, B06, D01 and D04 at once.

```python
# Scripts we can shape. Everything else falls through to "latn"/dflt.
def script_of(ch: str) -> str:
    cp = ord(ch)
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F \
       or 0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF \
       or 0xFE70 <= cp <= 0xFEFF:
        return "arab"
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return "hebr"
    return "latn"
```

**Font selection must change for RTL paragraphs (D01, D04, B06).** The current `tiro` fast-path at `converter.py:427-432` is what splits an Arabic sentence across two typefaces. Rule:

> If the paragraph's base direction is RTL, **never** use `tiro`. Route every character — Arabic letters, embedded Latin, digits, spaces and punctuation alike — through the Noto font, which covers Latin perfectly well. This gives one harmonised family per paragraph.

For LTR paragraphs keep the existing `tiro`-first behaviour byte-for-byte.

```python
def font_for(ch: str, base_rtl: bool, tiro_probe) -> str:
    if base_rtl:
        return "noto"                    # one family for the whole RTL paragraph
    try:
        if tiro_probe(ord(ch)) == ch:    # existing fontmap["tiro"].to_unichr probe
            return "tiro"
    except Exception:
        pass
    return "noto"


def segment_runs(chars, base_rtl, tiro_probe, marks) -> list[TextRun | PlaceholderRun]:
    items: list[TextRun | PlaceholderRun] = []
    for c in chars:
        if c.ch == OBJ:                          # formula placeholder
            items.append(PlaceholderRun(vid=marks[c.index], width=0.0, level=c.level))
            continue
        fk = font_for(c.ch, base_rtl, tiro_probe)
        sc = script_of(c.ch)
        last = items[-1] if items else None
        if (
            isinstance(last, TextRun)
            and last.level == c.level
            and last.font_key == fk
            and last.script == sc
        ):
            last.text += c.ch
        else:
            items.append(TextRun(text=c.ch, level=c.level, font_key=fk, script=sc))
    return items
```

> **Critical ordering detail.** Runs are produced in *visual* order, but HarfBuzz must be fed *logical* order. Since L2 reversal is applied to the whole sequence, the characters inside an RTL run are in reverse logical order at this point. Before shaping, re-reverse each RTL run's text so the shaper sees logical order, then shape with `direction="rtl"` — HarfBuzz returns glyphs already in visual (left-to-right) order, which is what we emit. This was verified: `shape(logical, "rtl")` and the visual sequence agree.

### 5.6 S4 — Shaping

```python
import functools

import uharfbuzz as hb
from fontTools.ttLib import TTFont


class Shaper:
    """Wraps one font file. Cache one instance per font path per process."""

    def __init__(self, font_path: str):
        blob = hb.Blob.from_file_path(font_path)
        self.face = hb.Face(blob)
        self.font = hb.Font(self.face)
        self.upem = self.face.upem
        tt = TTFont(font_path, lazy=True)
        self._hmtx = tt["hmtx"].metrics
        self._order = tt.getGlyphOrder()

    def _nominal(self, gid: int) -> int:
        return self._hmtx[self._order[gid]][0]

    @functools.lru_cache(maxsize=8192)
    def _shape_cached(self, text: str, direction: str, script: str, language: str):
        buf = hb.Buffer()
        buf.cluster_level = 1              # monotone chars: keeps clusters mappable
        buf.add_str(text)
        buf.direction = direction
        buf.script = script
        buf.language = language
        hb.shape(self.font, buf, {"kern": True, "liga": True})
        return (
            [(gi.codepoint, gi.cluster) for gi in buf.glyph_infos],
            [(p.x_advance, p.x_offset, p.y_offset) for p in buf.glyph_positions],
        )

    def shape(self, text, direction, script, language, size) -> list[ShapedGlyph]:
        infos, positions = self._shape_cached(text, direction, script, language)
        k = size / self.upem

        # cluster -> source character indices (fpdf2's approach, for ToUnicode)
        clusters = sorted({c for _, c in infos})
        mapping: dict[int, list[int]] = {}
        for i in range(len(text)):
            pos = bisect.bisect_right(clusters, i) - 1
            mapping.setdefault(clusters[max(pos, 0)], []).append(i)

        out = []
        for (gid, cluster), (xa, xo, yo) in zip(infos, positions):
            src = tuple(ord(text[i]) for i in mapping.get(cluster, ()))
            out.append(
                ShapedGlyph(
                    gid=gid,
                    x_advance=xa * k,
                    x_offset=xo * k,
                    y_offset=yo * k,
                    chars=src,
                    nominal_width=self._nominal(gid) * k,
                )
            )
        return out
```

**Language tags.** Pass the BCP-47 tag from `lang_out` (`"ar"`, `"fa"`, `"ur"`, `"he"`). This matters: Persian and Urdu select different `locl` forms of the same codepoints.

**Missing glyphs.** `gid == 0` is `.notdef` and renders as a tofu box. Log once per (font, codepoint) at DEBUG and drop the glyph rather than emitting a visible box.

### 5.7 S5 — Line breaking

Two defects here: A09 (per-character breaking severs joining runs) and **C01 (`brk` gating)**.

> **`brk` must stop gating wrapping.** Today `converter.py:454` only wraps when the source paragraph had a detected line break, so single-line sources (headings, captions, table cells, axis labels) run off the page. Wrapping becomes unconditional; `brk` is demoted to a *hint about whether reflow is desirable* and is used only to decide whether to attempt bbox expansion in S8.

Break-opportunity policy — no new dependency needed:

| Script class | Policy |
|---|---|
| CJK (Han, Hiragana, Katakana, Hangul) | break between any two clusters (preserves current behaviour) |
| Arabic, Hebrew, Latin, Cyrillic, ... | break at spaces, `U+200B` ZWSP, and after `-` / `U+00AD` |
| Overlong single word (exceeds the box on its own) | fall back to **cluster-boundary** breaking — never mid-cluster, so a ligature or a mark+base pair is never split |

```python
def break_lines(items, box_width, first_line_width) -> list[LayoutLine]:
    """Greedy wrap over runs. Never splits a ShapedGlyph cluster.

    `first_line_width` differs from `box_width` when the paragraph is indented.
    """
    # 1. flatten items into (item, glyph_index) break candidates
    # 2. accumulate widths; at each candidate record a break opportunity
    # 3. when width would exceed the current line's limit, cut at the last
    #    recorded opportunity; if none exists, cut at the last cluster boundary
    # 4. drop trailing spaces from the finished line (they must not count
    #    toward line width, unlike the current code)
    ...
```

Note step 4: the current implementation lets a trailing space consume width and influence the break decision. Fix while here.

### 5.8 S6 — Direction-aware placement

This replaces `converter.py:391-500`. The pen rules:

| | LTR | RTL |
|---|---|---|
| Line start x | `x0` (first line: `x`) | `x1` (first line: `x1 - (x - x0)`) |
| Advance | `pen += adv` | `pen -= adv` |
| Boundary test | `pen + adv > x1 + tol` | `pen - adv < x0 - tol` |
| Leading-space drop | `pen == x0` | `pen == x1` |

```python
def place_line(line: LayoutLine, x0: float, x1: float, base_rtl: bool,
               indent: float = 0.0) -> list[tuple]:
    """Assign absolute x to every glyph. Returns emission tuples.

    `line.items` is already in VISUAL order, so placement is a simple
    left-to-right sweep in BOTH directions -- the difference is only the
    starting edge and how leftover space is distributed.
    """
    if base_rtl:
        start = x1 - indent - line.width      # right-align the visual line
    else:
        start = x0 + indent
    pen = start
    out = []
    for item in line.items:
        if isinstance(item, PlaceholderRun):
            out.append(("formula", item.vid, pen))
            pen += item.width
        else:
            out.append(("text", item, pen))
            pen += item.width
    return out
```

> **Why this is simpler than it looks.** Once bidi has produced visual order, both directions are a left-to-right sweep. RTL only changes *where the line starts* (right-aligned instead of left-aligned) and the wrap decision. That keeps the emission code close to the original shape.

**B05 — indent mirroring.** `indent = pstk.x - pstk.x0` from the source. Apply it to the first line on the starting edge: `x0 + indent` for LTR, `x1 - indent` for RTL.

**B04 — edge tolerance.** Keep the existing `0.1 * size` tolerance so LTR output is unchanged. B09 confirmed no ink-overhang padding is needed for this font.

---

## 6. Converter integration (S7)

### 6.1 Emission — replacing `gen_op_txt`

The op format is unchanged. What changes is that a run may need several `Tm`-positioned pieces when glyphs carry offsets (harakat). Follow `fpdf2`'s pattern:

```python
def gen_op_glyphs(font, size, x, y, glyphs, raw):
    """Emit one shaped run, breaking out only for offset/forced glyphs."""
    ops = []
    buf = []
    pen = x
    run_x = x
    for g in glyphs:
        if g.force_positioning:
            if buf:
                ops.append(f"/{font} {size:f} Tf 1 0 0 1 {run_x:f} {y:f} Tm "
                           f"[<{raw(buf)}>] TJ ")
                buf = []
            ops.append(
                f"/{font} {size:f} Tf 1 0 0 1 "
                f"{pen + g.x_offset:f} {y + g.y_offset:f} Tm "
                f"[<{raw([g])}>] TJ "
            )
            pen += g.x_advance
            run_x = pen
            continue
        if not buf:
            run_x = pen
        buf.append(g)
        pen += g.x_advance
    if buf:
        ops.append(f"/{font} {size:f} Tf 1 0 0 1 {run_x:f} {y:f} Tm "
                   f"[<{raw(buf)}>] TJ ")
    return "".join(ops)
```

`raw_string` gains a glyph-list form; the existing character form stays for the LTR/`tiro` path:

```python
def raw_string_glyphs(glyphs) -> str:
    return "".join("%04x" % g.gid for g in glyphs)
```

> Because a mark's `x_advance` is 0, `pen` does not move and the following base glyph lands correctly. This is what fixes A06.

### 6.2 A07 — formula placement in RTL

The visual-order sweep in `place_line` already puts the placeholder at the correct visual x. The formula's own glyphs are always drawn **left-to-right internally** (a formula is a preserved image of the source, not text to reorder), starting at the placeholder's left edge:

```python
# `pen` is the placeholder's LEFT edge in both directions
for vch in var[vid]:
    emit(x=pen + vch.x0 - var[vid][0].x0,
         dy=fix + vch.y0 - var[vid][0].y0)
```

This is the existing code at `converter.py:461-471` with `x` replaced by the placeholder's left edge. No other change. `PlaceholderRun.width` is `vlen[vid]`, computed exactly as today at `converter.py:339-342`.

### 6.3 B07 — bullets

`converter.py:239-240` forces `cls = 0` for `•`, pinning the bullet at its source x. For RTL paragraphs, mirror the bullet's anchor across the paragraph box:

```python
# when the following text paragraph is RTL
bullet_x_mirrored = x0 + x1 - bullet_x1
```

Apply this only when `base_rtl` is true for the adjacent text paragraph; otherwise leave the existing behaviour untouched.

---

## 7. Vertical metrics & the fit ladder (S8)

### 7.1 Line-height floors (B01)

Measured ink heights at size 12 in `GoNotoKurrent-Regular.ttf`:

| content | above baseline | below baseline | total | implied min line height |
|---|---|---|---|---|
| Latin `Highlight jpqy` | 9.12pt | -2.88pt | 12.00pt | 1.00 |
| Arabic, unvocalized | 8.76pt | -2.53pt | 11.29pt | 0.94 |
| Arabic + tashkeel | 12.79pt | -2.53pt | 15.32pt | **1.28** |
| Arabic, tall + deep (`لآ جِّ`) | 11.21pt | -4.73pt | 15.94pt | **1.33** |

```python
LANG_LINEHEIGHT_MAP = {
    "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
    "ja": 1.1, "ko": 1.2, "en": 1.2,
    "ar": 1.45, "fa": 1.45, "ur": 1.55, "he": 1.35, "iw": 1.35,   # was ar: 1.0
    "ru": 0.8, "uk": 0.8, "ta": 0.8,
}

# Per-language floor used by the fit ladder. Detect tashkeel to pick the tight
# floor only when it is actually safe.
TASHKEEL = range(0x064B, 0x0653)


def line_height_floor(lang: str, text: str) -> float:
    if is_rtl_lang(lang):
        if any(ord(c) in TASHKEEL or ord(c) == 0x0670 for c in text):
            return 1.30
        return 1.15
    return 1.0
```

Urdu gets 1.55 because Nastaliq-style forms have a much steeper vertical cascade.

### 7.2 First-line ascent overshoot (B03)

The paragraph baseline `y` was placed for Latin ascent. Vocalized Arabic exceeds it by up to `(12.79 - 9.12) / 12 = 0.31 * size`.

```python
ascent = max((g.y_offset + glyph_ymax(g)) for g in first_line_glyphs)
overshoot = (y + ascent) - pstk[id].y1
if overshoot > 0:
    shift = min(overshoot, available_slack_below)   # never push past y0
    y -= shift
```

`available_slack_below` is `height - (lidx + 1) * size * line_height`. If there is no slack, log at DEBUG and accept the overshoot rather than clipping — a small collision reads better than a truncated line.

### 7.3 The fit ladder (B02, C02, C03, C04)

Adapted from BabelDOC's algorithm with Arabic floors. This replaces `converter.py:513-516`.

```python
def fit_paragraph(text, box, size, lang, *, brk):
    """Return (scale, line_height, lines) that fits, or the best effort."""
    scale = 1.0
    lh = LANG_LINEHEIGHT_MAP.get(lang, 1.1)
    lh_floor = line_height_floor(lang, text)
    while True:
        lines = layout(text, box, size * scale, lh)
        needed = len(lines) * size * scale * lh
        if needed <= box.height and not lines_overflow_x(lines, box):
            return scale, lh, lines
        if lh - 0.05 >= lh_floor:
            lh -= 0.05
            continue
        if brk and can_expand(box):        # only reflowable paragraphs may grow
            box = expand(box)              # respects page margins + neighbours
            continue
        if scale > 0.6:
            scale -= 0.05
        elif scale > 0.1:
            scale -= 0.10
        else:
            log.warning(
                "paragraph does not fit after scaling to %.2f; emitting with overflow",
                scale,
            )
            return scale, lh, lines
```

Ordering rationale, which differs slightly from BabelDOC: we exhaust **line height first**, then try **expansion**, then **scale**. Shrinking the font is the most visible degradation and Arabic loses legibility faster than Latin at small sizes (the marks collide before the letterforms do), so it goes last. Recommend a configurable floor `--min-font-scale` defaulting to `0.60`.

**C04 — verification pass.** After emission, assert every op's x lies within `[x0 - tol, x1 + tol]` and the last baseline within `[y0 - size, y1]`. Log a WARNING naming the page and the source text on violation. This is cheap and turns silent corruption into a diagnosable event.

**C05 — note for reviewers.** Fixing shaping makes Arabic *narrower* (0.52x–0.87x of the source), so most paragraphs will now fit where the broken width estimate previously forced wrapping. Expect LTR-target output to be unaffected (guarded by the golden test) but Arabic line counts to drop sharply versus any pre-fix snapshot.

---

## 8. Source-side fixes (S12)

### 8.1 E01 — `vflag` must not treat Arabic marks as formulas

`converter.py:215-224` currently returns `True` for any character in categories `Lm, Mn, Sk, Sm, Zl, Zp, Zs`. Verified misclassifications:

```
FATHA  U+064E  Mn -> formula      TATWEEL U+0640  Lm -> formula
DAMMA  U+064F  Mn -> formula      SHADDA  U+0651  Mn -> formula
KASRA  U+0650  Mn -> formula      SUKUN   U+0652  Mn -> formula
TANWEEN FATH U+064B Mn -> formula SUPERSCRIPT ALEF U+0670 Mn -> formula
HEBREW POINT QAMATS U+05B8 Mn -> formula
```

Add an early exemption for marks that belong to a text script:

```python
# Combining marks that are part of ordinary text, not mathematics.
TEXT_MARK_RANGES = (
    (0x0300, 0x036F),   # combining diacritics (Latin/Greek/Cyrillic)
    (0x064B, 0x065F),   # Arabic harakat
    (0x0670, 0x0670),   # superscript alef
    (0x06D6, 0x06ED),   # Quranic annotation marks
    (0x0591, 0x05C7),   # Hebrew niqqud + cantillation
    (0x0E31, 0x0E3A), (0x0E47, 0x0E4E),   # Thai
)


def _is_text_mark(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in TEXT_MARK_RANGES)
```

and in `vflag`, before the category test:

```python
if char and (_is_text_mark(ord(char[0])) or ord(char[0]) == 0x0640):
    return False        # tatweel and text marks are never formulas
```

Guard the change behind the existing `self.vchar` override so users who set `--vchar` keep full control.

### 8.2 E02 — direction-aware space and line-break detection

`converter.py:285-289` assumes a rightward-moving pen:

```python
if child.x0 > xt.x1 + 1:      # inline space
    sstk[-1] += " "
elif child.x1 < xt.x0:        # line break
    sstk[-1] += " "
    pstk[-1].brk = True
```

For an RTL source run the pen moves leftward, so the two branches swap meaning. Track the dominant direction of the current paragraph from the characters seen so far and select the comparison accordingly:

```python
para_rtl = _para_is_rtl(sstk[-1])          # first-strong over the accumulated text
if para_rtl:
    gap_inline = xt.x0 - child.x1          # next glyph sits to the LEFT
    wrapped    = child.x0 > xt.x1 + 1      # jumped back to the right margin
else:
    gap_inline = child.x0 - xt.x1
    wrapped    = child.x1 < xt.x0
if wrapped:
    sstk[-1] += " "
    pstk[-1].brk = True
elif gap_inline > 1:
    sstk[-1] += " "
```

### 8.3 E03 — normalizing source Arabic

Two independent problems in source PDFs:

1. **Presentation forms.** Producers may encode Arabic as `U+FB50`–`FDFF` / `U+FE70`–`FEFF`. Normalize to base letters before translation with `unicodedata.normalize("NFKC", text)`, which decomposes presentation forms back to their base codepoints. Note NFKC also splits the lam-alef ligature `U+FEFB` back into `U+0644 U+0627`, which is what we want.
2. **Visual ordering.** Many producers emit glyphs pre-reversed. Detect per paragraph: if the extracted text is RTL-script but the x-coordinates increase monotonically with content-stream order across a whole line, the run is stored visually and must be reversed to logical order before translation.

Both are heuristics. Gate them behind `--source-rtl {auto,visual,logical}` (default `auto`) so a user can force the correct interpretation when detection fails.

---

## 9. Configuration surface (S9)

### 9.1 CLI (`pdf2zh/pdf2zh.py`)

```
--rtl {auto,on,off}            default: auto   (derive from --lang-out)
--source-rtl {auto,visual,logical}  default: auto
--digit-form {auto,western,arabic}  default: auto (pass translator output through)
--min-font-scale FLOAT         default: 0.60
```

### 9.2 GUI (`pdf2zh/gui.py:102-113`)

```python
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
    "Arabic": "ar",        # NEW
    "Hebrew": "he",        # NEW
    "Persian": "fa",       # NEW
    "Urdu": "ur",          # NEW
}
```

### 9.3 D05 — digit form

```python
WESTERN = "0123456789"
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"


def normalize_digits(text: str, form: str) -> str:
    if form == "western":
        return text.translate(str.maketrans(ARABIC_INDIC, WESTERN))
    if form == "arabic":
        return text.translate(str.maketrans(WESTERN, ARABIC_INDIC))
    return text
```

Default `auto` = pass through. Recommend `western` for scientific documents, which is the prevailing convention in Arabic academic publishing and avoids mixing forms across sentences.

---

## 10. ToUnicode, subsetting & accessibility (S10)

### 10.1 A11 — `ToUnicode` for shaped glyphs

Each `ShapedGlyph` already carries `chars: tuple[int, ...]` from the cluster mapping. Accumulate a document-wide `dict[int, tuple[int, ...]]` (GID → source codepoints) during rendering, then inject a `/ToUnicode` CMap into the font dictionary with `pikepdf` (already a dependency) after `translate_stream` writes the document.

The CMap must use `bfrange`/`bfchar` entries whose destination is a UTF-16BE string — multi-codepoint destinations are legal and are exactly how a lam-alef ligature glyph maps back to two characters:

```
<1234> <0644 0627>      % one GID -> two source codepoints
```

Without this, copy/paste and screen readers get nothing usable from the Arabic output.

### 10.2 F03 — subsetting risk

`high_level.py:246-248` calls `doc_zh.subset_fonts(fallback=True)`. Today every emitted GID came from `noto.has_glyph()`, so PyMuPDF's own bookkeeping knows about it. HarfBuzz GIDs bypass that path.

Mitigation, in order:

1. Add an integration test that renders Arabic **with** subsetting and asserts the shaped GIDs still resolve (compare rendered pixmaps before and after `subset_fonts`).
2. If subsetting drops glyphs, force `skip_subset_fonts=True` whenever the target language is RTL — the flag already exists end-to-end (`pdf2zh.py:190`, `high_level.py:185`, `gui.py:642`).
3. Longer term, build the subset ourselves from the collected GID set with `fontTools.subset`, which also lets us emit the `ToUnicode` CMap in the same pass.

---

## 11. File-by-file change list

| File | Change | Stage |
|---|---|---|
| `pyproject.toml` | add `uharfbuzz>=0.39`, `python-bidi>=0.6,<0.7` | S0 |
| `pdf2zh/bidi_shape.py` | **new module** — Sections 5.1–5.8 | S1–S6 |
| `pdf2zh/converter.py:191-225` | `vflag` text-mark exemption | S12 |
| `pdf2zh/converter.py:239-240` | bullet mirroring for RTL | S7 |
| `pdf2zh/converter.py:285-289` | direction-aware space / break detection | S12 |
| `pdf2zh/converter.py:368-374` | `raw_string_glyphs` alongside `raw_string` | S4 |
| `pdf2zh/converter.py:377-380` | line-height map + floors | S8 |
| `pdf2zh/converter.py:385-386` | `gen_op_glyphs` alongside `gen_op_txt` | S7 |
| `pdf2zh/converter.py:391-529` | inner loop replaced by `bidi_shape` calls + fit ladder | S5–S8 |
| `pdf2zh/high_level.py:38-58` | RTL languages already covered by `noto_list`; add `fa` | S0 |
| `pdf2zh/high_level.py:246-248` | subsetting guard for RTL | S10 |
| `pdf2zh/gui.py:102-113` | add Arabic / Hebrew / Persian / Urdu | S9 |
| `pdf2zh/pdf2zh.py:64-80` | new CLI flags | S9 |
| `test/test_bidi_shape.py` | **new** — Section 12 | all |

### 11.1 The invariant that protects existing users

> For any `lang_out` where `is_rtl_lang()` is false, the emitted content stream must be **byte-identical** to the pre-change output.

Achieve this by keeping the existing character loop as the LTR path initially and routing only RTL paragraphs through `bidi_shape`. Once the golden test is green, LTR can be migrated onto the shaped path as a separate, individually revertable commit — that migration will change bytes (shaping adds kerning and ligatures) and needs its own review.

---

## 12. Test plan

### 12.1 Import-time self-test (guards the `python-bidi` internal API)

```python
def _self_test() -> None:
    """Fail loudly if python-bidi's internals changed under us."""
    from bidi import get_display
    for probe in ("المعادلة {v0} 42 NumPy",):
        chars = resolve_levels(probe, base_rtl=True)
        mine = "".join(c.ch for c in reorder_l2(chars))
        if mine != get_display(probe, base_dir="R"):
            raise RuntimeError(
                "python-bidi internal API mismatch; pin python-bidi>=0.6,<0.7"
            )
```

### 12.2 Unit tests

| Test | Asserts |
|---|---|
| `test_reorder_matches_reference` | index-tracked L2 reorder == `get_display()` across a fixture corpus of mixed Arabic/Latin/digit strings |
| `test_mirroring_applied` | `(` at odd level becomes `)`; `get_display` alone does not do this |
| `test_placeholder_survives_bidi` | `{v0}`, `{v1}`, `{v12}` round-trip through protect → bidi → restore with correct vids and correct visual order |
| `test_shaping_golden` | GID sequences for `الكتاب`, `لا إله`, `كِتَابٌ` match recorded goldens; `لا إله` yields **5** glyphs from 6 codepoints |
| `test_mark_zero_advance` | every glyph whose source codepoint is in `TASHKEEL` has `x_advance == 0` and a non-zero offset |
| `test_width_accuracy` | shaped width of the fixture strings is within 0.5% of the reference table in Appendix A.1 |
| `test_numbers_preserved` | `من 1990 إلى 2024` renders `1990` and `2024` in left-to-right glyph order |
| `test_vflag_arabic_marks` | harakat and tatweel return `False` from `vflag` |
| `test_rtl_source_spacing` | synthetic RTL `LTChar` sequence produces one space per word gap and `brk` only at real wraps |
| `test_ltr_output_unchanged` | **golden**: `test/file/*.pdf` translated to `zh` produces a byte-identical content stream |

### 12.3 Integration tests

- Render `test/file/translate.cli.plain.text.pdf` to `ar` with a stub translator returning fixed Arabic; assert the content stream's `Tm` x-values **decrease** across a line and that every glyph lies within `[x0, x1]`.
- Same document with `--skip-subset-fonts` off and on; compare rendered pixmaps to catch F03.
- A fixture with a formula placeholder mid-sentence; assert the formula's glyphs are emitted left-to-right at the placeholder's visual position.
- A single-line heading whose Arabic translation is deliberately long; assert it wraps (C01) rather than running off the page.

### 12.4 Visual regression

Render page pixmaps and diff against approved snapshots. Arabic defects are overwhelmingly visual — disjoined letters and misplaced harakat pass every text-level assertion. This is the only test that catches A01/A06 convincingly.

---

## 13. Risks, non-goals & open decisions

### 13.1 Risks

| Risk | Mitigation |
|---|---|
| `python-bidi`'s `bidi.algorithm` is an undocumented internal path | Version pin + import-time self-test (12.1). Fall back to vendoring `fpdf2/bidi.py` (LGPL, compatible) if it breaks |
| `subset_fonts` drops HarfBuzz-sourced GIDs | F03 mitigation ladder in Section 10.2 |
| `uharfbuzz` wheel availability for Python 3.11/3.12 on all target platforms | Verify in CI before merging S0; it ships manylinux/macos/win wheels today |
| Shaping cost on large documents | `lru_cache` on `(text, direction, script, language)`; shaping is per-run, not per-page, and runs repeat heavily in real documents |
| Fixing width estimation changes LTR line counts | Section 11.1 invariant keeps LTR on the old path until explicitly migrated |

### 13.2 Non-goals

- **Column-order flipping.** For a two-column paper translated to Arabic, native typesetting reads the right column first. The project's core promise is layout preservation, so columns stay where they are. Revisit only on explicit request.
- **Vertical scripts** (Mongolian, CJK vertical).
- **Kashida justification** — deferred to phase 4 (S11). Arabic sets ragged, exactly as the current LTR output does.
- **Re-translating source Arabic PDFs stored as scanned images** — out of scope, unchanged.

### 13.3 Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Column order | preserve | layout-preservation is the product promise |
| Overflow ladder order | line height → bbox expansion → font scale | font scaling is the most visible degradation, and Arabic marks collide before letterforms do |
| Min font scale | 0.60 default, configurable | below this Arabic diacritics stop being resolvable in print |
| Digit form | `auto` (pass through), `western` recommended | matches Arabic academic convention |
| Font for RTL paragraphs | Noto for everything, no `tiro` | one harmonised family; fixes D01/D04/B06 together |
| Bidi library | `python-bidi` primitives, not vendored | smallest maintenance surface; validated against reference |

### 13.4 Open questions for the maintainer

1. Should RTL support also land in the v2 (BabelDOC) kernel, or is v1 the only target? BabelDOC is LTR-only upstream, so v2 would need the same work donated there.
2. Is a new runtime dependency acceptable, or should `uharfbuzz`/`python-bidi` be an extra (`pip install pdf2zh[rtl]`) with a clear error when an RTL target is requested without them?
3. Is the byte-identical-LTR invariant (11.1) a hard release gate, or may the LTR path migrate to shaping in the same release?

---

## Appendix A — Measurements

All figures measured against `GoNotoKurrent-Regular.ttf` (the font `download_remote_fonts()` selects for `ar`) with `uharfbuzz` + `fontTools`.

### A.1 Shaping correctness and width error at 12pt

| text | current GIDs (isolated) | correct GIDs (shaped) | current width | correct width | error |
|---|---|---|---|---|---|
| `الكتاب` | `3921,4541,4432,5151,3921,3975` | `3975,3922,5155,4436,4543,3921` | 48.47 | 31.96 | **+51.7%** |
| `لا إله` | `4541,3921,3,3929,4541,4355` (6) | `4356,4543,3929,3878,4579` (**5**) | 30.37 | 21.91 | **+38.6%** |
| `مرحبا بالعالم` | 13 glyphs | 13 glyphs, all different | 81.97 | 57.82 | **+41.8%** |
| `كِتَابٌ` | 7 glyphs, all spacing | 7 glyphs, **3 with zero advance** | 37.27 | 24.83 | **+50.1%** |

Shaped `كِتَابٌ` positions — note the zero advances and non-zero offsets that the current code cannot express:

```
advances: [0, 993, 291, 0, 373, 0, 412]
offsets : [(302,-160), (0,0), (0,0), (29,27), (0,0), (21,0), (0,0)]
```

### A.2 Arabic vs English width ratio at 12pt

| English | en width | ar width | ratio |
|---|---|---|---|
| Introduction | 70.5 | 37.4 | 0.53x |
| The results show a significant improvement. | 248.1 | 128.5 | 0.52x |
| Machine learning models require large datasets. | 272.7 | 232.6 | 0.85x |
| Conclusion and future work | 155.8 | 135.3 | 0.87x |
| Table 1: Comparison of methods | 182.5 | 130.1 | 0.71x |

### A.3 Vertical ink extents at size 12

| content | above baseline | below baseline | total |
|---|---|---|---|
| Latin `Highlight jpqy` | 9.12pt | -2.88pt | 12.00pt |
| Arabic, unvocalized | 8.76pt | -2.53pt | 11.29pt |
| Arabic + tashkeel | 12.79pt | -2.53pt | 15.32pt |
| Arabic tall + deep | 11.21pt | -4.73pt | 15.94pt |

Font metrics: `upem=1000`, `hhea` ascent/descent `1069 / -293`, `usWinAscent/Descent` `1124 / -395`.

### A.4 Horizontal ink overhang (verified non-issue, B09)

| text | advance | ink extent | left overhang | right overhang |
|---|---|---|---|---|
| `الجملة العربية` | 63.62 | `[0.36, 62.84]` | -0.36 | -0.78 |
| `على المستوى` | 63.96 | `[0.36, 63.60]` | -0.36 | -0.36 |
| `في هذا البحث` | 67.42 | `[0.36, 66.88]` | -0.36 | -0.54 |
| `جميع النتائج` | 57.05 | `[0.12, 56.57]` | -0.12 | -0.48 |

All negative: ink stays inside the advance box. No edge padding required.

### A.5 Bidi behaviour on numbers (why UAX #9 is mandatory)

| logical | correct (UAX #9) | naive string reversal |
|---|---|---|
| `النتيجة 42 بالمئة` | `ةئملاب 42 ةجيتنلا` | `ةئملاب 24 ةجيتنلا` WRONG |
| `النتيجة ٤٢ بالمئة` | `ةئملاب ٤٢ ةجيتنلا` | `ةئملاب ٢٤ ةجيتنلا` WRONG |
| `من 1990 إلى 2024` | `2024 ىلإ 1990 نم` | `4202 ىلإ 0991 نم` WRONG |
| `القيمة 3.14 والنسبة 25%` | `%25 ةبسنلاو 3.14 ةميقلا` | `%52 ةبسنلاو 41.3 ةميقلا` WRONG |
| `ISBN 978-0-13-235088-4` | `ISBN 978-0-13-235088-4` | `4-880532-31-0-879 NBSI` WRONG |
| `درجة الحرارة -5.2 مئوية` | `ةيوئم 5.2- ةرارحلا ةجرد` | `ةيوئم 2.5- ةرارحلا ةجرد` WRONG |

Relevant bidi classes: `4`,`2` = **EN**; `٤`,`٢` = **AN**; `.` = **CS**; `-` = **ES**; `%` = **ET**; `،` = **CS**; `؛`,`؟` = **AL**; `(`,`)`,`[`,`]` = **ON**.

### A.6 Font capability (verified sufficient, F04)

```
numGlyphs: 64750                       (< 65536, so %04x GID encoding is safe)
Arabic-block codepoints in cmap: 255
Presentation-form codepoints:    752
GSUB scripts: DFLT, arab, hebr, latn
GSUB Arabic features: calt, ccmp, fina, init, isol, liga, medi, rlig
GPOS scripts: DFLT, arab, hebr, latn
GPOS Arabic features: curs, mark, mkmk
```

---

## Appendix B — Verified API notes and gotchas

Things that cost time to discover; recorded so the implementer does not rediscover them.

1. **`python-bidi.get_display()` does not apply UAX #9 rule L4 (mirroring).** Brackets in RTL runs come back unmirrored. Apply `bidi.algorithm.MIRRORED` (362 entries) yourself to odd-level characters.

2. **`resolve_neutral_types` and `resolve_implicit_levels` take a positional `debug` argument.** Calling them with one argument raises `TypeError: missing 1 required positional argument: 'debug'`.

3. **`{v0}` does not survive bidi.** Verified by codepoint: `007B 0076 0030 007D` becomes `007D 0076 0030 007B`. The cause is L2 *reordering*, not mirroring. Any implementation that reorders before extracting placeholders is broken.

4. **Never verify bidi output by reading terminal text.** Terminals apply their own bidi to your already-reordered string, so wrong output can look right. Always compare codepoint sequences. (This produced a false "placeholders are safe" conclusion during analysis.)

5. **`U+FFFC OBJECT REPLACEMENT CHARACTER`** — bidi class `ON`, `Bidi_Mirrored = No`, survives reordering. The correct placeholder sentinel.

6. **HarfBuzz returns RTL runs in visual order** (leftmost glyph first). For `الكتاب` the first output GID is `3975`, the final form of the *last* logical character. That is exactly what the PDF content stream needs — no post-reversal.

7. **`buf.cluster_level = 1`** is required for usable cluster indices when mapping glyphs back to source characters.

8. **Index-tracked reordering was validated** against `get_display()` on a mixed corpus and matched exactly on every case.

9. **PyMuPDF cannot help here.** `Font.has_glyph` / `char_lengths` do no shaping, and `TextWriter(right_to_left=True)` reorders without shaping (known disjoining bugs). Only PyMuPDF's `Story` shapes, and it cannot emit absolutely positioned runs.

---

## Sources

- [fpdf2 — Text Shaping](https://py-pdf.github.io/fpdf2/TextShaping.html) and [source](https://github.com/py-pdf/fpdf2) (`fpdf/fonts.py`, `fpdf/line_break.py`, `fpdf/bidi.py`) — reference implementation of HarfBuzz + UAX #9 in a Python PDF writer
- [BabelDOC typesetting implementation](https://funstory-ai.github.io/BabelDOC/ImplementationDetails/Typesetting/Typesetting/) — adaptive scaling ladder; states LTR-only limitation
- [BabelDOC repository](https://github.com/funstory-ai/BabelDOC)
- [PDFMathTranslate issue #1091 — "Translart Arabic"](https://github.com/PDFMathTranslate/PDFMathTranslate/issues/1091) — the open user request this work answers
- [UAX #9 — Unicode Bidirectional Algorithm](https://www.unicode.org/reports/tr9/)
- [UAX #14 — Unicode Line Breaking Algorithm](http://www.unicode.org/reports/tr14/tr14-45.html)
- [PyMuPDF TextWriter docs](https://pymupdf.readthedocs.io/en/latest/textwriter.html) and issues [#897](https://github.com/pymupdf/PyMuPDF/issues/897), [#1719](https://github.com/pymupdf/PyMuPDF/issues/1719), [#2199](https://github.com/pymupdf/PyMuPDF/issues/2199)
- [W3C ALReq — Arabic layout requirements, justification](https://github.com/w3c/alreq/wiki/Draft-for-%E2%80%9C4.2-Justification%E2%80%9D)
- [On Arabic justification — TypoArabic, University of Reading](https://research.reading.ac.uk/typoarabic/on-arabic-justification-part-1/)
- [uniseg — pure-Python UAX #14](https://uniseg-py.readthedocs.io/en/latest/introduction.html)
