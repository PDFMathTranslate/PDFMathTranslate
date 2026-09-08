# RTL field-test report #002 — layout, fit and translation coverage

**Test material:** *How to Teach Philosophy to Your Dog* (pages 1–100),
*Social Engineering: The Art of Human Hacking* (pages 1–100)
**Engine:** Google Translate, `-lo ar`
**Date:** 2026-09-06
**Follows:** `RTL_FIELD_REPORT_001.md` (bracket mirroring, marker corruption)

The bracket and marker defects from report #001 are confirmed fixed in the
field. This pass found five further defects, four of them fixed.

Warning counts over the same 100 pages, before and after:

| | before | after |
|---|---|---|
| `RTL paragraph does not fit after scaling to 0.60` | 2 | **0** |
| `translator dropped formula marker(s)` | 10 | **0** |

---

## Defect 4 — Italic body text was never translated (FIXED)

**ID:** G04 · **Severity:** blocker · **Status:** fixed

### Symptom

Large amounts of English survived untranslated: all dialogue in *Philosophy to
Your Dog* (which the book sets in italic), plus book titles, emphasis and
foreign phrases. Page 42 kept `No, not morbid at all.`, `That was a yawn of
excitement…`, `Got it, thanks.` entirely in English.

### Precise cause

`vflag()` classifies a run as a formula by font name, using a regex aimed at
LaTeX math fonts. It ends with `|.*Ital|`:

```
(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary
 |.*Mono|.*Code|.*Ital|.*Sym|.*Math)
```

`.*Ital` matches **any** font whose name contains "Ital". Verified against the
document's own fonts:

```
TimesNewRomanPS-ItalicMT     vflag-formula = True     <-- ordinary emphasis
TimesNewRomanPSMT            vflag-formula = False
```

So every italic run became a preserved `{vN}` formula: never sent to the
translator, re-drawn from the original glyphs. In a Word-, InDesign- or
Quark-produced PDF — where italic means emphasis, not mathematics — that
silently excludes a large share of the text.

This also explains the high `dropped formula marker` count: the more `{vN}`
markers a paragraph carries, the more chances Google has to mangle one.

### Fix

`converter.py` now exempts known **text** font families before applying the
LaTeX rule:

```python
TEXT_ITALIC_FONTS = (
    r"(Times|Arial|Helvetica|Georgia|Garamond|Minion|Calibri|Cambria|Baskerville"
    r"|Caslon|Palatino|Charter|Bookman|BookAntiqua|Century|Verdana|Tahoma|Segoe"
    r"|Lato|Merriweather|SourceSerif|SourceSans|Noto|Roboto|OpenSans|Liberation"
    r"|Nimbus|FreeSerif|FreeSans|DejaVu|Charis|Gentium|Alegreya|Lora|Spectral)"
)
```

LaTeX math italics are unaffected: they use CMMI/CMTI/CMSY names, still caught
by the `CM[^R]` rule ahead of the italic clause.

### Behaviour change — needs your confirmation

**This changes output for every language, not just Arabic.** Italic text in
Times/Arial/etc. documents is now *translated* instead of preserved. That is
correct for prose, but it is a deliberate departure from the previous
behaviour, so it is called out here rather than buried. `--vfont` still
overrides the whole rule if you want the old behaviour back:

```
--vfont "(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)"
```

---

## Defect 5 — Short paragraphs shrank to unreadable micro-text (FIXED)

**ID:** G05 · **Severity:** major · **Status:** fixed

### Symptom

Short quoted lines rendered at a fraction of body size — visually ~4pt,
illegible — and still overlapped their neighbours. From the log:

```
RTL paragraph does not fit after scaling to 0.60: '"في وقت متأخر من الليل؟"'
RTL paragraph does not fit after scaling to 0.60: 'الاجتماعية-'
```

### Precise cause

Instrumenting the failing paragraph gave the answer directly:

```
cls=7.0  box=[129.0,208.5]  w=79.5  region=(129.0,208.0)  overflow_x=0.0  lines=2
```

`overflow_x = 0.0` — the text **fits the column horizontally**. It simply needs
two lines in a box one line tall, because the English source (`Late at night?`)
was one short line and `pheight` is the source ink height.

The fit ladder responded by shrinking the font. But shrinking cannot reduce the
line count in a 79.5pt-wide box, so it ran all the way to the 0.60 floor,
producing micro-text that *still* overflowed — the worst of both outcomes.

### Fix

Three changes, in order of preference:

1. **Expand into the real column first.** `Paragraph` now records its layout
   region id (`cls`), and the renderer recovers that region's true left/right
   bounds from the layout model's segmentation map. A single-line source
   paragraph is no longer trapped in the ink extent of its own short line;
   it expands leftward first (RTL text is right-anchored, so growing left is
   least disruptive), then rightward.

2. **Use the whitespace that actually exists below.** Available height is now
   the distance to the nearest paragraph below that horizontally overlaps,
   not just the source box height. Where nothing is known to be below, the
   original height is kept — figures and rules are not in the paragraph list,
   so expanding blindly would risk colliding with them.

3. **Stop shrinking when it cannot help.** If the text already fits
   horizontally and a scale step fails to reduce the line count, the ladder
   stops and keeps the readable size:

   ```python
   if fits_h and nlines >= last_nlines:
       break  # vertical-only overflow; shrinking further just makes it unreadable
   ```

Readable text that overflows slightly beats unreadable text that overflows
anyway.

### Verification

0 fit failures across 100 pages, down from 2. The failing paragraph now renders
at full size.

---

## Defect 6 — `-p 0-100` silently translated the last page (FIXED)

**ID:** G06 · **Severity:** minor · **Status:** fixed · pre-existing

Page numbers are 1-based on the command line, so `-p 0-100` produced
`range(-1, 100)` — and index `-1` is Python's *last* page. Visible in your own
log:

```
pages=[-1, 0, 1, 2, ...]
```

The document's final page was translated instead of the first, silently.
`pdf2zh.py` now clamps at 0. Pre-existing and language-independent.

---

## Defect 7 — Consecutive paragraphs still overlap vertically (OPEN)

**ID:** G07 · **Severity:** major · **Status:** open — needs a design decision

Where a short source paragraph is immediately followed by another with no gap,
a 2–3 line Arabic translation still overlaps the paragraph below. Defect 5's
step 2 recovers genuine whitespace, but when there is none, there is nowhere
to go.

Fixing this properly means **pushing subsequent paragraphs down** — a
page-level vertical reflow. That changes the project's core "preserve the
original layout" contract: once one paragraph moves, everything below it moves,
and content can be pushed off the page.

Options, for your call:

1. **Accept overlap** (current). Layout is faithful; dense pages collide.
2. **Cascade downward within the column**, stopping at the bottom margin and
   accepting overflow past it. Readable, no longer position-faithful.
3. **Cascade with spill to a continuation page.** Most readable, furthest from
   the original layout, largest change.

BabelDOC takes roughly option 2 for LTR. I recommend 2 behind a flag
(`--reflow {off,column}`, default `off`) so the current guarantee is preserved
unless asked for.

---

## Defect 8 — Contents and numbered lists flattened (OPEN)

**ID:** G08 · **Severity:** major · **Status:** open — pre-existing, not RTL
(same as G03 in report #001)

The two-column contents page and in-text numbered lists collapse into one
running paragraph. Section A groups characters into paragraphs purely by the
layout model's region id (`converter.py`, `cls == xt_cls`), and the model
returns one region for the whole block.

Not RTL-specific — Chinese flattens identically. The fix belongs in section A:
split a region when successive lines share a left edge (list items) or when a
large horizontal gap separates two columns (contents leaders). That changes
paragraph segmentation for every language and needs its own evaluation set.

---

## Not a defect — `list index out of range`

```
ERROR:pdf2zh.converter:list index out of range   converter.py:400
```

This is logged inside `worker()`'s exception handler, i.e. it came out of
`self.translator.translate(s)` — Google's response parsing failing on one
request. `@retry(wait=wait_fixed(1))` retried it and it succeeded. Translator
layer, pre-existing, self-healing. It is worth hardening `GoogleTranslator`
separately, but it does not affect output.

---

## Status

| ID | Defect | Status |
|---|---|---|
| G04 | Italic body text never translated | **fixed** (behaviour change — see above) |
| G05 | Micro-text from futile font shrinking | **fixed** |
| G06 | `-p 0-N` off-by-one selecting last page | **fixed** |
| G07 | Consecutive paragraphs overlap vertically | open — design decision needed |
| G08 | Contents / numbered lists flattened | open — pre-existing, not RTL |
| — | `list index out of range` | translator layer, retried, benign |

**Tests:** 112 passed, 1 skipped, 1 pre-existing Windows path failure in
`test_kernel.py`.
