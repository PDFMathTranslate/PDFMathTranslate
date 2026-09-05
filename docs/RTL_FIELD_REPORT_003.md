# RTL field-test report #003 — punctuation, list markers, paragraph collisions

**Test material:** *How to Teach Philosophy to Your Dog* and
*Social Engineering: The Art of Human Hacking*, pages 1–100 each
**Date:** 2026-09-06
**Follows:** reports #001 (brackets, markers) and #002 (italics, fit, page range)

Regression over 200 pages, two books:

| | result |
|---|---|
| `does not fit after scaling` | 0 |
| `translator dropped formula marker(s)` | 0 |
| `ERROR` | 0 |

---

## Punctuation ordering — measured, and it is correct

The reported "punctuation appearing at the end instead of the beginning" was
tested rather than eyeballed, because rendered bidi text cannot be read
reliably (see the method note in report #001).

**Method.** 244 real translated paragraphs were pulled from the translation
cache, laid out in a box wide enough to force a single line, and the emitted
glyphs' *source characters* were compared, in visual order, against
`uba.display_order()` — the conformance-tested reference.

**Result.** 238 of 244 matched exactly. The 6 apparent failures were an
artefact of the checker: it counted the digits inside `{vN}` markers as
punctuation, while the engine correctly consumes those markers into formula
boxes. No real ordering defect exists.

> A note on expectations: in correct Arabic typesetting a sentence-final
> period sits at the **left** end of the line, because that is where the
> sentence ends in reading order. Word, LibreOffice and every browser render
> it the same way. It looks wrong to a reader used to Latin script, but it is
> right.

A first attempt at this measurement produced 199/244 "failures" and was wrong:
it decoded glyph IDs back to characters through the font cmap, and that map is
many-to-one, so the period's glyph resolved to a different codepoint and
appeared to be missing. Comparing source characters carried on each glyph
cluster is the correct method.

---

## Defect 9 — List bullets on the wrong side (FIXED)

**ID:** G09 · **Severity:** major · **Status:** fixed

### Symptom

This was the real cause of the "marks at the wrong end" report. On bulleted
pages every `•` sat to the **left** of the Arabic text. In Arabic the marker
belongs on the right, where the line begins.

### Cause

Section A pins bullets deliberately:

```python
if child.get_text() == "•":
    cls = 0          # 锚定文档中 bullet 的位置
```

`cls = 0` marks a preserved region, so the bullet becomes a formula anchored at
its **original absolute x** and is re-drawn there. That is correct for LTR and
wrong for RTL, where the whole text block flips to the other edge but the
marker does not follow.

### Fix

`converter.py` gains `mirror_bullet()`. For an RTL target, a preserved region
whose glyphs are all bullet characters is mirrored to the far side of the text
paragraph that shares its line, preserving the original marker-to-text gap:

```python
gap = host.x0 - para.x1      # original space between marker and text
para.x = para.x0 = host.x1 + gap
```

Only genuine bullet glyphs are moved (`• ‣ ▪ ▫ ◦ ● ○ · ∙ * – — -`). Equations
and other preserved regions are untouched, since mirroring a formula would put
it in the wrong place relative to its own text.

---

## Defect 7 (from report #002) — Paragraph collisions (LARGELY FIXED)

**ID:** G07 · **Severity:** major · **Status:** substantially fixed; residual
cases remain at page bottoms

### Symptom

Blocks of Arabic drawn on top of each other, producing unreadable dense
patches. Confirmed by measurement, not by eye: on the sample page, five pairs
of baselines sat 0.8–1.6pt apart where normal line spacing is ~10.2pt.

### Cause

A paragraph whose translation needs more lines than the source occupied spills
downward into the next paragraph. Report #002 recovered genuine whitespace
below each paragraph, but where the next paragraph starts immediately there is
none, and the two overlap.

### Fix

Layout is now split into two phases. `plan_rtl()` computes a paragraph's
layout, and `emit_rtl()` draws it; between them a collision pass runs
top-to-bottom over the page:

```python
for plan in sorted(rtl_plans, key=lambda p: -span(p)[0]):
    # push down only as far as needed to clear paragraphs already placed
    # in the same column, and never past the bottom of the page
```

Only paragraphs that would actually collide move, and only by the minimum
amount. Pages without collisions are byte-identical to before, so layout
fidelity is preserved wherever it can be.

Observed shifts on the sample page: 14.6pt, 5.4pt, 21.7pt. The dense
overlapping block in the middle of the page is gone.

### Residual limitation

Where a paragraph is already near the bottom of the page there is nowhere to
push it, the shift is clamped to zero, and the overlap remains. Resolving those
cases requires either spilling onto a continuation page or shrinking the font
below the readability floor — both worse trade-offs than a local overlap, and
both changes to the project's layout-preservation contract. Left as-is,
deliberately.

---

## Still open

| ID | Defect | Note |
|---|---|---|
| G08 | Contents pages and numbered lists flatten into one running paragraph | Pre-existing, not RTL. Section A groups by layout-model region id; the model returns one region for the whole block. Chinese flattens identically. Fixing it means splitting a region on shared left edges (list items) or wide horizontal gaps (contents leaders) — that changes paragraph segmentation for every language and needs its own evaluation. |
| G07 residual | Overlap at page bottoms | See above; needs a layout-contract decision. |
| G02 residual | Translator deletes a `{vN}` marker outright | Recoverable only upstream. `BaseTranslator` has unused `get_formular_placeholder` hooks the v1 converter ignores in favour of the hardcoded `{vN}`. |

---

## Status across all three reports

| ID | Defect | Status |
|---|---|---|
| G01 | Brackets mirrored twice, rendered backwards | fixed |
| G02 | Translator corrupts `{vN}` markers | mitigated; upstream fix pending |
| G03/G08 | Contents and lists flattened | open, pre-existing, not RTL |
| G04 | Italic body text never translated | fixed |
| G05 | Micro-text from futile font shrinking | fixed |
| G06 | `-p 0-N` off-by-one selecting last page | fixed |
| G07 | Paragraphs overlap vertically | largely fixed; residual at page bottoms |
| G09 | List bullets on the wrong side | fixed |

**Tests:** 112 passed, 1 skipped, 1 pre-existing Windows path failure in
`test_kernel.py`.

---

## Addendum — Defect 10: headings split one word per line (REGRESSION, FIXED)

**ID:** G10 · **Severity:** major · **Status:** fixed · **introduced by report #002**

### Symptom

Headings that previously rendered on one line broke into two, one word per
line — e.g. `ملاحظة المؤلف` (Author's Note) split as `ملاحظة` / `المؤلف`.

### Cause — my own regression

Report #002 added this rule to stop the micro-text problem:

```python
if fits_h and nlines >= last_nlines:
    break   # shrinking stopped helping
```

It steps the scale down 5% at a time and gives up as soon as one step fails to
reduce the line count. A heading typically needs 15–20% to pull back onto one
line, so a single 5% step never gets there and the rule bails out immediately,
leaving the heading wrapped.

A second, related fault: a wrapped heading passes the *vertical* fit test
(there is space below it), so the ladder broke out at the top and never even
reached the shrink logic.

### Fix — compute the scale instead of guessing it

The natural unwrapped width of the paragraph is now measured once, and the
required scale is derived directly rather than probed:

- **One-line sources** (headings, captions, table cells — `brk == False`) are
  treated as not fitting until they are one line. Required scale is
  `box_width / natural_width`. Applied when it is a moderate shrink
  (>= 0.75); below that, wrapping is accepted as the better outcome.
- **Vertical overflow**: line count grows roughly linearly with scale and
  occupied height roughly with its square, so the needed scale is about
  `sqrt(available_height * column_width / (natural_width * size * line_height))`.
  Applied in one step, floor at `--min-font-scale`.

Each iteration is guaranteed to reduce the scale by at least 0.02, so the loop
terminates.

This also fixed the residual page-bottom overlap from G07: dense full pages
(e.g. *Social Engineering* page 55) were overlapping for exactly the same
reason — the step-wise probe gave up before reaching a scale that fits. Those
pages now render cleanly.

### Verification

- `ملاحظة المؤلف` back on one line.
- *Social Engineering* page 55 renders with no overlap.
- 200 pages across both books: 0 fit failures, 0 dropped markers, 0 errors.

### Lesson

Both this defect and the micro-text defect it replaced came from probing a
continuous quantity in fixed steps and inferring "it cannot help" from one
failed step. Where the relationship can be computed — and here it can, from the
text's natural width — computing it is both correct and faster.
