# RTL field-test report #001 — Arabic output review

**Test material:** *How to Teach Philosophy to Your Dog*, pages 1–5, 10, 33, 50
**Engine:** Google Translate, `-lo ar`
**Reviewed from:** 23 side-by-side screenshots (original vs. translated)
**Date:** 2026-09-06

Overall the Arabic renders correctly: contextual joining, right-to-left flow,
right-aligned ragged-left paragraphs, word-boundary wrapping, tashkeel
positioning, Arabic punctuation, paragraph indentation, and embedded italic
Latin inside Arabic sentences all behave. Four defects were found. Two are
fixed in this pass; two are pre-existing and not RTL-specific.

---

## Defect 1 — Brackets rendered backwards (FIXED)

**ID:** G01 · **Severity:** blocker · **Status:** fixed + regression test

### Symptom

Every parenthetical in Arabic body text rendered inside-out. Instead of

> `(وأحيانًا في أماكن أبعد قليلاً)`

the output drew the closing glyph on the left and the opening glyph on the
right — visually `)…(`. Visible on page 10 (`and occasionally a little further
afield`), page 33 (`428/427–348/347 bce`), page 50 (`1671–1713`, `1694–1746`,
`pitié`).

### Precise cause: double mirroring

UAX #9 rule **L4** says a mirrorable character at an odd (RTL) embedding level
must be drawn with its mirrored glyph: the character `U+0028 (` is *displayed*
using the glyph of `U+0029 )`. That is what makes a parenthesis look correct
when the reading direction reverses.

We applied L4 ourselves in `bidi_shape.resolve_levels`. **HarfBuzz also applies
mirroring**, automatically, to any buffer whose direction is RTL. Because every
run is shaped with the direction implied by its own embedding level, HarfBuzz's
mirroring already *is* rule L4.

The two cancelled out. Traced directly:

```
logical text                      : 0627 0028 0628 0029 062C    ا ( ب ) ج
after our L4 in resolve_levels    : 0627 0029 0628 0028 062C    ا ) ب ( ج
raw HarfBuzz output (gid, cluster): [(4405,4), (12,3), (3975,2), (11,1), (3921,0)]
                                              ^^^^^^            ^^^^^^
   cluster 3 held U+0028 '(' -> HarfBuzz emitted gid 12 = U+0029 ')'
   cluster 1 held U+0029 ')' -> HarfBuzz emitted gid 11 = U+0028 '('
```

HarfBuzz mirrored a second time, restoring the original glyph shapes while the
*positions* were correctly reversed. Net effect: unmirrored glyphs in reversed
positions — exactly `)…(`.

### Why it happened

The specification in `RTL_ARABIC_SUPPORT.md` §5.4 correctly established that
`python-bidi.get_display()` does not apply L4, and concluded "so we must apply
it ourselves." That reasoning was sound for the *bidi* layer but never asked
the next question: does the **shaper** apply it? It does. The defect was
introduced by verifying the bidi layer in isolation — `uba.get_display()` is
conformance-correct and was tested against 91,707 Unicode cases — without an
end-to-end assertion on the glyph IDs that actually reach the PDF.

### Fix

`pdf2zh/bidi_shape.py` — `resolve_levels()` no longer mirrors:

```python
# NOTE: rule L4 (mirroring) is deliberately NOT applied here.  HarfBuzz
# mirrors mirrorable characters itself for any buffer whose direction is
# RTL, and because every run is shaped with the direction implied by its own
# embedding level, that is exactly L4.  Applying it here as well double
# mirrors: the two cancel and brackets come out backwards.
out.append(BidiChar(ch=ch, level=levels[i], index=i))
```

`uba.mirror_char()` is retained — it is still needed by `uba.get_display()`,
which is the reference oracle used in tests.

### Verification

New regression test `test_brackets_are_mirrored_exactly_once` decodes emitted
glyph IDs back through the font cmap and asserts the left-to-right bracket
sequence equals `uba.get_display()`. Confirmed on the real document:

| page | before | after |
|---|---|---|
| 10 | `)(` | `()` |
| 33 | `)(` | `()` |
| 50 | `)(` ×3 | `()` ×6 |

---

## Defect 2 — Formula markers corrupted by the translator (FIXED, with a caveat)

**ID:** G02 · **Severity:** major · **Status:** mitigated + warning added

### Symptom

Literal curly braces containing Arabic appeared in the output — `{الآية ٤}` on
pages 33 and 50 — and some italicised words (`Republic`, `pitié`) vanished
entirely from the translation.

### Precise cause

`converter.py` replaces formulas and italic runs with the marker `{vN}` before
translation. Google Translate **translates the marker text itself**. From the
translation cache:

```
source placeholders : ['0','1','2','3','4','5']
returned            : ['1','2','3']  +  raw brace groups ['{آية ٤}', '{آية ٥}']
```

Google read `v` as an abbreviation for *verse*, rendered it as `آية`, and
converted the digit to Arabic-Indic (`4` → `٤`). The result no longer matches
`\{\s*v([\d\s]+)\}`, so it survived into the output as literal text. `{v0}` was
dropped outright.

### Why it happened

The instruction "Keep the formula notation {v\*} unchanged" exists only in
`BaseTranslator.prompt()`, which is used by the LLM translators.
`GoogleTranslator` posts plain text to a web endpoint with no prompt and no
placeholder protection, so nothing defends the marker.

**This is pre-existing and not RTL-specific** — the same corruption occurs for
Chinese and every other target. Arabic makes it more visible because Google
both translates the token and transliterates the digits.

### Fix

`pdf2zh/bidi_shape.py` gains tolerant marker recovery:

- a looser pattern `\{[^{}]{0,32}\}` catches rewritten markers;
- the digits inside are extracted and parsed (`int()` accepts Arabic-Indic);
- the result is accepted **only if it names a formula that actually exists in
  this paragraph**, so ordinary braces in prose are never mistaken for markers,
  and a translator-invented marker cannot resurrect a formula.

`converter.py` computes the valid marker set from the source string and logs
when the translator loses one:

```
translator dropped formula marker(s) [0] from an RTL paragraph;
that content will be missing from the output
```

### Caveat

Recovery only works when *something* survives. When the translator deletes the
marker entirely (as with `{v0}`), the formula content is unrecoverable at
render time — the warning is now the visible signal rather than silent loss.

The durable fix belongs upstream in the translator layer: emit a marker that
survives machine translation. `BaseTranslator` already has unused hooks for
this (`get_formular_placeholder`, `get_rich_text_left_placeholder`) which the
v1 converter ignores in favour of the hardcoded `{vN}`. Wiring those up is
tracked as a follow-up; it is a translator-layer change affecting all
languages, so it is deliberately out of this RTL pass.

---

## Defect 3 — Table of contents flattened into a run-on paragraph (NOT FIXED)

**ID:** G03 · **Severity:** major · **Status:** open, pre-existing, not RTL

### Symptom

Page 4's two-column contents list

```
Walk 1     Good Dog, Bad Dog
Walk 2     Plato, Aristotle and the Good Life
```

became a single flowing Arabic paragraph with the numbers inlined.

### Cause

Section A of `receive_layout` groups characters into paragraphs by the layout
model's region id (`cls == xt_cls`, `converter.py:284`). The ONNX layout model
classifies the whole contents block as one region, so every entry is
concatenated into one string and sent to the translator as one sentence.

This is **entirely independent of RTL** — the same flattening happens for
Chinese. Fixing it means teaching section A to split a region on large
horizontal gaps or consistent line starts, which changes paragraph segmentation
for every language and needs its own evaluation.

---

## Non-defect — untranslated English fragments

`Author's Note`, `Prodogue`, `Further Reading`, `Acknowledgements`, `Index`,
and the subtitle *Exploring the Big Questions in Life* stayed in English.

This is correct behaviour on our side. Those are separate paragraphs whose
first strong character is Latin, so `detect_base_rtl()` routes them through the
unchanged LTR path; Google simply returned them unchanged. Nothing to fix.

---

## Status after this pass

| ID | Defect | Status |
|---|---|---|
| G01 | Brackets mirrored twice, rendered backwards | **fixed**, regression test added |
| G02 | Translator corrupts `{vN}` markers | **mitigated**, warning added; upstream fix pending |
| G03 | Contents block flattened into one paragraph | open, pre-existing, not RTL |
| — | Untranslated English fragments | working as intended |

**Tests:** 112 passed, 1 skipped. The single failure in `test_kernel.py`
(`test_output_defaults_to_input_parent`) is a pre-existing Windows path
separator assertion — `C:\some\dir` vs `/some/dir` — and reproduces with these
changes stashed.

## Method note

Two of my intermediate diagnoses in this session were wrong before the trace
above settled it, both from the same root cause: **rendered bidi text cannot be
verified by reading it**. Transcribing Arabic from a screenshot, or printing a
reordered string to a terminal, re-applies bidi to the transcription and can
make wrong output look right (and vice versa). Every conclusion here is based
on decoding emitted glyph IDs through the font cmap and comparing against
`uba.get_display()`. That is the only reliable check, and the new regression
test enforces it.
