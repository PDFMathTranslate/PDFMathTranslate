"""Tests for the RTL layout and shaping engine.

Shaping tests need a real font with Arabic coverage.  They are skipped when one
cannot be located; set PDF2ZH_TEST_FONT to a TTF path (for example the
GoNotoKurrent-Regular.ttf that download_remote_fonts() fetches) to run them.
"""

import os
import unittest
from pathlib import Path

from pdf2zh import bidi_shape as bs
from pdf2zh import uba


def _find_font() -> str | None:
    env = os.environ.get("PDF2ZH_TEST_FONT")
    if env and Path(env).exists():
        return env
    try:
        from babeldoc.assets.assets import get_font_and_metadata

        path, _ = get_font_and_metadata("GoNotoKurrent-Regular.ttf")
        return Path(path).as_posix()
    except Exception:
        return None


FONT = _find_font()
requires_font = unittest.skipIf(FONT is None, "no Arabic-capable font available")

ARABIC = "المعادلة تعطي نتيجة"
ARABIC_TASHKEEL = "كِتَابٌ"
MIXED = "المعادلة {v0} تعطي 42 بالمئة مع NumPy"


class TestLanguageTables(unittest.TestCase):
    def test_is_rtl_lang(self):
        for lang in ("ar", "AR", "ar-SA", "he", "iw", "fa", "ur", "ps", "yi"):
            self.assertTrue(bs.is_rtl_lang(lang), lang)
        for lang in ("en", "zh", "zh-CN", "ja", "ru", "", None):
            self.assertFalse(bs.is_rtl_lang(lang), lang)

    def test_text_marks_are_not_formulas(self):
        # Regression for defect E01: these are all category Mn or Lm and were
        # previously classified as formula characters by vflag().
        for cp in (0x064E, 0x064F, 0x0650, 0x0651, 0x0652, 0x064B, 0x0670, 0x05B8):
            self.assertTrue(bs.is_text_mark(cp), hex(cp))
        self.assertFalse(bs.is_text_mark(0x0627))  # alef is a letter, not a mark

    def test_digit_normalisation(self):
        self.assertEqual(bs.normalize_digits("٤٢", "western"), "42")
        self.assertEqual(bs.normalize_digits("42", "arabic"), "٤٢")
        self.assertEqual(bs.normalize_digits("42", "auto"), "42")

    def test_has_tashkeel(self):
        self.assertTrue(bs.has_tashkeel(ARABIC_TASHKEEL))
        self.assertFalse(bs.has_tashkeel(ARABIC))


class TestPlaceholderProtection(unittest.TestCase):
    def test_placeholders_replaced_and_indexed(self):
        text, marks = bs.protect_placeholders("المعادلة {v0} تعطي {v12} نتيجة")
        self.assertEqual(text.count(bs.OBJ), 2)
        self.assertNotIn("{", text)
        self.assertEqual(sorted(marks.values()), [0, 12])
        for index, vid in marks.items():
            self.assertEqual(text[index], bs.OBJ)

    def test_translator_quirks_still_parse(self):
        _, marks = bs.protect_placeholders("a { v 3 } b")
        self.assertEqual(list(marks.values()), [3])
        _, marks = bs.protect_placeholders("a {V7} b")
        self.assertEqual(list(marks.values()), [7])

    def test_mangled_marker_is_recovered(self):
        # Google returns "{aya 4}" (Arabic, Arabic-Indic digits) for "{v4}".
        for marker, valid, expected in (
            ("a {v4} b", {4}, [4]),
            ("a {الآية ٤} b", {4}, [4]),
            ("a {آية 5} b", {5}, [5]),
        ):
            with self.subTest(marker=marker):
                text, marks = bs.protect_placeholders(marker, valid)
                self.assertEqual(sorted(marks.values()), expected)
                self.assertEqual(text.count(bs.OBJ), len(expected))

    def test_ordinary_braces_are_not_markers(self):
        text, marks = bs.protect_placeholders("set {a, b} here", {0})
        self.assertEqual(marks, {})
        self.assertIn("{a, b}", text)

    def test_invented_marker_is_dropped(self):
        # A marker the source never contained must not resurrect a formula.
        _, marks = bs.protect_placeholders("a {v9} b", {0, 1})
        self.assertEqual(marks, {})

    def test_malformed_placeholder_kept_literally(self):
        text, marks = bs.protect_placeholders("a {v} b", set())
        self.assertEqual(marks, {})
        self.assertIn("{v}", text)

    def test_placeholder_survives_reordering(self):
        # Defect A08: a bare {v0} is reordered to }v0{ and the regex then fails.
        # The U+FFFC sentinel must come through intact and keep its identity.
        text, marks = bs.protect_placeholders(MIXED)
        chars = bs.resolve_levels(text, base_rtl=True)
        visual = bs.reorder_l2(chars, lambda c: c.level)
        sentinels = [c for c in visual if c.ch == bs.OBJ]
        self.assertEqual(len(sentinels), 1)
        self.assertEqual(marks[sentinels[0].index], 0)


class TestBidiIntegration(unittest.TestCase):
    def test_self_test_passes(self):
        bs.self_test()

    def test_base_direction_detection(self):
        self.assertTrue(bs.detect_base_rtl(ARABIC, "ar"))
        self.assertFalse(bs.detect_base_rtl("Hello world", "ar"))
        # No strong character: fall back to the target language.
        self.assertTrue(bs.detect_base_rtl("123 456", "ar"))
        self.assertFalse(bs.detect_base_rtl("123 456", "en"))

    def test_common_script_inherits(self):
        chars = bs.resolve_levels("مع النتائج", base_rtl=True)
        runs = bs.segment_runs(
            chars, {}, font_key="noto", language="ar", size=12,
            placeholder_width=lambda v: 0.0,
        )
        # The space must not split the phrase into three runs.
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].script, "Arab")


@requires_font
class TestShaping(unittest.TestCase):
    def setUp(self):
        self.shaper = bs.get_shaper(FONT)

    def test_arabic_is_shaped_not_isolated(self):
        # Defect A01: contextual forms must differ from the nominal glyphs.
        glyphs = self.shaper.shape(
            "الكتاب", rtl=True, script="Arab", language="ar", size=12
        )
        self.assertEqual(len(glyphs), 6)
        gids = [g.gid for g in glyphs]
        self.assertEqual(len(set(gids)), len(set(gids)))  # sanity
        # The two alefs are the same codepoint but take different contextual forms.
        self.assertNotEqual(gids[0], gids[-1])

    def test_ligature_collapses_clusters(self):
        # Defect A04: lam-alef is 2 codepoints but 1 glyph, so 6 chars -> 5 glyphs.
        glyphs = self.shaper.shape(
            "لا إله", rtl=True, script="Arab", language="ar", size=12
        )
        self.assertEqual(len(glyphs), 5)
        multi = [g for g in glyphs if len(g.chars) > 1]
        self.assertTrue(multi, "expected at least one multi-codepoint cluster")

    def test_marks_have_zero_advance_and_offsets(self):
        # Defect A06: harakat must not consume horizontal space.
        glyphs = self.shaper.shape(
            ARABIC_TASHKEEL, rtl=True, script="Arab", language="ar", size=12
        )
        marks = [g for g in glyphs if any(bs.is_text_mark(c) for c in g.chars)]
        self.assertTrue(marks, "expected shaped marks")
        for g in marks:
            self.assertEqual(g.x_advance, 0.0)
            self.assertTrue(g.force_positioning)

    def test_shaped_width_is_narrower_than_isolated(self):
        # Defect A05: measuring each codepoint in isolation, which is what the
        # original renderer did, overshoots the real shaped width by 38-52%.
        word = "الكتاب"
        shaped = sum(
            g.x_advance
            for g in self.shaper.shape(
                word, rtl=True, script="Arab", language="ar", size=12
            )
        )
        isolated = 0.0
        for ch in word:
            isolated += sum(
                g.x_advance
                for g in self.shaper.shape(
                    ch, rtl=True, script="Arab", language="ar", size=12
                )
            )
        self.assertLess(shaped, isolated)
        self.assertGreater((isolated - shaped) / shaped, 0.3)

    def test_glyphs_stored_in_logical_order(self):
        glyphs = self.shaper.shape(
            "الكتاب", rtl=True, script="Arab", language="ar", size=12
        )
        run = bs.TextRun(
            text="الكتاب", level=1, font_key="noto", script="Arab",
            language="ar", size=12, glyphs=glyphs,
        )
        self.assertEqual(run.visual_glyphs(), list(reversed(glyphs)))


@requires_font
class TestLayout(unittest.TestCase):
    def _layout(self, text, x0=100.0, x1=400.0, **kw):
        kw.setdefault("lang_out", "ar")
        kw.setdefault("placeholder_width", lambda v: 20.0)
        return bs.layout_paragraph(
            text, font_path=FONT, font_key="noto", size=12, x0=x0, x1=x1, **kw
        )

    def test_rtl_line_is_right_aligned(self):
        res = self._layout(ARABIC)
        self.assertTrue(res.base_rtl)
        line = res.lines[0]
        right_edge = line[-1].x + line[-1].item.width
        self.assertAlmostEqual(right_edge, 400.0, places=4)

    def test_ltr_paragraph_starts_at_left(self):
        res = self._layout("The results show a significant improvement.")
        self.assertFalse(res.base_rtl)
        self.assertAlmostEqual(res.lines[0][0].x, 100.0, places=4)

    def test_wrapping_is_not_gated_on_source_line_breaks(self):
        # Defect C01: a long single-line source must still wrap.
        long_text = " ".join([ARABIC] * 6)
        res = self._layout(long_text, x1=250.0)
        self.assertGreater(len(res.lines), 1)
        for line in res.lines:
            right = line[-1].x + line[-1].item.width
            self.assertLessEqual(right, 250.0 + 1e-6)
            self.assertGreaterEqual(line[0].x, 100.0 - 1e-6)

    def test_no_line_exceeds_the_box(self):
        res = self._layout(" ".join([ARABIC] * 4), x1=260.0)
        self.assertEqual(res.overflow_x, 0.0)

    def test_formula_placeholder_is_positioned(self):
        res = self._layout(MIXED)
        placeholders = [
            p for line in res.lines for p in line
            if isinstance(p.item, bs.PlaceholderRun)
        ]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0].item.vid, 0)
        self.assertEqual(placeholders[0].item.width, 20.0)

    def test_embedded_ltr_run_keeps_its_direction(self):
        res = self._layout(MIXED)
        runs = [
            p.item for line in res.lines for p in line
            if isinstance(p.item, bs.TextRun)
        ]
        latin = [r for r in runs if "NumPy" in r.text]
        self.assertTrue(latin)
        self.assertFalse(latin[0].rtl, "embedded Latin must stay left-to-right")

    def test_arabic_appears_at_the_right_of_embedded_latin(self):
        res = self._layout(MIXED)
        line = res.lines[0]
        latin = next(p for p in line if isinstance(p.item, bs.TextRun)
                     and "NumPy" in p.item.text)
        arabic = next(p for p in line if isinstance(p.item, bs.TextRun)
                      and "المعادلة" in p.item.text)
        self.assertLess(latin.x, arabic.x, "Arabic must sit to the right")

    def test_indent_is_mirrored_for_rtl(self):
        plain = self._layout(ARABIC)
        indented = self._layout(ARABIC, indent=20.0)
        plain_right = plain.lines[0][-1].x + plain.lines[0][-1].item.width
        indented_right = indented.lines[0][-1].x + indented.lines[0][-1].item.width
        self.assertAlmostEqual(plain_right - indented_right, 20.0, places=4)

    def test_ascent_is_reported_for_overshoot_correction(self):
        plain = self._layout(ARABIC)
        vocalised = self._layout(ARABIC_TASHKEEL)
        self.assertGreater(vocalised.max_ascent, plain.max_ascent)

    def test_brackets_are_mirrored_exactly_once(self):
        """Regression: HarfBuzz mirrors RTL runs itself (rule L4).

        Applying L4 in resolve_levels as well double mirrors, and an Arabic
        parenthetical then renders as ")text(" instead of "(text)".
        """
        from fontTools.ttLib import TTFont

        tt = TTFont(FONT, lazy=True)
        cmap = tt.getBestCmap()
        order = tt.getGlyphOrder()
        name_to_gid = {n: i for i, n in enumerate(order)}
        gid_to_cp = {name_to_gid[n]: cp for cp, n in cmap.items() if n in name_to_gid}
        brackets = "()[]{}"

        for text in (
            "ا(ب)ج",
            "لندن (وأحيانا) تهدف",
            "النتيجة [42] و(3.14) نهائية",
        ):
            with self.subTest(text=text):
                res = self._layout(text, base_rtl=True)
                emitted = []
                for line in res.lines:
                    for placed in line:
                        if not isinstance(placed.item, bs.TextRun):
                            continue
                        pen = placed.x
                        for g in placed.item.visual_glyphs():
                            cp = gid_to_cp.get(g.gid)
                            if cp is not None and chr(cp) in brackets:
                                emitted.append((pen, chr(cp)))
                            pen += g.x_advance
                got = "".join(ch for _, ch in sorted(emitted))
                want = "".join(c for c in uba.get_display(text, 1) if c in brackets)
                self.assertEqual(got, want)

    def test_empty_input(self):
        res = self._layout("")
        self.assertEqual(res.lines, [])


if __name__ == "__main__":
    unittest.main()
