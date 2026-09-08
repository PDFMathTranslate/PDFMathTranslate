"""Conformance tests for the Unicode Bidirectional Algorithm implementation.

The authoritative test is the Unicode Character Database's
`BidiCharacterTest.txt`.  It is not vendored (it is ~7 MB); when it is absent
the conformance test is skipped and only the hand-written cases run.

To run the full suite, download it next to this file or set BIDI_TEST_FILE:

    curl -o BidiCharacterTest.txt \\
        https://www.unicode.org/Public/UCD/latest/ucd/BidiCharacterTest.txt
"""

import os
import unittest
from pathlib import Path

from pdf2zh import uba


def _find_conformance_file() -> Path | None:
    env = os.environ.get("BIDI_TEST_FILE")
    if env and Path(env).exists():
        return Path(env)
    for candidate in (
        Path(__file__).parent / "BidiCharacterTest.txt",
        Path(__file__).parent / "file" / "BidiCharacterTest.txt",
    ):
        if candidate.exists():
            return candidate
    return None


class TestBidiConformance(unittest.TestCase):
    def test_bidi_character_test(self):
        path = _find_conformance_file()
        if path is None:
            self.skipTest("BidiCharacterTest.txt not available")
        total = level_fail = order_fail = 0
        first_failures = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split(";")
                if len(fields) < 5:
                    continue
                cps, direction, _para, levels_s, reorder_s = fields[:5]
                text = "".join(chr(int(c, 16)) for c in cps.split())
                para = None if direction == "2" else int(direction)
                total += 1

                levels, _ = uba.resolve(text, para)
                expected = levels_s.split()
                if not all(
                    e == "x" or levels[i] == int(e) for i, e in enumerate(expected)
                ):
                    level_fail += 1
                    if len(first_failures) < 5:
                        first_failures.append(("levels", cps, levels_s, levels))
                    continue

                expected_order = (
                    [int(x) for x in reorder_s.split()] if reorder_s.strip() else []
                )
                if uba.display_order(text, para) != expected_order:
                    order_fail += 1
                    if len(first_failures) < 5:
                        first_failures.append(("order", cps, reorder_s, None))

        self.assertGreater(total, 1000, "conformance file looks truncated")
        self.assertEqual(
            (level_fail, order_fail),
            (0, 0),
            f"{level_fail} level and {order_fail} order failures out of {total}; "
            f"first: {first_failures}",
        )


class TestBidiBasics(unittest.TestCase):
    def test_base_level_detection(self):
        self.assertEqual(uba.base_level("hello"), 0)
        self.assertEqual(uba.base_level("مرحبا"), 1)
        self.assertEqual(uba.base_level("123 مرحبا"), 1)  # digits are not strong
        self.assertEqual(uba.base_level("123"), 0)  # no strong character

    def test_numbers_keep_ltr_order_inside_rtl(self):
        # Regression for defect D06: naive reversal corrupts every number.
        for text, number in (
            ("من 1990 إلى 2024", "1990"),
            ("النتيجة 42 بالمئة", "42"),
            ("القيمة 3.14 والنسبة", "3.14"),
            ("ISBN 978-0-13-235088-4 المعرف", "978-0-13-235088-4"),
        ):
            with self.subTest(text=text):
                self.assertIn(number, uba.get_display(text, 1))

    def test_mirroring_applied(self):
        # Rule L4: brackets in an RTL run render mirrored.
        visual = uba.get_display("النتيجة (تقريبا) نهائية", 1)
        self.assertIn("(", visual)
        self.assertIn(")", visual)
        # The pair must still read as a well-formed pair left to right.
        self.assertLess(visual.index("("), visual.index(")"))

    def test_bracket_pairing_rule_n0(self):
        # The case python-bidi's legacy path gets wrong.
        levels, _ = uba.resolve("אב(גד[&ef].)gh", 0)
        self.assertEqual(levels[2], 0, "opening bracket must take the base level")

    def test_x9_characters_are_removed(self):
        self.assertTrue(uba.is_removed("‪"))  # LRE
        self.assertTrue(uba.is_removed("​" if False else "­"))  # SHY is BN
        self.assertFalse(uba.is_removed("a"))
        self.assertFalse(uba.is_removed("ا"))

    def test_empty_and_neutral_input(self):
        self.assertEqual(uba.resolve("", None), ([], 0))
        self.assertEqual(uba.get_display("   ", 0), "   ")


if __name__ == "__main__":
    unittest.main()
