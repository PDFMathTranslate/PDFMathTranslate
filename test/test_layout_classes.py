from pdf2zh.high_level import LAYOUT_EXCLUDED_CLASSES


def test_abandon_regions_are_not_excluded_from_text_translation():
    assert "abandon" not in LAYOUT_EXCLUDED_CLASSES
    assert {"figure", "table", "isolate_formula", "formula_caption"} == set(
        LAYOUT_EXCLUDED_CLASSES
    )
