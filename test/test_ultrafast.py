"""Offline integration check for native layout and the model-free CLI path."""

import tempfile
import logging
from threading import Event
from unittest.mock import Mock, patch

import pymupdf
import pytest

from pdf2zh.pdf2zh import main, parse_args


def test_ultrafast_rejects_other_backends():
    for options in (
        ["--mode", "precise"],
        ["--babeldoc"],
        ["--onnx", "model.onnx"],
        ["--interactive"],
    ):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--ultrafast", *options])
        assert exc.value.code == 2


def test_worker_defaults_and_overrides():
    from pdf2zh.kernel.protocol import TranslateRequest

    assert parse_args([]).thread == 4
    assert parse_args(["--ultrafast"]).thread == 16
    assert parse_args(["--ultrafast", "-t", "3"]).thread == 3
    assert TranslateRequest(files=[], ultrafast=True).thread == 16
    assert TranslateRequest(files=[], ultrafast=True, thread=4).thread == 4
    with pytest.raises(SystemExit):
        parse_args(["--ultrafast", "-t", "0"])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_ultrafast_cli_without_models_rendering_or_ocr(tmp_path, rotation, caplog):
    caplog.set_level(logging.INFO)
    pytest.importorskip("pdf_inspector")
    # Keep the translation cache outside the user's home during this check.
    with patch("os.path.expanduser", return_value=tempfile.mkdtemp()):
        from pdf2zh import high_level
    from pdf2zh.doclayout import ModelInstance, OnnxModel

    source = tmp_path / "native.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((20, 40), "Original paragraph")
        page.insert_text((20, 55), "continues on another line")
        page.insert_text((180, 40), "Second column")
        page.set_cropbox(pymupdf.Rect(10, 10, 290, 190))
        page.set_rotation(rotation)
        doc.new_page(width=300, height=200).insert_text((20, 40), "Another page")
        scan = doc.new_page(width=300, height=200)
        pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 20, 20), False)
        pix.clear_with(255)
        scan.insert_image(scan.rect, pixmap=pix)
        doc.save(source)
    font = tmp_path / "font.ttf"
    font.write_bytes(pymupdf.Font("helv").buffer)
    translator = Mock()
    next_page_started = Event()
    cross_page_overlap = []

    def translate(text):
        if "Another page" in text:
            next_page_started.set()
        if "Original paragraph" in text:
            cross_page_overlap.append(next_page_started.wait(timeout=2))
        return "Translated paragraph"

    translator.translate.side_effect = translate
    translator.lang_out = "en"
    factory = Mock(name="translator", return_value=translator)
    factory.name = "google"
    with (
        patch.object(
            OnnxModel, "load_available", side_effect=AssertionError("model loaded")
        ),
        patch.object(OnnxModel, "predict", side_effect=AssertionError("model called")),
        patch.object(
            pymupdf.Page, "get_pixmap", side_effect=AssertionError("page rendered")
        ),
        patch.object(
            high_level, "_ocr_pages", side_effect=AssertionError("OCR called")
        ),
        patch.object(
            pymupdf.Document,
            "subset_fonts",
            side_effect=AssertionError("fonts subsetted"),
        ),
        patch.object(high_level, "download_remote_fonts", return_value=str(font)),
        patch("pdf2zh.converter.GoogleTranslator", factory),
        patch.object(ModelInstance, "value", None),
    ):
        assert (
            main(
                [
                    str(source),
                    "--ultrafast",
                    "--ignore-cache",
                    "-lo",
                    "en",
                    "-o",
                    str(tmp_path),
                ]
            )
            == 0
        )
    assert translator.no_cache is True
    assert cross_page_overlap == [True]
    assert "16 workers" in caplog.text
    assert "[ultrafast perf] total:" in caplog.text
    for stage in (
        "layout_extract",
        "layout_mask",
        "paragraph_parse",
        "translation",
        "typesetting",
        "pdf_serialize",
    ):
        assert stage in caplog.text
    translated = [call.args[0] for call in translator.translate.call_args_list]
    assert any(
        "Original paragraph" in text and "continues" in text for text in translated
    )
    assert any("Second column" in text for text in translated)
    assert not any("Original" in text and "Second" in text for text in translated)
    with pymupdf.open(tmp_path / "native-mono.pdf") as doc:
        assert len(doc) == 3
        assert "Translated paragraph" in doc[0].get_text()
        assert doc[0].rotation == rotation
        assert "Translated paragraph" in doc[1].get_text()
        assert doc[2].get_text() == ""
        assert len(doc[2].get_images()) == 1
    with pymupdf.open(tmp_path / "native-dual.pdf") as doc:
        assert len(doc) == 6
        assert "Original paragraph" in doc[0].get_text()
        assert "Translated paragraph" in doc[1].get_text()


def test_no_cache_bypasses_reads_and_writes():
    from pdf2zh.translator import BaseTranslator

    translator = BaseTranslator("en", "zh", "test", False)
    translator.no_cache = True
    translator.cache = Mock()
    translator.cache.get.side_effect = AssertionError("cache read")
    translator.cache.set.side_effect = AssertionError("cache write")
    translator.do_translate = Mock(return_value="translated")
    assert translator.translate("source") == "translated"
    assert translator.translate("source") == "translated"
    assert translator.do_translate.call_count == 2
