from pathlib import Path

import pymupdf

from pdf2zh.markdown import export_markdown


def test_export_markdown(tmp_path):
    source_pdf = Path("test/file/translate.cli.plain.text.pdf")
    doc = pymupdf.open(source_pdf)
    try:
        md_path = export_markdown(
            doc,
            tmp_path,
            "plain-text",
            write_images=False,
        )
    finally:
        doc.close()

    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert content.strip(), "Markdown output should not be empty"
