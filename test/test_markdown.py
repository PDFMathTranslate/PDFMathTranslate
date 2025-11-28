from pathlib import Path

import pymupdf

from pdf2zh.markdown import export_markdown, _FootnoteEntry


SOURCE_PDF = Path("test/file/translate.cli.plain.text.pdf")


def test_export_markdown_keep_mode(tmp_path, monkeypatch):
    source_pdf = SOURCE_PDF

    call_state = {"count": 0}

    def fake_render(doc, **kwargs):
        call_state["count"] += 1
        image_path = kwargs.get("image_path") or ""
        write_images = kwargs.get("write_images", False)
        filename = Path(kwargs.get("filename", "doc.pdf")).name
        image_rel = f"{image_path}/{filename}-0000-00.png" if image_path else "inline.png"
        if write_images and image_path:
            image_dir = Path(image_path)
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{filename}-0000-00.png").write_text("fake")
        if call_state["count"] == 1:
            footnotes = []
            if kwargs.get("collect_footnotes"):
                footnotes = [_FootnoteEntry(page_number=1, kind="footnote", markdown="> note\n\n")]
            return f"## Überschrift\n![]({image_rel})\ntranslated line\n", footnotes
        return f"## **Heading**\n![]({image_rel})\nplain line\n", []

    monkeypatch.setattr(
        "pdf2zh.markdown._render_markdown_document",
        fake_render,
    )

    doc = pymupdf.open(source_pdf)
    reference = pymupdf.open(source_pdf)
    try:
        md_path = export_markdown(
            doc,
            tmp_path,
            "plain text",
            reference_doc=reference,
            write_images=True,
        )
    finally:
        doc.close()
        reference.close()

    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "# Überschrift" in content

    assets_dir = tmp_path / "plain text_assets"
    assert assets_dir.exists()
    files = list(assets_dir.iterdir())
    assert files, "Expected fake image to be written"
    assert "plain text_assets/plain-text.pdf-0000-00.png" in content
    assert "### Footnotes" in content
    assert "note" in content


def test_export_markdown_drop_mode(tmp_path, monkeypatch):
    source_pdf = SOURCE_PDF

    def fake_render(doc, **kwargs):
        return (
            "## Heading\ntranslated line\n",
            [
                _FootnoteEntry(
                    page_number=2,
                    kind="page-footer",
                    markdown="Permission to make digital copies",
                )
            ],
        )

    monkeypatch.setattr(
        "pdf2zh.markdown._render_markdown_document",
        fake_render,
    )

    doc = pymupdf.open(source_pdf)
    try:
        md_path = export_markdown(
            doc,
            tmp_path,
            "drop-mode",
            markdown_footnotes="drop",
            write_images=False,
        )
    finally:
        doc.close()

    content = md_path.read_text(encoding="utf-8")
    assert "### Footnotes" not in content
    assert "Permission to make digital copies" not in content
