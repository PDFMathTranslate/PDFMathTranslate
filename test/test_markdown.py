from pathlib import Path

import pymupdf

from pdf2zh.markdown import export_markdown


def test_export_markdown(tmp_path, monkeypatch):
    source_pdf = Path("test/file/translate.cli.plain.text.pdf")

    call_state = {"count": 0}

    def fake_to_markdown(doc, **kwargs):
        call_state["count"] += 1
        image_path = Path(kwargs["image_path"])
        image_path.mkdir(parents=True, exist_ok=True)
        filename = Path(kwargs["filename"]).name
        image_rel = f"{image_path.as_posix()}/{filename}-0000-00.png"
        (image_path / f"{filename}-0000-00.png").write_text("fake")
        if call_state["count"] == 1:
            return f"## Überschrift\n![]({image_rel})\ntranslated line\n"
        return f"## **Heading**\n![]({image_rel})\nplain line\n"

    monkeypatch.setattr("pdf2zh.markdown.pymupdf4llm.to_markdown", fake_to_markdown)

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
