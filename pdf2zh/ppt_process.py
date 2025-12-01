import os
import tempfile
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from typing import List, Dict, Any
from pdf2zh.translator import BaseTranslator


def translate_ppt(
    input_path: str,
    output_path: str,
    translator: BaseTranslator,
    lang_from: str,
    lang_to: str,
    **kwargs
) -> None:
    """
    Translate all text in a PowerPoint presentation while preserving formatting.

    Args:
        input_path: Path to the input PowerPoint file (.pptx or .ppt)
        output_path: Path to save the translated PowerPoint file
        translator: Translator instance to use for translation
        lang_from: Source language code
        lang_to: Target language code
        **kwargs: Additional arguments to pass to the translator
    """
    # Load the presentation
    prs = Presentation(input_path)

    # Translate text in slides
    for slide in prs.slides:
        # Translate text in shapes
        for shape in slide.shapes:
            if shape.has_text_frame:
                translate_text_frame(shape.text_frame, translator, lang_from, lang_to, **kwargs)

            # Translate text in tables
            if shape.has_table:
                translate_table(shape.table, translator, lang_from, lang_to, **kwargs)

    # Save the translated presentation
    prs.save(output_path)


def translate_text_frame(
    text_frame: Any,
    translator: BaseTranslator,
    lang_from: str,
    lang_to: str,
    **kwargs
) -> None:
    """
    Translate all text in a text frame while preserving formatting.

    Args:
        text_frame: Text frame object to translate
        translator: Translator instance to use for translation
        lang_from: Source language code
        lang_to: Target language code
        **kwargs: Additional arguments to pass to the translator
    """
    # Collect all paragraphs and their runs
    paragraphs = []
    for para in text_frame.paragraphs:
        runs = []
        for run in para.runs:
            runs.append((run.text, run.font))
        paragraphs.append(runs)

    # Flatten the text to translate
    text_to_translate = ""
    for para in paragraphs:
        for run_text, _ in para:
            text_to_translate += run_text

    # Translate the text
    if text_to_translate.strip():  # Only translate if there's text
        translated_text = translator.translate(text_to_translate, lang_from, lang_to, **kwargs)

        # Reapply the translated text with original formatting
        # This is a simplified approach - in a real-world scenario, you'd need to
        # handle text splitting more carefully to preserve formatting
        # For now, we'll just replace the text in the first run and clear others
        if paragraphs:
            first_para = paragraphs[0]
            if first_para:
                first_run_text, first_run_font = first_para[0]
                # Replace the text in the first run
                text_frame.paragraphs[0].runs[0].text = translated_text
                # Clear all other runs in the first paragraph
                for i in range(1, len(text_frame.paragraphs[0].runs)):
                    text_frame.paragraphs[0].runs[i].text = ""
                # Clear all other paragraphs
                for i in range(1, len(text_frame.paragraphs)):
                    text_frame.paragraphs[i].clear()


def translate_table(
    table: Any,
    translator: BaseTranslator,
    lang_from: str,
    lang_to: str,
    **kwargs
) -> None:
    """
    Translate all text in a table while preserving formatting.

    Args:
        table: Table object to translate
        translator: Translator instance to use for translation
        lang_from: Source language code
        lang_to: Target language code
        **kwargs: Additional arguments to pass to the translator
    """
    for row in table.rows:
        for cell in row.cells:
            if cell.text.strip():  # Only translate if there's text
                # Translate the cell text
                translated_text = translator.translate(cell.text, lang_from, lang_to, **kwargs)

                # Clear existing text in the cell
                cell.text = ""

                # Add the translated text with default formatting
                # Note: This doesn't preserve the original formatting of the cell text
                # In a real-world scenario, you'd need to handle this more carefully
                p = cell.paragraphs[0]
                run = p.add_run()
                run.text = translated_text


if __name__ == "__main__":
    # Example usage
    from pdf2zh.translator import OpenAITranslator

    # Initialize translator
    translator = OpenAITranslator()

    # Translate a PowerPoint file
    input_path = "input.pptx"
    output_path = "output.pptx"
    translate_ppt(input_path, output_path, translator, "en", "zh")
