import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from pptx import Presentation
from pptx.util import Pt
from pdf2zh.translator import BaseTranslator


def translate_ppt(file_path: str, translator: BaseTranslator, output_dir: str) -> str:
    """
    Translate the text content of a PowerPoint presentation while preserving formatting.

    Args:
        file_path: Path to the input PowerPoint file
        translator: Translator instance to use for translation
        output_dir: Directory to save the translated PowerPoint file

    Returns:
        Path to the translated PowerPoint file
    """
    # Load the presentation
    prs = Presentation(file_path)

    # Iterate through all slides
    for slide in prs.slides:
        # Iterate through all shapes in the slide
        for shape in slide.shapes:
            # Check if the shape has text frame
            if hasattr(shape, 'text_frame') and shape.text_frame is not None:
                # Iterate through all paragraphs in the text frame
                for paragraph in shape.text_frame.paragraphs:
                    # Iterate through all runs in the paragraph
                    for run in paragraph.runs:
                        # Translate the text
                        if run.text.strip():
                            translated_text = translator.translate(run.text)
                            # Replace the text while preserving formatting
                            run.text = translated_text

    # Save the translated presentation
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_translated.pptx")
    prs.save(output_path)

    return output_path


def process_ppt_file(
    file_path: str,
    translator: BaseTranslator,
    output_dir: str
) -> str:
    """
    Process a PowerPoint file: translate its content and save the result.

    Args:
        file_path: Path to the input PowerPoint file
        translator: Translator instance to use for translation
        output_dir: Directory to save the translated file

    Returns:
        Path to the translated PowerPoint file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Translate the PPT
    translated_path = translate_ppt(file_path, translator, output_dir)

    return translated_path
