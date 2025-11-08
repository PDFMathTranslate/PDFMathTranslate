from __future__ import annotations

import shutil
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import os

import pymupdf
import pymupdf4llm


PREFIX_CHARS = set("#>+-0123456789. \t")
WRAPPER_TOKENS = ("**", "__", "~~", "`", "_")


def export_markdown(
    doc: pymupdf.Document,
    output_dir: Path,
    base_name: str,
    *,
    reference_doc: Optional[pymupdf.Document] = None,
    write_images: bool = True,
    embed_images: bool = False,
    pages: Optional[list[int]] = None,
) -> Path:
    """
    Render the provided PyMuPDF document into Markdown via pymupdf4llm.

    Args:
        doc: Translated PyMuPDF document to render.
        output_dir: Destination directory for Markdown (and optional images).
        base_name: Base filename (without extension) used for the outputs.
        write_images: Whether to dump extracted images to disk.
        embed_images: Whether to embed images via data URIs instead of files.
        pages: Optional list of 0-based page indices to include.
    """

    if write_images and embed_images:
        raise ValueError("write_images and embed_images cannot both be True")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir: Optional[Path] = None
    image_path = output_dir

    assets_rel = None
    image_dir_token = None
    output_dir_abs = output_dir.resolve()
    assets_dir = None
    if write_images:
        assets_dir = output_dir / f"{base_name}_assets"
        assets_rel = Path(os.path.relpath(assets_dir.resolve(), output_dir_abs))
        image_dir_token = assets_dir.as_posix()
        assets_dir.mkdir(parents=True, exist_ok=True)
        image_path = assets_dir

    safe_pdf_name = f"{base_name.replace(' ', '-')}.pdf"

    markdown_text = pymupdf4llm.to_markdown(
        doc,
        filename=safe_pdf_name,
        write_images=write_images,
        embed_images=embed_images,
        image_path=str(image_path),
        pages=pages,
    )

    if reference_doc is not None:
        reference_assets_dir: Optional[Path] = None
        reference_image_path = image_path
        if write_images:
            reference_assets_dir = Path(tempfile.mkdtemp(prefix="pdf2zh-mdref-"))
            reference_image_path = reference_assets_dir

        reference_markdown = pymupdf4llm.to_markdown(
            reference_doc,
            filename=safe_pdf_name,
            write_images=write_images,
            embed_images=embed_images,
            image_path=str(reference_image_path),
            pages=pages,
        )

        markdown_text = _merge_markdown(reference_markdown, markdown_text)

        if reference_assets_dir is not None:
            shutil.rmtree(reference_assets_dir, ignore_errors=True)

    if write_images and assets_rel and image_dir_token:
        markdown_text = markdown_text.replace(
            image_dir_token,
            assets_rel.as_posix(),
        )

    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


def _merge_markdown(reference_text: str, translated_text: str) -> str:
    merged_lines = []
    ref_lines = reference_text.splitlines()
    trans_lines = translated_text.splitlines()

    for ref_line, trans_line in zip_longest(ref_lines, trans_lines, fillvalue=""):
        merged_lines.append(_merge_line(ref_line, trans_line))

    return "\n".join(merged_lines)


def _merge_line(reference_line: str, translated_line: str) -> str:
    if not reference_line:
        return translated_line
    if not translated_line:
        return reference_line

    stripped_translated = translated_line.strip()
    if stripped_translated.startswith("![]"):
        return translated_line

    prefix, remainder = _extract_prefix(reference_line)
    wrappers, _ = _extract_wrappers(remainder.strip())

    core = translated_line
    if prefix and core.startswith(prefix):
        core = core[len(prefix) :]

    leading_ws = len(core) - len(core.lstrip(" "))
    trailing_ws = len(core.rstrip(" ")) - len(core.strip(" "))
    inner = core.strip()

    if wrappers and inner:
        inner = _apply_wrappers(inner, wrappers)

    rebuilt_core = (" " * leading_ws) + inner + (" " * trailing_ws)
    return prefix + rebuilt_core


def _extract_prefix(line: str) -> tuple[str, str]:
    idx = 0
    while idx < len(line) and line[idx] in PREFIX_CHARS:
        idx += 1
    return line[:idx], line[idx:]


def _extract_wrappers(text: str) -> tuple[list[str], str]:
    wrappers: list[str] = []
    while text:
        for token in WRAPPER_TOKENS:
            if text.startswith(token) and text.endswith(token) and len(text) >= 2 * len(
                token
            ):
                text = text[len(token) : -len(token)]
                wrappers.append(token)
                break
        else:
            break
    return wrappers, text


def _apply_wrappers(content: str, wrappers: list[str]) -> str:
    result = content
    for token in reversed(wrappers):
        result = f"{token}{result}{token}"
    return result
