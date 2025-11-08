from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import os
import re

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers.pymupdf_rag import IdentifyHeaders


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

    translated_header_detector = CleanHeaderDetector(doc, pages=pages)

    markdown_text = pymupdf4llm.to_markdown(
        doc,
        filename=safe_pdf_name,
        hdr_info=translated_header_detector,
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

        reference_header_detector = CleanHeaderDetector(reference_doc, pages=pages)
        reference_markdown = pymupdf4llm.to_markdown(
            reference_doc,
            filename=safe_pdf_name,
            hdr_info=reference_header_detector,
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
        markdown_text = _rewrite_image_paths(markdown_text, assets_rel)

    markdown_text = _promote_primary_heading(markdown_text)

    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


def _merge_markdown(reference_text: str, translated_text: str) -> str:
    merged_lines: list[str] = []
    ref_lines = reference_text.splitlines()
    trans_lines = translated_text.splitlines()
    ref_len, trans_len = len(ref_lines), len(trans_lines)
    ref_idx = trans_idx = 0

    def is_blank(value: str) -> bool:
        return not value.strip()

    while ref_idx < ref_len or trans_idx < trans_len:
        ref_line = ref_lines[ref_idx] if ref_idx < ref_len else ""
        trans_line = trans_lines[trans_idx] if trans_idx < trans_len else ""

        stripped_ref = ref_line.strip()
        if stripped_ref.startswith("**==>") or stripped_ref.startswith("**Figure"):
            ref_idx += 1
            continue

        if ref_idx >= ref_len:
            merged_lines.append(trans_line)
            trans_idx += 1
            continue
        if trans_idx >= trans_len:
            merged_lines.append(ref_line)
            ref_idx += 1
            continue

        ref_blank = not stripped_ref
        trans_blank = is_blank(trans_line)
        ref_header = stripped_ref.startswith("#")
        trans_header = trans_line.lstrip().startswith("#")

        if ref_blank and trans_blank:
            merged_lines.append(trans_line)
            ref_idx += 1
            trans_idx += 1
            continue
        if ref_blank:
            ref_idx += 1
            continue
        if trans_blank:
            merged_lines.append(trans_line)
            trans_idx += 1
            continue
        if ref_header and not trans_header:
            ref_idx += 1
            continue
        if trans_header and not ref_header:
            merged_lines.append(trans_line)
            trans_idx += 1
            continue

        merged_lines.append(_merge_line(ref_line, trans_line))
        ref_idx += 1
        trans_idx += 1

    return "\n".join(merged_lines)


def _merge_line(reference_line: str, translated_line: str) -> str:
    if not reference_line:
        return translated_line
    if not translated_line:
        return reference_line

    stripped_translated = translated_line.strip()
    if stripped_translated.startswith("![]"):
        return translated_line

    ref_prefix, ref_body = _extract_prefix(reference_line)
    ref_wrappers, _ = _extract_wrappers(ref_body.strip())

    trans_prefix, trans_body = _extract_prefix(translated_line)
    prefix = trans_prefix if trans_prefix.strip() else ""
    if not prefix:
        ref_prefix_clean = ref_prefix.strip()
        if ref_prefix_clean and not ref_prefix_clean.startswith("#"):
            prefix = ref_prefix

    if prefix.strip().startswith("#"):
        ref_wrappers = []

    core = trans_body
    leading_ws = len(core) - len(core.lstrip(" "))
    trailing_ws = len(core.rstrip(" ")) - len(core.strip(" "))
    inner = core.strip()

    existing_wrappers, _ = _extract_wrappers(trans_body.strip())

    if ref_wrappers and inner and not existing_wrappers:
        inner = _apply_wrappers(inner, ref_wrappers)

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


def _rewrite_image_paths(markdown_text: str, assets_rel: Path | None) -> str:
    if not assets_rel:
        return markdown_text

    rel_prefix = assets_rel.as_posix()

    def replace(match: re.Match) -> str:
        original = match.group(1)
        if original.startswith(rel_prefix):
            return match.group(0)
        basename = os.path.basename(original)
        if not basename:
            return match.group(0)
        return f"![]({rel_prefix}/{basename})"

    return re.sub(r"!\[]\(([^)]+)\)", replace, markdown_text)


def _promote_primary_heading(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("## "):
            leading = len(line) - len(stripped)
            lines[idx] = (" " * leading) + "#" + stripped[2:]
            break
    return "\n".join(lines)


class CleanHeaderDetector:
    def __init__(self, doc: pymupdf.Document, pages: Optional[list[int]] = None):
        self._doc = doc
        self._pages = pages
        self._delegate = IdentifyHeaders(doc, pages=pages)
        self._prune_placeholder_sizes()

    def _prune_placeholder_sizes(self):
        removable = []
        for size in sorted(self._delegate.header_id.keys(), reverse=True):
            if self._size_is_placeholder_only(size):
                removable.append(size)
        for size in removable:
            self._delegate.header_id.pop(size, None)
        if self._delegate.header_id:
            sizes = sorted(self._delegate.header_id.keys(), reverse=True)
            self._delegate.header_id = {
                size: "#" * i + " " for i, size in enumerate(sizes, start=1)
            }
            self._delegate.body_limit = min(sizes) - 1

    def _size_is_placeholder_only(self, target_size: float) -> bool:
        doc = self._doc
        owns_doc = False
        if not isinstance(doc, pymupdf.Document):
            doc = pymupdf.open(doc)
            owns_doc = True
        try:
            pages = self._pages or range(doc.page_count)
            for pno in pages:
                page = doc.load_page(pno)
                blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
                for line in [
                    s
                    for b in blocks
                    for l in b["lines"]
                    for s in l["spans"]
                    if round(s["size"]) == target_size
                ]:
                    text = (line.get("text") or "").strip()
                    normalized = text.lstrip("* ").strip().lower()
                    if not (
                        normalized.startswith("==>")
                        or normalized.startswith("figure ")
                        or normalized.startswith("arxiv:")
                    ):
                        return False
            return True
        finally:
            if owns_doc:
                doc.close()

    def get_header_id(self, span: dict, page=None) -> str:
        return self._delegate.get_header_id(span, page=page)
