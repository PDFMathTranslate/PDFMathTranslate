from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import re

import pymupdf
from pymupdf4llm import parse_document
from pymupdf4llm.helpers import document_layout as doc_layout

PREFIX_CHARS = set("#>+-0123456789. \t")
WRAPPER_TOKENS = ("**", "__", "~~", "`", "_")
PLACEHOLDER_PATTERNS = (
    re.compile(r"^\*\*==>"),
    re.compile(r"^\*\*figure", re.IGNORECASE),
    re.compile(r"^\*\*table", re.IGNORECASE),
)


FOOTNOTE_INLINE = "inline"
FOOTNOTE_APPEND = "append"
FOOTNOTE_DROP = "drop"
FOOTNOTE_MODES = {FOOTNOTE_INLINE, FOOTNOTE_APPEND, FOOTNOTE_DROP}


@dataclass
class _FootnoteEntry:
    page_number: int
    kind: str
    markdown: str


def _render_markdown_document(
    doc: pymupdf.Document,
    *,
    filename: str,
    image_path: str,
    pages: Optional[list[int]],
    write_images: bool,
    embed_images: bool,
    footnote_mode: str,
    collect_footnotes: bool,
):
    if footnote_mode not in FOOTNOTE_MODES:
        raise ValueError(
            f"Invalid markdown footnote mode '{footnote_mode}'."
        )
    parsed_doc = parse_document(
        doc,
        filename=filename,
        image_path=image_path,
        pages=pages,
    )
    collected: list[_FootnoteEntry] = []
    if footnote_mode != FOOTNOTE_INLINE:
        collected = _extract_structural_footnotes(
            parsed_doc,
            footnote_mode,
            collect_footnotes,
        )
    markdown_text = parsed_doc.to_markdown(
        header=True,
        footer=(footnote_mode == FOOTNOTE_INLINE),
        write_images=write_images,
        embed_images=embed_images,
    )
    return markdown_text, collected


def _extract_structural_footnotes(
    parsed_doc,
    mode: str,
    capture: bool,
) -> list[_FootnoteEntry]:
    collected: list[_FootnoteEntry] = []
    pages = getattr(parsed_doc, "pages", []) or []
    for page in pages:
        boxes = list(getattr(page, "boxes", []) or [])
        filtered: list = []
        for box in boxes:
            kind = getattr(box, "boxclass", "")
            if kind in {"footnote", "page-footer"}:
                if capture and mode == FOOTNOTE_APPEND:
                    markdown = _box_to_markdown(kind, box)
                    if markdown.strip():
                        collected.append(
                            _FootnoteEntry(
                                page_number=getattr(page, "page_number", 0),
                                kind=kind,
                                markdown=markdown.strip(),
                            )
                        )
                continue
            filtered.append(box)
        page.boxes = filtered
    return collected


def _box_to_markdown(kind: str, box) -> str:
    textlines = getattr(box, "textlines", None)
    if not textlines:
        return ""
    if kind == "footnote":
        return doc_layout.footnote_to_md(textlines)
    return doc_layout.text_to_md(textlines)


def _format_footnote_section(entries: list[_FootnoteEntry]) -> str:
    if not entries:
        return ""
    lines = ["### Footnotes", ""]
    for entry in entries:
        kind_label = entry.kind.replace("-", " ").title()
        lines.append(f"**Page {entry.page_number} · {kind_label}**")
        lines.append("")
        lines.append(entry.markdown.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def export_markdown(
    doc: pymupdf.Document,
    output_dir: Path,
    base_name: str,
    *,
    reference_doc: Optional[pymupdf.Document] = None,
    write_images: bool = True,
    embed_images: bool = False,
    pages: Optional[list[int]] = None,
    markdown_footnotes: str = FOOTNOTE_APPEND,
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
        markdown_footnotes: Controls footnote placement: "inline", "append", or "drop".
    """

    if write_images and embed_images:
        raise ValueError("write_images and embed_images cannot both be True")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_rel: Optional[Path] = None
    image_dir_token: Optional[str] = None

    if write_images:
        assets_dir = output_dir / f"{base_name}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_dir_abs = output_dir.resolve()
        assets_rel = Path(os.path.relpath(assets_dir.resolve(), output_dir_abs))
        image_dir_token = assets_dir.as_posix()

    safe_pdf_name = f"{base_name.replace(' ', '-')}.pdf"

    translated_markdown, collected = _render_markdown_document(
        doc,
        filename=safe_pdf_name,
        image_path=image_dir_token or "",
        pages=pages,
        write_images=write_images,
        embed_images=embed_images,
        footnote_mode=markdown_footnotes,
        collect_footnotes=True,
    )

    if reference_doc is not None:
        # Reference Markdown is used purely for styling cues, so image extraction is unnecessary.
        reference_markdown, _ = _render_markdown_document(
            reference_doc,
            filename=safe_pdf_name,
            image_path="",
            pages=pages,
            write_images=False,
            embed_images=False,
            footnote_mode=markdown_footnotes,
            collect_footnotes=False,
        )

        markdown_text = _merge_markdown(reference_markdown, translated_markdown)
    else:
        markdown_text = translated_markdown

    if write_images and assets_rel and image_dir_token:
        markdown_text = markdown_text.replace(
            image_dir_token,
            assets_rel.as_posix(),
        )
        markdown_text = _rewrite_image_paths(markdown_text, assets_rel)

    markdown_text = _promote_primary_heading(markdown_text)

    if markdown_footnotes == FOOTNOTE_APPEND and collected:
        markdown_text = markdown_text.rstrip() + "\n\n" + _format_footnote_section(collected)

    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


def _merge_markdown(reference_text: str, translated_text: str) -> str:
    """Combine reference Markdown (with headings/styles) and translated Markdown line-by-line."""
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
        stripped_trans = trans_line.strip()

        if _is_placeholder(stripped_ref):
            merged_lines.append(trans_line)
            ref_idx += 1
            if trans_idx < trans_len:
                trans_idx += 1
            continue

        if ref_idx >= ref_len:
            merged_lines.append(trans_line)
            trans_idx += 1
            continue
        if trans_idx >= trans_len:
            break

        ref_blank = is_blank(ref_line)
        trans_blank = is_blank(trans_line)
        ref_header = stripped_ref.startswith("#")
        trans_header = stripped_trans.startswith("#")

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
            if trans_blank:
                merged_lines.append(_apply_reference_header(ref_line, trans_line))
                trans_idx += 1
            else:
                merged_lines.append(trans_line)
                trans_idx += 1
            ref_idx += 1
            continue
        if trans_header and not ref_header:
            merged_lines.append(trans_line)
            ref_idx += 1
            trans_idx += 1
            continue

        merged_lines.append(_merge_line(ref_line, trans_line))
        ref_idx += 1
        trans_idx += 1

    return "\n".join(merged_lines)


def _merge_line(reference_line: str, translated_line: str) -> str:
    """Merge a single line from reference/translated Markdown, preserving wrappers where appropriate."""
    if not reference_line:
        return translated_line
    if not translated_line:
        return reference_line

    stripped_translated = translated_line.strip()
    if stripped_translated.startswith("![]"):
        return translated_line

    ref_prefix, ref_body = _extract_prefix(reference_line)
    trans_prefix, trans_body = _extract_prefix(translated_line)

    prefix = trans_prefix if trans_prefix.strip() else ""
    if not prefix:
        ref_prefix_clean = ref_prefix.strip()
        if ref_prefix_clean and not ref_prefix_clean.startswith("#"):
            prefix = ref_prefix

    ref_wrappers = []
    if not prefix.strip().startswith("#") and not ref_prefix.strip().startswith("#"):
        ref_wrappers, _ = _extract_wrappers(ref_body.strip())

    core = trans_body
    leading_ws = len(core) - len(core.lstrip(" "))
    trailing_ws = len(core) - len(core.rstrip(" "))
    inner = core.strip()

    existing_wrappers, _ = _extract_wrappers(trans_body.strip())

    if (
        ref_wrappers
        and inner
        and not existing_wrappers
        and not prefix.strip().startswith("#")
    ):
        inner = _apply_wrappers(inner, ref_wrappers)

    rebuilt_core = (" " * leading_ws) + inner + (" " * trailing_ws)
    return prefix + rebuilt_core


def _extract_prefix(line: str) -> tuple[str, str]:
    """Return (prefix, body) where prefix contains heading/list markers."""
    idx = 0
    while idx < len(line) and line[idx] in PREFIX_CHARS:
        idx += 1
    return line[:idx], line[idx:]


def _extract_wrappers(text: str) -> tuple[list[str], str]:
    """Peel matching wrappers (**, __, ~~ …) from both ends of the text."""
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
    """Reapply wrappers to content in their original nesting order."""
    result = content
    for token in reversed(wrappers):
        result = f"{token}{result}{token}"
    return result


def _rewrite_image_paths(markdown_text: str, assets_rel: Path | None) -> str:
    """Repoint image references to the generated assets directory."""
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


def _apply_reference_header(reference_line: str, translated_line: str) -> str:
    """Apply the reference line's heading prefix to the translated line."""
    prefix, _ = _extract_prefix(reference_line)
    prefix = prefix.strip()
    if not prefix:
        prefix = "##"
    leading_ws = len(translated_line) - len(translated_line.lstrip(" "))
    core = translated_line.strip()
    if not core:
        core = reference_line.strip("# ").strip()
    else:
        core = core.lstrip("# ").strip()
    return (" " * leading_ws) + prefix + " " + core


def _promote_primary_heading(markdown_text: str) -> str:
    """Ensure the first sub-heading is promoted to a top-level # heading."""
    lines = markdown_text.splitlines()
    if any(line.lstrip().startswith("# ") for line in lines):
        return markdown_text

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("## "):
            leading = len(line) - len(stripped)
            lines[idx] = (" " * leading) + "# " + stripped[3:]
            break
    return "\n".join(lines)


def _is_placeholder(line: str) -> bool:
    """Return True if line matches a known placeholder pattern."""
    normalized = line.strip()
    if not normalized:
        return False
    return any(pattern.match(normalized) for pattern in PLACEHOLDER_PATTERNS)
