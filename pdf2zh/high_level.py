"""Functions that can be used for the most common use-cases for pdf2zh.six"""

import asyncio
import io
import json
import os
import re
import sys
import tempfile
import logging
import hashlib
from datetime import datetime
from asyncio import CancelledError
from pathlib import Path
from string import Template
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import requests
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font, Rect

from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import OnnxModel
from pdf2zh.pdfinterp import PDFPageInterpreterEx

from pdf2zh.config import ConfigManager
from babeldoc.assets.assets import get_font_and_metadata

NOTO_NAME = "noto"

logger = logging.getLogger(__name__)

noto_list = [
    "am",  # Amharic
    "ar",  # Arabic
    "bn",  # Bengali
    "bg",  # Bulgarian
    "chr",  # Cherokee
    "el",  # Greek
    "gu",  # Gujarati
    "iw",  # Hebrew
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ru",  # Russian
    "sr",  # Serbian
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "ur",  # Urdu
    "uk",  # Ukrainian
]


def check_files(files: List[str]) -> List[str]:
    files = [
        f for f in files if not f.startswith("http://")
    ]  # exclude online files, http
    files = [
        f for f in files if not f.startswith("https://")
    ]  # exclude online files, https
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files


def _sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-") or "unknown"


def _pages_signature(pages: Optional[list[int]]) -> str:
    if not pages:
        return "all"
    selected = sorted({p for p in pages if p >= 0})
    if not selected:
        return "all"
    # Convert 0-based to 1-based and compact ranges.
    one_based = [p + 1 for p in selected]
    ranges = []
    start = prev = one_based[0]
    for current in one_based[1:]:
        if current == prev + 1:
            prev = current
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = current
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return "_".join(ranges)


def _content_fingerprint(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(
        path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    )


def _load_translation_map(translation_file: str) -> Dict[str, str]:
    path = Path(translation_file)
    translation_map: Dict[str, str] = {}
    if path.is_dir():
        pages_dir = path / "pages" if (path / "pages").is_dir() else path
        for page_file in sorted(pages_dir.glob("*.json")):
            with open(page_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                continue
            if isinstance(loaded.get("translations"), dict):
                translation_map.update(loaded["translations"])
            elif all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in loaded.items()
            ):
                translation_map.update(loaded)
        if not translation_map:
            raise ValueError(
                "translation_file directory must contain JSON with translations."
            )
        return translation_map

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict) and isinstance(loaded.get("translations"), dict):
        return loaded["translations"]
    if isinstance(loaded, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
    ):
        return loaded
    raise ValueError(
        "translation_file must be a JSON object, {'translations': {...}}, or a pages directory."
    )


def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    ignore_cache: bool = False,
    source_data_path: str = None,
    translated_data_path: str = None,
    page_artifacts_dir: str = None,
    translation_file: str = None,
    **kwarg: Any,
) -> None:
    translation_map: Dict[str, str] = {}
    if translation_file:
        translation_map = _load_translation_map(translation_file)

    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
        ignore_cache,
        translation_map,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 新建一个 xref 存放新指令流
            page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    if source_data_path:
        source_data = {
            "lang_in": lang_in,
            "lang_out": lang_out,
            "pages": device.page_data,
        }
        _atomic_write_json(Path(source_data_path), source_data)
        logger.info(f"Source data saved to: {source_data_path}")

    # Batch mode: translate all collected texts and typeset deferred pages
    if device.batch_mode:
        device.flush_batch(obj_patch)

    device.close()

    if translated_data_path:
        translated_data = {"translations": device.translations}
        _atomic_write_json(Path(translated_data_path), translated_data)
        logger.info(f"Translated data saved to: {translated_data_path}")

    if page_artifacts_dir:
        page_dir = Path(page_artifacts_dir)
        page_dir.mkdir(parents=True, exist_ok=True)
        for page in device.page_data:
            page_number = int(page.get("page_number", 0))
            page_payload = {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "page_number": page_number,
                "paragraphs": page.get("paragraphs", []),
                "formulas": page.get("formulas", []),
                "translations": {},
            }
            for paragraph in page_payload["paragraphs"]:
                source_text = paragraph.get("source")
                if source_text is None:
                    continue
                if source_text in device.translations:
                    page_payload["translations"][source_text] = device.translations[
                        source_text
                    ]
            page_file = page_dir / f"{page_number + 1:04d}.json"
            _atomic_write_json(page_file, page_payload)
        logger.info(f"Page artifacts saved to: {page_artifacts_dir}")

    return obj_patch


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    source_data_path: str = None,
    translated_data_path: str = None,
    page_artifacts_dir: str = None,
    translation_file: str = None,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    def build_side_by_side_doc(
        source_doc: Document, translated_doc: Document, page_indexes: list[int]
    ) -> Document:
        dual_doc = Document()
        for p in page_indexes:
            source_rect = source_doc[p].rect
            translated_rect = translated_doc[p].rect
            left_width = float(source_rect.width)
            right_width = float(translated_rect.width)
            page_height = float(max(source_rect.height, translated_rect.height))

            dual_page = dual_doc.new_page(
                width=left_width + right_width, height=page_height
            )
            dual_page.show_pdf_page(
                Rect(0, 0, left_width, page_height), source_doc, p, keep_proportion=True
            )
            dual_page.show_pdf_page(
                Rect(left_width, 0, left_width + right_width, page_height),
                translated_doc,
                p,
                keep_proportion=True,
            )
        return dual_doc

    if pages:
        selected_pages = sorted({p for p in pages if 0 <= p < page_count})
        if not selected_pages:
            raise PDFValueError("No valid pages selected.")
        mono_doc = Document()
        for p in selected_pages:
            mono_doc.insert_pdf(doc_zh, from_page=p, to_page=p)
        dual_doc = build_side_by_side_doc(doc_en, doc_zh, selected_pages)
        if not skip_subset_fonts:
            mono_doc.subset_fonts(fallback=True)
            dual_doc.subset_fonts(fallback=True)
        # Avoid aggressive xref cleanup/object-stream rewrite because some PDFs
        # with broken references can hang in MuPDF's writer.
        return (
            mono_doc.write(deflate=True, garbage=0, use_objstms=0),
            dual_doc.write(deflate=True, garbage=0, use_objstms=0),
        )

    dual_doc = build_side_by_side_doc(doc_en, doc_zh, list(range(page_count)))
    if not skip_subset_fonts:
        doc_zh.subset_fonts(fallback=True)
        dual_doc.subset_fonts(fallback=True)
    # Avoid aggressive xref cleanup/object-stream rewrite because some PDFs
    # with broken references can hang in MuPDF's writer.
    return (
        doc_zh.write(deflate=True, garbage=0, use_objstms=0),
        dual_doc.write(deflate=True, garbage=0, use_objstms=0),
    )


def convert_to_pdfa(input_path, output_path):
    """
    Convert PDF to PDF/A format

    Args:
        input_path: Path to source PDF file
        output_path: Path to save PDF/A file
    """
    from pikepdf import Dictionary, Name, Pdf

    # Open the PDF file
    pdf = Pdf.open(input_path)

    # Add PDF/A conformance metadata
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # Create OutputIntent dictionary
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    # Add output intent to PDF root
    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    # Save as PDF/A
    pdf.save(output_path, linearize=True)
    pdf.close()


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    translation_file: str = None,
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = check_files(files)

    if missing_files:
        print("The following files do not exist:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("Some files do not exist.")

    output_dir = Path(output) if output else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = []

    for file in files:
        if type(file) is str and (
            file.startswith("http://") or file.startswith("https://")
        ):
            print("Online files detected, downloading...")
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        print(f"Writing the file: {file}...")
                        tmp_file.write(r.content)
                        file = tmp_file.name
                else:
                    r.raise_for_status()
            except Exception as e:
                raise PDFValueError(
                    f"Errors occur in downloading the PDF file. Please check the link(s).\nError:\n{e}"
                )
        filename = os.path.splitext(os.path.basename(file))[0]

        # If the commandline has specified converting to PDF/A format
        # --compatible / -cp
        if compatible:
            with tempfile.NamedTemporaryFile(
                suffix="-pdfa.pdf", delete=False
            ) as tmp_pdfa:
                print(f"Converting {file} to PDF/A format...")
                convert_to_pdfa(file, tmp_pdfa.name)
                doc_raw = open(tmp_pdfa.name, "rb")
                os.unlink(tmp_pdfa.name)
        else:
            doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        temp_dir = Path(tempfile.gettempdir())
        file_path = Path(file)
        try:
            if file_path.exists() and file_path.resolve().is_relative_to(
                temp_dir.resolve()
            ):
                file_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {file_path}", exc_info=True)

        doc_id = f"{_sanitize_slug(filename)}-{_content_fingerprint(s_raw)}"
        pages_sig = _pages_signature(pages)
        service_slug = _sanitize_slug(service)
        lang_pair = f"{_sanitize_slug(lang_in)}-{_sanitize_slug(lang_out)}"
        run_id = f"p{pages_sig}__{lang_pair}__{service_slug}"
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        artifacts_dir = output_dir / "artifacts" / doc_id
        json_root_dir = artifacts_dir / "json"
        pdf_root_dir = artifacts_dir / "pdf"
        run_name = f"{run_id}__{run_stamp}"
        run_dir = json_root_dir / run_name
        page_artifacts_dir = str(run_dir / "pages")
        source_data_path = None
        translated_data_path = None
        s_mono, s_dual = translate_stream(
            s_raw,
            **locals(),
        )
        run_file_base = f"{filename}__{run_id}__{run_stamp}"
        file_mono = pdf_root_dir / f"{run_file_base}-mono.pdf"
        file_dual = pdf_root_dir / f"{run_file_base}-dual.pdf"
        _atomic_write_bytes(file_mono, s_mono)
        _atomic_write_bytes(file_dual, s_dual)
        manifest = {
            "schema_version": 1,
            "doc_id": doc_id,
            "run_id": run_name,
            "filename": filename,
            "pages_signature": pages_sig,
            "selected_pages_1based": (
                [p + 1 for p in sorted({p for p in pages if p >= 0})] if pages else None
            ),
            "lang_in": lang_in,
            "lang_out": lang_out,
            "service": service,
            "created_at": datetime.strptime(run_stamp, "%Y%m%d-%H%M%S-%f").isoformat(timespec="seconds"),
            "outputs": {
                "mono_pdf": f"../pdf/{file_mono.name}",
                "dual_pdf": f"../pdf/{file_dual.name}",
            },
            "artifacts": {
                "pages_dir": "pages",
            },
        }
        _atomic_write_json(run_dir / "manifest.json", manifest)
        result_files.append((str(file_mono), str(file_dual)))

    return result_files


def download_remote_fonts(lang: str):
    lang = lang.lower()
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")

    # docker
    font_path = ConfigManager.get("NOTO_FONT_PATH", Path("/app", font_name).as_posix())
    if not Path(font_path).exists():
        font_path, _ = get_font_and_metadata(font_name)
        font_path = font_path.as_posix()

    logger.info(f"use font: {font_path}")

    return font_path
