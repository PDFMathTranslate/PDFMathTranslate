"""Fast kernel adapter — wraps existing pdf2zh.high_level.translate()."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from pdf2zh.kernel.protocol import TranslateRequest, TranslateResult

logger = logging.getLogger(__name__)


class LegacyKernel:
    """Kernel adapter for the original pdf2zh translation pipeline (fast mode)."""

    @property
    def name(self) -> str:
        return "fast"

    @property
    def version(self) -> str:
        from pdf2zh import __version__

        return __version__

    def is_available(self) -> bool:
        return True

    def translate(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]:
        from pdf2zh.doclayout import ModelInstance, OnnxModel
        from pdf2zh.high_level import translate

        # Ensure model is loaded
        if ModelInstance.value is None:
            ModelInstance.value = OnnxModel.load_available()

        # Build kwargs matching high_level.translate() signature
        kwargs: dict[str, Any] = {
            "files": request.files,
            "output": request.output,
            "lang_in": request.lang_in,
            "lang_out": request.lang_out,
            "service": request.service,
            "thread": request.thread,
            "vfont": request.vfont,
            "vchar": request.vchar,
            "callback": callback,
            "cancellation_event": cancellation_event,
            "model": ModelInstance.value,
            "envs": request.envs or {},
            "skip_subset_fonts": request.skip_subset_fonts,
            "ignore_cache": request.ignore_cache,
            "compatible": request.compatible,
        }

        if request.pages and isinstance(request.pages, list):
            kwargs["pages"] = request.pages

        if request.prompt:
            from string import Template

            kwargs["prompt"] = Template(request.prompt)

        if request.source_dir:
            kwargs["source_dir"] = request.source_dir

        result_files = translate(**kwargs)

        results = []
        for mono_path, dual_path in result_files:
            results.append(
                TranslateResult(
                    mono_pdf=Path(mono_path),
                    dual_pdf=Path(dual_path),
                )
            )
        return results

    async def translate_async(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]:
        return await asyncio.to_thread(
            self.translate, request, callback, cancellation_event
        )
