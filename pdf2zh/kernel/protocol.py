"""Kernel protocol — unified interface for translation kernels."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class TranslateRequest:
    """Unified translation request bridging CLI args to kernel-specific config."""

    files: list[str]
    lang_in: str = "en"
    lang_out: str = "zh"
    service: str = "google"
    pages: Optional[list[int] | str] = None
    output: str = ""
    thread: Optional[int] = None
    vfont: str = ""
    vchar: str = ""
    prompt: Optional[str] = None
    envs: Optional[dict] = field(default_factory=dict)
    debug: bool = False
    skip_subset_fonts: bool = False
    ignore_cache: bool = False
    compatible: bool = False
    ultrafast: bool = False

    def __post_init__(self):
        if self.thread is None:
            self.thread = 16 if self.ultrafast else 4


@dataclass
class TranslateResult:
    """Unified result from either kernel."""

    mono_pdf: Optional[Path | bytes] = None
    dual_pdf: Optional[Path | bytes] = None
    time_cost: float = 0.0


@runtime_checkable
class KernelProtocol(Protocol):
    """What every kernel must implement."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def translate(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]: ...

    async def translate_async(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]: ...

    def is_available(self) -> bool: ...
