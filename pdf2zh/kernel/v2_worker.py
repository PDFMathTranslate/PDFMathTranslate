#!/usr/bin/env python3
"""Subprocess worker — runs pdf2zh_next translation in an isolated venv.

Protocol:
  - stdin:  JSON array of CLI args (e.g. ["file.pdf", "--lang-out", "zh", "--siliconflowfree"])
  - stdout: JSON result (last line, after all progress events)
  - stderr: JSON-lines progress events and log output

This script is executed by PreciseKernel using the venv's Python interpreter.
v2's ConfigManager handles all config parsing from sys.argv + PDF2ZH_* env vars.

We call babeldoc directly (via create_babeldoc_config) instead of
do_translate_async_stream, because the latter now spawns a multiprocessing
subprocess internally — running that inside *this* subprocess causes
deadlocks and fd-inheritance issues on macOS/spawn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

_GOOGLE_TRANSLATE_CONNECT_TIMEOUT = 10
_GOOGLE_TRANSLATE_READ_TIMEOUT = 30


def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library log output doesn't pollute JSON results.

    We save the real stdout fd for writing the final JSON result.
    """
    real_stdout_fd = os.dup(1)  # save fd 1
    os.dup2(2, 1)  # point fd 1 → stderr
    return os.fdopen(real_stdout_fd, "w")


# Redirect before any pdf2zh_next imports (they configure logging on import)
_real_stdout = _redirect_stdout_to_stderr()

def _disable_translator_health_check():
    """Skip the blocking translate('Hello') health check in get_translator().

    The submodule's _create_translator_instance calls translator.translate("Hello")
    as a health check.  For SiliconFlowFree this hits the real translation proxy
    endpoint, which can hang for minutes (60s timeout × 3 retries + backoff).
    The translator's __init__ already validates the proxy via /check and /config,
    so this redundant health check just adds a failure mode with terrible UX.
    """
    import pdf2zh_next.translator.utils as _tu
    from pdf2zh_next.translator.base_translator import BaseTranslator

    _orig = _tu._create_translator_instance

    def _patched(settings, translator_config, rate_limiter, enforce_glossary_support=True):
        _orig_translate = BaseTranslator.translate
        BaseTranslator.translate = lambda self, *a, **kw: "ok"
        try:
            return _orig(settings, translator_config, rate_limiter, enforce_glossary_support)
        finally:
            BaseTranslator.translate = _orig_translate

    _tu._create_translator_instance = _patched


def _patch_google_request_timeout():
    """Bound precise Google translator requests so stalls fail instead of hanging.

    pdf2zh_next's Google translator uses requests.Session.get() without a timeout.
    When Google stops responding mid-request, the worker can sit forever inside
    Translate Paragraphs and the API/WebUI never see a terminal state.
    """
    import requests.sessions

    original_get = requests.sessions.Session.get
    if getattr(original_get, "__pdf2zh_google_timeout_patch__", False):
        return

    def _patched_get(self, url, *args, **kwargs):
        if (
            "timeout" not in kwargs
            and isinstance(url, str)
            and "translate.google.com/m" in url
        ):
            kwargs["timeout"] = (
                _GOOGLE_TRANSLATE_CONNECT_TIMEOUT,
                _GOOGLE_TRANSLATE_READ_TIMEOUT,
            )
        return original_get(self, url, *args, **kwargs)

    _patched_get.__pdf2zh_google_timeout_patch__ = True
    requests.sessions.Session.get = _patched_get


async def run_translation(cli_args: list[str]) -> dict:
    """Execute translation using v2's config parsing and babeldoc directly."""
    # Patch sys.argv so ConfigManager.initialize_config() picks up our args
    sys.argv = ["pdf2zh_next"] + cli_args

    from pathlib import Path

    from pdf2zh_next.config.main import ConfigManager
    from pdf2zh_next.high_level import create_babeldoc_config

    import babeldoc.assets.assets
    from babeldoc.format.pdf.high_level import do_translate, get_translation_stage
    from babeldoc.progress_monitor import ProgressMonitor

    settings = ConfigManager().initialize_config()

    # Extract input files from parsed settings
    input_files = list(settings.basic.input_files)
    settings.basic.input_files = set()

    results = []
    errors: list[dict] = []
    start_time = time.time()

    # Skip the blocking health-check translate("Hello") for SiliconFlowFree.
    # The translator __init__ already validates the proxy via /check and /config;
    # the redundant full-translation health check can hang for minutes when the
    # upstream API is slow or down.
    if "--siliconflowfree" in cli_args:
        _disable_translator_health_check()
        # Automatic term extraction is the first heavy LLM phase in BabelDOC and
        # commonly stalls around 30% with SiliconFlowFree due to long proxy timeouts.
        settings.translation.no_auto_extract_glossary = True
    elif "--google" in cli_args:
        _patch_google_request_timeout()

    # Ensure babeldoc assets are available
    babeldoc.assets.assets.warmup()

    for file_path in input_files:
        try:
            config = create_babeldoc_config(settings, Path(file_path))

            def progress_change_callback(**event):
                event_type = event.get("type", "")
                if event_type not in (
                    "progress_start",
                    "progress_update",
                    "progress_end",
                ):
                    return
                progress_event = {
                    "type": event_type,
                    "stage": event.get("stage", ""),
                    "stage_progress": event.get("stage_progress", 0.0),
                    "stage_current": event.get("stage_current", 0),
                    "stage_total": event.get("stage_total", 0),
                    "overall_progress": event.get("overall_progress", 0.0),
                    "part_index": event.get("part_index", 0),
                    "total_parts": event.get("total_parts", 0),
                }
                print(json.dumps(progress_event), file=sys.stderr, flush=True)

            with ProgressMonitor(
                get_translation_stage(config),
                progress_change_callback=progress_change_callback,
                report_interval=config.report_interval,
            ) as pm:
                tr = do_translate(pm, config)

            result = {
                "mono_pdf": str(tr.mono_pdf_path) if tr and tr.mono_pdf_path else None,
                "dual_pdf": str(tr.dual_pdf_path) if tr and tr.dual_pdf_path else None,
                "time_cost": tr.total_seconds if tr else 0.0,
            }
            results.append(result)

        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            errors.append(error_event)
            print(json.dumps(error_event), file=sys.stderr, flush=True)

    elapsed = time.time() - start_time
    return {"results": results, "time_cost": elapsed, "errors": errors}


def main():
    logging.basicConfig(level=logging.INFO)

    raw = sys.stdin.read()
    try:
        cli_args = json.loads(raw)
    except json.JSONDecodeError as e:
        error = {"type": "error", "message": f"Invalid JSON input: {e}"}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(1)

    if not isinstance(cli_args, list):
        error = {"type": "error", "message": "Expected JSON array of CLI args"}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(1)

    result = asyncio.run(run_translation(cli_args))
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.flush()

    # Treat as failure only when nothing succeeded.
    # babeldoc can emit recoverable error events while still producing outputs.
    if result.get("errors") and not result.get("results"):
        sys.exit(2)


if __name__ == "__main__":
    main()
