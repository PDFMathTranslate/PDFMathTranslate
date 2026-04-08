"""Precise kernel adapter — runs pip-installed pdf2zh_next in an isolated venv."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from pdf2zh.kernel.v2_bridge import request_to_cli_args, request_to_env
from pdf2zh.kernel.protocol import TranslateRequest, TranslateResult

logger = logging.getLogger(__name__)

# Use a dedicated user cache venv for precise kernel runtime.
_CACHE_DIR = Path.home() / ".cache" / "pdf2zh"
_VENV_DIR = _CACHE_DIR / "precise-kernel-venv"
_WORKER_SCRIPT = Path(__file__).resolve().parent / "v2_worker.py"


def _venv_python() -> str:
    """Return the path to the venv's Python interpreter."""
    if sys.platform == "win32":
        return str(_VENV_DIR / "Scripts" / "python.exe")
    return str(_VENV_DIR / "bin" / "python")


class PreciseKernel:
    """Kernel adapter for pdf2zh_next, running in a subprocess with isolated deps."""

    @property
    def name(self) -> str:
        return "precise"

    @property
    def version(self) -> str:
        if not self.is_available():
            return "unknown"
        try:
            result = subprocess.run(
                [
                    _venv_python(),
                    "-c",
                    "import pdf2zh_next; print(pdf2zh_next.__version__)",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def is_available(self) -> bool:
        """Check if venv exists and pdf2zh_next is importable."""
        return Path(_venv_python()).exists() and self._package_importable()

    def ensure_venv(self) -> None:
        """Create venv and install/upgrade pdf2zh_next from pip."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        venv_exists = Path(_venv_python()).exists()

        if venv_exists and self._package_importable():
            return

        if not venv_exists:
            logger.info("Creating precise kernel venv...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(_VENV_DIR)],
                check=True,
                timeout=60,
            )

        logger.info("Installing/upgrading pdf2zh-next from pip into precise venv...")
        subprocess.run(
            [
                _venv_python(),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pdf2zh-next",
            ],
            check=True,
            timeout=300,
        )

        logger.info("Precise kernel venv ready.")

    def _package_importable(self) -> bool:
        """Check if pdf2zh_next can be imported in the venv."""
        try:
            result = subprocess.run(
                [_venv_python(), "-c", "import pdf2zh_next"],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_subprocess_env(self, request: TranslateRequest) -> dict[str, str]:
        """Build environment for the subprocess with PDF2ZH_ prefixed vars."""
        env = os.environ.copy()
        env.update(request_to_env(request))
        return env

    def translate(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]:
        self.ensure_venv()

        cli_args = request_to_cli_args(request)
        input_json = json.dumps(cli_args)
        env = self._build_subprocess_env(request)

        cmd = [_venv_python(), str(_WORKER_SCRIPT)]
        logger.info("Starting precise kernel subprocess: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        stderr_lines: list[str] = []
        try:
            assert proc.stdin is not None
            assert proc.stderr is not None
            assert proc.stdout is not None
            proc.stdin.write(input_json)
            proc.stdin.close()

            for raw_line in proc.stderr:
                line = raw_line.rstrip("\r\n")
                if line:
                    stderr_lines.append(line)
                try:
                    event = self._parse_progress_event(line)
                    if not isinstance(event, dict):
                        raise ValueError("not a JSON object")
                    if callback:
                        callback(event)
                except (json.JSONDecodeError, ValueError):
                    # Preserve original v2 formatting (indentation/progress text).
                    sys.stderr.write(raw_line)
                    sys.stderr.flush()

            stdout = proc.stdout.read()
            proc.wait()

        except Exception:
            proc.kill()
            raise

        if proc.returncode != 0:
            detail = "\n".join(stderr_lines[-20:]) if stderr_lines else "(no stderr)"
            raise RuntimeError(
                f"Precise kernel subprocess failed (exit {proc.returncode}):\n{detail}"
            )

        try:
            result_data = json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON from worker: {stdout[:200]}")

        results = []
        for r in result_data.get("results", []):
            results.append(
                TranslateResult(
                    mono_pdf=Path(r["mono_pdf"]) if r.get("mono_pdf") else None,
                    dual_pdf=Path(r["dual_pdf"]) if r.get("dual_pdf") else None,
                    time_cost=result_data.get("time_cost", 0.0),
                )
            )
        return results

    @staticmethod
    def _parse_progress_event(line: str) -> dict[str, Any]:
        """Parse a JSON progress event from possibly mixed stderr output.

        Rich progress rendering may inject control text around event lines.
        Try full-line JSON first, then extract the JSON object between braces.
        """
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            start = line.find("{")
            end = line.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(line[start : end + 1])

    async def translate_async(
        self,
        request: TranslateRequest,
        callback: Any = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> list[TranslateResult]:
        self.ensure_venv()

        cli_args = request_to_cli_args(request)
        input_json = json.dumps(cli_args)
        env = self._build_subprocess_env(request)

        proc = await asyncio.create_subprocess_exec(
            _venv_python(),
            str(_WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout_bytes, stderr_bytes = await proc.communicate(input=input_json.encode())
        stdout = stdout_bytes.decode()

        if proc.returncode != 0:
            detail = stderr_bytes.decode()[-2000:] if stderr_bytes else "(no stderr)"
            raise RuntimeError(
                f"Precise kernel subprocess failed (exit {proc.returncode}):\n{detail}"
            )

        try:
            result_data = json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON from worker: {stdout[:200]}")

        results = []
        for r in result_data.get("results", []):
            results.append(
                TranslateResult(
                    mono_pdf=Path(r["mono_pdf"]) if r.get("mono_pdf") else None,
                    dual_pdf=Path(r["dual_pdf"]) if r.get("dual_pdf") else None,
                    time_cost=result_data.get("time_cost", 0.0),
                )
            )
        return results


def setup_precise_cli() -> None:
    """CLI entry point: provision the precise kernel venv."""
    logging.basicConfig(level=logging.INFO)
    PreciseKernel().ensure_venv()
