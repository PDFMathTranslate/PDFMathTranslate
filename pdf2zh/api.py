"""Lightweight REST API server for pdf2zh.

No Celery/Redis required — uses Python threading for background jobs.
Start with: pdf2zh --api
"""

import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from string import Template
from typing import Dict, Optional

from flask import Flask, request, jsonify, send_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def require_auth(token: str):
    """Decorator factory that validates Bearer token."""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != token:
                return jsonify({"error": "unauthorized"}), 401
            return f(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    filename: str = ""
    progress_current: int = 0
    progress_total: int = 0
    error: Optional[str] = None
    result_mono: Optional[bytes] = field(default=None, repr=False)
    result_dual: Optional[bytes] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    params: Dict = field(default_factory=dict)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    cancel_event: Optional[threading.Event] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "filename": self.filename,
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
            },
            "error": self.error,
            "created_at": self.created_at,
            "has_result": self.result_mono is not None,
        }


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, params: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            filename=filename,
            params=params,
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False
        job.cancel_event.set()
        return True

    def reset(self):
        with self._lock:
            for job in self._jobs.values():
                if job.status == JobStatus.RUNNING and job.cancel_event:
                    job.cancel_event.set()
            self._jobs.clear()

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

    def total_count(self) -> int:
        with self._lock:
            return len(self._jobs)


# ---------------------------------------------------------------------------
# Translation worker
# ---------------------------------------------------------------------------


def _run_translation(job: Job, model):
    import tqdm
    from pdf2zh.high_level import translate_stream

    def progress_callback(t: tqdm.tqdm):
        job.progress_current = t.n
        job.progress_total = t.total

    try:
        job.status = JobStatus.RUNNING
        params = job.params

        logger.info(
            "Job %s starting with params: lang_in=%r, lang_out=%r, service=%r, pages=%r, thread=%r",
            job.id,
            params.get("lang_in"),
            params.get("lang_out"),
            params.get("service"),
            params.get("pages"),
            params.get("thread"),
        )

        prompt = params.get("prompt")
        if prompt:
            prompt = Template(prompt)

        doc_mono, doc_dual = translate_stream(
            stream=params["stream"],
            pages=params.get("pages"),
            lang_in=params.get("lang_in", ""),
            lang_out=params.get("lang_out", ""),
            service=params.get("service", ""),
            thread=params.get("thread", 4),
            vfont=params.get("vfont", ""),
            vchar=params.get("vchar", ""),
            callback=progress_callback,
            cancellation_event=job.cancel_event,
            model=model,
            envs=params.get("envs"),
            prompt=prompt,
            skip_subset_fonts=params.get("skip_subset_fonts", False),
            ignore_cache=params.get("ignore_cache", False),
        )

        if job.cancel_event.is_set():
            job.status = JobStatus.CANCELLED
        else:
            logger.info(
                "Job %s finished: mono=%d bytes, dual=%d bytes, input=%d bytes",
                job.id,
                len(doc_mono) if doc_mono else 0,
                len(doc_dual) if doc_dual else 0,
                len(params.get("stream", b"")),
            )
            job.result_mono = doc_mono
            job.result_dual = doc_dual
            job.status = JobStatus.COMPLETED
    except Exception as e:
        if job.cancel_event and job.cancel_event.is_set():
            job.status = JobStatus.CANCELLED
        else:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.exception(f"Translation job {job.id} failed")
    finally:
        job.params.pop("stream", None)


# ---------------------------------------------------------------------------
# System resource helpers
# ---------------------------------------------------------------------------


def _get_cpu_info() -> dict:
    """Return CPU usage and memory stats."""
    import platform

    info = {
        "arch": platform.machine(),
        "cores": os.cpu_count(),
    }
    try:
        import psutil

        info["usage_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_mb": round(mem.total / 1024 / 1024),
            "used_mb": round(mem.used / 1024 / 1024),
            "percent": mem.percent,
        }
    except ImportError:
        info["usage_percent"] = None
        info["memory"] = None
    return info


def _get_gpu_info() -> list:
    """Return GPU utilisation when possible (NVIDIA via pynvml, else empty)."""
    gpus = []
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "gpu_util_percent": util.gpu,
                    "memory_total_mb": round(mem.total / 1024 / 1024),
                    "memory_used_mb": round(mem.used / 1024 / 1024),
                    "memory_percent": (
                        round(mem.used / mem.total * 100, 1) if mem.total else 0
                    ),
                }
            )
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return gpus


# ---------------------------------------------------------------------------
# Frontend → backend env-key mapping
# ---------------------------------------------------------------------------

# The web frontend uses lowercase keys (e.g. "siliconflow_api_key") while
# each translator class expects its own uppercase keys (e.g. "SILICON_API_KEY").
# This table maps frontend names to backend names so set_envs() can find them.
_ENV_KEY_MAP: dict[str, str] = {
    # OpenAI
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "openai_base_url": "OPENAI_BASE_URL",
    # Azure OpenAI
    "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
    "azure_openai_base_url": "AZURE_OPENAI_BASE_URL",
    "azure_openai_model": "AZURE_OPENAI_MODEL",
    "azure_openai_api_version": "AZURE_OPENAI_API_VERSION",
    # DeepSeek
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "deepseek_model": "DEEPSEEK_MODEL",
    # Ollama
    "ollama_host": "OLLAMA_HOST",
    "ollama_model": "OLLAMA_MODEL",
    # Xinference
    "xinference_host": "XINFERENCE_HOST",
    "xinference_model": "XINFERENCE_MODEL",
    # ModelScope
    "modelscope_api_key": "MODELSCOPE_API_KEY",
    "modelscope_model": "MODELSCOPE_MODEL",
    # Zhipu
    "zhipu_api_key": "ZHIPU_API_KEY",
    "zhipu_model": "ZHIPU_MODEL",
    # SiliconFlow
    "siliconflow_api_key": "SILICON_API_KEY",
    "siliconflow_model": "SILICON_MODEL",
    # Gemini
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_model": "GEMINI_MODEL",
    # Azure Translator
    "azure_api_key": "AZURE_API_KEY",
    "azure_endpoint": "AZURE_ENDPOINT",
    # Tencent
    "tencentcloud_secret_id": "TENCENTCLOUD_SECRET_ID",
    "tencentcloud_secret_key": "TENCENTCLOUD_SECRET_KEY",
    # AnythingLLM
    "anythingllm_apikey": "AnythingLLM_APIKEY",
    "anythingllm_url": "AnythingLLM_URL",
    # Dify
    "dify_apikey": "DIFY_API_KEY",
    "dify_url": "DIFY_API_URL",
    # Grok
    "grok_api_key": "GROK_API_KEY",
    "grok_model": "GROK_MODEL",
    # Groq
    "groq_api_key": "GROQ_API_KEY",
    "groq_model": "GROQ_MODEL",
    # QwenMt
    "qwenmt_api_key": "ALI_API_KEY",
    "qwenmt_model": "ALI_MODEL",
    "qwenmt_base_url": "ALI_BASE_URL",
    # OpenAI-compatible
    "openai_compatible_api_key": "OPENAILIKED_API_KEY",
    "openai_compatible_base_url": "OPENAILIKED_BASE_URL",
    "openai_compatible_model": "OPENAILIKED_MODEL",
    # Aliyun DashScope
    "aliyun_dashscope_api_key": "ALI_API_KEY",
    "aliyun_dashscope_model": "ALI_MODEL",
    "aliyun_dashscope_base_url": "ALI_BASE_URL",
    # DeepL
    "deepl_auth_key": "DEEPL_AUTH_KEY",
    # ClaudeCode
    "claude_code_path": "CLAUDE_CODE_PATH",
    "claude_code_model": "CLAUDE_CODE_MODEL",
}


def _remap_envs(envs: dict | None) -> dict | None:
    """Remap frontend env key names to backend translator env key names."""
    if not envs:
        return envs
    remapped = {}
    for k, v in envs.items():
        backend_key = _ENV_KEY_MAP.get(k, k)
        remapped[backend_key] = v
    return remapped


# ---------------------------------------------------------------------------
# Flask app factory & routes
# ---------------------------------------------------------------------------


def create_api_app(token: Optional[str], model) -> tuple:
    """Create Flask app and JobManager. Returns (app, jobs)."""
    app = Flask("pdf2zh-api")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
    jobs = JobManager()
    auth = require_auth(token) if token else lambda f: f

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "internal server error"}), 500

    @app.route("/v1/translate", methods=["POST"])
    @auth
    def create_translation():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        file = request.files["file"]
        stream = file.stream.read()
        if not stream:
            return jsonify({"error": "empty file"}), 400

        # The frontend may send params as top-level form fields OR inside
        # a nested JSON "data" field.  Merge both sources, preferring
        # explicit top-level form values over the JSON blob.
        data = {}
        if request.form.get("data"):
            try:
                data = json.loads(request.form["data"])
            except json.JSONDecodeError:
                return jsonify({"error": "invalid JSON in 'data' field"}), 400

        def _val(key, default=""):
            """Return a form field if present, else fall back to data JSON."""
            v = request.form.get(key)
            if v is not None and v != "":
                return v
            return data.get(key, default)

        backend = _val("backend", "fast")

        # Map display names (e.g. "Simplified Chinese") to language codes
        # (e.g. "zh").  The frontend stores and sends display names; the
        # translation engine expects ISO-style codes.
        _LANG_MAP = {
            "Simplified Chinese": "zh",
            "Traditional Chinese": "zh-TW",
            "English": "en",
            "French": "fr",
            "German": "de",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Spanish": "es",
            "Italian": "it",
        }

        def _resolve_lang(raw: str) -> str:
            return _LANG_MAP.get(raw, raw)  # pass through if already a code

        # Default service per backend when none specified
        service = _val("service", "")
        if not service:
            service = "google" if backend == "fast" else "siliconflowfree"

        # Validate service name against registry
        from pdf2zh.services import SERVICE_BY_NAME

        if service not in SERVICE_BY_NAME:
            return jsonify({"error": f"unknown service: {service!r}"}), 400

        # Parse pages — may arrive as JSON list or comma-separated string
        raw_pages = _val("pages", None)
        if isinstance(raw_pages, str):
            try:
                raw_pages = json.loads(raw_pages)
            except (json.JSONDecodeError, TypeError):
                raw_pages = None

        raw_thread = _val("thread", 4)
        if isinstance(raw_thread, str):
            try:
                raw_thread = int(raw_thread)
            except ValueError:
                raw_thread = 4

        params = {
            "stream": stream,
            "pages": raw_pages,
            "lang_in": _resolve_lang(_val("lang_in", "")),
            "lang_out": _resolve_lang(_val("lang_out", "")),
            "service": service,
            "thread": raw_thread,
            "vfont": _val("vfont", ""),
            "vchar": _val("vchar", ""),
            "envs": _remap_envs(data.get("envs")),
            "prompt": _val("prompt", None),
            "skip_subset_fonts": data.get("skip_subset_fonts", False),
            "ignore_cache": data.get("ignore_cache", False),
            "backend": backend,
        }

        job = jobs.create(filename=file.filename or "upload.pdf", params=params)

        # Resolve model from KernelRegistry when available, fall back to
        # the model passed at server startup.
        try:
            from pdf2zh.kernel.registry import KernelRegistry

            kernel = KernelRegistry.get(backend)
            run_model = getattr(kernel, "model", model)
        except Exception:
            run_model = model

        t = threading.Thread(
            target=_run_translation,
            args=(job, run_model),
            daemon=True,
        )
        job.thread = t
        t.start()
        return jsonify({"id": job.id}), 202

    @app.route("/v1/translate/<job_id>", methods=["GET"])
    @auth
    def get_translation(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job.to_dict())

    @app.route("/v1/translate/<job_id>/download/<fmt>", methods=["GET"])
    @auth
    def download_result(job_id: str, fmt: str):
        if fmt not in ("mono", "dual"):
            return jsonify({"error": "format must be 'mono' or 'dual'"}), 400
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        if job.status != JobStatus.COMPLETED:
            return (
                jsonify({"error": "job not completed", "status": job.status.value}),
                400,
            )
        data = job.result_mono if fmt == "mono" else job.result_dual
        if not data:
            return jsonify({"error": "result not available"}), 404
        basename = (
            job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
        )
        return send_file(
            io.BytesIO(data),
            mimetype="application/pdf",
            download_name=f"{basename}-{fmt}.pdf",
        )

    @app.route("/v1/status", methods=["GET"])
    @auth
    def health_status():
        return jsonify(
            {
                "status": "ok",
                "active_jobs": jobs.active_count(),
                "total_jobs": jobs.total_count(),
                "cpu": _get_cpu_info(),
                "gpu": _get_gpu_info(),
            }
        )

    @app.route("/v1/list", methods=["GET"])
    @auth
    def list_jobs():
        return jsonify({"jobs": jobs.list_all()})

    @app.route("/v1/translate/<job_id>/stop", methods=["POST"])
    @auth
    def stop_translation(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        if jobs.cancel(job_id):
            return jsonify({"message": "cancellation requested", "id": job_id})
        return jsonify({"error": "job is not running", "status": job.status.value}), 400

    @app.route("/v1/reboot", methods=["POST"])
    @auth
    def reboot():
        jobs.reset()
        return jsonify({"message": "server state reset"})

    @app.route("/v1/version", methods=["GET"])
    def get_version():
        from pdf2zh.kernel.legacy import LegacyKernel
        from pdf2zh.kernel.precise import PreciseKernel

        fast = LegacyKernel()
        precise = PreciseKernel()
        return jsonify(
            {
                "version": fast.version,
                "backends": {
                    "fast": {"version": fast.version, "available": fast.is_available()},
                    "precise": {
                        "version": precise.version if precise.is_available() else None,
                        "available": precise.is_available(),
                    },
                },
            }
        )

    @app.route("/v1/config", methods=["GET"])
    def get_config():
        from pdf2zh.kernel.legacy import LegacyKernel
        from pdf2zh.kernel.precise import PreciseKernel
        from pdf2zh.services import SERVICES

        fast = LegacyKernel()
        precise = PreciseKernel()
        return jsonify(
            {
                "services": [
                    {
                        "display": s.display,
                        "value": s.name,
                        "custom_prompt": s.custom_prompt,
                    }
                    for s in SERVICES
                ],
                "languages": {
                    "Simplified Chinese": "zh",
                    "Traditional Chinese": "zh-TW",
                    "English": "en",
                    "French": "fr",
                    "German": "de",
                    "Japanese": "ja",
                    "Korean": "ko",
                    "Russian": "ru",
                    "Spanish": "es",
                    "Italian": "it",
                },
                "backends": {
                    "fast": {"available": fast.is_available(), "version": fast.version},
                    "precise": {
                        "available": precise.is_available(),
                        "version": precise.version if precise.is_available() else None,
                    },
                },
                "default_backend": "fast",
            }
        )

    return app, jobs


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def run_api_server(
    host: str = "127.0.0.1",
    port: int = 8787,
    token: Optional[str] = None,
    model=None,
):
    if token is None:
        token = hashlib.sha1(os.urandom(32)).hexdigest()
        logger.info(f"Generated API token: {token}")
        print(f"\n{'=' * 60}")
        print(f"  API Token: {token}")
        print(f"  Keep this token secret. Use it in requests as:")
        print(f"  Authorization: Bearer {token}")
        print(f"{'=' * 60}\n")

    app, _jobs = create_api_app(token=token, model=model)
    logger.info(f"Starting API server on {host}:{port}")
    print(f"API server listening on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
