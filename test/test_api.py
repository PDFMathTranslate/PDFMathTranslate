"""Tests for the lightweight REST API server."""

import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_TOKEN = "test-token-abc123"
FAKE_PDF = b"%PDF-1.4 fake content"
FAKE_MONO = b"%PDF-1.4 mono result"
FAKE_DUAL = b"%PDF-1.4 dual result"


def _make_client(token=TEST_TOKEN):
    """Create a Flask test client with mocked model."""
    from pdf2zh.api import create_api_app

    model = MagicMock()
    app, jobs = create_api_app(token=token, model=model)
    app.config["TESTING"] = True
    return app.test_client(), jobs, model


def _auth_header(token=TEST_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, data=None, file_content=FAKE_PDF, filename="test.pdf"):
    """Helper to POST a file to /v1/translate."""
    payload = {"file": (io.BytesIO(file_content), filename)}
    form_data = {}
    if data is not None:
        import json

        form_data["data"] = json.dumps(data)
    return client.post(
        "/v1/translate",
        data={**payload, **form_data},
        headers=_auth_header(),
        content_type="multipart/form-data",
    )


class TestTokenAuth(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, _ = _make_client()

    def test_missing_auth_header_returns_401(self):
        resp = self.client.get("/v1/status")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"], "unauthorized")

    def test_invalid_token_returns_401(self):
        resp = self.client.get(
            "/v1/status", headers={"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual(resp.status_code, 401)

    def test_wrong_scheme_returns_401(self):
        resp = self.client.get(
            "/v1/status", headers={"Authorization": f"Basic {TEST_TOKEN}"}
        )
        self.assertEqual(resp.status_code, 401)

    def test_valid_token_passes(self):
        resp = self.client.get("/v1/status", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)


class TestStatusEndpoint(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, _ = _make_client()

    def test_health_check_returns_ok(self):
        resp = self.client.get("/v1/status", headers=_auth_header())
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["active_jobs"], 0)
        self.assertEqual(data["total_jobs"], 0)
        # CPU info should always be present
        self.assertIn("cpu", data)
        self.assertIn("cores", data["cpu"])
        self.assertIn("arch", data["cpu"])
        self.assertIsInstance(data["cpu"]["cores"], int)
        # GPU info should be a list (possibly empty)
        self.assertIn("gpu", data)
        self.assertIsInstance(data["gpu"], list)


class TestTranslateEndpoint(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, _ = _make_client()

    @patch("pdf2zh.api._run_translation")
    def test_upload_no_file_returns_400(self, mock_run):
        resp = self.client.post(
            "/v1/translate",
            headers=_auth_header(),
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no file", resp.get_json()["error"])

    @patch("pdf2zh.api._run_translation")
    def test_upload_empty_file_returns_400(self, mock_run):
        resp = _upload(self.client, file_content=b"")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.get_json()["error"])

    @patch("pdf2zh.api.threading.Thread")
    @patch("pdf2zh.api._run_translation")
    def test_upload_valid_file_returns_202(self, mock_run, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        resp = _upload(self.client)
        self.assertEqual(resp.status_code, 202)
        data = resp.get_json()
        self.assertIn("id", data)
        self.assertTrue(len(data["id"]) > 0)
        mock_thread.start.assert_called_once()

    @patch("pdf2zh.api.threading.Thread")
    @patch("pdf2zh.api._run_translation")
    def test_upload_invalid_json_data_returns_400(self, mock_run, mock_thread_cls):
        payload = {"file": (io.BytesIO(FAKE_PDF), "test.pdf"), "data": "not-json{"}
        resp = self.client.post(
            "/v1/translate",
            data=payload,
            headers=_auth_header(),
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("invalid JSON", resp.get_json()["error"])

    @patch("pdf2zh.api.threading.Thread")
    @patch("pdf2zh.api._run_translation")
    def test_upload_with_translation_params(self, mock_run, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        params = {"lang_in": "en", "lang_out": "zh", "service": "google"}
        resp = _upload(self.client, data=params)
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["id"]
        job = self.jobs.get(job_id)
        self.assertEqual(job.params["lang_in"], "en")
        self.assertEqual(job.params["lang_out"], "zh")
        self.assertEqual(job.params["service"], "google")


class TestJobLifecycle(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, self.model = _make_client()

    def test_get_nonexistent_job_returns_404(self):
        resp = self.client.get("/v1/translate/nonexistent", headers=_auth_header())
        self.assertEqual(resp.status_code, 404)

    @patch("pdf2zh.high_level.translate_stream", return_value=(FAKE_MONO, FAKE_DUAL))
    def test_job_completes_and_download_works(self, mock_translate):
        from pdf2zh.api import _run_translation, JobStatus

        # Create job directly via manager
        job = self.jobs.create("test.pdf", {"stream": FAKE_PDF})
        _run_translation(job, self.model)

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.progress_current, job.progress_total)
        self.assertGreater(job.progress_total, 0)

        # Check status
        resp = self.client.get(f"/v1/translate/{job.id}", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["has_result"])
        self.assertEqual(data["progress"]["current"], data["progress"]["total"])
        self.assertGreater(data["progress"]["total"], 0)

        # Download mono
        resp = self.client.get(
            f"/v1/translate/{job.id}/download/mono", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, FAKE_MONO)

        # Download dual
        resp = self.client.get(
            f"/v1/translate/{job.id}/download/dual", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, FAKE_DUAL)

    def test_download_invalid_format_returns_400(self):
        from pdf2zh.api import JobStatus

        job = self.jobs.create("test.pdf", {})
        job.status = JobStatus.COMPLETED
        job.result_mono = FAKE_MONO
        resp = self.client.get(
            f"/v1/translate/{job.id}/download/invalid", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 400)

    def test_download_incomplete_job_returns_400(self):
        job = self.jobs.create("test.pdf", {})
        resp = self.client.get(
            f"/v1/translate/{job.id}/download/mono", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not completed", resp.get_json()["error"])

    def test_download_nonexistent_job_returns_404(self):
        resp = self.client.get(
            "/v1/translate/nonexistent/download/mono", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 404)

    @patch("pdf2zh.kernel.registry.KernelRegistry.get")
    def test_precise_job_status_exposes_stage_progress(self, mock_kernel_get):
        from pdf2zh.api import _run_translation, JobStatus
        from pdf2zh.kernel.protocol import TranslateResult

        mock_kernel = MagicMock()

        def _fake_translate(_req, callback=None, cancellation_event=None):
            if callback:
                callback(
                    {
                        "event": "progress_update",
                        "stage": "layout_analysis",
                        "stage_progress": 0.25,
                        "stage_current": 1,
                        "stage_total": 4,
                        "overall_progress": 0.5,
                    }
                )
            return [TranslateResult(mono_pdf=None, dual_pdf=None)]

        mock_kernel.translate.side_effect = _fake_translate
        mock_kernel_get.return_value = mock_kernel

        job = self.jobs.create(
            "test.pdf",
            {
                "stream": FAKE_PDF,
                "backend": "precise",
                "service": "siliconflowfree",
                "lang_in": "en",
                "lang_out": "zh",
                "thread": 1,
            },
        )
        _run_translation(job, self.model)

        self.assertEqual(job.status, JobStatus.COMPLETED)
        resp = self.client.get(f"/v1/translate/{job.id}", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("stage", data)
        self.assertEqual(data["stage"]["name"], "layout_analysis")
        self.assertEqual(data["stage"]["event"], "progress_update")
        self.assertEqual(data["stage"]["current"], 1)
        self.assertEqual(data["stage"]["total"], 4)
        self.assertEqual(data["stage"]["progress"], 25.0)
        self.assertEqual(data["progress"]["current"], 100)
        self.assertEqual(data["progress"]["total"], 100)

    def test_dual_only_result_is_reported_as_available(self):
        from pdf2zh.api import JobStatus

        job = self.jobs.create("test.pdf", {})
        job.status = JobStatus.COMPLETED
        job.result_dual = FAKE_DUAL

        resp = self.client.get(f"/v1/translate/{job.id}", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["has_result"])
        self.assertFalse(data["has_mono_result"])
        self.assertTrue(data["has_dual_result"])

        resp = self.client.get(
            f"/v1/translate/{job.id}/download/dual", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, FAKE_DUAL)

    @patch("pdf2zh.api.threading.Thread")
    @patch("pdf2zh.api._run_translation")
    def test_list_jobs_returns_all(self, mock_run, mock_thread_cls):
        mock_thread_cls.return_value = MagicMock()
        # Create two jobs
        _upload(self.client)
        _upload(self.client)
        resp = self.client.get("/v1/list", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["jobs"]), 2)


class TestJobCancellation(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, _ = _make_client()

    def test_stop_nonexistent_job_returns_404(self):
        resp = self.client.post(
            "/v1/translate/nonexistent/stop", headers=_auth_header()
        )
        self.assertEqual(resp.status_code, 404)

    def test_stop_completed_job_returns_400(self):
        from pdf2zh.api import JobStatus

        job = self.jobs.create("test.pdf", {})
        job.status = JobStatus.COMPLETED
        resp = self.client.post(f"/v1/translate/{job.id}/stop", headers=_auth_header())
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not running", resp.get_json()["error"])

    def test_stop_running_job(self):
        from pdf2zh.api import JobStatus

        job = self.jobs.create("test.pdf", {})
        job.status = JobStatus.RUNNING
        resp = self.client.post(f"/v1/translate/{job.id}/stop", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        self.assertIn("cancellation requested", resp.get_json()["message"])
        self.assertTrue(job.cancel_event.is_set())


class TestReboot(unittest.TestCase):
    def setUp(self):
        self.client, self.jobs, _ = _make_client()

    def test_reboot_clears_all_jobs(self):
        from pdf2zh.api import JobStatus

        self.jobs.create("a.pdf", {})
        self.jobs.create("b.pdf", {})
        self.assertEqual(self.jobs.total_count(), 2)

        resp = self.client.post("/v1/reboot", headers=_auth_header())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.jobs.total_count(), 0)

    def test_reboot_cancels_running_jobs(self):
        from pdf2zh.api import JobStatus

        job = self.jobs.create("test.pdf", {})
        job.status = JobStatus.RUNNING
        self.client.post("/v1/reboot", headers=_auth_header())
        self.assertTrue(job.cancel_event.is_set())


if __name__ == "__main__":
    unittest.main()
