"""Tests for the kernel abstraction layer."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestKernelProtocol(unittest.TestCase):
    """Test TranslateRequest and TranslateResult dataclasses."""

    def test_translate_request_defaults(self):
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"])
        self.assertEqual(req.files, ["test.pdf"])
        self.assertEqual(req.lang_in, "en")
        self.assertEqual(req.lang_out, "zh")
        self.assertEqual(req.service, "google")
        self.assertEqual(req.thread, 4)
        self.assertIsNone(req.pages)
        self.assertFalse(req.debug)

    def test_translate_result_defaults(self):
        from pdf2zh.kernel.protocol import TranslateResult

        result = TranslateResult()
        self.assertIsNone(result.mono_pdf)
        self.assertIsNone(result.dual_pdf)
        self.assertEqual(result.time_cost, 0.0)

    def test_translate_result_with_paths(self):
        from pdf2zh.kernel.protocol import TranslateResult

        result = TranslateResult(
            mono_pdf=Path("/tmp/mono.pdf"),
            dual_pdf=Path("/tmp/dual.pdf"),
            time_cost=1.5,
        )
        self.assertEqual(result.mono_pdf, Path("/tmp/mono.pdf"))
        self.assertEqual(result.time_cost, 1.5)


class TestKernelRegistry(unittest.TestCase):
    """Test KernelRegistry singleton."""

    def setUp(self):
        from pdf2zh.kernel.registry import KernelRegistry

        self._registry = KernelRegistry
        self._orig_kernels = dict(KernelRegistry._kernels)
        self._orig_active = KernelRegistry._active

    def tearDown(self):
        self._registry._kernels = self._orig_kernels
        self._registry._active = self._orig_active

    def test_register_and_get(self):
        from pdf2zh.kernel.registry import KernelRegistry

        mock_kernel = MagicMock()
        mock_kernel.name = "test_kernel"
        mock_kernel.is_available.return_value = True

        KernelRegistry.register(mock_kernel)
        self.assertIs(KernelRegistry.get("test_kernel"), mock_kernel)

    def test_get_nonexistent_raises(self):
        from pdf2zh.kernel.registry import KernelRegistry

        with self.assertRaises(KeyError):
            KernelRegistry.get("nonexistent_kernel_xyz")

    def test_switch_and_active_name(self):
        from pdf2zh.kernel.registry import KernelRegistry

        mock_kernel = MagicMock()
        mock_kernel.name = "switchable"
        mock_kernel.is_available.return_value = True

        KernelRegistry.register(mock_kernel)
        KernelRegistry.switch("switchable")
        self.assertEqual(KernelRegistry.active_name(), "switchable")

    def test_switch_unavailable_raises(self):
        from pdf2zh.kernel.registry import KernelRegistry

        mock_kernel = MagicMock()
        mock_kernel.name = "unavailable"
        mock_kernel.is_available.return_value = False

        KernelRegistry.register(mock_kernel)
        with self.assertRaises(RuntimeError):
            KernelRegistry.switch("unavailable")

    def test_available_filters_correctly(self):
        from pdf2zh.kernel.registry import KernelRegistry

        k1 = MagicMock()
        k1.name = "avail1"
        k1.is_available.return_value = True

        k2 = MagicMock()
        k2.name = "unavail1"
        k2.is_available.return_value = False

        KernelRegistry.register(k1)
        KernelRegistry.register(k2)

        available = KernelRegistry.available()
        self.assertIn("avail1", available)
        self.assertNotIn("unavail1", available)


class TestLegacyKernelVersion(unittest.TestCase):

    def test_name(self):
        from pdf2zh.kernel.legacy import LegacyKernel

        k = LegacyKernel()
        self.assertEqual(k.name, "fast")

    def test_version(self):
        from pdf2zh.kernel.legacy import LegacyKernel

        k = LegacyKernel()
        self.assertEqual(k.version, "1.9.11")

    def test_is_available(self):
        from pdf2zh.kernel.legacy import LegacyKernel

        k = LegacyKernel()
        self.assertTrue(k.is_available())


class TestPreciseKernelVersion(unittest.TestCase):

    def test_name(self):
        from pdf2zh.kernel.precise import PreciseKernel

        k = PreciseKernel()
        self.assertEqual(k.name, "precise")

    def test_is_available_without_venv(self):
        from pdf2zh.kernel.precise import PreciseKernel

        k = PreciseKernel()
        result = k.is_available()
        self.assertIsInstance(result, bool)


class TestV2Bridge(unittest.TestCase):
    """Test v2 bridge v1 → v2 CLI args + env mapping."""

    def test_cli_args_basic(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(
            files=["test.pdf"],
            lang_in="en",
            lang_out="ja",
            service="google",
            thread=8,
        )
        args = request_to_cli_args(req)
        self.assertIn("test.pdf", args)
        self.assertIn("--lang-in", args)
        self.assertIn("en", args)
        self.assertIn("--lang-out", args)
        self.assertIn("ja", args)
        self.assertIn("--qps", args)
        self.assertIn("8", args)
        self.assertIn("--google", args)

    def test_cli_args_pages(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], pages=[0, 1, 4])
        args = request_to_cli_args(req)
        self.assertIn("--pages", args)
        self.assertIn("1,2,5", args)

    def test_cli_args_compatible(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], compatible=True)
        args = request_to_cli_args(req)
        self.assertIn("--enhance-compatibility", args)

    def test_cli_args_vfont_vchar(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], vfont="Arial", vchar="[a-z]")
        args = request_to_cli_args(req)
        self.assertIn("--formular-font-pattern", args)
        self.assertIn("Arial", args)
        self.assertIn("--formular-char-pattern", args)
        self.assertIn("[a-z]", args)

    def test_cli_args_prompt(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], prompt="Be formal")
        args = request_to_cli_args(req)
        self.assertIn("--custom-system-prompt", args)
        self.assertIn("Be formal", args)

    def test_cli_args_debug(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], debug=True)
        args = request_to_cli_args(req)
        self.assertIn("--debug", args)

    def test_cli_args_ignore_cache(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], ignore_cache=True)
        args = request_to_cli_args(req)
        self.assertIn("--ignore-cache", args)

    def test_cli_args_output(self):
        from pathlib import Path
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], output="/tmp/out")
        args = request_to_cli_args(req)
        self.assertIn("--output", args)
        idx = args.index("--output")
        self.assertEqual(args[idx + 1], str(Path("/tmp/out").resolve()))

    def test_cli_args_service_model_split(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], service="openai:gpt-4")
        args = request_to_cli_args(req)
        self.assertIn("--openai", args)

    def test_service_name_mapping(self):
        from pdf2zh.kernel.v2_bridge import SERVICE_NAME_MAP

        # Verify key service mappings exist (values are CLI flags)
        self.assertEqual(SERVICE_NAME_MAP["google"], "google")
        self.assertEqual(SERVICE_NAME_MAP["openai"], "openai")
        self.assertEqual(SERVICE_NAME_MAP["deepseek"], "deepseek")
        self.assertEqual(SERVICE_NAME_MAP["bing"], "bing")
        self.assertEqual(SERVICE_NAME_MAP["deepl"], "deepl")
        self.assertEqual(SERVICE_NAME_MAP["ollama"], "ollama")
        self.assertEqual(SERVICE_NAME_MAP["gemini"], "gemini")
        self.assertEqual(SERVICE_NAME_MAP["tencent"], "tencent")

    def test_request_to_env_maps_envs(self):
        from pdf2zh.kernel.v2_bridge import request_to_env
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(
            files=["test.pdf"],
            service="openai",
            envs={
                "OPENAI_API_KEY": "sk-test123",
                "OPENAI_BASE_URL": "https://custom.api",
            },
        )
        env = request_to_env(req)
        self.assertEqual(env["PDF2ZH_OPENAI_API_KEY"], "sk-test123")
        self.assertEqual(env["PDF2ZH_OPENAI_BASE_URL"], "https://custom.api")

    def test_request_to_env_service_model(self):
        from pdf2zh.kernel.v2_bridge import request_to_env
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(
            files=["test.pdf"],
            service="deepseek:deepseek-reasoner",
            envs={"DEEPSEEK_API_KEY": "ds-key"},
        )
        env = request_to_env(req)
        self.assertEqual(env["PDF2ZH_DEEPSEEK_API_KEY"], "ds-key")
        self.assertEqual(env["PDF2ZH_DEEPSEEK_MODEL"], "deepseek-reasoner")


class TestKernelInit(unittest.TestCase):

    def test_fast_registered(self):
        from pdf2zh.kernel import KernelRegistry

        kernel = KernelRegistry.get("fast")
        self.assertEqual(kernel.name, "fast")

    def test_precise_registered(self):
        from pdf2zh.kernel import KernelRegistry

        kernel = KernelRegistry.get("precise")
        self.assertEqual(kernel.name, "precise")

    def test_fast_in_available(self):
        from pdf2zh.kernel import KernelRegistry

        self.assertIn("fast", KernelRegistry.available())


class TestKernelVersions(unittest.TestCase):
    """Verify version information for both kernels."""

    def test_legacy_version_matches_package(self):
        from pdf2zh import __version__
        from pdf2zh.kernel.legacy import LegacyKernel

        k = LegacyKernel()
        self.assertEqual(k.version, __version__)

    def test_experimental_version_is_string(self):
        from pdf2zh.kernel.precise import PreciseKernel

        k = PreciseKernel()
        version = k.version
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)

    def test_experimental_version_semver_when_available(self):
        """When venv exists, version should be a proper semver."""
        import re
        from pdf2zh.kernel.precise import PreciseKernel

        k = PreciseKernel()
        if not k.is_available():
            self.skipTest("pdf2zh_next venv not available")
        self.assertRegex(k.version, r"^\d+\.\d+\.\d+")


class TestLegacyKernelTranslation(unittest.TestCase):
    """Test LegacyKernel.translate() with mocked high_level.translate()."""

    @patch("pdf2zh.high_level.translate")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_translate_single_file(self, mock_model_inst, mock_onnx, mock_translate):
        mock_model_inst.value = MagicMock()
        mock_translate.return_value = [("/tmp/mono.pdf", "/tmp/dual.pdf")]

        from pdf2zh.kernel.legacy import LegacyKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        k = LegacyKernel()
        req = TranslateRequest(files=["test.pdf"], service="google")
        results = k.translate(req)

        mock_translate.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mono_pdf, Path("/tmp/mono.pdf"))
        self.assertEqual(results[0].dual_pdf, Path("/tmp/dual.pdf"))

    @patch("pdf2zh.high_level.translate")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_translate_passes_all_params(
        self, mock_model_inst, mock_onnx, mock_translate
    ):
        mock_model_inst.value = MagicMock()
        mock_translate.return_value = []

        from pdf2zh.kernel.legacy import LegacyKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        k = LegacyKernel()
        req = TranslateRequest(
            files=["test.pdf"],
            lang_in="ja",
            lang_out="en",
            service="openai",
            thread=8,
            vfont="Arial",
            vchar="[a-z]",
            envs={"OPENAI_API_KEY": "sk-test"},
            ignore_cache=True,
            compatible=True,
        )
        k.translate(req)

        call_kwargs = mock_translate.call_args[1]
        self.assertEqual(call_kwargs["lang_in"], "ja")
        self.assertEqual(call_kwargs["lang_out"], "en")
        self.assertEqual(call_kwargs["service"], "openai")
        self.assertEqual(call_kwargs["thread"], 8)
        self.assertTrue(call_kwargs["ignore_cache"])
        self.assertTrue(call_kwargs["compatible"])

    @patch("pdf2zh.high_level.translate")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_translate_with_prompt_wraps_template(
        self, mock_model_inst, mock_onnx, mock_translate
    ):
        mock_model_inst.value = MagicMock()
        mock_translate.return_value = []

        from pdf2zh.kernel.legacy import LegacyKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        k = LegacyKernel()
        req = TranslateRequest(files=["test.pdf"], prompt="Be formal")
        k.translate(req)

        call_kwargs = mock_translate.call_args[1]
        from string import Template

        self.assertIsInstance(call_kwargs["prompt"], Template)
        self.assertEqual(call_kwargs["prompt"].template, "Be formal")


class TestPreciseKernelTranslation(unittest.TestCase):
    """Test PreciseKernel.translate() with mocked subprocess."""

    @patch("pdf2zh.kernel.precise.subprocess.Popen")
    @patch(
        "pdf2zh.kernel.precise.PreciseKernel.is_available",
        return_value=True,
    )
    @patch("pdf2zh.kernel.precise.PreciseKernel.ensure_venv")
    def test_translate_single_file(self, mock_venv, mock_avail, mock_popen):
        import json

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = json.dumps(
            {
                "results": [{"mono_pdf": "/tmp/mono.pdf", "dual_pdf": "/tmp/dual.pdf"}],
                "time_cost": 1.5,
            }
        )
        mock_proc.stderr = iter([])
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        from pdf2zh.kernel.precise import PreciseKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        k = PreciseKernel()
        req = TranslateRequest(files=["test.pdf"], service="google")
        results = k.translate(req)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mono_pdf, Path("/tmp/mono.pdf"))
        self.assertAlmostEqual(results[0].time_cost, 1.5)

    @patch("pdf2zh.kernel.precise.subprocess.Popen")
    @patch("pdf2zh.kernel.precise.PreciseKernel.ensure_venv")
    def test_translate_subprocess_failure_raises(self, mock_venv, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = ""
        mock_proc.stderr = iter([])
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        from pdf2zh.kernel.precise import PreciseKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        k = PreciseKernel()
        req = TranslateRequest(files=["test.pdf"])
        with self.assertRaises(RuntimeError):
            k.translate(req)

    @patch("pdf2zh.kernel.precise.subprocess.Popen")
    @patch(
        "pdf2zh.kernel.precise.PreciseKernel.is_available",
        return_value=True,
    )
    @patch("pdf2zh.kernel.precise.PreciseKernel.ensure_venv")
    def test_translate_progress_callback(self, mock_venv, mock_avail, mock_popen):
        import json

        progress_event = json.dumps(
            {
                "type": "progress_update",
                "stage": "Translating",
                "stage_progress": 50.0,
                "stage_current": 5,
                "stage_total": 10,
                "overall_progress": 25.0,
                "part_index": 1,
                "total_parts": 1,
            }
        )
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = json.dumps(
            {"results": [], "time_cost": 0.5}
        )
        mock_proc.stderr = iter([progress_event + "\n"])
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        from pdf2zh.kernel.precise import PreciseKernel
        from pdf2zh.kernel.protocol import TranslateRequest

        callback = MagicMock()
        k = PreciseKernel()
        req = TranslateRequest(files=["test.pdf"])
        k.translate(req, callback=callback)

        callback.assert_called_once()
        event_arg = callback.call_args[0][0]
        self.assertEqual(event_arg["type"], "progress_update")
        self.assertEqual(event_arg["stage"], "Translating")


class TestV2BridgeEndToEnd(unittest.TestCase):
    """End-to-end v2 bridge tests."""

    def test_all_services_produce_cli_flag(self):
        from pdf2zh.kernel.v2_bridge import SERVICE_NAME_MAP, request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        for v1_name, v2_flag in SERVICE_NAME_MAP.items():
            req = TranslateRequest(files=["test.pdf"], service=v1_name)
            args = request_to_cli_args(req)
            expected_flag = f"--{v2_flag.replace('_', '-')}"
            self.assertIn(
                expected_flag,
                args,
                f"Missing CLI flag {expected_flag} for service {v1_name}",
            )

    def test_cli_args_all_fields(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(
            files=["test.pdf"],
            service="deepseek:deepseek-chat",
            lang_in="en",
            lang_out="ja",
            thread=8,
            compatible=True,
            vfont="CMR",
            vchar="[0-9]",
            prompt="Be formal",
            ignore_cache=True,
            debug=True,
            output="/tmp/out",
        )
        args = request_to_cli_args(req)

        self.assertIn("--deepseek", args)
        self.assertIn("--enhance-compatibility", args)
        self.assertIn("--lang-in", args)
        self.assertIn("en", args)
        self.assertIn("--lang-out", args)
        self.assertIn("ja", args)
        self.assertIn("--qps", args)
        self.assertIn("8", args)
        self.assertIn("--formular-font-pattern", args)
        self.assertIn("CMR", args)
        self.assertIn("--formular-char-pattern", args)
        self.assertIn("[0-9]", args)
        self.assertIn("--custom-system-prompt", args)
        self.assertIn("Be formal", args)
        self.assertIn("--ignore-cache", args)
        self.assertIn("--debug", args)
        self.assertIn("--output", args)
        idx = args.index("--output")
        from pathlib import Path

        self.assertEqual(args[idx + 1], str(Path("/tmp/out").resolve()))

    def test_env_all_fields(self):
        from pdf2zh.kernel.v2_bridge import request_to_env
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(
            files=["a.pdf", "b.pdf"],
            service="gemini:gemini-pro",
            envs={"GEMINI_API_KEY": "key123"},
        )
        env = request_to_env(req)

        self.assertEqual(env["PDF2ZH_GEMINI_API_KEY"], "key123")
        self.assertEqual(env["PDF2ZH_GEMINI_MODEL"], "gemini-pro")

    def test_output_defaults_to_input_parent(self):
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["/some/dir/test.pdf"])
        args = request_to_cli_args(req)
        self.assertIn("--output", args)
        idx = args.index("--output")
        self.assertEqual(args[idx + 1], "/some/dir")

    def test_output_resolved_to_absolute_path(self):
        from pathlib import Path
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        # Relative input file → output should be resolved to absolute
        req = TranslateRequest(files=["relative/test.pdf"])
        args = request_to_cli_args(req)
        idx = args.index("--output")
        output_path = args[idx + 1]
        self.assertTrue(Path(output_path).is_absolute())

    def test_explicit_output_resolved_to_absolute(self):
        from pathlib import Path
        from pdf2zh.kernel.v2_bridge import request_to_cli_args
        from pdf2zh.kernel.protocol import TranslateRequest

        req = TranslateRequest(files=["test.pdf"], output="rel/out")
        args = request_to_cli_args(req)
        idx = args.index("--output")
        output_path = args[idx + 1]
        self.assertTrue(Path(output_path).is_absolute())


class TestCLIKernelPipeline(unittest.TestCase):
    """Test that CLI args correctly flow through the kernel layer."""

    @patch("pdf2zh.kernel.registry.KernelRegistry.switch")
    @patch("pdf2zh.kernel.registry.KernelRegistry.get")
    @patch("pdf2zh.doclayout.set_backend")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_cli_fast_mode_routes_through_kernel(
        self, mock_model_inst, mock_onnx, mock_set_backend, mock_get, mock_switch
    ):
        """Verify --mode fast goes through KernelRegistry."""
        mock_kernel = MagicMock()
        mock_kernel.translate.return_value = []
        mock_get.return_value = mock_kernel
        mock_model_inst.value = MagicMock()

        from pdf2zh.pdf2zh import main

        main(["test.pdf", "--mode", "fast"])

        mock_switch.assert_called_once_with("fast")
        mock_kernel.translate.assert_called_once()

    @patch("pdf2zh.kernel.registry.KernelRegistry.switch")
    @patch("pdf2zh.kernel.registry.KernelRegistry.get")
    @patch("pdf2zh.doclayout.set_backend")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_cli_precise_mode_routes_through_kernel(
        self, mock_model_inst, mock_onnx, mock_set_backend, mock_get, mock_switch
    ):
        """Verify --mode precise goes through KernelRegistry."""
        mock_kernel = MagicMock()
        mock_kernel.translate.return_value = []
        mock_get.return_value = mock_kernel
        mock_model_inst.value = MagicMock()

        from pdf2zh.pdf2zh import main

        main(["test.pdf", "--mode", "precise"])

        mock_switch.assert_called_once_with("precise")
        mock_kernel.translate.assert_called_once()

    @patch("pdf2zh.kernel.registry.KernelRegistry.switch")
    @patch("pdf2zh.kernel.registry.KernelRegistry.get")
    @patch("pdf2zh.doclayout.set_backend")
    @patch("pdf2zh.doclayout.OnnxModel")
    @patch("pdf2zh.doclayout.ModelInstance")
    def test_cli_service_model_syntax_passed_through(
        self, mock_model_inst, mock_onnx, mock_set_backend, mock_get, mock_switch
    ):
        """Verify 'openai:gpt-4' service syntax is passed to TranslateRequest."""
        mock_kernel = MagicMock()
        mock_kernel.translate.return_value = []
        mock_get.return_value = mock_kernel
        mock_model_inst.value = MagicMock()

        from pdf2zh.pdf2zh import main

        main(["test.pdf", "-s", "openai:gpt-4"])

        call_args = mock_kernel.translate.call_args[0][0]
        self.assertEqual(call_args.service, "openai:gpt-4")


if __name__ == "__main__":
    unittest.main()
