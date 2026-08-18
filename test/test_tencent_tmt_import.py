import importlib
import sys
import types
import unittest
from pathlib import Path


THIRD_PARTY_STUBS = (
    "deepl",
    "ollama",
    "openai",
    "requests",
    "xinference_client",
)

TMT_MODULE_NAMES = (
    "tencentcloud",
    "tencentcloud.common",
    "tencentcloud.tmt",
    "tencentcloud.tmt.v20180321",
    "tencentcloud.tmt.v20180321.models",
    "tencentcloud.tmt.v20180321.tmt_client",
)

# Surface of tencentcloud-sdk-python-tmt 3.1.129 (TextTranslateRequest removed).
TMT_3_1_129_MODEL_CLASSES = (
    "BoundingBox",
    "Coord",
    "ImageTranslateLLMRequest",
    "ImageTranslateLLMResponse",
    "RotateParagraphRect",
    "TransDetail",
)


def _pyproject_dependencies():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return list(data["project"]["dependencies"])


def _tmt_dependency_spec(dependencies):
    matches = [
        dep
        for dep in dependencies
        if dep.replace(" ", "").startswith("tencentcloud-sdk-python-tmt")
    ]
    if not matches:
        raise AssertionError(
            "pyproject.toml is missing tencentcloud-sdk-python-tmt"
        )
    return matches[0]


class TestTencentTmtImportIsolation(unittest.TestCase):
    def setUp(self):
        self._saved = {}
        self._added = []

    def tearDown(self):
        for name in self._added:
            sys.modules.pop(name, None)
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _remember(self, name):
        if name in self._saved or name in self._added:
            return
        if name in sys.modules:
            self._saved[name] = sys.modules[name]
        else:
            self._added.append(name)

    def _put_module(self, name, module):
        self._remember(name)
        sys.modules[name] = module
        if "." not in name:
            return
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, module)

    def _ensure_pkg(self, name):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__path__", None) is not None:
            self._remember(name)
            return existing
        module = types.ModuleType(name)
        module.__path__ = []
        self._put_module(name, module)
        return module

    def _install_missing_third_party_stubs(self):
        for name in THIRD_PARTY_STUBS:
            if name not in sys.modules:
                self._put_module(name, types.ModuleType(name))

        openai = sys.modules["openai"]
        if not hasattr(openai, "RateLimitError"):
            openai.RateLimitError = type("RateLimitError", (Exception,), {})

        if "tenacity" not in sys.modules:
            tenacity = types.ModuleType("tenacity")
            tenacity.retry = lambda *args, **kwargs: (lambda fn: fn)
            tenacity.retry_if_exception_type = lambda *args, **kwargs: None
            tenacity.stop_after_attempt = lambda *args, **kwargs: None
            tenacity.wait_exponential = lambda *args, **kwargs: None
            self._put_module("tenacity", tenacity)

        for pkg in (
            "azure",
            "azure.ai",
            "azure.ai.translation",
            "azure.core",
        ):
            if pkg not in sys.modules:
                self._ensure_pkg(pkg)

        if "azure.ai.translation.text" not in sys.modules:
            text = types.ModuleType("azure.ai.translation.text")
            text.TextTranslationClient = type("TextTranslationClient", (), {})
            self._put_module("azure.ai.translation.text", text)
        if "azure.core.credentials" not in sys.modules:
            creds = types.ModuleType("azure.core.credentials")
            creds.AzureKeyCredential = type("AzureKeyCredential", (), {})
            self._put_module("azure.core.credentials", creds)

        if "pdf2zh.cache" not in sys.modules:
            cache = types.ModuleType("pdf2zh.cache")
            cache.TranslationCache = type(
                "TranslationCache",
                (),
                {"__init__": lambda self, *args, **kwargs: None},
            )
            self._put_module("pdf2zh.cache", cache)

    def _install_incompatible_tmt_stub(self):
        """Install a 3.1.129-shaped TMT SDK: package present, text models gone."""
        for name in TMT_MODULE_NAMES:
            self._remember(name)
            sys.modules.pop(name, None)

        self._ensure_pkg("tencentcloud")
        common = types.ModuleType("tencentcloud.common")
        common.credential = types.SimpleNamespace()
        self._put_module("tencentcloud.common", common)

        self._ensure_pkg("tencentcloud.tmt")
        self._ensure_pkg("tencentcloud.tmt.v20180321")

        models = types.ModuleType("tencentcloud.tmt.v20180321.models")
        for class_name in TMT_3_1_129_MODEL_CLASSES:
            setattr(models, class_name, type(class_name, (), {}))
        self._put_module("tencentcloud.tmt.v20180321.models", models)

        client = types.ModuleType("tencentcloud.tmt.v20180321.tmt_client")
        client.TmtClient = type("TmtClient", (), {})
        self._put_module("tencentcloud.tmt.v20180321.tmt_client", client)

    def _import_translator_against_incompatible_tmt(self):
        self._remember("pdf2zh.translator")
        sys.modules.pop("pdf2zh.translator", None)
        self._install_missing_third_party_stubs()
        self._install_incompatible_tmt_stub()
        models = sys.modules["tencentcloud.tmt.v20180321.models"]
        self.assertFalse(hasattr(models, "TextTranslateRequest"))
        self.assertFalse(hasattr(models, "TextTranslateResponse"))
        return importlib.import_module("pdf2zh.translator")

    def test_translator_import_survives_incompatible_tmt_sdk(self):
        """#1167: other engines must still load when TMT 3.1.129 dropped text models."""
        translator = self._import_translator_against_incompatible_tmt()
        self.assertTrue(hasattr(translator, "GoogleTranslator"))
        self.assertTrue(hasattr(translator, "TencentTranslator"))

    def test_tencent_translator_raises_when_text_models_missing(self):
        translator = self._import_translator_against_incompatible_tmt()
        with self.assertRaises(ImportError) as ctx:
            translator.TencentTranslator("en", "zh", "test")
        message = str(ctx.exception)
        self.assertIn("TextTranslateRequest", message)
        self.assertIn("tencentcloud-sdk-python-tmt", message)

    def test_tmt_dependency_is_pinned_away_from_3_1_129(self):
        spec = _tmt_dependency_spec(_pyproject_dependencies())
        self.assertNotEqual(
            spec.strip(),
            "tencentcloud-sdk-python-tmt",
            "tencentcloud-sdk-python-tmt must be version-pinned",
        )
        self.assertTrue(
            any(token in spec for token in ("<", "<=", "==", "~=")),
            f"expected an upper bound or exact pin, got {spec!r}",
        )
        self.assertTrue(
            "3.1.121" in spec or "3.0.1257" in spec or "<3.1" in spec,
            f"pin should keep a release that still ships TextTranslateRequest: {spec!r}",
        )


if __name__ == "__main__":
    unittest.main()
