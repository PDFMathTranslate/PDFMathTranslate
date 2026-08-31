import sys
import types
import importlib


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _stub_deepl():
    module = _ensure_module("deepl")

    class Translator:
        def __init__(self, auth_key):
            self.auth_key = auth_key

        def translate_text(self, text, target_lang=None, source_lang=None):
            return types.SimpleNamespace(text=text)

    module.Translator = Translator


def _stub_ollama():
    module = _ensure_module("ollama")

    class ResponseError(Exception):
        pass

    class Client:
        def __init__(self, host=None):
            self.host = host

        def chat(self, **kwargs):
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content=kwargs.get("messages", [{}])[0].get("content", ""))
            )

    module.ResponseError = ResponseError
    module.Client = Client


def _stub_openai():
    module = _ensure_module("openai")

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class _ChatCompletions:
        def create(self, **kwargs):
            content = kwargs.get("messages", [{}])[0].get("content", "")
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        delta=types.SimpleNamespace(content=content),
                    )
                ]
            )

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.chat = _Chat()

    class AzureOpenAI(OpenAI):
        def __init__(self, azure_endpoint=None, azure_deployment=None, api_version=None, api_key=None):
            super().__init__(base_url=azure_endpoint, api_key=api_key)
            self.azure_deployment = azure_deployment
            self.api_version = api_version

    module.OpenAI = OpenAI
    module.AzureOpenAI = AzureOpenAI
    module.RateLimitError = RateLimitError
    module.BadRequestError = BadRequestError


def _stub_xinference_client():
    module = _ensure_module("xinference_client")

    class RESTfulClient:
        def __init__(self, host):
            self.host = host

        def get_model(self, model):
            return types.SimpleNamespace(
                chat=lambda **kwargs: {"choices": [{"message": {"content": ""}}]}
            )

    module.RESTfulClient = RESTfulClient


def _stub_azure():
    azure = _ensure_module("azure")
    azure.ai = _ensure_module("azure.ai")
    azure.ai.translation = _ensure_module("azure.ai.translation")
    text_module = _ensure_module("azure.ai.translation.text")
    azure.core = _ensure_module("azure.core")
    azure.core.credentials = _ensure_module("azure.core.credentials")

    class TextTranslationClient:
        def __init__(self, *args, **kwargs):
            pass

        def translate(self, body=None, from_language=None, to_language=None):
            translation = types.SimpleNamespace(text=body[0] if body else "")
            return [types.SimpleNamespace(translations=[translation])]

    class AzureKeyCredential:
        def __init__(self, key):
            self.key = key

    text_module.TextTranslationClient = TextTranslationClient
    azure.core.credentials.AzureKeyCredential = AzureKeyCredential


def _stub_tencentcloud():
    tencentcloud = _ensure_module("tencentcloud")
    tencentcloud.common = _ensure_module("tencentcloud.common")
    credential_module = _ensure_module("tencentcloud.common.credential")
    tencentcloud.tmt = _ensure_module("tencentcloud.tmt")
    v20180321 = _ensure_module("tencentcloud.tmt.v20180321")
    models_module = _ensure_module("tencentcloud.tmt.v20180321.models")
    tmt_client_module = _ensure_module("tencentcloud.tmt.v20180321.tmt_client")

    class DefaultCredentialProvider:
        def get_credential(self):
            raise EnvironmentError

    class Credential:
        def __init__(self, secret_id=None, secret_key=None):
            self.secret_id = secret_id
            self.secret_key = secret_key

    class TextTranslateRequest:
        def __init__(self):
            self.Source = None
            self.Target = None
            self.ProjectId = 0
            self.SourceText = ""

    class TextTranslateResponse:
        def __init__(self, target_text=""):
            self.TargetText = target_text

    class TmtClient:
        def __init__(self, cred, region):
            self.cred = cred
            self.region = region

        def TextTranslate(self, req):
            return TextTranslateResponse(req.SourceText)

    credential_module.DefaultCredentialProvider = DefaultCredentialProvider
    credential_module.Credential = Credential
    models_module.TextTranslateRequest = TextTranslateRequest
    models_module.TextTranslateResponse = TextTranslateResponse
    tmt_client_module.TmtClient = TmtClient
    v20180321.models = models_module
    v20180321.tmt_client = tmt_client_module


def _stub_gradio_pdf():
    module = _ensure_module("gradio_pdf")

    class PDF:
        def __init__(self, *args, **kwargs):
            pass

    module.PDF = PDF


def _stub_babeldoc():
    module = _ensure_module("babeldoc")
    module.__version__ = "0.0-test"
    docvision = _ensure_module("babeldoc.docvision")
    doclayout_module = _ensure_module("babeldoc.docvision.doclayout")

    class OnnxModel:
        @staticmethod
        def load_available():
            return types.SimpleNamespace()

    doclayout_module.OnnxModel = OnnxModel
    docvision.doclayout = doclayout_module


def _stub_pdf2zh_doclayout():
    pdf2zh_pkg = importlib.import_module("pdf2zh")
    module = _ensure_module("pdf2zh.doclayout")

    class OnnxModel:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def load_available():
            return types.SimpleNamespace()

    def set_backend(_backend):
        return None

    module.OnnxModel = getattr(module, "OnnxModel", OnnxModel)
    module.ModelInstance = getattr(
        module, "ModelInstance", types.SimpleNamespace(value=None)
    )
    module.set_backend = getattr(module, "set_backend", set_backend)
    setattr(pdf2zh_pkg, "doclayout", module)


def _stub_pdf2zh_high_level():
    pdf2zh_pkg = importlib.import_module("pdf2zh")
    module = _ensure_module("pdf2zh.high_level")

    def translate(**kwargs):
        return []

    def translate_stream(stream=None, **kwargs):
        return b"", b""

    module.translate = getattr(module, "translate", translate)
    module.translate_stream = getattr(module, "translate_stream", translate_stream)
    setattr(pdf2zh_pkg, "high_level", module)


_stub_deepl()
_stub_ollama()
_stub_openai()
_stub_xinference_client()
_stub_azure()
_stub_tencentcloud()
_stub_gradio_pdf()
_stub_babeldoc()
_stub_pdf2zh_doclayout()
_stub_pdf2zh_high_level()
