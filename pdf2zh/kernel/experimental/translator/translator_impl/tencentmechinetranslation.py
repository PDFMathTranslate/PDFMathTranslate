import logging

from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh.kernel.experimental.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tencentcloud.common import credential
from tencentcloud.tmt.v20180321.models import TextTranslateRequest
from tencentcloud.tmt.v20180321.models import TextTranslateResponse
from tencentcloud.tmt.v20180321.tmt_client import TmtClient

logger = logging.getLogger(__name__)


class TencentMechineTranslationTranslator(BaseTranslator):
    name = "tencent"
    lang_map = {"zh-cn": "zh", "zh-tw": "zh-TW", "zh-hk": "zh-TW"}

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        try:
            cred = credential.DefaultCredentialProvider().get_credential()
        except OSError:
            cred = credential.Credential(
                settings.translate_engine_settings.tencentcloud_secret_id,
                settings.translate_engine_settings.tencentcloud_secret_key,
            )
        self.client = TmtClient(cred, "ap-beijing")
        self.req = TextTranslateRequest()
        self.req.Source = self.lang_in
        self.req.Target = self.lang_out
        self.req.ProjectId = 0

    @retry(
        retry=retry_if_exception(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None):
        self.req.SourceText = text
        resp: TextTranslateResponse = self.client.TextTranslate(self.req)
        return resp.TargetText
