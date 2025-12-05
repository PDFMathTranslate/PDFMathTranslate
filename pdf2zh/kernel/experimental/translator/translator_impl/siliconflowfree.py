import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import httpx
from pdf2zh.kernel.experimental.config.model import SettingsModel
from pdf2zh.kernel.experimental.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh.kernel.experimental.translator.base_translator import BaseTranslator
from pdf2zh.kernel.experimental.translator.rate_limiter.qps_rate_limiter import QPSRateLimiter
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


class ServerNotAvailableError(Exception):
    pass


AVAILABLE_SERVER_ENDPOINTS = [
    "https://api1.pdf2zh-next.com/chatproxy",
    "https://api2.pdf2zh-next.com/chatproxy",
]


class SiliconFlowFreeTranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "siliconflowfree"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        self.settings = settings
        super().__init__(settings, rate_limiter)

        self.enable_json_mode = False
        if settings.translate_engine_settings.siliconflow_free_enable_json_mode:
            self.add_cache_impact_parameters("request_json_mode", True)
            self.enable_json_mode = True
        # CloudFlare has a timeout of 100 seconds
        self.client = httpx.Client(timeout=100)

        self.url = AVAILABLE_SERVER_ENDPOINTS[0]
        self.get_fast_service()
        pdf2zh_next_recommended_qps = 10
        pdf2zh_next_recommended_pool_max_workers = 100
        self.fetch_setting()

    def fetch_setting(self):
        try:
            response = self.client.get(f"{self.url}/config")
            if response.status_code == 200:
                resp = response.json()
                if resp["status"] == "ok":
                    qps = resp["qps"]
                    max_pool_size = resp["max_pool_size"]

                    assert isinstance(qps, int)
                    assert isinstance(max_pool_size, int)

                    assert qps > 0
                    assert max_pool_size > 0

                    pdf2zh_next_recommended_qps = qps
                    pdf2zh_next_recommended_pool_max_workers = max_pool_size

                    if isinstance(self.rate_limiter, QPSRateLimiter):
                        self.rate_limiter.set_max_qps(qps)
                        logger.info(f"Updated QPS rate limiter to {qps}")
                    logger.info(
                        f"Fetched setting and updated: qps: {qps}, max_pool_size: {max_pool_size}"
                    )

        except Exception as e:
            logger.error(f"Failed to fetch setting and update: {e}")

    def get_fast_service(self):
        """Find the fastest responding endpoint by sending parallel requests."""

        def test_endpoint_speed(endpoint):
            """Test a single endpoint 3 times and return (endpoint, total_response_time, success)."""
            total_response_time = 0
            success_count = 0

            for i in range(3):
                start_time = time.time()
                success = self.check_server_status(endpoint)
                response_time = time.time() - start_time
                total_response_time += response_time

                if success:
                    success_count += 1

                logger.debug(
                    f"Endpoint {endpoint} test {i + 1}/3: {response_time:.3f}s, success: {success}"
                )

            # Consider endpoint successful if at least 2 out of 3 tests succeed
            overall_success = success_count >= 2

            if not overall_success:
                total_response_time = float("inf")

            logger.info(
                f"Endpoint {endpoint} overall: {total_response_time:.3f}s total, {success_count}/3 successes"
            )
            return (endpoint, total_response_time, overall_success)

        # Test all endpoints in parallel
        fastest_endpoint = None
        fastest_time = float("inf")

        with ThreadPoolExecutor(
            max_workers=len(AVAILABLE_SERVER_ENDPOINTS)
        ) as executor:
            # Submit all endpoint tests
            future_to_endpoint = {
                executor.submit(test_endpoint_speed, endpoint): endpoint
                for endpoint in AVAILABLE_SERVER_ENDPOINTS
            }

            # Get results as they complete
            for future in as_completed(future_to_endpoint):
                endpoint, response_time, success = future.result()

                if success and response_time < fastest_time:
                    fastest_time = response_time
                    fastest_endpoint = endpoint
                    logger.info(
                        f"Found faster endpoint: {endpoint} ({response_time:.3f}s)"
                    )

        # Update the URL to use the fastest endpoint
        if fastest_endpoint:
            self.url = fastest_endpoint
            logger.info(
                f"Selected fastest endpoint: {fastest_endpoint} ({fastest_time:.3f}s)"
            )
        else:
            logger.warning("No available endpoints found, using default")
            self.url = AVAILABLE_SERVER_ENDPOINTS[0]  # Fallback to first endpoint

        return self.url

    def check_server_status(self, server_endpoint: str):
        try:
            response = self.client.post(f"{server_endpoint}/check", timeout=5)
            logger.info(
                f"Checking server status: {server_endpoint}, status code: {response.status_code}, response: {response.json()}"
            )
            if response.status_code == 200:
                resp = response.json()
                if resp["status"] == "ok":
                    return True
                else:
                    raise ServerNotAvailableError(
                        f"Server is not available, message: {resp['message']}"
                    )
            else:
                return False
        except ServerNotAvailableError as e:
            raise
        except Exception as e:
            return False

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        return self.do_llm_translate(
            f"You are a professional,authentic machine translation engine.\n\n;; Treat next line as plain text input and translate it into {self.lang_out}, output translation ONLY. If translation is unnecessary (e.g. proper nouns, codes, {'{{1}}, etc. '}), return the original text. NO explanations. NO notes. Input:\n\n{text}",
            rate_limit_params,
        )

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=30, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            request = {
                "text": text,
                "requestJsonMode": True,
            }
        else:
            request = {
                "text": text,
            }

        response = self.client.post(
            self.url,
            json=request,
            timeout=60,
        )
        if response.status_code == 429:
            raise RateLimitError
        response.raise_for_status()
        message = response.json()["content"]
        message = self._remove_cot_content(message)
        return message
