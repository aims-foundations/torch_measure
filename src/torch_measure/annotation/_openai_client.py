"""OpenAI API client — the only file that imports openai."""
from __future__ import annotations

import time

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
}


def _is_retryable(exc: BaseException) -> bool:
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code in (429, 500, 502, 503, 529):
            return True
        if isinstance(exc, openai.APIConnectionError):
            return True
    except ImportError:
        pass
    try:
        import httpx
        if isinstance(exc, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError)):
            return True
    except ImportError:
        pass
    return False


class OpenAIClient:
    """OpenAI API wrapper matching GeminiClient's generate() interface.

    Parameters
    ----------
    api_key:
        OpenAI API key.
    model:
        Pinned model string, e.g. "gpt-4o". No default — caller must supply
        the exact version to guarantee reproducibility.
    rpm:
        Optional rate limit (requests per minute). 0 disables throttling.
    """

    _MAX_TOKENS = 4096

    def __init__(self, api_key: str, model: str, rpm: int = 0) -> None:
        import openai
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._min_interval = (60.0 / rpm) if rpm > 0 else 0.0
        self._last_call_time: float = 0.0

        self._generate_with_retry = retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(min=2, max=256),
            stop=stop_after_attempt(10),
            reraise=True,
        )(self._call_api)

    def generate(self, prompt: str) -> tuple[str, str]:
        """Call the API and return (response_text, finish_reason).

        Retries up to 10 times on transient errors with exponential backoff
        (min 2 s, max 256 s), matching the paper's tenacity settings.
        """
        return self._generate_with_retry(prompt)

    def _call_api(self, prompt: str) -> tuple[str, str]:
        if self._min_interval > 0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""
        raw_reason = response.choices[0].finish_reason or "other"
        finish_reason = _FINISH_REASON_MAP.get(raw_reason, "other")
        return text, finish_reason
