"""Gemini API client — the only file that imports google.genai."""
from __future__ import annotations

import time

_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
}


def _is_retryable(exc: BaseException) -> bool:
    # Network-level transient errors — server closed keep-alive connection
    try:
        import httpx
        if isinstance(exc, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError)):
            return True
    except ImportError:
        pass
    # Gemini API errors
    try:
        from google.genai import errors as genai_errors
        if isinstance(exc, genai_errors.ServerError):
            return True
        if isinstance(exc, genai_errors.ClientError):
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            return code == 429
    except AttributeError:
        pass
    return False


class GeminiClient:
    """Thin wrapper around google.genai with retry and finish-reason normalisation.

    Parameters
    ----------
    api_key:
        Gemini API key.
    model:
        Pinned model string, e.g. "gemini-2.0-flash-001". No default — caller
        must supply the exact version to guarantee reproducibility.
    """

    _TEMPERATURE = 0.0
    _MAX_OUTPUT_TOKENS = 4096  # 2.5 Flash generates longer CoT than 2.0 Flash

    def __init__(self, api_key: str, model: str, rpm: int = 0) -> None:
        import google.genai as genai
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

        self._client = genai.Client(api_key=api_key)
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
        from google.genai import types as genai_types

        if self._min_interval > 0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=self._TEMPERATURE,
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
            ),
        )
        text: str = response.text or ""
        candidate = response.candidates[0] if response.candidates else None
        raw_reason = (
            candidate.finish_reason.name
            if candidate and candidate.finish_reason
            else "FINISH_REASON_UNSPECIFIED"
        )
        finish_reason = _FINISH_REASON_MAP.get(raw_reason, "other")
        return text, finish_reason
