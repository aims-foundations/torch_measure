"""Anthropic Claude API client — the only file that imports anthropic."""
from __future__ import annotations

import time

_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
}


def _is_retryable(exc: BaseException) -> bool:
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529:
            return True
        if isinstance(exc, anthropic.APIConnectionError):
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


class ClaudeClient:
    """Anthropic Claude API wrapper matching GeminiClient's generate() interface.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    model:
        Pinned model string, e.g. "claude-opus-4-8". No default — caller
        must supply the exact version to guarantee reproducibility.
    rpm:
        Optional rate limit (requests per minute). 0 disables throttling.
    """

    _MAX_TOKENS = 4096
    # Thinking is disabled: get_full_instruction() already requests text-based
    # chain-of-thought in the output, matching the paper's GPT-4o methodology.
    # Enabling thinking would add a second internal reasoning pass on top of the
    # in-output CoT, inflating cost and diverging from the paper's approach.
    _THINKING: dict = {"type": "disabled"}

    def __init__(self, api_key: str, model: str, rpm: int = 0) -> None:
        import anthropic
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

        self._client = anthropic.Anthropic(api_key=api_key)
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

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._MAX_TOKENS,
            thinking=self._THINKING,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        raw_reason = response.stop_reason or "other"
        finish_reason = _FINISH_REASON_MAP.get(raw_reason, "other")
        return text, finish_reason
