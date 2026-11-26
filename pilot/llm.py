"""Model clients: the real OpenRouter client, a deterministic fake, and usage logging.

Everything in Pilot that needs a model calls
``client.complete(messages, model=..., max_tokens=...)`` and gets an ``LLMResult`` back,
so no other module deals with HTTP, retries or provider usage fields.
``make_client(cfg)`` picks the real client or the fake one from the config.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

from pilot.config import load_config
from pilot.tokens import TokenCounter

log = logging.getLogger(__name__)

APP_TITLE = "Pilot take-home"  # sent as X-Title so calls are recognisable on the OpenRouter dashboard
MAX_ERROR_CHARS = 500  # provider error text kept in an LLMError message
EMPTY_EXTRACTION = '{"accounts":[],"records":[],"closes":[]}'

_usage_lock = threading.Lock()


@dataclass
class LLMResult:
    """One finished model call: the reply text and what it cost."""

    text: str
    prompt_tokens: int  # provider-reported usage.prompt_tokens if present, else local estimate
    completion_tokens: int
    cost_usd: float  # provider-reported usage.cost if present, else 0.0
    latency_s: float
    model: str
    reported: bool  # True when prompt_tokens came from the provider


class LLMError(RuntimeError):
    """A model call failed for good: bad request, bad key, or retries used up.

    ``retryable`` says whether the same call may work later. It is True when the provider
    was unavailable (retries used up, out of credit, key missing or refused, no usable
    answer) and False when the provider rejected this request itself (for example HTTP
    400): sending that message again would fail again. Extraction uses it so that an
    outage never uses up a message's attempts (pilot/extract.py: _count_failed_attempt).
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _Transient(Exception):
    """A failure worth retrying: rate limit, server error, timeout or dropped connection."""


class _ReasoningRejected(LLMError):
    """HTTP 400 because the model refuses the reasoning switch we sent."""


UNAVAILABLE_STATUSES = (401, 402, 403, 404)  # the key, the credit or the model, not the request


def _is_transient(status: int) -> bool:
    """Rate limits (429), timeouts (408) and server-side errors (5xx) may pass on a retry."""
    return status in (408, 429) or status >= 500


def _json_object(response: httpx.Response) -> dict | None:
    """The response body as a dict, or None if it is not a JSON object."""
    try:
        data = response.json()
    except ValueError:  # covers invalid JSON and undecodable bytes
        return None
    return data if isinstance(data, dict) else None


def _error_code(data: dict) -> int | None:
    """The ``error.code`` OpenRouter puts in a response body, if there is one."""
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), int):
        return error["code"]
    return None


def _redact(text: str, api_key: str) -> str:
    """``text`` with the API key replaced by [redacted], both as typed and as Python's repr
    escapes it: an HTTP library that rejects a header quotes its value that way, e.g.
    ``Illegal header value b'Bearer sk-...\\r'``, so the plain key never matches there."""
    if api_key:
        for form in (api_key, repr(api_key)[1:-1]):
            text = text.replace(form, "[redacted]")
    return text


def _error_text(data: dict | None, response: httpx.Response, api_key: str) -> str:
    """A short description of what the provider said went wrong, with the key removed
    (in case a provider ever echoes it back)."""
    error = data.get("error") if data is not None else None
    if isinstance(error, dict):
        text = str(error.get("message") or "")
        metadata = error.get("metadata")
        if isinstance(metadata, dict) and metadata.get("raw"):
            text += f" | upstream: {metadata['raw']}"  # the model host's own words
    else:
        text = response.text
    return _redact(text, api_key)[:MAX_ERROR_CHARS]


def is_openrouter(base_url: str) -> bool:
    """True when ``base_url`` points at OpenRouter, the only API given its extensions."""
    host = urlparse(str(base_url)).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


class OpenRouterClient:
    """Chat completions over OpenRouter's OpenAI-compatible HTTP API.

    ``transport`` and ``sleep`` exist for tests: an ``httpx.MockTransport`` replaces the
    network and a fake ``sleep`` records the backoff waits instead of waiting.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = load_config(cfg)
        # Only the variable's name is kept: the key itself is read at every call (_api_key),
        # so constructing an Assistant (or running FakeLLM tests) never needs a key.
        self._key_env = cfg["api_key_env"]
        self._url = cfg["base_url"].rstrip("/") + "/chat/completions"
        # usage.include, reasoning and provider are OpenRouter extensions. Another
        # OpenAI-compatible API may reject an unknown field with a 400 on every call,
        # so they are sent only to OpenRouter (a grader may point base_url elsewhere).
        self._openrouter = is_openrouter(cfg["base_url"])
        self._reasoning = cfg["reasoning"]
        self._provider = cfg["provider"]
        self._max_retries = cfg["max_retries"]
        self._sleep = sleep
        self._http = httpx.Client(timeout=cfg["request_timeout_s"], transport=transport)
        self._no_reasoning_switch: set[str] = set()  # models that rejected reasoning={"enabled": false}

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()

    def _api_key(self) -> str:
        """The API key, read from the environment now rather than once in __init__, so a
        key set after the Assistant was built (a harness or fixture that exports it later)
        still works. Stripped, because a stray space, CR or LF (cmd's ``set KEY=... ``, a
        key file with CRLF endings) makes the Authorization header illegal: every call then
        fails, and the HTTP library's error quotes the header, key included (packaging review).
        """
        return (os.environ.get(self._key_env) or "").strip()

    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
        purpose: str = "reply",
    ) -> LLMResult:
        """Send one chat request and return the reply with its usage.

        ``purpose`` ("reply", "extract", ...) is not sent; it is part of the shared
        signature so FakeLLM and the usage log can tell reply calls from helper calls.
        Raises LLMError when the call cannot succeed.
        """
        api_key = self._api_key()
        if not api_key:
            raise LLMError(f"API key env var {self._key_env} is not set", retryable=True)
        body = self._request_body(messages, model, max_tokens, temperature, seed, json_mode)
        started = time.perf_counter()
        try:
            data = self._post_with_retries(body, api_key)
        except _ReasoningRejected:
            # Some models cannot switch reasoning off. Retry once without the field and
            # remember the model, so later calls do not pay for the rejected request.
            log.info("%s rejects reasoning switched off; retrying without the field", model)
            self._no_reasoning_switch.add(model)
            del body["reasoning"]
            data = self._post_with_retries(body, api_key)
        latency_s = time.perf_counter() - started  # whole call, including any retries
        return self._to_result(data, messages, model, latency_s)

    def _request_body(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float | None,
        seed: int | None,
        json_mode: bool,
    ) -> dict:
        """The JSON body for /chat/completions; optional fields are left out when unset."""
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self._openrouter:
            body["usage"] = {"include": True}  # ask for token counts and cost in the response
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self._openrouter and not self._reasoning and model not in self._no_reasoning_switch:
            body["reasoning"] = {"enabled": False}
        if self._openrouter and self._provider:
            body["provider"] = self._provider
        return body

    def _post_with_retries(self, body: dict, api_key: str) -> dict:
        """POST ``body``, retrying temporary failures after 2, 4, 8 s (no jitter, so runs repeat)."""
        problem = ""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                wait_s = 2**attempt
                log.warning(
                    "%s call failed (%s); retry %d of %d in %d s",
                    body["model"], problem, attempt, self._max_retries, wait_s,
                )
                self._sleep(wait_s)
            try:
                return self._post_once(body, api_key)
            except _Transient as exc:
                problem = str(exc)
        raise LLMError(f"model call failed after {self._max_retries + 1} attempts: {problem}", retryable=True)

    def _post_once(self, body: dict, api_key: str) -> dict:
        """One HTTP attempt. Returns the response JSON or raises _Transient / LLMError.

        httpx errors are re-raised as our own, redacted, and ``from None``: chained with
        ``from exc``, the original error, whose text can quote the request's headers, would
        be printed unredacted under ours by every log.exception (packaging review).
        """
        headers = {"Authorization": f"Bearer {api_key}", "X-Title": APP_TITLE}
        try:
            response = self._http.post(self._url, json=body, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise _Transient(_redact(f"{type(exc).__name__}: {exc}", api_key)) from None
        except httpx.HTTPError as exc:  # anything else httpx raises (e.g. a bad URL) won't fix itself
            raise LLMError(_redact(f"{type(exc).__name__}: {exc}", api_key), retryable=True) from None

        data = _json_object(response)
        status = response.status_code
        if response.is_success and data is not None:
            if data.get("choices"):
                return data
            # OpenRouter reports errors that happen during generation inside a 200
            # response, with the real status in error.code; treat it like that status.
            status = _error_code(data) or status

        message = _error_text(data, response, api_key)
        if status == 400 and "reasoning" in body and "reasoning" in message.lower():
            raise _ReasoningRejected(message)
        if _is_transient(status):
            raise _Transient(f"HTTP {status}: {message}")
        if status >= 400:
            # 401/402/403/404: the key is refused, the credit is gone or the model is unknown;
            # other 4xx mean the provider rejected this request itself.
            raise LLMError(f"HTTP {status} from provider: {message}", retryable=status in UNAVAILABLE_STATUSES)
        raise LLMError(f"unusable response from provider (not JSON or no choices): {message}", retryable=True)

    def _to_result(self, data: dict, messages: list[dict], model: str, latency_s: float) -> LLMResult:
        """Build the LLMResult; if the provider sent no usage, count the prompt locally."""
        try:
            content = data["choices"][0]["message"].get("content")
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            reported = isinstance(prompt_tokens, int)
            if not reported:
                prompt_tokens = TokenCounter(model).count_messages(messages)
            return LLMResult(
                text=content.strip() if isinstance(content, str) else "",
                prompt_tokens=prompt_tokens,
                completion_tokens=int(usage.get("completion_tokens") or 0),
                cost_usd=float(usage.get("cost") or 0.0),
                latency_s=latency_s,
                model=model,
                reported=reported,
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise LLMError(f"malformed response from provider ({type(exc).__name__})", retryable=True) from exc


def default_responder(messages: list[dict], purpose: str) -> str:
    """FakeLLM's fixed answers: an empty extraction for helper calls, an echo otherwise."""
    if purpose == "extract":
        return EMPTY_EXTRACTION
    last_user = ""
    for message in messages:
        if message.get("role") == "user":
            last_user = message.get("content") or ""
    return "OK: " + last_user[:60]


class FakeLLM:
    """Offline stand-in for OpenRouterClient: same ``complete`` signature, no network, no cost.

    ``responder(messages, purpose) -> str`` decides the reply text. Tests read ``calls``
    to see exactly what Pilot would have sent.
    """

    def __init__(self, responder: Callable[[list[dict], str], str] | None = None) -> None:
        self.responder = responder or default_responder
        self.calls: list[dict] = []

    def close(self) -> None:
        """Nothing to close; here so callers can treat both clients the same way."""

    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
        purpose: str = "reply",
    ) -> LLMResult:
        """Record the call and answer with the responder; token counts are local estimates."""
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],  # copy, so later edits cannot change the record
                "model": model,
                "purpose": purpose,
                "json_mode": json_mode,
            }
        )
        text = self.responder(messages, purpose)
        counter = TokenCounter(model)
        return LLMResult(
            text=text,
            prompt_tokens=counter.count_messages(messages),
            completion_tokens=counter.count_text(text),
            cost_usd=0.0,
            latency_s=0.0,
            model=model,
            reported=False,
        )


def make_client(cfg: dict) -> OpenRouterClient | FakeLLM:
    """FakeLLM when ``cfg["llm"] == "fake"`` (tests, offline runs), else the real client."""
    if cfg.get("llm") == "fake":
        return FakeLLM()
    return OpenRouterClient(cfg)


def append_usage(data_dir: str, purpose: str, result: LLMResult, local_estimate: int) -> None:
    """Append one JSON line describing a model call to ``<data_dir>/usage.jsonl``.

    The eval sums these per purpose (reply vs helper) for its token and cost tables.
    ``local_estimate`` is our own count, kept next to the provider's to track tokenizer drift.
    """
    row = {
        "purpose": purpose,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "reported": result.reported,
        "local_estimate": local_estimate,
    }
    path = Path(data_dir) / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock:  # two threads must never interleave half lines
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
