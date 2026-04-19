"""
OpenRouter LLM client for the Mathematics Distillation Challenge:
Equational Theories, Stage 1.

The single entry point is :func:`call_llm`.  It handles request construction,
retry/backoff, and response parsing for the OpenRouter API.

All models are accessed through OpenRouter, which routes to the appropriate
upstream provider (Anthropic, OpenAI, Google, DeepSeek, etc.) automatically.

Retry and backoff
-----------------
Network errors and transient API errors (HTTP 429, 503, 529, any 5xx) are
retried with jittered exponential back-off:

* Initial interval: 1 s
* Multiplier: 2×
* Maximum interval: 30 s
* Total budget: 120 s
* Jitter: ±50 % per interval (uniform random)

Non-transient errors (auth failures, ``insufficient_quota``, malformed
requests) fail immediately without retrying.

API key rotation
----------------
``ProviderConfig.api_keys`` accepts a list of keys.  When more than one key
is supplied they are rotated round-robin across concurrent calls that share
the same *provider_name*, spreading rate-limit quota across the pool.

Model profiles
--------------
Each known model has a built-in profile controlling ``max_output_tokens``
(capped at 32 768), ``seed``, temperature, and whether extended reasoning is
enabled.  Unknown models fall through to defaults: ``max_tokens``, and any
reasoning / temperature settings from :class:`CompletionKwargs`.

For models without a built-in profile, pass
``CompletionKwargs(reasoning_effort="none")`` to explicitly disable reasoning,
or ``"low"``/``"medium"``/``"high"`` to enable it.

Provider routing
----------------
Set ``ProviderConfig.preferred_providers`` to steer OpenRouter to a specific
upstream (e.g. ``["DeepInfra"]``).  Set ``allow_fallbacks=False`` to make the
choice strict — OpenRouter returns an error rather than falling back to
another provider.

Use :mod:`models` (``models.py``) to drive these settings from a JSON config
file rather than hard-coding them.

``@provider_hint`` suffix
-------------------------
Append ``@<label>`` to a model ID to pin the upstream route for your own
bookkeeping.  The suffix is stripped before the request is sent::

    "deepseek/deepseek-v3.2@my-key"  →  API receives  "deepseek/deepseek-v3.2"

The hint must follow a ``/``-containing base model id; bare names with ``@``
are left unchanged.

Empty response retry
--------------------
Some models occasionally return an empty response with ``finish_reason ==
"stop"``.  The call is retried once in this case (unless a content refusal
was signalled).

When a model's profile has ``reasoning_mode != DISABLED`` and the response is
empty with ``finish_reason == "length"`` (hidden reasoning exhausted the token
budget), the call is retried once with a doubled token budget.

When a model returns an empty ``content`` field for any other reason, the
response text is treated as empty — hidden reasoning fields such as
``reasoning_content`` are **not** used as a fallback.

Logging
-------
All notable events are emitted through Python's standard :mod:`logging`
module under the logger named ``"llm"``.  Enable them in your application
with::

    import logging
    logging.basicConfig(level=logging.WARNING)

``WARNING`` covers retried errors and automatic retries (empty responses,
doubled-budget retries).  ``ERROR`` covers permanent failures and unexpected
response shapes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

__all__ = [
    "call_llm",
    "ProviderConfig",
    "CompletionKwargs",
    "LlmResponse",
    "LlmError",
    "EmptyApiKeyError",
    "HttpError",
    "ApiError",
    "normalize_openrouter_model_id",
]

_logger = logging.getLogger("llm")

_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Runtime configuration for a single LLM API provider.

    Parameters
    ----------
    api_keys:
        One or more OpenRouter API keys.  Multi-key support is useful when
        you have several keys and want to spread concurrent calls across
        their rate-limit quotas.  Blank / whitespace-only entries are
        silently ignored.  An empty list (after filtering) causes
        :exc:`EmptyApiKeyError` at call time.
    base_url:
        Override the default OpenRouter endpoint (``https://openrouter.ai/api/v1``).
        Useful for proxies or staging.
    preferred_providers:
        Ordered list of upstream provider names for OpenRouter routing.
        When non-empty, the request includes ``"provider": {"order": [...]}``.
        For example: ``["DeepInfra"]``.
    quantizations:
        Quantization filter for OpenRouter provider routing.  When non-empty,
        the request includes ``"provider": {"quantizations": [...]}``.
        For example: ``["bf16"]``.
    allow_fallbacks:
        When ``preferred_providers`` is set, controls whether OpenRouter may
        fall back to other providers if the preferred ones are unavailable.
        Defaults to ``False`` for strict provider pinning.
    """
    api_keys: list[str] = field(default_factory=list)
    base_url: Optional[str] = None
    preferred_providers: list[str] = field(default_factory=list)
    quantizations: list[str] = field(default_factory=list)
    allow_fallbacks: bool = False


@dataclass
class CompletionKwargs:
    """Completion parameters forwarded to the model.

    ``None`` fields are omitted from the API request so the provider's own
    defaults apply.

    Parameters
    ----------
    max_tokens:
        Maximum tokens in the model's response.
    temperature:
        Sampling temperature (0.0–2.0).
    reasoning_effort:
        Controls the reasoning budget.  For models with a built-in profile,
        this overrides the profile default.  For models without a profile,
        pass ``"none"`` to explicitly disable reasoning, or
        ``"low"``/``"medium"``/``"high"`` to enable it.
    seed:
        Optional integer seed for reproducible sampling.  When set, the
        value is included in the request body as ``"seed"``.
    """
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class LlmResponse:
    """A successful response from an LLM provider.

    Attributes
    ----------
    text:
        The model's generated text.
    tokens_in:
        Number of prompt tokens consumed, if reported by the provider.
    tokens_out:
        Number of completion tokens generated, if reported by the provider.
    finish_reason:
        The ``finish_reason`` returned by the API (e.g. ``"stop"``,
        ``"length"``, ``"content_filter"``).
    refusal:
        Non-``None`` when the model refused to answer due to content policy.
        When set, ``text`` is empty.
    """
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    finish_reason: Optional[str] = None
    refusal: Optional[str] = None
    actual_provider: Optional[str] = None


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class LlmError(Exception):
    """Base class for all LLM errors."""

    def is_overloaded(self) -> bool:
        """Return True when the upstream is overloaded or rate-limited."""
        return False

    def is_upstream_failure(self) -> bool:
        """Return True for transient errors that a circuit breaker should track."""
        return False


class NoProviderError(LlmError):
    """No provider is configured for the requested provider name."""
    def __init__(self, provider_name: str) -> None:
        super().__init__(f"no provider config for '{provider_name}'")
        self.provider_name = provider_name


class EmptyApiKeyError(LlmError):
    """The provider has no non-empty API key(s)."""
    def __init__(self, provider_name: str) -> None:
        super().__init__(f"empty API key for provider '{provider_name}'")
        self.provider_name = provider_name


class HttpError(LlmError):
    """A network-level error (connection refused, timeout, TLS, ...)."""
    def __init__(self, cause: Exception) -> None:
        super().__init__(f"HTTP error: {cause}")
        self.cause = cause

    def is_upstream_failure(self) -> bool:
        return True


class ApiError(LlmError):
    """The API returned a non-2xx HTTP status.

    Attributes
    ----------
    status:
        HTTP status code.
    body_snippet:
        First 512 bytes of the response body.
    error_type:
        Provider-specific error type string, if available.
    """
    def __init__(self, status: int, body_snippet: str, error_type: Optional[str] = None) -> None:
        tag = ""
        if status == 429:
            tag = " [rate-limited]"
        elif status in (503, 529):
            tag = " [overloaded]"
        if error_type:
            msg = f"API {status}{tag} ({error_type}): {body_snippet}"
        else:
            msg = f"API {status}{tag}: {body_snippet}"
        super().__init__(msg)
        self.status = status
        self.body_snippet = body_snippet
        self.error_type = error_type

    def is_overloaded(self) -> bool:
        return self.status in (429, 503, 529)

    def is_upstream_failure(self) -> bool:
        return _is_transient_api_error(self.status, self.error_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(s: str, max_bytes: int = 512) -> str:
    """Truncate *s* to at most *max_bytes* UTF-8 bytes."""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "..."


def _normalize_message_content(content, status: int) -> str:
    """Normalize OpenRouter ``message.content`` into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                text = part.get("text")
                if isinstance(text, str) and (part_type is None or part_type == "text"):
                    texts.append(text)
        return "".join(texts).strip()
    raise ApiError(status, f"unexpected message.content shape: {type(content).__name__}")


def _is_transient_api_error(status: int, error_type: Optional[str]) -> bool:
    if status == 429:
        return error_type != "insufficient_quota"
    return status >= 500


def _is_quota_error(status: int, message: str) -> bool:
    if status in (403, 429):
        lower = message.lower()
        return any(kw in lower for kw in ("key limit", "quota", "limit exceeded", "billing"))
    return False


# ---------------------------------------------------------------------------
# Model ID normalisation
# ---------------------------------------------------------------------------

def normalize_openrouter_model_id(model_id: str) -> str:
    """Strip the ``@provider_hint`` suffix from an OpenRouter model ID.

    The ``<model>@<provider_hint>`` format allows pinning the upstream route
    for bookkeeping while the API still receives only the base model ID.  The
    hint must follow a ``/``-containing base model ID; bare names with ``@``
    are left unchanged.

    Examples
    --------
    >>> normalize_openrouter_model_id("deepseek/deepseek-v3.2@my-key")
    'deepseek/deepseek-v3.2'
    >>> normalize_openrouter_model_id("custom-model@provider")
    'custom-model@provider'
    """
    trimmed = model_id.strip()
    if "@" in trimmed:
        base, _, hint = trimmed.rpartition("@")
        base = base.strip()
        if base and "/" in base and hint.strip():
            return base
    return trimmed


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

class _ReasoningMode(Enum):
    DISABLED = "disabled"
    LOW = "low"


@dataclass
class _ModelProfile:
    max_output_tokens: int = 32768
    seed: Optional[int] = None
    reasoning_mode: _ReasoningMode = _ReasoningMode.DISABLED
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None


_OPENROUTER_PROFILES: dict[str, _ModelProfile] = {
    "anthropic/claude-haiku-4.5":              _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "anthropic/claude-opus-4.6":               _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "anthropic/claude-sonnet-4.6":             _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "deepseek/deepseek-v3.2":                  _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "google/gemini-2.5-flash":                 _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "google/gemini-3.1-flash-lite-preview":    _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "google/gemini-3-flash-preview":           _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "google/gemini-3.1-pro-preview":           _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.LOW,      temperature=0.0,  reasoning_effort="low"),
    "z-ai/glm-5":                              _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "openai/gpt-5.4":                          _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=None),
    "openai/gpt-5-mini":                       _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.LOW,      temperature=None, reasoning_effort="low"),
    "openai/gpt-5-nano":                       _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.LOW,      temperature=None, reasoning_effort="low"),
    "openai/gpt-oss-120b":                     _ModelProfile(max_output_tokens=8192, seed=0,    reasoning_mode=_ReasoningMode.LOW,      temperature=0.0,  reasoning_effort="low"),
    "x-ai/grok-4.1-fast":                      _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "moonshotai/kimi-k2.5":                    _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.DISABLED, temperature=None),
    "minimax/minimax-m2.5":                    _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.LOW,      temperature=0.0,  reasoning_effort="low"),
    "qwen/qwen3.5-397b-a17b":                  _ModelProfile(seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "stepfun/step-3.5-flash":                  _ModelProfile(seed=None, reasoning_mode=_ReasoningMode.LOW,      temperature=0.0,  reasoning_effort="low"),
    # The three official Stage 1 evaluation models mirror evaluation_models.json.
    "google/gemma-4-31b-it":                   _ModelProfile(max_output_tokens=8192, seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
    "meta-llama/llama-3.3-70b-instruct":       _ModelProfile(max_output_tokens=8192, seed=0,    reasoning_mode=_ReasoningMode.DISABLED, temperature=0.0),
}


def _openrouter_model_profile(model_id: str) -> Optional[_ModelProfile]:
    return _OPENROUTER_PROFILES.get(model_id.strip().lower())


# ---------------------------------------------------------------------------
# API key rotation (thread-safe round-robin)
# ---------------------------------------------------------------------------

_KEY_INDICES: dict[str, int] = {}
_KEY_LOCK = threading.Lock()


def _choose_api_key(provider_name: str, config: ProviderConfig) -> Optional[str]:
    """Return the next API key for *provider_name*, rotating round-robin."""
    keys = [k.strip() for k in config.api_keys if k.strip()]
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    with _KEY_LOCK:
        cursor = _KEY_INDICES.get(provider_name, 0)
        key = keys[cursor % len(keys)]
        _KEY_INDICES[provider_name] = (cursor + 1) % len(keys)
    return key


# ---------------------------------------------------------------------------
# Exponential back-off retry loop
# ---------------------------------------------------------------------------

class _Transient(Exception):
    """Raised inside retry attempts to signal a retryable error."""
    def __init__(self, inner: LlmError) -> None:
        self.inner = inner


class _Permanent(Exception):
    """Raised inside retry attempts to signal a non-retryable error."""
    def __init__(self, inner: LlmError) -> None:
        self.inner = inner


async def _with_backoff(
    fn,
    initial: float = 1.0,
    multiplier: float = 2.0,
    max_interval: float = 30.0,
    max_elapsed: float = 120.0,
):
    """Retry *fn()* with exponential back-off.

    *fn* must be a coroutine function (``async def``) that raises
    :exc:`_Transient` for retryable errors or :exc:`_Permanent` for
    permanent ones.

    Delays are jittered by +/-50 % to spread retry storms.
    """
    interval = initial
    deadline = time.monotonic() + max_elapsed
    while True:
        try:
            return await fn()
        except _Permanent as e:
            raise e.inner
        except _Transient as e:
            remaining = deadline - time.monotonic()
            delay = min(interval, max_interval) * random.uniform(0.5, 1.5)
            if remaining <= 0 or remaining < delay:
                raise e.inner
            await asyncio.sleep(delay)
            interval = min(interval * multiplier, max_interval)


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

async def _call_once(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model_id: str,
    body: dict,
) -> LlmResponse:
    """Send one request with retry/backoff and return the parsed response."""
    async def attempt() -> LlmResponse:
        try:
            resp = await client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            )
        except httpx.RequestError as e:
            raise _Transient(HttpError(e))

        status = resp.status_code
        if 200 <= status < 300:
            try:
                data = resp.json()
            except Exception as e:
                preview = _truncate(resp.text)
                _logger.warning(
                    "Response deserialization failed for %s (transient, will retry): %s\nRaw body: %s",
                    model_id, e, preview,
                )
                raise _Transient(ApiError(status, f"deserialization error: {e}"))

            choices = data.get("choices") or []
            choice = choices[0] if choices else None
            message = (choice or {}).get("message") or {}
            refusal = message.get("refusal")
            content = _normalize_message_content(message.get("content"), status)
            finish_reason = (choice or {}).get("finish_reason")
            usage = data.get("usage") or {}

            # OpenRouter returns the actual upstream provider in `provider`.
            # If absent, parse it from the `@provider` suffix of the model field.
            actual_provider: Optional[str] = data.get("provider") or None
            if not actual_provider:
                resp_model = data.get("model", "")
                if "@" in resp_model:
                    actual_provider = resp_model.split("@", 1)[1] or None

            return LlmResponse(
                text=content,
                tokens_in=usage.get("prompt_tokens"),
                tokens_out=usage.get("completion_tokens"),
                finish_reason=finish_reason,
                refusal=refusal,
                actual_provider=actual_provider,
            )

        raw = resp.text
        # For 5xx, skip structured parse — the body is often HTML from a load balancer.
        if status >= 500:
            msg = raw
            error_type = None
        else:
            try:
                err_data = resp.json()
                msg = err_data.get("error", {}).get("message") or raw
                error_type = err_data.get("error", {}).get("type")
            except Exception:
                msg = raw
                error_type = None

        snippet = _truncate(msg)
        transient = _is_transient_api_error(status, error_type)
        if transient:
            _logger.warning(
                "API error %d for %s (transient, will retry): %s",
                status, model_id, snippet,
            )
        elif _is_quota_error(status, msg):
            _logger.warning(
                "API error %d for %s (provider quota/key-limit issue): %s",
                status, model_id, snippet,
            )
        else:
            _logger.error(
                "API error %d for %s: %s", status, model_id, snippet
            )
        err = ApiError(status, snippet, error_type)
        raise _Transient(err) if transient else _Permanent(err)

    return await _with_backoff(attempt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def call_llm(
    client: httpx.AsyncClient,
    provider_name: str,
    provider_config: ProviderConfig,
    model_id: str,
    prompt: str,
    kwargs: CompletionKwargs,
    system_prompt: Optional[str] = None,
) -> LlmResponse:
    """Call an LLM via OpenRouter and return its response.

    Parameters
    ----------
    client:
        A reusable ``httpx.AsyncClient``.  Create it once and share it across
        calls to take advantage of connection pooling.  Set a generous timeout
        (e.g. ``httpx.AsyncClient(timeout=300)``) to accommodate slow models.
    provider_name:
        A label you choose for this provider instance (e.g.
        ``"openrouter-main"``).  It is used as the key for round-robin API
        key rotation — always use the **same name** for the same provider so
        that rotation stays consistent across concurrent calls.  It also
        appears in log messages.
    provider_config:
        API keys, optional base URL override, and optional upstream provider
        preference list.  See :class:`ProviderConfig`.
    model_id:
        Model identifier string exactly as OpenRouter expects it
        (e.g. ``"deepseek/deepseek-v3.2"``).  You may append an
        ``@provider_hint`` suffix — it will be stripped before the request.
    prompt:
        The user-turn message text.
    kwargs:
        Completion parameters.  See :class:`CompletionKwargs`.
    system_prompt:
        Optional system message prepended before the user turn.  Pass
        ``None`` (the default) to omit it.

    Returns
    -------
    LlmResponse
        Contains the generated text, token counts, finish reason, and an
        optional refusal message.

    Raises
    ------
    EmptyApiKeyError
        ``provider_config.api_keys`` is empty or contains only whitespace.
    ApiError
        The API returned a non-transient error, or the 120-second retry
        budget was exhausted.
    HttpError
        A network-level error persisted after the retry budget was exhausted.
    """
    api_key = _choose_api_key(provider_name, provider_config)
    if api_key is None:
        raise EmptyApiKeyError(provider_name)

    base = (provider_config.base_url or _BASE_URL).rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    request_model_id = normalize_openrouter_model_id(model_id)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict = {"model": request_model_id, "messages": messages}

    profile = _openrouter_model_profile(request_model_id)
    is_reasoning = False

    max_tokens = kwargs.max_tokens if kwargs.max_tokens is not None else (
        profile.max_output_tokens if profile else 8192
    )
    if profile:
        max_tokens = min(max_tokens, profile.max_output_tokens)
        if profile.seed is not None:
            body["seed"] = profile.seed
        if kwargs.temperature is not None:
            body["temperature"] = kwargs.temperature
        elif profile.temperature is not None:
            body["temperature"] = profile.temperature
        # Reasoning: explicit kwargs take precedence over profile defaults,
        # mirroring how kwargs.seed overrides profile.seed below.
        if kwargs.reasoning_effort is not None:
            body["reasoning"] = {"effort": kwargs.reasoning_effort}
            if kwargs.reasoning_effort != "none":
                is_reasoning = True
        elif profile.reasoning_mode == _ReasoningMode.DISABLED:
            body["reasoning"] = {"effort": "none"}
        else:  # LOW
            is_reasoning = True
            effort = profile.reasoning_effort or "low"
            body["reasoning"] = {"effort": effort}
    else:
        if kwargs.temperature is not None:
            body["temperature"] = kwargs.temperature
        if kwargs.reasoning_effort is not None:
            body["reasoning"] = {"effort": kwargs.reasoning_effort}
            if kwargs.reasoning_effort != "none":
                is_reasoning = True

    body["max_tokens"] = max_tokens

    # Explicit seed from caller takes precedence over any profile default.
    if kwargs.seed is not None:
        body["seed"] = kwargs.seed

    # Provider routing.
    if provider_config.preferred_providers or provider_config.quantizations:
        prov: dict = {}
        if provider_config.preferred_providers:
            prov["order"] = provider_config.preferred_providers
        if provider_config.quantizations:
            prov["quantizations"] = provider_config.quantizations
        prov["allow_fallbacks"] = provider_config.allow_fallbacks
        body["provider"] = prov

    response = await _call_once(client, url, api_key, request_model_id, body)

    # Reasoning models may exhaust the token budget on hidden reasoning and
    # return no visible content with finish_reason=length.  Retry once with a
    # doubled budget to reduce spuriously empty answers.
    if (
        is_reasoning
        and not response.text.strip()
        and response.finish_reason == "length"
    ):
        retry_cap = profile.max_output_tokens if profile else 65536
        retry_max = min(max_tokens * 2, retry_cap)
        if retry_max > max_tokens:
            _logger.warning(
                "Reasoning response was empty with finish_reason=length; retrying once "
                "(model=%s, max_tokens=%d, retry_max_tokens=%d)",
                model_id, max_tokens, retry_max,
            )
            body["max_tokens"] = retry_max
            return await _call_once(client, url, api_key, request_model_id, body)

    # Some models occasionally return an empty response with
    # finish_reason=stop.  Retry once to reduce spurious empty answers.
    if not response.text.strip() and not response.refusal:
        _logger.warning(
            "Model returned empty response; retrying once (model=%s, finish_reason=%s)",
            model_id, response.finish_reason,
        )
        return await _call_once(client, url, api_key, request_model_id, body)

    return response
