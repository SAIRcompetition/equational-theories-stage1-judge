"""
Per-model configuration loader and call-parameter resolver for OpenRouter.

This module provides the bridge between a JSON model config file and the
:func:`~llm.call_llm` function.  It can be used in two ways:

**1. Load from a JSON config file** (recommended for multi-model setups)::

    from models import load_models, resolve

    models = load_models("evaluation_models.json")   # or omit path for default
    entry  = models["gpt-oss-120b"]
    name, pconfig, kwargs = resolve(entry, api_keys=["<your-openrouter-api-key>"])

    response = await call_llm(
        client, name, pconfig, entry.model_id, prompt, kwargs
    )

**2. Construct a :class:`ModelConfig` directly** (for use as a library)::

    from models import ModelConfig, resolve

    entry = ModelConfig(
        model_id   = "openai/gpt-oss-120b",
        provider   = "deepinfra/bf16",
        max_output_tokens = 8192,
        temperature       = 0.0,
        use_seed          = True,
        seed              = 0,
        reasoning_mode    = "low",
        reasoning_effort  = "low",
    )
    name, pconfig, kwargs = resolve(entry, api_keys=["<your-openrouter-api-key>"])

Config file format
------------------
See ``evaluation_models.json`` for the official Stage 1 evaluation model config.
The top-level keys are:

``defaults``
    Fields applied to every model entry when the per-model value is absent.
    Supported: ``allow_fallbacks``, ``max_output_tokens_cap``, and
    ``reasoning_mode``.

``models``
    A dict keyed by short model aliases (e.g. ``"gpt-oss-120b"``).  Each
    entry maps to a :class:`ModelConfig`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm import CompletionKwargs, ProviderConfig

__all__ = ["ModelConfig", "load_models", "resolve"]


# ---------------------------------------------------------------------------
# Provider name mapping
# ---------------------------------------------------------------------------

# Maps the lowercase-hyphen provider keys used in the JSON config to the
# display names that OpenRouter expects in provider.order.
_PROVIDER_NAMES: dict[str, str] = {
    "anthropic":       "Anthropic",
    "openai":          "OpenAI",
    "azure":           "Azure",
    "google-vertex":   "Google Vertex AI",
    "google-ai-studio":"Google AI Studio",
    "amazon-bedrock":  "Amazon Bedrock",
    "deepinfra":       "DeepInfra",
    "fireworks":       "Fireworks",
    "together":        "Together",
    "novita":          "Novita",
    "friendli":        "Friendli",
    "hyperbolic":      "Hyperbolic",
    "lepton":          "Lepton",
    "mancer":          "Mancer",
    "minimax":         "MiniMax",
    "parasail":        "Parasail",
    "recursal":        "Recursal",
    "stepfun":         "Stepfun",
    "xai":             "xAI",
}


def _openrouter_provider_name(provider: str) -> str:
    """Return the OpenRouter display name for a provider key.

    Tries the ``_PROVIDER_NAMES`` lookup first; if the key is unknown the
    original string is returned unchanged (so callers may pass display names
    directly as well).
    """
    return _PROVIDER_NAMES.get(provider.lower().strip(), provider)


def _parse_provider_tag(tag: str) -> tuple[str, "Optional[str]"]:
    """Parse a provider tag like ``"deepinfra/bf16"`` into (slug, quantization).

    If *tag* contains no ``/``, the quantization is ``None``.
    """
    if "/" in tag:
        slug, quant = tag.split("/", 1)
        return slug.strip(), quant.strip()
    return tag.strip(), None


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single model, independent of API keys.

    All fields map directly to JSON config entries.  Defaults mirror the
    ``[defaults]`` section of the config file.

    Parameters
    ----------
    model_id:
        Full OpenRouter model identifier, e.g. ``"deepseek/deepseek-v3.2"``.
    provider:
        Preferred upstream provider, optionally with quantization in tag
        format.  Examples: ``"deepinfra/bf16"``, ``"novita"``,
        ``"DeepInfra"``.  :func:`resolve` splits the tag (if present) and
        normalises the provider slug to its OpenRouter display name.
    allow_fallbacks:
        Whether OpenRouter may fall back to other providers when the preferred
        one is unavailable.  ``False`` means strict pinning.
    max_output_tokens:
        Maximum tokens the model may generate.
    temperature:
        Sampling temperature.  ``None`` defers to the provider default.
    use_seed:
        When ``True`` and ``seed`` is set, includes ``"seed"`` in the request.
    seed:
        Seed value for reproducible sampling.  Only sent when ``use_seed`` is
        ``True``.
    reasoning_mode:
        ``"disabled"`` — send ``{"reasoning": {"effort": "none"}}`` to the
        OpenRouter API, suppressing chain-of-thought tokens.
        ``"low"`` — enable reasoning at low effort.
        ``"on"`` — enable reasoning (for models that only support on/off
        without effort levels, e.g. gemma-4); resolves to effort ``"low"``.
        Any other non-``"disabled"`` value also enables reasoning.
    reasoning_effort:
        Explicit reasoning effort string (``"low"``, ``"medium"``,
        ``"high"``).  When ``reasoning_mode`` is not ``"disabled"`` and this
        is ``None``, defaults to ``"low"``.
    """
    model_id: str
    provider: str
    allow_fallbacks: bool = False
    max_output_tokens: int = 8192
    temperature: Optional[float] = 0.0
    use_seed: bool = False
    seed: Optional[int] = None
    reasoning_mode: str = "disabled"
    reasoning_effort: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON config loader
# ---------------------------------------------------------------------------

@dataclass
class _Defaults:
    allow_fallbacks: bool = False
    max_output_tokens_cap: int = 8192
    reasoning_mode: str = "disabled"


def _apply_defaults(raw: dict, defaults: _Defaults) -> ModelConfig:
    """Build a ModelConfig from a raw JSON dict, filling gaps from defaults."""
    max_out = raw.get("max_output_tokens", defaults.max_output_tokens_cap)
    max_out = min(max_out, defaults.max_output_tokens_cap)

    return ModelConfig(
        model_id          = raw["model"],
        provider          = raw["provider"],
        allow_fallbacks   = raw.get("allow_fallbacks",   defaults.allow_fallbacks),
        max_output_tokens = max_out,
        temperature       = raw.get("temperature",       0.0),
        use_seed          = raw.get("use_seed",          False),
        seed              = raw.get("seed"),
        reasoning_mode    = raw.get("reasoning_mode",    defaults.reasoning_mode),
        reasoning_effort  = raw.get("reasoning_effort"),
    )


def load_models(path: "str | Path" = "evaluation_models.json") -> dict[str, ModelConfig]:
    """Load all model configs from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON config file.  Defaults to ``evaluation_models.json``
        in the current working directory.

    Returns
    -------
    dict[str, ModelConfig]
        Maps short model aliases (e.g. ``"deepseek-v3-2"``) to their parsed
        :class:`ModelConfig` objects.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    KeyError
        A model entry is missing the required ``"model"`` or ``"provider"``
        field.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Models config not found: {p}\n"
            "Use evaluation_models.json or pass an explicit model config path."
        )

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    raw_defaults = data.get("defaults", {})
    defaults = _Defaults(
        allow_fallbacks      = raw_defaults.get("allow_fallbacks",      False),
        max_output_tokens_cap= raw_defaults.get("max_output_tokens_cap", 8192),
        reasoning_mode       = raw_defaults.get("reasoning_mode",       "disabled"),
    )

    return {
        key: _apply_defaults(entry, defaults)
        for key, entry in data.get("models", {}).items()
    }


# ---------------------------------------------------------------------------
# resolve — the middleware lib function
# ---------------------------------------------------------------------------

def resolve(
    entry: ModelConfig,
    api_keys: list[str],
    provider_name: Optional[str] = None,
) -> tuple[str, ProviderConfig, CompletionKwargs]:
    """Convert a :class:`ModelConfig` into ready-to-use :func:`~llm.call_llm` arguments.

    This is the central lib function.  Both this project (reading from
    ``evaluation_models.json``) and external projects (constructing
    :class:`ModelConfig` directly) call this to get the arguments that
    :func:`~llm.call_llm`
    requires beyond ``client``, ``model_id``, and ``prompt``.

    Parameters
    ----------
    entry:
        The per-model configuration.
    api_keys:
        One or more OpenRouter API keys.  Multiple keys are rotated
        round-robin across concurrent calls.
    provider_name:
        Optional label for API-key rotation bookkeeping.  Defaults to
        ``"openrouter-<provider>"`` (e.g. ``"openrouter-deepinfra"``).

    Returns
    -------
    provider_name : str
    provider_config : ProviderConfig
    completion_kwargs : CompletionKwargs

    Example
    -------
    ::

        name, pconfig, kwargs = resolve(entry, api_keys=["<your-openrouter-api-key>"])
        response = await call_llm(
            client, name, pconfig, entry.model_id, prompt, kwargs
        )
    """
    slug, quantization = _parse_provider_tag(entry.provider)
    or_name = _openrouter_provider_name(slug)
    pname = provider_name or f"openrouter-{slug.lower()}"

    provider_config = ProviderConfig(
        api_keys           = api_keys,
        preferred_providers= [or_name] if or_name else [],
        quantizations      = [quantization] if quantization else [],
        allow_fallbacks    = entry.allow_fallbacks,
    )

    # Reasoning effort: "none" when disabled, else the configured effort or "low".
    if entry.reasoning_mode == "disabled":
        r_effort: Optional[str] = "none"
    else:
        r_effort = entry.reasoning_effort or "low"

    completion_kwargs = CompletionKwargs(
        max_tokens       = entry.max_output_tokens,
        temperature      = entry.temperature,
        reasoning_effort = r_effort,
        seed             = entry.seed if entry.use_seed else None,
    )

    return pname, provider_config, completion_kwargs
