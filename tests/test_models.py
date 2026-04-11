"""Tests for models.py — provider tag parsing and resolve()."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import _parse_provider_tag, resolve, ModelConfig, load_models


# ---------------------------------------------------------------------------
# _parse_provider_tag
# ---------------------------------------------------------------------------

def test_parse_provider_tag_with_quantization():
    assert _parse_provider_tag("deepinfra/bf16") == ("deepinfra", "bf16")
    assert _parse_provider_tag("novita/fp4") == ("novita", "fp4")


def test_parse_provider_tag_without_quantization():
    assert _parse_provider_tag("deepinfra") == ("deepinfra", None)
    assert _parse_provider_tag("DeepInfra") == ("DeepInfra", None)


def test_parse_provider_tag_strips_whitespace():
    assert _parse_provider_tag("  deepinfra / bf16 ") == ("deepinfra", "bf16")
    assert _parse_provider_tag("  novita  ") == ("novita", None)


# ---------------------------------------------------------------------------
# resolve — quantizations routing
# ---------------------------------------------------------------------------

def test_resolve_provider_tag_sets_quantizations():
    entry = ModelConfig(
        model_id="openai/gpt-oss-120b",
        provider="deepinfra/bf16",
    )
    name, pconfig, kwargs = resolve(entry, api_keys=["sk-test"])
    assert pconfig.preferred_providers == ["DeepInfra"]
    assert pconfig.quantizations == ["bf16"]
    assert pconfig.allow_fallbacks is False


def test_resolve_provider_without_tag_has_no_quantizations():
    entry = ModelConfig(
        model_id="openai/gpt-oss-120b",
        provider="deepinfra",
    )
    name, pconfig, kwargs = resolve(entry, api_keys=["sk-test"])
    assert pconfig.preferred_providers == ["DeepInfra"]
    assert pconfig.quantizations == []


def test_resolve_reasoning_effort_passthrough():
    entry = ModelConfig(
        model_id="google/gemma-4-31b-it",
        provider="novita/bf16",
        reasoning_mode="on",
    )
    _, pconfig, kwargs = resolve(entry, api_keys=["sk-test"])
    # "on" without explicit effort → defaults to "low" (just enables thinking)
    assert kwargs.reasoning_effort == "low"
    assert pconfig.preferred_providers == ["Novita"]
    assert pconfig.quantizations == ["bf16"]


def test_resolve_reasoning_disabled():
    entry = ModelConfig(
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider="deepinfra/fp8",
        reasoning_mode="disabled",
    )
    _, pconfig, kwargs = resolve(entry, api_keys=["sk-test"])
    assert kwargs.reasoning_effort == "none"
    assert pconfig.preferred_providers == ["DeepInfra"]
    assert pconfig.quantizations == ["fp8"]


def test_official_config_disables_reasoning_for_gemma():
    models = load_models(os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_models.json"))
    entry = models["gemma-4-31b-it"]
    _, pconfig, kwargs = resolve(entry, api_keys=["sk-test"])
    assert kwargs.reasoning_effort == "none"
    assert pconfig.preferred_providers == ["Novita"]
    assert pconfig.quantizations == ["bf16"]
