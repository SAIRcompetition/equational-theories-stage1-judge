"""Tests for llm.py — key rotation, model profiles, helper functions."""
import json
import sys
import os
from types import SimpleNamespace

import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm import (
    CompletionKwargs,
    LlmResponse,
    ProviderConfig,
    _call_once,
    _choose_api_key, _KEY_INDICES, _KEY_LOCK,
    _openrouter_model_profile, _ReasoningMode,
    normalize_openrouter_model_id,
    _OPENROUTER_PROFILES,
    call_llm,
)


# ---------------------------------------------------------------------------
# choose_provider_api_key
# ---------------------------------------------------------------------------

def test_choose_provider_api_key_round_robin():
    cfg = ProviderConfig(api_keys=["k1", "k2", "k3"])
    # Use a fresh provider name to avoid interference with other tests.
    provider = "provider-rr-test-py"
    # Reset state
    with _KEY_LOCK:
        _KEY_INDICES.pop(provider, None)

    assert _choose_api_key(provider, cfg) == "k1"
    assert _choose_api_key(provider, cfg) == "k2"
    assert _choose_api_key(provider, cfg) == "k3"
    assert _choose_api_key(provider, cfg) == "k1"  # wraps around


def test_choose_provider_api_key_filters_empty_values():
    cfg = ProviderConfig(api_keys=["", "  ", "k-final"])
    provider = "provider-empty-test-py"
    with _KEY_LOCK:
        _KEY_INDICES.pop(provider, None)

    assert _choose_api_key(provider, cfg) == "k-final"


def test_choose_provider_api_key_empty_list_returns_none():
    cfg = ProviderConfig(api_keys=[])
    assert _choose_api_key("no-keys", cfg) is None


def test_choose_provider_api_key_single_key():
    cfg = ProviderConfig(api_keys=["only-key"])
    assert _choose_api_key("single-key", cfg) == "only-key"
    assert _choose_api_key("single-key", cfg) == "only-key"


# ---------------------------------------------------------------------------
# openrouter_model_profile
# ---------------------------------------------------------------------------

_EXPECTED_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "z-ai/glm-5",
    "openai/gpt-5.4",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-oss-120b",
    "x-ai/grok-4.1-fast",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-m2.5",
    "qwen/qwen3.5-397b-a17b",
    "stepfun/step-3.5-flash",
    "google/gemma-4-31b-it",
    "meta-llama/llama-3.3-70b-instruct",
]


def test_openrouter_model_profile_covers_known_models():
    for model_id in _EXPECTED_MODELS:
        profile = _openrouter_model_profile(model_id)
        assert profile is not None, f"missing profile for {model_id}"
        if model_id in {
            "openai/gpt-oss-120b",
            "google/gemma-4-31b-it",
            "meta-llama/llama-3.3-70b-instruct",
        }:
            assert profile.max_output_tokens == 8192, f"unexpected cap for {model_id}"
        else:
            assert profile.max_output_tokens == 32768, f"unexpected cap for {model_id}"


def test_openrouter_model_profile_reasoning_modes():
    gpt_54 = _openrouter_model_profile("openai/gpt-5.4")
    assert gpt_54 is not None
    assert gpt_54.reasoning_mode == _ReasoningMode.DISABLED
    assert gpt_54.reasoning_effort is None

    gemini = _openrouter_model_profile("google/gemini-3.1-pro-preview")
    assert gemini is not None
    assert gemini.reasoning_mode == _ReasoningMode.LOW
    assert gemini.reasoning_effort == "low"

    # gpt-oss-120b is a reasoning model despite not matching o1/o3/o4/gpt-5 prefix
    gpt_oss = _openrouter_model_profile("openai/gpt-oss-120b")
    assert gpt_oss is not None
    assert gpt_oss.reasoning_mode == _ReasoningMode.LOW
    assert gpt_oss.reasoning_effort == "low"
    assert gpt_oss.seed == 0

    gemma = _openrouter_model_profile("google/gemma-4-31b-it")
    assert gemma is not None
    assert gemma.reasoning_mode == _ReasoningMode.DISABLED
    assert gemma.reasoning_effort is None


def test_openrouter_model_profile_unknown_returns_none():
    assert _openrouter_model_profile("unknown/model-xyz") is None


# ---------------------------------------------------------------------------
# normalize_openrouter_model_id
# ---------------------------------------------------------------------------

def test_normalize_openrouter_model_id_strips_provider_hint_suffix():
    assert normalize_openrouter_model_id("deepseek/deepseek-v3.2@owen-openrouter") == "deepseek/deepseek-v3.2"
    assert normalize_openrouter_model_id(" openai/gpt-5-mini@openrouter-main ") == "openai/gpt-5-mini"
    assert normalize_openrouter_model_id("openai/gpt-5-mini") == "openai/gpt-5-mini"
    # Bare name without slash: leave unchanged
    assert normalize_openrouter_model_id("custom-model@provider-main") == "custom-model@provider-main"


@pytest.mark.asyncio
async def test_call_once_normalizes_list_message_content():
    class FakeResponse:
        status_code = 200
        text = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "VERDICT: TRUE"},
                                {"type": "reasoning", "text": "ignored"},
                                "\nREASONING: concise",
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "provider": "DeepInfra",
            }
        )

        def json(self):
            return json.loads(self.text)

    class FakeClient:
        async def post(self, *args, **kwargs):
            return FakeResponse()

    response = await _call_once(
        FakeClient(),
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="sk-test",
        model_id="openai/gpt-oss-120b",
        body={},
    )
    assert response.text == "VERDICT: TRUE\nREASONING: concise"
    assert response.actual_provider == "DeepInfra"


@pytest.mark.asyncio
async def test_call_once_retries_malformed_2xx_response_body():
    class MalformedResponse:
        status_code = 200
        text = "not-json"

        def json(self):
            raise ValueError("bad json")

    class ValidResponse:
        status_code = 200
        text = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "VERDICT: TRUE"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "provider": "DeepInfra",
            }
        )

        def json(self):
            return json.loads(self.text)

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            return MalformedResponse() if self.calls == 1 else ValidResponse()

    client = FakeClient()
    response = await _call_once(
        client,
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="sk-test",
        model_id="openai/gpt-oss-120b",
        body={},
    )

    assert client.calls == 2
    assert response.text == "VERDICT: TRUE"
    assert response.actual_provider == "DeepInfra"


@pytest.mark.asyncio
async def test_call_llm_kwargs_temperature_overrides_profile(monkeypatch):
    captured = {}

    async def fake_call_once(client, url, api_key, model_id, body):
        captured.update(body)
        return LlmResponse(text="VERDICT: TRUE", finish_reason="stop")

    monkeypatch.setattr("llm._call_once", fake_call_once)

    response = await call_llm(
        client=SimpleNamespace(),
        provider_name="openrouter-main",
        provider_config=ProviderConfig(api_keys=["sk-test"]),
        model_id="meta-llama/llama-3.3-70b-instruct",
        prompt="hello",
        kwargs=CompletionKwargs(max_tokens=128, temperature=0.7, reasoning_effort="none"),
    )
    assert response.text == "VERDICT: TRUE"
    assert captured["temperature"] == 0.7
    assert captured["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_call_llm_uses_profile_cap_when_kwargs_max_tokens_is_unset(monkeypatch):
    captured = {}

    async def fake_call_once(client, url, api_key, model_id, body):
        captured.update(body)
        return LlmResponse(text="VERDICT: TRUE", finish_reason="stop")

    monkeypatch.setattr("llm._call_once", fake_call_once)

    response = await call_llm(
        client=SimpleNamespace(),
        provider_name="openrouter-main",
        provider_config=ProviderConfig(api_keys=["sk-test"]),
        model_id="openai/gpt-oss-120b",
        prompt="hello",
        kwargs=CompletionKwargs(),
    )
    assert response.text == "VERDICT: TRUE"
    assert captured["max_tokens"] == 8192
