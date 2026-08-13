"""Unit tests for build_registry(): per-provider registration, credential gating,
and Ollama's optional-key / opt-in-flag / unreachable-provider behavior.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_gateway.core.config import Settings
from llm_gateway.core.exceptions import ProviderError
from llm_gateway.models.api import ChatMessage, ChatRequest
from llm_gateway.providers.factory import build_registry
from llm_gateway.providers.openai import OpenAIProvider

# conftest.py sets OPENAI/GROQ/ANTHROPIC/MISTRAL API keys as real OS env vars for
# the rest of the suite (pydantic-settings reads env vars regardless of
# `_env_file=None`), so an "empty" Settings for these tests must null them out
# explicitly rather than relying on them being unset.
_NO_EXISTING_PROVIDER_KEYS: dict[str, object] = {
    "OPENAI_API_KEY": None,
    "GROQ_API_KEY": None,
    "ANTHROPIC_API_KEY": None,
    "MISTRAL_API_KEY": None,
}


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **_NO_EXISTING_PROVIDER_KEYS, **overrides)  # type: ignore[arg-type]


async def test_no_keys_configured_registers_nothing() -> None:
    registry = build_registry(_settings())
    assert registry.all() == []
    await registry.close()


async def test_gemini_absent_without_a_key() -> None:
    registry = build_registry(_settings())
    assert registry.get("gemini") is None
    await registry.close()


async def test_gemini_registered_with_key_pricing_and_models() -> None:
    registry = build_registry(_settings(GEMINI_API_KEY="k"))
    provider = registry.get("gemini")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert provider.api_key == "k"
    assert provider.config.supported_models == ["gemini-2.0-flash", "gemini-2.5-pro"]
    for model in provider.config.supported_models:
        pricing = provider.config.pricing_for(model)
        assert pricing.input_cost_per_1m > 0
        assert pricing.output_cost_per_1m > 0
    await registry.close()


async def test_deepseek_absent_without_a_key() -> None:
    registry = build_registry(_settings())
    assert registry.get("deepseek") is None
    await registry.close()


async def test_deepseek_registered_with_default_base_url_and_pricing() -> None:
    registry = build_registry(_settings(DEEPSEEK_API_KEY="k"))
    provider = registry.get("deepseek")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert provider.api_key == "k"
    assert provider.config.supported_models == ["deepseek-chat", "deepseek-reasoner"]
    for model in provider.config.supported_models:
        pricing = provider.config.pricing_for(model)
        assert pricing.input_cost_per_1m > 0
        assert pricing.output_cost_per_1m > 0
    await registry.close()


async def test_deepseek_base_url_is_overridable_eg_opencode_go() -> None:
    """DEEPSEEK_BASE_URL is just a Settings field: pointing it at any other
    OpenAI-compatible host (e.g. OpenCode Go) needs no separate provider."""
    registry = build_registry(
        _settings(
            DEEPSEEK_API_KEY="opencode-go-key",
            DEEPSEEK_BASE_URL="https://opencode.ai/zen/v1",
        )
    )
    provider = registry.get("deepseek")
    assert provider is not None
    assert provider.base_url == "https://opencode.ai/zen/v1"
    assert provider.api_key == "opencode-go-key"
    await registry.close()


async def test_ollama_not_registered_by_default() -> None:
    registry = build_registry(_settings())
    assert registry.get("ollama") is None
    await registry.close()


async def test_ollama_registered_when_enabled_with_no_api_key_required() -> None:
    registry = build_registry(_settings(OLLAMA_ENABLED=True))
    provider = registry.get("ollama")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.api_key is None
    assert provider.config.supported_models == ["llama3.2"]
    # No Authorization header is sent when no key is configured.
    assert provider._headers() == {}
    await registry.close()


async def test_ollama_is_zero_cost() -> None:
    registry = build_registry(_settings(OLLAMA_ENABLED=True))
    provider = registry.get("ollama")
    assert provider is not None
    pricing = provider.config.pricing_for("llama3.2")
    assert pricing.input_cost_per_1m == 0.0
    assert pricing.output_cost_per_1m == 0.0
    await registry.close()


async def test_ollama_optional_api_key_and_custom_base_url() -> None:
    registry = build_registry(
        _settings(
            OLLAMA_ENABLED=True,
            OLLAMA_API_KEY="secured-key",
            OLLAMA_BASE_URL="http://192.168.1.50:11434/v1",
        )
    )
    provider = registry.get("ollama")
    assert provider is not None
    assert provider.api_key == "secured-key"
    assert provider.base_url == "http://192.168.1.50:11434/v1"
    assert provider._headers() == {"Authorization": "Bearer secured-key"}
    await registry.close()


@respx.mock
async def test_ollama_unreachable_at_request_time_raises_provider_error_not_crash() -> None:
    """Graceful degradation at the point of use: an unreachable Ollama maps to the
    same ProviderError every other provider gets on a connection failure — it
    does not propagate as an unhandled exception or crash the caller."""
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    registry = build_registry(_settings(OLLAMA_ENABLED=True))
    provider = registry.get("ollama")
    assert provider is not None

    request = ChatRequest(model="llama3.2", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderError):
        await provider.chat_completion(request)

    await registry.close()
