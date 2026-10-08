"""Configuration-driven provider construction for CLI/runtime entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness_core.config.credentials import CredentialResolver
from harness_core.providers.base import ModelProvider


def _provider_configs(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        # Preserve the existing default until users choose another provider.
        return [{"id": "openrouter", "enabled": True}]
    if isinstance(raw, dict):
        entries = []
        for name, value in raw.items():
            config = dict(value) if isinstance(value, dict) else {}
            entries.append({"id": name, **config})
        return entries
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict) and item.get("id")]
    raise ValueError("providers configuration must be a mapping or list")


def load_project_provider_config(project_root: str | Path) -> dict[str, Any]:
    """Load the project YAML used to configure runtime providers."""
    path = Path(project_root) / ".harness" / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not read provider configuration from {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Project configuration must be a YAML mapping")
    return data


def create_providers(
    config: dict[str, Any] | None = None,
    *,
    provider_ids: list[str] | None = None,
) -> list[ModelProvider]:
    """Create configured adapters using existing credentials and config.

    Expected form::

        providers:
          openrouter: {enabled: true}
          openai: {enabled: true, model: gpt-4o-mini}
          compatible-local: {type: openai-compatible, endpoint: http://localhost:8000/v1,
                             model: qwen, api_key_env: LOCAL_LLM_KEY}
    """
    from harness_core.providers.http_adapters import (
        AnthropicProvider,
        GeminiProvider,
        OpenAICompatibleProvider,
        OpenAIProvider,
    )
    from harness_core.providers.groq import GroqProvider
    from harness_core.providers.litellm import LiteLLMProvider
    from harness_core.providers.ninerouter import NineRouterProvider
    from harness_core.providers.nvidia import NvidiaProvider
    from harness_core.providers.ollama import OllamaProvider
    from harness_core.providers.openrouter import OpenRouterProvider

    root = config or {}
    configured = _provider_configs(root.get("providers"))
    selected = set(provider_ids or [])
    resolver = CredentialResolver()
    providers: list[ModelProvider] = []
    for spec in configured:
        name = str(spec.get("id", "")).strip().lower()
        if name == "google":
            name = "gemini"
        elif name == "ninerouter":
            name = "9router"
        if not name or spec.get("enabled", True) is False or (selected and name not in selected):
            continue
        endpoint = spec.get("endpoint") or spec.get("base_url")
        model = spec.get("model") or spec.get("default_model")
        env_name = spec.get("api_key_env")
        api_key = os.environ.get(str(env_name), "") if env_name else None
        if not api_key:
            credential = resolver.resolve(name)
            api_key = credential.api_key if credential else None
        else:
            credential = resolver.resolve(name)
        if credential:
            endpoint = endpoint or credential.base_url
            model = model or credential.default_model

        options = {"api_key": api_key, "base_url": endpoint, "default_model": model}
        if name == "openrouter":
            provider = OpenRouterProvider(api_key=api_key, base_url=endpoint, default_model=model)
        elif name == "openai":
            provider = OpenAIProvider(**options)
        elif name == "anthropic":
            provider = AnthropicProvider(api_key=api_key, base_url=endpoint or "https://api.anthropic.com", default_model=model or "claude-sonnet-4-20250514")
        elif name in {"gemini", "google"}:
            provider = GeminiProvider(api_key=api_key, base_url=endpoint or "https://generativelanguage.googleapis.com/v1beta", default_model=model or "gemini-2.5-flash")
        elif name == "nvidia":
            provider = NvidiaProvider(api_key=api_key, base_url=endpoint, default_model=model)
        elif name == "groq":
            provider = GroqProvider(api_key=api_key, base_url=endpoint, default_model=model)
        elif name == "ollama":
            provider = OllamaProvider(base_url=endpoint, default_model=model)
        elif name in {"9router", "ninerouter"}:
            provider = NineRouterProvider(api_key=api_key, base_url=endpoint or "http://localhost:20128/v1")
        elif name == "litellm":
            provider = LiteLLMProvider(api_key=api_key, base_url=endpoint, default_model=model)
        elif str(spec.get("type", "")).lower() in {"openai-compatible", "openai_compatible"}:
            if not endpoint:
                raise ValueError(f"Provider {name!r} requires an endpoint")
            provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=endpoint,
                default_model=model,
                provider_name=name,
            )
        else:
            raise ValueError(f"Unsupported provider configuration: {name!r}")
        from harness_core.providers.normalized import NormalizedModelProvider
        provider = NormalizedModelProvider(provider)
        provider.configured_models = _model_infos(name, spec)
        if model and not any(item.id == model for item in provider.configured_models):
            provider.configured_models.append(_model_infos(name, {"models": [{
                "id": model,
                "supports_tools": spec.get("supports_tools"),
                "supports_vision": spec.get("supports_vision"),
                "supports_structured_output": spec.get("supports_structured_output"),
                "supports_streaming": spec.get("supports_streaming"),
                "supports_reasoning": spec.get("supports_reasoning"),
                "context_window": spec.get("context_window"),
                "is_free": spec.get("is_free", False),
                "is_local": name == "ollama" or bool(spec.get("is_local", False)),
            }]})[0])
        providers.append(provider)
    if not providers:
        raise ValueError("No enabled, recognized model providers are configured")
    return providers


def _model_infos(provider: str, spec: dict[str, Any]):
    from harness_core.providers.base import ModelInfo

    result = []
    for item in spec.get("models", []) or []:
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict) or not item.get("id"):
            continue
        result.append(ModelInfo(
            id=str(item["id"]), name=str(item.get("name") or item["id"]), provider=provider,
            context_window=int(item.get("context_window") or 0),
            supports_tools=item.get("supports_tools"),
            supports_vision=item.get("supports_vision"),
            supports_structured_output=item.get("supports_structured_output"),
            supports_streaming=item.get("supports_streaming"),
            supports_reasoning=item.get("supports_reasoning"),
            is_free=bool(item.get("is_free", False)), is_local=bool(item.get("is_local", False)),
        ))
    return result
