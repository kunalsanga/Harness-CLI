"""
Provider discovery — normalizes provider ModelInfo into ModelProfile.

Integrates existing providers (OpenRouter, Ollama, LiteLLM) with ModelRegistry.
"""

from __future__ import annotations

import time
from typing import Any

from harness_core.models.types import ModelProfile
from harness_core.providers.base import ModelInfo, ModelProvider


def model_info_to_profile(info: ModelInfo) -> ModelProfile:
    """Convert a provider's ModelInfo to a ModelProfile."""
    return ModelProfile(
        model_id=info.id,
        provider=info.provider,
        display_name=info.name,
        context_window=info.context_window,
        input_cost_per_1k=info.cost_per_1k_input,
        output_cost_per_1k=info.cost_per_1k_output,
        supports_tools=info.supports_tools,
        supports_vision=info.supports_vision,
        supports_structured_output=info.supports_structured_output,
        supports_streaming=info.supports_streaming,
        supports_reasoning=info.supports_reasoning,
        is_free=info.is_free,
        is_local=info.is_local,
        tags=list(info.tags),
        discovered_at=time.time(),
        last_seen=time.time(),
    )


def estimate_capabilities(profile: ModelProfile) -> ModelProfile:
    """Estimate capabilities from provider metadata when no benchmark data exists.

    Uses heuristics from model ID, provider, and metadata.
    Does NOT fabricate scores — uses rough heuristics for routing.
    """
    model_id = profile.model_id.lower()
    provider = profile.provider.lower()

    # Heuristic coding capability from model naming
    coding_hints = {
        "deepseek-coder": 0.85,
        "deepseek-chat": 0.75,
        "codestral": 0.80,
        "qwen-coder": 0.80,
        "starcoder": 0.75,
        "codegemma": 0.70,
        "llama-3.1-70b": 0.70,
        "llama-3.1-8b": 0.55,
        "mistral-large": 0.75,
        "gpt-4": 0.85,
        "gpt-4o": 0.80,
        "gpt-4o-mini": 0.65,
        "claude-3.5-sonnet": 0.90,
        "claude-3-opus": 0.90,
        "gemini-pro": 0.75,
    }

    # Find best matching hint
    for hint, score in coding_hints.items():
        if hint in model_id:
            # Only set if not already benchmarked
            if not profile.capabilities.coding.is_measured:
                from harness_core.models.types import CapabilityConfidence, CapabilitySource
                profile.capabilities.coding.score = score
                profile.capabilities.coding.confidence = CapabilityConfidence.DECLARED
                profile.capabilities.coding.source = CapabilitySource.PROVIDER_METADATA
            break

    # Estimate tool use from provider metadata
    if profile.supports_tools and not profile.capabilities.tool_use.is_measured:
        from harness_core.models.types import CapabilityConfidence, CapabilitySource
        profile.capabilities.tool_use.score = 0.70
        profile.capabilities.tool_use.confidence = CapabilityConfidence.DECLARED
        profile.capabilities.tool_use.source = CapabilitySource.PROVIDER_METADATA

    return profile


async def discover_provider(
    provider: ModelProvider,
    estimate_caps: bool = True,
) -> list[ModelProfile]:
    """Discover models from a provider and convert to ModelProfiles."""
    try:
        model_infos = await provider.list_models()
    except Exception:
        return []

    profiles = []
    for info in model_infos:
        profile = model_info_to_profile(info)
        if estimate_caps:
            estimate_capabilities(profile)
        profiles.append(profile)

    return profiles
