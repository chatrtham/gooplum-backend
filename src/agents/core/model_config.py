"""Model configuration with predefined presets for agents."""

import os

from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Predefined model presets
MODEL_PRESETS = {
    # Anthropic models
    "claude-sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "description": "Claude Sonnet 4.5 - Best balance of speed and capability",
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "description": "Claude Haiku 4.5 - Fast and cost-effective",
    },
    "claude-opus": {
        "provider": "anthropic",
        "model": "claude-opus-4",
        "description": "Claude Opus 4 - Most capable",
    },
    # OpenAI models
    "gpt-5": {
        "provider": "openai",
        "model": "gpt-5",
        "description": "GPT-5 - OpenAI's flagship model",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "description": "GPT-5 Mini - Fast and affordable next-gen model",
    },
    # Google models
    "gemini-2-flash": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "description": "Gemini 2.0 Flash - Fast and capable",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash - Fast and capable Google model",
    },
    "gemini-2-pro": {
        "provider": "google",
        "model": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro - Most capable Google model",
    },
    # ZAI (custom OpenAI-compatible endpoint)
    "zai-glm": {
        "provider": "zai",
        "model": "glm-4.6",
        "description": "ZAI GLM 4.6 - Custom endpoint model",
    },
}


def get_available_presets() -> list[dict]:
    """Get list of available model presets with descriptions."""
    return [{"name": name, **preset} for name, preset in MODEL_PRESETS.items()]


def get_model_from_preset(preset_name: str) -> BaseChatModel:
    """Create a chat model from a preset name.

    Args:
        preset_name: Name of the preset (e.g., "claude-sonnet", "gpt-4o")

    Returns:
        Configured BaseChatModel instance

    Raises:
        ValueError: If preset is not found
    """
    if preset_name not in MODEL_PRESETS:
        available = list(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    preset = MODEL_PRESETS[preset_name]
    provider = preset["provider"]
    model_name = preset["model"]

    if provider == "anthropic":
        return init_chat_model(
            f"anthropic:{model_name}",
            temperature=0,
        )

    elif provider == "openai":
        return init_chat_model(
            f"openai:{model_name}",
            temperature=0,
        )

    elif provider == "google":
        # Google models via OpenAI-compatible endpoint
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    elif provider == "zai":
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("ZAI_API_KEY"),
            openai_api_base="https://api.z.ai/api/coding/paas/v4/",
        )

    else:
        raise ValueError(f"Unknown provider '{provider}' for preset '{preset_name}'")


def get_model_from_config(
    preset: str,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create a model from agent config.

    This is the main entry point used by agent_factory.

    Args:
        preset: Model preset name
        temperature: Temperature setting

    Returns:
        Configured BaseChatModel instance
    """
    return get_model_from_preset(preset, temperature)
