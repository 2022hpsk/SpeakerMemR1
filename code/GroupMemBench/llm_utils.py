from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Sequence

from openai import AzureOpenAI, OpenAI

AZURE_OPENAI_PROVIDER = "azure_openai"
OPENAI_PROVIDER = "openai"
DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"
MODEL_MAX_RETRIES = 5
MODEL_RETRY_DELAY_SECONDS = 10


def normalize_llm_provider(provider: Optional[str]) -> str:
    value = (provider or os.environ.get("LLM_PROVIDER") or AZURE_OPENAI_PROVIDER).strip().lower()
    aliases = {
        "azure": AZURE_OPENAI_PROVIDER,
        "azure_openai": AZURE_OPENAI_PROVIDER,
        "aoai": AZURE_OPENAI_PROVIDER,
        "openai": OPENAI_PROVIDER,
        "oai": OPENAI_PROVIDER,
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Expected one of: {AZURE_OPENAI_PROVIDER}, {OPENAI_PROVIDER}."
        )
    return normalized


def resolve_api_version(api_version: Optional[str] = None) -> str:
    return api_version or os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)


def resolve_api_key(provider: Optional[str] = None, api_key: Optional[str] = None) -> Optional[str]:
    normalized = normalize_llm_provider(provider)
    if api_key:
        return api_key
    if normalized == AZURE_OPENAI_PROVIDER:
        return os.environ.get("AZURE_OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def resolve_base_url(provider: Optional[str] = None, base_url: Optional[str] = None) -> Optional[str]:
    normalized = normalize_llm_provider(provider)
    if base_url:
        return base_url
    if normalized == AZURE_OPENAI_PROVIDER:
        return os.environ.get("AZURE_OPENAI_ENDPOINT")
    return os.environ.get("OPENAI_BASE_URL")


def create_chat_client(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_ad_token_provider: Any = None,
) -> Any:
    normalized = normalize_llm_provider(provider)
    if normalized == AZURE_OPENAI_PROVIDER:
        endpoint = resolve_base_url(normalized, azure_endpoint)
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT for Azure OpenAI client creation.")
        kwargs: Dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": resolve_api_version(api_version),
            "max_retries": 0,
        }
        if azure_ad_token_provider is not None:
            kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        else:
            key = resolve_api_key(normalized, api_key)
            if not key:
                raise ValueError("Missing AZURE_OPENAI_API_KEY for Azure OpenAI client creation.")
            kwargs["api_key"] = key
        return AzureOpenAI(**kwargs)

    key = resolve_api_key(normalized, api_key)
    if not key:
        raise ValueError("Missing OPENAI_API_KEY for OpenAI client creation.")
    kwargs = {"api_key": key, "max_retries": 0}
    resolved_base_url = resolve_base_url(normalized, base_url)
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def with_retry(func: Any, *args: Any, retries: int = MODEL_MAX_RETRIES, delay_seconds: int = MODEL_RETRY_DELAY_SECONDS, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry wrapper exhausted without capturing an exception.")


def model_uses_max_completion_tokens(model: str) -> bool:
    name = (model or "").strip().lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    if max_tokens is not None:
        if model_uses_max_completion_tokens(model):
            # Reasoning models (gpt-5 / o1 / o3 / o4) consume hidden reasoning
            # tokens out of max_completion_tokens before emitting output. The
            # callsites pick budgets tuned for gpt-4o, so bump them up with a
            # multiplier + floor to keep reasoning-heavy short calls alive.
            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
        else:
            kwargs["max_tokens"] = max_tokens
    if temperature is not None and not model_uses_max_completion_tokens(model):
        kwargs["temperature"] = temperature
    kwargs.update(extra)
    return kwargs


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Any:
    kwargs = build_chat_completion_kwargs(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    return with_retry(client.chat.completions.create, **kwargs)


def chat_completion_text(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> str:
    response = chat_completion(
        client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    content = response.choices[0].message.content if response.choices else ""
    return content or ""


def build_mem0_llm_config(
    *,
    provider: Optional[str],
    model: str,
    api_key: Optional[str],
    api_version: Optional[str],
    azure_endpoint: Optional[str],
    openai_base_url: Optional[str],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    normalized = normalize_llm_provider(provider)
    config: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if normalized == AZURE_OPENAI_PROVIDER:
        endpoint = resolve_base_url(normalized, azure_endpoint)
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT for mem0 Azure OpenAI config.")
        azure_kwargs: Dict[str, Any] = {
            "azure_deployment": model,
            "azure_endpoint": endpoint,
            "api_version": resolve_api_version(api_version),
        }
        key = resolve_api_key(normalized, api_key)
        if key:
            azure_kwargs["api_key"] = key
        config["azure_kwargs"] = azure_kwargs
        return {"provider": AZURE_OPENAI_PROVIDER, "config": config}

    key = resolve_api_key(normalized, api_key)
    if key:
        config["api_key"] = key
    resolved_openai_base_url = resolve_base_url(normalized, openai_base_url)
    if resolved_openai_base_url:
        config["openai_base_url"] = resolved_openai_base_url
    return {"provider": OPENAI_PROVIDER, "config": config}
