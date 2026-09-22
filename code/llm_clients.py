"""LLM client adapters used by the GroupMemBench evaluation.

All LLM calls in the upstream UCSB-NLP-Chang/GroupMemBench code use the OpenAI-style interface:

    client.chat.completions.create(model=, messages=[...], max_tokens=/
        max_completion_tokens=, temperature=) -> resp.choices[0].message.content

The default agent and judge model is ``gpt-5`` through OpenAI or Azure OpenAI.
Anthropic is also supported as an LLM endpoint
at ``ANTHROPIC_BASE_URL=https://api.anthropic.com`` through its native Messages API,
rather than ``chat.completions``.

This module provides ``make_client(...)``, returning a client with the same ``.chat.completions.create``
surface. The backend can be selected among several providers without modifying the cloned upstream repository
:

  * ``anthropic``       -> Anthropic Messages API (requires ANTHROPIC_API_KEY).
  * ``openai``/``local`` -> any OpenAI-compatible endpoint, such as a local vLLM
                          service (``vllm serve Qwen/Qwen2.5-7B-Instruct``). Set
                          OPENAI_BASE_URL + OPENAI_API_KEY(a nonempty key is sufficient for a local service).

Use it in an evaluation runner as follows:

    from llm_clients import make_client
    client = make_client(provider="anthropic")

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence





@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Response:
    choices: List[_Choice]
    usage: _Usage





def _split_system(messages: Sequence[Dict[str, Any]]):
    """OpenAI stores the system prompt in the messages list, while Anthropic expects it as a top-level `system`
    argument. This helper extracts the system prompt and passes the remaining messages through unchanged."""
    system_parts: List[str] = []
    convo: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content if isinstance(content, str) else str(content))
        else:
            convo.append({"role": role, "content": content})

    if not convo or convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "."})
    return "\n\n".join(system_parts), convo


class _AnthropicCompletions:
    def __init__(self, sdk_client: Any):
        self._c = sdk_client

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **_ignored: Any,
    ) -> _Response:
        system, convo = _split_system(messages)
        budget = max_tokens or max_completion_tokens or 100000
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(budget),
            "messages": convo,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        resp = self._c.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = _Usage(
            prompt_tokens=getattr(resp.usage, "input_tokens", 0),
            completion_tokens=getattr(resp.usage, "output_tokens", 0),
            total_tokens=getattr(resp.usage, "input_tokens", 0)
            + getattr(resp.usage, "output_tokens", 0),
        )
        return _Response(choices=[_Choice(message=_Message(content=text))], usage=usage)


class _Chat:
    def __init__(self, completions: Any):
        self.completions = completions


class AnthropicOpenAIClient:
    """Exposes `.chat.completions.create(...)` and uses the Anthropic SDK underneath."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 max_retries: int = 5):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No ANTHROPIC_API_KEY is available.The ANTHROPIC_BASE_URL endpoint is authenticated with "
                "a Claude Code OAuth token and cannot be used for batch API calls. Set "
                "ANTHROPIC_API_KEY in `.env` (or use --llm-provider local with a local "
                "vLLM service) to run the QA agent and judge."
            )
        client_kwargs: Dict[str, Any] = {"api_key": key, "max_retries": max_retries}
        if base_url or os.environ.get("ANTHROPIC_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.environ["ANTHROPIC_BASE_URL"]
        self._sdk = anthropic.Anthropic(**client_kwargs)
        self.chat = _Chat(_AnthropicCompletions(self._sdk))





def make_client(provider: str, *, api_key: Optional[str] = None,
                base_url: Optional[str] = None) -> Any:
    """Return a client exposing `.chat.completions.create` for the selected provider.

    Providers:
      - "anthropic": Anthropic Messages API (requires ANTHROPIC_API_KEY).
      - "deepseek": DeepSeek, an OpenAI-compatible endpoint at https://api.deepseek.com, requiring DEEPSEEK_API_KEY.
        requires DEEPSEEK_API_KEY; the default model is deepseek-v4-flash).
      - "openai" | "local": OpenAI-compatible endpoint; a nonempty key is sufficient for a local vLLM service.
        Local vLLM service; a nonempty key is sufficient).
    """
    p = (provider or "").strip().lower()
    if p == "anthropic":
        return AnthropicOpenAIClient(api_key=api_key, base_url=base_url)
    if p == "zju":

        from openai import OpenAI
        import httpx
        key = api_key or os.environ.get("ZJU_API_KEY")
        url = base_url or os.environ.get("ZJU_BASE_URL") or "https://api.zju.edu.cn/model/modelai/v1"
        if not key:
            raise RuntimeError("No ZJU_API_KEY is available. Set ZJU_API_KEY in `.env` before using this provider.")
        return OpenAI(api_key=key, base_url=url, max_retries=5,
                      http_client=httpx.Client(trust_env=False, timeout=120))
    if p == "deepseek":

        from openai import OpenAI

        key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No DEEPSEEK_API_KEY is available. Set DEEPSEEK_API_KEY in `.env` to run "
                "the QA agent and judge (default model: deepseek-v4-flash)."
            )
        url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com")
        return OpenAI(api_key=key, base_url=url, max_retries=5)
    if p in ("openai", "local", "vllm"):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs: Dict[str, Any] = {"api_key": key, "max_retries": 5}
        if url:
            kwargs["base_url"] = url
        return OpenAI(**kwargs)
    raise ValueError(f"Unknown provider: {provider!r}. Available providers: anthropic | deepseek | openai | local.")
