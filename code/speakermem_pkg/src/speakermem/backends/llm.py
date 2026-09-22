"""LLM backends: an abstract interface and an OpenAI-compatible implementation.

Memory writing can disable reasoning while answering can enable it through the thinking option.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLM(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, *, json_mode: bool = False,
             thinking: bool = True, temperature: float = 0.0,
             max_tokens: int = 100_000, model: Optional[str] = None) -> str:
        ...


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible endpoint. thinking=False disables the reasoning field for compatible DeepSeek endpoints."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 default_model: str = "deepseek-v4-flash", disable_thinking_extra_body: bool = True,
                 max_retries: int = 5, trust_env: bool = True):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY"
        url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")


        okw = dict(api_key=key, base_url=url, max_retries=max_retries, timeout=900.0)
        if not trust_env:
            import httpx
            okw["http_client"] = httpx.Client(trust_env=False, timeout=120)
        self._client = OpenAI(**okw)
        self.default_model = default_model
        self._disable_thinking_extra_body = disable_thinking_extra_body

    def chat(self, system, user, *, json_mode=False, thinking=True,
             temperature=0.0, max_tokens=100_000, model=None) -> str:
        kw: Dict = dict(model=model or self.default_model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if not thinking and self._disable_thinking_extra_body:
            kw["extra_body"] = {"thinking": {"type": "disabled"}}




        out, ch = "", None
        for attempt in range(3):
            if attempt == 2:
                kw["extra_body"] = {"thinking": {"type": "disabled"}}
            ch = self._client.chat.completions.create(**kw).choices[0]
            out = ch.message.content or ""
            if out.strip():
                break
            print(f"    [llm retry {attempt+1}/3] content is empty (finish={ch.finish_reason}, "
                  f"max_tokens={kw['max_tokens']})", flush=True)
        if not out.strip():
            print("    [llm warn] still empty after three attempts", flush=True)
        return out


class LocalHFLLM(LLM):
    """Local HuggingFace model (transformers) for CPU or GPU; loads weights directly without a serving process.
    Suitable for small local models such as Qwen2.5-0.5B/1.5B/3B. With a GPU it uses cuda and bf16; on CPU it falls back to fp32.
    json_mode relies on the prompt contract and robust parsing because local models do not provide native JSON mode."""

    def __init__(self, model_name: str, device: Optional[str] = None,
                 dtype: Optional[str] = None, max_new_tokens_cap: int = 3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = (torch.bfloat16 if (dtype == "bf16" or (dtype is None and self._dev == "cuda"))
              else torch.float32)
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=td,
            device_map=("auto" if self._dev == "cuda" else None))
        if self._dev != "cuda":
            self._model = self._model.to("cpu")
        self._model.eval()
        self._cap = max_new_tokens_cap

    def chat(self, system, user, *, json_mode=False, thinking=True,
             temperature=0.0, max_tokens=100_000, model=None) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking)
        except TypeError:
            text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        gen = dict(max_new_tokens=min(int(max_tokens), self._cap),
                   do_sample=temperature > 0,
                   pad_token_id=self._tok.eos_token_id)
        if temperature > 0:
            gen["temperature"] = float(temperature)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new, skip_special_tokens=True)
