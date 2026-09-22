"""A_ans answer agent: answer from S1 raw passages and the S2 structured-memory slice.

The output shape is constrained by the supplied evidence rather than by a template label:
  rows=["GROUP"] means the slice contains only the GROUP row, so the answerer naturally returns a single value.
"""
from __future__ import annotations

from typing import List, Optional

from ..types import MemoryEntry
from ..backends.llm import LLM
from ..config import AnswererConfig
from ..prompts import ANSWER_SYSTEM, ANSWER_USER_TMPL


def render_passages(entries: List[MemoryEntry]) -> str:
    """S1 raw passages."""
    if not entries:
        return "(none)"
    return "\n".join(e.render(i) for i, e in enumerate(entries, 1))


class Answerer:
    def __init__(self, llm_provider, cfg: AnswererConfig,
                 system_prompt: str = ANSWER_SYSTEM, user_tmpl: str = ANSWER_USER_TMPL):
        self._get_llm, self.cfg = llm_provider, cfg
        self.system_prompt, self.user_tmpl = system_prompt, user_tmpl

    def answer(self, question: str, entries: List[MemoryEntry],
               slice_text: str = "(empty)") -> str:
        user = self.user_tmpl.format(question=question,
                                     passages=render_passages(entries),
                                     slice=slice_text)
        return self._get_llm().chat(self.system_prompt, user, thinking=self.cfg.thinking,
                                    temperature=0.0, max_tokens=self.cfg.max_tokens,
                                    model=self.cfg.model)
