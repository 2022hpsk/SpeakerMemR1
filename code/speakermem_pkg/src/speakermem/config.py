"""Configuration: dataclasses collect the tunable retrieval, ingestion, and model settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WriterConfig:
    """A_mem ingestion configuration."""
    enabled: bool = True
    keep_verbatim: bool = True
    derive_every: str = "session"
    derive_window: int = 1
    model: str = "deepseek-v4-flash"
    thinking: bool = False




    max_tokens: int = 16384




    state_max_chars: int = 60000

    state_per_owner_k: int = 40
    verbatim_confidence: float = 0.6

    provenance: bool = True
    provenance_m: int = 3
    provenance_thresh: float = 0.25


@dataclass
class RetrieverConfig:
    """A_ret two-path retrieval configuration. S1 uses the raw track and S2 uses the derived track; their budgets are independent."""
    top_k: int = 10

    s1_enabled: bool = True
    s2_enabled: bool = True

    s1_k: int = 8
    s1_recall_n: int = 40
    llm_select: bool = True
    select_model: str = "deepseek-v4-flash"
    select_thinking: bool = False
    select_max_tokens: int = 100_000
    expand_max_rounds: int = 2
    expand_max_entries: int = 3
    context_window: int = 5




    ask_enabled: bool = True
    ask_max_rounds: int = 1
    ask_recall_n: int = 20

    ask_thinking: bool = False
    ask_max_tokens: int = 100_000

    s2_k: int = 2
    s2_derived_tau: float = 0.30


    s2_source_k: int = 1

    s2_layers: tuple = ()
    s2_fallback_raw: bool = True
    s2_raw_tau: float = 0.25
    allow_all_plus_group: bool = True


@dataclass
class AnswererConfig:
    enabled: bool = True
    model: str = "deepseek-v4-flash"
    thinking: bool = True


    max_tokens: int = 100_000


@dataclass
class SpeakerMemConfig:
    embed_model: str = "all-MiniLM-L6-v2"
    writer: WriterConfig = field(default_factory=WriterConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    answerer: AnswererConfig = field(default_factory=AnswererConfig)
