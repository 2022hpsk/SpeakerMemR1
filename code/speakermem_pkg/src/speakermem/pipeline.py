"""SpeakerMemory: a user-facing facade for one-line integration with the speaker-grounded memory framework.

    from speakermem import SpeakerMemory
    mem = SpeakerMemory(persist_path="mem.db")
    mem.ingest(messages)
    mem.add({"speaker":"Alice","content":"...","session":"s1","ts":"..."})
    mem.flush()
    entries = mem.retrieve("Who handles evaluation?", asker="Bob", k=10)
    answer  = mem.answer("Who handles evaluation?", asker="Bob")
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from .types import Message, MemoryEntry
from .config import SpeakerMemConfig
from .memory import SpeakerAwareMemory
from .backends.embedder import Embedder, SentenceTransformerEmbedder
from .backends.llm import LLM, OpenAICompatLLM
from .backends.vector_index import VectorIndex, NumpyIndex
from .backends.store import MemoryStore, SqliteStore, make_store
from .agents import Writer, Retriever, Answerer

MsgLike = Union[Message, Dict[str, Any]]


class SpeakerMemory:
    def __init__(self, *,
                 persist_path: Optional[str] = None,
                 store_backend: str = "sqlite",
                 config: Optional[SpeakerMemConfig] = None,
                 embedder: Optional[Embedder] = None,
                 llm: Optional[LLM] = None,
                 writer_llm: Optional[LLM] = None,
                 store: Optional[MemoryStore] = None,
                 index: Optional[VectorIndex] = None,
                 answer_system: Optional[str] = None,
                 answer_user_tmpl: Optional[str] = None):
        self.cfg = config or SpeakerMemConfig()
        self.embedder = embedder or SentenceTransformerEmbedder(self.cfg.embed_model)
        self._llm = llm
        self._writer_llm = writer_llm
        self.store = store or make_store(persist_path or ":memory:", store_backend)
        self.index = index or NumpyIndex()
        self.memory = SpeakerAwareMemory(self.store, self.index, self.embedder)

        self.writer = Writer(self.memory, self._lazy_writer_llm, self.cfg.writer)
        self.retriever = Retriever(self.memory, self.embedder, self.cfg.retriever,
                                   llm_provider=self._lazy_llm)
        kw = {}
        if answer_system: kw["system_prompt"] = answer_system
        if answer_user_tmpl: kw["user_tmpl"] = answer_user_tmpl
        self._answerer = Answerer(self._lazy_llm, self.cfg.answerer, **kw)

    def _lazy_llm(self) -> LLM:
        if self._llm is None:
            self._llm = OpenAICompatLLM()
        return self._llm

    def _lazy_writer_llm(self) -> LLM:
        return self._writer_llm or self._lazy_llm()


    @staticmethod
    def _to_msg(m: MsgLike) -> Message:
        return m if isinstance(m, Message) else Message.from_dict(m)

    def add(self, message: MsgLike) -> None:
        """Ingest one message online, store it verbatim, and trigger derived-memory writing."""
        self.writer.add(self._to_msg(message))

    def ingest(self, messages: List[MsgLike]) -> int:
        """Ingest messages in session and time order. Return the current number of memory entries."""
        self.writer.ingest([self._to_msg(m) for m in messages])
        return len(self.memory)

    def flush(self) -> None:
        self.writer.flush()


    def retrieve(self, query: str, asker: str = "", k: Optional[int] = None) -> List[MemoryEntry]:
        return self.retriever.retrieve(query, asker=asker, k=k)

    def answer(self, query: str, asker: str = "", k: Optional[int] = None,
               channel: Optional[str] = None) -> str:
        """Retrieve through both paths and answer. S1 provides raw evidence; S2 provides the structured slice for who, state changes, and absences."""
        raw, proj, cells = self.retriever.retrieve_dual(query, channel=channel, k=k)
        slice_text = self.retriever.render_slice(cells, proj.tense)
        return self._answerer.answer(query, raw, slice_text=slice_text)

    def answer_traced(self, query: str, asker: str = "", k: Optional[int] = None,
                      channel: Optional[str] = None) -> Dict[str, Any]:
        """Same as answer(), but also return intermediate products so a demo or debugger can inspect both retrieval paths."""
        raw, proj, cells = self.retriever.retrieve_dual(query, channel=channel, k=k)
        slice_text = self.retriever.render_slice(cells, proj.tense)
        return {"question": query,
                "projection": {"issue": proj.issue, "rows": proj.rows, "tense": proj.tense},
                "s1_raw": [e.render(i) for i, e in enumerate(raw, 1)],
                "s2_slice": slice_text,
                "answer": self._answerer.answer(query, raw, slice_text=slice_text)}


    def roster(self, channel: Optional[str] = None) -> List[str]:
        return self.memory.roster(channel)


    def stats(self) -> Dict[str, int]:
        from collections import Counter
        layers = Counter(e.layer for e in self.memory.active_entries())
        return {"total": len(self.memory), "roster": len(self.memory.roster()),
                "speakers": len(self.memory.speakers()),
                "chains": sum(1 for e in self.memory.active_entries() if e.superseded_by),
                **layers}

    def close(self):
        self.store.close()
