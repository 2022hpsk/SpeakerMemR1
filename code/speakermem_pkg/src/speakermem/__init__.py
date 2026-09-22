"""SpeakerMem is a speaker-grounded memory framework for multi-party conversations (training-free prototype / SpeakerMem-R1 inference scaffold).

Quick start:
    from speakermem import SpeakerMemory
    mem = SpeakerMemory(persist_path="mem.db")
    mem.ingest([{"speaker":"Alice","content":"I will lead model training.","session":"s1","ts":"2026-01-01T10:00"},
                {"speaker":"Bob","content":"I will handle data cleaning.","session":"s1","ts":"2026-01-01T10:01"}])
    mem.flush()
    print(mem.answer("Who handles evaluation?", asker="Carol"))
"""
from .pipeline import SpeakerMemory
from .types import Message, MemoryEntry, LAYERS, PER_SPEAKER_LAYERS, GROUP_LAYERS
from .config import (SpeakerMemConfig, WriterConfig, RetrieverConfig, AnswererConfig)
from .memory import SpeakerAwareMemory
from .agents import Writer, Retriever, Answerer
from .backends.embedder import Embedder, SentenceTransformerEmbedder
from .backends.llm import LLM, OpenAICompatLLM, LocalHFLLM
from .backends.vector_index import VectorIndex, NumpyIndex
from .backends.store import MemoryStore, JsonStore, SqliteStore, make_store

__version__ = "0.1.0"
__all__ = [
    "SpeakerMemory", "Message", "MemoryEntry", "SpeakerAwareMemory",
    "SpeakerMemConfig", "WriterConfig", "RetrieverConfig", "AnswererConfig",
    "Writer", "Retriever", "Answerer",
    "Embedder", "SentenceTransformerEmbedder", "LLM", "OpenAICompatLLM", "LocalHFLLM",
    "VectorIndex", "NumpyIndex", "MemoryStore", "JsonStore", "SqliteStore", "make_store",
    "LAYERS", "PER_SPEAKER_LAYERS", "GROUP_LAYERS",
]
