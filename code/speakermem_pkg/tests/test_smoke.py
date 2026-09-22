"""Smoke tests for ingestion, retrieval, and persistence without an LLM (writer.enabled=False).

Run:pip install -e ".[dev]" && pytest
"""
from speakermem import (SpeakerMemory, SpeakerMemConfig, WriterConfig, RetrieverConfig,
                        MemoryEntry)


def _mem(tmp_path=None):

    cfg = SpeakerMemConfig(writer=WriterConfig(enabled=False),
                           retriever=RetrieverConfig(llm_select=False))
    path = str(tmp_path / "m.db") if tmp_path else None
    return SpeakerMemory(persist_path=path, config=cfg)


MSGS = [
    {"speaker": "Alice", "content": "I will lead model training.", "session": "s1", "ts": "2026-01-01T10:00"},
    {"speaker": "Bob",   "content": "I will handle data cleaning.",   "session": "s1", "ts": "2026-01-01T10:01"},
    {"speaker": "Carol", "content": "I will track evaluation and metrics.",   "session": "s1", "ts": "2026-01-01T10:02"},
]


def test_ingest_and_layers():
    mem = _mem()
    n = mem.ingest(MSGS); mem.flush()
    assert n == 3
    st = mem.stats()
    assert st["total"] == 3 and st["speakers"] == 3
    assert st.get("per_speaker_episodic") == 3


def test_retrieve_speaker_aware():
    mem = _mem(); mem.ingest(MSGS); mem.flush()
    hits = mem.retrieve("What does Bob handle?", k=3)
    assert hits and hits[0].owner == "Bob"


def test_verbatim_preserved():
    mem = _mem(); mem.ingest(MSGS); mem.flush()
    contents = {e.content for e in mem.memory.active_entries()}
    assert "I will handle data cleaning." in contents


def test_persist_reload(tmp_path):
    mem = _mem(tmp_path); mem.ingest(MSGS); mem.flush(); mem.close()
    mem2 = _mem(tmp_path)
    assert len(mem2.memory) == 3
    assert mem2.retrieve("evaluation", k=2)


def test_entry_roundtrip():
    e = MemoryEntry(content="x", owner="A", source="A", layer="per_speaker_core", ts="2026-01-01")
    assert MemoryEntry.from_dict(e.to_dict()).owner == "A"
