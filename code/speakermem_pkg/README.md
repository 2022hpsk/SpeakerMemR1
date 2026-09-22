# SpeakerMem

SpeakerMem is a speaker-grounded memory framework for multi-party conversations such as group chats, meetings, and multi-agent sessions. It keeps raw utterances and derived memories separate, while preserving ownership, source, speaker identity, and temporal updates.

## Quick start

```python
from speakermem import SpeakerMemory

mem = SpeakerMemory(persist_path="mem.db")
mem.ingest([
    {"speaker": "Alice", "content": "I will lead model training.", "session": "s1", "ts": "2026-01-01T10:00"},
    {"speaker": "Bob", "content": "I will handle data cleaning.", "session": "s1", "ts": "2026-01-01T10:01"},
])
print(mem.answer("Who handles data cleaning?", asker="Carol"))
entries = mem.retrieve("Who handles data cleaning?", k=10)
```

The package can run verbatim ingestion and retrieval without an LLM. Derived-memory writing and answer generation use an OpenAI-compatible endpoint configured through `OPENAI_API_KEY`, `OPENAI_BASE_URL`, or `DEEPSEEK_API_KEY`.

## Installation

```bash
pip install -e .
pip install -e ".[faiss,chroma]"
```

## Memory layers

| Layer | Meaning |
| --- | --- |
| `per_speaker_core` | Stable facts, identity, persistent preferences, and durable stances about one speaker. |
| `per_speaker_episodic` | Verbatim utterances associated with a speaker and timestamp. |
| `per_speaker_profile` | Observations about one speaker made by another speaker. |
| `group_interaction` | Cross-speaker events, relationships, and group decisions. |
| `group_insight` | Group norms, consensus, and meta-observations. |

Each `MemoryEntry` stores an identifier, content, owner, source, layer, turn, timestamp, type, confidence, provenance links, and non-destructive update links.

## Ingestion

The ingestion pipeline has two tracks:

1. Every raw message is stored verbatim in `per_speaker_episodic`.
2. At the configured session or message boundary, the writer reads the current derived heads and the new message window, then emits `ADD`, `UPDATE`, or `NOOP` actions.

Updates are non-destructive: the previous entry remains available and the new entry links back to it. The speaker roster is collected directly from the message stream without an LLM.

## Retrieval

Retrieval combines a raw-evidence track with a structured-memory track. The raw track finds precise utterances and can expand local context. The structured track searches within speaker and group rows, follows update chains, and falls back to raw evidence when a row has no relevant derived memory. An optional cross-encoder can rerank the final candidates.

## Backends

| Component | Default | Optional alternatives |
| --- | --- | --- |
| Embeddings | `SentenceTransformerEmbedder` with `all-MiniLM-L6-v2` | Any `Embedder` implementation |
| LLM | `OpenAICompatLLM` | Any `LLM` implementation |
| Vector index | `NumpyIndex` | FAISS, Chroma, or Qdrant adapters |
| Storage | `SqliteStore` | `JsonStore` |

SQLite stores entries and embedding blobs in one file. The JSON store is a dependency-light alternative.

## Evaluation

`examples/socialmembench.py` runs the SpeakerMem pipeline on SocialMemBench. The top-level artifact README documents the included final outputs and the external benchmark data sources.

## License

The package is released under the license included in this directory. Benchmark licenses and citation requirements are listed in the top-level `THIRD_PARTY_NOTICES.md` file.
