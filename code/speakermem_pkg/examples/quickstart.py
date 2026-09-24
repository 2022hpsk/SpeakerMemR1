"""Minimal example: store a conversation and retrieve evidence without an LLM.

The local embedding model may be downloaded on first use.
Run from the repository root: python code/speakermem_pkg/examples/quickstart.py
"""
from speakermem import SpeakerMemory, SpeakerMemConfig, WriterConfig, RetrieverConfig


cfg = SpeakerMemConfig(
    writer=WriterConfig(enabled=False),
    retriever=RetrieverConfig(llm_select=False, ask_enabled=False, s2_enabled=False),
)
mem = SpeakerMemory(config=cfg)

msgs = [
    {"speaker": "Alice", "content": "I will lead model training.", "session": "s1"},
    {"speaker": "Bob", "content": "I will handle data cleaning.", "session": "s1"},
    {"speaker": "Carol", "content": "I will monitor evaluation and metrics.", "session": "s1"},
    {"speaker": "Dave", "content": "I will handle the frontend and demo.", "session": "s1"},
]
mem.ingest(msgs)
mem.flush()
print("stats:", mem.stats())

for q in ["Who handles evaluation?", "What does Bob handle?"]:
    print(f"\nQ: {q}")
    for entry in mem.retrieve(q, k=3):
        print("   ", entry.render())





