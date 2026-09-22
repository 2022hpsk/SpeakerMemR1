"""Minimal example: build memory, retrieve evidence, and answer questions.

The first two steps run without an LLM. Derived-memory writing and answer generation require DEEPSEEK_API_KEY or OPENAI_API_KEY.
Run with: python examples/quickstart.py
"""
from speakermem import SpeakerMemory, SpeakerMemConfig, WriterConfig


cfg = SpeakerMemConfig(writer=WriterConfig(enabled=False))
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





