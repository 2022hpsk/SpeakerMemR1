# GPT-5.6-Luna SpeakerMem (without ASK)

This directory contains the final benchmark summaries and question-level
records for the ASK-disabled SpeakerMem evaluation.

| Benchmark | Correct / total |
|---|---:|
| SocialMemBench | 664 / 1031 |
| GroupMemBench | 318 / 745 |
| EverMemBench-Dynamic | 1441 / 2400 |

The `summary_speakermem_*.json` files are regenerated from the complete raw
question-level JSONL records. The corresponding `metrics_speakermem_*.json`
files retain token-F1, EM, per-category, per-unit, and attribution aggregates.

