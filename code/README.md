# Code Layout

Run commands from the supplement root unless a command says otherwise. Relative
paths in the benchmark runners resolve from the `code/` directory.

- `bench_loaders.py` normalizes the four benchmark input formats.
- `run_baseline.py` runs BM25, dense retrieval, and full-context baselines.
- `speakermem_pkg/examples/socialmembench.py` runs the SpeakerMem pipeline.
- `run_locomo.py` runs the LoCoMo pipeline.
- `compute_metrics.py` recomputes binary accuracy, exact match, token-F1, and
  timing summaries from JSONL records.
- `rejudge_paper_rubric.py` is an optional SocialMem rubric scorer and may call
  an external LLM.
- `writer_training/` contains the trajectory loader, writer rollout, reward
  code, supervised recipe, and reinforced training recipe.
- `topk_sensitivity_full_llm/run_sensitivity.py` runs the fixed-memory,
  full-LLM System 1/System 2 retrieval-budget sensitivity analysis.

Complete benchmark payloads are intentionally omitted. Download and placement
instructions are in `DATA_SOURCES.md`. The writer-training directory contains
the included training and held-out files needed to inspect the writer recipes.
The sensitivity script requires the external held-out memory store and the
research repository that contains the ten-network evaluation inputs; pass that
location with `--project-root`.

## Environment variables

LLM clients read credentials from environment variables. Use `.env.example` as
a template and keep any resulting `.env` outside this directory. The offline
BM25 check does not require credentials.

## Output conventions

Evaluation output is written as:

```text
<results-dir>/<benchmark>/<unit>/<method>__<category>.jsonl
```

Benchmark summary JSON files aggregate correct and total counts per unit and
category. The sensitivity summary additionally reports question count, binary
accuracy, and token-F1. The included result JSONL files may contain
benchmark-derived questions, answers, and retrieved context. They are
experiment outputs, not replacements for the omitted source datasets.

The sensitivity analysis uses a separate result root:

```text
results/topk_sensitivity_full_llm/
├── sensitivity_results.json
├── sensitivity_results.csv
├── sensitivity_acc.svg
├── sensitivity_token_f1.svg
└── s1k*_s2k*_sourcek1/
```
