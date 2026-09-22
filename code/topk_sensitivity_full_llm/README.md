# Full-LLM Top-k Sensitivity Analysis

This directory contains the script used for the retrieval-budget sensitivity
study. It reuses the ten held-out SocialMemBench networks and 305 questions,
keeps the memory store fixed, and does not load an RL/SFT/Qwen Writer. The
full-LLM query/answer pipeline uses DeepSeek for the system components and the
same GPT-4o-mini judge configuration as the main evaluation.

The grid varies `s1_topk` in `{5, 10, 20}` and System 2 `k` in `{2, 4, 6}`;
`source-k=1` is fixed. The complete outputs are in
`results/topk_sensitivity_full_llm/`.

## Running

The public benchmark payloads, the held-out memory store, and model services are
external dependencies. From the supplement root, run a smoke check first:

```bash
python code/topk_sensitivity_full_llm/run_sensitivity.py \
  --project-root /path/to/root \
  --output-dir results/topk_sensitivity_full_llm --smoke
```

Then run the full 3x3 grid:

```bash
python code/topk_sensitivity_full_llm/run_sensitivity.py \
  --project-root /path/to/root \
  --output-dir results/topk_sensitivity_full_llm
```

The script writes per-configuration filter files, raw
question-level JSONL outputs, a complete JSON/CSV summary, and dependency-free
SVG plots. API credentials are read from the external project environment and
must not be committed to the supplement.
