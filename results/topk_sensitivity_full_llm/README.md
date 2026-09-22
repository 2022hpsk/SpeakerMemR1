# Full-LLM Top-k Sensitivity Results

These outputs cover the 3x3 retrieval-budget grid on the same ten held-out
SocialMemBench networks and 305 questions. System 1 top-k is `{5, 10, 20}`;
System 2 `k` is `{2, 4, 6}`; System 2 source-k is fixed to `1`. The memory
store and benchmark payloads are external inputs and are not included here.

- `s1k*_s2k*_sourcek1/`:  summaries, and question-level JSONL records.
- `sensitivity_results.json`: all nine configurations with question count,
  binary accuracy, and token-F1.

The summary uses the same GPT-4o-mini judge and token-F1 normalization as the
main evaluation. The result files are included experiment outputs; rerunning
the analysis requires the external data, memory store, and model services
described in `code/topk_sensitivity_full_llm/README.md`.
