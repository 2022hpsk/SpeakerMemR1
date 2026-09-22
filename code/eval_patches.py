"""A non-invasive runtime patch for the upstream GroupMemBench evaluator. The cloned
upstream code is left unchanged so it remains easy to update.

The patch addresses two evaluation issues when using a reasoning model:
1. **Responses can be truncated**: the upstream runner fixes the agent and judge
   output limits at 512 and 256 tokens, so the final verdict can be cut off.
2. **The default temperature is stochastic**: the upstream client uses 0.2, which
   can make repeated benchmark runs vary.

The patch replaces `eval_lib.call_chat` at runtime, raises the output floors, and
sets temperature to zero for deterministic judging. `run_qa` resolves the function
through the module namespace, so replacing that attribute is sufficient.
"""
from __future__ import annotations


def patch_token_budgets(min_judge: int = 100000, min_agent: int = 100000,
                        temperature: float = 0.0) -> None:


    import baselines.rag_common.eval_lib as elib
    from llm_utils import chat_completion_text

    def patched(client, model, system_prompt, user_prompt, max_tokens):

        floor = min_judge if max_tokens <= 256 else min_agent
        return chat_completion_text(
            client, model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            max_tokens=max(max_tokens, floor), temperature=temperature)

    elib.call_chat = patched
    print(f"[patch] completion limits: judge>={min_judge}, agent>={min_agent}, temperature={temperature}"
          f"(to avoid truncated answers)")
