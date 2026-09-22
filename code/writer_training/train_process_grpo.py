
"""Writer GRPO with on-policy memory, compact rewards, zero-variance skipping, and KL/gradient guards."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "writer_training"), str(HERE / "speakermem_pkg" / "src")]
from training_data import action_counts, load_networks
from training_rewards import QA_WEIGHT, MEMORY_WEIGHT, VALID_WEIGHT, reward_v4
from training_rollout import StableHashEmbedder, V4RolloutEnv, parse_actions
from speakermem.backends.llm import OpenAICompatLLM

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "8")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def trace_event(trace_path, event, **fields):
    """Durable JSONL trace: enough detail to audit an RL run without storing full prompts."""
    if not trace_path:
        return
    record = {"wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "event": event, **fields}
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()

def _terminal_qa_llm(model):
    from make_training_data import load_project_env
    load_project_env()
    return OpenAICompatLLM(default_model=model)

def _qa_subset(network, n):
    pairs = sorted(network.get("qa_pairs") or [], key=lambda q: str(q.get("id", "")))
    if n <= 0 or len(pairs) <= n: return pairs
    return [pairs[int(i * len(pairs) / n)] for i in range(n)]

def terminal_qa_score(env, network, llm, n=4, top_k=2):
    """Terminal QA increment: R_QA_delta = QA(S1+S2) - QA(S1 only)."""
    from speakermem.agents.answerer import Answerer
    from speakermem.agents.retriever import Retriever
    from speakermem.config import AnswererConfig, RetrieverConfig
    pairs = _qa_subset(network, n)
    if not pairs: return None
    retriever_cfg = RetrieverConfig(top_k=top_k, s1_k=top_k, s2_k=6, s2_source_k=3, ask_enabled=False)
    answer_cfg = AnswererConfig(model=llm.default_model, thinking=False, max_tokens=1024)
    retriever = Retriever(env.memory, env.embedder, retriever_cfg, llm_provider=lambda: llm)
    answerer = Answerer(lambda: llm, answer_cfg)
    combined_scores, s1_scores = [], []
    judge_system = "Judge whether Agent Answer correctly answers the Question against Gold Answer. Return JSON only: {\"correct\": true} or {\"correct\": false}."
    for q in pairs:
        try:
            question = str(q.get("question", ""))
            raw, projection, cells = retriever.retrieve_dual(question, k=top_k)

            def judge(answer):
                verdict = llm.chat(judge_system, "Question:\n%s\n\nGold Answer:\n%s\n\nAgent Answer:\n%s" % (q.get("question", ""), q.get("answer", ""), answer), json_mode=True, thinking=False, temperature=0.0, max_tokens=64)
                data = json.loads(verdict or "{}")
                value = data.get("correct")
                return 1.0 if value is True or str(value).lower() == "true" else 0.0

            combined = answerer.answer(question, raw, retriever.render_slice(cells, projection.tense))
            s1_only = answerer.answer(question, raw, "(empty)")
            combined_value = judge(combined)
            s1_value = judge(s1_only)
            combined_scores.append(combined_value)
            s1_scores.append(s1_value)
        except Exception as exc:
            print("    [terminal-qa warn] %s: %s" % (q.get("id", "?"), str(exc)[:80]), flush=True)
    if not combined_scores:
        return {"delta": 0.0, "combined": 0.0, "s1_only": 0.0, "judged": 0}
    combined = float(np.mean(combined_scores))
    s1_only = float(np.mean(s1_scores))
    return {"delta": combined - s1_only, "combined": combined,
            "s1_only": s1_only, "judged": len(combined_scores)}



class Policy:
    def __init__(self, model_path, *, device="cuda", dtype=None, load_ref=True,
                 use_vllm=False, vllm_gpu_util=0.8, vllm_max_len=32768,
                 train_device=None, rollout_device=None, ref_device=None, seed=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, (train_device or device)
        self.seed = seed
        if torch.cuda.is_available():
            torch.backends.cuda.enable_math_sdp(False)
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.rollout_device = rollout_device if rollout_device is not None else 0
        self.ref_device = ref_device or self.device
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa").to(self.device)
        self.model.config.use_cache = False; self.model.gradient_checkpointing_enable(); self.model.train()
        self.ref = None
        if load_ref:
            self.ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa").to(self.ref_device)
            self.ref.eval()
            for p in self.ref.parameters(): p.requires_grad_(False)


        self.rollout_model = None
        if rollout_device is not None and not use_vllm:
            self.rollout_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa").to(self.rollout_device)
            self.rollout_model.eval()
            for p in self.rollout_model.parameters(): p.requires_grad_(False)
        self.vllm = None
        if use_vllm:
            if not torch.cuda.is_available(): raise RuntimeError("vLLM rollout requires CUDA")
            torch.cuda.set_device(self.rollout_device)
            from vllm import LLM
            self.vllm = LLM(model=model_path, dtype="bfloat16", enforce_eager=True,
                            gpu_memory_utilization=vllm_gpu_util, max_model_len=vllm_max_len)
            self.sync_vllm()

    def sync_rollout(self):
        """Refresh the frozen rollout copy after every successful PPO update."""
        if self.rollout_model is not None:
            weights = {name: value.detach().to(self.rollout_device)
                       for name, value in self.model.state_dict().items()}
            self.rollout_model.load_state_dict(weights, strict=True)
        self.sync_vllm()

    def sync_vllm(self):
        if self.vllm is None: return
        items = [(name, value.detach().to(self.rollout_device)) for name, value in self.model.state_dict().items()]
        def load(worker, weights): worker.model_runner.model.load_weights(weights)
        self.vllm.collective_rpc(load, args=(items,))

    def prompt_ids(self, messages, max_tokens=None):
        encoded = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if max_tokens and ids.shape[1] > max_tokens:

            front = max_tokens // 2
            ids = self.torch.cat([ids[:, :front], ids[:, -(max_tokens - front):]], dim=1)
        return ids.to(self.device)

    def act_batch(self, batches, max_new_tokens=2048, temperature=0.8, top_p=0.95):
        torch = self.torch
        if self.vllm is not None:
            from vllm import SamplingParams
            sampling_kwargs = {"temperature": temperature, "top_p": top_p, "max_tokens": max_new_tokens}
            if self.seed is not None:
                sampling_kwargs["seed"] = int(self.seed)
            params = SamplingParams(**sampling_kwargs)
            outputs = self.vllm.chat(batches, params, use_tqdm=False)
            return [(o.outputs[0].text,
                     torch.tensor([list(o.outputs[0].token_ids) or [self.tok.eos_token_id]],
                                  device=self.device)) for o in outputs]
        rollout_model = self.rollout_model if self.rollout_model is not None else self.model
        rollout_device = self.rollout_device if self.rollout_model is not None else self.device
        shared_rollout = self.rollout_model is None
        was_training = rollout_model.training
        old_use_cache = rollout_model.config.use_cache
        self.tok.padding_side = "left"
        encoded = self.tok.apply_chat_template(batches, add_generation_prompt=True, padding=True,
                                               return_tensors="pt", return_dict=True)
        ids, mask = encoded["input_ids"].to(rollout_device), encoded["attention_mask"].to(rollout_device)
        if shared_rollout:
            rollout_model.eval()
            rollout_model.config.use_cache = True
        try:
            with torch.no_grad():
                out = rollout_model.generate(ids, attention_mask=mask, do_sample=True,
                                             temperature=temperature, top_p=top_p,
                                             max_new_tokens=max_new_tokens, use_cache=True,
                                             pad_token_id=self.tok.pad_token_id)
        finally:
            if shared_rollout:
                rollout_model.config.use_cache = old_use_cache
                if was_training:
                    rollout_model.train()
        result, start, eos = [], ids.shape[1], self.tok.eos_token_id
        for row in out:
            tokens = row[start:].tolist()
            if eos in tokens: tokens = tokens[:tokens.index(eos) + 1]
            tokens = tokens or [eos]
            tensor = torch.tensor([tokens], device=self.device)
            result.append((self.tok.decode(tokens, skip_special_tokens=True), tensor))
        return result

    def token_logp(self, prompt_ids, completion_ids, *, reference=False, with_grad=True, chunk=256):
        torch = self.torch; model = self.ref if reference else self.model
        model_device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(model_device)
        completion_ids = completion_ids.to(model_device)
        full = torch.cat([prompt_ids, completion_ids], dim=1); p = prompt_ids.shape[1]
        context = torch.enable_grad() if with_grad and not reference else torch.no_grad()
        with context:
            hidden = model.model(full).last_hidden_state; target = full[:, 1:]
            pieces = []
            for start in range(p - 1, hidden.shape[1] - 1, chunk):
                end = min(start + chunk, hidden.shape[1] - 1)
                logits = model.lm_head(hidden[:, start:end, :]); wanted = target[:, start:end]
                pieces.append((logits.gather(-1, wanted.unsqueeze(-1)).squeeze(-1)
                               - torch.logsumexp(logits, dim=-1)))
        result = torch.cat(pieces, dim=1).squeeze(0)
        return result.to(self.device) if reference and model_device != torch.device(self.device) else result


def optimizer_state_to(optimizer, device):
    """Keep Adam moments off GPU during forward/backward; move them only for optimizer.step."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if key != "step" and hasattr(value, "to"):
                state[key] = value.to(device)

def structural_failure_reasons(pred, completion_tokens, max_new_tokens, max_actions):
    """Return hard contract failures that must dominate memory/QA rewards."""
    report = pred.get("report") or {}
    reasons = []
    if not report.get("json_valid"):
        reasons.append("invalid_json")
    errors = report.get("errors") or []
    if errors:
        reasons.append("invalid_actions")
    attempted = int(report.get("attempted") or 0)
    if attempted > max_actions:
        reasons.append("too_many_actions")
    if completion_tokens >= max_new_tokens:
        reasons.append("max_tokens")
    return reasons


def apply_structural_failure_penalty(score, reasons, penalty=-1.0):
    """Apply bounded structural penalties without discarding useful partial credit.

    A completion may contain a valid action followed by one bad/repeated action.
    Replacing its whole reward with a constant made PPO unable to learn which
    part of the JSON was useful. Keep the diagnostics, but add small bounded
    penalties that are also preserved when terminal QA is backfilled.
    """
    score = dict(score)
    scale = min(1.0, abs(float(penalty)))
    local_penalty = 0.0
    if "invalid_json" in reasons:
        local_penalty -= 0.25 * scale
    if "invalid_actions" in reasons:
        local_penalty -= 0.15 * scale
    if "too_many_actions" in reasons:
        local_penalty -= 0.10 * scale
    if "max_tokens" in reasons:
        local_penalty -= 0.10 * scale
    score["hard_failure"] = 0.0
    score["structural_failure"] = 1.0 if reasons else 0.0
    score["hard_failure_reasons"] = list(reasons)
    score["hard_failure_penalty"] = local_penalty
    score["structural_penalty"] = local_penalty
    score["reward"] = float(score.get("reward", 0.0)) + local_penalty
    return score


def apply_terminal_qa_returns(trajectories, qa_scores, gamma=0.95):
    """Discount R_QA_delta back into each trajectory while retaining local structural penalties."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"qa return gamma must be in [0, 1], got {gamma}")
    summaries = []
    for candidate, trajectory in enumerate(trajectories):
        result = qa_scores[candidate] if candidate < len(qa_scores) else None
        if result is None:
            continue
        if isinstance(result, dict):
            terminal = min(1.0, max(-1.0, float(result.get("delta", 0.0))))
            terminal_combined = float(result.get("combined", 0.0))
            terminal_s1 = float(result.get("s1_only", 0.0))
            judged = int(result.get("judged", 0))
        else:
            terminal = min(1.0, max(-1.0, float(result)))
            terminal_combined = terminal_s1 = float("nan")
            judged = 0
        rewards = []
        for index, step in enumerate(trajectory):
            discount = gamma ** (len(trajectory) - 1 - index)
            discounted_qa = terminal * discount
            components = step["components"]
            total = (VALID_WEIGHT * components["valid"] +
                     MEMORY_WEIGHT * components["memory"] +
                     QA_WEIGHT * discounted_qa +
                     float(components.get("structural_penalty", 0.0)))
            components.update({"qa": discounted_qa, "qa_delta": discounted_qa,
                               "qa_terminal": terminal, "qa_delta_terminal": terminal,
                               "qa_combined_terminal": terminal_combined,
                               "qa_s1_only_terminal": terminal_s1, "qa_judged": judged,
                               "qa_discount": discount, "reward": total})
            step["reward"] = total
            rewards.append(total)
        summaries.append({"candidate": candidate, "terminal_qa": terminal,
                          "qa_delta": terminal, "qa_combined": terminal_combined,
                          "qa_s1_only": terminal_s1, "qa_judged": judged,
                          "rewards": rewards})
    return summaries

def rollout_group(policy, network, group_size, max_new_tokens, temperature,
                  env_factory=V4RolloutEnv, qa_llm=None, qa_per_network=0,
                  qa_top_k=2, qa_return_gamma=0.95, trace_path=None,
                  iteration=None, logprob_max_prompt_tokens=None,
                  hard_failure_penalty=-1.0):
    envs, trajectories = [env_factory() for _ in range(group_size)], [[] for _ in range(group_size)]
    total_positions = len(network["transitions"])
    terminal_qa_scores = [None] * group_size
    try:
        for pos, teacher_step in enumerate(network["transitions"]):
            pos_started = time.time()
            turns, sid, ts = teacher_step["messages"], teacher_step["session_id"], teacher_step.get("ts", "")
            prompts = [env.observe(turns, session_id=sid, ts=ts)["messages"] for env in envs]
            generation_started = time.time()
            generations, records = policy.act_batch(prompts, max_new_tokens, temperature), []
            generation_sec = time.time() - generation_started
            diagnostics = []
            for g, (text, completion_ids) in enumerate(generations):
                parsed = parse_actions(text)
                pred = envs[g].step(turns, parsed["actions"], session_id=sid, ts=ts, json_valid=parsed["json_valid"])
                report = pred.get("report", {})
                diagnostics.append({"candidate": g, "completion_tokens": int(completion_ids.shape[1]),
                    "completion_chars": len(text), "output_preview": text[:800],
                    "json_valid": bool(parsed["json_valid"]),
                    "actions": {kind: sum(1 for a in parsed["actions"] if a.get("action") == kind) for kind in ("ADD", "UPDATE", "NOOP")},
                    "report": {k: report.get(k) for k in ("attempted", "applied", "noop", "errors", "created_ids", "updated_from", "schema_valid_actions")}})
                records.append((completion_ids, pred))
            qa_scores, qa_sec = [None] * len(records), 0.0
            if qa_llm is not None and qa_per_network > 0 and pos == total_positions - 1:
                from concurrent.futures import ThreadPoolExecutor
                qa_started = time.time()
                with ThreadPoolExecutor(max_workers=min(8, len(envs))) as executor:
                    qa_scores = list(executor.map(lambda env: terminal_qa_score(env, network, qa_llm, qa_per_network, qa_top_k), envs))
                terminal_qa_scores = qa_scores
                qa_sec = time.time() - qa_started
            for g, (completion_ids, pred) in enumerate(records):
                score = reward_v4(pred, teacher_step)
                max_actions = max(8, 2 * len(turns))
                failure_reasons = structural_failure_reasons(
                    pred, int(completion_ids.shape[1]), max_new_tokens, max_actions)
                score = apply_structural_failure_penalty(
                    score, failure_reasons, hard_failure_penalty)
                diagnostics[g]["qa"] = qa_scores[g]
                diagnostics[g]["reward"] = score
                prompt_ids = policy.prompt_ids(prompts[g], max_tokens=logprob_max_prompt_tokens)
                diagnostics[g]["score_prompt_tokens"] = int(prompt_ids.shape[1])
                old_lp = policy.token_logp(prompt_ids, completion_ids, with_grad=False).detach()
                ref_lp = policy.token_logp(prompt_ids, completion_ids, reference=True).detach() if policy.ref is not None else old_lp
                trajectories[g].append({"pos": pos, "reward": score["reward"], "components": score, "prompt_ids": prompt_ids, "completion_ids": completion_ids, "old_lp": old_lp, "ref_lp": ref_lp})
            trace_event(trace_path, "rollout_transition", iteration=iteration, network_id=network["network_id"],
                position=pos, positions_total=total_positions, session_id=sid, ts=ts, messages=len(turns),
                generation_sec=round(generation_sec, 3), qa_sec=round(qa_sec, 3), total_sec=round(time.time()-pos_started, 3),
                candidates=diagnostics)
            print(f"[rollout] step={iteration} network={network['network_id']} pos={pos+1}/{total_positions} session={sid} "
                  f"gen_sec={generation_sec:.1f} qa_sec={qa_sec:.1f} total_sec={time.time()-pos_started:.1f} "
                  f"tokens={[d['completion_tokens'] for d in diagnostics]} rewards={[round(d['reward']['reward'],4) for d in diagnostics]}", flush=True)
        qa_returns = apply_terminal_qa_returns(trajectories, terminal_qa_scores, qa_return_gamma)
        trace_event(trace_path, "terminal_qa_return", iteration=iteration,
                    network_id=network["network_id"], gamma=qa_return_gamma, candidates=qa_returns)
        if qa_returns:
            print(f"[qa-return] step={iteration} network={network['network_id']} gamma={qa_return_gamma} "
                  f"terminal={[round(row['terminal_qa'], 4) for row in qa_returns]}", flush=True)
        return trajectories
    finally:
        for env in envs: env.close()

def assign_advantages(trajectories, min_std=0.02, clip=3.0,
                      hard_failure_advantage=0.0):
    """Set the advantage to zero at positions with no reward difference; do not divide by sd_floor to create artificial gradients."""
    skipped, stats = 0, {}
    positions = sorted({step["pos"] for trajectory in trajectories for step in trajectory})
    for pos in positions:
        steps = [trajectory[pos] for trajectory in trajectories if pos < len(trajectory)]
        rewards = np.asarray([step["reward"] for step in steps], dtype=float)
        mean, std = float(rewards.mean()), float(rewards.std()); stats[pos] = (mean, std)
        if std < min_std:
            any_hard_failure = False
            for step in steps:
                if step["components"].get("hard_failure"):
                    step["advantage"] = float(hard_failure_advantage)
                    any_hard_failure = True
                else:
                    step["advantage"] = 0.0
            if not any_hard_failure:
                skipped += 1
        else:
            for step in steps:
                advantage = float(np.clip((step["reward"] - mean) / std, -clip, clip))
                if step["components"].get("hard_failure"):
                    advantage = min(advantage, float(hard_failure_advantage))
                step["advantage"] = advantage
    return stats, skipped


def train(policy, networks, *, group_size=4, steps=100, max_new_tokens=2048,
          temperature=0.8, lr=1e-6, ppo_epochs=2, ppo_clip=0.2, kl_coef=0.1,
          min_reward_std=0.02, max_grad_norm=1.0, max_kl=0.02,
          save_dir=None, save_every=0, qa_llm=None, qa_per_network=0, qa_top_k=2, qa_return_gamma=0.95, trace_path=None, logprob_max_prompt_tokens=None,
          hard_failure_penalty=-1.0, hard_failure_advantage=0.0,
          start_step=0):
    torch = policy.torch; optimizer = torch.optim.AdamW(policy.model.parameters(), lr=lr)
    for iteration in range(start_step, steps):
        started = time.time(); network = networks[iteration % len(networks)]
        trace_event(trace_path, "update_start", iteration=iteration, network_id=network["network_id"], group_size=group_size, max_new_tokens=max_new_tokens)
        group = rollout_group(policy, network, group_size, max_new_tokens, temperature,
                              qa_llm=qa_llm, qa_per_network=qa_per_network,
                              qa_top_k=qa_top_k, qa_return_gamma=qa_return_gamma,
                              trace_path=trace_path, iteration=iteration,
                              logprob_max_prompt_tokens=logprob_max_prompt_tokens,
                              hard_failure_penalty=hard_failure_penalty)
        _, zero_positions = assign_advantages(
            group, min_reward_std,
            hard_failure_advantage=hard_failure_advantage)
        samples = [step for trajectory in group for step in trajectory if step["advantage"] != 0.0]
        if not samples:
            print(f"step={iteration} network={network['network_id']} skipped: all reward groups zero variance")
            continue
        last_kl, skipped_update = 0.0, False
        for _ in range(ppo_epochs):
            optimizer.zero_grad(set_to_none=True); kl_values = []
            for sample in samples:
                new_lp = policy.token_logp(sample["prompt_ids"], sample["completion_ids"])
                ratio = torch.exp(new_lp - sample["old_lp"]); advantage = sample["advantage"]
                surrogate = torch.min(ratio * advantage,
                                      torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantage)
                delta = sample["ref_lp"] - new_lp
                kl = (torch.exp(delta) - delta - 1).mean(); kl_values.append(kl.detach())
                ((-surrogate.mean() + kl_coef * kl) / len(samples)).backward()
            last_kl = float(torch.stack(kl_values).mean())
            if not math.isfinite(last_kl) or last_kl > max_kl:
                optimizer.zero_grad(set_to_none=True); skipped_update = True; break
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True); skipped_update = True; break
            optimizer_state_to(optimizer, policy.device)
            optimizer.step()
            optimizer_state_to(optimizer, "cpu")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not skipped_update: policy.sync_rollout()
        rewards = [step["reward"] for trajectory in group for step in trajectory]
        duration = time.time() - started
        summary = {"reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)), "zero_positions": zero_positions, "samples": len(samples), "kl": last_kl, "update": "SKIP" if skipped_update else "OK", "duration_sec": round(duration, 3)}
        trace_event(trace_path, "update_end", iteration=iteration, network_id=network["network_id"], **summary)
        print(f"step={iteration} network={network['network_id']} reward={summary['reward_mean']:.4f} "
              f"std={summary['reward_std']:.4f} zero_pos={zero_positions} samples={len(samples)} KL={last_kl:.4f} "
              f"update={summary['update']} sec={duration:.0f}", flush=True)
        if save_dir and not skipped_update:
            latest = HERE / f"{save_dir}-latest"; latest.mkdir(parents=True, exist_ok=True)
            policy.model.save_pretrained(latest); policy.tok.save_pretrained(latest)
            trace_event(trace_path, "checkpoint_latest", iteration=iteration, path=str(latest))
        if save_dir and save_every and (iteration + 1) % save_every == 0:
            path = HERE / f"{save_dir}-step{iteration + 1}"; path.mkdir(parents=True, exist_ok=True)
            policy.model.save_pretrained(path); policy.tok.save_pretrained(path)
            trace_event(trace_path, "checkpoint", iteration=iteration, path=str(path))


class DryPolicy:
    """Model-free environment dry-run; returns executable actions with measurable quality differences."""
    def __init__(self, teacher_actions):
        import torch
        self.torch, self.ref, self.calls, self.teacher_actions = torch, None, 0, teacher_actions
    def act_batch(self, prompts, *args):
        actions = self.teacher_actions[self.calls]; self.calls += 1; out = []
        for i, prompt in enumerate(prompts):
            current = json.loads(json.dumps(actions))
            if current and current[0].get("action") == "UPDATE":
                ids = re.findall(r"\n\s+([0-9a-f]{8}) \[", prompt[1]["content"])
                if ids: current[0]["entry_id"] = ids[-1]
            if i == len(prompts) - 1: current = [{"action": "NOOP"}]
            out.append((json.dumps({"actions": current}, ensure_ascii=False), self.torch.tensor([[1]])))
        return out
    def prompt_ids(self, messages, max_tokens=None): return self.torch.tensor([[1]])
    def token_logp(self, *args, **kwargs): return self.torch.zeros(1)


def dry_run():
    embedder = StableHashEmbedder(); teacher = V4RolloutEnv(embedder=embedder)
    specs = [([{"speaker": "A", "content": "Progress is 38%", "session": "s1"}],
              [{"action": "ADD", "owner": "A", "source": "A", "layer": "per_speaker_core",
                "utype": "fact", "content": "Project progress is 38%"}], "s1"),
             ([{"speaker": "A", "content": "Progress is 97%", "session": "s2"}], None, "s2")]
    transitions, actions = [], []
    first = teacher.step(specs[0][0], specs[0][1], session_id="s1"); transitions.append(first); actions.append(specs[0][1])
    old = first["post_state"]["heads"][0]["entry_id"]
    update = [{"action": "UPDATE", "entry_id": old, "content": "Project progress is 97%"}]
    transitions.append(teacher.step(specs[1][0], update, session_id="s2")); actions.append(update); teacher.close()
    net = {"network_id": "dry", "transitions": transitions}
    group = rollout_group(DryPolicy(actions), net, 2, 128, 0.8,
                          env_factory=lambda: V4RolloutEnv(embedder=embedder))
    _, skipped = assign_advantages(group)
    assert len(group) == 2 and all(len(t) == 2 for t in group) and skipped < 2
    print("OK reinforced-training dry-run: on-policy state, ADD/UPDATE/NOOP, reward and advantages")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="models/speakermem-writer-supervised")
    ap.add_argument("--data", nargs="+", default=["training_trajectories"])
    ap.add_argument("--G", type=int, default=4); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--start-step", type=int, default=0,
                    help="resume from this global update index; model must point to its checkpoint")
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None, help="local Python/NumPy/PyTorch/vLLM random seed; omitted preserves legacy behavior")
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--min-reward-std", type=float, default=0.02)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-kl", "--target-kl", dest="max_kl", type=float, default=0.02)
    ap.add_argument("--train-device", default=None); ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--ref-device", default=None, help="frozen reference GPU, e.g. cuda:1")
    ap.add_argument("--rollout-device", default=None, help="rollout GPU, e.g. cuda:1 (an independent Transformers copy is used by default)")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.8); ap.add_argument("--vllm-max-len", type=int, default=32768)
    ap.add_argument("--save-dir", default="models/speakermem-writer-supervised-rl")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--overwrite-output", action="store_true")
    ap.add_argument("--qa-per-network", type=int, default=4, help="fixed number of QA items evaluated by DeepSeek at the end of each policy rollout")
    ap.add_argument("--qa-top-k", type=int, default=2)
    ap.add_argument("--qa-return-gamma", type=float, default=0.95,
                    help="discount applied when the terminal QA reward is assigned to every transition in the same trajectory")
    ap.add_argument("--hard-failure-penalty", type=float, default=-1.0,
                    help="fallback reward for incomplete JSON, duplicate/invalid/excess actions, or generation cut off at the token limit")
    ap.add_argument("--hard-failure-advantage", type=float, default=0.0,
                    help="fixed advantage for compatibility with legacy hard-failure markers; disabled by default with local structural penalties")
    ap.add_argument("--qa-model", default="deepseek-v4-flash")
    ap.add_argument("--trace-jsonl", default=None, help="durable per-transition trace JSONL")
    ap.add_argument("--logprob-max-prompt-tokens", type=int, default=0, help="PPO scoring context cap; 0 keeps the exact full prompt")
    args = ap.parse_args()
    if args.dry_run: dry_run(); return
    import torch
    seed_everything(args.seed)
    networks = load_networks(args.data); print(f"networks={len(networks)} actions={action_counts(networks)}")
    if not torch.cuda.is_available(): raise RuntimeError("full reinforced training requires CUDA; use --dry-run for CPU validation")
    out = Path(args.save_dir) if Path(args.save_dir).is_absolute() else HERE / args.save_dir
    if out.exists() and not args.overwrite_output:
        raise FileExistsError(f"{out} exists; choose a new --save-dir or pass --overwrite-output")
    qa_llm = _terminal_qa_llm(args.qa_model) if args.qa_per_network > 0 else None
    trace_path = args.trace_jsonl or str(HERE / "logs" / "speakermem_reinforced_trace.jsonl")
    trace_event(trace_path, "run_config", model=args.model, seed=args.seed, data=args.data, G=args.G, steps=args.steps, start_step=args.start_step, max_new_tokens=args.max_new_tokens, temperature=args.temperature, lr=args.lr, ppo_epochs=args.ppo_epochs, clip=args.clip, kl_coef=args.kl_coef, target_kl=args.max_kl, qa_per_network=args.qa_per_network, qa_top_k=args.qa_top_k, qa_s2_k=6, qa_s2_source_k=3, qa_return_gamma=args.qa_return_gamma, qa_model=args.qa_model, hard_failure_penalty=args.hard_failure_penalty, hard_failure_advantage=args.hard_failure_advantage, logprob_max_prompt_tokens=args.logprob_max_prompt_tokens, ask_enabled=False, train_device=args.train_device, rollout_device=args.rollout_device, ref_device=args.ref_device, optimizer_state_offload="cpu", sdpa_math_backend=False, save_dir=args.save_dir)
    rollout_backend = "vllm" if args.vllm else ("separate-transformers" if args.rollout_device else "shared-transformers")
    print(f"terminal_qa={args.qa_per_network} model={args.qa_model} ask_enabled=False "
          f"train_device={args.train_device or 'cuda'} rollout_device={args.rollout_device or args.train_device or 'cuda'} "
          f"rollout_backend={rollout_backend}", flush=True)
    policy = Policy(str(HERE / args.model), dtype=torch.bfloat16, use_vllm=args.vllm,
                    vllm_gpu_util=args.vllm_gpu_util, vllm_max_len=args.vllm_max_len,
                    train_device=args.train_device, rollout_device=args.rollout_device, ref_device=args.ref_device, seed=args.seed)
    train(policy, networks, group_size=args.G, steps=args.steps, max_new_tokens=args.max_new_tokens,
          temperature=args.temperature, lr=args.lr, ppo_epochs=args.ppo_epochs,
          ppo_clip=args.clip, kl_coef=args.kl_coef, min_reward_std=args.min_reward_std,
          max_grad_norm=args.max_grad_norm, max_kl=args.max_kl,
          save_dir=args.save_dir, save_every=args.save_every, qa_llm=qa_llm,
          qa_per_network=args.qa_per_network, qa_top_k=args.qa_top_k,
          qa_return_gamma=args.qa_return_gamma, trace_path=trace_path,
          logprob_max_prompt_tokens=args.logprob_max_prompt_tokens,
          hard_failure_penalty=args.hard_failure_penalty,
          hard_failure_advantage=args.hard_failure_advantage,
          start_step=args.start_step)
    out.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(out); policy.tok.save_pretrained(out); print(f"OK saved {out}")


if __name__ == "__main__": main()
