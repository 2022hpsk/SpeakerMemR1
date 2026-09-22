
"""Qwen2.5-3B writer completion-only SFT."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "writer_training"), str(HERE / "speakermem_pkg" / "src")]
from training_data import action_counts, load_networks, load_sft_examples


def resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else HERE / p)


def encode(tokenizer, messages, target, device, max_length):
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
    completion = tokenizer(target + tokenizer.eos_token, add_special_tokens=False,
                           return_tensors="pt")["input_ids"]
    total = prompt.shape[1] + completion.shape[1]
    if total > max_length:
        raise ValueError(f"sample has {total} tokens > --max-length {max_length}; "
                         "increase max length or shorten the memory state, never silently truncate")
    inputs = __import__("torch").cat([prompt, completion], dim=1).to(device)
    labels = inputs.clone(); labels[:, :prompt.shape[1]] = -100
    return inputs, labels, total




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", nargs="+", default=["training_trajectories"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="models/speakermem-writer-supervised")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--overwrite-output", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    networks = load_networks(args.data)
    examples = load_sft_examples(args.data)
    print(f"networks={len(networks)} examples={len(examples)} actions={action_counts(networks)}")
    if args.validate_only: return

    out = Path(resolve(args.save_dir))
    if out.exists() and not args.overwrite_output:
        raise FileExistsError(f"{out} exists; choose a new --save-dir or pass --overwrite-output")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(resolve(args.model))
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(resolve(args.model), torch_dtype=dtype).to(device)
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def save(path):
        path.mkdir(parents=True, exist_ok=True)
        before = model.config.use_cache; model.config.use_cache = True
        model.save_pretrained(path); tokenizer.save_pretrained(path); model.config.use_cache = before
        (path / "training_meta.json").write_text(json.dumps(
            {"base_model": args.model, "data": args.data, "epochs": args.epochs,
             "prompt_schema": networks[0]["schema"], "prompt_version": networks[0]["prompt_version"]},
            ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        random.shuffle(examples); optimizer.zero_grad(set_to_none=True)
        loss_sum, token_max = 0.0, 0
        for i, ex in enumerate(examples):
            inputs, labels, length = encode(tokenizer, ex["messages"], ex["target"], device, args.max_length)
            loss = model(inputs, labels=labels).loss
            (loss / args.grad_accum).backward(); loss_sum += float(loss.detach()); token_max = max(token_max, length)
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        print(f"epoch={epoch + 1} train_loss={loss_sum / len(examples):.4f} max_tokens={token_max}", flush=True)
        if args.save_every_epoch: save(Path(f"{out}-ep{epoch + 1}"))
    save(out); print(f"OK saved supervised writer model to {out}")


if __name__ == "__main__": main()
