"""Evaluate SpeakerMem on the LoCoMo long-conversation benchmark.

The runner preserves the benchmark categories and reports two complementary
metrics: LLM-judge accuracy for comparison with memory-system evaluations, and
the official LoCoMo F1/string score. Image captions are appended to the message
text because some questions rely on caption evidence.

The public LoCoMo release contains ten conversations and 1,986 questions.

Usage:

  PYTHONPATH=.:GroupMemBench python3 run_locomo.py --limit-conv 1 --limit-per-cat 2 --max-sessions 3

  PYTHONPATH=.:GroupMemBench python3 run_locomo.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "speakermem_pkg" / "src"))
sys.path.insert(0, str(HERE / "GroupMemBench"))

from baselines.rag_common.eval_lib import load_env_file, read_text, split_reasoning_and_final, parse_judgment
load_env_file(str(HERE / ".env"))

from speakermem import SpeakerMemory
from speakermem.config import SpeakerMemConfig

CAT_NAME = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
ABSTAIN_GOLD = "There is no information available in the conversation to answer this question."



def parse_ts(s: str) -> str:
    """Parse a LoCoMo timestamp into ISO-like text; return the original value on failure."""
    try:
        return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return s


def load_conv(sample: dict, max_sessions: int = 0):
    """Load one LoCoMo conversation into messages, questions, and speakers."""
    conv = sample["conversation"]
    sess_keys = sorted([k for k in conv if re.fullmatch(r"session_\d+", k)],
                       key=lambda x: int(x.split("_")[1]))
    if max_sessions:
        sess_keys = sess_keys[:max_sessions]
    msgs, kept_dia = [], set()
    for sk in sess_keys:
        ts = parse_ts(conv.get(f"{sk}_date_time", ""))
        for t in conv[sk]:
            text = (t.get("text") or "").strip()
            cap = (t.get("blip_caption") or "").strip()
            if cap:
                text = f"{text} [shares a photo: {cap}]" if text else f"[shares a photo: {cap}]"
            if not text:
                continue
            kept_dia.add(t["dia_id"])
            msgs.append({"speaker": t["speaker"], "content": text,
                         "session": sk, "ts": ts, "msg_id": t["dia_id"]})
    qs = []
    for i, q in enumerate(sample["qa"]):
        cat = int(q.get("category", 0))
        if cat == 5:
            gold = ABSTAIN_GOLD
        else:
            gold = q.get("answer", "")

        ev = q.get("evidence") or []
        if max_sessions and ev and not any(e in kept_dia for e in ev):
            continue
        qs.append({"id": f"{sample['sample_id']}_q{i}", "question": q["question"],
                   "answer": str(gold), "category": CAT_NAME.get(cat, f"cat{cat}"),
                   "raw_category": cat, "evidence": ev})
    return msgs, qs, [conv.get("speaker_a", ""), conv.get("speaker_b", "")]



def _norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c.isalnum() or c.isspace())
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for w in p:
        if w in g:
            common[w] = min(p.count(w), g.count(w))
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def official_score(pred: str, q: dict) -> float:
    """Match the official eval_question_answering behavior."""
    cat = q["raw_category"]
    if cat == 5:
        low = (pred or "").lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0
    gold = q["answer"]
    if cat == 3:
        gold = gold.split(";")[0].strip()
    if cat == 1:
        subs = [x.strip() for x in re.split(r"[;,]", gold) if x.strip()] or [gold]
        return sum(max(f1_score(pred, s) for s in [sub]) for sub in subs) / len(subs)
    return f1_score(pred, gold)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "LoCoMo/locomo10.json"))
    ap.add_argument("--persist-dir", default=str(HERE / ".speakermem_locomo"))
    ap.add_argument("--results-dir", default=str(HERE / "results_locomo"))
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--qa-workers", type=int, default=8)
    ap.add_argument("--limit-conv", type=int, default=0, help=">0 evaluate only the first N conversations")
    ap.add_argument("--conv", default="", help="evaluate only the listed sample IDs (comma-separated)")
    ap.add_argument("--limit-per-cat", type=int, default=0, help=">0 evaluate only the first N questions in each category")
    ap.add_argument("--max-sessions", type=int, default=0, help=">0 ingest only the first N sessions per conversation")
    args = ap.parse_args()

    from eval_patches import patch_token_budgets
    patch_token_budgets()
    from llm_clients import make_client
    from baselines.rag_common.eval_lib import call_chat
    client = make_client("deepseek")
    judge_system = read_text(str(HERE / "GroupMemBench/prompts/hipporag_judge_system.txt"))

    data = json.load(open(args.data, encoding="utf-8"))
    if args.conv:
        want = {x.strip() for x in args.conv.split(",") if x.strip()}
        data = [x for x in data if x["sample_id"] in want]
        if not data:
            raise SystemExit(f"No requested sample IDs were found: {want}")
    if args.limit_conv:
        data = data[: args.limit_conv]
    Path(args.persist_dir).mkdir(parents=True, exist_ok=True)
    out_root = Path(args.results_dir); out_root.mkdir(parents=True, exist_ok=True)

    print(f"LoCoMo:{len(data)} conversations | model {args.model} | top_k={args.top_k}", flush=True)
    all_rec = []
    for sample in data:
        sid = sample["sample_id"]
        msgs, qs, speakers = load_conv(sample, args.max_sessions)
        if args.limit_per_cat:
            bycat, keep = defaultdict(int), []
            for q in qs:
                if bycat[q["category"]] < args.limit_per_cat:
                    bycat[q["category"]] += 1; keep.append(q)
            qs = keep
        cfg = SpeakerMemConfig()
        for c in (cfg.writer, cfg.retriever, cfg.answerer):
            for a in ("model", "select_model"):
                if hasattr(c, a):
                    setattr(c, a, args.model)
        db = Path(args.persist_dir) / f"locomo__{sid}.db"
        fresh = not db.exists()
        mem = SpeakerMemory(persist_path=str(db), config=cfg)
        if fresh:
            t0 = time.time()
            print(f"\n=== {sid}: ingest {len(msgs)} turns / {len(set(m['session'] for m in msgs))} session "
                  f"| speakers {speakers} ===", flush=True)
            CH = 500
            for i in range(0, len(msgs), CH):
                mem.ingest(msgs[i:i + CH])
                print(f"  [ingest] {min(i+CH,len(msgs))}/{len(msgs)} "
                      f"({time.time()-t0:.0f}s, derived {mem.stats().get('total',0)})", flush=True)
            mem.flush()
            print(f"  stats={mem.stats()} | roster={mem.roster()}", flush=True)
        else:
            print(f"\n=== {sid}: [reuse] {len(mem.memory)} entries | roster={mem.roster()} ===", flush=True)

        done = [0]

        def one(q):
            try:
                pred = (mem.answer(q["question"], k=args.top_k) or "").strip()
            except Exception as e:
                print(f"    [QA warn] {q['id']}: {type(e).__name__}", flush=True)
                pred = ""
            ju = (f"Question:\n{q['question']}\n\nGold Answer:\n{q['answer']}\n\n"
                  f"Agent Answer:\n{pred}\n")
            try:
                _, jf = split_reasoning_and_final(call_chat(client, args.model, judge_system, ju, 512))
                v = parse_judgment(jf)
            except Exception:
                jf, v = "ERROR", None
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  [QA] {sid}: {done[0]}/{len(qs)}", flush=True)
            return {"conv": sid, "id": q["id"], "category": q["category"], "raw_category": q["raw_category"],
                    "query": q["question"], "gold": q["answer"], "agent_answer": pred,
                    "judge_answer": jf,
                    "verdict": "correct" if v is True else "incorrect" if v is False else "unclear",
                    "official_f1": official_score(pred, q)}

        with ThreadPoolExecutor(max_workers=args.qa_workers) as ex:
            recs = list(ex.map(one, qs))
        mem.close()
        all_rec += recs
        c = sum(r["verdict"] == "correct" for r in recs)
        print(f"[locomo] {sid}: {len(recs)} questions | judge acc {c/max(len(recs),1)*100:.1f}% | "
              f"official F1 {sum(r['official_f1'] for r in recs)/max(len(recs),1)*100:.1f}", flush=True)
        (out_root / f"{sid}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")


    bycat = defaultdict(lambda: [0, 0.0, 0])
    for r in all_rec:
        b = bycat[r["category"]]
        b[0] += r["verdict"] == "correct"; b[1] += r["official_f1"]; b[2] += 1
    n = len(all_rec)
    print(f"\n{'='*68}\nLoCoMo summary({n} questions / {len(data)} conversations)\n{'='*68}")
    print(f"{'category':16s}{'questions':>6}{'judge acc':>12}{'official F1':>10}")
    for k in sorted(bycat, key=lambda x: -bycat[x][2]):
        c, f, t = bycat[k]
        print(f"{k:16s}{t:>6}{c/t*100:>11.1f}%{f/t*100:>10.1f}")
    C = sum(v[0] for v in bycat.values()); F = sum(v[1] for v in bycat.values())
    print(f"{'overall':16s}{n:>6}{C/n*100:>11.1f}%{F/n*100:>10.1f}")
    print("\nMetric note: judge accuracy is comparable with memory-system evaluations; official F1 follows the LoCoMo evaluation. Do not mix the two metrics.")
    (out_root / "summary.json").write_text(json.dumps(
        {"n": n, "judge_acc": C / n, "official_f1": F / n,
         "by_cat": {k: {"n": v[2], "judge_acc": v[0] / v[2], "official_f1": v[1] / v[2]}
                    for k, v in bycat.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_root}")


if __name__ == "__main__":
    main()
