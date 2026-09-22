#!/usr/bin/env python3
"""Run retrieval and question-answering baselines for GroupMemBench, SocialMemBench,
and EverMemBench.

The runner supports BM25, dense retrieval, and full-context evaluation. The
retrieval-only mode performs an offline evidence-recall check; the default mode
runs the answerer and judge through an external LLM client. The upstream
benchmark checkout is kept unchanged, and all adapters live in this file.

Example:

  python run_baseline.py --baseline bm25 --domain Finance --qtype multi_hop \
      --retrieval-only --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE / "GroupMemBench"
if not REPO.exists():
    sys.exit(f"[error] Upstream repository not found {REPO}. Clone it first:"
             f"git clone https://github.com/UCSB-NLP-Chang/GroupMemBench.git (or run setup.sh)")
sys.path.insert(0, str(REPO))

# Upstream modules (the repository has been added to sys.path)
from baselines.rag_common.eval_lib import (  # noqa: E402
    load_conversation_messages,
    load_env_file,
    load_questions,
    message_index_text,
    read_text,
    format_retrieved_message,
    split_reasoning_and_final,
    parse_judgment,
)
from rank_bm25 import BM25Okapi  # noqa: E402

DOMAINS = ["Finance", "Technology", "Healthcare", "Manufacturing"]
QTYPES = ["multi_hop", "knowledge_update", "term_ambiguity",
          "user_implicit", "temporal", "abstention"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenization; preserve numbers, timestamps, and user IDs, with no stop-word removal."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def conv_path(domain: str) -> Path:
    return REPO / "data" / "final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"


def questions_path(domain: str, qtype: str) -> Path:
    return REPO / "questions" / domain / f"{qtype}.jsonl"


# --------------------------------------------------------------------------- #
# Retrievers, independent from question answering and matching the upstream design
# --------------------------------------------------------------------------- #
def build_bm25_retriever(messages: List[Dict]) -> Callable[[str, int], List[int]]:
    """One message is one document; index only the message body and attach metadata when showing a passage to the agent."""
    corpus = [tokenize(message_index_text(m)) for m in messages]
    bm25 = BM25Okapi(corpus)

    def retrieve(query: str, k: int) -> List[int]:
        q = tokenize(query)
        if not q:
            return []
        scores = bm25.get_scores(q)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return retrieve


# Dense semantic retrieval baseline corresponding to Uncompressed/naive_rag and the GroupMem text-embedding-3-large baseline.
# Use a local sentence-transformers model instead of an embedding API; the interface matches BM25: retrieve(query, k) -> indices.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # overridden by --embed-model in main
FULL_CONTEXT_MAX_MSGS = 400             # full_context safety limit (truncate to avoid an oversized context)
_ST_CACHE: Dict[str, Any] = {}


def _get_st_model(name: str):
    import os as _os
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _os.environ.setdefault(_v, "8")
    if name not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_CACHE[name] = SentenceTransformer(name)
    return _ST_CACHE[name]


EMBED_STORE = HERE / ".embed_store"   # persistent message-embedding directory (reusable and ignored by git)


def build_embed_retriever(messages: List[Dict]) -> Callable[[str, int], List[int]]:
    """Dense RAG: embed each message and the query, then retrieve by cosine similarity with a local sentence-transformers model.
    The message-embedding matrix is cached by model and corpus hash under .embed_store/ and reused on later runs."""
    import numpy as np, threading, hashlib
    corpus = [message_index_text(m) for m in messages]
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "-", EMBED_MODEL_NAME)
    key = hashlib.sha1(("\n".join(corpus)).encode("utf-8")).hexdigest()[:16]
    cache_dir = EMBED_STORE / safe_model
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{key}_n{len(corpus)}.npy"

    model = _get_st_model(EMBED_MODEL_NAME)  # the query still needs encoding
    if cache.exists():
        M = np.load(cache)  # reuse cached message embeddings
    else:
        M = model.encode(corpus, normalize_embeddings=True, batch_size=128,
                         show_progress_bar=False).astype("float32")  # (N,d)
        np.save(cache, M)
    lock = threading.Lock()

    def retrieve(query: str, k: int) -> List[int]:
        with lock:  # ST encoding is not thread-safe; each query is short
            q = model.encode([query], normalize_embeddings=True)[0].astype("float32")
        scores = M @ q
        return np.argsort(-scores)[:k].tolist()

    return retrieve


def build_full_context_retriever(messages: List[Dict]) -> Callable[[str, int], List[int]]:
    """full_context (a mini-oracle): do not retrieve; return the conversation in order, truncated to the safety limit. Only small units are practical."""
    n = min(len(messages), FULL_CONTEXT_MAX_MSGS)
    idxs = list(range(n))

    def retrieve(query: str, k: int) -> List[int]:
        return idxs  # ignore k and return the full context

    return retrieve


BASELINES = {"bm25": build_bm25_retriever, "embed": build_embed_retriever,
             "full_context": build_full_context_retriever}


# --------------------------------------------------------------------------- #
# Retrieval-only proxy metric: whether the gold answer appears in the top-k passages
# --------------------------------------------------------------------------- #
def gold_recall_at_k(messages, questions, retrieve, k: int) -> Dict[str, float]:
    hits, total, skipped = 0, 0, 0
    for q in questions:
        gold = (q.get("answer") or "").strip().lower()
        # Abstention answers are generic phrases such as no information available and are not searchable evidence
        if not gold or gold.startswith("there is no information"):
            skipped += 1
            continue
        asker = q.get("asking_user_id") or ""
        query = f"{asker} {q['question']}" if asker else q["question"]
        idxs = retrieve(query, k)
        joined = " ".join(message_index_text(messages[i]).lower() for i in idxs if 0 <= i < len(messages))
        if gold in joined:
            hits += 1
        total += 1
    return {"recall_at_k": (hits / total if total else 0.0),
            "hits": hits, "scored": total, "skipped_abstention": skipped}


# --------------------------------------------------------------------------- #
# Additional metrics: token EM and token F1 as free references beyond judge accuracy
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().rstrip("."))


def token_f1(pred: str, gold: str) -> float:
    p, g = set(_norm(pred).split()), set(_norm(gold).split())
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    prec, rec = len(common) / len(p), len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> bool:
    return _norm(pred) == _norm(gold)


# --------------------------------------------------------------------------- #
# Global question-level parallel QA: pool all unit-category questions and process them concurrently.
#   Retrieval is read-only and each agent/judge task is stateless, so parallelism does not change accuracy.
#   This keeps workers busy even when some units have few questions.
# --------------------------------------------------------------------------- #
def run_global_qa(tasks, retrieves, msgs_by_unit, client, agent_model, judge_model,
                  agent_system, judge_system, top_k, workers, call_chat, progress_every=50):
    import time
    from concurrent.futures import ThreadPoolExecutor

    def _one(task):
        unit, q = task["unit"], task["q"]
        retrieve, messages = retrieves[unit], msgs_by_unit[unit]
        rec = {"unit": unit, "category": task["cat"], "id": q.get("id", ""),
               "asking_user_id": q.get("asking_user_id", "")}
        gold = q.get("answer", "")
        asker = q.get("asking_user_id") or ""
        query = f"{asker} {q['question']}" if asker else q["question"]
        t0 = time.time(); idxs = retrieve(query, top_k); rec["t_retrieve"] = time.time() - t0
        docs = [format_retrieved_message(messages[i]) for i in idxs if 0 <= i < len(messages)]
        passages = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))
        asker_line = f"Asking user: {asker}\n\n" if asker else ""
        agent_user = (f"{asker_line}Question:\n{q['question']}\n\nRetrieved passages:\n{passages}\n\n"
                      "Answer the question using the retrieved passages.")
        t0 = time.time(); ao = call_chat(client, agent_model, agent_system, agent_user, 512)
        rec["t_agent"] = time.time() - t0
        _, afinal = split_reasoning_and_final(ao)
        ju = f"Question:\n{q['question']}\n\nGold Answer:\n{gold}\n\nAgent Answer:\n{afinal}\n"
        t0 = time.time(); jo = call_chat(client, judge_model, judge_system, ju, 512)
        rec["t_judge"] = time.time() - t0
        _, jfinal = split_reasoning_and_final(jo)
        v = parse_judgment(jfinal)
        rec["verdict"] = "correct" if v is True else "incorrect" if v is False else "unclear"
        rec["em"] = exact_match(afinal, gold); rec["f1"] = token_f1(afinal, gold)
        # Record each question, retrieved context, answer, gold answer, judge rationale, and verdict for inspection
        rec["question"] = q["question"]
        rec["context"] = passages           # retrieved content supplied to the agent (top-k passages or full context)
        rec["agent_answer"] = afinal; rec["gold"] = gold
        rec["judge_answer"] = jfinal
        return rec

    t_wall = time.time(); records = []; done = 0; n = len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_one, tasks):
            records.append(r); done += 1
            if done % progress_every == 0 or done == n:
                acc = sum(1 for x in records if x["verdict"] == "correct") / done
                print(f"  progress {done}/{n} | live accuracy {acc*100:.1f}% | {time.time()-t_wall:.0f}s", flush=True)
    return records, round(time.time() - t_wall, 1)


def group_stats(recs: List[dict]) -> Dict[str, float]:
    n = len(recs)
    c = sum(1 for r in recs if r["verdict"] == "correct")
    inc = sum(1 for r in recs if r["verdict"] == "incorrect")
    unc = sum(1 for r in recs if r["verdict"] == "unclear")
    return {"accuracy": c / n if n else 0.0, "correct": c, "incorrect": inc, "unclear": unc,
            "total": n, "unclear_rate": unc / n if n else 0.0,
            "em": sum(1 for r in recs if r["em"]) / n if n else 0.0,
            "f1": sum(r["f1"] for r in recs) / n if n else 0.0,
            "sum_retrieve_s": round(sum(r["t_retrieve"] for r in recs), 1),
            "sum_agent_s": round(sum(r["t_agent"] for r in recs), 1),
            "sum_judge_s": round(sum(r["t_judge"] for r in recs), 1),
            "wall_s": 0.0}


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="bm25", choices=list(BASELINES))
    ap.add_argument("--benchmark", default="groupmem", choices=["groupmem", "socialmem", "evermem"],
                    help="benchmark: groupmem (domain units), socialmem (network units), or evermem (topic units)")
    ap.add_argument("--domain", "--unit", dest="domain", default="all",
                    help="unit filter: GroupMemBench domain, SocialMemBench network_id, or EverMemBench topic_id; or all")
    ap.add_argument("--qtype", default="all", help="question-category filter, or all")
    ap.add_argument("--limit", type=int, default=None, help="maximum questions per unit")
    ap.add_argument("--question-ids-file", default=None,
                    help="evaluate only question IDs listed in the file; each line may contain unit/id or an ID")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="local sentence-transformers model for the dense baseline")
    ap.add_argument("--retrieval-only", action="store_true", help="retrieval only; do not call an LLM")
    ap.add_argument("--llm-provider", default="deepseek",
                    choices=["deepseek", "anthropic", "openai", "local"])
    ap.add_argument("--agent-model", default="deepseek-v4-flash")
    ap.add_argument("--judge-model", default="deepseek-v4-flash")
    ap.add_argument("--agent-prompt", default=str(REPO / "prompts/hipporag_agent_system.txt"))
    ap.add_argument("--judge-prompt", default=str(REPO / "prompts/hipporag_judge_system.txt"))
    ap.add_argument("--env-file", default=str(HERE / ".env"), help="credential file; defaults to .env")
    ap.add_argument("--workers", type=int, default=8,
                    help="QA worker count; questions are independent and retrieval is read-only")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    question_ids = None
    if args.question_ids_file:
        question_ids = {line.strip() for line in Path(args.question_ids_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")}
        print(f"[question filter] loaded {len(question_ids)} question IDs")
    globals()["EMBED_MODEL_NAME"] = args.embed_model  # used by the dense baseline

    import time
    load_env_file(args.env_file)  # load keys from .env without overwriting existing variables

    from bench_loaders import iter_units  # shared loader for the three benchmarks

    client = None
    call_chat = None
    if not args.retrieval_only:
        from llm_clients import make_client
        from eval_patches import patch_token_budgets
        patch_token_budgets()  # use a large completion allowance and temperature=0 to avoid truncated Final lines
        import baselines.rag_common.eval_lib as _elib
        call_chat = _elib.call_chat   # use the patched call_chat
        client = make_client(args.llm_provider)  # missing keys produce a clear error
        agent_system = read_text(args.agent_prompt)
        judge_system = read_text(args.judge_prompt)

    os.makedirs(args.results_dir, exist_ok=True)
    summary: Dict[str, Dict] = {}

    # ── Stage 1: build all unit indexes and pool question-level tasks──
    retrieves: Dict[str, Callable] = {}
    msgs_by_unit: Dict[str, List[dict]] = {}
    idx_times: Dict[str, float] = {}
    tasks: List[dict] = []
    percat: Dict[tuple, int] = {}
    for unit, messages, questions in iter_units(args.benchmark, HERE, args.domain):
        t_idx = time.time()
        retrieves[unit] = BASELINES[args.baseline](messages)
        idx_times[unit] = time.time() - t_idx
        msgs_by_unit[unit] = messages
        print(f"[index] [{args.benchmark}] {unit}:{len(messages)} messages, index built {idx_times[unit]:.1f}s")
        for q in questions:
            cat = q.get("category", "unknown")
            if args.qtype != "all" and cat != args.qtype:
                continue
            if question_ids is not None:
                qid = str(q.get("id", ""))
                if qid not in question_ids and f"{unit}/{qid}" not in question_ids:
                    continue
            k = (unit, cat)
            percat[k] = percat.get(k, 0) + 1
            if args.limit and percat[k] > args.limit:   # limit questions per unit-category pair
                continue
            tasks.append({"unit": unit, "cat": cat, "q": q})
    print(f"[summary] {len(tasks)} questions, {len(retrieves)} units, total index time {sum(idx_times.values()):.1f}s")

    # ── Stage 2: evaluation ──
    qa_wall = 0.0
    if args.retrieval_only:
        groups: Dict[tuple, List[dict]] = {}
        for t in tasks:
            groups.setdefault((t["unit"], t["cat"]), []).append(t["q"])
        for (unit, cat), qs in groups.items():
            res = gold_recall_at_k(msgs_by_unit[unit], qs, retrieves[unit], args.top_k)
            summary[f"{unit}/{cat}"] = res
    else:
        print(f"[QA] global question-level parallelism with {args.workers} workers over {len(tasks)} questions ...")
        records, qa_wall = run_global_qa(
            tasks, retrieves, msgs_by_unit, client, args.agent_model, args.judge_model,
            agent_system, judge_system, args.top_k, args.workers, call_chat)
        # group by unit and category, then write question-level JSONL and summaries
        gr: Dict[tuple, List[dict]] = {}
        for r in records:
            gr.setdefault((r["unit"], r["category"]), []).append(r)
        for (unit, cat), recs in gr.items():
            out = Path(args.results_dir) / args.benchmark / unit / f"{args.baseline}__{cat}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            res = group_stats(recs)
            res["index_build_s"] = round(idx_times.get(unit, 0.0), 1)
            summary[f"{unit}/{cat}"] = res
        print(f"[QA] complete, wall time {qa_wall}s ({args.workers} workers)")

    # summary
    print("\n" + "=" * 64)
    report: Dict[str, Dict] = {"per_unit_cat": summary}
    if args.retrieval_only:
        by_cat: Dict[str, List[float]] = {}
        for key, res in summary.items():
            by_cat.setdefault(key.split("/", 1)[1], []).append(res["recall_at_k"])
        print(f"[{args.benchmark}] mean recall:")
        for cat in sorted(by_cat):
            v = by_cat[cat]; print(f"  {cat:18s}: {sum(v)/len(v)*100:5.1f}%  (n={len(v)})")
        allv = [r["recall_at_k"] for r in summary.values()]
        if allv: print(f"  {'overall':18s}: {sum(allv)/len(allv)*100:5.1f}%")
        report["overall"] = {"recall_at_k_mean": (sum(allv)/len(allv) if allv else 0.0)}
    else:
        # micro-average over questions rather than averaging unit-level accuracies
        def agg(keys):
            c = sum(summary[k]["correct"] for k in keys); inc = sum(summary[k]["incorrect"] for k in keys)
            unc = sum(summary[k]["unclear"] for k in keys); t = sum(summary[k]["total"] for k in keys)
            em = sum(summary[k]["em"] * summary[k]["total"] for k in keys)
            f1 = sum(summary[k]["f1"] * summary[k]["total"] for k in keys)
            return {"correct": c, "incorrect": inc, "unclear": unc, "total": t,
                    "accuracy": c / t if t else 0.0, "unclear_rate": unc / t if t else 0.0,
                    "em": em / t if t else 0.0, "f1": f1 / t if t else 0.0}
        # by category
        cats: Dict[str, List[str]] = {}
        for k in summary: cats.setdefault(k.split("/", 1)[1], []).append(k)
        print(f"[{args.benchmark}] by category (judge accuracy / EM / F1 / unclear rate):")
        report["per_category"] = {}
        for cat in sorted(cats):
            a = agg(cats[cat]); report["per_category"][cat] = a
            print(f"  {cat:16s}: acc {a['accuracy']*100:5.1f}%  EM {a['em']*100:5.1f}%  "
                  f"F1 {a['f1']*100:5.1f}%  unclear {a['unclear_rate']*100:4.1f}%  (n={a['total']})")
        # by unit
        units: Dict[str, List[str]] = {}
        for k in summary: units.setdefault(k.split("/", 1)[0], []).append(k)
        report["per_unit"] = {u: agg(ks) for u, ks in units.items()}
        # overall and timing
        ov = agg(list(summary)); report["overall"] = ov
        report["timing"] = {"qa_wall_s": qa_wall,
                            "index_build_s": round(sum(idx_times.values()), 1),
                            "agent_s_sum": round(sum(r["sum_agent_s"] for r in summary.values()), 1),
                            "judge_s_sum": round(sum(r["sum_judge_s"] for r in summary.values()), 1),
                            "retrieve_s_sum": round(sum(r["sum_retrieve_s"] for r in summary.values()), 1),
                            "workers": args.workers}
        print(f"  {'overall':16s}: acc {ov['accuracy']*100:5.1f}%  EM {ov['em']*100:5.1f}%  "
              f"F1 {ov['f1']*100:5.1f}%  unclear {ov['unclear_rate']*100:4.1f}%  (n={ov['total']})")
        print(f"  timing: QA wall time {qa_wall}s ({args.workers} workers) / index build {report['timing']['index_build_s']}s "
              f"/ agent {report['timing']['agent_s_sum']}s+judge {report['timing']['judge_s_sum']}s (sum of call times)")

    tag = "retrieval_only" if args.retrieval_only else "qa"
    summ_path = Path(args.results_dir) / f"summary_{args.benchmark}_{args.baseline}_{tag}.json"
    summ_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed summary written to {summ_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
