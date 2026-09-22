
"""Evaluate SocialMem on SocialMemBench with the same record format as the other baselines.

Use the `speakermem` package for memory construction and retrieval; use the benchmark prompts and DeepSeek for answering and judging.
Write records to results_speakermem_pkg/socialmem/<net>/speakermem__<cat>.jsonl for direct comparison with the other result files.

Usage from the artifact root:
  PYTHONPATH=.:GroupMemBench python3 speakermem_pkg/examples/socialmembench.py \
     --domain all --top-k 10
"""
from __future__ import annotations
import os, sys, json, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

REPRO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPRO)); sys.path.insert(0, str(REPRO / "GroupMemBench"))

from baselines.rag_common.eval_lib import (load_env_file, read_text,
                                           split_reasoning_and_final, parse_judgment)
from bench_loaders import iter_units
from speakermem import SpeakerMemory, SpeakerMemConfig
from speakermem.config import RetrieverConfig
from speakermem.agents.answerer import render_passages


def build_mem(persist_path, top_k, derive_window=0, llm=None, writer_llm=None, answer_system=None,
              model_name=None, writer_model=None, ask_enabled=False, route="both", s2_k=None, s2_source_k=None,
              s2_layer_set="all") -> SpeakerMemory:
    from speakermem.config import WriterConfig
    cfg = SpeakerMemConfig()
    cfg.retriever = RetrieverConfig(top_k=top_k)
    if route not in {"both", "s1", "s2"}:
        raise ValueError(f"Unknown QA route: {route}")
    cfg.retriever.s1_enabled = route in {"both", "s1"}
    cfg.retriever.s2_enabled = route in {"both", "s2"}

    cfg.retriever.ask_enabled = ask_enabled
    if s2_k is not None:
        cfg.retriever.s2_k = int(s2_k)
    if s2_source_k is not None:
        cfg.retriever.s2_source_k = int(s2_source_k)
    layer_sets = {
        "all": (),
        "per_speaker": ("per_speaker_core", "per_speaker_profile"),
        "group": ("group_interaction", "group_insight"),
    }
    if s2_layer_set not in layer_sets:
        raise ValueError(f"Unknown structured-layer set: {s2_layer_set}")
    cfg.retriever.s2_layers = layer_sets[s2_layer_set]
    if derive_window > 0:
        cfg.writer = WriterConfig(derive_every="message", derive_window=derive_window)
    if model_name:
        cfg.writer.model = model_name
        cfg.retriever.select_model = model_name
        cfg.answerer.model = model_name
        cfg.writer.max_tokens = 4096
    if writer_model:
        cfg.writer.model = writer_model
        cfg.writer.max_tokens = 4096
    kw = {}
    if llm is not None:
        kw["llm"] = llm
    if answer_system:
        kw["answer_system"] = answer_system
    return SpeakerMemory(persist_path=persist_path, config=cfg, writer_llm=writer_llm, **kw)


def run_unit(unit, messages, questions, mem, client, llm_model, agent_system, judge_system,
             top_k, results_dir, limit=None, qa_workers=16, bench="socialmem",
             answer_local=False, use_our_answer=False, question_ids=None) -> Dict:
    from baselines.rag_common.eval_lib import call_chat
    cats: Dict[str, List[dict]] = {}
    for q in questions:
        cats.setdefault(q.get("category", "unknown"), []).append(q)
    tasks = [(c, q) for c, qs in cats.items()
             for q in (qs[:limit] if limit else qs)
             if question_ids is None or q.get("id", "") in question_ids
             or f"{unit}/{q.get('id', '')}" in question_ids]
    import threading, time as _time
    _prog = {"n": 0, "t0": _time.time()}
    _plock = threading.Lock()
    _ntot = len(tasks)

    def _one(task):

        try:
            return _one_inner(task)
        except Exception as e:
            cat, q = task
            print(f"    [QA warn] {q.get('id','?')}: {type(e).__name__} {str(e)[:60]}", flush=True)
            return cat, {"id": q.get("id", ""), "query": q["question"], "gold": q.get("answer", ""),
                         "context": "", "agent_answer": "", "judge_answer": f"ERROR {type(e).__name__}",
                         "verdict": "incorrect"}

    def _one_inner(task):
        cat, q = task
        entries = mem.retrieve(q["question"], asker=q.get("asking_user_id", ""), k=top_k)
        passages = render_passages(entries)
        if use_our_answer:
            af = (mem.answer(q["question"], asker=q.get("asking_user_id", ""), k=top_k) or "").strip()

        elif answer_local:
            _, af = split_reasoning_and_final(mem._answerer.answer(q["question"], entries))
        else:
            au = (f"Question:\n{q['question']}\n\nRetrieved passages:\n{passages}\n\n"
                  "Answer the question using the retrieved passages.")
            _, af = split_reasoning_and_final(call_chat(client, llm_model, agent_system, au, 1024))
        ju = f"Question:\n{q['question']}\n\nGold Answer:\n{q.get('answer','')}\n\nAgent Answer:\n{af}\n"
        _, jf = split_reasoning_and_final(call_chat(client, llm_model, judge_system, ju, 512))
        v = parse_judgment(jf)
        with _plock:
            _prog["n"] += 1
            if _prog["n"] % 10 == 0 or _prog["n"] == _ntot:
                el = _time.time() - _prog["t0"]
                print(f"  [QA] {unit}: {_prog['n']}/{_ntot}  ({el:.0f}s, {_prog['n']/el:.2f}questions/s)", flush=True)
        return cat, {"id": q.get("id", ""), "query": q["question"], "gold": q.get("answer", ""),
                     "context": passages, "agent_answer": af, "judge_answer": jf,
                     "verdict": "correct" if v is True else "incorrect" if v is False else "unclear"}

    by_cat: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=qa_workers) as ex:
        for cat, rec in ex.map(_one, tasks):
            by_cat.setdefault(cat, []).append(rec)
    summary = {}
    for cat, recs in by_cat.items():
        out = results_dir / bench / unit / f"speakermem__{cat}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = sum(1 for r in recs if r["verdict"] == "correct")
        summary[f"{unit}/{cat}"] = {"correct": c, "total": len(recs)}
    n = len(tasks); cc = sum(s["correct"] for s in summary.values())
    print(f"[speakermem] {unit}: {n} questions, accuracy {cc/n*100:.1f}% ({cc}/{n})" if n else f"[speakermem] {unit}: 0", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="socialmem", choices=["socialmem", "groupmem", "evermem"])
    ap.add_argument("--domain", "--unit", dest="domain", default="all")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--derive-window", type=int, default=0,
                    help="derive memory in message-count windows for units without sessions; 0 uses sessions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--persist-dir", default=str(REPRO / ".speakermem_pkg_store"))
    ap.add_argument("--results-dir", default=str(REPRO / "results_speakermem_pkg"))
    ap.add_argument("--llm-model", default="deepseek-v4-flash")
    ap.add_argument("--qa-workers", type=int, default=16)
    ap.add_argument("--local-model", default=None,
                    help="use a local Hugging Face model for memory writing, retrieval, and answering; keep the judge on DeepSeek")
    ap.add_argument("--max-messages", type=int, default=0,
                    help=">0: ingest only the first N messages per unit for an end-to-end smoke test")
    ap.add_argument("--vllm-url", default=None,
                    help="an OpenAI-compatible vLLM endpoint used for memory writing, retrieval, and answering; the judge remains DeepSeek")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="model name loaded by the vLLM service")
    ap.add_argument("--writer-vllm-url", default=None,
                    help="the vLLM endpoint used only by the writer; selection, projection, and answering use DeepSeek")
    ap.add_argument("--writer-vllm-model", default=None,
                    help="model name served by the writer endpoint; defaults to --vllm-model")
    ap.add_argument("--our-answerer", action="store_true",
                    help="★ use the speaker-aware answerer and two-pass raw retrieval instead of the benchmark answer prompt")
    ap.add_argument("--provider", default="deepseek",
                    help="LLM provider: deepseek (default) or zju")
    ap.add_argument("--ask-enabled", action="store_true",
                    help="enable query expansion during retrieval; disabled by default")
    ap.add_argument("--qa-route", choices=["both", "s1", "s2"], default="both",
                    help="QA retrieval route: both uses raw and derived tracks; s1 uses raw evidence only; s2 uses derived memory only")
    ap.add_argument("--s2-k", type=int, default=None, help="number of derived evidence items per row")
    ap.add_argument("--s2-source-k", type=int, default=None, help="additional group-source items per row")
    ap.add_argument("--s2-layer-set", choices=["all", "per_speaker", "group"], default="all",
                    help="structured layers: all, per_speaker, or group")
    ap.add_argument("--question-ids-file", default=None,
                    help="run only question IDs listed in the file")
    ap.add_argument("--domain-list-file", default=None,
                    help="process only benchmark units listed in the file")
    ap.add_argument("--ingest-only", action="store_true",
                    help="ingest and validate memory only; do not run QA or judging")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a store only when its completion marker and input-message count are valid")
    ap.add_argument("--force-reingest", action="store_true",
                    help="delete the selected unit store and marker before ingesting it again")
    args = ap.parse_args()
    question_ids = None
    if args.question_ids_file:
        question_ids = {line.strip() for line in Path(args.question_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"[QA filter] run {len(question_ids)} selected question IDs", flush=True)
    selected_units = None
    if args.domain_list_file:
        selected_units = {line.strip() for line in Path(args.domain_list_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"[UNIT filter] process {len(selected_units)} selected units", flush=True)
    if args.resume and args.force_reingest:
        ap.error("--resume and --force-reingest cannot be used together")
    try:
        import torch; torch.set_num_threads(8)
    except Exception:
        pass
    load_env_file(str(REPRO / ".env"))
    from eval_patches import patch_token_budgets; patch_token_budgets()
    from llm_clients import make_client
    client = make_client(args.provider)
    agent_system = read_text(str(REPRO / "GroupMemBench/prompts/hipporag_agent_system.txt"))
    judge_system = read_text(str(REPRO / "GroupMemBench/prompts/hipporag_judge_system.txt"))
    persist_root = Path(args.persist_dir); persist_root.mkdir(parents=True, exist_ok=True)
    local_llm = None
    writer_llm = None
    if args.provider == "zju":
        from speakermem import OpenAICompatLLM
        zmodel = os.environ.get("ZJU_MODEL", "qwen3.5-27b-256k-local")
        local_llm = OpenAICompatLLM(api_key=os.environ.get("ZJU_API_KEY"),
                                    base_url=os.environ.get("ZJU_BASE_URL"),
                                    default_model=zmodel,
                                    disable_thinking_extra_body=False,
                                    trust_env=False)
        args.llm_model = zmodel
        print(f"[zju] memory writing, retrieval, answering, and judging use ZJU {zmodel} (QA workers={args.qa_workers})", flush=True)
    elif args.vllm_url:
        from speakermem import OpenAICompatLLM
        print(f"[vllm] use vLLM endpoint {args.vllm_url} with model {args.vllm_model} (QA workers={args.qa_workers})", flush=True)
        local_llm = OpenAICompatLLM(api_key="EMPTY", base_url=args.vllm_url,
                                    default_model=args.vllm_model,
                                    disable_thinking_extra_body=False)
    elif args.local_model:
        from speakermem import LocalHFLLM
        print(f"[local] loading local model {args.local_model} ...", flush=True)
        local_llm = LocalHFLLM(args.local_model)
        if args.qa_workers > 1:
            print("[local] local model generation is not thread-safe; setting qa-workers to 1", flush=True)
            args.qa_workers = 1
    if args.writer_vllm_url:
        from speakermem import OpenAICompatLLM
        writer_model = args.writer_vllm_model or args.vllm_model
        writer_llm = OpenAICompatLLM(api_key="EMPTY", base_url=args.writer_vllm_url,
                                     default_model=writer_model, disable_thinking_extra_body=False)
        print(f"[writer-vllm] writer uses {args.writer_vllm_url} with model {writer_model}; "
              "SELECT/PROJECT/ANSWER + QA judge=DeepSeek", flush=True)
    import re as _re
    all_summary = {}
    for unit, messages, questions in iter_units(args.benchmark, REPRO, args.domain):
        if selected_units is not None and str(unit) not in selected_units:
            continue
        if args.max_messages:
            messages = messages[:args.max_messages]
        safe = _re.sub(r"[^a-zA-Z0-9._-]", "-", str(unit))
        pkl = persist_root / f"{args.benchmark}__{safe}.db"
        marker = pkl.with_suffix(pkl.suffix + ".complete.json")
        expected_raw = len(messages)
        if args.force_reingest:

            for path in (pkl, Path(str(pkl) + "-wal"), Path(str(pkl) + "-shm"), marker):
                if path.exists():
                    path.unlink()
            print(f"\n=== {unit}: [force-reingest] store cleared ===", flush=True)

        completion = None
        if marker.exists():
            try:
                completion = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                completion = None
        valid_complete = bool(
            pkl.exists() and completion
            and completion.get("benchmark") == args.benchmark
            and str(completion.get("unit")) == str(unit)
            and completion.get("raw_messages") == expected_raw
        )
        if args.resume and not valid_complete:
            raise RuntimeError(
                f"{unit}: --resume requires a complete store whose input matches;"
                f"missing or invalid completion marker {marker.name}."
            )

        fresh = not pkl.exists()
        policy_model = (os.environ.get("ZJU_MODEL", "qwen3.5-27b-256k-local") if args.provider == "zju"
                        else (args.vllm_model if args.vllm_url else (args.local_model or None)))

        ans_sys = None if args.our_answerer else (agent_system if local_llm else None)
        mem = build_mem(str(pkl), args.top_k, derive_window=args.derive_window,
                        llm=local_llm, writer_llm=writer_llm, answer_system=ans_sys, model_name=policy_model,
                        writer_model=(args.writer_vllm_model or args.vllm_model) if args.writer_vllm_url else None,
                        ask_enabled=args.ask_enabled, route=args.qa_route, s2_k=args.s2_k, s2_source_k=args.s2_source_k,
                        s2_layer_set=args.s2_layer_set)
        if fresh:
            import time as _t
            msgs_in = [{**m, "speaker": m.get("author"), "session": m.get("_channel")} for m in messages]
            n_in = len(msgs_in); t0 = _t.time(); CH = 500
            print(f"\n=== {unit}: writing memory({n_in} messages)===", flush=True)
            for i in range(0, n_in, CH):
                mem.ingest(msgs_in[i:i + CH])
                done = min(i + CH, n_in); el = _t.time() - t0
                print(f"  [ingest] {unit}: {done}/{n_in} ({done/n_in*100:.0f}%, {el:.0f}s, "
                      f"{done/max(el,1):.1f}msgs/s, derived={mem.stats().get('total',0)})", flush=True)
            mem.flush()
            stats = mem.stats()
            raw_written = stats.get("per_speaker_episodic", 0)
            if raw_written != expected_raw:
                raise RuntimeError(
                    f"{unit}: raw-layer count mismatch: expected={expected_raw}, actual={raw_written};"
                    "refusing to write the completion marker."
                )
            marker.write_text(json.dumps({
                "benchmark": args.benchmark, "unit": str(unit),
                "raw_messages": expected_raw, "stats": stats,
                "derive_window": args.derive_window,
                "ask_enabled": args.ask_enabled,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  stats: {stats}\n  completion marker: {marker.name}", flush=True)
        else:
            status = "completion marker verified" if valid_complete else "existing store not validated"
            print(f"\n=== {unit}: [reuse] {len(mem.memory)} entries; {status} ===", flush=True)
        if args.ingest_only:
            mem.close()
            continue
        all_summary.update(run_unit(unit, messages, questions, mem, client, args.llm_model,
                                    agent_system, judge_system, args.top_k, Path(args.results_dir),
                                    limit=args.limit, qa_workers=args.qa_workers, bench=args.benchmark,
                                    answer_local=bool(local_llm), use_our_answer=args.our_answerer,
                                    question_ids=question_ids))
        mem.close()
    if args.ingest_only:
        print("\nIngestion complete: --ingest-only did not run QA or judging and wrote no summary.")
        return
    out = Path(args.results_dir, f"summary_speakermem_{args.benchmark}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    if out.exists():
        try:
            summary = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    summary.update(all_summary)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
