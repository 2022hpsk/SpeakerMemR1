"""A shared loader for the three multi-party memory benchmarks.

This module normalizes the native formats of GroupMemBench, SocialMemBench, and
EverMemBench into two structures consumed directly by run_baseline.py:
  - messages: list[dict]. Each record contains at least content for BM25 indexing,
    plus author/_channel/timestamp/msg_node metadata used to render passages for
    the LLM through eval_lib.format_retrieved_message.
  - questions: list[dict]. Each record contains id/question/answer/asking_user_id/category.
    Multiple-choice options are appended to question, and gold answers use the
    format "letter. text" for judge comparison.

Each benchmark is partitioned into independent units (GroupMemBench=domain,
SocialMemBench=network, EverMemBench=topic). Each unit receives its own BM25
index and retrieval pass; units are never mixed.

Interface: iter_units(benchmark, here, unit_filter="all") yields
(unit_name, messages, questions).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


import sys


def _gmb_helpers(here: Path):
    repo = here / "GroupMemBench"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from baselines.rag_common.eval_lib import load_conversation_messages, load_questions
    return load_conversation_messages, load_questions, repo


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _clean(s: Any) -> str:
    return html.unescape(s).strip() if isinstance(s, str) else ""





GMB_DOMAINS = ["Finance", "Technology", "Healthcare", "Manufacturing"]
GMB_QTYPES = ["multi_hop", "knowledge_update", "term_ambiguity",
              "user_implicit", "temporal", "abstention"]


def iter_groupmem(here: Path, unit_filter: str = "all") -> Iterator[Tuple[str, List[dict], List[dict]]]:
    load_conv, load_q, repo = _gmb_helpers(here)
    domains = GMB_DOMAINS if unit_filter == "all" else [unit_filter]
    for dom in domains:
        cp = repo / "data/final" / dom / f"synthetic_domain_channels_rolevariants_{dom}.json"
        if not cp.exists():
            continue
        messages = load_conv(str(cp))
        questions: List[dict] = []
        for qt in GMB_QTYPES:
            qp = repo / "questions" / dom / f"{qt}.jsonl"
            if not qp.exists():
                continue
            for q in load_q(str(qp)):
                questions.append({"id": q.get("id", ""), "question": q["question"],
                                  "answer": q.get("answer", ""),
                                  "asking_user_id": q.get("asking_user_id", ""),
                                  "category": qt})
        yield dom, messages, questions





def _mc_question(question: str, options: Dict[str, str]) -> str:
    """Append multiple-choice options to the question text."""
    if not options:
        return question
    lines = [f"{k}. {v}" for k, v in options.items()]
    return f"{question}\n\nOptions:\n" + "\n".join(lines)


def iter_socialmem(here: Path, unit_filter: str = "all") -> Iterator[Tuple[str, List[dict], List[dict]]]:
    base = here / "SocialMemBench"
    convs = _read_jsonl(base / "conversations.jsonl")
    qas = _read_jsonl(base / "qa.jsonl")


    by_net_msg: Dict[str, List[dict]] = {}
    for c in convs:
        m = {"content": _clean(c.get("message", "")),
             "author": c.get("speaker_display_name", "?"),
             "_channel": c.get("session_id", ""),
             "topic": c.get("session_topic", ""),
             "timestamp": str(c.get("timestamp", "")),
             "reply_to": c.get("reply_to_turn_id", ""),
             "msg_node": c.get("turn_id", "")}
        by_net_msg.setdefault(c.get("network_id", ""), []).append(m)
    for msgs in by_net_msg.values():
        msgs.sort(key=lambda x: (x["_channel"], x["timestamp"], x["msg_node"]))


    by_net_q: Dict[str, List[dict]] = {}
    for q in qas:
        opts = {}
        oj = q.get("options_json")
        if oj and oj not in ("[]", ""):
            try:
                parsed = json.loads(oj)
                if isinstance(parsed, dict):
                    opts = parsed
            except Exception:
                opts = {}
        gold = q.get("answer", "")
        if opts and q.get("correct_option"):
            letter = q["correct_option"]
            gold = f"{letter}. {opts.get(letter, gold)}"
        by_net_q.setdefault(q.get("network_id", ""), []).append({
            "id": q.get("qa_id", ""),
            "question": _mc_question(q["question"], opts),
            "answer": gold,
            "asking_user_id": "",
            "category": q.get("query_type", "unknown")})

    nets = sorted(by_net_msg) if unit_filter == "all" else [unit_filter]
    for net in nets:
        if net in by_net_msg and net in by_net_q:
            yield net, by_net_msg[net], by_net_q[net]





def iter_evermem(here: Path, unit_filter: str = "all") -> Iterator[Tuple[str, List[dict], List[dict]]]:
    base = here / "EverMemBench-Dynamic"
    dials = _read_jsonl(base / "dialogues.jsonl")
    qars = _read_jsonl(base / "qars.jsonl")


    by_topic_msg: Dict[str, List[dict]] = {}
    for row in dials:
        tid = str(row.get("topic_id", ""))
        date = str(row.get("date", ""))
        groups = row.get("dialogues", {}) or {}
        if not isinstance(groups, dict):
            continue
        for gname, msgs in groups.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                by_topic_msg.setdefault(tid, []).append({
                    "content": _clean(m.get("dialogue", "")),
                    "author": m.get("speaker", "?"),
                    "_channel": gname,
                    "timestamp": str(m.get("time", "")),
                    "msg_node": f"{date}|{gname}|{m.get('message_index', '')}"})
    for msgs in by_topic_msg.values():
        msgs.sort(key=lambda x: (x["timestamp"], x["_channel"], x["msg_node"]))

    by_topic_q: Dict[str, List[dict]] = {}
    for q in qars:
        tid = str(q.get("topic_id", ""))
        opts = q.get("options") if isinstance(q.get("options"), dict) else {}
        gold = q.get("A", "")
        if opts and isinstance(gold, str) and gold in opts:
            gold = f"{gold}. {opts[gold]}"
        by_topic_q.setdefault(tid, []).append({
            "id": q.get("id", ""),
            "question": _mc_question(q.get("Q", ""), opts),
            "answer": gold,
            "asking_user_id": "",
            "category": ("multiple_choice" if opts else "open_ended")})

    topics = sorted(by_topic_msg) if unit_filter == "all" else [unit_filter]
    for tid in topics:
        if tid in by_topic_msg and tid in by_topic_q:
            yield tid, by_topic_msg[tid], by_topic_q[tid]


LOADERS = {"groupmem": iter_groupmem, "socialmem": iter_socialmem, "evermem": iter_evermem}


def iter_units(benchmark: str, here: Path, unit_filter: str = "all"):
    if benchmark not in LOADERS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available loaders: {list(LOADERS)}")
    return LOADERS[benchmark](here, unit_filter)
