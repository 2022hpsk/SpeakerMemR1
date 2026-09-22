"""Shared utilities for RAG-style baselines (BM25, dense embedding).

Design intent: every step OUTSIDE of retrieval mirrors the agent-memory
baselines (mem0, hipporag, ...) byte-for-byte:
  - same conversation loader (sorted by channel/timestamp/msg_node)
  - same retrieved-passage header format used in the QA prompt
  - same gpt-5 agent + judge call shape
  - same JSONL output schema accepted by task_synthesis/summarize_typed_eval.py

The only thing each RAG baseline contributes is a ``retrieve(query) -> List[int]``
callable that returns indices into the shared message list. No LLM is used for
ingest / extraction; that's the whole point of the comparison.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from llm_utils import chat_completion_text


def load_env_file(env_path: str) -> None:
    if not env_path or not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                os.environ.setdefault(key, value)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_questions(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_conversation_messages(path: str) -> List[Dict[str, Any]]:
    """Same loader the mem0 baseline uses — keeps message order identical."""
    data = json.load(open(path, "r", encoding="utf-8"))
    messages: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for channel, msgs in data.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                m2 = dict(m)
                m2["_channel"] = channel
                messages.append(m2)
    messages.sort(key=lambda m: (m.get("_channel", ""), m.get("timestamp", ""), m.get("msg_node", "")))
    return messages


def message_index_text(message: Dict[str, Any]) -> str:
    """Text that the retriever indexes/scores. Per the baseline spec, this is
    the message content only — metadata stays out of the index but is attached
    to the passage when shown to the QA agent (see ``format_retrieved_message``)."""
    content = message.get("content", "")
    if not isinstance(content, str):
        return ""
    return html.unescape(content).strip()


def format_retrieved_message(message: Dict[str, Any]) -> str:
    """Build the same passage header the agent-memory baselines use, so the
    gpt-5 agent sees passages in an identical shape across all baselines."""
    author = message.get("author") or "?"
    tags: List[str] = [f"user={author}"]
    extra_keys = (
        ("role", "speaker_role"),
        ("_channel", "channel"),
        ("phase_name", "phase_name"),
        ("topic", "topic"),
        ("timestamp", "timestamp"),
        ("reply_to", "reply_to"),
        ("msg_node", "msg_node"),
    )
    for src_key, label in extra_keys:
        value = message.get(src_key)
        if value not in (None, ""):
            tags.append(f"{label}={value}")
    body = message.get("content", "")
    if isinstance(body, str):
        body = html.unescape(body).strip()
    else:
        body = ""
    return f"[{' / '.join(tags)}]\n{body}".strip()


def extract_final(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final:"):
            return ln.split(":", 1)[1].strip()
        if low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else text.strip()


def split_reasoning_and_final(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    final_idx = None
    for i in range(len(lines) - 1, -1, -1):
        low = lines[i].strip().lower()
        if low.startswith("final:") or low.startswith("final answer:"):
            final_idx = i
            break
    if final_idx is None:
        return text.strip(), extract_final(text)
    reasoning = "\n".join(lines[:final_idx]).strip()
    final = lines[final_idx].split(":", 1)[1].strip()
    return reasoning, final


def parse_judgment(text: str) -> Optional[bool]:
    final = extract_final(text).strip().lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None


def call_chat(client: Any, model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    return chat_completion_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )


def conversation_fingerprint(conversation_path: str) -> Dict[str, Any]:
    st = os.stat(conversation_path)
    return {
        "conversation_path": os.path.abspath(conversation_path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def conversation_id_from_path(path: str) -> str:
    return Path(path).stem


def run_qa(
    *,
    questions: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    retrieve: Callable[[str, int], List[int]],
    top_k: int,
    client: Any,
    agent_model: str,
    judge_model: str,
    agent_system: str,
    judge_system: str,
    output_jsonl: str,
    format_passage: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Tuple[int, int]:
    """Run the standard agent + judge loop given a retrieval callable.

    ``retrieve(query, k)`` must return up to ``k`` integer indices into
    ``messages``, ranked best-first. ``format_passage`` defaults to
    ``format_retrieved_message`` (the mem0/a_mem-style header) but can be
    overridden by baselines whose retrieval unit isn't a chat message
    (e.g. GraphRAG returns per-channel context blobs).
    """
    if format_passage is None:
        format_passage = format_retrieved_message
    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        eval_pbar = tqdm(questions, desc="Evaluating", unit="q", dynamic_ncols=True)
        for q in eval_pbar:
            question_text = q["question"]
            gold_answer = q.get("answer", "")
            asking_user_id = q.get("asking_user_id") or ""
            search_query = f"{asking_user_id} {question_text}" if asking_user_id else question_text

            indices = retrieve(search_query, top_k)
            retrieved_messages = [messages[i] for i in indices if 0 <= i < len(messages)]
            retrieved_docs = [format_passage(m) for m in retrieved_messages]
            passages_text = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs, start=1))

            asker_line = f"Asking user: {asking_user_id}\n\n" if asking_user_id else ""
            agent_user = (
                f"{asker_line}"
                f"Question:\n{question_text}\n\n"
                f"Retrieved passages:\n{passages_text}\n\n"
                "Answer the question using the retrieved passages."
            )
            agent_output = call_chat(client, agent_model, agent_system, agent_user, max_tokens=512)
            agent_reasoning, agent_final = split_reasoning_and_final(agent_output)

            judge_user = (
                f"Question:\n{question_text}\n\n"
                f"Gold Answer:\n{gold_answer}\n\n"
                f"Agent Answer:\n{agent_final}\n"
            )
            judge_output = call_chat(client, judge_model, judge_system, judge_user, max_tokens=256)
            judge_reasoning, judge_final = split_reasoning_and_final(judge_output)
            judgment = parse_judgment(judge_final)

            total += 1
            if judgment is True:
                correct += 1
                verdict = "Correct"
            elif judgment is False:
                verdict = "Incorrect"
            else:
                verdict = "Unclear"

            record = {
                "query": question_text,
                "asking_user_id": asking_user_id,
                "retrieved_docs": retrieved_docs,
                "agent_reasoning": agent_reasoning,
                "agent_answer": agent_final,
                "judge_reasoning": judge_reasoning,
                "judge_answer": judge_final,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            eval_pbar.set_postfix_str(f"acc={correct}/{total} ({(correct / total * 100):.1f}%)")

            qid = q.get("id", f"q_{total}")
            print(f"{qid}: {verdict}")

    return correct, total
