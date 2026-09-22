"""A_mem writer: an online two-track ingestion pipeline.

For each message (mem.add(message) / ingest(messages)):
  1. The message enters a per-session buffer.
  2. Track 1 is deterministic and LLM-free: every message is stored verbatim in per_speaker_episodic.
     It preserves the speaker, message time, and session turn without loss.
  3. At a session boundary or configured message interval, call the LLM.
     Track 2 reads current derived heads and the new message window, emits ADD/UPDATE/NOOP, and applies the actions.
  4. UPDATE is non-destructive: code creates the new entry, links the chain, and sets superseded_by.
  5. Register every speaker directly from the message stream without an LLM or inference.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from ..types import Message, DERIVED_LAYERS
from ..memory import SpeakerAwareMemory
from ..backends.llm import LLM
from ..config import WriterConfig
from ..prompts import WRITER_SYSTEM, WRITER_USER_TMPL


def _salvage_actions(raw: str) -> list:
    """Recover complete action objects from truncated JSON by balancing braces while skipping braces inside strings.

    A_mem emits an actions list. When generation is cut off at the token limit, only the final object is usually incomplete.
    Earlier closed objects remain valid and should not be discarded.
    Long messages can cut off a large batch at the token limit; the writer recovers the earlier complete actions.
    """
    if not raw:
        return []

    k = raw.find('"actions"')
    j = raw.find("[", k if k >= 0 else 0)
    body = raw[j + 1:] if j >= 0 else raw

    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        o = json.loads(body[start:i + 1], strict=False)
                        if isinstance(o, dict) and "action" in o:
                            out.append(o)
                    except Exception:
                        pass
                    start = -1
    return out


def _action_key(action: dict) -> tuple:
    """Action deduplication key; ADD/WRITE uses semantic fields, UPDATE uses target+new state."""
    kind = str(action.get("action", "")).upper()
    norm = lambda value: re.sub(r"[^\w]+", " ", str(value or "").lower()).strip()
    if kind in ("ADD", "WRITE"):
        return ("ADD", norm(action.get("owner")), norm(action.get("source")),
                norm(action.get("layer")), norm(action.get("utype")), norm(action.get("content")))
    if kind == "UPDATE":
        return ("UPDATE", str(action.get("entry_id", "")).strip(), norm(action.get("content")))
    return (kind,)


def _dedupe_actions(actions: list) -> tuple[list, int]:
    """Preserve first-seen order and remove identical ADD/UPDATE/NOOP actions."""
    seen, unique = set(), []
    for action in actions:
        if not isinstance(action, dict):
            unique.append(action); continue
        key = _action_key(action)
        if key not in seen:
            seen.add(key); unique.append(action)
    return unique, len(actions) - len(unique)


def _repeated_tail_length(actions: list) -> int:
    """Return the number of identical actions at the tail; fewer than three repetitions are not treated as a generation loop."""
    if not actions or not isinstance(actions[-1], dict):
        return 0
    tail_key = _action_key(actions[-1])
    run = 0
    for action in reversed(actions):
        if not isinstance(action, dict) or _action_key(action) != tail_key:
            break
        run += 1
    return run if run >= 3 else 0


class Writer:
    def __init__(self, memory: SpeakerAwareMemory, llm_provider, cfg: WriterConfig):

        self.mem, self._get_llm, self.cfg = memory, llm_provider, cfg
        self._buf: List[Message] = []
        self._cur_session: Optional[str] = None
        self._turn = 0
        self._last_flushed_session: Optional[str] = None
        self._sess_base = 0


    def add(self, msg: Message):
        if not msg.content:
            return
        if self._cur_session is None:
            self._cur_session = msg.session

        if msg.session != self._cur_session and self._buf:
            self._flush_session()
            self._cur_session = msg.session
        self._buf.append(msg)
        if self.cfg.derive_every == "message" and len(self._buf) >= self.cfg.derive_window:
            self._flush_session()

    def flush(self):
        if self._buf:
            self._flush_session()


    def ingest(self, messages: List[Message]):

        from itertools import groupby
        msgs = [m for m in messages if m.content]
        msgs_sorted = sorted(msgs, key=lambda m: (m.ts, m.msg_id))

        by_sess = {}
        for m in msgs_sorted:
            by_sess.setdefault(m.session, []).append(m)
        for sid in sorted(by_sess, key=lambda s: (by_sess[s][0].ts, s)):
            for m in by_sess[sid]:
                self.add(m)
            self.flush()


    def _flush_session(self):
        msgs, self._buf = self._buf, []
        turn = self._turn
        self._turn += 1
        speakers = sorted({m.speaker for m in msgs})
        sid = msgs[0].session if msgs else ""

        if sid != self._last_flushed_session:
            self._sess_base = 0
            self._last_flushed_session = sid

        self.mem.register_speakers(sid, speakers)

        ep_entries = []
        if self.cfg.keep_verbatim:
            for i, m in enumerate(msgs):
                e = self.mem.write(owner=m.speaker, source=m.speaker, layer="per_speaker_episodic",
                                   content=m.content, utype="utterance",
                                   confidence=self.cfg.verbatim_confidence, turn=turn, ts=m.ts,
                                   session=m.session, real_turn=self._sess_base + i)
                if e:
                    ep_entries.append(e)
        self._sess_base += len(msgs)

        sess_eps = []
        if self.cfg.provenance and ep_entries:
            for e in ep_entries:
                v = self.mem.index.get_vector(e.entry_id)
                if v is not None:
                    sess_eps.append((e.entry_id, v))

        if self.cfg.enabled:
            self._derive(sid, msgs, turn, speakers, sess_eps)

    def _link_provenance(self, entry, sess_eps):
        """Link a derived entry to the most similar source episode in the current session."""
        if not entry or not self.cfg.provenance or not sess_eps:
            return
        import numpy as np
        ev = self.mem.index.get_vector(entry.entry_id)
        if ev is None:
            return
        sims = sorted(((eid, float(np.dot(ev, v))) for eid, v in sess_eps), key=lambda x: -x[1])
        top = [eid for eid, s in sims[: self.cfg.provenance_m] if s >= self.cfg.provenance_thresh]
        if top:
            self.mem.add_from(entry.entry_id, top)

    def _derive(self, sid, msgs, turn, speakers, sess_eps=None, _repeat_retry=False):


        sess_eps = sess_eps or []
        convo = "\n".join(f"[{m.speaker} @ {m.ts}] {m.content}" for m in msgs)
        if not convo.strip():
            return



        state = self.mem.memory_state(max_chars=self.cfg.state_max_chars,
                                      query=convo, per_owner_k=self.cfg.state_per_owner_k)
        ts0_ = msgs[0].ts if msgs else ""
        max_actions = max(8, 2 * len(msgs))
        user = WRITER_USER_TMPL.format(speakers=", ".join(speakers), session=sid, ts=ts0_,
                                       state=state, convo=convo, max_actions=max_actions)
        def request():
            return self._get_llm().chat(WRITER_SYSTEM, user, json_mode=True, thinking=self.cfg.thinking,
                                        temperature=0.0, max_tokens=self.cfg.max_tokens, model=self.cfg.model)
        try:
            raw = request()
        except Exception as e:
            print(f"    [writer warn] {sid}: LLM call failed {str(e)[:70]}", flush=True)
            return
        recovered = False
        try:
            data = json.loads(raw or "{}", strict=False)
        except Exception:


            actions = _salvage_actions(raw or "")
            tail_run = _repeated_tail_length(actions)
            if not tail_run:
                print(f"    [writer reject] {sid}: incomplete JSON ({len(raw or '')} characters); batch skipped",
                      flush=True)
                return
            unique, repeated = _dedupe_actions(actions)
            print(f"    [writer recover] {sid}: truncated with repeated tail actions {tail_run} times;"
                  f"total={len(actions)}, unique={len(unique)}, duplicates={repeated}", flush=True)
            actions = unique
            recovered = True
            data = None
        if data is not None:
            actions = data.get("actions") if isinstance(data, dict) else None
            if not isinstance(actions, list):
                print(f"    [writer reject] {sid}: actions is not a list; batch skipped", flush=True)
                return
            unique, repeated = _dedupe_actions(actions)
        else:
            unique, repeated = actions, 0

        oversized = len(actions) > max_actions
        if (repeated or oversized) and not _repeat_retry and not recovered:
            print(f"    [writer retry] {sid}: actions={len(actions)}, duplicates={repeated}, "
                  f"oversized={oversized}; retrying once with the same prompt", flush=True)
            retry_raw = ""
            try:
                retry_raw = request(); retry_data = json.loads(retry_raw or "{}", strict=False)
                retry_actions = retry_data.get("actions") if isinstance(retry_data, dict) else None
                if not isinstance(retry_actions, list):
                    print(f"    [writer reject] {sid}: retry actions are not a list; batch skipped", flush=True)
                    return
            except Exception:
                retry_actions = _salvage_actions(retry_raw)
                retry_tail = _repeated_tail_length(retry_actions)
                if not retry_tail:
                    print(f"    [writer reject] {sid}: retry still returned incomplete JSON; batch skipped", flush=True)
                    return
                print(f"    [writer recover] {sid}: retrytruncated with repeated tail actions {retry_tail} times;"
                      f"total={len(retry_actions)}", flush=True)
            actions, unique = retry_actions, _dedupe_actions(retry_actions)[0]
            repeated = len(retry_actions) - len(unique)
            oversized = len(retry_actions) > max_actions
        if repeated or oversized:
            print(f"    [writer dedupe] {sid}: executing {len(unique)}/{len(actions)} unique actions "
                  f"(duplicates={repeated}, oversized={oversized})", flush=True)
            actions = unique
        ts0 = msgs[0].ts if msgs else ""
        for a in actions:
            try:
                act = str(a.get("action", "")).upper()
                if act in ("ADD", "WRITE"):
                    layer = str(a.get("layer", "")).strip()

                    if layer not in DERIVED_LAYERS:
                        print(f"    [writer warn] {sid}: invalid ADD layer={layer!r} ; entry discarded",
                              flush=True)
                        continue
                    e = self.mem.write(owner=str(a.get("owner", "")).strip() or "GROUP",
                                       source=str(a.get("source", "")).strip(),
                                       layer=layer,
                                       content=str(a.get("content", "")).strip(),
                                       utype=str(a.get("utype", "fact")).strip(), turn=turn, ts=ts0)
                    self._link_provenance(e, sess_eps)
                elif act == "UPDATE":

                    eid = str(a.get("entry_id", "")).strip()
                    e = self.mem.update(eid, str(a.get("content", "")).strip(), turn=turn, ts=ts0)
                    if e is None:


                        print(f"    [writer warn] {sid}: invalid UPDATE entry_id={eid!r}; entry discarded",
                              flush=True)
                    self._link_provenance(e, sess_eps)
            except Exception:
                continue
