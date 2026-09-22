
"""Writer rollout that reuses the production prompt, memory state, and non-destructive UPDATE semantics."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from speakermem.backends.embedder import Embedder, SentenceTransformerEmbedder
from speakermem.backends.store import MemoryStore, SqliteStore
from speakermem.backends.vector_index import NumpyIndex
from speakermem.memory import SpeakerAwareMemory
from speakermem.prompts import WRITER_SYSTEM, WRITER_USER_TMPL
from speakermem.types import (DERIVED_LAYERS, EPISODIC_LAYER, GROUP_LAYERS,
                              GROUP_OWNER, MemoryEntry, Message)

ALLOWED_ACTIONS = ("ADD", "UPDATE", "NOOP")
ALLOWED_UTYPES = ("fact", "stance", "observation", "decision", "relation")

_DEFAULT_EMBEDDER = None


def _default_embedder():
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    return _DEFAULT_EMBEDDER


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", (text or "").lower()).strip()
def _action_schema_valid(action) -> bool:
    if not isinstance(action, dict): return False
    kind = str(action.get("action", "")).upper()
    if kind == "NOOP": return set(action) == {"action"}
    if kind == "ADD":
        content = str(action.get("content", "")).strip()
        owner, source = str(action.get("owner", "")).strip(), str(action.get("source", "")).strip()
        layer, utype = str(action.get("layer", "")).strip(), str(action.get("utype", "")).strip()
        if not owner or not source or not content or layer not in DERIVED_LAYERS or utype not in ALLOWED_UTYPES: return False
        if layer in GROUP_LAYERS and owner != GROUP_OWNER: return False
        if layer in ("per_speaker_core", "per_speaker_profile") and owner == GROUP_OWNER: return False
        return True
    if kind == "UPDATE":
        content = str(action.get("content", "")).strip()
        return bool(str(action.get("entry_id", "")).strip() and content) and not (set(action) - {"action", "entry_id", "content"})
    return False


def parse_actions(raw: str) -> Dict[str, Any]:
    """Atomic parsing: incomplete JSON is rejected and no action is executed, matching the production writer."""
    text = raw or ""
    try:
        data = json.loads(text, strict=False)
    except Exception:
        return {"actions": [], "json_valid": False, "raw": text}
    actions = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(actions, list):
        return {"actions": [], "json_valid": False, "raw": text}
    return {"actions": actions, "json_valid": True, "raw": text}


class StableHashEmbedder(Embedder):
    """Used for offline tests; does not download a model."""
    def __init__(self, dim: int = 64): self.dim = dim
    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", (text or "").lower()):
                digest = hashlib.sha256(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1 if digest[4] & 1 else -1
            norm = float(np.linalg.norm(out[row]))
            if norm: out[row] /= norm
        return out


@dataclass
class ActionReport:
    json_valid: bool = True
    attempted: int = 0
    applied: int = 0
    noop: int = 0
    errors: List[str] = field(default_factory=list)
    created_ids: List[str] = field(default_factory=list)
    updated_from: List[str] = field(default_factory=list)
    schema_valid_actions: int = 0
    @property
    def valid(self): return self.json_valid and not self.errors and self.attempted > 0
    def to_dict(self):
        return {"json_valid": self.json_valid, "attempted": self.attempted,
                "applied": self.applied, "noop": self.noop, "errors": self.errors,
                "created_ids": self.created_ids, "updated_from": self.updated_from,
                "schema_valid_actions": self.schema_valid_actions,
                "valid": self.valid}


def _messages(turns: Sequence[Any], sid="", ts="") -> List[Message]:
    out = []
    for i, turn in enumerate(turns):
        if isinstance(turn, Message): msg = turn
        else:
            data = dict(turn)
            data.setdefault("session", data.get("session_id") or sid)
            data.setdefault("ts", data.get("timestamp") or ts)
            data.setdefault("msg_id", str(data.get("turn", i)))
            msg = Message.from_dict(data)
        if msg.content: out.append(msg)
    return out


class V4RolloutEnv:
    def __init__(self, *, embedder: Optional[Embedder] = None,
                 store: Optional[MemoryStore] = None, state_max_chars=60000,
                 state_per_owner_k=40, keep_verbatim=True, provenance=True,
                 provenance_m=3, provenance_thresh=0.25):
        self.embedder = embedder or _default_embedder()
        self.store = store or SqliteStore(":memory:")
        self.memory = SpeakerAwareMemory(self.store, NumpyIndex(), self.embedder)
        self.state_max_chars, self.state_per_owner_k = state_max_chars, state_per_owner_k
        self.keep_verbatim, self.provenance = keep_verbatim, provenance
        self.provenance_m, self.provenance_thresh = provenance_m, provenance_thresh
        self.turn, self._offsets = 0, {}

    def close(self): self.store.close()

    def observe(self, turns, *, session_id="", ts=""):
        msgs = _messages(turns, session_id, ts)
        sid = session_id or (msgs[0].session if msgs else "")
        ts0 = ts or (msgs[0].ts if msgs else "")
        speakers = sorted({m.speaker for m in msgs})
        convo = "\n".join(f"[{m.speaker} @ {m.ts}] {m.content}" for m in msgs)
        state = self.memory.memory_state(max_chars=self.state_max_chars, query=convo,
                                         per_owner_k=self.state_per_owner_k)
        max_actions = max(8, 2 * len(msgs))
        user = WRITER_USER_TMPL.format(speakers=", ".join(speakers), session=sid,
                                       ts=ts0, state=state, convo=convo,
                                       max_actions=max_actions)
        return {"messages": [{"role": "system", "content": WRITER_SYSTEM},
                             {"role": "user", "content": user}],
                "state": state, "session_id": sid, "ts": ts0}

    def validate_actions(self, actions, *, json_valid=True):
        r = ActionReport(json_valid=json_valid)
        if not isinstance(actions, list): r.errors.append("actions must be a list"); return r
        r.attempted = len(actions)
        r.schema_valid_actions = sum(1 for action in actions if _action_schema_valid(action))
        if not actions: r.errors.append("actions requires explicit NOOP when nothing changes"); return r
        kinds = [str(a.get("action", "")).upper() for a in actions if isinstance(a, dict)]
        if "NOOP" in kinds and len(actions) != 1: r.errors.append("NOOP must be the only action")
        seen_updates, seen_adds = set(), set()
        for i, a in enumerate(actions):
            p = f"actions[{i}]"
            if not isinstance(a, dict): r.errors.append(f"{p} must be an object"); continue
            kind = str(a.get("action", "")).upper()
            if kind not in ALLOWED_ACTIONS: r.errors.append(f"{p}.action={kind!r} is not a valid action"); continue
            if kind == "NOOP":
                if set(a) != {"action"}: r.errors.append(f"{p} NOOP has extra fields")
                continue
            content = str(a.get("content", "")).strip()
            if not content: r.errors.append(f"{p}.content is empty")
            if kind == "ADD":
                owner, source = str(a.get("owner", "")).strip(), str(a.get("source", "")).strip()
                layer, utype = str(a.get("layer", "")).strip(), str(a.get("utype", "")).strip()
                if not owner or not source: r.errors.append(f"{p} ADD requires owner and source")
                if layer not in DERIVED_LAYERS: r.errors.append(f"{p}.layer={layer!r} invalid")
                if utype not in ALLOWED_UTYPES: r.errors.append(f"{p}.utype={utype!r} invalid")
                if layer in GROUP_LAYERS and owner != GROUP_OWNER: r.errors.append(f"{p} group layer requires GROUP")
                if layer in ("per_speaker_core", "per_speaker_profile") and owner == GROUP_OWNER:
                    r.errors.append(f"{p} per-speaker layer cannot use GROUP")
                key = (owner, source, layer, _norm(content))
                if key in seen_adds: r.errors.append(f"{p} duplicate ADD")
                seen_adds.add(key)
            else:
                eid = str(a.get("entry_id", "")).strip(); old = self.memory.get(eid)
                if old is None: r.errors.append(f"{p}.entry_id={eid!r} is not in CURRENT MEMORY")
                elif old.superseded_by: r.errors.append(f"{p}.entry_id={eid!r} is not a head")
                elif _norm(old.content) == _norm(content): r.errors.append(f"{p} empty UPDATE; use NOOP")
                if eid in seen_updates: r.errors.append(f"{p} updates one entry twice")
                seen_updates.add(eid)
                extra = set(a) - {"action", "entry_id", "content"}
                if extra: r.errors.append(f"{p} UPDATE extra fields: {sorted(extra)}")
        return r

    def _write_raw(self, msgs):
        if not msgs: return []
        sid = msgs[0].session; self.memory.register_speakers(sid, {m.speaker for m in msgs})
        base, out = self._offsets.get(sid, 0), []
        if self.keep_verbatim:
            for i, m in enumerate(msgs):
                e = self.memory.write(m.speaker, m.speaker, EPISODIC_LAYER, m.content,
                                      utype="utterance", confidence=0.6, turn=self.turn,
                                      ts=m.ts, session=sid, real_turn=base + i)
                if e: out.append(e)
        self._offsets[sid] = base + len(msgs)
        return out

    def _provenance(self, entry, raws):
        if not entry or not self.provenance or not raws: return
        ev = self.memory.index.get_vector(entry.entry_id)
        scores = [] if ev is None else [(e.entry_id, float(np.dot(ev, self.memory.index.get_vector(e.entry_id)))) for e in raws]
        ids = [eid for eid, score in sorted(scores, key=lambda x: -x[1])[:self.provenance_m]
               if score >= self.provenance_thresh]
        if ids: self.memory.add_from(entry.entry_id, ids)

    def apply_actions(self, actions, *, ts="", raw_entries=(), json_valid=True):
        r = self.validate_actions(actions, json_valid=json_valid)
        if not isinstance(actions, list) or not r.valid: return r
        for a in actions:
            if not isinstance(a, dict): continue
            kind = str(a.get("action", "")).upper()
            if kind == "NOOP": r.noop += 1
            elif kind == "ADD":
                layer, content = str(a.get("layer", "")).strip(), str(a.get("content", "")).strip()
                if layer not in DERIVED_LAYERS or not content: continue
                e = self.memory.write(str(a.get("owner", "")).strip() or GROUP_OWNER,
                                      str(a.get("source", "")).strip(), layer, content,
                                      utype=str(a.get("utype", "fact")).strip(), turn=self.turn, ts=ts)
                if e: r.applied += 1; r.created_ids.append(e.entry_id); self._provenance(e, raw_entries)
            elif kind == "UPDATE":
                eid, content = str(a.get("entry_id", "")).strip(), str(a.get("content", "")).strip()
                old = self.memory.get(eid)
                if not old or old.superseded_by or not content: continue
                e = self.memory.update(eid, content, turn=self.turn, ts=ts)
                if e and e.entry_id != eid:
                    r.applied += 1; r.created_ids.append(e.entry_id); r.updated_from.append(eid); self._provenance(e, raw_entries)
        return r

    def step(self, turns, actions, *, session_id="", ts="", json_valid=True):
        obs = self.observe(turns, session_id=session_id, ts=ts)
        msgs = _messages(turns, obs["session_id"], obs["ts"])
        pre = self.snapshot(derived_only=True); raws = self._write_raw(msgs)
        report = self.apply_actions(actions, ts=obs["ts"], raw_entries=raws, json_valid=json_valid)
        result = {"turn": self.turn, "session_id": obs["session_id"], "ts": obs["ts"],
                  "messages": [m.__dict__ for m in msgs], "prompt": obs["messages"],
                  "pre_state_text": obs["state"], "actions": actions if isinstance(actions, list) else [],
                  "target_text": json.dumps({"actions": actions}, ensure_ascii=False, separators=(",", ":")),
                  "report": report.to_dict(), "pre_state": pre,
                  "post_state": self.snapshot(derived_only=True)}
        self.turn += 1
        return result

    def snapshot(self, *, derived_only=False):
        entries = self.memory.active_entries()
        if derived_only: entries = [e for e in entries if e.layer in DERIVED_LAYERS]
        entries = sorted(entries, key=lambda e: (e.turn_created, e.entry_id))
        heads = [e for e in entries if e.layer in DERIVED_LAYERS and not e.superseded_by]
        return {"entries": [e.to_dict() for e in entries], "heads": [e.to_dict() for e in heads],
                "update_edges": [{"old": e.entry_id, "new": e.superseded_by} for e in entries if e.superseded_by],
                "roster": self.memory.roster()}

    def clone(self):
        env = V4RolloutEnv(embedder=self.embedder, state_max_chars=self.state_max_chars,
                           state_per_owner_k=self.state_per_owner_k, keep_verbatim=self.keep_verbatim,
                           provenance=self.provenance, provenance_m=self.provenance_m,
                           provenance_thresh=self.provenance_thresh)
        env.memory.add([MemoryEntry.from_dict(e.to_dict()) for e in self.memory.active_entries()])
        env.memory.register_speakers("", self.memory.roster()); env.turn = self.turn
        env._offsets = dict(self._offsets)
        return env


def prompt_version():
    return hashlib.sha256((WRITER_SYSTEM + "\n---USER---\n" + WRITER_USER_TMPL).encode()).hexdigest()[:16]
