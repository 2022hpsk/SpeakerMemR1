"""Core data types: messages, memory entries, and memory layers.

SpeakerMem is a **speaker-grounded memory framework for multi-party conversations**: it treats who said what, who observed whom, and
how each stance changes over time as first-class information, indexed by speaker with one raw layer and four derived layers.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional




PER_SPEAKER_LAYERS = ("per_speaker_core", "per_speaker_episodic", "per_speaker_profile")
GROUP_LAYERS = ("group_interaction", "group_insight")
LAYERS = PER_SPEAKER_LAYERS + GROUP_LAYERS
EPISODIC_LAYER = "per_speaker_episodic"
DERIVED_LAYERS = ("per_speaker_core", "per_speaker_profile",
                  "group_interaction", "group_insight")

GROUP_OWNER = "GROUP"


@dataclass
class Message:
    """One multi-party conversation message, the minimum framework input."""
    speaker: str
    content: str
    session: str = ""
    ts: str = ""
    msg_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Message":
        """Accept common field aliases such as speaker/author/user and session/channel/_channel."""
        g = lambda *ks: next((d[k] for k in ks if d.get(k) not in (None, "")), "")
        return Message(
            speaker=str(g("speaker", "author", "user", "name") or "?"),
            content=str(d.get("content") or d.get("text") or "").strip(),
            session=str(g("session", "channel", "_channel", "session_id")),
            ts=str(g("ts", "timestamp", "time", "created_at")),
            msg_id=str(g("msg_id", "msg_node", "id")),
            meta={k: d[k] for k in ("topic",) if k in d},
        )


@dataclass
class MemoryEntry:
    """One memory entry. Separating who it is about (owner) from who said or observed it (source) supports cross-speaker references and observations.

    Field semantics:
      entry_id     8-character hexadecimal identifier used by UPDATE/PROMOTE/links.
      content      Memory text. An utterance stores the raw message verbatim; derived types store extracted facts.
      owner        Subject, or who the memory is about; per-speaker buckets use s_owner and the group layer uses GROUP.
      source       Source speaker and provenance; for self-reports, source equals owner.
      layer        One of five layers.
      turn_created Session turn index, used for before/after temporal queries.
      ts           Timestamp, used for calendar-date temporal queries.
      utype        fact/stance/observation/decision/relation/utterance.
      confidence   Confidence from 0 to 1, used for retrieval weighting and increased by UPDATE/PROMOTE.
      from_ids     **Source provenance**: entry IDs of raw episodes used to derive a core/profile/group entry.
                   Separate from links: from_ids records which raw episodes support the derived entry.
      links        Association and evolution edges; UPDATE links a new entry to the old entry and pairs with superseded_by to form an EVOLVED_FROM chain.
      superseded_by Non-destructive UPDATE: record the new ID when an entry is superseded; the old entry remains available for before/after queries.
    """
    content: str
    owner: str
    source: str
    layer: str
    turn_created: int = 0
    session: str = ""
    turn: int = 0
    ts: str = ""
    utype: str = "fact"
    confidence: float = 0.7
    from_ids: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    superseded_by: str = ""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


    def embed_text(self) -> str:
        """Text sent to the vector encoder, including a speaker anchor and timestamp to improve speaker- and time-aware retrieval."""
        tag = f"[{self.owner}]"
        if self.source and self.source != self.owner:
            tag += f"(by {self.source})"
        if self.ts:
            tag += f"@{self.ts[:10]}"
        return f"{tag} {self.content}"

    def render(self, i: Optional[int] = None) -> str:
        """Readable format for the answer agent, including ownership, layer, time, and superseded-state metadata."""
        meta = f"about={self.owner}"
        if self.source and self.source != self.owner:
            meta += f", said_by={self.source}"
        meta += f", {self.layer.replace('per_speaker_', '').replace('group_', 'group-')}"
        meta += f", t{self.turn_created}"
        if self.ts:
            meta += f" @{self.ts}"
        if self.superseded_by:
            meta += ", [earlier-state]"
        head = f"[{i}] " if i is not None else ""
        return f"{head}({meta}) {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryEntry":
        fields = {f for f in MemoryEntry.__dataclass_fields__}
        return MemoryEntry(**{k: v for k, v in d.items() if k in fields})
