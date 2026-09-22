"""SpeakerAwareMemory: a speaker-indexed memory container with ADD / UPDATE / NOOP actions.

Stores MemoryEntry objects in the persistent store and vector index while maintaining the in-memory _by_id view.
UPDATE is non-destructive: the old entry is retained, superseded_by is set, and the new entry links back to form a time chain.
The roster is collected from the message stream and is the sole source for questions about who never performs an action.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from .types import (MemoryEntry, LAYERS, GROUP_LAYERS, PER_SPEAKER_LAYERS, GROUP_OWNER,
                    EPISODIC_LAYER, DERIVED_LAYERS)
from .backends.embedder import Embedder
from .backends.vector_index import VectorIndex
from .backends.store import MemoryStore


def _norm_text(s: str) -> str:
    """Comparison normalization: remove case, punctuation, and redundant whitespace when checking whether UPDATE changes anything."""
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


class SpeakerAwareMemory:
    def __init__(self, store: MemoryStore, index: VectorIndex, embedder: Embedder):
        self.store, self.index, self.embedder = store, index, embedder
        self._by_id: Dict[str, MemoryEntry] = {}
        self._roster_cache: Dict[str, List[str]] = {}
        self._load()



        self.roster()


    def _load(self):
        entries = self.store.all_entries()
        if not entries:
            return
        self._by_id = {e.entry_id: e for e in entries}
        vecs = self.store.vectors()
        missing = [e for e in entries if e.entry_id not in vecs]
        if missing:
            mv = self.embedder.encode([e.embed_text() for e in missing])
            for e, v in zip(missing, mv):
                vecs[e.entry_id] = v
        for e in entries:
            self.index.upsert(e.entry_id, vecs[e.entry_id])


    def add(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        entries = [e for e in entries if e.content]
        if not entries:
            return []
        vecs = self.embedder.encode([e.embed_text() for e in entries])
        self.store.add_many(list(zip(entries, vecs)))
        for e, v in zip(entries, vecs):
            self._by_id[e.entry_id] = e
            self.index.upsert(e.entry_id, v)
        return entries

    def write(self, owner, source, layer, content, *, utype="fact", confidence=0.7,
              turn=0, ts="", links=None, session="", real_turn=0) -> Optional[MemoryEntry]:
        if layer not in LAYERS:
            layer = "per_speaker_episodic"






        if owner == GROUP_OWNER and layer not in GROUP_LAYERS:
            layer = "group_interaction"
        if layer in GROUP_LAYERS:
            owner = GROUP_OWNER
        e = MemoryEntry(content=content, owner=owner or GROUP_OWNER,
                        source=source or owner or GROUP_OWNER, layer=layer, turn_created=turn,
                        session=session, turn=real_turn,
                        ts=ts, utype=utype, confidence=confidence, links=list(links or []))
        return (self.add([e]) or [None])[0]

    def update(self, entry_id, new_content, *, source="", turn=0, ts="", utype=None):
        """Non-destructive UPDATE: retain the old entry, set superseded_by, and link the new entry back to it.

        No-op UPDATE safeguard: if normalized old and new content match, do not create a chain and return the old entry.
          In a large evaluation batch, many adjacent UPDATE pairs
          had identical content because the LLM treated seeing a related entry as requiring an update.
          Long duplicate chains add no information and, with tense=full,
          fill the slice with repeated history. The prompt asks for NOOP when content is unchanged, and this code enforces that rule deterministically."""
        old = self._by_id.get(entry_id)
        if not old or not new_content:
            return None
        if _norm_text(old.content) == _norm_text(new_content):
            return old
        new = self.write(old.owner, source or old.source, old.layer, new_content,
                         utype=utype or old.utype, confidence=min(1.0, old.confidence + 0.1),
                         turn=turn, ts=ts, links=[old.entry_id])
        if new:
            old.superseded_by = new.entry_id
            self.store.update(old)
        return new

    def add_links(self, entry_id, target_ids):
        """Append deduplicated evolution or association links to an entry and persist them."""
        e = self._by_id.get(entry_id)
        if not e:
            return
        new = [t for t in target_ids if t and t != entry_id and t not in e.links]
        if new:
            e.links.extend(new)
            self.store.update(e)

    def add_from(self, entry_id, source_ep_ids):
        """Append deduplicated provenance source-episode IDs to a derived entry and persist them."""
        e = self._by_id.get(entry_id)
        if not e:
            return
        new = [t for t in source_ep_ids if t and t != entry_id and t not in e.from_ids]
        if new:
            e.from_ids.extend(new)
            self.store.update(e)


    def get(self, entry_id) -> Optional[MemoryEntry]:
        return self._by_id.get(entry_id)

    def active_entries(self) -> List[MemoryEntry]:
        """All entries; this method name is retained for caller compatibility."""
        return list(self._by_id.values())

    def by_owner(self, owner) -> List[MemoryEntry]:
        return [e for e in self._by_id.values() if e.owner == owner]

    def by_layer(self, layer) -> List[MemoryEntry]:
        return [e for e in self._by_id.values() if e.layer == layer]

    def speakers(self) -> List[str]:
        return sorted({e.owner for e in self._by_id.values() if e.owner != GROUP_OWNER})

    def episodics_in_window(self, session: str, turn_lo: int, turn_hi: int) -> List[MemoryEntry]:
        """Return utterances in a session whose real turns fall in [lo, hi], sorted by turn.
        The retriever uses this to disambiguate and complete a hit by expanding around its source turn."""
        return sorted(
            (e for e in self._by_id.values()
             if e.utype == "utterance"
             and e.session == session and turn_lo <= e.turn <= turn_hi),
            key=lambda e: e.turn)


    def register_speakers(self, channel: str, speakers) -> None:
        """Register speakers observed in the message stream without an LLM or inference."""
        self.store.add_roster(channel or "", speakers)
        self._roster_cache.clear()

    def roster(self, channel: Optional[str] = None) -> List[str]:
        """Return all group members. If the roster is empty, infer members from existing entry owners.

        Cache the roster in memory: it is fixed during writes and read by multithreaded QA, so querying SQLite on every read
          can trigger SQLite thread-affinity errors.
          The cache avoids database access during reads and is thread-safe and faster."""
        key = channel or ""
        if key not in self._roster_cache:
            self._roster_cache[key] = self.store.roster(channel) or self.speakers()
        return self._roster_cache[key]


    def head(self, entry_id: str) -> Optional[MemoryEntry]:
        """Follow superseded_by to the chain tail, which is the current value for the slot."""
        e = self._by_id.get(entry_id)
        seen = set()
        while e is not None and e.superseded_by and e.superseded_by not in seen:
            seen.add(e.entry_id)
            nxt = self._by_id.get(e.superseded_by)
            if nxt is None:
                break
            e = nxt
        return e

    def chain(self, entry_id: str) -> List[MemoryEntry]:
        """Return the complete time chain for an entry, ordered from old to new."""
        e = self._by_id.get(entry_id)
        if e is None:
            return []
        back, seen = [], set()
        cur = e
        while cur is not None and cur.entry_id not in seen:
            seen.add(cur.entry_id)
            back.append(cur)
            prev = next((self._by_id[l] for l in cur.links if l in self._by_id), None)
            cur = prev
        out = list(reversed(back))
        cur = e
        while cur is not None and cur.superseded_by and cur.superseded_by not in seen:
            seen.add(cur.superseded_by)
            cur = self._by_id.get(cur.superseded_by)
            if cur is not None:
                out.append(cur)
        return out


    def derived_of(self, owner: str, layers=None) -> List[MemoryEntry]:
        """Return derived memory for one row (a person or GROUP), optionally filtered by layer."""
        allowed = set(layers or DERIVED_LAYERS)
        return [e for e in self._by_id.values()
                if e.owner == owner and e.layer in allowed]

    def episodic_of(self, owner: str) -> List[MemoryEntry]:
        """Return all raw entries for one row, used by the second level of S2 retrieval."""
        return [e for e in self._by_id.values()
                if e.owner == owner and e.layer == EPISODIC_LAYER]

    def spoken_by(self, speaker: str, layers=None) -> List[MemoryEntry]:
        """Derived memories spoken by this person on behalf of the group: owner=GROUP and source=this person.

        Why it is needed: S2 partitions by owner, while group decisions have owner=GROUP, so the proposer
        row may be empty and questions about who proposed X would rely on S1.
        The provenance is already present in source; this method makes it searchable.
        This does not move the entry to the personal row. The entry remains a single GROUP entry, but it
        is also visible in the proposer row without creating a duplicate chain."""
        allowed = set(layers or GROUP_LAYERS)
        return [e for e in self._by_id.values()
                if e.layer in allowed and e.source == speaker and e.owner != speaker]

    def row_search(self, owner: str, qv, k: int = 3, derived: bool = True,
                   min_cos: float = 0.0, layers=None):
        """Search within the specified row by vector similarity and filter irrelevant entries with min_cos.

        This is the key difference from S1: S1 competes across the full store, while S2 fixes the row first so every row can contribute.
        min_cos is essential: without it, every row would return irrelevant memories for a query such as a weekend check-in,
          making every cell nonempty and preventing the raw-evidence fallback from ever running."""
        pool = self.derived_of(owner, layers=layers) if derived else self.episodic_of(owner)
        if not pool or qv is None:
            return []
        return self._search_pool(pool, qv, k, min_cos)

    def row_search_as_source(self, speaker: str, qv, k: int = 1, min_cos: float = 0.0, layers=None):
        """Search memories spoken by this person on behalf of the group; this is an extra quota independent of row_search,
        so it does not displace the row quota for personal memories."""
        return self._search_pool(self.spoken_by(speaker, layers=layers), qv, k, min_cos)

    def _search_pool(self, pool: List[MemoryEntry], qv, k: int, min_cos: float):
        if not pool or qv is None:
            return []
        allow = {e.entry_id for e in pool}
        hits = self.index.search(qv, max(k, 1), allow=allow)
        return [(self._by_id[i], c) for i, c in hits if i in self._by_id and c >= min_cos]


    def memory_state(self, max_chars: int = 12000, query: str = "",
                     per_owner_k: int = 0) -> str:
        """Render the current memory state for A_mem grouped by owner, showing only each chain head.
        Showing only heads makes each slot appear exactly once and maps each entry_id to one slot, so
        the LLM only chooses whether new information belongs to an existing slot or is new.

        When per_owner_k>0 and a query is provided, retrieve the most relevant top-k entries separately for each row.
          This is necessary because feeding the full store works only for small stores. SocialMem networks contain about 40 derived entries and fit comfortably;
          a GroupMem domain may contain tens of thousands of messages and derived entries, while a 12,000-character state fits only
          about 100 entries. Dropping an entire speaker when the state is full is worse because later speakers remain invisible,
          so their memories can never be updated and are repeatedly added.
          Per-row top-k ensures that every member with memory appears, the size is rows times k,
          independent of store size, and structurally matches S2 retrieval.
        """
        heads = [e for e in self._by_id.values()
                 if e.layer in DERIVED_LAYERS and not e.superseded_by]
        if not heads:
            return "(no derived memory yet)"
        by_owner: Dict[str, List[MemoryEntry]] = {}
        for e in heads:
            by_owner.setdefault(e.owner, []).append(e)

        qv = None
        if per_owner_k > 0 and query and self.embedder is not None:
            try:
                qv = self.embedder.encode_one(query)
            except Exception:
                qv = None





        per_owner_chars = max(240, max_chars // max(len(by_owner), 1))
        lines = []
        for owner in sorted(by_owner, key=lambda o: (o == GROUP_OWNER, o)):
            grp = by_owner[owner]
            if qv is not None and len(grp) > 1:

                hits = self.index.search(qv, len(grp), allow={e.entry_id for e in grp})
                ranked = [self._by_id[i] for i, _ in hits if i in self._by_id]
                grp = ranked or grp
            else:
                grp = sorted(grp, key=lambda e: e.turn_created, reverse=True)
            if per_owner_k > 0:
                grp = grp[:per_owner_k]
            blk, used = [f"  [{owner}]"], 0
            for e in sorted(grp, key=lambda x: x.turn_created):
                src = f"(by {e.source})" if e.source and e.source != e.owner else ""
                lyr = e.layer.replace("per_speaker_", "").replace("group_", "g-")
                ln = f"    {e.entry_id} [{lyr}|{e.owner}{src}@{e.ts[:10]}] {e.content}"
                if used + len(ln) > per_owner_chars and len(blk) > 1:
                    blk.append(f"    ... ({len(grp) - len(blk) + 1} more of {owner} omitted)")
                    break
                blk.append(ln); used += len(ln)
            lines.append("\n".join(blk))
        return "\n".join(lines)

    def __len__(self):
        return len(self._by_id)

    def _render_state(self, groups) -> str:
        """Render owner-to-entry groups as text with entry IDs."""
        lines = []
        for owner, es in groups.items():
            if not es:
                continue
            lines.append(f"  [{owner}]")
            for e in es:
                src = f"(by {e.source})" if e.source != e.owner else ""
                lines.append(f"    {e.entry_id} <{e.layer.replace('per_speaker_','').replace('group_','g-')}>"
                             f"{src} {e.content[:90]}")
        return "\n".join(lines) if lines else "(no derived memory yet)"

    def state_summary(self, max_entries: int = 60, derived_only: bool = True,
                      active_speakers=None, per_speaker_k: int = 8, group_k: int = 10,
                      query_vec=None, update_candidates_n: int = 24) -> str:
        """Render a current-memory summary with entry IDs so A_mem can detect changed entries and emit UPDATE actions,
        or repeated observations and promote them. Exclude verbatim utterances so they do not overwhelm derived memory.

        Selection strategies:
        - active_speakers=None: take the last max_entries entries by global recency. This is the naive dyadic behavior.
        - when active_speakers is provided: for each present speaker, take entries from that owner bucket
          top-k by embedding similarity when query_vec is available, otherwise by recency, then add the top group_k GROUP entries.
          This ensures that every present speaker brings existing memory and that quiet or older speakers are not crowded out."""
        active = [e for e in self._by_id.values()
                  if not e.superseded_by and (not derived_only or e.utype != "utterance")]
        if not active:
            return "(no derived memory yet)"

        if not active_speakers:
            from collections import OrderedDict
            groups: "OrderedDict[str, list]" = OrderedDict()
            for e in active[-max_entries:]:
                groups.setdefault(e.owner, []).append(e)
            return self._render_state(groups)


        def _rank(entries):
            if query_vec is not None:
                def sim(e):
                    v = self.index.get_vector(e.entry_id)
                    return float(np.dot(query_vec, v)) if v is not None else -1.0
                return sorted(entries, key=sim, reverse=True)
            return list(reversed(entries))

        by_owner: Dict[str, list] = {}
        for e in active:
            by_owner.setdefault(e.owner, []).append(e)
        from collections import OrderedDict
        groups = OrderedDict()
        for sp in active_speakers:
            groups[sp] = _rank(by_owner.get(sp, []))[:per_speaker_k]
        if GROUP_OWNER in by_owner:
            groups[GROUP_OWNER] = _rank(by_owner[GROUP_OWNER])[:group_k]
        base = self._render_state(groups)


        if query_vec is not None and update_candidates_n:
            allow = {e.entry_id for e in active}
            by_id = {e.entry_id: e for e in active}
            hits = self.index.search(query_vec, update_candidates_n, allow=allow)
            cand = [by_id[eid] for eid, _ in hits if eid in by_id]
            if cand:
                lines = ["  [MOST RELEVANT existing memory to THIS session — prefer UPDATE on these "
                         "entry_ids if a person's state changed, instead of writing a duplicate]"]
                for e in cand:
                    src = f"(by {e.source})" if e.source != e.owner else ""
                    lyr = e.layer.replace("per_speaker_", "").replace("group_", "g-")
                    lines.append(f"    {e.entry_id} <{lyr}>[{e.owner}]{src} {e.content[:90]}")
                return "\n".join(lines) + "\n" + base
        return base
