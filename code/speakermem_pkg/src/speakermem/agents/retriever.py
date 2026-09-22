"""A_ret retriever with two parallel paths.

    S1 similarity path -> searches only the raw track and provides precise evidence for what was said.
    S2 structure path -> traverses only the derived track and provides evidence about who, what role, and how a state changed.

The paths run in parallel with independent budgets; S2 does not prune S1.
Otherwise precision pruning can remove coverage and silently skip quiet members.

S1: retrieve raw passages with vectors, select ranked or expanded evidence, and expand the local context window.
    A2b ask can issue one follow-up query when the final evidence is incomplete.
S2: project issue/rows/tense, then retrieve top-k2 for the issue within each row.
    Fold update chains by tense and fall back to raw row evidence when derived memory is empty.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import MemoryEntry, GROUP_OWNER, EPISODIC_LAYER, DERIVED_LAYERS
from ..memory import SpeakerAwareMemory
from ..backends.embedder import Embedder
from ..config import RetrieverConfig
from ..prompts import (SELECT_SYSTEM, SELECT_USER_TMPL, ASK_SYSTEM, ASK_USER_TMPL,
                       PROJECT_SYSTEM, PROJECT_USER_TMPL)


class Projection:
    """S2 projection: which rows, issue, and temporal state to retrieve."""

    def __init__(self, issue: str, rows: List[str], tense: str = "head"):
        self.issue, self.rows, self.tense = issue, rows, tense

    def __repr__(self):
        return f"Projection(issue={self.issue!r}, rows={self.rows}, tense={self.tense!r})"


class Cell:
    """One cell of the perspective tensor: memory for one row and one issue (a time chain or an empty cell)."""

    def __init__(self, owner: str, chain: List[MemoryEntry], from_raw: bool = False,
                 as_source: bool = False):
        self.owner, self.chain, self.from_raw = owner, chain, from_raw

        self.as_source = as_source

    @property
    def empty(self) -> bool:
        return not self.chain

    def render(self, tense: str = "head") -> str:
        if self.empty:
            return f"  {self.owner}  (none)"
        cur = self.chain[-1]
        mark = " [from raw]" if self.from_raw else ""

        if self.owner == GROUP_OWNER:
            mark += " [group-level: a decision/norm/shared fact, NOT any individual's stance]"
        elif self.as_source:
            mark += (f" [{self.owner} SAID this on behalf of the group -"
                     f" it is {self.owner}'s doing, but it is a group-level item,"
                     f" not {self.owner}'s personal stance]")
        out = [f"  {self.owner}  *current: {cur.content}{mark}"]
        if tense == "full" and len(self.chain) > 1:
            for e in self.chain[:-1]:
                out.append(f"              - history: {e.content} @{e.ts[:10]}")
        return "\n".join(out)


class Retriever:
    def __init__(self, memory: SpeakerAwareMemory, embedder: Embedder, cfg: RetrieverConfig,
                 llm_provider=None):
        self.mem, self.embedder, self.cfg = memory, embedder, cfg
        self._get_llm = llm_provider




    def s1_raw(self, query: str, k: Optional[int] = None) -> List[MemoryEntry]:
        c = self.cfg
        k = k or c.s1_k
        raw = [e for e in self.mem.active_entries() if e.layer == EPISODIC_LAYER]
        if not raw:
            return []
        by_id = {e.entry_id: e for e in raw}
        qv = self.embedder.encode_one(query)
        hits = self.mem.index.search(qv, max(c.s1_recall_n, k), allow=set(by_id))
        pool = [eid for eid, _ in hits]
        if not pool:
            return []
        if not (c.llm_select and self._get_llm is not None):
            return [by_id[i] for i in pool[:k]]











        max_rounds = c.expand_max_rounds
        in_pool = set(pool)
        ranked, expand = self._select(query, pool, by_id, allow_expand=(max_rounds > 0))
        pulls = 0
        while expand and pulls < max_rounds:
            added = False
            for e in self._expand_context([by_id[i] for i in expand if i in by_id]):
                if e.entry_id not in in_pool:
                    by_id.setdefault(e.entry_id, e)
                    in_pool.add(e.entry_id)
                    pool.append(e.entry_id)
                    added = True
            pulls += 1
            if not added:
                break
            ranked, expand = self._select(query, pool, by_id, allow_expand=(pulls < max_rounds))

        def _cut(rk):




            seen = set(rk)
            return [by_id[i] for i in (list(rk) + [i for i in pool if i not in seen])[:k]
                    if i in by_id]

        final = _cut(ranked)

        for _ in range(c.ask_max_rounds):
            aq = self._ask(query, final)
            if not aq:
                break
            grew = False
            for eid, _s in self.mem.index.search(self.embedder.encode_one(aq),
                                                 c.ask_recall_n, allow=set(by_id)):
                if eid not in in_pool:
                    in_pool.add(eid); pool.append(eid); grew = True
            if not grew:
                break
            ranked, _ = self._select(query, pool, by_id, allow_expand=False)
            final = _cut(ranked)
        return final

    def _ask(self, query: str, material: List[MemoryEntry]) -> str:
        """Inspect the current evidence and return a query only when something is missing; otherwise return an empty string."""
        if not (self.cfg.ask_enabled and self._get_llm is not None and material):
            return ""
        try:
            raw = self._get_llm().chat(
                ASK_SYSTEM,
                ASK_USER_TMPL.format(question=query,
                                     material="\n".join(e.render(n) for n, e
                                                        in enumerate(material, 1))),
                json_mode=True, thinking=self.cfg.ask_thinking, temperature=0.0,
                max_tokens=self.cfg.ask_max_tokens, model=self.cfg.select_model)
            return (json.loads(raw or "{}", strict=False).get("ask") or "").strip()
        except Exception:
            return ""

    def _expand_context(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Expand each raw hit to its surrounding session/turn context window."""
        w = self.cfg.context_window
        out: Dict[str, MemoryEntry] = {}
        for e in entries[: self.cfg.expand_max_entries]:
            if not e.session:
                continue
            for ep in self.mem.episodics_in_window(e.session, e.turn - w, e.turn + w):
                out.setdefault(ep.entry_id, ep)
        return list(out.values())

    def _select(self, query: str, pool_ids: List[str], by_id, allow_expand: bool = True):
        if not pool_ids:
            return [], []
        cand = "\n".join(by_id[i].render(n) for n, i in enumerate(pool_ids, 1))
        user = SELECT_USER_TMPL.format(question=query, candidates=cand)
        if not allow_expand:
            user += '\n\nDo NOT request expansion now: return "expand":[] and only the final ranking.'
        try:
            raw = self._get_llm().chat(SELECT_SYSTEM, user, json_mode=True,
                                       thinking=self.cfg.select_thinking, temperature=0.0,
                                       max_tokens=self.cfg.select_max_tokens, model=self.cfg.select_model)
            d = json.loads(raw or "{}", strict=False)
            order = d.get("ranked", [])
            exp = d.get("expand", []) if allow_expand else []
        except Exception:
            return [], []

        def _map(nums):
            out, seen = [], set()
            for n in nums:
                try:
                    idx = int(n) - 1
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(pool_ids) and pool_ids[idx] not in seen:
                    out.append(pool_ids[idx]); seen.add(pool_ids[idx])
            return out
        return _map(order), _map(exp)




    def project(self, query: str, channel: Optional[str] = None) -> Projection:
        """1. Projection: ask the LLM for issue / rows / tense and fall back to conservative defaults on failure."""
        roster = self.mem.roster(channel)
        if self._get_llm is None:
            return Projection(query, ["ALL"], "head")
        user = PROJECT_USER_TMPL.format(question=query, roster=", ".join(roster))
        try:
            raw = self._get_llm().chat(PROJECT_SYSTEM, user, json_mode=True,
                                       thinking=self.cfg.select_thinking, temperature=0.0,
                                       max_tokens=self.cfg.select_max_tokens, model=self.cfg.select_model)
            d = json.loads(raw or "{}", strict=False)
            issue = str(d.get("issue") or query).strip() or query
            rows = d.get("rows") or ["ALL"]
            if isinstance(rows, str):
                rows = [rows]
            rows = [str(r).strip() for r in rows if str(r).strip()]
            tense = "full" if str(d.get("tense", "head")).lower().startswith("f") else "head"
        except Exception:
            return Projection(query, ["ALL"], "head")






        return Projection(issue, rows or ["ALL"], tense)

    def _resolve_rows(self, rows: List[str], channel: Optional[str]) -> List[str]:
        """Resolve rows into concrete row names."""
        roster = self.mem.roster(channel)
        out: List[str] = []
        for r in rows:
            ru = r.upper()
            if ru == "ALL":
                out.extend(roster)
            elif ru == "GROUP":
                out.append(GROUP_OWNER)
            else:

                m = next((s for s in roster if s.lower() == r.lower()), r)
                out.append(m)
        seen, uniq = set(), []
        for o in out:
            if o not in seen:
                seen.add(o); uniq.append(o)
        return uniq

    def s2_slice(self, query: str, proj: Optional[Projection] = None,
                 channel: Optional[str] = None) -> Tuple[Projection, List[Cell]]:
        """Retrieve top-k2 per row and issue, fold chains by tense, and fall back to raw evidence when derived memory is empty."""
        c = self.cfg
        proj = proj or self.project(query, channel)
        qv = self.embedder.encode_one(proj.issue)
        cells: List[Cell] = []
        rows = self._resolve_rows(proj.rows, channel)
        group_in_rows = GROUP_OWNER in rows
        for owner in rows:


            src_cells: List[Cell] = []
            if owner != GROUP_OWNER and not group_in_rows and c.s2_source_k > 0:
                for e, _ in self.mem.row_search_as_source(owner, qv, k=c.s2_source_k,
                                                          min_cos=c.s2_derived_tau, layers=c.s2_layers):
                    hd = self.mem.head(e.entry_id) or e
                    ch = self.mem.chain(e.entry_id) if proj.tense == "full" else [hd]
                    src_cells.append(Cell(owner, ch, as_source=True))

            hits = self.mem.row_search(owner, qv, k=c.s2_k, derived=True,
                                       min_cos=c.s2_derived_tau, layers=c.s2_layers)
            if hits:
                chains: List[List[MemoryEntry]] = []
                seen_head = set()
                for e, _ in hits:
                    hd = self.mem.head(e.entry_id) or e
                    if hd.entry_id in seen_head:
                        continue
                    seen_head.add(hd.entry_id)
                    ch = self.mem.chain(e.entry_id) if proj.tense == "full" else [hd]
                    chains.append(ch)
                for ch in chains[: c.s2_k]:
                    cells.append(Cell(owner, ch))
                cells.extend(src_cells)
                continue
            if src_cells:
                cells.extend(src_cells)
                continue

            got = None
            if c.s2_fallback_raw:
                rh = self.mem.row_search(owner, qv, k=1, derived=False)
                if rh and rh[0][1] >= c.s2_raw_tau:
                    got = rh[0][0]
            cells.append(Cell(owner, [got] if got else [], from_raw=bool(got)))
        return proj, cells

    @staticmethod
    def render_slice(cells: List[Cell], tense: str = "head") -> str:
        if not cells:
            return "(empty)"

        ordered = ([c for c in cells if c.owner != GROUP_OWNER]
                   + [c for c in cells if c.owner == GROUP_OWNER])
        return "\n".join(c.render(tense) for c in ordered)




    def retrieve(self, query: str, asker: str = "", k: Optional[int] = None) -> List[MemoryEntry]:
        raw = self.s1_raw(query, k=k) if self.cfg.s1_enabled else []
        _, cells = self.s2_slice(query) if self.cfg.s2_enabled else (Projection(query, [], "head"), [])
        out, seen = list(raw), {e.entry_id for e in raw}
        for cell in cells:
            for e in cell.chain:
                if e.entry_id not in seen:
                    seen.add(e.entry_id); out.append(e)
        return out

    def retrieve_dual(self, query: str, channel: Optional[str] = None,
                      k: Optional[int] = None):
        """Return raw passages, the Projection, and the tensor slice for separate rendering by the answerer."""
        raw = self.s1_raw(query, k=k) if self.cfg.s1_enabled else []
        if self.cfg.s2_enabled:
            proj, cells = self.s2_slice(query, channel=channel)
        else:
            proj, cells = Projection(query, [], "head"), []
        return raw, proj, cells
