"""Storage backends: a simple JSON store and a recommended SQLite store with queryable fields and persisted embedding blobs.

JSON requires recomputing all embeddings on load and is less suitable for incremental concurrent access.
SQLite stores float32 vectors in one standard-library database and supports owner and layer queries.
"""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import MemoryEntry


class MemoryStore(ABC):
    @abstractmethod
    def add_many(self, items: List[Tuple[MemoryEntry, Optional[np.ndarray]]]) -> None: ...
    @abstractmethod
    def update(self, entry: MemoryEntry) -> None: ...
    @abstractmethod
    def all_entries(self) -> List[MemoryEntry]: ...
    @abstractmethod
    def vectors(self) -> Dict[str, np.ndarray]: ...

    def add_roster(self, channel: str, speakers) -> None:
        r = getattr(self, "_roster", None)
        if r is None:
            r = self._roster = {}
        r.setdefault(channel or "", set()).update(s for s in speakers if s)

    def roster(self, channel: Optional[str] = None) -> List[str]:
        r = getattr(self, "_roster", None) or {}
        if channel is None:
            return sorted({s for v in r.values() for s in v})
        return sorted(r.get(channel, set()))

    def close(self) -> None: ...


class JsonStore(MemoryStore):
    """Simple JSON store: entries are saved in .json and optional vectors in a matching .npz file. Missing vectors are recomputed by the caller after loading."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.vpath = self.path.with_suffix(".vecs.npz")
        self._entries: List[MemoryEntry] = []
        self._vecs: Dict[str, np.ndarray] = {}
        if self.path.exists():
            self._entries = [MemoryEntry.from_dict(d) for d in json.load(open(self.path))]
        if self.vpath.exists():
            z = np.load(self.vpath)
            self._vecs = {k: z[k] for k in z.files}

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump([e.to_dict() for e in self._entries], open(self.path, "w"), ensure_ascii=False)
        if self._vecs:
            np.savez(self.vpath, **self._vecs)

    def add_many(self, items):
        for e, v in items:
            self._entries.append(e)
            if v is not None:
                self._vecs[e.entry_id] = np.asarray(v, dtype=np.float32)
        self._flush()

    def update(self, entry):
        for i, e in enumerate(self._entries):
            if e.entry_id == entry.entry_id:
                self._entries[i] = entry
                break
        self._flush()

    def all_entries(self):
        return list(self._entries)

    def vectors(self):
        return dict(self._vecs)


class SqliteStore(MemoryStore):
    """SQLite store with an entries table and vector blobs. Supports incremental writes and field queries without recomputing embeddings."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS entries(
                entry_id TEXT PRIMARY KEY, content TEXT, owner TEXT, source TEXT,
                layer TEXT, turn_created INTEGER, ts TEXT, utype TEXT, confidence REAL,
                links TEXT, from_ids TEXT, superseded_by TEXT, active INTEGER, vec BLOB,
                session TEXT, turn INTEGER)""")

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS roster(
                channel TEXT, speaker TEXT, PRIMARY KEY(channel, speaker))""")

        for col, typ in (("session", "TEXT"), ("turn", "INTEGER")):
            try:
                self._db.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_owner ON entries(owner)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_layer ON entries(layer)")
        self._db.commit()

    @staticmethod
    def _row(e: MemoryEntry, v: Optional[np.ndarray]):
        return (e.entry_id, e.content, e.owner, e.source, e.layer, e.turn_created, e.ts,
                e.utype, e.confidence, json.dumps(e.links), json.dumps(e.from_ids),
                e.superseded_by, 1,
                (np.asarray(v, dtype=np.float32).tobytes() if v is not None else None),
                e.session, e.turn)

    def add_many(self, items):
        self._db.executemany(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [self._row(e, v) for e, v in items])
        self._db.commit()

    def update(self, entry):
        self._db.execute(
            "UPDATE entries SET content=?,layer=?,confidence=?,links=?,from_ids=?,"
            "superseded_by=? WHERE entry_id=?",
            (entry.content, entry.layer, entry.confidence, json.dumps(entry.links),
             json.dumps(entry.from_ids), entry.superseded_by, entry.entry_id))
        self._db.commit()

    def all_entries(self):
        out = []
        for r in self._db.execute(
                "SELECT entry_id,content,owner,source,layer,turn_created,ts,utype,confidence,"
                "links,from_ids,superseded_by,session,turn FROM entries"):
            out.append(MemoryEntry(
                entry_id=r[0], content=r[1], owner=r[2], source=r[3], layer=r[4],
                turn_created=r[5], ts=r[6], utype=r[7], confidence=r[8],
                links=json.loads(r[9] or "[]"), from_ids=json.loads(r[10] or "[]"),
                superseded_by=(r[11] or ""),
                session=(r[12] or ""), turn=(r[13] or 0)))
        return out

    def vectors(self):
        out = {}
        for eid, blob in self._db.execute("SELECT entry_id,vec FROM entries WHERE vec IS NOT NULL"):
            out[eid] = np.frombuffer(blob, dtype=np.float32)
        return out


    def add_roster(self, channel: str, speakers):
        self._db.executemany("INSERT OR IGNORE INTO roster VALUES (?,?)",
                             [(channel or "", s) for s in speakers if s])
        self._db.commit()

    def roster(self, channel: Optional[str] = None):
        if channel is None:
            rows = self._db.execute("SELECT speaker FROM roster")
        else:
            rows = self._db.execute("SELECT speaker FROM roster WHERE channel=?", (channel,))
        return sorted({r[0] for r in rows})

    def close(self):
        self._db.close()


def make_store(path: str, backend: str = "sqlite") -> MemoryStore:
    return SqliteStore(path) if backend == "sqlite" else JsonStore(path)
