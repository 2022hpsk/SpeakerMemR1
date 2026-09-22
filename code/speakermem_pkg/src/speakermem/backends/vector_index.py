"""Vector index: an abstract interface and a dependency-free NumPy cosine implementation."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, entry_id: str, vec: np.ndarray) -> None: ...
    @abstractmethod
    def remove(self, entry_id: str) -> None: ...
    @abstractmethod
    def search(self, qvec: np.ndarray, n: int, allow: set | None = None) -> List[Tuple[str, float]]:
        """Return [(entry_id, cosine)] in descending score order; restrict the search to allow when it is non-empty."""
        ...

    def get_vector(self, entry_id: str):
        """Return the normalized vector for an entry, or None when it is unavailable."""
        return None


class NumpyIndex(VectorIndex):
    """In-memory cosine index: L2-normalized vectors make dot products equal cosine similarity. Fast enough for small and medium stores (<10^5 entries).

    Design: `_vecs` (entry_id to vector) is the **source of truth**; `_mat`/`_ids`/`_pos` are rebuilt-on-demand
    derived caches marked by `_dirty`. **Incremental upserts never discard existing vectors**; rebuilds are locked and safe for concurrent QA.
    """

    def __init__(self):
        self._vecs: Dict[str, np.ndarray] = {}
        self._ids: List[str] = []
        self._pos: Dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._dirty = True
        self._lock = threading.Lock()

    def upsert(self, entry_id: str, vec: np.ndarray) -> None:
        with self._lock:
            self._vecs[entry_id] = np.asarray(vec, dtype=np.float32)
            self._dirty = True

    def remove(self, entry_id: str) -> None:
        with self._lock:
            if self._vecs.pop(entry_id, None) is not None:
                self._dirty = True

    def _ensure(self):
        """Ensure the matrix cache is current under the lock and return a (mat, ids) snapshot."""
        with self._lock:
            if self._dirty:
                self._ids = list(self._vecs.keys())
                self._pos = {i: k for k, i in enumerate(self._ids)}
                self._mat = (np.vstack([self._vecs[i] for i in self._ids]) if self._ids
                             else np.zeros((0, 1), dtype=np.float32))
                self._dirty = False
            return self._mat, self._ids

    def get_vector(self, entry_id):
        return self._vecs.get(entry_id)

    def search(self, qvec, n, allow=None):
        mat, ids = self._ensure()
        if not ids:
            return []
        sims = mat @ np.asarray(qvec, dtype=np.float32)
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for idx in order:
            eid = ids[idx]
            if allow is not None and eid not in allow:
                continue
            out.append((eid, float(sims[idx])))
            if len(out) >= n:
                break
        return out
