"""Embedding backends: an abstract interface and a sentence-transformers default implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedder(ABC):
    """Encode text as L2-normalized vectors so a dot product is cosine similarity."""
    dim: int

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Default local sentence-transformers implementation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._model.encode(
            texts, normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=False), dtype=np.float32)
