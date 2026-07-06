"""In-memory vector index with cosine similarity search.

Implemented with pure numpy (no external vector database) to keep the
retrieval algorithm transparent: embeddings are L2-normalized, so cosine
similarity is a single matrix multiplication followed by a top-k selection
using argpartition, which runs in O(n) instead of a full O(n log n) sort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class VectorIndex:
    """Stores (embedding, item_id) pairs and answers top-k similarity queries."""

    def __init__(self, embeddings: np.ndarray | None = None, ids: List[str] | None = None):
        self.embeddings = embeddings if embeddings is not None else np.empty((0, 0))
        self.ids = ids or []

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Append new items to the index."""
        if len(embeddings) != len(ids):
            raise ValueError("embeddings and ids must have the same length")
        if len(self.ids) == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k most similar items to a query embedding.

        query: array of shape (d,) or (1, d), assumed L2-normalized.
        Returns a list of (item_id, cosine_similarity) sorted descending.
        """
        if len(self.ids) == 0:
            return []
        q = query.reshape(-1)
        scores = self.embeddings @ q                       # cosine similarity
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]          # O(n) top-k selection
        top = top[np.argsort(-scores[top])]                # sort only the k winners
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        ids = json.loads((directory / "ids.json").read_text())
        return cls(embeddings, ids)
