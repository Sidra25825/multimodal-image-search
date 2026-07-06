"""Unit tests for the vector index (pure numpy, no model download needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.index import VectorIndex


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_search_returns_most_similar_first():
    rng = np.random.default_rng(42)
    embs = normalize(rng.normal(size=(100, 512)).astype(np.float32))
    ids = [f"img_{i}" for i in range(100)]
    index = VectorIndex()
    index.add(embs, ids)

    # Query identical to item 7 must rank it first with similarity ~1
    results = index.search(embs[7], k=5)
    assert results[0][0] == "img_7"
    assert abs(results[0][1] - 1.0) < 1e-5
    # Scores must be sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_index():
    embs = normalize(np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32))
    index = VectorIndex()
    index.add(embs, ["a", "b", "c"])
    assert len(index.search(embs[0], k=10)) == 3


def test_empty_index():
    assert VectorIndex().search(np.ones(8), k=5) == []


def test_save_and_load(tmp_path):
    embs = normalize(np.random.default_rng(1).normal(size=(10, 16)).astype(np.float32))
    index = VectorIndex()
    index.add(embs, [f"x{i}" for i in range(10)])
    index.save(tmp_path / "store")
    loaded = VectorIndex.load(tmp_path / "store")
    assert loaded.ids == index.ids
    assert np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.search(embs[3], k=1)[0][0] == "x3"


def test_incremental_add():
    rng = np.random.default_rng(2)
    a = normalize(rng.normal(size=(5, 16)).astype(np.float32))
    b = normalize(rng.normal(size=(4, 16)).astype(np.float32))
    index = VectorIndex()
    index.add(a, [f"a{i}" for i in range(5)])
    index.add(b, [f"b{i}" for i in range(4)])
    assert len(index) == 9
    assert index.search(b[2], k=1)[0][0] == "b2"
