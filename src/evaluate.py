"""Retrieval evaluation: Recall@K and Mean Reciprocal Rank (MRR).

Given a folder of images and a captions.csv file with columns
(filename, caption), this module embeds every caption as a query,
retrieves images from the index, and measures how often the correct
image appears in the top K results.

Usage:
    python -m src.evaluate --images data/images --captions data/captions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .embedder import ClipEmbedder
from .index import VectorIndex


def load_captions(csv_path: str | Path) -> Dict[str, str]:
    """Read captions.csv into a {filename: caption} mapping."""
    captions: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions[row["filename"].strip()] = row["caption"].strip()
    return captions


def recall_at_k(ranks: List[int], k: int) -> float:
    """Fraction of queries whose correct item ranked within the top k (1-indexed)."""
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


def mean_reciprocal_rank(ranks: List[int]) -> float:
    return float(np.mean([1.0 / r for r in ranks]))


def evaluate(images_dir: str | Path, captions_csv: str | Path,
             ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """Run text-to-image retrieval evaluation and return metrics."""
    images_dir = Path(images_dir)
    captions = load_captions(captions_csv)

    filenames = [f for f in captions if (images_dir / f).exists()]
    if not filenames:
        raise FileNotFoundError("No captioned images found in the images directory")

    embedder = ClipEmbedder()

    print(f"Embedding {len(filenames)} images...")
    images = [Image.open(images_dir / f) for f in filenames]
    image_embs = embedder.embed_images(images)

    index = VectorIndex()
    index.add(image_embs, filenames)

    print(f"Embedding {len(filenames)} caption queries...")
    text_embs = embedder.embed_texts([captions[f] for f in filenames])

    ranks: List[int] = []
    for i, fname in enumerate(filenames):
        results = index.search(text_embs[i], k=len(filenames))
        retrieved = [r[0] for r in results]
        ranks.append(retrieved.index(fname) + 1)

    metrics = {f"Recall@{k}": recall_at_k(ranks, k) for k in ks}
    metrics["MRR"] = mean_reciprocal_rank(ranks)
    metrics["queries"] = len(ranks)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text-to-image retrieval")
    parser.add_argument("--images", required=True, help="Directory of images")
    parser.add_argument("--captions", required=True, help="CSV with filename,caption columns")
    args = parser.parse_args()

    metrics = evaluate(args.images, args.captions)
    print("\nText-to-image retrieval results")
    print("-" * 33)
    for name, value in metrics.items():
        if name == "queries":
            print(f"{name:<12} {value}")
        else:
            print(f"{name:<12} {value:.3f}")


if __name__ == "__main__":
    main()
