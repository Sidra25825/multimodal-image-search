"""CLI tool: build and save a vector index from a folder of images.

Usage:
    python scripts/build_index.py --images data/images --out index_store
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.embedder import ClipEmbedder
from src.index import VectorIndex

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default="index_store")
    args = parser.parse_args()

    paths = [p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTENSIONS]
    print(f"Found {len(paths)} images")

    embedder = ClipEmbedder()
    images = [Image.open(p) for p in paths]
    embeddings = embedder.embed_images(images)

    index = VectorIndex()
    index.add(embeddings, [p.name for p in paths])
    index.save(args.out)
    print(f"Index with {len(index)} items saved to {args.out}/")


if __name__ == "__main__":
    main()
