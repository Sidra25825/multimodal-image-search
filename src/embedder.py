"""CLIP-based embedder for images and text.

Wraps the OpenAI CLIP model (via Hugging Face Transformers) and exposes
two simple methods: embed_images() and embed_texts(). Both return
L2-normalized numpy arrays so that cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    """Thin wrapper around a CLIP model for multimodal embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Embed a list of PIL images. Returns array of shape (n, d)."""
        feats = []
        for i in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns array of shape (n, d)."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        f = self.model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    @torch.no_grad()
    def zero_shot_classify(self, image: Image.Image, labels: List[str]) -> List[tuple]:
        """Zero-shot image classification.

        Compares one image against a set of candidate text labels and
        returns (label, probability) pairs sorted by probability.
        """
        prompts = [f"a photo of a {label.strip()}" for label in labels]
        image_emb = self.embed_images([image])          # (1, d)
        text_emb = self.embed_texts(prompts)            # (n, d)
        logits = 100.0 * image_emb @ text_emb.T         # CLIP temperature scaling
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
        ranked = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)
        return ranked
