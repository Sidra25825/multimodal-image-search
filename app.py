"""Multimodal Image Search - Streamlit app.

Two features, both powered by CLIP:
1. Text-to-image search: type a natural language query, retrieve the
   closest images from your uploaded library.
2. Zero-shot classification: upload one image and a list of candidate
   labels, get a probability for each label without any training.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from src.embedder import ClipEmbedder
from src.index import VectorIndex

st.set_page_config(page_title="Multimodal Image Search", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading CLIP model (first run only)...")
def get_embedder() -> ClipEmbedder:
    return ClipEmbedder()


def init_state() -> None:
    if "index" not in st.session_state:
        st.session_state.index = VectorIndex()
    if "image_store" not in st.session_state:
        st.session_state.image_store = {}  # id -> PIL image


init_state()

st.title("🔍 Multimodal Image Search")
st.caption(
    "Search your image library with natural language, powered by CLIP "
    "(Contrastive Language-Image Pre-training)."
)

tab_search, tab_classify, tab_about = st.tabs(
    ["Text → Image Search", "Zero-Shot Classification", "How It Works"]
)

with tab_search:
    col_upload, col_query = st.columns([1, 2])

    with col_upload:
        st.subheader("1. Build your library")
        files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if files:
            new_files = [f for f in files if f.name not in st.session_state.image_store]
            if new_files:
                embedder = get_embedder()
                images = [Image.open(io.BytesIO(f.read())).convert("RGB") for f in new_files]
                embs = embedder.embed_images(images)
                st.session_state.index.add(embs, [f.name for f in new_files])
                for f, img in zip(new_files, images):
                    st.session_state.image_store[f.name] = img
        st.metric("Images indexed", len(st.session_state.index))

    with col_query:
        st.subheader("2. Search with words")
        query = st.text_input(
            "Describe what you are looking for",
            placeholder="a dog playing in the snow",
        )
        top_k = st.slider("Number of results", 1, 12, 4)

        if query and len(st.session_state.index) > 0:
            embedder = get_embedder()
            q_emb = embedder.embed_texts([query])[0]
            results = st.session_state.index.search(q_emb, k=top_k)

            cols = st.columns(min(4, len(results)))
            for i, (image_id, score) in enumerate(results):
                with cols[i % len(cols)]:
                    st.image(
                        st.session_state.image_store[image_id],
                        caption=f"{image_id}  |  similarity {score:.3f}",
                        use_container_width=True,
                    )
        elif query:
            st.info("Upload some images first, then search.")

with tab_classify:
    st.subheader("Zero-shot image classification")
    st.write(
        "CLIP can classify an image against **any** labels you invent, "
        "with no training. Try it:"
    )
    col_img, col_labels = st.columns(2)

    with col_img:
        single = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "webp"], key="clf")
        if single:
            image = Image.open(single).convert("RGB")
            st.image(image, use_container_width=True)

    with col_labels:
        labels_text = st.text_area(
            "Candidate labels (one per line)",
            value="cat\ndog\ncar\nmountain landscape\nfood",
            height=150,
        )
        if single and labels_text.strip():
            labels = [l.strip() for l in labels_text.splitlines() if l.strip()]
            embedder = get_embedder()
            ranked = embedder.zero_shot_classify(image, labels)
            st.write("**Predictions:**")
            for label, prob in ranked:
                st.progress(prob, text=f"{label}: {prob:.1%}")

with tab_about:
    st.subheader("Architecture")
    st.markdown(
        """
**Pipeline:** images and text are projected into the *same* 512-dimensional
embedding space by CLIP's dual encoders (a Vision Transformer for images and
a Transformer for text). Because both embeddings are L2-normalized, cosine
similarity reduces to a dot product, and retrieval is a single matrix
multiplication followed by an O(n) top-k selection with `argpartition`.

**Components:**
- `src/embedder.py` wraps the CLIP model and handles batching and normalization
- `src/index.py` implements the vector index and top-k search from scratch in numpy
- `src/evaluate.py` measures retrieval quality with Recall@K and Mean Reciprocal Rank

**Evaluation:** on a captioned image set, each caption is used as a query and
we check whether the correct image ranks in the top K results. See the README
for reproduction instructions.
"""
    )
