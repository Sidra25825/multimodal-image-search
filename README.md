# 🔍 Multimodal Image Search

Search an image library using natural language, powered by **CLIP** (Contrastive Language-Image Pre-training). Type *"a dog playing in the snow"* and instantly retrieve the closest matching images, no tags, no metadata, no training required.

**Live demo:** *(add your Streamlit Cloud or Hugging Face Spaces link here)*

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange) ![Tests](https://img.shields.io/badge/tests-5%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Text → Image Search:** natural language queries against your own image library
- **Zero-Shot Classification:** classify any image against arbitrary labels you invent on the spot, with no training
- **Retrieval Evaluation:** reproducible Recall@K and Mean Reciprocal Rank benchmarks
- **From-Scratch Vector Index:** cosine similarity search implemented in pure numpy with O(n) top-k selection, so the retrieval algorithm is fully transparent

## How It Works

CLIP projects images and text into the **same** 512-dimensional embedding space using two encoders: a Vision Transformer for images and a Transformer for text. Both outputs are L2-normalized, so cosine similarity reduces to a dot product:

```
query text ──► Text Encoder ──► q ∈ R⁵¹²  ─┐
                                            ├──►  scores = E · q  ──► top-k results
image library ──► Image Encoder ──► E ∈ Rⁿˣ⁵¹² ─┘
```

Top-k retrieval uses `np.argpartition`, which selects the k best candidates in O(n) time and only sorts those k winners, instead of sorting the entire score vector.

## Project Structure

```
├── app.py                  # Streamlit web app (search + zero-shot classification)
├── src/
│   ├── embedder.py         # CLIP wrapper: batched, normalized image/text embeddings
│   ├── index.py            # Vector index: cosine similarity + O(n) top-k search
│   └── evaluate.py         # Recall@K and MRR evaluation pipeline
├── scripts/
│   └── build_index.py      # CLI: embed a folder of images and persist the index
├── tests/
│   └── test_index.py       # Unit tests for the retrieval logic
└── data/
    └── captions_example.csv
```

## Quickstart

```bash
git clone https://github.com/Sidra25825/multimodal-image-search.git
cd multimodal-image-search
pip install -r requirements.txt
streamlit run app.py
```

The CLIP model (~600 MB) downloads automatically on first run.

## Evaluation

Retrieval quality is measured with text-to-image **Recall@K** and **MRR**: every caption is used as a query, and we check whether the correct image appears in the top K results.

```bash
python -m src.evaluate --images data/images --captions data/captions.csv
```

| Metric   | Score |
|----------|-------|
| Recall@1 | *run to reproduce* |
| Recall@5 | *run to reproduce* |
| MRR      | *run to reproduce* |

The captions file is a simple CSV with `filename,caption` columns, see `data/captions_example.csv`.

## Running the Tests

```bash
python -m pytest tests/ -v
```

## Deploy Your Own

**Streamlit Community Cloud (free):** push this repo to GitHub, go to [share.streamlit.io](https://share.streamlit.io), select the repo and `app.py`, done.

**Hugging Face Spaces (free):** create a new Space with the Streamlit SDK and push this repo to it.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, CLIP (ViT-B/32), NumPy, Streamlit, pytest

## License

MIT
