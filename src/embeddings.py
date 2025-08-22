# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from .config import EMBEDDING_MODEL

_model = None
_index = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def query_top_k(index, query_emb, k=5):
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    return D, I
