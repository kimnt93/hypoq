"""
Embedding Vector Engine for AutoResearch.
Supports dual providers:
1. Google Gemini Embeddings (text-embedding-004) via Google AI Studio API (free tier).
2. Open-source Hugging Face models optimized for systems under 8GB RAM (default: all-MiniLM-L6-v2, 384-dim, ~120MB RAM).

Enforces L2 unit-normalization for fast dot-product cosine similarity.
"""

import os
import sys
import json
import struct
import numpy as np
import requests
from typing import List, Optional, Tuple, Dict, Any

DEFAULT_HF_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

_HF_MODEL_INSTANCE = None


def get_active_provider() -> str:
    """
    Determines active embedding provider based on environment:
    - If EMBEDDING_PROVIDER == 'hf' or 'local': HuggingFace open-source model.
    - If GEMINI_API_KEY or GOOGLE_API_KEY is present: Google Gemini text-embedding-004.
    - Default: HuggingFace open-source model under 8GB RAM.
    """
    provider_override = os.environ.get("EMBEDDING_PROVIDER", "").lower().strip()
    if provider_override in ["hf", "local", "huggingface"]:
        return "huggingface"
    if provider_override in ["gemini", "google"]:
        return "gemini"

    # Auto-detect Gemini API key
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"

    return "huggingface"


def get_gemini_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _get_hf_model():
    """Lazily loads Hugging Face model onto CPU/CUDA (RAM footprint < 200MB)."""
    global _HF_MODEL_INSTANCE
    if _HF_MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Runs efficiently on CPU under 8GB RAM constraint
            _HF_MODEL_INSTANCE = SentenceTransformer(DEFAULT_HF_MODEL, device="cpu")
        except Exception as e:
            print(f"[Warning] Failed to load SentenceTransformer ({DEFAULT_HF_MODEL}): {e}")
            _HF_MODEL_INSTANCE = False
    return _HF_MODEL_INSTANCE


def embed_via_gemini(text: str, api_key: str) -> np.ndarray:
    """Calls Google AI Studio Gemini text-embedding-004 REST endpoint."""
    url = f"{GEMINI_API_URL}?key={api_key}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text[:8000]}]  # Cap text to 8k chars
        }
    }
    headers = {"Content-Type": "application/json"}
    
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    if res.status_code == 200:
        data = res.json()
        values = data.get("embedding", {}).get("values", [])
        if values:
            vec = np.array(values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)
    raise RuntimeError(f"Gemini API embedding error ({res.status_code}): {res.text}")


def embed_via_hf(text: str) -> np.ndarray:
    """Embeds text using open-source Hugging Face model under 8GB RAM."""
    model = _get_hf_model()
    if model:
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    # Deterministic fallback vector if offline/no model (384 dimensions)
    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


def get_embedding(text: str) -> np.ndarray:
    """
    Generates a unit-normalized Float32 vector embedding for text.
    Uses Gemini text-embedding-004 if API key present, otherwise HuggingFace local model.
    """
    provider = get_active_provider()
    clean_text = text.strip() if text else ""
    if not clean_text:
        clean_text = "empty research hypothesis"

    if provider == "gemini":
        api_key = get_gemini_api_key()
        if api_key:
            try:
                return embed_via_gemini(clean_text, api_key)
            except Exception as e:
                print(f"[Warning] Gemini API call failed: {e}. Falling back to HuggingFace.")
                return embed_via_hf(clean_text)

    return embed_via_hf(clean_text)


def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Generates embeddings for a batch of texts."""
    provider = get_active_provider()
    
    if provider == "huggingface":
        model = _get_hf_model()
        if model:
            vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return vecs.astype(np.float32)

    # Iterative fallback
    vectors = [get_embedding(t) for t in texts]
    return np.vstack(vectors).astype(np.float32)


def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serializes a Float32 numpy array into raw byte string for SQLite BLOB or Postgres BYTEA."""
    return vec.astype(np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserializes raw binary bytes into a 1D Float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between 1D query_vec and 2D doc_vectors (N, D).
    Assumes vectors are L2-normalized; dot-product is equivalent to cosine similarity.
    """
    if len(doc_vectors) == 0:
        return np.array([], dtype=np.float32)
    # Dot product
    sims = np.dot(doc_vectors, query_vec)
    return np.clip(sims, -1.0, 1.0)
