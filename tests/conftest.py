import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Deterministic embeddings so tests never download or load a neural model."""
    import hypoq.db as db

    def fake(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=16).astype("float32")

    monkeypatch.setattr(db, "get_embedding", fake)
    monkeypatch.setattr(db, "get_active_provider", lambda: "test")


@pytest.fixture
def db_path(tmp_path):
    from hypoq.db import init_db

    path = str(tmp_path / "test.db")
    init_db(path)
    return path
