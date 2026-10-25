import os

from api import BASE_DIR, resolve_inside


def test_resolve_inside_accepts_real_files():
    pdfs = [os.path.relpath(os.path.join(r, f), BASE_DIR) for r, _, fs in os.walk(os.path.join(BASE_DIR, "researches")) for f in fs if f.endswith(".pdf")]
    if pdfs:
        assert resolve_inside(pdfs[0], "researches", ".pdf") is not None


def test_resolve_inside_blocks_traversal_and_wrong_types():
    assert resolve_inside("../../etc/passwd", "data", ".parquet") is None
    assert resolve_inside("../.bashrc", "researches", ".pdf") is None
    assert resolve_inside("hypoq.db", "data", ".parquet") is None
    assert resolve_inside("README.md", "researches", ".pdf") is None
    assert resolve_inside("data/../README.md", "data", ".md") is None
