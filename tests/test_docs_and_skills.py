import ast
import glob
import os
import re
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = ["README.md", "AGENTS.md", "DATA.md", "RESEARCH.md"] + sorted(glob.glob(os.path.join(ROOT, ".agents", "skills", "*", "SKILL.md")))


def _read(path):
    return open(os.path.join(ROOT, path) if not os.path.isabs(path) else path, encoding="utf-8").read()


def test_python_snippets_in_skills_are_valid_syntax():
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, ".agents", "skills", "*", "SKILL.md"))):
        for i, block in enumerate(re.findall(r"```python\n(.*?)```", _read(path), re.S)):
            try:
                ast.parse(textwrap.dedent(block))
            except SyntaxError as exc:
                bad.append(f"{os.path.basename(os.path.dirname(path))} block {i}: {exc}")
    assert not bad, bad


def test_documented_data_files_exist():
    placeholders = {"data/domain/name*.parquet"}
    missing = []
    for path in DOCS:
        for m in re.finditer(r"data/[A-Za-z_]+/[A-Za-z0-9_*\.]+\.parquet", _read(path)):
            p = m.group(0)
            if p in placeholders:
                continue
            if not glob.glob(os.path.join(ROOT, p)):
                missing.append(f"{path}: {p}")
    assert not missing, missing


def test_every_skill_has_frontmatter_and_is_listed_in_agents_md():
    agents = _read("AGENTS.md")
    for path in glob.glob(os.path.join(ROOT, ".agents", "skills", "*", "SKILL.md")):
        name = os.path.basename(os.path.dirname(path))
        head = _read(path).split("---")[1]
        assert f"name: {name}" in head and "description:" in head, name
        assert f"`{name}`" in agents, f"{name} missing from AGENTS.md"


def test_no_stale_project_names_or_removed_datasets():
    banned = ["autoresearch.db", "from autoresearch", "vietnam_macro", "vietnam_monetary", "vn_etf_nav_aum", "data/sentiment/", "paper-hub"]
    hits = []
    for path in DOCS:
        text = _read(path)
        hits += [f"{path}: {b}" for b in banned if b in text]
    assert not hits, hits
