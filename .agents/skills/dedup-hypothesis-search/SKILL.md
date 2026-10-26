---
name: dedup-hypothesis-search
description: Search existing research hypotheses using dense semantic vector embeddings (Google Gemini text-embedding-004 or open-source Hugging Face all-MiniLM-L6-v2) across SQLite or PostgreSQL backends to eliminate duplicate experiments.
version: 2.0.0
author: hypoq
tags: [search, deduplication, semantic-search, vector-embeddings, gemini, huggingface, postgres, sqlite]
---

# Deduplication & Hypothesis Search Skill

## Overview
To prevent wasted compute and redundant empirical exploration, this skill provides dense semantic vector search over historical hypotheses stored in `hypoq.db` (SQLite) or PostgreSQL (configured via `DATABASE_URL`). It replaces rigid keyword matching with neural embeddings that compute cosine similarity over hypothesis concepts, null/alt formulations, and empirical findings.

### Embedding Providers Supported
1. **Google Gemini (`text-embedding-004`)**: Activated when `GEMINI_API_KEY` is provided. Free tier, 768 dimensions.
2. **Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)**: Default open-source model. 384 dimensions, < 200MB RAM, optimized for machines under 8GB RAM.
### Database lifecycle and changing the embedding method
`hypoq.db` is git-ignored. It is rebuilt from the committed raw records (`researches/*/results/hypothesis_record.json`) by `make db-import`, which creates the embeddings with the active method. Each vector is stored with the id of the method that produced it (`huggingface:<model>` or `gemini:<model>`).
After changing `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` or the Gemini key, run `make db-migrate` (add `FORCE=1` to redo all): vectors whose id differs from the active method are recomputed from the stored text. `python run.py --search` also migrates on demand. If the method cannot produce vectors, the commands fail with an explanation; no substitute vectors are ever created. Never compare or mix vectors of different method ids.

## When to Use
- Before launching a new experimental research run.
- When an agent is prompted with a research question to see if previous experiments accepted, rejected, or found inconclusive evidence on the same topic.
- To discover prior test statistics and refine an alternative angle or model specification.

## CLI Usage
Execute semantic search from the terminal:
```bash
# Semantic concept search (matches even if exact words differ)
python run.py --search="interest rate hike shock transmission to cross border ETFs"

# Dynamic portfolio optimization search
python run.py --search="reinforcement learning policy for portfolio rebalancing"

# Set database backend via environment variable (SQLite or PostgreSQL)
DATABASE_URL="postgresql://user:pass@localhost:5432/hypoq" python run.py --search="order flow imbalance"
```

## Python Integration
```python
from hypoq.db import search_hypotheses

def query_semantic_dedup(query_str: str, top_k: int = 5, min_score: float = 0.3):
    """
    Executes dense semantic vector similarity search against the hypotheses database.
    Returns matches ranked by cosine similarity score (0.0 to 1.0).
    """
    matches = search_hypotheses(query_str, top_k=top_k, min_score=min_score)
    for m in matches:
        print(f"[{m['similarity_score']:.4f}] ID {m['id']} ({m['status']}): {m['title']}")
    return matches
```

## Deduplication Strategy Guidelines
- If a query returns a match with high relevance score and status **`ACCEPTED`**:
  Do NOT rerun the exact same specification. Instead, test an extension (e.g., higher frequency, sub-sample stability, interaction with macro regimes).
- If a query returns a match with status **`REJECTED`**:
  Examine the failure reason (e.g. transaction cost erosion, lack of statistical power). Do not repeat the failed pipeline unless introducing a fundamentally different conditioning mechanism or data source.
