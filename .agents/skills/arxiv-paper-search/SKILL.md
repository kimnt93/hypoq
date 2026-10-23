---
name: arxiv-paper-search
description: Search academic literature via the arXiv API, fetch abstracts, authors, and BibTeX citations to position research hypotheses against existing literature.
version: 1.0.0
author: kim / vnresearch
tags: [arxiv, literature, academic, papers, citations, bibtex]
---

# arXiv Paper Search Skill

## Overview
This skill provides automated access to the arXiv API and academic discovery protocols, enabling agents to search for prior literature, examine theoretical models, locate benchmark equations, and pull clean BibTeX references.

## When to Use
- When formulating an initial hypothesis to establish whether the mechanism has been studied.
- When finding foundational papers to cite in `references.bib` and Section 1 of the report.
- When retrieving baseline econometric specifications or machine learning architectures.

## Python arXiv Search Helper
Use standard library `urllib` and `xml.etree.ElementTree` (zero external dependencies) to query arXiv:

```python
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict

def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Searches arXiv API for papers matching query in quantitative finance or economics.
    """
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = base_url + urllib.parse.urlencode(params)
    
    req = urllib.request.Request(url, headers={"User-Agent": "AutoResearch/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_data = resp.read()
        
    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text[:10]
        entry_id = entry.find("atom:id", ns).text
        
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        
        results.append({
            "title": title,
            "authors": authors,
            "published": published,
            "arxiv_id": entry_id.split("/")[-1],
            "url": entry_id,
            "summary": summary[:400] + "..."
        })
    return results

def format_bibtex(paper: Dict) -> str:
    """Formats an arXiv paper entry into a BibTeX record."""
    first_author_surname = paper["authors"][0].split()[-1].lower() if paper["authors"] else "unknown"
    year = paper["published"][:4]
    key = f"{first_author_surname}{year}{paper['arxiv_id'].split('v')[0].replace('.', '')}"
    authors_str = " and ".join(paper["authors"])
    
    return f"""@article{{{key},
  title={{{paper['title']}}},
  author={{{authors_str}}},
  journal={{arXiv preprint arXiv:{paper['arxiv_id']}}},
  year={{{year}}},
  url={{{paper['url']}}}
}}"""
```
