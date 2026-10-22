#!/usr/bin/env python3
"""
anymd Document Parser Wrapper (SylphxAI/anymd).
Provides unified helpers to convert and parse PDFs, Word docs, spreadsheets,
and web documents into clean Markdown for data processing pipelines.
"""

import os
import subprocess
import shutil

ANYMD_BIN = shutil.which("anymd") or "/home/kim/.bun/bin/anymd"

def parse_to_markdown(file_or_url, pages=None, max_tokens=None, output_path=None):
    """
    Parses a local file (PDF, DOCX, XLSX, HTML) or web URL into clean Markdown using anymd.
    """
    cmd = [ANYMD_BIN, str(file_or_url)]
    if pages:
        cmd.extend(["-p", str(pages)])
    if max_tokens:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if output_path:
        cmd.extend(["-o", str(output_path)])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error parsing {file_or_url} with anymd: {e.stderr.strip()}")
        return ""
    except Exception as e:
        print(f"Unexpected error running anymd on {file_or_url}: {e}")
        return ""


def search_document(query, file_or_dir):
    """
    Searches across documents using anymd's native BM25/ranked search.
    """
    cmd = [ANYMD_BIN, "search", str(query), str(file_or_dir)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except Exception as e:
        print(f"Error searching with anymd: {e}")
        return ""


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if target == "--help":
        print("Usage: python3 anymd_parser.py <file_or_url> [pages]")
    else:
        pgs = sys.argv[2] if len(sys.argv) > 2 else None
        print(parse_to_markdown(target, pages=pgs))
