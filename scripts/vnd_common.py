#!/usr/bin/env python3
"""
VNDirect finfo API helper that never silently truncates.

The finfo API caps rows per call at `size`; a large company has ~16,000 financial-statement rows, ~3x the size used by the
original scripts, so history was cut off. The server accepts very large pages, so `vnd_get_data` requests everything in one call
(size=50000), then compares the number of rows received with the `totalElements` the server reports and warns on any mismatch.
(Page 2+ responses omit `totalPages`, and unsorted pages are not stable across calls, so paging is deliberately avoided.)
"""
import re
import time
from typing import List, Optional

import requests

PAGE_SIZE = 50000


def vnd_get_data(url: str, headers: dict, timeout: int = 60, retries: int = 3) -> Optional[List[dict]]:
    """All rows of a finfo query, or None when it failed on every attempt and page size."""
    m = re.search(r"size=(\d+)", url)
    original = int(m.group(1)) if m else None
    sizes = [PAGE_SIZE, 10000] + ([original] if original else [])
    for size in dict.fromkeys(sizes):  # largest first; some endpoints answer HTTP 500 to oversized pages
        sized = re.sub(r"size=\d+", f"size={size}", url) if m else f"{url}{'&' if '?' in url else '?'}size={size}"
        for attempt in range(retries):
            try:
                r = requests.get(sized, headers=headers, timeout=max(timeout, 60))
                if r.status_code == 200:
                    payload = r.json()
                    rows = payload.get("data", [])
                    total = payload.get("totalElements")
                    if total is not None and total != len(rows):
                        print(f"      ! WARNING: {url.split('?')[0]} returned {len(rows):,} of {total:,} rows (truncated at size={size})")
                    return rows
                if r.status_code >= 500 and attempt == 0:
                    break  # oversized page rejected: try the next, smaller size immediately
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2 ** attempt)
    return None
