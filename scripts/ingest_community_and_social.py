#!/usr/bin/env python3
"""
Global Community & Social Financial Discussions Ingestion Pipeline.

Downloads RAW public discussion data from free, key-less sources:
1. Truth Social archive of Donald Trump posts (CNN open data mirror, refreshed continuously)
2. Hacker News (Algolia API) stories on macro, central-bank, tariff, crypto and market topics
3. Stack Exchange (money / quant / economics) top finance questions

STRICT MANDATE:
Only raw text, timestamps and engagement counts are stored. No sentiment labels, polarity scores
or return reactions are computed here: AI agents derive those themselves.
Nothing in this module is hand-written or simulated; every row comes from the HTTP responses.
Re-running the script refreshes the files to the latest available data.

Reddit and X (Twitter) block anonymous API access, so they are intentionally not scraped.

All datasets follow the geographic prefix standard `global_*` and are saved under data/community/.
"""

import html
import os
import re
import time
from datetime import datetime, timezone

import polars as pl
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMMUNITY_DIR = os.path.join(DATA_DIR, "community")
os.makedirs(COMMUNITY_DIR, exist_ok=True)

HEADERS = {"User-Agent": "hypoq-research/2.0 (open data ingestion)"}
TRUTH_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"
HN_URL = "https://hn.algolia.com/api/v1/search_by_date"
SE_URL = "https://api.stackexchange.com/2.3/questions"

HN_QUERIES = [
    "federal reserve", "inflation", "interest rates", "recession", "tariffs",
    "bitcoin", "stock market", "treasury yields", "oil price", "bank run",
]
HN_START_YEAR = 2015
HN_MIN_POINTS = 10
SE_SITES = ["money", "quant", "economics"]
SE_PAGES_PER_SITE = 15

_TAG_RE = re.compile(r"<[^>]+>")


def _get_json(url: str, params: dict = None, retries: int = 4):
    """GET JSON with exponential back-off; returns None when all attempts fail."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"    ! HTTP {r.status_code} from {url}")
        except (requests.RequestException, ValueError) as exc:
            print(f"    ! {type(exc).__name__}: {exc}")
        time.sleep(2 ** attempt)
    return None


def _clean_text(raw: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", raw or "")).strip()


def _write(df: pl.DataFrame, name: str) -> str:
    out = os.path.join(COMMUNITY_DIR, name)
    df.write_parquet(out, compression="zstd")
    print(f"    - Saved {os.path.relpath(out, BASE_DIR)} ({len(df):,} rows x {len(df.columns)} cols)")
    return out


def build_vip_social_posts_raw():
    """Donald Trump's Truth Social posts: raw text, UTC timestamp, replies, reblogs, favourites."""
    print("\n>>> [1/3] Downloading Truth Social archive (Donald Trump)...")
    data = _get_json(TRUTH_ARCHIVE_URL)
    if not data:
        print("    ! Truth Social archive unavailable; skipping.")
        return None
    rows = [
        {
            "datetime_utc": p["created_at"],
            "author": "donald_trump",
            "platform": "truth_social",
            "post_id": str(p["id"]),
            "text": _clean_text(p.get("content")),
            "url": p.get("url"),
            "has_media": bool(p.get("media")),
            "replies_count": int(p.get("replies_count") or 0),
            "reposts_count": int(p.get("reblogs_count") or 0),
            "likes_count": int(p.get("favourites_count") or 0),
        }
        for p in data
    ]
    df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("datetime_utc").str.to_datetime(time_unit="ms", time_zone="UTC").dt.replace_time_zone(None))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .unique(subset="post_id")
        .sort("datetime_utc")
    )
    return _write(df, "global_vip_social_posts_raw.parquet")


def build_hackernews_posts():
    """Hacker News stories matching macro/market keywords, one request window per quarter and query."""
    print("\n>>> [2/3] Downloading Hacker News finance & macro stories...")
    now = datetime.now(timezone.utc)
    windows = []
    for year in range(HN_START_YEAR, now.year + 1):
        for q in range(4):
            start = datetime(year, 3 * q + 1, 1, tzinfo=timezone.utc)
            end = datetime(year + (q == 3), (3 * q + 4 - 1) % 12 + 1, 1, tzinfo=timezone.utc)
            if start > now:
                break
            windows.append((int(start.timestamp()), int(end.timestamp())))

    rows = {}
    for query in HN_QUERIES:
        for lo, hi in windows:
            payload = _get_json(
                HN_URL,
                {
                    "query": query,
                    "tags": "story",
                    "numericFilters": f"created_at_i>={lo},created_at_i<{hi},points>={HN_MIN_POINTS}",
                    "hitsPerPage": 100,
                },
            )
            for hit in (payload or {}).get("hits", []):
                oid = hit["objectID"]
                if oid in rows:
                    rows[oid]["matched_queries"].add(query)
                    continue
                rows[oid] = {
                    "datetime_utc": datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc).replace(tzinfo=None),
                    "post_id": oid,
                    "author": hit.get("author"),
                    "title": hit.get("title"),
                    "body_text": _clean_text(hit.get("story_text")),
                    "url": hit.get("url"),
                    "points": int(hit.get("points") or 0),
                    "num_comments": int(hit.get("num_comments") or 0),
                    "matched_queries": {query},
                }
            time.sleep(0.15)
        print(f"    - query '{query}' done ({len(rows):,} unique stories so far)")
    if not rows:
        print("    ! Hacker News unavailable; skipping.")
        return None
    for r in rows.values():
        r["matched_queries"] = ",".join(sorted(r["matched_queries"]))
    df = (
        pl.DataFrame(list(rows.values()))
        .with_columns(pl.col("datetime_utc").cast(pl.Datetime("ms")))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .sort("datetime_utc")
    )
    return _write(df, "global_hackernews_finance_posts.parquet")


def build_stackexchange_questions():
    """Most-voted finance questions from Stack Exchange money / quant / economics sites."""
    print("\n>>> [3/3] Downloading Stack Exchange finance questions...")
    rows = []
    for site in SE_SITES:
        for page in range(1, SE_PAGES_PER_SITE + 1):
            payload = _get_json(
                SE_URL,
                {
                    "site": site, "page": page, "pagesize": 100,
                    "order": "desc", "sort": "votes", "filter": "withbody",
                },
            )
            if not payload:
                break
            for q in payload.get("items", []):
                rows.append(
                    {
                        "datetime_utc": datetime.fromtimestamp(q["creation_date"], tz=timezone.utc).replace(tzinfo=None),
                        "site": site,
                        "post_id": f"{site}_{q['question_id']}",
                        "title": html.unescape(q.get("title", "")),
                        "body_text": _clean_text(q.get("body")),
                        "tags": ",".join(q.get("tags", [])),
                        "score": int(q.get("score") or 0),
                        "view_count": int(q.get("view_count") or 0),
                        "answer_count": int(q.get("answer_count") or 0),
                        "url": q.get("link"),
                    }
                )
            if not payload.get("has_more"):
                break
            time.sleep(max(float(payload.get("backoff", 0)), 0.3))
        print(f"    - site '{site}' done ({len(rows):,} questions so far)")
    if not rows:
        print("    ! Stack Exchange unavailable; skipping.")
        return None
    df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("datetime_utc").cast(pl.Datetime("ms")))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .unique(subset="post_id")
        .sort("datetime_utc")
    )
    return _write(df, "global_stackexchange_finance_questions.parquet")


def run_all():
    build_vip_social_posts_raw()
    build_hackernews_posts()
    build_stackexchange_questions()


if __name__ == "__main__":
    run_all()
