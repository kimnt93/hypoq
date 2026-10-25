#!/usr/bin/env python3
"""
Global Community & Social Financial Discussions Ingestion Pipeline.

Downloads RAW public discussion data from free, key-less sources:
1. Truth Social archive of Donald Trump posts (CNN open data mirror, refreshed continuously)
2. Hacker News (Algolia API) stories on macro, central-bank, tariff, crypto and market topics
3. Stack Exchange (money / quant / economics) top finance questions
4. Federal Reserve monetary-policy press releases (FOMC statements) and speeches / testimony, full text since 2016
5. SEC EDGAR 8-K current reports (filing time, items) for the US mega-caps in data/market/us_equities_daily.parquet
6. Mastodon public hashtag timelines on markets, macro and crypto

STRICT MANDATE:
Only raw text, timestamps and engagement counts are stored. No sentiment labels, polarity scores
or return reactions are computed here: AI agents derive those themselves.
Nothing in this module is hand-written or simulated; every row comes from the HTTP responses.
Re-running the script refreshes the files to the latest available data.

Reddit and X (Twitter) require an authenticated account (www.reddit.com is unreachable anonymously and old.reddit.com
demands a login), so they are intentionally not scraped. SEC EDGAR requires a descriptive User-Agent: set HYPOQ_CONTACT.

All datasets follow the geographic prefix standard `global_*` and are saved under data/community/.
"""

import concurrent.futures
import html
import json
import os
import re
import time
from datetime import datetime, timezone

import polars as pl
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMMUNITY_DIR = os.path.join(DATA_DIR, "community")
os.makedirs(COMMUNITY_DIR, exist_ok=True)

CONTACT = os.environ.get("HYPOQ_CONTACT", "research@hypoq.invalid")  # SEC fair access: set HYPOQ_CONTACT to your own email
HEADERS = {"User-Agent": f"hypoq-research/2.0 ({CONTACT})"}
FED_BASE = "https://www.federalreserve.gov"
SINCE = "2016-01-01"
MASTODON_URL = "https://mastodon.social/api/v1/timelines/tag/{tag}"
MASTODON_TAGS = ["stocks", "economy", "inflation", "bitcoin", "crypto", "markets", "fed", "trading", "investing"]
MASTODON_PAGES = 30
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
    print("\n>>> [1/6] Downloading Truth Social archive (Donald Trump)...")
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
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(pl.col("datetime_utc").str.to_datetime(time_unit="ms", time_zone="UTC").dt.replace_time_zone(None))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .unique(subset="post_id")
        .sort("datetime_utc")
    )
    return _write(df, "global_vip_social_posts_raw.parquet")


def build_hackernews_posts():
    """Hacker News stories matching macro/market keywords, one request window per quarter and query."""
    print("\n>>> [2/6] Downloading Hacker News finance & macro stories...")
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
        pl.DataFrame(list(rows.values()), infer_schema_length=None)
        .with_columns(pl.col("datetime_utc").cast(pl.Datetime("ms")))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .sort("datetime_utc")
    )
    return _write(df, "global_hackernews_finance_posts.parquet")


def build_stackexchange_questions():
    """Most-voted finance questions from Stack Exchange money / quant / economics sites."""
    print("\n>>> [3/6] Downloading Stack Exchange finance questions...")
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
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(pl.col("datetime_utc").cast(pl.Datetime("ms")))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .unique(subset="post_id")
        .sort("datetime_utc")
    )
    return _write(df, "global_stackexchange_finance_questions.parquet")


def _fed_json(name: str):
    try:
        r = requests.get(f"{FED_BASE}/json/{name}", headers=HEADERS, timeout=90)
        r.raise_for_status()
        return json.loads(r.content.decode("utf-8-sig"))
    except (requests.RequestException, ValueError) as exc:
        print(f"    ! Fed feed {name} unavailable: {exc}")
        return None


def _fed_article_text(path: str) -> str:
    for attempt in range(3):
        try:
            r = requests.get(FED_BASE + path, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                node = BeautifulSoup(r.text, "html.parser").select_one("#article")
                return node.get_text("\n", strip=True) if node else ""
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return ""


def build_fed_communications():
    """FOMC statements (monetary-policy press releases) and Fed speeches/testimony with full text."""
    print("\n>>> [4/6] Downloading Federal Reserve statements and speeches...")
    press = _fed_json("ne-press.json") or []
    speeches = _fed_json("ne-speeches.json") or []
    items = []
    for p in press:
        if p.get("pt") == "Monetary Policy":
            items.append(("press_release", p.get("d"), p.get("t"), None, None, p.get("l")))
    for sp in speeches:
        items.append(("speech", sp.get("d"), sp.get("t"), sp.get("s"), sp.get("lo"), sp.get("l")))
    parsed = []
    for kind, d, title, speaker, venue, link in items:
        try:
            ts = datetime.strptime(d, "%m/%d/%Y %I:%M:%S %p")
        except (TypeError, ValueError):
            continue
        if ts.strftime("%Y-%m-%d") >= SINCE and link and link.endswith(".htm"):
            parsed.append((kind, ts, title, speaker, venue, link))
    if not parsed:
        print("    ! Federal Reserve feeds unavailable; skipping.")
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        texts = list(ex.map(lambda x: _fed_article_text(x[5]), parsed))
    rows = [
        {
            "datetime_utc": ts, "doc_type": kind, "title": html.unescape(title or ""), "speaker": html.unescape(speaker or "") or None,
            "venue": html.unescape(venue or "") or None,
            "url": FED_BASE + link, "text": text,
        }
        for (kind, ts, title, speaker, venue, link), text in zip(parsed, texts)
        if text
    ]
    df = (
        pl.DataFrame(rows, infer_schema_length=None)
        # the Fed feed publishes US/Eastern wall-clock times: convert to naive UTC like every other dataset
        .with_columns(pl.col("datetime_utc").cast(pl.Datetime("ms")).dt.replace_time_zone("America/New_York", ambiguous="earliest")
                      .dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias("datetime_utc"))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .unique(subset="url")
        .sort("datetime_utc")
    )
    return _write(df, "us_fed_communications.parquet")


def build_sec_8k_filings():
    """SEC EDGAR 8-K current reports (event timestamps and item codes) for the US equities in the data lake."""
    print("\n>>> [5/6] Downloading SEC EDGAR 8-K filings...")
    eq_path = os.path.join(DATA_DIR, "market", "us_equities_daily.parquet")
    if not os.path.exists(eq_path):
        print("    ! data/market/us_equities_daily.parquet missing; skipping.")
        return None
    symbols = pl.read_parquet(eq_path)["symbol"].unique().to_list()
    tickers = _get_json("https://www.sec.gov/files/company_tickers.json")
    if not tickers:
        return None
    cik_by_ticker = {v["ticker"]: int(v["cik_str"]) for v in tickers.values()}
    rows = []
    for sym in sorted(symbols):
        cik = cik_by_ticker.get(sym.replace("-", "-"))
        if cik is None:
            print(f"    ! no CIK for {sym}")
            continue
        sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if not sub:
            continue
        blocks = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            if extra.get("filingTo", "9999") >= SINCE:
                more = _get_json(f"https://data.sec.gov/submissions/{extra['name']}")
                if more:
                    blocks.append(more)
        for blk in blocks:
            for i, form in enumerate(blk["form"]):
                if form not in ("8-K", "8-K/A") or blk["filingDate"][i] < SINCE:
                    continue
                acc = blk["accessionNumber"][i]
                rows.append(
                    {
                        "symbol": sym, "cik": cik, "form": form, "accession_number": acc,
                        "filing_date": blk["filingDate"][i], "accepted_utc": blk["acceptanceDateTime"][i],
                        "report_date": blk["reportDate"][i] or None, "items": blk["items"][i],
                        "document_description": blk["primaryDocDescription"][i],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{blk['primaryDocument'][i]}",
                    }
                )
        print(f"    - {sym}: {sum(1 for r in rows if r['symbol'] == sym)} 8-K filings")
        time.sleep(0.2)
    if not rows:
        return None
    df = (
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            pl.col("accepted_utc").str.to_datetime(strict=False, time_zone="UTC").dt.replace_time_zone(None).cast(pl.Datetime("ms")).alias("datetime_utc"),
            pl.col("filing_date").str.to_date("%Y-%m-%d").alias("date"),
            pl.col("report_date").str.to_date("%Y-%m-%d", strict=False),
        )
        .drop(["accepted_utc", "filing_date"])
        .unique(subset="accession_number")
        .sort(["datetime_utc", "symbol"])
    )
    return _write(df, "us_sec_8k_filings.parquet")


def build_mastodon_posts():
    """Public Mastodon hashtag timelines (mastodon.social): raw posts with engagement counts."""
    print("\n>>> [6/6] Downloading Mastodon public hashtag timelines...")
    rows = {}
    for tag in MASTODON_TAGS:
        max_id = None
        for _ in range(MASTODON_PAGES):
            params = {"limit": 40}
            if max_id:
                params["max_id"] = max_id
            page = _get_json(MASTODON_URL.format(tag=tag), params)
            if not page:
                break
            for p in page:
                if p["id"] in rows:
                    rows[p["id"]]["tags_matched"].add(tag)
                    continue
                rows[p["id"]] = {
                    "datetime_utc": p["created_at"], "post_id": p["id"], "author": p["account"]["acct"],
                    "language": p.get("language"), "text": _clean_text(p.get("content")), "url": p.get("url"),
                    "replies_count": int(p.get("replies_count") or 0), "reposts_count": int(p.get("reblogs_count") or 0),
                    "likes_count": int(p.get("favourites_count") or 0), "tags_matched": {tag},
                    "hashtags": ",".join(t["name"] for t in p.get("tags", [])),
                }
            max_id = page[-1]["id"]
            time.sleep(0.3)
        print(f"    - #{tag} done ({len(rows):,} unique posts so far)")
    if not rows:
        return None
    for r in rows.values():
        r["tags_matched"] = ",".join(sorted(r["tags_matched"]))
    df = (
        pl.DataFrame(list(rows.values()), infer_schema_length=None)
        .with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None).cast(pl.Datetime("ms")))
        .with_columns(pl.col("datetime_utc").dt.date().alias("date"))
        .filter(pl.col("text").str.len_chars() > 0)
        .sort("datetime_utc")
    )
    return _write(df, "global_mastodon_finance_posts.parquet")


def run_all():
    build_vip_social_posts_raw()
    build_hackernews_posts()
    build_stackexchange_questions()
    build_fed_communications()
    build_sec_8k_filings()
    build_mastodon_posts()


if __name__ == "__main__":
    run_all()
