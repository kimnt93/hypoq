#!/usr/bin/env python3
"""
Keep DATA.md numerically in sync with data/:
- regenerates the directory inventory tree (rows x cols per file, keeping each file's description)
- refreshes the file/row totals in the headings
- refreshes the date range, record count and column count in every `* **Coverage**:` bullet that belongs to a dataset section
Descriptions and field tables stay hand-written; run `python scripts/update_data_docs.py` after `make download`.
"""
import glob
import os
import re

import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(BASE_DIR, "DATA.md")
FOLDER_ORDER = ["macro", "market", "community", "crypto", "market_structure", "financials", "cross_asset"]
NEW_DESCRIPTIONS = {
    "vn_cpi_inflation_annual.parquet": "World Bank annual CPI inflation for Vietnam (reference-period dated)",
    "us_fed_communications.parquet": "FOMC statements, minutes and Fed speeches/testimony, full text, UTC release times",
    "us_sec_8k_filings.parquet": "SEC EDGAR 8-K filings of the US mega-caps: acceptance time, item codes, document link",
    "global_mastodon_finance_posts.parquet": "Public Mastodon posts on markets, macro and crypto hashtags (raw text, engagement)",
    "us_fed_fomc_rate_decisions_pit.parquet": "Point-in-time FOMC decisions: announcement time, target range before/after, change in bp",
}


def main():
    text = open(DOC, encoding="utf-8").read()
    files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "*", "*.parquet")))
    desc = {}
    for m in re.finditer(r"([a-z0-9_]+\.parquet)\s+#\s*[\d,]+ rows x \d+ cols \| (.*)$", text, re.M):
        desc[m.group(1)] = m.group(2).strip()
    desc.update({k: v for k, v in NEW_DESCRIPTIONS.items() if k not in desc})

    info, by_folder, total = {}, {}, 0
    for f in files:
        lf = pl.scan_parquet(f)
        n = lf.select(pl.len()).collect().item()
        cols = lf.collect_schema()
        dmin = dmax = None
        if "date" in cols and cols["date"] in (pl.Date, pl.Datetime):
            r = lf.select(pl.col("date").min().alias("a"), pl.col("date").max().alias("b")).collect().row(0)
            dmin, dmax = [str(x)[:10] for x in r]
        name = os.path.basename(f)
        info[name] = (n, len(cols), dmin, dmax)
        by_folder.setdefault(f.split(os.sep)[-2], []).append((name, n, len(cols)))
        total += n

    lines = ["data/"]
    folders = [x for x in FOLDER_ORDER if x in by_folder] + sorted(set(by_folder) - set(FOLDER_ORDER))
    for i, folder in enumerate(folders):
        last = i == len(folders) - 1
        lines.append(("└── " if last else "├── ") + folder + "/")
        for j, (n, r, c) in enumerate(by_folder[folder]):
            pre = "    " if last else "│   "
            lines.append(f"{pre}{'└── ' if j == len(by_folder[folder]) - 1 else '├── '}{n:<44} # {r:,} rows x {c} cols | {desc.get(n, '')}")
    text = re.sub(r"```text\ndata/.*?```", "```text\n" + "\n".join(lines) + "\n```", text, count=1, flags=re.S)
    text = re.sub(r"\(\d+ Parquet Files, [\d,]+ Rows\)", f"({len(files)} Parquet Files, {total:,} Rows)", text)
    text = re.sub(r"all \d+ empirical datasets", f"all {len(files)} empirical datasets", text)

    # per-dataset coverage bullets
    parts = re.split(r"(?m)^(?=#### )", text)
    out = []
    for part in parts:
        m = re.match(r"#### \w\. `([a-z0-9_]+\.parquet)`", part)
        if m and m.group(1) in info:
            n, c, dmin, dmax = info[m.group(1)]
            def fix(line):
                if dmin:
                    line = re.sub(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", f"{dmin} to {dmax}", line, count=1)
                line = re.sub(r"\(([\d,]+)( [^,)]*)", lambda mm: f"({n:,}{mm.group(2)}", line, count=1)
                line = re.sub(r"(\d+) columns", f"{c} columns", line, count=1)
                return line
            part = re.sub(r"(?m)^\* \*\*Coverage\*\*:.*$", lambda mm: fix(mm.group(0)), part, count=1)
        out.append(part)
    open(DOC, "w", encoding="utf-8").write("".join(out))

    # headline counts in README.md and AGENTS.md ("70 Parquet files ... 3.1M rows")
    millions = f"{total / 1e6:.1f}M"
    for name in ("README.md", "AGENTS.md"):
        path = os.path.join(BASE_DIR, name)
        t = open(path, encoding="utf-8").read()
        t = re.sub(r"\d+ Parquet files", f"{len(files)} Parquet files", t)
        t = re.sub(r"(\d+ Parquet files(?: under `data/`)?,? \()?[\d.]+M\+? rows", lambda m: (m.group(1) or "") + f"{millions} rows", t)
        open(path, "w", encoding="utf-8").write(t)
    print(f"DATA.md synchronised: {len(files)} files, {total:,} rows")


if __name__ == "__main__":
    main()
