#!/usr/bin/env python3
"""
res_017 (Return decomposition): do US mega-caps earn their return overnight rather than during the trading day?

For each stock-day, with the open adjusted by the same dividend/split factor as the close (factor = adj_close / close):
  overnight_t = open_adj_t / adj_close_{t-1} - 1,   intraday_t = adj_close_t / open_adj_t - 1.
Daily cross-sectional equal-weight averages give two series. Statistic: mean(overnight - intraday). H0: <= 0. H1: > 0 (one-sided, recentred stationary bootstrap, block 10).
"""
import os
import sys
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

# --- metric helpers copied from .agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py (hypoq/ is frozen; each study is self-contained) ---

def _stationary_indices(rng: np.random.Generator, n: int, block: float, n_boot: int) -> np.ndarray:
    """(n_boot, n) index matrix of Politis-Romano stationary-bootstrap resamples with mean block length `block`."""
    restart = rng.random((n_boot, n)) < 1.0 / block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for k in range(1, n):
        idx[:, k] = np.where(restart[:, k], starts[:, k], (idx[:, k - 1] + 1) % n)
    return idx

def stationary_bootstrap_mean_pvalue(x: np.ndarray, n_boot: int = 5000, block: float = 10.0, seed: int = 11) -> float:
    """One-sided p-value for H1: mean(x) > 0, recentred stationary bootstrap."""
    idx = _stationary_indices(np.random.default_rng(seed), len(x), block, n_boot)
    means = x[idx].mean(axis=1)
    return float((np.sum(means - means.mean() >= x.mean()) + 1) / (n_boot + 1))

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Overnight versus Intraday Returns of US Mega-Cap Stocks"


def run():
    t0 = time.time()
    d = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_equities_daily.parquet")).select(["date", "symbol", "open", "close", "adj_close"])
         .drop_nulls().filter((pl.col("open") > 0) & (pl.col("close") > 0)).sort(["symbol", "date"]))
    d = d.with_columns((pl.col("open") * pl.col("adj_close") / pl.col("close")).alias("open_adj"))
    d = d.with_columns((pl.col("open_adj") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("on"), (pl.col("adj_close") / pl.col("open_adj") - 1).alias("intra")).drop_nulls(["on", "intra"])
    n_raw = d.height
    d = d.filter((pl.col("on").abs() < 0.5) & (pl.col("intra").abs() < 0.5))  # drops vendor errors; real moves beyond 50% in one half-day are not in this universe
    n_dropped = n_raw - d.height
    daily = d.group_by("date").agg(pl.col("on").mean().alias("on"), pl.col("intra").mean().alias("intra"), pl.len().alias("n")).sort("date")
    on, intra = daily["on"].to_numpy(), daily["intra"].to_numpy()
    diff = on - intra
    p = stationary_bootstrap_mean_pvalue(diff, n_boot=5000, block=10, seed=11)
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"
    ann = lambda x: float(np.prod(1 + x) ** (252 / len(x)) - 1)
    by_year = (daily.with_columns(pl.col("date").dt.year().alias("year")).group_by("year").agg(pl.col("on").mean().alias("on"), pl.col("intra").mean().alias("intra")).sort("year"))

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.plot(daily["date"].to_list(), np.cumprod(1 + on), label=f"overnight ({ann(on)*100:.1f}% p.a.)", color="#1f77b4")
    ax.plot(daily["date"].to_list(), np.cumprod(1 + intra), label=f"intraday ({ann(intra)*100:.1f}% p.a.)", color="#d62728")
    ax.set_ylabel("growth of 1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_overnight_vs_intraday.png"))
    plt.close()

    rws = "\n".join(f"{int(y)} & {a*100:.3f}\\% & {b*100:.3f}\\% \\\\" for y, a, b in by_year.iter_rows())
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Mean daily overnight and intraday return of the equal-weight portfolio}\n\\begin{tabular}{lcc}\n\\toprule\n"
             "Year & Overnight & Intraday \\\\\n\\midrule\n" + rws + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_overnight_intraday_by_year.tex"), "w") as f:
        f.write(table)
    sample = f"{daily['date'].min()} to {daily['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the mean overnight return does not exceed the mean intraday return; $H_1$: it does",
        "Data": f"{len(daily):,} sessions of {d['symbol'].n_unique()} US mega-caps, {sample}; {n_dropped} stock-days with a half-day move beyond 50\\% dropped as data errors",
        "Method": "Open adjusted by the close adjustment factor; equal-weight daily averages; one-sided recentred stationary bootstrap of the mean difference (5,000 draws, block 10)",
        "Results": f"Mean overnight {on.mean()*1e4:.2f} bp versus intraday {intra.mean()*1e4:.2f} bp per day (difference {diff.mean()*1e4:.2f} bp, $p={p:.4f}$); annualised {ann(on)*100:.1f}\\% versus {ann(intra)*100:.1f}\\%.",
    }
    body = ("\\section{Design}\nThe decomposition follows \\citet{cliff2008overnight}: the open is scaled by the dividend and split factor of the same day so that "
            "the overnight and intraday legs multiply to the adjusted close-to-close return. Equal weights across the available stocks give two daily series whose mean difference is tested.\n\n"
            "\\section{Results}\n\\input{results/table1_overnight_intraday_by_year.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_overnight_vs_intraday.png}\n\\caption{Growth of one unit earned only overnight and only intraday.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nOpening prices on Yahoo Finance are official opening auctions; they exclude pre-market trades, so the overnight leg is the return to the opening auction, which is "
            "tradable only with market-on-open orders. The universe is today's mega-caps (survivorship) and costs are not modelled.\n")
    bib = ("@article{cliff2008overnight,\n  title={Overnight and intraday returns},\n  author={Cliff, Michael and Cooper, Michael J and Gulen, Huseyin},\n  journal={Working paper, Virginia Tech},\n  year={2008}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_sessions": len(daily), "n_stock_days_dropped": n_dropped, "sample": sample, "mean_overnight": float(on.mean()), "mean_intraday": float(intra.mean()),
         "mean_difference": float(diff.mean()), "p_one_sided": p, "ann_overnight": ann(on), "ann_intraday": ann(intra), "by_year": by_year.to_dicts()},
        abstract, body, bib,
        statement="US mega-cap stocks earn a higher average return overnight than during the trading day.",
        null_hypothesis="Mean overnight return minus mean intraday return of the equal-weight portfolio is not positive.",
        alt_hypothesis="The mean difference is positive (one-sided stationary-bootstrap p < 0.05).",
        domain="market_microstructure", datasets=["data/market/us_equities_daily.parquet"],
        methodology="Overnight/intraday decomposition with adjusted opens and a stationary-bootstrap mean test",
        metric_name="mean overnight minus intraday daily return", metric_value=float(diff.mean()), p_value=p,
        summary_findings=f"Overnight {on.mean()*1e4:.2f} bp vs intraday {intra.mean()*1e4:.2f} bp per day (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
