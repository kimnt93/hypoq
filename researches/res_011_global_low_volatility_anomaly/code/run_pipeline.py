#!/usr/bin/env python3
"""
res_011 (Cross-sectional sort): is the low-volatility anomaly present in a pooled universe of large caps from US, Europe, Japan, HK, Korea, China and Asia ex-Japan?

Each month-end t every stock with 252 observed daily returns is ranked on its trailing 252-day return volatility (information up to t).
Quintile 1 (lowest volatility) and quintile 5 (highest) are held equal-weighted over month t+1 in local-currency returns.
H0: Sharpe(Q1) <= Sharpe(Q5). H1: Sharpe(Q1) > Sharpe(Q5): one-sided stationary-bootstrap p < 0.05 (monthly data, mean block 3).
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
from scipy import stats as sps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "The Low-Volatility Anomaly in a Pooled Universe of Large-Cap Equities from Seven Regions"
REGIONS = ["us", "eu", "jp", "hk", "kr", "cn", "asia"]
WINDOW, MIN_STOCKS = 252, 40


# --- inference helpers (self-contained: the hypoq package is frozen; see the hypothesis-testing-stats skill) ---

def sharpe(x: np.ndarray, periods: int) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))

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

def stationary_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, periods: int = 252, n_boot: int = 5000,
                                     block: float = 10.0, seed: int = 11) -> Tuple[float, float]:
    """One-sided p-value for H1: Sharpe(a) > Sharpe(b), recentred under H0. Returns (annualised difference, p)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = _stationary_indices(rng, n, block, n_boot)
    sh = lambda m: m.mean(axis=1) / m.std(axis=1, ddof=1)
    obs = a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)
    diffs = sh(a[idx]) - sh(b[idx])
    p = float((np.sum(diffs - diffs.mean() >= obs) + 1) / (n_boot + 1))
    return float(obs * np.sqrt(periods)), p

def monthly_panel() -> pl.DataFrame:
    frames = []
    for reg in REGIONS:
        d = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", f"{reg}_equities_daily.parquet")).select(["date", "symbol", "returns"]).drop_nulls("returns")
        frames.append(d.with_columns(pl.lit(reg).alias("region")))
    # four tickers are listed in two regional files: keep one copy so no stock is counted twice
    d = pl.concat(frames).unique(subset=["symbol", "date"], keep="first", maintain_order=True).sort(["symbol", "date"])
    d = d.with_columns(pl.col("returns").rolling_std(WINDOW).over("symbol").alias("vol"),
                       pl.col("returns").log1p().alias("lr"),
                       (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("ym"))
    m = (d.group_by(["symbol", "region", "ym"]).agg(pl.col("vol").last().alias("vol"), pl.col("lr").sum().alias("lr"), pl.col("date").max().alias("last_date"))
         .sort(["symbol", "ym"]))
    m = m.with_columns(pl.col("ym").shift(-1).over("symbol").alias("ym_next"), (pl.col("lr").shift(-1).over("symbol")).alias("lr_next"))
    # keep only truly consecutive months (a skipped month would pair vol with a later return)
    def nxt(ym): return ym + 1 if ym % 100 < 12 else (ym // 100 + 1) * 100 + 1
    m = m.with_columns(pl.col("ym").map_elements(nxt, return_dtype=pl.Int64).alias("ym_expected"))
    m = m.filter((pl.col("ym_next") == pl.col("ym_expected")) & pl.col("vol").is_not_null() & pl.col("lr_next").is_not_null())
    return m.with_columns((pl.col("lr_next").exp() - 1).alias("ret_next"))


def run():
    t0 = time.time()
    m = monthly_panel()
    rows = []
    for ym, g in m.group_by("ym", maintain_order=True):
        if g.height < MIN_STOCKS:
            continue
        g = g.with_columns((pl.col("vol").rank("ordinal") - 1).alias("rk"))
        q = (g["rk"] * 5 // g.height).to_numpy()
        r = g["ret_next"].to_numpy()
        rows.append((ym[0], r[q == 0].mean(), r[q == 4].mean(), r.mean(), g.height))
    rows.sort()
    ym = [r[0] for r in rows]
    q1, q5, mkt = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    n_stocks_med = int(np.median([r[4] for r in rows]))
    diff, p = stationary_bootstrap_sharpe_diff(q1, q5, periods=12, n_boot=5000, block=3, seed=11)
    spread = q1 - q5
    t_spread = float(sps.ttest_1samp(spread, 0.0).statistic)
    mean_spread = float(spread.mean())
    pf = lambda x: {"ann_return": float(np.prod(1 + x) ** (12 / len(x)) - 1), "ann_vol": float(x.std(ddof=1) * np.sqrt(12)), "sharpe": sharpe(x, 12),
                    "max_drawdown": float((np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min())}
    perf = {"Q1 low vol": pf(q1), "Q5 high vol": pf(q5), "All stocks": pf(mkt)}
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    for lab, x, c in [("Q1 low vol", q1, "#1f77b4"), ("Q5 high vol", q5, "#d62728"), ("All stocks", mkt, "#7f7f7f")]:
        ax.plot(range(len(x)), np.cumprod(1 + x), label=f"{lab} (Sharpe {perf[lab]['sharpe']:.2f})", color=c)
    ax.set_xticks(range(0, len(ym), 24))
    ax.set_xticklabels([f"{y // 100}" for y in ym[::24]])
    ax.set_ylabel("growth of 1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_low_vol_quintiles.png"))
    plt.close()

    rws = "\n".join(f"{k} & {v['ann_return']*100:.1f}\\% & {v['ann_vol']*100:.1f}\\% & {v['sharpe']:.2f} & {v['max_drawdown']*100:.1f}\\% \\\\" for k, v in perf.items())
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Equal-weight volatility quintiles, monthly rebalanced, local-currency returns}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Portfolio & Return & Vol & Sharpe & Max DD \\\\\n\\midrule\n" + rws + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_quintiles.tex"), "w") as f:
        f.write(table)
    sample = f"{ym[0] // 100}-{ym[0] % 100:02d} to {ym[-1] // 100}-{ym[-1] % 100:02d} (formation months)"
    abstract = {
        "Hypothesis": "$H_0$: the Sharpe ratio of the lowest-volatility quintile does not exceed that of the highest-volatility quintile; $H_1$: it does",
        "Data": f"{len(rows)} monthly cross-sections, median {n_stocks_med} stocks from {len(REGIONS)} regions, {sample}",
        "Method": "Month-end sort on trailing 252-day volatility, equal-weight quintiles, one-sided stationary-bootstrap test of the Sharpe difference (5,000 draws, block 3)",
        "Results": f"Sharpe {perf['Q1 low vol']['sharpe']:.2f} versus {perf['Q5 high vol']['sharpe']:.2f} (difference {diff:.2f}, one-sided $p={p:.4f}$); "
                   f"mean monthly Q1--Q5 spread {mean_spread*100:.2f}\\% ($t={t_spread:.2f}$).",
    }
    body = ("\\section{Design}\nThe sort variable uses returns up to the month-end of formation and portfolios are held over the following calendar month "
            "\\citep{baker2011benchmarks,ang2006cross}. Stocks enter only after 252 observed returns, and months without a consecutive successor are discarded rather than interpolated. "
            "Returns are in local currency and are not converted, so cross-region currency moves are not part of the test.\n\n"
            "\\section{Results}\n\\input{results/table1_quintiles.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_low_vol_quintiles.png}\n\\caption{Growth of one unit in each volatility quintile.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe universe is a fixed list of today's large caps, so survivorship bias is present and probably flatters high-volatility winners. "
            "There are about 130 stocks, so a quintile holds roughly 25 names; the result need not extend to small caps or to transaction-cost-aware implementations.\n")
    bib = ("@article{baker2011benchmarks,\n  title={Benchmarks as limits to arbitrage: Understanding the low-volatility anomaly},\n  author={Baker, Malcolm and Bradley, Brendan and Wurgler, Jeffrey},\n"
           "  journal={Financial Analysts Journal},\n  volume={67},\n  number={1},\n  pages={40--54},\n  year={2011}\n}\n"
           "@article{ang2006cross,\n  title={The cross-section of volatility and expected returns},\n  author={Ang, Andrew and Hodrick, Robert J and Xing, Yuhang and Zhang, Xiaoyan},\n"
           "  journal={The Journal of Finance},\n  volume={61},\n  number={1},\n  pages={259--299},\n  year={2006}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_months": len(rows), "median_stocks": n_stocks_med, "sample": sample, "performance": perf, "sharpe_difference": diff, "bootstrap_p_one_sided": p,
         "mean_monthly_spread_q1_minus_q5": mean_spread, "spread_t": t_spread},
        abstract, body, bib,
        statement="Low-volatility large-cap stocks earn a higher risk-adjusted return than high-volatility large-cap stocks.",
        null_hypothesis="Sharpe(lowest-volatility quintile) <= Sharpe(highest-volatility quintile).",
        alt_hypothesis="Sharpe(lowest-volatility quintile) > Sharpe(highest-volatility quintile) (one-sided stationary-bootstrap p < 0.05).",
        domain="fundamental_factors", datasets=[f"data/market/{r}_equities_daily.parquet" for r in REGIONS],
        methodology="Monthly sort on trailing 252-day volatility, equal-weight quintiles, stationary-bootstrap Sharpe difference test",
        metric_name="annualised Sharpe difference (Q1 minus Q5)", metric_value=diff, p_value=p,
        summary_findings=f"Sharpe Q1 {perf['Q1 low vol']['sharpe']:.2f} vs Q5 {perf['Q5 high vol']['sharpe']:.2f}; spread {mean_spread*100:.2f}%/month (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
