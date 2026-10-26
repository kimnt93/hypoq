#!/usr/bin/env python3
"""
res_018 (Cross-sectional sort): does 12-1 month momentum work in a pooled universe of large caps from seven regions?

At each month-end t, stocks with 11 consecutive months of returns are ranked on the cumulative log return of months t-11..t-1 (the month t return is skipped).
Quintile 5 (winners) and quintile 1 (losers) are held equal-weighted over month t+1 (local-currency returns).
H0: Sharpe(winners) <= Sharpe(losers). H1: Sharpe(winners) > Sharpe(losers): one-sided stationary-bootstrap p < 0.05 (monthly data, mean block 3).
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

# --- metric helpers copied from .agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py (hypoq/ is frozen; each study is self-contained) ---

def sharpe(r: np.ndarray, periods: int, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio; `rf` is the risk-free rate per period (0 is fine for long/short or excess returns)."""
    x = r - rf
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
    idx = _stationary_indices(rng, len(a), block, n_boot)
    sh = lambda m: m.mean(axis=1) / m.std(axis=1, ddof=1)
    obs = a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)
    diffs = sh(a[idx]) - sh(b[idx])
    return float(obs * np.sqrt(periods)), float((np.sum(diffs - diffs.mean() >= obs) + 1) / (n_boot + 1))

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Twelve-Minus-One Month Momentum in a Pooled Universe of Large-Cap Equities from Seven Regions"
REGIONS = ["us", "eu", "jp", "hk", "kr", "cn", "asia"]
FORM, MIN_STOCKS = 11, 40


def monthly_panel() -> pl.DataFrame:
    frames = [pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", f"{r}_equities_daily.parquet")).select(["date", "symbol", "returns"]).drop_nulls("returns") for r in REGIONS]
    # four tickers are listed in two regional files: keep one copy so no stock is counted twice
    d = pl.concat(frames).unique(subset=["symbol", "date"], keep="first", maintain_order=True).sort(["symbol", "date"])
    d = d.with_columns(pl.col("returns").log1p().alias("lr"), (pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).alias("mi"))
    m = d.group_by(["symbol", "mi"]).agg(pl.col("lr").sum().alias("lr")).sort(["symbol", "mi"])
    # formation return over months t-11..t-1 using only consecutive calendar months
    m = m.with_columns(pl.col("lr").shift(1).rolling_sum(FORM).over("symbol").alias("mom"),
                       (pl.col("mi").shift(1) - pl.col("mi").shift(FORM)).over("symbol").alias("span"),
                       (pl.col("mi") - pl.col("mi").shift(FORM)).over("symbol").alias("span_full"),
                       pl.col("lr").shift(-1).over("symbol").alias("lr_next"), pl.col("mi").shift(-1).over("symbol").alias("mi_next"))
    m = m.filter((pl.col("span") == FORM - 1) & (pl.col("span_full") == FORM) & (pl.col("mi_next") == pl.col("mi") + 1) & pl.col("mom").is_not_null() & pl.col("lr_next").is_not_null())
    return m.with_columns((pl.col("lr_next").exp() - 1).alias("ret_next"))


def run():
    t0 = time.time()
    m = monthly_panel()
    rows = []
    for mi, g in m.group_by("mi", maintain_order=True):
        if g.height < MIN_STOCKS:
            continue
        g = g.with_columns((pl.col("mom").rank("ordinal") - 1).alias("rk"))
        q = (g["rk"] * 5 // g.height).to_numpy()
        r = g["ret_next"].to_numpy()
        rows.append((mi[0], r[q == 4].mean(), r[q == 0].mean(), r.mean(), g.height))
    rows.sort()
    mi = [r[0] for r in rows]
    win, los, mkt = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    n_med = int(np.median([r[4] for r in rows]))
    diff, p = stationary_bootstrap_sharpe_diff(win, los, periods=12, n_boot=5000, block=3, seed=11)
    spread = win - los
    t_spread = float(sps.ttest_1samp(spread, 0.0).statistic)
    pf = lambda x: {"ann_return": float(np.prod(1 + x) ** (12 / len(x)) - 1), "ann_vol": float(x.std(ddof=1) * np.sqrt(12)), "sharpe": sharpe(x, 12),
                    "max_drawdown": float((np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min())}
    perf = {"Winners (Q5)": pf(win), "Losers (Q1)": pf(los), "All stocks": pf(mkt)}
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    for lab, x, c in [("Winners (Q5)", win, "#1f77b4"), ("Losers (Q1)", los, "#d62728"), ("All stocks", mkt, "#7f7f7f")]:
        ax.plot(range(len(x)), np.cumprod(1 + x), label=f"{lab} (Sharpe {perf[lab]['sharpe']:.2f})", color=c)
    ticks = list(range(0, len(mi), 24))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str((mi[i] - 1) // 12) for i in ticks])
    ax.set_ylabel("growth of 1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_momentum_quintiles.png"))
    plt.close()

    rws = "\n".join(f"{k} & {v['ann_return']*100:.1f}\\% & {v['ann_vol']*100:.1f}\\% & {v['sharpe']:.2f} & {v['max_drawdown']*100:.1f}\\% \\\\" for k, v in perf.items())
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Equal-weight momentum quintiles, monthly rebalanced, local-currency returns}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Portfolio & Return & Vol & Sharpe & Max DD \\\\\n\\midrule\n" + rws + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_momentum_quintiles.tex"), "w") as f:
        f.write(table)
    fmt = lambda k: f"{(k - 1) // 12}-{(k - 1) % 12 + 1:02d}"
    sample = f"{fmt(mi[0])} to {fmt(mi[-1])} (formation months)"
    abstract = {
        "Hypothesis": "$H_0$: the Sharpe ratio of past winners does not exceed that of past losers; $H_1$: it does",
        "Data": f"{len(rows)} monthly cross-sections, median {n_med} stocks from {len(REGIONS)} regions, {sample}",
        "Method": "Month-end sort on the 12-1 month cumulative return, equal-weight quintiles, one-sided stationary-bootstrap test of the Sharpe difference (5,000 draws, block 3)",
        "Results": f"Sharpe {perf['Winners (Q5)']['sharpe']:.2f} versus {perf['Losers (Q1)']['sharpe']:.2f} (difference {diff:.2f}, one-sided $p={p:.4f}$); "
                   f"mean monthly winners-minus-losers spread {spread.mean()*100:.2f}\\% ($t={t_spread:.2f}$).",
    }
    body = ("\\section{Design}\nMomentum is the cumulative return over months $t-11$ to $t-1$, skipping the most recent month to avoid short-term reversal "
            "\\citep{jegadeesh1993returns}. Stocks enter only with eleven consecutive observed months. Returns are in local currency.\n\n\\section{Results}\n"
            "\\input{results/table1_momentum_quintiles.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_momentum_quintiles.png}\n\\caption{Growth of one unit in the winner, loser and all-stock portfolios.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe universe is a fixed list of today's large caps: survivorship bias helps past losers that recovered and hides delisted ones, "
            "so it works against momentum in both tails in unknown proportions. About 130 stocks give quintiles of roughly 25 names, and the decade contains a few strong momentum crashes.\n")
    bib = ("@article{jegadeesh1993returns,\n  title={Returns to buying winners and selling losers: Implications for stock market efficiency},\n  author={Jegadeesh, Narasimhan and Titman, Sheridan},\n  journal={The Journal of Finance},\n"
           "  volume={48},\n  number={1},\n  pages={65--91},\n  year={1993}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_months": len(rows), "median_stocks": n_med, "sample": sample, "performance": perf, "sharpe_difference": diff, "bootstrap_p_one_sided": p,
         "mean_monthly_spread_winners_minus_losers": float(spread.mean()), "spread_t": t_spread},
        abstract, body, bib,
        statement="Large-cap stocks with high 12-1 month returns earn a higher risk-adjusted return than stocks with low past returns.",
        null_hypothesis="Sharpe(winner quintile) <= Sharpe(loser quintile).",
        alt_hypothesis="Sharpe(winner quintile) > Sharpe(loser quintile) (one-sided stationary-bootstrap p < 0.05).",
        domain="fundamental_factors", datasets=[f"data/market/{r}_equities_daily.parquet" for r in REGIONS],
        methodology="Monthly sort on 12-1 month return, equal-weight quintiles, stationary-bootstrap Sharpe difference test",
        metric_name="annualised Sharpe difference (winners minus losers)", metric_value=diff, p_value=p,
        summary_findings=f"Sharpe winners {perf['Winners (Q5)']['sharpe']:.2f} vs losers {perf['Losers (Q1)']['sharpe']:.2f}; spread {spread.mean()*100:.2f}%/month (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
