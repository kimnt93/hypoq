#!/usr/bin/env python3
"""
res_014 (Event-type test): are gold, long Treasuries and Bitcoin safe havens on the worst S&P 500 days?

Following Baur and Lucey, an asset is a safe haven if its mean return is positive on days when the S&P 500 return is in its worst decile.
Assets: GLD and TLT (data/market/global_etfs_daily.parquet, US sessions) and BTC-USD (data/crypto/crypto_top_assets_daily.parquet).
For each asset: H0 mean return on those days <= 0, H1 > 0; one-sided recentred stationary bootstrap (block 3) on the time-ordered crash-day returns.
Bonferroni alpha = 0.05 / 3. ACCEPTED if at least one asset is a safe haven at the corrected level, REJECTED otherwise.
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
TITLE = "Safe Havens in Equity Crashes: Gold, Long Treasuries and Bitcoin on the Worst US Equity Days"
ASSETS = ["GLD", "TLT", "BTC-USD"]


def run():
    t0 = time.time()
    spx = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC").select(["date", pl.col("returns").alias("spx")]).drop_nulls()
    etf = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "global_etfs_daily.parquet")).filter(pl.col("symbol").is_in(["GLD", "TLT"])).select(["date", "symbol", "returns"])
    btc = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "crypto", "crypto_top_assets_daily.parquet")).filter(pl.col("symbol") == "BTC-USD").select(["date", "symbol", "returns"])
    long = pl.concat([etf, btc]).drop_nulls()
    cut = float(spx["spx"].quantile(0.10))
    crash = spx.filter(pl.col("spx") <= cut)
    rows, rets = [], {}
    alpha = 0.05 / len(ASSETS)
    for a in ASSETS:
        r = crash.join(long.filter(pl.col("symbol") == a).select(["date", pl.col("returns").alias("r")]), on="date", how="inner").sort("date")["r"].to_numpy()
        allr = long.filter(pl.col("symbol") == a)["returns"].to_numpy()
        p = stationary_bootstrap_mean_pvalue(r, n_boot=5000, block=3, seed=11)
        rows.append({"asset": a, "n_crash_days": int(len(r)), "mean_crash": float(r.mean()), "mean_all_days": float(allr.mean()), "p_one_sided": p, "haven_at_bonferroni": bool(p < alpha)})
        rets[a] = r
    decision = "ACCEPTED" if any(r["haven_at_bonferroni"] for r in rows) else "REJECTED"
    best = min(rows, key=lambda r: r["p_one_sided"])

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.bar([r["asset"] for r in rows], [r["mean_crash"] * 100 for r in rows], color="#1f77b4", label="worst-decile S&P days")
    ax.plot([r["asset"] for r in rows], [r["mean_all_days"] * 100 for r in rows], "o", color="#d62728", label="all days")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean daily return (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_safe_haven_returns.png"))
    plt.close()

    trs = "\n".join(f"{r['asset']} & {r['n_crash_days']} & {r['mean_crash']*100:.3f}\\% & {r['mean_all_days']*100:.3f}\\% & {r['p_one_sided']:.4f} \\\\" for r in rows)
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Mean asset return on the worst 10\\% of S\\&P 500 days}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Asset & Crash days & Mean (crash) & Mean (all days) & One-sided $p$ \\\\\n\\midrule\n" + trs + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_safe_haven.tex"), "w") as f:
        f.write(table)
    abstract = {
        "Hypothesis": "$H_0$: the mean return of the asset on the worst-decile S\\&P 500 days is not positive; $H_1$: it is positive (safe haven), for gold, long Treasuries or Bitcoin",
        "Data": f"daily returns of GLD, TLT and BTC-USD; crash threshold {cut*100:.2f}\\% (10th percentile of S\\&P 500 daily returns)",
        "Method": f"One-sided recentred stationary bootstrap of the mean on crash days; Bonferroni $\\alpha={alpha:.4f}$ for three assets",
        "Results": "; ".join(f"{r['asset']} mean {r['mean_crash']*100:.2f}\\% ($p={r['p_one_sided']:.4f}$)" for r in rows) + ".",
    }
    body = ("\\section{Design}\nA safe haven is an asset that is uncorrelated or negatively correlated with equities in market stress \\citep{baur2010gold}. "
            "Crash days are defined by the 10th percentile of the whole-sample S\\&P 500 return distribution, which is fixed before looking at the assets.\n\n"
            "\\section{Results}\n\\input{results/table1_safe_haven.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_safe_haven_returns.png}\n\\caption{Mean daily return on crash days versus all days.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nBitcoin trades around the clock and its daily close (00:00 UTC) is not synchronous with the US close, which can blur the same-day relation. "
            "The crash days cluster in a few episodes (2018, 2020, 2022), so the effective number of independent crashes is small.\n")
    bib = ("@article{baur2010gold,\n  title={Is gold a safe haven? International evidence},\n  author={Baur, Dirk G and McDermott, Thomas K},\n  journal={Journal of Banking and Finance},\n"
           "  volume={34},\n  number={8},\n  pages={1886--1898},\n  year={2010}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"crash_threshold": cut, "bonferroni_alpha": alpha, "assets": rows},
        abstract, body, bib,
        statement="At least one of gold, long Treasuries and Bitcoin earns a positive average return on the worst S&P 500 days.",
        null_hypothesis="For each asset the mean return on worst-decile S&P 500 days is not positive.",
        alt_hypothesis="For at least one asset the mean is positive (one-sided stationary-bootstrap p < 0.05/3).",
        domain="macro_transmission", datasets=["data/market/us_indices_daily.parquet", "data/market/global_etfs_daily.parquet", "data/crypto/crypto_top_assets_daily.parquet"],
        methodology="Baur-Lucey safe-haven definition, stationary-bootstrap mean test, Bonferroni correction over three assets",
        metric_name=f"mean return of {best['asset']} on worst-decile S&P days", metric_value=best["mean_crash"], p_value=best["p_one_sided"],
        summary_findings="; ".join(f"{r['asset']} {r['mean_crash']*100:.2f}% (p={r['p_one_sided']:.4f})" for r in rows),
    )


if __name__ == "__main__":
    run()
