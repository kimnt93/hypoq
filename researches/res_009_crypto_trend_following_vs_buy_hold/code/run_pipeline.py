#!/usr/bin/env python3
"""
res_009 (Strategy test): does a 30-day trend filter improve the risk-adjusted return of a crypto portfolio?

Universe: the 11 largest crypto assets in data/crypto/crypto_top_assets_daily.parquet (each enters when its history starts).
Strategy (fixed in advance, no tuning): at the close of day t hold asset i on day t+1 iff its trailing 30-day log return is positive,
otherwise hold cash at 0%. Capital is split equally over the assets that exist on that day. Cost: 10 bps per unit of one-way turnover.
Benchmark: the same equal-weight portfolio always invested (daily rebalanced, same cost on its own turnover).
H0: Sharpe(trend) <= Sharpe(benchmark). H1: Sharpe(trend) > Sharpe(benchmark): one-sided stationary-bootstrap p < 0.05.
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

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Trend Filtering in Crypto Markets: A Thirty-Day Time-Series Momentum Rule versus Equal-Weight Buy-and-Hold"
LOOKBACK, COST, PPY = 30, 0.0010, 365


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

def perf(r):
    eq = np.cumprod(1 + r)
    return {"ann_return": float(np.prod(1 + r) ** (PPY / len(r)) - 1), "ann_vol": float(r.std(ddof=1) * np.sqrt(PPY)),
            "sharpe": sharpe(r, PPY), "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min())}


def run():
    t0 = time.time()
    d = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "crypto", "crypto_top_assets_daily.parquet")).select(["date", "symbol", "returns"])
    wide = d.pivot(on="symbol", index="date", values="returns").sort("date")
    dates = wide["date"].to_list()
    R = wide.drop("date").to_numpy().astype(float)
    exists = ~np.isnan(R)
    logr = np.log1p(np.where(exists, R, 0.0))
    # trailing sum over the last LOOKBACK observed days (NaN until a full window of real returns exists)
    cs = np.cumsum(logr, axis=0)
    cnt = np.cumsum(exists, axis=0)
    trail = np.full_like(R, np.nan)
    trail[LOOKBACK:] = cs[LOOKBACK:] - cs[:-LOOKBACK]
    full = np.zeros_like(R, dtype=bool)
    full[LOOKBACK:] = (cnt[LOOKBACK:] - cnt[:-LOOKBACK]) == LOOKBACK
    sig = (full & (trail > 0)).astype(float)                       # decided at close t
    sig_next = np.vstack([np.zeros((1, R.shape[1])), sig[:-1]])     # held over day t -> signal from t-1 close
    ex_next = exists                                                # asset must have a return on the holding day
    start = int(np.argmax(full.any(axis=1))) + 1
    Rz = np.where(exists, R, 0.0)

    def run_book(hold):
        n_avail = np.maximum(exists.sum(axis=1), 1)
        w = hold * ex_next / n_avail[:, None]
        gross = (w * Rz).sum(axis=1)
        turn = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, R.shape[1])))).sum(axis=1)
        return gross - COST * turn, turn

    trend, turn_t = run_book(sig_next)
    bench, turn_b = run_book(np.ones_like(R))
    trend, bench, turn_t, turn_b = trend[start:], bench[start:], turn_t[start:], turn_b[start:]
    diff, p = stationary_bootstrap_sharpe_diff(trend, bench, periods=PPY, n_boot=5000, block=10, seed=11)
    pt, pb = perf(trend), perf(bench)
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"
    exposure = float((sig_next[start:] * ex_next[start:]).sum() / ex_next[start:].sum())

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.plot(dates[start:], np.cumprod(1 + trend), label=f"30d trend (Sharpe {pt['sharpe']:.2f})", color="#1f77b4")
    ax.plot(dates[start:], np.cumprod(1 + bench), label=f"buy-and-hold (Sharpe {pb['sharpe']:.2f})", color="#7f7f7f")
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_trend_vs_buy_hold.png"))
    plt.close()

    sample = f"{dates[start]} to {dates[-1]}"
    rows = "\n".join(f"{n} & {m['ann_return']*100:.1f}\\% & {m['ann_vol']*100:.1f}\\% & {m['sharpe']:.2f} & {m['max_drawdown']*100:.1f}\\% & {tn*100:.1f}\\% \\\\"
                     for n, m, tn in [("30-day trend", pt, turn_t.mean()), ("Buy-and-hold", pb, turn_b.mean())])
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Net performance after 10 bps one-way costs}\n\\begin{tabular}{lccccc}\n\\toprule\n"
             "Portfolio & Return & Vol & Sharpe & Max DD & Daily turnover \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_performance.tex"), "w") as f:
        f.write(table)
    abstract = {
        "Hypothesis": "$H_0$: the Sharpe ratio of the 30-day trend rule does not exceed that of equal-weight buy-and-hold; $H_1$: it does",
        "Data": f"{R.shape[1]} large crypto assets, {len(trend):,} daily observations, {sample}",
        "Method": "Long-or-cash rule on the sign of the trailing 30-day return, 10 bps costs, one-sided stationary-bootstrap test of the Sharpe difference (5,000 draws)",
        "Results": f"Sharpe {pt['sharpe']:.2f} versus {pb['sharpe']:.2f} (difference {diff:.2f}, one-sided $p={p:.4f}$); maximum drawdown "
                   f"{pt['max_drawdown']*100:.1f}\\% versus {pb['max_drawdown']*100:.1f}\\%.",
    }
    body = ("\\section{Design}\nThe rule uses only returns up to the close of day $t-1$ to hold on day $t$ \\citep{moskowitz2012time}. "
            "The look-back and the cost are fixed before the evaluation and are not tuned. An asset is included only after it has a full 30-day history.\n\n"
            f"\\section{{Results}}\nThe rule is invested in {exposure*100:.1f}\\% of asset-days.\n\n\\input{{results/table1_performance.tex}}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_trend_vs_buy_hold.png}\n\\caption{Growth of one unit of capital.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe universe is today's largest assets, so there is survivorship bias that favours buy-and-hold. Returns are daily closes "
            "of Yahoo Finance quotes without spreads or funding costs. The comparison is one pre-specified rule, not a search over parameters.\n")
    bib = ("@article{moskowitz2012time,\n  title={Time series momentum},\n  author={Moskowitz, Tobias J and Ooi, Yao Hua and Pedersen, Lasse Heje},\n"
           "  journal={Journal of Financial Economics},\n  volume={104},\n  number={2},\n  pages={228--250},\n  year={2012}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(trend), "n_assets": int(R.shape[1]), "sample": sample, "lookback_days": LOOKBACK, "cost_per_unit_turnover": COST,
         "trend": pt, "buy_hold": pb, "sharpe_difference": diff, "bootstrap_p_one_sided": p, "exposure_share": exposure,
         "avg_daily_turnover_trend": float(turn_t.mean()), "avg_daily_turnover_buy_hold": float(turn_b.mean())},
        abstract, body, bib,
        statement="A 30-day time-series momentum filter raises the Sharpe ratio of an equal-weight crypto portfolio.",
        null_hypothesis="Sharpe(30d trend, net of costs) <= Sharpe(equal-weight buy-and-hold, net of costs).",
        alt_hypothesis="Sharpe(30d trend) > Sharpe(buy-and-hold) (one-sided stationary-bootstrap p < 0.05).",
        domain="crypto", datasets=["data/crypto/crypto_top_assets_daily.parquet"],
        methodology="Pre-specified long-or-cash trend rule, cost-adjusted, stationary-bootstrap Sharpe difference test",
        metric_name="annualised Sharpe difference (trend minus buy-and-hold)", metric_value=diff, p_value=p,
        summary_findings=f"Sharpe {pt['sharpe']:.2f} vs {pb['sharpe']:.2f}; max drawdown {pt['max_drawdown']*100:.1f}% vs {pb['max_drawdown']*100:.1f}% (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
