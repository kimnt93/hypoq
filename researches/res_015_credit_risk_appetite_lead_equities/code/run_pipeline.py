#!/usr/bin/env python3
"""
res_015 (Predictive regression): does credit-market risk appetite lead equities?

x_t = 5-day return of HYG (high-yield bonds) minus 5-day return of IEF (7-10y Treasuries), a free proxy for the credit risk premium, known at the close of day t.
y_t = SPY return over the next five sessions. Model y = a + b x + e, Newey-West HAC with 10 lags. H0: b = 0. H1: b > 0 (credit leads equities).
ACCEPTED only if both the HAC one-sided p < 0.05 and a 2,000-draw circular-shift placebo p < 0.05.
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
import statsmodels.api as sm
from scipy import stats as sps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

# --- metric helpers copied from .agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py (hypoq/ is frozen; each study is self-contained) ---

def hac_ols(y: np.ndarray, x: np.ndarray, lags: int):
    """OLS of y on a constant and x (1-d or 2-d) with Newey-West HAC errors (statsmodels results object)."""
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def circular_shift_pvalue(y: np.ndarray, x: np.ndarray, lags: int, n_draws: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """
    Placebo for y_t = a + b x_t + e_t: rotate x by a random offset (keeps its autocorrelation, breaks only the alignment with y) and
    recompute the HAC t-statistic of b. Returns (observed t, one-sided p for H1: b > 0).
    """
    rng = np.random.default_rng(seed)
    n, lo = len(y), lags + 10
    t_obs = float(hac_ols(y, x, lags).tvalues[1])
    sims = np.array([hac_ols(y, np.roll(x, int(rng.integers(lo, n - lo))), lags).tvalues[1] for _ in range(n_draws)])
    return t_obs, float((np.sum(sims >= t_obs) + 1) / (n_draws + 1))

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Does High-Yield Credit Lead Equities? Relative Bond Performance and Next-Week Stock Returns"
H, LAGS = 5, 10


def run():
    t0 = time.time()
    e = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "global_etfs_daily.parquet")).filter(pl.col("symbol").is_in(["HYG", "IEF", "SPY"])).select(["date", "symbol", "returns"])
    w = e.pivot(on="symbol", index="date", values="returns").sort("date").drop_nulls()
    lr = lambda c: pl.col(c).log1p()
    w = w.with_columns((lr("HYG") - lr("IEF")).alias("spr"), lr("SPY").alias("ls"))
    w = w.with_columns(pl.col("spr").rolling_sum(H).alias("x"), pl.col("ls").rolling_sum(H).alias("own"),
                       sum([pl.col("ls").shift(-k) for k in range(1, H + 1)]).alias("y")).drop_nulls(["x", "own", "y"])
    y, x = w["y"].to_numpy(), w["x"].to_numpy()
    fit = hac_ols(y, x, LAGS)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    _, p_placebo = circular_shift_pvalue(y, x, LAGS, n_draws=2000, seed=7)
    ctrl = hac_ols(y, np.column_stack([x, w["own"].to_numpy()]), LAGS)  # robustness: control for SPY's own trailing 5-day return
    beta_c, t_c = float(ctrl.params[1]), float(ctrl.tvalues[1])
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"
    q = np.quantile(x, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 4)
    qmeans = [float(y[bins == b].mean()) for b in range(5)]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.bar(range(1, 6), [m * 100 for m in qmeans], color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("quintile of 5-day HYG minus IEF return (5 = credit outperforming)")
    ax.set_ylabel("mean next-5-day SPY return (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_credit_quintile_spy.png"))
    plt.close()

    sample = f"{w['date'].min()} to {w['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the five-day relative return of high-yield versus Treasury bonds does not predict the next five-day SPY return; $H_1$: the relation is positive",
        "Data": f"{len(w):,} daily observations of HYG, IEF and SPY, {sample}",
        "Method": f"OLS with Newey--West HAC errors ({LAGS} lags) and a 2,000-draw circular-shift placebo; the decision needs both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.4f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.4f}$.",
    }
    body = ("\\section{Design}\nCredit markets are often argued to price risk earlier than equities \\citep{gilchrist2012credit}. "
            "The predictor is observed at the close of day $t$ and the outcome covers days $t+1$ to $t+5$. Total-return ETFs are used because they are free and liquid; "
            "they embed duration, so the Treasury leg is a duration-adjusted benchmark rather than a pure credit spread.\n\n\\section{Results}\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_credit_quintile_spy.png}\n\\caption{Mean next-week SPY return by quintile of the credit signal.}\n\\end{figure}\n\n"
            f"Adding SPY's own trailing five-day return as a control gives a coefficient of {beta_c:.4f} ($t={t_c:.2f}$) on the credit signal.\n\n"
            "\\section{Limitations}\nHYG and IEF differ in duration, so the signal also reflects interest-rate moves. The overlap of five-day windows reduces the effective sample size. "
            "Daily closes of ETFs and the index are not perfectly synchronous.\n")
    bib = ("@article{gilchrist2012credit,\n  title={Credit spreads and business cycle fluctuations},\n  author={Gilchrist, Simon and Zakraj{\\v{s}}ek, Egon},\n  journal={American Economic Review},\n"
           "  volume={102},\n  number={4},\n  pages={1692--1720},\n  year={2012}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(w), "sample": sample, "horizon_days": H, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac, "placebo_p_one_sided": p_placebo,
         "r2": float(fit.rsquared), "beta_with_spy_control": beta_c, "t_with_spy_control": t_c, "quintile_means": qmeans},
        abstract, body, bib,
        statement="Outperformance of high-yield bonds over Treasuries over five days predicts a higher SPY return over the next five days.",
        null_hypothesis="The slope of the next-5-day SPY return on the 5-day HYG-minus-IEF return is zero.",
        alt_hypothesis="The slope is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="macro_transmission", datasets=["data/market/global_etfs_daily.parquet"],
        methodology="Overlapping-horizon predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="slope of next-5d SPY return on 5d HYG-IEF return", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.4f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, R2={fit.rsquared:.4f}.",
    )


if __name__ == "__main__":
    run()
