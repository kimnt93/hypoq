#!/usr/bin/env python3
"""
res_019 (Predictive regression, Vietnam): does a low market P/E predict higher VN-Index returns over the next year?

Model: log(VNINDEX_{t+252} / VNINDEX_t) = a + b * PE_t + e, daily observations with 252-day overlap, Newey-West HAC with 252 lags.
PE_t is the VN-Index trailing P/E published by HOSE at the close of day t. H0: b = 0. H1: b < 0 (cheap markets earn more).
ACCEPTED only if both the HAC one-sided p < 0.05 and a 2,000-draw circular-shift placebo p < 0.05 (applied to -PE so that the alternative is b > 0).
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
TITLE = "Valuation and Subsequent Returns in Vietnam: Does a Low Market P/E Predict Higher One-Year VN-Index Returns?"
H = 252


def run():
    t0 = time.time()
    pe = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "vn_market_pe_daily.parquet")).filter(pl.col("symbol") == "VNINDEX").select(["date", "pe"]).drop_nulls().filter(pl.col("pe") > 0)
    px = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "vn_indices_daily.parquet")).filter(pl.col("symbol") == "VNINDEX").select(["date", "close"]).sort("date")
    df = (px.join(pe, on="date", how="inner").sort("date").with_columns((pl.col("close").shift(-H) / pl.col("close")).log().alias("fwd")).drop_nulls())
    y, x = df["fwd"].to_numpy(), df["pe"].to_numpy()
    fit = hac_ols(y, x, H)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.cdf(t_hac))          # H1: b < 0
    t_obs, p_placebo = circular_shift_pvalue(y, -x, H, n_draws=2000, seed=7)
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"
    q = np.quantile(x, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 4)
    qmeans = [float(y[bins == b].mean()) for b in range(5)]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.bar(range(1, 6), [m * 100 for m in qmeans], color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("P/E quintile (1 = cheapest)")
    ax.set_ylabel("mean next-252-day VN-Index log return (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_pe_quintile_forward_return.png"))
    plt.close()

    sample = f"{df['date'].min()} to {df['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the VN-Index P/E has no predictive power for the 252-session-ahead VN-Index return; $H_1$: the slope is negative",
        "Data": f"{len(df):,} daily observations, {sample}; P/E range {x.min():.1f} to {x.max():.1f}",
        "Method": f"OLS with Newey--West HAC errors ({H} lags) and a 2,000-draw circular-shift placebo; the decision needs both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.4f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.3f}$.",
    }
    body = ("\\section{Design}\nValuation ratios are a classical predictor of long-horizon returns \\citep{campbell1988stock}. The P/E at the close of day $t$ is paired with the log return "
            "over the following 252 sessions; the overlap makes the HAC lag equal to the horizon and the circular-shift placebo is the main robustness check.\n\n\\section{Results}\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_pe_quintile_forward_return.png}\n\\caption{Mean one-year-ahead VN-Index return by P/E quintile.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe sample holds about eight non-overlapping years, so the effective sample is under ten independent observations; the P/E uses trailing earnings which "
            "are reported with a lag and may be revised, a point-in-time earnings series is not free.\n")
    bib = ("@article{campbell1988stock,\n  title={Stock prices, earnings, and expected dividends},\n  author={Campbell, John Y and Shiller, Robert J},\n  journal={The Journal of Finance},\n"
           "  volume={43},\n  number={3},\n  pages={661--676},\n  year={1988}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "sample": sample, "horizon_days": H, "beta": beta, "hac_t": t_hac, "hac_p_one_sided_negative": p_hac, "placebo_p_one_sided": p_placebo,
         "r2": float(fit.rsquared), "quintile_means": qmeans},
        abstract, body, bib,
        statement="A low VN-Index trailing P/E predicts a higher VN-Index return over the next 252 sessions.",
        null_hypothesis="The slope of the 252-session-ahead VN-Index log return on the P/E is zero.",
        alt_hypothesis="The slope is negative (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="fundamental_factors", datasets=["data/macro/vn_market_pe_daily.parquet", "data/market/vn_indices_daily.parquet"],
        methodology="Overlapping-horizon predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="slope of 252d forward return on P/E", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.4f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, R2={fit.rsquared:.3f}.",
    )


if __name__ == "__main__":
    run()
