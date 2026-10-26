#!/usr/bin/env python3
"""
res_013 (Predictive regression): do VIX spikes predict higher S&P 500 returns over the next week?

z_t = (VIX_t - mean of the last 60 closes) / std of the last 60 closes (information up to the close of day t).
Model: log(P_{t+5} / P_t) = a + b * z_t + e, Newey-West HAC with 10 lags (5-day overlap). H0: b = 0. H1: b > 0 (fear spikes mean-revert into higher returns).
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
TITLE = "Fear Spikes and Next-Week Returns: Does an Elevated VIX Predict Higher US Equity Returns?"
WIN, H, LAGS = 60, 5, 10


def run():
    t0 = time.time()
    vix = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_yield_curve_daily.parquet")).select(["date", "cboe_vix"]).drop_nulls().sort("date")
    px = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC").select(["date", "close"]).sort("date")
    df = (px.join(vix, on="date", how="inner").sort("date")
          .with_columns(((pl.col("cboe_vix") - pl.col("cboe_vix").rolling_mean(WIN)) / pl.col("cboe_vix").rolling_std(WIN)).alias("z"),
                        (pl.col("close").shift(-H) / pl.col("close")).log().alias("fwd")).drop_nulls())
    y, x = df["fwd"].to_numpy(), df["z"].to_numpy()
    fit = hac_ols(y, x, LAGS)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    _, p_placebo = circular_shift_pvalue(y, x, LAGS, n_draws=2000, seed=7)
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"
    hi = x > 2.0
    mean_hi, mean_lo = (float(y[hi].mean()) if hi.any() else float("nan")), float(y[~hi].mean())
    q = np.quantile(x, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 4)
    qmeans = [float(y[bins == b].mean()) for b in range(5)]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.bar(range(1, 6), [m * 100 for m in qmeans], color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("VIX z-score quintile (5 = highest)")
    ax.set_ylabel("mean next-5-day S&P 500 log return (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_vix_z_quintile_forward_return.png"))
    plt.close()

    sample = f"{df['date'].min()} to {df['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the 60-day VIX z-score has no predictive power for the S\\&P 500 return over the next five sessions; $H_1$: the slope is positive",
        "Data": f"{len(df):,} daily observations, {sample}; {int(hi.sum())} sessions with $z>2$",
        "Method": f"OLS with Newey--West HAC errors ({LAGS} lags) and a 2,000-draw circular-shift placebo; the decision needs both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.5f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.4f}$; mean next-week return "
                   f"{mean_hi*100:.2f}\\% after $z>2$ versus {mean_lo*100:.2f}\\% otherwise.",
    }
    body = ("\\section{Design}\nThe VIX z-score is computed from the 60 closes ending on day $t$ and the outcome is the log return from the close of $t$ to the close of $t+5$, "
            "so predictor and outcome do not overlap \\citep{whaley2000investor}. The five-day overlap between neighbouring outcomes is handled with HAC errors (10 lags) "
            "and a circular-shift placebo.\n\n\\section{Results}\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_vix_z_quintile_forward_return.png}\n"
            "\\caption{Mean next-week S\\&P 500 return by quintile of the VIX z-score.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe sample contains few independent volatility spikes (2018, 2020, 2022, 2025), so the effective sample is small. "
            "No transaction costs or timing frictions are modelled and the result is not a trading rule.\n")
    bib = ("@article{whaley2000investor,\n  title={The investor fear gauge},\n  author={Whaley, Robert E},\n  journal={The Journal of Portfolio Management},\n"
           "  volume={26},\n  number={3},\n  pages={12--17},\n  year={2000}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "sample": sample, "horizon_days": H, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac, "placebo_p_one_sided": p_placebo,
         "r2": float(fit.rsquared), "n_sessions_z_gt_2": int(hi.sum()), "mean_fwd_after_z_gt_2": mean_hi, "mean_fwd_otherwise": mean_lo, "quintile_means": qmeans},
        abstract, body, bib,
        statement="A VIX level that is high relative to its 60-day history predicts a higher S&P 500 return over the next five sessions.",
        null_hypothesis="The slope of the 5-day-ahead S&P 500 log return on the 60-day VIX z-score is zero.",
        alt_hypothesis="The slope is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="macro_transmission", datasets=["data/macro/us_yield_curve_daily.parquet", "data/market/us_indices_daily.parquet"],
        methodology="Overlapping-horizon predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="slope of 5d forward return on VIX z-score", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.5f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, R2={fit.rsquared:.4f}.",
    )


if __name__ == "__main__":
    run()
