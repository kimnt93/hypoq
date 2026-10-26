#!/usr/bin/env python3
"""
res_008 (Predictive regression): does a flatter US yield curve predict lower S&P 500 returns over the next six months?

Model: log(P_{t+126}/P_t) = a + b * (10y - 3m spread)_t + e, daily observations with 126-day overlap, Newey-West HAC errors (126 lags).
H0: b = 0. H1: b > 0 (steeper curve -> higher future returns, i.e. a flat or inverted curve predicts weak returns).
Decision rule fixed in advance: ACCEPTED only if BOTH the HAC one-sided p < 0.05 AND the circular-shift placebo p < 0.05
(overlapping returns make the HAC p-value alone unreliable in a ten-year sample with two inversion episodes).
"""
import os
import sys
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import polars as pl
from scipy import stats as sps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Does the Slope of the US Yield Curve Predict Six-Month-Ahead US Equity Returns? A Predictive Regression with Overlap-Robust Inference"
H = 126


# --- inference helpers (self-contained: the hypoq package is frozen; see the hypothesis-testing-stats skill) ---

def hac_ols(y: np.ndarray, x: np.ndarray, lags: int):
    """OLS of y on a constant and x (1-d or 2-d) with Newey-West HAC errors."""
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def circular_shift_pvalue(y: np.ndarray, x: np.ndarray, lags: int, n_draws: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """
    Placebo for a predictive regression y_t = a + b x_t + e_t: rotate x by a random offset (keeping its autocorrelation and the
    link between observations inside x intact, breaking only the alignment with y) and recompute the HAC t-statistic of b.
    Returns (observed t, one-sided p for H1: b > 0).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    t_obs = float(hac_ols(y, x, lags).tvalues[1])
    lo = lags + 10
    sims = np.empty(n_draws)
    for i in range(n_draws):
        s = int(rng.integers(lo, n - lo))
        sims[i] = hac_ols(y, np.roll(x, s), lags).tvalues[1]
    return t_obs, float((np.sum(sims >= t_obs) + 1) / (n_draws + 1))

def run():
    t0 = time.time()
    yc = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_yield_curve_daily.parquet")).select(["date", "yield_spread_10y_3m"]).drop_nulls()
    px = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC").select(["date", "close"]).sort("date")
    df = (px.join(yc, on="date", how="inner").sort("date")
          .with_columns((pl.col("close").shift(-H) / pl.col("close")).log().alias("fwd")).drop_nulls())
    y, x = df["fwd"].to_numpy(), df["yield_spread_10y_3m"].to_numpy()
    fit = hac_ols(y, x, H)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    t_obs, p_placebo = circular_shift_pvalue(y, x, H, n_draws=2000, seed=7)
    inv = x < 0
    n_inv_days = int(inv.sum())
    mean_inv = float(y[inv].mean()) if n_inv_days else float("nan")
    mean_norm = float(y[~inv].mean())
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    sc = ax.scatter(x, y * 100, s=4, c=np.arange(len(x)), cmap="viridis", alpha=0.7)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, (fit.params[0] + beta * xs) * 100, color="#d62728", lw=1.5, label=f"OLS slope {beta*100:.2f} pp per pp")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("10y - 3m Treasury spread (pp)")
    ax.set_ylabel("S&P 500 log return over next 126 sessions (%)")
    ax.legend()
    plt.colorbar(sc, label="time (early to late)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_spread_vs_forward_return.png"))
    plt.close()

    sample = f"{df['date'].min()} to {df['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the 10y--3m spread has no predictive power for 126-session-ahead S\\&P 500 returns; $H_1$: the slope coefficient is positive",
        "Data": f"{len(df):,} daily observations, {sample}; {n_inv_days} sessions with an inverted curve",
        "Method": f"OLS with Newey--West HAC errors ({H} lags) and a 2,000-draw circular-shift placebo; ACCEPTED only if both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.4f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.3f}$. "
                   f"",
    }
    body = ("\\section{Design}\nThe regressor is the 10-year minus 3-month Treasury spread observed at the close of day $t$; the outcome is the log "
            "return of the S\\&P 500 from the close of $t$ to the close of $t+126$, so there is no look-ahead \\citep{estrella1998predicting}. "
            "Overlapping outcomes induce strong serial correlation, therefore HAC errors with a lag equal to the horizon are used and a circular-shift placebo "
            "provides a non-parametric reference distribution for the $t$-statistic.\n\n"
            "\\section{Results}\n"
            f"Mean forward return is {mean_inv*100:.2f}\\% after inverted-curve sessions ({n_inv_days} sessions) and {mean_norm*100:.2f}\\% otherwise.\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_spread_vs_forward_return.png}\n"
            "\\caption{Six-month-ahead S\\&P 500 log return against the 10y--3m spread.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe sample covers one full cycle with only two inversion episodes, so the effective number of independent observations is "
            "far smaller than the number of rows; the placebo accounts for this only partly. The spread is a market series and is not revised.\n")
    bib = ("@article{estrella1998predicting,\n  title={Predicting US recessions: Financial variables as leading indicators},\n  author={Estrella, Arturo and Mishkin, Frederic S},\n"
           "  journal={Review of Economics and Statistics},\n  volume={80},\n  number={1},\n  pages={45--61},\n  year={1998}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "sample": sample, "horizon_days": H, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac, "placebo_p_one_sided": p_placebo,
         "r2": float(fit.rsquared), "n_inverted_sessions": n_inv_days, "mean_fwd_return_inverted": mean_inv, "mean_fwd_return_not_inverted": mean_norm},
        abstract, body, bib,
        statement="A flatter or inverted 10y-3m Treasury spread predicts lower S&P 500 returns over the next 126 sessions.",
        null_hypothesis="The slope coefficient of the 126-session-ahead S&P 500 log return on the 10y-3m spread is zero.",
        alt_hypothesis="The slope coefficient is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="macro_transmission", datasets=["data/macro/us_yield_curve_daily.parquet", "data/market/us_indices_daily.parquet"],
        methodology="Overlapping-horizon predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="slope of 126d forward return on 10y-3m spread", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.4f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, R2={fit.rsquared:.3f}.",
    )


if __name__ == "__main__":
    run()
