#!/usr/bin/env python3
"""
res_012 (Predictive regression, Vietnam derivatives): does the VN30 futures basis predict the next-day VN30 index return?

basis_t = F1M_close_t / VN30_close_t - 1 at the close of day t (both are published by the exchange at the close).
Model: r_{VN30,t+1} = a + b * basis_t + e, Newey-West HAC (5 lags). H0: b = 0. H1: b > 0 (one-sided).
ACCEPTED only if both the HAC p < 0.05 and a 2,000-draw circular-shift placebo p < 0.05.
The front-month contract rolls, so the basis converges to zero near expiry; days with |basis| > 10% are treated as data errors and dropped, never clipped.
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
TITLE = "Futures Basis and Next-Day Spot Returns on the Vietnamese VN30 Index"
LAGS = 5


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
    fut = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "vn_vn30f1m_derivatives_daily.parquet")).select(["date", pl.col("close").alias("f")])
    spot = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "vn_indices_daily.parquet")).filter(pl.col("symbol") == "VN30")
            .sort("date").select(["date", pl.col("close").alias("s")]).with_columns(pl.col("s").pct_change().alias("r")))
    df = spot.join(fut, on="date", how="inner").sort("date").with_columns((pl.col("f") / pl.col("s") - 1).alias("basis"))
    n_raw = df.height
    df = df.filter(pl.col("basis").abs() <= 0.10)
    n_dropped = n_raw - df.height
    df = df.with_columns(pl.col("r").shift(-1).alias("r_next")).drop_nulls(["r_next", "basis"])
    y, x = df["r_next"].to_numpy(), df["basis"].to_numpy()
    fit = hac_ols(y, x, LAGS)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    t_obs, p_placebo = circular_shift_pvalue(y, x, LAGS, n_draws=2000, seed=7)
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"
    top = x > np.quantile(x, 0.8)
    bot = x < np.quantile(x, 0.2)
    spread = float(y[top].mean() - y[bot].mean())

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    q = np.quantile(x, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 4)
    ax.bar(range(1, 6), [y[bins == b].mean() * 100 for b in range(5)], color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("basis quintile (1 = lowest basis)")
    ax.set_ylabel("mean next-day VN30 return (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_basis_quintile_next_day_return.png"))
    plt.close()

    sample = f"{df['date'].min()} to {df['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the VN30 futures basis at day $t$ has no relation to the VN30 return on $t+1$; $H_1$: the relation is positive",
        "Data": f"{len(df):,} sessions, {sample} ({n_dropped} sessions with $|$basis$|>10\\%$ dropped)",
        "Method": f"OLS with Newey--West HAC errors ({LAGS} lags) and a 2,000-draw circular-shift placebo; ACCEPTED only if both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.4f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.4f}$; "
                   f"top-minus-bottom basis quintile next-day return {spread*100:.3f}\\%.",
    }
    body = ("\\section{Design}\nThe VN30 futures are the only liquid index derivative in Vietnam and retail-heavy, so the basis may capture sentiment before it reaches the "
            "cash market \\citep{hasbrouck2003intraday}. Both prices are daily closes of the same session; the return of the next session is the outcome, hence there is no overlap with the predictor.\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_basis_quintile_next_day_return.png}\n"
            "\\caption{Mean next-day VN30 return by basis quintile.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe series is a continuous front-month series, so the basis is mechanically linked to time to expiry and to dividends, which are not modelled. "
            "Daily closes cannot reveal intraday lead--lag effects.\n")
    bib = ("@article{hasbrouck2003intraday,\n  title={Intraday price formation in US equity index markets},\n  author={Hasbrouck, Joel},\n"
           "  journal={The Journal of Finance},\n  volume={58},\n  number={6},\n  pages={2375--2400},\n  year={2003}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "n_dropped_basis_gt_10pct": n_dropped, "sample": sample, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac,
         "placebo_p_one_sided": p_placebo, "r2": float(fit.rsquared), "top_minus_bottom_quintile_next_day": spread, "mean_basis": float(x.mean())},
        abstract, body, bib,
        statement="A higher VN30 futures basis at the close predicts a higher VN30 index return on the next trading day.",
        null_hypothesis="The coefficient of the next-day VN30 return on the closing futures basis is zero.",
        alt_hypothesis="The coefficient is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="market_microstructure", datasets=["data/market/vn_vn30f1m_derivatives_daily.parquet", "data/market/vn_indices_daily.parquet"],
        methodology="Predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="beta of next-day VN30 return on futures basis", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.4f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}.",
    )


if __name__ == "__main__":
    run()
