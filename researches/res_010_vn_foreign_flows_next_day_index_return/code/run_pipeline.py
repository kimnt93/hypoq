#!/usr/bin/env python3
"""
res_010 (Predictive regression, Vietnam): do net foreign flows predict the next-session VN-Index return?

Flow imbalance_t = sum_i net_val_{i,t} / sum_i (buy_val_{i,t} + sell_val_{i,t}) over the 65 stocks in data/market/vn_foreign_trading_daily.parquet
(foreign buy minus sell value divided by foreign turnover). It is published after the close of day t and used only for day t+1.
Model: r_{VNINDEX,t+1} = a + b * imbalance_t + e, Newey-West HAC (5 lags). H0: b = 0. H1: b > 0 (one-sided).
ACCEPTED only if both the HAC p < 0.05 and a 2,000-draw circular-shift placebo p < 0.05. Sessions where the vendor reports no foreign turnover are dropped, never zero-filled.
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
TITLE = "Foreign Investor Order Flow and Next-Day Returns on the Vietnamese Equity Market"
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
    ft = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "vn_foreign_trading_daily.parquet"))
    flow = (ft.group_by("date").agg(pl.col("net_val").sum().alias("net"), (pl.col("buy_val") + pl.col("sell_val")).sum().alias("turn"), pl.len().alias("n_stocks"))
            .filter(pl.col("turn") > 0).with_columns((pl.col("net") / pl.col("turn")).alias("imb")).sort("date"))
    ix = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "vn_indices_daily.parquet")).filter(pl.col("symbol") == "VNINDEX")
          .sort("date").with_columns(pl.col("close").pct_change().alias("r")).select(["date", "r"]).drop_nulls())
    df = ix.join(flow.select(["date", "imb", "n_stocks"]), on="date", how="inner").sort("date")
    df = df.with_columns(pl.col("r").shift(-1).alias("r_next")).drop_nulls("r_next")
    y, x = df["r_next"].to_numpy(), df["imb"].to_numpy()
    fit = hac_ols(y, x, LAGS)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    t_obs, p_placebo = circular_shift_pvalue(y, x, LAGS, n_draws=2000, seed=7)
    ctrl = hac_ols(y, np.column_stack([x, df["r"].to_numpy()]), LAGS)
    beta_c, t_c = float(ctrl.params[1]), float(ctrl.tvalues[1])
    contemp = float(np.corrcoef(df["r"].to_numpy(), x)[0, 1])
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    q = np.quantile(x, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 9)
    means = [y[bins == b].mean() * 100 for b in range(10)]
    ax.bar(range(1, 11), means, color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("foreign flow imbalance decile (1 = heaviest net selling)")
    ax.set_ylabel("mean next-day VN-Index return (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_flow_decile_next_day_return.png"))
    plt.close()

    sample = f"{df['date'].min()} to {df['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: foreign flow imbalance on day $t$ has no relation to the VN-Index return on $t+1$; $H_1$: the relation is positive",
        "Data": f"{len(df):,} sessions, {sample}; imbalance aggregated over up to {int(df['n_stocks'].max())} stocks",
        "Method": f"OLS with Newey--West HAC errors ({LAGS} lags) and a 2,000-draw circular-shift placebo; ACCEPTED only if both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.4f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.4f}$; "
                   f"controlling for the same-day return $t={t_c:.2f}$.",
    }
    body = ("\\section{Design}\nVietnamese exchanges publish foreign buy and sell values at the end of each session. The imbalance is therefore known at the close "
            "of day $t$ and is paired with the return of day $t+1$ only. Sessions with zero reported foreign turnover are vendor gaps and are dropped, never filled. "
            "Foreign ownership limits make flows informative or merely price-following, see \\citet{griffin2004foreign}.\n\n"
            f"\\section{{Results}}\nThe contemporaneous correlation between the imbalance and the same-day return is {contemp:.3f}, so flows largely follow returns. "
            f"With the same-day return as a control the coefficient on the imbalance is {beta_c:.4f} ($t={t_c:.2f}$).\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_flow_decile_next_day_return.png}\n"
            "\\caption{Mean next-day VN-Index return by decile of the foreign flow imbalance.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe 65 stocks are those currently tracked, so delisted names are missing. Flows are aggregated in VND without adjustment for "
            "market capitalisation.\n")
    bib = ("@article{griffin2004foreign,\n  title={Foreign investor flows and stock returns},\n  author={Griffin, John M and Nardari, Federico and Stulz, Ren{\\'e} M},\n"
           "  journal={Review of Financial Studies},\n  volume={17},\n  number={4},\n  pages={1093--1131},\n  year={2004}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "sample": sample, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac, "placebo_p_one_sided": p_placebo, "r2": float(fit.rsquared),
         "beta_with_return_control": beta_c, "t_with_return_control": t_c, "contemporaneous_corr": contemp},
        abstract, body, bib,
        statement="Net foreign buying on a Vietnamese trading day predicts a higher VN-Index return on the next trading day.",
        null_hypothesis="The coefficient of the next-day VN-Index return on the foreign flow imbalance is zero.",
        alt_hypothesis="The coefficient is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="market_microstructure", datasets=["data/market/vn_foreign_trading_daily.parquet", "data/market/vn_indices_daily.parquet"],
        methodology="Predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="beta of next-day VN-Index return on foreign flow imbalance", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.4f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, contemporaneous corr {contemp:.2f}.",
    )


if __name__ == "__main__":
    run()
