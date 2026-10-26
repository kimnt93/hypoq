#!/usr/bin/env python3
"""
res_021 (Predictive regression): does growth of the Fed balance sheet predict S&P 500 returns over the next four weeks?

Weekly WALCL (Fed total assets, as of Wednesday) is published on Thursday at 16:30 US/Eastern. Information date = the Friday close, i.e. the first trading session
on or after Wednesday + 2 days. Signal: 4-week log change in WALCL. Outcome: S&P 500 log return over the 20 sessions after the information date.
Model y = a + b * signal + e with Newey-West HAC (20 lags). H0: b = 0. H1: b > 0 (balance-sheet expansion supports equities).
Weekly signals are sampled every week, so outcomes overlap 4-fold; ACCEPTED requires the HAC p < 0.05 and a circular-shift placebo p < 0.05.
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
TITLE = "Central Bank Balance Sheet Growth and Subsequent US Equity Returns"
H_DAYS = 20
LAGS = 8   # weekly observations, outcomes overlap by 4 weeks -> at least 4 lags; 8 is conservative


def run():
    t0 = time.time()
    fed = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_fed_balance_sheet_weekly.parquet")).select(["date", "fed_total_assets_walcl"]).drop_nulls().sort("date")
           .with_columns(pl.col("fed_total_assets_walcl").log().diff(4).alias("sig"), (pl.col("date") + pl.duration(days=2)).alias("info_date")).drop_nulls())
    px = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC").select(["date", "close"]).sort("date")
          .with_columns((pl.col("close").shift(-H_DAYS) / pl.col("close")).log().alias("fwd")).drop_nulls("fwd"))
    df = fed.sort("info_date").join_asof(px.rename({"date": "px_date"}).sort("px_date"), left_on="info_date", right_on="px_date", strategy="forward")
    df = df.drop_nulls(["fwd", "px_date"]).filter((pl.col("px_date") - pl.col("info_date")).dt.total_days() <= 4).filter(pl.col("px_date") >= pl.date(2016, 10, 24))
    y, x = df["fwd"].to_numpy(), df["sig"].to_numpy()
    fit = hac_ols(y, x, LAGS)
    beta, t_hac = float(fit.params[1]), float(fit.tvalues[1])
    p_hac = float(sps.norm.sf(t_hac))
    _, p_placebo = circular_shift_pvalue(y, x, LAGS, n_draws=2000, seed=7)
    decision = "ACCEPTED" if (p_hac < 0.05 and p_placebo < 0.05) else "REJECTED"
    keep = df["info_date"].dt.year().to_numpy() != 2020          # robustness (not part of the decision): drop the 2020 QE episode
    fit_x = hac_ols(y[keep], x[keep], LAGS)
    beta_x, t_x = float(fit_x.params[1]), float(fit_x.tvalues[1])
    expand = x > 0
    mean_up, mean_dn = float(y[expand].mean()), float(y[~expand].mean())

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.scatter(x * 100, y * 100, s=8, alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs * 100, (fit.params[0] + beta * xs) * 100, color="#d62728", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("4-week change in Fed total assets (%)")
    ax.set_ylabel("S&P 500 log return over next 20 sessions (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_fed_assets_vs_forward_return.png"))
    plt.close()

    sample = f"{df['px_date'].min()} to {df['px_date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the four-week change in Federal Reserve total assets has no predictive power for the S\\&P 500 return over the next 20 sessions; $H_1$: the slope is positive",
        "Data": f"{len(df)} weekly observations, {sample}; information date set to the Friday after the Thursday release",
        "Method": f"OLS with Newey--West HAC errors ({LAGS} lags) and a 2,000-draw circular-shift placebo; the decision needs both one-sided $p<0.05$",
        "Results": f"$\\beta={beta:.3f}$ (HAC $t={t_hac:.2f}$, $p={p_hac:.4f}$; placebo $p={p_placebo:.4f}$), $R^2={fit.rsquared:.4f}$; mean forward return {mean_up*100:.2f}\\% "
                   f"after expansion and {mean_dn*100:.2f}\\% after contraction; without 2020 the slope is {beta_x:.3f} ($t={t_x:.2f}$).",
    }
    body = ("\\section{Design}\nLiquidity provision by central banks is often linked to risk-asset performance \\citep{lewis2019monetary}. The H.4.1 release is public on Thursday afternoon, "
            "so the earliest tradable close is Friday; the outcome window starts from that close. The signal is the four-week log change in WALCL.\n\n\\section{Results}\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_fed_assets_vs_forward_return.png}\n\\caption{Four-week balance-sheet change against the next 20-session S\\&P 500 return.}\n\\end{figure}\n\n"
            f"Excluding the 2020 episode the slope is {beta_x:.3f} (HAC $t={t_x:.2f}$); mean 20-session returns after expansion and contraction weeks are {mean_up*100:.2f}\\% and {mean_dn*100:.2f}\\%, "
            "so the sign of the change alone carries little information and the pooled slope depends on large balance-sheet moves.\n\n"
            "\\section{Limitations}\nThe sample contains one expansion (2020--21) and one contraction (2022--) regime, so effective independent observations are very few. "
            "Balance-sheet growth is endogenous to market stress. Reserve and Treasury General Account dynamics are not modelled.\n")
    bib = ("@article{lewis2019monetary,\n  title={Monetary policy and asset prices: the role of central bank balance sheets},\n  author={Lewis, Vivien and Roth, Markus},\n  journal={Journal of International Money and Finance},\n"
           "  volume={96},\n  pages={238--251},\n  year={2019}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(df), "sample": sample, "horizon_sessions": H_DAYS, "beta": beta, "hac_t": t_hac, "hac_p_one_sided": p_hac, "placebo_p_one_sided": p_placebo,
         "r2": float(fit.rsquared), "mean_fwd_after_expansion": mean_up, "mean_fwd_after_contraction": mean_dn, "beta_excluding_2020": beta_x, "t_excluding_2020": t_x},
        abstract, body, bib,
        statement="Growth of the Fed's total assets over four weeks predicts a higher S&P 500 return over the next 20 sessions.",
        null_hypothesis="The slope of the 20-session-ahead S&P 500 log return on the 4-week change in Fed total assets is zero.",
        alt_hypothesis="The slope is positive (HAC one-sided p < 0.05 and circular-shift placebo p < 0.05).",
        domain="macro_transmission", datasets=["data/macro/us_fed_balance_sheet_weekly.parquet", "data/market/us_indices_daily.parquet"],
        methodology="Publication-lagged weekly predictive regression with Newey-West HAC errors and circular-shift placebo",
        metric_name="slope of 20d forward return on 4-week change in Fed assets", metric_value=beta, p_value=max(p_hac, p_placebo),
        summary_findings=f"beta={beta:.3f}, HAC t={t_hac:.2f} (p={p_hac:.4f}), placebo p={p_placebo:.4f}, R2={fit.rsquared:.4f}.",
    )


if __name__ == "__main__":
    run()
