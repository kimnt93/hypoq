#!/usr/bin/env python3
"""
res_007 (Event study): are S&P 500 moves larger on FOMC announcement days than on ordinary sessions?

Events: scheduled and unscheduled FOMC statements from the Federal Reserve press-release feed (data/macro/us_fed_fomc_rate_decisions_pit.parquet).
The statement is released at 14:00 US/Eastern, before the 16:00 close, so day 0 is the release date in US/Eastern.
Statistic: mean |S&P 500 close-to-close return| on FOMC days. Placebo: 10,000 random draws of equally many non-FOMC sessions
(sessions within one day of a meeting are excluded from the pool). H0: equal. H1: larger (one-sided).
Secondary (descriptive, Bonferroni-corrected with the primary test): the change in VIX on FOMC days.
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
from hypoq.report import latex_escape

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Federal Reserve Announcement Days and US Equity Volatility: An Event Study of FOMC Statements, 2016--2026"
N_DRAWS = 10000


# --- inference helpers (self-contained: the hypoq package is frozen; see the hypothesis-testing-stats skill) ---

def perm_placebo_mean(event_values: np.ndarray, pool: np.ndarray, n_draws: int = 10000, seed: int = 5) -> Tuple[float, float]:
    """Mean of `event_values` vs. the distribution of means of equally sized random draws from `pool` (one-sided p: larger)."""
    rng = np.random.default_rng(seed)
    k = len(event_values)
    sims = rng.choice(pool, size=(n_draws, k)).mean(axis=1)
    obs = float(event_values.mean())
    return float(sims.mean()), float((np.sum(sims >= obs) + 1) / (n_draws + 1))

def run():
    t0 = time.time()
    fm = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_fed_fomc_rate_decisions_pit.parquet"))
    et = pl.col("announced_at").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York").dt.date()
    fm = fm.with_columns(et.alias("day0")).select(["day0", "action", "rate_change_bps"])

    spx = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet"))
           .filter(pl.col("symbol") == "^GSPC").select(["date", "returns"]).drop_nulls().sort("date"))
    vix = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_yield_curve_daily.parquet")).select(["date", "cboe_vix"]).drop_nulls().sort("date")
    df = (spx.join(vix, on="date", how="left").sort("date")
          .with_columns(pl.col("cboe_vix").pct_change().alias("dvix"), pl.col("returns").abs().alias("absret"))
          .with_columns(pl.col("date").is_in(fm["day0"].to_list()).alias("is_event")))
    n_events_listed = fm.height
    dates = df["date"].to_list()
    ev_idx = [i for i, e in enumerate(df["is_event"].to_list()) if e]
    near = set()
    for i in ev_idx:
        near.update(range(i - 1, i + 2))
    mask_pool = np.array([i not in near for i in range(len(dates))])
    ev = df.filter(pl.col("is_event")).drop_nulls(["returns", "dvix"])
    pool_abs = df["absret"].to_numpy()[mask_pool]
    pool_dvix = df["dvix"].to_numpy()[mask_pool]
    pool_ok = ~np.isnan(pool_abs) & ~np.isnan(pool_dvix)
    pool_abs, pool_dvix = pool_abs[pool_ok], pool_dvix[pool_ok]
    n_ev = ev.height

    ev_abs = ev["absret"].to_numpy()
    ev_dvix = ev["dvix"].to_numpy()
    placebo_abs, p_abs = perm_placebo_mean(ev_abs, pool_abs, N_DRAWS, seed=5)
    placebo_dvix, p_dvix_low = perm_placebo_mean(-ev_dvix, -pool_dvix, N_DRAWS, seed=6)  # H1: VIX falls more on FOMC days
    alpha_each = 0.05 / 2
    decision = "ACCEPTED" if p_abs < alpha_each else "REJECTED"

    ev_signed = ev["returns"].to_numpy()
    t_signed = float(sps.ttest_1samp(ev_signed, 0.0).statistic)
    by_action = (ev.join(fm.unique("day0"), left_on="date", right_on="day0").group_by("action")
                 .agg(pl.len().alias("n"), pl.col("absret").mean().alias("abs"), pl.col("returns").mean().alias("ret"),
                      pl.col("dvix").mean().alias("dvix")).sort("n", descending=True))

    sims = np.random.default_rng(5).choice(pool_abs, size=(N_DRAWS, n_ev)).mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.hist(sims * 100, bins=40, color="#7f7f7f", alpha=0.75, label="placebo (ordinary sessions)")
    ax.axvline(ev_abs.mean() * 100, color="#d62728", lw=2, label=f"FOMC days: {ev_abs.mean()*100:.2f}%")
    ax.set_xlabel("mean |S&P 500 return| (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_fomc_abs_return_placebo.png"))
    plt.close()

    rows = "\n".join(f"{latex_escape(a)} & {n} & {ab*100:.2f}\\% & {r*100:.2f}\\% & {d*100:.2f}\\% \\\\" for a, n, ab, r, d in by_action.iter_rows())
    table = ("\\begin{table}[H]\n\\centering\n\\caption{S\\&P 500 and VIX on FOMC announcement days by policy action}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Action & Events & Mean $|$return$|$ & Mean return & Mean $\\Delta$VIX \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_fomc_by_action.tex"), "w") as f:
        f.write(table)

    sample = f"{ev['date'].min()} to {ev['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: mean $|$return$|$ of the S\\&P 500 on FOMC announcement days equals that of ordinary sessions; $H_1$: it is larger",
        "Data": f"{n_ev} FOMC statements ({n_events_listed} listed) matched to S\\&P 500 and VIX closes, {sample}",
        "Method": f"{N_DRAWS:,}-draw placebo over ordinary sessions (meeting $\\pm1$ day excluded); Bonferroni $\\alpha=0.025$ for two outcomes",
        "Results": f"Mean $|$return$|$ {ev_abs.mean()*100:.3f}\\% versus placebo {placebo_abs*100:.3f}\\% (one-sided $p={p_abs:.4f}$); "
                   f"mean $\\Delta$VIX {ev_dvix.mean()*100:.2f}\\% versus placebo {-placebo_dvix*100:.2f}\\% ($p={p_dvix_low:.4f}$ for a larger fall). "
                   f"",
    }
    body = ("\\section{Design}\nFOMC statements are released at 14:00 US/Eastern, two hours before the cash close, so the close-to-close return of the release "
            "date contains the full announcement reaction together with 5.5 hours of pre-announcement trading, which dilutes the measured effect. "
            "The placebo draws ordinary sessions from the same sample, so volatility regimes are matched on average "
            "\\citep{lucca2015premium}.\n\n\\section{Results}\n\\input{results/table1_fomc_by_action.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_fomc_abs_return_placebo.png}\n"
            "\\caption{Mean absolute S\\&P 500 return on FOMC days against the placebo distribution.}\n\\end{figure}\n\n"
            f"The signed mean return on FOMC days is {ev_signed.mean()*100:.3f}\\% (t={t_signed:.2f}).\n\n"
            "\\section{Limitations}\nThe sample is short (about ten years, one tightening cycle) and clustered in 2020 when unscheduled meetings coincided with the "
            "COVID shock; a regime-matched placebo is left to follow-up work. Intraday data would isolate the announcement window but are not free.\n")
    bib = ("@article{lucca2015premium,\n  title={The pre-FOMC announcement drift},\n  author={Lucca, David O and Moench, Emanuel},\n"
           "  journal={The Journal of Finance},\n  volume={70},\n  number={1},\n  pages={329--371},\n  year={2015}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_events": n_ev, "sample": sample, "mean_abs_return_event": float(ev_abs.mean()), "mean_abs_return_placebo": placebo_abs,
         "p_abs_return_one_sided": p_abs, "mean_dvix_event": float(ev_dvix.mean()), "mean_dvix_placebo": -placebo_dvix,
         "p_dvix_larger_fall_one_sided": p_dvix_low, "signed_mean_return": float(ev_signed.mean()), "signed_t": t_signed,
         "by_action": by_action.to_dicts()}, abstract, body, bib,
        statement="The S&P 500 moves more on FOMC announcement days than on ordinary sessions.",
        null_hypothesis="Mean |S&P 500 return| on FOMC announcement days equals the mean on ordinary sessions.",
        alt_hypothesis="Mean |S&P 500 return| on FOMC days is larger (one-sided placebo p < 0.025 after Bonferroni for two outcomes).",
        domain="macro_transmission",
        datasets=["data/macro/us_fed_fomc_rate_decisions_pit.parquet", "data/market/us_indices_daily.parquet", "data/macro/us_yield_curve_daily.parquet"],
        methodology="Event study with ordinary-session placebo resampling (10,000 draws) and Bonferroni correction",
        metric_name="mean |S&P 500 return| on FOMC days", metric_value=float(ev_abs.mean()), p_value=p_abs,
        summary_findings=f"FOMC-day mean |return| {ev_abs.mean()*100:.3f}% vs placebo {placebo_abs*100:.3f}% over {n_ev} meetings (p={p_abs:.4f}).",
    )


if __name__ == "__main__":
    run()
