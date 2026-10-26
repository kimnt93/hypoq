#!/usr/bin/env python3
"""
res_020 (Calendar effect): do crypto returns differ by day of the week?

y_t = equal-weight daily return of the crypto assets available on day t (data/crypto/crypto_top_assets_daily.parquet, 24/7 market, UTC daily bars).
Model: y_t = a + sum_{d=2..7} b_d * 1[weekday_t = d] + e (Monday is the base), Newey-West HAC with 7 lags.
H0: b_2 = ... = b_7 = 0 (all weekdays have the same mean). H1: at least one differs. Joint Wald test (chi-square, 6 df), p < 0.05.
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

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Day-of-the-Week Effects in Cryptocurrency Returns"
LAGS = 7
NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def run():
    t0 = time.time()
    d = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "crypto", "crypto_top_assets_daily.parquet")).select(["date", "symbol", "returns"]).drop_nulls("returns")
    ew = d.group_by("date").agg(pl.col("returns").mean().alias("r"), pl.len().alias("n")).sort("date").filter(pl.col("n") >= 3)
    ew = ew.with_columns((pl.col("date").dt.weekday() - 1).alias("wd"))
    y = ew["r"].to_numpy()
    wd = ew["wd"].to_numpy()
    X = np.column_stack([(wd == k).astype(float) for k in range(1, 7)])
    fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
    R = np.zeros((6, 7))
    R[np.arange(6), np.arange(1, 7)] = 1.0
    w = fit.wald_test(R, use_f=False, scalar=True)
    stat, p = float(w.statistic), float(w.pvalue)
    means = [float(y[wd == k].mean()) for k in range(7)]
    best, worst = int(np.argmax(means)), int(np.argmin(means))
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.bar(NAMES, [m * 100 for m in means], color="#1f77b4")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean daily return of the equal-weight crypto basket (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_crypto_day_of_week.png"))
    plt.close()

    rws = "\n".join(f"{NAMES[k]} & {int((wd == k).sum())} & {means[k]*100:.3f}\\% & {float(y[wd == k].std(ddof=1))*100:.2f}\\% \\\\" for k in range(7))
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Equal-weight crypto basket return by weekday (UTC)}\n\\begin{tabular}{lccc}\n\\toprule\n"
             "Weekday & Days & Mean & Std. dev. \\\\\n\\midrule\n" + rws + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_crypto_weekday.tex"), "w") as f:
        f.write(table)
    sample = f"{ew['date'].min()} to {ew['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the mean daily return of an equal-weight crypto basket is the same on all weekdays; $H_1$: at least one weekday differs",
        "Data": f"{len(ew):,} days with at least three assets, {sample}",
        "Method": f"Weekday dummies with Newey--West HAC errors ({LAGS} lags) and a joint Wald test of the six dummies",
        "Results": f"Wald $\\chi^2_6={stat:.2f}$, $p={p:.4f}$; highest mean {NAMES[best]} ({means[best]*100:.3f}\\%), lowest {NAMES[worst]} ({means[worst]*100:.3f}\\%).",
    }
    body = ("\\section{Design}\nCrypto markets trade every day, so weekday patterns can arise from the weekly rhythm of institutional and retail activity \\citep{caporale2019day}. "
            "The test is a single joint hypothesis on all weekdays rather than a search for the best day, which avoids data-snooping across seven comparisons.\n\n\\section{Results}\n"
            "\\input{results/table1_crypto_weekday.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_crypto_day_of_week.png}\n\\caption{Mean daily return by weekday.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nDaily bars close at 00:00 UTC, so the weekday label follows UTC rather than any local trading day. The basket includes more assets in later years. "
            "The decade has a few extreme days, and the joint test is not robust to a single outlier driving one weekday's mean.\n")
    bib = ("@article{caporale2019day,\n  title={Day of the week effect in cryptocurrency markets},\n  author={Caporale, Guglielmo Maria and Plastun, Alex},\n  journal={Finance Research Letters},\n"
           "  volume={31},\n  pages={258--269},\n  year={2019}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_obs": len(ew), "sample": sample, "wald_chi2": stat, "wald_p": p, "weekday_means": dict(zip(NAMES, means)), "best_day": NAMES[best], "worst_day": NAMES[worst]},
        abstract, body, bib,
        statement="The mean daily return of an equal-weight crypto basket differs across weekdays.",
        null_hypothesis="All six weekday-dummy coefficients are zero (equal mean return on every weekday).",
        alt_hypothesis="At least one coefficient differs from zero (HAC joint Wald p < 0.05).",
        domain="crypto", datasets=["data/crypto/crypto_top_assets_daily.parquet"],
        methodology="Weekday-dummy regression with Newey-West HAC errors and a joint Wald test",
        metric_name="HAC Wald chi-square (6 df) for equal weekday means", metric_value=stat, p_value=p,
        summary_findings=f"Wald chi2={stat:.2f} (p={p:.4f}); best {NAMES[best]} {means[best]*100:.3f}%, worst {NAMES[worst]} {means[worst]*100:.3f}%.",
    )


if __name__ == "__main__":
    run()
