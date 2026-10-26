#!/usr/bin/env python3
"""
res_016 (Calendar effect): is the S&P 500 return higher around the turn of the month?

Turn-of-month (ToM) sessions: the last trading session of each month and the first three sessions of the next month (window fixed in advance, Ariel / Lakonishok-Smidt).
Statistic: mean daily S&P 500 return on ToM sessions. Placebo: 10,000 draws of equally many non-ToM sessions. H0: equal. H1: ToM mean is larger (one-sided p < 0.05).
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

# --- metric helpers copied from .agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py (hypoq/ is frozen; each study is self-contained) ---

def perm_placebo_mean(event_values: np.ndarray, pool: np.ndarray, n_draws: int = 10000, seed: int = 5) -> Tuple[float, float]:
    """Mean of `event_values` vs. means of equally sized random draws from `pool` of non-event values (one-sided p: larger)."""
    sims = np.random.default_rng(seed).choice(pool, size=(n_draws, len(event_values))).mean(axis=1)
    return float(sims.mean()), float((np.sum(sims >= event_values.mean()) + 1) / (n_draws + 1))

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "The Turn-of-the-Month Effect in the US Stock Index, 2016--2026"


def run():
    t0 = time.time()
    d = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC")
         .select(["date", "returns"]).drop_nulls().sort("date")
         .with_columns((pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("ym")))
    d = d.with_columns(pl.col("date").rank("ordinal").over("ym").alias("pos"), pl.len().over("ym").alias("n"))
    # month-edge sessions only count when the month is complete at both ends of the data
    first_ym, last_ym = int(d["ym"].min()), int(d["ym"].max())
    d = d.filter((pl.col("ym") > first_ym) & (pl.col("ym") < last_ym))
    d = d.with_columns(((pl.col("pos") <= 3) | (pl.col("pos") == pl.col("n"))).alias("tom"))
    tom, rest = d.filter(pl.col("tom"))["returns"].to_numpy(), d.filter(~pl.col("tom"))["returns"].to_numpy()
    placebo_mean, p = perm_placebo_mean(tom, rest, n_draws=10000, seed=5)
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"
    share_days = float(len(tom) / len(d))
    total = float(d["returns"].sum())
    share_ret = float(tom.sum() / total) if total != 0 else float("nan")
    by_year = (d.with_columns(pl.col("date").dt.year().alias("year")).group_by(["year", "tom"]).agg(pl.col("returns").mean()).pivot(on="tom", index="year", values="returns").sort("year"))

    sims = np.random.default_rng(5).choice(rest, size=(10000, len(tom))).mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.hist(sims * 100, bins=40, color="#7f7f7f", alpha=0.75, label="placebo (other sessions)")
    ax.axvline(tom.mean() * 100, color="#d62728", lw=2, label=f"turn-of-month: {tom.mean()*100:.3f}%")
    ax.set_xlabel("mean daily return (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_tom_placebo.png"))
    plt.close()

    rws = "\n".join(f"{int(r['year'])} & {r['true']*100:.3f}\\% & {r['false']*100:.3f}\\% \\\\" for r in by_year.iter_rows(named=True) if r.get("true") is not None and r.get("false") is not None)
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Mean daily S\\&P 500 return by year}\n\\begin{tabular}{lcc}\n\\toprule\n"
             "Year & Turn-of-month & Other sessions \\\\\n\\midrule\n" + rws + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_tom_by_year.tex"), "w") as f:
        f.write(table)
    sample = f"{d['date'].min()} to {d['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: the mean daily S\\&P 500 return on turn-of-month sessions equals that of other sessions; $H_1$: it is larger",
        "Data": f"{len(d):,} sessions, {sample}; {len(tom)} turn-of-month sessions ({share_days*100:.1f}\\% of days)",
        "Method": "Last session of the month plus the first three of the next; 10,000-draw placebo over other sessions",
        "Results": f"Mean {tom.mean()*100:.3f}\\% versus placebo {placebo_mean*100:.3f}\\% (one-sided $p={p:.4f}$); the window carries {share_ret*100:.0f}\\% of the cumulative log-return sum.",
    }
    body = ("\\section{Design}\nThe window is the classic one of \\citet{lakonishok1988turn}, fixed in advance. The first and last month of the data are dropped because their edges are incomplete. "
            "The placebo draws the same number of sessions from outside the window, which keeps the volatility regime matched on average.\n\n\\section{Results}\n"
            "\\input{results/table1_tom_by_year.tex}\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_tom_placebo.png}\n\\caption{Turn-of-month mean return against the placebo distribution.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nA calendar effect documented in the literature may be arbitraged away; ten years contain about 120 windows. Placebo draws ignore serial dependence, "
            "which is small for daily index returns.\n")
    bib = ("@article{lakonishok1988turn,\n  title={Are seasonal anomalies real? A ninety-year perspective},\n  author={Lakonishok, Josef and Smidt, Seymour},\n  journal={Review of Financial Studies},\n"
           "  volume={1},\n  number={4},\n  pages={403--425},\n  year={1988}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_sessions": len(d), "n_tom": int(len(tom)), "sample": sample, "mean_tom": float(tom.mean()), "mean_other": float(rest.mean()), "placebo_mean": placebo_mean,
         "p_one_sided": p, "share_of_days": share_days, "share_of_return": share_ret},
        abstract, body, bib,
        statement="S&P 500 returns are higher on the last session of the month and the first three sessions of the next month.",
        null_hypothesis="Mean daily S&P 500 return on turn-of-month sessions equals the mean on other sessions.",
        alt_hypothesis="The turn-of-month mean is larger (one-sided placebo p < 0.05).",
        domain="fundamental_factors", datasets=["data/market/us_indices_daily.parquet"],
        methodology="Calendar-window placebo test with 10,000 draws of non-window sessions",
        metric_name="mean daily return on turn-of-month sessions", metric_value=float(tom.mean()), p_value=p,
        summary_findings=f"ToM mean {tom.mean()*100:.3f}% vs placebo {placebo_mean*100:.3f}% over {len(tom)} sessions (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
