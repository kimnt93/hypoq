#!/usr/bin/env python3
"""
res_002: Do Donald Trump's tariff-related Truth Social posts move US equity returns or implied volatility?

H0: S&P 500 |return| and VIX change on the first trading day after a tariff post equal those on ordinary days.
H1: they differ (two-sided).
Posts are matched by keyword only (no sentiment model). The reaction window is strictly AFTER the post date,
so no post-event information leaks into the baseline. The placebo draws random non-event days.
"""
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.db import upsert_hypothesis, record_experiment
from hypoq.report import write_paper, compile_pdf

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
RES_DIR = os.path.join(RESEARCH_DIR, "results")

TITLE = "Tariff-Related Executive Social Media Posts and Next-Day US Equity and Implied Volatility Reactions"
KEYWORD = r"\btariffs?\b"
N_PLACEBO = 5000
BASELINE_WINDOW = 250
ALPHA = 0.05


def load():
    px = (
        pl.read_parquet(os.path.join(DATA_DIR, "market", "us_indices_daily.parquet"))
        .filter(pl.col("symbol") == "^GSPC").select(["date", "returns"]).sort("date")
        .join(pl.read_parquet(os.path.join(DATA_DIR, "macro", "global_cross_asset_volatility_daily.parquet"))
              .select(["date", "vix_equity"]), on="date", how="inner")
        .sort("date")
        .with_columns(
            pl.col("vix_equity").pct_change().alias("dvix"),
            # backward-looking baseline only: mean of the previous 250 sessions, shifted one day
            (pl.col("returns") - pl.col("returns").rolling_mean(BASELINE_WINDOW).shift(1)).alias("ar"),
        )
        .with_columns(pl.col("ar").abs().alias("abs_ar"))
        .drop_nulls()
    )
    posts = pl.read_parquet(os.path.join(DATA_DIR, "community", "global_vip_social_posts_raw.parquet"))
    return px, posts


def placebo(values: np.ndarray, mask: np.ndarray, stat) -> tuple:
    rng = np.random.default_rng(7)
    obs = stat(values[mask])
    pool = values[~mask]
    k = int(mask.sum())
    draws = np.array([stat(rng.choice(pool, size=k, replace=False)) for _ in range(N_PLACEBO)])
    p = float((np.sum(np.abs(draws - draws.mean()) >= abs(obs - draws.mean())) + 1) / (N_PLACEBO + 1))
    return float(obs), p, draws


def run():
    t0 = time.time()
    px, posts = load()
    tariff = posts.filter(pl.col("text").str.to_lowercase().str.contains(KEYWORD))
    first = px["date"].min()
    event_posts = tariff.filter(pl.col("date") >= first)
    # reaction day = first trading day strictly after the post date
    ev = (
        event_posts.select((pl.col("date") + pl.duration(days=1)).alias("date")).unique().sort("date")
        .join_asof(px.select("date").with_columns(pl.col("date").alias("trade_date")), on="date", strategy="forward")
        .drop_nulls().select("trade_date").unique()
    )
    px = px.filter(pl.col("date") >= ev["trade_date"].min())
    mask = px["date"].is_in(ev["trade_date"].to_list()).to_numpy()
    n_ev, n_all = int(mask.sum()), len(px)
    print(f">>> {len(tariff)} tariff posts -> {n_ev} distinct reaction days out of {n_all} sessions")

    out = {}
    draws_by = {}
    for name, col, stat in [("abs_return", "abs_ar", np.mean), ("signed_return", "ar", np.mean), ("vix_change", "dvix", np.mean)]:
        obs, p, draws = placebo(px[col].to_numpy(), mask, stat)
        out[name] = {"event_mean": obs, "baseline_mean": float(px[col].to_numpy()[~mask].mean()), "placebo_p": p}
        draws_by[name] = draws
    bonf = ALPHA / len(out)
    accepted_h1 = any(v["placebo_p"] < bonf for v in out.values())
    decision = "ACCEPTED" if accepted_h1 else "REJECTED"   # ACCEPTED = H1 supported

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200)
    for ax, (name, label) in zip(axes, [("abs_return", "Mean |abnormal return|"), ("signed_return", "Mean abnormal return"),
                                         ("vix_change", "Mean VIX % change")]):
        ax.hist(draws_by[name], bins=40, color="#7f7f7f", alpha=0.7)
        ax.axvline(out[name]["event_mean"], color="#d62728", lw=2)
        ax.set_title(f"{label}\nplacebo p = {out[name]['placebo_p']:.3f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_tariff_event_placebo.png"))
    plt.close()

    rows = "\n".join(
        f"{lbl} & {out[k]['event_mean']*100:.3f} & {out[k]['baseline_mean']*100:.3f} & {out[k]['placebo_p']:.4f} \\\\"
        for k, lbl in [("abs_return", "$|$AR$|$ S\\&P 500 (\\%)"), ("signed_return", "AR S\\&P 500 (\\%)"), ("vix_change", "$\\Delta$VIX (\\%)")]
    )
    table = ("\\begin{table}[h!]\n\\centering\n\\caption{Reaction on the first trading day after a tariff post}\n\\label{tab:event}\n"
             "\\begin{tabular}{lccc}\n\\toprule\nMeasure & Event days & Other days & Placebo $p$ \\\\\n\\midrule\n"
             + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open(os.path.join(RES_DIR, "table1_event_study.tex"), "w") as f:
        f.write(table)

    results = {"title": TITLE, "decision": decision, "n_tariff_posts": len(tariff), "n_event_days": n_ev,
               "n_sessions": n_all, "bonferroni_alpha": bonf, "measures": out,
               "sample_start": str(px["date"].min()), "sample_end": str(px["date"].max())}
    with open(os.path.join(RES_DIR, "empirical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    abstract = {
        "Hypothesis": "$H_0$: next-day S\\&P 500 absolute abnormal return, signed abnormal return and VIX change after a tariff-related "
                      "post equal those of ordinary days; $H_1$: they differ",
        "Data": f"{len(tariff)} Truth Social posts matching the keyword `tariff', {n_ev} distinct reaction days among {n_all:,} sessions "
                f"({px['date'].min()} to {px['date'].max()})",
        "Method": f"Reaction window is the first session strictly after the post; baseline is the trailing {BASELINE_WINDOW}-day mean; "
                  f"{N_PLACEBO}-draw placebo over non-event days; Bonferroni $\\alpha={bonf:.4f}$",
        "Results": "; ".join(f"{k.replace('_', ' ')}: event {v['event_mean']*100:.3f}\\% vs other {v['baseline_mean']*100:.3f}\\% "
                              f"(placebo $p={v['placebo_p']:.3f}$)" for k, v in out.items()) + f". Decision: \\textbf{{{decision}}}",
    }
    body = ("\\section{Design}\nPosts are selected by keyword only, so no text classifier can leak sentiment. Using the session after "
            "the post date guarantees that the measured reaction is not contaminated by the same-day move that may have prompted the post "
            "\\citep{mackinlay1997event}. Event days are not independent when posts cluster; the placebo draws random days "
            "from the same sample rather than assuming normal errors.\n\n\\section{Results}\n\\input{results/table1_event_study.tex}\n\n"
            "\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=\\textwidth]{fig1_tariff_event_placebo.png}\n"
            "\\caption{Observed event-day mean (red) against the placebo distribution.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nNext-day windows discard the intraday reaction. Keyword matching ignores context. Truth Social "
            "coverage starts in 2022, so tariff episodes before that are missing.\n")
    bib = ("@article{mackinlay1997event,\n  title={Event studies in economics and finance},\n  author={MacKinlay, A Craig},\n"
           "  journal={Journal of Economic Literature},\n  volume={35},\n  number={1},\n  pages={13--39},\n  year={1997}\n}\n")
    write_paper(RESEARCH_DIR, TITLE, abstract, body, bib)
    pdf = compile_pdf(RESEARCH_DIR)

    rel = os.path.relpath(RESEARCH_DIR, PROJECT_ROOT)
    best = min(out.values(), key=lambda v: v["placebo_p"])
    hid = upsert_hypothesis(
        title=TITLE,
        statement="Tariff-related executive posts are followed by abnormal next-session S&P 500 returns or VIX changes.",
        null_hypothesis="Mean next-day |AR|, AR and dVIX on event days equal those on non-event days.",
        alt_hypothesis="At least one of the three means differs from the non-event baseline (Bonferroni alpha 0.0167).",
        domain="sentiment_events", session_id="sess_hypoq_examples", agent_name="hypoq-pipeline",
        dataset_dependencies=["data/community/global_vip_social_posts_raw.parquet", "data/market/us_indices_daily.parquet",
                              "data/macro/global_cross_asset_volatility_daily.parquet"],
        methodology="Keyword event study, trailing-mean abnormal returns, placebo resampling",
        metric_name="min placebo p", metric_value=best["placebo_p"], p_value=best["placebo_p"], status=decision,
        summary_findings="; ".join(f"{k}: event {v['event_mean']*100:.3f}% vs {v['baseline_mean']*100:.3f}% (p={v['placebo_p']:.3f})"
                                   for k, v in out.items()),
        report_dir=rel, pdf_path=f"{rel}/main.pdf" if pdf else None,
    )
    record_experiment(hypothesis_id=hid, session_id="sess_hypoq_examples", agent_name="hypoq-pipeline",
                      code_path=f"{rel}/code/run_pipeline.py", duration_seconds=time.time() - t0, metrics=results)
    print(f">>> {decision}; hypothesis #{hid}")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    run()
