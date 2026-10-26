#!/usr/bin/env python3
"""
res_001: Cross-border equity return transmission (US -> Europe, US -> Japan).

H0: the previous US session carries no information about the next European / Japanese session (beta = 0).
H1: beta > 0.
Every number in tables, figures, the paper and the database is computed here.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.db import upsert_hypothesis, record_experiment, DEFAULT_DB_PATH
from hypoq.report import write_paper, compile_pdf

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
RES_DIR = os.path.join(RESEARCH_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

TITLE = "Cross-Border Return Transmission from the US Session to European and Japanese Equity Markets"
ALPHA = 0.05
N_PLACEBO = 2000
HAC_LAGS = 5


def load_returns() -> pl.DataFrame:
    def one(path, symbol, name):
        return (
            pl.read_parquet(os.path.join(DATA_DIR, "market", path))
            .filter(pl.col("symbol") == symbol)
            .select(["date", "returns"])
            .rename({"returns": name})
        )

    return (
        one("us_indices_daily.parquet", "^GSPC", "us")
        .join(one("eu_indices_daily.parquet", "^STOXX50E", "eu"), on="date", how="inner")
        .join(one("asia_indices_daily.parquet", "^N225", "jp"), on="date", how="inner")
        .sort("date")
        .drop_nulls()
    )


def lead_lag(y: np.ndarray, x: np.ndarray):
    """OLS of y_t on x_{t-1} with Newey-West HAC errors, plus a circular-shift placebo for the t-stat."""
    yy, xx = y[1:], x[:-1]
    fit = sm.OLS(yy, sm.add_constant(xx)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    rng = np.random.default_rng(42)
    placebo_t = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        shift = int(rng.integers(HAC_LAGS + 10, len(xx) - HAC_LAGS - 10))
        f = sm.OLS(yy, sm.add_constant(np.roll(xx, shift))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        placebo_t[i] = f.tvalues[1]
    # one-sided: H1 is beta > 0
    one_sided_p = float(fit.pvalues[1] / 2) if fit.params[1] > 0 else 1.0 - float(fit.pvalues[1] / 2)
    placebo_p = float((np.sum(placebo_t >= fit.tvalues[1]) + 1) / (N_PLACEBO + 1))
    return {
        "beta": float(fit.params[1]), "t": float(fit.tvalues[1]), "p": one_sided_p,
        "r2": float(fit.rsquared), "n": int(len(yy)), "placebo_p": placebo_p,
    }, placebo_t


def fevd_spillover(data: np.ndarray, order) -> tuple:
    """Diebold-Yilmaz total spillover index (%) from a Cholesky FEVD at horizon 10 for a given ordering."""
    res = VAR(data[:, order]).fit(maxlags=5, ic="bic")
    decomp = res.fevd(10).decomp[:, -1, :]  # (variable, shock) at last horizon
    off_diag = decomp.sum() - np.trace(decomp)
    return float(100.0 * off_diag / decomp.sum()), int(res.k_ar)


def run():
    t0 = time.time()
    df = load_returns()
    us, eu, jp = (df[c].to_numpy() for c in ("us", "eu", "jp"))
    print(f">>> {len(df):,} aligned sessions {df['date'].min()} -> {df['date'].max()}")

    res_jp, placebo_jp = lead_lag(jp, us)
    res_eu, _ = lead_lag(eu, us)

    mat = df.select(["us", "eu", "jp"]).to_numpy()
    si_fwd, lag_fwd = fevd_spillover(mat, [0, 1, 2])
    si_rev, _ = fevd_spillover(mat, [2, 1, 0])

    bonferroni = ALPHA / 2
    accepted = all(r["p"] < bonferroni and r["placebo_p"] < ALPHA for r in (res_jp, res_eu))
    decision = "ACCEPTED" if accepted else "REJECTED"

    # Figure: Japan(t) vs US(t-1) with the placebo t distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    axes[0].scatter(us[:-1] * 100, jp[1:] * 100, s=6, alpha=0.3, color="#1f77b4")
    xs = np.linspace(us.min(), us.max(), 50) * 100
    axes[0].plot(xs, res_jp["beta"] * xs, color="#d62728", lw=2)
    axes[0].set_xlabel("S&P 500 return, t-1 [%]")
    axes[0].set_ylabel("Nikkei 225 return, t [%]")
    axes[0].set_title("Lead-lag regression")
    axes[1].hist(placebo_jp, bins=40, color="#7f7f7f", alpha=0.7)
    axes[1].axvline(res_jp["t"], color="#d62728", lw=2, label=f"observed t = {res_jp['t']:.2f}")
    axes[1].set_xlabel("HAC t-statistic under circular-shift placebo")
    axes[1].set_title(f"Placebo distribution ({N_PLACEBO} shifts)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_volatility_spillover.png"))
    plt.close()

    def row(label, r):
        return (f"{label} & {r['beta']:.4f} & {r['t']:.2f} & {r['p']:.2e} & {r['placebo_p']:.4f} "
                f"& {r['r2']:.4f} & {r['n']:,} \\\\")

    table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Next-session response to the prior US session (HAC Newey--West, one-sided $H_1:\\beta>0$)}\n"
        "\\label{tab:spillover}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Target & $\\beta$ & HAC $t$ & $p$ & Placebo $p$ & $R^2$ & $N$ \\\\\n\\midrule\n"
        + row("Nikkei 225 ($t$)", res_jp) + "\n" + row("Euro Stoxx 50 ($t$)", res_eu)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(RES_DIR, "table1_volatility_spillover.tex"), "w") as f:
        f.write(table)

    results = {
        "title": TITLE, "decision": decision, "alpha_bonferroni": bonferroni,
        "japan": res_jp, "europe": res_eu,
        "fevd_total_spillover_pct": {"order_us_eu_jp": si_fwd, "order_jp_eu_us": si_rev},
        "var_lag_order": lag_fwd, "sample_start": str(df["date"].min()), "sample_end": str(df["date"].max()),
    }
    with open(os.path.join(RES_DIR, "empirical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    abstract = {
        "Hypothesis": "$H_0:\\beta_{\\text{US}\\to\\text{Asia/Europe}}=0$ versus $H_1:\\beta>0$ for the next-session response "
                      "of the Nikkei 225 and Euro Stoxx 50 to the prior S\\&P 500 return",
        "Data": f"Daily index returns, {df['date'].min()} to {df['date'].max()}, {len(df):,} sessions common to all three markets",
        "Method": f"OLS with Newey--West HAC errors ({HAC_LAGS} lags), a {N_PLACEBO}-draw circular-shift placebo, "
                  f"and a Cholesky-FEVD spillover index from a BIC-selected VAR({lag_fwd})",
        "Results": f"Nikkei: $\\beta={res_jp['beta']:.3f}$, $t={res_jp['t']:.2f}$, placebo $p={res_jp['placebo_p']:.4f}$; "
                   f"Euro Stoxx: $\\beta={res_eu['beta']:.3f}$, $t={res_eu['t']:.2f}$, placebo $p={res_eu['placebo_p']:.4f}$. "
                   f"Decision threshold: Bonferroni $\\alpha={bonferroni}$.",
    }
    body = (
        "\\section{Hypothesis and Design}\n"
        "Asian and European sessions close before the US session, so the previous US return is strictly predetermined "
        "\\citep{engle1990meteor}. The test is a one-sided HAC regression with a circular-shift placebo that preserves "
        "serial dependence while breaking the cross-market alignment.\n\n"
        "\\section{Results}\n"
        f"The FEVD total spillover index is {si_fwd:.1f}\\% with ordering US, EU, JP and {si_rev:.1f}\\% with the reverse "
        "ordering; Cholesky FEVD is order dependent, so the gap between the two is reported as sensitivity "
        "\\citep{diebold2012better}.\n\n"
        "\\input{results/table1_volatility_spillover.tex}\n\n"
        "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{fig1_volatility_spillover.png}\n"
        "\\caption{Lead-lag fit and placebo distribution.}\n\\end{figure}\n\n"
        "\\section{Limitations}\n"
        "Only sessions present in all three calendars are used; no transaction costs or trading rule are evaluated; "
        "returns rather than realised volatility are modelled.\n"
    )
    bib = open(os.path.join(RESEARCH_DIR, "references.bib")).read() if os.path.exists(
        os.path.join(RESEARCH_DIR, "references.bib")) else ""
    write_paper(RESEARCH_DIR, TITLE, abstract, body, bib)
    pdf = compile_pdf(RESEARCH_DIR)

    rel = os.path.relpath(RESEARCH_DIR, PROJECT_ROOT)
    hid = upsert_hypothesis(
        title=TITLE,
        statement="The previous US session predicts the next European and Japanese session returns.",
        null_hypothesis="beta = 0 for US(t-1) -> JP(t) and US(t-1) -> EU(t).",
        alt_hypothesis="beta > 0 for both targets (Bonferroni alpha 0.025 and placebo p < 0.05).",
        domain="macro_transmission",
        session_id="sess_hypoq_examples",
        agent_name="hypoq-pipeline",
        dataset_dependencies=[
            "data/market/us_indices_daily.parquet", "data/market/eu_indices_daily.parquet",
            "data/market/asia_indices_daily.parquet",
        ],
        methodology="HAC lead-lag OLS, circular-shift placebo, Cholesky FEVD spillover index",
        metric_name="HAC t-stat (Nikkei)", metric_value=res_jp["t"], p_value=res_jp["p"], status=decision,
        summary_findings=(f"Nikkei beta={res_jp['beta']:.3f} (t={res_jp['t']:.2f}); "
                          f"Euro Stoxx beta={res_eu['beta']:.3f} (t={res_eu['t']:.2f}); FEVD spillover {si_fwd:.1f}%."),
        report_dir=rel, pdf_path=f"{rel}/main.pdf" if pdf else None,
    )
    record_experiment(
        hypothesis_id=hid, session_id="sess_hypoq_examples", agent_name="hypoq-pipeline",
        code_path=f"{rel}/code/run_pipeline.py", duration_seconds=time.time() - t0,
        metrics={"japan": res_jp, "europe": res_eu, "fevd_pct": si_fwd},
    )
    print(f">>> {decision}: Nikkei t={res_jp['t']:.2f}, Euro Stoxx t={res_eu['t']:.2f}; hypothesis #{hid}")


if __name__ == "__main__":
    run()
