#!/usr/bin/env python3
"""
res_006 (Event study): do SEC 8-K Item 2.02 (results of operations) filings move US mega-cap prices?

Events: 8-K filings with Item 2.02 for the US equities in the data lake. Day 0 is the first trading session whose close
follows the EDGAR acceptance time (accepted before 16:00 US/Eastern on a trading day -> that day, otherwise the next session).
Market model: OLS of the stock on the S&P 500 over trading days [-250, -31], estimated strictly before the event.
Statistic: mean |CAR[0,+1]|. Placebo: for every event draw a random non-event session of the same stock and compute the same CAR;
repeat 5,000 times. H0: |CAR| on filing days equals |CAR| on ordinary days. H1: it is larger (one-sided).
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Earnings Filings and Price Reactions: An Event Study of SEC 8-K Item 2.02 Announcements for US Mega-Caps"
EST_START, EST_END = 250, 31
N_PLACEBO = 5000
EXCLUDE_DAYS = 3


def abnormal_returns() -> pl.DataFrame:
    eq = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_equities_daily.parquet")).select(["date", "symbol", "returns"]).drop_nulls()
    mk = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet"))
          .filter(pl.col("symbol") == "^GSPC").select(["date", pl.col("returns").alias("rm")]).drop_nulls())
    df = eq.join(mk, on="date").sort(["symbol", "date"])
    n = EST_START - EST_END + 1
    # estimation window for day t = observations t-250 .. t-31: shift by 31 then roll 220 (all strictly before t)
    r_s, m_s = pl.col("returns").shift(EST_END), pl.col("rm").shift(EST_END)
    df = df.with_columns(
        r_s.rolling_mean(n).over("symbol").alias("mr"),
        m_s.rolling_mean(n).over("symbol").alias("mm"),
        (r_s * m_s).rolling_mean(n).over("symbol").alias("mrm"),
        (m_s * m_s).rolling_mean(n).over("symbol").alias("mmm"),
    ).with_columns(
        ((pl.col("mrm") - pl.col("mr") * pl.col("mm")) / (pl.col("mmm") - pl.col("mm") ** 2)).alias("beta"),
    ).with_columns((pl.col("mr") - pl.col("beta") * pl.col("mm")).alias("alpha")).with_columns(
        (pl.col("returns") - pl.col("alpha") - pl.col("beta") * pl.col("rm")).alias("ar")
    )
    return df.with_columns((pl.col("ar") + pl.col("ar").shift(-1).over("symbol")).alias("car01")).select(["date", "symbol", "ar", "car01"])


def event_days(ar: pl.DataFrame) -> pl.DataFrame:
    f = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "community", "us_sec_8k_filings.parquet"))
    f = f.filter((pl.col("form") == "8-K") & pl.col("items").str.contains(r"(^|,)2\.02(,|$)")).drop_nulls("datetime_utc")
    et = pl.col("datetime_utc").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    f = f.with_columns(et.alias("et")).with_columns(
        pl.when(pl.col("et").dt.hour() >= 16).then(pl.col("et").dt.date() + pl.duration(days=1)).otherwise(pl.col("et").dt.date()).alias("target")
    )
    days = ar.select(["symbol", pl.col("date").alias("session")]).sort(["symbol", "session"])
    ev = f.sort("target").join_asof(days.sort("session"), left_on="target", right_on="session", by="symbol", strategy="forward")
    return ev.drop_nulls("session").select(["symbol", pl.col("session").alias("date"), "datetime_utc", "accession_number"]).unique(subset=["symbol", "date"])


def run():
    t0 = time.time()
    ar = abnormal_returns()
    ev = event_days(ar)
    obs = ev.join(ar, on=["symbol", "date"], how="inner").drop_nulls("car01")
    n_ev = obs.height
    print(f">>> {n_ev} Item 2.02 events across {obs['symbol'].n_unique()} stocks, {obs['date'].min()} .. {obs['date'].max()}")
    abs_obs = float(obs["car01"].abs().mean())
    signed_mean = float(obs["car01"].mean())
    t_signed = float(stats.ttest_1samp(obs["car01"].to_numpy(), 0.0).statistic)

    # placebo pools: non-event sessions of the same stock, at least 3 sessions away from any event
    pools = {}
    ev_by_sym = {s: set(g["date"].to_list()) for s, g in ev.group_by("symbol")}
    for sym, g in ar.drop_nulls("car01").group_by("symbol"):
        sym = sym[0]
        dates = g["date"].to_list()
        ev_idx = {i for i, d in enumerate(dates) if d in ev_by_sym.get(sym, set())}
        bad = {j for i in ev_idx for j in range(i - EXCLUDE_DAYS, i + EXCLUDE_DAYS + 1)}
        pools[sym] = np.abs(np.array([c for i, c in enumerate(g["car01"].to_list()) if i not in bad]))
    counts = obs.group_by("symbol").len()
    rng = np.random.default_rng(3)
    sims = np.zeros(N_PLACEBO)
    for sym, k in counts.iter_rows():
        pool = pools[sym]
        sims += rng.choice(pool, size=(N_PLACEBO, k)).sum(axis=1)
    sims /= n_ev
    p = float((np.sum(sims >= abs_obs) + 1) / (N_PLACEBO + 1))
    placebo_mean = float(sims.mean())
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.hist(sims * 100, bins=40, color="#7f7f7f", alpha=0.75, label="placebo (non-event days)")
    ax.axvline(abs_obs * 100, color="#d62728", lw=2, label=f"8-K 2.02 days: {abs_obs*100:.2f}%")
    ax.set_xlabel("mean |CAR[0,+1]| (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_earnings_car_placebo.png"))
    plt.close()

    by_year = (obs.with_columns(pl.col("date").dt.year().alias("year")).group_by("year")
               .agg(pl.len().alias("n"), pl.col("car01").abs().mean().alias("abs_car")).sort("year"))
    rows = "\n".join(f"{y} & {n} & {a*100:.2f}\\% \\\\" for y, n, a in by_year.iter_rows())
    table = ("\\begin{table}[h!]\n\\centering\n\\caption{Mean $|$CAR$[0,+1]|$ of Item 2.02 filings by year}\n\\begin{tabular}{lcc}\n\\toprule\n"
             "Year & Events & Mean $|$CAR$|$ \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_event_study_car.tex"), "w") as f:
        f.write(table)

    sample = f"{obs['date'].min()} to {obs['date'].max()}"
    abstract = {
        "Hypothesis": "$H_0$: mean $|$CAR$[0,+1]|$ on 8-K Item 2.02 filing days equals that of ordinary days; $H_1$: it is larger",
        "Data": f"{n_ev:,} Item 2.02 filings by {obs['symbol'].n_unique()} US mega-caps, {sample}, with EDGAR acceptance timestamps",
        "Method": f"Market-model abnormal returns (S\\&P 500, estimation window $[-{EST_START},-{EST_END}]$), reaction day set by the acceptance time in US/Eastern, "
                  f"{N_PLACEBO:,}-draw placebo over non-event sessions of the same stock",
        "Results": f"Mean $|$CAR$|$ {abs_obs*100:.2f}\\% versus placebo {placebo_mean*100:.2f}\\% (one-sided $p={p:.4f}$); signed mean CAR {signed_mean*100:.3f}\\% "
                   f"($t={t_signed:.2f}$). Decision: \\textbf{{{decision}}}",
    }
    body = ("\\section{Design}\nEvents come from the SEC submissions API: Form 8-K filings whose items include 2.02. A filing accepted at or after 16:00 "
            "US/Eastern (or on a non-trading day) is assigned to the next session so that the measured reaction cannot precede the disclosure "
            "\\citep{mackinlay1997event}. The two-day CAR starts on that session. The placebo preserves each stock's own volatility regime.\n\n"
            "\\section{Results}\n\\input{results/table1_event_study_car.tex}\n\n"
            "\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_earnings_car_placebo.png}\n\\caption{Observed mean absolute CAR against the placebo distribution.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nFilings accepted during trading hours include a pre-announcement part of the day, which understates the reaction. "
            "Earnings dates are not exogenous to volatility. The universe is today's mega-caps, so the results do not generalise to small or delisted firms. "
            "The signed CAR needs a surprise measure (consensus data is not free) and is reported only descriptively.\n")
    bib = ("@article{mackinlay1997event,\n  title={Event studies in economics and finance},\n  author={MacKinlay, A Craig},\n  journal={Journal of Economic Literature},\n"
           "  volume={35},\n  number={1},\n  pages={13--39},\n  year={1997}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_events": n_ev, "n_stocks": obs["symbol"].n_unique(), "mean_abs_car01": abs_obs, "placebo_mean_abs_car01": placebo_mean, "placebo_p_one_sided": p,
         "signed_mean_car01": signed_mean, "signed_t": t_signed, "sample": sample, "by_year": by_year.to_dicts()}, abstract, body, bib,
        statement="US mega-cap prices react abnormally strongly on the session following an 8-K Item 2.02 earnings filing.",
        null_hypothesis="Mean |CAR[0,+1]| on filing sessions equals mean |CAR[0,+1]| on non-event sessions of the same stocks.",
        alt_hypothesis="Mean |CAR[0,+1]| on filing sessions is larger (one-sided placebo p < 0.05).",
        domain="sentiment_events", datasets=["data/community/us_sec_8k_filings.parquet", "data/market/us_equities_daily.parquet", "data/market/us_indices_daily.parquet"],
        methodology="Market-model event study with acceptance-time alignment and same-stock placebo resampling",
        metric_name="mean |CAR[0,+1]|", metric_value=abs_obs, p_value=p,
        summary_findings=f"Mean |CAR| {abs_obs*100:.2f}% vs placebo {placebo_mean*100:.2f}% over {n_ev} filings (p={p:.4f}).",
    )


if __name__ == "__main__":
    run()
