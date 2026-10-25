#!/usr/bin/env python3
"""
res_003 (Machine Learning): can gradient boosting rank next-week returns of US mega-caps from price/volume features?

H0: out-of-sample mean cross-sectional rank IC of the gradient-boosting forecast is zero.
H1: it is positive (one-sided, HAC t-test with 5 lags because 5-day labels overlap).
Walk-forward: retrain every calendar year on all earlier data with a 5-day embargo. Features use information up to
day t only; the label is the close-to-close return from t to t+5. Benchmarks: ridge and 12-1 momentum.
Known bias (reported in the paper): the 35 tickers are today's mega-caps, so the universe is survivorship-biased.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Gradient-Boosted Ranking of Next-Week Returns in US Mega-Cap Equities: A Walk-Forward Test"
HORIZON = 5
FIRST_TEST_YEAR = 2019
FEATURES = ["ret_1", "ret_5", "ret_21", "ret_63", "mom_12_1", "vol_21", "vol_63", "vol_ratio", "volume_z"]


def build_panel() -> pl.DataFrame:
    df = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_equities_daily.parquet")).sort(["symbol", "date"])
    g = lambda e: e.over("symbol")
    df = df.with_columns(
        g(pl.col("close").pct_change(1)).alias("ret_1"),
        g(pl.col("close").pct_change(5)).alias("ret_5"),
        g(pl.col("close").pct_change(21)).alias("ret_21"),
        g(pl.col("close").pct_change(63)).alias("ret_63"),
        (g(pl.col("close").shift(21)) / g(pl.col("close").shift(126)) - 1).alias("mom_12_1"),
        g(pl.col("returns").rolling_std(21)).alias("vol_21"),
        g(pl.col("returns").rolling_std(63)).alias("vol_63"),
        ((pl.col("volume") + 1).log() - g((pl.col("volume") + 1).log().rolling_mean(63))).alias("volume_z"),
        (g(pl.col("close").shift(-HORIZON)) / pl.col("close") - 1).alias("fwd"),
    ).with_columns((pl.col("vol_21") / pl.col("vol_63")).alias("vol_ratio"))
    # rank-normalise features and label within each date (cross-sectional, no look-ahead)
    ranked = [((pl.col(c).rank().over("date") - 0.5) / pl.col(c).count().over("date") - 0.5).alias(c) for c in FEATURES + ["fwd"]]
    keep = df.select(["date", "symbol", "fwd"] + FEATURES).drop_nulls(FEATURES)
    ranked_df = keep.with_columns(ranked).rename({"fwd": "fwd_rank"})
    return keep.select(["date", "symbol", "fwd"]).join(ranked_df.drop([]), on=["date", "symbol"])


def daily_ic(frame: pl.DataFrame, pred_col: str) -> np.ndarray:
    ics = []
    for _, g in frame.group_by("date", maintain_order=True):
        if g["fwd"].null_count() or g.height < 10:
            continue
        ics.append(spearmanr(g[pred_col].to_numpy(), g["fwd"].to_numpy())[0])
    return np.array(ics)


def hac_mean(x: np.ndarray):
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})
    t = float(fit.tvalues[0])
    p_one = float(fit.pvalues[0] / 2) if fit.params[0] > 0 else 1.0 - float(fit.pvalues[0] / 2)
    return float(fit.params[0]), t, p_one


def run():
    t0 = time.time()
    panel = build_panel()
    years = sorted(set(panel["date"].dt.year().to_list()))
    test_years = [y for y in years if y >= FIRST_TEST_YEAR]
    preds = []
    for y in test_years:
        start = pl.date(y, 1, 1)
        embargo_end = pl.date(y, 1, 1) - pl.duration(days=HORIZON * 2)
        train = panel.filter((pl.col("date") < embargo_end) & pl.col("fwd").is_not_null())
        test = panel.filter((pl.col("date") >= start) & (pl.col("date") < pl.date(y + 1, 1, 1)) & pl.col("fwd").is_not_null())
        if train.height < 5000 or test.height == 0:
            continue
        Xtr, ytr = train.select(FEATURES).to_numpy(), train["fwd_rank"].to_numpy()
        Xte = test.select(FEATURES).to_numpy()
        gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
        ridge = Ridge(alpha=10.0).fit(Xtr, ytr)
        preds.append(test.with_columns(
            pl.Series("gbm", gbm.predict(Xte)), pl.Series("ridge", ridge.predict(Xte)),
            pl.col("mom_12_1").alias("momentum"),
        ))
        print(f"    fold {y}: train {train.height:,} rows, test {test.height:,} rows")
    oos = pl.concat(preds)
    out, series = {}, {}
    for name in ("gbm", "ridge", "momentum"):
        ic = daily_ic(oos, name)
        mean, t, p = hac_mean(ic)
        out[name] = {"mean_ic": mean, "hac_t": t, "p_one_sided": p, "n_days": int(len(ic)), "ic_positive_share": float((ic > 0).mean())}
        series[name] = ic
    # decile spread (top vs bottom third of the 35 names) for GBM, gross of costs
    spreads = []
    for _, g in oos.group_by("date", maintain_order=True):
        if g.height < 15:
            continue
        g = g.sort("gbm")
        k = max(g.height // 3, 1)
        spreads.append(g["fwd"].tail(k).mean() - g["fwd"].head(k).mean())
    sp_mean, sp_t, _ = hac_mean(np.array(spreads))
    out["gbm_top_minus_bottom_third_5d"] = {"mean": sp_mean, "hac_t": sp_t, "note": "gross of costs, overlapping 5-day windows"}

    decision = "ACCEPTED" if out["gbm"]["p_one_sided"] < 0.05 else "REJECTED"
    n_stocks = panel["symbol"].n_unique()
    sample = f"{oos['date'].min()} to {oos['date'].max()}"

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    for name, c in (("gbm", "#d62728"), ("ridge", "#1f77b4"), ("momentum", "#7f7f7f")):
        ax.plot(np.cumsum(series[name]), label=f"{name} (mean IC {out[name]['mean_ic']:.4f})", color=c)
    ax.set_xlabel("out-of-sample trading day")
    ax.set_ylabel("cumulative daily rank IC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_cumulative_ic.png"))
    plt.close()

    rows = "\n".join(f"{n} & {out[n]['mean_ic']:.4f} & {out[n]['hac_t']:.2f} & {out[n]['p_one_sided']:.4f} & {out[n]['ic_positive_share']*100:.1f}\\% \\\\"
                     for n in ("gbm", "ridge", "momentum"))
    table = ("\\begin{table}[h!]\n\\centering\n\\caption{Out-of-sample daily rank IC against the 5-day forward return}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Model & Mean IC & HAC $t$ & One-sided $p$ & Days IC $>0$ \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_ml_performance.tex"), "w") as f:
        f.write(table)

    g = out["gbm"]
    abstract = {
        "Hypothesis": "$H_0$: the out-of-sample mean cross-sectional rank IC of a gradient-boosting forecast of the 5-day return is zero; $H_1$: it is positive",
        "Data": f"Daily prices and volume of {n_stocks} US mega-cap stocks, out-of-sample {sample} ({g['n_days']:,} days)",
        "Method": f"Walk-forward from {FIRST_TEST_YEAR} with annual retraining, a {HORIZON*2}-day embargo, cross-sectionally rank-normalised features, "
                  "ridge and 12-1 momentum benchmarks, HAC (5 lags) inference",
        "Results": f"GBM mean IC {g['mean_ic']:.4f} (HAC $t={g['hac_t']:.2f}$, one-sided $p={g['p_one_sided']:.3f}$); ridge {out['ridge']['mean_ic']:.4f}; "
                   f"momentum {out['momentum']['mean_ic']:.4f}. Decision: \\textbf{{{decision}}}",
    }
    body = ("\\section{Design}\nFeatures are 1/5/21/63-day returns, 12-1 momentum, 21/63-day volatility, their ratio and a volume z-score, each "
            "rank-normalised across stocks on the same date. The label is the cross-sectional rank of the close-to-close return from $t$ to $t+5$. "
            "Models are refit each January on data that end ten calendar days earlier, so no label overlaps the test year \\citep{gu2020empirical}.\n\n"
            "\\section{Results}\n\\input{results/table1_ml_performance.tex}\n\n"
            f"Gross of costs, the top-third minus bottom-third GBM spread earns {sp_mean*100:.3f}\\% per 5 days (HAC $t={sp_t:.2f}$).\n\n"
            "\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=0.85\\textwidth]{fig1_cumulative_ic.png}\n\\caption{Cumulative daily rank IC.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nThe universe consists of today's mega-caps, which is survivorship-biased and flatters every model; with only 35 names the daily IC "
            "is noisy; spreads ignore trading costs and borrow fees.\n")
    bib = ("@article{gu2020empirical,\n  title={Empirical asset pricing via machine learning},\n  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},\n"
           "  journal={The Review of Financial Studies},\n  volume={33},\n  number={5},\n  pages={2223--2273},\n  year={2020}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision, {"models": out, "sample": sample, "n_stocks": n_stocks, "horizon_days": HORIZON}, abstract, body, bib,
        statement="Gradient boosting on price and volume features ranks 5-day returns of US mega-caps better than chance out of sample.",
        null_hypothesis="Mean out-of-sample cross-sectional rank IC of the GBM forecast equals zero.",
        alt_hypothesis="Mean out-of-sample rank IC is positive (one-sided HAC p < 0.05).",
        domain="fundamental_factors", datasets=["data/market/us_equities_daily.parquet"],
        methodology="Walk-forward HistGradientBoostingRegressor vs Ridge vs 12-1 momentum; HAC inference on daily rank IC",
        metric_name="GBM mean rank IC", metric_value=g["mean_ic"], p_value=g["p_one_sided"],
        summary_findings=f"GBM IC {g['mean_ic']:.4f} (t={g['hac_t']:.2f}); ridge {out['ridge']['mean_ic']:.4f}; momentum {out['momentum']['mean_ic']:.4f}.",
    )


if __name__ == "__main__":
    run()
