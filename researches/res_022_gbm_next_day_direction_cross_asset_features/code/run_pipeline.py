#!/usr/bin/env python3
"""
res_022 (Machine learning): can gradient boosting predict the next-day direction of the S&P 500 from cross-asset features?

Features at the close of day t (all published by then): S&P 500 return and its 5/20-day sums, VIX level change, 20-day realised volatility, 10y-3m spread change,
returns of TLT, GLD, HYG and IEF, and the HYG-IEF 5-day return. Target: 1 if the S&P 500 return on day t+1 is positive.
Walk-forward: yearly folds, each trained on all earlier data with a 5-day embargo, fixed hyper-parameters (HistGradientBoostingClassifier, depth 3, 150 iterations, learning rate 0.05).
Statistic: pooled out-of-sample ROC AUC. H0: AUC = 0.5. H1: AUC > 0.5. P-value from 5,000 label permutations of the out-of-sample targets. Baseline: logistic regression on the same features.
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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

# --- metric helpers copied from .agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py (hypoq/ is frozen; each study is self-contained) ---

def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney)."""
    r = stats.rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Gradient Boosting and Next-Day Equity Direction: A Walk-Forward Test with Cross-Asset Features"
EMBARGO, MIN_TRAIN_YEARS, N_PERM = 5, 3, 5000


def build() -> pl.DataFrame:
    spx = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "us_indices_daily.parquet")).filter(pl.col("symbol") == "^GSPC").select(["date", pl.col("returns").alias("spx")]).drop_nulls()
    mac = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "macro", "us_yield_curve_daily.parquet")).select(["date", "cboe_vix", "yield_spread_10y_3m"]).drop_nulls()
    etf = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "global_etfs_daily.parquet")).filter(pl.col("symbol").is_in(["TLT", "GLD", "HYG", "IEF"]))
           .select(["date", "symbol", "returns"]).pivot(on="symbol", index="date", values="returns"))
    df = spx.join(mac, on="date").join(etf, on="date").sort("date").drop_nulls()
    df = df.with_columns(
        pl.col("spx").rolling_sum(5).alias("spx5"), pl.col("spx").rolling_sum(20).alias("spx20"), pl.col("spx").rolling_std(20).alias("rv20"),
        pl.col("cboe_vix").diff().alias("dvix"), pl.col("yield_spread_10y_3m").diff().alias("dspr"),
        ((pl.col("HYG") - pl.col("IEF")).rolling_sum(5)).alias("credit5"),
        (pl.col("spx").shift(-1) > 0).cast(pl.Int8).alias("y"), pl.col("spx").shift(-1).alias("ret_next"),
    )
    return df.drop_nulls()


FEATURES = ["spx", "spx5", "spx20", "rv20", "dvix", "cboe_vix", "dspr", "TLT", "GLD", "HYG", "IEF", "credit5"]


def run():
    t0 = time.time()
    df = build()
    years = df["date"].dt.year().to_numpy()
    X, y = df.select(FEATURES).to_numpy(), df["y"].to_numpy()
    first_test = int(years.min()) + MIN_TRAIN_YEARS
    oos_idx, p_gbm, p_lr = [], [], []
    for yr in range(first_test, int(years.max()) + 1):
        te = np.where(years == yr)[0]
        tr = np.where(years < yr)[0]
        tr = tr[tr < te.min() - EMBARGO]
        if len(te) == 0 or len(tr) < 250:
            continue
        gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        oos_idx.append(te)
        p_gbm.append(gbm.predict_proba(X[te])[:, 1])
        p_lr.append(lr.predict_proba(X[te])[:, 1])
    idx = np.concatenate(oos_idx)
    pg, pl_ = np.concatenate(p_gbm), np.concatenate(p_lr)
    yt = y[idx]
    a_gbm, a_lr = auc(yt, pg), auc(yt, pl_)
    rng = np.random.default_rng(3)
    perm = np.array([auc(rng.permutation(yt), pg) for _ in range(N_PERM)])
    p = float((np.sum(perm >= a_gbm) + 1) / (N_PERM + 1))
    acc = float(np.mean((pg > 0.5) == (yt == 1)))
    base = float(max(yt.mean(), 1 - yt.mean()))
    ret = df["ret_next"].to_numpy()[idx]
    long_short = np.where(pg > 0.5, ret, -ret)
    sharpe_ls = float(long_short.mean() / long_short.std(ddof=1) * np.sqrt(252))
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    ax.hist(perm, bins=40, color="#7f7f7f", alpha=0.75, label="AUC under permuted labels")
    ax.axvline(a_gbm, color="#d62728", lw=2, label=f"GBM AUC {a_gbm:.3f}")
    ax.axvline(a_lr, color="#1f77b4", lw=2, ls="--", label=f"logistic AUC {a_lr:.3f}")
    ax.set_xlabel("out-of-sample ROC AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_auc_permutation.png"))
    plt.close()

    dates = df["date"].to_numpy()[idx]
    sample = f"{str(dates.min())[:10]} to {str(dates.max())[:10]}"
    abstract = {
        "Hypothesis": "$H_0$: the out-of-sample ROC AUC of the gradient-boosting classifier for the next-day S\\&P 500 direction equals 0.5; $H_1$: it is larger",
        "Data": f"{len(idx):,} out-of-sample sessions, {sample}, {len(FEATURES)} cross-asset features published by the close",
        "Method": f"Yearly walk-forward folds with a {EMBARGO}-day embargo, fixed hyper-parameters, logistic-regression baseline, {N_PERM:,}-draw label-permutation test",
        "Results": f"GBM AUC {a_gbm:.4f} (permutation $p={p:.4f}$), logistic AUC {a_lr:.4f}; accuracy {acc*100:.1f}\\% versus {base*100:.1f}\\% for always predicting the majority class.",
    }
    body = ("\\section{Design}\nThe target is the sign of the next-day S\\&P 500 return, a task for which return predictability is known to be weak \\citep{gu2020empirical}. "
            "Features use only information published by the close of day $t$; each yearly fold trains on earlier years with an embargo of five sessions before the test year. "
            "Hyper-parameters are fixed before the evaluation, and nothing is tuned on out-of-sample data.\n\n\\section{Results}\n"
            f"A long-short rule that follows the predicted sign has an annualised Sharpe ratio of {sharpe_ls:.2f} before costs (descriptive only).\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{fig1_auc_permutation.png}\n\\caption{Out-of-sample AUC against the permutation distribution.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nPermuting labels treats daily directions as exchangeable, which is a good approximation for daily index returns but ignores volatility clustering. "
            "An AUC above 0.5 does not imply a tradable edge after costs. All features come from Yahoo Finance and FRED closes that are not perfectly synchronous.\n")
    bib = ("@article{gu2020empirical,\n  title={Empirical asset pricing via machine learning},\n  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},\n  journal={Review of Financial Studies},\n"
           "  volume={33},\n  number={5},\n  pages={2223--2273},\n  year={2020}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"n_oos": int(len(idx)), "sample": sample, "features": FEATURES, "auc_gbm": a_gbm, "auc_logistic": a_lr, "permutation_p_one_sided": p,
         "accuracy_gbm": acc, "majority_baseline_accuracy": base, "long_short_sharpe_before_costs": sharpe_ls},
        abstract, body, bib,
        statement="Gradient boosting on cross-asset features predicts the direction of the next-day S&P 500 return better than chance.",
        null_hypothesis="The out-of-sample ROC AUC of the classifier equals 0.5.",
        alt_hypothesis="The out-of-sample ROC AUC is above 0.5 (one-sided label-permutation p < 0.05).",
        domain="fundamental_factors", datasets=["data/market/us_indices_daily.parquet", "data/macro/us_yield_curve_daily.parquet", "data/market/global_etfs_daily.parquet"],
        methodology="Walk-forward HistGradientBoostingClassifier with embargo, logistic baseline and label-permutation AUC test",
        metric_name="out-of-sample ROC AUC", metric_value=a_gbm, p_value=p,
        summary_findings=f"GBM AUC {a_gbm:.4f} (p={p:.4f}), logistic {a_lr:.4f}, accuracy {acc*100:.1f}% vs majority {base*100:.1f}%.",
    )


if __name__ == "__main__":
    run()
