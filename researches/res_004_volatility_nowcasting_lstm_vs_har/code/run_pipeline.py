#!/usr/bin/env python3
"""
res_004 (Deep Learning): does an LSTM beat the HAR-RV benchmark for 5-day-ahead volatility forecasts?

Assets: S&P 500 and Bitcoin. Volatility proxy: Garman-Klass daily variance from OHLC. Target: log of the mean GK variance
over days t+1..t+5. Models: HAR (OLS on daily / 5-day / 22-day log variance) and an LSTM on 22-day windows (5 seeds averaged).
Split is strictly chronological: train/validation before 2023-01-01, test afterwards. Loss: QLIKE on the variance forecast;
significance: one-sided Diebold-Mariano test with HAC (5 lags) on the QLIKE differential, Bonferroni over the two assets.
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
import torch
from scipy.stats import norm
from torch import nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize
from hypoq.report import latex_escape as ltx

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Recurrent Neural Networks versus HAR for Five-Day-Ahead Volatility Forecasts of Equities and Bitcoin"
SPLIT = "2023-01-01"
H, WIN, SEEDS = 5, 22, 5
ASSETS = {"S&P 500": ("market/us_indices_daily.parquet", "^GSPC"), "Bitcoin": ("crypto/crypto_top_assets_daily.parquet", "BTC-USD")}
torch.set_num_threads(4)


class LSTMNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, 32, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def load(path, symbol) -> pl.DataFrame:
    df = pl.read_parquet(os.path.join(PROJECT_ROOT, "data", path)).filter(pl.col("symbol") == symbol).sort("date")
    df = df.filter((pl.col("open") > 0) & (pl.col("high") >= pl.col("low")) & (pl.col("low") > 0)).with_columns(
        (0.5 * (pl.col("high") / pl.col("low")).log() ** 2 - (2 * np.log(2) - 1) * (pl.col("close") / pl.col("open")).log() ** 2).clip(lower_bound=1e-8).alias("gk_var"),
        (pl.col("close") / pl.col("open")).log().alias("oc"),
    ).with_columns(pl.col("gk_var").log().alias("lv"))
    return df.with_columns(
        pl.col("lv").rolling_mean(5).alias("lv5"),
        pl.col("lv").rolling_mean(22).alias("lv22"),
        pl.col("gk_var").rolling_mean(H).shift(-H).alias("fut_var"),
    ).with_columns(pl.col("fut_var").log().alias("y"))


def qlike(var_true, var_pred):
    ratio = var_true / var_pred
    return ratio - np.log(ratio) - 1


def dm_one_sided(loss_a, loss_b):
    """H1: model A (LSTM) has lower loss than B (HAR). Returns (mean diff, t, one-sided p)."""
    d = loss_b - loss_a
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": H})
    t = float(fit.tvalues[0])
    return float(d.mean()), t, float(1 - norm.cdf(t))


def make_windows(feat: np.ndarray, y: np.ndarray, idx: np.ndarray):
    X = np.stack([feat[i - WIN + 1: i + 1] for i in idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y[idx], dtype=torch.float32)


def run_asset(name, path, symbol):
    df = load(path, symbol).drop_nulls(["lv", "lv5", "lv22"])
    dates = df["date"].to_numpy()
    n_all = df.height
    feat = np.column_stack([df["lv"].to_numpy(), df["oc"].to_numpy(), df["lv5"].to_numpy()])
    y = df["y"].to_numpy()
    valid = ~np.isnan(y)
    is_train = (dates < np.datetime64(SPLIT)) & valid
    test_idx = np.where((dates >= np.datetime64(SPLIT)) & valid)[0]
    train_idx = np.where(is_train)[0]
    train_idx = train_idx[train_idx >= max(WIN, 22)]
    # purge: the last H training labels overlap the test period
    train_idx = train_idx[train_idx < np.where(dates >= np.datetime64(SPLIT))[0][0] - H]
    cut = int(len(train_idx) * 0.8)
    fit_idx, val_idx = train_idx[:cut], train_idx[cut:]

    # HAR benchmark
    har_X = lambda idx: sm.add_constant(np.column_stack([df["lv"].to_numpy()[idx], df["lv5"].to_numpy()[idx], df["lv22"].to_numpy()[idx]]))
    har = sm.OLS(y[train_idx], har_X(train_idx)).fit()
    har_pred = har.predict(har_X(test_idx))

    # LSTM, standardised with training statistics only
    mu, sd = feat[fit_idx].mean(0), feat[fit_idx].std(0) + 1e-9
    fz = (feat - mu) / sd
    ymu, ysd = y[fit_idx].mean(), y[fit_idx].std()
    Xtr, ytr = make_windows(fz, (y - ymu) / ysd, fit_idx)
    Xva, yva = make_windows(fz, (y - ymu) / ysd, val_idx)
    Xte, _ = make_windows(fz, (y - ymu) / ysd, test_idx)
    preds = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        net = LSTMNet(feat.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_state, patience = float("inf"), None, 0
        for epoch in range(120):
            net.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), 64):
                bi = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.mse_loss(net(Xtr[bi]), ytr[bi]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = nn.functional.mse_loss(net(Xva), yva).item()
            if vloss < best - 1e-5:
                best, best_state, patience = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 12:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(net(Xte).numpy() * ysd + ymu)
    lstm_pred = np.mean(preds, axis=0)

    y_test = y[test_idx]
    resid_sd = float(np.std(y[train_idx] - har.predict(har_X(train_idx))))
    # exp of a log forecast is the median, not the mean: add the same variance correction for both models
    var_true = np.exp(y_test)
    var_har, var_lstm = np.exp(har_pred + 0.5 * resid_sd ** 2), np.exp(lstm_pred + 0.5 * resid_sd ** 2)
    q_har, q_lstm = qlike(var_true, var_har), qlike(var_true, var_lstm)
    mse_har, mse_lstm = float(np.mean((y_test - har_pred) ** 2)), float(np.mean((y_test - lstm_pred) ** 2))
    diff, t, p = dm_one_sided(q_lstm, q_har)
    return {
        "n_test": int(len(test_idx)), "test_start": str(dates[test_idx[0]])[:10], "test_end": str(dates[test_idx[-1]])[:10],
        "qlike_har": float(q_har.mean()), "qlike_lstm": float(q_lstm.mean()),
        "mse_log_har": mse_har, "mse_log_lstm": mse_lstm,
        "dm_mean_diff_har_minus_lstm": diff, "dm_t": t, "dm_p_one_sided": p,
        "seeds": SEEDS, "har_betas": [float(b) for b in har.params],
    }, (dates[test_idx], y_test, har_pred, lstm_pred)


def run():
    t0 = time.time()
    res, plots = {}, {}
    for name, (path, sym) in ASSETS.items():
        print(f">>> {name}")
        res[name], plots[name] = run_asset(name, path, sym)
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items() if k != "har_betas"})
    alpha = 0.05 / len(ASSETS)
    wins = [n for n, r in res.items() if r["dm_p_one_sided"] < alpha]
    losses = [n for n, r in res.items() if r["qlike_lstm"] > r["qlike_har"]]
    decision = "ACCEPTED" if len(wins) == len(ASSETS) else ("INCONCLUSIVE" if wins else "REJECTED")

    fig, axes = plt.subplots(1, len(ASSETS), figsize=(11, 3.6), dpi=200)
    for ax, (name, (d, yt, hp, lp)) in zip(np.atleast_1d(axes), plots.items()):
        ax.plot(d, np.exp(0.5 * yt) * np.sqrt(252) * 100, color="#7f7f7f", lw=0.8, label="realised (GK)")
        ax.plot(d, np.exp(0.5 * hp) * np.sqrt(252) * 100, color="#1f77b4", lw=0.8, label="HAR")
        ax.plot(d, np.exp(0.5 * lp) * np.sqrt(252) * 100, color="#d62728", lw=0.8, label="LSTM")
        ax.set_title(f"{name}: 5-day volatility, annualised % (test)", fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_volatility_forecasts.png"))
    plt.close()

    rows = "\n".join(
        f"{ltx(n)} & {r['qlike_har']:.4f} & {r['qlike_lstm']:.4f} & {r['mse_log_har']:.4f} & {r['mse_log_lstm']:.4f} & {r['dm_t']:.2f} & {r['dm_p_one_sided']:.4f} \\\\"
        for n, r in res.items())
    table = ("\\begin{table}[h!]\n\\centering\n\\caption{Out-of-sample forecast loss (lower is better) and one-sided Diebold--Mariano test, H$_1$: LSTM better}\n"
             "\\begin{tabular}{lcccccc}\n\\toprule\nAsset & QLIKE HAR & QLIKE LSTM & MSE-log HAR & MSE-log LSTM & DM $t$ & $p$ \\\\\n\\midrule\n"
             + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_dl_volatility_performance.tex"), "w") as f:
        f.write(table)

    best_name = min(res, key=lambda n: res[n]["dm_p_one_sided"])
    abstract = {
        "Hypothesis": "$H_0$: an LSTM and the HAR benchmark have equal expected QLIKE loss for the 5-day-ahead variance; $H_1$: the LSTM has lower loss",
        "Data": "Daily OHLC of the S\\&P 500 and Bitcoin; " + "; ".join(f"{ltx(n)} test {r['test_start']} to {r['test_end']} ({r['n_test']:,} days)" for n, r in res.items()),
        "Method": f"Garman--Klass variance, HAR on log variance, LSTM on {WIN}-day windows with early stopping and {SEEDS} seeds, chronological split at {SPLIT}, "
                  f"one-sided DM test with HAC errors, Bonferroni $\\alpha={alpha}$",
        "Results": "; ".join(f"{ltx(n)}: QLIKE HAR {r['qlike_har']:.4f} vs LSTM {r['qlike_lstm']:.4f} (DM $p={r['dm_p_one_sided']:.3f}$)" for n, r in res.items())
                   + f". Decision: \\textbf{{{decision}}}",
    }
    body = ("\\section{Design}\nThe target is the log mean Garman--Klass variance of the next five sessions. HAR regresses it on the current, weekly and monthly "
            "log variance \\citep{corsi2009simple}; the LSTM sees the same information as 22-day windows. Both receive the same log-normal bias "
            "correction when converted to variance. Training labels that overlap the test period are purged.\n\n"
            "\\section{Results}\n\\input{results/table1_dl_volatility_performance.tex}\n\n"
            "\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=\\textwidth]{fig1_volatility_forecasts.png}\n\\caption{Test-period forecasts.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nGarman--Klass from daily bars is a noisy proxy for realised variance; one chronological split is used rather than rolling re-estimation; "
            "the network is small and its hyper-parameters were not tuned on the test period.\n")
    bib = ("@article{corsi2009simple,\n  title={A simple approximate long-memory model of realized volatility},\n  author={Corsi, Fulvio},\n"
           "  journal={Journal of Financial Econometrics},\n  volume={7},\n  number={2},\n  pages={174--196},\n  year={2009}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision, {"assets": res, "bonferroni_alpha": alpha, "split": SPLIT}, abstract, body, bib,
        statement="An LSTM forecasts 5-day-ahead variance of equities and Bitcoin better than the HAR benchmark out of sample.",
        null_hypothesis="LSTM and HAR have equal expected QLIKE loss (Diebold-Mariano).",
        alt_hypothesis="LSTM has lower QLIKE loss (one-sided DM p < 0.025 per asset).",
        domain="macro_transmission", datasets=[f"data/{p}" for p, _ in ASSETS.values()],
        methodology="Garman-Klass variance; HAR OLS vs LSTM (PyTorch, 5 seeds); QLIKE; DM test with HAC errors",
        metric_name="DM t-stat (best asset)", metric_value=res[best_name]["dm_t"], p_value=res[best_name]["dm_p_one_sided"],
        summary_findings="; ".join(f"{n}: QLIKE HAR {r['qlike_har']:.4f}, LSTM {r['qlike_lstm']:.4f}, DM p={r['dm_p_one_sided']:.3f}" for n, r in res.items()),
    )


if __name__ == "__main__":
    run()
