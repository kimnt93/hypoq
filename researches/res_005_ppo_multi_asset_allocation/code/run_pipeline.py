#!/usr/bin/env python3
"""
res_005 (Reinforcement Learning): does PPO allocation across global ETFs beat equal weight after transaction costs?

Assets: SPY, TLT, GLD, EEM, HYG (total returns from adjusted closes). Action = target weights (softmax, fully invested).
Reward = log(1 + w_t . r_{t+1} - cost * turnover). The decision at close t is applied to the return of t+1 (no look-ahead).
Train: 2016-2022 (random 1-year episodes). Test: 2023 onward, deterministic policy, never used for tuning.
Primary pre-specified benchmark: daily-rebalanced equal weight with the same costs.
H0: Sharpe(PPO) <= Sharpe(equal weight). H1: Sharpe(PPO) > Sharpe(equal weight): one-sided stationary-bootstrap p-value.
"""
import os
import sys
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gymnasium import spaces
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from hypoq.pipeline import finalize

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TITLE = "Proximal Policy Optimisation for Multi-Asset ETF Allocation under Transaction Costs"
ASSETS = ["SPY", "TLT", "GLD", "EEM", "HYG"]
SPLIT = np.datetime64("2023-01-01")
LOOKBACK, COST, TIMESTEPS, SEEDS, EPISODE = 20, 0.0005, 120_000, 3, 252
N_BOOT, BLOCK = 5000, 20


def load_returns():
    df = (pl.read_parquet(os.path.join(PROJECT_ROOT, "data", "market", "global_etfs_daily.parquet"))
          .filter(pl.col("symbol").is_in(ASSETS)).sort(["symbol", "date"])
          .with_columns(pl.col("adj_close").pct_change().over("symbol").alias("tr")))
    wide = df.pivot(on="symbol", index="date", values="tr").sort("date").select(["date"] + ASSETS).drop_nulls()
    return wide["date"].to_numpy(), wide.select(ASSETS).to_numpy()


class AllocationEnv(gym.Env):
    def __init__(self, rets, lookback=LOOKBACK, episode=EPISODE, random_start=True, scale=None):
        super().__init__()
        self.r, self.L, self.episode, self.random_start = rets, lookback, episode, random_start
        self.scale = scale if scale is not None else rets.std(0) + 1e-9
        n = rets.shape[1]
        self.action_space = spaces.Box(-3.0, 3.0, (n,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (lookback * n + n,), np.float32)

    def _obs(self):
        window = self.r[self.t - self.L: self.t] / self.scale
        return np.concatenate([window.ravel(), self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = self.r.shape[1]
        hi = len(self.r) - self.episode - 1
        self.t = int(self.np_random.integers(self.L, hi)) if self.random_start else self.L
        self.end = self.t + self.episode if self.random_start else len(self.r) - 1
        self.w = np.ones(n) / n
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        target = np.exp(a - a.max())
        target /= target.sum()
        turnover = np.abs(target - self.w).sum()
        gross = float(target @ self.r[self.t])  # weights chosen at close t earn the return of day t (index t = next day)
        net = gross - COST * turnover
        drift = target * (1 + self.r[self.t])
        self.w = drift / drift.sum()
        self.t += 1
        done = self.t >= self.end
        return self._obs(), float(np.log1p(max(net, -0.99))), done, False, {"net": net, "target": target}


def fixed_weight_backtest(rets, weights):
    """Daily-rebalanced fixed weights with the same cost model (cost on drift back to target)."""
    w_prev, nets = weights.copy(), []
    for t in range(len(rets)):
        turnover = np.abs(weights - w_prev).sum()
        nets.append(float(weights @ rets[t]) - COST * turnover)
        drift = weights * (1 + rets[t])
        w_prev = drift / drift.sum()
    return np.array(nets)


def stats(r):
    ann = 252
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"ann_return": float(np.prod(1 + r) ** (ann / len(r)) - 1), "ann_vol": float(r.std(ddof=1) * np.sqrt(ann)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ann)), "max_drawdown": float(dd)}


def stationary_bootstrap_sharpe_diff(a, b, n_boot=N_BOOT, block=BLOCK, seed=11):
    """One-sided p-value for H1: Sharpe(a) > Sharpe(b), recentred under H0 (stationary bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    sh = lambda x: x.mean() / x.std(ddof=1)
    obs = sh(a) - sh(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        pos = int(rng.integers(n))
        for k in range(n):
            idx[k] = pos
            pos = int(rng.integers(n)) if rng.random() < 1.0 / block else (pos + 1) % n
        diffs[i] = sh(a[idx]) - sh(b[idx])
    p = float((np.sum(diffs - diffs.mean() >= obs) + 1) / (n_boot + 1))
    return float(obs * np.sqrt(252)), p


def run():
    t0 = time.time()
    dates, rets = load_returns()
    train_mask = dates < SPLIT
    tr, te = rets[train_mask], rets[~train_mask]
    scale = tr.std(0) + 1e-9
    models = []
    for seed in range(SEEDS):
        env = AllocationEnv(tr, scale=scale)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=1024, batch_size=128, learning_rate=3e-4, gamma=0.99, ent_coef=0.005,
                    policy_kwargs=dict(net_arch=[64, 64]))
        model.learn(total_timesteps=TIMESTEPS)
        models.append(model)
        print(f"    PPO seed {seed} trained ({TIMESTEPS:,} steps)")

    def ensemble(obs):
        return np.mean([m.predict(obs, deterministic=True)[0] for m in models], axis=0)

    # test env must use the TRAIN scale: build it explicitly
    env_test = AllocationEnv(te, random_start=False, scale=scale)
    obs, _ = env_test.reset()
    nets, ws, done = [], [], False
    while not done:
        obs, _, done, _, info = env_test.step(ensemble(obs))
        nets.append(info["net"])
        ws.append(info["target"])
    ppo, w_path = np.array(nets), np.array(ws)
    test_rets = te[LOOKBACK:LOOKBACK + len(ppo)]
    test_dates = dates[~train_mask][LOOKBACK:LOOKBACK + len(ppo)]
    n_assets = len(ASSETS)
    ew = fixed_weight_backtest(test_rets, np.ones(n_assets) / n_assets)
    spy = test_rets[:, 0]
    sixty_forty = fixed_weight_backtest(test_rets, np.array([0.6, 0.4, 0, 0, 0]))
    curves = {"PPO ensemble": ppo, "Equal weight": ew, "SPY buy-and-hold": spy, "60/40 SPY/TLT": sixty_forty}
    perf = {k: stats(v) for k, v in curves.items()}
    avg_turnover = float(np.abs(np.diff(w_path, axis=0)).sum(1).mean())
    diff, p = stationary_bootstrap_sharpe_diff(ppo, ew)
    decision = "ACCEPTED" if p < 0.05 else "REJECTED"

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    for (k, v), c in zip(curves.items(), ("#d62728", "#1f77b4", "#7f7f7f", "#2ca02c")):
        ax.plot(test_dates, np.cumprod(1 + v), label=f"{k} (Sharpe {perf[k]['sharpe']:.2f})", color=c, lw=1.2)
    ax.set_ylabel("growth of 1 (net of costs)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_rl_equity_curves.png"))
    plt.close()

    rows = "\n".join(f"{k} & {v['ann_return']*100:.2f}\\% & {v['ann_vol']*100:.2f}\\% & {v['sharpe']:.2f} & {v['max_drawdown']*100:.1f}\\% \\\\" for k, v in perf.items())
    table = ("\\begin{table}[H]\n\\centering\n\\caption{Out-of-sample performance net of 5 bps cost per unit turnover}\n\\begin{tabular}{lcccc}\n\\toprule\n"
             "Strategy & Ann. return & Ann. vol. & Sharpe & Max drawdown \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    os.makedirs(os.path.join(RESEARCH_DIR, "results"), exist_ok=True)
    with open(os.path.join(RESEARCH_DIR, "results", "table1_rl_performance.tex"), "w") as f:
        f.write(table)

    sample = f"{test_dates[0]} to {test_dates[-1]}"
    sample = sample.replace("T00:00:00.000000000", "").replace("T00:00:00", "")
    abstract = {
        "Hypothesis": "$H_0$: the Sharpe ratio of a PPO allocation policy does not exceed that of daily-rebalanced equal weight; $H_1$: it does",
        "Data": f"Total returns of {', '.join(ASSETS)}; training before 2023, out-of-sample {sample[:10]} to {sample[-10:]} ({len(ppo):,} days)",
        "Method": f"PPO ({SEEDS} seeds, {TIMESTEPS:,} steps each, averaged), softmax weights, 5 bps cost per unit turnover, decision at close $t$ applied to day $t{{+}}1$, "
                  "one-sided stationary-bootstrap test of the Sharpe difference",
        "Results": f"PPO Sharpe {perf['PPO ensemble']['sharpe']:.2f} vs equal weight {perf['Equal weight']['sharpe']:.2f} (difference {diff:.2f}, $p={p:.3f}$). "
                   f"",
    }
    body = ("\\section{Design}\nThe environment exposes the last 20 daily returns of each asset (scaled by training volatility) and the current drifted weights. "
            "The agent outputs unconstrained scores mapped to long-only fully invested weights by a softmax; reward is the log net return. "
            "Training episodes are random one-year windows before 2023; the test period is simulated once and never used for tuning \\citep{moody2001learning}.\n\n"
            "\\section{Results}\n\\input{results/table1_rl_performance.tex}\n\n"
            f"The ensemble policy turns over {avg_turnover*100:.1f}\\% of the portfolio per day on average.\n\n"
            "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.85\\textwidth]{fig1_rl_equity_curves.png}\n\\caption{Out-of-sample growth of one unit.}\n\\end{figure}\n\n"
            "\\section{Limitations}\nFive ETFs and a single three-year test window give low statistical power; cash and financing are ignored; "
            "PPO hyper-parameters were fixed a priori, not searched, and results vary across seeds (averaged here).\n")
    bib = ("@article{moody2001learning,\n  title={Learning to trade via direct reinforcement},\n  author={Moody, John and Saffell, Matthew},\n"
           "  journal={IEEE Transactions on Neural Networks},\n  volume={12},\n  number={4},\n  pages={875--889},\n  year={2001}\n}\n")
    finalize(
        RESEARCH_DIR, t0, TITLE, decision,
        {"performance": perf, "sharpe_difference_vs_equal_weight": diff, "bootstrap_p_one_sided": p, "avg_daily_turnover": avg_turnover,
         "n_test_days": len(ppo), "timesteps_per_seed": TIMESTEPS, "seeds": SEEDS, "cost_per_unit_turnover": COST, "assets": ASSETS}, abstract, body, bib,
        statement="A PPO allocation agent trained on 2016-2022 ETF returns beats daily-rebalanced equal weight out of sample after costs.",
        null_hypothesis="Sharpe(PPO) <= Sharpe(equal weight).",
        alt_hypothesis="Sharpe(PPO) > Sharpe(equal weight) (one-sided stationary bootstrap p < 0.05).",
        domain="fundamental_factors", datasets=["data/market/global_etfs_daily.parquet"],
        methodology="Stable-Baselines3 PPO on a custom gymnasium allocation environment; stationary bootstrap Sharpe test",
        metric_name="Sharpe difference vs equal weight", metric_value=diff, p_value=p,
        summary_findings="; ".join(f"{k}: Sharpe {v['sharpe']:.2f}, return {v['ann_return']*100:.1f}%" for k, v in perf.items()),
    )


if __name__ == "__main__":
    run()
