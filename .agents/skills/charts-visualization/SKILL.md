---
name: charts-visualization
description: Render publication-grade financial charts and econometric figures using Polars and Seaborn/Matplotlib, outputting exclusively high-DPI PNGs.
version: 1.0.0
author: kim / vnresearch
tags: [visualization, seaborn, matplotlib, polars, charts, figures]
---

# Charts & Visualization Skill

## Overview
This skill defines formatting rules and code recipes for creating figures in quantitative research reports. Every empirical finding, return path, event-study response, or correlation matrix must be visually communicated with clarity, high resolution, and aesthetic quality.

## When to Use
- Generating figures for the LaTeX research report under `figures/fig*.png`.
- Visualizing time-series basis spreads, rolling volatilities, drawdown charts, or cumulative active alpha.
- Plotting event-study abnormal return windows $[-10, +10]$ with confidence bands.

## Non-Negotiable Styling Rules
1. **Output Format**: Strictly save **PNG** files (`dpi=200` to `300`). Zero PDF reports in `vnresearch/`.
2. **Typography**: Clean sans-serif fonts, explicit axis labels with units (e.g. `Return (%)`, `Spread (bps)`, `Date`).
3. **Palette**: Professional colorblind-safe palettes (e.g., Deep, Muted, Colorblind, or `#1f77b4`, `#d62728`, `#2ca02c`).
4. **Layout**: Tight layout (`plt.tight_layout()` or `bbox_inches="tight"`), no unnecessary borders, soft grid (`alpha=0.3`).

## Implementation Recipes

### Recipe 1: Cumulative Performance and Underwater Drawdown
```python
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_performance_and_drawdown(dates, strategy_returns, benchmark_returns, out_path: str):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # Cumulative curves
    strat_cum = np.cumprod(1 + strategy_returns) - 1
    bench_cum = np.cumprod(1 + benchmark_returns) - 1
    
    ax1.plot(dates, strat_cum * 100, label="Strategy Alpha (Net)", color="#1f77b4", lw=2)
    ax1.plot(dates, bench_cum * 100, label="Benchmark (VNINDEX)", color="#7f7f7f", lw=1.5, ls="--")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Empirical Out-of-Sample Performance", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Underwater Drawdown
    peak = np.maximum.accumulate(strat_cum + 1)
    drawdown = (strat_cum + 1 - peak) / peak * 100
    ax2.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.4, label="Strategy Drawdown")
    ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.set_ylim(min(drawdown.min() * 1.1, -5), 0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
```

### Recipe 2: Event Study Abnormal Return Dynamics
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_event_study_car(event_window: np.ndarray, car_mean: np.ndarray, car_se: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    ax.plot(event_window, car_mean * 100, marker="o", color="#004c6d", lw=2, label="Cumulative Abnormal Return (CAR)")
    ax.fill_between(
        event_window,
        (car_mean - 1.96 * car_se) * 100,
        (car_mean + 1.96 * car_se) * 100,
        color="#257d9d",
        alpha=0.25,
        label="95% Confidence Band"
    )
    ax.axvline(0, color="crimson", ls="--", lw=1.5, label="Effective Event Date")
    ax.axhline(0, color="black", ls=":", lw=1)
    
    ax.set_title("Rebalancing Event Study: Cumulative Abnormal Return Window", fontsize=12, fontweight="bold")
    ax.set_xlabel("Event Time (Sessions relative to Rebalance)", fontsize=10)
    ax.set_ylabel("CAR (%)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
```
