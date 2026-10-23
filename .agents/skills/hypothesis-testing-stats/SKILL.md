---
name: hypothesis-testing-stats
description: Statistical and econometric hypothesis testing methods, including Newey-West HAC standard errors, ADF unit root tests, event studies, and calendar placebo tests.
version: 1.0.0
author: kim / vnresearch
tags: [statistics, econometrics, hypothesis-testing, newey-west, stationarity]
---

# Hypothesis Testing & Statistics Skill

## Overview
Every empirical report in this system must rigorously test a falsifiable hypothesis. This skill outlines the econometric methods, statistical tests, error correction routines, and decision frameworks required to produce peer-reviewed results.

## When to Use
- When formulating formal Null ($H_0$) and Alternative ($H_1$) hypotheses.
- When estimating time-series regressions where serial correlation and heteroskedasticity violate standard OLS assumptions.
- When conducting index rebalancing or announcement event studies.
- When deciding whether a result is **ACCEPTED**, **REJECTED**, or **INCONCLUSIVE**.

## Standard Decision Framework
1. **Significance Threshold**: $\alpha = 0.05$ (two-tailed $p$-value $< 0.05$ or $|t| \ge 1.96$). For multiple testing, apply Benjamini-Hochberg FDR correction.
2. **Economic Threshold**: Effect size must exceed realistic market frictions (e.g. minimum 15 bps two-way trading fee and borrow cost).
3. **Decisions**:
   - **ACCEPTED**: $H_0$ rejected, statistical significance confirmed, economic hurdle satisfied.
   - **REJECTED**: $H_0$ fails to be rejected ($p \ge 0.05$ or coefficient sign opposite).
   - **INCONCLUSIVE**: Insufficient sample size, extreme parameter instability, or failure of placebo test.

## Implementation Protocols

### 1. Augmented Dickey-Fuller (ADF) Stationarity Verification
Financial predictors and spreads must be verified for covariance stationarity before regression:
```python
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series: np.ndarray, name: str = "Series"):
    clean = series[~np.isnan(series)]
    res = adfuller(clean, autolag="AIC")
    stat, pval, used_lag = res[0], res[1], res[2]
    is_stationary = pval < 0.05
    print(f"[ADF] {name}: Stat={stat:.4f}, p-val={pval:.4e}, Lags={used_lag} -> Stationary: {is_stationary}")
    return {"stat": float(stat), "p_value": float(pval), "is_stationary": bool(is_stationary)}
```

### 2. Time-Series Regression with Newey-West HAC Robust Covariance
Financial return regressions exhibit residual autocorrelation and time-varying volatility. Standard OLS standard errors underestimate uncertainty.
```python
import numpy as np
import statsmodels.api as sm

def run_hac_regression(y: np.ndarray, X: np.ndarray, max_lags: int = None):
    """
    Fits OLS with Newey-West HAC standard errors.
    Optimal lag length defaults to Bartlett rule: L = 4 * (T / 100)^(2/9)
    """
    T = len(y)
    if max_lags is None:
        max_lags = int(4 * (T / 100.0) ** (2.0 / 9.0))
    
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    
    return {
        "params": results.params.tolist(),
        "bse": results.bse.tolist(),
        "tvalues": results.tvalues.tolist(),
        "pvalues": results.pvalues.tolist(),
        "rsquared": float(results.rsquared),
        "rsquared_adj": float(results.rsquared_adj),
        "f_pvalue": float(results.f_pvalue) if results.f_pvalue is not None else None,
        "nobs": int(results.nobs)
    }
```

### 3. Calendar Placebo Bootstrap Resampling
To rule out spurious correlation or sample mining, generate pseudo-event distributions:
```python
import numpy as np

def run_calendar_placebo(actual_stat: float, series: np.ndarray, n_draws: int = 1000, seed: int = 42):
    np.random.seed(seed)
    placebo_stats = []
    T = len(series)
    for _ in range(n_draws):
        shuffled = np.random.permutation(series)
        pseudo_metric = np.mean(shuffled[:20]) - np.mean(shuffled[20:40])
        placebo_stats.append(pseudo_metric)
    
    placebo_stats = np.array(placebo_stats)
    p_placebo = np.mean(np.abs(placebo_stats) >= np.abs(actual_stat))
    return {
        "actual_stat": float(actual_stat),
        "placebo_mean": float(np.mean(placebo_stats)),
        "placebo_p_value": float(p_placebo),
        "is_robust": bool(p_placebo < 0.05)
    }
```
