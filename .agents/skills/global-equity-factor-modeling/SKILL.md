---
name: global-equity-factor-modeling
description: Construct, evaluate, and backtest cross-sectional and time-series factor models (Fama-French, momentum, quality, low-volatility, size, and value) across US, European, and Asian equity markets.
version: 1.0.0
author: AI Quantitative Research Network
tags: [equities, factors, fama-french, momentum, multi-region, asia, europe, us, portfolio-construction]
---

# Global Equity Factor Modeling Skill

## Overview
This skill specifies the operational framework for multi-region equity factor research. It enables systematic testing of whether empirical asset pricing anomalies (Size, Value, Momentum, Quality, Low-Volatility) exhibit cross-regional portability across the **United States**, **Europe**, and **Asia**.

Empirical datasets leveraged:
* `data/market/us_equities_daily.parquet` & `us_indices_daily.parquet`
* `data/market/eu_equities_daily.parquet` & `eu_indices_daily.parquet`
* `data/market/asia_equities_daily.parquet` & `asia_indices_daily.parquet`
* `data/market/global_etfs_daily.parquet` (Factor benchmarks: SPY, IWM, QQQ, EFA, EEM, INDA, FXI, EWJ)

---

## 1. Cross-Sectional Factor Normalization Protocol

Before computing factor quintile spreads or Fama-MacBeth regressions:
1. **Cross-Sectional Winsorization**: Trim extreme outliers at the 1st and 99th percentiles:
   $$\tilde{f}_{i, t} = \min\left(\max(f_{i, t}, Q_{0.01}(f_t)), Q_{0.99}(f_t)\right)$$
2. **Cross-Sectional Z-Scoring**: Zero-center and normalize unit variance per date $t$:
   $$z_{i, t} = \frac{\tilde{f}_{i, t} - \mu_{f, t}}{\sigma_{f, t}}$$

```python
import polars as pl
import numpy as np

def compute_cross_sectional_factor_scores(df_equities: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    """
    Normalizes a factor column within each calendar date across available universe.
    """
    return (
        df_equities
        .with_columns([
            pl.col(factor_col).quantile(0.01).over("date").alias("q01"),
            pl.col(factor_col).quantile(0.99).over("date").alias("q99"),
        ])
        .with_columns(
            pl.col(factor_col).clip(pl.col("q01"), pl.col("q99")).alias("factor_clipped")
        )
        .with_columns([
            pl.col("factor_clipped").mean().over("date").alias("factor_mean"),
            pl.col("factor_clipped").std().over("date").alias("factor_std"),
        ])
        .with_columns(
            ((pl.col("factor_clipped") - pl.col("factor_mean")) / pl.col("factor_std")).alias(f"{factor_col}_zscore")
        )
        .drop(["q01", "q99", "factor_clipped", "factor_mean", "factor_std"])
    )
```

---

## 2. Multi-Region Momentum Factor Formulation (Carhart 12-2)

Momentum $MOM_{i, t}$ is calculated over the past 252 trading days skipping the most recent 21 days (reversal avoidance):
$$MOM_{i, t} = \prod_{s=22}^{252} (1 + R_{i, t-s}) - 1$$

```python
import polars as pl

def compute_12_2_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes 12-2 momentum factor across multi-region equities.
    df schema must include ['symbol', 'date', 'close'].
    """
    return (
        df.sort(["symbol", "date"])
        .with_columns([
            # 21-day lag price and 252-day lag price
            pl.col("close").shift(21).over("symbol").alias("p_lag21"),
            pl.col("close").shift(252).over("symbol").alias("p_lag252")
        ])
        .with_columns(
            ((pl.col("p_lag21") / pl.col("p_lag252")) - 1.0).alias("factor_mom_12_2")
        )
        .drop(["p_lag21", "p_lag252"])
    )
```

---

## 3. Fama-MacBeth 2-Step Cross-Sectional Regressions

Evaluates factor premia $\gamma_t$ with Newey--West HAC correction across time:
* **Step 1 (Time-Series)**: Estimate factor loadings $\beta_i$ for each asset.
* **Step 2 (Cross-Sectional)**: For each period $t$, regress forward returns on estimated loadings:
  $$R_{i, t+1} = \gamma_{0, t} + \sum_{k=1}^K \gamma_{k, t} \beta_{i, k} + \eta_{i, t+1}$$
* **HAC Premia Average**:
  $$\bar{\gamma}_k = \frac{1}{T} \sum_{t=1}^T \gamma_{k, t}, \quad t\text{-stat} = \frac{\bar{\gamma}_k}{\text{HAC-SE}(\gamma_k)}$$

```python
import numpy as np
import statsmodels.api as sm

def fama_macbeth_evaluation(cross_sectional_results: list) -> dict:
    """
    Computes time-series mean and Newey-West HAC standard errors of daily factor risk premia.
    cross_sectional_results: list of estimated gamma vectors from each date t.
    """
    gammas = np.array(cross_sectional_results) # (T, K)
    T, K = gammas.shape
    
    results = {}
    for k in range(K):
        series = gammas[:, k]
        ones = np.ones(T)
        ols = sm.OLS(series, ones).fit(cov_type='HAC', cov_kwds={'maxlags': int(4 * (T / 100)**(2/9))})
        results[f"factor_{k}"] = {
            "mean_premium_bps": float(ols.params[0] * 10000),
            "t_statistic": float(ols.tvalues[0]),
            "p_value": float(ols.pvalues[0]),
            "is_significant": bool(ols.pvalues[0] < 0.05)
        }
    return results
```

---

## 4. Multi-Region Portability Hypothesis Test ($H_0$)
When testing an anomaly discovered in the US across European and Asian markets:
* **Null Hypothesis ($H_0$)**: Factor risk premium in Region $M$ is statistically indistinguishable from zero ($\bar{\gamma}_{k, M} \le 0$).
* **Alternative Hypothesis ($H_1$)**: Factor risk premium in Region $M$ is strictly positive ($\bar{\gamma}_{k, M} > 0, t \ge 1.96$) net of regional transactions costs.

**Regional Transaction Cost Hurdles**:
* **US Large-Cap**: 10 bps two-way.
* **European Large-Cap**: 15 bps two-way (stamp duty / transaction taxes where applicable).
* **Asian Equities**: 25 bps two-way (higher retail participation, wider spreads).
