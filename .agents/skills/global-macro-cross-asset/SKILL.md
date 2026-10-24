---
name: global-macro-cross-asset
description: Model cross-border macroeconomic transmission, sovereign yield curves (US Treasuries, Bunds, JGBs), currency pass-through (USD, EUR, JPY, CNY, VND), and commodity price shock propagation using Vector Autoregressions (VAR/VECM), local projections, and Diebold-Yilmaz connectedness.
version: 1.0.0
author: AI Quantitative Research Network
tags: [macro, yield-curve, cross-asset, fx-transmission, commodities, var, vecm, diebold-yilmaz, global]
---

# Global Macro & Cross-Asset Transmission Skill

## Overview
This skill establishes standardized quantitative methodologies for modeling international macroeconomic linkages, sovereign term structure dynamics, foreign exchange pass-through, and commodity price shock propagation across **Asia**, the **United States**, **Europe**, and global markets.

Empirical datasets leveraged:
* `data/macro/us_yield_curve_daily.parquet` (3M T-bill, 5Y/10Y T-notes, 30Y T-bond, 10Y-3M slope, CBOE VIX)
* `data/macro/us_macro_daily.parquet` & `us_fed_balance_sheet_weekly.parquet` (DFF, EFFR, SOFR, Fed Balance Sheet)
* `data/macro/eu_macro_daily.parquet` (EUR/USD, GBP/USD, EUR/GBP, EUR/CHF, EUR/JPY)
* `data/macro/asia_macro_daily.parquet` (USD/JPY, USD/CNY, USD/KRW, USD/INR, USD/SGD, USD/HKD, AUD/USD, USD/VND)
* `data/macro/global_commodities_daily.parquet` (Energy, Precious & Industrial Metals, Agricultural commodities)
* `data/market/global_etfs_daily.parquet` (Sovereign bonds, credit spreads, equity benchmarks)

---

## 1. Time-Zone Causality Invariants (Non-Negotiable)
When modeling cross-border market returns across Asia, Europe, and the US:
$$\text{Asia Close (UTC 06:00–08:00)} \prec \text{Europe Close (UTC 16:30)} \prec \text{US Close (UTC 21:00)}$$
1. **Zero Look-Ahead**: Asian same-day returns $R_{\text{Asia}, t}$ cannot be explained by US close returns $R_{\text{US}, t}$ on calendar date $t$. US close on date $t-1$ precedes Asia open on date $t$.
2. **Lagged Identification**: In regressions estimating US monetary policy or Treasury yield impact on Asian equities:
   $$R_{\text{Asia}, t} = \alpha + \beta_1 \Delta y_{\text{US}, t-1} + \beta_2 R_{\text{US}, t-1} + \epsilon_t$$

---

## 2. Yield Curve Term Structure Factor Extraction

Extract Level ($L_t$), Slope ($S_t$), and Curvature ($C_t$) using the empirical Nelson--Siegel decomposition:
$$y_t(\tau) = L_t + S_t \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} \right) + C_t \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau} \right)$$
Or the principal empirical proxies:
* **Level**: $y_{10Y, t}$
* **Slope**: $y_{10Y, t} - y_{3M, t}$ (or $10Y - 2Y$)
* **Curvature**: $2 \cdot y_{5Y, t} - (y_{3M, t} + y_{10Y, t})$

```python
import polars as pl
import numpy as np
import statsmodels.api as sm

def analyze_yield_curve_regime(parquet_path: str = "data/macro/us_yield_curve_daily.parquet"):
    df = pl.read_parquet(parquet_path).sort("date")
    
    # Calculate daily changes in term premia & slope
    df_metrics = df.with_columns([
        (pl.col("tnote_10y") - pl.col("tbill_3m")).alias("term_spread_10y_3m"),
        (2 * pl.col("tnote_5y") - (pl.col("tbill_3m") + pl.col("tnote_10y"))).alias("curvature_factor"),
        pl.col("tnote_10y").diff().alias("d_10y"),
        pl.col("cboe_vix").diff().alias("d_vix")
    ]).drop_nulls()

    return df_metrics
```

---

## 3. Diebold-Yilmaz Cross-Asset Volatility Connectedness

Quantifies directional volatility spillovers across equities, commodities, bonds, and currencies using generalized forecast error variance decomposition (GFEVD) from a Vector Autoregression (VAR):
$$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} \left( e_i' A_h \Sigma e_j \right)^2}{\sum_{h=0}^{H-1} \left( e_i' A_h \Sigma A_h' e_i \right)}$$

```python
import numpy as np
import polars as pl
from statsmodels.tsa.api import VAR

def estimate_cross_asset_spillover(returns_matrix: np.ndarray, maxlags: int = 2, horizon: int = 10):
    """
    Computes total spillover index across global asset classes.
    returns_matrix: (T, N) array of stationary returns or log volatilities.
    """
    model = VAR(returns_matrix)
    results = model.fit(maxlags=maxlags, ic='bic')
    
    fevd = results.fevd(horizon)
    decomp = fevd.decomp # shape (N, horizon, N)
    
    # Extract horizon H normalized variance shares
    P_H = decomp[:, horizon - 1, :]
    N = P_H.shape[0]
    
    # Row normalize to ensure sum to 100%
    P_norm = P_H / P_H.sum(axis=1, keepdims=True) * 100.0
    
    # Off-diagonal sum (spillover)
    total_spillover = (P_norm.sum() - np.trace(P_norm)) / N
    
    return {
        "total_spillover_index": float(total_spillover),
        "variance_matrix": P_norm,
        "var_aic": float(results.aic),
        "var_bic": float(results.bic)
    }
```

---

## 4. Currency Pass-Through & Uncovered Interest Parity (UIP)

Tests exchange rate sensitivity to cross-border policy rate differentials:
$$\Delta s_{t+k} = \alpha + \beta (i_{t} - i^*_{t}) + \gamma \Delta \text{VIX}_t + \epsilon_{t+k}$$
* Standard UIP posits $\beta = 1$.
* The empirical "Forward Premium Puzzle" shows $\beta < 0$ or insignificant in emerging Asia due to sovereign risk premia and FX interventions.

```python
import polars as pl
import statsmodels.api as sm

def test_fx_rate_sensitivity(
    fx_path: str = "data/macro/asia_macro_daily.parquet",
    yc_path: str = "data/macro/us_yield_curve_daily.parquet",
    pair_symbol: str = "USDJPY=X"
):
    df_fx = pl.read_parquet(fx_path).filter(pl.col("symbol") == pair_symbol).sort("date")
    df_yc = pl.read_parquet(yc_path).sort("date")

    joined = df_fx.join(df_yc, on="date", how="inner").drop_nulls()
    
    # Regression: FX returns on US 10Y yield changes and VIX changes
    joined = joined.with_columns([
        pl.col("tnote_10y").diff().alias("d_us10y"),
        pl.col("cboe_vix").diff().alias("d_vix")
    ]).drop_nulls()

    y = joined["returns"].to_numpy()[1:]
    X = sm.add_constant(joined.select(["d_us10y", "d_vix"]).to_numpy()[1:])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "pair": pair_symbol,
        "beta_us10y": float(model.params[1]),
        "tstat_us10y": float(model.tvalues[1]),
        "pvalue_us10y": float(model.pvalues[1]),
        "beta_vix": float(model.params[2]),
        "tstat_vix": float(model.tvalues[2]),
        "r2": float(model.rsquared)
    }
```

---

## 5. Invariants & Empirical Verification
1. **Stationarity**: All macro level series (yields, FX levels, commodity spot prices) must be checked with Augmented Dickey-Fuller (`adfuller`). First differences $\Delta y_t$ or log returns $\ln(P_t/P_{t-1})$ must be used unless cointegration is established via Johansen trace test.
2. **Robust Covariance**: Never report standard OLS standard errors on time-series macro data. Always apply Newey--West HAC standard errors with Bartlett kernel.
3. **Transaction Costs**: Any cross-border currency carry trade or cross-asset commodity momentum strategy must deduct realistic bid-ask spreads (min 3-5 bps on G10 FX, 15-25 bps on EM FX).
