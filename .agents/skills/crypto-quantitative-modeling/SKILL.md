---
name: crypto-quantitative-modeling
description: Quantitative modeling of cryptocurrency market microstructure, high-volatility regime switching, cross-asset spillover (crypto vs equities and gold), momentum and mean-reversion anomalies, and tail risk estimation.
version: 1.0.0
author: hypoq
tags: [crypto, bitcoin, ethereum, volatility, cross-asset, tail-risk, momentum, var-cvar, econometrics]
---

# Cryptocurrency Quantitative Modeling Skill

## Overview
This skill defines the empirical modeling protocol for digital assets and cryptocurrencies. Cryptocurrencies operate 24/7/365 with fat-tailed return distributions, regime shifts, and intermittent liquidity shocks.

Empirical datasets leveraged:
* `data/crypto/crypto_top_assets_daily.parquet` (Daily OHLCV, returns, and 30-day realized volatility for BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT, NEAR)
* `data/crypto/crypto_market_daily.parquet` (ETH/BTC ratio, aggregate volume proxy, 30-day rolling correlation with S&P 500 and Gold)
* `data/market/us_indices_daily.parquet` (`^GSPC`, `^NDX`)
* `data/macro/global_commodities_daily.parquet` (`GC=F` Gold, `CL=F` Crude Oil)

---

## 1. Calendar Alignment & Zero Look-Ahead Invariants
Crypto trades continuously on Saturdays and Sundays, while traditional equity and debt markets operate on Monday–Friday trading calendars:
1. **Never Forward Fill Equity into Crypto**: Do not create fictitious Saturday/Sunday stock prices.
2. **Backward Join Strategy**: When aligning crypto with traditional macro/equity dates $t$, always match the crypto observation corresponding exactly to traditional market close date $t$:
   ```python
   # Match crypto close strictly to equity market date
   df_aligned = df_equity.join(df_crypto, on="date", how="inner")
   ```
3. **Weekend Accumulator**: When analyzing Monday market openings, the cumulative weekend return of Bitcoin ($\prod_{d \in \{\text{Sat}, \text{Sun}\}} (1 + R_d) - 1$) serves as an exogenous weekend information shock to traditional markets.

---

## 2. Realized Volatility & Extreme Value Estimators

Standard close-to-close sample volatility understates intraday volatility. Use Parkinson (1980) or Garman--Klass (1980) extreme value estimators:
$$\sigma_{\text{Parkinson}}^2 = \frac{1}{4 \ln 2} \ln \left( \frac{H_t}{L_t} \right)^2$$
$$\sigma_{\text{GK}}^2 = 0.511 \left[ \ln \left( \frac{H_t}{L_t} \right) \right]^2 - 0.019 \left[ \ln \left( \frac{C_t}{O_t} \right) \ln \left( \frac{H_t L_t}{O_t^2} \right) - 2 \ln \left( \frac{H_t}{O_t} \right) \ln \left( \frac{L_t}{O_t} \right) \right] - 0.383 \left[ \ln \left( \frac{C_t}{O_t} \right) \right]^2$$

```python
import polars as pl
import numpy as np

def compute_crypto_volatility_estimators(parquet_path: str = "data/crypto/crypto_top_assets_daily.parquet"):
    df = pl.read_parquet(parquet_path)
    
    df_vol = df.with_columns([
        # Log Parkinson Volatility
        (np.sqrt((1.0 / (4.0 * np.log(2.0))) * (pl.col("high") / pl.col("low")).log() ** 2) * np.sqrt(365)).alias("vol_parkinson_annualized"),
        # Standard Close-to-Close 30-day realized volatility
        pl.col("volatility_30d")
    ])
    return df_vol
```

---

## 3. Macro-Crypto Spillover & Correlation Dynamics

Models whether Bitcoin acts as "Digital Gold" (inflation hedge / safe haven) or a "High-Beta Tech Asset" (correlated with S&P 500 / Nasdaq):
$$R_{\text{BTC}, t} = \alpha + \beta_1 R_{\text{SP500}, t} + \beta_2 R_{\text{Gold}, t} + \beta_3 \Delta \text{VIX}_t + \epsilon_t$$

```python
import polars as pl
import statsmodels.api as sm

def estimate_macro_crypto_beta():
    df_mkt = pl.read_parquet("data/crypto/crypto_market_daily.parquet").drop_nulls()
    
    # 30-day rolling correlation summary
    summary = df_mkt.select([
        pl.col("btc_sp500_corr_30d").mean().alias("mean_corr_sp500"),
        pl.col("btc_sp500_corr_30d").quantile(0.10).alias("q10_corr_sp500"),
        pl.col("btc_sp500_corr_30d").quantile(0.90).alias("q90_corr_sp500"),
        pl.col("btc_gold_corr_30d").mean().alias("mean_corr_gold"),
        pl.col("btc_gold_corr_30d").quantile(0.10).alias("q10_corr_gold"),
        pl.col("btc_gold_corr_30d").quantile(0.90).alias("q90_corr_gold"),
    ])
    return summary
```

---

## 4. Tail Risk: Value-at-Risk (VaR) & Expected Shortfall (CVaR)

Because crypto exhibits excess kurtosis ($\kappa > 5$), parametric Gaussian VaR severely underestimates tail probability. Always compute:
1. **Historical Non-Parametric VaR**: Empirical quantile at $\alpha = 0.01, 0.05$.
2. **Cornish-Fisher Modified VaR**: Adjusted for skewness $S$ and kurtosis $K$.
3. **Expected Shortfall (CVaR)**: Conditional expectation of loss exceeding VaR:
   $$\text{CVaR}_\alpha = -\mathbb{E}[R_t \mid R_t \le -\text{VaR}_\alpha]$$

```python
import numpy as np
import polars as pl
from scipy import stats

def compute_crypto_tail_risk(returns: np.ndarray, alpha: float = 0.05):
    """
    Computes Historical VaR and Expected Shortfall for a cryptocurrency return vector.
    """
    clean_rets = returns[~np.isnan(returns)]
    if len(clean_rets) < 100:
        return {"error": "Insufficient sample size"}

    # Historical VaR
    var_hist = -np.percentile(clean_rets, alpha * 100)
    
    # Expected Shortfall (CVaR)
    cvar_hist = -np.mean(clean_rets[clean_rets <= -var_hist])
    
    # Cornish-Fisher Expansion
    z = stats.norm.ppf(1 - alpha)
    s = stats.skew(clean_rets)
    k = stats.kurtosis(clean_rets)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * (s**2) / 36
    var_cf = -(np.mean(clean_rets) - z_cf * np.std(clean_rets))

    return {
        "var_historical_95": float(var_hist),
        "cvar_expected_shortfall_95": float(cvar_hist),
        "var_cornish_fisher_95": float(var_cf),
        "skewness": float(s),
        "excess_kurtosis": float(k),
        "n_obs": len(clean_rets)
    }
```

---

## 5. Execution Rules & Frictions
* **Friction Standard**: Any backtested crypto anomaly (momentum, mean-reversion, cross-pair arbitrage) must account for a minimum of **25 bps to 50 bps** two-way trading fee (taker fee + slippage).
* **Funding Rate Regimes**: In perpetual futures studies, funding rate payments must be accounted for on an 8-hour basis.
